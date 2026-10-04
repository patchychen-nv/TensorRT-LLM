# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Script real executor transitions across replicated and plain ADP schedulers."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from _torch.executor.kv_cache.test_dkv_scheduler import _PageManager, _scheduler
from dkv_test_utils import LockstepTpGroup, make_request

from tensorrt_llm._torch.pyexecutor.dkv import DkvInvariantChecker, sync_dkv_sample_results
from tensorrt_llm._torch.pyexecutor.llm_request import FinishReason, LlmRequest, LlmRequestState
from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor
from tensorrt_llm._torch.pyexecutor.resource_manager import ResourceManager, ResourceManagerType
from tensorrt_llm._torch.pyexecutor.sampler import SampleState
from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests
from tensorrt_llm._torch.pyexecutor.seq_slot_manager import SeqSlotManager

pytestmark = pytest.mark.cpu_only

_ARRIVALS = (
    ((10, 1, 8), (11, 1, 8)),
    ((20, 1, 4), (21, 1, 4)),
    (),
    ((40, 0, 4), (41, 1, 8), (42, 0, 4)),
    ((50, 0, 8), (51, 1, 4)),
)


class _ScriptPages(_PageManager):
    """Bounded pages with observable lifecycle calls and resident dummy backing."""

    dkv_scope = "lifecycle"

    def add_dummy_requests(
        self, request_ids: list[int], token_nums=None, *, is_gen: bool, prepare_resource: bool
    ) -> list[LlmRequest]:
        assert token_nums is None and is_gen and prepare_resource
        requests = []
        for request_id in request_ids:
            request = make_request(request_id, prompt_len=2, is_dummy=True)
            request.state = LlmRequestState.GENERATION_IN_PROGRESS
            self.kv_cache_map[request_id] = SimpleNamespace(capacity=2, is_active=True)
            self.trace.append(("dummy", request_id))
            requests.append(request)
        return requests

    def prepare_resources(self, batch: ScheduledRequests) -> None:
        self.trace.append(
            ("prepare_resources", tuple(req.py_request_id for req in batch.all_requests()))
        )

    def update_context_resources(self, batch: ScheduledRequests) -> None:
        self.trace.append(("commit", tuple(req.py_request_id for req in batch.context_requests)))
        assert all(request.is_finished for request in batch.context_requests)

    def update_resources(self, batch: ScheduledRequests, *args) -> None:
        self.trace.append(
            ("update_resources", tuple(req.py_request_id for req in batch.all_requests()))
        )


class _FixedSampler:
    """Deterministic CPU forward completion; executor owns all surrounding control flow."""

    def __init__(self) -> None:
        self.sampled: list[tuple[int, ...]] = []

    @staticmethod
    def beam_width(requests: list[LlmRequest]) -> int:
        return 1

    @staticmethod
    def is_generation_model() -> bool:
        return True

    @staticmethod
    def setup_sampler_step(batch: ScheduledRequests) -> None:
        assert all(request.seq_slot is not None for request in batch.all_requests())

    def sample_async(self, batch: ScheduledRequests, outputs, prefix_sum) -> SampleState:
        requests = [request for request in batch.all_requests() if not request.is_dummy]
        self.sampled.append(tuple(request.py_request_id for request in requests))
        return SampleState(requests=requests)

    @staticmethod
    def update_requests(state: SampleState, resource_manager: ResourceManager) -> None:
        for request in state.requests:
            request.add_new_token(100 + request.py_request_id, 0)
            request.py_decoding_iter = 1
            request.finish_by_reason(FinishReason.LENGTH)


def _executor(dist, *, dkv_enabled: bool) -> PyExecutor:
    executor = PyExecutor.__new__(PyExecutor)
    executor.dist = dist
    executor.dkv_enabled = dkv_enabled
    executor.enable_attention_dp = True
    executor.disable_overlap_scheduler = True
    executor.model_engine = SimpleNamespace(is_warmup=False, route_capture=None)
    executor.draft_model_engine = None
    executor.is_warmup = False
    executor._dkv_forward_dummies = []
    executor._dkv_sampler_errors = []
    executor._dkv_commit_reason = None
    executor._dkv_freed_request_ids = []
    executor.active_requests = []
    executor.iter_counter = 0
    executor.stream_interval = 1
    executor.force_terminate_ctx_for_partial_reuse = False
    executor.enable_joint_kv_cache_reuse = False
    executor.gather_all_responses = False
    executor.result_wait_queues = {}
    executor._prefetched_request_ids = set()
    executor._disagg_pp_termination_handler = None
    executor.perf_manager = Mock()
    executor._disagg_coordinator = Mock()
    executor._maybe_attach_ctx_usage = Mock()
    executor._handle_errors = Mock(side_effect=AssertionError("Unexpected sampler error"))
    executor.sampler = _FixedSampler()
    executor.kv_cache_manager = _ScriptPages(pages=64, block_size=4, tp_rank=dist.tp_rank)
    executor.scheduler = _scheduler(
        executor.kv_cache_manager,
        group_size=2 if dkv_enabled else None,
        batch_size=2,
        tokens=8,
    )
    executor.resource_manager = ResourceManager(
        {
            ResourceManagerType.KV_CACHE_MANAGER: executor.kv_cache_manager,
            ResourceManagerType.SEQ_SLOT_MANAGER: SeqSlotManager(3),
        }
    )
    return executor


def _run_script(*, dkv_enabled: bool) -> tuple[list[dict], LockstepTpGroup]:
    group = LockstepTpGroup(2)
    executors = [_executor(dist, dkv_enabled=dkv_enabled) for dist in group.ranks]

    def run(dist) -> dict:
        executor = executors[dist.tp_rank]
        manager = executor.kv_cache_manager
        emitted = []

        def enqueue(responses) -> None:
            records = [(request_id, dist.tp_rank) for request_id, _ in responses]
            gathered = dist.tp_gather(records)
            if gathered is not None:
                emitted.extend(record for rank_records in gathered for record in rank_records)

        executor._enqueue_responses = enqueue
        if dkv_enabled:
            executor._dkv_invariant_checker = DkvInvariantChecker(dist, enabled=True)
            executor._initialize_dkv_forward_dummies()
        dummy_baseline = [
            (
                request.get_tokens(0),
                request.state,
                manager.kv_cache_map[request.py_request_id].capacity,
            )
            for request in executor._dkv_forward_dummies
        ]
        initial_pages = manager.used_pages
        forwarded = []
        page_snapshots = []
        dummy_iterations = 0

        for iteration, arrivals in enumerate(_ARRIVALS):
            executor.iter_counter = iteration
            for request_id, owner, length in arrivals:
                if dkv_enabled or owner == dist.tp_rank:
                    request = make_request(
                        request_id, compute_rank=owner, local_rank=dist.tp_rank, prompt_len=length
                    )
                    request.create_response = Mock(wraps=request.create_response)
                    executor.active_requests.append(request)
            scheduled = executor.scheduler.schedule_request(executor.active_requests, set())
            global_batch = ScheduledRequests()
            global_batch.reset_context_requests(scheduled.context_requests)
            global_batch.generation_requests = scheduled.generation_requests
            global_batch.paused_requests = scheduled.paused_requests
            global_batch.recompute_paused_requests = scheduled.recompute_paused_requests
            forward_batch = executor._dkv_forward_batch(global_batch)
            real_forwarded = [
                req.py_request_id for req in forward_batch.all_requests() if not req.is_dummy
            ]
            if not dkv_enabled and not forward_batch.batch_size:
                # ADP's local dummy is ephemeral; its id is intentionally excluded from parity.
                dummy = make_request(0, is_dummy=True, prompt_len=2)
                dummy.state = LlmRequestState.GENERATION_IN_PROGRESS
                forward_batch.generation_requests = [dummy]
            forwarded.append(real_forwarded)
            assert executor._can_queue(global_batch, forward_batch) == (True, True)
            if dkv_enabled and not forwarded[-1]:
                assert forward_batch.all_requests() == [executor._dkv_forward_dummies[dist.tp_rank]]
                dummy_iterations += 1

            executor.resource_manager.prepare_resources(
                global_batch, forward_batch=forward_batch if dkv_enabled else None
            )
            for request in global_batch.all_requests():
                if dkv_enabled and not request.py_dkv_is_local:
                    assert request.seq_slot is None
            stats = executor._collect_scheduled_batch_stats(forward_batch)
            assert stats.num_ctx_requests == len(forwarded[-1])
            assert stats.num_gen_requests == 0
            executor._setup_sampler_step(forward_batch)
            sample_state = executor._sample_async(
                forward_batch, {"logits": torch.zeros(forward_batch.batch_size, 1)}
            )
            executor._update_request_states(global_batch)
            executor._update_requests(sample_state, executor.resource_manager)
            if dkv_enabled:
                sync_dkv_sample_results(
                    dist,
                    global_batch.all_requests(),
                    forward_batch.all_requests(),
                    executor._dkv_sampler_errors,
                )
            executor._update_v2_context_resources(global_batch)
            finished = executor._handle_responses()
            for request in finished:
                if dkv_enabled and not request.py_dkv_is_local:
                    request.create_response.assert_not_called()
                    assert request.get_num_tokens(0) == request.orig_prompt_len
                else:
                    request.create_response.assert_called_with(False, dist.rank)
            executor.resource_manager.update_resources(
                global_batch, forward_batch=forward_batch if dkv_enabled else None
            )
            if not dkv_enabled:
                for request in forward_batch.generation_requests:
                    executor.resource_manager.free_resources(request)
            assert not executor._dkv_sampler_errors
            assert [
                (req.get_tokens(0), req.state, manager.kv_cache_map[req.py_request_id].capacity)
                for req in executor._dkv_forward_dummies
            ] == dummy_baseline
            page_snapshots.append(manager.used_pages)

        assert not executor.active_requests
        assert manager.used_pages == initial_pages
        if dkv_enabled and dist.tp_rank == 0:
            assert dummy_iterations == 3
            assert len(executor.sampler.sampled) == 2
        return {
            "forwarded": forwarded,
            "emitted": emitted,
            "pages": page_snapshots,
            "lifecycle": manager.trace,
        }

    results = group.run(run)
    assert group.traces[0] == group.traces[1]
    return results, group


def test_five_iteration_dkv_lifecycle_matches_adp(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TRTLLM_DKV_DUAL_LEDGER", "1")
    adp, _ = _run_script(dkv_enabled=False)
    dkv, group = _run_script(dkv_enabled=True)
    assert [rank["forwarded"] for rank in dkv] == [rank["forwarded"] for rank in adp]
    expected = [(request_id, owner) for arrivals in _ARRIVALS for request_id, owner, _ in arrivals]
    assert sorted(dkv[0]["emitted"]) == sorted(expected)
    assert sorted(adp[0]["emitted"]) == sorted(expected)
    assert dkv[1]["emitted"] == []
    assert dkv[0]["pages"] == dkv[1]["pages"]
    assert dkv[0]["lifecycle"] == dkv[1]["lifecycle"]
    assert sum(step[2] == "tp_gather" for step in group.traces[0]) == len(_ARRIVALS)


def test_debug_forward_participation_rejects_missing_rank_dummy() -> None:
    group = LockstepTpGroup(2)
    executors = [_executor(dist, dkv_enabled=True) for dist in group.ranks]

    def run(dist) -> None:
        executor = executors[dist.tp_rank]
        executor._dkv_invariant_checker = DkvInvariantChecker(dist, enabled=True)
        executor._initialize_dkv_forward_dummies()
        request = make_request(8, compute_rank=1, local_rank=dist.tp_rank)
        request.context_chunk_size = request.orig_prompt_len
        batch = ScheduledRequests()
        batch.append_context_request(request)
        executor.active_requests = [request]
        forward = executor._dkv_forward_batch(batch)
        if dist.tp_rank == 0:
            forward.generation_requests.clear()
        executor._can_queue(batch, forward)

    with pytest.raises(RuntimeError, match="forward participation differs"):
        group.run(run)


def test_globally_empty_batch_keeps_resident_dummies_out_of_forward() -> None:
    group = LockstepTpGroup(2)
    executors = [_executor(dist, dkv_enabled=True) for dist in group.ranks]

    def run(dist) -> None:
        executor = executors[dist.tp_rank]
        executor._dkv_invariant_checker = DkvInvariantChecker(dist, enabled=True)
        executor._initialize_dkv_forward_dummies()
        pages = executor.kv_cache_manager.used_pages
        batch = ScheduledRequests()
        forward = executor._dkv_forward_batch(batch)
        assert forward.batch_size == 0
        assert executor._can_queue(batch, forward) == (False, False)
        assert executor.kv_cache_manager.used_pages == pages
        assert executor.active_requests == []

    group.run(run)
    assert group.traces[0] == group.traces[1]
