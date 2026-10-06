# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Run the real ``PyExecutor._executor_loop`` on every rank of a lockstep group, on CPU.

Each rank gets an executor whose scheduler, sampler stand-in, sequence slots and KV page
bookkeeping are real objects or deterministic fakes. The loop, S-control, S-sample, the response
and release commit points and the end-of-iteration check all run unmodified; the harness only
records the order in which the phases of one iteration are entered.
"""

from contextlib import nullcontext
from dataclasses import dataclass, field
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import torch
from _torch.executor.kv_cache.test_dkv_scheduler import _PageManager, _scheduler
from dkv_test_utils import LockstepTpGroup, make_request

from tensorrt_llm._torch.disaggregation.kv_cache_transceiver import CtxTransferStatus
from tensorrt_llm._torch.disaggregation.orchestration.transfer_manager import AsyncTransferManager
from tensorrt_llm._torch.pyexecutor.dkv import DkvInvariantChecker, compute_ownership
from tensorrt_llm._torch.pyexecutor.dkv_plan import (
    PageCost,
    build_dkv_plan,
    layer_types_from_compress_ratios,
    plan_fingerprint,
)
from tensorrt_llm._torch.pyexecutor.llm_request import FinishReason, LlmRequest, LlmRequestState
from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor
from tensorrt_llm._torch.pyexecutor.resource_manager import ResourceManager, ResourceManagerType
from tensorrt_llm._torch.pyexecutor.sampler import SampleState
from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests
from tensorrt_llm._torch.pyexecutor.seq_slot_manager import SeqSlotManager

_PY_EXECUTOR = "tensorrt_llm._torch.pyexecutor.py_executor"


class LoopPages(_PageManager):
    """Bounded KV pages with the manager hooks the executor loop calls."""

    dkv_scope = "loop"

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.freed: list[int] = []
        self.released_index_slots: list[int] = []
        self.on_free = None

    def add_dummy_requests(
        self, request_ids, token_nums=None, *, is_gen: bool, prepare_resource: bool
    ) -> list[LlmRequest]:
        requests = []
        for request_id in request_ids:
            request = make_request(request_id, prompt_len=2, is_dummy=True)
            request.state = LlmRequestState.GENERATION_IN_PROGRESS
            self.kv_cache_map[request_id] = SimpleNamespace(capacity=2, is_active=True)
            requests.append(request)
        return requests

    def prepare_resources(self, batch: ScheduledRequests, *args, **kwargs) -> None:
        pass

    def update_context_resources(self, batch: ScheduledRequests) -> None:
        pass

    def update_resources(self, batch: ScheduledRequests, *args, **kwargs) -> None:
        pass

    def free_resources(self, request: LlmRequest, *args, **kwargs) -> None:
        self.freed.append(request.py_request_id)
        if self.on_free is not None:
            self.on_free(request.py_request_id)
        super().free_resources(request)

    def release_index_slot(self, request_id: int) -> None:
        self.released_index_slots.append(request_id)

    def consume_dkv_trace(self) -> list:
        trace, self.trace = self.trace, []
        return trace

    def get_dkv_control_digest(self):
        return self.used_pages, ((self.pages - self.used_pages,),)

    def get_dkv_state_fingerprint(self) -> list[tuple]:
        return sorted(
            (request_id, cache.capacity) for request_id, cache in self.kv_cache_map.items()
        )


class FinishingSampler:
    """Finishes a request when its prefill completes; failures can be injected per request."""

    def __init__(self, fail_ids: set[int] | None = None) -> None:
        self.fail_ids = fail_ids or set()
        self.sampled: list[tuple[int, ...]] = []

    @staticmethod
    def beam_width(requests: list[LlmRequest]) -> int:
        return 1

    @staticmethod
    def is_generation_model() -> bool:
        return True

    @staticmethod
    def setup_sampler_step(batch: ScheduledRequests) -> None:
        pass

    def sample_async(self, batch: ScheduledRequests, outputs, prefix_sum) -> SampleState:
        requests = [request for request in batch.all_requests() if not request.is_dummy]
        self.sampled.append(tuple(request.py_request_id for request in requests))
        failing = [request for request in requests if request.py_request_id in self.fail_ids]
        if failing:
            raise RuntimeError(f"injected sampler failure {[r.py_request_id for r in failing]}")
        return SampleState(requests=requests)

    @staticmethod
    def update_requests(state: SampleState, resource_manager: ResourceManager) -> None:
        for request in state.requests:
            if request.context_remaining_length == 0:
                request.add_new_token(100 + request.py_request_id, 0)
                request.py_decoding_iter = 1
                request.finish_by_reason(FinishReason.LENGTH)


@dataclass
class LoopScript:
    """New requests per iteration as ``(request_id, compute_rank, prompt_len)``.

    The loop leaves when the script is exhausted and no request is active; that condition is part
    of the replicated state, so every rank leaves in the same iteration.
    """

    arrivals: list[list[tuple[int, int, int]]]
    pages: int = 64
    batch_size: int = 2
    max_num_tokens: int = 8
    chunked: bool = False
    fail_ids: set[int] = field(default_factory=set)
    debug: bool = False
    dual_ledger: bool = True
    # ``(rank, iteration)`` at which that rank's forward raises.
    forward_error: tuple[int, int] | None = None
    # ``(rank, iteration)`` after which that rank holds a KV cache that no other rank has.
    leak: tuple[int, int] | None = None
    # A sampler failure is charged to the error budget of the failing rank as fatal.
    fatal_budget: bool = False
    # Requests that only prefill and then send their KV; the owner's transceiver reports the
    # transfer complete from this iteration on.
    context_only: set[int] = field(default_factory=set)
    transfer_done_iteration: int = 0
    # Under the layer-split layout every rank sends the layers it owns of a context-only request,
    # and the request is released once all of them have reported.
    layer_split: bool = False
    # Rank -> the iteration from which its transfers complete, instead of ``transfer_done_iteration``.
    transfer_done_by_rank: dict[int, int] = field(default_factory=dict)
    # ``(rank, iteration)``: from that iteration on, the transfers of that rank end in an error.
    transfer_fails: tuple[int, int] | None = None
    # The layer-split layout: the loop plans the data plane of every iteration and drains it.
    streamer: bool = False
    # ``(rank, iteration)`` at which that rank plans without the first request of the batch.
    plan_skew: tuple[int, int] | None = None
    # The loop does not drain the data plane before the commits (the streamer ignores the drain).
    skip_drain: bool = False


class _PageCosts:
    """A plan cost model of invented sizes: some pages to fetch, some to write back."""

    def page_cost(self, layer: int, kind, history: int, chunk: int) -> PageCost:
        return PageCost(512, history // 4 % 3, 1 + chunk // 4)


class PlanStreamer:
    """What the loop sees of ``DkvStreamer``: it builds the plan, takes it and is drained."""

    NUM_LAYERS = 6

    def __init__(
        self, executor: PyExecutor, group_size: int, skew, record, skip_drain: bool = False
    ) -> None:
        self._executor = executor
        self._group_size = group_size
        self._skew = skew
        self._record = record
        self._skip_drain = skip_drain
        self._drained = True
        self._owners = compute_ownership(self.NUM_LAYERS, group_size)
        self._layer_types = layer_types_from_compress_ratios([1, 4, 128] * 2)
        self.plans: list = []
        self.planned: list = []
        self.last_plan = None

    def plan_for(self, requests):
        requests = list(requests)
        if self._skew == (self._executor.dist.tp_rank, self._executor.iter_counter):
            requests = requests[1:]
        self.planned.append(requests)
        return build_dkv_plan(
            requests,
            self._owners,
            self._layer_types,
            _PageCosts(),
            group_size=self._group_size,
            ring_depth=2,
        )

    def set_plan(self, plan, iteration: int = 0) -> None:
        self.plans.append(plan)
        self.last_plan = plan
        self._drained = False
        self._record("plan", plan_fingerprint(plan))

    def drain(self, timeout: float) -> list:
        self._record("drain", timeout)
        self._drained = not self._skip_drain
        return []

    def assert_idle(self, what: str) -> None:
        if not self._drained:
            raise RuntimeError(
                f"DKV invariant violation: {what} while the data plane of rank "
                f"{self._executor.dist.tp_rank} is not drained"
            )


@dataclass
class RankRun:
    """What one rank observed while the loop ran."""

    events: list[tuple] = field(default_factory=list)
    emitted: list[tuple[int, int]] = field(default_factory=list)
    errors: list[tuple[str, tuple[int, ...]]] = field(default_factory=list)
    pages: LoopPages | None = None
    sends: list[int] = field(default_factory=list)
    status_polls: int = 0
    initial_pages: int = 0
    final_pages: int = 0
    executor: PyExecutor | None = None
    group: LockstepTpGroup | None = None


def _attach_transceiver(executor: PyExecutor, script: LoopScript, run: RankRun, record) -> None:
    """Context-only requests send through a mock transceiver and the real coordinator.

    Only the owner's transceiver is ever asked to send or to report status, unless the script is
    layer-split, where every rank is. Its transfers complete from ``script.transfer_done_iteration``
    (or the iteration its rank has in ``script.transfer_done_by_rank``) on, or end in an error from
    the iteration of ``script.transfer_fails``.
    """

    def status(at_least: int) -> CtxTransferStatus:
        run.status_polls += 1
        rank = executor.dist.tp_rank
        in_transfer = executor.async_transfer_manager.requests_in_transfer()
        sent = [rid for rid in run.sends if rid in in_transfer]
        if script.transfer_fails is not None and (
            rank == script.transfer_fails[0] and executor.iter_counter >= script.transfer_fails[1]
        ):
            return CtxTransferStatus([], sent)
        done_from = script.transfer_done_by_rank.get(rank, script.transfer_done_iteration)
        if executor.iter_counter < done_from:
            return CtxTransferStatus([], [])
        return CtxTransferStatus(sent, [])

    def send(request: LlmRequest) -> None:
        run.sends.append(request.py_request_id)
        record("send", request.py_request_id)
        # A real transceiver attaches the context phase parameters that the response carries.
        request.create_response = Mock(return_value=SimpleNamespace(result=SimpleNamespace()))

    executor.kv_cache_transceiver = Mock(
        kv_transfer_timeout_ms=60000,
        pipeline_transfer_enabled=False,
        supports_inflight_cancellation=False,
        check_context_transfer_status=Mock(side_effect=status),
        has_retired_send_session=Mock(return_value=False),
        cancel_request=Mock(return_value=True),
        respond_and_send_async=Mock(side_effect=send),
    )
    # KVCacheManagerV2 has no store-and-pin, so production runs the manager without storing blocks;
    # the page manager has no such methods, so a call would fail the test.
    executor.async_transfer_manager = AsyncTransferManager(
        executor.resource_manager, should_store_blocks=False
    )
    executor.__dict__.pop("_disagg_coordinator", None)


def build_loop_executor(dist, script: LoopScript, run: RankRun) -> PyExecutor:
    """An executor around the real loop, with every phase recording its entry in ``run.events``."""
    executor = PyExecutor.__new__(PyExecutor)
    executor.dist = dist
    executor.dkv_enabled = True
    executor.enable_attention_dp = True
    executor.attention_dp_enable_balance = False
    executor.enable_batch_waiting = False
    executor.is_encoder_decoder = False
    executor.disable_overlap_scheduler = True
    executor.model_engine = SimpleNamespace(
        is_warmup=False, route_capture=None, is_spec_decode=False
    )
    executor.draft_model_engine = None
    executor._is_warmup = False
    executor.device_id = 0
    executor.iter_counter = 0
    executor.active_requests = []
    executor.inflight_req_ids = set()
    executor.canceled_req_ids = []
    executor.waiting_queue = []
    executor.is_shutdown = False
    executor.stream_interval = 1
    executor.force_terminate_ctx_for_partial_reuse = False
    executor.enable_joint_kv_cache_reuse = False
    executor.gather_all_responses = False
    executor.result_wait_queues = {}
    executor._prefetched_request_ids = set()
    executor._disagg_pp_termination_handler = None
    executor._fatal_error = None
    executor._pending_transfer_responses = []
    executor._pending_response_terminations = []
    executor._dkv_forward_dummies = []
    executor.enable_kv_cache_events = False
    executor._dkv_sampler_errors = []
    executor._dkv_fatal_messages = []
    executor._dkv_freed_request_ids = []
    executor._dkv_commit_reason = None
    executor._error_budget = Mock(consume=Mock(return_value=False), budget=5.0)
    executor.enable_iter_perf_stats = False
    executor._resource_governor_enabled = False
    executor._is_kv_manager_v2 = True
    executor._mm_encoder_item_scheduling_enabled = False
    executor.is_benchmark_disagg = False
    executor.kv_connector_manager = None
    executor.guided_decoder = None
    executor.drafter = None
    executor.speculation_gate = None
    executor.dwdp_manager = None
    executor.kv_cache_transceiver = None
    executor._profile_enabled = False
    executor.control_requests = []
    executor.hang_detector = MagicMock()
    executor.perf_manager = MagicMock(
        enabled=False, create_timing_events=Mock(return_value=(None, None, None))
    )
    executor._profiler = Mock(return_value=nullcontext(Mock()))
    executor._disagg_coordinator = Mock()
    executor.async_transfer_manager = Mock(
        requests_in_transfer=Mock(return_value={}),
        has_any_inflight_requests=Mock(return_value=False),
    )
    executor.executor_request_queue = Mock(enqueue_shutdown_request=Mock())
    executor._event_loop_completed = False
    executor._maybe_attach_ctx_usage = Mock()
    executor._maybe_record_hang_diagnostic_phase = Mock()

    pages = LoopPages(pages=script.pages, block_size=4, tp_rank=dist.tp_rank)
    run.pages = pages
    executor.kv_cache_manager = pages
    executor.scheduler = _scheduler(
        pages,
        group_size=dist.tp_size,
        batch_size=script.batch_size,
        tokens=script.max_num_tokens,
        chunked=script.chunked,
    )
    executor.scheduler.dkv_dual_ledger_enabled = script.dual_ledger
    executor.sampler = FinishingSampler(script.fail_ids if dist.tp_rank == 1 else set())
    executor.resource_manager = ResourceManager(
        {
            ResourceManagerType.KV_CACHE_MANAGER: pages,
            ResourceManagerType.SEQ_SLOT_MANAGER: SeqSlotManager(script.batch_size * dist.tp_size),
        }
    )
    executor._dkv_invariant_checker = DkvInvariantChecker(dist, enabled=script.debug)
    executor._initialize_dkv_forward_dummies()
    run.initial_pages = pages.used_pages

    def record(name: str, value=None):
        run.events.append((executor.iter_counter, name, value))

    pages.on_free = lambda request_id: record("free", request_id)
    executor.dkv_layer_split = script.layer_split
    if script.context_only:
        _attach_transceiver(executor, script, run, record)
    if script.streamer:
        executor.dkv_streamer = PlanStreamer(
            executor, dist.tp_size, script.plan_skew, record, script.skip_drain
        )
        executor.dkv_staging_settings = {"transport_timeout_s": 7.0}

    def ids(batch: ScheduledRequests) -> list[tuple[int, bool]]:
        return [(req.py_request_id, req.is_dummy) for req in batch.all_requests()]

    # -- scripted arrivals and the real scheduler -----------------------------------------
    real_schedule = executor._schedule

    def prepare_and_schedule():
        record("schedule")
        if executor.iter_counter >= len(script.arrivals) and not executor.active_requests:
            executor.is_shutdown = True
            return None, None
        arrivals = (
            script.arrivals[executor.iter_counter]
            if executor.iter_counter < len(script.arrivals)
            else []
        )
        for request_id, owner, length in arrivals:
            executor.active_requests.append(
                make_request(
                    request_id,
                    compute_rank=owner,
                    local_rank=dist.tp_rank,
                    prompt_len=length,
                    context_only=request_id in script.context_only,
                )
            )
        batch, _, _ = real_schedule()
        return batch, None

    executor._prepare_and_schedule_batch = prepare_and_schedule

    # -- phases that only record -----------------------------------------------------------
    for name in (
        "_check_benchmark_disagg_gate",
        "_handle_dynamic_draft_len",
        "_commit_kv_cache_stats",
        "_maybe_prefetch_next_iter_mm_encoders",
        "_handle_guided_decoder_errors",
        "_release_unused_connector_reservations",
        "_finalize_adp_dummy_allocation",
        "_kv_connector_terminate_requests",
        "_flush_iter_stats_synced",
        "_revert_gen_alloc",
        "_terminate_recompute_paused_requests",
        "_pause_recompute_paused_requests",
        "_can_pause_for_rebalance",
        "_prepare_disagg_gen_transmission_complete",
    ):
        setattr(executor, name, Mock(return_value=None))
    executor._check_benchmark_disagg_gate = Mock(return_value=(True, False))
    executor._can_pause_for_rebalance = Mock(return_value=False)

    def update_v2_context_resources(batch: ScheduledRequests) -> None:
        # The check that the real method makes before it commits the context.
        executor._check_dkv_data_plane_idle("the context commit")
        record("context_commit")

    executor._update_v2_context_resources = Mock(side_effect=update_v2_context_resources)

    real_forward_batch = executor._dkv_forward_batch

    def forward_batch(batch: ScheduledRequests) -> ScheduledRequests:
        local = real_forward_batch(batch)
        record("forward_batch", ids(local))
        return local

    executor._dkv_forward_batch = forward_batch

    def forward_step(batch: ScheduledRequests):
        record("forward", ids(batch))
        if script.forward_error == (dist.tp_rank, executor.iter_counter):
            raise RuntimeError("injected forward failure")
        return {"logits": torch.zeros(max(batch.batch_size, 1), 1)}

    executor._forward_step = forward_step
    if script.fatal_budget and dist.tp_rank == 1:
        executor._error_budget.consume.return_value = True

    real_prepare = executor.resource_manager.prepare_resources
    real_update = executor.resource_manager.update_resources

    def prepare_resources(batch, *args, **kwargs):
        record("prepare_resources", ids(batch))
        return real_prepare(batch, *args, **kwargs)

    def update_resources(batch, *args, **kwargs):
        record("update_resources", ids(batch))
        if script.leak == (dist.tp_rank, executor.iter_counter):
            pages.kv_cache_map[10_000] = SimpleNamespace(capacity=4, is_active=True)
        return real_update(batch, *args, **kwargs)

    executor.resource_manager.prepare_resources = prepare_resources
    executor.resource_manager.update_resources = update_resources

    real_sync_samples = executor._sync_dkv_samples

    def sync_samples(scheduled, forward):
        record("sample_sync", ids(scheduled))
        real_sync_samples(scheduled, forward)

    executor._sync_dkv_samples = sync_samples
    real_control = executor._sync_dkv_control

    def sync_control():
        record("control")
        real_control()

    executor._sync_dkv_control = sync_control
    real_handle_errors = executor._handle_errors

    def handle_errors(message=None, *args, **kwargs):
        # The coordinator names the message ``error_msg``; the loop passes it by position.
        message = kwargs.pop("error_msg", message)
        requests = kwargs.get("requests")
        run.errors.append(
            (message, tuple(sorted(r.py_request_id for r in requests)) if requests else ())
        )
        record("errors", message)
        return real_handle_errors(message, *args, **kwargs)

    executor._handle_errors = handle_errors

    def enqueue(responses) -> None:
        records = [(request_id, dist.tp_rank) for request_id, _ in responses]
        gathered = dist.tp_gather(records)
        if gathered is not None:
            run.emitted.extend(record_ for rank_records in gathered for record_ in rank_records)

    executor._enqueue_responses = enqueue
    return executor


def run_loop(
    group_size: int, script: LoopScript, *, timeout: float = 20.0, keep_runs: list | None = None
) -> list[RankRun]:
    """Run the real executor loop on ``group_size`` lockstep ranks until the script drains.

    ``keep_runs`` receives the per-rank records even when the loop raises, for diagnosis.
    """
    group = LockstepTpGroup(group_size, timeout=timeout)
    runs = [RankRun() for _ in range(group_size)]
    if keep_runs is not None:
        keep_runs.extend(runs)

    def rank_main(dist) -> None:
        run = runs[dist.tp_rank]
        executor = build_loop_executor(dist, script, run)
        run.executor = executor
        executor._executor_loop()
        run.final_pages = run.pages.used_pages

    with (
        patch(f"{_PY_EXECUTOR}.torch.cuda.set_device", Mock()),
        patch(f"{_PY_EXECUTOR}.cudart.cudaSetDevice", Mock()),
        patch(f"{_PY_EXECUTOR}.CUASSERT", Mock()),
    ):
        group.run(rank_main)
    for run in runs:
        run.group = group
    return runs
