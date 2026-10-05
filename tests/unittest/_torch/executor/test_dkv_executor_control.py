# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Actual executor control, cancellation and free commits across replicated ranks."""

from contextlib import nullcontext
from queue import Queue
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import pytest
from dkv_test_utils import LockstepTpGroup, make_request

from tensorrt_llm._torch.pyexecutor.dkv import DkvInvariantChecker
from tensorrt_llm._torch.pyexecutor.llm_request import (
    FinishReason,
    LlmRequest,
    LlmRequestState,
    LlmRequestType,
    SamplingConfig,
)
from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor
from tensorrt_llm._torch.pyexecutor.resource_manager import ResourceManagerType
from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests

pytestmark = pytest.mark.cpu_only


class _WaitingQueue(list):
    def remove_by_ids(self, request_ids: set[int]) -> None:
        assert not self


def _executor(dist, *, debug: bool = True) -> PyExecutor:
    executor = PyExecutor.__new__(PyExecutor)
    executor.dist = dist
    executor.dkv_enabled = True
    executor.enable_attention_dp = True
    executor.disable_overlap_scheduler = True
    executor.iter_counter = 0
    executor.active_requests = []
    executor._dkv_commit_reason = None
    executor._dkv_freed_request_ids = []
    executor._dkv_fatal_messages = []
    executor._dkv_sampler_errors = []
    executor._dkv_invariant_checker = SimpleNamespace(enabled=debug)
    executor._pending_transfer_responses = []
    executor._pending_response_terminations = []
    executor._fatal_error = None
    executor._error_budget = Mock(consume=Mock(return_value=False), budget=5.0)
    executor.kv_cache_manager = Mock(get_dkv_control_digest=Mock(return_value=(2, ((100,),))))
    executor.resource_manager = Mock()
    executor.kv_connector_manager = None
    executor.kv_cache_transceiver = None
    executor._disagg_pp_termination_handler = None
    executor._disagg_coordinator = Mock()
    executor.async_transfer_manager = Mock(requests_in_transfer=Mock(return_value={}))
    executor._prefetched_request_ids = set()
    executor.waiting_queue = _WaitingQueue()
    executor.executor_request_queue = Mock(get_request_queue=Mock(return_value=Queue()))
    executor.gather_all_responses = False
    executor.result_wait_queues = {}
    executor.stream_interval = 1
    executor.force_terminate_ctx_for_partial_reuse = False
    executor.perf_manager = Mock()
    executor._maybe_attach_ctx_usage = Mock()
    executor.canceled_req_ids = []
    executor.is_shutdown = False
    executor.emitted = []

    def enqueue(responses) -> None:
        records = [
            (request_id, dist.tp_rank, response.error_msg) for request_id, response in responses
        ]
        gathered = dist.tp_gather(records)
        if gathered is not None:
            executor.emitted.extend(record for rank in gathered for record in rank)

    executor._enqueue_responses = enqueue
    return executor


def _batch(requests: list[LlmRequest]) -> ScheduledRequests:
    result = ScheduledRequests()
    for request in requests:
        request.context_chunk_size = request.orig_prompt_len
    result.reset_context_requests(requests)
    return result


def _freed(executor: PyExecutor) -> list[int]:
    return [
        call.args[0].py_request_id
        for call in executor.resource_manager.free_resources.call_args_list
    ]


def _prepare_idle_loop(executor: PyExecutor) -> list[str]:
    executor.device_id = 0
    executor._profiler = Mock(return_value=nullcontext(Mock()))
    executor.hang_detector = MagicMock()
    executor.enable_iter_perf_stats = False
    executor._resource_governor_enabled = False
    executor._is_kv_manager_v2 = False
    executor._mm_encoder_item_scheduling_enabled = False
    executor.is_benchmark_disagg = False
    executor.guided_decoder = None
    executor.drafter = None
    executor.speculation_gate = None
    executor.resource_manager.resource_managers = {}
    executor._check_benchmark_disagg_gate = Mock(return_value=(True, False))
    executor._terminate_requests = Mock()
    executor._pause_requests = Mock()
    executor._revert_gen_alloc = Mock()
    executor._finalize_adp_dummy_allocation = Mock()
    executor._kv_connector_terminate_requests = Mock()
    executor._flush_iter_stats_synced = Mock()
    sequence = []
    control = executor._sync_dkv_control

    def sync() -> None:
        sequence.append("control")
        control()

    def schedule():
        sequence.append("schedule")
        return (ScheduledRequests(), None) if executor.iter_counter == 0 else (None, None)

    executor._sync_dkv_control = sync
    executor.disagg.handle_errors_synced.side_effect = lambda: sequence.append("disagg")
    executor._prepare_and_schedule_batch = schedule
    return sequence


def test_idle_control_does_not_flush_or_add_a_second_collective() -> None:
    group = LockstepTpGroup(2)

    def run(dist) -> None:
        executor = _executor(dist, debug=False)
        for iteration in range(3):
            executor.iter_counter = iteration
            executor._sync_dkv_control()
        assert executor.emitted == []
        assert _freed(executor) == []

    group.run(run)
    assert group.traces[0] == group.traces[1]
    assert [entry[2] for entry in group.traces[0]] == ["tp_allgather"] * 3


@pytest.mark.parametrize("dkv_enabled", [False, True])
def test_main_loop_control_precedes_scheduling_and_removes_trailing_dkv_flush(
    monkeypatch: pytest.MonkeyPatch, dkv_enabled: bool
) -> None:
    module = "tensorrt_llm._torch.pyexecutor.py_executor"
    monkeypatch.setattr(f"{module}.torch.cuda.set_device", Mock())
    monkeypatch.setattr(f"{module}.cudart.cudaSetDevice", Mock())
    monkeypatch.setattr(f"{module}.CUASSERT", Mock())
    group = LockstepTpGroup(2)

    def run(dist) -> None:
        executor = _executor(dist, debug=False)
        executor.dkv_enabled = dkv_enabled
        sequence = _prepare_idle_loop(executor)
        executor._executor_loop()
        assert executor._event_loop_completed
        expected = ["control", "schedule"] if dkv_enabled else ["disagg", "schedule"]
        assert sequence == expected * 2
        if dkv_enabled:
            executor.disagg.handle_errors_synced.assert_not_called()
            executor.disagg.handle_timeouts_synced.assert_not_called()
            executor.disagg.check_transfer_timeouts.assert_not_called()

    group.run(run)
    assert group.traces[0] == group.traces[1]
    kinds = [entry[2] for entry in group.traces[0]]
    assert len(kinds) == 3
    assert kinds.count("tp_gather") == (1 if dkv_enabled else 2)


def test_main_loop_cancels_a_stalled_request_without_a_forward(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = "tensorrt_llm._torch.pyexecutor.py_executor"
    monkeypatch.setattr(f"{module}.torch.cuda.set_device", Mock())
    monkeypatch.setattr(f"{module}.cudart.cudaSetDevice", Mock())
    monkeypatch.setattr(f"{module}.CUASSERT", Mock())
    group = LockstepTpGroup(2)

    def run(dist) -> list:
        executor = _executor(dist, debug=False)
        _prepare_idle_loop(executor)
        executor._forward_step = Mock(side_effect=AssertionError("Unexpected forward"))
        request = make_request(8, compute_rank=1, local_rank=dist.tp_rank)
        executor.active_requests = [request]
        executor.canceled_req_ids = [8]
        executor._executor_loop()
        assert FinishReason.CANCELLED in request.finish_reasons
        assert not executor.active_requests
        assert not executor.canceled_req_ids
        assert _freed(executor) == [8]
        executor._forward_step.assert_not_called()
        return executor.emitted

    assert group.run(run) == [[(8, 1, None)], []]
    assert group.traces[0] == group.traces[1]
    assert [entry[2] for entry in group.traces[0]] == [
        "tp_allgather",
        "tp_gather",
        "tp_allgather",
        "tp_gather",
    ]


def test_healthy_iteration_collectives_do_not_exceed_adp_baseline() -> None:
    group = LockstepTpGroup(2)

    def run(dist) -> None:
        executor = _executor(dist, debug=False)
        request = make_request(8, compute_rank=1, local_rank=dist.tp_rank)
        executor.active_requests = [request]
        global_batch = _batch([request])
        forward = global_batch.local_view()
        if not request.py_dkv_is_local:
            forward.generation_requests = [make_request(0, is_dummy=True)]
        executor._sync_dkv_control()
        assert executor._can_queue(global_batch, forward) == (True, True)
        executor._update_request_states(global_batch)
        if request.py_dkv_is_local:
            request.add_new_token(42, 0)
            request.py_decoding_iter = 1
            request.finish_by_reason(FinishReason.LENGTH)
        executor._sync_dkv_samples(global_batch, forward)
        executor._handle_responses()
        assert _freed(executor) == [8]

    group.run(run)
    assert group.traces[0] == group.traces[1]
    # ADP has queue allgather, response gather and an unconditional trailing flush gather.
    assert [entry[2] for entry in group.traces[0]] == ["tp_allgather", "tp_allgather", "tp_gather"]


def test_rank_one_sampler_failure_is_deferred_then_freed_once_on_both_ranks() -> None:
    group = LockstepTpGroup(2)

    def run(dist) -> dict:
        executor = _executor(dist)
        request = make_request(8, compute_rank=1, local_rank=dist.tp_rank)
        executor.active_requests = [request]
        global_batch = _batch([request])
        forward = global_batch.local_view()
        if dist.tp_rank == 1:
            executor._stage_dkv_sampler_error("injected sampler failure", [request])
        executor._sync_dkv_samples(global_batch, forward)
        assert global_batch.batch_size == forward.batch_size == 0
        assert not executor.active_requests
        assert _freed(executor) == []
        assert len(executor._pending_response_terminations) == 1
        assert len(executor._pending_transfer_responses) == dist.tp_rank
        executor.iter_counter += 1
        executor._sync_dkv_control()
        assert _freed(executor) == [8]
        assert not executor._pending_transfer_responses
        assert not executor._pending_response_terminations
        assert not executor.is_shutdown
        executor.iter_counter += 1
        executor._sync_dkv_control()
        assert _freed(executor) == [8]
        assert not executor._dkv_freed_request_ids
        return {
            "emitted": executor.emitted,
            "budget_calls": executor._error_budget.consume.call_count,
        }

    result = group.run(run)
    assert result[0]["emitted"] == [(8, 1, "injected sampler failure")]
    assert result[1]["emitted"] == []
    assert [rank["budget_calls"] for rank in result] == [0, 1]
    assert group.traces[0] == group.traces[1]


def test_rank_local_fatal_budget_shuts_down_every_rank_at_control() -> None:
    group = LockstepTpGroup(2)

    def run(dist) -> dict:
        executor = _executor(dist)
        request = make_request(8, compute_rank=1, local_rank=dist.tp_rank)
        executor.active_requests = [request]
        global_batch = _batch([request])
        forward = global_batch.local_view()
        if dist.tp_rank == 1:
            executor._error_budget.consume.return_value = True
            executor._stage_dkv_sampler_error("fatal sampler budget", [request])
        executor._sync_dkv_samples(global_batch, forward)
        assert executor._fatal_error is None
        assert not executor.is_shutdown
        executor._sync_dkv_control()
        assert executor.is_shutdown
        assert isinstance(executor._fatal_error, RuntimeError)
        assert "rank 1: fatal sampler budget" in str(executor._fatal_error)
        assert _freed(executor) == [8]
        executor.executor_request_queue.enqueue_shutdown_request.assert_called_once()
        return {
            "emitted": executor.emitted,
            "budget_calls": executor._error_budget.consume.call_count,
        }

    result = group.run(run)
    assert len(result[0]["emitted"]) == 1
    assert result[0]["emitted"][0][:2] == (8, 1)
    assert [rank["budget_calls"] for rank in result] == [0, 1]
    assert group.traces[0] == group.traces[1]


@pytest.mark.parametrize("method", ["_terminate_request", "_free_request_resources"])
def test_free_outside_commit_window_is_rejected_before_resources_mutate(method: str) -> None:
    executor = _executor(LockstepTpGroup(1).ranks[0])
    with pytest.raises(RuntimeError, match="outside.*commit window"):
        getattr(executor, method)(make_request(8))
    assert _freed(executor) == []
    assert executor._dkv_freed_request_ids == []


@pytest.mark.parametrize("reason", ["activation", "responses", "control", "fatal"])
def test_each_commit_window_records_real_frees_and_restores_scope(reason: str) -> None:
    executor = _executor(LockstepTpGroup(1).ranks[0])
    with executor._dkv_commit_window(reason):
        executor._terminate_request(make_request(8))
    assert executor._dkv_commit_reason is None
    assert executor._dkv_freed_request_ids == [8]
    assert _freed(executor) == [8]


def test_different_free_sets_are_detected_without_extra_failure_collective() -> None:
    group = LockstepTpGroup(2)

    def run(dist) -> None:
        executor = _executor(dist)
        executor._dkv_freed_request_ids = [8 + dist.tp_rank]
        executor._sync_dkv_control()

    with pytest.raises(RuntimeError, match="freed request IDs differ"):
        group.run(run)
    assert group.traces[0] == group.traces[1]
    assert len(group.traces[0]) == 1


def test_capacity_divergence_is_checked_without_debug() -> None:
    group = LockstepTpGroup(2)

    def run(dist) -> None:
        executor = _executor(dist, debug=False)
        executor.kv_cache_manager.get_dkv_control_digest.return_value = (
            2,
            ((100 + dist.tp_rank,),),
        )
        executor._sync_dkv_control()

    with pytest.raises(RuntimeError, match="capacity digest differs"):
        group.run(run)
    assert group.traces[0] == group.traces[1]


def _iteration_end_executor(dist, *, enabled: bool = True, **changes) -> PyExecutor:
    """Executor whose replicated end-of-iteration state can differ per rank."""
    executor = _executor(dist)
    executor._dkv_invariant_checker = DkvInvariantChecker(dist, enabled=enabled)
    fingerprint = changes.get("kv_state", [("pool", 4, 2)])
    executor.kv_cache_manager = SimpleNamespace(get_dkv_state_fingerprint=lambda: fingerprint)
    in_transfer = changes.get("in_transfer", ())
    executor.kv_cache_transceiver = object() if in_transfer else None
    executor.async_transfer_manager = Mock(
        requests_in_transfer=Mock(return_value={request_id: None for request_id in in_transfer})
    )
    executor.active_requests = changes.get("active", [make_request(8), make_request(9)])
    executor._dkv_freed_request_ids = list(changes.get("freed", ()))
    return executor


def test_iteration_end_check_passes_when_replicated_state_agrees() -> None:
    group = LockstepTpGroup(2)

    def run(dist) -> None:
        executor = _iteration_end_executor(dist, in_transfer=(5, 3), freed=(8,))
        executor._check_dkv_iteration_end(include_freed=True)

    group.run(run)
    assert [len(trace) for trace in group.traces] == [2, 2]


@pytest.mark.parametrize(
    "divergence, tag",
    [
        ("kv_state", "end of iteration kv state"),
        ("in_transfer", "end of iteration in transfer"),
        ("active", "end of iteration active"),
    ],
)
def test_iteration_end_check_reports_the_diverging_record(divergence: str, tag: str) -> None:
    group = LockstepTpGroup(2)

    def run(dist) -> None:
        changes = {}
        if divergence == "kv_state":
            changes["kv_state"] = [("pool", 4, 2 + dist.tp_rank)]
        elif divergence == "in_transfer":
            changes["in_transfer"] = (5,) if dist.tp_rank else (5, 6)
        executor = _iteration_end_executor(dist, **changes)
        if divergence == "active":
            executor.active_requests[1].context_current_position = dist.tp_rank
        executor._check_dkv_iteration_end()

    with pytest.raises(RuntimeError, match="DKV invariant violation") as caught:
        group.run(run)
    assert f"rank1 differs from rank0: tag {tag}" in str(caught.value)


def test_iteration_end_check_includes_freed_requests_only_when_requested() -> None:
    def run(include_freed: bool):
        def per_rank(dist) -> None:
            executor = _iteration_end_executor(dist, freed=(8 + dist.tp_rank,))
            executor._check_dkv_iteration_end(include_freed=include_freed)

        return per_rank

    LockstepTpGroup(2).run(run(False))
    with pytest.raises(RuntimeError, match="end of iteration freed"):
        LockstepTpGroup(2).run(run(True))


def test_iteration_end_check_is_free_when_debug_is_disabled() -> None:
    group = LockstepTpGroup(2)

    def run(dist) -> None:
        executor = _iteration_end_executor(dist, enabled=False, kv_state=[("pool", dist.tp_rank)])
        executor._check_dkv_iteration_end(include_freed=True)

    group.run(run)
    assert [len(trace) for trace in group.traces] == [1, 1]


def _context_transfer(rank: int) -> LlmRequest:
    request = LlmRequest(
        request_id=8,
        input_tokens=list(range(8)),
        max_new_tokens=1,
        sampling_config=SamplingConfig(1),
        is_streaming=False,
        llm_request_type=LlmRequestType.LLMREQUEST_TYPE_CONTEXT_ONLY,
    )
    request.py_dkv_compute_rank = 1
    request.py_dkv_is_local = rank == 1
    request.state = LlmRequestState.DISAGG_CONTEXT_TRANS_IN_PROGRESS
    return request


@pytest.mark.parametrize("in_active", [False, True])
def test_context_transfer_cancellation_stays_pending_without_local_transceiver_calls(
    in_active: bool,
) -> None:
    group = LockstepTpGroup(2)

    def run(dist) -> tuple:
        executor = _executor(dist)
        request = _context_transfer(dist.tp_rank)
        executor.active_requests = [request] if in_active else []
        executor.async_transfer_manager.requests_in_transfer.return_value = {8: request}
        executor.kv_cache_transceiver = Mock()
        executor.kv_cache_transceiver.has_inflight_transfer.return_value = dist.tp_rank == 1
        executor.canceled_req_ids = [8]
        assert executor._is_request_in_transmission(request)
        executor._handle_canceled_requests()
        executor.kv_cache_transceiver.has_inflight_transfer.assert_not_called()
        executor.disagg.request_cancellation.assert_not_called()
        assert _freed(executor) == []
        return executor.canceled_req_ids, request.state

    states = group.run(run)
    assert states == [([8], LlmRequestState.DISAGG_CONTEXT_TRANS_IN_PROGRESS)] * 2


def test_aggregate_cancel_marks_both_replicas_and_response_commit_frees_them() -> None:
    group = LockstepTpGroup(2)

    def run(dist) -> dict:
        executor = _executor(dist)
        request = make_request(8, compute_rank=1, local_rank=dist.tp_rank)
        executor.active_requests = [request]
        executor.canceled_req_ids = [8]
        executor._handle_canceled_requests()
        assert not executor.canceled_req_ids
        assert FinishReason.CANCELLED in request.finish_reasons
        assert _freed(executor) == []
        executor._handle_responses()
        executor._sync_dkv_control()
        assert _freed(executor) == [8]
        assert not executor.active_requests
        return {"emitted": executor.emitted}

    result = group.run(run)
    assert result[0]["emitted"] == [(8, 1, None)]
    assert result[1]["emitted"] == []
    assert group.traces[0] == group.traces[1]


_UNREPLICATED_LOOP_FEATURES = {
    "benchmark disaggregation": ("is_benchmark_disagg", True),
    "multimodal encoder scheduling": ("_mm_encoder_item_scheduling_enabled", True),
    "a KV cache connector": ("kv_connector_manager", object()),
    "guided decoding": ("guided_decoder", object()),
    "a drafter": ("drafter", object()),
    "a speculation gate": ("speculation_gate", object()),
}


def _plain_loop_executor() -> PyExecutor:
    executor = _executor(LockstepTpGroup(1).ranks[0])
    executor.resource_manager = SimpleNamespace(resource_managers={})
    executor.is_benchmark_disagg = False
    executor._mm_encoder_item_scheduling_enabled = False
    executor.kv_connector_manager = None
    executor.guided_decoder = None
    executor.drafter = None
    executor.speculation_gate = None
    return executor


def test_loop_accepts_a_plain_dkv_configuration() -> None:
    _plain_loop_executor()._validate_dkv_loop_features()


@pytest.mark.parametrize("feature", list(_UNREPLICATED_LOOP_FEATURES))
def test_loop_rejects_features_whose_control_flow_is_not_replicated(feature: str) -> None:
    executor = _plain_loop_executor()
    attribute, value = _UNREPLICATED_LOOP_FEATURES[feature]
    setattr(executor, attribute, value)
    with pytest.raises(RuntimeError, match=f"not supported with {feature}$"):
        executor._validate_dkv_loop_features()


def test_loop_rejects_a_speculative_resource_manager_and_names_every_feature() -> None:
    executor = _plain_loop_executor()
    executor.resource_manager = SimpleNamespace(
        resource_managers={ResourceManagerType.SPEC_RESOURCE_MANAGER: object()}
    )
    executor.drafter = object()
    with pytest.raises(RuntimeError) as caught:
        executor._validate_dkv_loop_features()
    assert "a drafter, a speculative-decoding resource manager" in str(caught.value)
