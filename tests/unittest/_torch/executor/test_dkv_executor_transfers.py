# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Executor transfer commits preserve response ownership and replicated frees."""

from contextlib import nullcontext
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from dkv_test_utils import LockstepTpGroup

from tensorrt_llm._torch.disaggregation.kv_cache_transceiver import CtxTransferStatus
from tensorrt_llm._torch.disaggregation.orchestration.transfer_manager import AsyncTransferManager
from tensorrt_llm._torch.disaggregation.transceiver import KvCacheTransceiverV2
from tensorrt_llm._torch.pyexecutor.disagg_adapter import PyExecutorEffects
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
from tensorrt_llm._torch.pyexecutor.llm_request import (
    LlmRequest,
    LlmRequestState,
    LlmRequestType,
    SamplingConfig,
)
from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor
from tensorrt_llm._torch.pyexecutor.resource_manager import ResourceManagerType
from tensorrt_llm._torch.pyexecutor.scheduler import FCFSWaitingQueue
from tensorrt_llm.mapping import Mapping

pytestmark = pytest.mark.cpu_only


def _request(rank: int) -> LlmRequest:
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
    request.context_current_position = request.orig_prompt_len
    request.state = LlmRequestState.GENERATION_IN_PROGRESS
    return request


def _executor(dist) -> PyExecutor:
    executor = PyExecutor.__new__(PyExecutor)
    executor.dist = dist
    executor.dkv_enabled = True
    executor.enable_attention_dp = True
    executor.disable_overlap_scheduler = True
    executor.iter_counter = 0
    executor.active_requests = []
    executor.canceled_req_ids = []
    executor._dkv_commit_reason = None
    executor._dkv_freed_request_ids = []
    executor._dkv_fatal_messages = []
    executor._dkv_invariant_checker = SimpleNamespace(enabled=True)
    executor._pending_transfer_responses = []
    executor._pending_response_terminations = []
    executor._fatal_error = None
    executor._error_budget = Mock()
    executor.kv_cache_manager = Mock(get_dkv_control_digest=Mock(return_value=(0, ((100,),))))
    executor.kv_cache_manager.store_blocks_for_reuse.return_value = 8
    executor.resource_manager = Mock()
    executor.resource_manager.resource_managers = {
        ResourceManagerType.KV_CACHE_MANAGER: executor.kv_cache_manager
    }
    executor.async_transfer_manager = AsyncTransferManager(executor.resource_manager)
    executor.kv_cache_transceiver = Mock(
        kv_transfer_timeout_ms=60000,
        pipeline_transfer_enabled=False,
        supports_inflight_cancellation=False,
        check_context_transfer_status=Mock(return_value=CtxTransferStatus([], [])),
        has_retired_send_session=Mock(return_value=False),
    )
    executor.kv_connector_manager = None
    executor._disagg_pp_termination_handler = None
    executor._prefetched_request_ids = set()
    executor.gather_all_responses = False
    executor.result_wait_queues = {8: object()}
    executor.force_terminate_ctx_for_partial_reuse = False
    executor.perf_manager = Mock()
    executor.stream_interval = 1
    executor._maybe_attach_ctx_usage = Mock()
    executor.emitted = []
    executor.freed = []

    def enqueue(responses) -> None:
        records = [(request_id, dist.tp_rank) for request_id, _ in responses]
        gathered = dist.tp_allgather(records)
        executor.emitted.extend(record for rank in gathered for record in rank)

    def free(request: LlmRequest) -> None:
        assert executor._dkv_commit_reason == "control"
        executor.freed.append((request.py_request_id, executor.iter_counter))

    executor._enqueue_responses = Mock(side_effect=enqueue)
    executor.resource_manager.free_resources.side_effect = free
    return executor


def test_transfer_completion_waits_for_next_control_then_flushes_every_rank() -> None:
    group = LockstepTpGroup(2)

    def run(dist) -> None:
        executor = _executor(dist)
        request = _request(dist.tp_rank)
        request.create_response = Mock(
            return_value=SimpleNamespace(result=SimpleNamespace(cached_tokens=0))
        )
        executor.active_requests = [request]
        executor._send_kv_async([request])
        executor.kv_cache_manager.release_index_slot.assert_called_once_with(8)
        assert executor.kv_cache_transceiver.respond_and_send_async.call_count == dist.tp_rank
        assert executor.async_transfer_manager.requests_in_transfer() == {8: request}
        if dist.tp_rank == 1:
            executor.kv_cache_transceiver.check_context_transfer_status.return_value = (
                CtxTransferStatus([8], [])
            )
        executor.disagg.reap_context_sends(0)
        assert executor.freed == []
        executor.kv_cache_manager.unpin_blocks_by_id.assert_not_called()

        executor.iter_counter = 1
        executor._sync_dkv_control()
        assert executor.freed == [(8, 1)]
        assert executor.emitted == [(8, 1)]
        assert executor.active_requests == []
        assert executor.async_transfer_manager.requests_in_transfer() == {}
        executor.kv_cache_manager.unpin_blocks_by_id.assert_called_once_with(8)
        assert request.create_response.call_count == dist.tp_rank
        assert executor._pending_response_terminations == []
        assert executor._pending_transfer_responses == []
        if dist.tp_rank == 0:
            executor.kv_cache_transceiver.check_context_transfer_status.assert_not_called()

        executor.iter_counter = 2
        executor._sync_dkv_control()
        assert executor.freed == [(8, 1)]
        executor._enqueue_responses.assert_called_once()

    group.run(run)
    assert group.traces[0] == group.traces[1]
    assert [entry[2] for entry in group.traces[0]] == ["tp_allgather"] * 3


def test_idle_control_polls_inflight_context_sends_without_a_forward() -> None:
    def run(dist) -> None:
        executor = _executor(dist)
        request = _request(dist.tp_rank)
        executor.active_requests = [request]
        executor.async_transfer_manager.start_transfer(request)
        for iteration in range(3):
            executor.iter_counter = iteration
            executor._sync_dkv_control()
        assert executor.freed == []
        assert executor.active_requests == [request]
        assert executor.kv_cache_transceiver.check_context_transfer_status.call_count == (
            3 if dist.tp_rank == 1 else 0
        )
        executor._enqueue_responses.assert_not_called()

    LockstepTpGroup(2).run(run)


@pytest.mark.parametrize("dkv_enabled", [False, True])
@pytest.mark.parametrize("transfer_pending", [False, True])
def test_fetch_does_not_block_control_progress_for_dkv_transfers(
    dkv_enabled: bool, transfer_pending: bool
) -> None:
    def run(dist) -> None:
        executor = _executor(dist)
        executor.dkv_enabled = dkv_enabled
        executor.is_shutdown = False
        executor._disable_mpi = False
        executor.control_requests = []
        executor.request_accumulated = []
        executor.hang_detector = SimpleNamespace(pause=nullcontext)
        executor.executor_request_queue = Mock(get_from_request_queue=Mock(return_value=[]))
        executor.request_broadcaster = Mock()
        executor._handle_special_queue_items = lambda items: items
        waiting_queue = FCFSWaitingQueue()
        if transfer_pending:
            executor.async_transfer_manager.start_transfer(_request(dist.tp_rank))

        def broadcast(items):
            gathered = dist.tp_allgather(items)
            return gathered[0], None

        executor.request_broadcaster.broadcast.side_effect = broadcast
        executor._fetch_and_enqueue_requests(waiting_queue, total_num_live_requests=0)
        if dist.tp_rank == 0:
            expected_timeout = timedelta(0) if dkv_enabled and transfer_pending else None
            executor.executor_request_queue.get_from_request_queue.assert_called_once_with(
                expected_timeout
            )
        else:
            executor.executor_request_queue.get_from_request_queue.assert_not_called()
        assert not waiting_queue

    LockstepTpGroup(2).run(run)


def test_shutdown_waits_for_departed_context_transfers_to_commit() -> None:
    def run(dist) -> None:
        executor = _executor(dist)
        executor.is_shutdown = True
        executor.waiting_queue = FCFSWaitingQueue()
        request = _request(dist.tp_rank)
        executor.async_transfer_manager.start_transfer(request)
        assert executor.active_requests == []
        assert not executor.should_stop_processing
        if dist.tp_rank == 1:
            executor.kv_cache_transceiver.check_context_transfer_status.return_value = (
                CtxTransferStatus([8], [])
            )
        executor._sync_dkv_control()
        assert executor.freed == [(8, 0)]
        assert executor.should_stop_processing

    LockstepTpGroup(2).run(run)


def test_fatal_control_preserves_pages_until_inflight_fabric_is_stopped() -> None:
    def run(dist) -> None:
        executor = _executor(dist)
        request = _request(dist.tp_rank)
        executor.active_requests = [request]
        executor.async_transfer_manager.start_transfer(request)
        executor._handle_errors = Mock()
        if dist.tp_rank == 1:
            executor._dkv_fatal_messages = ["fatal sampler failure"]
        with pytest.raises(RuntimeError, match="Fatal DKV error: rank 1: fatal sampler failure"):
            executor._sync_dkv_control()
        executor._handle_errors.assert_not_called()
        executor._enqueue_responses.assert_not_called()
        executor.kv_cache_manager.unpin_blocks_by_id.assert_not_called()
        assert executor.freed == []
        assert executor.async_transfer_manager.requests_in_transfer() == {8: request}
        assert executor.active_requests == [request]
        assert executor._dkv_commit_reason is None

    LockstepTpGroup(2).run(run)


@pytest.mark.parametrize("owner_timed_out", [False, True])
def test_response_pass_ignores_local_transfer_observations(owner_timed_out: bool) -> None:
    def run(dist) -> None:
        executor = _executor(dist)
        request = _request(dist.tp_rank)
        executor.active_requests = [request]
        executor.async_transfer_manager.start_transfer(request)
        request.create_response = Mock(return_value=None)
        request.py_kv_transfer_timed_out = owner_timed_out and request.py_dkv_is_local
        executor.disagg.fail_timed_out = Mock()
        executor.disagg.request_cancellation = Mock()
        assert executor._handle_responses() == []
        assert executor.active_requests == []
        assert executor.freed == []
        assert executor.async_transfer_manager.requests_in_transfer() == {8: request}
        executor.disagg.request_cancellation.assert_not_called()
        executor.disagg.fail_timed_out.assert_not_called()
        assert request.create_response.call_count == dist.tp_rank

    LockstepTpGroup(2).run(run)


def test_adapter_can_defer_replica_termination_without_a_response() -> None:
    executor = PyExecutor.__new__(PyExecutor)
    executor._pending_transfer_responses = []
    executor._pending_response_terminations = []
    executor._terminate_request = Mock()
    request = _request(0)
    PyExecutorEffects(executor).stage_transfer_response(8, None, request)
    assert executor._pending_transfer_responses == []
    assert executor._pending_response_terminations == [request]
    executor._terminate_request.assert_not_called()


@pytest.mark.parametrize("admission_failed", [False, True])
def test_transfer_error_emits_once_before_or_after_context_response(admission_failed: bool) -> None:
    def run(dist) -> None:
        executor = _executor(dist)
        request = _request(dist.tp_rank)
        executor.active_requests = [request]
        if admission_failed:
            executor.kv_cache_transceiver.respond_and_send_async.side_effect = RuntimeError(
                "send admission failed"
            )
            executor.kv_cache_transceiver.cancel_request.return_value = True
            # Native context metadata is absent when admission throws.
            request.create_response = Mock(wraps=request.create_response)
        else:
            request.create_response = Mock(
                return_value=SimpleNamespace(result=SimpleNamespace(cached_tokens=0))
            )
        executor._send_kv_async([request])
        assert executor._handle_responses() == []
        assert executor.active_requests == []
        assert executor.freed == []
        assert request.py_dkv_context_response_sent == (
            not admission_failed and request.py_dkv_is_local
        )
        if admission_failed:
            request.create_response.assert_not_called()
            assert executor.emitted == []
        else:
            assert executor.emitted == [(8, 1)]
            if dist.tp_rank == 1:
                executor.kv_cache_transceiver.check_context_transfer_status.return_value = (
                    CtxTransferStatus([], [8])
                )
        executor.iter_counter = 1
        executor._sync_dkv_control()
        assert executor.freed == [(8, 1)]
        assert executor.emitted == [(8, 1)]
        assert executor.async_transfer_manager.requests_in_transfer() == {}
        executor._error_budget.consume.assert_not_called()
        assert request.state == LlmRequestState.GENERATION_COMPLETE

    LockstepTpGroup(2).run(run)


@pytest.mark.parametrize("timed_out", [False, True])
def test_reaped_failure_before_context_publication_sends_only_an_error(timed_out: bool) -> None:
    def run(dist) -> None:
        executor = _executor(dist)
        request = _request(dist.tp_rank)
        request.create_response = Mock(wraps=request.create_response)
        executor.active_requests = [request]
        if request.py_dkv_is_local:
            request.py_kv_transfer_timed_out = timed_out
            executor.kv_cache_transceiver.check_context_transfer_status.return_value = (
                CtxTransferStatus([], [8])
            )
        executor._send_kv_async([request])
        assert executor._handle_responses() == []
        assert executor.active_requests == []
        request.create_response.assert_not_called()
        assert not request.py_dkv_context_response_sent
        assert executor.emitted == []
        assert executor.freed == []
        executor.iter_counter = 1
        executor._sync_dkv_control()
        assert executor.emitted == [(8, 1)]
        assert executor.freed == [(8, 1)]
        assert executor.async_transfer_manager.requests_in_transfer() == {}
        assert request.state == LlmRequestState.GENERATION_COMPLETE

    LockstepTpGroup(2).run(run)


def _runtime_executor() -> PyExecutor:
    executor = PyExecutor.__new__(PyExecutor)
    executor.kv_cache_manager = KVCacheManagerV2.__new__(KVCacheManagerV2)
    executor.kv_cache_manager.enable_block_reuse = True
    executor.kv_cache_transceiver = KvCacheTransceiverV2.__new__(KvCacheTransceiverV2)
    executor.kv_cache_transceiver._enable_pipelined_transfer = False
    executor.kv_cache_transceiver._ctx_need_tp_sync = False
    executor.kv_cache_transceiver._ctx_need_pp_sync = False
    executor.kv_cache_transceiver._fp4_mla_bridge_enabled = False
    executor.kv_cache_transceiver.kv_transfer_timeout_ms = 60000
    executor.is_encoder_decoder = False
    executor.is_benchmark_disagg = False
    executor._mm_encoder_item_scheduling_enabled = False
    executor.attention_dp_enable_balance = False
    executor.dist = SimpleNamespace(
        mapping=Mapping(world_size=2, tp_size=2, moe_ep_size=2, enable_attention_dp=True)
    )
    return executor


def test_runtime_accepts_python_v2_without_status_collectives() -> None:
    _runtime_executor()._validate_dkv_runtime()


@pytest.mark.parametrize(
    "attribute,value,message",
    [
        ("_enable_pipelined_transfer", True, "Pipelined transfer"),
        ("_ctx_need_tp_sync", True, "must not use TP or PP collectives"),
        ("_ctx_need_pp_sync", True, "must not use TP or PP collectives"),
        ("_fp4_mla_bridge_enabled", True, "FP4 MLA transfer bridge"),
        ("kv_transfer_timeout_ms", None, "finite positive"),
        ("kv_transfer_timeout_ms", 0, "finite positive"),
    ],
)
def test_runtime_rejects_unsupported_transceiver_features(
    attribute: str, value: object, message: str
) -> None:
    executor = _runtime_executor()
    setattr(executor.kv_cache_transceiver, attribute, value)
    with pytest.raises(ValueError, match=message):
        executor._validate_dkv_runtime()
