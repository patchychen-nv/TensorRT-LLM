# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Replicated context transfer ownership and physical-retirement contracts."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from tensorrt_llm._torch.disaggregation.kv_cache_transceiver import CtxTransferStatus
from tensorrt_llm._torch.disaggregation.orchestration.coordinator import DisaggTransferCoordinator
from tensorrt_llm._torch.disaggregation.orchestration.interfaces import DkvTransferEvent
from tensorrt_llm._torch.disaggregation.orchestration.transfer_manager import AsyncTransferManager
from tensorrt_llm._torch.pyexecutor.llm_request import (
    FinishReason,
    LlmRequest,
    LlmRequestState,
    SamplingConfig,
)
from tensorrt_llm._torch.pyexecutor.resource_manager import ResourceManagerType
from tensorrt_llm.bindings.internal.batch_manager import LlmRequestType

pytestmark = pytest.mark.cpu_only


class _Registry:
    def __init__(self):
        self.active = []
        self.canceled = []

    def active_requests(self):
        return self.active

    def contains(self, request):
        return request in self.active

    def remove(self, request):
        self.active.remove(request)

    def canceled_request_ids(self):
        return self.canceled


def _request(request_id=7, owner=1):
    request = Mock()
    request.py_request_id = request_id
    request.request_id = request_id
    request.py_disaggregated_params = None
    request.py_dkv_compute_rank = owner
    request.is_dummy = False
    request.is_child = False
    request.is_context_only_request = True
    request.is_disagg_generation_transmission_in_progress = False
    request.is_finished_due_to_cancellation = False
    request.is_context_finished = True
    request.is_finished_due_to_length = False
    request.state = LlmRequestState.GENERATION_IN_PROGRESS
    request.py_kv_transfer_start_time = None
    request.py_kv_transfer_timed_out = False
    request.cached_tokens = 0
    request.create_response.return_value = SimpleNamespace(result=SimpleNamespace())
    return request


def _rank(rank, requests):
    registry = _Registry()
    registry.active = requests
    kv = Mock()
    kv.store_blocks_for_reuse.side_effect = lambda request, pin: request.py_request_id
    seq = Mock()
    transfers = AsyncTransferManager(
        SimpleNamespace(
            resource_managers={
                ResourceManagerType.KV_CACHE_MANAGER: kv,
                ResourceManagerType.SEQ_SLOT_MANAGER: seq,
            }
        )
    )
    transceiver = Mock()
    transceiver.kv_transfer_timeout_ms = 1000
    transceiver.check_context_transfer_status.return_value = CtxTransferStatus([], [])
    transceiver.has_retired_send_session.return_value = False
    transceiver.cancel_request.return_value = True
    effects = Mock()
    dist = Mock(rank=rank, tp_rank=rank, tp_size=2, pp_size=1, world_size=2)
    for name in ("tp_allgather", "tp_allreduce", "allreduce", "allgather"):
        getattr(dist, name).side_effect = AssertionError("Unexpected transfer collective")
    draft = Mock()
    coordinator = DisaggTransferCoordinator(
        transceiver=transceiver,
        transfer_manager=transfers,
        kv_cache_manager=kv,
        draft_kv_cache_manager=draft,
        dist=dist,
        effects=effects,
        registry=registry,
        enable_attention_dp=True,
        force_terminate_ctx_for_partial_reuse=False,
        is_kv_manager_v2=True,
        is_local=lambda request: request.py_dkv_compute_rank == rank,
    )
    return SimpleNamespace(
        coordinator=coordinator,
        registry=registry,
        kv=kv,
        draft=draft,
        seq=seq,
        transfers=transfers,
        transceiver=transceiver,
        effects=effects,
    )


def test_both_replicas_pin_and_release_index_but_only_owner_sends():
    for rank in range(2):
        request = _request()
        worker = _rank(rank, [request])
        worker.coordinator.send_completed_context([request])
        worker.coordinator.send_completed_context([request])

        worker.kv.release_index_slot.assert_called_once_with(7)
        worker.draft.release_index_slot.assert_called_once_with(7)
        worker.seq.free_resources.assert_called_once_with(request)
        worker.kv.store_blocks_for_reuse.assert_called_once_with(request, True)
        assert list(worker.transfers.requests_in_transfer()) == [7]
        assert request.state == LlmRequestState.DISAGG_CONTEXT_TRANS_IN_PROGRESS
        assert worker.transceiver.respond_and_send_async.call_count == rank
        assert (request.py_kv_transfer_start_time is not None) == bool(rank)


@pytest.mark.parametrize("reason", ["failed", "removed", "canceled", "dummy", "unfinished"])
def test_failed_and_ineligible_contexts_never_release_an_index(reason):
    request = _request()
    worker = _rank(1, [request])
    if reason == "failed":
        request.state = LlmRequestState.DISAGG_TRANS_ERROR
    elif reason == "removed":
        worker.registry.remove(request)
        request.state = LlmRequestState.GENERATION_COMPLETE
    elif reason == "canceled":
        worker.registry.canceled.append(7)
    elif reason == "dummy":
        request.is_dummy = True
    else:
        request.is_context_finished = False
    worker.coordinator.send_completed_context([request])
    worker.kv.release_index_slot.assert_not_called()
    worker.transceiver.respond_and_send_async.assert_not_called()
    assert not worker.transfers.has_any_inflight_requests()


def _native_context_request():
    request = LlmRequest(
        request_id=7,
        input_tokens=list(range(8)),
        max_new_tokens=1,
        sampling_config=SamplingConfig(1),
        is_streaming=False,
        llm_request_type=LlmRequestType.LLMREQUEST_TYPE_CONTEXT_ONLY,
    )
    request.py_dkv_compute_rank = 1
    request.context_chunk_size = 8
    request.move_to_next_context_chunk()
    request.state = LlmRequestState.GENERATION_IN_PROGRESS
    return request


def test_length_completion_enters_transfer_even_in_generation_complete_state():
    request = _native_context_request()
    request.add_new_token(42, 0)
    request.finish_by(FinishReason.LENGTH, 0)
    assert request.state == LlmRequestState.GENERATION_COMPLETE
    assert request.is_finished_due_to_length
    worker = _rank(1, [request])
    worker.coordinator.send_completed_context([request])
    assert list(worker.transfers.requests_in_transfer()) == [7]
    worker.transceiver.respond_and_send_async.assert_called_once_with(request)


def test_owner_observation_does_not_release_until_shared_commit():
    workers = [_rank(rank, [_request()]) for rank in range(2)]
    for worker in workers:
        worker.coordinator.send_completed_context(worker.registry.active)
    workers[1].transceiver.check_context_transfer_status.return_value = CtxTransferStatus([7], [])
    for worker in workers:
        worker.coordinator.reap_context_sends()
        worker.kv.unpin_blocks_by_id.assert_not_called()
        worker.effects.stage_transfer_response.assert_not_called()
        assert list(worker.transfers.requests_in_transfer()) == [7]
    workers[0].transceiver.check_context_transfer_status.assert_not_called()
    events = workers[1].coordinator.collect_dkv_transfer_events()
    assert events == (DkvTransferEvent(7, 1, "completed"),)
    for rank, worker in enumerate(workers):
        request = worker.registry.active[0]
        assert worker.coordinator.commit_dkv_transfer_events(events)
        worker.kv.unpin_blocks_by_id.assert_called_once_with(7)
        assert not worker.transfers.has_any_inflight_requests()
        assert request.create_response.call_count == rank
        response = request.create_response.return_value if rank else None
        worker.effects.stage_transfer_response.assert_called_once_with(7, response, request)
        worker.effects.terminate_request.assert_not_called()


@pytest.mark.parametrize("outcome", ["failed", "timed_out"])
@pytest.mark.parametrize("still_active", [False, True])
def test_failure_releases_every_replica_even_after_context_response(outcome, still_active):
    for rank in range(2):
        request = _request()
        worker = _rank(rank, [request])
        worker.coordinator.send_completed_context([request])
        if not still_active:
            worker.registry.remove(request)
        event = DkvTransferEvent(7, 1, outcome, "native failure")
        assert worker.coordinator.commit_dkv_transfer_events([event])
        assert not worker.transfers.has_any_inflight_requests()
        worker.kv.unpin_blocks_by_id.assert_called_once_with(7)
        request.create_response.assert_not_called()
        worker.effects.fail_requests.assert_called_once_with(
            "native failure", [request], charge_budget=False
        )


def test_wallclock_timeout_cannot_release_while_native_reader_is_active():
    request = _request()
    worker = _rank(1, [request])
    with patch(
        "tensorrt_llm._torch.disaggregation.orchestration.coordinator.time.monotonic",
        return_value=10,
    ):
        worker.coordinator.send_completed_context([request])
    worker.transceiver.cancel_request.side_effect = [False, False, True]
    with patch(
        "tensorrt_llm._torch.disaggregation.orchestration.coordinator.time.monotonic",
        return_value=12,
    ):
        for _ in range(2):
            assert worker.coordinator.collect_dkv_transfer_events() == ()
            assert list(worker.transfers.requests_in_transfer()) == [7]
            worker.kv.unpin_blocks_by_id.assert_not_called()
        events = worker.coordinator.collect_dkv_transfer_events()
    assert len(events) == 1 and events[0].outcome == "timed_out"
    worker.kv.unpin_blocks_by_id.assert_not_called()
    worker.coordinator.commit_dkv_transfer_events(events)
    worker.kv.unpin_blocks_by_id.assert_called_once_with(7)


def test_replica_does_not_poll_or_cancel_based_on_local_clock():
    request = _request()
    worker = _rank(0, [request])
    worker.coordinator.send_completed_context([request])
    request.py_kv_transfer_start_time = 0
    assert worker.coordinator.collect_dkv_transfer_events() == ()
    assert not request.py_kv_transfer_timed_out
    worker.transceiver.cancel_request.assert_not_called()
    worker.transceiver.check_context_transfer_status.assert_not_called()


@pytest.mark.parametrize("admission_failure", ["exception", "rejected"])
def test_send_admission_failure_is_staged_only_after_native_retirement(admission_failure):
    request = _request()
    worker = _rank(1, [request])
    if admission_failure == "exception":
        worker.transceiver.respond_and_send_async.side_effect = RuntimeError("submit failed")
    else:

        def reject(req):
            req.state = LlmRequestState.DISAGG_TRANS_ERROR

        worker.transceiver.respond_and_send_async.side_effect = reject
    worker.coordinator.send_completed_context([request])
    assert request.state == LlmRequestState.DISAGG_CONTEXT_TRANS_IN_PROGRESS
    assert worker.coordinator.dkv_context_send_failed(request)
    worker.transceiver.cancel_request.side_effect = [False, True]
    assert worker.coordinator.collect_dkv_transfer_events() == ()
    events = worker.coordinator.collect_dkv_transfer_events()
    assert len(events) == 1 and events[0].outcome == "failed"
    worker.kv.unpin_blocks_by_id.assert_not_called()
    worker.effects.fail_requests.assert_not_called()


def test_status_ids_are_normalized_to_executor_request_ids():
    request = _request()
    request.py_disaggregated_params = SimpleNamespace(disagg_request_id=123, ctx_usage=None)
    worker = _rank(1, [request])
    worker.coordinator.send_completed_context([request])
    worker.transceiver.check_context_transfer_status.return_value = CtxTransferStatus([123], [])
    assert worker.coordinator.collect_dkv_transfer_events() == (
        DkvTransferEvent(7, 1, "completed"),
    )


def test_remote_cancellation_is_a_retired_failure_even_without_status_id():
    request = _request()
    worker = _rank(1, [request])
    worker.coordinator.send_completed_context([request])
    worker.transceiver.has_retired_send_session.return_value = True
    events = worker.coordinator.collect_dkv_transfer_events()
    assert len(events) == 1 and events[0].outcome == "failed"
    assert worker.coordinator.dkv_context_send_failed(request)


@pytest.mark.parametrize("outcome", ["completed", "failed", "timed_out"])
def test_terminal_observation_before_publication_suppresses_only_failed_response(outcome):
    request = _native_context_request()
    worker = _rank(1, [request])
    worker.coordinator.send_completed_context([request])
    if outcome == "failed":
        worker.transceiver.check_context_transfer_status.return_value = CtxTransferStatus([], [7])
    elif outcome == "timed_out":
        request.py_kv_transfer_timed_out = True
    else:
        worker.transceiver.check_context_transfer_status.return_value = CtxTransferStatus([7], [])

    worker.coordinator.reap_context_sends()

    assert worker.coordinator.dkv_context_send_failed(request) == (outcome != "completed")
    assert request.state == LlmRequestState.DISAGG_CONTEXT_TRANS_IN_PROGRESS
    worker.kv.unpin_blocks_by_id.assert_not_called()


def test_commit_orders_releases_by_request_id():
    requests = [_request(9), _request(2)]
    worker = _rank(0, requests)
    worker.coordinator.send_completed_context(requests)
    worker.coordinator.commit_dkv_transfer_events(
        [DkvTransferEvent(9, 1, "completed"), DkvTransferEvent(2, 1, "completed")]
    )
    assert [call.args[0] for call in worker.kv.unpin_blocks_by_id.call_args_list] == [2, 9]
    assert [call.args[0] for call in worker.effects.stage_transfer_response.call_args_list] == [
        2,
        9,
    ]


@pytest.mark.parametrize("malformed", ["unknown", "owner", "duplicate", "state", "outcome"])
def test_invalid_event_is_rejected_before_any_release(malformed):
    request = _request()
    worker = _rank(0, [request])
    worker.coordinator.send_completed_context([request])
    events = [DkvTransferEvent(7, 1, "completed")]
    if malformed == "unknown":
        events.append(DkvTransferEvent(8, 1, "completed"))
    elif malformed == "owner":
        events = [DkvTransferEvent(7, 0, "completed")]
    elif malformed == "duplicate":
        events *= 2
    elif malformed == "state":
        request.state = LlmRequestState.GENERATION_COMPLETE
    else:
        events = [DkvTransferEvent(7, 1, "invalid")]
    with pytest.raises(RuntimeError, match="Invalid DKV transfer event"):
        worker.coordinator.commit_dkv_transfer_events(events)
    worker.kv.unpin_blocks_by_id.assert_not_called()
    assert list(worker.transfers.requests_in_transfer()) == [7]


def test_legacy_error_and_timeout_votes_are_disabled_for_dkv():
    worker = _rank(0, [])
    worker.coordinator.handle_errors_synced()
    worker.coordinator.handle_timeouts_synced()
    worker.effects.fail_requests.assert_not_called()
