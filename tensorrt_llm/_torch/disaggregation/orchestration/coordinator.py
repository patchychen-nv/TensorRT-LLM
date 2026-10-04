# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Executor-facing entry points for disaggregated KV transfer.

The executor loops call the disagg state machine only through a
``DisaggTransferCoordinator``. This module must not import ``PyExecutor`` or
hold a reference to it: services are injected, executor-owned behavior is
reached through the ``interfaces`` Protocols.
"""

import os
import time
import traceback
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Sequence, Set, Tuple

from tensorrt_llm._torch.disaggregation.base.transfer import get_unique_rid
from tensorrt_llm._torch.disaggregation.kv_cache_transceiver import (
    is_disagg_inflight_cancel_enabled,
)
from tensorrt_llm._torch.distributed.communicator import ReduceOp
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest, LlmRequestState
from tensorrt_llm._utils import nvtx_range
from tensorrt_llm.disaggregated_params import DisaggScheduleStyle
from tensorrt_llm.logger import logger

from .admission import DisaggTransferAdmissionController
from .interfaces import ActiveRequestRegistry, DkvTransferEvent, ExecutorEffects

if TYPE_CHECKING:
    from tensorrt_llm._torch.pyexecutor.scheduler.scheduler import ScheduledRequests


def is_gen_only_no_context_benchmark() -> bool:
    """Whether the ``gen_only_no_context`` benchmark skips KV transfer."""
    return os.getenv("TRTLLM_DISAGG_BENCHMARK_GEN_ONLY") == "1"


def uses_async_gen_transfer() -> bool:
    """Whether generation KV transfers can remain in flight across iterations."""
    return (
        not is_gen_only_no_context_benchmark()
        and os.getenv("TRTLLM_DISABLE_KV_CACHE_TRANSFER_OVERLAP") != "1"
    )


def transfer_window_bypass_eligible(transceiver, dist, is_kv_manager_v2: bool) -> bool:
    """Whether this runtime may skip the executor-level transfer window.

    ``max_tokens_in_buffer`` describes the C++ transceiver's physical buffer.
    The asynchronous Python transceiver does not consume it, and with KV cache
    manager V2 and PP1 its generation requests stay bounded by the scheduler's
    inline KV admission, so no second budget is applied. Other configurations
    keep the window.
    """
    return (
        transceiver is not None
        and transceiver.consumes_transfer_buffer is False
        and uses_async_gen_transfer()
        and dist.pp_size == 1
        and is_kv_manager_v2
    )


def attach_ctx_usage(request: LlmRequest, response) -> None:
    """Copy gen-first context usage from the transfer aux data onto the response."""
    disagg_params = request.py_disaggregated_params
    if disagg_params is not None and disagg_params.ctx_usage is not None:
        response.result.ctx_usage = disagg_params.ctx_usage


class DisaggTransferCoordinator:
    """Disagg transfer entry points used by every executor loop variant.

    Several methods perform rank-consensus collectives inside the transceiver
    or over ``dist``; every rank must call them the same number of times per
    iteration. The loops therefore call them unconditionally and rely on the
    coordinator (or ``NoopDisaggCoordinator``) to be a no-op when
    disaggregation is off.
    """

    def __init__(
        self,
        *,
        transceiver,
        transfer_manager,
        kv_cache_manager,
        dist,
        effects: ExecutorEffects,
        registry: ActiveRequestRegistry,
        enable_attention_dp: bool,
        force_terminate_ctx_for_partial_reuse: bool,
        draft_kv_cache_manager=None,
        admission_controller: Optional[DisaggTransferAdmissionController] = None,
        is_kv_manager_v2: bool = False,
        is_local: Optional[Callable[[LlmRequest], bool]] = None,
    ) -> None:
        self._transceiver = transceiver
        self._transfers = transfer_manager
        self._kv_cache_manager = kv_cache_manager
        self._draft_kv_cache_manager = draft_kv_cache_manager
        self._dist = dist
        self._effects = effects
        self._registry = registry
        self._enable_attention_dp = enable_attention_dp
        self._force_terminate_ctx_for_partial_reuse = force_terminate_ctx_for_partial_reuse
        # Transfer-window budget; None or a disabled controller means no window.
        self._admission_controller = admission_controller
        self._is_kv_manager_v2 = is_kv_manager_v2
        self._is_local = is_local
        self._dkv_transfer_events: Dict[int, DkvTransferEvent] = {}
        self._dkv_send_errors: Dict[int, str] = {}
        self._dkv_fatal_messages: List[str] = []
        self._dkv_stalled_request_ids: Set[int] = set()
        # Context sends that failed after leaving the transfer manager; applied
        # at the next rank-synchronized error pass.
        self._pending_ctx_transfer_failures: Set[int] = set()
        # Timed-out generation requests whose error response waits for the
        # ADP-safe consensus point.
        self._pending_timed_out_requests: List[LlmRequest] = []
        # Requests whose timed-out transfer was already cancelled in flight.
        self._timed_out_ctx_cancelled_ids: Set[int] = set()
        self._timed_out_gen_cancelled_ids: Set[int] = set()
        self._inflight_cancel_unsupported_logged = False

    # -- loop head -----------------------------------------------------------

    @nvtx_range("handle_errors_synced")
    def handle_errors_synced(self) -> None:
        """Rank-safe disagg cache error and poison handler.

        Called from the top of every executor iteration. Buffer poison is
        reduced over the full executor world because one poisoned PP/DP rank
        requires the whole distributed executor to stop. ADP TP ranks then
        vote on failed request IDs and fail matching local replicas together;
        otherwise the downstream ``tp_gather`` in ``_enqueue_responses``
        deadlocks or leaves peer replicas running.
        """
        if self._is_local is not None:
            return
        pending_ids = self.take_pending_context_failures()
        pending_requests = (
            [req for req in self._registry.active_requests() if get_unique_rid(req) in pending_ids]
            if pending_ids
            else []
        )
        for request in pending_requests:
            request.state = LlmRequestState.DISAGG_TRANS_ERROR

        if self.inflight_cancel_active():
            local_poisoned = self._transceiver.has_poisoned_transfer_buffer()
            if self._dist.world_size != 1:
                any_poisoned = bool(self._dist.allreduce(int(local_poisoned), op=ReduceOp.MAX))
            else:
                any_poisoned = local_poisoned
            if any_poisoned:
                self._effects.fail_fatal(
                    "Disagg KV cache transfer buffer is poisoned; process restart is required"
                )
                return

        if not (self._enable_attention_dp and self._dist.world_size != 1):
            if pending_requests:
                self._check_transfer_errors("context requests")
            return

        local_error_requests = [
            req
            for req in self._registry.active_requests()
            if req.state == LlmRequestState.DISAGG_TRANS_ERROR
        ]
        local_vote = {
            "error_ids": [self._vote_id(req) for req in local_error_requests],
            "blocked_ids": [
                self._vote_id(req)
                for req in local_error_requests
                if self._is_error_cleanup_blocked(req)
            ],
        }
        all_votes = self._dist.tp_allgather(local_vote)
        voted_error_ids = {rid for vote in all_votes for rid in vote["error_ids"]}
        blocked_error_ids = {rid for vote in all_votes for rid in vote["blocked_ids"]}
        ready_error_ids = voted_error_ids - blocked_error_ids
        if not ready_error_ids:
            return
        local_voted_error_requests = [
            req for req in self._registry.active_requests() if self._vote_id(req) in ready_error_ids
        ]
        logger.warning(
            f"Disagg KV cache transfer error: rank={self._dist.rank} "
            f"local_err_count={len(local_error_requests)}, "
            f"voted_err_count={len(voted_error_ids)}, "
            f"blocked_err_count={len(voted_error_ids & blocked_error_ids)}"
        )
        self._effects.fail_requests(
            "Disagg KV cache transfer error", local_voted_error_requests, charge_budget=False
        )

    @nvtx_range("prepare_context_schedulable")
    def prepare_context_schedulable(self, new_requests: List[LlmRequest]) -> None:
        """Let the transceiver gate generation-first context requests.

        Context-first context requests are schedulable at once; for
        generation-first ones the transceiver decides when the peer is ready.
        """
        gen_first_ctx_requests = [
            req
            for req in new_requests
            if req.is_context_only_request
            and req.py_disaggregated_params.schedule_style == DisaggScheduleStyle.GENERATION_FIRST
        ]
        # Always call prepare_context_requests, with new requests or without,
        # so the consensus inside it can promote requests whose peer info has
        # arrived on every rank.
        self._transceiver.prepare_context_requests(gen_first_ctx_requests)

    @nvtx_range("poll_gen_transfers")
    def poll_gen_transfers(self) -> None:
        """Poll receive-side transfers and their timeouts; rank-synchronized."""
        if not uses_async_gen_transfer():
            return
        # Gen-transfer status performs cross-rank consensus internally. Enter
        # it symmetrically; ranks with no ready local future contribute an
        # empty ready set.
        self.reap_gen_receives(0)
        if self.inflight_cancel_active():
            self._cancel_timed_out_gen_transfers()
            self._check_gen_transfer_errors_consensus()

    @nvtx_range("check_transfer_timeouts")
    def check_transfer_timeouts(self, only_with_context_sends: bool = False) -> None:
        """Flag transfers that exceeded ``kv_transfer_timeout_ms``.

        ``only_with_context_sends`` keeps the post-batch call sites gated on an
        in-flight context send, as they were before the extraction.
        """
        if only_with_context_sends and not self._transfers.has_any_inflight_requests():
            return
        timeout_ms = self._transceiver.kv_transfer_timeout_ms
        if timeout_ms is None:
            return

        def flag_if_timed_out(req: LlmRequest, kind: str) -> None:
            if req.py_kv_transfer_start_time is None:
                return
            elapsed_ms = (time.monotonic() - req.py_kv_transfer_start_time) * 1000
            if elapsed_ms > timeout_ms and not req.py_kv_transfer_timed_out:
                verb = (
                    "Requesting cancellation for"
                    if self.inflight_cancel_active()
                    else "Observed timeout on"
                )
                logger.warning(
                    f"{verb} {kind} request {req.py_request_id} due to KV cache "
                    f"transfer timeout: elapsed {elapsed_ms:.0f}ms > "
                    f"kv_transfer_timeout_ms={timeout_ms}ms"
                )
                req.py_kv_transfer_timed_out = True

        # Context requests start their clock on the last chunk, which is also
        # when they enter the transfer manager, so this covers the whole
        # context side.
        for req in self._transfers.requests_in_transfer().values():
            if self._is_local is not None and not self._is_local(req):
                continue
            flag_if_timed_out(req, "context")
        for req in self._registry.active_requests():
            if req.is_disagg_generation_transmission_in_progress:
                flag_if_timed_out(req, "generation")

    # -- scheduling ----------------------------------------------------------

    def admit(self, fitting_gen_init: List[LlmRequest]) -> Tuple[List[LlmRequest], bool]:
        """Select the gen-init requests that may start receiving this iteration.

        Returns ``(admitted, blocked_by_active_transfers)``.
        """
        # gen_only_no_context has no CTX worker and does not transfer data.
        # Real synchronous gen_only transfers still honor the budget to bound
        # the number of blocking transfers started in one executor iteration.
        if is_gen_only_no_context_benchmark():
            return fitting_gen_init, False

        if not (self._transfer_window_is_active() and fitting_gen_init):
            return fitting_gen_init, False

        controller = self._admission_controller
        admission_result = controller.select(self._registry.active_requests(), fitting_gen_init)
        if admission_result.deferred_request_count > 0:
            logger.debug(
                "Disagg transfer admission deferred "
                f"{admission_result.deferred_request_count} requests; "
                f"active transfer blocks={admission_result.active_transfer_blocks}, "
                f"admitted transfer blocks={admission_result.admitted_transfer_blocks}, "
                f"budget={controller.max_transfer_blocks}"
            )

        self.revert_deferred_gen_init(fitting_gen_init, admission_result.admitted_requests)

        return (
            admission_result.admitted_requests,
            admission_result.is_blocked_by_active_transfers(),
        )

    def revert_deferred_gen_init(
        self, candidates: List[LlmRequest], admitted: List[LlmRequest]
    ) -> None:
        """Release KV allocated for candidates that were not admitted.

        Scheduler V2 allocates KV while evaluating generation-init requests.
        This reconciliation is required both after transfer-window admission
        and when a PP follower's local candidates differ from the canonical
        schedule propagated by rank 0.
        """
        if not (self._is_kv_manager_v2 and candidates):
            return

        admitted_request_ids = {request.py_request_id for request in admitted}
        deferred_requests = [
            request for request in candidates if request.py_request_id not in admitted_request_ids
        ]
        if deferred_requests:
            self._effects.revert_ctx_alloc(deferred_requests)

    def _transfer_window_is_active(self) -> bool:
        """Whether the executor-level transfer window bounds admission."""
        return (
            self._admission_controller is not None
            and self._admission_controller.enabled()
            and not transfer_window_bypass_eligible(
                self._transceiver, self._dist, self._is_kv_manager_v2
            )
        )

    @nvtx_range("receive_gen_init")
    def receive_gen_init(self, admitted: List[LlmRequest]) -> None:
        """Prepare executor resources for the admitted gen-init requests, then
        start their KV receive in the configured transfer mode."""
        if not admitted:
            return
        self._effects.prepare_gen_resources(admitted)

        # gen_only_no_context has no CTX worker, so mark each request as
        # transmission-complete immediately.
        if is_gen_only_no_context_benchmark():
            for req in admitted:
                req.state = LlmRequestState.DISAGG_GENERATION_TRANS_COMPLETE
            return

        if not uses_async_gen_transfer():
            # Resources have already been prepared for every request in this
            # batch. Drain all synchronous receives even after one fails so no
            # prepared request is left in DISAGG_GENERATION_INIT.
            for req in admitted:
                self._transceiver.request_and_receive_sync(req)
            self._check_transfer_errors("generation requests")
            return

        for req in admitted:
            self._transceiver.request_and_receive_async(req)

        if self._transceiver.kv_transfer_timeout_ms is not None:
            for req in admitted:
                if req.state == LlmRequestState.DISAGG_GENERATION_TRANS_IN_PROGRESS:
                    req.py_kv_transfer_start_time = time.monotonic()

        self.reap_gen_receives(0)

    def poll_progress_when_idle(self) -> None:
        """Reap completed context KV transfers so their blocks can be freed.

        A synchronous GEN receive blocks rank-locally, so a multi-rank worker
        must not enter the context status collective here. A single-rank
        worker cannot diverge and polls only while a send is in flight.
        """
        uses_synchronous_gen_transfer = (
            not uses_async_gen_transfer() and not is_gen_only_no_context_benchmark()
        )
        should_poll_synchronous_context_status = (
            uses_synchronous_gen_transfer
            and self._dist.world_size == 1
            and self._transfers.has_any_inflight_requests()
        )
        if uses_synchronous_gen_transfer and not should_poll_synchronous_context_status:
            return

        self.reap_context_sends(0)

    # -- batch execution -----------------------------------------------------

    def completed_gen_receives(self, scheduled_batch: "ScheduledRequests") -> List[LlmRequest]:
        """Generation requests in the batch whose KV receive has completed.

        Read-only: the executor prepares batch-level resources for these
        requests while they are still transmission-complete, then calls
        ``try_finish_gen_receive`` for each request in the batch.
        """
        return [
            req
            for req in scheduled_batch.generation_requests
            if req.is_disagg_generation_transmission_complete
        ]

    def try_finish_gen_receive(self, request: LlmRequest) -> bool:
        """Close out a completed KV receive and hand the request to generation.

        Returns False, touching nothing, when the request is no longer
        transmission-complete (batch-level preparation may have failed and
        terminated it). Errors from the transfer side propagate.
        """
        if not request.is_disagg_generation_transmission_complete:
            return False
        request.state = LlmRequestState.GENERATION_IN_PROGRESS
        # commit_blocks_for_reuse requires the position to be at the prompt end.
        request.context_current_position = request.prompt_len
        self._transceiver.commit_blocks_for_reuse(request)
        request.py_kv_transfer_start_time = None
        request.py_kv_transfer_timed_out = False
        return True

    def send_completed_context(self, requests: List[LlmRequest]) -> None:
        """Start async KV sends for finished context-only requests."""
        if self._is_local is not None:
            self._send_dkv_context(requests)
            return
        # Do not send more chunks after an in-flight cancellation.
        cancel_pending_ids = set(self._registry.canceled_request_ids())
        bridge_enabled = getattr(self._transceiver, "_fp4_mla_bridge_enabled", False) is True
        for req in requests:
            if not req.is_context_only_request or req.is_finished_due_to_cancellation:
                continue
            request_id = req.parent_request_id if req.is_child else req.py_request_id
            if request_id in cancel_pending_ids:
                continue
            if self._transceiver.has_retired_send_session(req):
                # The peer registration went away with the session, so no
                # further slice can land.
                continue
            if req.is_context_finished or req.is_finished_due_to_length:
                # Forward is done: release the IndexMapper slot on every KV
                # manager that has one so new requests can reuse it. KV blocks
                # stay allocated for the transfer.
                for manager in (self._kv_cache_manager, self._draft_kv_cache_manager):
                    if hasattr(manager, "release_index_slot"):
                        manager.release_index_slot(req.py_request_id)
                # start_transfer commits the request's blocks to the reuse tree
                # and pins them; it must run before respond_and_send_async
                # sends the final slice and (for the Python transceiver) moves
                # the request toward completion.
                self._transfers.start_transfer(req)
                self._transceiver.respond_and_send_async(req)
                # Bridge validation can reject before a transfer session exists.
                # Release the claim right away: there is no physical accessor
                # whose retirement the reap could poll.
                if (
                    bridge_enabled
                    and req.state == LlmRequestState.DISAGG_TRANS_ERROR
                    and not self._transceiver.has_inflight_transfer(req)
                ):
                    self.release_transfer(req)
                    continue
                if self._transceiver.kv_transfer_timeout_ms is not None:
                    req.py_kv_transfer_start_time = time.monotonic()
            elif (
                self._transceiver.pipeline_transfer_enabled
                and req.state != LlmRequestState.GENERATION_COMPLETE
            ):
                # Intermediate chunk of a pipelined transfer. GENERATION_COMPLETE
                # means an error path already failed and freed this request, so
                # its chunk bounds are unset.
                self._transceiver.respond_and_send_async(req)

    def _send_dkv_context(self, requests: List[LlmRequest]) -> None:
        canceled_ids = set(self._registry.canceled_request_ids())
        in_transfer = self._transfers.requests_in_transfer()
        for request in requests:
            if (
                request.is_dummy
                or not request.is_context_only_request
                or request.is_finished_due_to_cancellation
                or not self._registry.contains(request)
                or request.state == LlmRequestState.DISAGG_TRANS_ERROR
                or self._vote_id(request) in canceled_ids
                or request.py_request_id in in_transfer
                or not (request.is_context_finished or request.is_finished_due_to_length)
            ):
                continue
            for manager in (self._kv_cache_manager, self._draft_kv_cache_manager):
                if hasattr(manager, "release_index_slot"):
                    manager.release_index_slot(request.py_request_id)
            self._transfers.start_transfer(request)
            if not self._is_local(request):
                continue
            request.py_kv_transfer_start_time = time.monotonic()
            try:
                self._transceiver.respond_and_send_async(request)
            except Exception as error:
                # A backend can raise after submitting a physical read. The
                # claim stays pinned until polling or cancellation retires it.
                self._dkv_send_errors[request.py_request_id] = str(error)
            if request.state == LlmRequestState.DISAGG_TRANS_ERROR:
                self._dkv_send_errors.setdefault(
                    request.py_request_id, "Context KV transfer was rejected"
                )
            # Native admission failures are local observations until S-control.
            request.state = LlmRequestState.DISAGG_CONTEXT_TRANS_IN_PROGRESS

    def _stage_dkv_transfer_event(
        self, request: LlmRequest, *, reported_completed: bool = False
    ) -> None:
        """Stage the terminal outcome of one local send.

        A completion reported by the transport wins over the local timeout flag: the pages were
        delivered, so a deadline that expired in the same poll must not fail the request.
        """
        request_id = request.py_request_id
        if request_id in self._dkv_transfer_events:
            return
        error_message = self._dkv_send_errors.get(request_id, "")
        if reported_completed and not error_message:
            outcome = "completed"
        elif request.py_kv_transfer_timed_out:
            outcome = "timed_out"
            error_message = f"Request {request_id} timed out (KV transfer)"
        else:
            outcome = "failed"
            error_message = error_message or "Context KV transfer failed"
        self._dkv_transfer_events[request_id] = DkvTransferEvent(
            request_id, request.py_dkv_compute_rank, outcome, error_message
        )

    def _poll_dkv_context_sends(self) -> None:
        local_requests = {
            request.py_request_id: request
            for request in self._transfers.requests_in_transfer().values()
            if self._is_local(request)
        }
        if not local_requests:
            return
        # V2 attention-DP context polling has no TP/PP collective. Only a
        # compute rank owns a send session; terminal results guarantee that
        # physical writers no longer reference the request's pages.
        status = self._transceiver.check_context_transfer_status(0)
        by_transfer_id = {get_unique_rid(request): request for request in local_requests.values()}
        failed_ids = set(status.error_request_ids)
        for transfer_id in sorted(set(status.completed_request_ids) | failed_ids):
            request = by_transfer_id.get(transfer_id)
            if request is None:
                raise RuntimeError(f"DKV transfer status has unknown request {transfer_id}")
            self._stage_dkv_transfer_event(
                request, reported_completed=transfer_id not in failed_ids
            )

        for request_id, request in sorted(local_requests.items()):
            if request_id in self._dkv_transfer_events:
                continue
            if request.py_kv_transfer_timed_out or request_id in self._dkv_send_errors:
                # A timeout alone does not establish physical quiescence.
                if self.request_cancellation(request):
                    self._stage_dkv_transfer_event(request)
            elif self._transceiver.has_retired_send_session(request):
                # V2 also retires remote cancellations without a completion ID.
                self._stage_dkv_transfer_event(request)
        for request in local_requests.values():
            request.state = LlmRequestState.DISAGG_CONTEXT_TRANS_IN_PROGRESS

    def _stalled_dkv_transfers(self) -> List[str]:
        """Describe local sends still unresolved after twice the transfer timeout.

        The first timeout requests cancellation; a send that stays unresolved for as long again
        cannot be released, because the fabric may still read its pages. Each stall is reported
        once.
        """
        timeout_ms = self._transceiver.kv_transfer_timeout_ms
        if timeout_ms is None:
            return []
        now = time.monotonic()
        stalled = []
        for request_id, request in sorted(self._transfers.requests_in_transfer().items()):
            if (
                not self._is_local(request)
                or request_id in self._dkv_transfer_events
                or request_id in self._dkv_stalled_request_ids
                or request.py_kv_transfer_start_time is None
            ):
                continue
            elapsed_ms = (now - request.py_kv_transfer_start_time) * 1000
            if elapsed_ms > 2 * timeout_ms:
                self._dkv_stalled_request_ids.add(request_id)
                stalled.append(
                    f"KV transfer of request {request_id} has no terminal result after "
                    f"{elapsed_ms:.0f}ms (kv_transfer_timeout_ms={timeout_ms}ms)"
                )
        return stalled

    def collect_dkv_transfer_events(self) -> Tuple[DkvTransferEvent, ...]:
        """Observe local sends without releasing any replicated resources.

        A failure while polling only happens on a compute rank, so it is not raised here: the
        executor publishes it through ``take_dkv_fatal_messages`` and every rank stops together.
        """
        if self._is_local is None:
            return ()
        try:
            self.check_transfer_timeouts()
            self._poll_dkv_context_sends()
            self._dkv_fatal_messages.extend(self._stalled_dkv_transfers())
        except Exception as error:
            logger.error(f"DKV context transfer polling failed: {error}")
            logger.error(traceback.format_exc())
            self._dkv_fatal_messages.append(f"DKV context transfer polling failed: {error}")
        return tuple(self._dkv_transfer_events[key] for key in sorted(self._dkv_transfer_events))

    def take_dkv_fatal_messages(self) -> Tuple[str, ...]:
        """Drain the fatal conditions found by the DKV transfer path on this rank."""
        messages = tuple(self._dkv_fatal_messages)
        self._dkv_fatal_messages.clear()
        return messages

    def dkv_context_send_failed(self, request: LlmRequest) -> bool:
        """Whether a local failure makes the context success response unsafe."""
        event = self._dkv_transfer_events.get(request.py_request_id)
        return request.py_request_id in self._dkv_send_errors or (
            event is not None and event.outcome != "completed"
        )

    def commit_dkv_transfer_events(self, events: Sequence[DkvTransferEvent]) -> bool:
        """Release agreed transfers in ID order inside the S-control window.

        Returns whether the executor must collectively flush staged responses
        and terminations before it schedules another batch.

        Response creation runs on the compute rank only. An exception raised there is not
        replicated: it ends the event loop of that rank and the rank-crash hard kill stops the
        group, which is the accepted outcome for a fault that cannot be agreed on beforehand.
        """
        if self._is_local is None:
            if events:
                raise RuntimeError("DKV transfer events require a DKV coordinator")
            return False
        requests = self._transfers.requests_in_transfer()
        ordered = sorted(events, key=lambda event: event.request_id)
        seen = set()
        for event in ordered:
            request = requests.get(event.request_id)
            if (
                event.request_id in seen
                or request is None
                or request.py_dkv_compute_rank != event.compute_rank
                or request.state != LlmRequestState.DISAGG_CONTEXT_TRANS_IN_PROGRESS
                or event.outcome not in ("completed", "failed", "timed_out")
            ):
                raise RuntimeError(f"Invalid DKV transfer event: {event}")
            seen.add(event.request_id)
        needs_flush = False
        for event in ordered:
            request = requests[event.request_id]
            self._dkv_transfer_events.pop(event.request_id, None)
            self._dkv_send_errors.pop(event.request_id, None)
            request.py_kv_transfer_start_time = None
            request.py_kv_transfer_timed_out = event.outcome == "timed_out"
            if event.outcome != "completed":
                request.state = LlmRequestState.DISAGG_TRANS_ERROR
                if not self._transfers.end_transfer(request):
                    raise RuntimeError("DKV context transfer has an unexpected extra owner")
                self._effects.fail_requests(
                    event.error_message or "Context KV transfer failed",
                    [request],
                    charge_budget=False,
                )
                needs_flush = True
                continue
            response = None
            if self._registry.contains(request) and self._is_local(request):
                response = request.create_response(False, self._dist.rank)
                if response:
                    response.result.cached_tokens = request.cached_tokens
                    attach_ctx_usage(request, response)
            if not self._transfers.end_transfer(request):
                raise RuntimeError("DKV context transfer has an unexpected extra owner")
            if self._registry.contains(request):
                self._registry.remove(request)
            self._effects.stage_transfer_response(event.request_id, response, request)
            needs_flush = True
        return needs_flush

    @nvtx_range("reap_context_sends")
    def reap_context_sends(self, at_least: int = 0) -> None:
        """Poll send-side transfers and release settled requests."""
        if self._is_local is not None:
            self._poll_dkv_context_sends()
            return
        ctx_status = self._transceiver.check_context_transfer_status(at_least)
        failed_req_ids = set(ctx_status.error_request_ids)
        completed_req_ids = set(ctx_status.completed_request_ids) | failed_req_ids

        requests_in_transfer = self._transfers.requests_in_transfer()
        for request_id in completed_req_ids:
            if request_id not in requests_in_transfer:
                if request_id in failed_req_ids:
                    self._pending_ctx_transfer_failures.add(request_id)
                else:
                    logger.warning(f"Request {request_id} not found in transfer manager")
                continue
            request = requests_in_transfer[request_id]
            if request_id in failed_req_ids:
                # Past the context phase: writing the error state here is safe.
                request.state = LlmRequestState.DISAGG_TRANS_ERROR
            self.release_transfer(request)

        # Releases above may have changed the set of requests in transfer.
        requests_in_transfer = self._transfers.requests_in_transfer()
        for request_id in list(requests_in_transfer.keys()):
            request = requests_in_transfer[request_id]
            if (
                not request.py_kv_transfer_timed_out
                or request_id in completed_req_ids
                or request_id in self._timed_out_ctx_cancelled_ids
            ):
                continue
            if not self.request_cancellation(request):
                continue
            if self.inflight_cancel_active():
                self._timed_out_ctx_cancelled_ids.add(request_id)
                logger.warning(
                    f"Cancelled timed-out context KV transfer for request "
                    f"{request.py_request_id}; waiting for C++ transfer status "
                    "to report final cleanup"
                )
            else:
                # Legacy timeout behavior: a queued transfer that can be
                # cancelled is released from the async manager immediately.
                request.py_kv_transfer_start_time = None
                request.state = LlmRequestState.DISAGG_CONTEXT_COMPLETE
                self.release_transfer(request)

        self._check_transfer_errors("context requests")

    @nvtx_range("reap_gen_receives")
    def reap_gen_receives(self, at_least: int = 0) -> None:
        """Poll receive-side transfers; rank-synchronized inside the transceiver."""
        gen_status = self._transceiver.check_gen_transfer_status(at_least)
        if gen_status.cancelled_requests:
            user_canceled_ids = set(self._registry.canceled_request_ids())
            for req in gen_status.cancelled_requests:
                req_id = req.parent_request_id if req.is_child else req.py_request_id
                if req_id not in user_canceled_ids:
                    req.state = LlmRequestState.DISAGG_TRANS_ERROR
        if not self.inflight_cancel_active():
            self._check_transfer_errors("generation requests")

    def release_transfer(self, request: LlmRequest) -> None:
        """Release one transfer claim and terminate once the last owner releases.

        The transceiver and the KV connector can both hold a claim on the same
        request; ``AsyncTransferManager`` counts them. A request that is still
        active gets its response created here (the transfer completed before
        the response pass could run) and staged for the rank-synchronized
        flush; a failed transfer only releases its claim so the synchronized
        error path can respond once every owner is done.
        """
        transfer_failed = request.state == LlmRequestState.DISAGG_TRANS_ERROR
        if self._registry.contains(request):
            if transfer_failed:
                self._transfers.end_transfer(request)
                return
            # Create the response while the state is still TRANS_IN_PROGRESS
            # (required by C++ createResult).
            response = request.create_response(False, self._dist.rank)
            if response:
                response.result.cached_tokens = request.cached_tokens
                attach_ctx_usage(request, response)
            released = self._transfers.end_transfer(request)
            if released:
                self._registry.remove(request)
            if response:
                self._effects.stage_transfer_response(
                    request.py_request_id, response, request if released else None
                )
            elif released:
                self._effects.terminate_request(request)
            return
        if self._transfers.end_transfer(request):
            if transfer_failed:
                return
            # Skip if the PP=1 early path already terminated this request;
            # under PP>1 that path is off, so terminate here on completion.
            if not self._force_terminate_ctx_for_partial_reuse:
                self._effects.terminate_request(request)

    # -- timeouts and cancellation -------------------------------------------

    def inflight_cancel_active(self) -> bool:
        """Whether timed-out transfers are cancelled in flight."""
        if not is_disagg_inflight_cancel_enabled():
            return False
        supports = getattr(self._transceiver, "supports_inflight_request_cancellation", None)
        if callable(supports) and supports() is True:
            return True
        if not self._inflight_cancel_unsupported_logged:
            logger.warning(
                "TRTLLM_DISAGG_ENABLE_INFLIGHT_CANCEL=1 was requested, but "
                f"{type(self._transceiver).__name__} does not advertise in-flight "
                "request cancellation support. Cancellation and transfer-buffer "
                "quarantine are currently scoped to the C++ NIXL transceiver "
                "with the UCX plugin; using the existing timeout and "
                "cancellation behavior for this transceiver."
            )
            self._inflight_cancel_unsupported_logged = True
        return False

    def request_cancellation(self, request: LlmRequest) -> bool:
        """Best-effort cancellation that leaves ownership intact on errors."""
        try:
            return self._transceiver.cancel_request(request)
        except Exception as error:
            logger.error(
                f"KV transfer cancellation failed for request "
                f"{request.py_request_id}; will retry: {error}"
            )
            return False

    def fail_timed_out(self, requests: List[LlmRequest]) -> None:
        """Fail generation requests whose transfer timed out.

        Under multi-rank ADP the error response enters a collective, so it is
        deferred to ``handle_timeouts_synced``.
        """
        if self._enable_attention_dp and self._dist.world_size != 1:
            self._pending_timed_out_requests.extend(requests)
            return
        for req in requests:
            self._effects.fail_requests(
                f"Request {req.py_request_id} timed out", [req], charge_budget=False
            )

    def handle_timeouts_synced(self) -> None:
        """ADP-safe drain of the KV-transfer-timeout consensus collective.

        Reached the same number of times on every rank per iteration; non-ADP
        runs failed timeouts inline and the buffer is empty here.
        """
        if self._is_local is not None:
            return
        if not (self._enable_attention_dp and self._dist.world_size != 1):
            return
        timed_out = self._pending_timed_out_requests
        self._pending_timed_out_requests = []
        any_timed_out = bool(self._dist.tp_allgather_int64([bool(timed_out)]).any())
        if any_timed_out:
            self._effects.fail_requests(
                "Request timed out (KV transfer)", timed_out, charge_budget=False
            )

    def take_pending_context_failures(self) -> Set[int]:
        """Drain context sends that failed after leaving the transfer manager."""
        pending = self._pending_ctx_transfer_failures
        self._pending_ctx_transfer_failures = set()
        return pending

    def forget_request(self, request_id: int) -> None:
        """Drop per-request cancellation bookkeeping once the request is freed."""
        self._timed_out_ctx_cancelled_ids.discard(request_id)
        self._timed_out_gen_cancelled_ids.discard(request_id)
        self._dkv_transfer_events.pop(request_id, None)
        self._dkv_send_errors.pop(request_id, None)
        self._dkv_stalled_request_ids.discard(request_id)

    @nvtx_range("cancel_timed_out_gen_transfers")
    def _cancel_timed_out_gen_transfers(self) -> None:
        """Request cancellation for timed-out generation transfers.

        Rank-synchronized: under attention-DP each TP rank owns a different
        request subset, but the later error responses pass through a TP
        collective, so the decision is based on the TP-wide id union rather
        than a rank-local timeout observation.
        """
        timeout_ms = self._transceiver.kv_transfer_timeout_ms
        if timeout_ms is None:
            return

        requests_in_transfer = {
            req.py_request_id: req
            for req in self._registry.active_requests()
            if req.is_disagg_generation_transmission_in_progress
        }
        current_time = time.monotonic()
        for request in requests_in_transfer.values():
            if request.py_kv_transfer_start_time is None:
                continue
            elapsed_ms = (current_time - request.py_kv_transfer_start_time) * 1000
            if elapsed_ms > timeout_ms and not request.py_kv_transfer_timed_out:
                logger.warning(
                    f"Requesting cancellation for generation request "
                    f"{request.py_request_id} due to KV cache transfer timeout"
                )
                request.py_kv_transfer_timed_out = True

        user_canceled_ids = set(self._registry.canceled_request_ids())
        local_timed_out_ids = sorted(
            request_id
            for request_id, request in requests_in_transfer.items()
            if request.py_kv_transfer_timed_out
            and request_id not in user_canceled_ids
            and request_id not in self._timed_out_gen_cancelled_ids
        )

        if self._dist.tp_size > 1:
            any_timed_out = self._dist.tp_allreduce(int(bool(local_timed_out_ids)), op=ReduceOp.MAX)
        else:
            any_timed_out = int(bool(local_timed_out_ids))
        if not any_timed_out:
            return

        if self._dist.tp_size > 1:
            gathered = self._dist.tp_allgather(local_timed_out_ids)
            timed_out_ids = sorted(set().union(*gathered))
        else:
            timed_out_ids = local_timed_out_ids

        for request_id in timed_out_ids:
            request = requests_in_transfer.get(request_id)
            if request is None:
                continue
            # A peer rank may have crossed the timeout first. Mirror the
            # TP-wide decision locally so a failed cancel attempt keeps
            # retrying even if this rank's wall clock had not yet expired.
            request.py_kv_transfer_timed_out = True
            if request_id in self._timed_out_gen_cancelled_ids:
                continue
            if self.request_cancellation(request):
                self._timed_out_gen_cancelled_ids.add(request_id)
                logger.warning(
                    f"Cancelled timed-out generation KV transfer for request "
                    f"{request.py_request_id}; waiting for C++ transfer status "
                    "to report final cleanup"
                )

    @nvtx_range("check_gen_transfer_errors_consensus")
    def _check_gen_transfer_errors_consensus(self) -> None:
        """Flush generation transfer errors through a TP-uniform path."""
        error_requests = [
            req for req in self._requests_in_error_state() if req.is_generation_only_request
        ]
        local_needs_flush = bool(error_requests)
        if self._dist.tp_size > 1:
            any_needs_flush = self._dist.tp_allreduce(int(local_needs_flush), op=ReduceOp.MAX)
        else:
            any_needs_flush = int(local_needs_flush)
        if not any_needs_flush:
            return
        self._effects.fail_requests(
            "Error in kv cache transfer for generation requests",
            error_requests,
            charge_budget=False,
        )

    # -- transfer errors -----------------------------------------------------

    def _check_transfer_errors(self, kind: str) -> None:
        """Fail requests whose transfer errored, rank-locally.

        Under multi-rank ADP this is a no-op: errors are handled by
        ``handle_errors_synced`` at the loop top.
        """
        if self._enable_attention_dp and self._dist.world_size != 1:
            return
        error_requests = self._requests_in_error_state()
        if error_requests:
            self._effects.fail_requests(
                f"Error in kv cache transfer for {kind}", error_requests, charge_budget=False
            )

    def _requests_in_error_state(self) -> List[LlmRequest]:
        return [
            req
            for req in self._registry.active_requests()
            if req.state == LlmRequestState.DISAGG_TRANS_ERROR
            and not self._is_error_cleanup_blocked(req)
        ]

    def _is_error_cleanup_blocked(self, request: LlmRequest) -> bool:
        """Whether a failed request must wait: the cancel path owns it, or a
        transfer owner still holds its context blocks."""
        if self._vote_id(request) in self._registry.canceled_request_ids():
            return True
        return (
            getattr(request, "is_context_only_request", False) is True
            and request.py_request_id in self._transfers.requests_in_transfer()
        )

    @staticmethod
    def _vote_id(request: LlmRequest) -> int:
        return request.parent_request_id if request.is_child else request.py_request_id

    # -- loop tail -----------------------------------------------------------

    def pace_idle(self) -> None:
        """Sleep briefly when only a KV transfer completing can make progress.

        Call this at the end of an iteration that queued nothing, after the
        pass has drained its ready work, so the pending-transfer check sees the
        state that work left behind. The check is rank-local; the sleep only
        paces and never gates a collective.
        """
        # Context sends are tracked by the transfer manager; generation
        # receives live in the request state, so both directions are covered.
        waiting_on_transfer = self._transfers.has_any_inflight_requests() or any(
            req.is_disagg_generation_init_state or req.is_disagg_generation_transmission_in_progress
            for req in self._registry.active_requests()
        )
        if waiting_on_transfer:
            time.sleep(0.001)


class NoopDisaggCoordinator(DisaggTransferCoordinator):
    """Coordinator used when the executor has no KV cache transceiver.

    A null object: no state, no dependencies, no side effects. The loops call
    every entry point unconditionally, so each one is a no-op here. Transceiver
    enablement is rank-uniform within a collective group, so every rank skips
    the same collectives symmetrically.
    """

    def __init__(self) -> None:
        super().__init__(
            transceiver=None,
            transfer_manager=None,
            kv_cache_manager=None,
            dist=None,
            effects=None,
            registry=None,
            enable_attention_dp=False,
            force_terminate_ctx_for_partial_reuse=False,
        )

    def handle_errors_synced(self) -> None:
        return None

    def admit(self, fitting_gen_init: List[LlmRequest]) -> Tuple[List[LlmRequest], bool]:
        return fitting_gen_init, False

    def revert_deferred_gen_init(
        self, candidates: List[LlmRequest], admitted: List[LlmRequest]
    ) -> None:
        return None

    def receive_gen_init(self, admitted: List[LlmRequest]) -> None:
        return None

    def completed_gen_receives(self, scheduled_batch: "ScheduledRequests") -> List[LlmRequest]:
        return []

    def try_finish_gen_receive(self, request: LlmRequest) -> bool:
        return False

    def prepare_context_schedulable(self, new_requests: List[LlmRequest]) -> None:
        return None

    def poll_gen_transfers(self) -> None:
        return None

    def poll_progress_when_idle(self) -> None:
        return None

    def check_transfer_timeouts(self, only_with_context_sends: bool = False) -> None:
        return None

    def send_completed_context(self, requests: List[LlmRequest]) -> None:
        return None

    def reap_context_sends(self, at_least: int = 0) -> None:
        return None

    def collect_dkv_transfer_events(self) -> Tuple[DkvTransferEvent, ...]:
        return ()

    def take_dkv_fatal_messages(self) -> Tuple[str, ...]:
        return ()

    def dkv_context_send_failed(self, request: LlmRequest) -> bool:
        return False

    def commit_dkv_transfer_events(self, events: Sequence[DkvTransferEvent]) -> bool:
        return False

    def reap_gen_receives(self, at_least: int = 0) -> None:
        return None

    def release_transfer(self, request: LlmRequest) -> None:
        raise RuntimeError(
            "release_transfer has no transceiver to serve; connector-only "
            "releases are handled by the executor"
        )

    def inflight_cancel_active(self) -> bool:
        return False

    def request_cancellation(self, request: LlmRequest) -> bool:
        return True

    def fail_timed_out(self, requests: List[LlmRequest]) -> None:
        # Without a transceiver no request can time out on a KV transfer.
        return None

    def handle_timeouts_synced(self) -> None:
        return None

    def take_pending_context_failures(self) -> Set[int]:
        return set()

    def forget_request(self, request_id: int) -> None:
        return None

    def pace_idle(self) -> None:
        return None
