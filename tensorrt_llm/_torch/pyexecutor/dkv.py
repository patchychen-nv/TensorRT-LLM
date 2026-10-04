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
"""DKV: layer-split distributed KV storage with a group-replicated KV lifecycle.

All DKV-related helper code lives in this module for now; it will be split
into a package once the streamer/staging components land.
"""

import hashlib
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from tensorrt_llm._torch.disaggregation.orchestration.interfaces import DkvTransferEvent
from tensorrt_llm._torch.distributed.communicator import Distributed
from tensorrt_llm.logger import logger

from .llm_request import FinishReason, LlmRequest, LlmRequestState


@dataclass(frozen=True)
class DkvSampleCompletion:
    """One compute rank's completion, excluding generated token values."""

    request_id: int
    state: int
    finish_reasons: tuple[int, ...]


@dataclass(frozen=True)
class DkvSamplerError:
    """A sampler failure and the local real requests affected by it."""

    message: str
    request_ids: tuple[int, ...]


@dataclass(frozen=True)
class DkvSamplePayload:
    """The single S-sample exchange, including pre-exchange validation failures."""

    completions: tuple[DkvSampleCompletion, ...]
    errors: tuple[DkvSamplerError, ...]
    validation_errors: tuple[str, ...]


def sync_dkv_sample_results(
    dist: Distributed,
    global_requests: Sequence[LlmRequest],
    local_requests: Sequence[LlmRequest],
    errors: Sequence[tuple[str, tuple[int, ...]]] = (),
) -> tuple[DkvSamplerError, ...]:
    """Replicate single-token prefill completion before committing global KV.

    All ranks must call after global context cursor advancement and local sampler
    updates. Intermediate chunks contribute no completion. Validation failures
    are raised after the exchange, before changing replicas. Valid sampler
    failures are returned in rank order for symmetric lifecycle handling; a
    failure takes precedence over a partially updated completion.
    """
    global_by_id: dict[int, LlmRequest] = {}
    validation_errors: list[str] = []
    for request in global_requests:
        if request.is_dummy:
            continue
        request_id = request.py_request_id
        if request_id in global_by_id:
            validation_errors.append(f"duplicate global request {request_id}")
        global_by_id[request_id] = request
        compute_rank = request.py_dkv_compute_rank
        if not isinstance(compute_rank, int) or not 0 <= compute_rank < dist.tp_size:
            validation_errors.append(f"invalid compute rank for request {request_id}")
        if request.py_dkv_is_local != (compute_rank == dist.tp_rank):
            validation_errors.append(f"inconsistent locality for request {request_id}")
        if not request.py_dkv_is_local and request.finish_reasons != [FinishReason.NOT_FINISHED]:
            validation_errors.append(f"non-local request {request_id} already has a finish reason")

    local_ids: set[int] = set()
    completions = []
    for request in local_requests:
        if request.is_dummy:
            continue
        request_id = request.py_request_id
        if request_id in local_ids:
            validation_errors.append(f"duplicate local request {request_id}")
        local_ids.add(request_id)
        if global_by_id.get(request_id) is not request:
            validation_errors.append(f"local request {request_id} is not its global replica")
        if not request.py_dkv_is_local or request.py_dkv_compute_rank != dist.tp_rank:
            validation_errors.append(f"non-local request {request_id} in local sample batch")
        if request.is_finished:
            completions.append(
                DkvSampleCompletion(
                    request_id,
                    request.state_value,
                    tuple(reason.value for reason in request.finish_reasons),
                )
            )
    expected_local_ids = {
        request_id
        for request_id, request in global_by_id.items()
        if request.py_dkv_compute_rank == dist.tp_rank
    }
    if local_ids != expected_local_ids:
        validation_errors.append(
            f"local sample batch differs from compute-rank ownership: "
            f"expected={sorted(expected_local_ids)}, actual={sorted(local_ids)}"
        )

    payload = DkvSamplePayload(
        tuple(completions),
        tuple(DkvSamplerError(message, tuple(request_ids)) for message, request_ids in errors),
        tuple(validation_errors),
    )
    all_payloads = dist.tp_allgather(payload)
    if len(all_payloads) != dist.tp_size or any(
        not isinstance(rank_payload, DkvSamplePayload) for rank_payload in all_payloads
    ):
        raise RuntimeError(f"DKV S-sample invalid rank payloads: {all_payloads!r}")
    problems = [
        f"rank {rank}: {message}"
        for rank, rank_payload in enumerate(all_payloads)
        for message in rank_payload.validation_errors
    ]
    received: dict[int, DkvSampleCompletion] = {}
    sampler_errors: list[DkvSamplerError] = []
    failed_ids: set[int] = set()
    for rank, rank_payload in enumerate(all_payloads):
        for error in rank_payload.errors:
            if (
                not isinstance(error, DkvSamplerError)
                or not isinstance(error.message, str)
                or not isinstance(error.request_ids, tuple)
                or any(type(request_id) is not int for request_id in error.request_ids)
            ):
                problems.append(f"rank {rank} reported an invalid sampler error")
                continue
            if len(error.request_ids) != len(set(error.request_ids)):
                problems.append(f"rank {rank} reported duplicate sampler error request IDs")
            for request_id in error.request_ids:
                request = global_by_id.get(request_id)
                if request is None or request.py_dkv_compute_rank != rank:
                    problems.append(f"rank {rank} reported sampler error for unowned {request_id}")
                failed_ids.add(request_id)
            sampler_errors.append(error)
        for completion in rank_payload.completions:
            request_id = completion.request_id
            request = global_by_id.get(request_id)
            if request is None or request.py_dkv_compute_rank != rank:
                problems.append(f"rank {rank} reported completion for unowned {request_id}")
            if request_id in received:
                problems.append(f"duplicate completion for request {request_id}")
            received[request_id] = completion
            if completion.state != LlmRequestState.GENERATION_COMPLETE.value:
                problems.append(f"invalid completed state for request {request_id}")
            if len(completion.finish_reasons) != 1 or any(
                reason == FinishReason.NOT_FINISHED.value for reason in completion.finish_reasons
            ):
                problems.append(f"invalid single-beam finish reason for request {request_id}")
            else:
                try:
                    FinishReason(completion.finish_reasons[0])
                except ValueError:
                    problems.append(f"unknown finish reason for request {request_id}")
    if problems:
        raise RuntimeError("DKV S-sample invariant violation: " + "; ".join(problems))
    expected_completions = {
        request_id
        for request_id, request in global_by_id.items()
        if request.context_remaining_length == 0 and request_id not in failed_ids
    }
    healthy_completions = received.keys() - failed_ids
    if healthy_completions != expected_completions:
        raise RuntimeError(
            "DKV S-sample completions differ from finished prefill: "
            f"missing={sorted(expected_completions - healthy_completions)}, "
            f"unexpected={sorted(healthy_completions - expected_completions)}"
        )
    for request_id, request in global_by_id.items():
        if request.py_dkv_is_local or request_id not in healthy_completions:
            continue
        completion = received[request_id]
        request.finish_by_reason(FinishReason(completion.finish_reasons[0]))
        request.state = LlmRequestState(completion.state)
    return tuple(sampler_errors)


@dataclass(frozen=True)
class DkvControlDigest:
    """Always-on logical capacity counters at one S-control boundary."""

    iter_counter: int
    free_pages: tuple[tuple[int, ...], ...]
    index_mapper_used: int
    active_request_count: int


@dataclass(frozen=True)
class DkvControlPayload:
    """Local events and replicated state exchanged once on every DKV loop."""

    digest: DkvControlDigest
    fatal_messages: tuple[str, ...] = ()
    has_pending_responses: bool = False
    debug_enabled: bool = False
    freed_request_ids: tuple[int, ...] = ()
    transfer_events: tuple[DkvTransferEvent, ...] = ()


@dataclass(frozen=True)
class DkvControlResult:
    """Identical actions for every rank after a successful S-control exchange."""

    fatal_messages: tuple[str, ...]
    has_pending_responses: bool
    transfer_events: tuple[DkvTransferEvent, ...] = ()


def _valid_control_digest(digest: DkvControlDigest) -> bool:
    if not isinstance(digest, DkvControlDigest):
        return False
    counters = (digest.iter_counter, digest.index_mapper_used, digest.active_request_count)
    return (
        all(type(value) is int and value >= 0 for value in counters)
        and isinstance(digest.free_pages, tuple)
        and all(
            isinstance(level, tuple) and all(type(value) is int and value >= 0 for value in level)
            for level in digest.free_pages
        )
    )


def sync_dkv_control(dist: Distributed, payload: DkvControlPayload) -> DkvControlResult:
    """Validate replicated state and combine local events in one collective.

    D12 capacity counters are checked even with debugging disabled. Debug C4
    free sets share this exchange, including diagnostics on
    mismatch, so no failure-only collective is necessary. Fatal events are
    returned for the executor's collectively aligned shutdown path.
    """
    all_payloads = dist.tp_allgather(payload)
    problems = []
    transfer_events: dict[int, DkvTransferEvent] = {}
    if len(all_payloads) != dist.tp_size:
        problems.append(f"expected {dist.tp_size} rank payloads, got {len(all_payloads)}")
    for rank, rank_payload in enumerate(all_payloads):
        if not isinstance(rank_payload, DkvControlPayload):
            problems.append(f"rank {rank}: invalid control payload")
            continue
        if not _valid_control_digest(rank_payload.digest):
            problems.append(f"rank {rank}: invalid D12 digest")
        if not isinstance(rank_payload.fatal_messages, tuple) or any(
            not isinstance(message, str) for message in rank_payload.fatal_messages
        ):
            problems.append(f"rank {rank}: invalid fatal messages")
        if type(rank_payload.has_pending_responses) is not bool:
            problems.append(f"rank {rank}: invalid pending-response flag")
        if type(rank_payload.debug_enabled) is not bool:
            problems.append(f"rank {rank}: invalid debug flag")
        if not isinstance(rank_payload.freed_request_ids, tuple) or any(
            type(request_id) is not int for request_id in rank_payload.freed_request_ids
        ):
            problems.append(f"rank {rank}: invalid free request IDs")
        if not rank_payload.debug_enabled and rank_payload.freed_request_ids:
            problems.append(f"rank {rank}: debug state supplied with debugging disabled")
        if not isinstance(rank_payload.transfer_events, tuple):
            problems.append(f"rank {rank}: invalid transfer events")
            continue
        for event in rank_payload.transfer_events:
            if (
                not isinstance(event, DkvTransferEvent)
                or type(event.request_id) is not int
                or type(event.compute_rank) is not int
                or event.compute_rank != rank
                or event.outcome not in ("completed", "failed", "timed_out")
                or not isinstance(event.error_message, str)
            ):
                problems.append(f"rank {rank}: invalid transfer event {event!r}")
                continue
            if event.request_id in transfer_events:
                problems.append(f"rank {rank}: duplicate transfer event for {event.request_id}")
            transfer_events[event.request_id] = event
    if not problems:
        reference = all_payloads[0]
        for rank, rank_payload in enumerate(all_payloads[1:], start=1):
            if rank_payload.digest != reference.digest:
                problems.append(f"rank {rank}: D12 digest differs from rank 0")
            if rank_payload.debug_enabled != reference.debug_enabled:
                problems.append(f"rank {rank}: debug flag differs from rank 0")
            if rank_payload.debug_enabled and (
                rank_payload.freed_request_ids != reference.freed_request_ids
            ):
                problems.append(f"rank {rank}: C4 free request IDs differ from rank 0")
    if problems:
        summaries = "\n".join(
            f"  rank {rank}: {rank_payload!r}" for rank, rank_payload in enumerate(all_payloads)
        )
        raise RuntimeError(
            "DKV S-control invariant violation: " + "; ".join(problems) + "\n" + summaries
        )
    return DkvControlResult(
        fatal_messages=tuple(
            f"rank {rank}: {message}"
            for rank, rank_payload in enumerate(all_payloads)
            for message in rank_payload.fatal_messages
        ),
        has_pending_responses=any(
            rank_payload.has_pending_responses for rank_payload in all_payloads
        ),
        transfer_events=tuple(transfer_events[key] for key in sorted(transfer_events)),
    )


class DkvInvariantChecker:
    """Debug-only cross-rank consistency checker for DKV.

    DKV correctness relies on every rank independently deriving identical
    scheduling decisions from replicated state. A divergence is silent and
    surfaces much later (e.g. as a mismatched MoE collective). When enabled
    (TRTLLM_DKV_DEBUG=1) this checker fingerprints per-iteration decisions,
    allgathers one digest per rank, and on mismatch dumps a per-rank diff
    and raises at the exact iteration the divergence appeared.
    """

    def __init__(self, dist: Distributed, enabled: bool | None = None) -> None:
        self.dist = dist
        if enabled is None:
            enabled = os.environ.get("TRTLLM_DKV_DEBUG", "0") not in ("0", "false", "False")
        self.enabled = enabled
        enabled_by_rank = self.dist.tp_allgather(self.enabled)
        if any(flag != self.enabled for flag in enabled_by_rank):
            raise RuntimeError(
                "DKV invariant checker enable flags differ across ranks: "
                f"{enabled_by_rank}; set TRTLLM_DKV_DEBUG consistently"
            )

    def check(self, iter_counter: int, tag: str, items: object) -> None:
        """Assert that ``items`` is identical on every rank of the TP group.

        ``items`` is an ordered sequence of fingerprint tuples, e.g.
        ``[(request_id, context_current_position, context_chunk_size,
        state_value), ...]``. Preserve lifecycle operation order and include
        dummy requests: DKV dummy lifecycles must agree across ranks too.
        """
        self.check_many(iter_counter, {tag: items})

    def check_many(self, iter_counter: int, items_by_tag: Mapping[str, object]) -> None:
        """Compare ordered fingerprints using one collective on the success path.

        Values must have deterministic representations across processes. Tag
        insertion order, iteration number and each value's order are included
        so a collective at a different checkpoint cannot silently match.
        """
        if not self.enabled:
            return
        # sha256 over repr, NOT builtin hash(): str hashing is salted per
        # process (PYTHONHASHSEED) and would differ across ranks.
        payload = (iter_counter, list(items_by_tag.items()))
        digest = hashlib.sha256(repr(payload).encode()).hexdigest()
        all_digests = self.dist.tp_allgather(digest)
        if all(d == all_digests[0] for d in all_digests):
            return
        all_items = self.dist.tp_allgather(payload)
        report = [
            f"DKV invariant violation at iter {iter_counter}, tags={list(items_by_tag)}",
            f"  rank0: {all_items[0]!r}",
        ]
        reference = all_items[0]
        for rank, rank_items in enumerate(all_items[1:], start=1):
            if rank_items == reference:
                continue
            report.append(f"  rank{rank}: {rank_items!r}")
        msg = "\n".join(report)
        logger.error(msg)
        raise RuntimeError(msg)
