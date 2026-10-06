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
"""DKV: group-replicated KV lifecycle for attention-DP prefill.

Every rank runs the KV cache lifecycle of every request, so the ranks must reach
the same decisions from the same replicated state. This module holds the two
lockstep exchanges that keep them aligned and the checker that detects drift:

* the sample sync: one allgather per iteration in which the compute rank of each
  request publishes the sampler outcome (completion, finish reason, error) so
  every replica reaches the same state before KV is committed;
* the control sync: one allgather at the top of every iteration that carries the
  capacity digest, fatal errors, pending responses and transfer events;
* ``DkvInvariantChecker``: startup agreement plus debug-only fingerprint checks;
* layer ownership: which rank stores the KV of which layer under the ``layer_split`` layout.
"""

import hashlib
import os
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass

from tensorrt_llm._torch.disaggregation.orchestration.interfaces import DkvTransferEvent
from tensorrt_llm._torch.distributed.communicator import Distributed
from tensorrt_llm.logger import logger

from .dkv_types import LifecycleKey
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
    """The single S-sample exchange, including pre-exchange validation failures.

    ``batch_digest`` is ``(request count, digest)`` of the global batch as this rank sees it, so a
    rank whose replica of the batch diverged is reported by every rank at the same moment.
    """

    completions: tuple[DkvSampleCompletion, ...]
    errors: tuple[DkvSamplerError, ...]
    validation_errors: tuple[str, ...]
    batch_digest: tuple[int, int] = (0, 0)


def _digest_text(text: str) -> int:
    """64-bit digest that is identical across processes (unlike the salted builtin ``hash``)."""
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big")


def digest_global_batch(requests: Iterable[LlmRequest]) -> tuple[int, int]:
    """Order-sensitive digest of ``(id, compute rank, context position)`` of the real requests."""
    records = [
        (request.py_request_id, request.py_dkv_compute_rank, request.context_current_position)
        for request in requests
        if not request.is_dummy
    ]
    return len(records), _digest_text(repr(records))


def _valid_batch_digest(digest: object) -> bool:
    return (
        isinstance(digest, tuple)
        and len(digest) == 2
        and all(type(value) is int and value >= 0 for value in digest)
    )


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
        digest_global_batch(global_requests),
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
    for rank, rank_payload in enumerate(all_payloads):
        if not _valid_batch_digest(rank_payload.batch_digest):
            problems.append(f"rank {rank}: invalid global batch digest")
        elif rank_payload.batch_digest != all_payloads[0].batch_digest:
            problems.append(
                f"rank {rank}: global batch differs from rank 0 "
                f"({rank_payload.batch_digest[0]} requests, digest {rank_payload.batch_digest[1]} "
                f"vs {all_payloads[0].batch_digest[0]} requests, "
                f"digest {all_payloads[0].batch_digest[1]})"
            )
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
    in_transfer_count: int = 0
    in_transfer_digest: int = 0


def digest_request_ids(request_ids: Iterable[int]) -> int:
    """Order-independent 64-bit digest of request ids, identical across processes."""
    return _digest_text(",".join(str(request_id) for request_id in sorted(request_ids)))


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
    counters = (
        digest.iter_counter,
        digest.index_mapper_used,
        digest.active_request_count,
        digest.in_transfer_count,
        digest.in_transfer_digest,
    )
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

    Capacity counters are checked even with debugging disabled. The debug-only
    freed-request sets share this exchange, including diagnostics on
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
            problems.append(f"rank {rank}: invalid capacity digest")
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
                problems.append(f"rank {rank}: capacity digest differs from rank 0")
            if rank_payload.debug_enabled != reference.debug_enabled:
                problems.append(f"rank {rank}: debug flag differs from rank 0")
            if rank_payload.debug_enabled and (
                rank_payload.freed_request_ids != reference.freed_request_ids
            ):
                problems.append(f"rank {rank}: freed request IDs differ from rank 0")
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


_MAX_RANK_REPORT_CHARS = 4000


def _first_record_difference(reference: object, other: object) -> str:
    """Name the first record where ``other`` departs from ``reference``."""
    if isinstance(reference, (list, tuple)) and isinstance(other, (list, tuple)):
        for index, (left, right) in enumerate(zip(reference, other)):
            if left != right:
                return f"record {index}: {left!r} != {right!r}"
        if len(reference) != len(other):
            return f"{len(reference)} records != {len(other)} records"
    return f"{reference!r} != {other!r}"


def _first_payload_difference(reference: tuple, other: tuple) -> str:
    """Summarize where one rank's ``(iter, [(tag, items), ...])`` differs from rank 0."""
    if reference[0] != other[0]:
        return f"iteration {reference[0]} != {other[0]}"
    reference_tags = [tag for tag, _ in reference[1]]
    other_tags = [tag for tag, _ in other[1]]
    if reference_tags != other_tags:
        return f"tags {reference_tags} != {other_tags}"
    for (tag, left), (_, right) in zip(reference[1], other[1]):
        if left != right:
            return f"tag {tag}, {_first_record_difference(left, right)}"
    return "no difference found"


def _bounded_repr(value: object) -> str:
    text = repr(value)
    if len(text) > _MAX_RANK_REPORT_CHARS:
        return text[:_MAX_RANK_REPORT_CHARS] + f"... ({len(text)} characters)"
    return text


class DkvInvariantChecker:
    """Cross-rank consistency checker for DKV; per-iteration checks are debug-only.

    DKV correctness relies on every rank independently deriving identical
    scheduling decisions from replicated state. A divergence is silent and
    surfaces much later (e.g. as a mismatched MoE collective). Construction
    always agrees on the debug flag and on the startup settings that every rank
    must share. When debugging is enabled (TRTLLM_DKV_DEBUG=1) the checker
    fingerprints per-iteration decisions, allgathers one digest per rank, and on
    mismatch reports the first differing record per rank and raises at the exact
    iteration the divergence appeared.
    """

    def __init__(
        self,
        dist: Distributed,
        enabled: bool | None = None,
        startup_settings: Mapping[str, object] | None = None,
    ) -> None:
        self.dist = dist
        if enabled is None:
            enabled = os.environ.get("TRTLLM_DKV_DEBUG", "0") not in ("0", "false", "False")
        self.enabled = enabled
        settings = tuple(sorted((startup_settings or {}).items()))
        by_rank = self.dist.tp_allgather((self.enabled, settings))
        if any(rank_enabled != self.enabled for rank_enabled, _ in by_rank):
            raise RuntimeError(
                "DKV invariant checker enable flags differ across ranks: "
                f"{[rank_enabled for rank_enabled, _ in by_rank]}; set TRTLLM_DKV_DEBUG "
                "consistently"
            )
        if any(rank_settings != settings for _, rank_settings in by_rank):
            raise RuntimeError(
                "DKV startup settings differ across ranks: "
                + "; ".join(
                    f"rank {rank}: {dict(rank_settings)}"
                    for rank, (_, rank_settings) in enumerate(by_rank)
                )
                + ". Set the corresponding environment variables identically on every rank."
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
        report = [f"DKV invariant violation at iter {iter_counter}, tags={list(items_by_tag)}"]
        for rank, rank_items in enumerate(all_items[1:], start=1):
            if rank_items != all_items[0]:
                report.append(
                    f"  rank{rank} differs from rank0: "
                    f"{_first_payload_difference(all_items[0], rank_items)}"
                )
        for rank, rank_items in enumerate(all_items):
            report.append(f"  rank{rank}: {_bounded_repr(rank_items)}")
        msg = "\n".join(report)
        logger.error(msg)
        raise RuntimeError(msg)


def compute_ownership(num_layers: int, group_size: int) -> tuple[int, ...]:
    """The rank that stores the KV of every layer under the ``layer_split`` layout.

    Layers are split into contiguous ranges whose sizes differ by at most one; the first
    ``num_layers % group_size`` ranks take the extra layer. The table is a pure function of its
    arguments, so every rank derives the same one without communication.

    Raises:
        ValueError: ``num_layers`` or ``group_size`` is below one.
    """
    if num_layers < 1 or group_size < 1:
        raise ValueError(
            f"num_layers and group_size must be positive, got {num_layers} and {group_size}"
        )
    base, extra = divmod(num_layers, group_size)
    owners: list[int] = []
    for rank in range(group_size):
        owners.extend([rank] * (base + (1 if rank < extra else 0)))
    return tuple(owners)


def owned_layers(owner_of_layer: Sequence[int], rank: int) -> tuple[int, ...]:
    """The layers whose KV ``rank`` stores, in increasing order."""
    return tuple(layer for layer, owner in enumerate(owner_of_layer) if owner == rank)


def ownership_fingerprint(owner_of_layer: Sequence[int]) -> str:
    """A digest of the ownership table that is identical across processes."""
    return hashlib.sha256(repr(tuple(owner_of_layer)).encode()).hexdigest()


def validate_ownership(
    owner_of_layer: Sequence[int],
    life_cycles_of_layer: Sequence[Collection[LifecycleKey]],
    group_size: int,
) -> None:
    """Check that every rank can hold every KV life cycle of the model.

    The prefix-match length is decided by which life cycles exist for a block, so a rank that
    lacks one of them would reuse a different number of tokens than the others and the replicated
    lifecycle would diverge. Every rank must therefore own layers that cover all the life cycles
    that any layer of the model takes part in.

    Args:
        owner_of_layer: The rank that stores each layer.
        life_cycles_of_layer: The semantic life cycles each layer's KV lives in.
        group_size: The number of ranks.

    Raises:
        ValueError: The table does not cover the layers, names a rank outside the group, or leaves
            a rank without some life cycle.
    """
    if len(owner_of_layer) != len(life_cycles_of_layer):
        raise ValueError(
            f"The ownership table has {len(owner_of_layer)} layers, the model {len(life_cycles_of_layer)}"
        )
    outside = sorted({owner for owner in owner_of_layer if not 0 <= owner < group_size})
    if outside:
        raise ValueError(f"Ranks {outside} own layers but the group has {group_size} ranks")
    required = set().union(*life_cycles_of_layer) if life_cycles_of_layer else set()
    for rank in range(group_size):
        layers = owned_layers(owner_of_layer, rank)
        covered = (
            set().union(*(life_cycles_of_layer[layer] for layer in layers)) if layers else set()
        )
        if covered != required:
            raise ValueError(
                f"{len(owner_of_layer)} layers on {group_size} ranks is not supported with "
                f"dkv_config layer_split yet: rank {rank} would own layers {list(layers)}, which "
                f"lack the KV life cycles {sorted(required - covered, key=repr)} that other layers "
                "use. Use a smaller attention-DP group."
            )
