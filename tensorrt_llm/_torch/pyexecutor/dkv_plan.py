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
"""Data-plane plan of the DKV ``layer_split`` layout.

Under ``layer_split`` the KV cache of a layer lives on one rank, the *owner* of the layer, while the
forward of a request runs on its *compute rank*, which is usually another rank. Around each layer
the owner sends the compute rank the cached KV the layer reads (a *fetch*, step ``F(layer)``) and
the compute rank sends the new KV back (a *writeback*, step ``W(layer)``). This module decides, for
one iteration, which messages exist, how many bytes each carries and in which order every rank
issues them. It performs no communication and imports only the standard library, so a plan can be
built, fingerprinted and checked (see ``dkv_plan_simulator``) without a GPU.

Every rank builds the plan itself from replicated state and nothing is exchanged, so the ranks
agree only because ``build_dkv_plan`` is a pure function of

* the global scheduled batch: for each request its id, compute rank, the number of history tokens
  already in the cache, the chunk size and whether it is a dummy;
* the ownership table (the owner of every layer);
* the cost model, which names the buffers of every layer and turns (layer, buffer, history, chunk)
  into bytes and pages.

It reads no clock, environment or device state and iterates only sequences, never a set or mapping
whose order could differ between processes. Requests are put in a canonical order, ascending
compute rank and then request id, whatever order the caller lists them in. A dummy request adds
nothing to the plan.

Messages. The KV of a layer is held in one or more *buffers* (a data role of the cache manager:
key and value, compressed KV, a compressor state, ...), named by the cost model in an order every
rank shares. The unit of communication is one message per (step, compute rank): it carries, back to
back in request order and for each request in buffer order, the segments of every request that is
computed on that rank. A message that would carry no byte is not part of the plan. When the owner
of a layer is also the compute rank of some requests, their part of a step is a local gather (fetch)
or scatter (writeback) instead of a message.

Order. A rank issues the operations of its data stream one after another, and a send or receive
blocks until its peer posts the matching operation. The ranks therefore have to agree on the order
of the steps and on the order of the messages inside a step. Every rank issues the steps in one
global order (``global_step_order``) driven by hooks that run on all ranks:

* before the first layer, ``F(0)`` to ``F(ring_depth - 2)``: the prefetch of the first layers;
* at the top of layer ``l``, ``W(l - 1)`` and then ``F(l + ring_depth - 1)`` (each if it exists):
  the writeback of the layer that just finished and the prefetch of the layer ``ring_depth - 1``
  layers ahead, which reuses the staging slot of layer ``l - 1``;
* after the last layer, ``W(L - 1)``.

Inside a step the messages are ordered by ascending compute rank. A rank derives its own operations
by filtering the step for the messages it takes part in, so for every pair of ranks the sender and
the receiver see the same messages in the same order. In one step a rank never both sends and
receives: the owner only sends in a fetch and only receives in a writeback, every other rank does
the opposite.

Usage::

    from tensorrt_llm._torch.pyexecutor.dkv import compute_ownership

    owners = compute_ownership(num_layers, group_size)
    plan = build_dkv_plan(batch, owners, costs, group_size=group_size, ring_depth=2)
    fingerprint = plan_fingerprint(plan)
"""

import hashlib
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import IntEnum
from typing import NamedTuple, Protocol

__all__ = [
    "ISSUE_AT_BEGIN",
    "Direction",
    "DkvPlan",
    "OpAction",
    "PageCost",
    "PlanRequest",
    "PlanStep",
    "RankOp",
    "Segment",
    "StagingCostModel",
    "StepId",
    "Transfer",
    "build_dkv_plan",
    "global_step_order",
    "plan_canonical_form",
    "plan_fingerprint",
    "step_label",
]

# The issue point of the steps that are enqueued before the first layer.
ISSUE_AT_BEGIN = -1


class Direction(IntEnum):
    """Which way a step moves KV.

    ``FETCH``: the owner of a layer sends the cached KV to the compute rank.
    ``WRITEBACK``: the compute rank sends the new KV back to the owner.
    """

    FETCH = 0
    WRITEBACK = 1


class OpAction(IntEnum):
    """What one rank does in a message: send it, receive it, or copy locally.

    ``LOCAL`` is the gather (fetch) or scatter (writeback) of a layer whose owner is also the
    compute rank; it involves no peer.
    """

    SEND = 0
    RECV = 1
    LOCAL = 2


@dataclass(frozen=True, slots=True)
class PageCost:
    """The staging costs of one request's share of one buffer in one layer.

    Attributes:
        page_bytes: Bytes of one page of the buffer, one block's worth. Positive.
        fetch_pages: Pages the owner sends before the chunk is computed.
        writeback_pages: Pages the compute rank returns after the chunk was computed.
    """

    page_bytes: int
    fetch_pages: int
    writeback_pages: int


class StagingCostModel(Protocol):
    """Where the plan gets its buffers and sizes from.

    The plan's logic only adds and orders what the cost model returns, so a different model (other
    buffers, page sizes, windows) changes the numbers and nothing else.
    """

    def buffers_of(self, layer: int) -> tuple[str, ...]:
        """The buffers of ``layer``: distinct names, in the order the segments of a request list
        them.

        Must give the same answer on every rank.
        """
        ...

    def page_cost(self, layer: int, buffer: str, history: int, chunk: int) -> PageCost:
        """The costs of a request's share of ``buffer`` in ``layer``.

        The cost model must be a pure function of its arguments and give the same answer on every
        rank. It is asked only for the buffers ``buffers_of`` names and only for requests that are
        not dummies.

        Args:
            layer: The model layer.
            buffer: A buffer of ``layer``.
            history: Tokens of the request that are already in the cache.
            chunk: Tokens of the request that are computed in this iteration.

        Returns:
            The page size and the page counts of the two directions. A count of zero means the
            request has nothing of this buffer to move that way.
        """
        ...


@dataclass(frozen=True, slots=True)
class PlanRequest:
    """A request of the global scheduled batch, as the plan sees it.

    Attributes:
        request_id: Unique across the batch.
        compute_rank: The rank that runs the forward of the request.
        context_current_position: Tokens of the request already in the KV cache (the history).
        context_chunk_size: Tokens of the request computed in this iteration.
        is_dummy: Whether the request only keeps its rank running a forward. A dummy never moves
            KV.
    """

    request_id: int
    compute_rank: int
    context_current_position: int
    context_chunk_size: int
    is_dummy: bool = False


@dataclass(frozen=True, slots=True)
class Segment:
    """The part of a message that belongs to one request and one buffer.

    Attributes:
        request_id: The request the pages belong to.
        buffer: The buffer of the pages.
        pages: Number of pages.
        nbytes: Bytes of the segment, ``pages`` times the page size of the buffer in that layer.
    """

    request_id: int
    buffer: str
    pages: int
    nbytes: int


@dataclass(frozen=True, slots=True)
class Transfer:
    """One message, or one local copy when ``owner == compute``.

    A fetch moves ``segments`` from the owner to the compute rank, a writeback from the compute
    rank to the owner. The segments are concatenated in this order: requests by ascending id and,
    for each request, the buffers in the cost model's order.

    Attributes:
        direction: Fetch or writeback.
        layer: The layer the data belongs to.
        owner: The rank that owns the layer.
        compute: The rank that computes the requests of the segments.
        segments: What the message carries; never empty.
    """

    direction: Direction
    layer: int
    owner: int
    compute: int
    segments: tuple[Segment, ...]

    @property
    def nbytes(self) -> int:
        """Bytes of the whole message."""
        return sum(segment.nbytes for segment in self.segments)

    @property
    def pages(self) -> int:
        """Pages of the whole message."""
        return sum(segment.pages for segment in self.segments)

    @property
    def is_local(self) -> bool:
        """Whether the owner computes the requests itself, so no message is needed."""
        return self.owner == self.compute

    @property
    def label(self) -> str:
        """The step and the compute rank, for example ``F(7)->2``."""
        return f"{step_label(self.direction, self.layer)}->{self.compute}"


@dataclass(frozen=True, slots=True)
class RankOp:
    """What one rank does for one transfer.

    Attributes:
        action: Whether the rank sends, receives or copies locally.
        transfer: The transfer the operation belongs to.
    """

    action: OpAction
    transfer: Transfer

    @property
    def rank(self) -> int:
        """The rank that performs the operation."""
        transfer = self.transfer
        if self.action is OpAction.LOCAL:
            return transfer.owner
        owner_sends = transfer.direction is Direction.FETCH
        return transfer.owner if (self.action is OpAction.SEND) == owner_sends else transfer.compute

    @property
    def peer(self) -> int:
        """The other end of the message; the rank itself for a local copy."""
        transfer = self.transfer
        return transfer.compute if self.rank == transfer.owner else transfer.owner

    @property
    def nbytes(self) -> int:
        """Bytes of the operation."""
        return self.transfer.nbytes

    def describe(self) -> str:
        """A short text for messages, for example ``send F(7)->2 4096 B to rank 2``."""
        transfer = self.transfer
        text = f"{transfer.label} {transfer.nbytes} B"
        if self.action is OpAction.LOCAL:
            verb = "gather" if transfer.direction is Direction.FETCH else "scatter"
            return f"local {verb} {text}"
        if self.action is OpAction.SEND:
            return f"send {text} to rank {self.peer}"
        return f"receive {text} from rank {self.peer}"


@dataclass(frozen=True, slots=True)
class PlanStep:
    """One step of the global order: all the messages of a fetch or a writeback of one layer.

    Attributes:
        direction: Fetch or writeback.
        layer: The layer of the step.
        issue_point: The hook that enqueues the step on every rank: ``ISSUE_AT_BEGIN`` before the
            first layer, a layer index ``l`` at the top of layer ``l`` and the number of layers
            after the last layer.
        transfers: The transfers ordered by ascending compute rank.
    """

    direction: Direction
    layer: int
    issue_point: int
    transfers: tuple[Transfer, ...]

    def ops_for(self, rank: int) -> tuple[RankOp, ...]:
        """The operations ``rank`` performs in this step, in the order it must issue them."""
        ops = []
        fetch = self.direction is Direction.FETCH
        for transfer in self.transfers:
            if transfer.owner == transfer.compute:
                if rank == transfer.owner:
                    ops.append(RankOp(OpAction.LOCAL, transfer))
            elif rank == transfer.owner:
                ops.append(RankOp(OpAction.SEND if fetch else OpAction.RECV, transfer))
            elif rank == transfer.compute:
                ops.append(RankOp(OpAction.RECV if fetch else OpAction.SEND, transfer))
        return tuple(ops)


class StepId(NamedTuple):
    """A step of the global order and the hook that enqueues it."""

    direction: Direction
    layer: int
    issue_point: int


def step_label(direction: Direction, layer: int) -> str:
    """The name of a step in messages: ``F(7)`` for the fetch and ``W(7)`` for the writeback."""
    return f"{'F' if direction is Direction.FETCH else 'W'}({layer})"


def global_step_order(num_layers: int, ring_depth: int) -> tuple[StepId, ...]:
    """The order in which every rank enqueues the steps of an iteration.

    ``F(0)`` to ``F(ring_depth - 2)`` come first. Then, at the top of each layer ``l``, ``W(l - 1)``
    (for ``l >= 1``) and ``F(l + ring_depth - 1)`` (if that layer exists). ``W(num_layers - 1)``
    comes last. Every fetch and every writeback appears exactly once.

    Args:
        num_layers: Number of layers of the model, at least 1.
        ring_depth: Slots of the staging ring, at least 1. A fetch is issued ``ring_depth - 1``
            layers ahead of the layer that consumes it.

    Raises:
        ValueError: ``num_layers`` or ``ring_depth`` is below one.
    """
    if num_layers < 1 or ring_depth < 1:
        raise ValueError(
            f"num_layers and ring_depth must be positive, got {num_layers} and {ring_depth}"
        )
    order = [
        StepId(Direction.FETCH, layer, ISSUE_AT_BEGIN)
        for layer in range(min(ring_depth - 1, num_layers))
    ]
    for layer in range(num_layers):
        if layer >= 1:
            order.append(StepId(Direction.WRITEBACK, layer - 1, layer))
        ahead = layer + ring_depth - 1
        if ahead < num_layers:
            order.append(StepId(Direction.FETCH, ahead, layer))
    order.append(StepId(Direction.WRITEBACK, num_layers - 1, num_layers))
    return tuple(order)


@dataclass(frozen=True, repr=False)
class DkvPlan:
    """The data plane of one iteration, identical on every rank.

    Attributes:
        group_size: Number of ranks of the DKV group. Every rank runs the forward of every layer.
        ring_depth: Slots of the staging ring.
        owner_of_layer: The owner rank of every layer.
        requests: The batch in canonical order (ascending compute rank, then request id),
            dummies included.
        steps: Every step, in the global order of ``global_step_order``, also those without a
            transfer.
    """

    group_size: int
    ring_depth: int
    owner_of_layer: tuple[int, ...]
    requests: tuple[PlanRequest, ...]
    steps: tuple[PlanStep, ...]

    def __repr__(self) -> str:
        transfers = sum(len(step.transfers) for step in self.steps)
        return (
            f"DkvPlan(group_size={self.group_size}, ring_depth={self.ring_depth}, "
            f"layers={self.num_layers}, requests={len(self.requests)}, transfers={transfers})"
        )

    @property
    def num_layers(self) -> int:
        """Number of layers of the model."""
        return len(self.owner_of_layer)

    def step(self, direction: Direction, layer: int) -> PlanStep:
        """The step ``F(layer)`` or ``W(layer)``.

        Raises:
            ValueError: The plan has no such layer.
        """
        for step in self.steps:
            if step.direction is direction and step.layer == layer:
                return step
        raise ValueError(
            f"{step_label(direction, layer)} is not in a plan of {self.num_layers} layers"
        )

    def transfers(self, direction: Direction, layer: int) -> tuple[Transfer, ...]:
        """The messages of one step, ordered by compute rank."""
        return self.step(direction, layer).transfers

    def steps_at(self, issue_point: int) -> tuple[PlanStep, ...]:
        """The steps that the hook ``issue_point`` enqueues, in the order to enqueue them."""
        return tuple(step for step in self.steps if step.issue_point == issue_point)

    def begin_steps(self) -> tuple[PlanStep, ...]:
        """The steps enqueued before the first layer."""
        return self.steps_at(ISSUE_AT_BEGIN)

    def layer_steps(self, layer: int) -> tuple[PlanStep, ...]:
        """The steps enqueued at the top of ``layer``."""
        return self.steps_at(layer)

    def end_steps(self) -> tuple[PlanStep, ...]:
        """The steps enqueued after the last layer."""
        return self.steps_at(self.num_layers)

    def rank_program(self, rank: int) -> tuple[RankOp, ...]:
        """Everything ``rank`` does on its data stream, in issue order.

        Raises:
            ValueError: ``rank`` is not a rank of the group.
        """
        if not 0 <= rank < self.group_size:
            raise ValueError(f"rank {rank} is not in a group of {self.group_size} ranks")
        return tuple(op for step in self.steps for op in step.ops_for(rank))


def _check_geometry(owner_of_layer: tuple[int, ...], group_size: int, ring_depth: int) -> None:
    if group_size < 1 or ring_depth < 1:
        raise ValueError(
            f"group_size and ring_depth must be positive, got {group_size} and {ring_depth}"
        )
    if not owner_of_layer:
        raise ValueError("the ownership table has no layer")
    outside = sorted({owner for owner in owner_of_layer if not 0 <= owner < group_size})
    if outside:
        raise ValueError(f"ranks {outside} own layers but the group has {group_size} ranks")


def _canonical_requests(
    requests: Iterable[PlanRequest], group_size: int
) -> tuple[PlanRequest, ...]:
    ordered = tuple(sorted(requests, key=lambda r: (r.compute_rank, r.request_id)))
    for request in ordered:
        if not 0 <= request.compute_rank < group_size:
            raise ValueError(
                f"request {request.request_id} is computed on rank {request.compute_rank}, "
                f"but the group has {group_size} ranks"
            )
        if request.context_current_position < 0 or request.context_chunk_size < 0:
            raise ValueError(
                f"request {request.request_id} has history {request.context_current_position} "
                f"and chunk {request.context_chunk_size}; both must not be negative"
            )
    ids = sorted(request.request_id for request in ordered)
    for first, second in zip(ids, ids[1:]):
        if first == second:
            raise ValueError(f"request id {first} appears more than once in the batch")
    return ordered


def _check_buffers(layer: int, buffers: tuple[str, ...]) -> None:
    if not all(isinstance(buffer, str) for buffer in buffers) or len(set(buffers)) < len(buffers):
        raise ValueError(
            f"the cost model names the buffers {buffers!r} for layer {layer}; "
            "they must be distinct strings"
        )


def _segments(
    costs: StagingCostModel,
    layer: int,
    buffers: tuple[str, ...],
    requests: Sequence[PlanRequest],
) -> tuple[list[Segment], list[Segment]]:
    """The fetch and writeback segments of ``requests`` for the buffers of one layer."""
    fetch: list[Segment] = []
    writeback: list[Segment] = []
    for request in requests:
        history = request.context_current_position
        chunk = request.context_chunk_size
        for buffer in buffers:
            cost = costs.page_cost(layer, buffer, history, chunk)
            if cost.page_bytes < 1 or cost.fetch_pages < 0 or cost.writeback_pages < 0:
                raise ValueError(
                    f"the cost model returned {cost} for layer {layer}, buffer {buffer}, "
                    f"history {history} and chunk {chunk}"
                )
            if cost.fetch_pages:
                nbytes = cost.fetch_pages * cost.page_bytes
                fetch.append(Segment(request.request_id, buffer, cost.fetch_pages, nbytes))
            if cost.writeback_pages:
                nbytes = cost.writeback_pages * cost.page_bytes
                writeback.append(Segment(request.request_id, buffer, cost.writeback_pages, nbytes))
    return fetch, writeback


def build_dkv_plan(
    requests: Iterable[PlanRequest],
    owner_of_layer: Sequence[int],
    costs: StagingCostModel,
    *,
    group_size: int,
    ring_depth: int,
) -> DkvPlan:
    """Build the data-plane plan of one iteration.

    A pure function of its arguments: calling it on every rank with the same global batch gives
    equal plans, whatever the order of ``requests``. Only the costs of requests that are not
    dummies are asked of ``costs``.

    Args:
        requests: The global scheduled batch of all ranks, dummies included.
        owner_of_layer: The owner rank of every layer, as ``compute_ownership`` returns it.
        costs: The buffers of every layer and their sizes.
        group_size: Number of ranks of the DKV group.
        ring_depth: Slots of the staging ring, at least 1.

    Returns:
        The plan with one message per (step, compute rank) that moves bytes.

    Raises:
        ValueError: The geometry or the batch is inconsistent: a rank outside the group, a repeated
            request id, a negative history or chunk, or a cost model that names a buffer twice or
            not by a string, or returns a non-positive page size or a negative page count.
    """
    owners = tuple(owner_of_layer)
    _check_geometry(owners, group_size, ring_depth)
    batch = _canonical_requests(requests, group_size)

    real_by_rank: list[list[PlanRequest]] = [[] for _ in range(group_size)]
    for request in batch:
        if not request.is_dummy:
            real_by_rank[request.compute_rank].append(request)

    fetches: list[tuple[Transfer, ...]] = []
    writebacks: list[tuple[Transfer, ...]] = []
    for layer, owner in enumerate(owners):
        buffers = tuple(costs.buffers_of(layer))
        _check_buffers(layer, buffers)
        layer_fetches: list[Transfer] = []
        layer_writebacks: list[Transfer] = []
        for compute, real in enumerate(real_by_rank):
            if not real:
                continue
            fetch_segments, writeback_segments = _segments(costs, layer, buffers, real)
            if fetch_segments:
                layer_fetches.append(
                    Transfer(Direction.FETCH, layer, owner, compute, tuple(fetch_segments))
                )
            if writeback_segments:
                layer_writebacks.append(
                    Transfer(Direction.WRITEBACK, layer, owner, compute, tuple(writeback_segments))
                )
        fetches.append(tuple(layer_fetches))
        writebacks.append(tuple(layer_writebacks))

    steps = tuple(
        PlanStep(
            step_id.direction,
            step_id.layer,
            step_id.issue_point,
            (fetches if step_id.direction is Direction.FETCH else writebacks)[step_id.layer],
        )
        for step_id in global_step_order(len(owners), ring_depth)
    )
    return DkvPlan(group_size, ring_depth, owners, batch, steps)


def plan_canonical_form(plan: DkvPlan) -> tuple[object, ...]:
    """The plan as nested tuples of integers, booleans and strings only.

    The form does not depend on the process: no hash of a string or an object, no set, no
    mapping. ``plan_fingerprint`` hashes it, and it can be compared to find where two plans
    differ.
    """
    return (
        "dkv_plan",
        plan.group_size,
        plan.ring_depth,
        plan.owner_of_layer,
        tuple(
            (
                request.request_id,
                request.compute_rank,
                request.context_current_position,
                request.context_chunk_size,
                request.is_dummy,
            )
            for request in plan.requests
        ),
        tuple(
            (
                int(step.direction),
                step.layer,
                step.issue_point,
                tuple(
                    (
                        transfer.owner,
                        transfer.compute,
                        tuple(
                            (segment.request_id, segment.buffer, segment.pages, segment.nbytes)
                            for segment in transfer.segments
                        ),
                    )
                    for transfer in step.transfers
                ),
            )
            for step in plan.steps
        ),
    )


def plan_fingerprint(plan: DkvPlan) -> str:
    """A digest of the plan that is identical on every rank that built an equal plan.

    It is a sha256 over the representation of ``plan_canonical_form``, not Python's ``hash``,
    which differs between processes. Ranks compare fingerprints to check that they agree on the
    plan.
    """
    return hashlib.sha256(repr(plan_canonical_form(plan)).encode()).hexdigest()
