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
"""Moves the KV of a layer between the cache manager and the staging area around its attention.

The attention backend calls ``on_layer(l)`` at the top of every layer and ``end_forward()`` after
the last one. ``LoopbackStreamer`` serves a rank that owns every layer: the cached pages of the
layer are copied into the slot before its attention runs and the pages the new tokens wrote are
copied back after it, all on the stream of the forward pass, so the copies are ordered with the
kernels and need no events.

``DkvStreamer`` serves the ``layer_split`` layout, where the KV of a layer lives on the rank that
owns the layer and is computed on another. It carries out a ``DkvPlan`` on a data stream of its own:
it packs the pages of a message, sends it, receives the messages of its peers and unpacks them, in
the order the plan fixes for every rank, and it orders the data stream with the stream of the
forward pass by events.
"""

import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import NamedTuple, Protocol

import torch

from .dkv_plan import (
    DEADLINE_OF_KIND,
    DeadlineClass,
    Direction,
    DkvPlan,
    OpAction,
    PageCost,
    PlanRequest,
    PlanStep,
    RankOp,
    Transfer,
    build_dkv_plan,
    layer_types_from_compress_ratios,
    step_label,
)
from .dkv_staging import (
    BAD_PAGE_INDEX,
    DkvPageCopier,
    DkvStagedKvView,
    RequestSpan,
    StagingComponent,
    StagingKind,
    StagingLayout,
    pool_page_addresses,
)
from .dkv_transport import DkvTransport


class LoopbackStreamer:
    """Fetches and writes back the pages of one layer at a time, between the staging area and the
    cache manager of the same rank."""

    def __init__(
        self,
        manager,
        view: DkvStagedKvView,
        copier: DkvPageCopier | None = None,
        *,
        fill: str = "",
    ) -> None:
        """``fill`` (``"zero"`` or ``"nan"``, empty: none) overwrites the slots of a layer before
        its pages are fetched."""
        self._manager = manager
        self._fill = fill
        self._view = view
        self._layout = view.layout
        self._pool = view.pool
        self._copier = copier or DkvPageCopier()
        self._request_ids: list[int] = []
        self._history: list[int] = []
        self._new: list[int] = []
        self._spans: dict[StagingKind, tuple[RequestSpan, ...]] = {}
        self._written_layer: int | None = None
        self.bytes_fetched = 0
        self.bytes_written_back = 0

    def begin_iteration(
        self,
        request_ids: Sequence[int],
        history: Sequence[int],
        new: Sequence[int],
        spans: dict[StagingKind, tuple[RequestSpan, ...]],
    ) -> None:
        """Remember the staged batch; its pages are copied layer by layer."""
        self._request_ids = list(request_ids)
        self._history = list(history)
        self._new = list(new)
        self._spans = spans
        self._written_layer = None

    def on_layer(self, layer: int) -> None:
        """Called at the top of a layer: return the previous layer's pages, bring this layer's in."""
        if self._written_layer is not None:
            self._write_back(self._written_layer)
        if self._fill:
            self._pool.fill_layer(layer, self._fill)
        self._fetch(layer)
        self._written_layer = layer

    def end_forward(self) -> None:
        """Called after the last layer: return its pages."""
        if self._written_layer is not None:
            self._write_back(self._written_layer)
            self._written_layer = None

    def _pages(self, layer: int, fetch: bool) -> dict:
        """The (pool address, staging address) pairs of every component of ``layer``."""
        layout, pool, manager = self._layout, self._pool, self._manager
        stream_pairs: dict = {}
        for kind in layout.kinds:
            if layer not in layout.layers_of(kind):
                continue
            components = [c for c in layout.components if c.kind is kind]
            for index, request_id in enumerate(self._request_ids):
                if request_id not in manager.kv_cache_map:
                    continue
                history, new = self._history[index], self._new[index]
                span = self._spans[kind][index]
                first, count = (layout.fetch_range if fetch else layout.writeback_range)(
                    kind, history, new
                )
                if not count:
                    continue
                table = layout.block_table(kind, layer, span, first + count)
                for component in components:
                    indices = manager.get_cache_indices(request_id, layer, component.attention_type)
                    addresses = pool_page_addresses(
                        manager, layer, component.attention_type, indices
                    )
                    base = pool.layer_pointer(layer, component)
                    page_bytes = layout.page_bytes(component)
                    pairs = stream_pairs.setdefault(component, [])
                    for block in range(first, first + count):
                        if (
                            table[block] == BAD_PAGE_INDEX
                            or block >= len(addresses)
                            or addresses[block] is None
                        ):
                            continue
                        pairs.append((addresses[block], base + table[block] * page_bytes))
        return stream_pairs

    def _fetch(self, layer: int) -> None:
        stream = torch.cuda.current_stream().cuda_stream
        for component, pairs in self._pages(layer, fetch=True).items():
            if pairs:
                page_bytes = self._layout.page_bytes(component)
                self._copier.gather(
                    [pool for pool, _ in pairs],
                    [staging for _, staging in pairs],
                    page_bytes,
                    stream,
                )
                self.bytes_fetched += page_bytes * len(pairs)

    def _write_back(self, layer: int) -> None:
        stream = torch.cuda.current_stream().cuda_stream
        for component, pairs in self._pages(layer, fetch=False).items():
            if pairs:
                page_bytes = self._layout.page_bytes(component)
                self._copier.scatter(
                    [staging for _, staging in pairs],
                    [pool for pool, _ in pairs],
                    page_bytes,
                    stream,
                )
                self.bytes_written_back += page_bytes * len(pairs)


class LayoutCostModel:
    """The sizes of the staging layout in the form the plan asks for them."""

    def __init__(self, layout: StagingLayout) -> None:
        self._layout = layout

    def page_cost(self, layer: int, kind: StagingKind, history: int, chunk: int) -> PageCost:
        layout = self._layout
        page_bytes = layout.kind_page_bytes(kind)
        if chunk < 1:
            return PageCost(page_bytes, 0, 0)
        return PageCost(
            page_bytes,
            layout.fetch_range(kind, history, chunk)[1],
            layout.writeback_range(kind, history, chunk)[1],
        )


def max_message_bytes(layout: StagingLayout) -> int:
    """An upper bound of the size of any message of an iteration.

    A message holds pages of the kinds of one deadline class of the requests that one rank
    computes, and those pages fit the slots of the kinds, so the slots of a class bound it.
    """
    per_deadline: dict = {}
    for kind in layout.kinds:
        deadline = DEADLINE_OF_KIND[kind]
        per_deadline[deadline] = per_deadline.get(deadline, 0) + layout.slot_pages(
            kind
        ) * layout.kind_page_bytes(kind)
    return max(per_deadline.values(), default=0)


@dataclass
class DataPlaneStats:
    """What the data plane of one rank has moved since its streamer was built."""

    iterations: int = 0
    messages_sent: int = 0
    messages_received: int = 0
    bytes_sent: int = 0
    bytes_received: int = 0
    local_copies: int = 0
    bytes_local: int = 0


class _MessagePage(NamedTuple):
    """One page of a message: a block of a request in one buffer of a kind."""

    component: StagingComponent
    request_id: int
    block: int
    offset: int


# The multiplier of the position of a word in the checksum: 2**64 divided by the golden ratio.
_CHECKSUM_STRIDE = -7046029254386353131


def message_checksum(data: torch.Tensor) -> torch.Tensor:
    """A 64-bit checksum of ``data``, a tensor of bytes whose size is a multiple of 8.

    The words are combined with their position, so exchanged words change it, and a changed word
    always does, since the exclusive or with a constant is a bijection of the word.
    """
    words = data.view(torch.int64)
    positions = torch.arange(1, words.numel() + 1, dtype=torch.int64, device=data.device)
    return (words ^ (positions * _CHECKSUM_STRIDE)).sum()


class DataPlaneDebug(Protocol):
    """The checks of the data plane that run on its stream and cost a pass over every message."""

    def checksum(self, key: tuple, role: str, data: torch.Tensor) -> None:
        """Record the checksum of ``data`` for the message ``key`` in the role ``role``.

        The checksum is computed on the data stream, when it gets to it.
        """
        ...

    def corrupt(self, data: torch.Tensor, kind: str) -> None:
        """Damage ``data`` on the data stream: ``"flip"`` one byte or ``"zero"`` all of them."""
        ...

    def collect(self) -> list[tuple[tuple, str, int]]:
        """The ``(message, role, checksum)`` records of the finished iteration; clears them."""
        ...


class DeviceDataPlaneDebug:
    """``DataPlaneDebug`` for CUDA tensors: the checks run on ``stream``."""

    def __init__(self, stream) -> None:
        self._stream = stream
        self._pending: list[tuple[tuple, str, torch.Tensor]] = []

    def checksum(self, key: tuple, role: str, data: torch.Tensor) -> None:
        with torch.cuda.stream(self._stream):
            self._pending.append((key, role, message_checksum(data)))

    def corrupt(self, data: torch.Tensor, kind: str) -> None:
        with torch.cuda.stream(self._stream):
            if kind == "zero":
                data.zero_()
            else:
                data[data.numel() // 2].bitwise_xor_(0xFF)

    def collect(self) -> list[tuple[tuple, str, int]]:
        pending, self._pending = self._pending, []
        if not pending:
            return []
        # The data stream is done: the host has seen the event that ends the iteration.
        values = torch.stack([value for _, _, value in pending]).tolist()
        return [(key, role, int(value)) for (key, role, _), value in zip(pending, values)]


_FAULT_KINDS = ("flip", "zero")
_DIRECTIONS = {"fetch": Direction.FETCH, "writeback": Direction.WRITEBACK}


@dataclass(frozen=True)
class DataPlaneFault:
    """A message to damage after it was received, to see that the checksums notice.

    Attributes:
        iteration: The iteration of the executor.
        layer: The layer whose message is damaged.
        direction: Whether the damaged message is a fetch or a writeback.
        kind: ``"flip"`` one byte or ``"zero"`` the message.
        rank: The rank that receives the message; ``None``: whichever receives it.
    """

    iteration: int
    layer: int
    direction: Direction
    kind: str = "flip"
    rank: int | None = None

    @classmethod
    def parse(cls, text: str) -> "DataPlaneFault":
        """Read ``iteration:layer:fetch|writeback[:flip|zero[:rank]]``."""
        parts = text.split(":")
        try:
            if not 3 <= len(parts) <= 5 or parts[2] not in _DIRECTIONS:
                raise ValueError
            kind = parts[3] if len(parts) > 3 else "flip"
            if kind not in _FAULT_KINDS:
                raise ValueError
            return cls(
                int(parts[0]),
                int(parts[1]),
                _DIRECTIONS[parts[2]],
                kind,
                int(parts[4]) if len(parts) > 4 else None,
            )
        except ValueError:
            raise ValueError(
                f"{text!r} is not iteration:layer:fetch|writeback[:flip|zero[:rank]]"
            ) from None


def message_key(transfer: Transfer) -> tuple[int, int, int, int, int]:
    """What names a message in every iteration: the same on its two ends."""
    return (
        int(transfer.direction),
        transfer.layer,
        int(transfer.deadline),
        transfer.owner,
        transfer.compute,
    )


def find_data_plane_mismatches(
    records_by_rank: Sequence[Sequence[tuple[tuple, str, int]]], plan: DkvPlan | None = None
) -> list[str]:
    """Describe the messages of an iteration whose checksums disagree.

    A message that crossed ranks has the checksum of what the sender packed, of what the receiver
    got, and of the pages the receiver unpacked it into read back; a message that was copied on one
    rank has the checksum of the source pages and of the destination pages. Each of them has to be
    equal. A message that one end recorded and the other did not, which means that the two ranks
    did not run the same plan, is reported as well.
    """
    roles_of: dict[tuple, dict[str, list[tuple[int, int]]]] = {}
    for rank, records in enumerate(records_by_rank):
        for key, role, value in records:
            roles_of.setdefault(tuple(key), {}).setdefault(role, []).append((rank, value))
    segments = {}
    if plan is not None:
        for step in plan.steps:
            for transfer in step.transfers:
                segments[message_key(transfer)] = transfer
    problems = []
    for key, roles in sorted(roles_of.items()):
        direction, layer, deadline, owner, compute = key
        name = f"{step_label(Direction(direction), layer)}/{DeadlineClass(deadline).name.lower()}"
        transfer = segments.get(key)
        requests = (
            sorted({segment.request_id for segment in transfer.segments}) if transfer else "?"
        )
        where = f"{name} of requests {requests} (owner rank {owner}, compute rank {compute})"
        if owner == compute:
            expected = {"source", "stored"}
        else:
            expected = {"sent", "received", "stored"}
        if set(roles) != expected or any(len(entries) != 1 for entries in roles.values()):
            problems.append(f"{where}: recorded {sorted(roles)} instead of {sorted(expected)}")
            continue
        values = {role: entries[0][1] for role, entries in roles.items()}
        reference = values["source" if owner == compute else "sent"]
        for role in sorted(expected):
            if values[role] != reference:
                rank = roles[role][0][0]
                problems.append(
                    f"{where}: rank {rank} has {role} {values[role]:#x}, not {reference:#x}"
                )
    return problems


class DkvStreamer:
    """Carries out the data plane of the ``layer_split`` layout around the layers of a forward pass.

    The rank computes the requests it was given and owns the layers ``owner_of_layer`` assigns to
    it. For each iteration the executor builds a ``DkvPlan`` (``plan_for``) from the replicated
    scheduling state and hands it over (``set_plan``). The attention backend then calls
    ``begin_iteration`` (through the staged view), ``on_layer`` at the top of every layer and
    ``end_forward`` after the last one, and the executor calls ``drain`` when the iteration ends.

    Everything the plan lists for this rank runs on ``data_stream``, one operation after the other
    in the order of the plan: the owner packs the pages of a message from its cache manager and
    sends them, the compute rank receives them and unpacks them into the slot of the layer, and the
    other way round for the pages the new tokens wrote. The data stream waits for the forward pass
    by events, and the forward pass waits for the fetch of a layer:

    * the first operation of an iteration waits for the pages of the cache manager to be ready;
    * the operations at the top of layer ``l`` wait for the kernels of the layers before ``l``, so
      the writeback sends what they produced and the prefetch may overwrite their slots;
    * the attention of layer ``l`` waits for the fetch of layer ``l``.

    ``drain`` waits on the host for the data stream. After it returns the cache manager may free or
    move any page.

    No call blocks the host except ``drain`` (and the transport, if it has to). A rank without a
    plan, as in a warm-up or profiling forward pass, does nothing.
    """

    # How often drain polls the data stream, in seconds.
    _POLL_SECONDS = 0.0005

    def __init__(
        self,
        manager,
        view: DkvStagedKvView,
        transport: DkvTransport,
        owner_of_layer: Sequence[int],
        rank: int,
        *,
        group_size: int,
        copier: DkvPageCopier | None = None,
        data_stream=None,
        current_stream: Callable[[], object] | None = None,
        fill: str = "",
        debug: DataPlaneDebug | None = None,
        fault: DataPlaneFault | None = None,
    ) -> None:
        """Build the streamer of ``rank``.

        Args:
            manager: The cache manager that holds the pages of the layers this rank owns.
            view: The staged view whose pool holds the slots of the forward pass.
            transport: Sends and receives the messages.
            owner_of_layer: The owner rank of every layer of the model.
            rank: This rank in the group.
            group_size: The number of ranks of the group.
            copier: Packs and unpacks the pages.
            data_stream: The CUDA stream of the data plane; a new one by default.
            current_stream: Returns the stream of the forward pass; the current stream by default.
            fill: ``"zero"`` or ``"nan"`` overwrites a layer's slots before they are fetched.
            debug: Checksums every message at both ends and the pages it came from or went to;
                ``drain`` returns the records for the executor to compare across the ranks.
            fault: A message to damage after it was received; it needs ``debug``.
        """
        layout = view.layout
        owners = tuple(owner_of_layer)
        if len(owners) != len(layout.geometry.compress_ratios):
            raise ValueError(
                f"the ownership table has {len(owners)} layers, the model "
                f"{len(layout.geometry.compress_ratios)}"
            )
        if not 0 <= rank < group_size:
            raise ValueError(f"rank {rank} is not in a group of {group_size} ranks")
        if fault is not None and debug is None:
            raise ValueError("a data plane fault is only noticed by the checksums of debug")
        self._manager = manager
        self._layout = layout
        self._pool = view.pool
        self._transport = transport
        self._owners = owners
        self._rank = rank
        self._group_size = group_size
        self._copier = copier or DkvPageCopier()
        self._data_stream = data_stream if data_stream is not None else torch.cuda.Stream()
        self._current_stream = current_stream or torch.cuda.current_stream
        self._fill = fill
        self._costs = LayoutCostModel(layout)
        self._layer_types = layer_types_from_compress_ratios(layout.geometry.compress_ratios)
        size = max_message_bytes(layout)
        device = view.pool.buffer.device
        self._send_buffer = torch.empty(size, dtype=torch.uint8, device=device)
        self._recv_buffer = torch.empty(size, dtype=torch.uint8, device=device)
        self._debug = debug
        self._fault = fault
        # Where the pages of a message are read back to checksum them.
        self._scratch = torch.empty(size, dtype=torch.uint8, device=device) if debug else None
        self._iteration = 0
        self.last_plan: DkvPlan | None = None
        self.stats = DataPlaneStats()
        self._plan: DkvPlan | None = None
        self._requests: dict[int, PlanRequest] = {}
        self._computes = False
        self._begun = False
        self._local_index: dict[int, int] = {}
        self._spans: dict[StagingKind, tuple[RequestSpan, ...]] = {}
        self._fetch_done: dict[int, object] = {}
        self._data_done = None
        self._indices: dict[tuple[int, int, object], Sequence[int]] = {}
        self._page_tables: dict[tuple[int, object], tuple[int, int]] = {}

    # ---- the plan -------------------------------------------------------------------------

    def plan_for(self, requests: Iterable[PlanRequest]) -> DkvPlan:
        """The plan of an iteration whose global scheduled batch is ``requests``."""
        return build_dkv_plan(
            requests,
            self._owners,
            self._layer_types,
            self._costs,
            group_size=self._group_size,
            ring_depth=self._layout.geometry.ring_depth,
        )

    def set_plan(self, plan: DkvPlan, iteration: int = 0) -> None:
        """Carry out ``plan`` in the next forward pass, the pass of iteration ``iteration``.

        Raises:
            RuntimeError: The previous plan has not been carried out, or ``plan`` was built for a
                different group, ownership table or ring.
        """
        if self._plan is not None:
            raise RuntimeError("the plan of the previous forward pass was not carried out")
        if (
            plan.group_size != self._group_size
            or plan.owner_of_layer != self._owners
            or plan.ring_depth != self._layout.geometry.ring_depth
        ):
            raise RuntimeError(
                f"the plan is for {plan.group_size} ranks, ring depth {plan.ring_depth} and "
                f"owners {plan.owner_of_layer}, the streamer for {self._group_size} ranks, "
                f"ring depth {self._layout.geometry.ring_depth} and owners {self._owners}"
            )
        self._plan = plan
        self._iteration = iteration
        self._requests = {request.request_id: request for request in plan.requests}
        self._computes = any(
            request.compute_rank == self._rank and not request.is_dummy for request in plan.requests
        )
        self._begun = False
        self._local_index = {}
        self._spans = {}
        self._fetch_done = {}
        self._indices = {}
        self._page_tables = {}

    # ---- hooks of the forward pass --------------------------------------------------------

    def begin_iteration(
        self,
        request_ids: Sequence[int],
        history: Sequence[int],
        new: Sequence[int],
        spans: dict[StagingKind, tuple[RequestSpan, ...]],
    ) -> None:
        """Called once the requests of the forward pass are placed in the slots."""
        plan = self._plan
        if plan is None:
            return
        self._local_index = {request_id: index for index, request_id in enumerate(request_ids)}
        self._spans = spans
        for request in plan.requests:
            if request.compute_rank != self._rank or request.is_dummy:
                continue
            index = self._local_index.get(request.request_id)
            if (
                index is None
                or history[index] != request.context_current_position
                or new[index] != request.context_chunk_size
            ):
                staged = None if index is None else (history[index], new[index])
                raise RuntimeError(
                    f"rank {self._rank}: the plan has request {request.request_id} with "
                    f"{request.context_current_position} cached and {request.context_chunk_size} "
                    f"new tokens, the staged batch has (cached, new) = {staged}"
                )
        if not self._begun:
            self._begin()

    def on_layer(self, layer: int) -> None:
        """Called at the top of a layer, on the stream of the forward pass."""
        plan = self._plan
        if plan is None:
            return
        if not self._begun:
            self._begin()
        stream = self._current_stream()
        if layer >= 1 and self._computes:
            self._data_stream.wait_event(stream.record_event())
        for step in plan.layer_steps(layer):
            self._issue(step)
        fetched = self._fetch_done.pop(layer, None)
        if fetched is not None:
            stream.wait_event(fetched)

    def end_forward(self) -> None:
        """Called after the last layer: send back what the last layer produced."""
        plan = self._plan
        if plan is None:
            return
        if not self._begun:
            self._begin()
        if self._computes:
            self._data_stream.wait_event(self._current_stream().record_event())
        for step in plan.end_steps():
            self._issue(step)
        self._data_done = self._data_stream.record_event()
        self.last_plan = plan
        self._plan = None
        self.stats.iterations += 1

    def assert_idle(self, what: str) -> None:
        """Raise unless the data plane of the last iteration is done and no iteration is open.

        A page may only be freed or moved by the cache manager while the data plane is idle.

        Raises:
            RuntimeError: ``what`` is about to happen while the data plane is still running.
        """
        if self._plan is not None or self._data_done is not None:
            raise RuntimeError(
                f"DKV invariant violation: {what} while the data plane of rank {self._rank} "
                "is not drained"
            )

    def drain(self, timeout: float) -> list[tuple[tuple, str, int]]:
        """Wait until the data stream has finished the last forward pass.

        Returns:
            The checksum records of the iteration when the streamer is in debug, else nothing.

        Raises:
            RuntimeError: The data stream is not done after ``timeout`` seconds, which means that a
                peer does not take part in the plan or the interconnect failed.
        """
        done, self._data_done = self._data_done, None
        if done is None:
            return []
        deadline = time.monotonic() + timeout
        while not done.query():
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    f"rank {self._rank}: the DKV data plane did not finish within {timeout} s"
                )
            time.sleep(self._POLL_SECONDS)
        return self._debug.collect() if self._debug is not None else []

    # ---- issuing the steps ----------------------------------------------------------------

    def _begin(self) -> None:
        self._begun = True
        self._data_stream.wait_event(self._current_stream().record_event())
        for step in self._plan.begin_steps():
            self._issue(step)

    def _issue(self, step: PlanStep) -> None:
        fetch = step.direction is Direction.FETCH
        if fetch and self._computes and self._fill:
            with torch.cuda.stream(self._data_stream):
                self._pool.fill_layer(step.layer, self._fill)
        for op in step.ops_for(self._rank):
            self._run(op)
        # A rank that computes waits for the fetch of a layer also when the layer has nothing to
        # fetch: the step follows the writeback of the layer whose slot this layer computes in, so
        # the layer must not start before the data stream has got past it.
        if fetch and self._computes:
            self._fetch_done[step.layer] = self._data_stream.record_event()

    def _run(self, op: RankOp) -> None:
        transfer = op.transfer
        pages = self._message_pages(transfer)
        key = message_key(transfer)
        if op.action is OpAction.LOCAL:
            fetch = transfer.direction is Direction.FETCH
            pool = [self._pool_address(transfer.layer, page) for page in pages]
            staging = [self._staging_address(transfer.layer, page) for page in pages]
            source, destination = (pool, staging) if fetch else (staging, pool)
            if self._debug is not None:
                self._checksum_pages(key, "source", transfer.nbytes, pages, source)
            self._copy(
                [(page.component, to, at) for page, to, at in zip(pages, destination, source)]
            )
            if self._debug is not None:
                self._checksum_pages(key, "stored", transfer.nbytes, pages, destination)
            self.stats.local_copies += 1
            self.stats.bytes_local += transfer.nbytes
            return
        sending = op.action is OpAction.SEND
        page_address = self._pool_address if op.rank == transfer.owner else self._staging_address
        buffer = self._send_buffer if sending else self._recv_buffer
        message = buffer[: transfer.nbytes]
        base = buffer.data_ptr()
        page_at = [page_address(transfer.layer, page) for page in pages]
        if sending:
            self._copy(
                [(page.component, base + page.offset, at) for page, at in zip(pages, page_at)]
            )
            if self._debug is not None:
                self._debug.checksum(key, "sent", message)
            self._transport.send(message, op.peer, self._data_stream)
            self.stats.messages_sent += 1
            self.stats.bytes_sent += transfer.nbytes
        else:
            self._transport.recv(message, op.peer, self._data_stream)
            if self._fault is not None and self._is_faulty(transfer):
                self._debug.corrupt(message, self._fault.kind)
            if self._debug is not None:
                self._debug.checksum(key, "received", message)
            self._copy(
                [(page.component, at, base + page.offset) for page, at in zip(pages, page_at)]
            )
            if self._debug is not None:
                self._checksum_pages(key, "stored", transfer.nbytes, pages, page_at)
            self.stats.messages_received += 1
            self.stats.bytes_received += transfer.nbytes

    def _is_faulty(self, transfer: Transfer) -> bool:
        fault = self._fault
        return (
            fault.iteration == self._iteration
            and fault.layer == transfer.layer
            and fault.direction is transfer.direction
            and fault.rank in (None, self._rank)
        )

    def _checksum_pages(
        self, key: tuple, role: str, nbytes: int, pages: Sequence[_MessagePage], addresses
    ) -> None:
        """Read the pages at ``addresses`` back into the order of a message and checksum them."""
        base = self._scratch.data_ptr()
        self._copy([(page.component, base + page.offset, at) for page, at in zip(pages, addresses)])
        self._debug.checksum(key, role, self._scratch[:nbytes])

    def _copy(self, pairs: Sequence[tuple[StagingComponent, int, int]]) -> None:
        """Copy the pages ``(component, destination, source)``, one launch per page size."""
        by_size: dict[int, list[tuple[int, int]]] = {}
        for component, destination, source in pairs:
            by_size.setdefault(self._layout.page_bytes(component), []).append((destination, source))
        stream = self._data_stream.cuda_stream
        for page_bytes, group in by_size.items():
            self._copier.copy(group, page_bytes, stream)

    # ---- where the pages are --------------------------------------------------------------

    def _message_pages(self, transfer: Transfer) -> list[_MessagePage]:
        """The pages of a message in the order they are laid out, buffer by buffer in a segment."""
        layout = self._layout
        fetch = transfer.direction is Direction.FETCH
        pages: list[_MessagePage] = []
        offset = 0
        for segment in transfer.segments:
            request = self._requests[segment.request_id]
            history, chunk = request.context_current_position, request.context_chunk_size
            span_of = layout.fetch_range if fetch else layout.writeback_range
            first, count = span_of(segment.kind, history, chunk)
            if count != segment.pages:
                raise RuntimeError(
                    f"rank {self._rank}: {transfer.label} of request {segment.request_id} has "
                    f"{segment.pages} {segment.kind.label} pages in the plan, the layout {count}"
                )
            for component in layout.components_of(segment.kind):
                page_bytes = layout.page_bytes(component)
                for block in range(first, first + count):
                    pages.append(_MessagePage(component, segment.request_id, block, offset))
                    offset += page_bytes
        if offset != transfer.nbytes:
            raise RuntimeError(
                f"rank {self._rank}: {transfer.label} has {transfer.nbytes} bytes in the plan, "
                f"the layout {offset}"
            )
        return pages

    def _pool_address(self, layer: int, page: _MessagePage) -> int:
        """The address of a page in the cache manager of this rank."""
        attention_type = page.component.attention_type
        key = (page.request_id, layer, attention_type)
        indices = self._indices.get(key)
        if indices is None:
            indices = self._manager.get_cache_indices(page.request_id, layer, attention_type)
            self._indices[key] = indices
        if page.block >= len(indices) or indices[page.block] == BAD_PAGE_INDEX:
            raise RuntimeError(
                f"rank {self._rank}: the cache manager has no page for block {page.block} of "
                f"request {page.request_id} in layer {layer} ({attention_type.name}), which the "
                "plan moves"
            )
        table = self._page_tables.get((layer, attention_type))
        if table is None:
            buffer = self._manager.get_buffers(layer, attention_type)
            table = (buffer.data_ptr(), buffer.stride(0) * buffer.element_size())
            self._page_tables[layer, attention_type] = table
        return table[0] + indices[page.block] * table[1]

    def _staging_address(self, layer: int, page: _MessagePage) -> int:
        """The address of a page in the slot of ``layer`` of this rank."""
        layout = self._layout
        component = page.component
        index = self._local_index.get(page.request_id)
        if index is None:
            raise RuntimeError(
                f"rank {self._rank}: request {page.request_id} is not in the staged batch"
            )
        span = self._spans[component.kind][index]
        if not span.first_block <= page.block < span.first_block + span.num_pages:
            raise RuntimeError(
                f"rank {self._rank}: block {page.block} of request {page.request_id} is outside "
                f"the staged blocks {span.first_block}..{span.first_block + span.num_pages - 1} "
                f"of {component.kind.label}"
            )
        entry = (
            layout.slot_page_offset(layer, component.kind)
            + span.page_offset
            + page.block
            - span.first_block
        )
        return self._pool.layer_pointer(layer, component) + entry * layout.page_bytes(component)
