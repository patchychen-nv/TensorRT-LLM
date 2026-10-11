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
forward pass by events. The plan is resolved to device addresses once per iteration, when the
staged batch is known, so that the hooks of the forward pass only enqueue work.
"""

import functools
import os
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import NamedTuple, Protocol

import numpy as np
import torch

from .dkv_plan import (
    DEADLINE_OF_KIND,
    ISSUE_AT_BEGIN,
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
    """The sizes of the staging layout in the form the plan asks for them.

    The cost of a kind depends on the history and the chunk of the request only, so the costs of
    the requests of an iteration are computed once and looked up for every layer.
    """

    # Costs remembered before the table is emptied; a few per request and kind are in use.
    _MAX_REMEMBERED = 4096

    def __init__(self, layout: StagingLayout) -> None:
        self._layout = layout
        self._costs: dict[tuple[StagingKind, int, int], PageCost] = {}

    def page_cost(self, layer: int, kind: StagingKind, history: int, chunk: int) -> PageCost:
        key = (kind, history, chunk)
        cost = self._costs.get(key)
        if cost is None:
            layout = self._layout
            page_bytes = layout.kind_page_bytes(kind)
            if chunk < 1:
                cost = PageCost(page_bytes, 0, 0)
            else:
                cost = PageCost(
                    page_bytes,
                    layout.fetch_range(kind, history, chunk)[1],
                    layout.writeback_range(kind, history, chunk)[1],
                )
            if len(self._costs) >= self._MAX_REMEMBERED:
                self._costs.clear()
            self._costs[key] = cost
        return cost


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


def max_peer_step_bytes(layout: StagingLayout) -> int:
    """An upper bound of the bytes one rank exchanges with one peer in one step.

    The messages of a step to one peer are the deadline classes of one layer, each at most once,
    and their pages fit the slots of the kinds, so all the slots together bound them.
    """
    return sum(layout.slot_pages(kind) * layout.kind_page_bytes(kind) for kind in layout.kinds)


def arena_peers(group_size: int) -> int:
    """The peers that each of the two message arenas of a rank has room for, a step's worth each."""
    return max(group_size - 1, 1)


# Every message of a step starts at a multiple of this in the message arena.
_MESSAGE_ALIGNMENT = 256


@dataclass
class DataPlaneStats:
    """What the data plane of one rank has moved since its streamer was built, and what it cost.

    ``hook_seconds`` is the host time of the hooks of the forward pass, which run between the
    launches of its layers and so add to a pass that is bound by the host, and ``drain_seconds`` the
    host time that ``drain`` waited for the data stream, the part of the data plane that the
    forward pass did not hide. Parts of ``hook_seconds`` are told apart: ``compile_seconds``
    resolved the plan of an iteration to device addresses, ``copy_seconds`` was spent inside the
    page copies and ``transport_seconds`` inside the transport calls; the last two enqueue work
    on the GPU and wait only when its queue is full. With the C++ executor of the program,
    ``native_seconds`` is the time of its calls, which do the copies and the transport.
    ``fetch_wait_seconds`` is the GPU time the compute stream spent waiting for layer fetches,
    measured only with ``TRTLLM_DKV_WAIT_TIMING=1``.
    """

    iterations: int = 0
    messages_sent: int = 0
    messages_received: int = 0
    bytes_sent: int = 0
    bytes_received: int = 0
    local_copies: int = 0
    bytes_local: int = 0
    bytes_local_fetch: int = 0
    bytes_local_writeback: int = 0
    hook_seconds: float = 0.0
    compile_seconds: float = 0.0
    copy_seconds: float = 0.0
    transport_seconds: float = 0.0
    native_seconds: float = 0.0
    drain_seconds: float = 0.0
    fetch_wait_seconds: float = 0.0


def _timed_hook(method):
    """Add the host time of a hook of the forward pass to ``stats.hook_seconds``.

    A pass without a plan, as in a warm-up, does nothing and costs nothing.
    """

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        if self._plan is None:
            return method(self, *args, **kwargs)
        started = time.perf_counter()
        try:
            return method(self, *args, **kwargs)
        finally:
            self.stats.hook_seconds += time.perf_counter() - started

    return wrapper


class _Run(NamedTuple):
    """The pages of a message that one buffer of a kind holds for one request.

    The pages are the blocks ``first`` to ``first + count - 1`` of the request, and the message
    holds them back to back from byte ``offset``.
    """

    component: StagingComponent
    request_id: int
    first: int
    count: int
    offset: int


# One side of a copy of the pages of a run, as ``DkvPageCopier.copy_runs`` reads it: a base
# address, an address step and the host address of the int32 page indices of the blocks, or 0 when
# the pages follow each other.
_Side = tuple[int, int, int]


class _CompiledOp(NamedTuple):
    """One operation of the data plane with its copies resolved to device addresses."""

    op: RankOp
    key: tuple[int, int, int, int, int]
    # The region of the message arena the message occupies; None for a local copy.
    message: torch.Tensor | None
    # The copies the operation issues, as the runs of ``DkvPageCopier.copy_runs``. A send packs
    # pages into its message, a receive unpacks its message into pages and a local copy moves
    # pages between the cache manager and the staging area.
    copies: list[int]
    runs: tuple[_Run, ...]
    # Where the pages of every run are on this rank, as copy sides, for the checksums of debug:
    # the pages a message was packed from or unpacked into, or (source, destination) of a local
    # copy.
    pages: tuple[tuple[_Side, ...], ...]


class _CompiledStep(NamedTuple):
    """What one rank does in one step, resolved to addresses and grouped by kind of operation."""

    step: PlanStep
    # The layer whose slots are overwritten before its fetch (the debug fill), else None.
    fill_layer: int | None
    local: tuple[_CompiledOp, ...]
    sends: tuple[_CompiledOp, ...]
    recvs: tuple[_CompiledOp, ...]
    # Record the event the attention of the layer waits for, once the fetch is enqueued.
    record_fetch: bool


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
        iteration: The iteration of the executor; ``None``: every iteration.
        layer: The layer whose message is damaged.
        direction: Whether the damaged message is a fetch or a writeback.
        kind: ``"flip"`` one byte or ``"zero"`` the message.
        rank: The rank that receives the message; ``None``: whichever receives it.
    """

    iteration: int | None
    layer: int
    direction: Direction
    kind: str = "flip"
    rank: int | None = None

    @classmethod
    def parse(cls, text: str) -> "DataPlaneFault":
        """Read ``iteration|*:layer:fetch|writeback[:flip|zero[:rank]]``."""
        parts = text.split(":")
        try:
            if not 3 <= len(parts) <= 5 or parts[2] not in _DIRECTIONS:
                raise ValueError
            kind = parts[3] if len(parts) > 3 else "flip"
            if kind not in _FAULT_KINDS:
                raise ValueError
            return cls(
                None if parts[0] == "*" else int(parts[0]),
                int(parts[1]),
                _DIRECTIONS[parts[2]],
                kind,
                int(parts[4]) if len(parts) > 4 else None,
            )
        except ValueError:
            raise ValueError(
                f"{text!r} is not iteration|*:layer:fetch|writeback[:flip|zero[:rank]]"
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

    When the staged batch is known the plan is compiled: every operation of this rank is resolved
    to the device addresses of its pages and to its region of the message arena, step by step. The
    hooks then only enqueue the work of their steps on ``data_stream``: the owner packs the pages
    of its messages from its cache manager and sends them, the compute rank receives them and
    unpacks them into the slot of the layer, and the other way round for the pages the new tokens
    wrote. All the messages a rank sends or receives in one step are issued together when the
    transport can group them. The data stream waits for the forward pass by events, and the
    forward pass waits for the fetch of a layer:

    * the first operation of an iteration waits for the pages of the cache manager to be ready;
    * the operations at the top of layer ``l`` wait for the kernels of the layers before ``l``, so
      the writeback sends what they produced and the prefetch may overwrite their slots; a rank
      that computes nothing waits as well, which keeps it in step with the others;
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
        timing_event_factory: Callable[[], torch.cuda.Event] | None = None,
        group_messages: bool = True,
        native: bool = True,
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
            timing_event_factory: Creates a timing-enabled event for fetch waits. Used only with
                ``TRTLLM_DKV_WAIT_TIMING=1``; CUDA events by default.
            group_messages: Hand all the messages of a step to the transport in one call when it
                has a ``group_send_recv`` method that is not None; else one message at a time.
            native: Carry out the program with the C++ executor the transport makes with
                ``native_data_plane``, if it has one: a hook of the forward pass is then one call.
                Not with ``debug``, ``fault`` or ``fill``, which act on the messages and slots in
                Python, nor without grouped messages or the affine page map of the cache manager.
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
        self._group_send_recv = (
            getattr(transport, "group_send_recv", None) if group_messages else None
        )
        self._owners = owners
        self._rank = rank
        self._group_size = group_size
        self._copier = copier or DkvPageCopier()
        self._data_stream = data_stream if data_stream is not None else torch.cuda.Stream()
        self._current_stream = current_stream or torch.cuda.current_stream
        self._wait_timing = os.environ.get("TRTLLM_DKV_WAIT_TIMING") == "1"
        self._timing_event_factory = timing_event_factory or functools.partial(
            torch.cuda.Event, enable_timing=True
        )
        self._fetch_wait_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
        self._fill = fill
        self._costs = LayoutCostModel(layout)
        self._layer_types = layer_types_from_compress_ratios(layout.geometry.compress_ratios)
        # Every message of a step has its own region of the arena, so the messages of a step can
        # be packed first and issued together: a rank exchanges at most max_peer_step_bytes with
        # each of its peers in a step, in at most one message per deadline class.
        arena = arena_peers(group_size) * (
            max_peer_step_bytes(layout) + len(DeadlineClass) * _MESSAGE_ALIGNMENT
        )
        device = view.pool.buffer.device
        self._send_arena = torch.empty(arena, dtype=torch.uint8, device=device)
        self._recv_arena = torch.empty(arena, dtype=torch.uint8, device=device)
        self._debug = debug
        self._fault = fault
        # Where the pages of a message are read back to checksum them.
        self._scratch = (
            torch.empty(max_message_bytes(layout), dtype=torch.uint8, device=device)
            if debug
            else None
        )
        self._iteration = 0
        self.last_plan: DkvPlan | None = None
        self.stats = DataPlaneStats()
        self._plan: DkvPlan | None = None
        self._requests: dict[int, PlanRequest] = {}
        self._computes = False
        self._begun = False
        self._program: dict[int, list[_CompiledStep]] | None = None
        self._local_index: dict[int, int] = {}
        self._spans: dict[StagingKind, tuple[RequestSpan, ...]] = {}
        self._fetch_done: dict[int, object] = {}
        self._data_done = None
        # The events of the iteration, one per use, recorded again every iteration.
        self._events: dict[tuple, object] = {}
        # A cache manager that maps blocks to pages by an affine function of their base page index
        # gives the addresses of a run without a call per (request, layer, kind).
        self._affine_pages = callable(
            getattr(manager, "get_cache_index_affine", None)
        ) and callable(getattr(manager, "get_base_page_indices", None))
        self._affines: dict[tuple[int, object], tuple[int, int, int]] = {}
        self._base_indices: dict[tuple[int, int], np.ndarray] = {}
        self._indices: dict[tuple[int, int, object], np.ndarray] = {}
        self._page_tables: dict[tuple[int, object], tuple[int, int]] = {}
        make_native = getattr(transport, "native_data_plane", None) if native else None
        self._native = (
            make_native(len(owners), self._data_stream)
            if make_native is not None
            and debug is None
            and fault is None
            and not fill
            and self._group_send_recv is not None
            and self._affine_pages
            else None
        )
        self._native_static = False
        self._native_compiled = False
        self._native_kinds: dict[StagingKind, int] = {}
        self._native_components: dict[StagingComponent, int] = {}
        self._native_groups = 0
        # Per issue point: messages and bytes sent and received, local copies and their bytes.
        self._native_counts: dict[int, list[int]] = {}
        # The base page indices the executor reads while it builds the copies of the iteration.
        self._native_indices: list[np.ndarray] = []

    @property
    def uses_native(self) -> bool:
        """Whether the C++ executor carries out the program."""
        return self._native is not None

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
        self._program = None
        self._local_index = {}
        self._spans = {}
        self._fetch_done = {}
        self._base_indices = {}
        self._indices = {}
        self._native_compiled = False
        self._native_counts = {}
        self._native_indices = []

    # ---- hooks of the forward pass --------------------------------------------------------

    @_timed_hook
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

    @_timed_hook
    def on_layer(self, layer: int) -> None:
        """Called at the top of a layer, on the stream of the forward pass."""
        plan = self._plan
        if plan is None:
            return
        if not self._begun:
            self._begin()
        if self._native is not None:
            self._run_native(layer)
            return
        stream = self._current_stream()
        if layer >= 1:
            # Also on a rank that computes nothing: the host of an idle rank runs far ahead of its
            # GPU, and a receive that is enqueued without this wait starts at once and spins for a
            # message that is not sent before the compute rank reaches the layer. The spinning
            # kernel holds multiprocessors that the layers of this rank need to get through the MoE
            # exchange, which the sender of the message waits for.
            self._data_stream.wait_event(self._record(stream, ("compute", layer)))
        for compiled in self._program.get(layer, ()):
            self._issue(compiled)
        fetched = self._fetch_done.pop(layer, None)
        if fetched is not None:
            if self._wait_timing:
                start, end = self._timing_event_factory(), self._timing_event_factory()
                stream.record_event(start)
                stream.wait_event(fetched)
                stream.record_event(end)
                self._fetch_wait_events.append((start, end))
            else:
                stream.wait_event(fetched)

    @_timed_hook
    def end_forward(self) -> None:
        """Called after the last layer: send back what the last layer produced."""
        plan = self._plan
        if plan is None:
            return
        if not self._begun:
            self._begin()
        if self._native is not None:
            self._run_native(plan.num_layers)
        else:
            self._data_stream.wait_event(self._record(self._current_stream(), ("end",)))
            for compiled in self._program.get(plan.num_layers, ()):
                self._issue(compiled)
        self._data_done = self._record(self._data_stream, ("done",))
        self.last_plan = plan
        self._plan = None
        self._program = None
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
        started = time.monotonic()
        deadline = started + timeout
        while not done.query():
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    f"rank {self._rank}: the DKV data plane did not finish within {timeout} s"
                )
            time.sleep(self._POLL_SECONDS)
        self.stats.drain_seconds += time.monotonic() - started
        for start, end in self._fetch_wait_events:
            end.synchronize()
            self.stats.fetch_wait_seconds += start.elapsed_time(end) / 1000.0
        self._fetch_wait_events.clear()
        if self._native is not None:
            if self._wait_timing:
                self.stats.fetch_wait_seconds += self._native.fetch_wait_ms() / 1000.0
            self._native_indices = []
        return self._debug.collect() if self._debug is not None else []

    # ---- compiling the plan ---------------------------------------------------------------

    def _compile(self) -> None:
        """Resolve every operation of this rank to addresses, step by step."""
        started = time.perf_counter()
        program: dict[int, list[_CompiledStep]] = {}
        for step in self._plan.steps:
            fetch = step.direction is Direction.FETCH
            fill_layer = step.layer if fetch and self._computes and self._fill else None
            # A rank that computes waits for the fetch of a layer also when the layer has nothing
            # to fetch: the step follows the writeback of the layer whose slot this layer computes
            # in, so the layer must not start before the data stream has got past it.
            record_fetch = fetch and self._computes
            ops = step.ops_for(self._rank)
            if not ops and fill_layer is None and not record_fetch:
                continue
            local: list[_CompiledOp] = []
            sends: list[_CompiledOp] = []
            recvs: list[_CompiledOp] = []
            send_offset = recv_offset = 0
            for op in ops:
                if op.action is OpAction.LOCAL:
                    local.append(self._compile_local(op))
                elif op.action is OpAction.SEND:
                    compiled, send_offset = self._compile_message(op, self._send_arena, send_offset)
                    sends.append(compiled)
                else:
                    compiled, recv_offset = self._compile_message(op, self._recv_arena, recv_offset)
                    recvs.append(compiled)
            program.setdefault(step.issue_point, []).append(
                _CompiledStep(
                    step, fill_layer, tuple(local), tuple(sends), tuple(recvs), record_fetch
                )
            )
        self._program = program
        self.stats.compile_seconds += time.perf_counter() - started

    def _compile_local(self, op: RankOp) -> _CompiledOp:
        transfer = op.transfer
        layer = transfer.layer
        runs = tuple(self._message_runs(transfer))
        pool = tuple(self._pool_side(layer, run) for run in runs)
        staging = tuple(self._staging_side(layer, run) for run in runs)
        if transfer.direction is Direction.FETCH:
            sources, destinations = pool, staging
        else:
            sources, destinations = staging, pool
        return _CompiledOp(
            op,
            message_key(transfer),
            None,
            self._copy_runs(runs, destinations, sources),
            runs,
            (sources, destinations),
        )

    def _compile_message(
        self, op: RankOp, arena: torch.Tensor, offset: int
    ) -> tuple[_CompiledOp, int]:
        """Resolve a send or receive whose message starts at ``offset`` of ``arena``.

        Returns the operation and the offset of the next message of the step.
        """
        transfer = op.transfer
        layer = transfer.layer
        runs = tuple(self._message_runs(transfer))
        end = offset + transfer.nbytes
        if end > arena.numel():
            raise RuntimeError(
                f"rank {self._rank}: the messages of {step_label(transfer.direction, layer)} need "
                f"{end} bytes, the message arena has {arena.numel()}"
            )
        message = arena[offset:end]
        base = arena.data_ptr() + offset
        side_of = self._pool_side if op.rank == transfer.owner else self._staging_side
        pages = tuple(side_of(layer, run) for run in runs)
        in_message = tuple(self._message_side(base, run) for run in runs)
        if op.action is OpAction.SEND:
            copies = self._copy_runs(runs, in_message, pages)
        else:
            copies = self._copy_runs(runs, pages, in_message)
        compiled = _CompiledOp(op, message_key(transfer), message, copies, runs, (pages,))
        return compiled, -(-end // _MESSAGE_ALIGNMENT) * _MESSAGE_ALIGNMENT

    # ---- compiling the plan for the C++ executor ------------------------------------------

    def _set_native_static(self) -> None:
        """Hand the executor the addresses that stay the same: pools, slots, page sizes."""
        layout = self._layout
        num_layers = len(self._owners)
        self._native_kinds = {kind: index for index, kind in enumerate(layout.kinds)}
        self._native_components = {
            component: index for index, component in enumerate(layout.components)
        }
        attention_types = list(dict.fromkeys(c.attention_type for c in layout.components))
        attention_index = {
            attention_type: index for index, attention_type in enumerate(attention_types)
        }
        components = np.array(
            [
                [
                    self._native_kinds[component.kind],
                    attention_index[component.attention_type],
                    layout.page_bytes(component),
                ]
                for component in layout.components
            ],
            dtype=np.int64,
        ).reshape(len(layout.components), 3)
        pool = np.zeros((num_layers, len(attention_types), 3), dtype=np.int64)
        pool[:, :, 0] = -1
        staging = np.zeros((num_layers, len(layout.components)), dtype=np.int64)
        slots = np.zeros((num_layers, len(layout.kinds)), dtype=np.int64)
        groups = 0
        for component, index in self._native_components.items():
            kind = self._native_kinds[component.kind]
            for layer in layout.layers_of(component.kind):
                staging[layer, index] = self._pool.layer_pointer(layer, component)
                slots[layer, kind] = layout.slot_page_offset(layer, component.kind)
                if self._owners[layer] == self._rank:
                    group, step, origin = self._affine(layer, component.attention_type)
                    pool[layer, attention_index[component.attention_type]] = (group, step, origin)
                    groups = max(groups, group + 1)
        self._native.set_static(
            torch.from_numpy(components),
            torch.from_numpy(pool),
            torch.from_numpy(staging),
            torch.from_numpy(slots),
        )
        self._native_groups = groups
        self._native_static = True

    def _compile_native(self) -> None:
        """Compile the plan of the iteration into the tables of the C++ executor.

        The runs of the messages of a transfer depend only on its direction, the layer type, the
        deadline class and the compute rank, so they are built once per such template and placed
        in a layer and a message by the operations of the steps. The executor resolves them to
        addresses when it issues a step.
        """
        started = time.perf_counter()
        if not self._native_static:
            self._set_native_static()
        plan = self._plan
        layout = self._layout
        slots: dict[int, int] = {}
        requests: list[int] = []
        indices: list[np.ndarray] = []
        for request in plan.requests:
            if request.is_dummy:
                continue
            slots[request.request_id] = len(slots)
            requests.append(request.request_id)
            for group in range(self._native_groups):
                # A copy: the manager's table is valid only until the next lifecycle operation.
                table = np.array(
                    self._manager.get_base_page_indices(request.request_id, group), dtype=np.int32
                )
                indices.append(table)
                requests += (table.ctypes.data if len(table) else 0, len(table))
            index = (
                self._local_index.get(request.request_id)
                if request.compute_rank == self._rank
                else None
            )
            for kind in layout.kinds:
                if index is None:
                    requests += (-1, -1, -1)
                else:
                    span = self._spans[kind][index]
                    requests += (span.first_block, span.page_offset, span.num_pages)
        templates: dict[tuple, tuple[int, int]] = {}
        offsets = [0]
        runs: list[int] = []
        steps: list[int] = []
        ops: list[int] = []
        counts: dict[int, list[int]] = {}
        layer_types = self._layer_types
        for step in plan.steps:
            fetch = step.direction is Direction.FETCH
            record_fetch = fetch and self._computes
            step_ops = step.ops_for(self._rank)
            if not step_ops and not record_fetch:
                continue
            first = len(ops) // 5
            send_offset = recv_offset = 0
            tally = counts.setdefault(step.issue_point, [0] * 8)
            for op in step_ops:
                transfer = op.transfer
                key = (
                    transfer.direction,
                    layer_types[transfer.layer],
                    transfer.deadline,
                    transfer.compute,
                )
                template = templates.get(key)
                if template is None:
                    template = (len(templates), self._native_template(transfer, slots, runs))
                    templates[key] = template
                    offsets.append(len(runs) // 5)
                nbytes = transfer.nbytes
                if template[1] != nbytes:
                    raise RuntimeError(
                        f"rank {self._rank}: {transfer.label} has {nbytes} bytes in the plan, "
                        f"other layers of its type {template[1]}"
                    )
                if op.action is OpAction.LOCAL:
                    offset = 0
                    tally[4] += 1
                    tally[5] += nbytes
                    tally[6 if fetch else 7] += nbytes
                elif op.action is OpAction.SEND:
                    offset = send_offset
                    send_offset = self._next_message(send_offset, transfer, self._send_arena)
                    tally[0] += 1
                    tally[1] += nbytes
                else:
                    offset = recv_offset
                    recv_offset = self._next_message(recv_offset, transfer, self._recv_arena)
                    tally[2] += 1
                    tally[3] += nbytes
                ops += (int(op.action), template[0], op.peer, offset, nbytes)
            steps += (
                step.issue_point,
                int(step.direction),
                step.layer,
                int(record_fetch),
                first,
                len(step_ops),
            )
        width = 1 + 2 * self._native_groups + 3 * len(layout.kinds)
        self._native.set_iteration(
            torch.from_numpy(np.array(requests, dtype=np.int64).reshape(-1, width)),
            torch.from_numpy(np.array(offsets, dtype=np.int64)),
            torch.from_numpy(np.array(runs, dtype=np.int64).reshape(-1, 5)),
            torch.from_numpy(np.array(steps, dtype=np.int64).reshape(-1, 6)),
            torch.from_numpy(np.array(ops, dtype=np.int64).reshape(-1, 5)),
            self._send_arena.data_ptr(),
            self._recv_arena.data_ptr(),
        )
        self._native_indices = indices
        self._native_counts = counts
        self._native_compiled = True
        self.stats.compile_seconds += time.perf_counter() - started

    def _native_template(self, transfer: Transfer, slots: dict[int, int], runs: list[int]) -> int:
        """Append the runs of the messages of ``transfer`` to ``runs``; returns their bytes."""
        layout = self._layout
        span_of = (
            layout.fetch_range if transfer.direction is Direction.FETCH else layout.writeback_range
        )
        offset = 0
        for segment in transfer.segments:
            request = self._requests[segment.request_id]
            first, count = span_of(
                segment.kind, request.context_current_position, request.context_chunk_size
            )
            if count != segment.pages:
                raise RuntimeError(
                    f"rank {self._rank}: {transfer.label} of request {segment.request_id} has "
                    f"{segment.pages} {segment.kind.label} pages in the plan, the layout {count}"
                )
            if count == 0:
                continue
            slot = slots[segment.request_id]
            for component in layout.components_of(segment.kind):
                runs += (self._native_components[component], slot, first, count, offset)
                offset += count * layout.page_bytes(component)
        if offset != transfer.nbytes:
            raise RuntimeError(
                f"rank {self._rank}: {transfer.label} has {transfer.nbytes} bytes in the plan, "
                f"the layout {offset}"
            )
        return offset

    def _next_message(self, offset: int, transfer: Transfer, arena: torch.Tensor) -> int:
        """Where the message after ``transfer``'s, which starts at ``offset``, starts."""
        end = offset + transfer.nbytes
        if end > arena.numel():
            raise RuntimeError(
                f"rank {self._rank}: the messages of "
                f"{step_label(transfer.direction, transfer.layer)} need {end} bytes, the message "
                f"arena has {arena.numel()}"
            )
        return -(-end // _MESSAGE_ALIGNMENT) * _MESSAGE_ALIGNMENT

    def _run_native(self, issue_point: int) -> None:
        """Enqueue the steps of ``issue_point`` with the C++ executor."""
        started = time.perf_counter()
        try:
            self._native.run(issue_point, self._current_stream().cuda_stream, self._wait_timing)
        except RuntimeError as error:
            raise RuntimeError(
                f"rank {self._rank}: the data plane at issue point {issue_point}: {error}"
            ) from error
        finally:
            self.stats.native_seconds += time.perf_counter() - started
        counts = self._native_counts.get(issue_point)
        if counts is not None:
            stats = self.stats
            stats.messages_sent += counts[0]
            stats.bytes_sent += counts[1]
            stats.messages_received += counts[2]
            stats.bytes_received += counts[3]
            stats.local_copies += counts[4]
            stats.bytes_local += counts[5]
            stats.bytes_local_fetch += counts[6]
            stats.bytes_local_writeback += counts[7]

    def _copy_runs(
        self, runs: Sequence[_Run], destinations: Sequence[_Side], sources: Sequence[_Side]
    ) -> list[int]:
        """The runs of a copy from ``sources`` to ``destinations`` as ``copy_runs`` reads them."""
        page_bytes = self._layout.page_bytes
        flat: list[int] = []
        for run, destination, source in zip(runs, destinations, sources):
            flat.append(page_bytes(run.component))
            flat.append(run.count)
            flat.extend(destination)
            flat.extend(source)
        return flat

    # ---- issuing the steps ----------------------------------------------------------------

    def _record(self, stream, slot: tuple) -> object:
        """Record the event of ``slot`` on ``stream``; one event per slot, reused every iteration."""
        event = stream.record_event(self._events.get(slot))
        self._events[slot] = event
        return event

    def _begin(self) -> None:
        self._begun = True
        if self._native is not None:
            if not self._native_compiled:
                self._compile_native()
            self._run_native(ISSUE_AT_BEGIN)
            return
        if self._program is None:
            self._compile()
        self._data_stream.wait_event(self._record(self._current_stream(), ("begin",)))
        for compiled in self._program.get(ISSUE_AT_BEGIN, ()):
            self._issue(compiled)

    def _issue(self, compiled: _CompiledStep) -> None:
        """Enqueue the operations of one step on the data stream."""
        if compiled.fill_layer is not None:
            with torch.cuda.stream(self._data_stream):
                self._pool.fill_layer(compiled.fill_layer, self._fill)
        stream = self._data_stream.cuda_stream
        debug = self._debug
        stats = self.stats
        for op in compiled.local:
            nbytes = op.op.transfer.nbytes
            if debug is not None:
                self._checksum_pages(op.key, "source", nbytes, op.runs, op.pages[0])
            self._copy(op.copies, stream, op.op.transfer.label)
            if debug is not None:
                self._checksum_pages(op.key, "stored", nbytes, op.runs, op.pages[1])
            stats.local_copies += 1
            stats.bytes_local += nbytes
            if op.op.transfer.direction is Direction.FETCH:
                stats.bytes_local_fetch += nbytes
            else:
                stats.bytes_local_writeback += nbytes
        if compiled.sends:
            for op in compiled.sends:
                self._copy(op.copies, stream, op.op.transfer.label)
                if debug is not None:
                    debug.checksum(op.key, "sent", op.message)
            self._transfer(compiled.sends, ())
            stats.messages_sent += len(compiled.sends)
            stats.bytes_sent += sum(op.op.transfer.nbytes for op in compiled.sends)
        if compiled.recvs:
            self._transfer((), compiled.recvs)
            for op in compiled.recvs:
                nbytes = op.op.transfer.nbytes
                if self._fault is not None and self._is_faulty(op.op.transfer):
                    debug.corrupt(op.message, self._fault.kind)
                if debug is not None:
                    debug.checksum(op.key, "received", op.message)
                self._copy(op.copies, stream, op.op.transfer.label)
                if debug is not None:
                    self._checksum_pages(op.key, "stored", nbytes, op.runs, op.pages[0])
            stats.messages_received += len(compiled.recvs)
            stats.bytes_received += sum(op.op.transfer.nbytes for op in compiled.recvs)
        if compiled.record_fetch:
            layer = compiled.step.layer
            self._fetch_done[layer] = self._record(self._data_stream, ("fetch", layer))

    def _transfer(self, sends: Sequence[_CompiledOp], recvs: Sequence[_CompiledOp]) -> None:
        """Hand the messages of a step to the transport: in one group when it offers one."""
        started = time.perf_counter()
        if self._group_send_recv is not None:
            self._group_send_recv(
                [(op.message, op.op.peer) for op in sends],
                [(op.message, op.op.peer) for op in recvs],
                self._data_stream,
            )
        else:
            for op in sends:
                self._transport.send(op.message, op.op.peer, self._data_stream)
            for op in recvs:
                self._transport.recv(op.message, op.op.peer, self._data_stream)
        self.stats.transport_seconds += time.perf_counter() - started

    def _is_faulty(self, transfer: Transfer) -> bool:
        fault = self._fault
        return (
            fault.iteration in (None, self._iteration)
            and fault.layer == transfer.layer
            and fault.direction is transfer.direction
            and fault.rank in (None, self._rank)
        )

    def _checksum_pages(
        self, key: tuple, role: str, nbytes: int, runs: Sequence[_Run], pages: Sequence[_Side]
    ) -> None:
        """Read the pages at ``pages`` back into the order of a message and checksum them."""
        base = self._scratch.data_ptr()
        in_scratch = tuple(self._message_side(base, run) for run in runs)
        self._copy(
            self._copy_runs(runs, in_scratch, pages),
            self._data_stream.cuda_stream,
            f"the {role} checksum of {step_label(Direction(key[0]), key[1])}",
        )
        self._debug.checksum(key, role, self._scratch[:nbytes])

    def _copy(self, copies: Sequence[int], stream: int, what: str) -> None:
        started = time.perf_counter()
        try:
            self._copier.copy_runs(copies, stream)
        except RuntimeError as error:
            raise RuntimeError(f"rank {self._rank}: {what}: {error}") from error
        finally:
            self.stats.copy_seconds += time.perf_counter() - started

    # ---- where the pages are --------------------------------------------------------------

    def _message_runs(self, transfer: Transfer) -> list[_Run]:
        """The pages of a message in the order they are laid out, buffer by buffer in a segment."""
        layout = self._layout
        fetch = transfer.direction is Direction.FETCH
        span_of = layout.fetch_range if fetch else layout.writeback_range
        runs: list[_Run] = []
        offset = 0
        for segment in transfer.segments:
            request = self._requests[segment.request_id]
            history, chunk = request.context_current_position, request.context_chunk_size
            first, count = span_of(segment.kind, history, chunk)
            if count != segment.pages:
                raise RuntimeError(
                    f"rank {self._rank}: {transfer.label} of request {segment.request_id} has "
                    f"{segment.pages} {segment.kind.label} pages in the plan, the layout {count}"
                )
            for component in layout.components_of(segment.kind):
                runs.append(_Run(component, segment.request_id, first, count, offset))
                offset += count * layout.page_bytes(component)
        if offset != transfer.nbytes:
            raise RuntimeError(
                f"rank {self._rank}: {transfer.label} has {transfer.nbytes} bytes in the plan, "
                f"the layout {offset}"
            )
        return runs

    def _message_side(self, base: int, run: _Run) -> _Side:
        """The pages of a run in a message buffer that starts at ``base``: back to back."""
        return (base + run.offset, self._layout.page_bytes(run.component), 0)

    def _pool_side(self, layer: int, run: _Run) -> _Side:
        """The pages of a run in the cache manager of this rank: through their page indices.

        The copier reads the indices from the table of the cache manager itself (or from a copy of
        the list an older manager gives) and reports a block without a page.
        """
        attention_type = run.component.attention_type
        if self._affine_pages:
            group, step, origin = self._affine(layer, attention_type)
            indices = self._base_indices.get((run.request_id, group))
            if indices is None:
                # A copy: the manager's table is a view that is valid only until the next
                # lifecycle operation of the request, and the copier reads the indices during
                # the whole forward pass.
                indices = np.array(
                    self._manager.get_base_page_indices(run.request_id, group), dtype=np.int32
                )
                self._base_indices[run.request_id, group] = indices
        else:
            origin, step = self._page_table(layer, attention_type)
            key = (run.request_id, layer, attention_type)
            indices = self._indices.get(key)
            if indices is None:
                indices = np.asarray(
                    self._manager.get_cache_indices(run.request_id, layer, attention_type),
                    dtype=np.int32,
                )
                self._indices[key] = indices
        if len(indices) < run.first + run.count:
            raise RuntimeError(
                f"rank {self._rank}: the cache manager has no page for block {len(indices)} of "
                f"request {run.request_id} in layer {layer} ({attention_type.name}), which the "
                "plan moves"
            )
        return (origin, step, indices.ctypes.data + 4 * run.first)

    def _affine(self, layer: int, attention_type) -> tuple[int, int, int]:
        """``(layer group, address step, address origin)`` of the pages of a buffer.

        Block ``b`` of a request is at ``origin + base[b] * step`` where ``base`` is its base
        page indices in the layer group. Looked up once per buffer.
        """
        key = (layer, attention_type)
        affine = self._affines.get(key)
        if affine is None:
            group, scale, offset = self._manager.get_cache_index_affine(layer, attention_type)
            base, stride = self._page_table(layer, attention_type)
            affine = (group, scale * stride, base + offset * stride)
            self._affines[key] = affine
        return affine

    def _page_table(self, layer: int, attention_type) -> tuple[int, int]:
        """Where the pages of a buffer of the cache manager start and how far apart they are.

        The pools of the cache manager are allocated once, so this is looked up once.
        """
        table = self._page_tables.get((layer, attention_type))
        if table is None:
            buffer = self._manager.get_buffers(layer, attention_type)
            table = (buffer.data_ptr(), buffer.stride(0) * buffer.element_size())
            self._page_tables[layer, attention_type] = table
        return table

    def _staging_side(self, layer: int, run: _Run) -> _Side:
        """The pages of a run in the slot of ``layer`` of this rank: back to back."""
        layout = self._layout
        component = run.component
        index = self._local_index.get(run.request_id)
        if index is None:
            raise RuntimeError(
                f"rank {self._rank}: request {run.request_id} is not in the staged batch"
            )
        span = self._spans[component.kind][index]
        if not (
            span.first_block <= run.first
            and run.first + run.count <= span.first_block + span.num_pages
        ):
            raise RuntimeError(
                f"rank {self._rank}: blocks {run.first}..{run.first + run.count - 1} of request "
                f"{run.request_id} are outside the staged blocks {span.first_block}.."
                f"{span.first_block + span.num_pages - 1} of {component.kind.label}"
            )
        page_bytes = layout.page_bytes(component)
        entry = (
            layout.slot_page_offset(layer, component.kind)
            + span.page_offset
            + run.first
            - span.first_block
        )
        begin = self._pool.layer_pointer(layer, component) + entry * page_bytes
        return (begin, page_bytes, 0)
