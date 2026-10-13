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
"""Moves the KV of a layer between the owners' caches and the working cache around its attention.

``DkvStreamer`` serves the ``layer_split`` layout, where the KV of a layer lives in the cache
manager of the rank that owns the layer and is computed in the working cache (``dkv_working``) of
the rank that runs the request. It carries out a ``DkvPlan`` on a data stream of its own: it packs
the pages of a message, sends it, receives the messages of its peers and unpacks them, in the order
the plan fixes for every rank, and it orders the data stream with the stream of the forward pass by
events. The plan is resolved to device addresses once per iteration, when the working caches of the
batch are open, so that the hooks of the forward pass only enqueue work.

Nothing here knows the model: the buffers of a layer, the blocks a request moves and the address of
every page come from what both cache managers publish (``manager_buffers``, ``page_affine``).
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
    ISSUE_AT_BEGIN,
    Direction,
    DkvPlan,
    OpAction,
    PlanRequest,
    PlanStep,
    RankOp,
    Transfer,
    build_dkv_plan,
    step_label,
)
from .dkv_staging import DkvPageCopier
from .dkv_transport import DkvTransport
from .dkv_working import (
    BufferCostModel,
    DkvBuffer,
    PageAffine,
    base_page_indices,
    fetch_blocks,
    manager_buffers,
    page_affine,
    staged_blocks,
    writeback_blocks,
)


def arena_peers(group_size: int) -> int:
    """The peers that each of the two message arenas of a rank has room for, a step's worth each."""
    return max(group_size - 1, 1)


def max_peer_step_bytes(working) -> int:
    """An upper bound of the bytes one rank exchanges with one peer in one step.

    The messages of a step to one peer hold the pages of one layer for the requests one rank
    computes, and those pages fit the working cache: for every buffer of the layer, the pages of
    its life cycle times the bytes of a page.
    """
    counts = working.kv_cache_manager_py_config.lifecycle_slot_counts
    if counts is None:
        raise ValueError("the working cache has no fixed page counts")
    per_layer: dict[int, int] = {}
    for entry in manager_buffers(working).values():
        group = int(working.layer_to_pool_mapping_dict[entry.layer_id])
        per_layer[entry.buffer.layer] = (
            per_layer.get(entry.buffer.layer, 0) + int(counts[0][group]) * entry.buffer.page_bytes
        )
    return max(per_layer.values(), default=0)


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
    """The pages of a message that one buffer holds for one request.

    The pages are the blocks ``first`` to ``first + count - 1`` of the request, and the message
    holds them back to back from byte ``offset``.
    """

    buffer: DkvBuffer
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
    key: tuple[int, int, int, int]
    # The region of the message arena the message occupies; None for a local copy.
    message: torch.Tensor | None
    # The copies the operation issues, as the runs of ``DkvPageCopier.copy_runs``. A send packs
    # pages into its message, a receive unpacks its message into pages and a local copy moves
    # pages between the owner's cache and the working cache.
    copies: list[int]
    runs: tuple[_Run, ...]
    # Where the pages of every run are on this rank, as copy sides, for the checksums of debug:
    # the pages a message was packed from or unpacked into, or (source, destination) of a local
    # copy.
    pages: tuple[tuple[_Side, ...], ...]


class _CompiledStep(NamedTuple):
    """What one rank does in one step, resolved to addresses and grouped by kind of operation."""

    step: PlanStep
    # The copies that overwrite the working pages of the layer before its fetch (the debug fill).
    fill: list[int]
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


def message_key(transfer: Transfer) -> tuple[int, int, int, int]:
    """What names a message in every iteration: the same on its two ends."""
    return (int(transfer.direction), transfer.layer, transfer.owner, transfer.compute)


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
        direction, layer, owner, compute = key
        name = step_label(Direction(direction), layer)
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


_FILL_BYTES = {"zero": 0x00, "nan": 0xFF}


class DkvStreamer:
    """Carries out the data plane of the ``layer_split`` layout around the layers of a forward pass.

    The rank computes the requests it was given in its working cache and owns the layers
    ``owner_of_layer`` assigns to it. For each iteration the executor builds a ``DkvPlan``
    (``plan_for``) from the replicated scheduling state and hands it over (``set_plan``). Once the
    working caches of the batch are open, ``begin_iteration`` is called, then ``on_layer`` at the
    top of every layer and ``end_forward`` after the last one, and the executor calls ``drain`` when
    the iteration ends.

    When the batch is known the plan is compiled: every operation of this rank is resolved to the
    device addresses of its pages and to its region of the message arena, step by step. The hooks
    then only enqueue the work of their steps on ``data_stream``: the owner packs the pages of its
    messages from its cache and sends them, the compute rank receives them and unpacks them into
    its working cache, and the other way round for the pages the new tokens wrote. All the messages
    a rank sends or receives in one step are issued together when the transport can group them.
    The data stream waits for the forward pass by events, and the forward pass waits for the fetch
    of a layer:

    * the first operation of an iteration waits for the pages of the caches to be ready;
    * the operations at the top of layer ``l`` wait for the kernels of the layers before ``l``, so
      the writeback sends what they produced and the prefetch may overwrite the pages they read; a
      rank that computes nothing waits as well, which keeps it in step with the others;
    * the attention of layer ``l`` waits for the fetch of layer ``l``.

    ``drain`` waits on the host for the data stream. After it returns the cache managers may free or
    move any page.

    No call blocks the host except ``drain`` (and the transport, if it has to). A rank without a
    plan, as in a warm-up or profiling forward pass, does nothing.
    """

    # How often drain polls the data stream, in seconds.
    _POLL_SECONDS = 0.0005

    def __init__(
        self,
        manager,
        working,
        transport: DkvTransport | None,
        owner_of_layer: Sequence[int],
        rank: int,
        *,
        group_size: int,
        ring_depth: int = 2,
        copier: DkvPageCopier | None = None,
        data_stream=None,
        current_stream: Callable[[], object] | None = None,
        fill: str = "",
        debug: DataPlaneDebug | None = None,
        fault: DataPlaneFault | None = None,
        timing_event_factory: Callable[[], torch.cuda.Event] | None = None,
        group_messages: bool = True,
        device: torch.device | None = None,
    ) -> None:
        """Build the streamer of ``rank``.

        Args:
            manager: The cache manager that holds the pages of the layers this rank owns.
            working: The working cache manager the attention of this rank reads, which covers
                every layer.
            transport: Sends and receives the messages; None for a group of one rank.
            owner_of_layer: The owner rank of every layer of the model.
            rank: This rank in the group.
            group_size: The number of ranks of the group.
            ring_depth: How many layers ahead of its attention a layer is fetched, plus one.
            copier: Packs and unpacks the pages.
            data_stream: The CUDA stream of the data plane; a new one by default.
            current_stream: Returns the stream of the forward pass; the current stream by default.
            fill: ``"zero"`` or ``"nan"`` overwrites the working pages of a layer before they are
                fetched, so that a page that is read without having been fetched or written shows.
            debug: Checksums every message at both ends and the pages it came from or went to;
                ``drain`` returns the records for the executor to compare across the ranks.
            fault: A message to damage after it was received; it needs ``debug``.
            timing_event_factory: Creates a timing-enabled event for fetch waits. Used only with
                ``TRTLLM_DKV_WAIT_TIMING=1``; CUDA events by default.
            group_messages: Hand all the messages of a step to the transport in one call when it
                has a ``group_send_recv`` method that is not None; else one message at a time.
            device: Where the message buffers live, the device of both caches; the current CUDA
                device by default.
        """
        owners = tuple(owner_of_layer)
        if not 0 <= rank < group_size:
            raise ValueError(f"rank {rank} is not in a group of {group_size} ranks")
        if fault is not None and debug is None:
            raise ValueError("a data plane fault is only noticed by the checksums of debug")
        if fill and fill not in _FILL_BYTES:
            raise ValueError(f"unknown staging fill {fill!r}: use 'zero' or 'nan'")
        if transport is None and group_size != 1:
            raise ValueError("a group of more than one rank needs a transport")
        working_buffers = manager_buffers(working)
        layers = sorted({name[0] for name in working_buffers})
        if layers != list(range(len(owners))):
            raise ValueError(
                f"the ownership table has {len(owners)} layers, the working cache holds the "
                f"layers {layers}"
            )
        self._manager = manager
        self._working = working
        self._transport = transport
        self._group_send_recv = (
            getattr(transport, "group_send_recv", None)
            if group_messages and transport is not None
            else None
        )
        self._owners = owners
        self._rank = rank
        self._group_size = group_size
        self._ring_depth = ring_depth
        self._costs = BufferCostModel(
            [entry.buffer for entry in working_buffers.values()],
            int(working.kv_cache_manager_py_config.tokens_per_block),
        )
        self._tokens_per_block = self._costs.tokens_per_block
        self._working_affines: dict[tuple[int, str], PageAffine] = {
            name: page_affine(working, entry.layer_id, entry.role)
            for name, entry in working_buffers.items()
        }
        self._owner_affines: dict[tuple[int, str], PageAffine] = {}
        for name, entry in manager_buffers(manager).items():
            if owners[name[0]] != rank:
                continue
            reference = working_buffers.get(name)
            if reference is None or reference.buffer != entry.buffer:
                raise ValueError(
                    f"the cache of rank {rank} holds {entry.buffer} of layer {name[0]}, the "
                    f"working cache {None if reference is None else reference.buffer}"
                )
            self._owner_affines[name] = page_affine(manager, entry.layer_id, entry.role)
        for layer, owner in enumerate(owners):
            if owner != rank:
                continue
            missing = [
                role
                for role in self._costs.buffers_of(layer)
                if (layer, role) not in self._owner_affines
            ]
            if missing:
                raise ValueError(f"rank {rank} owns layer {layer} but holds no {missing} of it")
        self._copier = copier or DkvPageCopier()
        self._data_stream = data_stream if data_stream is not None else torch.cuda.Stream()
        self._current_stream = current_stream or torch.cuda.current_stream
        self._wait_timing = os.environ.get("TRTLLM_DKV_WAIT_TIMING") == "1"
        self._timing_event_factory = timing_event_factory or functools.partial(
            torch.cuda.Event, enable_timing=True
        )
        self._fetch_wait_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
        self._fill = fill
        # Every message of a step has its own region of the arena, so the messages of a step can
        # be packed first and issued together: a rank exchanges at most max_peer_step_bytes with
        # each of its peers in a step, in one message.
        step_bytes = max_peer_step_bytes(working)
        arena = arena_peers(group_size) * (step_bytes + _MESSAGE_ALIGNMENT)
        if device is None:
            device = torch.device("cuda", torch.cuda.current_device())
        self._send_arena = torch.empty(arena, dtype=torch.uint8, device=device)
        self._recv_arena = torch.empty(arena, dtype=torch.uint8, device=device)
        self._debug = debug
        self._fault = fault
        # Where the pages of a message are read back to checksum them.
        self._scratch = torch.empty(step_bytes, dtype=torch.uint8, device=device) if debug else None
        # The page the debug fill copies over the working pages of a layer.
        largest_page = max((buffer.page_bytes for buffer in self._buffers()), default=16)
        self._fill_page = (
            torch.full((largest_page,), _FILL_BYTES[fill], dtype=torch.uint8, device=device)
            if fill
            else None
        )
        self._iteration = 0
        self.last_plan: DkvPlan | None = None
        self.stats = DataPlaneStats()
        self._plan: DkvPlan | None = None
        # The next forward pass carries out the plan of the batch it stages (``set_staged_plan``).
        self._plan_staged_batch = False
        self._requests: dict[int, PlanRequest] = {}
        self._computes = False
        self._begun = False
        self._program: dict[int, list[_CompiledStep]] | None = None
        self._fetch_done: dict[int, object] = {}
        self._data_done = None
        # The events of the iteration, one per use, recorded again every iteration.
        self._events: dict[tuple, object] = {}
        # Copies of the base page index tables the copies of the iteration read, by (owner side?,
        # request, layer group): a manager's table is a view valid only until the next lifecycle
        # operation of the request.
        self._indices: dict[tuple[bool, int, int], np.ndarray] = {}

    @property
    def uses_native(self) -> bool:
        """Whether a C++ executor carries out the program; this streamer issues from Python."""
        return False

    @property
    def message_bytes(self) -> int:
        """Bytes of the buffers that hold the messages of a step, sent and received."""
        return self._send_arena.numel() + self._recv_arena.numel()

    @property
    def costs(self) -> BufferCostModel:
        """The buffers of every layer and the pages a request moves of them."""
        return self._costs

    def _buffers(self) -> Iterable[DkvBuffer]:
        for layer in range(len(self._owners)):
            for role in self._costs.buffers_of(layer):
                yield self._costs.buffer(layer, role)

    # ---- the plan -------------------------------------------------------------------------

    def plan_for(self, requests: Iterable[PlanRequest]) -> DkvPlan:
        """The plan of an iteration whose global scheduled batch is ``requests``."""
        return build_dkv_plan(
            requests,
            self._owners,
            self._costs,
            group_size=self._group_size,
            ring_depth=self._ring_depth,
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
            or plan.ring_depth != self._ring_depth
        ):
            raise RuntimeError(
                f"the plan is for {plan.group_size} ranks, ring depth {plan.ring_depth} and "
                f"owners {plan.owner_of_layer}, the streamer for {self._group_size} ranks, "
                f"ring depth {self._ring_depth} and owners {self._owners}"
            )
        self._plan = plan
        self._plan_staged_batch = False
        self._iteration = iteration
        self._requests = {request.request_id: request for request in plan.requests}
        self._computes = any(
            request.compute_rank == self._rank and not request.is_dummy for request in plan.requests
        )
        self._begun = False
        self._program = None
        self._fetch_done = {}
        self._indices = {}

    def set_staged_plan(self, iteration: int = 0) -> None:
        """Carry out, in the next forward pass, the plan of the batch the working cache stages for it.

        Only for a group of one rank, which owns every layer and so needs no other rank to agree
        on its plan: it is built from the requests and token counts that the attention stages,
        whatever kind of request they are.

        Raises:
            RuntimeError: The group has more than one rank, or the previous plan has not been
                carried out.
        """
        if self._group_size != 1:
            raise RuntimeError("only a group of one rank plans the batch its working cache stages")
        if self._plan is not None or self._plan_staged_batch:
            raise RuntimeError("the plan of the previous forward pass was not carried out")
        self._plan_staged_batch = True
        self._iteration = iteration

    # ---- hooks of the forward pass --------------------------------------------------------

    def begin_iteration(
        self, request_ids: Sequence[int], history: Sequence[int], new: Sequence[int]
    ) -> None:
        """Called once the working caches of the requests of the forward pass are open."""
        if self._plan_staged_batch:
            self.set_plan(
                self.plan_for(
                    PlanRequest(request_id, 0, cached, chunk)
                    for request_id, cached, chunk in zip(request_ids, history, new)
                ),
                self._iteration,
            )
        self._begin_iteration(request_ids, history, new)

    @_timed_hook
    def _begin_iteration(
        self, request_ids: Sequence[int], history: Sequence[int], new: Sequence[int]
    ) -> None:
        plan = self._plan
        if plan is None:
            return
        local = {
            request_id: (cached, chunk)
            for request_id, cached, chunk in zip(request_ids, history, new)
        }
        for request in plan.requests:
            if request.compute_rank != self._rank or request.is_dummy:
                continue
            staged = local.get(request.request_id)
            if staged != (request.context_current_position, request.context_chunk_size):
                raise RuntimeError(
                    f"rank {self._rank}: the plan has request {request.request_id} with "
                    f"{request.context_current_position} cached and {request.context_chunk_size} "
                    f"new tokens, the working cache has (cached, new) = {staged}"
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
        # A forward pass that staged no batch has nothing to plan.
        self._plan_staged_batch = False
        plan = self._plan
        if plan is None:
            return
        if not self._begun:
            self._begin()
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

        A page may only be freed or moved by a cache manager while the data plane is idle.

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
        self._indices = {}
        return self._debug.collect() if self._debug is not None else []

    # ---- compiling the plan ---------------------------------------------------------------

    def _compile(self) -> None:
        """Resolve every operation of this rank to addresses, step by step."""
        started = time.perf_counter()
        program: dict[int, list[_CompiledStep]] = {}
        for step in self._plan.steps:
            fetch = step.direction is Direction.FETCH
            fill = self._fill_copies(step.layer) if fetch and self._computes and self._fill else []
            # A rank that computes waits for the fetch of a layer also when the layer has nothing
            # to fetch: the step follows the writeback of the layer before, so the layer must not
            # start before the data stream has got past it.
            record_fetch = fetch and self._computes
            ops = step.ops_for(self._rank)
            if not ops and not fill and not record_fetch:
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
                _CompiledStep(step, fill, tuple(local), tuple(sends), tuple(recvs), record_fetch)
            )
        self._program = program
        self.stats.compile_seconds += time.perf_counter() - started

    def _compile_local(self, op: RankOp) -> _CompiledOp:
        transfer = op.transfer
        layer = transfer.layer
        runs = tuple(self._message_runs(transfer))
        pool = tuple(self._owner_side(layer, run) for run in runs)
        working = tuple(self._working_side(layer, run) for run in runs)
        if transfer.direction is Direction.FETCH:
            sources, destinations = pool, working
        else:
            sources, destinations = working, pool
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
        side_of = self._owner_side if op.rank == transfer.owner else self._working_side
        pages = tuple(side_of(layer, run) for run in runs)
        in_message = tuple(self._message_side(base, run) for run in runs)
        if op.action is OpAction.SEND:
            copies = self._copy_runs(runs, in_message, pages)
        else:
            copies = self._copy_runs(runs, pages, in_message)
        compiled = _CompiledOp(op, message_key(transfer), message, copies, runs, (pages,))
        return compiled, -(-end // _MESSAGE_ALIGNMENT) * _MESSAGE_ALIGNMENT

    def _fill_copies(self, layer: int) -> list[int]:
        """Copies of the fill page over every working page this rank's requests have in ``layer``."""
        flat: list[int] = []
        source = self._fill_page.data_ptr()
        for request in self._plan.requests:
            if request.compute_rank != self._rank or request.is_dummy:
                continue
            history, chunk = request.context_current_position, request.context_chunk_size
            for role in self._costs.buffers_of(layer):
                buffer = self._costs.buffer(layer, role)
                first, count = staged_blocks(buffer.key, history, chunk, self._tokens_per_block)
                if not count:
                    continue
                run = _Run(buffer, request.request_id, first, count, 0)
                flat.append(buffer.page_bytes)
                flat.append(count)
                flat.extend(self._working_side(layer, run))
                flat.extend((source, 0, 0))
        return flat

    def _copy_runs(
        self, runs: Sequence[_Run], destinations: Sequence[_Side], sources: Sequence[_Side]
    ) -> list[int]:
        """The runs of a copy from ``sources`` to ``destinations`` as ``copy_runs`` reads them."""
        flat: list[int] = []
        for run, destination, source in zip(runs, destinations, sources):
            flat.append(run.buffer.page_bytes)
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
        if self._program is None:
            self._compile()
        self._data_stream.wait_event(self._record(self._current_stream(), ("begin",)))
        for compiled in self._program.get(ISSUE_AT_BEGIN, ()):
            self._issue(compiled)

    def _issue(self, compiled: _CompiledStep) -> None:
        """Enqueue the operations of one step on the data stream."""
        stream = self._data_stream.cuda_stream
        if compiled.fill:
            self._copy(compiled.fill, stream, f"the fill of layer {compiled.step.layer}")
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
        fetch = transfer.direction is Direction.FETCH
        blocks_of = fetch_blocks if fetch else writeback_blocks
        runs: list[_Run] = []
        offset = 0
        for segment in transfer.segments:
            request = self._requests[segment.request_id]
            buffer = self._costs.buffer(transfer.layer, segment.buffer)
            first, count = blocks_of(
                buffer.key,
                request.context_current_position,
                request.context_chunk_size,
                self._tokens_per_block,
            )
            if count != segment.pages:
                raise RuntimeError(
                    f"rank {self._rank}: {transfer.label} of request {segment.request_id} has "
                    f"{segment.pages} pages of {segment.buffer} in the plan, the rules {count}"
                )
            runs.append(_Run(buffer, segment.request_id, first, count, offset))
            offset += count * buffer.page_bytes
        if offset != transfer.nbytes:
            raise RuntimeError(
                f"rank {self._rank}: {transfer.label} has {transfer.nbytes} bytes in the plan, "
                f"the rules {offset}"
            )
        return runs

    def _message_side(self, base: int, run: _Run) -> _Side:
        """The pages of a run in a message buffer that starts at ``base``: back to back."""
        return (base + run.offset, run.buffer.page_bytes, 0)

    def _owner_side(self, layer: int, run: _Run) -> _Side:
        """The pages of a run in the cache of this rank, the owner of ``layer``."""
        affine = self._owner_affines[layer, run.buffer.role]
        return self._indexed_side(self._manager, False, affine, layer, run, "the cache")

    def _working_side(self, layer: int, run: _Run) -> _Side:
        """The pages of a run in the working cache of this rank, which computes the request."""
        affine = self._working_affines[layer, run.buffer.role]
        return self._indexed_side(self._working, True, affine, layer, run, "the working cache")

    def _indexed_side(
        self, manager, working: bool, affine: PageAffine, layer: int, run: _Run, what: str
    ) -> _Side:
        """A copy side over the pages of a run that a manager's base page indices name.

        The copier reads the indices from a copy of the manager's table, which stays alive until
        the iteration is drained, and reports a block without a page.
        """
        key = (working, run.request_id, affine.layer_group)
        indices = self._indices.get(key)
        if indices is None:
            if run.request_id not in manager.kv_cache_map:
                raise RuntimeError(
                    f"rank {self._rank}: request {run.request_id} has no pages in {what}"
                )
            indices = np.array(
                base_page_indices(manager, run.request_id, affine.layer_group), dtype=np.int32
            )
            self._indices[key] = indices
        if len(indices) < run.first + run.count:
            raise RuntimeError(
                f"rank {self._rank}: {what} has no page for block {len(indices)} of request "
                f"{run.request_id} in layer {layer} ({run.buffer.role}), which the plan moves"
            )
        return (affine.base, affine.step, indices.ctypes.data + 4 * run.first)
