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
"""CUDA streams, events, the page copy kernel, the interconnect and the cache managers of the DKV
data plane on threads and host memory.

``DkvStreamer`` orders the work of two streams with events and exchanges messages with its peers.
These stand-ins run that logic on host memory, one thread per stream, so the order of the work, the
pairing of the messages and the hazards between the streams can be tested on every machine. A stream
runs the work enqueued on it one item after the other, as a CUDA stream does; a wait on an event
that is never recorded and a message that nobody receives fail after a timeout instead of hanging;
and a send finishes only when its receive has taken the message, which is the strictest behaviour
an interconnect may have. ``jitter`` makes every item wait a random time first, so that a missing
event shows up as a wrong result and not by luck of the schedule.
"""

import ctypes
import itertools
import queue
import random
import threading
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import torch

from tensorrt_llm._torch.pyexecutor.dkv_staging import expand_runs
from tensorrt_llm._torch.pyexecutor.dkv_types import LifecycleKey
from tensorrt_llm._torch.pyexecutor.dkv_working import staged_blocks
from tensorrt_llm.runtime.kv_cache_manager_v2 import (
    BAD_PAGE_INDEX,
    AttentionLayerConfig,
    BufferConfig,
    PageIndexMode,
)


class FakeEvent:
    """An event that the stream records when it reaches it.

    Like a CUDA event it can be recorded again and again: a wait enqueued after a record waits for
    that record and not for a later one, and ``query`` asks about the latest record. An event that
    was never recorded is never reached, so a wait on it times out instead of passing.
    """

    def __init__(
        self, *, enable_timing: bool = False, clock: Callable[[], float] = time.perf_counter
    ) -> None:
        self._condition = threading.Condition()
        self._armed = 0  # records enqueued so far
        self._reached = 0  # records the stream has carried out
        self._enable_timing = enable_timing
        self._clock = clock
        self._timestamp: float | None = None

    def arm(self) -> int:
        """Count a record that is about to be enqueued; returns its generation."""
        with self._condition:
            self._armed += 1
            return self._armed

    @property
    def armed(self) -> int:
        with self._condition:
            return self._armed

    def record(self, generation: int | None = None) -> None:
        with self._condition:
            if self._enable_timing:
                self._timestamp = self._clock()
            if generation is None:
                generation = self._armed = self._armed + 1
            self._reached = max(self._reached, generation)
            self._condition.notify_all()

    def query(self) -> bool:
        with self._condition:
            return self._armed > 0 and self._reached >= self._armed

    def wait(self, timeout: float, generation: int | None = None) -> None:
        with self._condition:
            target = self._armed if generation is None else generation
            reached = self._condition.wait_for(
                lambda: target > 0 and self._reached >= target, timeout
            )
            if not reached:
                raise TimeoutError("an event was waited for but never recorded")

    def synchronize(self) -> None:
        self.wait(10.0)

    def elapsed_time(self, end_event: "FakeEvent") -> float:
        if self._timestamp is None or end_event._timestamp is None:
            raise RuntimeError("elapsed time requires two recorded timing events")
        return (end_event._timestamp - self._timestamp) * 1000.0


class FakeStream:
    """A CUDA stream: a thread that runs the enqueued work in order."""

    _handles = itertools.count(1)
    _registry: dict[int, "FakeStream"] = {}

    def __init__(
        self, name: str, *, timeout: float = 10.0, jitter: float = 0.0, seed: int = 0
    ) -> None:
        self.name = name
        self.cuda_stream = next(self._handles)
        FakeStream._registry[self.cuda_stream] = self
        self.timeout = timeout
        self.errors: list[BaseException] = []
        self._jitter = jitter
        self._random = random.Random(seed)
        self._queue: queue.Queue[Callable[[], None] | None] = queue.Queue()
        self._thread = threading.Thread(target=self._run, name=f"fake-stream-{name}", daemon=True)
        self._thread.start()

    @classmethod
    def of_handle(cls, handle: int) -> "FakeStream":
        return cls._registry[handle]

    def _run(self) -> None:
        while True:
            work = self._queue.get()
            if work is None:
                return
            if self._jitter:
                time.sleep(self._random.uniform(0.0, self._jitter))
            try:
                work()
            except BaseException as error:  # a stream keeps going; the test inspects ``errors``
                self.errors.append(error)

    def enqueue(self, work: Callable[[], None]) -> None:
        self._queue.put(work)

    def record_event(self, event: FakeEvent | None = None) -> FakeEvent:
        event = event if event is not None else FakeEvent()
        generation = event.arm()
        self.enqueue(lambda: event.record(generation))
        return event

    def wait_event(self, event: FakeEvent) -> None:
        # As a CUDA stream does: wait for the record that exists now, not for a later one.
        generation = event.armed
        self.enqueue(lambda: event.wait(self.timeout, generation))

    def synchronize(self) -> None:
        self.record_event().wait(self.timeout)

    def close(self) -> None:
        self._queue.put(None)
        self._thread.join(timeout=self.timeout)


class FakeNetwork:
    """The wires between the ranks of a group: one queue of messages per ordered pair."""

    def __init__(self, size: int, *, timeout: float = 10.0) -> None:
        self.size = size
        self.timeout = timeout
        self._channels = {
            (source, destination): queue.Queue()
            for source in range(size)
            for destination in range(size)
            if source != destination
        }
        self.sent: dict[tuple[int, int], list[int]] = {key: [] for key in self._channels}
        # The sizes of the messages that the receiver has taken, in the order it took them.
        self.received: dict[tuple[int, int], list[int]] = {key: [] for key in self._channels}

    def channel(
        self, source: int, destination: int
    ) -> "queue.Queue[tuple[bytes, threading.Event]]":
        return self._channels[source, destination]


class FakeTransport:
    """``DkvTransport`` of one rank on a ``FakeNetwork``, with the semantics of the real one.

    ``grouped`` offers ``group_send_recv`` as the NCCL transport does; without it the streamer
    sends and receives one message at a time.
    """

    def __init__(self, network: FakeNetwork, rank: int, *, grouped: bool = True) -> None:
        self._network = network
        self._rank = rank
        # Hook for fault injection: called with the bytes of a message before it is delivered.
        self.corrupt: Callable[[int, int, bytearray], None] | None = None
        if not grouped:
            self.group_send_recv = None

    def group_send_recv(
        self,
        sends: Sequence[tuple[torch.Tensor, int]],
        recvs: Sequence[tuple[torch.Tensor, int]],
        stream: FakeStream,
    ) -> None:
        """All the messages of a step: the sends first, then the receives, in list order."""
        for buffer, peer in sends:
            self.send(buffer, peer, stream)
        for buffer, peer in recvs:
            self.recv(buffer, peer, stream)

    def send(self, buffer: torch.Tensor, peer: int, stream: FakeStream) -> None:
        network, rank = self._network, self._rank

        def work() -> None:
            payload = bytearray(buffer.numpy().tobytes())
            network.sent[rank, peer].append(len(payload))
            if self.corrupt is not None:
                self.corrupt(rank, peer, payload)
            taken = threading.Event()
            network.channel(rank, peer).put((bytes(payload), taken))
            if not taken.wait(network.timeout):
                raise TimeoutError(f"rank {rank}: rank {peer} did not receive the message")

        stream.enqueue(work)

    def recv(self, buffer: torch.Tensor, peer: int, stream: FakeStream) -> None:
        network, rank = self._network, self._rank

        def work() -> None:
            try:
                payload, taken = network.channel(peer, rank).get(timeout=network.timeout)
            except queue.Empty:
                raise TimeoutError(f"rank {rank}: rank {peer} sent no message") from None
            try:
                if len(payload) != buffer.numel():
                    raise RuntimeError(
                        f"rank {rank} receives {buffer.numel()} bytes from rank {peer}, "
                        f"which sent {len(payload)}"
                    )
                buffer.copy_(torch.frombuffer(bytearray(payload), dtype=torch.uint8))
                network.received[peer, rank].append(len(payload))
            finally:
                taken.set()

        stream.enqueue(work)


class FakeCopier:
    """``DkvPageCopier`` on host memory: the copies run on the stream they are enqueued on.

    The page index tables of the runs are read when the copy is enqueued, as the real copier
    reads them.
    """

    def copy(self, pairs: Sequence[tuple[int, int]], num_bytes: int, stream: int) -> None:
        if num_bytes <= 0 or num_bytes % 16:
            raise ValueError(f"num_bytes must be a positive multiple of 16, got {num_bytes}")
        tasks = list(pairs)

        def work() -> None:
            for destination, source in tasks:
                ctypes.memmove(destination, source, num_bytes)

        FakeStream.of_handle(stream).enqueue(work)

    def copy_addresses(
        self, destinations: np.ndarray, sources: np.ndarray, num_bytes: int, stream: int
    ) -> None:
        if len(destinations) != len(sources):
            raise ValueError(
                f"{len(destinations)} destination addresses but {len(sources)} source addresses"
            )
        self.copy(list(zip(destinations.tolist(), sources.tolist())), num_bytes, stream)

    def copy_runs(self, runs: Sequence[int], stream: int) -> None:
        for page_bytes, destinations, sources in expand_runs(runs):
            self.copy_addresses(destinations, sources, page_bytes, stream)


class HostDataPlaneDebug:
    """``DataPlaneDebug`` on host memory: the checks run on the fake stream, in its order."""

    def __init__(self, stream: FakeStream) -> None:
        self._stream = stream
        self._records: list[tuple[tuple, str, int]] = []

    def checksum(self, key: tuple, role: str, data: torch.Tensor) -> None:
        from tensorrt_llm._torch.pyexecutor.dkv_streamer import message_checksum

        def work() -> None:
            self._records.append((key, role, int(message_checksum(data))))

        self._stream.enqueue(work)

    def corrupt(self, data: torch.Tensor, kind: str) -> None:
        def work() -> None:
            if kind == "zero":
                data.zero_()
            else:
                data[data.numel() // 2] ^= 0xFF

        self._stream.enqueue(work)

    def collect(self) -> list[tuple[tuple, str, int]]:
        records, self._records = self._records, []
        return records


@dataclass(frozen=True)
class FakeBuffer:
    """A buffer of a model layer: its data role, the bytes of one block, a multiple of 16, and the
    tokens its life cycle keeps behind the newest one (0: the whole history)."""

    role: str
    page_bytes: int
    window: int = 0

    def lifecycle(self) -> LifecycleKey:
        return LifecycleKey(False, self.window, 0, False)


class _FakeRequestCache:
    """The base page index of every block of one request in every layer group, as a V2 request
    cache keeps them. A lifecycle operation replaces a table, so an earlier view goes stale."""

    def __init__(self, num_groups: int) -> None:
        self.tables = [np.zeros(0, dtype=np.int32) for _ in range(num_groups)]

    def get_base_page_indices(self, group: int) -> np.ndarray:
        return self.tables[group]


class _FakeImpl:
    """The address arithmetic the V2 storage publishes for a buffer."""

    def __init__(self, cache: "FakeV2Cache") -> None:
        self._cache = cache

    def get_page_index_converter(self, layer_id: int, role: str) -> SimpleNamespace:
        group, page_bytes, _ = self._cache._placement[layer_id, role]
        members = len(self._cache._pool_members[group, page_bytes])
        return SimpleNamespace(expansion=1, scale=members)

    def get_page_stride(self, layer_id: int, role: str) -> int:
        return self._cache._placement[layer_id, role][1]

    def get_mem_pool_base_address(self, layer_id: int, role: str, mode) -> int:
        if mode != PageIndexMode.SHARED:
            raise NotImplementedError("the fake publishes base addresses for shared indices only")
        group, page_bytes, position = self._cache._placement[layer_id, role]
        return self._cache._pools[group, page_bytes].data_ptr() + position * page_bytes


class FakeV2Cache:
    """The pages of the layers one cache manager holds, in host memory, in the shape of the V2
    cache manager that the data plane reads.

    Every model layer is one V2 layer per window of its buffers. A window is a life cycle and a
    layer group: every request has one base page index per block and group, drawn in random order
    from the free slots so that indices are unrelated to the position of a block. The buffers of a
    group with the same page size share a pool whose slot holds one page of each, as V2 coalesces
    them, so a page lies at the pool base, plus the offset of its buffer in the slot, plus the base
    page index times the size of the slot.

    ``prepare`` is what the cache manager of an owner does for a request before an iteration: it
    gives the request the pages its life cycles keep up to the end of the chunk and frees those of
    blocks that left a window. ``open`` and ``close`` are what the working cache does around one
    iteration: a request gets fresh pages, which hold ``poison``, and gives them all back.
    """

    def __init__(
        self,
        layers: Sequence[Sequence[FakeBuffer]],
        held_layers: Iterable[int],
        *,
        tokens_per_block: int,
        slots: int,
        seed: int = 0,
        poison: int = 0,
    ) -> None:
        self._random = random.Random(seed)
        self._tokens_per_block = tokens_per_block
        self._poison = poison
        self.impl = _FakeImpl(self)
        self.kv_cache_map: dict[int, _FakeRequestCache] = {}
        self.layer_to_pool_mapping_dict: dict[int, int] = {}
        self.held_layers = tuple(sorted(set(held_layers)))
        self._windows: list[int] = []
        self._model_layer: dict[int, int] = {}
        self._layer_id: dict[tuple[int, str], int] = {}
        # (layer id, role) -> (group, page bytes, position of the buffer in the pool's slot)
        self._placement: dict[tuple[int, str], tuple[int, int, int]] = {}
        self._pool_members: dict[tuple[int, int], list[tuple[int, str]]] = {}
        configs = []
        for layer in self.held_layers:
            by_window: dict[int, list[FakeBuffer]] = {}
            for buffer in layers[layer]:
                by_window.setdefault(buffer.window, []).append(buffer)
            for window, buffers in by_window.items():
                layer_id = len(configs)
                configs.append(
                    AttentionLayerConfig(
                        layer_id=layer_id,
                        buffers=[BufferConfig(role=b.role, size=b.page_bytes) for b in buffers],
                        sliding_window_size=window or None,
                    )
                )
                if window not in self._windows:
                    self._windows.append(window)
                group = self._windows.index(window)
                self.layer_to_pool_mapping_dict[layer_id] = group
                self._model_layer[layer_id] = layer
                for buffer in buffers:
                    members = self._pool_members.setdefault((group, buffer.page_bytes), [])
                    self._placement[layer_id, buffer.role] = (
                        group,
                        buffer.page_bytes,
                        len(members),
                    )
                    members.append((layer_id, buffer.role))
                    self._layer_id[layer, buffer.role] = layer_id
        self._pools = {
            (group, page_bytes): torch.full(
                (slots, page_bytes * len(members)), poison, dtype=torch.uint8
            )
            for (group, page_bytes), members in self._pool_members.items()
        }
        self._free = [list(range(slots)) for _ in self._windows]
        for free in self._free:
            self._random.shuffle(free)
        self.kv_cache_manager_py_config = SimpleNamespace(
            layers=configs,
            tokens_per_block=tokens_per_block,
            lifecycle_slot_counts=[[slots] * len(self._windows)],
        )

    def get_disagg_ignored_roles(self) -> tuple:
        return ()

    def dkv_model_layer_of(self, layer_id: int, role: str) -> int:
        return self._model_layer[layer_id]

    def page(self, request_id: int, layer: int, role: str, block: int) -> torch.Tensor | None:
        """The page of a block of a request, or None when the block has no page."""
        layer_id = self._layer_id[layer, role]
        group, page_bytes, position = self._placement[layer_id, role]
        table = self.kv_cache_map[request_id].tables[group]
        if block >= len(table) or table[block] == BAD_PAGE_INDEX:
            return None
        start = position * page_bytes
        return self._pools[group, page_bytes][int(table[block]), start : start + page_bytes]

    def prepare(self, request_id: int, history: int, chunk: int) -> None:
        cache = self.kv_cache_map.setdefault(request_id, _FakeRequestCache(len(self._windows)))
        for group, window in enumerate(self._windows):
            key = LifecycleKey(False, window, 0, False)
            first, count = staged_blocks(key, history, chunk, self._tokens_per_block)
            free = self._free[group]
            table = cache.tables[group].tolist()
            while len(table) < first + count:
                table.append(free.pop() if len(table) >= first else BAD_PAGE_INDEX)
            for block in range(first):
                if table[block] != BAD_PAGE_INDEX:
                    free.insert(self._random.randrange(len(free) + 1), table[block])
                    table[block] = BAD_PAGE_INDEX
            cache.tables[group] = np.array(table, dtype=np.int32)

    def open(self, request_id: int, history: int, chunk: int) -> None:
        if request_id in self.kv_cache_map:
            raise RuntimeError(f"request {request_id} is already open")
        self.prepare(request_id, history, chunk)
        for (group, page_bytes), pool in self._pools.items():
            for slot in self.kv_cache_map[request_id].tables[group]:
                if slot != BAD_PAGE_INDEX:
                    pool[int(slot)].fill_(self._poison)

    def release(self, request_id: int) -> None:
        cache = self.kv_cache_map.pop(request_id, None)
        if cache is None:
            return
        for group, table in enumerate(cache.tables):
            self._free[group].extend(int(slot) for slot in table if slot != BAD_PAGE_INDEX)

    close = release
