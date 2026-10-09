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
"""CUDA streams, events, the page copy kernel and the interconnect of the DKV data plane on threads.

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
from collections.abc import Callable, Sequence

import numpy as np
import torch

from tensorrt_llm._torch.pyexecutor.dkv_staging import BAD_PAGE_INDEX, StagingLayout


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
    """``DkvPageCopier`` on host memory: the copies run on the stream they are enqueued on."""

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


class FakeKvManager:
    """The pages of the layers one rank owns, in host memory, in the shape the streamer reads.

    Every owned (layer, cache role) has a buffer of pages and every request a list of page indices
    into it, drawn in random order from the free pages so that indices and addresses are unrelated
    to the position of a block. ``prepare`` is what the cache manager does for a request before an
    iteration: it gives the request the pages its new tokens need and frees the pages of blocks
    that left the window of a sliding-window kind.
    """

    def __init__(
        self,
        layout: StagingLayout,
        owned_layers: Sequence[int],
        pages: int,
        seed: int = 0,
        *,
        affine: bool = True,
    ) -> None:
        """``affine`` offers the page tables the way the DeepSeek-V4 manager does, as base page
        indices per layer group with an affine map per buffer; without it the streamer asks for
        the page indices of every (request, layer, buffer)."""
        self.layout = layout
        self.kv_cache_map: dict[int, object] = {}
        self._random = random.Random(seed)
        self._buffers: dict[tuple[int, object], torch.Tensor] = {}
        self._free: dict[tuple[int, object], list[int]] = {}
        self._indices: dict[tuple[int, int, object], list[int]] = {}
        for component in layout.components:
            for layer in layout.layers_of(component.kind):
                if layer in owned_layers:
                    key = (layer, component.attention_type)
                    self._buffers[key] = torch.zeros(
                        pages, layout.page_bytes(component), dtype=torch.uint8
                    )
                    free = list(range(pages))
                    self._random.shuffle(free)
                    self._free[key] = free
        # Every buffer is a layer group of its own: the fake has no pool shared between layers.
        self._group_keys = list(self._buffers)
        self._groups = {key: group for group, key in enumerate(self._group_keys)}
        if not affine:
            self.get_cache_index_affine = None
            self.get_base_page_indices = None

    def get_buffers(self, layer: int, attention_type) -> torch.Tensor:
        return self._buffers[layer, attention_type]

    def get_cache_indices(self, request_id: int, layer: int, attention_type) -> list[int]:
        return list(self._indices[request_id, layer, attention_type])

    def get_cache_index_affine(self, layer: int, attention_type) -> tuple[int, int, int]:
        return self._groups[layer, attention_type], 1, 0

    def get_base_page_indices(self, request_id: int, group: int) -> np.ndarray:
        layer, attention_type = self._group_keys[group]
        return np.asarray(
            self._indices.get((request_id, layer, attention_type), []), dtype=np.int32
        )

    def prepare(self, request_id: int, history: int, chunk: int) -> None:
        self.kv_cache_map[request_id] = object()
        layout = self.layout
        for component in layout.components:
            for layer in layout.layers_of(component.kind):
                key = (layer, component.attention_type)
                if key not in self._buffers:
                    continue
                first, pages = layout.block_range(component.kind, history, chunk)
                indices = self._indices.setdefault((request_id, *key), [])
                free = self._free[key]
                while len(indices) < first + pages:
                    indices.append(free.pop())
                for block in range(first):
                    if indices[block] != BAD_PAGE_INDEX:
                        free.insert(self._random.randrange(len(free) + 1), indices[block])
                        indices[block] = BAD_PAGE_INDEX

    def release(self, request_id: int) -> None:
        self.kv_cache_map.pop(request_id, None)
        for key in [key for key in self._indices if key[0] == request_id]:
            _, layer, attention_type = key
            for index in self._indices.pop(key):
                if index != BAD_PAGE_INDEX:
                    self._free[layer, attention_type].append(index)
