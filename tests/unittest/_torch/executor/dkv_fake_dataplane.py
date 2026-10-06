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

import torch

from tensorrt_llm._torch.pyexecutor.dkv_staging import BAD_PAGE_INDEX, StagingLayout


class FakeEvent:
    """An event that the stream records when it reaches it."""

    def __init__(self) -> None:
        self._recorded = threading.Event()

    def record(self) -> None:
        self._recorded.set()

    def query(self) -> bool:
        return self._recorded.is_set()

    def wait(self, timeout: float) -> None:
        if not self._recorded.wait(timeout):
            raise TimeoutError("an event was waited for but never recorded")


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

    def record_event(self) -> FakeEvent:
        event = FakeEvent()
        self.enqueue(event.record)
        return event

    def wait_event(self, event: FakeEvent) -> None:
        self.enqueue(lambda: event.wait(self.timeout))

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

    def channel(
        self, source: int, destination: int
    ) -> "queue.Queue[tuple[bytes, threading.Event]]":
        return self._channels[source, destination]


class FakeTransport:
    """``DkvTransport`` of one rank on a ``FakeNetwork``, with the semantics of the real one."""

    def __init__(self, network: FakeNetwork, rank: int) -> None:
        self._network = network
        self._rank = rank
        # Hook for fault injection: called with the bytes of a message before it is delivered.
        self.corrupt: Callable[[int, int, bytearray], None] | None = None

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


class FakeKvManager:
    """The pages of the layers one rank owns, in host memory, in the shape the streamer reads.

    Every owned (layer, cache role) has a buffer of pages and every request a list of page indices
    into it, drawn in random order from the free pages so that indices and addresses are unrelated
    to the position of a block. ``prepare`` is what the cache manager does for a request before an
    iteration: it gives the request the pages its new tokens need and frees the pages of blocks
    that left the window of a sliding-window kind.
    """

    def __init__(
        self, layout: StagingLayout, owned_layers: Sequence[int], pages: int, seed: int = 0
    ) -> None:
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

    def get_buffers(self, layer: int, attention_type) -> torch.Tensor:
        return self._buffers[layer, attention_type]

    def get_cache_indices(self, request_id: int, layer: int, attention_type) -> list[int]:
        return list(self._indices[request_id, layer, attention_type])

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
