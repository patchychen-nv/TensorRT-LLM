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
"""The data plane of the layer-split layout: ``DkvStreamer`` carries out a plan on threads.

Every rank of a group runs a streamer on host memory with stand-ins for the streams, the copy kernel,
the interconnect and the two cache managers (``dkv_fake_dataplane``): the cache of the layers the
rank owns and the working cache of the requests it computes. A stand-in for the attention of a
layer reads the pages the streamer fetched into the working cache, checks them against what an
earlier iteration wrote, and writes the pages its new tokens touch. The working cache opens fresh
pages that hold poison every iteration, so only what was fetched can pass. After the last iteration
the pages in the cache of the owner of each layer must hold what the compute ranks wrote. The runs
chunk prompts over several iterations, so every direction of every step is exercised, with the
ring depths and the placements the plan is checked for, and with random delays on the streams. The
mutations at the end remove one of the events of the streamer and the test has to notice.
"""

import threading
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pytest
import torch
from dkv_fake_dataplane import (
    FakeBuffer,
    FakeCopier,
    FakeEvent,
    FakeNetwork,
    FakeStream,
    FakeTransport,
    FakeV2Cache,
    HostDataPlaneDebug,
)

from tensorrt_llm._torch.pyexecutor.dkv import compute_ownership, owned_layers
from tensorrt_llm._torch.pyexecutor.dkv_plan import Direction, OpAction, PlanRequest, build_dkv_plan
from tensorrt_llm._torch.pyexecutor.dkv_streamer import (
    DataPlaneFault,
    DkvStreamer,
    find_data_plane_mismatches,
    max_peer_step_bytes,
    message_checksum,
)
from tensorrt_llm._torch.pyexecutor.dkv_working import (
    BufferCostModel,
    fetch_blocks,
    manager_buffers,
    writeback_blocks,
)

pytestmark = pytest.mark.cpu_only

_BLOCK = 128
_POISON = 0xA5
_CHUNK = 128
_TIMEOUT = 15.0
_HOST = torch.device("cpu")

# The buffers of three kinds of layers, after DeepSeek-V4: sliding-window attention only; sparse
# attention over KV compressed 4 times, with an indexer and compressor states that keep the last
# few tokens; and attention over KV compressed 128 times, with a state that keeps a window.
_SWA = FakeBuffer("swa", 512, 128)
_KINDS = {
    1: (_SWA,),
    4: (
        _SWA,
        FakeBuffer("compressed", 128),
        FakeBuffer("indexer", 64),
        FakeBuffer("state", 256, 8),
        FakeBuffer("indexer_state", 64, 8),
    ),
    128: (_SWA, FakeBuffer("compressed", 32), FakeBuffer("state", 1024, 128)),
}
_LAYERS = tuple(_KINDS[ratio] for ratio in (1, 4, 128, 4, 128, 1, 4, 128))
_NUM_LAYERS = len(_LAYERS)


def _cache(held_layers, *, slots: int, seed: int = 0) -> FakeV2Cache:
    return FakeV2Cache(
        _LAYERS, held_layers, tokens_per_block=_BLOCK, slots=slots, seed=seed, poison=_POISON
    )


def _costs() -> BufferCostModel:
    working = _cache(range(_NUM_LAYERS), slots=1)
    return BufferCostModel([entry.buffer for entry in manager_buffers(working).values()], _BLOCK)


@dataclass(frozen=True)
class Scenario:
    """Requests that prefill in chunks of ``_CHUNK`` tokens on the ranks of a group.

    ``prompts`` maps a request id to its prompt length, ``ranks`` to its compute rank and
    ``arrival`` to the iteration in which it enters the batch.
    """

    group_size: int
    ring_depth: int
    prompts: dict[int, int]
    ranks: dict[int, int]
    arrival: dict[int, int] = field(default_factory=dict)

    def batch(self, iteration: int) -> list[PlanRequest]:
        """The global scheduled batch of an iteration, with a dummy for every idle rank."""
        requests = []
        for request_id, prompt in self.prompts.items():
            start = self.arrival.get(request_id, 0)
            progress = _progress(prompt, iteration - start)
            if iteration >= start and progress < prompt:
                chunk = min(_CHUNK, prompt - progress)
                requests.append(PlanRequest(request_id, self.ranks[request_id], progress, chunk))
        busy = {request.compute_rank for request in requests}
        for rank in range(self.group_size):
            if rank not in busy:
                requests.append(PlanRequest(1000 + rank, rank, 0, 1, is_dummy=True))
        return requests

    @property
    def iterations(self) -> int:
        return max(
            self.arrival.get(request_id, 0) + -(-prompt // _CHUNK)
            for request_id, prompt in self.prompts.items()
        )


def _progress(prompt: int, iterations_done: int) -> int:
    return min(prompt, max(0, iterations_done) * _CHUNK)


def _pattern(request_id: int, layer: int, index: int, block: int, nbytes: int) -> torch.Tensor:
    seed = (request_id * 131 + layer * 31 + index * 7 + block * 13) % 251 + 1
    return ((torch.arange(nbytes, dtype=torch.int64) * 7 + seed) % 251).to(torch.uint8)


def _whose(page: torch.Tensor) -> str:
    """What a page holds, for the message of a failed check."""
    if bool((page == _POISON).all()):
        return "poison"
    if not bool(page.any()):
        return "zeros"
    for request_id in range(1, 9):
        for layer, buffers in enumerate(_LAYERS):
            for index, buffer in enumerate(buffers):
                if buffer.page_bytes != page.numel():
                    continue
                for block in range(10):
                    if torch.equal(page, _pattern(request_id, layer, index, block, page.numel())):
                        return (
                            f"the page of request {request_id}, layer {layer}, {buffer.role}, "
                            f"block {block}"
                        )
    return "something else"


class _NoWait(FakeStream):
    """A stream that ignores the events it is told to wait for: the mutation under test."""

    def wait_event(self, event) -> None:
        pass


@dataclass
class RankReport:
    stats: object
    errors: list[str]
    largest_message: int
    # The bound of the bytes a rank exchanges with one peer in a step, and its message buffer.
    step_limit: int
    buffer_bytes: int
    # The checksum records and the plan of every iteration, when the run is in debug.
    records: list = field(default_factory=list)
    plans: list = field(default_factory=list)


class Stall:
    """Holds the forward pass of one rank after the attention of one layer until it is released."""

    def __init__(self, rank: int, layer: int) -> None:
        self.rank = rank
        self.layer = layer
        self.reached = threading.Event()
        self.release = threading.Event()

    def hold(self) -> None:
        self.reached.set()
        if not self.release.wait(_TIMEOUT):
            raise TimeoutError("the stalled forward pass was never released")


@dataclass(frozen=True)
class Options:
    data_jitter: float = 0.0
    exec_jitter: float = 0.0
    seed: int = 0
    # The mutation: which stream ignores the events it waits for ("data", "exec" or "").
    deaf: str = ""
    # The fault: ``(rank, n)`` flips a byte of the n-th message that the rank sends.
    corrupt: tuple[int, int] = (-1, -1)
    # Checksum every message at both ends; ``fault`` damages a message after it was received.
    debug: bool = False
    fault: DataPlaneFault | None = None
    # Stops the forward pass of one rank in the middle of an iteration.
    stall: Stall | None = None
    # The transport takes all the messages of a step in one call, as the NCCL one does.
    grouped: bool = True


def _corrupting(transport: FakeTransport, rank: int, options: Options) -> FakeTransport:
    target, number = options.corrupt
    if rank == target:
        sent = []

        def flip(source: int, destination: int, payload: bytearray) -> None:
            sent.append(destination)
            if len(sent) - 1 == number:
                payload[len(payload) // 2] ^= 0xFF

        transport.corrupt = flip
    return transport


def _run_rank(
    rank: int,
    scenario: Scenario,
    network: FakeNetwork,
    options: Options,
    barrier: threading.Barrier,
) -> RankReport:
    group_size = scenario.group_size
    owners = compute_ownership(_NUM_LAYERS, group_size)
    slots = sum(-(-prompt // _BLOCK) for prompt in scenario.prompts.values()) + 2
    manager = _cache(owned_layers(owners, rank), slots=slots, seed=rank)
    working = _cache(range(_NUM_LAYERS), slots=slots, seed=100 + rank)
    seed = options.seed * 100 + rank
    data_class = _NoWait if options.deaf == "data" else FakeStream
    exec_class = _NoWait if options.deaf == "exec" else FakeStream
    data_stream = data_class(f"data{rank}", timeout=_TIMEOUT, jitter=options.data_jitter, seed=seed)
    exec_stream = exec_class(
        f"exec{rank}", timeout=_TIMEOUT, jitter=options.exec_jitter, seed=seed + 50
    )
    streamer = DkvStreamer(
        manager,
        working,
        _corrupting(FakeTransport(network, rank, grouped=options.grouped), rank, options),
        owners,
        rank,
        group_size=group_size,
        ring_depth=scenario.ring_depth,
        copier=FakeCopier(),
        data_stream=data_stream,
        current_stream=lambda: exec_stream,
        debug=HostDataPlaneDebug(data_stream) if options.debug else None,
        fault=options.fault,
        timing_event_factory=lambda: FakeEvent(enable_timing=True),
        device=_HOST,
    )
    written: dict[tuple[int, int, str], set[int]] = {}
    errors: list[str] = []
    records: list = []
    plans: list = []
    try:
        for iteration in range(scenario.iterations):
            batch = scenario.batch(iteration)
            local = [r for r in batch if r.compute_rank == rank and not r.is_dummy]
            for request in batch:
                if not request.is_dummy:
                    manager.prepare(
                        request.request_id,
                        request.context_current_position,
                        request.context_chunk_size,
                    )
            streamer.set_plan(streamer.plan_for(batch), iteration)
            for request in local:
                working.open(
                    request.request_id, request.context_current_position, request.context_chunk_size
                )
            barrier.wait(timeout=_TIMEOUT)
            streamer.begin_iteration(
                [r.request_id for r in local],
                [r.context_current_position for r in local],
                [r.context_chunk_size for r in local],
            )
            for layer in range(_NUM_LAYERS):
                streamer.on_layer(layer)
                exec_stream.enqueue(_attention(layer, local, working))
                if options.stall is not None and (rank, layer) == (
                    options.stall.rank,
                    options.stall.layer,
                ):
                    exec_stream.enqueue(options.stall.hold)
            _record_writes(written, batch, manager)
            streamer.end_forward()
            records.append(streamer.drain(_TIMEOUT))
            plans.append(streamer.last_plan)
            exec_stream.synchronize()
            for request in local:
                working.close(request.request_id)
            for stream in (data_stream, exec_stream):
                errors.extend(f"{stream.name}: {error}" for error in stream.errors)
                stream.errors.clear()
        errors.extend(_check_owner_pages(rank, manager, written))
        largest = max((size for sizes in network.sent.values() for size in sizes), default=0)
        return RankReport(
            streamer.stats,
            errors,
            largest,
            max_peer_step_bytes(working),
            streamer._send_arena.numel(),
            records,
            plans,
        )
    finally:
        exec_stream.close()
        data_stream.close()


def _attention(layer: int, local: list, working: FakeV2Cache):
    """Stand-in for the attention of a layer: read what was fetched, write the new pages."""

    def work() -> None:
        for request in local:
            history, chunk = request.context_current_position, request.context_chunk_size
            for index, buffer in enumerate(_LAYERS[layer]):
                first, count = fetch_blocks(buffer.lifecycle(), history, chunk, _BLOCK)
                for block in range(first, first + count):
                    expected = _pattern(request.request_id, layer, index, block, buffer.page_bytes)
                    page = working.page(request.request_id, layer, buffer.role, block)
                    if page is None or not torch.equal(page, expected):
                        raise AssertionError(
                            f"layer {layer}: block {block} of {buffer.role} of request "
                            f"{request.request_id} was not fetched, the working cache holds "
                            f"{'no page' if page is None else _whose(page)}"
                        )
                first, count = writeback_blocks(buffer.lifecycle(), history, chunk, _BLOCK)
                for block in range(first, first + count):
                    working.page(request.request_id, layer, buffer.role, block).copy_(
                        _pattern(request.request_id, layer, index, block, buffer.page_bytes)
                    )

    return work


def _record_writes(written, batch, manager: FakeV2Cache) -> None:
    """Note the pages that the iteration writes back into the cache manager of this rank."""
    for request in batch:
        if request.is_dummy:
            continue
        history, chunk = request.context_current_position, request.context_chunk_size
        for layer in manager.held_layers:
            for buffer in _LAYERS[layer]:
                first, count = writeback_blocks(buffer.lifecycle(), history, chunk, _BLOCK)
                key = (request.request_id, layer, buffer.role)
                written.setdefault(key, set()).update(range(first, first + count))


def _check_owner_pages(rank: int, manager: FakeV2Cache, written) -> list[str]:
    problems = []
    for (request_id, layer, role), blocks in sorted(written.items()):
        index = [buffer.role for buffer in _LAYERS[layer]].index(role)
        for block in sorted(blocks):
            page = manager.page(request_id, layer, role, block)
            if page is None:  # the block left the window of its life cycle
                continue
            if not torch.equal(page, _pattern(request_id, layer, index, block, page.numel())):
                problems.append(
                    f"rank {rank}: block {block} of request {request_id} in layer {layer} "
                    f"({role}) was not written back, the page holds {_whose(page)}"
                )
    return problems


def run_group(
    scenario: Scenario, options: Options = Options(), network: FakeNetwork | None = None
) -> list[RankReport]:
    network = network or FakeNetwork(scenario.group_size, timeout=_TIMEOUT)
    barrier = threading.Barrier(scenario.group_size)
    reports: list[RankReport | None] = [None] * scenario.group_size
    failures: list[BaseException] = []

    def target(rank: int) -> None:
        try:
            reports[rank] = _run_rank(rank, scenario, network, options, barrier)
        except BaseException as error:  # the other ranks are told by the barrier timeout
            failures.append(error)
            barrier.abort()

    threads = [
        threading.Thread(target=target, args=(rank,), daemon=True)
        for rank in range(scenario.group_size)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=60.0)
    assert not any(thread.is_alive() for thread in threads), "a rank did not finish"
    if failures:
        raise failures[0]
    return reports


def _spread(group_size: int, prompts: dict[int, int]) -> dict[int, int]:
    return {request_id: request_id % group_size for request_id in prompts}


_PROMPTS = {1: 300, 2: 700, 3: 150, 4: 1000, 5: 90}


@pytest.mark.parametrize("ring_depth", [1, 2, 3])
@pytest.mark.parametrize("group_size", [1, 2, 3, 4])
def test_the_pages_of_every_layer_reach_the_compute_rank_and_come_back(
    group_size: int, ring_depth: int
) -> None:
    scenario = Scenario(
        group_size, ring_depth, _PROMPTS, _spread(group_size, _PROMPTS), arrival={4: 1, 5: 2}
    )
    options = Options(data_jitter=0.001, exec_jitter=0.001, seed=group_size)
    reports = run_group(scenario, options)
    for rank, report in enumerate(reports):
        assert report.errors == [], f"rank {rank}"
        assert report.largest_message <= report.step_limit <= report.buffer_bytes
    stats = [report.stats for report in reports]
    assert sum(s.bytes_sent for s in stats) == sum(s.bytes_received for s in stats)
    assert sum(s.messages_sent for s in stats) == sum(s.messages_received for s in stats)
    assert all(s.iterations == scenario.iterations for s in stats)
    # The hooks of the forward pass and the wait of the drain cost the host some time.
    assert all(s.hook_seconds > 0 and s.drain_seconds >= 0 for s in stats)
    if group_size > 1:
        assert sum(s.bytes_sent for s in stats) > 0
    assert sum(s.bytes_local for s in stats) > 0


def test_the_pages_arrive_when_the_transport_takes_one_message_at_a_time() -> None:
    # A transport that takes one message at a time pairs the messages up like the grouped one.
    scenario = Scenario(3, 2, _PROMPTS, _spread(3, _PROMPTS), arrival={4: 1, 5: 2})
    options = Options(data_jitter=0.001, exec_jitter=0.001, seed=7, grouped=False)
    reports = run_group(scenario, options)
    for rank, report in enumerate(reports):
        assert report.errors == [], f"rank {rank}"
    stats = [report.stats for report in reports]
    assert sum(s.bytes_sent for s in stats) == sum(s.bytes_received for s in stats) > 0
    assert sum(s.messages_sent for s in stats) == sum(s.messages_received for s in stats)
    assert all(s.compile_seconds > 0 for s in stats)
    for rank, report in enumerate(reports):
        local_bytes = Counter()
        sent = 0
        for plan in report.plans:
            for step in plan.steps:
                for transfer in step.transfers:
                    if transfer.owner == transfer.compute == rank:
                        local_bytes[transfer.direction] += transfer.nbytes
                sent += sum(op.action is OpAction.SEND for op in step.ops_for(rank))
        assert report.stats.bytes_local_fetch == local_bytes[Direction.FETCH]
        assert report.stats.bytes_local_writeback == local_bytes[Direction.WRITEBACK]
        assert report.stats.bytes_local == sum(local_bytes.values())
        # One message per peer and step: what the plan says this rank sends.
        assert report.stats.messages_sent == sent


@pytest.mark.parametrize("group_size", [2, 3, 4])
def test_ranks_without_requests_still_serve_the_layers_they_own(group_size: int) -> None:
    prompts = {1: 400, 2: 260}
    scenario = Scenario(group_size, 2, prompts, {1: 0, 2: 0})
    reports = run_group(scenario, Options(data_jitter=0.001, exec_jitter=0.001))
    for rank, report in enumerate(reports):
        assert report.errors == [], f"rank {rank}"
    assert reports[group_size - 1].stats.messages_sent > 0
    assert reports[group_size - 1].stats.messages_received > 0


def test_a_rank_that_computes_nothing_receives_a_layer_only_after_its_forward_pass_got_there() -> (
    None
):
    """The host of an idle rank runs far ahead of its GPU. A receive that does not wait for the
    layers before it would start at once and spin on the GPU until the compute rank sends."""
    group_size, stalled_layer = 2, 6
    scenario = Scenario(group_size, 2, {1: 100}, {1: 0})
    owners = compute_ownership(_NUM_LAYERS, group_size)
    plan = build_dkv_plan(scenario.batch(0), owners, _costs(), group_size=group_size, ring_depth=2)

    def messages(layers) -> int:
        return sum(len(plan.transfers(Direction.WRITEBACK, layer)) for layer in layers)

    served = [layer for layer in range(_NUM_LAYERS) if owners[layer] == 1]
    # The writeback of layer l is enqueued at the top of layer l + 1, behind the attention of l.
    reachable = [layer for layer in served if layer < stalled_layer]
    assert 0 < messages(reachable) < messages(served)

    network = FakeNetwork(group_size, timeout=_TIMEOUT)
    stall = Stall(rank=1, layer=stalled_layer)
    outcome: list = []

    def run() -> None:
        try:
            outcome.append(run_group(scenario, Options(stall=stall), network))
        except BaseException as error:  # reported by the assertion below
            outcome.append(error)

    runner = threading.Thread(target=run, daemon=True)
    runner.start()
    assert stall.reached.wait(_TIMEOUT)
    time.sleep(0.5)  # a data stream that did not wait would have taken every message by now
    received_while_stalled = len(network.received[0, 1])
    stall.release.set()
    runner.join(timeout=60.0)
    assert not runner.is_alive(), "the group did not finish after the stall was released"
    assert not isinstance(outcome[0], BaseException), outcome[0]
    assert [report.errors for report in outcome[0]] == [[], []]
    assert received_while_stalled == messages(reachable)
    assert len(network.received[0, 1]) == messages(served)


def _busy_scenario() -> Scenario:
    prompts = {request_id: 120 + 97 * request_id for request_id in range(1, 9)}
    return Scenario(
        4,
        2,
        prompts,
        _spread(4, prompts),
        arrival={request_id: request_id // 3 for request_id in prompts},
    )


@pytest.mark.parametrize("seed", range(3))
def test_a_busy_group_with_many_requests_on_every_rank(seed: int) -> None:
    options = Options(data_jitter=0.0015, exec_jitter=0.0015, seed=seed)
    reports = run_group(_busy_scenario(), options)
    assert [report.errors for report in reports] == [[], [], [], []]


# What the tests above need from the test itself


def test_a_stream_that_waits_for_a_never_recorded_event_fails_instead_of_hanging() -> None:
    stream = FakeStream("probe", timeout=0.2)
    stream.wait_event(FakeEvent())
    stream.record_event().wait(2.0)
    assert len(stream.errors) == 1 and isinstance(stream.errors[0], TimeoutError)
    stream.close()


def test_the_cost_model_names_the_buffers_of_every_layer_and_their_pages() -> None:
    costs = _costs()
    for layer, buffers in enumerate(_LAYERS):
        # Every rank lists the buffers of a layer in the same order: by role.
        assert costs.buffers_of(layer) == tuple(sorted(buffer.role for buffer in buffers))
        for buffer in buffers:
            for history, chunk in [(0, 128), (128, 128), (300, 100), (1000, 24), (5, 1)]:
                cost = costs.page_cost(layer, buffer.role, history, chunk)
                key = buffer.lifecycle()
                assert cost.page_bytes == buffer.page_bytes
                assert cost.fetch_pages == fetch_blocks(key, history, chunk, _BLOCK)[1]
                assert cost.writeback_pages == writeback_blocks(key, history, chunk, _BLOCK)[1]


# The mutations: without one of the events of the streamer the scenario must fail.

_MUTATION_SCENARIO = Scenario(3, 2, {1: 500, 2: 520, 3: 380}, {1: 0, 2: 1, 3: 2}, arrival={})


def test_a_forward_pass_that_does_not_wait_for_the_fetch_reads_pages_that_have_not_arrived() -> (
    None
):
    reports = run_group(_MUTATION_SCENARIO, Options(data_jitter=0.01, deaf="exec"))
    errors = [error for report in reports for error in report.errors]
    assert any("was not fetched" in error for error in errors), errors


def test_a_data_stream_that_does_not_wait_for_the_forward_pass_corrupts_the_cache() -> None:
    reports = run_group(_MUTATION_SCENARIO, Options(exec_jitter=0.01, deaf="data"))
    errors = [error for report in reports for error in report.errors]
    assert errors, "the corruption of the owner pages or of the working cache went unnoticed"


@pytest.mark.parametrize("which", ["first", "middle", "last"])
def test_a_byte_flipped_in_a_message_is_noticed(which: str) -> None:
    scenario = Scenario(2, 2, {1: 400, 2: 520}, {1: 0, 2: 1})
    owners = compute_ownership(_NUM_LAYERS, 2)
    sent = sum(
        op.action is OpAction.SEND
        for iteration in range(scenario.iterations)
        for op in build_dkv_plan(
            scenario.batch(iteration), owners, _costs(), group_size=2, ring_depth=2
        ).rank_program(0)
    )
    # Rank 0 fetches four layers for rank 1 in four iterations and writes four back in four.
    assert sent == 32
    number = {"first": 0, "middle": sent // 2, "last": sent - 1}[which]
    reports = run_group(scenario, Options(corrupt=(0, number)))
    errors = [error for report in reports for error in report.errors]
    assert errors, f"the flip of the byte of message {number} went unnoticed"


# One rank, to see what the streamer does with a plan that does not fit and with no plan at all


class _Single:
    def __init__(
        self,
        ring_depth: int = 2,
        *,
        timing_event_factory: Callable[[], FakeEvent] | None = None,
    ) -> None:
        self.owners = compute_ownership(_NUM_LAYERS, 1)
        self.data = FakeStream("data", timeout=1.0)
        self.exec = FakeStream("exec", timeout=1.0)
        self.manager = _cache(range(_NUM_LAYERS), slots=8)
        self.working = _cache(range(_NUM_LAYERS), slots=8, seed=1)
        self.streamer = DkvStreamer(
            self.manager,
            self.working,
            FakeTransport(FakeNetwork(1), 0),
            self.owners,
            0,
            group_size=1,
            ring_depth=ring_depth,
            copier=FakeCopier(),
            data_stream=self.data,
            current_stream=lambda: self.exec,
            timing_event_factory=timing_event_factory,
            device=_HOST,
        )

    def stage(self, request_id: int, history: int, chunk: int) -> None:
        """What the executor does for a request before the forward pass, in both caches."""
        self.manager.prepare(request_id, history, chunk)
        self.working.open(request_id, history, chunk)

    def run(self, request_ids: list[int], history: list[int], new: list[int]) -> None:
        """The hooks of one forward pass, then the drain."""
        streamer = self.streamer
        streamer.begin_iteration(request_ids, history, new)
        for layer in range(_NUM_LAYERS):
            streamer.on_layer(layer)
        streamer.end_forward()
        streamer.drain(5.0)

    def close(self) -> None:
        self.data.close()
        self.exec.close()


@pytest.fixture
def single():
    rank = _Single()
    yield rank
    rank.close()


def test_a_forward_pass_without_a_plan_does_nothing(single) -> None:
    streamer = single.streamer
    single.run([1], [0], [128])
    assert streamer.stats == type(streamer.stats)()
    assert single.data._queue.empty() and single.exec._queue.empty()


def test_a_plan_has_to_be_carried_out_before_the_next_one_is_set(single) -> None:
    streamer = single.streamer
    plan = streamer.plan_for([PlanRequest(1, 0, 0, 128)])
    streamer.set_plan(plan)
    with pytest.raises(RuntimeError, match="previous forward pass"):
        streamer.set_plan(plan)


def test_a_plan_of_another_ring_is_rejected(single) -> None:
    other = _Single(ring_depth=3)
    try:
        plan = other.streamer.plan_for([PlanRequest(1, 0, 0, 128)])
        with pytest.raises(RuntimeError, match="ring depth"):
            single.streamer.set_plan(plan)
    finally:
        other.close()


def test_the_compiled_plan_keeps_its_own_copy_of_the_page_index_tables(single) -> None:
    """A manager's table is a view that its next lifecycle operation invalidates.

    The copies of the forward pass are enqueued during the whole pass, so the plan must not point
    into that view.
    """
    single.stage(1, 128, 128)
    views: dict[tuple[bool, int, int], np.ndarray] = {}
    for working, cache in ((False, single.manager), (True, single.working)):
        request = cache.kv_cache_map[1]
        original = request.get_base_page_indices

        def persistent(group: int, original=original, working=working) -> np.ndarray:
            return views.setdefault((working, 1, group), original(group))

        request.get_base_page_indices = persistent
    streamer = single.streamer
    streamer.set_plan(streamer.plan_for([PlanRequest(1, 0, 128, 128)]))
    streamer.begin_iteration([1], [128], [128])
    assert {working for working, _, _ in streamer._indices} == {False, True}
    for key, indices in streamer._indices.items():
        assert not np.shares_memory(indices, views[key])
        views[key][:] = -1  # what a lifecycle operation leaves behind
    for layer in range(_NUM_LAYERS):
        streamer.on_layer(layer)
    streamer.end_forward()
    streamer.drain(5.0)
    assert single.data.errors == []
    assert streamer.stats.local_copies > 0


def test_a_group_of_one_plans_the_batch_its_working_cache_stages(single) -> None:
    """A rank that owns every layer moves whatever the attention stages: a decoding request as
    much as a chunk of a prompt."""
    single.stage(1, 300, 1)
    single.stage(2, 0, 128)

    def fetched():
        for layer, buffers in enumerate(_LAYERS):
            for index, buffer in enumerate(buffers):
                first, count = fetch_blocks(buffer.lifecycle(), 300, 1, _BLOCK)
                for block in range(first, first + count):
                    yield (
                        layer,
                        buffer.role,
                        block,
                        _pattern(1, layer, index, block, buffer.page_bytes),
                    )

    for layer, role, block, pattern in fetched():
        single.manager.page(1, layer, role, block).copy_(pattern)
    streamer = single.streamer
    streamer.set_staged_plan(4)
    with pytest.raises(RuntimeError, match="previous forward pass"):
        streamer.set_staged_plan(5)
    single.run([1, 2], [300, 0], [1, 128])
    planned = [
        (r.request_id, r.compute_rank, r.context_current_position, r.context_chunk_size)
        for r in streamer.last_plan.requests
    ]
    assert planned == [(1, 0, 300, 1), (2, 0, 0, 128)]
    for layer, role, block, pattern in fetched():
        assert torch.equal(single.working.page(1, layer, role, block), pattern), (layer, role)
    assert single.data.errors == []
    # A forward pass that stages no batch leaves nothing planned for the next one.
    streamer.set_staged_plan(5)
    streamer.end_forward()
    streamer.set_staged_plan(6)


def test_only_a_group_of_one_plans_the_batch_it_stages() -> None:
    owners = compute_ownership(_NUM_LAYERS, 2)
    stream = FakeStream("never used")
    try:
        streamer = DkvStreamer(
            _cache(owned_layers(owners, 0), slots=2),
            _cache(range(_NUM_LAYERS), slots=2),
            FakeTransport(FakeNetwork(2), 0),
            owners,
            0,
            group_size=2,
            copier=FakeCopier(),
            data_stream=stream,
            device=_HOST,
        )
        with pytest.raises(RuntimeError, match="only a group of one rank"):
            streamer.set_staged_plan()
    finally:
        stream.close()


def test_a_staged_batch_that_is_not_the_one_of_the_plan_is_rejected(single) -> None:
    streamer = single.streamer
    streamer.set_plan(streamer.plan_for([PlanRequest(1, 0, 128, 128)]))
    with pytest.raises(RuntimeError, match="request 1"):
        streamer.begin_iteration([1], [128], [64])


def test_a_request_without_pages_in_the_working_cache_is_named(single) -> None:
    single.manager.prepare(1, 128, 128)
    streamer = single.streamer
    streamer.set_plan(streamer.plan_for([PlanRequest(1, 0, 128, 128)]))
    with pytest.raises(RuntimeError, match="request 1 has no pages in the working cache"):
        streamer.begin_iteration([1], [128], [128])


def test_a_second_begin_of_the_same_iteration_does_not_issue_the_first_fetches_again(
    single,
) -> None:
    streamer = single.streamer
    single.stage(1, 128, 128)
    streamer.set_plan(streamer.plan_for([PlanRequest(1, 0, 128, 128)]))
    streamer.begin_iteration([1], [128], [128])
    copies = streamer.stats.local_copies
    assert copies > 0
    streamer.begin_iteration([1], [128], [128])
    assert streamer.stats.local_copies == copies


def test_the_data_plane_is_idle_only_when_it_is_drained(single) -> None:
    streamer = single.streamer
    single.stage(1, 0, 128)
    streamer.assert_idle("a page is freed")
    streamer.set_plan(streamer.plan_for([PlanRequest(1, 0, 0, 128)]))
    with pytest.raises(RuntimeError, match="a page is freed while the data plane of rank 0"):
        streamer.assert_idle("a page is freed")
    streamer.begin_iteration([1], [0], [128])
    for layer in range(_NUM_LAYERS):
        streamer.on_layer(layer)
    streamer.end_forward()
    # The plan is carried out, but the host has not seen the data stream finish.
    with pytest.raises(RuntimeError, match="not drained"):
        streamer.assert_idle("a page is moved")
    streamer.drain(5.0)
    streamer.assert_idle("a page is moved")


def test_drain_gives_up_when_the_data_stream_never_finishes(single) -> None:
    streamer = single.streamer
    single.data.wait_event(FakeEvent())
    streamer._data_done = single.data.record_event()
    with pytest.raises(RuntimeError, match="did not finish"):
        streamer.drain(0.05)


def test_fetch_wait_timing_brackets_compute_waits_and_accumulates_once_per_iteration(
    monkeypatch,
) -> None:
    monkeypatch.setenv("TRTLLM_DKV_WAIT_TIMING", "1")
    # Each layer contributes a distinct duration, independent of thread scheduling.
    durations = [(layer + 1) / 1000.0 for layer in range(_NUM_LAYERS)]
    timestamps = iter(
        timestamp
        for iteration in range(2)
        for layer, duration in enumerate(durations)
        for timestamp in (iteration * 100 + layer, iteration * 100 + layer + duration)
    )
    events: list[FakeEvent] = []

    def event_factory() -> FakeEvent:
        event = FakeEvent(enable_timing=True, clock=lambda: next(timestamps))
        events.append(event)
        return event

    rank = _Single(timing_event_factory=event_factory)
    calls: list[tuple[str, FakeEvent | None]] = []
    record_event, wait_event = rank.exec.record_event, rank.exec.wait_event

    def record(event: FakeEvent | None = None) -> FakeEvent:
        calls.append(("record", event))
        return record_event(event)

    def wait(event: FakeEvent) -> None:
        calls.append(("wait", event))
        wait_event(event)

    monkeypatch.setattr(rank.exec, "record_event", record)
    monkeypatch.setattr(rank.exec, "wait_event", wait)
    try:
        streamer = rank.streamer
        for iteration in range(2):
            history = iteration * 128
            rank.stage(1, history, 128)
            streamer.set_plan(streamer.plan_for([PlanRequest(1, 0, history, 128)]))
            streamer.begin_iteration([1], [history], [128])
            for layer in range(_NUM_LAYERS):
                streamer.on_layer(layer)
                assert calls[-3:] == [
                    ("record", events[-2]),
                    ("wait", calls[-2][1]),
                    ("record", events[-1]),
                ]
                assert calls[-2][1] not in events
            streamer.end_forward()
            assert streamer.stats.fetch_wait_seconds == pytest.approx(iteration * sum(durations))
            streamer.drain(5.0)
            assert streamer.stats.fetch_wait_seconds == pytest.approx(
                (iteration + 1) * sum(durations)
            )
            assert streamer._fetch_wait_events == []
            streamer.drain(5.0)
            assert streamer.stats.fetch_wait_seconds == pytest.approx(
                (iteration + 1) * sum(durations)
            )
            rank.working.close(1)
        assert len(events) == 4 * _NUM_LAYERS
        assert rank.data.errors == rank.exec.errors == []
    finally:
        rank.close()


@pytest.mark.parametrize("setting", [None, "0"])
def test_disabled_fetch_wait_timing_creates_no_events(monkeypatch, setting: str | None) -> None:
    if setting is None:
        monkeypatch.delenv("TRTLLM_DKV_WAIT_TIMING", raising=False)
    else:
        monkeypatch.setenv("TRTLLM_DKV_WAIT_TIMING", setting)

    def forbidden_event() -> FakeEvent:
        pytest.fail("disabled fetch wait timing must not create timing events")

    rank = _Single(timing_event_factory=forbidden_event)
    try:
        streamer = rank.streamer
        rank.stage(1, 128, 128)
        streamer.set_plan(streamer.plan_for([PlanRequest(1, 0, 128, 128)]))
        rank.run([1], [128], [128])
        assert streamer.stats.fetch_wait_seconds == 0.0
        assert rank.data.errors == rank.exec.errors == []
    finally:
        rank.close()


def test_fetch_wait_timing_measures_only_ranks_that_compute(monkeypatch) -> None:
    monkeypatch.setenv("TRTLLM_DKV_WAIT_TIMING", "1")
    scenario = Scenario(2, 2, {1: 300}, {1: 0})
    reports = run_group(scenario, Options(data_jitter=0.001))
    assert [report.errors for report in reports] == [[], []]
    assert reports[0].stats.fetch_wait_seconds > 0.0
    assert reports[1].stats.fetch_wait_seconds == 0.0


# A streamer and caches that do not fit together


def test_a_working_cache_that_lacks_a_layer_is_refused() -> None:
    stream = FakeStream("never used")
    try:
        with pytest.raises(ValueError, match="the working cache holds the layers"):
            DkvStreamer(
                _cache(range(_NUM_LAYERS), slots=2),
                _cache(range(_NUM_LAYERS - 1), slots=2),
                None,
                compute_ownership(_NUM_LAYERS, 1),
                0,
                group_size=1,
                copier=FakeCopier(),
                data_stream=stream,
                device=_HOST,
            )
    finally:
        stream.close()


def test_a_rank_whose_cache_lacks_a_layer_it_owns_is_refused() -> None:
    owners = compute_ownership(_NUM_LAYERS, 2)
    held = [layer for layer in owned_layers(owners, 1) if layer != owned_layers(owners, 1)[-1]]
    stream = FakeStream("never used")
    try:
        with pytest.raises(ValueError, match="rank 1 owns layer"):
            DkvStreamer(
                _cache(held, slots=2),
                _cache(range(_NUM_LAYERS), slots=2),
                FakeTransport(FakeNetwork(2), 1),
                owners,
                1,
                group_size=2,
                copier=FakeCopier(),
                data_stream=stream,
                device=_HOST,
            )
    finally:
        stream.close()


# The debug checksums: every message at both ends and on the pages it came from or went to


def _mismatches(reports: list[RankReport]) -> list[str]:
    problems = []
    for iteration in range(len(reports[0].records)):
        problems += find_data_plane_mismatches(
            [report.records[iteration] for report in reports], reports[0].plans[iteration]
        )
    return problems


@pytest.mark.parametrize("group_size", [1, 2, 3])
def test_the_checksums_of_a_healthy_run_agree(group_size: int) -> None:
    prompts = {1: 400, 2: 520, 3: 130}
    scenario = Scenario(group_size, 2, prompts, _spread(group_size, prompts))
    reports = run_group(scenario, Options(debug=True, data_jitter=0.001))
    assert [report.errors for report in reports] == [[]] * group_size
    assert _mismatches(reports) == []
    roles = Counter(
        role for report in reports for records in report.records for _, role, _ in records
    )
    # The test is not vacuous: every message was checksummed where it was packed, where it arrived
    # and in the pages it was unpacked into, and every local copy at both of its ends.
    if group_size > 1:
        assert roles["sent"] == roles["received"] > 0
        assert roles["stored"] == roles["received"] + roles["source"]
    assert roles["source"] > 0


@pytest.mark.parametrize("kind", ["flip", "zero"])
@pytest.mark.parametrize(
    ("direction", "iteration", "layer", "rank"),
    [
        (Direction.WRITEBACK, 0, 5, 1),
        (Direction.FETCH, 1, 1, 1),
        (Direction.FETCH, 1, 6, 0),
        (Direction.WRITEBACK, 1, 2, 0),
    ],
)
def test_a_damaged_message_is_named_by_the_checksums(
    kind: str, direction: Direction, iteration: int, layer: int, rank: int
) -> None:
    scenario = Scenario(2, 2, {1: 400, 2: 520}, {1: 0, 2: 1})
    fault = DataPlaneFault(iteration, layer, direction, kind, rank)
    problems = _mismatches(run_group(scenario, Options(debug=True, fault=fault)))
    label = ("F" if direction is Direction.FETCH else "W") + f"({layer}) of requests"
    # A step carries one message to the rank, whatever buffers the layer has. It arrived different
    # from what was sent, and so did the pages it was unpacked into; nothing else differs.
    assert len(problems) == 2 and all(label in problem for problem in problems), problems
    assert all(f"rank {rank} has " in problem for problem in problems), problems


def test_a_run_whose_fault_never_happens_has_no_mismatch() -> None:
    scenario = Scenario(2, 2, {1: 400, 2: 520}, {1: 0, 2: 1})
    fault = DataPlaneFault(99, 1, Direction.FETCH)
    assert _mismatches(run_group(scenario, Options(debug=True, fault=fault))) == []


def test_the_mismatch_finder_names_what_differs_and_what_is_missing() -> None:
    fetch = (int(Direction.FETCH), 3, 0, 1)
    assert (
        find_data_plane_mismatches(
            [[(fetch, "sent", 7)], [(fetch, "received", 7), (fetch, "stored", 7)]]
        )
        == []
    )
    (problem,) = find_data_plane_mismatches(
        [[(fetch, "sent", 7)], [(fetch, "received", 7), (fetch, "stored", 9)]]
    )
    assert "rank 1 has stored 0x9, not 0x7" in problem and "F(3) of requests" in problem
    assert "owner rank 0, compute rank 1" in problem
    (problem,) = find_data_plane_mismatches([[(fetch, "sent", 7)], []])
    assert "recorded ['sent'] instead of ['received', 'sent', 'stored']" in problem
    local = (int(Direction.WRITEBACK), 1, 2, 2)
    assert find_data_plane_mismatches([[], [], [(local, "source", 5), (local, "stored", 5)]]) == []
    (problem,) = find_data_plane_mismatches([[], [], [(local, "source", 5), (local, "stored", 6)]])
    assert "W(1) of requests" in problem and "rank 2 has stored" in problem


def test_the_checksum_depends_on_every_word_and_on_its_position() -> None:
    data = torch.arange(64, dtype=torch.uint8)
    reference = message_checksum(data)
    assert message_checksum(data.clone()) == reference
    flipped = data.clone()
    flipped[17] ^= 1
    assert message_checksum(flipped) != reference
    swapped = data.clone()
    swapped[:8], swapped[8:16] = data[8:16].clone(), data[:8].clone()
    assert message_checksum(swapped) != reference
    assert message_checksum(torch.zeros(64, dtype=torch.uint8)) != message_checksum(
        torch.zeros(72, dtype=torch.uint8)
    )


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("3:7:fetch", DataPlaneFault(3, 7, Direction.FETCH)),
        ("0:2:writeback:zero:1", DataPlaneFault(0, 2, Direction.WRITEBACK, "zero", 1)),
        ("5:0:fetch:flip", DataPlaneFault(5, 0, Direction.FETCH, "flip", None)),
        ("*:3:writeback", DataPlaneFault(None, 3, Direction.WRITEBACK)),
    ],
)
def test_a_fault_is_read_from_its_text(text: str, expected: DataPlaneFault) -> None:
    assert DataPlaneFault.parse(text) == expected


@pytest.mark.parametrize(
    "text",
    ["", "1:2", "1:2:both", "a:2:fetch", "1:2:fetch:melt", "1:2:fetch:flip:x", "1:2:3:4:5:6"],
)
def test_a_fault_that_is_not_in_the_form_is_rejected(text: str) -> None:
    with pytest.raises(ValueError, match="is not iteration"):
        DataPlaneFault.parse(text)


def test_a_fault_needs_the_checksums_to_be_noticed() -> None:
    stream = FakeStream("never used")
    try:
        with pytest.raises(ValueError, match="only noticed by the checksums"):
            DkvStreamer(
                _cache(range(_NUM_LAYERS), slots=2),
                _cache(range(_NUM_LAYERS), slots=2),
                FakeTransport(FakeNetwork(1), 0),
                compute_ownership(_NUM_LAYERS, 1),
                0,
                group_size=1,
                copier=FakeCopier(),
                data_stream=stream,
                fault=DataPlaneFault(0, 0, Direction.FETCH),
                device=_HOST,
            )
    finally:
        stream.close()
