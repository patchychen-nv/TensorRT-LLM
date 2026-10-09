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

Every rank of a group runs a streamer on host memory with stand-ins for the streams, the copy kernel
and the interconnect (``dkv_fake_dataplane``). A stand-in for the attention of a layer reads the
pages the streamer fetched into the slot of the layer, checks them against what an earlier
iteration wrote, and writes the pages its new tokens touch. After the last iteration the pages in
the cache manager of the owner of each layer must hold what the compute ranks wrote. The runs chunk
prompts over several iterations, so every direction of every step is exercised, with the ring
depths and the placements the plan is checked for, and with random delays on the streams. The
mutations at the end remove one of the events of the streamer and the test has to notice.
"""

import threading
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest
import torch
from dkv_fake_dataplane import (
    FakeCopier,
    FakeEvent,
    FakeKvManager,
    FakeNetwork,
    FakeStream,
    FakeTransport,
    HostDataPlaneDebug,
)

from tensorrt_llm._torch.pyexecutor.dkv import compute_ownership, owned_layers
from tensorrt_llm._torch.pyexecutor.dkv_plan import (
    Direction,
    PlanRequest,
    build_dkv_plan,
    layer_types_from_compress_ratios,
)
from tensorrt_llm._torch.pyexecutor.dkv_staging import (
    BAD_PAGE_INDEX,
    StagingGeometry,
    StagingLayout,
    StagingPool,
)
from tensorrt_llm._torch.pyexecutor.dkv_streamer import (
    DataPlaneFault,
    DkvStreamer,
    LayoutCostModel,
    find_data_plane_mismatches,
    max_message_bytes,
    message_checksum,
)

pytestmark = pytest.mark.cpu_only

_RATIOS = (1, 4, 128, 4, 128, 1, 4, 128)
_POISON = 0xA5
_CHUNK = 128
_TIMEOUT = 15.0


def _geometry(ring_depth: int) -> StagingGeometry:
    return StagingGeometry(
        compress_ratios=_RATIOS,
        tokens_per_block=128,
        head_dim=64,
        index_head_dim=128,
        has_fp8_kv_cache=True,
        indexer_k_dtype="fp8",
        window_size=128,
        max_num_tokens=512,
        max_staging_tokens=4096,
        max_batch_size=4,
        ring_depth=ring_depth,
    )


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


def _whose(page: torch.Tensor, layout: StagingLayout) -> str:
    """What a page holds, for the message of a failed check."""
    if bool((page == _POISON).all()):
        return "poison"
    if not bool(page.any()):
        return "zeros"
    for request_id in range(1, 9):
        for layer in range(len(_RATIOS)):
            for index, component in enumerate(layout.components):
                if layer not in layout.layers_of(component.kind):
                    continue
                if layout.page_bytes(component) != page.numel():
                    continue
                for block in range(10):
                    if torch.equal(page, _pattern(request_id, layer, index, block, page.numel())):
                        return (
                            f"the page of request {request_id}, layer {layer}, "
                            f"{component.attention_type.name}, block {block}"
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
    # The cache manager maps blocks to pages by an affine function of their base page index.
    affine: bool = True


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
    layout = StagingLayout(_geometry(scenario.ring_depth))
    pool = StagingPool(layout, device="cpu")
    pool.buffer.fill_(_POISON)
    owners = compute_ownership(len(_RATIOS), group_size)
    blocks = sum(
        -(-prompt // _geometry(1).tokens_per_block) for prompt in scenario.prompts.values()
    )
    manager = FakeKvManager(
        layout, owned_layers(owners, rank), pages=blocks + 2, seed=rank, affine=options.affine
    )
    seed = options.seed * 100 + rank
    data_class = _NoWait if options.deaf == "data" else FakeStream
    exec_class = _NoWait if options.deaf == "exec" else FakeStream
    data_stream = data_class(f"data{rank}", timeout=_TIMEOUT, jitter=options.data_jitter, seed=seed)
    exec_stream = exec_class(
        f"exec{rank}", timeout=_TIMEOUT, jitter=options.exec_jitter, seed=seed + 50
    )
    streamer = DkvStreamer(
        manager,
        SimpleNamespace(layout=layout, pool=pool),
        _corrupting(FakeTransport(network, rank, grouped=options.grouped), rank, options),
        owners,
        rank,
        group_size=group_size,
        copier=FakeCopier(),
        data_stream=data_stream,
        current_stream=lambda: exec_stream,
        debug=HostDataPlaneDebug(data_stream) if options.debug else None,
        fault=options.fault,
        timing_event_factory=lambda: FakeEvent(enable_timing=True),
    )
    written: dict[tuple[int, int, object], set[int]] = {}
    errors: list[str] = []
    records: list = []
    plans: list = []
    try:
        for iteration in range(scenario.iterations):
            batch = scenario.batch(iteration)
            local = [request for request in batch if request.compute_rank == rank]
            for request in batch:
                if not request.is_dummy:
                    manager.prepare(
                        request.request_id,
                        request.context_current_position,
                        request.context_chunk_size,
                    )
            streamer.set_plan(streamer.plan_for(batch), iteration)
            placement = [(r.context_current_position, r.context_chunk_size) for r in local]
            spans = {kind: layout.request_spans(kind, placement) for kind in layout.kinds}
            barrier.wait(timeout=_TIMEOUT)
            streamer.begin_iteration(
                [r.request_id for r in local],
                [r.context_current_position for r in local],
                [r.context_chunk_size for r in local],
                spans,
            )
            for layer in range(len(_RATIOS)):
                streamer.on_layer(layer)
                exec_stream.enqueue(_attention(layer, local, spans, layout, pool))
                if options.stall is not None and (rank, layer) == (
                    options.stall.rank,
                    options.stall.layer,
                ):
                    exec_stream.enqueue(options.stall.hold)
            _record_writes(written, batch, layout, manager)
            streamer.end_forward()
            records.append(streamer.drain(_TIMEOUT))
            plans.append(streamer.last_plan)
            exec_stream.synchronize()
            for stream in (data_stream, exec_stream):
                errors.extend(f"{stream.name}: {error}" for error in stream.errors)
                stream.errors.clear()
        errors.extend(_check_owner_pages(rank, layout, manager, written))
        largest = max((size for sizes in network.sent.values() for size in sizes), default=0)
        return RankReport(
            streamer.stats, errors, largest, streamer._send_arena.numel(), records, plans
        )
    finally:
        exec_stream.close()
        data_stream.close()


def _attention(layer, local, spans, layout, pool):
    """Stand-in for the attention of a layer: read what was fetched, write the new pages."""

    def staged_page(component, index, block):
        span = spans[component.kind][index]
        page = span.page_offset + block - span.first_block
        return pool.slot(layer, component).view(-1, layout.page_bytes(component))[page]

    def work() -> None:
        for index, request in enumerate(local):
            if request.is_dummy:
                continue
            history, chunk = request.context_current_position, request.context_chunk_size
            for kind in layout.kinds:
                if layer not in layout.layers_of(kind):
                    continue
                for component in layout.components_of(kind):
                    nbytes = layout.page_bytes(component)
                    slot = layout.components.index(component)
                    first, count = layout.fetch_range(kind, history, chunk)
                    for block in range(first, first + count):
                        expected = _pattern(request.request_id, layer, slot, block, nbytes)
                        page = staged_page(component, index, block)
                        if not torch.equal(page, expected):
                            raise AssertionError(
                                f"layer {layer}: block {block} of {component.attention_type.name} "
                                f"of request {request.request_id} was not fetched, the slot holds "
                                f"{_whose(page, layout)}"
                            )
                    first, count = layout.writeback_range(kind, history, chunk)
                    for block in range(first, first + count):
                        staged_page(component, index, block).copy_(
                            _pattern(request.request_id, layer, slot, block, nbytes)
                        )

    return work


def _record_writes(written, batch, layout, manager) -> None:
    """Note the pages that the iteration writes back into the cache manager of this rank."""
    for request in batch:
        if request.is_dummy:
            continue
        history, chunk = request.context_current_position, request.context_chunk_size
        for kind in layout.kinds:
            first, count = layout.writeback_range(kind, history, chunk)
            for component in layout.components_of(kind):
                for layer in layout.layers_of(kind):
                    if (layer, component.attention_type) in manager._buffers:
                        key = (request.request_id, layer, component.attention_type)
                        written.setdefault(key, set()).update(range(first, first + count))


def _check_owner_pages(rank, layout, manager, written) -> list[str]:
    problems = []
    for (request_id, layer, attention_type), blocks in sorted(
        written.items(), key=lambda item: (item[0][0], item[0][1], item[0][2].value)
    ):
        if (layer, attention_type) not in manager._buffers:
            continue
        component = next(
            c
            for c in layout.components
            if c.attention_type is attention_type and layer in layout.layers_of(c.kind)
        )
        slot = layout.components.index(component)
        indices = manager.get_cache_indices(request_id, layer, attention_type)
        buffer = manager.get_buffers(layer, attention_type)
        for block in sorted(blocks):
            if block >= len(indices) or indices[block] == BAD_PAGE_INDEX:
                continue
            expected = _pattern(request_id, layer, slot, block, layout.page_bytes(component))
            if not torch.equal(buffer[indices[block]], expected):
                problems.append(
                    f"rank {rank}: block {block} of request {request_id} in layer {layer} "
                    f"({attention_type.name}) was not written back, the page holds "
                    f"{_whose(buffer[indices[block]], layout)}"
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
    reports = run_group(scenario, Options(data_jitter=0.001, exec_jitter=0.001, seed=group_size))
    for rank, report in enumerate(reports):
        assert report.errors == [], f"rank {rank}"
        assert report.largest_message <= report.buffer_bytes
    stats = [report.stats for report in reports]
    assert sum(s.bytes_sent for s in stats) == sum(s.bytes_received for s in stats)
    assert sum(s.messages_sent for s in stats) == sum(s.messages_received for s in stats)
    assert all(s.iterations == scenario.iterations for s in stats)
    # The hooks of the forward pass and the wait of the drain cost the host some time.
    assert all(s.hook_seconds > 0 and s.drain_seconds >= 0 for s in stats)
    if group_size > 1:
        assert sum(s.bytes_sent for s in stats) > 0
    assert sum(s.bytes_local for s in stats) > 0


@pytest.mark.parametrize("grouped,affine", [(False, True), (True, False), (False, False)])
def test_the_pages_arrive_without_grouped_messages_and_without_the_affine_page_map(
    grouped: bool, affine: bool
) -> None:
    # A transport that takes one message at a time pairs the messages up like the grouped one,
    # and a cache manager that only lists page indices per (request, layer, buffer) is served too.
    scenario = Scenario(3, 2, _PROMPTS, _spread(3, _PROMPTS), arrival={4: 1, 5: 2})
    options = Options(data_jitter=0.001, exec_jitter=0.001, seed=7, grouped=grouped, affine=affine)
    reports = run_group(scenario, options)
    for rank, report in enumerate(reports):
        assert report.errors == [], f"rank {rank}"
    stats = [report.stats for report in reports]
    assert sum(s.bytes_sent for s in stats) == sum(s.bytes_received for s in stats) > 0
    assert sum(s.messages_sent for s in stats) == sum(s.messages_received for s in stats)
    assert all(s.compile_seconds > 0 for s in stats)
    for rank, report in enumerate(reports):
        local_bytes = Counter()
        for plan in report.plans:
            for step in plan.steps:
                for transfer in step.transfers:
                    if transfer.owner == transfer.compute == rank:
                        local_bytes[transfer.direction] += transfer.nbytes
        assert report.stats.bytes_local_fetch == local_bytes[Direction.FETCH]
        assert report.stats.bytes_local_writeback == local_bytes[Direction.WRITEBACK]
        assert report.stats.bytes_local == sum(local_bytes.values())


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
    owners = compute_ownership(len(_RATIOS), group_size)
    plan = build_dkv_plan(
        scenario.batch(0),
        owners,
        layer_types_from_compress_ratios(_RATIOS),
        LayoutCostModel(StagingLayout(_geometry(2))),
        group_size=group_size,
        ring_depth=2,
    )

    def messages(layers) -> int:
        return sum(len(plan.transfers(Direction.WRITEBACK, layer)) for layer in layers)

    served = [layer for layer in range(len(_RATIOS)) if owners[layer] == 1]
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


@pytest.mark.parametrize("seed", range(3))
def test_a_busy_group_with_many_requests_on_every_rank(seed: int) -> None:
    prompts = {request_id: 120 + 97 * request_id for request_id in range(1, 9)}
    scenario = Scenario(
        4,
        2,
        prompts,
        _spread(4, prompts),
        arrival={request_id: request_id // 3 for request_id in prompts},
    )
    reports = run_group(scenario, Options(data_jitter=0.0015, exec_jitter=0.0015, seed=seed))
    assert [report.errors for report in reports] == [[], [], [], []]


# What the tests above need from the test itself


def test_a_stream_that_waits_for_a_never_recorded_event_fails_instead_of_hanging() -> None:
    stream = FakeStream("probe", timeout=0.2)
    stream.wait_event(FakeEvent())
    stream.record_event().wait(2.0)
    assert len(stream.errors) == 1 and isinstance(stream.errors[0], TimeoutError)
    stream.close()


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
    assert errors, "the corruption of the owner pages or of the slots went unnoticed"


@pytest.mark.parametrize("number", [0, 7, 40])
def test_a_byte_flipped_in_a_message_is_noticed(number: int) -> None:
    scenario = Scenario(2, 2, {1: 400, 2: 520}, {1: 0, 2: 1})
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
        self.layout = StagingLayout(_geometry(ring_depth))
        self.pool = StagingPool(self.layout, device="cpu")
        self.owners = compute_ownership(len(_RATIOS), 1)
        self.data = FakeStream("data", timeout=1.0)
        self.exec = FakeStream("exec", timeout=1.0)
        self.manager = FakeKvManager(self.layout, range(len(_RATIOS)), pages=8)
        self.streamer = DkvStreamer(
            self.manager,
            SimpleNamespace(layout=self.layout, pool=self.pool),
            FakeTransport(FakeNetwork(1), 0),
            self.owners,
            0,
            group_size=1,
            copier=FakeCopier(),
            data_stream=self.data,
            current_stream=lambda: self.exec,
            timing_event_factory=timing_event_factory,
        )

    def spans(self, placement):
        return {kind: self.layout.request_spans(kind, placement) for kind in self.layout.kinds}

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
    streamer.begin_iteration([1], [0], [128], single.spans([(0, 128)]))
    for layer in range(len(_RATIOS)):
        streamer.on_layer(layer)
    streamer.end_forward()
    streamer.drain(1.0)
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


def test_a_staged_batch_that_is_not_the_one_of_the_plan_is_rejected(single) -> None:
    streamer = single.streamer
    streamer.set_plan(streamer.plan_for([PlanRequest(1, 0, 128, 128)]))
    with pytest.raises(RuntimeError, match="request 1"):
        streamer.begin_iteration([1], [128], [64], single.spans([(128, 64)]))


def test_a_second_begin_of_the_same_iteration_does_not_issue_the_first_fetches_again(
    single,
) -> None:
    streamer = single.streamer
    single.manager.prepare(1, 128, 128)
    streamer.set_plan(streamer.plan_for([PlanRequest(1, 0, 128, 128)]))
    streamer.begin_iteration([1], [128], [128], single.spans([(128, 128)]))
    copies = streamer.stats.local_copies
    assert copies > 0
    streamer.begin_iteration([1], [128], [128], single.spans([(128, 128)]))
    assert streamer.stats.local_copies == copies


def test_the_data_plane_is_idle_only_when_it_is_drained(single) -> None:
    streamer = single.streamer
    single.manager.prepare(1, 0, 128)
    streamer.assert_idle("a page is freed")
    streamer.set_plan(streamer.plan_for([PlanRequest(1, 0, 0, 128)]))
    with pytest.raises(RuntimeError, match="a page is freed while the data plane of rank 0"):
        streamer.assert_idle("a page is freed")
    streamer.begin_iteration([1], [0], [128], single.spans([(0, 128)]))
    for layer in range(len(_RATIOS)):
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
    durations = [(layer + 1) / 1000.0 for layer in range(len(_RATIOS))]
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
            rank.manager.prepare(1, history, 128)
            streamer.set_plan(streamer.plan_for([PlanRequest(1, 0, history, 128)]))
            streamer.begin_iteration([1], [history], [128], rank.spans([(history, 128)]))
            for layer in range(len(_RATIOS)):
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
        assert len(events) == 4 * len(_RATIOS)
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
        rank.manager.prepare(1, 128, 128)
        streamer.set_plan(streamer.plan_for([PlanRequest(1, 0, 128, 128)]))
        streamer.begin_iteration([1], [128], [128], rank.spans([(128, 128)]))
        for layer in range(len(_RATIOS)):
            streamer.on_layer(layer)
        streamer.end_forward()
        streamer.drain(5.0)
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
    ("direction", "iteration", "layer", "rank", "messages"),
    [
        # The layer has a message for each of its deadline classes: one for a layer with sliding
        # window attention only (5), two for a layer compressed 128 times (2) and three for one
        # compressed 4 times (1, 6).
        (Direction.WRITEBACK, 0, 5, 1, 1),
        (Direction.FETCH, 1, 1, 1, 3),
        (Direction.FETCH, 1, 6, 0, 3),
        (Direction.WRITEBACK, 1, 2, 0, 2),
    ],
)
def test_a_damaged_message_is_named_by_the_checksums(
    kind: str, direction: Direction, iteration: int, layer: int, rank: int, messages: int
) -> None:
    scenario = Scenario(2, 2, {1: 400, 2: 520}, {1: 0, 2: 1})
    fault = DataPlaneFault(iteration, layer, direction, kind, rank)
    problems = _mismatches(run_group(scenario, Options(debug=True, fault=fault)))
    label = ("F" if direction is Direction.FETCH else "W") + f"({layer})/"
    # Each damaged message arrived different from what was sent, and so did the pages it was
    # unpacked into; nothing else differs.
    assert len(problems) == 2 * messages and all(label in problem for problem in problems), problems
    assert all(f"rank {rank} has " in problem for problem in problems), problems


def test_a_run_whose_fault_never_happens_has_no_mismatch() -> None:
    scenario = Scenario(2, 2, {1: 400, 2: 520}, {1: 0, 2: 1})
    fault = DataPlaneFault(99, 1, Direction.FETCH)
    assert _mismatches(run_group(scenario, Options(debug=True, fault=fault))) == []


def test_the_mismatch_finder_names_what_differs_and_what_is_missing() -> None:
    fetch = (int(Direction.FETCH), 3, 2, 0, 1)
    assert (
        find_data_plane_mismatches(
            [[(fetch, "sent", 7)], [(fetch, "received", 7), (fetch, "stored", 7)]]
        )
        == []
    )
    (problem,) = find_data_plane_mismatches(
        [[(fetch, "sent", 7)], [(fetch, "received", 7), (fetch, "stored", 9)]]
    )
    assert "rank 1 has stored 0x9, not 0x7" in problem and "F(3)/attention_kv" in problem
    assert "owner rank 0, compute rank 1" in problem
    (problem,) = find_data_plane_mismatches([[(fetch, "sent", 7)], []])
    assert "recorded ['sent'] instead of ['received', 'sent', 'stored']" in problem
    local = (int(Direction.WRITEBACK), 1, 0, 2, 2)
    assert find_data_plane_mismatches([[], [], [(local, "source", 5), (local, "stored", 5)]]) == []
    (problem,) = find_data_plane_mismatches([[], [], [(local, "source", 5), (local, "stored", 6)]])
    assert "W(1)/state" in problem and "rank 2 has stored" in problem


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
    layout = StagingLayout(_geometry(2))
    stream = FakeStream("never used")
    try:
        with pytest.raises(ValueError, match="only noticed by the checksums"):
            DkvStreamer(
                FakeKvManager(layout, range(len(_RATIOS)), pages=2),
                SimpleNamespace(layout=layout, pool=StagingPool(layout, device="cpu")),
                FakeTransport(FakeNetwork(1), 0),
                compute_ownership(len(_RATIOS), 1),
                0,
                group_size=1,
                copier=FakeCopier(),
                data_stream=stream,
                fault=DataPlaneFault(0, 0, Direction.FETCH),
            )
    finally:
        stream.close()


# The cost model and the sizes the plan relies on


@pytest.mark.parametrize("ring_depth", [1, 2])
def test_the_cost_model_gives_the_layout_ranges(ring_depth: int) -> None:
    layout = StagingLayout(_geometry(ring_depth))
    costs = LayoutCostModel(layout)
    for kind in layout.kinds:
        for history, chunk in [(0, 128), (128, 128), (300, 100), (1000, 24), (5, 1)]:
            cost = costs.page_cost(0, kind, history, chunk)
            assert cost.page_bytes == layout.kind_page_bytes(kind)
            assert cost.fetch_pages == layout.fetch_range(kind, history, chunk)[1]
            assert cost.writeback_pages == layout.writeback_range(kind, history, chunk)[1]
        assert costs.page_cost(0, kind, 100, 0).fetch_pages == 0


def test_a_message_never_exceeds_the_buffer_that_holds_it() -> None:
    layout = StagingLayout(_geometry(2))
    limit = max_message_bytes(layout)
    assert limit > 0
    for kind in layout.kinds:
        assert layout.slot_pages(kind) * layout.kind_page_bytes(kind) <= limit
