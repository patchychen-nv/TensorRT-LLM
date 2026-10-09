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
"""The data-plane plan of the DKV layer-split layout and the deadlock check of its operations.

The plan and the simulator import only the standard library. They are imported from the package
where it can be imported and loaded straight from their files where it cannot (a host without
torch), so this file runs in both places. The ownership tables come from ``compute_ownership`` in
the same way: imported where the package imports, and compiled from the source of the function in
``dkv.py`` where not, so the tables are always those of the real function.
"""

import ast
import dataclasses
import hashlib
import importlib
import itertools
import os
import random
import subprocess
import sys
import types
from collections import Counter
from collections.abc import Callable, Sequence
from pathlib import Path

import pytest

pytestmark = pytest.mark.cpu_only

_REPO = Path(__file__).resolve().parents[4]
_PYEXECUTOR = _REPO / "tensorrt_llm" / "_torch" / "pyexecutor"


def _import_plan_modules() -> tuple[types.ModuleType, types.ModuleType]:
    try:
        from tensorrt_llm._torch.pyexecutor import dkv_plan, dkv_plan_simulator
    except ImportError:
        package = sys.modules.get("_dkv_plan_pkg")
        if package is None:
            package = types.ModuleType("_dkv_plan_pkg")
            package.__path__ = [str(_PYEXECUTOR)]
            sys.modules["_dkv_plan_pkg"] = package
        dkv_plan = importlib.import_module("_dkv_plan_pkg.dkv_plan")
        dkv_plan_simulator = importlib.import_module("_dkv_plan_pkg.dkv_plan_simulator")
    return dkv_plan, dkv_plan_simulator


def _load_compute_ownership() -> Callable[[int, int], tuple[int, ...]]:
    try:
        from tensorrt_llm._torch.pyexecutor.dkv import compute_ownership
    except ImportError:
        path = _PYEXECUTOR / "dkv.py"
        tree = ast.parse(path.read_text(), filename=str(path))
        function = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "compute_ownership"
        )
        namespace: dict[str, object] = {}
        exec(compile(ast.Module([function], []), str(path), "exec"), namespace)
        compute_ownership = namespace["compute_ownership"]
    return compute_ownership


plan_lib, simulator_lib = _import_plan_modules()
compute_ownership = _load_compute_ownership()

DEADLINE_OF_KIND = plan_lib.DEADLINE_OF_KIND
DeadlineClass = plan_lib.DeadlineClass
Direction = plan_lib.Direction
LayerType = plan_lib.LayerType
OpAction = plan_lib.OpAction
PageCost = plan_lib.PageCost
PlanRequest = plan_lib.PlanRequest
RankOp = plan_lib.RankOp
StagingKind = plan_lib.StagingKind
build_dkv_plan = plan_lib.build_dkv_plan
global_step_order = plan_lib.global_step_order
kinds_of_layer = plan_lib.kinds_of_layer
layer_types_from_compress_ratios = plan_lib.layer_types_from_compress_ratios
plan_canonical_form = plan_lib.plan_canonical_form
plan_fingerprint = plan_lib.plan_fingerprint
step_label = plan_lib.step_label
ProblemKind = simulator_lib.ProblemKind
expand_plan = simulator_lib.expand_plan
simulate = simulator_lib.simulate

FETCH, WRITEBACK = Direction.FETCH, Direction.WRITEBACK
SWA_ONLY, CSA, HCA = LayerType.SWA_ONLY, LayerType.CSA, LayerType.HCA


# A cost model in the shape of the DeepSeek-V4 layout. The numbers are not the real ones, but the
# page counts follow the same rules: a kind stores one row per ``tokens_per_row`` tokens, a
# windowed kind caches only the last window of rows, a fetch covers the cached rows and a
# writeback the pages that the chunk's new rows touch.


@dataclasses.dataclass(frozen=True)
class _Geometry:
    tokens_per_row: int
    rows_per_page: int
    page_bytes: int
    window_rows: int | None


_GEOMETRY = {
    StagingKind.SWA: _Geometry(1, 128, 64 * 1024, 128),
    StagingKind.COMPRESS_R4: _Geometry(4, 32, 16 * 1024, None),
    StagingKind.COMPRESS_R128: _Geometry(128, 1, 2 * 1024, None),
    StagingKind.INDEXER_COMPRESS: _Geometry(4, 32, 4 * 1024, None),
    StagingKind.STATE_CSA: _Geometry(1, 8, 64 * 1024, 8),
    StagingKind.STATE_HCA: _Geometry(1, 128, 256 * 1024, 128),
    StagingKind.STATE_INDEXER: _Geometry(1, 8, 16 * 1024, 8),
}


def _pages_covering(first_row: int, end_row: int, rows_per_page: int) -> int:
    if end_row <= first_row:
        return 0
    return (end_row - 1) // rows_per_page - first_row // rows_per_page + 1


class _V4Costs:
    """Page sizes and counts in the shape of the V4 layout.

    ``scale_by_layer`` makes the page size differ from layer to layer, so that a plan that takes a
    size from the wrong layer is noticed.
    """

    def __init__(self, *, scale_by_layer: bool = False) -> None:
        self.scale_by_layer = scale_by_layer

    def page_cost(self, layer: int, kind: StagingKind, history: int, chunk: int) -> PageCost:
        geometry = _GEOMETRY[kind]
        cached_rows = history // geometry.tokens_per_row
        first_cached = (
            0 if geometry.window_rows is None else max(0, cached_rows - geometry.window_rows)
        )
        fetch = _pages_covering(first_cached, cached_rows, geometry.rows_per_page)
        written_rows = (history + chunk) // geometry.tokens_per_row
        writeback = _pages_covering(cached_rows, written_rows, geometry.rows_per_page)
        scale = 1 + layer % 3 if self.scale_by_layer else 1
        return PageCost(geometry.page_bytes * scale, fetch, writeback)


def _expected_bytes(
    batch: Sequence, layer_types: Sequence, costs: _V4Costs, direction: Direction
) -> list[int]:
    """The bytes of every layer that the cost model says the real requests move, without the
    plan."""
    per_layer = [0] * len(layer_types)
    for request in batch:
        if request.is_dummy:
            continue
        for layer, layer_type in enumerate(layer_types):
            for kind in kinds_of_layer(layer_type):
                cost = costs.page_cost(
                    layer, kind, request.context_current_position, request.context_chunk_size
                )
                pages = cost.fetch_pages if direction is FETCH else cost.writeback_pages
                per_layer[layer] += pages * cost.page_bytes
    return per_layer


# The models, batches and plans the tests use.


def _alternating(num_layers: int) -> list[int]:
    """The compression ratios of DeepSeek-V4-Pro: two layers without compression, then compressed
    sparse (ratio 4) and heavily compressed (ratio 128) layers alternating."""
    return [1, 1] + [4 if layer % 2 == 0 else 128 for layer in range(num_layers - 2)]


_MODELS = {
    "v4_43_layers": layer_types_from_compress_ratios(_alternating(43)),
    "v4_61_layers": layer_types_from_compress_ratios(_alternating(61)),
}
_V4_43 = _MODELS["v4_43_layers"]
_PLACEMENTS = [
    "single_rank",
    "every_rank",
    "mixed_with_dummy_only_ranks",
    "last_rank",
    "first_chunks",
    "dummy_only",
]


def _batch(placement: str, group_size: int) -> list:
    """A global batch in which every rank without a real request holds a dummy, as the executor
    arranges it."""
    batch = []
    ids = itertools.count(1)
    served: set[int] = set()

    def real(rank: int, history: int, chunk: int) -> None:
        batch.append(PlanRequest(next(ids), rank, history, chunk))
        served.add(rank)

    def dummy(rank: int) -> None:
        # A dummy that claims history and a chunk: the plan has to ignore it all the same.
        batch.append(PlanRequest(10_000 + rank, rank, 999, 1, is_dummy=True))
        served.add(rank)

    if placement == "single_rank":
        for history, chunk in ((0, 4096), (777, 513), (60_000, 8192)):
            real(0, history, chunk)
    elif placement == "every_rank":
        for rank in range(group_size):
            real(rank, 1000 * (rank + 1) + 3, 513)
            real(rank, 0, 2048)
    elif placement == "mixed_with_dummy_only_ranks":
        for rank in range(0, group_size, 2):
            real(rank, 500 * (rank + 1) + 7, 300)
            dummy(rank)
    elif placement == "last_rank":
        real(group_size - 1, 4097, 1024)
        real(group_size - 1, 129, 7)
    elif placement == "first_chunks":
        for rank in range(group_size):
            real(rank, 0, 1000 + rank)
    elif placement != "dummy_only":
        raise ValueError(placement)
    for rank in range(group_size):
        if rank not in served:
            dummy(rank)
    return batch


def _plan(batch, owners, layer_types, *, group_size, ring_depth=2, costs=None):
    return build_dkv_plan(
        batch,
        owners,
        layer_types,
        costs or _V4Costs(),
        group_size=group_size,
        ring_depth=ring_depth,
    )


def _check_plan_invariants(plan, batch: Sequence, costs: _V4Costs) -> None:
    """What every plan must satisfy, checked against the batch and the cost model."""
    by_id = {request.request_id: request for request in batch}
    real_ids: dict[int, set[int]] = {}
    for request in batch:
        if not request.is_dummy:
            real_ids.setdefault(request.compute_rank, set()).add(request.request_id)
    assert [(s.direction, s.layer, s.issue_point) for s in plan.steps] == list(
        global_step_order(plan.num_layers, plan.ring_depth)
    )
    for step in plan.steps:
        order = [(transfer.compute, transfer.deadline) for transfer in step.transfers]
        assert order == sorted(order) and len(set(order)) == len(order)
        for transfer in step.transfers:
            assert (transfer.direction, transfer.layer) == (step.direction, step.layer)
            assert transfer.owner == plan.owner_of_layer[transfer.layer]
            assert transfer.segments
            ids = {segment.request_id for segment in transfer.segments}
            assert ids <= real_ids.get(transfer.compute, set())
            keys = [(segment.request_id, segment.kind) for segment in transfer.segments]
            assert keys == sorted(keys)
            layer_kinds = set(kinds_of_layer(plan.layer_types[transfer.layer]))
            for segment in transfer.segments:
                assert segment.kind in layer_kinds
                assert DEADLINE_OF_KIND[segment.kind] is transfer.deadline
                request = by_id[segment.request_id]
                cost = costs.page_cost(
                    transfer.layer,
                    segment.kind,
                    request.context_current_position,
                    request.context_chunk_size,
                )
                pages = cost.fetch_pages if step.direction is FETCH else cost.writeback_pages
                assert pages > 0
                assert (segment.pages, segment.nbytes) == (pages, pages * cost.page_bytes)


# Layer types and the global step order


def test_layer_types_follow_the_compression_ratios() -> None:
    assert layer_types_from_compress_ratios([0, 1, 4, 128, 64]) == (
        SWA_ONLY,
        SWA_ONLY,
        CSA,
        HCA,
        HCA,
    )
    assert Counter(_V4_43) == {SWA_ONLY: 2, CSA: 21, HCA: 20}
    assert _V4_43[:4] == (SWA_ONLY, SWA_ONLY, CSA, HCA)
    assert Counter(_MODELS["v4_61_layers"]) == {SWA_ONLY: 2, CSA: 30, HCA: 29}


def test_kinds_and_deadline_classes_of_the_layer_types() -> None:
    assert set(DEADLINE_OF_KIND) == set(StagingKind)
    assert plan_lib.deadline_classes_of_layer(SWA_ONLY) == (DeadlineClass.ATTENTION_KV,)
    assert plan_lib.deadline_classes_of_layer(HCA) == (
        DeadlineClass.STATE,
        DeadlineClass.ATTENTION_KV,
    )
    assert plan_lib.deadline_classes_of_layer(CSA) == (
        DeadlineClass.STATE,
        DeadlineClass.INDEXER,
        DeadlineClass.ATTENTION_KV,
    )
    for layer_type in LayerType:
        kinds = kinds_of_layer(layer_type)
        assert list(kinds) == sorted(set(kinds))
        assert StagingKind.SWA in kinds
        # A layer holds the state of its own compressor and indexer, and of no other.
        assert (StagingKind.STATE_CSA in kinds) == (layer_type is CSA)
        assert (StagingKind.STATE_HCA in kinds) == (layer_type is HCA)
        assert (StagingKind.STATE_INDEXER in kinds) == (layer_type is CSA)
        assert (StagingKind.INDEXER_COMPRESS in kinds) == (layer_type is CSA)


@pytest.mark.parametrize(
    ("num_layers", "ring_depth", "expected"),
    [
        (4, 1, "F0 W0 F1 W1 F2 W2 F3 W3"),
        (4, 2, "F0 F1 W0 F2 W1 F3 W2 W3"),
        (4, 3, "F0 F1 F2 W0 F3 W1 W2 W3"),
        (2, 4, "F0 F1 W0 W1"),
        (1, 1, "F0 W0"),
        (1, 3, "F0 W0"),
    ],
)
def test_global_step_order_is_the_documented_one(
    num_layers: int, ring_depth: int, expected: str
) -> None:
    order = global_step_order(num_layers, ring_depth)
    assert " ".join(f"{'F' if s.direction is FETCH else 'W'}{s.layer}" for s in order) == expected


def test_hooks_enqueue_the_steps_at_the_documented_points() -> None:
    by_hook: dict[int, list[str]] = {}
    for step in global_step_order(5, 3):
        by_hook.setdefault(step.issue_point, []).append(step_label(step.direction, step.layer))
    assert by_hook == {
        -1: ["F(0)", "F(1)"],  # before the first layer
        0: ["F(2)"],
        1: ["W(0)", "F(3)"],
        2: ["W(1)", "F(4)"],
        3: ["W(2)"],
        4: ["W(3)"],
        5: ["W(4)"],  # after the last layer
    }


@pytest.mark.parametrize("ring_depth", [1, 2, 3, 4])
@pytest.mark.parametrize("num_layers", [1, 2, 3, 7, 43])
def test_every_step_is_issued_once_and_after_the_events_it_waits_for(
    num_layers: int, ring_depth: int
) -> None:
    order = global_step_order(num_layers, ring_depth)
    assert sorted((s.direction, s.layer) for s in order) == sorted(
        itertools.product(Direction, range(num_layers))
    )
    # The hooks run in increasing order, so the global order must never return to an earlier hook.
    assert [s.issue_point for s in order] == sorted(s.issue_point for s in order)
    for step in order:
        if step.direction is WRITEBACK:
            # It waits for the attention of its layer, which records its event during that layer,
            # so the hook at the top of the next layer is the first that can enqueue it.
            assert step.issue_point == step.layer + 1
        else:
            reused = step.layer - ring_depth  # the layer that used the staging slot before
            if reused >= 0:
                # It waits for the end of that layer, which is recorded at the top of the layer
                # after it.
                assert step.issue_point == reused + 1
            else:
                assert step.issue_point <= 0  # nothing to wait for
    for hook in sorted({s.issue_point for s in order}):
        directions = [s.direction for s in order if s.issue_point == hook]
        # The writeback of the layer that just ended goes before the prefetch.
        assert directions in (
            [FETCH] * len(directions) if hook < 0 else [WRITEBACK, FETCH],
            [FETCH],
            [WRITEBACK],
        )


@pytest.mark.parametrize(("num_layers", "ring_depth"), [(0, 2), (4, 0), (-1, 1), (3, -2)])
def test_a_step_order_needs_layers_and_a_ring(num_layers: int, ring_depth: int) -> None:
    with pytest.raises(ValueError, match="must be positive"):
        global_step_order(num_layers, ring_depth)


# The plan as a pure function


def test_a_plan_has_the_messages_the_cost_model_implies() -> None:
    """Sizes computed by hand for one request with 300 history tokens and a chunk of 100 tokens on
    rank 1, whose layers are all owned by rank 0."""
    batch = [PlanRequest(7, 1, 300, 100), PlanRequest(8, 0, 999, 5, is_dummy=True)]
    plan = _plan(batch, (0, 0, 0), (SWA_ONLY, CSA, HCA), group_size=2)

    def table(direction: Direction, layer: int) -> list:
        return [
            (
                t.deadline.name,
                t.owner,
                t.compute,
                [(s.kind.name, s.pages, s.nbytes) for s in t.segments],
                t.pages,
                t.nbytes,
            )
            for t in plan.transfers(direction, layer)
        ]

    assert table(FETCH, 0) == [("ATTENTION_KV", 0, 1, [("SWA", 2, 131072)], 2, 131072)]
    assert table(WRITEBACK, 0) == [("ATTENTION_KV", 0, 1, [("SWA", 2, 131072)], 2, 131072)]
    indexer_fetch = [("INDEXER_COMPRESS", 3, 12288), ("STATE_INDEXER", 2, 32768)]
    assert table(FETCH, 1) == [
        ("STATE", 0, 1, [("STATE_CSA", 2, 131072)], 2, 131072),
        ("INDEXER", 0, 1, indexer_fetch, 5, 45056),
        ("ATTENTION_KV", 0, 1, [("SWA", 2, 131072), ("COMPRESS_R4", 3, 49152)], 5, 180224),
    ]
    indexer_writeback = [("INDEXER_COMPRESS", 2, 8192), ("STATE_INDEXER", 13, 212992)]
    assert table(WRITEBACK, 1) == [
        ("STATE", 0, 1, [("STATE_CSA", 13, 851968)], 13, 851968),
        ("INDEXER", 0, 1, indexer_writeback, 15, 221184),
        ("ATTENTION_KV", 0, 1, [("SWA", 2, 131072), ("COMPRESS_R4", 2, 32768)], 4, 163840),
    ]
    assert table(FETCH, 2) == [
        ("STATE", 0, 1, [("STATE_HCA", 2, 524288)], 2, 524288),
        ("ATTENTION_KV", 0, 1, [("SWA", 2, 131072), ("COMPRESS_R128", 2, 4096)], 4, 135168),
    ]
    assert plan.transfers(WRITEBACK, 2, DeadlineClass.STATE)[0].nbytes == 524288
    assert plan.transfers(WRITEBACK, 2, DeadlineClass.INDEXER) == ()


def test_the_same_input_gives_equal_plans() -> None:
    batch = _batch("every_rank", 4)
    owners = compute_ownership(43, 4)
    first = _plan(batch, owners, _V4_43, group_size=4)
    second = _plan(list(batch), tuple(owners), _V4_43, group_size=4)
    assert first == second
    assert hash(first) == hash(second)
    assert plan_fingerprint(first) == plan_fingerprint(second)


def test_the_plan_does_not_depend_on_the_order_of_the_requests() -> None:
    owners = (0, 1, 2, 2, 1)
    layer_types = (CSA, HCA, SWA_ONLY, CSA, HCA)
    batch = [
        PlanRequest(5, 2, 1200, 300),
        PlanRequest(2, 0, 0, 512),
        PlanRequest(9, 1, 4097, 8),
        PlanRequest(3, 1, 130, 1000),
        PlanRequest(7, 0, 33, 2, is_dummy=True),
    ]
    reference = _plan(batch, owners, layer_types, group_size=3)
    for permutation in itertools.permutations(batch):
        plan = _plan(permutation, owners, layer_types, group_size=3)
        assert plan == reference
        assert plan_fingerprint(plan) == plan_fingerprint(reference)


def test_a_large_plan_does_not_depend_on_the_order_of_the_requests_either() -> None:
    owners = compute_ownership(43, 8)
    batch = _batch("every_rank", 8) + [PlanRequest(30_000, 3, 70, 70)]
    reference = _plan(batch, owners, _V4_43, group_size=8, ring_depth=3)
    shuffler = random.Random(2026)
    for _ in range(4):
        shuffled = list(batch)
        shuffler.shuffle(shuffled)
        assert _plan(shuffled, owners, _V4_43, group_size=8, ring_depth=3) == reference


def test_requests_are_ordered_by_compute_rank_then_id_and_so_are_the_segments() -> None:
    batch = [
        PlanRequest(9, 1, 400, 10),
        PlanRequest(3, 1, 400, 10),
        PlanRequest(4, 0, 400, 10),
        PlanRequest(7, 1, 400, 10),
        PlanRequest(1, 2, 400, 10, is_dummy=True),
    ]
    plan = _plan(batch, (2, 2), (SWA_ONLY, SWA_ONLY), group_size=3)
    assert [(r.compute_rank, r.request_id) for r in plan.requests] == [
        (0, 4),
        (1, 3),
        (1, 7),
        (1, 9),
        (2, 1),
    ]
    transfers = plan.transfers(FETCH, 0)
    # Rank 0 comes before rank 1 whatever the ids, and a message lists its requests by id.
    assert [t.compute for t in transfers] == [0, 1]
    assert [s.request_id for s in transfers[1].segments] == [3, 7, 9]


def test_a_batch_may_be_any_iterable() -> None:
    batch = [PlanRequest(1, 0, 10, 5), PlanRequest(2, 1, 20, 5)]
    from_list = _plan(batch, (0,), (SWA_ONLY,), group_size=2, ring_depth=1)
    assert from_list == _plan(iter(batch), (0,), (SWA_ONLY,), group_size=2, ring_depth=1)
    assert from_list == _plan(tuple(reversed(batch)), [0], [SWA_ONLY], group_size=2, ring_depth=1)


@pytest.mark.parametrize("group_size", [2, 4])
def test_dummy_requests_add_nothing(group_size: int) -> None:
    owners = compute_ownership(43, group_size)
    real = _batch("every_rank", group_size)
    dummies = [
        PlanRequest(20_000 + 2 * rank + k, rank, history, chunk, is_dummy=True)
        for rank in range(group_size)
        for k, (history, chunk) in enumerate(((0, 1), (50_000, 7)))
    ]
    plain = _plan(real, owners, _V4_43, group_size=group_size)
    padded = _plan(real + dummies, owners, _V4_43, group_size=group_size)
    assert plain.steps == padded.steps
    assert padded.requests != plain.requests
    assert sum(r.is_dummy for r in padded.requests) == 2 * group_size


def test_dummy_requests_are_never_costed() -> None:
    asked: list[tuple[int, int]] = []

    class Spy(_V4Costs):
        def page_cost(self, layer, kind, history, chunk):
            asked.append((history, chunk))
            return super().page_cost(layer, kind, history, chunk)

    batch = [PlanRequest(1, 0, 100, 10), PlanRequest(2, 1, 54_321, 77, is_dummy=True)]
    build_dkv_plan(batch, (1, 1), (CSA, HCA), Spy(), group_size=2, ring_depth=2)
    assert asked and set(asked) == {(100, 10)}


def test_a_batch_of_only_dummies_has_steps_but_no_transfer() -> None:
    plan = _plan(_batch("dummy_only", 4), compute_ownership(43, 4), _V4_43, group_size=4)
    assert len(plan.steps) == 2 * 43
    assert not any(step.transfers for step in plan.steps)
    assert all(plan.rank_program(rank) == () for rank in range(4))


def test_nothing_that_moves_no_byte_is_a_message() -> None:
    owners = compute_ownership(43, 4)
    first_chunks = _plan(_batch("first_chunks", 4), owners, _V4_43, group_size=4)
    # Without history there is nothing to fetch, but the new KV still goes back.
    assert not any(s.transfers for s in first_chunks.steps if s.direction is FETCH)
    assert all(s.transfers for s in first_chunks.steps if s.direction is WRITEBACK)
    no_chunk = _plan([PlanRequest(1, 0, 5000, 0)], owners, _V4_43, group_size=4)
    assert not any(s.transfers for s in no_chunk.steps if s.direction is WRITEBACK)
    assert all(s.transfers for s in no_chunk.steps if s.direction is FETCH)


@pytest.mark.parametrize("scale_by_layer", [False, True])
@pytest.mark.parametrize("placement", _PLACEMENTS)
def test_byte_totals_match_the_cost_model(placement: str, scale_by_layer: bool) -> None:
    costs = _V4Costs(scale_by_layer=scale_by_layer)
    batch = _batch(placement, 4)
    plan = _plan(batch, compute_ownership(43, 4), _V4_43, group_size=4, costs=costs)
    _check_plan_invariants(plan, batch, costs)
    for direction in Direction:
        expected = _expected_bytes(batch, _V4_43, costs, direction)
        moved = [
            sum(t.nbytes for t in plan.transfers(direction, layer)) for layer in range(len(_V4_43))
        ]
        assert moved == expected
        assert sum(moved) == sum(
            t.nbytes for s in plan.steps if s.direction is direction for t in s.transfers
        )


def test_the_owner_computing_its_own_requests_is_a_local_copy_not_a_message() -> None:
    group_size = 4
    owners = compute_ownership(43, group_size)
    batch = [PlanRequest(1, 2, 5000, 700), PlanRequest(2, 2, 90, 33)] + [
        PlanRequest(100 + rank, rank, 0, 1, is_dummy=True) for rank in (0, 1, 3)
    ]
    plan = _plan(batch, owners, _V4_43, group_size=group_size)
    for step in plan.steps:
        for transfer in step.transfers:
            assert transfer.compute == 2
            assert transfer.is_local == (owners[transfer.layer] == 2)
    for rank in range(group_size):
        for op in plan.rank_program(rank):
            owner = owners[op.transfer.layer]
            if op.action is OpAction.LOCAL:
                assert rank == op.peer == owner == 2
            elif op.transfer.direction is FETCH:
                # The owner sends and the computing rank receives.
                assert (op.action, rank, op.peer) == (
                    (OpAction.SEND, owner, 2) if rank == owner else (OpAction.RECV, 2, owner)
                )
            else:
                assert (op.action, rank, op.peer) == (
                    (OpAction.RECV, owner, 2) if rank == owner else (OpAction.SEND, 2, owner)
                )
    owned_by_two = {layer for layer, owner in enumerate(owners) if owner == 2}
    for direction in Direction:
        local = {
            op.transfer.layer
            for op in plan.rank_program(2)
            if op.action is OpAction.LOCAL and op.transfer.direction is direction
        }
        assert local == owned_by_two
    # A rank that owns layers and computes none of the requests only serves the others.
    assert OpAction.LOCAL not in {op.action for op in plan.rank_program(0)}
    assert {op.action for op in plan.rank_program(0)} == {OpAction.SEND, OpAction.RECV}


def test_a_group_of_one_rank_needs_no_message() -> None:
    batch = [PlanRequest(1, 0, 4000, 300), PlanRequest(2, 0, 0, 20)]
    plan = _plan(batch, (0,) * 43, _V4_43, group_size=1)
    assert plan.steps and all(t.is_local for s in plan.steps for t in s.transfers)
    assert {op.action for op in plan.rank_program(0)} == {OpAction.LOCAL}
    assert simulate(plan).ok


@pytest.mark.parametrize("placement", _PLACEMENTS)
@pytest.mark.parametrize("ring_depth", [1, 2, 3])
def test_the_rank_programs_split_the_plan_into_matching_pairs(
    placement: str, ring_depth: int
) -> None:
    plan = _plan(
        _batch(placement, 4), compute_ownership(43, 4), _V4_43, group_size=4, ring_depth=ring_depth
    )
    expected: Counter = Counter()
    for step in plan.steps:
        for transfer in step.transfers:
            if transfer.is_local:
                expected[(transfer, OpAction.LOCAL)] += 1
            else:
                expected[(transfer, OpAction.SEND)] += 1
                expected[(transfer, OpAction.RECV)] += 1
        # In one step a rank only sends or only receives: the owner sends in a fetch and receives
        # in a writeback, every other rank does the opposite.
        owner = plan.owner_of_layer[step.layer]
        for rank in range(4):
            roles = {op.action for op in step.ops_for(rank)} - {OpAction.LOCAL}
            if step.direction is FETCH:
                allowed = {OpAction.SEND} if rank == owner else {OpAction.RECV}
            else:
                allowed = {OpAction.RECV} if rank == owner else {OpAction.SEND}
            assert roles <= allowed
    actual: Counter = Counter()
    positions = {(s.direction, s.layer): i for i, s in enumerate(plan.steps)}
    for rank in range(4):
        program = plan.rank_program(rank)
        for op in program:
            assert op.rank == rank
            actual[(op.transfer, op.action)] += 1
        # Every rank walks the steps in the global order.
        steps = [positions[(op.transfer.direction, op.transfer.layer)] for op in program]
        assert steps == sorted(steps)
    assert actual == expected


def test_a_dummy_only_rank_still_serves_the_layers_it_owns() -> None:
    owners = compute_ownership(43, 4)
    plan = _plan(_batch("mixed_with_dummy_only_ranks", 4), owners, _V4_43, group_size=4)
    served = {op.transfer.layer for op in plan.rank_program(1)}
    assert served and served <= {layer for layer, owner in enumerate(owners) if owner == 1}
    assert {op.action for op in plan.rank_program(1)} == {OpAction.SEND, OpAction.RECV}


def test_the_hook_accessors_partition_the_steps() -> None:
    plan = _plan(
        _batch("every_rank", 2), compute_ownership(6, 2), (CSA,) * 6, group_size=2, ring_depth=3
    )
    pieces = [plan.begin_steps(), *(plan.layer_steps(layer) for layer in range(6))]
    pieces.append(plan.end_steps())
    assert [step for piece in pieces for step in piece] == list(plan.steps)
    assert [(s.direction, s.layer) for s in plan.begin_steps()] == [(FETCH, 0), (FETCH, 1)]
    assert [(s.direction, s.layer) for s in plan.layer_steps(2)] == [(WRITEBACK, 1), (FETCH, 4)]
    assert [(s.direction, s.layer) for s in plan.end_steps()] == [(WRITEBACK, 5)]
    with pytest.raises(ValueError, match="not in a plan of 6 layers"):
        plan.step(FETCH, 6)
    with pytest.raises(ValueError, match="not in a group of 2 ranks"):
        plan.rank_program(2)


# Invalid input


def _build(**change):
    arguments = {
        "requests": [PlanRequest(1, 0, 100, 10), PlanRequest(2, 1, 50, 10)],
        "owner_of_layer": (0, 1),
        "layer_types": (CSA, HCA),
        "costs": _V4Costs(),
        "group_size": 2,
        "ring_depth": 2,
    } | change
    requests = arguments.pop("requests")
    owners = arguments.pop("owner_of_layer")
    layer_types = arguments.pop("layer_types")
    costs = arguments.pop("costs")
    return build_dkv_plan(requests, owners, layer_types, costs, **arguments)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"group_size": 0}, "must be positive"),
        ({"ring_depth": 0}, "must be positive"),
        ({"owner_of_layer": ()}, "no layer"),
        ({"layer_types": (CSA,)}, "has 2 layers, the layer types 1"),
        ({"owner_of_layer": (0, 2)}, r"ranks \[2\] own layers but the group has 2 ranks"),
        ({"owner_of_layer": (0, -1)}, r"ranks \[-1\] own layers"),
        ({"requests": [PlanRequest(1, 2, 0, 1)]}, "request 1 is computed on rank 2"),
        ({"requests": [PlanRequest(1, 0, -1, 1)]}, "request 1 has history -1 and chunk 1"),
        ({"requests": [PlanRequest(1, 0, 1, -5)]}, "request 1 has history 1 and chunk -5"),
        (
            {"requests": [PlanRequest(4, 0, 1, 1), PlanRequest(4, 1, 1, 1)]},
            "request id 4 appears more than once",
        ),
    ],
)
def test_an_inconsistent_batch_or_geometry_is_rejected(change: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _build(**change)


@pytest.mark.parametrize(
    "cost",
    [PageCost(0, 1, 1), PageCost(-4096, 1, 1), PageCost(4096, -1, 1), PageCost(4096, 1, -1)],
)
def test_a_cost_model_with_a_bad_page_size_or_count_is_rejected(cost) -> None:
    class Bad:
        def page_cost(self, layer, kind, history, chunk):
            return cost

    with pytest.raises(ValueError, match="the cost model returned"):
        _build(costs=Bad())


# The fingerprint


def _fingerprint_of(**change) -> str:
    base = {
        "requests": [
            PlanRequest(1, 0, 4000, 300),
            PlanRequest(2, 1, 100, 60),
            PlanRequest(3, 1, 0, 9, is_dummy=True),
        ],
        "owner_of_layer": (0, 1, 1, 0),
        "layer_types": (CSA, HCA, SWA_ONLY, CSA),
    }
    return plan_fingerprint(_build(**(base | change)))


@pytest.mark.parametrize(
    "change",
    [
        {"ring_depth": 3},
        {"owner_of_layer": (0, 1, 0, 0)},
        {"layer_types": (CSA, HCA, SWA_ONLY, HCA)},
        {"costs": _V4Costs(scale_by_layer=True)},
        {"requests": [PlanRequest(1, 0, 4001, 300), PlanRequest(2, 1, 100, 60)]},
        {"requests": [PlanRequest(1, 0, 4000, 301), PlanRequest(2, 1, 100, 60)]},
        {"requests": [PlanRequest(1, 1, 4000, 300), PlanRequest(2, 1, 100, 60)]},
        {"requests": [PlanRequest(5, 0, 4000, 300), PlanRequest(2, 1, 100, 60)]},
        # The dummies are part of the batch that every rank has to agree on.
        {"requests": [PlanRequest(1, 0, 4000, 300), PlanRequest(2, 1, 100, 60)]},
        {"requests": [PlanRequest(1, 0, 4000, 300), PlanRequest(2, 1, 100, 60, is_dummy=True)]},
    ],
)
def test_the_fingerprint_changes_with_every_input(change: dict) -> None:
    assert _fingerprint_of(**change) != _fingerprint_of()


def test_the_fingerprint_is_a_digest_of_plain_data() -> None:
    """Strings, integers and tuples print the same in every process, unlike the hash of a string
    or an object, so ranks that build equal plans get the same fingerprint."""
    plan = _plan(_batch("every_rank", 4), compute_ownership(43, 4), _V4_43, group_size=4)
    form = plan_canonical_form(plan)

    def plain(value: object) -> bool:
        if isinstance(value, tuple):
            return all(plain(item) for item in value)
        return type(value) in (int, str, bool)

    assert plain(form)
    assert plan_fingerprint(plan) == hashlib.sha256(repr(form).encode()).hexdigest()


_FINGERPRINT_PROGRAM = """
import importlib
import random
import sys
import types

package = types.ModuleType("_dkv_plan_pkg")
package.__path__ = [{path!r}]
sys.modules["_dkv_plan_pkg"] = package
plan_lib = importlib.import_module("_dkv_plan_pkg.dkv_plan")


class Costs:
    def page_cost(self, layer, kind, history, chunk):
        return plan_lib.PageCost(512 * (1 + int(kind)), history // 100 % 5, chunk // 50 % 4)


requests = [plan_lib.PlanRequest(i + 1, i % 4, 137 * i, 61 * i + 5, i % 5 == 4) for i in range(12)]
random.Random(int(sys.argv[1])).shuffle(requests)
layer_types = plan_lib.layer_types_from_compress_ratios([1, 1] + [4, 128] * 6)
owners = [layer * 4 // 14 for layer in range(14)]
plan = plan_lib.build_dkv_plan(
    requests, owners, layer_types, Costs(), group_size=4, ring_depth=3
)
print(plan_lib.plan_fingerprint(plan))
"""


def test_processes_with_other_hash_seeds_and_request_orders_build_the_same_plan() -> None:
    """The plan of ranks that are separate processes cannot depend on the hash seed of a process,
    which orders the sets and mappings of strings and enums, nor on the order of the requests."""
    program = _FINGERPRINT_PROGRAM.format(path=str(Path(plan_lib.__file__).parent))
    fingerprints = set()
    for hash_seed, shuffle_seed in ((0, 1), (1, 2), (2, 3), (4242, 4), (31337, 5)):
        done = subprocess.run(
            [sys.executable, "-c", program, str(shuffle_seed)],
            env={**os.environ, "PYTHONHASHSEED": str(hash_seed)},
            capture_output=True,
            text=True,
            check=True,
        )
        fingerprints.add(done.stdout.strip())
    assert len(fingerprints) == 1 and len(next(iter(fingerprints))) == 64


def test_a_plan_prints_briefly() -> None:
    """A failing assertion on plans must not print thousands of transfers."""
    plan = _plan(_batch("every_rank", 4), compute_ownership(43, 4), _V4_43, group_size=4)
    assert len(repr(plan)) < 200


# What the plan and the simulator may depend on


@pytest.mark.parametrize(
    ("module", "allowed"),
    [
        (
            plan_lib,
            {"collections", "dataclasses", "enum", "hashlib", "types", "typing", ".dkv_types"},
        ),
        (simulator_lib, {"collections", "dataclasses", "enum", ".dkv_plan"}),
        (sys.modules[StagingKind.__module__], {"enum", "typing"}),
    ],
    ids=["dkv_plan", "dkv_plan_simulator", "dkv_types"],
)
def test_the_modules_import_nothing_but_the_standard_library(
    module: types.ModuleType, allowed: set
) -> None:
    """A plan that cannot import a clock, the environment or a device cannot read one."""
    tree = ast.parse(Path(module.__file__).read_text())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level:
            imported.add("." * node.level + (node.module or ""))
        elif isinstance(node, ast.ImportFrom):
            imported.add((node.module or "").split(".")[0])
    assert imported <= allowed
    assert all(name.startswith(".") or name in sys.stdlib_module_names for name in imported)


# The simulator on good plans


@pytest.mark.parametrize("placement", _PLACEMENTS)
@pytest.mark.parametrize("ring_depth", [1, 2, 3])
@pytest.mark.parametrize("model", list(_MODELS))
@pytest.mark.parametrize("group_size", [2, 3, 4, 8])
def test_plans_of_the_ownership_tables_of_compute_ownership_are_deadlock_free(
    group_size: int, model: str, ring_depth: int, placement: str
) -> None:
    layer_types = _MODELS[model]
    owners = compute_ownership(len(layer_types), group_size)
    batch = _batch(placement, group_size)
    costs = _V4Costs(scale_by_layer=True)
    plan = _plan(
        batch, owners, layer_types, group_size=group_size, ring_depth=ring_depth, costs=costs
    )
    _check_plan_invariants(plan, batch, costs)
    report = simulate(plan)
    assert report.ok, str(report)
    assert report.graph_checked
    assert report.num_messages == sum(not t.is_local for s in plan.steps for t in s.transfers)
    assert report.num_ops == sum(len(plan.rank_program(rank)) for rank in range(group_size))
    assert (report.num_ops == 0) == (placement == "dummy_only")
    # The messages of a step issued as one group that finishes as a whole cannot deadlock either.
    grouped = simulate(plan, grouped=True)
    assert grouped.ok, str(grouped)
    assert grouped.num_nodes >= report.num_nodes


@pytest.mark.parametrize("ring_depth", [1, 2, 3])
@pytest.mark.parametrize(("group_size", "num_layers"), [(2, 6), (3, 4)])
def test_every_ownership_table_of_a_small_model_is_deadlock_free(
    group_size: int, num_layers: int, ring_depth: int
) -> None:
    """The argument does not depend on the table, so it holds for all of them, contiguous or not."""
    layer_types = (CSA, SWA_ONLY, HCA, CSA, HCA, SWA_ONLY)[:num_layers]
    batches = [
        _batch(placement, group_size)
        for placement in ("every_rank", "mixed_with_dummy_only_ranks", "last_rank")
    ]
    for owners in itertools.product(range(group_size), repeat=num_layers):
        for batch in batches:
            plan = _plan(batch, owners, layer_types, group_size=group_size, ring_depth=ring_depth)
            for grouped in (False, True):
                report = simulate(plan, grouped=grouped)
                assert report.ok, f"owners {owners}, grouped {grouped}: {report}"


@pytest.mark.parametrize("ring_depth", [1, 2, 3])
def test_random_ownership_tables_and_batches_of_a_larger_model_are_deadlock_free(
    ring_depth: int,
) -> None:
    chooser = random.Random(20261005 + ring_depth)
    layer_types = _V4_43[:17]
    for group_size in (3, 5, 8):
        for _ in range(6):
            owners = [chooser.randrange(group_size) for _ in layer_types]
            batch = [
                PlanRequest(
                    index + 1,
                    chooser.randrange(group_size),
                    chooser.choice([0, 5, 500, 12_345]),
                    chooser.choice([1, 300, 4096]),
                    is_dummy=chooser.random() < 0.3,
                )
                for index in range(2 * group_size)
            ]
            plan = _plan(batch, owners, layer_types, group_size=group_size, ring_depth=ring_depth)
            for grouped in (False, True):
                report = simulate(plan, grouped=grouped)
                assert report.ok, f"owners {owners}, grouped {grouped}: {report}"


@pytest.mark.parametrize("grouped", [False, True])
@pytest.mark.parametrize("ring_depth", [1, 2, 3])
@pytest.mark.parametrize("group_size", [2, 4, 8])
@pytest.mark.parametrize("placement", ["every_rank", "mixed_with_dummy_only_ranks"])
def test_every_wait_of_a_plan_raises_the_potential_of_the_deadlock_argument(
    group_size: int, ring_depth: int, placement: str, grouped: bool
) -> None:
    """The graph has no cycle because a potential grows along every wait. Attention of layer l is
    4l, the MoE exchange of layer l 4l + 1, the MoE 4l + 2, the writeback of layer l 4l + 3 and the
    fetch of layer m, which is enqueued at the top of layer m - D + 1, is 4(m - D + 1) - 1/2. A
    wait between two operations of the same step is the order of one data stream, in which the
    messages of a step are sorted by compute rank and deadline class; a group of messages has the
    potential of its step and waits only for its members."""
    layer_types = _MODELS["v4_43_layers"]
    plan = _plan(
        _batch(placement, group_size),
        compute_ownership(43, group_size),
        layer_types,
        group_size=group_size,
        ring_depth=ring_depth,
    )
    simulation = simulator_lib._Simulation(plan, expand_plan(plan), grouped=grouped)
    assert not simulation.run().problems
    graph = simulation.graph
    order = global_step_order(plan.num_layers, plan.ring_depth)

    def step_potential(direction, layer: int) -> float:
        if direction is FETCH:
            return 4 * (layer - ring_depth + 1) - 0.5
        return 4 * layer + 3

    def potential(node: int) -> float:
        kind, first, second, _, _ = graph.keys[node]
        if kind == "attention":
            return 4 * second
        if kind == "moe":
            return 4 * second + 2
        if kind == "exchange":
            return 4 * first + 1
        if kind == "group":
            return step_potential(order[second].direction, order[second].layer)
        transfer = simulation.programs[first][second].transfer  # a message or a local copy
        return step_potential(transfer.direction, transfer.layer)

    groups = 0
    for node, prerequisites in enumerate(graph.prerequisites):
        groups += graph.keys[node][0] == "group"
        for prerequisite, reason in prerequisites.items():
            before, after = potential(prerequisite), potential(node)
            assert before < after or (
                before == after and reason[0] in ("stream", "group member")
            ), f"{graph.keys[prerequisite]} -> {graph.keys[node]} ({reason[0]})"
    # A step sends an owner's messages of several deadline classes and peers together.
    assert (groups > 0) == grouped


def test_a_report_counts_what_it_checked() -> None:
    plan = _plan(_batch("every_rank", 4), compute_ownership(43, 4), _V4_43, group_size=4)
    report = simulate(plan)
    assert report.ok and report.graph_checked
    local = sum(t.is_local for s in plan.steps for t in s.transfers)
    # Per layer an attention and a MoE node on every rank and one for the MoE exchange; then one
    # node per message and per local copy.
    assert report.num_nodes == 43 * (2 * 4 + 1) + report.num_messages + local
    assert report.num_edges > report.num_nodes
    assert str(report).startswith("deadlock free: ")


# Bad plans, made by editing the per-rank sequences of a good one


def _step_of(op) -> tuple:
    return op.transfer.direction, op.transfer.layer


def _blocks(program: Sequence) -> list[list]:
    """The operations of a program grouped by step."""
    blocks: list[list] = []
    for op in program:
        if blocks and _step_of(blocks[-1][0]) == _step_of(op):
            blocks[-1].append(op)
        else:
            blocks.append([op])
    return blocks


def _swap_steps(program: Sequence, first: tuple, second: tuple) -> list:
    blocks = _blocks(program)
    steps = [_step_of(block[0]) for block in blocks]
    i, j = steps.index(first), steps.index(second)
    blocks[i], blocks[j] = blocks[j], blocks[i]
    return [op for block in blocks for op in block]


def _kinds(report) -> set:
    return {problem.kind for problem in report.problems}


def _text(report, kind) -> str:
    return " ".join(problem.message for problem in report.of_kind(kind))


def test_swapping_a_writeback_and_the_next_fetch_on_every_rank_deadlocks_through_the_slot() -> None:
    """The fetch of layer 3 overwrites the staging slot of layer 1, which W(1) still has to send."""
    batch = [PlanRequest(1, 0, 300, 70), PlanRequest(2, 1, 500, 90)]
    plan = _plan(batch, (0, 1, 1, 0), (CSA,) * 4, group_size=2, ring_depth=2)
    programs = expand_plan(plan)
    assert simulate(plan, programs).ok
    swapped = [_swap_steps(program, (WRITEBACK, 1), (FETCH, 3)) for program in programs]
    report = simulate(plan, swapped)
    assert not report.ok
    assert {ProblemKind.CYCLE, ProblemKind.STEP_ORDER} <= _kinds(report)
    cycle = _text(report, ProblemKind.CYCLE)
    assert "deadlock" in cycle and "overwrites a staging slot" in cycle
    assert "W(1)" in cycle and "F(3)" in cycle and "rank 0" in cycle
    order = _text(report, ProblemKind.STEP_ORDER)
    assert "rank 0 issues F(3)" in order and "rank 1 issues F(3)" in order
    assert "puts W(1) first" in order


def test_swapping_two_steps_on_one_rank_only_is_a_head_to_head_deadlock() -> None:
    """Rank 0 sends W(1) and then receives F(3); rank 1 sends F(3) first: each waits for the other."""
    batch = [PlanRequest(1, 0, 300, 70), PlanRequest(2, 1, 0, 1, is_dummy=True)]
    plan = _plan(batch, (0, 1, 1, 1), (CSA,) * 4, group_size=2, ring_depth=2)
    programs = expand_plan(plan)
    assert simulate(plan, programs).ok
    programs[1] = _swap_steps(programs[1], (WRITEBACK, 1), (FETCH, 3))
    report = simulate(plan, programs)
    # Every message still finds its partner; only two steps are in another order on one rank.
    assert not {ProblemKind.UNPAIRED, ProblemKind.SIZE_MISMATCH} & _kinds(report)
    assert {ProblemKind.CYCLE, ProblemKind.STEP_ORDER} <= _kinds(report)
    cycle = _text(report, ProblemKind.CYCLE)
    assert "W(1)" in cycle and "F(3)" in cycle
    assert "from rank 0 to rank 1" in cycle and "from rank 1 to rank 0" in cycle


@pytest.mark.parametrize("silent", [1, 3])
def test_a_dummy_only_rank_that_skips_its_owner_duties_hangs_the_others(silent: int) -> None:
    batch = _batch("mixed_with_dummy_only_ranks", 4)  # ranks 1 and 3 hold dummies only
    plan = _plan(batch, compute_ownership(43, 4), _V4_43, group_size=4)
    assert simulate(plan).ok
    programs = expand_plan(plan)
    assert programs[silent], "a dummy-only rank owns layers, so it has duties"
    programs[silent] = []
    report = simulate(plan, programs)
    assert not report.ok
    assert ProblemKind.UNPAIRED in _kinds(report)
    text = _text(report, ProblemKind.UNPAIRED)
    assert f"rank {silent} posts only 0" in text
    assert "waits forever" in text and "blocks forever" in text
    # Programs that hang by themselves are not analysed further.
    assert not report.graph_checked


def test_a_rank_that_leaves_out_one_message_misaligns_and_hangs_its_peer() -> None:
    plan = _plan(_batch("every_rank", 4), compute_ownership(43, 4), _V4_43, group_size=4)
    programs = expand_plan(plan)
    victim = next(i for i, op in enumerate(programs[1]) if op.action is OpAction.SEND)
    dropped = programs[1].pop(victim)
    report = simulate(plan, programs)
    assert ProblemKind.UNPAIRED in _kinds(report)
    text = _text(report, ProblemKind.UNPAIRED)
    assert f"rank {dropped.peer} posts" in text and "rank 1 posts" in text
    # Everything that followed the missing message is now paired with its neighbour.
    assert _kinds(report) & {ProblemKind.SIZE_MISMATCH, ProblemKind.STEP_MISMATCH}


def test_a_receive_hoisted_above_the_peer_visible_writeback_breaks_the_pairing() -> None:
    """The owner of layer l - 1 is rank 1, the owner of layer l + D - 1 is rank 0, and both have
    requests. Rank 1 sees that its part of W(l - 1) is a local copy, enqueues the receive of
    F(l + D - 1) right after it and only then receives what rank 0 writes back. Rank 0 sends the
    writeback first, so every message of rank 0 meets the wrong receive."""
    batch = [PlanRequest(1, 0, 300, 70), PlanRequest(2, 1, 500, 90)]
    plan = _plan(batch, (0, 1, 1, 0), (CSA,) * 4, group_size=2, ring_depth=2)
    programs = expand_plan(plan)
    assert simulate(plan, programs).ok
    blocks = _blocks(programs[1])
    steps = [_step_of(block[0]) for block in blocks]
    writeback, fetch = steps.index((WRITEBACK, 1)), steps.index((FETCH, 3))
    assert fetch == writeback + 1
    local = [op for op in blocks[writeback] if op.action is OpAction.LOCAL]
    remote = [op for op in blocks[writeback] if op.action is not OpAction.LOCAL]
    assert local and remote
    hoisted: list = []
    for index, block in enumerate(blocks):
        if index == writeback:
            hoisted += local + blocks[fetch] + remote
        elif index != fetch:
            hoisted += block
    programs[1] = hoisted
    report = simulate(plan, programs)
    assert not report.ok
    sizes = _text(report, ProblemKind.SIZE_MISMATCH)
    assert "from rank 0 to rank 1" in sizes
    assert "W(1)" in sizes and "F(3)" in sizes
    assert "transport hangs or corrupts" in sizes
    assert ProblemKind.STEP_ORDER in _kinds(report)


def test_messages_of_equal_size_in_the_wrong_order_are_still_caught() -> None:
    """The transport would not notice since the sizes agree: the data lands in the wrong place."""
    batch = [PlanRequest(1, 0, 300, 70), PlanRequest(2, 1, 0, 1, is_dummy=True)]
    plan = _plan(batch, (1,) * 4, (HCA,) * 4, group_size=2, ring_depth=4)
    assert [t.nbytes for t in plan.transfers(FETCH, 1)] == [
        t.nbytes for t in plan.transfers(FETCH, 2)
    ]
    programs = expand_plan(plan)
    assert simulate(plan, programs).ok
    programs[1] = _swap_steps(programs[1], (FETCH, 1), (FETCH, 2))
    report = simulate(plan, programs)
    assert ProblemKind.SIZE_MISMATCH not in _kinds(report)
    text = _text(report, ProblemKind.STEP_MISMATCH)
    assert "silently delivered" in text and "F(1)" in text and "F(2)" in text


def test_a_message_that_is_a_little_larger_on_one_side_is_a_size_mismatch() -> None:
    batch = [PlanRequest(1, 0, 300, 70), PlanRequest(2, 1, 40, 5)]
    plan = _plan(batch, (0, 1), (CSA,) * 2, group_size=2)
    programs = expand_plan(plan)
    index = next(i for i, op in enumerate(programs[1]) if op.action is OpAction.RECV)
    op = programs[1][index]
    segment = op.transfer.segments[0]
    larger = dataclasses.replace(segment, pages=segment.pages + 1, nbytes=segment.nbytes + 16)
    grown = dataclasses.replace(op.transfer, segments=(larger, *op.transfer.segments[1:]))
    programs[1][index] = dataclasses.replace(op, transfer=grown)
    text = _text(simulate(plan, programs), ProblemKind.SIZE_MISMATCH)
    assert f"{op.nbytes} B sent, {op.nbytes + 16} B expected" in text
    assert f"rank {op.peer} op" in text and op.transfer.label in text


def test_a_sender_that_serves_its_peers_one_after_the_other_deadlocks_in_the_moe_exchange() -> None:
    """Rank 2 owns every layer and computes nothing. It serves rank 0 through all layers before it
    serves rank 1. Every message still finds its partner. But the fetch of layer 1 on rank 0 waits
    for the end of layer 0 there, the MoE of layer 0 is collective and needs the attention of
    rank 1, and that waits for a fetch which rank 2 posts only after everything it does for rank 0.
    Without the collective MoE, rank 0 would simply run ahead."""
    batch = [
        PlanRequest(1, 0, 300, 70),
        PlanRequest(2, 1, 300, 70),
        PlanRequest(3, 2, 0, 1, is_dummy=True),
    ]
    plan = _plan(batch, (2,) * 4, (HCA,) * 4, group_size=3, ring_depth=1)
    programs = expand_plan(plan)
    assert simulate(plan, programs).ok
    programs[2] = sorted(programs[2], key=lambda op: op.peer)  # a stable sort, peer by peer
    report = simulate(plan, programs)
    assert not {ProblemKind.UNPAIRED, ProblemKind.SIZE_MISMATCH} & _kinds(report)
    assert {ProblemKind.CYCLE, ProblemKind.STEP_ORDER} <= _kinds(report)
    cycle = _text(report, ProblemKind.CYCLE)
    assert "MoE exchange of layer" in cycle and "collective over all ranks" in cycle
    assert all(f"rank {rank}" in cycle for rank in range(3))


def test_a_fetch_ahead_of_the_fetch_of_the_layer_whose_slot_it_reuses_deadlocks() -> None:
    """With one slot, the fetch of layer 1 overwrites the slot of layer 0 and so waits for the end
    of layer 0, whose attention waits for the fetch of layer 0 behind it on the data stream. The
    request has no chunk, so nothing is written back and only the fetches are at stake. Both ends
    of the message wait for the end of layer 0: the receiver for the slot, and the sender because
    the fetch is enqueued at the top of layer 1."""
    batch = [PlanRequest(1, 1, 300, 0), PlanRequest(2, 0, 0, 1, is_dummy=True)]
    plan = _plan(batch, (0, 0, 0), (CSA,) * 3, group_size=2, ring_depth=1)
    programs = expand_plan(plan)
    assert simulate(plan, programs).ok
    swapped = [_swap_steps(program, (FETCH, 0), (FETCH, 1)) for program in programs]
    report = simulate(plan, swapped)
    assert {ProblemKind.CYCLE, ProblemKind.STEP_ORDER} <= _kinds(report)
    assert not {ProblemKind.UNPAIRED, ProblemKind.SIZE_MISMATCH} & _kinds(report)
    cycle = _text(report, ProblemKind.CYCLE)
    assert (
        "overwrites the staging slot of an earlier layer" in cycle
        or "behind the MoE of the layer before" in cycle
    )
    assert "F(0)" in cycle and "F(1)" in cycle


def test_a_writeback_enqueued_ahead_of_the_fetch_of_its_own_layer_deadlocks() -> None:
    """The writeback of layer 0 sends what the attention of layer 0 produced, and that attention
    waits for the fetch of layer 0 behind the writeback on the data stream."""
    batch = [PlanRequest(1, 0, 300, 70), PlanRequest(2, 1, 400, 70)]
    plan = _plan(batch, (0, 0, 0), (CSA,) * 3, group_size=2, ring_depth=1)
    programs = expand_plan(plan)
    assert simulate(plan, programs).ok
    swapped = [_swap_steps(program, (FETCH, 0), (WRITEBACK, 0)) for program in programs]
    report = simulate(plan, swapped)
    assert {ProblemKind.CYCLE, ProblemKind.STEP_ORDER} <= _kinds(report)
    cycle = _text(report, ProblemKind.CYCLE)
    assert "sends what the attention produced" in cycle
    assert "W(0)" in cycle and "F(0)" in cycle


class _FetchingLayers:
    """One page of 4 KiB per request that fetches in the given layers; nothing is written back."""

    def __init__(self, layers: set[int]) -> None:
        self.layers = layers

    def page_cost(self, layer: int, kind: StagingKind, history: int, chunk: int) -> PageCost:
        return PageCost(4096, int(layer in self.layers and history > 0), 0)


def test_a_layer_that_fetches_nothing_still_runs_after_the_layer_before() -> None:
    """Layer 1 fetches nothing, so only the order of the compute stream holds its attention back.
    The fetch of layer 2 waits for the end of layer 1, which waits for layer 0 and so for the
    fetch of layer 0 behind it on the data stream."""
    batch = [PlanRequest(1, 0, 100, 70), PlanRequest(2, 1, 100, 70)]
    plan = build_dkv_plan(
        batch,
        (0, 0, 1),
        (SWA_ONLY,) * 3,
        _FetchingLayers({0, 2}),
        group_size=2,
        ring_depth=1,
    )
    assert not plan.transfers(FETCH, 1)
    programs = expand_plan(plan)
    assert simulate(plan, programs).ok
    swapped = [_swap_steps(program, (FETCH, 0), (FETCH, 2)) for program in programs]
    report = simulate(plan, swapped)
    assert ProblemKind.CYCLE in _kinds(report)
    assert "computes layer 1 after the layer before" in _text(report, ProblemKind.CYCLE)


def test_programs_that_do_not_fit_the_plan_are_reported_and_not_analysed() -> None:
    batch = [PlanRequest(1, 0, 300, 70), PlanRequest(2, 1, 40, 5)]
    plan = _plan(batch, (0, 1), (CSA,) * 2, group_size=2)
    programs = expand_plan(plan)
    assert "2 ranks but 1 programs" in simulate(plan, programs[:1]).problems[0].message
    report = simulate(plan, [programs[1], programs[0]])
    assert _kinds(report) == {ProblemKind.MALFORMED}
    assert "is an operation of rank" in report.problems[0].message
    assert report.num_nodes == 0 and not report.graph_checked


def test_a_rank_that_messages_itself_is_reported() -> None:
    plan = _plan([PlanRequest(1, 0, 300, 70)], (0, 1), (CSA,) * 2, group_size=2)
    programs = expand_plan(plan)
    transfer = dataclasses.replace(programs[0][0].transfer, owner=0, compute=0)
    programs[0].append(RankOp(OpAction.SEND, transfer))
    report = simulate(plan, programs)
    assert "message from the rank to itself" in _text(report, ProblemKind.MALFORMED)


# The simulator against an operational model of the same rules


def _writes_slot(op) -> bool:
    """A receive or local gather of a fetch writes the staging slot of its layer."""
    return op.transfer.direction is FETCH and op.action is not OpAction.SEND


def _reads_slot(op) -> bool:
    """A send or local scatter of a writeback reads the staging slot of its layer."""
    return op.transfer.direction is WRITEBACK and op.action is not OpAction.RECV


def _completes(plan, programs: Sequence[Sequence]) -> bool:
    """Whether every operation and every layer finishes, by running the rules instead of building
    a graph: streams advance whenever their next operation is ready and a message finishes when its
    send and its receive are both at the head of their streams and ready."""
    group_size, num_layers, ring_depth = plan.group_size, plan.num_layers, plan.ring_depth
    partner: dict[tuple[int, int], tuple[int, int]] = {}
    sends: dict[tuple[int, int], list[int]] = {}
    receives: dict[tuple[int, int], list[int]] = {}
    for rank, program in enumerate(programs):
        for index, op in enumerate(program):
            if op.action is OpAction.SEND:
                sends.setdefault((rank, op.peer), []).append(index)
            elif op.action is OpAction.RECV:
                receives.setdefault((op.peer, rank), []).append(index)
    for (sender, receiver), send_ops in sends.items():
        for i, j in zip(send_ops, receives.get((sender, receiver), [])):
            partner[(sender, i)] = (receiver, j)
            partner[(receiver, j)] = (sender, i)

    finished = [0] * group_size  # the operations of a data stream finish in order
    attention: list[set[int]] = [set() for _ in range(group_size)]
    moe: list[set[int]] = [set() for _ in range(group_size)]
    # The hook that enqueues the steps; the data stream waits for the layers before the hook.
    issue_point = {
        (step.direction, step.layer): step.issue_point
        for step in global_step_order(num_layers, ring_depth)
    }

    def done(rank: int, index: int) -> bool:
        return index < finished[rank]

    def ready(rank: int, index: int) -> bool:
        op = programs[rank][index]
        transfer = op.transfer
        if index != finished[rank]:
            return False
        issued = issue_point[(transfer.direction, transfer.layer)]
        if issued >= 1 and issued - 1 not in moe[rank]:
            return False
        if _writes_slot(op):
            earlier = transfer.layer - ring_depth
            if earlier >= 0 and earlier not in moe[rank]:
                return False
            for other_index, other in enumerate(programs[rank]):
                if _reads_slot(other) and other.transfer.layer == earlier:
                    if not done(rank, other_index):
                        return False
        if _reads_slot(op) and transfer.layer not in attention[rank]:
            return False
        return True

    progress = True
    while progress:
        progress = False
        for rank in range(group_size):
            for layer in range(num_layers):
                if layer not in attention[rank] and (layer == 0 or layer - 1 in moe[rank]):
                    fetched = all(
                        done(rank, index)
                        for index, op in enumerate(programs[rank])
                        if _writes_slot(op) and op.transfer.layer == layer
                    )
                    if fetched:
                        attention[rank].add(layer)
                        progress = True
                if layer not in moe[rank] and all(layer in attention[r] for r in range(group_size)):
                    moe[rank].add(layer)
                    progress = True
            if finished[rank] == len(programs[rank]):
                continue
            index = finished[rank]
            op = programs[rank][index]
            if op.action is OpAction.LOCAL:
                if ready(rank, index):
                    finished[rank] += 1
                    progress = True
            elif (rank, index) in partner:
                other, other_index = partner[(rank, index)]
                if (
                    ready(rank, index)
                    and other_index == finished[other]
                    and ready(other, other_index)
                ):
                    finished[rank] += 1
                    finished[other] += 1
                    progress = True
    return all(finished[r] == len(programs[r]) for r in range(group_size)) and all(
        len(attention[r]) == num_layers and len(moe[r]) == num_layers for r in range(group_size)
    )


def _hangs(report) -> bool:
    return bool(_kinds(report) & {ProblemKind.CYCLE, ProblemKind.UNPAIRED})


def _mutations(programs: list[list], plan, chooser: random.Random) -> list[tuple[str, list[list]]]:
    """Programs that a careless implementation of the plan might issue."""
    steps = [(s.direction, s.layer) for s in plan.steps if s.transfers]
    group_size = len(programs)
    mutated: list[tuple[str, list[list]]] = []
    for _ in range(3):
        first, second = chooser.sample(steps, 2)
        ranks = chooser.sample(range(group_size), chooser.randint(1, group_size))
        edited = [list(program) for program in programs]
        try:
            for rank in ranks:
                edited[rank] = _swap_steps(edited[rank], first, second)
        except ValueError:  # a rank that does not take part in one of the steps
            continue
        mutated.append((f"swap {first} and {second} on ranks {sorted(ranks)}", edited))
    rank = chooser.randrange(group_size)
    shuffled = [list(program) for program in programs]
    chooser.shuffle(shuffled[rank])
    mutated.append((f"shuffle rank {rank}", shuffled))
    peer_major = [list(program) for program in programs]
    peer_major[rank] = sorted(peer_major[rank], key=lambda op: op.peer)
    mutated.append((f"peer-major order on rank {rank}", peer_major))
    if programs[rank]:
        dropped = [list(program) for program in programs]
        del dropped[rank][chooser.randrange(len(dropped[rank]))]
        mutated.append((f"drop an operation of rank {rank}", dropped))
    return mutated


@pytest.mark.parametrize("ring_depth", [1, 2, 3])
def test_the_simulator_agrees_with_an_operational_model_on_good_and_mutated_programs(
    ring_depth: int,
) -> None:
    """Two implementations of the same rules, one a graph and one a run, must agree on whether a set
    of programs hangs. Good plans must run through; so must some mutations and others must not."""
    chooser = random.Random(7 * ring_depth)
    hangs = completes = 0
    for group_size, num_layers in ((2, 3), (2, 4), (3, 4)):
        layer_types = (CSA, SWA_ONLY, HCA, CSA)[:num_layers]
        for _ in range(12):
            owners = [chooser.randrange(group_size) for _ in range(num_layers)]
            batch = [
                PlanRequest(
                    rank + 1,
                    rank,
                    chooser.choice([0, 60, 700]),
                    chooser.choice([0, 9, 400]),
                    is_dummy=chooser.random() < 0.25,
                )
                for rank in range(group_size)
            ]
            plan = _plan(batch, owners, layer_types, group_size=group_size, ring_depth=ring_depth)
            programs = expand_plan(plan)
            assert _completes(plan, programs) and simulate(plan, programs).ok
            if not any(programs):
                continue
            for what, edited in _mutations(programs, plan, chooser):
                report = simulate(plan, edited)
                if _hangs(report):
                    hangs += 1
                else:
                    completes += 1
                assert _hangs(report) == (not _completes(plan, edited)), (
                    f"{what} on owners {owners}, ring depth {ring_depth}\n{report}"
                )
    # The comparison is only worth something if both outcomes occur.
    assert hangs >= 20 and completes >= 20
