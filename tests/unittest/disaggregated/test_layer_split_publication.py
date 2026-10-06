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
"""A layer-split context group is published to the generation workers as a pipeline.

Every rank of the group holds the layers of one range of the model for every request, which is what
a stage of a pipeline holds. The group is published as one tensor-parallel rank with a stage per
rank, so a generation worker finds every rank among the writers of a request.
"""

from types import SimpleNamespace

import pytest
from test_peer import _make_peer_registrar_and_peer_ri, make_rankinfo

from tensorrt_llm._torch.disaggregation.native.rank_info import RankInfo, layer_split_coverage_gap
from tensorrt_llm._torch.disaggregation.native.transfer import TransferWorker
from tensorrt_llm._torch.disaggregation.transceiver import layer_split_layer_counts

pytestmark = pytest.mark.cpu_only

_LAYERS = [11, 11, 11, 10]


def _published_rank(rank: int, group_size: int = 4, layers=_LAYERS) -> RankInfo:
    """The rank info of a rank of an attention-DP group after it joined the group."""
    info = make_rankinfo(
        "ctx",
        instance_rank=rank,
        tp_size=group_size,
        tp_rank=rank,
        dp_size=group_size,
        dp_rank=rank,
        enable_attention_dp=True,
        layer_num_per_pp=[layers[rank]],
    )
    endpoints = [f"tcp://rank{peer}" for peer in range(group_size)]
    worker = SimpleNamespace(_rank_info=info)
    TransferWorker.populate_instance_and_rank_info(worker, endpoints, list(layers), virtual_pp=True)
    return info


def _ranges(layers_per_rank):
    first = 0
    owned = []
    for count in layers_per_rank:
        owned.append(list(range(first, first + count)))
        first += count
    return owned


def test_the_group_is_published_as_a_pipeline_with_a_stage_per_rank():
    for rank in range(4):
        info = _published_rank(rank)
        assert (info.pp_size, info.pp_rank) == (4, rank)
        assert (info.tp_size, info.tp_rank) == (1, 0)
        assert (info.dp_size, info.dp_rank) == (1, 0)
        assert not info.attention.enable_attention_dp
        assert info.layer_num_per_pp == _LAYERS
        assert len(info.sender_endpoints) == 4
        # Whichever rank a generation worker asks, it reads the same description of the group.
        assert RankInfo.from_bytes(info.to_bytes()) == info


def test_a_group_that_is_not_published_as_a_pipeline_keeps_its_attention_dp_description():
    info = make_rankinfo(
        "ctx", tp_size=4, tp_rank=2, dp_size=4, dp_rank=2, enable_attention_dp=True
    )
    worker = SimpleNamespace(_rank_info=info)
    TransferWorker.populate_instance_and_rank_info(worker, ["a", "b", "c", "d"], [2])
    assert (info.tp_size, info.tp_rank, info.pp_size, info.dp_size, info.dp_rank) == (4, 2, 1, 4, 2)
    assert info.attention.enable_attention_dp


@pytest.mark.parametrize("gen_tp", [1, 2, 4])
def test_a_generation_worker_receives_a_request_from_every_rank_of_the_group(gen_tp):
    ctx = _published_rank(0)
    for gen_rank in range(gen_tp):
        gen = make_rankinfo(
            "gen", tp_size=gen_tp, tp_rank=gen_rank, pp_size=1, layer_num_per_pp=[sum(_LAYERS)]
        )
        registrar, ctx_info = _make_peer_registrar_and_peer_ri(gen, ctx)
        # The context rank that computed a request does not matter: the request belongs to the
        # one data-parallel rank of the published group.
        overlap = registrar.get_peer_overlap(ctx_info, 0)
        assert overlap.overlap_pp_size == 4
        assert sorted(overlap.ranks) == [0, 1, 2, 3]
        assert len(overlap.ranks) == 4


def test_the_ranks_are_the_stages_of_consecutive_layer_ranges():
    owned = _ranges(_LAYERS)
    assert layer_split_layer_counts(owned) == _LAYERS
    assert layer_split_layer_counts([[0], [1], [2]]) == [1, 1, 1]
    assert layer_split_layer_counts([list(range(5))]) == [5]


@pytest.mark.parametrize(
    "owned",
    [
        [[0, 1], [3, 4]],
        [[2, 3], [0, 1]],
        [[0, 1], [2, 4]],
        [[0, 1], []],
        [[0, 2], [1]],
    ],
)
def test_ranks_that_do_not_own_consecutive_ranges_in_rank_order_cannot_be_published(owned):
    with pytest.raises(ValueError, match="consecutive layer ranges"):
        layer_split_layer_counts(owned)


# What a receiver does with the ranks that write to it


def _piece(expected: int, authorized):
    from test_task_handle import _sole_piece

    from tensorrt_llm import DisaggregatedParams
    from tensorrt_llm._torch.disaggregation.native.transfer import KVRecvTask

    task = KVRecvTask(7, _sole_piece(), 0, DisaggregatedParams(disagg_request_id=7), aux_slot=None)
    task.expected_transfers = expected
    task.authorized_writers = authorized
    return task


def test_a_report_from_a_rank_the_request_was_not_published_to_fails_the_piece():
    from tensorrt_llm._torch.disaggregation.native.transfer import TaskStatus

    task = _piece(2, {0, 1})
    assert task.note_writer_report(0, True) == (True, False)
    assert task.note_writer_report(5, True) == (False, False)
    assert task.status is TaskStatus.ERROR
    assert task.is_done
    assert "published to ranks [0, 1] only" in str(task._exception)


def test_every_authorized_writer_completes_the_piece_once():
    task = _piece(4, {0, 1, 2, 3})
    reports = [task.note_writer_report(rank, True) for rank in range(4)]
    assert reports == [(True, False)] * 3 + [(True, True)]


def test_a_piece_that_was_not_dispatched_counts_the_reports_that_it_gets():
    task = _piece(2, None)
    assert task.note_writer_report(7, True) == (True, False)
    assert task.note_writer_report(9, True) == (True, True)


# What a generation worker finds out about a layer-split context group when it registers


def test_the_layer_split_mark_is_sent_only_by_a_group_that_has_it():
    group = _published_rank(1)
    assert group.layer_split
    assert RankInfo.from_bytes(group.to_bytes()).layer_split
    ordinary = make_rankinfo("ctx")
    assert not ordinary.layer_split
    # A worker of an earlier version still reads the description of an ordinary instance.
    assert b"layer_split" not in ordinary.to_bytes()
    assert not RankInfo.from_bytes(ordinary.to_bytes()).layer_split


def test_a_rank_info_with_a_field_this_version_does_not_know_is_refused():
    import msgpack

    data = msgpack.unpackb(make_rankinfo("ctx").to_bytes(), strict_map_key=False)
    data["something_newer"] = 1
    with pytest.raises(ValueError, match=r"fields \['something_newer'\]"):
        RankInfo.from_bytes(msgpack.packb(data))


def test_a_layer_split_group_must_cover_the_layers_of_the_worker():
    from tensorrt_llm._torch.disaggregation.native.rank_info import validate_layer_split_peer

    group = _published_rank(0)
    gen = make_rankinfo("gen", tp_size=2, layer_num_per_pp=[sum(_LAYERS)])
    validate_layer_split_peer(gen, group)
    short = make_rankinfo("gen", tp_size=2, layer_num_per_pp=[sum(_LAYERS) + 1])
    with pytest.raises(ValueError, match="does not cover the 44 layers"):
        validate_layer_split_peer(short, group)
    # Only a layer-split group has to add up: a pipeline may leave out the layers of another side.
    validate_layer_split_peer(short, make_rankinfo("ctx", layer_num_per_pp=[1]))


@pytest.mark.parametrize(
    ("layers", "stage", "writers"),
    [
        ([22, 21], 0, [0, 1]),
        ([22, 21], 1, [2, 3]),
        # A stage of the generation worker may share a context rank with its neighbour.
        ([15, 14, 14], 0, [0, 1]),
        ([15, 14, 14], 1, [1, 2]),
        ([15, 14, 14], 2, [2, 3]),
    ],
)
def test_a_pipelined_generation_worker_is_written_by_the_ranks_that_hold_its_layers(
    layers, stage, writers
):
    ctx = _published_rank(0)
    gen = make_rankinfo(
        "gen", tp_size=1, pp_size=len(layers), pp_rank=stage, layer_num_per_pp=layers
    )
    registrar, ctx_info = _make_peer_registrar_and_peer_ri(gen, ctx)
    overlap = registrar.get_peer_overlap(ctx_info, 0)
    assert sorted(overlap.ranks) == writers
    assert layer_split_coverage_gap(gen, ctx, overlap.ranks) is None


def test_the_layers_of_a_worker_must_be_covered_exactly_by_the_ranks_that_write_to_it():
    ctx = _published_rank(0)
    whole = make_rankinfo("gen", tp_size=2, layer_num_per_pp=[sum(_LAYERS)])
    assert layer_split_coverage_gap(whole, ctx, [0, 1, 2, 3]) is None
    assert "send 33 of the 43 layers of this rank (0..42)" in layer_split_coverage_gap(
        whole, ctx, [0, 1, 2]
    )
    assert "rank 4 is not one of the 4 ranks" in layer_split_coverage_gap(
        whole, ctx, [0, 1, 2, 3, 4]
    )
    second_half = make_rankinfo("gen", tp_size=1, pp_size=2, pp_rank=1, layer_num_per_pp=[22, 21])
    assert layer_split_coverage_gap(second_half, ctx, [2, 3]) is None
    # Ranks that hold layers of another stage do not make up for the layers that are missing.
    assert "send 11 of the 21 layers of this rank (22..42)" in layer_split_coverage_gap(
        second_half, ctx, [0, 1, 2]
    )
