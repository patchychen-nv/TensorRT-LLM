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
"""DKV request counts retain compute-rank ownership through ADP stats fanout."""

from dataclasses import fields, replace

import pytest
from dkv_test_utils import LockstepTpGroup

from tensorrt_llm._torch.pyexecutor.adp_iter_stats import ADPIterStatsBuffer
from tensorrt_llm._torch.pyexecutor.scheduler.adp_router import RankIterStatsPayload, RankState
from tensorrt_llm.bindings.executor import InflightBatchingStats, IterationStats

pytestmark = pytest.mark.cpu_only


def _stats(iteration: int, count: int) -> IterationStats:
    stats = IterationStats()
    stats.iter = iteration
    stats.num_new_active_requests = count
    stats.num_active_requests = 2 * count
    stats.num_completed_requests = count
    stats.num_queued_requests = 7
    stats.new_active_requests_queue_latency_ms = 3.5
    ifb = InflightBatchingStats()
    ifb.num_context_requests = count
    ifb.num_ctx_tokens = count * 8
    ifb.num_queued_context_requests = 7
    stats.inflight_batching_stats = ifb
    return stats


@pytest.mark.parametrize("dkv_enabled", [False, True])
@pytest.mark.parametrize("counts", [(0, 1), (2, 3), (0, 1, 2, 3)])
def test_request_count_fanout_uses_local_dkv_counts_only(counts, dkv_enabled: bool) -> None:
    group = LockstepTpGroup(len(counts))

    def run_rank(dist):
        buffer = ADPIterStatsBuffer()
        for iteration in range(3):
            buffer.queue(
                _stats(iteration, counts[dist.tp_rank]),
                is_rank0=dist.tp_rank == 0,
                dkv_enabled=dkv_enabled,
            )
            state = RankState(rank=dist.tp_rank, iter_stats=buffer.next_payload())
            all_states = [
                RankState.deserialize(data) for data in dist.tp_allgather(state.serialize())
            ]
            records = buffer.finalize(all_states, is_rank0=dist.tp_rank == 0)
            if dist.tp_rank != 0:
                assert records == []
                continue
            rows = [record.stats for record in records]
            assert len(rows) == len(counts)
            assert [row.inflight_batching_stats.num_context_requests for row in rows] == list(
                counts
            )
            assert [row.num_new_active_requests for row in rows] == (
                list(counts) if dkv_enabled else [counts[0]] + [0] * (len(counts) - 1)
            )
            assert [row.num_completed_requests for row in rows] == (
                list(counts) if dkv_enabled else [counts[0]] + [0] * (len(counts) - 1)
            )
            assert [row.num_active_requests for row in rows] == (
                [2 * count for count in counts] if dkv_enabled else [2 * counts[0]] * len(counts)
            )
            assert sum(row.num_queued_requests for row in rows) == 7
            assert sum(row.inflight_batching_stats.num_queued_context_requests for row in rows) == 7
            assert [row.new_active_requests_queue_latency_ms for row in rows] == [3.5] + [0.0] * (
                len(counts) - 1
            )
            if dkv_enabled:
                assert sum(row.num_completed_requests for row in rows) == sum(counts)

    group.run(run_rank)
    assert [len(trace) for trace in group.traces] == [3] * len(counts)


def test_idle_rank_synthetic_payload_preserves_zero_dkv_counts() -> None:
    group = LockstepTpGroup(2)

    def run_rank(dist):
        buffer = ADPIterStatsBuffer()
        if dist.tp_rank == 0:
            buffer.queue(_stats(7, 2), is_rank0=True, dkv_enabled=True)
        records = []
        for _ in range(2):
            state = RankState(rank=dist.tp_rank)
            state.copy_iter_stats_from(buffer.next_payload())
            all_states = [
                RankState.deserialize(data) for data in dist.tp_allgather(state.serialize())
            ]
            records = buffer.finalize(all_states, is_rank0=dist.tp_rank == 0)
        if dist.tp_rank == 0:
            assert [row.stats.num_new_active_requests for row in records] == [2, 0]
            assert [row.stats.num_active_requests for row in records] == [4, 0]
            assert [row.stats.num_completed_requests for row in records] == [2, 0]
        assert buffer.next_payload() is None

    group.run(run_rank)


def test_stats_payload_accepts_legacy_wire_fields_without_local_count_semantics() -> None:
    payload = RankIterStatsPayload.deserialize([1, 7, 1, 8, 0, 0, 0, 0, 0])
    assert payload.has_local_request_counts == 0
    assert payload.num_new_active_requests == 0
    assert payload.num_active_requests == 0
    assert payload.num_completed_requests == 0


_DKV_COUNT_FIELDS = (
    "has_local_request_counts",
    "num_new_active_requests",
    "num_active_requests",
    "num_completed_requests",
)


def test_dkv_counts_precede_the_measurement_snapshot_at_the_end_of_the_payload() -> None:
    names = tuple(field.name for field in fields(RankIterStatsPayload))
    assert names[-5:-1] == _DKV_COUNT_FIELDS
    assert names[-1] == "dkv_measurement"


def test_payload_without_dkv_counts_keeps_its_original_width() -> None:
    payload = RankIterStatsPayload(has_iter_stats=1, iter_stats_iter=4, num_gen_requests=2)
    assert len(payload.serialize()) == 9
    state = RankState(rank=3, num_active_requests=5, iter_stats=payload)
    assert len(state.serialize()) == 13
    assert RankState.deserialize(state.serialize()) == state


@pytest.mark.parametrize("name", _DKV_COUNT_FIELDS)
def test_any_nonzero_dkv_count_is_transported_with_the_full_width(name: str) -> None:
    payload = replace(RankIterStatsPayload(has_iter_stats=1), **{name: 3})
    assert len(payload.serialize()) == 13
    assert RankIterStatsPayload.deserialize(payload.serialize()) == payload
    state = RankState(rank=1, iter_stats=payload)
    assert len(state.serialize()) == 17
    assert RankState.deserialize(state.serialize()) == state


def test_ranks_with_different_payload_widths_still_exchange_and_decode() -> None:
    group = LockstepTpGroup(2)

    def run_rank(dist):
        counts = RankIterStatsPayload(has_local_request_counts=1, num_active_requests=2)
        state = RankState(
            rank=dist.tp_rank, iter_stats=counts if dist.tp_rank == 0 else RankIterStatsPayload()
        )
        return [RankState.deserialize(data) for data in dist.tp_allgather(state.serialize())]

    states = group.run(run_rank)[0]
    assert states[0].iter_stats.num_active_requests == 2
    assert states[1].iter_stats == RankIterStatsPayload()


def _snapshot(rank: int, requests: int) -> dict:
    return {
        "rank": rank,
        "group_size": 2,
        "counters": {"request_count": requests},
        "cache_tiers": ["gpu"],
        "pools_by_level": [[{"slot_sizes": [8], "total": 32}]],
    }


def test_measurement_snapshot_travels_with_the_payload_and_widens_it() -> None:
    payload = RankIterStatsPayload(has_iter_stats=1, dkv_measurement=_snapshot(1, 2))
    values = payload.serialize()
    assert len(values) == 14 and values[-1] == _snapshot(1, 2)
    assert RankIterStatsPayload.deserialize(values) == payload
    state = RankState(rank=1, iter_stats=payload)
    assert len(state.serialize()) == 18
    assert RankState.deserialize(state.serialize()) == state


def test_each_rank_reports_its_own_measurement_through_the_stats_fanout() -> None:
    group = LockstepTpGroup(2)

    def run_rank(dist):
        # Rank 1 has no work, so the fanout gives it a zero payload that still reports its counters.
        buffer = ADPIterStatsBuffer(
            measurement_provider=lambda: _snapshot(dist.tp_rank, 5 * dist.tp_rank)
        )
        if dist.tp_rank == 0:
            buffer.queue(
                _stats(7, 2), is_rank0=True, dkv_enabled=True, dkv_measurement=_snapshot(0, 9)
            )
        records = []
        for _ in range(2):
            state = RankState(rank=dist.tp_rank)
            state.copy_iter_stats_from(buffer.next_payload())
            all_states = [
                RankState.deserialize(data) for data in dist.tp_allgather(state.serialize())
            ]
            records = buffer.finalize(all_states, is_rank0=dist.tp_rank == 0)
        return [(record.attention_dp_rank, record.dkv_measurement) for record in records]

    rows = group.run(run_rank)[0]
    assert rows == [(0, _snapshot(0, 9)), (1, _snapshot(1, 5))]


def test_rows_without_a_provider_or_snapshot_carry_none() -> None:
    group = LockstepTpGroup(2)

    def run_rank(dist):
        buffer = ADPIterStatsBuffer()
        if dist.tp_rank == 0:
            buffer.queue(_stats(7, 2), is_rank0=True)
        records = []
        for _ in range(2):
            state = RankState(rank=dist.tp_rank)
            state.copy_iter_stats_from(buffer.next_payload())
            all_states = [
                RankState.deserialize(data) for data in dist.tp_allgather(state.serialize())
            ]
            records = buffer.finalize(all_states, is_rank0=dist.tp_rank == 0)
        return [record.dkv_measurement for record in records]

    assert group.run(run_rank)[0] == [None, None]
