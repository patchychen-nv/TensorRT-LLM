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
"""S-control event fan-in and always-on capacity-digest consistency with strict TP exchange."""

from dataclasses import replace

import pytest
from dkv_test_utils import LockstepTpGroup

from tensorrt_llm._torch.disaggregation.orchestration.interfaces import DkvTransferEvent
from tensorrt_llm._torch.pyexecutor.dkv import (
    DkvControlDigest,
    DkvControlPayload,
    DkvControlResult,
    digest_request_ids,
    sync_dkv_control,
)

pytestmark = pytest.mark.cpu_only


def test_transfer_events_are_ordered_within_the_single_control_exchange() -> None:
    group = LockstepTpGroup(3)
    events = (
        DkvTransferEvent(9, 0, "completed"),
        DkvTransferEvent(3, 1, "timed_out", "transfer deadline"),
        DkvTransferEvent(7, 2, "failed", "sender error"),
    )
    results = group.run(
        lambda dist: sync_dkv_control(dist, _payload(transfer_events=(events[dist.tp_rank],)))
    )
    assert results == [DkvControlResult((), False, (events[1], events[2], events[0]))] * 3
    assert [len(trace) for trace in group.traces] == [1, 1, 1]


@pytest.mark.parametrize(
    "events",
    [
        [DkvTransferEvent(4, 1, "completed")],
        ("invalid",),
        (DkvTransferEvent(True, 1, "completed"),),
        (DkvTransferEvent(4, True, "completed"),),
        (DkvTransferEvent(4, 0, "completed"),),
        (DkvTransferEvent(4, 1, "pending"),),
        (DkvTransferEvent(4, 1, "failed", None),),
        (DkvTransferEvent(4, 1, "completed"), DkvTransferEvent(4, 1, "failed")),
    ],
)
def test_invalid_transfer_events_fail_on_every_rank_without_an_extra_collective(events) -> None:
    group = LockstepTpGroup(2)

    def run_rank(dist):
        payload = _payload(transfer_events=events if dist.tp_rank else ())
        with pytest.raises(RuntimeError, match="transfer event") as caught:
            sync_dkv_control(dist, payload)
        return str(caught.value)

    errors = group.run(run_rank)
    assert errors[0] == errors[1]
    assert [len(trace) for trace in group.traces] == [1, 1]


def test_transfer_id_cannot_be_claimed_by_two_owners() -> None:
    group = LockstepTpGroup(2)

    def run_rank(dist):
        event = DkvTransferEvent(4, dist.tp_rank, "completed")
        with pytest.raises(RuntimeError, match="duplicate transfer event") as caught:
            sync_dkv_control(dist, _payload(transfer_events=(event,)))
        return str(caught.value)

    errors = group.run(run_rank)
    assert errors[0] == errors[1]


def test_request_id_digest_ignores_order_and_distinguishes_sets() -> None:
    assert digest_request_ids([3, 1, 2]) == digest_request_ids((1, 2, 3))
    assert digest_request_ids([]) != digest_request_ids([0])
    assert digest_request_ids([1, 2]) != digest_request_ids([12])
    assert digest_request_ids([1, 2, 3]) != digest_request_ids([1, 2, 4])
    assert 0 <= digest_request_ids([5]) < 2**64


def _payload(**changes) -> DkvControlPayload:
    return DkvControlPayload(
        digest=DkvControlDigest(
            iter_counter=7,
            free_pages=((24, 16, 8), (12, 4)),
            index_mapper_used=3,
            active_request_count=2,
        ),
        **changes,
    )


def test_idle_iterations_each_use_one_control_collective_without_a_flush() -> None:
    group = LockstepTpGroup(3)

    def run_rank(dist):
        for iteration in range(3):
            payload = _payload()
            payload = replace(payload, digest=replace(payload.digest, iter_counter=iteration))
            assert sync_dkv_control(dist, payload) == DkvControlResult((), False)

    group.run(run_rank)
    assert [len(trace) for trace in group.traces] == [3, 3, 3]
    assert all(step[2] == "tp_allgather" for trace in group.traces for step in trace)


def test_pending_response_and_fatal_events_are_combined_in_rank_order() -> None:
    group = LockstepTpGroup(3)

    def run_rank(dist):
        messages = ("first failure", "second failure") if dist.tp_rank == 2 else ()
        if dist.tp_rank == 0:
            messages = ("budget exhausted",)
        return sync_dkv_control(
            dist,
            _payload(fatal_messages=messages, has_pending_responses=dist.tp_rank == 1),
        )

    expected = DkvControlResult(
        ("rank 0: budget exhausted", "rank 2: first failure", "rank 2: second failure"), True
    )
    assert group.run(run_rank) == [expected] * group.size
    assert [len(trace) for trace in group.traces] == [1, 1, 1]


def test_transfer_events_of_eight_ranks_are_sorted_by_request_id() -> None:
    group = LockstepTpGroup(8)
    request_ids = (41, 7, 23, 3, 88, 12, 60, 31)
    events = tuple(DkvTransferEvent(request_ids[rank], rank, "completed") for rank in range(8))
    results = group.run(
        lambda dist: sync_dkv_control(dist, _payload(transfer_events=(events[dist.tp_rank],)))
    )
    ordered = tuple(sorted(events, key=lambda event: event.request_id))
    assert results == [DkvControlResult((), False, ordered)] * 8
    assert [len(trace) for trace in group.traces] == [1] * 8


def test_fatal_events_and_pending_responses_of_eight_ranks_reach_every_rank_in_rank_order() -> None:
    group = LockstepTpGroup(8)

    def run_rank(dist):
        messages = (f"failure on {dist.tp_rank}",) if dist.tp_rank in (3, 7) else ()
        return sync_dkv_control(
            dist, _payload(fatal_messages=messages, has_pending_responses=dist.tp_rank == 5)
        )

    expected = DkvControlResult(("rank 3: failure on 3", "rank 7: failure on 7"), True)
    assert group.run(run_rank) == [expected] * 8
    assert [len(trace) for trace in group.traces] == [1] * 8


@pytest.mark.parametrize(
    "diverging",
    [(7,), (1, 6), (1, 2, 3, 4, 5, 6, 7)],
    ids=["last-rank", "two-ranks", "all-but-rank-0"],
)
def test_every_diverging_rank_of_eight_is_named_on_every_rank(diverging) -> None:
    group = LockstepTpGroup(8)

    def run_rank(dist):
        payload = _payload()
        if dist.tp_rank in diverging:
            payload = replace(payload, digest=replace(payload.digest, active_request_count=3))
        with pytest.raises(RuntimeError, match="capacity digest differs") as caught:
            sync_dkv_control(dist, payload)
        return str(caught.value)

    errors = group.run(run_rank)
    assert len(set(errors)) == 1
    for rank in range(1, 8):
        named = f"rank {rank}: capacity digest differs from rank 0" in errors[0]
        assert named == (rank in diverging), (rank, errors[0])
    assert [len(trace) for trace in group.traces] == [1] * 8


def test_a_diverging_rank_zero_makes_every_other_rank_of_eight_differ_from_it() -> None:
    group = LockstepTpGroup(8)

    def run_rank(dist):
        payload = _payload()
        if dist.tp_rank == 0:
            payload = replace(payload, digest=replace(payload.digest, index_mapper_used=4))
        with pytest.raises(RuntimeError, match="capacity digest differs") as caught:
            sync_dkv_control(dist, payload)
        return str(caught.value)

    errors = group.run(run_rank)
    assert len(set(errors)) == 1
    assert all(
        f"rank {rank}: capacity digest differs from rank 0" in errors[0] for rank in range(1, 8)
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"iter_counter": 8},
        {"free_pages": ((23, 16, 8), (12, 4))},
        {"free_pages": ((16, 24, 8), (12, 4))},
        {"free_pages": ((24, 16), (8, 12, 4))},
        {"index_mapper_used": 4},
        {"active_request_count": 3},
        {"in_transfer_count": 1},
        {"in_transfer_digest": digest_request_ids([4])},
    ],
)
def test_capacity_digest_divergence_is_fatal_even_with_debug_disabled(changes) -> None:
    group = LockstepTpGroup(2)

    def run_rank(dist):
        payload = _payload()
        assert payload.debug_enabled is False
        if dist.tp_rank == 1:
            payload = replace(payload, digest=replace(payload.digest, **changes))
        with pytest.raises(RuntimeError, match="capacity digest differs") as caught:
            sync_dkv_control(dist, payload)
        assert "rank 0: DkvControlPayload" in str(caught.value)
        assert "rank 1: DkvControlPayload" in str(caught.value)
        return str(caught.value)

    errors = group.run(run_rank)
    assert errors[0] == errors[1]
    assert [len(trace) for trace in group.traces] == [1, 1]


def test_freed_request_set_is_piggybacked_on_the_same_collective() -> None:
    group = LockstepTpGroup(2)
    expected = DkvControlResult((), False)
    assert group.run(
        lambda dist: sync_dkv_control(dist, _payload(debug_enabled=True, freed_request_ids=(7, 8)))
    ) == [expected, expected]
    assert [len(trace) for trace in group.traces] == [1, 1]


@pytest.mark.parametrize("rank_one_ids", [(7,), (7, 9), (7, 7, 8), (8, 7)])
def test_different_and_duplicate_frees_are_detected(rank_one_ids) -> None:
    group = LockstepTpGroup(2)

    def run_rank(dist):
        freed_ids = (7, 8) if dist.tp_rank == 0 else rank_one_ids
        with pytest.raises(RuntimeError, match="freed request IDs differ"):
            sync_dkv_control(dist, _payload(debug_enabled=True, freed_request_ids=freed_ids))

    group.run(run_rank)
    assert [len(trace) for trace in group.traces] == [1, 1]


def test_debug_flag_divergence_is_rejected_at_control_boundary() -> None:
    group = LockstepTpGroup(2)

    def run_rank(dist):
        with pytest.raises(RuntimeError, match="debug flag differs"):
            sync_dkv_control(dist, _payload(debug_enabled=dist.tp_rank == 1))

    group.run(run_rank)


@pytest.mark.parametrize(
    "changes, diagnostic",
    [
        ({"fatal_messages": ["failure"]}, "invalid fatal messages"),
        ({"fatal_messages": (42,)}, "invalid fatal messages"),
        ({"has_pending_responses": 1}, "invalid pending-response flag"),
        ({"debug_enabled": 1}, "invalid debug flag"),
        ({"freed_request_ids": [7]}, "invalid free request IDs"),
        ({"freed_request_ids": ("7",)}, "invalid free request IDs"),
        ({"freed_request_ids": (True,)}, "invalid free request IDs"),
        ({"freed_request_ids": (7,)}, "debug state supplied with debugging disabled"),
        ({"digest": None}, "invalid capacity digest"),
    ],
)
def test_malformed_control_payload_fails_on_every_rank(changes, diagnostic: str) -> None:
    group = LockstepTpGroup(2)

    def run_rank(dist):
        payload = replace(_payload(), **changes) if dist.tp_rank == 1 else _payload()
        with pytest.raises(RuntimeError, match=diagnostic):
            sync_dkv_control(dist, payload)

    group.run(run_rank)
    assert [len(trace) for trace in group.traces] == [1, 1]


@pytest.mark.parametrize(
    "changes",
    [
        {"iter_counter": -1},
        {"index_mapper_used": True},
        {"active_request_count": -1},
        {"in_transfer_count": -1},
        {"in_transfer_digest": True},
        {"free_pages": [[24, 16, 8], [12, 4]]},
        {"free_pages": ((24, -1),)},
        {"free_pages": ((24, True),)},
    ],
)
def test_malformed_capacity_digest_is_rejected_after_exchange(changes) -> None:
    group = LockstepTpGroup(2)

    def run_rank(dist):
        payload = _payload()
        if dist.tp_rank == 1:
            payload = replace(payload, digest=replace(payload.digest, **changes))
        with pytest.raises(RuntimeError, match="invalid capacity digest"):
            sync_dkv_control(dist, payload)

    group.run(run_rank)
    assert [len(trace) for trace in group.traces] == [1, 1]


def test_wrong_payload_type_is_reported_without_a_diagnostic_collective() -> None:
    group = LockstepTpGroup(2)

    def run_rank(dist):
        with pytest.raises(RuntimeError, match="invalid control payload"):
            sync_dkv_control(dist, None if dist.tp_rank else _payload())

    group.run(run_rank)
    assert [len(trace) for trace in group.traces] == [1, 1]
