# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The context-worker exit of the real executor loop: the senders, symmetric release."""

from collections import Counter

import pytest
from dkv_loop_harness import LoopScript, RankRun, run_loop

pytestmark = pytest.mark.cpu_only


def _script(group_size: int, owners: list[int], **changes) -> LoopScript:
    """One context-only request per entry of ``owners``, all arriving in the first iteration."""
    ids = [10 + index for index in range(len(owners))]
    arrivals = [[(request_id, owner, 4) for request_id, owner in zip(ids, owners)]]
    arrivals += [[] for _ in range(6)]
    return LoopScript(
        arrivals,
        max_num_tokens=16,
        batch_size=len(owners) + 1,
        context_only=set(ids),
        **changes,
    )


def _iterations(run: RankRun, name: str) -> dict[int, int]:
    """Iteration in which each request id entered the named phase."""
    return {value: iteration for iteration, event, value in run.events if event == name}


@pytest.mark.parametrize("group_size", [2, 4, 8])
def test_only_the_owner_sends_and_polls_while_every_rank_starts_the_transfer(
    group_size: int,
) -> None:
    owners = [1] * 3
    runs = run_loop(group_size, _script(group_size, owners, transfer_done_iteration=3))
    ids = [10, 11, 12]
    for rank, run in enumerate(runs):
        assert run.sends == (ids if rank == 1 else [])
        assert (run.status_polls > 0) == (rank == 1)
        # Both replicas and the owner give back the index slot once.
        assert sorted(run.pages.released_index_slots) == ids


@pytest.mark.parametrize("group_size", [2, 4, 8])
def test_pages_are_released_in_the_same_iteration_on_every_rank_after_completion(
    group_size: int,
) -> None:
    owners = [1, 0, 1]
    runs = run_loop(group_size, _script(group_size, owners, transfer_done_iteration=3))
    ids = [10, 11, 12]
    sent = [_iterations(run, "send") for run in runs]
    freed = [_iterations(run, "free") for run in runs]
    assert all(rank_freed == freed[0] for rank_freed in freed), "ranks released at different times"
    assert set(freed[0]) == set(ids)
    for owner_rank, request_id in zip(owners, ids):
        # The send starts in the prefill iteration; the release waits for the observed completion.
        assert sent[owner_rank][request_id] == 0
        assert freed[0][request_id] >= 3
    assert all(run.final_pages == run.initial_pages for run in runs)


@pytest.mark.parametrize("group_size", [2, 4, 8])
def test_one_reply_per_context_request_comes_from_its_owner(group_size: int) -> None:
    owners = [1, 0, 1]
    runs = run_loop(group_size, _script(group_size, owners, transfer_done_iteration=2))
    expected = {10 + index: owner for index, owner in enumerate(owners)}
    assert Counter(request_id for request_id, _ in runs[0].emitted) == Counter(
        dict.fromkeys(expected, 1)
    )
    assert all(rank == expected[request_id] for request_id, rank in runs[0].emitted)


def test_a_sampler_failure_in_the_prefill_iteration_keeps_that_request_out_of_the_transfer() -> (
    None
):
    owners = [0, 1]
    script = _script(2, owners, transfer_done_iteration=2)
    script.fail_ids = {11}
    runs = run_loop(2, script)
    for run in runs:
        assert 11 not in run.pages.released_index_slots
        assert 10 in run.pages.released_index_slots
        assert run.sends == ([10] if run is runs[0] else [])
        assert [ids for _, ids in run.errors] == [(11,)]
        assert run.final_pages == run.initial_pages
    assert runs[0].pages.freed == runs[1].pages.freed


def test_a_late_completion_keeps_the_pages_on_every_rank_until_it_is_observed() -> None:
    runs = run_loop(2, _script(2, [1], transfer_done_iteration=5))
    for run in runs:
        freed_at = _iterations(run, "free")[10]
        assert freed_at >= 5
        assert run.final_pages == run.initial_pages
    assert _iterations(runs[0], "free") == _iterations(runs[1], "free")


# The layer-split layout: every rank sends the layers it owns of a request

_FAILED = 99


@pytest.mark.parametrize("group_size", [2, 4, 8])
def test_every_rank_of_a_layer_split_group_sends_and_polls_and_the_owner_alone_replies(
    group_size: int,
) -> None:
    owners = [1, 0, 1]
    runs = run_loop(
        group_size, _script(group_size, owners, layer_split=True, transfer_done_iteration=3)
    )
    ids = [10, 11, 12]
    for run in runs:
        assert sorted(run.sends) == ids
        assert run.status_polls > 0
        assert sorted(run.pages.released_index_slots) == ids
        assert run.final_pages == run.initial_pages
    expected = {10 + index: owner for index, owner in enumerate(owners)}
    assert Counter(request_id for request_id, _ in runs[0].emitted) == Counter(
        dict.fromkeys(expected, 1)
    )
    assert all(rank == expected[request_id] for request_id, rank in runs[0].emitted)
    freed = [_iterations(run, "free") for run in runs]
    assert all(rank_freed == freed[0] for rank_freed in freed), "ranks released at different times"


@pytest.mark.parametrize("group_size", [2, 4])
def test_a_layer_split_request_is_released_when_the_last_rank_has_completed(
    group_size: int,
) -> None:
    done = {rank: 1 + rank for rank in range(group_size)}
    runs = run_loop(
        group_size,
        _script(group_size, [0], layer_split=True, transfer_done_by_rank=done),
    )
    freed = [_iterations(run, "free") for run in runs]
    assert all(rank_freed == freed[0] for rank_freed in freed)
    assert freed[0][10] >= max(done.values())
    assert all(run.final_pages == run.initial_pages for run in runs)


def test_a_layer_split_request_fails_on_every_rank_when_one_rank_reports_an_error() -> None:
    """The ranks whose sends are unfinished are told to cancel them, and then all release."""
    group_size = 4
    runs = run_loop(
        group_size,
        _script(
            group_size,
            [2],
            layer_split=True,
            transfer_done_iteration=_FAILED,
            transfer_fails=(1, 2),
        ),
    )
    freed = [_iterations(run, "free") for run in runs]
    assert all(rank_freed == freed[0] for rank_freed in freed)
    for rank, run in enumerate(runs):
        assert [ids for _, ids in run.errors] == [(10,)], rank
        assert run.final_pages == run.initial_pages
        cancelled = run.executor.kv_cache_transceiver.cancel_request.call_count
        # The rank that reported the error has nothing left to cancel; the others cancel theirs.
        assert (cancelled > 0) == (rank != 1), (rank, cancelled)
    assert len({message for run in runs for message, _ in run.errors}) == 1
