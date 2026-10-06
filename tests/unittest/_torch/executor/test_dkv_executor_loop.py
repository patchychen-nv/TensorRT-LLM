# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The real executor loop keeps every rank in lockstep and orders S-sample before the commits."""

from collections import Counter, defaultdict
from unittest.mock import patch

import pytest
from dkv_loop_harness import LoopScript, RankRun, run_loop

pytestmark = pytest.mark.cpu_only

# Phases of an iteration whose global batch is not empty, in the order the loop enters them.
_BUSY = [
    "control",
    "schedule",
    "forward_batch",
    "prepare_resources",
    "forward",
    "sample_sync",
    "context_commit",
    "update_resources",
]
_IDLE = ["control", "schedule", "forward_batch"]
# Recorded for observation only; they are not phases the loop enters.
_OBSERVATIONS = {"free", "send", "errors"}


def _arrivals(group_size: int, per_iteration: tuple[int, ...]) -> list[list[tuple[int, int, int]]]:
    """Requests whose owners rotate over the group, so every rank is busy and idle in turn."""
    request_id = 10
    script = []
    for iteration, count in enumerate(per_iteration):
        arrivals = []
        for index in range(count):
            arrivals.append((request_id, (iteration + index) % group_size, 4 + 4 * (index % 2)))
            request_id += 1
        script.append(arrivals)
    return script


def _phases(run: RankRun) -> dict[int, list[str]]:
    by_iteration: dict[int, list[str]] = defaultdict(list)
    for iteration, name, _ in run.events:
        if name not in _OBSERVATIONS:
            by_iteration[iteration].append(name)
    return by_iteration


@pytest.mark.parametrize("group_size", [2, 4, 8])
def test_every_iteration_enters_the_same_phases_in_the_documented_order(group_size: int) -> None:
    arrivals = _arrivals(group_size, (2, 1, 0, 3, 2))
    runs = run_loop(group_size, LoopScript(arrivals, batch_size=2, max_num_tokens=16))
    phases = [_phases(run) for run in runs]
    assert all(rank == phases[0] for rank in phases), "ranks entered different phases"
    scheduled = [name for iteration in sorted(phases[0]) for name in phases[0][iteration]]
    assert scheduled[-2:] == ["control", "schedule"], "the loop must leave right after scheduling"
    for iteration, names in phases[0].items():
        assert names in (_BUSY, _IDLE, ["control", "schedule"]), (iteration, names)
    # S-sample sits between the forward and the commits that may free replicated resources.
    assert any(names == _BUSY for names in phases[0].values())
    assert [len(trace) for trace in runs[0].group.traces] == [len(runs[0].group.traces[0])] * (
        group_size
    )


@pytest.mark.parametrize("group_size", [2, 4, 8])
def test_forward_batches_hold_only_owned_requests_or_the_resident_dummy(group_size: int) -> None:
    arrivals = _arrivals(group_size, (2, 1, 0, 3, 2))
    owners = {request_id: owner for batch in arrivals for request_id, owner, _ in batch}
    runs = run_loop(group_size, LoopScript(arrivals, batch_size=2, max_num_tokens=16))
    for rank, run in enumerate(runs):
        dummy_id = run.executor._dkv_forward_dummies[rank].py_request_id
        for iteration, name, value in run.events:
            if name != "forward":
                continue
            real = [request_id for request_id, dummy in value if not dummy]
            assert all(owners[request_id] == rank for request_id in real), (iteration, value)
            assert real or value == [(dummy_id, True)], (iteration, value)


@pytest.mark.parametrize("group_size", [2, 4, 8])
def test_each_request_is_answered_once_by_its_owner_and_pages_return(group_size: int) -> None:
    arrivals = _arrivals(group_size, (2, 1, 0, 3, 2))
    owners = {request_id: owner for batch in arrivals for request_id, owner, _ in batch}
    runs = run_loop(group_size, LoopScript(arrivals, batch_size=2, max_num_tokens=16))
    emitted = runs[0].emitted
    assert Counter(request_id for request_id, _ in emitted) == Counter(dict.fromkeys(owners, 1))
    assert all(rank == owners[request_id] for request_id, rank in emitted)
    assert all(not run.emitted for run in runs[1:])
    for run in runs:
        assert run.final_pages == run.initial_pages
        assert not run.executor.active_requests
    assert all(run.pages.freed == runs[0].pages.freed for run in runs)
    assert sorted(runs[0].pages.freed) == sorted(owners)


@pytest.mark.parametrize("group_size", [2, 4, 8])
def test_healthy_collectives_per_iteration_do_not_exceed_the_adp_baseline(group_size: int) -> None:
    arrivals = _arrivals(group_size, (2, 1, 0, 3, 2))
    runs = run_loop(group_size, LoopScript(arrivals, batch_size=2, max_num_tokens=16))
    phases = _phases(runs[0])
    busy = sum(names == _BUSY for names in phases.values())
    idle = sum(names == _IDLE for names in phases.values())
    # The checker's startup agreement, then per iteration. Busy: S-control, S-sample and the
    # response gather (the ADP baseline also uses three). Idle: S-control only. The final
    # iteration: S-control and the shutdown flush.
    assert len(runs[0].group.traces[0]) == 1 + 3 * busy + idle + 2


@pytest.mark.parametrize("dual_ledger", [True, False])
def test_dual_ledger_switch_keeps_the_loop_in_lockstep(dual_ledger: bool) -> None:
    arrivals = _arrivals(2, (2, 3, 0, 2))
    runs = run_loop(2, LoopScript(arrivals, max_num_tokens=16, dual_ledger=dual_ledger))
    assert runs[0].pages.freed == runs[1].pages.freed
    assert all(run.final_pages == run.initial_pages for run in runs)


def test_debug_checks_pass_through_a_healthy_run() -> None:
    arrivals = _arrivals(2, (2, 1, 0, 3))
    runs = run_loop(2, LoopScript(arrivals, max_num_tokens=16, debug=True))
    assert all(run.final_pages == run.initial_pages for run in runs)


def test_a_rank_local_sampler_failure_is_failed_on_every_rank_in_the_same_iteration() -> None:
    arrivals = _arrivals(2, (2, 2, 0))
    owners = {request_id: owner for batch in arrivals for request_id, owner, _ in batch}
    failing = next(request_id for request_id, owner in owners.items() if owner == 1)
    runs = run_loop(2, LoopScript(arrivals, max_num_tokens=16, fail_ids={failing}))
    failed_iterations = [
        [iteration for iteration, name, _ in run.events if name == "errors"] for run in runs
    ]
    assert len(failed_iterations[0]) == 1 and failed_iterations[0] == failed_iterations[1]
    for run in runs:
        assert [ids for _, ids in run.errors] == [(failing,)]
        assert f"injected sampler failure [{failing}]" in run.errors[0][0]
        assert run.final_pages == run.initial_pages
    assert Counter(request_id for request_id, _ in runs[0].emitted) == Counter(
        dict.fromkeys(owners, 1)
    )
    assert runs[0].pages.freed == runs[1].pages.freed


def test_an_exhausted_error_budget_shuts_every_rank_down_together() -> None:
    arrivals = _arrivals(2, (2, 2, 0, 0))
    owners = {request_id: owner for batch in arrivals for request_id, owner, _ in batch}
    failing = next(request_id for request_id, owner in owners.items() if owner == 1)
    runs = run_loop(
        2, LoopScript(arrivals, max_num_tokens=16, fail_ids={failing}, fatal_budget=True)
    )
    for run in runs:
        assert isinstance(run.executor._fatal_error, RuntimeError)
        assert "rank 1: " in str(run.executor._fatal_error)
        assert run.executor.is_shutdown
        assert run.final_pages == run.initial_pages
    shutdown_iterations = [
        [
            iteration
            for iteration, name, value in run.events
            if name == "errors" and "rank 1" in value
        ]
        for run in runs
    ]
    assert shutdown_iterations[0] and shutdown_iterations[0] == shutdown_iterations[1]


def test_a_forward_exception_leaves_the_loop_instead_of_being_absorbed() -> None:
    arrivals = _arrivals(2, (2, 2, 1))
    with pytest.raises(RuntimeError, match="injected forward failure"):
        run_loop(2, LoopScript(arrivals, max_num_tokens=16, forward_error=(1, 1)))


def test_debug_state_check_names_the_iteration_where_kv_state_diverged() -> None:
    arrivals = _arrivals(2, (2, 2, 1, 0))
    with pytest.raises(RuntimeError, match=r"iter 1, tags=.*end of iteration kv state"):
        run_loop(2, LoopScript(arrivals, max_num_tokens=16, debug=True, leak=(1, 1)))


def test_capacity_digest_catches_diverged_kv_state_even_without_debug() -> None:
    arrivals = _arrivals(2, (2, 2, 1, 0))
    with pytest.raises(RuntimeError, match="capacity digest differs"):
        run_loop(2, LoopScript(arrivals, max_num_tokens=16, debug=False, leak=(1, 1)))


@pytest.mark.parametrize("group_size", [2, 4, 8])
def test_a_kv_stall_is_recovered_in_lockstep_and_every_request_still_completes(
    group_size: int,
) -> None:
    # The resident dummies keep one page each. Every request then holds one of the remaining pages
    # and none can get a second one: nothing is schedulable until the last started request gives
    # its page back.
    arrivals = [[(10 + rank, rank, 8) for rank in range(group_size)], [], [], [], [], [], []]
    script = LoopScript(
        arrivals, pages=2 * group_size, batch_size=group_size, max_num_tokens=4, chunked=True
    )
    with patch("tensorrt_llm._torch.pyexecutor.scheduler.scheduler_v2.logger.warning") as warning:
        runs = run_loop(group_size, script)
    # The patched logger sees every warning of the process, not only those of this loop.
    recoveries = [
        call.args[0] for call in warning.call_args_list if call.args[0].startswith("DKV stall")
    ]
    restarted = {message.split("request ")[1].split(" ")[0] for message in recoveries}
    assert restarted == {str(10 + group_size - 1)}, "only the last started request restarts"
    assert len(recoveries) == group_size, "every rank logs the same recovery once"
    owners = {10 + rank: rank for rank in range(group_size)}
    assert Counter(request_id for request_id, _ in runs[0].emitted) == Counter(
        dict.fromkeys(owners, 1)
    )
    assert all(run.final_pages == run.initial_pages for run in runs)
    assert all(run.pages.freed == runs[0].pages.freed for run in runs)


# The layer-split layout: the loop plans the data plane of an iteration once the pages are
# allocated, and drains it after S-sample, before the commits that may free a page.

_BUSY_LAYER_SPLIT = [
    "control",
    "schedule",
    "forward_batch",
    "prepare_resources",
    "plan",
    "forward",
    "sample_sync",
    "drain",
    "context_commit",
    "update_resources",
]


@pytest.mark.parametrize("group_size", [2, 4, 8])
def test_the_data_plane_is_planned_before_the_forward_and_drained_before_the_commits(
    group_size: int,
) -> None:
    arrivals = _arrivals(group_size, (2, 1, 0, 3, 2))
    runs = run_loop(group_size, LoopScript(arrivals, max_num_tokens=16, streamer=True))
    phases = [_phases(run) for run in runs]
    assert all(rank == phases[0] for rank in phases), "ranks entered different phases"
    assert any(names == _BUSY_LAYER_SPLIT for names in phases[0].values())
    for iteration, names in phases[0].items():
        assert names in (_BUSY_LAYER_SPLIT, _IDLE, ["control", "schedule"]), (iteration, names)
    # The plan is a function of the replicated batch: every rank derived the same one.
    fingerprints = [
        [(iteration, value) for iteration, name, value in run.events if name == "plan"]
        for run in runs
    ]
    assert fingerprints[0] and all(rank == fingerprints[0] for rank in fingerprints)
    assert all(run.final_pages == run.initial_pages for run in runs)


@pytest.mark.parametrize("group_size", [2, 4])
def test_an_idle_rank_is_in_the_plan_as_the_owner_of_its_layers(group_size: int) -> None:
    # Every request computes on rank 0, so every other rank runs its resident dummy.
    arrivals = [[(10, 0, 8)], [(11, 0, 8)], [], []]
    runs = run_loop(group_size, LoopScript(arrivals, max_num_tokens=16, streamer=True))
    for run in runs:
        dummies = {r.py_request_id for r in run.executor._dkv_forward_dummies[1:]}
        for plan in run.executor.dkv_streamer.plans:
            idle = [request for request in plan.requests if request.is_dummy]
            assert {request.request_id for request in idle} == dummies
            assert sorted(request.compute_rank for request in idle) == list(range(1, group_size))
            real = [request for request in plan.requests if not request.is_dummy]
            assert all(request.compute_rank == 0 for request in real)
            # Layers that another rank owns are fetched from it and written back to it.
            assert any(transfer.owner != 0 for step in plan.steps for transfer in step.transfers)


def test_the_drain_has_the_deadline_of_the_startup_settings() -> None:
    arrivals = _arrivals(2, (2, 1))
    runs = run_loop(2, LoopScript(arrivals, max_num_tokens=16, streamer=True))
    deadlines = {value for run in runs for _, name, value in run.events if name == "drain"}
    assert deadlines == {7.0}


def test_the_plan_check_names_the_iteration_where_the_plans_of_the_ranks_differ() -> None:
    arrivals = _arrivals(2, (2, 2, 1, 0))
    with pytest.raises(RuntimeError, match=r"iter 1, tags=.*data plane plan"):
        run_loop(
            2, LoopScript(arrivals, max_num_tokens=16, debug=True, streamer=True, plan_skew=(1, 1))
        )


def test_a_plan_that_differs_goes_unnoticed_without_the_debug_check() -> None:
    # Without the check the divergence shows as a hang of the data plane, which the drain and the
    # hang detector bound; the loop itself carries on.
    arrivals = _arrivals(2, (2, 2, 1, 0))
    runs = run_loop(
        2, LoopScript(arrivals, max_num_tokens=16, debug=False, streamer=True, plan_skew=(1, 1))
    )
    fingerprints = [[value for _, name, value in run.events if name == "plan"] for run in runs]
    assert fingerprints[0] != fingerprints[1]
