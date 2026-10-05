# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reuse workloads are deterministic and honor prefix sharing, turn order and concurrency."""

import importlib.util
from collections import Counter
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parents[3] / "integration/defs/dkv/dkv_workloads.py"
_SPEC = importlib.util.spec_from_file_location("dkv_workloads_under_test", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
_WORKLOADS = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_WORKLOADS)

pytestmark = pytest.mark.cpu_only

_ZIPF = dict(prefixes=4, alpha=1.2, prefix_tokens=64, suffix_tokens=8, vocab=1000)


def test_zipf_workload_is_deterministic_and_popular_prefixes_dominate() -> None:
    first = _WORKLOADS.zipf_prefix_workload(400, seed=3, **_ZIPF)
    again = _WORKLOADS.zipf_prefix_workload(400, seed=3, **_ZIPF)
    other = _WORKLOADS.zipf_prefix_workload(400, seed=4, **_ZIPF)
    assert first == again and first != other
    popularity = Counter(request.session for request in first)
    assert popularity[0] > popularity[1] > popularity[3]
    assert sum(popularity.values()) == 400


def test_zipf_requests_share_only_their_prefix() -> None:
    requests = _WORKLOADS.zipf_prefix_workload(60, seed=1, **_ZIPF)
    by_session: dict[int, list[list[int]]] = {}
    for request in requests:
        assert len(request.prompt) == 72
        by_session.setdefault(request.session, []).append(request.prompt)
    for prompts in by_session.values():
        assert len({tuple(prompt[:64]) for prompt in prompts}) == 1
        assert len({tuple(prompt[64:]) for prompt in prompts}) == len(prompts)
    assert len({tuple(prompts[0][:64]) for prompts in by_session.values()}) == len(by_session)
    assert all(token >= 2 for request in requests for token in request.prompt)


def test_multi_turn_prompts_extend_the_previous_turn_of_their_session() -> None:
    requests = _WORKLOADS.multi_turn_workload(
        sessions=3, turns=4, first_tokens=20, turn_tokens=5, vocab=500, seed=2
    )
    assert [request.turn for request in requests[:6]] == [0, 0, 0, 1, 1, 1]
    last: dict[int, list[int]] = {}
    for request in requests:
        assert len(request.prompt) == 20 + 5 * request.turn
        if request.session in last:
            assert request.prompt[: len(last[request.session])] == last[request.session]
            assert request.prompt != last[request.session]
        last[request.session] = request.prompt
    assert len({tuple(prompt[:20]) for prompt in last.values()}) == 3


def test_trace_workload_orders_by_arrival_and_counts_turns_per_session() -> None:
    rows = [
        {"arrival_s": 2.0, "session": 7, "prompt": [1, 2]},
        {"arrival_s": 0.5, "session": 7, "prompt": [1]},
        {"arrival_s": 1.0, "session": 9, "prompt": [5]},
    ]
    requests = _WORKLOADS.trace_workload(rows)
    assert [(r.session, r.turn, r.arrival_s) for r in requests] == [
        (7, 0, 0.5),
        (9, 0, 1.0),
        (7, 1, 2.0),
    ]


class _Clock:
    """A clock that only advances when the runner sleeps, so arrival times are deterministic."""

    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


def test_run_workload_bounds_concurrency_and_keeps_turn_order() -> None:
    requests = _WORKLOADS.multi_turn_workload(
        sessions=3, turns=3, first_tokens=4, turn_tokens=2, vocab=100, seed=0
    )
    events: list[tuple[str, int]] = []
    in_flight = 0
    peak = 0

    def submit(request):
        nonlocal in_flight, peak
        events.append(("start", request.index))
        in_flight += 1
        peak = max(peak, in_flight)

        def wait() -> int:
            nonlocal in_flight
            events.append(("end", request.index))
            in_flight -= 1
            return request.turn * 2

        return wait

    clock = _Clock()
    results = _WORKLOADS.run_workload(
        submit, requests, concurrency=2, clock=clock, sleep=clock.sleep
    )
    assert [result.index for result in results] == list(range(9))
    assert peak == 2
    assert [result.cached_tokens for result in results] == [r.turn * 2 for r in requests]
    ended = {index: position for position, (kind, index) in enumerate(events) if kind == "end"}
    started = {index: position for position, (kind, index) in enumerate(events) if kind == "start"}
    for request in requests:
        if request.turn:
            previous = next(
                r for r in requests if r.session == request.session and r.turn == request.turn - 1
            )
            assert ended[previous.index] < started[request.index]


def test_run_workload_waits_for_arrival_offsets() -> None:
    requests = _WORKLOADS.trace_workload(
        [{"arrival_s": 0.0, "prompt": [1]}, {"arrival_s": 5.0, "prompt": [2]}]
    )
    clock = _Clock()
    results = _WORKLOADS.run_workload(
        lambda request: (lambda: 0), requests, concurrency=4, clock=clock, sleep=clock.sleep
    )
    assert clock.sleeps == [5.0]
    assert len(results) == 2


def test_run_workload_rejects_a_workload_that_can_never_make_progress() -> None:
    orphan = _WORKLOADS.WorkloadRequest(0, session=0, prompt=[1], turn=1)
    with pytest.raises(RuntimeError, match="blocked"):
        _WORKLOADS.run_workload(lambda request: (lambda: 0), [orphan], concurrency=1)
    with pytest.raises(ValueError, match="concurrency"):
        _WORKLOADS.run_workload(lambda request: (lambda: 0), [], concurrency=0)


@pytest.mark.parametrize(
    "arguments",
    [
        dict(count=0, prefixes=1, alpha=1.0, prefix_tokens=1, suffix_tokens=1, vocab=10),
        dict(count=1, prefixes=1, alpha=-1.0, prefix_tokens=1, suffix_tokens=1, vocab=10),
    ],
)
def test_zipf_workload_rejects_degenerate_arguments(arguments: dict) -> None:
    with pytest.raises(ValueError):
        _WORKLOADS.zipf_prefix_workload(**arguments)
