# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Deterministic reuse workloads: Zipf-shared prefixes, multi-turn sessions and trace replay.

The generators work on token ids, so they do not depend on a tokenizer, and every generator is a
pure function of its arguments and seed: two runs of an experiment, ADP and DKV, submit identical
requests.
"""

import bisect
import itertools
import random
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class WorkloadRequest:
    """One request of a workload.

    ``session`` names the prefix group (Zipf) or the conversation (multi-turn); ``turn`` orders
    the requests of one conversation, whose later turns must wait for the earlier ones.
    ``arrival_s`` is the offset from the start of the run for trace replay, 0 for closed-loop runs.
    """

    index: int
    session: int
    prompt: list[int]
    turn: int = 0
    arrival_s: float = 0.0


@dataclass(frozen=True)
class WorkloadResult:
    index: int
    session: int
    turn: int
    prompt_tokens: int
    cached_tokens: int
    latency_s: float


def _tokens(rng: random.Random, count: int, vocab: int) -> list[int]:
    # Token 0 and 1 are commonly special; stay clear of them.
    return [rng.randrange(2, vocab) for _ in range(count)]


def zipf_prefix_workload(
    count: int,
    *,
    prefixes: int,
    alpha: float,
    prefix_tokens: int,
    suffix_tokens: int,
    vocab: int,
    seed: int = 0,
) -> list[WorkloadRequest]:
    """Requests that share one of ``prefixes`` fixed prefixes, popular ones far more often.

    Prefix ``k`` (from 0) is chosen with weight ``1 / (k + 1) ** alpha``. Every request has its own
    random suffix, so only the prefix can be reused.
    """
    if min(count, prefixes, prefix_tokens, suffix_tokens) < 1 or alpha < 0:
        raise ValueError("count, prefixes and token counts must be positive and alpha non-negative")
    rng = random.Random(seed)
    bodies = [_tokens(rng, prefix_tokens, vocab) for _ in range(prefixes)]
    cumulative = list(itertools.accumulate((k + 1) ** -alpha for k in range(prefixes)))
    requests = []
    for index in range(count):
        session = bisect.bisect_left(cumulative, rng.random() * cumulative[-1])
        prompt = bodies[session] + _tokens(rng, suffix_tokens, vocab)
        requests.append(WorkloadRequest(index, session, prompt))
    return requests


def multi_turn_workload(
    *,
    sessions: int,
    turns: int,
    first_tokens: int,
    turn_tokens: int,
    vocab: int,
    seed: int = 0,
) -> list[WorkloadRequest]:
    """Conversations whose every turn re-sends the whole history and adds new tokens.

    Turn ``t`` of a session starts with the complete prompt of turn ``t - 1``. Requests are ordered
    turn by turn across the sessions, which is the order a closed-loop client submits them in.
    """
    if min(sessions, turns, first_tokens, turn_tokens) < 1:
        raise ValueError("sessions, turns and token counts must be positive")
    rng = random.Random(seed)
    history = [_tokens(rng, first_tokens, vocab) for _ in range(sessions)]
    requests = []
    for turn in range(turns):
        for session in range(sessions):
            if turn:
                history[session] = history[session] + _tokens(rng, turn_tokens, vocab)
            requests.append(WorkloadRequest(len(requests), session, list(history[session]), turn))
    return requests


def trace_workload(rows: Sequence[dict]) -> list[WorkloadRequest]:
    """Requests from recorded rows ``{"arrival_s", "session", "prompt"}``, ordered by arrival."""
    ordered = sorted(rows, key=lambda row: row["arrival_s"])
    turns: dict[int, int] = {}
    requests = []
    for index, row in enumerate(ordered):
        session = row.get("session", index)
        turn = turns.get(session, 0)
        turns[session] = turn + 1
        requests.append(
            WorkloadRequest(index, session, list(row["prompt"]), turn, float(row["arrival_s"]))
        )
    return requests


def run_workload(
    submit: Callable[[WorkloadRequest], Callable[[], int]],
    requests: Sequence[WorkloadRequest],
    *,
    concurrency: int,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> list[WorkloadResult]:
    """Run a workload with at most ``concurrency`` requests in flight.

    ``submit(request)`` starts a request and returns a callable that waits for it and returns the
    number of prompt tokens served from cache. A request waits for the previous turn of its session
    to finish and, when it has an arrival offset, for that offset to pass.
    """
    if concurrency < 1:
        raise ValueError("concurrency must be positive")
    started = clock()
    pending = sorted(requests, key=lambda request: (request.arrival_s, request.index))
    in_flight: list[tuple[WorkloadRequest, float, Callable[[], int]]] = []
    done_turns: dict[int, int] = {}
    results: dict[int, WorkloadResult] = {}

    def finish(entry) -> None:
        request, submitted, wait = entry
        cached = wait()
        results[request.index] = WorkloadResult(
            request.index,
            request.session,
            request.turn,
            len(request.prompt),
            cached,
            clock() - submitted,
        )
        done_turns[request.session] = request.turn + 1

    while pending or in_flight:
        ready = next(
            (
                request
                for request in pending
                if done_turns.get(request.session, 0) >= request.turn
                and clock() - started >= request.arrival_s
            ),
            None,
        )
        if ready is not None and len(in_flight) < concurrency:
            pending.remove(ready)
            in_flight.append((ready, clock(), submit(ready)))
        elif in_flight:
            finish(in_flight.pop(0))
        else:
            wait = min(request.arrival_s for request in pending) - (clock() - started)
            if wait <= 0:
                raise RuntimeError("The workload is blocked on turns that were never submitted")
            sleep(wait)
    return [results[index] for index in sorted(results)]
