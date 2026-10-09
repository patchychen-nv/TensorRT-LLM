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
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal


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
    ttft_server_s: float | None = None
    queue_s: float | None = None
    submitted_at: float | None = None
    completed_at: float | None = None
    ttft_server_raw_s: float | None = None
    queue_raw_s: float | None = None
    server_timing_error: str | None = None


@dataclass(frozen=True)
class WorkloadCompletion:
    """One completed response, with an optional timestamp from its response consumer."""

    cached_tokens: int
    ttft_server_s: float | None = None
    queue_s: float | None = None
    completed_at: float | None = None
    ttft_server_raw_s: float | None = None
    queue_raw_s: float | None = None
    server_timing_error: str | None = None


_WARMUP_SEED_OFFSET = 1_000_003


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


def chat_workload(
    *,
    sessions: int,
    turns: tuple[int, int],
    first_tokens: Sequence[int],
    turn_tokens: Sequence[int],
    vocab: int,
    seed: int = 0,
    interleave: Literal["rounds", "random"] = "rounds",
) -> list[WorkloadRequest]:
    """Conversations of different length whose every turn re-sends the whole history.

    A session draws its first-turn length from ``first_tokens``, its per-turn growth from
    ``turn_tokens`` and its number of turns from the inclusive range ``turns``. ``rounds`` orders the
    requests turn by turn across the sessions, the order a closed-loop client submits them in;
    ``random`` interleaves the sessions at random and keeps the turns of one session in order.
    """
    low, high = turns
    if not first_tokens or not turn_tokens:
        raise ValueError("first_tokens and turn_tokens need at least one length")
    if min(sessions, low, *first_tokens, *turn_tokens) < 1 or high < low:
        raise ValueError("sessions, turns and token counts must be positive and turns ordered")
    rng = random.Random(seed)
    plans: list[list[list[int]]] = []
    for _ in range(sessions):
        history = _tokens(rng, rng.choice(first_tokens), vocab)
        growth = rng.choice(turn_tokens)
        plan = [list(history)]
        for _ in range(rng.randint(low, high) - 1):
            history = history + _tokens(rng, growth, vocab)
            plan.append(list(history))
        plans.append(plan)
    if interleave == "random":
        order = [session for session, plan in enumerate(plans) for _ in plan]
        rng.shuffle(order)
    else:
        order = [
            session
            for turn in range(max(map(len, plans)))
            for session, plan in enumerate(plans)
            if turn < len(plan)
        ]
    served = [0] * sessions
    requests = []
    for session in order:
        requests.append(
            WorkloadRequest(
                len(requests), session, plans[session][served[session]], served[session]
            )
        )
        served[session] += 1
    return requests


def history_hit_ceiling(requests: Sequence[WorkloadRequest]) -> int:
    """Prompt tokens a cache that keeps everything serves when each turn extends the previous one.

    Every request after the first of its session can match the whole prompt of the session's
    previous request, so a lifecycle visible to all ranks reaches exactly this many hits.
    """
    previous: dict[int, int] = {}
    total = 0
    for request in sorted(requests, key=lambda r: (r.session, r.turn)):
        total += previous.get(request.session, 0)
        previous[request.session] = len(request.prompt)
    return total


def prefix_bodies(requests: Sequence[WorkloadRequest], prefix_tokens: int) -> list[list[int]]:
    """The shared prefix of every Zipf session that occurs in ``requests``, in session order."""
    bodies: dict[int, list[int]] = {}
    for request in requests:
        bodies.setdefault(request.session, request.prompt[:prefix_tokens])
    return [bodies[session] for session in sorted(bodies)]


def warmup_prompts(*, count: int, tokens: int, vocab: int, seed: int = 0) -> list[list[int]]:
    """Random prompts for warming kernels up, drawn apart from the workload's own seed stream."""
    rng = random.Random(seed + _WARMUP_SEED_OFFSET)
    return [_tokens(rng, tokens, vocab) for _ in range(count)]


def summarize_results(
    results: Sequence[WorkloadResult],
    *,
    elapsed_seconds: float | None = None,
    computed_context_tokens: int | None = None,
) -> dict:
    """Hits, latency percentiles and throughput over the measured request intervals.

    ``computed_context_tokens`` comes from executor scheduling counters. The elapsed time excludes
    warm-up, priming and waits for the measurement streams to settle between intervals.
    """
    by_turn: dict[int, dict[str, int]] = {}
    for result in results:
        row = by_turn.setdefault(
            result.turn, {"requests": 0, "eligible_tokens": 0, "cached_tokens": 0}
        )
        row["requests"] += 1
        row["eligible_tokens"] += result.prompt_tokens - 1
        row["cached_tokens"] += result.cached_tokens
    eligible = sum(row["eligible_tokens"] for row in by_turn.values())
    cached = sum(row["cached_tokens"] for row in by_turn.values())

    def percentile(values: Sequence[float], fraction: float) -> float | None:
        ordered = sorted(values)
        return ordered[min(len(ordered) - 1, int(fraction * len(ordered)))] if ordered else None

    latencies = [result.latency_s for result in results]
    server = [result.ttft_server_s for result in results if result.ttft_server_s is not None]
    queues = [result.queue_s for result in results if result.queue_s is not None]
    timing_errors: dict[str, int] = {}
    for result in results:
        if result.server_timing_error is not None:
            reason = result.server_timing_error
            timing_errors[reason] = timing_errors.get(reason, 0) + 1
    duration = elapsed_seconds if elapsed_seconds and elapsed_seconds > 0 else None
    return {
        "requests": len(results),
        "prompt_tokens": sum(result.prompt_tokens for result in results),
        "eligible_tokens": eligible,
        "cached_tokens": cached,
        "cached_over_eligible": cached / eligible if eligible else None,
        "by_turn": by_turn,
        "requests_with_cached_tokens": sum(1 for result in results if result.cached_tokens > 0),
        **{f"latency_p{p}_s": percentile(latencies, p / 100) for p in (50, 90, 99)},
        **{f"ttft_client_p{p}_s": percentile(latencies, p / 100) for p in (50, 90, 99)},
        **{f"ttft_server_p{p}_s": percentile(server, p / 100) for p in (50, 90, 99)},
        "queue_p99_s": percentile(queues, 0.99),
        "server_timing_requests": len(server),
        "server_timing_complete": bool(results) and len(server) == len(results),
        "server_timing_errors": timing_errors,
        "ttft_source": "server" if results and len(server) == len(results) else "client",
        "measured_seconds": elapsed_seconds,
        "computed_context_tokens": computed_context_tokens,
        "requests_per_second": len(results) / duration if duration else None,
        "computed_context_tokens_per_second": (
            computed_context_tokens / duration
            if duration and computed_context_tokens is not None
            else None
        ),
    }


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
    submit: Callable[[WorkloadRequest], Callable[[], int | WorkloadCompletion | None]],
    requests: Sequence[WorkloadRequest],
    *,
    concurrency: int,
    completed_turns: Mapping[int, int] | None = None,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    idle: Callable[[], None] | None = None,
) -> list[WorkloadResult]:
    """Run a workload with at most ``concurrency`` requests in flight.

    ``submit(request)`` starts a request and returns a poll callable: ``None`` means still pending,
    a ``WorkloadCompletion`` gives the response metrics. A legacy blocking callable returning only
    cached tokens is also accepted, but cannot measure out-of-order completion. A request waits for
    the previous turn of its session
    to finish and, when it has an arrival offset, for that offset to pass; offsets count from the
    start of each call. ``completed_turns`` maps a session to the number of its turns an earlier call
    already served, so a closed-loop workload can be run in several consecutive parts.
    """
    if concurrency < 1:
        raise ValueError("concurrency must be positive")
    started = clock()
    pending = sorted(requests, key=lambda request: (request.arrival_s, request.index))
    in_flight: list[
        tuple[WorkloadRequest, float, Callable[[], int | WorkloadCompletion | None]]
    ] = []
    done_turns: dict[int, int] = dict(completed_turns or {})
    results: dict[int, WorkloadResult] = {}

    def finish(entry, completion: WorkloadCompletion) -> None:
        request, submitted, _ = entry
        finished = completion.completed_at if completion.completed_at is not None else clock()
        latency = finished - submitted
        ttft, queue = completion.ttft_server_s, completion.queue_s
        timing_error = completion.server_timing_error
        if ttft is not None and ttft > latency:
            ttft, queue = None, None
            timing_error = "server_ttft_exceeds_client"
        results[request.index] = WorkloadResult(
            request.index,
            request.session,
            request.turn,
            len(request.prompt),
            completion.cached_tokens,
            latency,
            ttft,
            queue,
            submitted,
            finished,
            completion.ttft_server_raw_s,
            completion.queue_raw_s,
            timing_error,
        )
        done_turns[request.session] = max(done_turns.get(request.session, 0), request.turn + 1)

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
            completed = []
            for entry in in_flight:
                response = entry[2]()
                if response is not None:
                    if isinstance(response, int):
                        response = WorkloadCompletion(response)
                    finish(entry, response)
                    completed.append(entry)
            for entry in completed:
                in_flight.remove(entry)
            if idle is not None:
                idle()
            if not completed:
                sleep(0.001)
        else:
            wait = min(request.arrival_s for request in pending) - (clock() - started)
            if wait <= 0:
                raise RuntimeError("The workload is blocked on turns that were never submitted")
            sleep(wait)
    return [results[index] for index in sorted(results)]
