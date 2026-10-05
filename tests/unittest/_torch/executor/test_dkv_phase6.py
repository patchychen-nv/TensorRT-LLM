# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The parts of the Phase 6 runner that need no GPU: options, waiting for the streams, tables."""

import dataclasses
import importlib.util
import json
import sys
from pathlib import Path

import pytest

_DKV_DIR = Path(__file__).resolve().parents[3] / "integration/defs/dkv"
sys.path.insert(0, str(_DKV_DIR))
try:
    _SPEC = importlib.util.spec_from_file_location(
        "dkv_phase6_runner_under_test", _DKV_DIR / "dkv_phase6_runner.py"
    )
    assert _SPEC is not None and _SPEC.loader is not None
    _RUNNER = importlib.util.module_from_spec(_SPEC)
    _SPEC.loader.exec_module(_RUNNER)
finally:
    sys.path.remove(str(_DKV_DIR))
_TABLES = _RUNNER.tables

pytestmark = pytest.mark.cpu_only


def _options(**changes):
    return dataclasses.replace(_RUNNER.Phase6Options(model="model", mode="dkv"), **changes)


def test_modes_are_named_after_their_router() -> None:
    assert _RUNNER.parse_mode("dkv") == _RUNNER.Mode("dkv", True, None)
    assert _RUNNER.parse_mode("adp") == _RUNNER.Mode("adp", False, None)
    assert _RUNNER.parse_mode("adp_kv:0.25") == _RUNNER.Mode("adp_kv_b0.25", False, 0.25)
    for spec in ("kv", "adp_kv", "adp_kv:x", ""):
        with pytest.raises(ValueError):
            _RUNNER.parse_mode(spec)


def test_the_capacity_limit_is_either_a_token_count_or_a_quota() -> None:
    by_tokens = _RUNNER.kv_cache_options(_options(kv_max_tokens=1000))
    assert by_tokens == {
        "enable_partial_reuse": True,
        "event_buffer_max_size": 262144,
        "max_tokens": 1000,
    }
    by_quota = _RUNNER.kv_cache_options(_options(kv_quota_gib=1.5, kv_max_tokens=1000))
    assert by_quota["max_tokens"] is None
    assert by_quota["max_gpu_total_bytes"] == 3 << 29


def test_options_survive_a_round_trip_through_json() -> None:
    options = _options(workload="chat", interleave="random", kv_quota_gib=2.0, prime=True)
    again = _RUNNER.Phase6Options(**json.loads(json.dumps(dataclasses.asdict(options))))
    assert again == options


def test_requests_and_priming_prompts_follow_the_workload_options() -> None:
    zipf = _options(
        requests=40, prefixes=4, prefix_blocks=2, tokens_per_block=8, suffix_tokens=3, prime=True
    )
    requests = _RUNNER.build_requests(zipf, vocab=100)
    assert len(requests) == 40 and all(len(request.prompt) == 19 for request in requests)
    primers = _RUNNER.priming_prompts(zipf, requests)
    assert 1 < len(primers) <= 4 and all(len(prompt) == 16 for prompt in primers)
    assert _RUNNER.priming_prompts(dataclasses.replace(zipf, prime=False), requests) == []
    chat = _options(workload="chat", sessions=5, turns_min=2, turns_max=3, first_tokens="6,9")
    requests = _RUNNER.build_requests(chat, vocab=100)
    assert {request.session for request in requests} == set(range(5))
    assert _RUNNER.priming_prompts(dataclasses.replace(chat, prime=True), requests) == []


def test_the_hit_ceiling_is_what_a_cache_every_rank_can_read_serves() -> None:
    chat = _options(workload="chat", sessions=5, turns_min=2, turns_max=4, first_tokens="6,9")
    requests = _RUNNER.build_requests(chat, vocab=100)
    previous: dict[int, int] = {}
    expected = 0
    for request in requests:  # rounds order: the turns of a session are in order
        expected += previous.get(request.session, 0)
        previous[request.session] = len(request.prompt)
    assert _RUNNER.hit_ceiling(chat, requests) == expected > 0
    zipf = _options(requests=30, prefixes=3, prefix_blocks=2, tokens_per_block=8, suffix_tokens=2)
    requests = _RUNNER.build_requests(zipf, vocab=100)
    distinct = len({request.session for request in requests})
    assert _RUNNER.hit_ceiling(zipf, requests) == (30 - distinct) * 16
    primed = dataclasses.replace(zipf, prime=True)
    assert _RUNNER.hit_ceiling(primed, requests) == 30 * 16


class _Clock:
    """Time moves only when something waits, so the polling loops end deterministically."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


class _Streams:
    """An LLM whose iteration stats count the requests it has been told about."""

    def __init__(self, clock: _Clock, group: int) -> None:
        self.clock = clock
        self.group = group
        self.counted = [0] * group
        self.events: list[list[dict]] = []
        self.iteration = 0

    def get_kv_cache_events(self, timeout: float) -> list[dict]:
        self.clock.sleep(timeout)
        return self.events.pop(0) if self.events else []

    def get_stats(self, timeout: float) -> list[dict]:
        self.clock.sleep(timeout)
        self.iteration += 1
        return [
            {
                "iter": self.iteration,
                "attentionDpRank": rank,
                "dkvMeasurement": {"rank": rank, "counters": {"request_count": count}},
            }
            for rank, count in enumerate(self.counted)
        ]


def _observer(tmp_path: Path, streams: _Streams):
    return _RUNNER.Observer(
        streams, streams.group, tmp_path, clock=streams.clock, sleep=streams.clock.sleep
    )


def test_settling_waits_for_every_request_to_be_counted_and_for_quiet_events(
    tmp_path: Path,
) -> None:
    streams = _Streams(_Clock(), group=2)
    streams.counted = [1, 0]
    streams.events = [[{"event_id": 0}], [{"event_id": 1}]]
    observer = _observer(tmp_path, streams)
    observer.pull()
    streams.counted = [2, 1]
    snapshots = observer.settle(3)
    assert [snapshot["counters"]["request_count"] for snapshot in snapshots] == [2, 1]
    assert [event["event_id"] for event in observer.events] == [0, 1]
    assert len((tmp_path / "events.jsonl").read_text().splitlines()) == 2
    assert (tmp_path / "iteration-stats.jsonl").exists()
    assert streams.clock.now >= 1.0


def test_settling_fails_when_requests_are_missing_or_too_many(tmp_path: Path) -> None:
    streams = _Streams(_Clock(), group=2)
    streams.counted = [1, 1]
    with pytest.raises(TimeoutError, match="2 of 3"):
        _observer(tmp_path, streams).settle(3, deadline_s=5.0)
    with pytest.raises(RuntimeError, match="counted 2"):
        _observer(tmp_path, streams).settle(1)


def test_an_event_stream_that_never_goes_quiet_is_reported(tmp_path: Path) -> None:
    streams = _Streams(_Clock(), group=1)
    streams.events = [[{"event_id": index}] for index in range(10_000)]
    with pytest.raises(TimeoutError, match="idle"):
        _observer(tmp_path, streams).drain(deadline_s=3.0)


class _Future:
    def __init__(self, timeouts: int) -> None:
        self.timeouts = timeouts

    def result(self, timeout: float):
        if self.timeouts:
            self.timeouts -= 1
            raise TimeoutError
        return type("Done", (), {"cached_tokens": 7})()


def test_waiting_for_a_request_keeps_draining_the_streams(tmp_path: Path) -> None:
    streams = _Streams(_Clock(), group=1)
    observer = _observer(tmp_path, streams)
    assert _RUNNER._await_cached_tokens(_Future(2), observer, 60.0) == 7
    assert streams.iteration == 2
    with pytest.raises(TimeoutError):
        _RUNNER._await_cached_tokens(_Future(10**9), observer, 0.0)


def _report(
    *,
    hit: float = 0.5,
    matched: int = 50,
    scheduled: int = 50,
    reasons: list[str] | None = None,
    drops: int = 0,
    removed: int = 0,
    requests: int = 4,
    complete: bool = True,
    by_rank: tuple[int, ...] = (2, 2),
) -> dict:
    return {
        "global": {
            "request_count": requests,
            "token_prefix_hit_rate": hit,
            "request_prefix_hit_rate": hit,
            "matched_prefix_tokens": matched,
            "scheduled_context_tokens": scheduled,
        },
        "load": {
            "request_count_by_rank": list(by_rank),
            "scheduled_context_tokens_by_rank": [scheduled // 2] * 2,
            "per_iteration_token_variance": 4.0,
            "per_iteration_token_imbalance": 2.0,
            "iterations_measured": 3,
        },
        "storage": {
            "complete": complete,
            "duplicate_storage_ratio": 0.5,
            "removed_block_copies": removed,
            "last_tier_capacity_dropped_pages": drops,
        },
        "validity": {"invalid_reasons": reasons or []},
        "capacity_comparison": None,
        "capacity_by_rank": [
            {
                "pools_by_level": [
                    [
                        {
                            "slot_sizes": [8],
                            "total": 100,
                            "free": 90,
                            "evictable": 6,
                            "available": 96,
                        },
                        {
                            "slot_sizes": [64],
                            "total": 20,
                            "free": 10,
                            "evictable": 6,
                            "available": 16,
                        },
                    ]
                ]
            }
        ],
    }


def _result(name: str, beta: float | None = None, **report) -> dict:
    options = {
        "workload": "zipf",
        "group": 2,
        "requests": 4,
        "prefixes": 2,
        "alpha": 1.0,
        "prefix_blocks": 2,
        "tokens_per_block": 8,
        "suffix_tokens": 3,
        "concurrency": 4,
        "warmup": 2,
        "kv_max_tokens": 1000,
        "max_batch_size": 2,
        "max_num_tokens": 64,
    }
    body = _report(**report)
    return {
        "mode": {"name": name, "dkv": name == "dkv", "beta": beta},
        "options": options,
        "workload": {"requests": 4, "hit_ceiling_tokens": 90},
        "phases": {"phase1": {"report": body}},
        "report": body,
        "client": {
            "cached_over_eligible": 0.5,
            "eligible_tokens": 100,
            "cached_tokens": report.get("matched", 50),
            "by_turn": {"0": {"cached_tokens": 0, "eligible_tokens": 10}},
            "latency_p50_s": 0.1,
            "latency_p99_s": 0.2,
        },
        "build_seconds": 10.0,
        "wall_seconds": 15.0,
    }


def test_removals_without_capacity_drops_are_labelled_apart() -> None:
    only = ["removals_without_verified_equal_capacity"]
    assert _TABLES.gate_label(_report()) == "yes"
    assert _TABLES.gate_label(_report(reasons=only, removed=3)) == (
        "NO (removals only, no capacity drops)"
    )
    assert _TABLES.gate_label(_report(reasons=only, drops=2)) == "NO"
    assert _TABLES.gate_label(_report(reasons=[*only, "incomplete_event_observation"])) == "NO"


def test_modes_are_ordered_adp_then_kv_aware_by_weight_then_dkv() -> None:
    results = [_result("dkv"), _result("adp_kv_b4", 4.0), _result("adp"), _result("adp_kv_b1", 1.0)]
    names = [result["mode"]["name"] for result in sorted(results, key=_TABLES.mode_order)]
    assert names == ["adp", "adp_kv_b1", "adp_kv_b4", "dkv"]


def test_tables_compare_every_mode_with_the_adp_baseline() -> None:
    text = _TABLES.comparison_tables(
        [_result("adp", hit=0.2, matched=20, scheduled=80), _result("dkv", scheduled=40)]
    )
    assert "| adp | yes | 0.200 |" in text
    assert "| dkv | yes | 0.500 |" in text
    assert "| 50 | 40 | 0.500 | 0.500 |" in text  # the context tokens saved: 1 - 40 / 80
    assert "requests by rank" in text and "[2, 2]" in text
    assert "G=2, 4 requests over 2 Zipf" in text
    assert "90 matched tokens, 0.900 of the eligible tokens" in text


def test_the_seed_summary_has_a_column_per_label() -> None:
    text = _TABLES.seed_summary(
        {
            "seed0": [_result("adp", hit=0.2), _result("dkv", hit=0.9)],
            "seed1": [_result("adp", hit=0.3), _result("dkv", hit=0.8)],
        }
    )
    assert "| mode | seed0 | seed1 |" in text
    assert "| adp | 0.200 | 0.300 |" in text and "| dkv | 0.900 | 0.800 |" in text
    assert "requests per rank, max / min" in text


def test_the_capacity_table_lists_pages_hits_and_the_equal_capacity_verdict() -> None:
    adp = _result("adp", hit=0.2, matched=18, drops=7)
    dkv = _result("dkv", hit=0.45, matched=45, drops=11)
    dkv["options"]["kv_quota_gib"] = 4.0
    dkv["report"]["capacity_comparison"] = {
        "equal_usable_capacity": True,
        "tolerance_pages": 8,
        "pools": [{"usable_gap_pages": -6}, {"usable_gap_pages": 2}],
    }
    text = _TABLES.capacity_table({"equal capacity": [adp, dkv]})
    assert "| equal capacity | adp | n/a | 100 / 20 | 0.200 | 0.200 | 7 | 1.00 |" in text
    assert "| equal capacity | dkv | 4.000 | 100 / 20 | 0.450 | 0.500 | 11 | 1.00 |" in text
    assert "True (gaps [-6, 2], tolerance 8)" in text
    plain = _TABLES.capacity_table({"a": [_result("adp")], "b": [_result("dkv")]})
    assert plain.count("\n| ") >= 3 and "| - |" in plain


@pytest.mark.parametrize(
    ("report", "problem"),
    [
        (dict(requests=3), "request count"),
        (dict(complete=False), "event stream"),
        (dict(matched=40), "client and counter hits differ"),
    ],
)
def test_a_run_whose_counts_do_not_add_up_is_flagged(report: dict, problem: str) -> None:
    result = _result("dkv")
    result["report"] = _report(**report)
    result["client"]["cached_tokens"] = 50
    (line,) = _TABLES.check_runs([result])
    assert "INCOMPLETE" in line and problem in line
    (clean,) = _TABLES.check_runs([_result("dkv")])
    assert "INCOMPLETE" not in clean


def test_the_report_commands_read_a_results_directory(tmp_path: Path, capfd) -> None:
    for name in ("adp", "dkv"):
        directory = tmp_path / name
        directory.mkdir()
        (directory / "result.json").write_text(json.dumps(_result(name)))
    assert _RUNNER.main(["report", str(tmp_path)]) == 0
    assert "| dkv | yes |" in capfd.readouterr().out
    assert _RUNNER.main(["check", str(tmp_path)]) == 0
    assert _RUNNER.main(["summary", f"a={tmp_path}", f"b={tmp_path}"]) == 0
    assert _RUNNER.main(["capacity", f"a={tmp_path}"]) == 0
    assert "| a | dkv |" in capfd.readouterr().out
    broken = _result("dkv")
    broken["report"] = _report(requests=1)
    (tmp_path / "dkv" / "result.json").write_text(json.dumps(broken))
    assert _RUNNER.main(["check", str(tmp_path)]) == 1


def test_the_command_line_rejects_an_unknown_mode(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        _RUNNER.main(["run", "--model", "m", "--out", str(tmp_path), "--modes", "adp,nope"])
    with pytest.raises(SystemExit):
        _RUNNER.main(["worker", "--out", str(tmp_path)])


def test_a_worker_takes_its_options_from_a_file_or_from_flags(tmp_path: Path, monkeypatch) -> None:
    served = []
    monkeypatch.setattr(_RUNNER, "run_mode", lambda options, out: served.append((options, out)))
    options = _options(workload="chat", group=8, kv_quota_gib=3.0, mode="adp_kv:1")
    path = tmp_path / "options.json"
    path.write_text(json.dumps(dataclasses.asdict(options)))
    assert _RUNNER.main(["worker", "--options", str(path), "--out", str(tmp_path / "a")]) == 0
    flags = ["--model", "model", "--mode", "adp_kv:1", "--group", "8", "--workload", "chat"]
    flags += ["--kv-quota-gib", "3.0"]
    assert _RUNNER.main(["worker", *flags, "--out", str(tmp_path / "b")]) == 0
    assert [out for _, out in served] == [tmp_path / "a", tmp_path / "b"]
    assert served[0][0] == served[1][0] == options
