# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tables and run checks over the results written by ``dkv_phase6_runner``.

Everything here works on the ``result.json`` dictionaries, so it needs neither a GPU nor the
runtime. A directory of results holds one sub-directory per mode with a ``result.json``.
"""

import json
import statistics
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path

REMOVALS_ONLY = ["removals_without_verified_equal_capacity"]


def mode_order(result: Mapping) -> tuple[int, float]:
    """Sort key: ADP first, then the KV-aware routers by their weight, DKV last."""
    mode = result["mode"]
    if mode["name"] == "adp":
        return (0, 0.0)
    if mode["beta"] is not None:
        return (1, mode["beta"])
    return (2, 0.0)


def load_results(directories: Iterable[Path]) -> list[dict]:
    """Every ``<directory>/<mode>/result.json``, in the order of :func:`mode_order`."""
    results = [
        json.loads(path.read_text())
        for directory in directories
        for path in sorted(Path(directory).glob("*/result.json"))
    ]
    return sorted(results, key=mode_order)


def gate_label(report: Mapping) -> str:
    """Whether the hit-rate comparison is accepted, and why not when it is not.

    A report that is invalid only because blocks were removed, while no page was dropped for lack
    of capacity, is the usual picture of multi-turn traffic: extending a conversation replaces the
    partial block that ended the previous turn. It is labelled apart so the reader can judge it.
    """
    reasons = report["validity"]["invalid_reasons"]
    if not reasons:
        return "yes"
    if reasons == REMOVALS_ONLY and report["storage"].get("last_tier_capacity_dropped_pages") == 0:
        return "NO (removals only, no capacity drops)"
    return "NO"


def _number(value: float | int | None, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _cv(values: Sequence[int]) -> float | None:
    if not values or statistics.mean(values) == 0:
        return None
    return statistics.pstdev(values) / statistics.mean(values)


def _table(head: Sequence[str], rows: Iterable[Sequence[str]]) -> list[str]:
    lines = ["| " + " | ".join(head) + " |", "|" + " --- |" * len(head)]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    lines.append("")
    return lines


def _workload_line(options: Mapping) -> str:
    if options.get("workload", "zipf") == "chat":
        shape = (
            f"{options['sessions']} conversations of {options['turns_min']}-{options['turns_max']} "
            f"turns, first turn {options['first_tokens']} tokens, growth {options['turn_tokens']} "
            f"tokens per turn, {options.get('interleave', 'rounds')} order"
        )
    else:
        primed = ", primed" if options.get("prime") else ""
        shape = (
            f"{options['requests']} requests over {options['prefixes']} Zipf(alpha={options['alpha']})"
            f" prefixes of {options['prefix_blocks']} blocks x {options['tokens_per_block']} tokens, "
            f"suffix {options['suffix_tokens']} tokens{primed}"
        )
    quota = (
        f"quota {options['kv_quota_gib']} GiB"
        if options.get("kv_quota_gib")
        else f"max_tokens {options['kv_max_tokens']}"
    )
    return (
        f"G={options['group']}, {shape}, concurrency {options['concurrency']}, "
        f"warm-up {options['warmup']}, KV {quota}, batch {options['max_batch_size']}, "
        f"max_num_tokens {options['max_num_tokens']}"
    )


def comparison_tables(results: Sequence[Mapping]) -> str:
    """Markdown tables for the runs of one workload: hits, load, duplication, validity."""
    if not results:
        return "no results\n"
    adp = next((r for r in results if r["mode"]["name"] == "adp"), None)
    adp_scheduled = (
        adp["report"]["global"]["scheduled_context_tokens"]
        if adp is not None and "global" in adp["report"]
        else None
    )
    phases = sorted(results[0]["phases"])
    lines = [_workload_line(results[0]["options"]), ""]
    ceiling = results[0]["workload"].get("hit_ceiling_tokens")
    eligible = results[0]["client"].get("eligible_tokens")
    if ceiling is not None and eligible:
        lines += [
            f"Hit ceiling of a cache that keeps everything and every rank can read: {ceiling} "
            f"matched tokens, {ceiling / eligible:.3f} of the eligible tokens.",
            "",
        ]
    rows = []
    for result in results:
        report = result["report"]
        name = result["mode"]["name"]
        if "global" not in report:
            rows.append([name, f"report error: {report.get('report_error')}"])
            continue
        counters = report["global"]
        saved = None
        if adp_scheduled and name != "adp":
            saved = 1 - counters["scheduled_context_tokens"] / adp_scheduled
        rows.append(
            [
                name,
                gate_label(report),
                _number(counters["token_prefix_hit_rate"]),
                *[
                    _number(
                        result["phases"][p]["report"].get("global", {}).get("token_prefix_hit_rate")
                    )
                    for p in phases
                ],
                _number(counters["request_prefix_hit_rate"]),
                _number(counters["matched_prefix_tokens"]),
                _number(counters["scheduled_context_tokens"]),
                _number(saved),
                _number(result["client"]["cached_over_eligible"]),
            ]
        )
    lines += _table(
        [
            "mode",
            "valid",
            "token hit rate",
            *[f"hit {p}" for p in phases],
            "request hit rate",
            "matched tokens",
            "scheduled ctx tokens",
            "ctx saved vs adp",
            "client cached/eligible",
        ],
        rows,
    )
    rows = []
    for result in results:
        report = result["report"]
        if "load" not in report:
            continue
        load = report["load"]
        rows.append(
            [
                result["mode"]["name"],
                str(load["request_count_by_rank"]),
                str(load["scheduled_context_tokens_by_rank"]),
                _number(_cv(load["scheduled_context_tokens_by_rank"])),
                _number(load["per_iteration_token_variance"], 0),
                _number(load["per_iteration_token_imbalance"], 0),
                _number(load["iterations_measured"]),
                _number(report["storage"].get("duplicate_storage_ratio")),
                _number(report["storage"].get("removed_block_copies")),
                _number(report["storage"].get("last_tier_capacity_dropped_pages")),
            ]
        )
    lines += _table(
        [
            "mode",
            "requests by rank",
            "scheduled ctx tokens by rank",
            "CV of ctx tokens",
            "per-iteration token variance",
            "per-iteration imbalance",
            "iterations",
            "duplicate storage ratio",
            "removed block copies",
            "capacity-dropped pages",
        ],
        rows,
    )
    turns = sorted({int(turn) for r in results for turn in r["client"].get("by_turn", {})})
    if len(turns) > 1:
        rows = []
        for result in results:
            by_turn = result["client"]["by_turn"]
            rows.append(
                [
                    result["mode"]["name"],
                    *[
                        _number(
                            by_turn[str(t)]["cached_tokens"] / by_turn[str(t)]["eligible_tokens"]
                        )
                        if str(t) in by_turn
                        else "-"
                        for t in turns
                    ],
                ]
            )
        lines += _table(["mode", *[f"client hit turn {t}" for t in turns]], rows)
    rows = []
    for result in results:
        report = result["report"]
        reasons = report.get("validity", {}).get("invalid_reasons", [report.get("report_error")])
        client = result["client"]
        comparison = report.get("capacity_comparison")
        rows.append(
            [
                result["mode"]["name"],
                ", ".join(map(str, reasons)) or "-",
                "-"
                if comparison is None
                else f"{comparison['equal_usable_capacity']} (tolerance {comparison['tolerance_pages']})",
                _number(result["build_seconds"], 0),
                _number(result["wall_seconds"] - result["build_seconds"], 0),
                f"{_number(client['latency_p50_s'])} / {_number(client['latency_p99_s'])}",
            ]
        )
    lines += _table(
        [
            "mode",
            "invalid reasons",
            "equal usable capacity",
            "LLM build s",
            "run wall s",
            "client latency p50 / p99 s (informational)",
        ],
        rows,
    )
    return "\n".join(lines)


_SUMMARY_ROWS: list[tuple[str, Callable[[Mapping], float | int | None], int]] = [
    ("token hit rate (matched / eligible)", lambda r: r["global"]["token_prefix_hit_rate"], 3),
    ("request hit rate", lambda r: r["global"]["request_prefix_hit_rate"], 3),
    ("context tokens still computed", lambda r: r["global"]["scheduled_context_tokens"], 0),
    (
        "requests per rank, max / min",
        lambda r: max(r["load"]["request_count_by_rank"])
        / max(1, min(r["load"]["request_count_by_rank"])),
        2,
    ),
    (
        "coefficient of variation of the context tokens of the ranks",
        lambda r: _cv(r["load"]["scheduled_context_tokens_by_rank"]),
        3,
    ),
    (
        "per-iteration token variance across ranks",
        lambda r: r["load"]["per_iteration_token_variance"],
        0,
    ),
    (
        "logical duplicate storage ratio",
        lambda r: r["storage"].get("duplicate_storage_ratio"),
        3,
    ),
    (
        "capacity-dropped pages (physical eviction)",
        lambda r: r["storage"].get("last_tier_capacity_dropped_pages"),
        0,
    ),
]


def seed_summary(groups: Mapping[str, Sequence[Mapping]]) -> str:
    """One row per mode and one column per label, for runs of the same workload.

    ``groups`` maps a label (a seed, a quota, a group size) to the results of its modes.
    """
    by_label = {
        label: {result["mode"]["name"]: result for result in results}
        for label, results in groups.items()
    }
    modes = sorted(
        {name for results in by_label.values() for name in results},
        key=lambda name: mode_order(
            next(results[name] for results in by_label.values() if name in results)
        ),
    )
    lines: list[str] = []
    for title, getter, digits in _SUMMARY_ROWS:
        rows = []
        for mode in modes:
            cells = []
            for label in by_label:
                result = by_label[label].get(mode)
                if result is None or "global" not in result["report"]:
                    cells.append("-")
                else:
                    cells.append(_number(getter(result["report"]), digits))
            rows.append([mode, *cells])
        lines.append(f"**{title}**\n")
        lines += _table(["mode", *by_label], rows)
    return "\n".join(lines)


def check_runs(results: Sequence[Mapping]) -> list[str]:
    """One line per run, ending in ``INCOMPLETE`` when the interval cannot be trusted.

    A run is complete when its report was built, every submitted request was counted by an owner,
    and the event stream of every rank is complete. The hits the client saw (``cached_tokens``) must
    equal the hits the ranks counted.
    """
    lines = []
    for result in results:
        name = result["mode"]["name"]
        report = result["report"]
        if "global" not in report:
            lines.append(f"{name}: REPORT ERROR {report.get('report_error')} INCOMPLETE")
            continue
        counters, storage = report["global"], report["storage"]
        expected = result["workload"]["requests"]
        problems = []
        if counters["request_count"] != expected:
            problems.append("request count")
        if not storage["complete"]:
            problems.append("event stream")
        if any("report_error" in phase["report"] for phase in result["phases"].values()):
            problems.append("phase report")
        if counters["matched_prefix_tokens"] != result["client"]["cached_tokens"]:
            problems.append("client and counter hits differ")
        lines.append(
            f"{name}: requests {counters['request_count']}/{expected} "
            f"complete={storage['complete']} drops={storage['last_tier_capacity_dropped_pages']} "
            f"removed={storage['removed_block_copies']} matched={counters['matched_prefix_tokens']} "
            f"cached={result['client']['cached_tokens']} gate={gate_label(report)}"
            + (f" INCOMPLETE ({', '.join(problems)})" if problems else "")
        )
    return lines
