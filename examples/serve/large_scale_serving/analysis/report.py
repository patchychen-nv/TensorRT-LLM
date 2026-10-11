#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build the per-request table, the engine iteration tables and one HTML report for a serving run.

    python3 analysis/report.py <attempt_dir> [--hour 2026-09-09T13] [--out DIR]
    python3 analysis/report.py <attempt_dir>/request_trace/2026-09-09T13        # same as --hour
    python3 analysis/report.py --index-only                                      # just rebuild _reports/index.html

With --hour only the requests of that UTC hour are read, and the engine logs are cut to the window
[first request start, last request finish] of those requests. Without it the whole attempt is used.

Output (default: _reports/<run name>[_<hour>]/ beside this checkout):
    requests.csv        one row per request                      (requests_table.py)
    ctx_iters.csv       one row per (ctx worker, iteration), ranks pooled
    ctx_rank_iters.csv  one row per (ctx worker, iteration, rank)
    gen_iters.csv / gen_rank_iters.csv   the same for the generation worker
    worker_iters.csv / worker_rank_iters.csv   instead of the four above on an aggregated run (one
                        server doing prefill and decode; its log is server.log)
    REPORT.html         §0 headline metrics, percentile tables, per-instance engine tables, charts
    summary.json        §0 as data plus run / attempt / hour / topology / GPUs, for a collector to plot across reports
    model.json          empirical fits and validation metrics (with --model)
    notes.json, engine_meta.json   what --from-csv needs to re-render the page without the logs

The deployment mode is read off the logs: ctx-N.log / gen-N.log is disaggregated, server.log alone
is aggregated. The request table picks its join key and TTFT definition the same way (requests_table.py).
"""
from __future__ import annotations

import argparse
import bisect
import html
import json
import math
import re
import statistics
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import charts
from common import (
    TIME_COLUMNS,
    effective_isl_new,
    local_tz,
    percentile,
    read_csv_rows,
    stats,
    utc_iso,
    write_csv,
    yaml_scalar,
)
from engine_iters import INSTANCE_COLUMNS, RANK_COLUMNS, TIER_COUNTERS, build_engine
from requests_table import COLUMNS as REQUEST_COLUMNS
from requests_table import build_requests
from rollup import build_rollup

REPORTS_ROOT = Path(__file__).resolve().parent.parent / "_reports"

# Latency phases in the order they happen inside one request, nested by containment: (column, label, level).
# A level-n row lies inside the nearest level-(n-1) row above it. The proxy fronting ctx and gen workers
# parses the body before it dispatches; the ctx worker's own overhead, queue and prefill follow; the KV
# transfer and the gen decode fill the span after the first token.
LATENCY_TREE = [
    ("e2e_ms", "E2E (arrival → response end)", 0),
    ("ttft_ms", "TTFT (proxy arrival → first token)", 1),
    ("proxy_dispatch_ms", "Proxy pre-dispatch (arrival → ctx dispatch)", 2),
    ("parse_ms", "Body parse (arrival → trace hook)", 3),
    ("server_overhead_ms", "ctx server overhead (HTTP arrival → engine enqueue)", 2),
    ("prefill_queue_ms", "Prefill queue (ctx enqueue → first scheduled)", 2),
    ("prefill_ms", "Prefill (first scheduled → first token)", 2),
    ("gpu_prefill_ms", "GPU prefill forward (ctx)", 3),
    ("decode_ms", "Decode (E2E − TTFT)", 1),
    ("kv_transfer_ms", "KV transfer (gen)", 2),
    ("gen_decode_ms", "Decode on gen (first → last token)", 2),
]
AGG_LATENCY_TREE = [  # one server, one worker record per request: the body parse sits inside the server overhead
    ("e2e_ms", "E2E (arrival → response end)", 0),
    ("ttft_ms", "TTFT (server arrival → first token)", 1),
    ("server_overhead_ms", "Server overhead (HTTP arrival → engine enqueue)", 2),
    ("parse_ms", "Body parse (arrival → trace hook)", 3),
    ("prefill_queue_ms", "Engine queue (enqueue → first scheduled)", 2),
    ("prefill_ms", "Prefill (first scheduled → first token)", 2),
    ("decode_ms", "Decode (E2E − TTFT)", 1),
    ("gen_decode_ms", "Decode on the worker (first → last token)", 2),
]
TOKEN_METRICS = [  # (column, label, decimals)
    ("isl_total", "ISL total", 0), ("isl_cached", "ISL cached", 0), ("isl_new", "ISL new", 0), ("osl", "OSL", 0),
    ("cache_hit_ratio", "Cache hit ratio (cached / total)", 3),
    ("kv_transfer_bytes", "KV transfer bytes", 0),
    ("mtp_acceptance", "Draft acceptance rate (accepted / proposed draft tokens, gen)", 3),
]
AGG_TOKEN_METRICS = [
    ("isl_total", "ISL total", 0), ("isl_cached", "ISL cached", 0), ("isl_new", "ISL new", 0), ("osl", "OSL", 0),
    ("cache_hit_ratio", "Cache hit ratio (cached / total)", 3),
    ("mtp_acceptance", "Draft acceptance rate (accepted / proposed draft tokens)", 3),
]


def metrics_for(mode: str) -> tuple[list, list]:
    """(latency tree, token metrics) for the deployment mode."""
    return (AGG_LATENCY_TREE, AGG_TOKEN_METRICS) if mode == "agg" else (LATENCY_TREE, TOKEN_METRICS)


# ---------------------------------------------------------------- formatting
def esc(text) -> str:
    return html.escape(str(text))


def num(value, digits: int = 0) -> str:
    if value is None:
        return "—"
    return f"{value:,.{digits}f}"


def dur(ms) -> str:
    if ms is None:
        return "—"
    return f"{ms / 1000:,.2f} s" if abs(ms) >= 1000 else f"{ms:,.0f} ms"


def pct(value) -> str:
    return "—" if value is None else f"{value * 100:.1f}%"


def table(header: list[str], rows: list[list], caption: str = "") -> str:
    head = "".join(f"<th>{esc(h)}</th>" for h in header)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>" for row in rows)
    cap = f"<caption>{esc(caption)}</caption>" if caption else ""
    return f"<table>{cap}<thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def latency_rows(done: list[dict], tree) -> list[list]:
    """One row per phase: the label indented by nesting level, n / mean / p50 / p90 / p99, and its share of E2E.

    Share = Σ phase / Σ E2E over the completed requests that have the phase, i.e. time-weighted: a long
    request counts for its length, and sibling phases add up to about their parent's share.
    """
    out = []
    for column, label, level in tree:
        s = stats(r.get(column) for r in done)
        both = [r for r in done if r.get(column) is not None and r.get("e2e_ms")]
        share = sum(r[column] for r in both) / sum(r["e2e_ms"] for r in both) if both else None
        out.append([f'<span class="lvl{level}">{esc(label)}</span>', s["n"],
                    dur(s["mean"]), dur(s["p50"]), dur(s["p90"]), dur(s["p99"]), pct(share)])
    return out


def token_rows(done: list[dict], metrics) -> list[list]:
    out = []
    for column, label, digits in metrics:
        s = stats(r.get(column) for r in done)
        out.append([esc(label), s["n"], num(s["mean"], digits), num(s["p50"], digits), num(s["p90"], digits), num(s["p99"], digits)])
    return out


def mean(values) -> float | None:
    kept = [v for v in values if v is not None]
    return statistics.fmean(kept) if kept else None


def ratio_cells(values: list, fmt) -> list[str]:
    """Each value formatted, then its ratio to the smallest non-zero value of the column: "1.08 s (1.12×)".

    The column is one metric across the ranks of one worker, so the ratio reads as "how many times the
    best rank"; a zero or missing value carries no ratio.
    """
    floor = min((v for v in values if v), default=None)
    return [fmt(v) + (f" ({v / floor:.2f}×)" if v and floor else "") for v in values]


def rank_label(rank):
    """Ranks arrive as ints from the JSON traces and as floats from a CSV round trip; print them as ints."""
    return int(rank) if isinstance(rank, float) and rank.is_integer() else rank


def tier_cell(rank_rows: list[dict], counter: str) -> str:
    """Cumulative cross-tier block count at the end of the window: each rank's last counter, summed.

    The counters run since engine start, so on an hour report this includes earlier hours.
    """
    last_of_rank: dict = {}
    for r in rank_rows:
        last_of_rank[r["rank"]] = r.get(f"{counter}_total")
    cum = [v for v in last_of_rank.values() if v is not None]
    return num(sum(cum)) if cum else "—"


# ---------------------------------------------------------------- §0 metrics
BUCKET_S = 60.0                                   # ratio-of-sums and per-GPU rows are taken per minute
SPREAD = (0.10, 0.25, 0.50, 0.75, 0.90, 0.99)
SPREAD_COLS = ("mean",) + tuple(f"p{int(q * 100)}" for q in SPREAD)
HIGHLIGHT = {"mean", "p10", "p50", "p90"}         # drawn in red
PARALLEL_KEYS = (("tp", "tensor_parallel_size"), ("pp", "pipeline_parallel_size"), ("cp", "context_parallel_size"))


def dist(values) -> dict:
    """n, mean and the SPREAD percentiles over the non-missing values."""
    kept = sorted(v for v in values if v is not None)
    out = {"n": len(kept), "mean": statistics.fmean(kept) if kept else None}
    out.update({f"p{int(q * 100)}": percentile(kept, q) for q in SPREAD})
    return out


def gpus_per_instance(config: Path) -> tuple[int, str] | None:
    """tp × pp × cp of one server config (each rank is one GPU), with the factors spelled out."""
    if not config.exists():
        return None
    sizes = [(label, yaml_scalar([config], key) or 1) for label, key in PARALLEL_KEYS]
    return math.prod(v for _, v in sizes), " × ".join(f"{label} {v}" for label, v in sizes)


def instance_count(disagg_config: Path, section: str) -> int | None:
    """`num_instances` of the context_servers / generation_servers block, or its url count."""
    if not disagg_config.exists():
        return None
    block = re.search(rf"^{section}:[ \t]*\n((?:[ \t]+\S.*\n?)*)", disagg_config.read_text(), re.M)
    if not block:
        return None
    hit = re.search(r"^[ \t]+num_instances:[ \t]*(\d+)", block.group(1), re.M)
    if hit:
        return int(hit.group(1))
    return len(re.findall(r"^[ \t]+-[ \t]*\S+:\d+", block.group(1), re.M)) or None


def gpu_count(attempt: Path, mode: str, engine: dict) -> tuple[int | None, str]:
    """(total GPUs serving the run, how the number was reached).

    From the config snapshots: instances × tp × pp × cp per role, one GPU per rank. The worker logs
    give the same count as distinct ranks per worker and are quoted as the cross-check; they are the
    fallback when no config snapshot sits beside the logs.
    """
    roles = {w["worker"]: w["role"] for w in engine.get("workers", [])}
    ranks: dict[str, set] = defaultdict(set)
    for key in ("ctx_rank", "gen_rank", "mixed_rank"):
        for r in engine.get(key, []):
            ranks[r["worker"]].add(r["rank"])
    per_role: dict[str, list[int]] = defaultdict(list)
    for worker, seen in ranks.items():
        per_role[roles.get(worker, "?")].append(len(seen))
    observed = sum(sum(v) for v in per_role.values())
    observed_how = " + ".join(f"{role} {len(v)} × {v[0]}" if len(set(v)) == 1 else f"{role} {sum(v)}"
                              for role, v in sorted(per_role.items()))

    if mode == "agg":
        plan = [("server", attempt / "server_config.yaml", None)]
    else:
        plan = [("ctx", attempt / "ctx_config.yaml", "context_servers"), ("gen", attempt / "gen_config.yaml", "generation_servers")]
    total, parts = 0, []
    for role, config, section in plan:
        per = gpus_per_instance(config)
        if per is None:
            continue
        n = (instance_count(attempt / "disagg_config.yaml", section) if section else 1) or len(per_role.get(role, [])) or 1
        total += n * per[0]
        parts.append(f"{role} {n} × ({per[1]})")
    if parts:
        check = (f"; the worker logs show {observed_how} = {observed} ranks" + ("" if observed == total else " — MISMATCH")
                 if observed else "; no worker log to cross-check")
        return total, f"GPUs = {total}: {' + '.join(parts)} from the config snapshots{check}."
    if observed:
        return observed, f"GPUs = {observed} = distinct ranks in the worker logs ({observed_how}); no config snapshot to read tp/pp/cp from."
    return None, "GPUs unknown: no config snapshot and no worker log; the per-GPU rows are empty."


NO_SESSION = "_no_session"
OCCUPANCY_POINTS = 3600   # grid points for the occupancy series: one per second on an hour report


def occupancy(intervals: list[tuple[float, float]], grid: list[float]) -> list[tuple[float, float]]:
    """How many [start, end) intervals cover each grid point."""
    starts = sorted(a for a, _ in intervals)
    ends = sorted(b for _, b in intervals)
    return [(t, bisect.bisect_right(starts, t) - bisect.bisect_right(ends, t)) for t in grid]


def instance_occupancy(rows: list[dict], lo: float, hi: float) -> tuple[list[tuple[str, list]], dict]:
    """The four occupancy series of §0 and a few numbers about them.

    Client requests: one per conversation, keyed on the trace's session -- the conversation_id the serve
    path itself resolves (headers, else prompt_cache_key), i.e. what the router uses for affinity -- from
    the start of its first turn to the end of its last turn loaded. The tool-call gaps between turns count,
    because the next turn comes back for the same KV; no cap on the gap. A request without a key is its own
    single-turn conversation. Turns in flight: the server-side requests, arrival to response end. Prefill:
    from ctx first scheduled (first token minus the prefill span) to the first token. Decode: from the first
    token to the response end. A turn still open at the end of the window ends there.
    """
    kept = [r for r in rows if r.get("started_at") is not None and r["status"] not in ("rejected_400", "no_response")]
    end_of = lambda r: r["finished_at"] if r.get("finished_at") is not None else hi
    turns = [(r["started_at"], end_of(r)) for r in kept]
    by_session: dict[str, list] = defaultdict(list)
    for r in kept:
        key = r.get("session") if r.get("session") and r["session"] != NO_SESSION else r.get("trace_id") or id(r)
        by_session[key].append((r["started_at"], end_of(r)))
    sessions = [(min(a for a, _ in t), max(b for _, b in t)) for t in by_session.values()]
    prefill, decode = [], []
    for r in kept:
        if r.get("ttft_ms") is None:
            continue
        first = r["started_at"] + r["ttft_ms"] / 1000.0
        if r.get("prefill_ms") is not None:
            prefill.append((first - r["prefill_ms"] / 1000.0, first))
        decode.append((first, end_of(r)))
    gaps = []
    for t in by_session.values():
        t.sort()
        gaps += [b[0] - a[1] for a, b in zip(t, t[1:]) if b[0] > a[1]]
    step = max(1.0, (hi - lo) / OCCUPANCY_POINTS)
    grid = [lo + i * step for i in range(int((hi - lo) / step) + 1)]
    series = [("client requests in the instance (conversations, tool-call gaps included)", occupancy(sessions, grid)),
              ("turns in flight (server requests)", occupancy(turns, grid)),
              ("turns in prefill", occupancy(prefill, grid)),
              ("turns in decode", occupancy(decode, grid))]
    facts = {"conversations": len(sessions), "multi_turn": sum(1 for t in by_session.values() if len(t) > 1),
             "turns": len(kept), "with_key": sum(1 for r in kept if r.get("session") and r["session"] != NO_SESSION),
             "gap_p50": percentile(sorted(gaps), 0.5), "gap_p90": percentile(sorted(gaps), 0.9), "gaps": len(gaps)}
    return series, facts


def metrics_section(rows: list[dict], mode: str, total_gpu: int | None, gpu_how: str,
                    tokens_per_block: int | None) -> str:
    """§0: the headline numbers as mean / p10 / p25 / p50 / p75 / p90 / p99, in three blocks.

    Per-request rows are distributions over the completed requests. The Σ / Σ and per-GPU rows
    are taken per BUCKET_S bucket keyed on the request's finish time (first and last partial
    bucket dropped) for the percentiles, while their mean cell is the same quantity over the
    whole window -- Σ over every completed request divided by Σ e2e or by window length × GPUs.
    The two differ: the whole-window ratio weights each minute by the request-seconds it carried.
    """
    agg = mode == "agg"
    reqs, source = [], Counter()
    for r in rows:
        if (r["status"] != "completed" or not r.get("e2e_ms") or r["e2e_ms"] <= 0
                or r.get("started_at") is None or r.get("finished_at") is None):
            continue
        new, how = effective_isl_new(r, mode, tokens_per_block)
        source[how] += 1
        reqs.append({"e2e": r["e2e_ms"] / 1000.0, "ttft": None if r.get("ttft_ms") is None else r["ttft_ms"] / 1000.0,
                     "osl": r.get("osl"), "new": new, "start": r["started_at"], "finish": r["finished_at"]})
    parts = ["<h2>0 · Metrics</h2>"]
    if not reqs:
        return parts[0] + "<p class='sub'>No completed requests.</p>"
    lo, hi = min(q["start"] for q in reqs), max(q["finish"] for q in reqs)
    span = hi - lo
    buckets: dict[int, list[dict]] = defaultdict(list)
    for q in reqs:
        buckets[math.floor(q["finish"] / BUCKET_S)].append(q)
    full = [buckets.get(b, []) for b in range(math.floor(lo / BUCKET_S) + 1, math.floor(hi / BUCKET_S))]
    with_osl = [q for q in reqs if q["osl"] is not None]
    with_new = [q for q in with_osl if q["new"] is not None]
    decode = [q for q in with_osl if q["ttft"] is not None and q["osl"] >= 2 and q["e2e"] > q["ttft"]]

    def cells(label: str, d: dict, fmt, indent: bool = False) -> list:
        out = [f'<span class="ind">{esc(label)}</span>' if indent else esc(label), d["n"]]
        for col in SPREAD_COLS:
            text = fmt(d[col])
            out.append(f'<span class="hl">{text}</span>' if col in HIGHLIGHT else text)
        return out

    def per_request(label, values, fmt, indent=False) -> list:
        return cells(label, dist(values), fmt, indent)

    def windowed(label, per_bucket, whole, fmt, keep_empty: bool, indent=False) -> list:
        """Percentiles over the full buckets, mean cell = the whole-window value."""
        d = dist(per_bucket(g) for g in full if g or keep_empty)
        d["mean"] = whole
        return cells(label, d, fmt, indent)

    def ratio(top, group) -> float | None:
        kept = [q for q in group if q["osl"] is not None and (top is osl or q["new"] is not None)]
        bottom = sum(q["e2e"] for q in kept)
        return sum(top(q) for q in kept) / bottom if bottom else None

    def per_gpu(top, group, seconds) -> float | None:
        if not total_gpu:
            return None
        kept = [q for q in group if q["osl"] is not None and (top is osl or top is one or q["new"] is not None)]
        return sum(top(q) for q in kept) / seconds / total_gpu

    osl = lambda q: q["osl"]
    total = lambda q: q["new"] + q["osl"]
    new = lambda q: q["new"]
    one = lambda q: 1
    sec = lambda s: dur(None if s is None else s * 1000.0)
    tps = lambda v: num(v, 1)
    rps = lambda v: num(v, 3)

    e2e_rows = [
        per_request("E2E latency", (q["e2e"] for q in reqs), sec),
        per_request("Output tokens / E2E (tok/s)", (q["osl"] / q["e2e"] for q in with_osl), tps),
        windowed("Σ output tokens / Σ E2E (tok/s, 1-min buckets)", lambda g: ratio(osl, g), ratio(osl, reqs), tps, False, indent=True),
        per_request("(ISL new + OSL) / E2E (tok/s)", ((q["new"] + q["osl"]) / q["e2e"] for q in with_new), tps),
        windowed("Σ (ISL new + OSL) / Σ E2E (tok/s, 1-min buckets)", lambda g: ratio(total, g), ratio(total, reqs), tps, False, indent=True),
    ]
    tput_rows = [
        windowed("Requests / s / GPU", lambda g: per_gpu(one, g, BUCKET_S), per_gpu(one, reqs, span), rps, True),
        windowed("(ISL new + OSL) tokens / s / GPU", lambda g: per_gpu(total, g, BUCKET_S), per_gpu(total, reqs, span), tps, True),
        windowed("Output tokens / s / GPU", lambda g: per_gpu(osl, g, BUCKET_S), per_gpu(osl, reqs, span), tps, True),
        windowed("Input tokens (ISL new) / s / GPU", lambda g: per_gpu(new, g, BUCKET_S), per_gpu(new, reqs, span), tps, True),
    ]
    slo_rows = [
        per_request("TTFT", (q["ttft"] for q in reqs), sec),
        per_request("TPS / user = (OSL − 1) / (E2E − TTFT)", ((q["osl"] - 1) / (q["e2e"] - q["ttft"]) for q in decode), tps),
    ]
    header = ["metric", "n"] + list(SPREAD_COLS)
    ttft_how = ("TTFT is the server's HTTP arrival to its first token." if agg else
                "TTFT is the proxy's arrival to the first token it forwarded, the ctx worker's, before the KV transfer; "
                "the decode span under TPS / user therefore includes the transfer and the proxy hop.")
    fallback = (f" On {source['blocks']} requests the /v1/responses usage reported cached == total (the pre-09-16 proxy bug) and "
                f"ISL new is the ctx worker's newly allocated KV blocks × {tokens_per_block}." if source["blocks"] else "")
    missing = (f" {source['missing']} completed requests have no ISL new and are left out of the ISL-new rows." if source["missing"] else "")
    parts.append(
        f"<p class='sub'>Completed requests only: {len(reqs)} of {len(rows)}, finishing between {utc_iso(lo)} and {utc_iso(hi)} "
        f"({span / 60:.1f} min). Per-request rows are distributions over those requests. Rows marked 1-min buckets "
        f"(and the whole Throughput block) are computed per minute of finish time over the {len(full)} full minutes inside "
        f"the window; their percentiles run over the minutes and their <span class='hl'>mean</span> is the whole-window "
        f"value: Σ over all requests divided by Σ E2E, or by {span / 60:.1f} min × GPUs. That is the request-second-weighted "
        f"mean of the minutes, not their plain average. An empty minute counts as 0 in Throughput and is skipped in the Σ / Σ rows. "
        f"{esc(gpu_how)} ISL new = usage ISL total − cached.{fallback}{missing} {ttft_how} TPS / user is 1 / TPOT, "
        f"over requests with at least two output tokens.</p>")
    parts.append('<div class="block"><h3>E2E</h3>' + table(header, e2e_rows) + "</div>")
    parts.append('<div class="block"><h3>Throughput</h3>' + table(header, tput_rows) + "</div>")
    parts.append('<div class="block"><h3>Token-level SLO</h3>' + table(header, slo_rows) + "</div>")

    series, facts = instance_occupancy(rows, lo, hi)
    levels = [[esc(name), num(statistics.fmean(v for _, v in pts), 1), num(percentile(sorted(v for _, v in pts), 0.5)),
               num(percentile(sorted(v for _, v in pts), 0.9)), num(max(v for _, v in pts))] for name, pts in series if pts]
    gap = (f" Between two turns of one conversation the client is away (tool call) for {dur(facts['gap_p50'] * 1000)} p50, "
           f"{dur(facts['gap_p90'] * 1000)} p90 ({facts['gaps']} gaps)." if facts["gaps"] else "")
    key_note = ("" if facts["with_key"] == facts["turns"] else
                f" {facts['turns'] - facts['with_key']} turns have no session key and count as single-turn conversations.")
    parts.append('<div class="block"><h3>In the instance over time</h3>'
                 f"<p class='sub'>Client requests are conversations: the trace's session key, the conversation_id the server "
                 f"resolves for routing affinity (Codex thread id here), groups the turns one agent run sends, and a conversation "
                 f"stays in the instance from its first turn's arrival to its last turn's end, tool-call gaps included with no cap, "
                 f"because the next turn comes back for the same KV. {facts['conversations']} "
                 f"conversations over {facts['turns']} turns here, {facts['multi_turn']} of them with more than one turn.{gap}"
                 f"{key_note} Turns in flight are the server-side requests; prefill runs from ctx first scheduled to the first token, "
                 f"decode from the first token to the response end. Rejected and "
                 f"unanswered requests are left out; a turn still open at the end of the window ends there.</p>"
                 + charts.line_chart(series, "Requests in the instance", "requests")
                 + table(["series", "mean", "p50", "p90", "max"], levels, "occupancy over the window, sampled per second")
                 + "</div>")
    return "".join(parts)


# ---------------------------------------------------------------- summary.json
_DEPLOY_HEADER = re.compile(r"instance=(\S+)\s+nodes=(\d+)\s+world_size=(\d+)\s+ctx_instances=(\d+)\s+gen_instances=(\d+)")
_KEY_BY_LABEL = (  # §0 row label prefix -> stable key a collector plots by; anything else gets a slug of its label
    ("E2E latency", "e2e_s"), ("Output tokens / E2E", "osl_per_e2e"), ("Σ output tokens / Σ E2E", "sum_osl_per_sum_e2e"),
    ("(ISL new + OSL) / E2E", "total_per_e2e"), ("Σ (ISL new + OSL) / Σ E2E", "sum_total_per_sum_e2e"),
    ("Requests / s / GPU", "rps_per_gpu"), ("(ISL new + OSL) tokens / s / GPU", "total_tok_s_gpu"),
    ("Output tokens / s / GPU", "out_tok_s_gpu"), ("Input tokens (ISL new) / s / GPU", "in_new_tok_s_gpu"),
    ("TTFT", "ttft_s"), ("TPS / user", "tps_user"),
)


def deployment_meta(attempt: Path) -> dict:
    """Instance, topology tag and commit from the fleetctl files beside the attempt directories.

    `deployment.yaml` carries `# instance=i02 nodes=8 world_size=32 ctx_instances=6 gen_instances=1`,
    `run_metadata.txt` carries `topology=6xctx4 + 1xgen8 on 8x4` and `commit=`. Without them the instance
    is the run name's last `_` field and the server counts come from the log names. `label` is the tag
    reports are compared by: `6P1D`, `4P1D`, `AGG`, with `-hostKV` when the run name says so.
    """
    run = run_name(attempt)
    meta = {"instance": None, "nodes": None, "world_size": None, "ctx_instances": None, "gen_instances": None,
            "topology_raw": None, "commit": None}
    dep = attempt.parent / "deployment.yaml"
    if dep.exists():
        for line in dep.read_text(errors="replace").splitlines()[:12]:
            hit = _DEPLOY_HEADER.search(line)
            if hit:
                meta.update(instance=hit.group(1), nodes=int(hit.group(2)), world_size=int(hit.group(3)),
                            ctx_instances=int(hit.group(4)), gen_instances=int(hit.group(5)))
                break
    run_meta = attempt.parent / "run_metadata.txt"
    if run_meta.exists():
        for line in run_meta.read_text(errors="replace").splitlines():
            key, _, value = line.partition("=")
            if key.strip() == "topology":
                meta["topology_raw"] = value.strip()
            elif key.strip() == "commit":
                meta["commit"] = value.strip()[:10]
    if meta["ctx_instances"] is None:
        n_ctx, n_gen = len(list(attempt.glob("ctx-*.log"))), len(list(attempt.glob("gen-*.log")))
        if n_ctx or n_gen:
            meta.update(ctx_instances=n_ctx, gen_instances=n_gen)
        elif (attempt / "server.log").exists():
            meta.update(ctx_instances=0, gen_instances=0)
    if meta["instance"] is None and "_" in run:
        meta["instance"] = run.rsplit("_", 1)[-1]
    if meta["ctx_instances"]:
        label = f"{meta['ctx_instances']}P{meta['gen_instances']}D"
    elif meta["ctx_instances"] == 0:
        label = "AGG"
    else:
        label = "?"
    if "host-kv" in run or "hostkv" in run.lower():
        label += "-hostKV"
    meta["label"] = label
    return meta


def section0_data(rows: list[dict], mode: str, total_gpu: int | None, gpu_how: str,
                  tokens_per_block: int | None) -> list[dict]:
    """§0 as data: each table of metrics_section() as a block of rows with the raw numbers.

    Rather than keeping a second copy of the section's arithmetic, this renders the section with the
    value formatters (dur / num / pct) swapped for markers that carry the raw number and its unit, and
    reads the marked-up tables back. Whatever rows §0 gains appear here without a change.
    """
    g = globals()
    saved = {name: g[name] for name in ("dur", "num", "pct")}
    g["dur"] = lambda ms: "\x01s:%s\x01" % ("" if ms is None else repr(ms / 1000.0))
    g["num"] = lambda v, digits=0: "\x01n:%s\x01" % ("" if v is None else repr(float(v)))
    g["pct"] = lambda v: "\x01r:%s\x01" % ("" if v is None else repr(float(v)))
    try:
        text = metrics_section(rows, mode, total_gpu, gpu_how, tokens_per_block)
    finally:
        g.update(saved)
    blocks = []
    for name, body in re.findall(r"<h3>(.*?)</h3>\s*(?:<p[^>]*>.*?</p>\s*)?<table>(.*?)</table>", text, re.S):
        header = [html.unescape(re.sub(r"<[^>]+>", "", c)) for c in re.findall(r"<th>(.*?)</th>", body, re.S)]
        out_rows = []
        for tr in re.findall(r"<tr>(.*?)</tr>", body, re.S):
            cells = re.findall(r"<td>(.*?)</td>", tr, re.S)
            if len(cells) != len(header):
                continue
            label = html.unescape(re.sub(r"<[^>]+>", "", cells[0])).strip()
            key = next((k for prefix, k in _KEY_BY_LABEL if label.startswith(prefix)), None) \
                or re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")
            row = {"key": key, "label": label, "indent": 'class="ind"' in cells[0], "unit": None}
            for col, cell in zip(header[1:], cells[1:]):
                mark = re.search(r"\x01([snr]):([^\x01]*)\x01", cell)
                if mark:
                    kind, raw = mark.group(1), mark.group(2)
                    row[col] = float(raw) if raw else None
                    row["unit"] = {"s": "s", "r": "ratio",
                                   "n": "req/s" if "Requests / s" in label else "tok/s" if "tok" in label.lower() else "count"}[kind]
                else:
                    plain = html.unescape(re.sub(r"<[^>]+>", "", cell)).strip().replace(",", "")
                    try:
                        row[col] = int(plain)
                    except ValueError:
                        row[col] = None if plain in ("", "—") else plain
            out_rows.append(row)
        blocks.append({"name": html.unescape(name), "rows": out_rows})
    return blocks


def summary_payload(rows: list[dict], attempt: Path, hour: str | None, mode: str, total_gpu: int | None,
                    gpu_how: str, tokens_per_block: int | None, budgets: dict | None = None) -> dict:
    """What summary.json holds: the §0 data plus what identifies the report, for a collector to plot across reports."""
    done = [r for r in rows if r["status"] == "completed" and r.get("started_at") is not None and r.get("finished_at") is not None]
    lo = min((r["started_at"] for r in done), default=None)
    hi = max((r["finished_at"] for r in done), default=None)
    return {
        "schema": 1, "built_at": time.time(), "run": run_name(attempt),
        "attempt": attempt.name if re.fullmatch(r"attempt-\d+", attempt.name) else None, "hour": hour, "mode": mode,
        **deployment_meta(attempt), "gpus": total_gpu, "gpu_how": gpu_how, "budgets": budgets or {},
        "requests": {"total": len(rows), **dict(Counter(r["status"] for r in rows))},
        "n_completed": len(done),
        "window_start": utc_iso(lo) if lo is not None else None, "window_end": utc_iso(hi) if hi is not None else None,
        "span_s": (hi - lo) if lo is not None and hi is not None else None,
        "report": "REPORT.html",
        "blocks": section0_data(rows, mode, total_gpu, gpu_how, tokens_per_block),
    }


# ---------------------------------------------------------------- sections
def requests_section(rows: list[dict], mode: str = "disagg") -> str:
    agg = mode == "agg"
    latency_tree, token_metrics = metrics_for(mode)
    parts = ["<h2>1 · Requests</h2>"]
    by_route = Counter((r["route"], r["status"]) for r in rows)
    parts.append(table(["route", "status", "requests"],
                       [[esc(route), esc(status), n] for (route, status), n in sorted(by_route.items())]))
    cols = ["metric", "n", "mean", "p50", "p90", "p99"]
    done = [r for r in rows if r["status"] == "completed"]
    how = ("TTFT is the server's own HTTP arrival to its first token; server overhead, engine queue, prefill and "
           "decode come from the worker's perf record of the same request, joined on the server request id (client_id). "
           "Prefill here is first scheduled to first token: with a KV-cache hit that is one chunk, a cold long prompt "
           "spans several chunked iterations shared with the decode batch."
           if agg else
           "TTFT is the proxy's arrival to the first token it forwarded (the ctx worker's token, before KV transfer); "
           "ctx and gen phases come from the worker perf records of the same disagg_request_id.")
    parts.append(f"<h3>Latency</h3><p class='sub'>Completed requests only ({len(done)} of {len(rows)}); "
                 "an errored or disconnected stream has a truncated E2E. The rows are the phases of a request in the order "
                 "they happen; an indented row lies inside the span above it. Share of E2E = Σ phase / Σ E2E over the "
                 f"requests that have the phase, so it is weighted by request length. {how}</p>"
                 + table(cols + ["share of E2E"], latency_rows(done, latency_tree)))
    parts.append("<h3>Tokens</h3>" + table(cols, token_rows(done, token_metrics)))
    parts.append('<p class="sub">ISL / OSL / cached come from the usage block the client received. Traces recorded before '
                 "2026-09-16 carry the /v1/responses proxy bug that reports cached_tokens equal to the whole prompt; there "
                 "isl_cached, isl_new and cache_hit_ratio on that route are wrong and ctx_blocks_reused × tokens_per_block in "
                 "requests.csv is the engine-side measurement. The draft acceptance rate is the worker's "
                 "speculative_decoding.acceptance_rate per request: accepted / proposed draft tokens over its decode.</p>")

    if agg:
        # One worker, so the routing decision is which attention-DP rank took the request. Each latency
        # cell is the rank's p50 followed by its ratio to the best rank's p50.
        routed = [r for r in rows if r["routed_rank"] is not None]
        rank_ids = sorted({r["routed_rank"] for r in routed})
        by_rank = {rank: [r for r in routed if r["routed_rank"] == rank and r["status"] == "completed"] for rank in rank_ids}
        p50 = lambda column, rank: stats(r[column] for r in by_rank[rank])["p50"]
        columns = [ratio_cells([p50(column, rank) for rank in rank_ids], dur) for column in ("ttft_ms", "prefill_queue_ms", "gen_decode_ms")]
        per_rank = [[rank_label(rank), sum(1 for r in routed if r["routed_rank"] == rank), *cells]
                    for rank, *cells in zip(rank_ids, *columns)]
        parts.append(f"<h3>By rank (from the routing trace)</h3><p class='sub'>{len(routed)} of {len(rows)} requests have a "
                     "routing decision (adp_route_trace.jsonl, joined on the server request id); the rank is where the "
                     "request's prefill and decode ran. p50 over completed requests; in brackets the ratio to the lowest rank.</p>"
                     + table(["rank", "requests", "TTFT p50", "engine queue p50", "decode p50"], per_rank))
    else:
        per_instance = []
        for inst in sorted({r["ctx_instance"] for r in rows if r["ctx_instance"]}):
            sub = [r for r in rows if r["ctx_instance"] == inst]
            ranks = Counter(r["routed_rank"] for r in sub)
            per_instance.append([esc(inst), len(sub), " / ".join(str(ranks.get(k, 0)) for k in sorted(ranks)),
                                 dur(stats(r["ttft_ms"] for r in sub)["p50"]), dur(stats(r["prefill_ms"] for r in sub)["p50"]),
                                 num(sum(r["ctx_blocks_new"] or 0 for r in sub)), num(sum(r["ctx_blocks_reused"] or 0 for r in sub))])
        parts.append("<h3>By context instance (from the routing trace)</h3>" + table(
            ["ctx instance", "requests", "per rank", "TTFT p50", "prefill p50", "Σ ctx blocks new", "Σ ctx blocks reused"], per_instance))

    parts.append('<div class="grid-2">' + "".join([
        charts.histogram([r["isl_total"] for r in done], "ISL total", "tokens", log_x=True),
        charts.histogram([r["isl_new"] for r in done], "ISL new (usage)", "tokens", log_x=True),
        charts.histogram([r["osl"] for r in done], "OSL", "tokens", log_x=True),
        charts.histogram([r["ttft_ms"] for r in done], "TTFT", "ms", log_x=True),
        charts.histogram([r["e2e_ms"] for r in done], "E2E", "ms", log_x=True),
        charts.histogram([r["prefill_ms"] for r in done], "Prefill" if agg else "Prefill (ctx)", "ms", log_x=True),
    ] + ([charts.histogram([r["prefill_queue_ms"] for r in done], "Engine queue", "ms", log_x=True),
          charts.histogram([r["mtp_acceptance"] for r in done], "Draft acceptance rate", "accepted / proposed")] if agg else [])
    ) + "</div>")
    return "".join(parts)


def last_instance(rows: list[dict], worker: str) -> list[dict]:
    """Rows of the engine that served traffic: the highest engine_instance seen for the worker."""
    mine = [r for r in rows if r["worker"] == worker]
    if not mine:
        return []
    last = max(r["engine_instance"] for r in mine)
    return [r for r in mine if r["engine_instance"] == last]


TIER_CHARTS = (("kv_offload_blocks_total", "offloaded to host (cumulative)"),
               ("kv_onboard_blocks_total", "onboarded back to GPU (cumulative)"),
               ("kv_host_dropped_blocks_total", "dropped from host (cumulative)"))


def budget_note(title: str, items: list[tuple[str, str]], quota: dict) -> str:
    """The highlighted first line of an engine section: the budgets the tables are read against."""
    cells = " · ".join(f"{esc(k)} = <b>{v}</b>" for k, v in items)
    kv = (f" · KV quota per rank GPU <b>{quota['device_quota_gib']:.1f} GiB</b>, host <b>{quota.get('host_quota_gib', 0):.1f} GiB</b>"
          if quota.get("device_quota_gib") else "")
    return f'<p class="note"><b>{esc(title)}:</b> {cells}{kv}</p>'


def kv_row(worker: str, rows: list[dict], rrows: list[dict]) -> list:
    """One worker's KV cache line: utilisation, pool fill peak, cumulative hit rate, cross-tier totals."""
    return [esc(worker), num(mean(r["kv_cache_util_mean"] for r in rows), 3),
            pct(max((r["kv_pool_filled_ratio"] for r in rrows if r["kv_pool_filled_ratio"] is not None), default=None)),
            num(rows[-1]["kv_hit_rate_cum"] if rows else None, 3),
            *[tier_cell(rrows, c) for c in TIER_COUNTERS]]


KV_HEADER = ["worker", "kv_cache_util", "pool filled peak", "hit rate (cum, end)",
             "offload blocks (cum)", "onboard blocks (cum)", "host dropped blocks (cum)"]


def rank_table(worker: str, rrows: list[dict], spec: list[tuple[str, object, object]]) -> list[list]:
    """Per-rank rows of one worker. `spec` is (header, value(rows of the rank), formatter); every cell carries
    its ratio to the lowest rank of the same column (ratio_cells)."""
    by_rank = defaultdict(list)
    for r in rrows:
        by_rank[r["rank"]].append(r)
    rank_ids = sorted(by_rank)
    columns = [ratio_cells([value(by_rank[k]) for k in rank_ids], fmt) for _, value, fmt in spec]
    return [[esc(worker), rank_label(k), *cells] for k, *cells in zip(rank_ids, *columns)]


def last_of(rows: list[dict], key: str):
    return rows[-1][key] if rows else None


STEP_SPEC = [  # the two step clocks, per rank; both describe the same iteration (engine_iters.realign_device_step)
    ("host step", lambda rs: mean(r["host_step_ms"] for r in rs), dur),
    ("device step", lambda rs: mean(r["device_step_ms"] for r in rs), dur),
]
STEP_NOTE = ("Host step is the wall clock of the iteration (time.time() across the loop body, so Σ host step equals the "
             "window length); device step is the same span measured with CUDA events on the GPU stream. The log prints "
             "the device value one iteration late (the ping-pong event pair), so the tables shift it back onto the "
             "iteration it measured; neither is GPU busy time. In the per-rank tables every cell carries its ratio to "
             "the lowest rank in brackets.")


def tier_charts(worker: str, rrows: list[dict], quotas: dict) -> str:
    """Cumulative cross-tier pages per rank over time, against what one rank's GPU pool and host tier hold.

    Unit: pages, one slot per pool group; on a single-pool-group model that is one KV block of
    tokens_per_block tokens across all layers, the same unit as kv_free_blocks / kv_capacity_blocks.
    Host capacity is the GPU pool's slot count scaled by host quota / device quota, i.e. the same
    slot size the GPU pool was observed to use.
    """
    by_rank = defaultdict(list)
    for r in rrows:
        by_rank[r["rank"]].append(r)
    gpu_cap = max((r["kv_capacity_blocks"] for r in rrows if r["kv_capacity_blocks"] is not None), default=None)
    host_cap = None
    if gpu_cap and quotas.get("device_quota_gib") and quotas.get("host_quota_gib"):
        host_cap = gpu_cap * quotas["host_quota_gib"] / quotas["device_quota_gib"]
    refs = [("GPU pool per rank", gpu_cap), ("host tier per rank", host_cap)]
    return "".join(
        charts.line_chart([(f"rank {k}", [(r["timestamp"], r[column]) for r in v]) for k, v in sorted(by_rank.items())],
                          f"{worker}: pages {label}", "pages (1 page = 1 KV block, all layers)", reference_lines=refs)
        for column, label in TIER_CHARTS)


def prefill_section(iters: list[dict], ranks: list[dict], max_num_tokens: int | None, quotas: dict,
                    tokens_per_block: int | None = None, max_batch_size: int | None = None) -> str:
    workers = sorted({r["worker"] for r in iters})
    rank_count = max((r["ranks"] for r in iters), default=None)
    quota = quotas.get(workers[0], {}) if workers else {}
    parts = ["<h2>2 · Prefill workers (ctx)</h2>",
             budget_note("Budgets per rank per iteration (ctx)",
                         [("max_num_tokens", num(max_num_tokens)), ("max_batch_size", num(max_batch_size)),
                          ("tokens_per_block", num(tokens_per_block)), ("workers × ranks", f"{len(workers)} × {num(rank_count)}")], quota),
             '<p class="sub">Attention-DP pads (an idle rank\'s near-budget dummy) count as 0 tokens but stay in every rank average. '
             "Prefill requests / ctx tokens are the log\'s scheduled requests / num_ctx_tokens of the rank. Worker means are over "
             "iterations with prefill work (a ctx worker logs only while it has some); per-rank averages over all of the rank\'s "
             "iterations, a pad counting 0. Skew is (max − mean) / mean across the ranks of one iteration and reads ranks − 1 when "
             "a single rank is busy. Σ paused = requests admitted but set aside for lack of KV blocks or budget, summed over the "
             "window. Hit rate is the log\'s cumulative reused / (reused + missed) since engine start; cross-tier counters are "
             f"cumulative pages since engine start, drawn per rank against one rank\'s GPU pool and host tier. {STEP_NOTE}</p>"]
    schedule, kv, per_rank = [], [], []
    for worker in workers:
        rows, rrows = last_instance(iters, worker), last_instance(ranks, worker)
        busy = [r for r in rows if r["has_prefill"]]
        pads = sum(1 for r in rrows if r["is_adp_pad"])
        schedule.append([esc(worker), len(rows), len(busy), pct(pads / len(rrows)) if rrows else "—",
                         num(mean((r["scheduled_requests_sum"] - r["idle_ranks"]) / r["ranks"] for r in busy if r["ranks"]), 2),
                         num(mean(r["ctx_tokens_mean"] for r in busy)), num(mean(r["token_budget_util"] for r in busy), 3),
                         num(mean(r["busy_ranks"] for r in busy), 2), num(mean(r["ctx_tokens_rank_skew"] for r in busy), 2),
                         num(sum(r["paused_requests_sum"] or 0 for r in rows)),
                         dur(mean(r["host_step_ms_mean"] for r in rows)), dur(mean(r["device_step_ms_mean"] for r in rows))])
        kv.append(kv_row(worker, rows, rrows))
        total_real = sum(r["ctx_tokens_real"] for r in rrows) or 1
        per_rank += rank_table(worker, rrows, [
            ("Σ prefill requests", lambda rs: sum(r["scheduled_requests"] for r in rs if not r["is_adp_pad"]), num),
            ("Σ real ctx tokens", lambda rs: sum(r["ctx_tokens_real"] for r in rs), num),
            ("share", lambda rs: sum(r["ctx_tokens_real"] for r in rs) / total_real, pct),
            ("avg sched reqs", lambda rs: mean(0 if r["is_adp_pad"] else r["scheduled_requests"] for r in rs), lambda v: num(v, 2)),
            ("avg sched tokens", lambda rs: mean(r["total_tokens_real"] for r in rs), lambda v: num(v, 1)),
            ("idle iterations (pads)", lambda rs: sum(1 for r in rs if r["is_adp_pad"]), num),
            ("Σ paused", lambda rs: sum(r["paused_requests"] or 0 for r in rs), num),
            ("hit rate (cum, end)", lambda rs: last_of(rs, "kv_hit_rate_cum"), lambda v: num(v, 3)),
            ("pool filled (end)", lambda rs: last_of(rs, "kv_pool_filled_ratio"), pct),
            ("capacity blocks", lambda rs: last_of(rs, "kv_capacity_blocks"), num),
            *STEP_SPEC])
    parts.append(table(["worker", "iterations", "with prefill", "pad share", "avg prefill reqs / rank", "avg ctx tokens / rank",
                        "budget util", "busy ranks", "rank skew", "Σ paused", "host step", "device step"],
                       schedule, "scheduling: one row per context worker, ranks pooled"))
    parts.append(table(KV_HEADER, kv, "KV cache: one row per context worker, ranks pooled"))
    parts.append(table(["worker", "rank", "Σ prefill requests", "Σ real ctx tokens", "share", "avg sched reqs", "avg sched tokens",
                        "idle iterations (pads)", "Σ paused", "hit rate (cum, end)", "pool filled (end)", "capacity blocks",
                        "host step", "device step"], per_rank, "per rank: where the prefill tokens landed; brackets = ratio to the lowest rank"))
    for i, worker in enumerate(workers):
        rows, rrows = last_instance(iters, worker), last_instance(ranks, worker)
        by_rank = defaultdict(list)
        for r in rrows:
            by_rank[r["rank"]].append(r)
        series = lambda key: [(f"rank {k}", [(r["timestamp"], r[key]) for r in v]) for k, v in sorted(by_rank.items())]
        parts.append(f'<details {"open" if i == 0 else ""}><summary>{esc(worker)} charts</summary><div class="grid-2">'
                     + charts.line_chart([("pooled", [(r["timestamp"], r["token_budget_util"]) for r in rows])],
                                         f"{worker}: token budget utilization", "share of max_num_tokens", y_max=1.0)
                     + charts.line_chart(series("kv_hit_rate_cum"), f"{worker}: KV hit rate (cumulative) per rank", "reused / (reused + missed)", y_max=1.0)
                     + charts.line_chart(series("kv_pool_filled_ratio"), f"{worker}: KV pool filled per rank", "1 − free / capacity", y_max=1.0)
                     + charts.line_chart([("host step", [(r["timestamp"], r["host_step_ms_mean"]) for r in rows]),
                                          ("device step (realigned)", [(r["timestamp"], r["device_step_ms_mean"]) for r in rows])],
                                         f"{worker}: step time (mean over ranks)", "ms", unit=" ms")
                     + tier_charts(worker, rrows, quotas.get(worker, {}))
                     + "</div></details>")
    return "".join(parts)


def decode_section(iters: list[dict], ranks: list[dict], max_batch_size: int | None, quotas: dict,
                   tokens_per_block: int | None = None, max_num_tokens: int | None = None) -> str:
    workers = sorted({r["worker"] for r in iters})
    rank_count = max((r["ranks"] for r in iters), default=None)
    quota = quotas.get(workers[0], {}) if workers else {}
    parts = ["<h2>3 · Generation worker (gen)</h2>",
             budget_note("Budgets per rank per iteration (gen)",
                         [("max_batch_size", num(max_batch_size)), ("max_num_tokens", num(max_num_tokens)),
                          ("tokens_per_block", num(tokens_per_block)), ("workers × ranks", f"{len(workers)} × {num(rank_count)}")], quota),
             '<p class="sub">An idle attention-DP rank carries one dummy decode request (one scheduled request, kv_cache_util 0); '
             "it counts as 0 work. Decode requests and token slots are per iteration on ONE rank, averaged over the ranks "
             "(multiply by the rank count for the worker total). Token slots are the log\'s num_generation_tokens: requests × "
             "(draft length + 1), rounded up to the CUDA-graph batch size and aligned across ranks in a pure-decode iteration — "
             "what the scheduler charges against max_num_tokens, not the tokens accepted (draft acceptance rate in §1). "
             "tokens per request = slots per real request: 1 without speculative decoding, draft length + 1 with it, more where "
             "the padding shows. Worker means are over iterations with at least one real request; per-rank averages over all of "
             "the rank\'s iterations, an idle one counting 0. Σ paused = requests admitted but set aside for lack of KV blocks or "
             "budget, summed over the window. Block reuse is off on this worker, so the hit rate reads empty. Cross-tier counters "
             f"are cumulative pages since engine start. {STEP_NOTE}</p>"]
    schedule, kv, per_rank = [], [], []
    for worker in workers:
        rows, rrows = last_instance(iters, worker), last_instance(ranks, worker)
        busy = [r for r in rows if r["has_decode"]]
        idle_share = sum(r["idle_ranks"] for r in rows) / sum(r["ranks"] for r in rows) if rows else None
        schedule.append([esc(worker), len(rows), len(busy), pct(idle_share),
                         num(mean(r["decode_requests_sum"] / r["ranks"] for r in busy if r["ranks"]), 2),
                         num(mean(r["gen_tokens_sum"] / r["ranks"] for r in busy if r["ranks"]), 1),
                         num(mean(r["tokens_per_request"] for r in busy), 2), num(mean(r["batch_occupancy"] for r in busy), 3),
                         num(sum(r["paused_requests_sum"] or 0 for r in rows)),
                         dur(mean(r["host_step_ms_mean"] for r in rows)), dur(mean(r["device_step_ms_mean"] for r in rows))])
        kv.append(kv_row(worker, rows, rrows))
        per_rank += rank_table(worker, rrows, [
            ("avg sched reqs", lambda rs: mean(r["decode_requests_real"] for r in rs), lambda v: num(v, 2)),
            ("avg sched tokens (slots)", lambda rs: mean(r["gen_tokens_real"] for r in rs), lambda v: num(v, 1)),
            ("idle iterations", lambda rs: sum(1 for r in rs if r["is_adp_pad"]), num),
            ("Σ paused", lambda rs: sum(r["paused_requests"] or 0 for r in rs), num),
            ("kv_cache_util (mean)", lambda rs: mean(r["kv_cache_util"] for r in rs), lambda v: num(v, 3)),
            ("pool filled (end)", lambda rs: last_of(rs, "kv_pool_filled_ratio"), pct),
            ("capacity blocks", lambda rs: last_of(rs, "kv_capacity_blocks"), num),
            *STEP_SPEC])
    parts.append(table(["worker", "iterations", "with decode", "idle rank share", "avg decode reqs / rank", "avg decode token slots / rank",
                        "tokens per request", "batch occupancy", "Σ paused", "host step", "device step"],
                       schedule, "scheduling: one row per generation worker, ranks pooled"))
    parts.append(table(KV_HEADER, kv, "KV cache: one row per generation worker, ranks pooled"))
    parts.append(table(["worker", "rank", "avg sched reqs", "avg sched tokens (slots)", "idle iterations", "Σ paused", "kv_cache_util (mean)",
                        "pool filled (end)", "capacity blocks", "host step", "device step"],
                       per_rank, "per rank: where the decode batch landed; brackets = ratio to the lowest rank"))
    for worker in workers:
        rows, rrows = last_instance(iters, worker), last_instance(ranks, worker)
        busy = [r for r in rows if r["has_decode"]]
        by_rank = defaultdict(list)
        for r in rrows:
            by_rank[r["rank"]].append(r)
        parts.append('<div class="grid-2">'
                     + charts.line_chart([("all ranks", [(r["timestamp"], r["decode_requests_sum"]) for r in rows])], f"{worker}: num_reqs per iteration", "real decode requests")
                     + charts.line_chart([(f"rank {k}", [(r["timestamp"], r["gen_tokens_real"]) for r in v]) for k, v in sorted(by_rank.items())],
                                         f"{worker}: num_tokens per iteration per rank", "decode token slots (real requests)")
                     + charts.line_chart([("pooled", [(r["timestamp"], r["tokens_per_request"]) for r in busy])], f"{worker}: tokens per request per iteration", "tokens")
                     + charts.line_chart([(f"rank {k}", [(r["timestamp"], r["kv_cache_util"]) for r in v]) for k, v in sorted(by_rank.items())], f"{worker}: kv_cache_util per rank", "share of blocks pinned", y_max=1.0)
                     + charts.line_chart([("host step", [(r["timestamp"], r["host_step_ms_mean"]) for r in rows]),
                                          ("device step (realigned)", [(r["timestamp"], r["device_step_ms_mean"]) for r in rows])],
                                         f"{worker}: step time (mean over ranks)", "ms", unit=" ms")
                     + tier_charts(worker, rrows, quotas.get(worker, {}))
                     + "</div>")
    return "".join(parts)


def step_split(rows: list[dict]) -> list[list]:
    """Step time of one worker's iterations by what they carried: a prefill chunk, decode only, or dummies only.

    Both clocks describe the same iteration here: host_step_time is printed for the loop that just
    finished, and engine_iters shifts prev_device_step_time back onto the iteration it measured.
    """
    kinds = (("with a prefill chunk", [r for r in rows if r["has_prefill"]]),
             ("decode only", [r for r in rows if r["decode_only"]]),
             ("idle (dummies only)", [r for r in rows if not r["has_prefill"] and not r["has_decode"]]))
    out = []
    for label, sub in kinds:
        host, device = stats(r["host_step_ms_mean"] for r in sub), stats(r["device_step_ms_mean"] for r in sub)
        out.append([esc(label), len(sub), pct(len(sub) / len(rows)) if rows else "—",
                    dur(host["mean"]), dur(host["p50"]), dur(host["p90"]), dur(host["p99"]),
                    dur(device["mean"]), dur(device["p50"]), dur(device["p90"]), dur(device["p99"]),
                    num(mean(r["ctx_tokens_mean"] for r in sub)),
                    num(mean(r["decode_requests_sum"] / r["ranks"] for r in sub if r["ranks"]), 2)])
    return out


def worker_section(iters: list[dict], ranks: list[dict], max_num_tokens: int | None, max_batch_size: int | None,
                   quotas: dict, tokens_per_block: int | None = None) -> str:
    """The one worker of an aggregated deployment: prefill and decode on the same ranks, same iterations."""
    rank_count = max((r["ranks"] for r in iters), default=None)
    quota = next(iter(quotas.values()), {}) if quotas else {}
    parts = ["<h2>2 · Worker (prefill + decode)</h2>",
             budget_note("Budgets per rank per iteration",
                         [("max_num_tokens", num(max_num_tokens) + " (ctx tokens + decode token slots)"), ("max_batch_size", num(max_batch_size)),
                          ("tokens_per_block", num(tokens_per_block)), ("attention-DP ranks", num(rank_count))], quota),
             '<p class="sub">One aggregated server: every rank runs prefill chunks and decode steps in the same iteration. An idle '
             "rank carries one dummy decode request (one scheduled request, kv_cache_util 0) and counts as 0 work. Prefill requests "
             "/ ctx tokens are the log's num_ctx_requests / num_ctx_tokens; decode requests = scheduled − prefill; decode token slots "
             "= num_generation_tokens, i.e. requests × (draft length + 1) rounded up to the CUDA-graph batch size — what the scheduler "
             "charges against max_num_tokens, not the tokens accepted (draft acceptance rate in §1). Worker averages are over "
             "iterations with real work; per-rank averages over all of the rank's iterations, an idle one counting 0. Σ paused = "
             "requests the scheduler had admitted but set aside for lack of KV blocks or budget, summed over the window. Hit rate is "
             "the log's cumulative reused / (reused + missed) since engine start; cross-tier counters are cumulative pages since engine "
             f"start, drawn per rank against one rank's GPU pool and host tier. {STEP_NOTE}</p>"]
    workers = sorted({r["worker"] for r in iters})
    schedule, kv, per_rank, steps = [], [], [], []
    for worker in workers:
        rows, rrows = last_instance(iters, worker), last_instance(ranks, worker)
        busy = [r for r in rows if r["has_prefill"] or r["has_decode"]]
        idle_share = sum(r["idle_ranks"] for r in rows) / sum(r["ranks"] for r in rows) if rows else None

        def per_rank_mean(key, sub=busy):
            return mean(r[key] / r["ranks"] for r in sub if r["ranks"] and r[key] is not None)

        schedule.append([esc(worker), len(rows), sum(1 for r in rows if r["has_prefill"]), sum(1 for r in rows if r["decode_only"]),
                         pct(idle_share),
                         num(per_rank_mean("ctx_requests_sum"), 2), num(per_rank_mean("decode_requests_sum"), 2),
                         num(mean(r["ctx_tokens_mean"] for r in busy)), num(per_rank_mean("gen_tokens_sum"), 1),
                         num(mean(r["total_budget_util"] for r in busy), 3), num(mean(r["batch_occupancy"] for r in busy), 3),
                         num(sum(r["paused_requests_sum"] or 0 for r in rows)),
                         dur(mean(r["host_step_ms_mean"] for r in rows)), dur(mean(r["device_step_ms_mean"] for r in rows))])
        kv.append(kv_row(worker, rows, rrows))
        steps += [[esc(worker)] + row for row in step_split(rows)]
        total_real = sum(r["ctx_tokens_real"] for r in rrows) or 1
        per_rank += rank_table(worker, rrows, [
            ("Σ prefill requests", lambda rs: sum(r["ctx_requests"] or 0 for r in rs), num),
            ("Σ real ctx tokens", lambda rs: sum(r["ctx_tokens_real"] for r in rs), num),
            ("share", lambda rs: sum(r["ctx_tokens_real"] for r in rs) / total_real, pct),
            ("avg sched reqs", lambda rs: mean((r["ctx_requests"] or 0) + r["decode_requests_real"] for r in rs), lambda v: num(v, 2)),
            ("avg sched tokens", lambda rs: mean(r["total_tokens_real"] for r in rs), lambda v: num(v, 1)),
            ("idle iterations", lambda rs: sum(1 for r in rs if r["is_adp_pad"]), num),
            ("Σ paused", lambda rs: sum(r["paused_requests"] or 0 for r in rs), num),
            ("hit rate (cum, end)", lambda rs: last_of(rs, "kv_hit_rate_cum"), lambda v: num(v, 3)),
            ("pool filled (end)", lambda rs: last_of(rs, "kv_pool_filled_ratio"), pct),
            ("capacity blocks", lambda rs: last_of(rs, "kv_capacity_blocks"), num),
            *STEP_SPEC])

    parts.append(table(["worker", "iterations", "with prefill", "decode only", "idle rank share",
                        "avg prefill reqs / rank", "avg decode reqs / rank", "avg ctx tokens / rank", "avg decode token slots / rank",
                        "budget util (ctx + decode)", "batch occupancy", "Σ paused", "host step", "device step"],
                       schedule, "scheduling: one row per worker, ranks pooled"))
    parts.append(table(KV_HEADER, kv, "KV cache: one row per worker, ranks pooled"))
    parts.append(table(["worker", "iterations", "n", "share", "host step mean", "p50", "p90", "p99",
                        "device step mean", "p50", "p90", "p99", "avg ctx tokens / rank", "avg decode reqs / rank"], steps,
                       "step time by what the iteration carried: a prefill chunk on some rank stretches the step for every rank"))
    parts.append(table(["worker", "rank", "Σ prefill requests", "Σ real ctx tokens", "share", "avg sched reqs (prefill + decode)",
                        "avg sched tokens (prefill + decode)", "idle iterations", "Σ paused", "hit rate (cum, end)",
                        "pool filled (end)", "capacity blocks", "host step", "device step"],
                       per_rank, "per rank: where the work landed; brackets = ratio to the lowest rank"))
    for i, worker in enumerate(workers):
        rows, rrows = last_instance(iters, worker), last_instance(ranks, worker)
        by_rank = defaultdict(list)
        for r in rrows:
            by_rank[r["rank"]].append(r)
        series = lambda key: [(f"rank {k}", [(r["timestamp"], r[key]) for r in v]) for k, v in sorted(by_rank.items())]
        with_prefill = [r for r in rows if r["has_prefill"]]
        decode_only = [r for r in rows if r["decode_only"]]
        parts.append(f'<details {"open" if i == 0 else ""}><summary>{esc(worker)} charts</summary><div class="grid-2">'
                     + charts.line_chart(series("decode_requests_real"), f"{worker}: decode requests per iteration per rank",
                                         "real decode requests", reference_lines=[("max_batch_size", max_batch_size)])
                     + charts.line_chart(series("ctx_tokens_real"), f"{worker}: ctx (prefill) tokens per iteration per rank", "tokens")
                     + charts.line_chart(series("gen_tokens_real"), f"{worker}: decode token slots per iteration per rank",
                                         "tokens (requests × (draft + 1), CUDA-graph padded)")
                     + charts.line_chart(series("total_tokens_real"), f"{worker}: ctx + decode tokens per iteration per rank",
                                         "tokens charged against max_num_tokens", reference_lines=[("max_num_tokens", max_num_tokens)])
                     + charts.line_chart([("mean over ranks", [(r["timestamp"], r["total_budget_util"]) for r in rows])],
                                         f"{worker}: token budget utilization (ctx + decode)", "share of max_num_tokens", y_max=1.0)
                     + charts.line_chart(series("paused_requests"), f"{worker}: paused requests per rank", "requests")
                     + charts.line_chart(series("kv_cache_util"), f"{worker}: kv_cache_util per rank", "share of blocks pinned", y_max=1.0)
                     + charts.line_chart(series("kv_pool_filled_ratio"), f"{worker}: KV pool filled per rank", "1 − free / capacity", y_max=1.0)
                     + charts.line_chart(series("kv_hit_rate_cum"), f"{worker}: KV hit rate (cumulative) per rank", "reused / (reused + missed)", y_max=1.0)
                     + charts.line_chart([("with a prefill chunk", [(r["timestamp"], r["host_step_ms_mean"]) for r in with_prefill]),
                                          ("decode only", [(r["timestamp"], r["host_step_ms_mean"]) for r in decode_only])],
                                         f"{worker}: host step by iteration kind", "ms", unit=" ms")
                     + charts.line_chart([("with a prefill chunk", [(r["timestamp"], r["device_step_ms_mean"]) for r in with_prefill]),
                                          ("decode only", [(r["timestamp"], r["device_step_ms_mean"]) for r in decode_only])],
                                         f"{worker}: device step (realigned) by iteration kind", "ms", unit=" ms")
                     + tier_charts(worker, rrows, quotas.get(worker, {}))
                     + "</div></details>")
    return "".join(parts)


# ---------------------------------------------------------------- index
def write_index(root: Path) -> Path | None:
    """One page listing every report under `root`, newest first, with a few headline numbers."""
    if not root.exists():
        return None
    rows = []
    for report in sorted(root.glob("*/REPORT.html"), key=lambda p: p.stat().st_mtime, reverse=True):
        folder = report.parent
        requests_csv = folder / "requests.csv"
        done = []
        if requests_csv.exists():
            import csv
            with requests_csv.open(encoding="utf-8") as handle:
                done = [r for r in csv.DictReader(handle) if r.get("status") == "completed"]
        ttft = stats(float(r["ttft_ms"]) for r in done if r.get("ttft_ms"))
        e2e = stats(float(r["e2e_ms"]) for r in done if r.get("e2e_ms"))
        isl = stats(float(r["isl_total"]) for r in done if r.get("isl_total"))
        built = datetime.fromtimestamp(report.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
        files = " · ".join(f'<a href="{folder.name}/{f.name}">{f.name}</a>' for f in sorted(folder.iterdir()) if f.name != "REPORT.html")
        rows.append([f'<a href="{folder.name}/REPORT.html">{esc(folder.name)}</a>', built, len(done),
                     dur(ttft["p50"]), dur(e2e["p50"]), num(isl["p50"]), files])
    page = (f"<!doctype html><meta charset=utf-8><title>reports</title>{charts.FONT_LINKS}<style>{charts.CSS}</style>"
            f"<h1>Reports</h1><p class='sub'>{esc(root)} · {len(rows)} reports, newest first · completed requests only</p>"
            + table(["report", "built", "requests", "TTFT p50", "E2E p50", "ISL p50", "files"], rows))
    out = root / "index.html"
    out.write_text(page, encoding="utf-8")
    return out


# ---------------------------------------------------------------- driver
def resolve_paths(target: Path, hour: str | None) -> tuple[Path, str | None]:
    """Accept an attempt dir, or a request_trace/<hour> dir which implies both."""
    target = target.resolve()
    if target.parent.name == "request_trace":
        return target.parent.parent, target.name
    return target, hour


def run_name(attempt: Path) -> str:
    return attempt.parent.name if re.fullmatch(r"attempt-\d+", attempt.name) else attempt.name


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("target", type=Path, nargs="?", help="attempt directory, or its request_trace/<UTC hour> directory")
    parser.add_argument("--index-only", action="store_true", help=f"only rebuild {REPORTS_ROOT}/index.html")
    parser.add_argument("--hour", default=None, help="UTC hour bucket to restrict to, e.g. 2026-09-09T13")
    parser.add_argument("--out", type=Path, default=None, help=f"output directory (default: {REPORTS_ROOT}/<run>[_<hour>])")
    parser.add_argument("--no-index", action="store_true",
                        help="read every log in full instead of resuming from the "
                             "checkpoint index. Slower on a long-lived instance; "
                             "use it to check the index against a full pass.")
    parser.add_argument("--json", action="store_true",
                        help="also write rollup.json: this hour reduced to percentiles, counts and "
                             "bounded samples, for a collector to keep. A few hundred kilobytes.")
    parser.add_argument("--json-only", action="store_true",
                        help="write only rollup.json and stop, skipping the CSVs and the HTML page")
    parser.add_argument("--log-tz", default=None, metavar="ZONE",
                        help="time zone of the worker logs' naive `timestamp` field, e.g. America/Los_Angeles "
                             "(default: this machine's zone; the workers and the analysis must agree)")
    parser.add_argument("--index-dir", type=Path, default=None, metavar="DIR",
                        help="where the worker-log checkpoint index lives (default: <attempt>/.engine_index). "
                             "Point it somewhere writable when the attempt directory is another user's.")
    parser.add_argument("--jobs", type=int, default=None, metavar="N",
                        help="worker logs read in parallel, one process each (default: all of them; 1 = in-process)")
    parser.add_argument("--from-csv", action="store_true",
                        help="re-render REPORT.html and summary.json from the CSVs, notes.json and engine_meta.json "
                             "already in --out; the attempt is only consulted for its config snapshots")
    parser.add_argument(
        "--model", action="store_true",
        help="fit empirical performance models, write model.json, and include them in the report",
    )
    args = parser.parse_args()
    if args.index_only or args.target is None:
        print(f"index -> {write_index(REPORTS_ROOT)}")
        return 0
    log_tz = ZoneInfo(args.log_tz) if args.log_tz else local_tz()
    attempt, hour = resolve_paths(args.target, args.hour)
    out = args.out or REPORTS_ROOT / (run_name(attempt) + (f"_{hour}" if hour else ""))
    out.mkdir(parents=True, exist_ok=True)
    if args.from_csv:
        return rerender(attempt, hour, out, with_model=args.model)

    # B: requests
    requests, notes = build_requests(attempt, hour)
    starts = [r["started_at"] for r in requests if r["started_at"] is not None]
    ends = [r["finished_at"] for r in requests if r["finished_at"] is not None]
    window = (min(starts), max(ends)) if hour and starts and ends else (None, None)
    if hour and window[1] is not None:
        # Responses of this hour's requests may land in the next hour's file (read by build_requests);
        # the engine section still describes this hour, so its window ends at the hour boundary.
        hour_end = datetime.strptime(hour, "%Y-%m-%dT%H").replace(tzinfo=ZoneInfo("UTC")).timestamp() + 3600.0
        window = (window[0], min(window[1], hour_end))

    # A: engine iterations (worker logs are stamped in the worker's local zone; assumed to be this machine's)
    max_num_tokens = yaml_scalar([attempt / "ctx_config.yaml", attempt / "server_config.yaml"], "max_num_tokens")
    max_batch_size = yaml_scalar([attempt / "gen_config.yaml", attempt / "server_config.yaml"], "max_batch_size")
    tokens_per_block = yaml_scalar([attempt / "ctx_config.yaml", attempt / "server_config.yaml"], "tokens_per_block")
    ctx_max_batch_size = yaml_scalar([attempt / "ctx_config.yaml"], "max_batch_size")   # the other two budgets, for the
    gen_max_num_tokens = yaml_scalar([attempt / "gen_config.yaml"], "max_num_tokens")   # highlighted lines of §2 / §3
    # The index lives beside the logs it describes, so it travels with the run
    # and a second reader of the same attempt inherits the work of the first.
    index_dir = args.index_dir or attempt / ".engine_index"
    engine = build_engine(attempt, log_tz, window, max_num_tokens, max_batch_size,
                          None if args.no_index else index_dir, jobs=args.jobs)

    mode = engine.get("mode", "disagg")
    prefill_ranks = engine["mixed_rank"] if mode == "agg" else engine["ctx_rank"]
    real_tokens = sum(r["ctx_tokens_real"] for r in prefill_ranks)
    perf_new_tokens = sum(r["ctx_blocks_new"] or 0 for r in requests) * (tokens_per_block or 0)
    pad_sizes = Counter(int(r["ctx_tokens"]) for r in engine["ctx_rank"] if r["is_adp_pad"])
    total_gpu, gpu_how = gpu_count(attempt, mode, engine)
    notes.append(f"{'Aggregated' if mode == 'agg' else 'Disaggregated'} deployment; workers: "
                 f"{', '.join(w['worker'] for w in engine['workers'])}; budgets max_num_tokens={max_num_tokens}, "
                 f"max_batch_size={max_batch_size}, tokens_per_block={tokens_per_block}. {gpu_how}")
    for w in engine["workers"]:
        if w.get("device_quota_gib"):
            notes.append(f"{w['worker']}: KV quota per rank GPU {w['device_quota_gib']:.2f} GiB, host "
                         f"{w.get('host_quota_gib', 0):.2f} GiB, {w.get('kv_bytes_per_token', 0):,.0f} bytes per token "
                         f"({(w.get('kv_bytes_per_token') or 0) * (tokens_per_block or 0) / 2**20:.2f} MiB per {tokens_per_block}-token block).")
    if mode == "agg":
        dummies = sum(1 for r in engine["mixed_rank"] if r["is_adp_pad"])
        pads = (f"Idle-rank check: {dummies:,} of {len(engine['mixed_rank']):,} rank-iterations were the idle dummy "
                "(one scheduled request, kv_cache_util 0) and count as 0 work; an aggregated worker has no attention-DP context pads")
    else:
        pads = f"Pad check: {sum(pad_sizes.values()):,} attention-DP pads removed (sizes seen: {dict(pad_sizes)})"
    notes.append(f"{pads}; real ctx tokens in window {real_tokens:,.0f} vs KV blocks newly allocated × tokens_per_block "
                 f"{perf_new_tokens:,.0f} → ratio {real_tokens / perf_new_tokens:.4f}." if perf_new_tokens else
                 f"{pads}; KV-block cross-check skipped: no KV block counts in the perf records.")
    if window[0] is not None:
        notes.append(f"Window: {utc_iso(window[0])} to {utc_iso(window[1])} (UTC), from the hour's requests; "
                     f"engine iterations outside it are dropped after differencing. Worker log stamps read as {log_tz}.")

    budgets = {
        "max_num_tokens": max_num_tokens, "max_batch_size": max_batch_size,
        "tokens_per_block": tokens_per_block, "ctx_max_batch_size": ctx_max_batch_size,
        "gen_max_num_tokens": gen_max_num_tokens,
    }
    model_payload = None
    if args.model:
        from model_view import build_model_payload, write_model

        model_payload = build_model_payload(
            requests, engine, window, budgets,
            {
                "run": run_name(attempt), "attempt": attempt.name, "hour": hour,
                "mode": mode, "gpus": total_gpu, "source": "raw", "log_tz": str(log_tz),
                "selection": "hourly_request_cohorts" if hour else "whole_attempt",
            },
        )
        write_model(out, model_payload)
        print(f"  models -> {out / 'model.json'}")
    if args.json:
        # Written before the CSVs and the page, so a collector that only wants
        # this is not held up by the parts it will not read.
        payload = build_rollup(attempt, hour, window, requests, engine, budgets, notes)
        (out / "rollup.json").write_text(json.dumps(payload, allow_nan=False), encoding="utf-8")
        print(f"  rollup -> {out / 'rollup.json'}")
        if args.json_only:
            return 0

    # What a re-render needs and cannot recompute from the CSVs: the notes and the workers' quotas.
    (out / "notes.json").write_text(json.dumps(notes, indent=1), encoding="utf-8")
    (out / "engine_meta.json").write_text(json.dumps({"mode": mode, "workers": engine["workers"], "window": list(window),
                                                      "log_tz": str(log_tz), "budgets": budgets}, indent=1), encoding="utf-8")
    counts = render_report(out, attempt, hour, requests, engine, notes, mode, budgets, total_gpu, gpu_how,
                           ctx_max_batch_size, gen_max_num_tokens, write_tables=True, model_payload=model_payload)
    print(f"{len(requests)} requests, {counts} -> {out}")
    for note in notes:
        print("  note: " + note)
    if out.parent == REPORTS_ROOT:  # a report inside the root refreshes the index that links them all
        print(f"index -> {write_index(REPORTS_ROOT)}")
    return 0


def render_report(out: Path, attempt: Path, hour: str | None, requests: list[dict], engine: dict, notes: list[str],
                  mode: str, budgets: dict, total_gpu: int | None, gpu_how: str,
                  ctx_max_batch_size: int | None, gen_max_num_tokens: int | None, write_tables: bool,
                  model_payload: dict | None = None) -> str:
    """The CSVs (unless re-rendering), summary.json and REPORT.html. Returns the iteration counts for the log line."""
    max_num_tokens, max_batch_size, tokens_per_block = (budgets.get(k) for k in ("max_num_tokens", "max_batch_size", "tokens_per_block"))
    quotas = {w["worker"]: w for w in engine["workers"]}
    if mode == "agg":
        if write_tables:
            write_csv(out / "worker_iters.csv", engine["mixed_iters"], INSTANCE_COLUMNS, TIME_COLUMNS)
            write_csv(out / "worker_rank_iters.csv", engine["mixed_rank"], RANK_COLUMNS, TIME_COLUMNS)
        engine_sections = [worker_section(engine["mixed_iters"], engine["mixed_rank"], max_num_tokens, max_batch_size, quotas,
                                          tokens_per_block)]
        counts = f"{len(engine['mixed_iters'])} worker iterations"
    else:
        if write_tables:
            write_csv(out / "ctx_iters.csv", engine["ctx_iters"], INSTANCE_COLUMNS, TIME_COLUMNS)
            write_csv(out / "ctx_rank_iters.csv", engine["ctx_rank"], RANK_COLUMNS, TIME_COLUMNS)
            write_csv(out / "gen_iters.csv", engine["gen_iters"], INSTANCE_COLUMNS, TIME_COLUMNS)
            write_csv(out / "gen_rank_iters.csv", engine["gen_rank"], RANK_COLUMNS, TIME_COLUMNS)
        engine_sections = [prefill_section(engine["ctx_iters"], engine["ctx_rank"], max_num_tokens, quotas, tokens_per_block, ctx_max_batch_size),
                           decode_section(engine["gen_iters"], engine["gen_rank"], max_batch_size, quotas, tokens_per_block, gen_max_num_tokens)]
        counts = f"{len(engine['ctx_iters'])} ctx and {len(engine['gen_iters'])} gen iterations"
    if write_tables:
        write_csv(out / "requests.csv", requests, REQUEST_COLUMNS, TIME_COLUMNS)

    payload = summary_payload(requests, attempt, hour, mode, total_gpu, gpu_how, tokens_per_block, budgets)
    (out / "summary.json").write_text(json.dumps(payload, allow_nan=False, indent=1), encoding="utf-8")

    title = f"{run_name(attempt)}" + (f" · {hour} UTC" if hour else "")
    page = [f"<!doctype html><meta charset=utf-8><title>{esc(title)}</title>{charts.FONT_LINKS}<style>{charts.CSS}</style>",
            f"<h1>{esc(title)}</h1><p class='sub'>{esc(attempt)}</p>",
            "<ul>" + "".join(f"<li>{esc(n)}</li>" for n in notes) + "</ul>",
            metrics_section(requests, mode, total_gpu, gpu_how, tokens_per_block),
            requests_section(requests, mode), *engine_sections]
    if model_payload is not None:
        from model_view import model_section

        page.append(model_section(model_payload))
    (out / "REPORT.html").write_text("".join(page), encoding="utf-8")
    return counts


def rerender(attempt: Path, hour: str | None, out: Path, with_model: bool = False) -> int:
    """REPORT.html and summary.json again from what a previous run left in `out`, without the logs or the trace.

    For changes to the page itself. Floats come back from the CSVs with 4 decimals, so the numbers can
    differ from the original in the last place; the notes are the original run's, from notes.json.
    """
    meta_path = out / "engine_meta.json"
    if not meta_path.exists():
        print(f"cannot re-render: {meta_path} missing (the report predates --from-csv)")
        return 2
    meta = json.loads(meta_path.read_text())
    notes = json.loads((out / "notes.json").read_text()) if (out / "notes.json").exists() else []
    mode, budgets = meta["mode"], meta["budgets"]
    requests = read_csv_rows(
        out / "requests.csv",
        text_columns=(
            "rid", "trace_id", "session", "route", "status", "model",
            "ctx_instance", "route_phase", "validation_errors",
        ),
    )
    for r in requests:  # ids and enum-like columns are text in the table
        for key in ("rid", "trace_id", "route", "status", "model", "ctx_instance", "route_phase", "validation_errors"):
            if r.get(key) is not None:
                r[key] = str(r[key])
    engine = {"workers": meta["workers"], "mode": mode}
    if mode == "agg":
        engine.update(mixed_iters=read_csv_rows(out / "worker_iters.csv"), mixed_rank=read_csv_rows(out / "worker_rank_iters.csv"),
                      ctx_iters=[], ctx_rank=[], gen_iters=[], gen_rank=[])
    else:
        engine.update(ctx_iters=read_csv_rows(out / "ctx_iters.csv"), ctx_rank=read_csv_rows(out / "ctx_rank_iters.csv"),
                      gen_iters=read_csv_rows(out / "gen_iters.csv"), gen_rank=read_csv_rows(out / "gen_rank_iters.csv"),
                      mixed_iters=[], mixed_rank=[])
    total_gpu, gpu_how = gpu_count(attempt, mode, engine)
    ctx_max_batch_size = yaml_scalar([attempt / "ctx_config.yaml"], "max_batch_size")
    gen_max_num_tokens = yaml_scalar([attempt / "gen_config.yaml"], "max_num_tokens")
    model_payload = None
    if with_model:
        from model_view import build_model_payload, write_model

        model_payload = build_model_payload(
            requests, engine, tuple(meta.get("window", (None, None))), budgets,
            {
                "run": run_name(attempt), "attempt": attempt.name, "hour": hour,
                "mode": mode, "gpus": total_gpu, "source": "csv", "log_tz": meta.get("log_tz"),
                "selection": "hourly_request_cohorts" if hour else "whole_attempt",
            },
        )
        write_model(out, model_payload)
    counts = render_report(out, attempt, hour, requests, engine, notes, mode, budgets, total_gpu, gpu_how,
                           ctx_max_batch_size, gen_max_num_tokens, write_tables=False, model_payload=model_payload)
    print(f"re-rendered: {len(requests)} requests, {counts} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
