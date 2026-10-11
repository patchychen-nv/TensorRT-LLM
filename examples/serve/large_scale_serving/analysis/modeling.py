# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Empirical models over parsed request and engine tables, without rereading logs.

Models describe observed conditions. Session spans do not identify completed
agent jobs, and neither token slots nor device timeline spans measure GPU work.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator

from common import stats
from model_fit import fit_linear

__all__ = ["build_models"]

_FEATURES = {
    "new_input_tokens": ("_model_isl_new", "tokens", False),
    "total_input_tokens": ("isl_total", "tokens", False),
    "kv_transfer_bytes": ("kv_transfer_bytes", "bytes", False),
    "ctx_tokens_per_rank": ("ctx_tokens_mean", "tokens/rank", False),
    "decode_requests_per_rank": ("decode_requests_sum", "requests/rank", True),
    "decode_slots_per_rank": ("gen_tokens_sum", "slots/rank", True),
    "cached_kv_tokens_per_rank": ("cached_kv_tokens_sum", "tokens/rank", True),
}
_SCHEDULING_FIELDS = (
    "ranks",
    "idle_ranks",
    "busy_ranks",
    "token_budget_util",
    "batch_occupancy",
    "total_budget_util",
    "ctx_tokens_rank_skew",
    "decode_requests_rank_skew",
    "kv_cache_util_mean",
    "kv_pool_filled_mean",
    "paused_requests_sum",
    "host_step_ms_mean",
    "device_step_ms_mean",
    "device_step_rank_skew",
)


def _number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) else None


def _nonnegative(value: object) -> float | None:
    number = _number(value)
    return number if number is not None and number >= 0 else None


def _interval(row: dict) -> tuple[float, float] | None:
    start, end = _number(row.get("started_at")), _number(row.get("finished_at"))
    if row.get("status") != "completed" or start is None or end is None or end < start:
        return None
    return start, end


def _window(requests: list[dict], window: tuple) -> tuple[float, float] | None:
    low, high = (_number(value) for value in window)
    intervals = [_interval(row) for row in requests]
    valid = [interval for interval in intervals if interval is not None]
    if low is None and valid:
        low = min(start for start, _ in valid)
    if high is None and valid:
        high = max(end for _, end in valid)
    return (low, high) if low is not None and high is not None and high > low else None


def _request_exclusion(row: dict, window: tuple[float, float] | None) -> str | None:
    if row.get("status") != "completed":
        return "not_completed"
    interval = _interval(row)
    if interval is None:
        return "invalid_or_missing_interval"
    if window is None:
        return "no_valid_window"
    if interval[0] < window[0] or interval[1] > window[1] or interval[0] >= window[1]:
        return "not_fully_contained_in_window"
    return None


def _model(
    rows: list[dict],
    identifier: str,
    label: str,
    group: dict,
    names: list[str],
    target: str,
    window: tuple[float, float] | None,
    *,
    request: bool,
    invalid_device_iters: set | None = None,
) -> dict:
    exclusions: Counter = Counter()

    def samples() -> Iterator[tuple[float, tuple[float, ...], float]]:
        for row in rows:
            reason = _request_exclusion(row, window) if request else None
            if not request and (row.get("worker"), row.get("engine_instance"), row.get("iter")) in (
                invalid_device_iters or ()
            ):
                reason = "incomplete_rank_device_timing"
            if reason:
                exclusions[reason] += 1
                continue
            stamp = _number(row.get("started_at" if request else "timestamp"))
            observed = _nonnegative(row.get(target))
            predictors = []
            for name in names:
                field, _, per_rank = _FEATURES[name]
                value = _nonnegative(row.get(field))
                if per_rank:
                    ranks = _nonnegative(row.get("ranks"))
                    value = value / ranks if value is not None and ranks else None
                predictors.append(value)
            if stamp is None or observed is None or any(value is None for value in predictors):
                exclusions["missing_nonfinite_or_negative_model_value"] += 1
                continue
            yield stamp, tuple(predictors), observed

    fit = fit_linear(samples(), names, target=target, target_unit="ms")
    return {
        "id": identifier,
        "label": label,
        "group": group,
        "feature_units": {name: _FEATURES[name][1] for name in names},
        "filtering": {
            "n_candidates": len(rows),
            "n_excluded": sum(exclusions.values()),
            "excluded_counts": dict(exclusions),
        },
        "notes": [],
        "fit": fit,
    }


def _request_models(
    requests: list[dict], window: tuple[float, float] | None, mode: str
) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in requests:
        groups[
            (
                row.get("ctx_instance"),
                row.get("_model_isl_source", "missing"),
                row.get("model"),
                row.get("route"),
            )
        ].append(row)
    models = []
    for (worker, source, model, route), rows in groups.items():
        group = {"ctx_instance": worker, "token_source": source, "model": model, "route": route}
        result = _model(
            rows,
            "prefill",
            "Request prefill latency",
            group,
            ["new_input_tokens", "total_input_tokens"],
            "prefill_ms",
            window,
            request=True,
        )
        result["notes"] = [
            "Prefill spans first scheduled to first token; chunking and scheduling gaps may contribute.",
            "Block-derived new input is a block-granularity allocation proxy.",
        ]
        if worker is None:
            result["notes"].append(
                "Context worker identity is unavailable; this group may mix workers."
            )
        if model is None:
            result["notes"].append("Model identity is unavailable; this group may mix models.")
        models.append(result)
    if mode == "disagg":
        result = _model(
            requests,
            "kv_transfer",
            "KV transfer latency",
            {"path": "aggregate"},
            ["kv_transfer_bytes"],
            "kv_transfer_ms",
            window,
            request=True,
        )
        result["notes"] = [
            "Generation worker identity is absent; transfer paths are pooled.",
            "The coefficient describes observed transfer latency, not physical link bandwidth.",
        ]
        models.append(result)
    return models


def _kind(row: dict, role: str) -> str:
    prefill_value = _nonnegative(row.get("ctx_tokens_mean")) if role != "gen" else 0
    decode_value = _nonnegative(row.get("decode_requests_sum")) if role != "ctx" else 0
    if prefill_value is None or decode_value is None:
        return "unknown"
    prefill, decode = prefill_value > 0, decode_value > 0
    return (
        "mixed"
        if prefill and decode
        else "prefill"
        if prefill
        else "decode_only"
        if decode
        else "idle"
    )


def _means(rows: list[dict]) -> dict:
    output = {}
    for field in _SCHEDULING_FIELDS:
        count, total = 0, 0.0
        for row in rows:
            value = _nonnegative(row.get(field))
            if value is not None:
                count += 1
                total += value
        output[field] = {"n": count, "mean": total / count if count else None}
    return output


def _engine_models(engine: dict, window: tuple[float, float] | None) -> tuple[list, list, dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    scheduling_groups: dict[tuple, list[dict]] = defaultdict(list)
    exclusions: Counter = Counter()
    reuse: dict[tuple, list[float]] = defaultdict(lambda: [0, 0.0, 0.0])
    invalid_device_iters: dict[str, set] = defaultdict(set)
    invalid_cache_iters: dict[str, set] = defaultdict(set)
    expected_ranks: dict[tuple, float] = defaultdict(float)
    for role in ("ctx", "gen", "mixed"):
        coverage: dict[tuple, list[int]] = defaultdict(lambda: [0, 0, 0])
        for row in engine.get(f"{role}_iters", []):
            stamp = _number(row.get("timestamp"))
            if window is None or stamp is None or not window[0] <= stamp < window[1]:
                exclusions["outside_window_or_missing_timestamp"] += 1
                continue
            if (
                row.get("worker") is None
                or row.get("engine_instance") is None
                or not _nonnegative(row.get("ranks"))
            ):
                exclusions["missing_identity_or_invalid_rank_count"] += 1
                continue
            key = (role, row["worker"], row["engine_instance"])
            expected_ranks[key] = max(expected_ranks[key], row["ranks"])
            scheduling_groups[key].append(row)
            groups[(*key, row["ranks"], _kind(row, role))].append(row)
        for row in engine.get(f"{role}_rank", []):
            stamp = _number(row.get("timestamp"))
            if window is None or stamp is None or not window[0] <= stamp < window[1]:
                continue
            iteration = (row.get("worker"), row.get("engine_instance"), row.get("iter"))
            counters = coverage[iteration]
            counters[0] += 1
            counters[1] += _nonnegative(row.get("device_step_ms")) is not None
            counters[2] += _nonnegative(row.get("cached_kv_tokens")) is not None
            reused = _nonnegative(row.get("kv_reused_blocks_delta"))
            missed = _nonnegative(row.get("kv_missed_blocks_delta"))
            if reused is not None and missed is not None:
                totals = reuse[(role, row.get("worker"), row.get("engine_instance"))]
                totals[0] += 1
                totals[1] += reused
                totals[2] += missed
            else:
                exclusions["rank_kv_delta_missing_nonfinite_or_negative"] += 1
        for row in engine.get(f"{role}_iters", []):
            stamp = _number(row.get("timestamp"))
            if window is None or stamp is None or not window[0] <= stamp < window[1]:
                continue
            iteration = (row.get("worker"), row.get("engine_instance"), row.get("iter"))
            expected = expected_ranks.get((role, *iteration[:2]))
            observed, device_count, cache_count = coverage.get(iteration, (0, 0, 0))
            if (
                not expected
                or row.get("ranks") != expected
                or observed != expected
                or device_count != expected
            ):
                invalid_device_iters[role].add(iteration)
            if _nonnegative(row.get("cached_kv_tokens_sum")) is not None and (
                not expected or observed != expected or cache_count != expected
            ):
                invalid_cache_iters[role].add(iteration)
        del coverage
    models = []
    for (role, worker, instance, ranks, kind), rows in groups.items():
        if kind in ("idle", "unknown"):
            exclusions[f"{kind}_iteration_not_fitted"] += len(rows)
            continue
        names = ["ctx_tokens_per_rank"] if kind in ("prefill", "mixed") else []
        if kind in ("decode_only", "mixed"):
            names += ["decode_requests_per_rank", "decode_slots_per_rank"]
        cache_coverage = sum(
            _nonnegative(row.get("cached_kv_tokens_sum")) is not None
            and (worker, instance, row.get("iter")) not in invalid_cache_iters[role]
            for row in rows
        )
        if kind in ("decode_only", "mixed") and cache_coverage == len(rows):
            names.append("cached_kv_tokens_per_rank")
        group = {
            "role": role,
            "worker": worker,
            "engine_instance": instance,
            "ranks": ranks,
            "iteration_kind": kind,
        }
        result = _model(
            rows,
            "device_step",
            "Engine device step span",
            group,
            names,
            "device_step_ms_mean",
            window,
            request=False,
            invalid_device_iters=invalid_device_iters[role],
        )
        result["notes"] = [
            "Device span is already aligned to its iteration and is a GPU timeline span, not busy time.",
            "Decode slots include MTP/CUDA graph padding; they are not accepted output tokens.",
            f"Cached-context coverage: {cache_coverage}/{len(rows)}; used only with complete coverage.",
            "Cached-context coverage requires paired rank values for the complete inferred rank count.",
            "Partial rank timing targets are excluded using the accompanying rank table.",
            "Expected ranks use the maximum observed count per worker/lifetime; configuration is not verified.",
        ]
        models.append(result)
    scheduling = []
    for key, rows in scheduling_groups.items():
        count, reused, missed = reuse[key]
        scheduling.append(
            {
                "group": dict(zip(("role", "worker", "engine_instance"), key)),
                "n_iterations": len(rows),
                "weighting": "iteration-weighted; not wall-time GPU utilization",
                "iteration_kinds": dict(Counter(_kind(row, key[0]) for row in rows)),
                "metrics": _means(rows),
                "kv_reuse": {
                    "n_rank_samples": count,
                    "reused_blocks": reused,
                    "missed_blocks": missed,
                    "hit_rate": reused / (reused + missed) if reused + missed else None,
                    "scope": "All iteration kinds for this worker/lifetime in the window; paired rank deltas only.",
                },
            }
        )
    return models, scheduling, dict(exclusions)


def _flow(requests: list[dict], window: tuple[float, float] | None) -> dict:
    result = {"status": "unavailable", "reason": "No positive observation window."}
    if window is None:
        return result
    low, high = window
    intervals = [interval for row in requests if (interval := _interval(row)) is not None]
    contained = [(start, end) for start, end in intervals if low <= start < high and end <= high]
    area = sum(max(0.0, min(end, high) - max(start, low)) for start, end in intervals)
    duration = high - low
    mean_latency = (
        sum(end - start for start, end in contained) / len(contained) if contained else None
    )
    rate = len(contained) / duration
    return {
        "status": "available",
        "window_start": low,
        "window_end": high,
        "duration_s": duration,
        "n_observed_completed": len(intervals),
        "n_fully_contained": len(contained),
        "observed_request_seconds": area,
        "observed_mean_concurrency": area / duration,
        "contained_request_rate_per_s": rate,
        "contained_mean_latency_s": mean_latency,
        "contained_rate_times_latency": rate * mean_latency if mean_latency is not None else None,
        "notes": [
            "Request-seconds integrate completed observed intervals clipped to this window exactly.",
            "Unanswered intervals are excluded, so this is not total system concurrency.",
            "Hourly trace cohorts omit prior-hour carry-in and may omit later responses.",
            "Contained rate times mean latency is a subset accounting identity, not a Little-law validation.",
        ],
    }


def _union_duration(intervals: Iterable[tuple[float, float]]) -> float:
    duration, last_end = 0.0, None
    for start, end in sorted(intervals):
        duration += end - start if last_end is None else max(0.0, end - max(start, last_end))
        last_end = end if last_end is None else max(last_end, end)
    return duration


def _sessions(requests: list[dict], window: tuple[float, float] | None) -> dict:
    groups: dict[str, list[tuple[float, float]]] = defaultdict(list)
    missing, excluded = 0, 0
    for row in requests:
        session = row.get("session")
        if session in (None, "", "_no_session"):
            missing += 1
            continue
        interval = _interval(row)
        if (
            interval is None
            or window is None
            or interval[1] <= window[0]
            or interval[0] >= window[1]
        ):
            excluded += 1
            continue
        groups[str(session)].append((max(interval[0], window[0]), min(interval[1], window[1])))
    metrics: dict[str, list[float]] = defaultdict(list)
    for intervals in groups.values():
        span = max(end for _, end in intervals) - min(start for start, _ in intervals)
        total = sum(end - start for start, end in intervals)
        union = _union_duration(intervals)
        for name, value in (
            ("observed_span_s", span),
            ("request_duration_sum_s", total),
            ("request_union_s", union),
            ("idle_gap_s", max(0.0, span - union)),
            ("overlap_s", max(0.0, total - union)),
        ):
            metrics[name].append(value)
    return {
        "status": "available" if groups else "unavailable",
        "n_known_sessions": len(groups),
        "n_complete_intervals": sum(len(intervals) for intervals in groups.values()),
        "n_missing_session": missing,
        "n_excluded_intervals": excluded,
        "metrics": {name: stats(values) for name, values in metrics.items()},
        "boundary_status": "Session and job completion unknown; intervals are window-clipped.",
        "notes": [
            "Only completed observed requests with known session keys contribute.",
            "Union duration handles concurrent turns; gaps cannot be attributed to tools.",
            "Missing or out-of-window turns can inflate apparent gaps or truncate session spans.",
            "Session identity may span jobs or merge sanitized keys; no completed-job counts are inferred.",
        ],
    }


def build_models(
    requests: list[dict], engine: dict, window: tuple, budgets: dict, metadata: dict
) -> dict:
    """Build empirical fits and descriptive flow/session summaries.

    Args:
        requests: Parsed rows enriched with ``_model_isl_new`` and
            ``_model_isl_source`` by the report's effective-input policy.
        engine: Parsed ``ctx``, ``gen``, and ``mixed`` rank/iteration tables.
        window: Epoch-second start/end; missing endpoints use completed requests.
        budgets: Configuration snapshot, preserved for interpretation.
        metadata: Deployment mode, GPU count, source (raw/csv), hour, optional run.

    Returns:
        A JSON-serializable schema-1 object. Insufficient data yields explicit
        unavailable fits; no predictions outside the training domain are made.
    """
    resolved = _window(requests, window)
    engine_models, scheduling, engine_exclusions = _engine_models(engine, resolved)
    request_exclusions = Counter(
        reason for row in requests if (reason := _request_exclusion(row, resolved))
    )
    limitations = [
        "Fits are observational associations under this configuration, not causal capacity or scaling forecasts.",
        "Request fits use completed intervals fully inside the same window as engine observations.",
        "Request and engine samples are modeled separately; no unobserved request-to-iteration join is assumed.",
        "Cohort truncation, recovered arrival clocks and coarse engine timestamps limit temporal interpretation.",
        "No job boundaries, tool events or branch dependencies exist in these tables.",
    ]
    if metadata.get("source") == "csv":
        limitations.append(
            "Legacy CSV timestamps may be quantized to seconds; numeric columns are rounded. Precision is lost."
        )
    if len(metadata.get("source_windows", [])) > 1:
        limitations.append(
            "Merged windows use their enclosing time span; gaps between source windows are not "
            "observed engine idle time. Request occupancy covers only the supplied completed cohorts."
        )
    return {
        "schema": 1,
        "metadata": {
            **metadata,
            "budgets": budgets,
            "window": list(resolved) if resolved else None,
        },
        "quality": {
            "n_requests": len(requests),
            "request_excluded_counts": dict(request_exclusions),
            "token_sources": dict(
                Counter(row.get("_model_isl_source", "missing") for row in requests)
            ),
            "engine_excluded_counts": engine_exclusions,
        },
        "request_models": _request_models(requests, resolved, metadata.get("mode", "unknown")),
        "engine_models": engine_models,
        "scheduling": scheduling,
        "flow": _flow(requests, resolved),
        "sessions": _sessions(requests, resolved),
        "limitations": limitations,
        "job_metrics": {
            "status": "unavailable",
            "reason": "Requires job boundaries, tool events and fork/join dependencies; session completion is unknown.",
        },
    }
