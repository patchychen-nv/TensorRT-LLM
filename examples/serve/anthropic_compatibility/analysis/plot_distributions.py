#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate pooled per-turn distribution PNG and HTML from audit timelines.

Usage:
    python3 plot_distributions.py \
        --series "DeepSeek V4=/path/to/v4/analysis/timeline.csv" \
        --series "GLM 5.2=/path/to/glm/analysis/timeline.csv" \
        --out-dir /path/to/output \
        --title "SWE-bench Verified 100"
"""

from __future__ import annotations

import argparse
import csv
import html
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


COLORS = [
    "#2369bd",
    "#e36a2e",
    "#1b9e77",
    "#8e63ce",
    "#d6a000",
    "#d43f70",
]
SURFACE = "#fbfbfa"
GRID = "#deddd8"
INK = "#171717"
MUTED = "#66645f"


@dataclass(frozen=True)
class Metric:
    columns: tuple[str, ...]
    title: str
    xlabel: str
    scale: float = 1.0
    ratio: bool = False
    percentile_format: str = ",.0f"
    compact_thousands: bool = False
    matched_tool_loop_only: bool = False
    first_agent_turn_only: bool = False
    session_turn_count: bool = False
    measurable_decode_only: bool = False


# A decode rate is a token count divided by the window between the first and
# last generated token. When a response arrives as a single chunk that window
# collapses toward zero and the quotient explodes -- one such turn has been
# observed at 66,860 tok/s from a 0.344ms window. That is an unmeasurable rate,
# not a fast one, and leaving it in drags the pooled mean far above every
# percentile. Real windows sit three orders of magnitude above this threshold
# (the next smallest observed is 134ms), so it only catches degenerate ones.
MIN_MEASURABLE_DECODE_MS = 1.0


METRICS = [
    Metric(
        ("isl_total",),
        "Total ISL",
        "Tokens",
        percentile_format=".0f",
        compact_thousands=True,
    ),
    Metric(
        ("isl_cached",),
        "Cached ISL",
        "Tokens",
        percentile_format=".0f",
        compact_thousands=True,
    ),
    Metric(
        ("isl_new",),
        "New / uncached ISL",
        "Tokens",
        percentile_format=".0f",
        compact_thousands=True,
    ),
    Metric(
        ("isl_total",),
        "First agent-turn ISL (session / task)",
        "Tokens",
        percentile_format=".0f",
        compact_thousands=True,
        first_agent_turn_only=True,
    ),
    Metric(("osl_model_tokens",), "Output length (OSL)", "Tokens"),
    Metric(
        (),
        "Completed turns per session / task",
        "Turns",
        percentile_format=".0f",
        session_turn_count=True,
    ),
    Metric(
        ("server_ttft_ms",),
        "Time to first token (TTFT)",
        "Seconds",
        1000.0,
        percentile_format=".2f",
    ),
    Metric(
        ("server_total_ms",),
        "Total server latency",
        "Seconds",
        1000.0,
        percentile_format=".2f",
    ),
    Metric(
        ("output_tps_per_user",),
        "Decode throughput (TPS / user)",
        "Tokens / second",
        percentile_format=".1f",
        measurable_decode_only=True,
    ),
    Metric(
        ("actual_cache_hit_ratio",),
        "Actual cache-hit ratio",
        "Percent",
        0.01,
        ratio=True,
        percentile_format=".1f",
    ),
    Metric(
        ("tool_loop_gap_ms", "tool_loop_gap_min_ms"),
        "Tool-loop gap",
        "Seconds",
        1000.0,
        percentile_format=".2f",
        matched_tool_loop_only=True,
    ),
]


def _parse_series(value: str) -> tuple[str, Path]:
    try:
        label, path = value.split("=", maxsplit=1)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"invalid series {value!r}; expected LABEL=/path/to/timeline.csv"
        ) from error
    if not label.strip() or not path.strip():
        raise argparse.ArgumentTypeError(
            f"invalid series {value!r}; label and timeline path must be non-empty"
        )
    return label.strip(), Path(path).expanduser()


def _read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"timeline CSV not found: {path}")
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _completed_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [
        row
        for row in rows
        if row.get("status", "").lower() == "completed"
    ]


def _number(row: dict[str, str], column: str) -> float | None:
    value = row.get(column)
    if value in ("", "None", None):
        return None
    try:
        number = float(value)
    except ValueError:
        return None
    return number if np.isfinite(number) else None


def _numbers(rows: list[dict[str, str]], column: str) -> np.ndarray:
    values = [
        number
        for row in rows
        if (number := _number(row, column)) is not None
    ]
    return np.asarray(values, dtype=float)


def _sum_column(rows: list[dict[str, str]], column: str) -> float:
    return float(np.sum(_numbers(rows, column)))


def _measurable_decode(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Drop turns whose decode window is too short to yield a meaningful rate.

    See ``MIN_MEASURABLE_DECODE_MS``.
    """
    kept = []
    for row in rows:
        decode_ms = _number(row, "server_decode_ms")
        if decode_ms is not None and decode_ms >= MIN_MEASURABLE_DECODE_MS:
            kept.append(row)
    return kept


def _format_tokens(value: float) -> str:
    if value >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if value >= 1_000:
        return f"{value / 1_000:.1f}K"
    return f"{value:,.0f}"


def _format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "N/A"
    if seconds >= 3600:
        return f"{seconds / 3600:.2f}h"
    if seconds >= 60:
        return f"{seconds / 60:.1f}min"
    return f"{seconds:.1f}s"


def _format_percent(numerator: float, denominator: float) -> str:
    if denominator <= 0:
        return "N/A"
    return f"{numerator / denominator * 100:.1f}%"


def _parse_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _trace_wall_clock_seconds(rows: list[dict[str, str]]) -> float | None:
    starts = [
        timestamp
        for row in rows
        if (timestamp := _parse_timestamp(row.get("started_at"))) is not None
    ]
    finishes = [
        timestamp
        for row in rows
        if (timestamp := _parse_timestamp(row.get("finished_at"))) is not None
    ]
    if not starts or not finishes:
        return None
    return max(0.0, (max(finishes) - min(starts)).total_seconds())


def _is_warm_turn(row: dict[str, str]) -> bool:
    turn_index = _number(row, "session_turn_index")
    return turn_index is not None and turn_index > 1


def _percentiles(
    values: np.ndarray,
    percentiles: tuple[int, ...],
    number_format: str,
    unit: str,
) -> str:
    if not len(values):
        return "N/A"
    formatted = [
        format(value, number_format)
        for value in np.percentile(values, percentiles)
    ]
    return " / ".join(formatted) + unit


def _tool_gap_values(rows: list[dict[str, str]]) -> np.ndarray:
    values = []
    for row in rows:
        if row.get("tool_loop_matched", "").lower() != "true":
            continue
        gap = _number(row, "tool_loop_gap_ms")
        if gap is None:
            gap = _number(row, "tool_loop_gap_max_ms")
        if gap is not None and gap >= 0:
            values.append(gap)
    return np.asarray(values, dtype=float)


def _summarize_run(rows: list[dict[str, str]]) -> dict[str, str]:
    completed = _completed_rows(rows)
    warm = [row for row in completed if _is_warm_turn(row)]

    total_isl = _sum_column(completed, "isl_total")
    cached_isl = _sum_column(completed, "isl_cached")
    new_isl = _sum_column(completed, "isl_new")
    output_tokens = _sum_column(completed, "osl_model_tokens")
    warm_isl = _sum_column(warm, "isl_total")
    warm_cached = _sum_column(warm, "isl_cached")
    tool_calls = _sum_column(completed, "tool_call_count")
    tool_results = _sum_column(completed, "tool_result_count")
    tool_result_errors = _sum_column(completed, "tool_result_error_count")
    tool_gaps_ms = _tool_gap_values(completed)

    statuses = [row.get("status", "").lower() for row in rows]
    failed_requests = sum(
        status in {"error", "failed", "stream_error"}
        for status in statuses
    )
    cancelled_requests = sum("cancel" in status for status in statuses)

    return {
        "Sessions": f"{len({row.get('session_id') for row in rows if row.get('session_id')}):,}",
        "API requests": f"{len(rows):,}",
        "Completed turns": f"{len(completed):,}",
        "Failed requests": f"{failed_requests:,}",
        "Cancelled requests": f"{cancelled_requests:,}",
        "Trace elapsed time (max finish - min start)": _format_duration(
            _trace_wall_clock_seconds(rows)
        ),
        "Summed request latency (sum server_total_ms)": _format_duration(
            _sum_column(completed, "server_total_ms") / 1000.0
        ),
        "Total processed tokens": _format_tokens(total_isl + output_tokens),
        "Total input ISL": _format_tokens(total_isl),
        "Cache-read ISL": (
            f"{_format_tokens(cached_isl)} ({_format_percent(cached_isl, total_isl)})"
        ),
        "New/computed ISL": (
            f"{_format_tokens(new_isl)} ({_format_percent(new_isl, total_isl)})"
        ),
        "Model output tokens": _format_tokens(output_tokens),
        "Warm cache-hit ratio": _format_percent(warm_cached, warm_isl),
        "Warm TTFT p50 / p95": _percentiles(
            _numbers(warm, "server_ttft_ms"),
            (50, 95),
            ",.0f",
            "ms",
        ),
        "Decode TPS/user p50": _percentiles(
            _numbers(_measurable_decode(completed), "output_tps_per_user"),
            (50,),
            ".1f",
            "",
        ),
        "Tool calls": f"{tool_calls:,.0f}",
        "Total tool-call time": _format_duration(
            float(np.sum(tool_gaps_ms)) / 1000.0
        ),
        "Tool-loop gap p50 / p95": _percentiles(
            tool_gaps_ms / 1000.0,
            (50, 95),
            ".2f",
            "s",
        ),
        "Tool-result error rate": (
            f"{_format_percent(tool_result_errors, tool_results)} "
            f"({tool_result_errors:,.0f}/{tool_results:,.0f})"
        ),
    }


def _markdown_escape(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def _write_run_summary(
    path: Path,
    title: str,
    series_rows: dict[str, list[dict[str, str]]],
) -> None:
    summaries = {
        label: _summarize_run(rows)
        for label, rows in series_rows.items()
    }
    metric_names = list(next(iter(summaries.values())))
    labels = list(summaries)
    lines = [
        f"# {title} — run summary",
        "",
        "| Metric | " + " | ".join(_markdown_escape(label) for label in labels) + " |",
        "|---|" + "|".join("---:" for _ in labels) + "|",
    ]
    for metric_name in metric_names:
        values = [
            _markdown_escape(summaries[label][metric_name])
            for label in labels
        ]
        lines.append(
            f"| {_markdown_escape(metric_name)} | "
            + " | ".join(values)
            + " |"
        )
    lines.extend(
        [
            "",
            "Token and performance aggregates include completed turns only. "
            "`Total processed tokens = total input ISL + model output tokens`.",
            "",
            "`New/computed ISL` includes uncached input and cache-creation input. "
            "Warm metrics exclude the first turn of every session.",
            "",
            "`Trace elapsed time = max(finished_at) - min(started_at)` across all "
            "requests. It includes client, tool, and idle gaps; overlapping requests "
            "occupy the same wall-clock interval.",
            "",
            "`Summed request latency = sum(server_total_ms)` across completed requests. "
            "It excludes gaps outside the server and counts overlapping request latency "
            "more than once.",
            "",
            "`Total tool-call time` sums one matched response-to-next-request gap "
            "per tool-calling turn, so parallel tool calls in one turn are not double-counted.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _numeric(rows: list[dict[str, str]], metric: Metric) -> np.ndarray:
    if metric.session_turn_count:
        counts: dict[str, int] = {}
        for row in rows:
            session_id = row.get("session_id")
            if session_id:
                counts[session_id] = counts.get(session_id, 0) + 1
        return np.asarray(list(counts.values()), dtype=float)

    if metric.first_agent_turn_only:
        first_rows: dict[str, tuple[float, dict[str, str]]] = {}
        for row in rows:
            session_id = row.get("session_id")
            turn_index = _number(row, "session_turn_index")
            tool_count = _number(row, "tool_definition_count")
            if (
                not session_id
                or turn_index is None
                or tool_count is None
                or tool_count <= 0
            ):
                continue
            previous = first_rows.get(session_id)
            if previous is None or turn_index < previous[0]:
                first_rows[session_id] = (turn_index, row)
        rows = [item[1] for item in first_rows.values()]

    values: list[float] = []
    for row in rows:
        if (
            metric.matched_tool_loop_only
            and row.get("tool_loop_matched", "").lower() != "true"
        ):
            continue

        if metric.measurable_decode_only:
            decode_ms = _number(row, "server_decode_ms")
            if decode_ms is None or decode_ms < MIN_MEASURABLE_DECODE_MS:
                continue

        value = next(
            (
                row.get(column)
                for column in metric.columns
                if row.get(column) not in ("", "None", None)
            ),
            None,
        )
        if value is None:
            continue
        try:
            number = float(value) / metric.scale
        except ValueError:
            continue
        if np.isfinite(number) and number >= 0:
            values.append(number)
    return np.asarray(values, dtype=float)


def _bin_edges(arrays: list[np.ndarray], ratio: bool) -> np.ndarray:
    nonempty = [array for array in arrays if len(array)]
    if not nonempty:
        return np.linspace(0.0, 1.0, 41)
    combined = np.concatenate(nonempty)
    if ratio:
        return np.linspace(0.0, 100.0, 41)

    positive = combined[combined > 0]
    if not len(positive):
        return np.linspace(0.0, 1.0, 41)

    low = float(positive.min())
    high = float(positive.max())
    if high <= low:
        return np.linspace(0.0, high * 1.1, 41)

    logarithmic = np.geomspace(low, high, 43)
    if np.any(combined == 0):
        return np.concatenate(([0.0], logarithmic))
    return logarithmic


def _summary_text(data: dict[str, np.ndarray], metric: Metric) -> str:
    lines = []
    for label, values in data.items():
        if not len(values):
            continue
        mean = np.mean(values)
        p50, p75, p99 = np.percentile(values, [50, 75, 99])

        def format_value(value: float) -> str:
            if metric.compact_thousands:
                if value < 1000.0:
                    return f"{value:.0f}"
                return f"{value / 1000.0:.1f}K"
            return format(value, metric.percentile_format)

        lines.append(
            f"{label:<14} n={len(values):>5}  "
            f"mean={format_value(mean)}  p50={format_value(p50)}  "
            f"p75={format_value(p75)}  p99={format_value(p99)}"
        )
    return "\n".join(lines)


def _style_axes(ax: plt.Axes, ecdf_ax: plt.Axes) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", labelsize=8.5, colors=MUTED)
    ecdf_ax.tick_params(axis="y", labelsize=8.5, colors=MUTED)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    for spine in ecdf_ax.spines.values():
        spine.set_visible(False)


def _plot_distribution(
    ax: plt.Axes,
    data: dict[str, np.ndarray],
    colors: dict[str, str],
    metric: Metric,
) -> None:
    arrays = list(data.values())
    edges = _bin_edges(arrays, metric.ratio)

    for label, values in data.items():
        if not len(values):
            continue
        weights = np.full(len(values), 100.0 / len(values))
        ax.hist(
            values,
            bins=edges,
            weights=weights,
            histtype="stepfilled",
            alpha=0.24,
            linewidth=1.1,
            edgecolor=colors[label],
            color=colors[label],
        )

    ecdf_ax = ax.twinx()
    for label, values in data.items():
        if not len(values):
            continue
        ordered = np.sort(values)
        cumulative = np.arange(1, len(ordered) + 1) / len(ordered) * 100.0
        ecdf_ax.plot(ordered, cumulative, color=colors[label], linewidth=1.7)

    if metric.ratio:
        ax.set_xlim(0, 100)
    else:
        nonempty = [array for array in arrays if len(array)]
        combined = np.concatenate(nonempty) if nonempty else np.array([])
        positive = combined[combined > 0]
        if len(positive):
            linear_threshold = max(float(np.percentile(positive, 1)) / 2.0, 1e-3)
            ax.set_xscale("symlog", linthresh=linear_threshold)

    ax.set_title(metric.title, loc="left", fontsize=12.5, fontweight="bold", color=INK)
    ax.set_xlabel(metric.xlabel, fontsize=9.5, color=MUTED)
    ax.set_ylabel("Requests per bin (%)", fontsize=9.5, color=MUTED)
    ecdf_ax.set_ylabel("ECDF (%)", fontsize=9.5, color=MUTED)
    ecdf_ax.set_ylim(0, 102)
    _style_axes(ax, ecdf_ax)

    ax.text(
        0.015,
        0.97,
        _summary_text(data, metric),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.8,
        color=INK,
        fontfamily="monospace",
        linespacing=1.45,
        bbox={
            "boxstyle": "round,pad=0.32",
            "facecolor": SURFACE,
            "edgecolor": GRID,
            "alpha": 0.93,
        },
        zorder=10,
    )


def _write_html(
    path: Path,
    png_name: str,
    summary_name: str,
    title: str,
) -> None:
    escaped_title = html.escape(title)
    escaped_png_name = html.escape(png_name, quote=True)
    escaped_summary_name = html.escape(summary_name, quote=True)
    document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escaped_title}</title>
  <style>
    body {{ margin: 0; padding: 24px; background: #f4f6f8;
            color: #172033; font-family: system-ui, sans-serif; }}
    main {{ max-width: 1500px; margin: 0 auto; }}
    header {{ display: flex; align-items: center; justify-content: space-between;
              gap: 16px; margin-bottom: 16px; }}
    h1 {{ margin: 0; font-size: clamp(20px, 3vw, 30px); }}
    nav {{ display: flex; gap: 8px; }}
    a {{ flex: none; padding: 9px 14px; border-radius: 8px; color: white;
         background: #2457a7; text-decoration: none; font-weight: 600; }}
    figure {{ margin: 0; padding: 12px; border: 1px solid #d8dee8;
              border-radius: 12px; background: white;
              box-shadow: 0 4px 16px rgb(23 32 51 / 8%); }}
    img {{ display: block; width: 100%; height: auto; }}
  </style>
</head>
<body>
  <main>
    <header>
      <h1>{escaped_title}</h1>
      <nav>
        <a href="{escaped_summary_name}">Run summary</a>
        <a href="{escaped_png_name}" download>Download PNG</a>
      </nav>
    </header>
    <figure>
      <img src="{escaped_png_name}" alt="{escaped_title}">
    </figure>
  </main>
</body>
</html>
"""
    path.write_text(document, encoding="utf-8")


def _plot(
    series_rows: dict[str, list[dict[str, str]]],
    output_png: Path,
    title: str,
) -> None:
    colors = {
        label: COLORS[index % len(COLORS)]
        for index, label in enumerate(series_rows)
    }
    figure, axes = plt.subplots(
        4,
        3,
        figsize=(22, 23),
        facecolor=SURFACE,
        gridspec_kw={"hspace": 0.44, "wspace": 0.30},
    )
    for ax, metric in zip(axes.flat, METRICS):
        data = {
            label: _numeric(rows, metric)
            for label, rows in series_rows.items()
        }
        _plot_distribution(ax, data, colors, metric)
    for ax in axes.flat[len(METRICS):]:
        figure.delaxes(ax)

    handles = [
        plt.Line2D([0], [0], color=colors[label], linewidth=3, label=label)
        for label in series_rows
    ]
    figure.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.976),
        ncol=min(len(handles), 4),
        frameon=False,
        fontsize=11,
    )
    figure.suptitle(
        f"{title} — pooled per-turn distributions",
        fontsize=18.5,
        fontweight="bold",
        color=INK,
        y=0.992,
    )
    figure.text(
        0.5,
        0.965,
        "Completed requests only; missing values are excluded, not replaced with zero. "
        "Histograms show request share per bin; lines show ECDF.",
        ha="center",
        va="top",
        fontsize=10,
        color=MUTED,
    )
    figure.savefig(output_png, dpi=180, bbox_inches="tight", facecolor=SURFACE)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--series",
        action="append",
        type=_parse_series,
        required=True,
        metavar="LABEL=TIMELINE_CSV",
        help="Label and timeline.csv path; repeat to compare runs",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--title", default="Anthropic audit trace")
    parser.add_argument("--output-stem", default="distribution_dashboard")
    args = parser.parse_args()

    timeline_rows = {
        label: _read_rows(timeline_path)
        for label, timeline_path in args.series
    }
    completed_series_rows = {
        label: _completed_rows(rows)
        for label, rows in timeline_rows.items()
    }
    completed_turns = sum(len(rows) for rows in completed_series_rows.values())
    if completed_turns == 0:
        raise ValueError("no completed turns found in the supplied timelines")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    output_png = args.out_dir / f"{args.output_stem}.png"
    output_html = args.out_dir / f"{args.output_stem}.html"
    output_summary = args.out_dir / "run_summary.md"
    _plot(completed_series_rows, output_png, args.title)
    _write_run_summary(output_summary, args.title, timeline_rows)
    _write_html(
        output_html,
        output_png.name,
        output_summary.name,
        f"{args.title} — pooled per-turn distributions",
    )
    print(
        f"Wrote {output_png}, {output_html}, and {output_summary} "
        f"from {completed_turns} completed turns"
    )


if __name__ == "__main__":
    main()
