#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate per-turn dashboard PNG from an analyze_audit.py output directory.

Usage:
    python3 plot_dashboard.py <analysis_dir> [--out dashboard.png] [--title "..."]

<analysis_dir> must contain timeline.csv (produced by analyze_audit.py).
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker


# ---------------------------------------------------------------------------
# Palette (reference instance – light mode)
# ---------------------------------------------------------------------------
CAT_COLORS_LIST = [
    "#2a78d6",  # blue
    "#eb6834",  # orange
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#e87ba4",  # magenta
    "#008300",  # green
    "#4a3aa7",  # violet
    "#e34948",  # red
]
# Secondary encoding: once the 8 hues are exhausted they repeat, so session
# N and session N+8 share a hue and are told apart by texture instead.
HATCHES = ["", "///", "...", "xxx", "\\\\\\"]
SURFACE      = "#fcfcfb"
GRID         = "#e1e0d9"
INK_MUT      = "#898781"
INK_SEC      = "#52514e"
INK_PRI      = "#0b0b0b"
ISL_CACHED_C = "#2a78d6"   # sequential blue step 450
ISL_NEW_C    = "#9ec5f4"   # sequential blue step 200


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def fv(r: dict, col: str, default: float = 0.0) -> float:
    v = r.get(col, "")
    if v in ("", "None", None):
        return default
    try:
        return float(v)
    except ValueError:
        return default


def add_stats(ax, data, fmt=".0f", unit="", exclude_zeros=False):
    arr = np.asarray(data, dtype=float)
    arr = arr[~np.isnan(arr)]
    if exclude_zeros:
        arr = arr[arr > 0]
    if len(arr) == 0:
        return
    mean = np.mean(arr)
    p50  = np.percentile(arr, 50)
    p75  = np.percentile(arr, 75)
    p99  = np.percentile(arr, 99)

    def fmtv(v):
        s = f"{v:{fmt}}"
        return s + unit if unit else s

    text = f"mean {fmtv(mean)}\n p50  {fmtv(p50)}\n p75  {fmtv(p75)}\n p99  {fmtv(p99)}"
    ax.text(
        0.985, 0.97, text,
        transform=ax.transAxes, ha="right", va="top",
        fontsize=7.5, color=INK_SEC, fontfamily="monospace", linespacing=1.6,
        bbox=dict(boxstyle="round,pad=0.35", facecolor=SURFACE,
                  edgecolor=GRID, linewidth=0.8, alpha=0.92),
        zorder=10,
    )


def session_style(index: int) -> tuple[str, str]:
    """Return (color, hatch) for the Nth session.

    Hues repeat past 8 sessions; the hatch changes on each wrap so a repeated
    hue is never ambiguous.
    """
    color = CAT_COLORS_LIST[index % len(CAT_COLORS_LIST)]
    hatch = HATCHES[(index // len(CAT_COLORS_LIST)) % len(HATCHES)]
    return color, hatch


def apply_hatches(bar_container, hatches) -> None:
    """Stamp per-bar hatch patterns; hatch ink is the surface color."""
    for patch, hatch in zip(bar_container.patches, hatches):
        if hatch:
            patch.set_hatch(hatch)
            patch.set_edgecolor(SURFACE)
            patch.set_linewidth(0)


def style_ax(ax, title, ylabel, yunit="", x_arr=None):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, fontsize=11, fontweight="bold", color=INK_PRI, pad=8, loc="left")
    ax.set_ylabel(ylabel + (f" ({yunit})" if yunit else ""), fontsize=9, color=INK_SEC)
    ax.set_xlabel("Global request index", fontsize=9, color=INK_MUT)
    ax.tick_params(colors=INK_MUT, labelsize=8)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.spines["bottom"].set_visible(True)
    ax.spines["bottom"].set_color(GRID)
    ax.spines["left"].set_visible(True)
    ax.spines["left"].set_color(GRID)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    if x_arr is not None:
        ax.set_xlim(x_arr.min() - 0.8, x_arr.max() + 0.8)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analysis_dir", type=Path,
                        help="Directory produced by analyze_audit.py (must contain timeline.csv)")
    parser.add_argument("--out", type=Path, default=None,
                        help="Output PNG path (default: <analysis_dir>/dashboard.png)")
    parser.add_argument("--title", default="",
                        help="Extra text appended to the figure super-title")
    args = parser.parse_args()

    csv_path = args.analysis_dir / "timeline.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"timeline.csv not found in {args.analysis_dir}")

    # Resolve the run label. The analysis dir is expected to sit inside the run
    # dir (the layout analyze_audit.py produces); if it does not, we cannot
    # infer a name, so demand an explicit --title rather than mislabel the plot.
    run_dir = args.analysis_dir.resolve().parent
    if args.title:
        run_label = args.title
    elif (run_dir / "anthropic_audit.jsonl").exists():
        run_label = run_dir.name
    else:
        raise SystemExit(
            f"error: cannot infer the run label — {run_dir}/anthropic_audit.jsonl "
            f"does not exist, so {args.analysis_dir} does not look like it sits "
            f"inside a run directory. Pass --title explicitly."
        )

    out_path = args.out or (args.analysis_dir / "dashboard.png")

    with csv_path.open() as f:
        rows = list(csv.DictReader(f))

    x    = np.array([int(r["global_request_index"]) for r in rows])
    sids = [r["session_id"] for r in rows]
    unique_sessions = list(dict.fromkeys(sids))

    session_styles = {s: session_style(i) for i, s in enumerate(unique_sessions)}
    bar_colors  = [session_styles[s][0] for s in sids]
    bar_hatches = [session_styles[s][1] for s in sids]

    isl_cached = np.array([fv(r, "isl_cached")       for r in rows])
    isl_new    = np.array([fv(r, "isl_new")           for r in rows])
    isl_total  = isl_cached + isl_new
    osl        = np.array([fv(r, "osl_model_tokens")  for r in rows])
    ttft       = np.array([fv(r, "server_ttft_ms")    for r in rows])
    total_ms   = np.array([fv(r, "server_total_ms")   for r in rows])
    hit_ratio  = np.array([fv(r, "actual_cache_hit_ratio") for r in rows])
    reuse_opp  = np.array([fv(r, "current_reuse_opportunity_ratio", float("nan")) for r in rows])
    tool_count = np.array([fv(r, "tool_call_count")   for r in rows])
    tool_gap   = np.array([fv(r, "tool_loop_gap_min_ms", float("nan")) for r in rows])
    tps_raw    = np.array([fv(r, "output_tps_per_user", float("nan")) for r in rows])
    TPS_CAP    = 200.0
    tps_capped = np.where(tps_raw > TPS_CAP, float("nan"), tps_raw)

    BAR_W = 0.85

    fig, axes = plt.subplots(
        4, 2, figsize=(18, 20), facecolor=SURFACE,
        gridspec_kw={"hspace": 0.55, "wspace": 0.18},
    )
    axes = axes.flatten()

    # ---- Panel 0: ISL breakdown --------------------------------------------
    # Cached vs new is a part-to-whole magnitude split, so it wears a sequential
    # blue ramp rather than the session hues. Session identity rides a thin
    # colour rug below the baseline, keeping the stack itself unambiguous.
    ax = axes[0]
    ax.bar(x, isl_cached, width=BAR_W, color=ISL_CACHED_C, label="Cached ISL", zorder=2)
    ax.bar(x, isl_new, width=BAR_W, bottom=isl_cached, color=ISL_NEW_C, label="New ISL", zorder=2)
    style_ax(ax, "ISL breakdown (cached + new)", "Tokens", "tokens", x)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(
        lambda v, _: f"{v/1000:.0f}k" if v >= 1000 else f"{v:.0f}"
    ))

    isl_top  = isl_total.max() * 1.08
    rug_h    = isl_top * 0.028
    rug_base = -rug_h * 1.9
    rug = ax.bar(x, rug_h, bottom=rug_base, width=BAR_W,
                 color=bar_colors, zorder=2)
    apply_hatches(rug, bar_hatches)
    ax.set_yticks([t for t in ax.get_yticks() if t >= 0])
    ax.set_ylim(rug_base - rug_h * 0.4, isl_top)
    ax.text(-0.012, rug_base + rug_h / 2, "session",
            transform=ax.get_yaxis_transform(), ha="right", va="center",
            fontsize=7, color=INK_MUT, style="italic")

    ax.legend(
        handles=[mpatches.Patch(color=ISL_CACHED_C, label="Cached ISL"),
                 mpatches.Patch(color=ISL_NEW_C,    label="New ISL")],
        fontsize=8, frameon=False, loc="upper left", labelcolor=INK_SEC,
    )
    add_stats(ax, isl_total, fmt=",.0f", unit=" tok")

    # ---- Panel 1: OSL ------------------------------------------------------
    ax = axes[1]
    bars = ax.bar(x, osl, width=BAR_W, color=bar_colors, zorder=2)
    apply_hatches(bars, bar_hatches)
    style_ax(ax, "Output length (OSL)", "Tokens", "tokens", x)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:.0f}"))
    add_stats(ax, osl, fmt=".0f", unit=" tok")

    # ---- Panel 2: TTFT -----------------------------------------------------
    ax = axes[2]
    ttft_sec = ttft / 1000.0
    bars = ax.bar(x, ttft_sec, width=BAR_W, color=bar_colors, zorder=2)
    apply_hatches(bars, bar_hatches)
    style_ax(ax, "Time to first token (TTFT)", "Seconds", "s", x)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:.0f}s"))
    for xi, yi in zip(x, ttft_sec):
        if yi > 30:
            ax.annotate(f"{yi:.0f}s", xy=(xi, yi), xytext=(xi, yi + 1),
                        ha="center", va="bottom", fontsize=7, color=INK_SEC)
    add_stats(ax, ttft_sec, fmt=".1f", unit="s")

    # ---- Panel 3: Total latency --------------------------------------------
    ax = axes[3]
    total_sec = total_ms / 1000.0
    bars = ax.bar(x, total_sec, width=BAR_W, color=bar_colors, zorder=2)
    apply_hatches(bars, bar_hatches)
    style_ax(ax, "Total server latency", "Seconds", "s", x)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:.0f}s"))
    add_stats(ax, total_sec, fmt=".1f", unit="s")

    # ---- Panel 4: Cache hit ratio ------------------------------------------
    ax = axes[4]
    bars = ax.bar(x, hit_ratio * 100, width=BAR_W, color=bar_colors, zorder=2)
    apply_hatches(bars, bar_hatches)
    style_ax(ax, "Actual cache-hit ratio", "Hit rate", "%", x)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    reuse_pct  = reuse_opp * 100
    valid_mask = ~np.isnan(reuse_pct)
    if valid_mask.any():
        ax.plot(x[valid_mask], reuse_pct[valid_mask],
                color=INK_SEC, linewidth=1.5, linestyle="--", alpha=0.6,
                label="Reuse opportunity", zorder=3)

    # Flag turns below the threshold. A full-height span would read as a second
    # series stacked above the bar, so the threshold gets a hairline and the
    # offending turns get a caret in the margin below the baseline instead.
    LOW_REUSE_PCT = 80.0
    ax.axhline(LOW_REUSE_PCT, color="#e34948", linewidth=1.0,
               linestyle=(0, (4, 3)), alpha=0.75, zorder=3,
               label=f"{LOW_REUSE_PCT:.0f}% threshold")

    low = (hit_ratio * 100) < LOW_REUSE_PCT
    ax.set_ylim(-9, 110)
    if low.any():
        ax.scatter(x[low], np.full(low.sum(), -4.5),
                   marker="^", s=26, color="#e34948",
                   clip_on=False, zorder=4)
        ax.text(-0.012, -4.5, "low", transform=ax.get_yaxis_transform(),
                ha="right", va="center", fontsize=7,
                color="#e34948", style="italic")
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    # Bars fill the plot area top to bottom here, so an inside legend would sit
    # on top of data; park it above the axes opposite the left-aligned title.
    ax.legend(fontsize=7, frameon=False, labelcolor=INK_SEC, ncol=2,
              loc="lower right", bbox_to_anchor=(1.0, 1.01),
              handlelength=1.8, columnspacing=1.4)
    add_stats(ax, hit_ratio * 100, fmt=".1f", unit="%")

    # ---- Panel 5: Decode TPS -----------------------------------------------
    # A turn whose decode window is ~0ms has an unmeasurable rate, not a zero
    # one. Drawing it as a zero-height bar would read as "no throughput" —
    # exactly backwards — so those turns get no bar and a hollow marker instead.
    ax = axes[5]
    measurable = ~np.isnan(tps_capped)
    tps_bars = ax.bar(x[measurable], tps_capped[measurable], width=BAR_W,
                      color=[c for c, m in zip(bar_colors, measurable) if m],
                      zorder=2)
    apply_hatches(tps_bars, [h for h, m in zip(bar_hatches, measurable) if m])
    style_ax(ax, "Decode throughput (TPS / user)", "Tokens / s", "tok/s", x)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:.0f}"))

    if (~measurable).any():
        y_mark = tps_capped[measurable].max() * 0.04 if measurable.any() else 1.0
        ax.scatter(x[~measurable], np.full((~measurable).sum(), y_mark),
                   marker="o", s=42, facecolors="none",
                   edgecolors=INK_MUT, linewidths=1.2, zorder=4)
        ax.text(0.01, 0.97,
                f"○ = decode window ≈0ms, rate unmeasurable "
                f"({int((~measurable).sum())} turn(s), excluded from stats)",
                transform=ax.transAxes, ha="left", va="top",
                fontsize=7, color=INK_MUT, style="italic")
    add_stats(ax, tps_capped, fmt=".1f", unit=" tok/s")

    # ---- Panel 6: Tool call count ------------------------------------------
    ax = axes[6]
    bars = ax.bar(x, tool_count, width=BAR_W, color=bar_colors, zorder=2)
    apply_hatches(bars, bar_hatches)
    style_ax(ax, "Tool calls emitted per turn", "Count", "calls", x)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{int(v)}"))
    ax.set_ylim(0, tool_count.max() + 1)
    add_stats(ax, tool_count, fmt=".1f", unit=" calls")

    # ---- Panel 7: Tool loop gap --------------------------------------------
    ax = axes[7]
    gap_sec = np.where(np.isnan(tool_gap), 0, tool_gap) / 1000.0
    bars = ax.bar(x, gap_sec, width=BAR_W, color=bar_colors, zorder=2)
    apply_hatches(bars, bar_hatches)
    style_ax(ax, "Tool loop gap (client exec time)", "Seconds", "s", x)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:.0f}s"))
    ax.text(0.01, 0.97, "0 = no tool call or unmatched",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=7, color=INK_MUT, style="italic")
    add_stats(ax, tool_gap / 1000.0, fmt=".1f", unit="s", exclude_zeros=True)

    # ---- Session legend ----------------------------------------------------
    session_labels = {s: f"S{i+1} ({s[:8]}…)" for i, s in enumerate(unique_sessions)}
    legend_handles = [
        mpatches.Patch(facecolor=session_styles[s][0], label=session_labels[s],
                       hatch=session_styles[s][1], edgecolor=SURFACE, linewidth=0)
        for s in unique_sessions
    ]
    fig.legend(handles=legend_handles, loc="lower center",
               ncol=min(len(unique_sessions), 6),
               fontsize=9, frameon=False, labelcolor=INK_SEC,
               bbox_to_anchor=(0.5, 0.01))

    fig.suptitle(
        f"Per-turn metrics — {run_label}",
        fontsize=13, fontweight="bold", color=INK_PRI, y=0.995,
    )

    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
