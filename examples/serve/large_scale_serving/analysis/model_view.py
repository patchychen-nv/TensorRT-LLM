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
"""Build and render empirical models without importing the report entry point."""

from __future__ import annotations

import html
import json
from pathlib import Path

from common import effective_isl_new
from modeling import build_models

__all__ = ["build_model_payload", "model_section", "write_model"]

_REQUEST_FIELDS = (
    "status",
    "started_at",
    "finished_at",
    "session",
    "ctx_instance",
    "model",
    "route",
    "isl_total",
    "prefill_ms",
    "kv_transfer_ms",
    "kv_transfer_bytes",
)


def build_model_payload(
    requests: list[dict],
    engine: dict,
    window: tuple[float | None, float | None],
    budgets: dict,
    metadata: dict,
) -> dict:
    """Build models using the same effective input token definition as report §0.

    Args:
        requests: Request rows; these are not modified.
        engine: Parsed instance and rank iteration tables.
        window: Analysis bounds in epoch seconds, or unspecified whole-run bounds.
        budgets: Serving token and batch budgets from the configuration snapshots.
        metadata: Run identity, deployment mode, GPU count and raw/CSV provenance.

    Returns:
        A JSON-compatible model payload with validation metrics and limitations.
    """
    enriched = []
    for row in requests:
        new, source = effective_isl_new(
            row, metadata.get("mode", "disagg"), budgets.get("tokens_per_block")
        )
        enriched.append(
            {
                **{key: row.get(key) for key in _REQUEST_FIELDS},
                "_model_isl_new": new,
                "_model_isl_source": source,
            }
        )
    return build_models(enriched, engine, window, budgets, metadata)


def write_model(out: Path, payload: dict) -> None:
    """Write model.json under the output directory, rejecting non-finite numbers."""
    out.mkdir(parents=True, exist_ok=True)
    (out / "model.json").write_text(
        json.dumps(payload, allow_nan=False, indent=2), encoding="utf-8"
    )


def _text(value: object) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:,.5g}"
    return html.escape(str(value))


def _table(headers: list[str], rows: list[list]) -> str:
    head = "".join(f"<th>{_text(value)}</th>" for value in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{_text(value)}</td>" for value in row) + "</tr>" for row in rows
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def _notes(notes: list[str]) -> str:
    return "<ul>" + "".join(f"<li>{_text(note)}</li>" for note in notes) + "</ul>"


def _group(group: dict) -> str:
    return ", ".join(f"{key}={value}" for key, value in group.items())


def _fit_section(models: list[dict], title: str) -> str:
    parts = [f"<h3>{_text(title)}</h3>"]
    if not models:
        return parts[0] + "<p>No eligible groups in the supplied data.</p>"
    for model in models:
        fit = model["fit"]
        parts.append(f"<h4>{_text(model['label'])}</h4><p>{_text(_group(model['group']))}</p>")
        parts.append(
            _table(
                ["status", "candidates", "valid", "sampled", "train", "holdout"],
                [
                    [
                        fit["status"],
                        model["filtering"]["n_candidates"],
                        fit.get("n_valid"),
                        fit.get("n_sampled"),
                        fit.get("n_train"),
                        fit.get("n_test"),
                    ]
                ],
            )
        )
        if fit.get("status") != "fitted":
            parts.append(f"<p>{_text(fit.get('reason'))}</p>")
        else:
            parts.append(
                _table(
                    ["partition", "MAE (ms)", "RMSE (ms)", "R²"],
                    [
                        [
                            name,
                            (fit.get(name) or {}).get("mae"),
                            (fit.get(name) or {}).get("rmse"),
                            (fit.get(name) or {}).get("r2"),
                        ]
                        for name in ("train", "test")
                    ],
                )
            )
            parts.append(
                f"<p>Predicted {_text(fit['target'])} = intercept + Σ coefficient × feature.</p>"
            )
            ranges = fit.get("feature_ranges", {})
            parts.append(
                _table(
                    [
                        "term",
                        "coefficient",
                        "feature unit",
                        "training min",
                        "training max",
                        "dropped",
                    ],
                    [
                        [
                            name,
                            value,
                            model.get("feature_units", {}).get(name, "—"),
                            ranges.get(name, {}).get("min"),
                            ranges.get(name, {}).get("max"),
                            fit.get("dropped_features", {}).get(name),
                        ]
                        for name, value in fit["coefficients"].items()
                    ],
                )
            )
        parts.append(_notes(model.get("notes", [])))
        details = {
            "filtering": model["filtering"],
            "split": fit.get("split"),
            "time_bounds": fit.get("time_bounds"),
            "sampling": fit.get("sampling"),
        }
        parts.append(
            "<details><summary>Sample selection and validation</summary><pre>"
            + _text(json.dumps(details, indent=2))
            + "</pre></details>"
        )
    return "".join(parts)


def model_section(payload: dict) -> str:
    """Render a safe HTML fragment for both REPORT.html and the CSV-only MODEL.html."""
    metadata = payload["metadata"]
    parts = [
        "<h2>Empirical performance models</h2>",
        "<p>Models describe the observed deployment and feature ranges. Earlier timestamps train "
        "the model; later timestamps validate it. Coefficients are associations, and extrapolation "
        "does not establish capacity or the benefit of changing the deployment.</p>",
        f"<p>Source: {_text(metadata.get('source'))}. Full results: <a href='model.json'>model.json</a>.</p>",
        _notes(payload.get("limitations", [])),
        _fit_section(payload["request_models"], "Request latency models"),
        _fit_section(payload["engine_models"], "Engine step models"),
        "<h3>Scheduling and KV reuse</h3>",
    ]
    for scheduling in payload["scheduling"]:
        parts.append(f"<h4>{_text(_group(scheduling['group']))}</h4>")
        parts.append(
            f"<p>{_text(scheduling.get('weighting'))}; "
            f"{_text(scheduling['n_iterations'])} iterations.</p>"
        )
        parts.append(
            _table(
                ["metric", "n", "mean"],
                [
                    [key, value.get("n"), value.get("mean")]
                    for key, value in scheduling["metrics"].items()
                ],
            )
        )
        parts.append(
            _table(
                ["KV metric", "value"],
                [[key, value] for key, value in scheduling["kv_reuse"].items()],
            )
        )
    parts.append("<h3>Observed request flow</h3>")
    flow = payload["flow"]
    parts.append(
        _table(
            ["metric", "value"],
            [[key, value] for key, value in flow.items() if not isinstance(value, (dict, list))],
        )
    )
    parts.append(_notes(flow.get("notes", [])))
    sessions = payload["sessions"]
    parts.append("<h3>Observed conversation spans</h3>")
    parts.append(
        _table(
            ["metric", "value"],
            [
                [key, value]
                for key, value in sessions.items()
                if not isinstance(value, (dict, list))
            ],
        )
    )
    parts.append(
        _table(
            ["metric", "n", "mean", "p50", "p90"],
            [
                [key, value.get("n"), value.get("mean"), value.get("p50"), value.get("p90")]
                for key, value in sessions.get("metrics", {}).items()
            ],
        )
    )
    parts.append(_notes(sessions.get("notes", [])))
    parts.append(f"<p>Job metrics: {_text(payload['job_metrics']['reason'])}</p>")
    parts.append(
        "<details><summary>Data quality</summary><pre>"
        + _text(json.dumps(payload["quality"], indent=2))
        + "</pre></details>"
    )
    return "".join(parts)
