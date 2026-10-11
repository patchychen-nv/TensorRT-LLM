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
"""Fit models from existing report tables, optionally combining consecutive UTC hours."""

from __future__ import annotations

import argparse
import csv
import gzip
import html
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import charts
from common import read_csv_rows
from model_view import build_model_payload, model_section, write_model

__all__ = ["load_reports", "main"]

_REQUEST_TEXT = (
    "rid",
    "trace_id",
    "session",
    "route",
    "status",
    "model",
    "ctx_instance",
    "route_phase",
    "validation_errors",
)
_DEPLOYMENT_FIELDS = (
    "run",
    "attempt",
    "mode",
    "gpus",
    "instance",
    "nodes",
    "world_size",
    "ctx_instances",
    "gen_instances",
    "topology_raw",
    "commit",
)
_ENGINE_TABLES = {
    "agg": {"mixed_iters": "worker_iters.csv", "mixed_rank": "worker_rank_iters.csv"},
    "disagg": {
        "ctx_iters": "ctx_iters.csv",
        "ctx_rank": "ctx_rank_iters.csv",
        "gen_iters": "gen_iters.csv",
        "gen_rank": "gen_rank_iters.csv",
    },
}


@dataclass
class _Report:
    path: Path
    summary: dict
    engine_meta: dict
    hour_epoch: float | None
    window: tuple[float | None, float | None]


def _json_object(path: Path) -> dict:
    if not path.is_file():
        raise ValueError(f"Missing {path}; provide a report directory produced by report.py.")
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object.")
    return value


def _load_metadata(path: Path) -> _Report:
    summary = _json_object(path / "summary.json")
    meta = _json_object(path / "engine_meta.json")
    if not isinstance(summary.get("run"), str) or not summary["run"]:
        raise ValueError(f"{path}/summary.json is missing the run identity.")
    if meta.get("mode") not in _ENGINE_TABLES or summary.get("mode") != meta.get("mode"):
        raise ValueError(f"{path}: summary and engine metadata need the same agg/disagg mode.")
    budgets = meta.get("budgets")
    if not isinstance(budgets, dict):
        raise ValueError(f"{path}/engine_meta.json is missing the budgets object.")
    for key in (
        "max_num_tokens",
        "max_batch_size",
        "tokens_per_block",
        "ctx_max_batch_size",
        "gen_max_num_tokens",
    ):
        value = budgets.get(key)
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
        ):
            raise ValueError(
                f"{path}/engine_meta.json: budget {key} must be a positive integer or null."
            )
    if summary.get("budgets") and summary["budgets"] != budgets:
        raise ValueError(f"{path}: summary and engine budgets differ; regenerate the report.")
    workers = meta.get("workers")
    if not isinstance(workers, list) or any(
        not isinstance(worker, dict) or not isinstance(worker.get("worker"), str)
        for worker in workers
    ):
        raise ValueError(f"{path}/engine_meta.json needs a workers list with worker identities.")
    worker_names = [worker["worker"] for worker in workers]
    if len(worker_names) != len(set(worker_names)):
        raise ValueError(f"{path}/engine_meta.json contains duplicate worker identities.")
    gpus = summary.get("gpus")
    if gpus is not None and (
        isinstance(gpus, bool)
        or not isinstance(gpus, (int, float))
        or not math.isfinite(gpus)
        or gpus <= 0
        or gpus != int(gpus)
    ):
        raise ValueError(f"{path}/summary.json: gpus must be a positive integer or null.")
    hour = summary.get("hour")
    hour_epoch = None
    if hour is not None:
        if not isinstance(hour, str):
            raise ValueError(f"{path}/summary.json: hour must use YYYY-MM-DDTHH UTC format.")
        try:
            stamp = datetime.strptime(hour, "%Y-%m-%dT%H").replace(tzinfo=timezone.utc)
        except ValueError as error:
            raise ValueError(
                f"{path}: invalid UTC hour {hour!r}; expected YYYY-MM-DDTHH."
            ) from error
        if stamp.strftime("%Y-%m-%dT%H") != hour:
            raise ValueError(f"{path}: hour must use YYYY-MM-DDTHH UTC format.")
        hour_epoch = stamp.timestamp()
    bounds = meta.get("window")
    if (
        not isinstance(bounds, list)
        or len(bounds) != 2
        or any(
            value is not None
            and (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            )
            for value in bounds
        )
    ):
        raise ValueError(f"{path}/engine_meta.json: window must contain two epoch bounds or nulls.")
    if all(value is not None for value in bounds) and bounds[1] < bounds[0]:
        raise ValueError(f"{path}/engine_meta.json: window end precedes its start.")
    return _Report(path, summary, meta, hour_epoch, tuple(bounds))


def _worker_topology(workers: list[dict]) -> list[dict]:
    # These counters describe the portion scanned, not a deployment change.
    return sorted(
        [
            {
                key: value
                for key, value in worker.items()
                if key not in ("iter_lines", "engine_instances")
            }
            for worker in workers
        ],
        key=lambda worker: worker["worker"],
    )


def _validate_compatibility(reports: list[_Report]) -> None:
    if len(reports) < 2:
        return
    first = reports[0]
    if any(report.hour_epoch is None for report in reports):
        raise ValueError(
            "Combining reports requires distinct consecutive UTC hours; whole-run reports cannot be merged."
        )
    for report in reports[1:]:
        for key in _DEPLOYMENT_FIELDS:
            if report.summary.get(key) != first.summary.get(key):
                raise ValueError(
                    f"Incompatible {key}: {first.path} and {report.path}; fit separately."
                )
        for key in ("budgets", "log_tz"):
            if report.engine_meta.get(key) != first.engine_meta.get(key):
                raise ValueError(
                    f"Incompatible {key}: {first.path} and {report.path}; fit separately."
                )
        if _worker_topology(report.engine_meta["workers"]) != _worker_topology(
            first.engine_meta["workers"]
        ):
            raise ValueError(
                f"Incompatible worker topology/quotas: {first.path} and {report.path}; fit separately."
            )
    epochs = sorted(report.hour_epoch for report in reports)
    if any(right - left != 3600 for left, right in zip(epochs, epochs[1:])):
        raise ValueError(
            "Reports must have distinct consecutive UTC hours (no duplicates or gaps)."
        )


def _read_table(path: Path, *, request: bool = False, rank: bool = False) -> list[dict]:
    candidates = (path, path.with_suffix(path.suffix + ".gz"))
    found = next((candidate for candidate in candidates if candidate.is_file()), None)
    if found is None:
        raise ValueError(
            f"Missing {path.name} (or .gz) in {path.parent}; regenerate report.py output with CSV tables."
        )
    opener = gzip.open if found.suffix == ".gz" else open
    with opener(found, "rt", encoding="utf-8", newline="") as handle:
        header = next(csv.reader(handle), [])
    required = (
        {"status", "started_at", "finished_at"}
        if request
        else {
            "worker",
            "engine_instance",
            "iter",
            "timestamp",
        }
    )
    if rank:
        required.add("rank")
    if not header or not required.issubset(header) or len(set(header)) != len(header):
        missing = ", ".join(sorted(required - set(header)))
        raise ValueError(
            f"{found}: invalid CSV header; missing columns: {missing or 'empty or duplicate columns'}."
        )
    rows = read_csv_rows(found, text_columns=_REQUEST_TEXT if request else ("worker",))
    if not request and any(
        any(row.get(key) is None for key in required - {"timestamp"}) for row in rows
    ):
        raise ValueError(f"{found}: an engine row is missing its worker/iteration/rank identity.")
    if not request:
        identity_fields = (
            ("engine_instance", "iter", "rank") if rank else ("engine_instance", "iter")
        )
        if any(
            isinstance(row.get(key), bool) or not isinstance(row.get(key), int) or row[key] < 0
            for row in rows
            for key in identity_fields
        ):
            raise ValueError(
                f"{found}: engine instance, iteration and rank must be nonnegative integers."
            )
    return rows


def _same_measurement(left: object, right: object) -> bool:
    if isinstance(left, int) and isinstance(right, int):
        return left == right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isclose(left, right, rel_tol=0, abs_tol=1e-4)
    return left == right


def _deduplicate(rows: list[dict], *, request: bool = False, rank: bool = False) -> list[dict]:
    unique = []
    identities: dict[tuple, dict] = {}
    for row in rows:
        if request:
            identity = next(
                ((key, row[key]) for key in ("trace_id", "rid") if row.get(key) is not None), None
            )
        else:
            fields = (
                ("worker", "engine_instance", "iter", "rank")
                if rank
                else (
                    "worker",
                    "engine_instance",
                    "iter",
                )
            )
            identity = tuple(row.get(key) for key in fields)
        if identity is None:
            unique.append(row)
        elif identity not in identities:
            identities[identity] = row
            unique.append(row)
        else:
            previous = identities[identity]
            for key, value in row.items():
                if previous.get(key) is None:
                    previous[key] = value
                elif value is not None and not _same_measurement(previous[key], value):
                    raise ValueError(
                        f"Conflicting duplicate {'request' if request else 'engine'} row {identity!r}, "
                        f"column {key!r}; regenerate consistent reports."
                    )
    time_key = "started_at" if request else "timestamp"
    return sorted(unique, key=lambda row: (row.get(time_key) is None, row.get(time_key) or 0))


def load_reports(paths: list[Path]) -> tuple[list[dict], dict, tuple, dict, dict]:
    """Read compatible CSV reports and merge their observations before fitting.

    Args:
        paths: One report directory, or directories for consecutive UTC hours.

    Returns:
        Requests, engine tables, observation window, budgets, and source metadata.
    """
    if not paths:
        raise ValueError("Provide at least one report directory.")
    reports = [_load_metadata(path.resolve()) for path in paths]
    _validate_compatibility(reports)
    reports.sort(key=lambda report: report.hour_epoch or 0)
    first = reports[0]
    mode = first.engine_meta["mode"]
    engine = {
        f"{role}_{grain}": [] for role in ("ctx", "gen", "mixed") for grain in ("iters", "rank")
    }
    engine.update(mode=mode, workers=first.engine_meta["workers"])
    requests = []
    for report in reports:
        requests.extend(_read_table(report.path / "requests.csv", request=True))
        for name, filename in _ENGINE_TABLES[mode].items():
            engine[name].extend(_read_table(report.path / filename, rank=name.endswith("rank")))
    requests = _deduplicate(requests, request=True)
    for name in _ENGINE_TABLES[mode]:
        engine[name] = _deduplicate(engine[name], rank=name.endswith("rank"))
    metadata = {key: first.summary.get(key) for key in _DEPLOYMENT_FIELDS}
    metadata.update(
        source="csv",
        reports=[str(report.path) for report in reports],
        hours=[report.summary.get("hour") for report in reports],
        hour=first.summary.get("hour") if len(reports) == 1 else None,
        log_tz=first.engine_meta.get("log_tz"),
        source_windows=[list(report.window) for report in reports],
        source_request_windows=[
            [report.summary.get("window_start"), report.summary.get("window_end")]
            for report in reports
        ],
        selection="hourly_request_cohorts" if first.hour_epoch is not None else "whole_run",
    )
    windows = [report.window for report in reports]
    if len(reports) > 1:
        windows = [
            (
                report.window[0] if report.window[0] is not None else report.hour_epoch,
                report.window[1] if report.window[1] is not None else report.hour_epoch + 3600,
            )
            for report in reports
        ]
        if any(report.window != window for report, window in zip(reports, windows)):
            metadata["window_inference"] = "Missing hourly bounds use their nominal UTC hour."
    window = (
        min((start for start, _ in windows if start is not None), default=None),
        max((end for _, end in windows if end is not None), default=None),
    )
    return requests, engine, window, first.engine_meta["budgets"], metadata


def main(argv: list[str] | None = None) -> int:
    """Write model.json and MODEL.html using existing report artifacts only."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", metavar="REPORT_DIR", nargs="+", type=Path)
    parser.add_argument("--out", metavar="OUTPUT_DIR", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        requests, engine, window, budgets, metadata = load_reports(args.reports)
        payload = build_model_payload(requests, engine, window, budgets, metadata)
        args.out.mkdir(parents=True, exist_ok=True)
        write_model(args.out, payload)
        title = html.escape(f"{metadata['run']} · Performance models")
        page = (
            f"<!doctype html><meta charset=utf-8><title>{title}</title>"
            f"{charts.FONT_LINKS}<style>{charts.CSS}</style><h1>{title}</h1>"
            + model_section(payload)
        )
        (args.out / "MODEL.html").write_text(page, encoding="utf-8")
    except (ValueError, OSError, csv.Error) as error:
        parser.exit(2, f"model_report.py: {error}\n")
    print(f"{len(requests)} requests from {len(args.reports)} reports -> {args.out / 'model.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
