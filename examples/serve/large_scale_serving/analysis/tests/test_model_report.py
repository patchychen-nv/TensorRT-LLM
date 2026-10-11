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
"""Exercise report merging and CSV-only model output with synthetic artifacts."""

from __future__ import annotations

import csv
import gzip
import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
from pathlib import Path

from common import utc_iso
from engine_iters import INSTANCE_COLUMNS, RANK_COLUMNS
from model_report import load_reports, main
from requests_table import COLUMNS as REQUEST_COLUMNS


class ModelReportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def _write_table(self, path: Path, columns: list[str], rows: list[dict]) -> None:
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "wt", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)

    def _report(self, name: str, hour: str | None = "2026-09-28T08", mode: str = "disagg") -> Path:
        path = self.root / name
        path.mkdir()
        start = (
            datetime.strptime(hour or "2026-09-28T08", "%Y-%m-%dT%H")
            .replace(tzinfo=timezone.utc)
            .timestamp()
        )
        budgets = {
            "max_num_tokens": 8192,
            "max_batch_size": 128,
            "tokens_per_block": 64,
            "ctx_max_batch_size": 64,
            "gen_max_num_tokens": 4096,
        }
        summary = {
            "run": "synthetic",
            "attempt": "attempt-1",
            "hour": hour,
            "mode": mode,
            "gpus": 8,
            "budgets": budgets,
            "window_start": utc_iso(start),
            "window_end": utc_iso(start + 3601),
        }
        workers = (
            [{"worker": "server", "role": "mixed", "device_quota_gib": 20}]
            if mode == "agg"
            else [
                {"worker": "ctx-0", "role": "ctx", "device_quota_gib": 20},
                {"worker": "gen-0", "role": "gen", "device_quota_gib": 40},
            ]
        )
        meta = {
            "mode": mode,
            "budgets": budgets,
            "workers": workers,
            "window": [start, start + 3600],
            "log_tz": "UTC",
        }
        (path / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
        (path / "engine_meta.json").write_text(json.dumps(meta), encoding="utf-8")
        self._write_table(path / "requests.csv", REQUEST_COLUMNS, [])
        for prefix in ("worker",) if mode == "agg" else ("ctx", "gen"):
            self._write_table(path / f"{prefix}_iters.csv", INSTANCE_COLUMNS, [])
            self._write_table(path / f"{prefix}_rank_iters.csv", RANK_COLUMNS, [])
        return path

    def _change_json(self, path: Path, filename: str, changes: dict) -> None:
        target = path / filename
        payload = json.loads(target.read_text(encoding="utf-8"))
        payload.update(changes)
        target.write_text(json.dumps(payload), encoding="utf-8")

    def test_gzip_preserves_numeric_and_boolean_identifiers(self) -> None:
        path = self._report("first")
        row = {
            "rid": "0001",
            "trace_id": "001",
            "session": "True",
            "status": "completed",
            "started_at": "2026-09-28T08:00:01Z",
            "finished_at": "2026-09-28T08:00:02Z",
        }
        (path / "requests.csv").unlink()
        self._write_table(path / "requests.csv.gz", REQUEST_COLUMNS, [row])
        requests, _, _, _, metadata = load_reports([path])
        self.assertEqual(requests[0]["rid"], "0001")
        self.assertEqual(requests[0]["trace_id"], "001")
        self.assertEqual(requests[0]["session"], "True")
        self.assertIsInstance(requests[0]["started_at"], float)
        self.assertEqual(metadata["source"], "csv")
        self.assertEqual(metadata["selection"], "hourly_request_cohorts")

    def test_consecutive_hours_merge_rows_and_boundary_measurements(self) -> None:
        first = self._report("first")
        second = self._report("second", "2026-09-28T09")
        boundary = {
            "worker": "gen-0",
            "engine_instance": 2,
            "iter": 500,
            "timestamp": "2026-09-28T09:00:00Z",
            "device_step_ms_mean": None,
            "host_step_ms_mean": 8,
        }
        self._write_table(first / "gen_iters.csv", INSTANCE_COLUMNS, [boundary])
        self._write_table(
            second / "gen_iters.csv", INSTANCE_COLUMNS, [{**boundary, "device_step_ms_mean": 7}]
        )
        rank = {
            "worker": "gen-0",
            "engine_instance": 2,
            "iter": 500,
            "rank": 0,
            "timestamp": "2026-09-28T09:00:00Z",
            "device_step_ms": 7,
        }
        self._write_table(first / "gen_rank_iters.csv", RANK_COLUMNS, [rank])
        self._write_table(second / "gen_rank_iters.csv", RANK_COLUMNS, [rank, {**rank, "rank": 1}])
        identified = {"trace_id": "same", "status": "completed"}
        unidentified = {"status": "unanswered"}
        for path in (first, second):
            self._write_table(path / "requests.csv", REQUEST_COLUMNS, [identified, unidentified])
        requests, engine, window, _, metadata = load_reports([second, first])
        self.assertEqual(len(requests), 3)
        self.assertEqual(len(engine["gen_iters"]), 1)
        self.assertEqual(engine["gen_iters"][0]["device_step_ms_mean"], 7)
        self.assertEqual(len(engine["gen_rank"]), 2)
        self.assertEqual(window[1] - window[0], 7200)
        self.assertIsNone(metadata["hour"])
        self.assertEqual(metadata["hours"], ["2026-09-28T08", "2026-09-28T09"])
        self.assertEqual(len(metadata["source_windows"]), 2)

    def test_missing_ids_are_distinct_and_rid_fallback_deduplicates(self) -> None:
        path = self._report("first")
        self._write_table(
            path / "requests.csv",
            REQUEST_COLUMNS,
            [{"rid": "001", "status": "completed"}] * 2 + [{"status": "unanswered"}] * 2,
        )
        requests, *_ = load_reports([path])
        self.assertEqual(len(requests), 3)

    def test_conflicting_duplicate_rejected(self) -> None:
        path = self._report("first")
        self._write_table(
            path / "requests.csv",
            REQUEST_COLUMNS,
            [{"trace_id": "same", "e2e_ms": 1}, {"trace_id": "same", "e2e_ms": 2}],
        )
        with self.assertRaisesRegex(ValueError, "Conflicting duplicate request.*e2e_ms"):
            load_reports([path])

    def test_duplicate_epoch_timestamps_and_counts_compare_exactly(self) -> None:
        path = self._report("first")
        row = {
            "worker": "gen-0",
            "engine_instance": 2,
            "iter": 1,
            "timestamp": "2026-09-28T08:00:01Z",
            "gen_tokens_sum": 10000000000,
        }
        for changes, field in (
            ({"timestamp": "2026-09-28T08:00:02Z"}, "timestamp"),
            ({"gen_tokens_sum": 10000000001}, "gen_tokens_sum"),
        ):
            self._write_table(path / "gen_iters.csv", INSTANCE_COLUMNS, [row, {**row, **changes}])
            with (
                self.subTest(field=field),
                self.assertRaisesRegex(ValueError, f"Conflicting duplicate engine.*{field}"),
            ):
                load_reports([path])

    def test_malformed_budget_and_engine_identity_rejected(self) -> None:
        path = self._report("first")
        self._write_table(
            path / "gen_iters.csv",
            INSTANCE_COLUMNS,
            [{"worker": "gen-0", "engine_instance": 2, "iter": "invalid"}],
        )
        with self.assertRaisesRegex(ValueError, "must be nonnegative integers"):
            load_reports([path])
        self._change_json(path, "engine_meta.json", {"budgets": {"tokens_per_block": "invalid"}})
        with self.assertRaisesRegex(ValueError, "budget tokens_per_block"):
            load_reports([path])

    def test_deployment_and_timezone_mismatches_rejected(self) -> None:
        first = self._report("first")
        for field, value in (
            ("run", "other"),
            ("attempt", "attempt-2"),
            ("gpus", None),
            ("nodes", 2),
            ("topology_raw", "4P1D"),
        ):
            with self.subTest(field=field):
                second = self._report(f"second-{field}", "2026-09-28T09")
                self._change_json(second, "summary.json", {field: value})
                with self.assertRaisesRegex(ValueError, f"Incompatible {field}"):
                    load_reports([first, second])
        second = self._report("timezone", "2026-09-28T09")
        self._change_json(second, "engine_meta.json", {"log_tz": "Asia/Shanghai"})
        with self.assertRaisesRegex(ValueError, "Incompatible log_tz"):
            load_reports([first, second])

    def test_budget_and_quota_mismatches_rejected(self) -> None:
        first = self._report("first")
        for key in ("ctx_max_batch_size", "gen_max_num_tokens"):
            second = self._report(key, "2026-09-28T09")
            budgets = json.loads((second / "summary.json").read_text())["budgets"]
            budgets[key] *= 2
            self._change_json(second, "summary.json", {"budgets": budgets})
            self._change_json(second, "engine_meta.json", {"budgets": budgets})
            with self.assertRaisesRegex(ValueError, "Incompatible budgets"):
                load_reports([first, second])
        second = self._report("quota", "2026-09-28T09")
        workers = json.loads((second / "engine_meta.json").read_text())["workers"]
        workers[0]["device_quota_gib"] = 40
        self._change_json(second, "engine_meta.json", {"workers": workers})
        with self.assertRaisesRegex(ValueError, "worker topology/quotas"):
            load_reports([first, second])

    def test_runtime_worker_counters_do_not_prevent_merging(self) -> None:
        first = self._report("first")
        second = self._report("second", "2026-09-28T09")
        workers = json.loads((second / "engine_meta.json").read_text())["workers"]
        workers[0].update(iter_lines=10000, engine_instances=3)
        self._change_json(second, "engine_meta.json", {"workers": list(reversed(workers))})
        load_reports([first, second])

    def test_duplicate_gapped_and_whole_run_merges_rejected(self) -> None:
        first = self._report("first")
        for name, hour in (
            ("duplicate", "2026-09-28T08"),
            ("gap", "2026-09-28T10"),
            ("whole", None),
        ):
            second = self._report(name, hour)
            with (
                self.subTest(hour=hour),
                self.assertRaisesRegex(ValueError, "consecutive UTC hours"),
            ):
                load_reports([first, second])

    def test_unknown_gpu_count_allowed_for_single_report(self) -> None:
        path = self._report("first", None)
        self._change_json(path, "summary.json", {"gpus": None})
        self._change_json(path, "engine_meta.json", {"window": [None, None]})
        _, _, window, _, metadata = load_reports([path])
        self.assertIsNone(metadata["gpus"])
        self.assertEqual(window, (None, None))

    def test_aggregated_engine_mapping(self) -> None:
        path = self._report("first", mode="agg")
        self._write_table(
            path / "worker_iters.csv",
            INSTANCE_COLUMNS,
            [
                {
                    "worker": "server",
                    "engine_instance": 2,
                    "iter": 1,
                    "timestamp": "2026-09-28T08:00:01Z",
                }
            ],
        )
        _, engine, *_ = load_reports([path])
        self.assertEqual(engine["mode"], "agg")
        self.assertEqual(len(engine["mixed_iters"]), 1)
        self.assertEqual(engine["ctx_iters"], [])

    def test_empty_header_tables_valid_but_missing_or_bad_header_rejected(self) -> None:
        path = self._report("first")
        requests, engine, *_ = load_reports([path])
        self.assertEqual(requests, [])
        self.assertEqual(engine["ctx_iters"], [])
        (path / "ctx_iters.csv").unlink()
        with self.assertRaisesRegex(ValueError, "Missing ctx_iters.csv"):
            load_reports([path])
        (path / "ctx_iters.csv").write_text("\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "invalid CSV header"):
            load_reports([path])

    def test_cli_writes_only_model_outputs(self) -> None:
        path = self._report("first")
        (path / "REPORT.html").write_text("original report", encoding="utf-8")
        originals = {file.name: file.read_bytes() for file in path.iterdir()}
        with redirect_stdout(io.StringIO()):
            self.assertEqual(main([str(path), "--out", str(path)]), 0)
        self.assertEqual(
            set(file.name for file in path.iterdir()) - originals.keys(),
            {"model.json", "MODEL.html"},
        )
        for name, data in originals.items():
            self.assertEqual((path / name).read_bytes(), data)
        payload = json.loads((path / "model.json").read_text())
        self.assertEqual(payload["metadata"]["source"], "csv")
        self.assertIn("model.json", (path / "MODEL.html").read_text())

    def test_cli_malformed_metadata_is_actionable(self) -> None:
        path = self._report("first")
        (path / "summary.json").write_text("{broken", encoding="utf-8")
        stderr = io.StringIO()
        with redirect_stderr(stderr), self.assertRaises(SystemExit) as raised:
            main([str(path), "--out", str(self.root / "output")])
        self.assertEqual(raised.exception.code, 2)
        self.assertIn("model_report.py:", stderr.getvalue())
        self.assertNotIn("Traceback", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
