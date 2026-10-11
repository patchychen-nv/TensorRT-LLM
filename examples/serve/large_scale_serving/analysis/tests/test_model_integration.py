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
"""Exercise report integration using synthetic rows, with raw parsers mocked."""

from __future__ import annotations

import contextlib
import copy
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import report
from common import effective_isl_new, read_csv_rows
from model_view import build_model_payload, model_section
from requests_table import COLUMNS


def _requests() -> list[dict]:
    rows = []
    for index in range(40):
        start = 100.25 + index * 10
        new = 50 + index
        total = 1000 + index % 7 * 32
        prefill = 3 + 0.05 * new + 0.002 * total
        row = dict.fromkeys(COLUMNS)
        row.update(
            trace_id=f"trace-{index}",
            rid=str(index),
            session="001" if index < 20 else "1",
            route="/v1/chat/completions",
            status="completed",
            model="<synthetic>",
            started_at=start,
            handler_entry_at=start + 0.01,
            finished_at=start + 2.5,
            e2e_ms=2500.0,
            ttft_ms=10 + prefill,
            prefill_ms=prefill,
            decode_ms=2490 - prefill,
            isl_total=total,
            isl_cached=total - new,
            isl_new=new,
            osl=20,
            ctx_instance="server",
            routed_rank=0,
        )
        rows.append(row)
    return rows


def _engine() -> dict:
    return {
        "mode": "agg",
        "workers": [],
        **{f"{role}_{kind}": [] for role in ("ctx", "gen", "mixed") for kind in ("iters", "rank")},
    }


class ReportModelIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.attempt = self.root / "run" / "attempt-001"
        self.attempt.mkdir(parents=True)
        (self.attempt / "server_config.yaml").write_text(
            "tensor_parallel_size: 1\nmax_num_tokens: 512\nmax_batch_size: 16\n"
            "tokens_per_block: 64\n",
            encoding="utf-8",
        )
        self.out = self.root / "report"

    def _run_raw(self, *options: str) -> None:
        args = [
            "report.py",
            str(self.attempt),
            "--hour",
            "1970-01-01T00",
            "--log-tz",
            "UTC",
            "--out",
            str(self.out),
            *options,
        ]
        with (
            patch.object(sys, "argv", args),
            patch.object(report, "build_requests", return_value=(_requests(), [])),
            patch.object(report, "build_engine", return_value=_engine()),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(report.main(), 0)

    def test_raw_report_and_csv_rerender(self) -> None:
        self._run_raw("--model")
        payload = json.loads((self.out / "model.json").read_text())
        self.assertEqual(payload["metadata"]["source"], "raw")
        fit = payload["request_models"][0]["fit"]
        self.assertEqual(fit["status"], "fitted")
        self.assertLess(fit["test"]["rmse"], 1e-9)
        page = (self.out / "REPORT.html").read_text()
        self.assertIn("Empirical performance models", page)
        self.assertIn("&lt;synthetic&gt;", page)
        self.assertNotIn("<synthetic>", page)
        args = [
            "report.py",
            str(self.attempt),
            "--hour",
            "1970-01-01T00",
            "--out",
            str(self.out),
            "--from-csv",
            "--model",
        ]
        with (
            patch.object(sys, "argv", args),
            patch.object(report, "build_requests", side_effect=AssertionError("raw trace read")),
            patch.object(report, "build_engine", side_effect=AssertionError("raw log read")),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(report.main(), 0)
        payload = json.loads((self.out / "model.json").read_text())
        self.assertEqual(payload["metadata"]["source"], "csv")
        self.assertEqual(payload["sessions"]["n_known_sessions"], 2)
        self.assertTrue(any("quantized" in note for note in payload["limitations"]))

    def test_json_only_still_writes_model(self) -> None:
        self._run_raw("--model", "--json", "--json-only")
        self.assertTrue((self.out / "model.json").is_file())
        self.assertTrue((self.out / "rollup.json").is_file())
        self.assertFalse((self.out / "REPORT.html").exists())
        self.assertFalse((self.out / "requests.csv").exists())

    def test_model_is_opt_in(self) -> None:
        with patch(
            "model_view.build_models", side_effect=AssertionError("model unexpectedly called")
        ):
            self._run_raw()
        self.assertFalse((self.out / "model.json").exists())
        self.assertNotIn("Empirical performance models", (self.out / "REPORT.html").read_text())

    def test_shared_token_policy_and_unmodified_input(self) -> None:
        rows = _requests()
        rows[0].update(route="/v1/responses", isl_cached=1000, isl_new=0, ctx_blocks_new=2)
        before = copy.deepcopy(rows)
        payload = build_model_payload(
            rows,
            _engine(),
            (0, 1000),
            {"tokens_per_block": 64},
            {"mode": "disagg", "source": "raw"},
        )
        self.assertEqual(rows, before)
        self.assertEqual(effective_isl_new(rows[0], "disagg", 64), (128, "blocks"))
        self.assertEqual(effective_isl_new(rows[0], "agg", 64), (None, "missing"))
        self.assertTrue(
            any(
                model["group"].get("token_source") == "blocks"
                for model in payload["request_models"]
            )
        )
        self.assertIn("insufficient_samples", model_section(payload))
        json.dumps(payload, allow_nan=False)

    def test_csv_text_columns_preserve_identity(self) -> None:
        path = self.root / "identity.csv"
        path.write_text("session,rid,osl\n001,False,2\n1,001,3\n", encoding="utf-8")
        rows = read_csv_rows(path, text_columns=("session", "rid"))
        self.assertEqual(
            rows,
            [
                {"session": "001", "rid": "False", "osl": 2},
                {"session": "1", "rid": "001", "osl": 3},
            ],
        )

    def test_malformed_csv_reports_source_location(self) -> None:
        path = self.root / "malformed.csv"
        path.write_text("session,osl\nthread,2,unexpected\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, r"malformed.csv:2: CSV row"):
            read_csv_rows(path)


if __name__ == "__main__":
    unittest.main()
