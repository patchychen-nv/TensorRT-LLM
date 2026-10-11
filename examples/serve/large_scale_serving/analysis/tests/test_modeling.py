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
"""Synthetic tests of cohort boundaries, session overlaps and model grouping."""

import json
import unittest

from modeling import build_models


def _request(start: float, end: float, **fields: object) -> dict:
    return {"started_at": start, "finished_at": end, "status": "completed", **fields}


def _build(
    requests: list[dict], engine: dict | None = None, window: tuple = (0, 100), **metadata: object
) -> dict:
    return build_models(requests, engine or {}, window, {}, {"mode": "disagg", **metadata})


class ModelingTests(unittest.TestCase):
    def test_clipped_flow_and_contained_identity(self) -> None:
        result = _build(
            [
                _request(-2, 2),
                _request(2, 4),
                _request(8, 12),
                _request(1, 9, status="no_response"),
            ],
            window=(0, 10),
        )
        flow = result["flow"]
        self.assertEqual(flow["observed_request_seconds"], 6)
        self.assertEqual(flow["observed_mean_concurrency"], 0.6)
        self.assertEqual(flow["n_fully_contained"], 1)
        self.assertEqual(flow["contained_mean_latency_s"], 2)
        self.assertEqual(flow["contained_rate_times_latency"], 0.2)
        self.assertEqual(
            result["quality"]["request_excluded_counts"],
            {"not_fully_contained_in_window": 2, "not_completed": 1},
        )

    def test_overlapping_sessions_use_union_not_duration_sum(self) -> None:
        requests = [
            _request(0, 5, session="a"),
            _request(2, 4, session="a"),
            _request(3, 7, session="a"),
            _request(9, 10, session="a"),
            _request(1, 3, session="_no_session"),
        ]
        sessions = _build(requests, window=(0, 10))["sessions"]
        metrics = sessions["metrics"]
        self.assertEqual(sessions["n_known_sessions"], 1)
        self.assertEqual(sessions["n_missing_session"], 1)
        for name, expected in {
            "observed_span_s": 10,
            "request_duration_sum_s": 12,
            "request_union_s": 8,
            "idle_gap_s": 2,
            "overlap_s": 4,
        }.items():
            self.assertEqual(metrics[name]["mean"], expected)

    def test_request_groups_keep_sources_and_routes_separate(self) -> None:
        requests = []
        for source, route in (("usage", "chat"), ("blocks", "chat"), ("usage", "responses")):
            requests.extend(
                _request(
                    index,
                    index + 0.5,
                    ctx_instance=0,
                    model="m",
                    route=route,
                    _model_isl_source=source,
                    _model_isl_new=index,
                    isl_total=2 * index + index % 3,
                    prefill_ms=3 * index + 2,
                )
                for index in range(40)
            )
        models = [model for model in _build(requests)["request_models"] if model["id"] == "prefill"]
        self.assertEqual(len(models), 3)
        self.assertTrue(all(model["fit"]["status"] == "fitted" for model in models))
        self.assertEqual(
            {(model["group"]["token_source"], model["group"]["route"]) for model in models},
            {("usage", "chat"), ("blocks", "chat"), ("usage", "responses")},
        )

    def test_missing_inputs_not_zero_and_strict_json(self) -> None:
        requests = [
            _request(1, 2, _model_isl_source="missing", prefill_ms=3, isl_total=5),
            _request(
                3,
                4,
                _model_isl_source="missing",
                _model_isl_new=float("nan"),
                prefill_ms=-1,
                isl_total=5,
            ),
        ]
        result = _build(requests, source="csv")
        model = result["request_models"][0]
        self.assertEqual(model["filtering"]["n_excluded"], 2)
        self.assertEqual(model["fit"]["status"], "unavailable")
        self.assertEqual(model["fit"]["n_valid"], 0)
        self.assertTrue(any("quantized" in note for note in result["limitations"]))
        json.dumps(result, allow_nan=False)

    def test_engine_roles_lifetimes_and_optional_cache(self) -> None:
        engine: dict[str, list] = {"ctx_iters": [], "ctx_rank": [], "gen_iters": [], "gen_rank": []}
        for role in ("ctx", "gen"):
            for lifetime in (0, 1):
                for index in range(40):
                    fields = (
                        {"ctx_tokens_mean": index + 1, "decode_requests_sum": None}
                        if role == "ctx"
                        else {
                            "ctx_tokens_mean": None,
                            "decode_requests_sum": index + 1,
                            "gen_tokens_sum": 3 * (index + 1),
                        }
                    )
                    identity = {
                        "worker": f"{role}-0",
                        "engine_instance": lifetime,
                        "iter": index,
                        "timestamp": index,
                    }
                    engine[f"{role}_iters"].append(
                        {**identity, **fields, "ranks": 2, "device_step_ms_mean": 2 * index + 1}
                    )
                    for rank in range(2):
                        engine[f"{role}_rank"].append(
                            {
                                **identity,
                                "rank": rank,
                                "device_step_ms": 2 * index + 1,
                                "kv_reused_blocks_delta": 1,
                                "kv_missed_blocks_delta": 3,
                            }
                        )
        result = _build([], engine)
        self.assertEqual(len(result["engine_models"]), 4)
        for model in result["engine_models"]:
            self.assertEqual(model["fit"]["status"], "fitted")
            self.assertNotIn("cached_kv_tokens_per_rank", model["fit"]["feature_names"])
            self.assertEqual(
                model["group"]["iteration_kind"],
                "prefill" if model["group"]["role"] == "ctx" else "decode_only",
            )
        self.assertTrue(all(item["kv_reuse"]["hit_rate"] == 0.25 for item in result["scheduling"]))

    def test_partial_rank_device_timing_is_excluded(self) -> None:
        identity = {"worker": "ctx-0", "engine_instance": 0, "iter": 1, "timestamp": 1}
        engine = {
            "ctx_iters": [{**identity, "ranks": 2, "ctx_tokens_mean": 1, "device_step_ms_mean": 5}],
            "ctx_rank": [
                {**identity, "rank": 0, "device_step_ms": 5},
                {**identity, "rank": 1, "device_step_ms": None},
            ],
        }
        model = _build([], engine)["engine_models"][0]
        self.assertEqual(
            model["filtering"]["excluded_counts"], {"incomplete_rank_device_timing": 1}
        )
        self.assertEqual(model["fit"]["n_valid"], 0)

    def test_missing_whole_iteration_and_partial_cache(self) -> None:
        identity = {"worker": "gen-0", "engine_instance": 0, "iter": 1, "timestamp": 1}
        first = {
            **identity,
            "ranks": 2,
            "decode_requests_sum": 2,
            "gen_tokens_sum": 4,
            "device_step_ms_mean": 5,
            "cached_kv_tokens_sum": 20,
        }
        engine = {
            "gen_iters": [first, {**first, "iter": 2, "timestamp": 2}],
            "gen_rank": [
                {**identity, "rank": 0, "device_step_ms": 5, "cached_kv_tokens": 20},
                {**identity, "rank": 1, "device_step_ms": 5, "cached_kv_tokens": None},
            ],
        }
        model = _build([], engine)["engine_models"][0]
        self.assertEqual(
            model["filtering"]["excluded_counts"], {"incomplete_rank_device_timing": 1}
        )
        self.assertEqual(model["fit"]["n_valid"], 1)
        self.assertNotIn("cached_kv_tokens_per_rank", model["fit"]["feature_names"])

    def test_empty_and_whole_attempt_windows(self) -> None:
        empty = _build([], window=(None, None))
        self.assertEqual(empty["flow"]["status"], "unavailable")
        self.assertEqual(empty["job_metrics"]["status"], "unavailable")
        result = _build([_request(20, 23), _request(25, 30)], window=(None, None))
        self.assertEqual(result["metadata"]["window"], [20, 30])
        self.assertEqual(result["flow"]["observed_request_seconds"], 8)
        json.dumps(result, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
