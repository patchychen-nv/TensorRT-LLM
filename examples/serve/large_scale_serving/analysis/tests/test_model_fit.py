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
"""Synthetic numerical and time-split checks; no trace inputs or GPUs needed."""

from __future__ import annotations

import json
import math
import unittest

from model_fit import fit_linear


class LinearFitTests(unittest.TestCase):
    def test_known_multivariate_relationship(self) -> None:
        samples = [
            (float(index), (float(index), float(index % 7)), 12.0 + 2.5 * index - 3 * (index % 7))
            for index in range(100)
        ]
        result = fit_linear(samples, ["tokens", "batch"], target="latency")
        self.assertEqual(result["status"], "fitted")
        self.assertAlmostEqual(result["coefficients"]["intercept"], 12.0)
        self.assertAlmostEqual(result["coefficients"]["tokens"], 2.5)
        self.assertAlmostEqual(result["coefficients"]["batch"], -3.0)
        self.assertEqual((result["n_train"], result["n_test"]), (80, 20))
        self.assertLess(result["test"]["rmse"], 1e-10)
        self.assertAlmostEqual(result["test"]["r2"], 1.0)
        self.assertEqual(result["feature_ranges"]["tokens"], {"min": 0.0, "max": 79.0})
        json.dumps(result, allow_nan=False)

    def test_holdout_does_not_change_coefficients_or_scaling(self) -> None:
        samples = [(float(index), (float(index),), 3.0 + 2 * index) for index in range(100)]
        for index in range(80, 100):
            samples[index] = (float(index), (float(index * 1000),), 103.0 + 2000 * index)
        result = fit_linear(reversed(samples), ["tokens"], target="latency")
        self.assertAlmostEqual(result["coefficients"]["intercept"], 3.0)
        self.assertAlmostEqual(result["coefficients"]["tokens"], 2.0)
        self.assertLess(result["train"]["rmse"], 1e-10)
        self.assertAlmostEqual(result["test"]["mae"], 100.0)
        self.assertEqual(result["feature_ranges"]["tokens"]["max"], 79.0)
        self.assertEqual(result["time_bounds"]["train"], {"start": 0.0, "end": 79.0})
        self.assertEqual(result["time_bounds"]["test"], {"start": 80.0, "end": 99.0})

    def test_drops_constant_and_collinear_predictors(self) -> None:
        samples = [
            (float(index), (float(index), float(index * 2 + 3), 7.0), 4.0 + 5 * index)
            for index in range(100)
        ]
        result = fit_linear(samples, ["tokens", "duplicate", "constant"], target="latency")
        self.assertEqual(result["status"], "fitted")
        self.assertEqual(
            result["dropped_features"],
            {"duplicate": "collinear_in_training", "constant": "constant_in_training"},
        )
        self.assertAlmostEqual(result["coefficients"]["tokens"], 5.0)
        self.assertAlmostEqual(result["coefficients"]["intercept"], 4.0)
        self.assertEqual(result["coefficients"]["duplicate"], 0.0)
        self.assertLess(result["test"]["rmse"], 1e-10)

    def test_constant_target_intercept_only(self) -> None:
        result = fit_linear(
            [(float(index), (1.0,), 7.0) for index in range(100)], ["x"], target="y"
        )
        self.assertEqual(result["status"], "fitted")
        self.assertAlmostEqual(result["coefficients"]["intercept"], 7.0)
        self.assertIsNone(result["train"]["r2"])
        self.assertIsNone(result["test"]["r2"])
        self.assertLess(result["test"]["rmse"], 1e-12)

    def test_invalid_samples_are_counted(self) -> None:
        samples = [(float(index), (float(index),), 2.0 * index) for index in range(30)]
        samples.extend(
            [
                (None, (1.0,), 2.0),
                (1.0, (None,), 2.0),
                (1.0, (1.0,), None),
                (1.0, (1.0, 2.0), 3.0),
                (1.0, (math.inf,), 2.0),
                (math.nan, (1.0,), 2.0),
                (1.0, (1.0,), "bad"),
                None,
            ]
        )
        result = fit_linear(samples, ["x"], target="y")
        self.assertEqual(result["status"], "fitted")
        self.assertEqual((result["n_total"], result["n_valid"], result["n_invalid"]), (38, 30, 8))
        self.assertEqual(
            result["invalid_counts"], {"missing": 3, "dimension": 1, "nonfinite": 2, "malformed": 2}
        )
        json.dumps(result, allow_nan=False)

    def test_insufficient_and_single_timestamp(self) -> None:
        result = fit_linear([], ["x"], target="y")
        self.assertEqual(result["status"], "unavailable")
        self.assertIn("insufficient_samples", result["reason"])
        result = fit_linear(
            [(1.0, (float(index),), 1.0) for index in range(100)], ["x"], target="y"
        )
        self.assertEqual(result["status"], "unavailable")
        self.assertIn("no_temporal_holdout", result["reason"])
        json.dumps(result, allow_nan=False)

    def test_same_timestamp_never_crosses_holdout(self) -> None:
        samples = [(float(index // 30), (float(index),), 2.0 * index) for index in range(100)]
        result = fit_linear(samples, ["x"], target="y")
        self.assertEqual((result["n_train"], result["n_test"]), (90, 10))
        self.assertLess(
            result["time_bounds"]["train"]["end"], result["time_bounds"]["test"]["start"]
        )

    def test_reservoir_cap_is_deterministic(self) -> None:
        samples = [(float(index), (float(index),), float(index % 11)) for index in range(1000)]
        first = fit_linear(iter(samples), ["x"], target="y", max_samples=100)
        second = fit_linear(iter(samples), ["x"], target="y", max_samples=100)
        self.assertEqual(first, second)
        self.assertEqual(first["n_sampled"], 100)
        self.assertEqual(first["n_valid"], 1000)
        self.assertLess(first["time_bounds"]["train"]["end"], first["time_bounds"]["test"]["start"])
        self.assertGreater(first["time_bounds"]["test"]["end"], 900)

    def test_large_finite_values_and_tiny_features(self) -> None:
        samples = [(float(index), (index * 1e-300,), 2.0 + index * 0.01) for index in range(100)]
        result = fit_linear(samples, ["x"], target="y")
        self.assertEqual(result["status"], "fitted")
        self.assertAlmostEqual(result["coefficients"]["x"] / 1e298, 1.0)
        self.assertLess(result["test"]["rmse"], 1e-12)
        samples = [(float(index), (index / 100.0,), 1e307 + index * 1e304) for index in range(100)]
        result = fit_linear(samples, ["x"], target="y")
        self.assertEqual(result["status"], "fitted")
        self.assertAlmostEqual(result["coefficients"]["intercept"] / 1e307, 1.0)
        json.dumps(result, allow_nan=False)

    def test_unrepresentable_coefficient_is_unavailable(self) -> None:
        samples = [(float(index), (index * 1e-310,), float(index)) for index in range(100)]
        result = fit_linear(samples, ["x"], target="y")
        self.assertEqual(result["status"], "unavailable")
        self.assertIn("numerical_overflow", result["reason"])
        json.dumps(result, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
