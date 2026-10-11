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
"""Bounded, dependency-free linear fits with chronological validation."""

from __future__ import annotations

import math
import random
from collections.abc import Iterable

__all__ = ["fit_linear"]

_Sample = tuple[float, tuple[float, ...], float]
_MIN_TRAIN = 20
_MIN_TEST = 5
_RANK_TOLERANCE = 1e-10


def _validate_sample(sample: object, feature_count: int) -> tuple[_Sample | None, str | None]:
    if not isinstance(sample, (tuple, list)) or len(sample) != 3:
        return None, "malformed"
    timestamp, features, target = sample
    if features is None:
        return None, "missing"
    if not isinstance(features, (tuple, list)) or len(features) != feature_count:
        return None, "dimension"
    values = (timestamp, *features, target)
    if any(value is None or value == "" for value in values):
        return None, "missing"
    try:
        numbers = tuple(float(value) for value in values)
    except (TypeError, ValueError):
        return None, "malformed"
    except OverflowError:
        return None, "nonfinite"
    if not all(math.isfinite(value) for value in numbers):
        return None, "nonfinite"
    return (numbers[0], numbers[1:-1], numbers[-1]), None


def _dot(left: list[float], right: list[float]) -> float:
    return math.fsum(a * b for a, b in zip(left, right))


def _finite(value: float) -> float:
    if not math.isfinite(value):
        raise OverflowError("An intermediate result exceeds floating-point range")
    return value


def _product_ratio(value: float, numerator: float, denominator: float) -> float:
    # Avoid overflowing an intermediate product when the final coefficient is finite.
    value_mantissa, value_exponent = math.frexp(value)
    numerator_mantissa, numerator_exponent = math.frexp(numerator)
    denominator_mantissa, denominator_exponent = math.frexp(denominator)
    return _finite(
        math.ldexp(
            value_mantissa * numerator_mantissa / denominator_mantissa,
            value_exponent + numerator_exponent - denominator_exponent,
        )
    )


def _solve(
    samples: list[_Sample], feature_names: list[str]
) -> tuple[dict[str, float], dict[str, str], list[tuple[int, float, float, float]], float, float]:
    """Fit centered/scaled columns with twice-orthogonalized modified Gram-Schmidt."""
    count = len(samples)
    root_count = math.sqrt(count)
    orthogonal = [[1.0 / root_count] * count]
    triangular = [[root_count]]
    active: list[tuple[int, float, float, float]] = []
    dropped: dict[str, str] = {}

    for index, name in enumerate(feature_names):
        values = [sample[1][index] for sample in samples]
        if min(values) == max(values):
            dropped[name] = "constant_in_training"
            continue
        scale = max(abs(value) for value in values)
        scaled = [value / scale for value in values]
        center = math.fsum(scaled) / count
        column = [value - center for value in scaled]
        norm = math.sqrt(_dot(column, column))
        if norm == 0.0:
            dropped[name] = "constant_in_training"
            continue
        column = [value / norm for value in column]
        projections = [0.0] * len(orthogonal)
        # A second pass controls loss of orthogonality for nearly dependent predictors.
        for _ in range(2):
            for previous, basis in enumerate(orthogonal):
                projection = _dot(basis, column)
                projections[previous] += projection
                column = [value - projection * item for value, item in zip(column, basis)]
        residual_norm = math.sqrt(_dot(column, column))
        if residual_norm <= _RANK_TOLERANCE:
            dropped[name] = "collinear_in_training"
            continue
        orthogonal.append([value / residual_norm for value in column])
        triangular.append([*projections, residual_norm])
        active.append((index, scale, center, norm))

    target_scale = max(abs(sample[2]) for sample in samples) or 1.0
    scaled_targets = [sample[2] / target_scale for sample in samples]
    solution = [_dot(basis, scaled_targets) for basis in orthogonal]
    for index in range(len(solution) - 1, -1, -1):
        remainder = math.fsum(
            triangular[later][index] * solution[later] for later in range(index + 1, len(solution))
        )
        solution[index] = (solution[index] - remainder) / triangular[index][index]

    coefficients = {name: 0.0 for name in feature_names}
    intercept = solution[0]
    normalized: list[tuple[int, float, float, float]] = []
    for (index, scale, center, norm), coefficient in zip(active, solution[1:]):
        normalized_coefficient = coefficient / norm
        coefficients[feature_names[index]] = _product_ratio(
            normalized_coefficient, target_scale, scale
        )
        intercept -= normalized_coefficient * center
        normalized.append((index, scale, center, normalized_coefficient))
    coefficients["intercept"] = _finite(intercept * target_scale)
    return coefficients, dropped, normalized, solution[0], target_scale


def _predict(
    samples: list[_Sample],
    normalized: list[tuple[int, float, float, float]],
    intercept: float,
    target_scale: float,
) -> list[float]:
    predictions = []
    for _, features, _ in samples:
        prediction = intercept + math.fsum(
            _finite(coefficient * _finite(features[index] / scale - center))
            for index, scale, center, coefficient in normalized
        )
        predictions.append(_finite(prediction * target_scale))
    return predictions


def _errors(samples: list[_Sample], predictions: list[float]) -> dict[str, float | None]:
    actual = [sample[2] for sample in samples]
    scale = (
        max(max(abs(value) for value in actual), max(abs(value) for value in predictions)) or 1.0
    )
    scaled_actual = [value / scale for value in actual]
    residual = [value - prediction / scale for value, prediction in zip(scaled_actual, predictions)]
    squared_error = _dot(residual, residual)
    mean = math.fsum(scaled_actual) / len(samples)
    total = math.fsum((value - mean) ** 2 for value in scaled_actual)
    r2 = 1.0 - squared_error / total if total and min(actual) != max(actual) else None
    return {
        "mae": _finite(math.fsum(abs(value) for value in residual) / len(samples) * scale),
        "rmse": _finite(math.sqrt(squared_error / len(samples)) * scale),
        "r2": r2 if r2 is not None and math.isfinite(r2) else None,
    }


def fit_linear(
    samples: Iterable[tuple[float, tuple[float, ...], float]],
    feature_names: list[str],
    *,
    target: str,
    target_unit: str = "ms",
    max_samples: int = 20000,
) -> dict:
    """Fit a linear model on earlier samples and validate on a later holdout.

    Args:
        samples: Iterable of (epoch seconds, predictor tuple, target) observations.
            Missing, malformed, dimension-mismatched and nonfinite rows are counted
            and excluded before deterministic reservoir sampling.
        feature_names: Unique predictor names in tuple order; ``intercept`` is reserved.
        target: Descriptive target name for the serialized result.
        target_unit: Unit of the target, predictions, MAE and RMSE.
        max_samples: Maximum retained observations; memory scales with this bound.

    Returns:
        JSON-compatible fit metadata, coefficients in original units, training
        ranges and separate training/holdout errors. The nearest timestamp boundary
        to an 80/20 split with at least 20 training and 5 holdout observations is
        used. Identical timestamps never cross the split. All preprocessing and
        coefficients use training samples only; the model is never refit on holdout.
        Insufficient data or numerical overflow produces ``status=unavailable``
        with a reason. Constant/collinear predictors have zero coefficients and
        explicit drop reasons. An intercept-only fit remains valid.

    Raises:
        ValueError: If predictor names or the sampling bound are invalid.
    """
    if not isinstance(max_samples, int) or max_samples < 1:
        raise ValueError("max_samples must be a positive integer")
    if any(
        not isinstance(name, str) or not name or name == "intercept" for name in feature_names
    ) or len(set(feature_names)) != len(feature_names):
        raise ValueError("feature_names must be unique nonempty strings excluding 'intercept'")

    invalid_counts = dict.fromkeys(("malformed", "dimension", "missing", "nonfinite"), 0)
    result = {
        "status": "unavailable",
        "reason": None,
        "target": target,
        "target_unit": target_unit,
        "feature_names": list(feature_names),
        "n_total": 0,
        "n_valid": 0,
        "n_invalid": 0,
        "n_sampled": 0,
        "n_train": 0,
        "n_test": 0,
        "invalid_counts": invalid_counts,
        "coefficients": {},
        "dropped_features": {},
        "feature_ranges": {},
        "train": {"mae": None, "rmse": None, "r2": None},
        "test": {"mae": None, "rmse": None, "r2": None},
        "time_bounds": {"train": None, "test": None},
        "sampling": {"method": "reservoir", "seed": 0, "max_samples": max_samples},
        "method": "train-centered/scaled OLS; reorthogonalized QR; intercept included",
        "split": "nearest distinct-timestamp boundary to 80/20; minimum train=20, test=5",
    }
    retained: list[_Sample] = []
    generator = random.Random(0)
    for sample in samples:
        result["n_total"] += 1
        valid, reason = _validate_sample(sample, len(feature_names))
        if valid is None:
            invalid_counts[reason] += 1
            result["n_invalid"] += 1
            continue
        result["n_valid"] += 1
        if len(retained) < max_samples:
            retained.append(valid)
        else:
            index = generator.randrange(result["n_valid"])
            if index < max_samples:
                retained[index] = valid
    retained.sort(key=lambda sample: sample[0])
    result["n_sampled"] = len(retained)
    if len(retained) < _MIN_TRAIN + _MIN_TEST:
        result["reason"] = (
            "insufficient_samples: require at least 20 training and 5 holdout samples"
        )
        return result
    boundaries = [
        index
        for index in range(_MIN_TRAIN, len(retained) - _MIN_TEST + 1)
        if retained[index - 1][0] < retained[index][0]
    ]
    if not boundaries:
        result["reason"] = "no_temporal_holdout: no timestamp boundary meets minimum sample counts"
        return result
    split = min(boundaries, key=lambda index: abs(index - 0.8 * len(retained)))
    training, testing = retained[:split], retained[split:]
    result["n_train"], result["n_test"] = len(training), len(testing)
    result["time_bounds"] = {
        "train": {"start": training[0][0], "end": training[-1][0]},
        "test": {"start": testing[0][0], "end": testing[-1][0]},
    }
    result["feature_ranges"] = {
        name: {
            "min": min(sample[1][index] for sample in training),
            "max": max(sample[1][index] for sample in training),
        }
        for index, name in enumerate(feature_names)
    }
    try:
        coefficients, dropped, normalized, intercept, target_scale = _solve(training, feature_names)
        train_errors = _errors(training, _predict(training, normalized, intercept, target_scale))
        test_errors = _errors(testing, _predict(testing, normalized, intercept, target_scale))
    except (OverflowError, ZeroDivisionError):
        result["reason"] = "numerical_overflow: coefficients or errors exceed floating-point range"
        return result
    result.update(
        status="fitted",
        coefficients=coefficients,
        dropped_features=dropped,
        train=train_errors,
        test=test_errors,
    )
    return result
