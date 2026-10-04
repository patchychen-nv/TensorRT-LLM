# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Option B precision must reject shared reusable prefixes and invalid numerical controls."""

import copy
import importlib.util
from pathlib import Path

import pytest
import torch

_MODULE_PATH = Path(__file__).resolve().parents[3] / "integration/defs/dkv/dkv_precision.py"
_SPEC = importlib.util.spec_from_file_location("dkv_precision_policy_under_test", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_POLICY = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_POLICY)

pytestmark = pytest.mark.cpu_only


def _run(steps: int = 1) -> dict:
    return {
        "tokens": [[2] * steps],
        "logits": [torch.tensor([[0.0, 0.5, 1.0]] * steps)],
    }


def test_reuse_off_allows_shared_few_shot_prefixes() -> None:
    _POLICY.validate_precision_inputs(
        enable_block_reuse=False,
        tokens_per_block=32,
        prompts=[[1] * 256, [1] * 256],
    )


def test_reuse_off_does_not_need_tokenized_prompts() -> None:
    _POLICY.validate_precision_inputs(enable_block_reuse=False, tokens_per_block=32)


def test_reuse_on_accepts_only_sub_block_shared_prefixes() -> None:
    _POLICY.validate_precision_inputs(
        enable_block_reuse=True,
        tokens_per_block=4,
        prompts=[[1, 2, 3, 4, 5], [1, 2, 3, 9, 5], [1, 2, 3], [1, 2, 3]],
    )


@pytest.mark.parametrize(
    "prompts",
    [
        [[1, 2, 3, 4], [1, 2, 3, 4]],
        [[1, 2, 3, 4], [1, 2, 3, 4, 9]],
        [[1, 2, 3, 4, 8], [1, 2, 3, 4, 9]],
        [[1, 2, 3, 4], [9, 9, 9, 9], [1, 2, 3, 4]],
    ],
    ids=["exact-block", "strict-prefix", "different-suffix", "earlier-request"],
)
def test_reuse_on_rejects_shared_full_blocks(prompts: list[list[int]]) -> None:
    with pytest.raises(ValueError, match="share at least 4 prefix tokens"):
        _POLICY.validate_precision_inputs(
            enable_block_reuse=True, tokens_per_block=4, prompts=prompts
        )


@pytest.mark.parametrize("prompts", [None, [], [[]]])
def test_reuse_on_rejects_missing_prompt_evidence(prompts: list[list[int]] | None) -> None:
    with pytest.raises(ValueError, match="prompt"):
        _POLICY.validate_precision_inputs(
            enable_block_reuse=True, tokens_per_block=4, prompts=prompts
        )


@pytest.mark.parametrize("reuse", [None, "False", 0])
def test_implicit_reuse_setting_is_not_precision_evidence(reuse: object) -> None:
    with pytest.raises(ValueError, match="explicit boolean"):
        _POLICY.validate_precision_inputs(enable_block_reuse=reuse, tokens_per_block=4)


@pytest.mark.parametrize("tokens_per_block", [0, -1, True])
def test_invalid_block_size_is_rejected(tokens_per_block: int) -> None:
    with pytest.raises(ValueError, match="positive tokens_per_block"):
        _POLICY.validate_precision_inputs(
            enable_block_reuse=False, tokens_per_block=tokens_per_block
        )


@pytest.mark.parametrize("steps", [1, 8], ids=["aggregate", "disaggregated"])
def test_exact_control_requires_and_accepts_exact_dkv(steps: int) -> None:
    control = _run(steps)
    result = _POLICY.compare_precision_runs(control, copy.deepcopy(control), copy.deepcopy(control))
    assert result["comparison"] == "bitwise"
    assert result["atol"] == result["rtol"] == 0


def test_exact_control_never_relaxes_tolerance_for_dkv() -> None:
    control, replay, dkv = _run(), _run(), _run()
    dkv["logits"][0][0, 0] = 1e-5
    with pytest.raises(AssertionError, match="Reproducible ADP logits differ"):
        _POLICY.compare_precision_runs(control, replay, dkv)


def test_non_bitwise_control_uses_fixed_tolerance_and_matching_tokens() -> None:
    control, replay, dkv = _run(), _run(), _run()
    replay["logits"][0][0, 0] = 0.005
    dkv["logits"][0][0, 0] = 0.004
    result = _POLICY.compare_precision_runs(control, replay, dkv)
    assert result["comparison"] == "token_and_logits_tolerance"
    assert result["atol"] == result["rtol"] == 1e-2


def test_dkv_must_match_both_tolerated_control_runs() -> None:
    control, replay, dkv = _run(), _run(), _run()
    replay["logits"][0][0, 0] = 0.005
    dkv["logits"][0][0, 0] = -0.009
    with pytest.raises(AssertionError):
        _POLICY.compare_precision_runs(control, replay, dkv)


def test_unstable_control_tokens_fail_before_inspecting_dkv() -> None:
    control, replay = _run(), _run()
    replay["tokens"][0] = [1]
    with pytest.raises(AssertionError, match="ADP control tokens are not reproducible"):
        _POLICY.compare_precision_runs(control, replay, {})


def test_control_tolerance_failure_never_becomes_dkv_success() -> None:
    control, replay = _run(), _run()
    replay["logits"][0][0, 0] = 0.03
    with pytest.raises(AssertionError, match="ADP control logits exceed"):
        _POLICY.compare_precision_runs(control, replay, copy.deepcopy(control))


def test_matching_logits_do_not_override_different_dkv_tokens() -> None:
    control, replay, dkv = _run(), _run(), _run()
    dkv["tokens"][0] = [1]
    with pytest.raises(AssertionError, match="ADP and DKV generated different tokens"):
        _POLICY.compare_precision_runs(control, replay, dkv)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_logits_are_never_precision_success(value: float) -> None:
    control = _run()
    control["logits"][0][0, 0] = value
    with pytest.raises(AssertionError, match="finite floating-point"):
        _POLICY.compare_precision_runs(control, copy.deepcopy(control), copy.deepcopy(control))


@pytest.mark.parametrize("values", [torch.ones(1), torch.ones(2, 3), torch.ones(1, 0)])
def test_logits_must_cover_every_generated_token(values: torch.Tensor) -> None:
    control, replay, dkv = _run(), _run(), _run()
    dkv["logits"][0] = values
    with pytest.raises(AssertionError, match="logits must have shape"):
        _POLICY.compare_precision_runs(control, replay, dkv)


def test_logits_vocabulary_shape_cannot_broadcast() -> None:
    control, replay, dkv = _run(), _run(), _run()
    dkv["logits"][0] = torch.ones(1, 1)
    with pytest.raises(AssertionError, match="logits shapes differ"):
        _POLICY.compare_precision_runs(control, replay, dkv)


def test_missing_requests_cannot_be_silently_truncated() -> None:
    control, replay, dkv = _run(), _run(), _run()
    dkv["tokens"].append([2])
    with pytest.raises(AssertionError, match="one token sequence and logits tensor per request"):
        _POLICY.compare_precision_runs(control, replay, dkv)


def test_empty_runs_are_not_precision_evidence() -> None:
    with pytest.raises(AssertionError, match="per request"):
        _POLICY.compare_precision_runs(
            {"tokens": [], "logits": []}, {"tokens": [], "logits": []}, {"tokens": [], "logits": []}
        )
