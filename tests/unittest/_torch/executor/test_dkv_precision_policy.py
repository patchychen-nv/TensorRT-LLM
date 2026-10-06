# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DKV precision must reject shared reusable prefixes and invalid numerical controls."""

import copy
import dataclasses
import importlib.util
import math
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
        enable_partial_reuse=False,
        prompts=[[1, 2, 3, 4, 5], [1, 2, 3, 9, 5], [1, 2, 3], [1, 2, 3]],
    )


@pytest.mark.parametrize("partial", [None, True, 0])
def test_reuse_on_requires_partial_reuse_to_be_explicitly_off(partial: object) -> None:
    with pytest.raises(ValueError, match="enable_partial_reuse=False"):
        _POLICY.validate_precision_inputs(
            enable_block_reuse=True,
            tokens_per_block=4,
            enable_partial_reuse=partial,
            prompts=[[1, 2, 3, 4], [5, 6, 7, 8]],
        )


def test_partial_reuse_is_irrelevant_when_reuse_is_off() -> None:
    _POLICY.validate_precision_inputs(
        enable_block_reuse=False, tokens_per_block=4, enable_partial_reuse=True
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
            enable_block_reuse=True,
            tokens_per_block=4,
            enable_partial_reuse=False,
            prompts=prompts,
        )


@pytest.mark.parametrize("prompts", [None, [], [[]]])
def test_reuse_on_rejects_missing_prompt_evidence(prompts: list[list[int]] | None) -> None:
    with pytest.raises(ValueError, match="prompt"):
        _POLICY.validate_precision_inputs(
            enable_block_reuse=True,
            tokens_per_block=4,
            enable_partial_reuse=False,
            prompts=prompts,
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


_PROMPTS = 12
_VOCABULARY = 64


def _base_logits(seed: int = 0) -> torch.Tensor:
    """Independent sharp distributions per prompt, standing in for distinct natural prompts."""
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(_PROMPTS, _VOCABULARY, generator=generator) * 5


def _noisy_run(base: torch.Tensor, noise: float, seed: int) -> dict:
    generator = torch.Generator().manual_seed(seed)
    logits = base + noise * torch.randn(base.shape, generator=generator)
    return {
        "tokens": [[int(row.argmax())] for row in logits],
        "logits": [row.unsqueeze(0) for row in logits],
    }


def _noisy_runs(noise: float = 0.1) -> tuple[dict, dict, dict]:
    base = _base_logits()
    return _noisy_run(base, noise, 1), _noisy_run(base, noise, 2), _noisy_run(base, noise, 3)


def test_noisy_policy_accepts_dkv_inside_the_adp_noise() -> None:
    control, replay, dkv = _noisy_runs()
    result = _POLICY.compare_precision_runs(control, replay, dkv, _POLICY.NOISY_POLICY)
    assert result["comparison"] == "noise_envelope"
    assert result["decisive_prompts"] >= _PROMPTS // 2
    assert 0 < result["mean_adp_self_tv"] < 0.2
    assert result["mean_extra_tv"] < _POLICY.NOISY_POLICY.max_mean_extra_tv
    separation = _POLICY.assess_prompt_separation(
        control, result["mean_adp_self_tv"], _POLICY.NOISY_POLICY
    )
    assert separation["p10_prompt_separation"] >= separation["required_separation"]


def test_noisy_policy_rejects_a_request_that_read_another_requests_kv() -> None:
    control, replay, dkv = _noisy_runs()
    dkv["logits"][3] = control["logits"][5].clone()
    dkv["tokens"][3] = control["tokens"][5]
    with pytest.raises(AssertionError, match="decisive top-two margin|differs from ADP"):
        _POLICY.compare_precision_runs(control, replay, dkv, _POLICY.NOISY_POLICY)


def test_noisy_policy_rejects_extra_distance_even_without_a_token_flip() -> None:
    control, replay, dkv = _noisy_runs(noise=0.05)
    # Keep the argmax but move most of the probability mass elsewhere.
    shifted = dkv["logits"][2].clone()
    runner_up = int(shifted.topk(2).indices[0, 1])
    shifted[0, runner_up] = shifted[0, int(dkv["tokens"][2][0])] - 0.01
    dkv["logits"][2] = shifted
    with pytest.raises(AssertionError, match="differs from ADP"):
        _POLICY.compare_precision_runs(control, replay, dkv, _POLICY.NOISY_POLICY)


def test_noisy_policy_tolerates_a_token_flip_inside_the_top_two_margin() -> None:
    control, replay, dkv = _noisy_runs(noise=0.05)
    for run, winner in ((control, 0), (replay, 1), (dkv, 0)):
        logits = run["logits"][4].clone()
        top = logits.max()
        logits[0, :2] = torch.tensor([top + 0.05 * (winner == 0), top + 0.05 * (winner == 1)])
        run["logits"][4] = logits
        run["tokens"][4] = [winner]
    result = _POLICY.compare_precision_runs(control, replay, dkv, _POLICY.NOISY_POLICY)
    assert result["decisive_prompts"] <= _PROMPTS - 1


_POSITIONS = 4


def _continuation_run(base: torch.Tensor, noise: float, seed: int) -> dict:
    """Runs of several generated tokens per prompt: ``base`` is [prompts, positions, vocabulary]."""
    generator = torch.Generator().manual_seed(seed)
    logits = base + noise * torch.randn(base.shape, generator=generator)
    return {
        "tokens": [[int(row.argmax()) for row in prompt] for prompt in logits],
        "logits": list(logits),
    }


def _continuation_runs(noise: float = 0.001) -> tuple[dict, dict, dict]:
    generator = torch.Generator().manual_seed(0)
    base = torch.randn(_PROMPTS, _POSITIONS, _VOCABULARY, generator=generator) * 5
    return tuple(_continuation_run(base, noise, seed) for seed in (1, 2, 3))


def _make_near_tie(run: dict, prompt: int, position: int, winner: int) -> None:
    """Make two tokens of one position nearly equal and let ``winner`` take the argmax."""
    logits = run["logits"][prompt].clone()
    top = logits[position].max()
    logits[position, :2] = torch.tensor([top + 0.05 * (winner == 0), top + 0.05 * (winner == 1)])
    run["logits"][prompt] = logits
    run["tokens"][prompt][position] = winner


def test_softmax_distance_averages_over_the_generated_positions() -> None:
    first = torch.tensor([[4.0, 0.0], [0.0, 4.0]])
    second = torch.tensor([[4.0, 0.0], [4.0, 0.0]])
    expected = 0.5 * _POLICY.softmax_distance(first[1:], second[1:])
    assert _POLICY.softmax_distance(first, second) == pytest.approx(expected)


def test_noisy_policy_compares_every_position_of_a_continuation_that_agrees() -> None:
    control, replay, dkv = _continuation_runs()
    result = _POLICY.compare_precision_runs(control, replay, dkv, _POLICY.NOISY_POLICY)
    assert result["compared_positions"] == _PROMPTS * _POSITIONS
    assert result["decisive_positions"] >= _PROMPTS * _POSITIONS // 2
    assert result["mean_extra_tv"] < _POLICY.NOISY_POLICY.max_mean_extra_tv


def test_noisy_policy_stops_comparing_after_a_flip_inside_the_top_two_margin() -> None:
    control, replay, dkv = _continuation_runs()
    for run, winner in ((control, 0), (replay, 1), (dkv, 0)):
        _make_near_tie(run, prompt=2, position=1, winner=winner)
    # Once the runs chose different tokens, the later positions answer different questions.
    generator = torch.Generator().manual_seed(9)
    dkv["logits"][2][2:] = torch.randn(_POSITIONS - 2, _VOCABULARY, generator=generator) * 5
    dkv["tokens"][2][2:] = [int(row.argmax()) for row in dkv["logits"][2][2:]]
    result = _POLICY.compare_precision_runs(control, replay, dkv, _POLICY.NOISY_POLICY)
    assert result["compared_positions"] == _PROMPTS * _POSITIONS - (_POSITIONS - 2)


def test_noisy_policy_rejects_a_decisive_flip_at_a_later_position() -> None:
    control, replay, dkv = _continuation_runs()
    margins = _POLICY.top_two_margins(dkv["logits"][5])
    position = int((margins >= _POLICY.NOISY_POLICY.min_margin).nonzero()[-1])
    assert position > 0
    dkv["tokens"][5][position] = int(dkv["logits"][5][position].topk(2).indices[1])
    with pytest.raises(AssertionError, match=f"decisive top-two margin at position {position}"):
        _POLICY.compare_precision_runs(control, replay, dkv, _POLICY.NOISY_POLICY)


def test_noisy_policy_rejects_extra_distance_over_a_whole_continuation() -> None:
    control, replay, dkv = _continuation_runs()
    shifted = dkv["logits"][4].clone()
    for position in range(_POSITIONS):
        # Keep the argmax but move half of the probability mass to the runner-up.
        runner_up = int(shifted[position].topk(2).indices[1])
        shifted[position, runner_up] = shifted[position, dkv["tokens"][4][position]] - 0.01
    dkv["logits"][4] = shifted
    with pytest.raises(AssertionError, match="differs from ADP"):
        _POLICY.compare_precision_runs(control, replay, dkv, _POLICY.NOISY_POLICY)


def test_noisy_policy_needs_enough_decisive_prompts_to_compare_tokens() -> None:
    base = torch.zeros(_PROMPTS, _VOCABULARY)
    base[:, 0] = 0.1
    runs = [_noisy_run(base, 0.01, seed) for seed in (1, 2, 3)]
    with pytest.raises(AssertionError, match="cannot compare tokens"):
        _POLICY.compare_precision_runs(*runs, _POLICY.NOISY_POLICY)


def test_noisy_policy_rejects_an_adp_control_that_is_too_noisy() -> None:
    control, replay, dkv = _noisy_runs(noise=6.0)
    with pytest.raises(AssertionError, match="ADP control is noisier"):
        _POLICY.compare_precision_runs(control, replay, dkv, _POLICY.NOISY_POLICY)


_ALLOWANCE = dataclasses.replace(_POLICY.NOISY_POLICY, mean_extra_sigmas=2.0)


def _extra_distance_runs(extras: list[float]) -> tuple[dict, dict, dict]:
    """Runs of one two-token position per prompt: both ADP runs agree, and the DKV distribution of
    prompt ``i`` lies ``extras[i]`` from theirs in total variation."""

    def logits(probability: float) -> torch.Tensor:
        return torch.tensor([[math.log(probability / (1 - probability)), 0.0]])

    control = {"tokens": [[0]] * len(extras), "logits": [logits(0.9)] * len(extras)}
    dkv = {"tokens": [[0]] * len(extras), "logits": [logits(0.9 - extra) for extra in extras]}
    return control, copy.deepcopy(control), dkv


def test_the_mean_extra_distance_of_scattered_prompts_may_exceed_the_allowance_by_its_error() -> (
    None
):
    # The mean is 0.03 and its standard error 0.0086: a correct run lies there now and then.
    runs = _extra_distance_runs([0.06, 0.0] * 7)
    with pytest.raises(AssertionError, match="of which 0.0200 is allowed"):
        _POLICY.compare_precision_runs(*runs, _POLICY.NOISY_POLICY)
    result = _POLICY.compare_precision_runs(*runs, _ALLOWANCE)
    assert result["mean_extra_tv"] == pytest.approx(0.03)


def test_a_bias_that_every_prompt_shares_gets_no_allowance() -> None:
    runs = _extra_distance_runs([0.03] * 14)
    with pytest.raises(AssertionError, match="of which 0.0200 is allowed"):
        _POLICY.compare_precision_runs(*runs, _ALLOWANCE)


def test_a_prompt_that_is_far_off_is_not_excused_by_the_allowance() -> None:
    runs = _extra_distance_runs([0.3] + [0.0] * 13)
    with pytest.raises(AssertionError, match="One prompt differs"):
        _POLICY.compare_precision_runs(*runs, _ALLOWANCE)


def test_the_allowance_needs_a_spread_to_estimate() -> None:
    assert _POLICY.mean_extra_limit(_ALLOWANCE, [0.5]) == _ALLOWANCE.max_mean_extra_tv
    assert _POLICY.mean_extra_limit(_ALLOWANCE, [0.01, 0.03]) > _ALLOWANCE.max_mean_extra_tv
    noisy = _POLICY.NOISY_POLICY
    assert _POLICY.mean_extra_limit(noisy, [0.01, 0.2]) == noisy.max_mean_extra_tv


def test_exact_policy_still_rejects_noise_that_a_noisy_policy_accepts() -> None:
    control, replay, dkv = _noisy_runs(noise=0.3)
    with pytest.raises(AssertionError):
        _POLICY.compare_precision_runs(control, replay, dkv, _POLICY.EXACT_POLICY)


def test_failing_requests_names_none_for_a_dkv_run_inside_the_noise() -> None:
    control, replay, dkv = _noisy_runs()
    assert _POLICY.failing_requests(control, replay, dkv, _POLICY.NOISY_POLICY) == []


def test_failing_requests_names_the_request_that_read_another_requests_kv() -> None:
    control, replay, dkv = _noisy_runs()
    dkv["logits"][3] = control["logits"][5].clone()
    dkv["tokens"][3] = control["tokens"][5]
    assert _POLICY.failing_requests(control, replay, dkv, _POLICY.NOISY_POLICY) == [3]
    with pytest.raises(AssertionError):
        _POLICY.compare_precision_runs(control, replay, dkv, _POLICY.NOISY_POLICY)


def test_failing_requests_names_a_request_that_only_moved_probability_mass() -> None:
    control, replay, dkv = _noisy_runs(noise=0.05)
    shifted = dkv["logits"][2].clone()
    runner_up = int(shifted.topk(2).indices[0, 1])
    shifted[0, runner_up] = shifted[0, int(dkv["tokens"][2][0])] - 0.01
    dkv["logits"][2] = shifted
    assert _POLICY.failing_requests(control, replay, dkv, _POLICY.NOISY_POLICY) == [2]


def test_failing_requests_names_every_corrupted_request() -> None:
    control, replay, dkv = _noisy_runs()
    for victim, source in ((1, 4), (8, 9)):
        dkv["logits"][victim] = control["logits"][source].clone()
        dkv["tokens"][victim] = control["tokens"][source]
    assert _POLICY.failing_requests(control, replay, dkv, _POLICY.NOISY_POLICY) == [1, 8]


def test_failing_requests_ignores_a_flip_inside_the_top_two_margin() -> None:
    control, replay, dkv = _continuation_runs()
    for run, winner in ((control, 0), (replay, 1), (dkv, 0)):
        _make_near_tie(run, prompt=2, position=1, winner=winner)
    assert _POLICY.failing_requests(control, replay, dkv, _POLICY.NOISY_POLICY) == []


def test_far_requests_names_the_request_that_read_another_requests_kv() -> None:
    control, replay, dkv = _noisy_runs()
    assert _POLICY.far_requests(control, replay, dkv, _POLICY.NOISY_POLICY) == []
    dkv["logits"][3] = control["logits"][5].clone()
    dkv["tokens"][3] = control["tokens"][5]
    assert _POLICY.far_requests(control, replay, dkv, _POLICY.NOISY_POLICY) == [3]


def test_far_requests_leaves_out_a_flip_at_a_tie_but_failing_requests_names_it() -> None:
    """A top-two gap of exactly ``min_margin`` is decisive for the margin check, and a flip across
    it is a distance of 0.245, which is not far. The noise of a model can do that on its own."""
    confident = torch.tensor([[3.0, 0.0]])
    tie = torch.tensor([[0.5, 0.0]])
    control = {"tokens": [[0]] * 6, "logits": [confident] * 4 + [tie] + [confident]}
    replay = copy.deepcopy(control)
    dkv = copy.deepcopy(control)
    dkv["tokens"][4] = [1]
    dkv["logits"][4] = tie.flip(dims=[1])
    assert _POLICY.failing_requests(control, replay, dkv, _POLICY.NOISY_POLICY) == [4]
    assert _POLICY.far_requests(control, replay, dkv, _POLICY.NOISY_POLICY) == []


def test_far_requests_counts_logits_that_are_not_finite_as_far() -> None:
    control, replay, dkv = _noisy_runs()
    dkv["logits"][2] = torch.full_like(dkv["logits"][2], float("nan"))
    assert _POLICY.far_requests(control, replay, dkv, _POLICY.NOISY_POLICY) == [2]


def test_far_requests_under_the_exact_policy_are_the_failing_requests() -> None:
    control = _noisy_run(_base_logits(), 0.0, 1)
    replay, dkv = copy.deepcopy(control), copy.deepcopy(control)
    dkv["logits"][6][0, 0] += 1e-5
    assert _POLICY.far_requests(control, replay, dkv) == _POLICY.failing_requests(
        control, replay, dkv
    )
    assert _POLICY.far_requests(control, replay, dkv) == [6]


def test_failing_requests_under_the_exact_policy_sees_the_smallest_difference() -> None:
    control = _noisy_run(_base_logits(), 0.0, 1)
    replay, dkv = copy.deepcopy(control), copy.deepcopy(control)
    assert _POLICY.failing_requests(control, replay, dkv) == []
    dkv["logits"][6][0, 0] += 1e-5
    assert _POLICY.failing_requests(control, replay, dkv) == [6]
    dkv["tokens"][9] = [(dkv["tokens"][9][0] + 1) % _VOCABULARY]
    assert _POLICY.failing_requests(control, replay, dkv) == [6, 9]


def test_failing_requests_under_the_exact_policy_uses_the_tolerance_for_a_tolerated_control() -> (
    None
):
    control = _noisy_run(_base_logits(), 0.0, 1)
    replay, dkv = copy.deepcopy(control), copy.deepcopy(control)
    replay["logits"][4][0, 0] += 0.005
    dkv["logits"][4][0, 0] += 0.004
    assert _POLICY.failing_requests(control, replay, dkv) == []
    dkv["logits"][4][0, 0] += 0.05
    assert _POLICY.failing_requests(control, replay, dkv) == [4]


def test_prompt_separation_rejects_prompts_with_similar_distributions() -> None:
    similar = {
        "tokens": [[0]] * 6,
        "logits": [torch.tensor([[3.0, 0.0, 0.0]]) + 0.01 * index for index in range(6)],
    }
    with pytest.raises(AssertionError, match="not separable"):
        _POLICY.assess_prompt_separation(similar, 0.0, _POLICY.EXACT_POLICY)


def test_prompt_separation_scales_with_the_noise() -> None:
    control, _, _ = _noisy_runs()
    _POLICY.assess_prompt_separation(control, 0.05, _POLICY.NOISY_POLICY)
    with pytest.raises(AssertionError, match="not separable"):
        _POLICY.assess_prompt_separation(control, 0.5, _POLICY.NOISY_POLICY)


def _word_encode(text: str, special: bool = False) -> list[int]:
    ids = [sum(map(ord, word)) % 50021 + 1 for word in text.split()]
    return [0, *ids] if special else ids


@pytest.mark.parametrize("tokens_per_block", [32, 128])
def test_prompt_set_is_discriminating_and_has_no_shared_leading_block(
    tokens_per_block: int,
) -> None:
    prompts, answers = _POLICY.build_precision_prompts(_word_encode, tokens_per_block)
    assert len(prompts) == len(_POLICY.PRECISION_PLACEMENT) == 14
    assert [len(prompt) for prompt in prompts[:10]] == list(_POLICY._PARAGRAPH_LENGTHS)
    assert len({tuple(prompt[:tokens_per_block]) for prompt in prompts}) == len(prompts)
    assert [index for index, _ in answers] == [10, 11, 12, 13]
    assert {0, 1} == set(_POLICY.PRECISION_PLACEMENT)
    # Both ranks see consecutive requests as well as alternation.
    assert any(a == b for a, b in zip(_POLICY.PRECISION_PLACEMENT, _POLICY.PRECISION_PLACEMENT[1:]))
    assert any(a != b for a, b in zip(_POLICY.PRECISION_PLACEMENT, _POLICY.PRECISION_PLACEMENT[1:]))


@pytest.mark.parametrize("group_size", [3, 4, 8])
def test_precision_placement_covers_every_rank_of_a_larger_group(group_size: int) -> None:
    placement = _POLICY.precision_placement(group_size)
    assert len(placement) == len(_POLICY.PRECISION_PLACEMENT)
    assert set(placement) == set(range(group_size))


def test_precision_placement_keeps_the_two_rank_pattern() -> None:
    assert _POLICY.precision_placement(2) is _POLICY.PRECISION_PLACEMENT


def test_burst_prompts_have_the_requested_lengths_and_unique_leading_blocks() -> None:
    lengths = (64, 256, 700)
    prompts = _POLICY.build_burst_prompts(_word_encode, 20, lengths)
    assert [len(prompt) for prompt in prompts] == [lengths[i % 3] for i in range(20)]
    assert len({tuple(prompt[:32]) for prompt in prompts}) == 20
    _POLICY.validate_precision_inputs(
        enable_block_reuse=True, tokens_per_block=32, enable_partial_reuse=False, prompts=prompts
    )


def test_prefix_pairs_hold_a_prompt_and_the_prompt_with_one_token_more() -> None:
    prompts = _POLICY.build_prefix_pairs(_word_encode, 5, 96)
    firsts, seconds = prompts[0::2], prompts[1::2]
    assert [len(prompt) for prompt in firsts] == [96] * 5
    assert [len(prompt) for prompt in seconds] == [97] * 5
    assert all(second[:96] == first for first, second in zip(firsts, seconds))
    assert len({tuple(prompt[:32]) for prompt in firsts}) == 5


def test_prefix_pairs_end_in_the_given_tails_in_turn() -> None:
    tails = ["The capital of France is", "Two plus two equals"]
    prompts = _POLICY.build_prefix_pairs(_word_encode, 3, 96, tails=tails)
    seconds = prompts[1::2]
    assert [second[96:] for second in seconds] == [_word_encode(tails[i % 2]) for i in range(3)]
    assert all(second[:96] == first for first, second in zip(prompts[0::2], seconds))


def test_known_answers_detect_a_degraded_control() -> None:
    answers = [(0, ("paris",)), (1, ("four", "4"))]
    words = {1: " Paris", 2: " four", 3: " cheese"}

    def decode(ids: list[int]) -> str:
        return words[ids[0]]

    run = {"tokens": [[1], [2]], "logits": []}
    _POLICY.assert_known_answers(run, answers, decode, "ADP control")
    run["tokens"][1] = [3]
    with pytest.raises(AssertionError, match="ADP control answered 'cheese' for prompt 1"):
        _POLICY.assert_known_answers(run, answers, decode, "ADP control")
