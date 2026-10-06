# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Numerical acceptance rules and the discriminating prompt set for the DKV test suites."""

import itertools
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TypedDict

import torch

_LOGITS_ATOL = 1e-2
_LOGITS_RTOL = 1e-2


class PrecisionRun(TypedDict):
    """One full-vocabulary tensor, shaped [generated tokens, vocabulary], per request."""

    tokens: list[list[int]]
    logits: list[torch.Tensor]


@dataclass(frozen=True)
class PrecisionPolicy:
    """How closely DKV must reproduce the attention-DP (ADP) control of one model.

    A model whose ADP runs reproduce (``noisy=False``) must match bit for bit, or within the fixed
    tolerance when the control itself is only tolerance-reproducible.

    A model whose ADP runs differ from each other (``noisy=True``) is judged against its own
    run-to-run noise. Tokens must agree wherever the top-two logit margin is at least ``min_margin``,
    because a flip is expected only where that gap is inside the noise. A continuation of several
    tokens is compared up to the first position where the three runs choose different tokens, since
    later positions no longer share a prefix. The total-variation (TV) distance of the softmax
    distributions between DKV and ADP may exceed the ADP run-to-run distance by at most
    ``max_mean_extra_tv`` on average and ``max_prompt_extra_tv`` for any single prompt.
    """

    name: str
    noisy: bool = False
    atol: float = _LOGITS_ATOL
    rtol: float = _LOGITS_RTOL
    min_margin: float = 0.5
    min_decisive_fraction: float = 0.5
    max_adp_self_tv: float = 0.2
    max_mean_extra_tv: float = 0.02
    max_prompt_extra_tv: float = 0.25
    # Different prompts must be far apart compared with the noise, or the gate cannot tell a request
    # that read another request's KV from a correct one.
    min_prompt_separation: float = 0.5
    min_separation_over_noise: float = 5.0


EXACT_POLICY = PrecisionPolicy("exact")
NOISY_POLICY = PrecisionPolicy("noisy", noisy=True)


def validate_precision_inputs(
    *,
    enable_block_reuse: bool,
    tokens_per_block: int,
    enable_partial_reuse: bool | None = None,
    prompts: Sequence[Sequence[int]] | None = None,
) -> None:
    """Require explicit reuse-off, or prove that no submitted prompt can hit another's prefix.

    With reuse enabled, partial-block reuse must be off and the prompts must include the entire
    submission history of the executor, including warmup requests and repetitions, with no two
    prompts sharing a full leading block. Prompts shorter than a block cannot match without
    partial reuse, so they take no part in the uniqueness check.
    """
    if type(enable_block_reuse) is not bool:
        raise ValueError("Precision requires an explicit boolean enable_block_reuse")
    if type(tokens_per_block) is not int or tokens_per_block <= 0:
        raise ValueError("Precision requires a positive tokens_per_block")
    if not enable_block_reuse:
        return
    if enable_partial_reuse is not False:
        raise ValueError("Reuse-enabled precision requires enable_partial_reuse=False")
    if not prompts:
        raise ValueError("Reuse-enabled precision requires the complete prompt history")
    first_blocks: dict[tuple[int, ...], int] = {}
    for index, prompt in enumerate(prompts):
        if not prompt:
            raise ValueError("Precision prompts must not be empty")
        if len(prompt) < tokens_per_block:
            continue
        block = tuple(prompt[:tokens_per_block])
        if block in first_blocks:
            raise ValueError(
                f"DKV precision prompts {first_blocks[block]} and {index} share at least "
                f"{tokens_per_block} prefix tokens; disable block reuse"
            )
        first_blocks[block] = index


def _validate_run(run: PrecisionRun, label: str) -> None:
    assert len(run["tokens"]) == len(run["logits"]) > 0, (
        f"{label} requires one token sequence and logits tensor per request"
    )
    for index, (tokens, values) in enumerate(zip(run["tokens"], run["logits"], strict=True)):
        assert tokens, f"{label} request {index} has no generated tokens"
        assert isinstance(values, torch.Tensor), f"{label} request {index} lacks logits"
        assert values.ndim == 2 and values.shape[0] == len(tokens) and values.shape[1] > 0, (
            f"{label} request {index} logits must have shape [generated tokens, vocabulary]"
        )
        assert values.is_floating_point() and torch.isfinite(values).all(), (
            f"{label} request {index} logits must be finite floating-point values"
        )


def _validate_matching_logits(first: PrecisionRun, second: PrecisionRun, label: str) -> None:
    assert len(first["logits"]) == len(second["logits"]), f"{label} request counts differ"
    for index, (left, right) in enumerate(zip(first["logits"], second["logits"], strict=True)):
        assert left.shape == right.shape, f"{label} request {index} logits shapes differ"
        assert left.dtype == right.dtype, f"{label} request {index} logits dtypes differ"


def validate_adp_control(
    control: PrecisionRun, replay: PrecisionRun, policy: PrecisionPolicy = EXACT_POLICY
) -> bool:
    """Reject unstable ADP tokens or logits beyond the policy's tolerance before judging DKV.

    Noisy policies only check the run shapes here; their tolerance is applied by the paired
    comparison. Returns whether the two ADP runs are bitwise identical.
    """
    _validate_run(control, "ADP control")
    _validate_run(replay, "ADP replay")
    _validate_matching_logits(control, replay, "ADP control/replay")
    bitwise = all(
        torch.equal(first, second)
        for first, second in zip(control["logits"], replay["logits"], strict=True)
    )
    if policy.noisy:
        return bitwise
    assert control["tokens"] == replay["tokens"], "ADP control tokens are not reproducible"
    if not bitwise:
        for first, second in zip(control["logits"], replay["logits"], strict=True):
            torch.testing.assert_close(
                first,
                second,
                atol=policy.atol,
                rtol=policy.rtol,
                msg="ADP control logits exceed the fixed precision tolerance",
            )
    return bitwise


def softmax_distance(first: torch.Tensor, second: torch.Tensor) -> float:
    """Total-variation distance of two [positions, vocabulary] logit tensors, per-position mean."""
    left = first.double().softmax(-1)
    right = second.double().softmax(-1)
    return (0.5 * (left - right).abs().sum(-1)).mean().item()


def top_two_margins(logits: torch.Tensor) -> torch.Tensor:
    """Gap between the largest two logits at every generated position."""
    top = logits.topk(2, dim=-1).values
    return top[:, 0] - top[:, 1]


def _comparable_length(*token_runs: Sequence[int]) -> int:
    """Leading positions whose logits the runs computed from the same prefix.

    Once two runs choose different tokens, their later positions answer different questions, so
    only the positions up to and including the first disagreement can be compared.
    """
    for position in range(len(token_runs[0])):
        if len({tokens[position] for tokens in token_runs}) > 1:
            return position + 1
    return len(token_runs[0])


def assess_prompt_separation(run: PrecisionRun, noise: float, policy: PrecisionPolicy) -> dict:
    """Require different prompts to be far apart relative to the policy's noise.

    A gate whose prompts produce similar distributions cannot distinguish a request that read the
    wrong KV from a correct one. The 10th percentile pair distance is used so one accidental
    near-duplicate does not hide an otherwise discriminating set. ``noise`` is the mean ADP
    run-to-run distance reported by ``compare_precision_runs``.
    """
    distances = sorted(
        softmax_distance(first, second)
        for first, second in itertools.combinations(run["logits"], 2)
    )
    assert distances, "Separation needs at least two prompts"
    tenth_percentile = distances[len(distances) // 10]
    required = max(policy.min_prompt_separation, policy.min_separation_over_noise * noise)
    assert tenth_percentile >= required, (
        f"Prompts are not separable: the 10th percentile pair distance {tenth_percentile:.4f} is "
        f"below {required:.4f}"
    )
    return {"p10_prompt_separation": tenth_percentile, "required_separation": required}


def _paired_tv_summary(control: PrecisionRun, replay: PrecisionRun, dkv: PrecisionRun) -> dict:
    self_distances = []
    extra_distances = []
    for index in range(len(control["logits"])):
        length = _comparable_length(
            control["tokens"][index], replay["tokens"][index], dkv["tokens"][index]
        )
        first, second, actual = (run["logits"][index][:length] for run in (control, replay, dkv))
        self_distance = softmax_distance(first, second)
        cross = (softmax_distance(first, actual) + softmax_distance(second, actual)) / 2
        self_distances.append(self_distance)
        extra_distances.append(cross - self_distance)
    count = len(self_distances)
    return {
        "mean_adp_self_tv": sum(self_distances) / count,
        "mean_extra_tv": sum(extra_distances) / count,
        "max_extra_tv": max(extra_distances),
        "extra_tv_by_prompt": extra_distances,
    }


def _compared_request(runs: Sequence[PrecisionRun], index: int) -> tuple[int, torch.Tensor, bool]:
    """How many positions of one request the runs can be compared on, and what they saw there.

    Returns the compared length, the smallest top-two gap any run saw at each compared position,
    and whether the runs ended the compared positions on different tokens.
    """
    tokens = [run["tokens"][index] for run in runs]
    length = _comparable_length(*tokens)
    margins = (
        torch.stack([top_two_margins(run["logits"][index][:length]) for run in runs])
        .min(dim=0)
        .values
    )
    flipped = len({sequence[length - 1] for sequence in tokens}) > 1
    return length, margins, flipped


def _compare_noisy(
    control: PrecisionRun, replay: PrecisionRun, dkv: PrecisionRun, policy: PrecisionPolicy
) -> dict:
    summary = _paired_tv_summary(control, replay, dkv)
    assert summary["mean_adp_self_tv"] <= policy.max_adp_self_tv, (
        f"ADP control is noisier than the policy allows: mean TV {summary['mean_adp_self_tv']:.4f}"
    )
    runs = (control, replay, dkv)
    compared_positions = 0
    decisive_positions = 0
    decisive_prompts = 0
    for index in range(len(control["logits"])):
        length, margins, flipped = _compared_request(runs, index)
        assert not flipped or margins[-1] < policy.min_margin, (
            f"Prompt {index} has a decisive top-two margin at position {length - 1} but generated "
            "different tokens"
        )
        decided = int((margins >= policy.min_margin).sum())
        compared_positions += length
        decisive_positions += decided
        decisive_prompts += decided == length
    assert decisive_positions >= policy.min_decisive_fraction * compared_positions, (
        f"Only {decisive_positions} of {compared_positions} compared positions have a top-two "
        f"margin of at least {policy.min_margin}; the gate cannot compare tokens"
    )
    assert summary["mean_extra_tv"] <= policy.max_mean_extra_tv, (
        f"DKV differs from ADP by {summary['mean_extra_tv']:.4f} more than ADP differs from itself "
        f"(ADP differs from itself by {summary['mean_adp_self_tv']:.4f}; by prompt: "
        f"{[round(value, 4) for value in summary['extra_tv_by_prompt']]})"
    )
    assert summary["max_extra_tv"] <= policy.max_prompt_extra_tv, (
        f"One prompt differs from ADP by {summary['max_extra_tv']:.4f} more than ADP differs from "
        "itself"
    )
    return {
        "requests": len(control["tokens"]),
        "comparison": "noise_envelope",
        "compared_positions": compared_positions,
        "decisive_positions": decisive_positions,
        "decisive_prompts": decisive_prompts,
        **{key: value for key, value in summary.items() if key != "extra_tv_by_prompt"},
    }


def compare_precision_runs(
    control: PrecisionRun,
    replay: PrecisionRun,
    dkv: PrecisionRun,
    policy: PrecisionPolicy = EXACT_POLICY,
) -> dict:
    """Judge DKV against two ADP runs of the same prompts under the model's policy."""
    bitwise_control = validate_adp_control(control, replay, policy)
    _validate_run(dkv, "DKV")
    _validate_matching_logits(control, dkv, "ADP/DKV")
    if policy.noisy:
        return _compare_noisy(control, replay, dkv, policy)
    assert control["tokens"] == dkv["tokens"], "ADP and DKV generated different tokens"
    for first, second, actual in zip(
        control["logits"], replay["logits"], dkv["logits"], strict=True
    ):
        if bitwise_control:
            assert torch.equal(first, actual), "Reproducible ADP logits differ from DKV logits"
        else:
            torch.testing.assert_close(first, actual, atol=policy.atol, rtol=policy.rtol)
            torch.testing.assert_close(second, actual, atol=policy.atol, rtol=policy.rtol)
    return {
        "requests": len(control["tokens"]),
        "bitwise_adp_control": bitwise_control,
        "comparison": "bitwise" if bitwise_control else "token_and_logits_tolerance",
        "atol": 0 if bitwise_control else policy.atol,
        "rtol": 0 if bitwise_control else policy.rtol,
        "mean_adp_self_tv": _paired_tv_summary(control, replay, dkv)["mean_adp_self_tv"],
    }


def failing_requests(
    control: PrecisionRun,
    replay: PrecisionRun,
    dkv: PrecisionRun,
    policy: PrecisionPolicy = EXACT_POLICY,
) -> list[int]:
    """Indices of the requests whose DKV result breaks the policy on its own.

    ``compare_precision_runs`` judges the whole set and stops at the first violation. This names
    every request that violates a per-request rule, so a test that corrupts one request can show
    that the gate objects to that request and to no other. The rules that only hold for the set as
    a whole, such as the mean extra distance, stay with ``compare_precision_runs``.
    """
    bitwise_control = validate_adp_control(control, replay, policy)
    _validate_run(dkv, "DKV")
    _validate_matching_logits(control, dkv, "ADP/DKV")
    failing = []
    if policy.noisy:
        extra_tv = _paired_tv_summary(control, replay, dkv)["extra_tv_by_prompt"]
        for index in range(len(control["logits"])):
            _, margins, flipped = _compared_request((control, replay, dkv), index)
            decisive_flip = flipped and margins[-1] >= policy.min_margin
            if decisive_flip or extra_tv[index] > policy.max_prompt_extra_tv:
                failing.append(index)
        return failing
    for index in range(len(control["logits"])):
        first, second, actual = (run["logits"][index] for run in (control, replay, dkv))
        if bitwise_control:
            same_logits = torch.equal(first, actual)
        else:
            same_logits = all(
                torch.allclose(reference, actual, atol=policy.atol, rtol=policy.rtol)
                for reference in (first, second)
            )
        if control["tokens"][index] != dkv["tokens"][index] or not same_logits:
            failing.append(index)
    return failing


# Real English paragraphs and code of different lengths and styles. Distinct, natural prompts keep
# the logits of different requests far apart, which is what lets the gate notice a request that read
# the wrong KV.
_PARAGRAPHS = (
    "The history of computing is usually told as a sequence of machines, but it is better understood "
    "as a sequence of abstractions. Early programmers wired circuits by hand; later they wrote "
    "assembly, then compiled languages, then libraries that hid entire subsystems behind a single "
    "call. Each layer traded some control for a large gain in productivity, and each layer "
    "eventually became the thing that the next generation took for granted.",
    "In a typical distributed training job, the slowest worker determines the pace of everyone "
    "else. Gradients must be exchanged before the optimizer can step, so any imbalance in compute, "
    "memory bandwidth or network latency shows up directly as idle time on the faster devices. "
    "Engineers therefore spend as much effort balancing the work as they do optimizing individual "
    "kernels.",
    "To make a simple tomato sauce, warm olive oil in a wide pan and add thinly sliced garlic. "
    "Before the garlic browns, pour in crushed tomatoes and a pinch of salt. Let the sauce simmer "
    "gently for twenty minutes, stirring occasionally, then finish with torn basil leaves and a "
    "spoonful of the pasta water so that it clings to the noodles.",
    "Photosynthesis converts light energy into chemical energy stored in glucose. In the "
    "light-dependent reactions, chlorophyll absorbs photons and drives the transfer of electrons "
    "along a chain of proteins, producing ATP and NADPH. In the Calvin cycle, those molecules power "
    "the fixation of carbon dioxide into sugars that the plant uses for growth.",
    "def merge_sorted(left, right):\n    result = []\n    i = j = 0\n"
    "    while i < len(left) and j < len(right):\n        if left[i] <= right[j]:\n"
    "            result.append(left[i])\n            i += 1\n        else:\n"
    "            result.append(right[j])\n            j += 1\n    result.extend(left[i:])\n"
    "    result.extend(right[j:])\n    return result\n",
    "The treaty established a framework for cooperation between the signatory states, requiring "
    "each party to notify the others before undertaking any action that might affect shared "
    "waterways. Disputes were to be settled first by direct negotiation and, failing that, by an "
    "arbitration panel whose members were chosen in equal numbers by each side.",
    "Large language models predict the next token given all previous tokens. During generation the "
    "key and value vectors of earlier tokens are cached so that each new token only requires "
    "attention against stored state instead of recomputing the whole prefix. The size of that "
    "cache grows linearly with sequence length and quickly dominates memory use at long contexts.",
    "A lighthouse keeper once wrote in his journal that the sea is never the same twice. On calm "
    "mornings it lay flat and silver, and on stormy nights it climbed the rocks as though trying "
    "to reach the lamp. He learned to read the weather from the colour of the horizon and the "
    "sound of the gulls long before the barometer moved.",
    "The patient presented with a three-day history of fever, productive cough and shortness of "
    "breath. Examination revealed crackles at the right lung base, and a chest radiograph showed a "
    "lobar consolidation. Treatment with an appropriate antibiotic was started, and the patient "
    "was advised to rest, drink plenty of fluids and return if symptoms worsened.",
    "Walking through the old market at dawn, you can smell fresh bread, roasted coffee and damp "
    "stone all at once. Vendors unload crates of oranges, the fishmonger arranges the morning catch "
    "on a bed of ice, and somewhere a radio plays a song that everyone seems to know. By nine "
    "o'clock the narrow lanes are crowded and the quiet is gone.",
)

# Sentences whose continuation a capable model gets right, and the accepted first words.
KNOWN_ANSWERS = (
    ("The capital of France is", ("paris",)),
    ("Two plus two equals", ("four", "4")),
    ("The chemical symbol for gold is", ("au",)),
    ("The largest planet in our solar system is", ("jupiter",)),
)

# Token budget of each paragraph prompt; the mix covers sub-block and multi-block lengths.
_PARAGRAPH_LENGTHS = (40, 72, 100, 130, 192, 256, 320, 384, 56, 160)

# Ranks of the 14 prompts (ten paragraphs, then the four known answers): runs of consecutive
# requests on one rank alternate with single requests, so both ranks idle and work in turn.
PRECISION_PLACEMENT = (0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1)


def precision_placement(group_size: int) -> tuple[int, ...]:
    """Rank of each of the 14 prompts in a group of ``group_size`` ranks.

    Two ranks take the pattern above; a larger group takes the prompts round-robin, so every rank
    computes at least one of them.
    """
    if group_size == 2:
        return PRECISION_PLACEMENT
    return tuple(index % group_size for index in range(len(PRECISION_PLACEMENT)))


def build_precision_prompts(
    encode: Callable[[str, bool], list[int]], tokens_per_block: int
) -> tuple[list[list[int]], list[tuple[int, tuple[str, ...]]]]:
    """Tokenize the discriminating prompt set.

    ``encode(text, special)`` returns token ids and adds the leading special tokens only when
    ``special`` is true. Returns the 14 prompts and, for the known-answer prompts, ``(prompt
    index, accepted words)``. Every paragraph prompt starts with a unique tag, so no two prompts
    share a leading block.
    """
    prompts: list[list[int]] = []
    for index, length in enumerate(_PARAGRAPH_LENGTHS):
        ids = list(encode(f"Request {index:04d}. ", True))
        paragraph = index
        while len(ids) < length:
            ids.extend(encode(" " + _PARAGRAPHS[paragraph % len(_PARAGRAPHS)], False))
            paragraph += 1
        prompts.append(ids[:length])
    answers = []
    for sentence, accepted in KNOWN_ANSWERS:
        answers.append((len(prompts), accepted))
        prompts.append(list(encode(sentence, True)))
    validate_precision_inputs(
        enable_block_reuse=True,
        tokens_per_block=tokens_per_block,
        enable_partial_reuse=False,
        prompts=prompts,
    )
    assert len(prompts) == len(PRECISION_PLACEMENT)
    return prompts, answers


def build_burst_prompts(
    encode: Callable[[str, bool], list[int]], count: int, lengths: Sequence[int]
) -> list[list[int]]:
    """Distinct natural-text prompts of the given token lengths, cycling through ``lengths``.

    Every prompt starts with a unique tag, so no two prompts share a leading block.
    """
    prompts = []
    for index in range(count):
        length = lengths[index % len(lengths)]
        ids = list(encode(f"Burst {index:05d}. ", True))
        paragraph = index
        while len(ids) < length:
            ids.extend(encode(" " + _PARAGRAPHS[paragraph % len(_PARAGRAPHS)], False))
            paragraph += 1
        prompts.append(ids[:length])
    return prompts


def assert_known_answers(
    run: PrecisionRun,
    answers: Sequence[tuple[int, tuple[str, ...]]],
    decode: Callable[[list[int]], str],
    label: str,
) -> None:
    """Check that a run still answers the known sentences, so a degraded control is noticed."""
    for index, accepted in answers:
        word = decode(run["tokens"][index][:1]).strip().lower()
        assert word in accepted, (
            f"{label} answered {word!r} for prompt {index}; expected {accepted}"
        )
