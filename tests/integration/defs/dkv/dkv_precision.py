# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Numerical acceptance rules for the Option B DKV test suites."""

from collections.abc import Sequence
from typing import TypedDict

import torch

_LOGITS_ATOL = 1e-2
_LOGITS_RTOL = 1e-2


class PrecisionRun(TypedDict):
    """One full-vocabulary tensor, shaped [generated tokens, vocabulary], per request."""

    tokens: list[list[int]]
    logits: list[torch.Tensor]


def validate_precision_inputs(
    *,
    enable_block_reuse: bool,
    tokens_per_block: int,
    prompts: Sequence[Sequence[int]] | None = None,
) -> None:
    """Require explicit reuse-off or prove all submitted prompts lack a shared full block.

    With reuse enabled, prompts must include the entire submission history of the executor,
    including warmup requests and repetitions, rather than only the current batch.
    """
    if type(enable_block_reuse) is not bool:
        raise ValueError("Precision requires an explicit boolean enable_block_reuse")
    if type(tokens_per_block) is not int or tokens_per_block <= 0:
        raise ValueError("Precision requires a positive tokens_per_block")
    if not enable_block_reuse:
        return
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
                f"Option B precision prompts {first_blocks[block]} and {index} share at least "
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


def validate_adp_control(control: PrecisionRun, replay: PrecisionRun) -> bool:
    """Reject unstable ADP tokens or logits beyond fixed tolerance before evaluating DKV."""
    _validate_run(control, "ADP control")
    _validate_run(replay, "ADP replay")
    _validate_matching_logits(control, replay, "ADP control/replay")
    assert control["tokens"] == replay["tokens"], "ADP control tokens are not reproducible"
    bitwise = all(
        torch.equal(first, second)
        for first, second in zip(control["logits"], replay["logits"], strict=True)
    )
    if not bitwise:
        for first, second in zip(control["logits"], replay["logits"], strict=True):
            torch.testing.assert_close(
                first,
                second,
                atol=_LOGITS_ATOL,
                rtol=_LOGITS_RTOL,
                msg="ADP control logits exceed the fixed precision tolerance",
            )
    return bitwise


def compare_precision_runs(control: PrecisionRun, replay: PrecisionRun, dkv: PrecisionRun) -> dict:
    """Require bitwise DKV logits for an exact ADP control, otherwise a fixed tolerance."""
    bitwise_control = validate_adp_control(control, replay)
    _validate_run(dkv, "DKV")
    _validate_matching_logits(control, dkv, "ADP/DKV")
    assert control["tokens"] == dkv["tokens"], "ADP and DKV generated different tokens"
    for first, second, actual in zip(
        control["logits"], replay["logits"], dkv["logits"], strict=True
    ):
        if bitwise_control:
            assert torch.equal(first, actual), "Reproducible ADP logits differ from DKV logits"
        else:
            torch.testing.assert_close(first, actual, atol=_LOGITS_ATOL, rtol=_LOGITS_RTOL)
            torch.testing.assert_close(second, actual, atol=_LOGITS_ATOL, rtol=_LOGITS_RTOL)
    return {
        "requests": len(control["tokens"]),
        "bitwise_adp_control": bitwise_control,
        "comparison": "bitwise" if bitwise_control else "token_and_logits_tolerance",
        "atol": 0 if bitwise_control else _LOGITS_ATOL,
        "rtol": 0 if bitwise_control else _LOGITS_RTOL,
    }
