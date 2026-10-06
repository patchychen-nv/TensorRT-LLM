# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Four-GPU context-first NIXL transfers with replicated DKV context state."""

from pathlib import Path

import pytest
import torch

from .dkv_models import MODEL_IDS, MODELS, DkvModel, PromptTokenizer
from .dkv_precision import (
    EXACT_POLICY,
    KNOWN_ANSWERS,
    PrecisionPolicy,
    assert_known_answers,
    assess_prompt_separation,
    build_precision_prompts,
    build_prefix_pairs,
)
from .dkv_transfer_runner import run_transfer_gate

# The pairs of prompts of the reuse gate: only the second prompt of each is compared with the
# controls, and the mean of the extra distance over so few requests has a spread that has to stay
# well inside its threshold.
_REUSE_PAIRS = 12


def run_context_transfer_gate(
    tmp_path: Path, model: DkvModel, policy: PrecisionPolicy | None = None
) -> None:
    """Hand the precision prompts from a DKV context worker to ordinary generation, and judge it.

    The context workers are built with the layout ``DKV_TEST_KV_LAYOUT`` names. The run is judged
    by ``policy``, which defaults to that of the model.
    """
    if torch.cuda.device_count() < 4:
        pytest.skip("DKV context TP2 + ordinary generation TP2 requires four GPUs")
    if not Path(model.path()).is_dir():
        pytest.skip(f"{model.pytest_id} checkpoint is not available at {model.path()}")
    tokenizer = PromptTokenizer(model.path())
    prompts, answers = build_precision_prompts(tokenizer.encode, model.tokens_per_block)
    policy = policy or model.precision or EXACT_POLICY
    work_dir = tmp_path / "transfer"
    result = run_transfer_gate(
        model.path(),
        str(work_dir),
        moe_backend=model.moe_backend or "CUTLASS",
        disable_finalize_fusion=model.disable_finalize_fusion,
        tokens_per_block=model.tokens_per_block,
        transfer_timeout_ms=30000,
        policy=policy,
        prompts=prompts,
    )
    assert result["lifecycle"]["status"] == "passed"
    assert result["precision"]["status"] == "passed"

    if model.precision is None:
        # Its outputs only have to match the control; they are no oracle for the known answers.
        return
    control = torch.load(work_dir / "adp-control" / "outputs.pt", weights_only=True)
    dkv = torch.load(work_dir / "dkv" / "outputs.pt", weights_only=True)
    assess_prompt_separation(control, result["precision"]["mean_adp_self_tv"], policy)
    assert_known_answers(control, answers, tokenizer.decode, "ADP control")
    assert_known_answers(dkv, answers, tokenizer.decode, "DKV")


def run_context_reuse_gate(
    tmp_path: Path, model: DkvModel, policy: PrecisionPolicy | None = None
) -> None:
    """Hand prompts whose prefix the context worker finds in its cache to ordinary generation.

    The prompts come in pairs, and the two of a pair run on different ranks of the context group.
    The first is a text of three blocks, and the second is the first followed by a sentence with a
    known answer, so it finds the whole first prompt in the cache, which another rank computed, and
    hands generation that cached KV together with its own. Only the second prompts are compared
    with the ADP controls, since the continuation of a text that ends in the middle of a sentence is
    no answer that the model is sure of. The context worker is built with the layout
    ``DKV_TEST_KV_LAYOUT`` names; the ADP controls do not reuse. The run is judged by ``policy``,
    which defaults to that of the model.
    """
    if torch.cuda.device_count() < 4:
        pytest.skip("DKV context TP2 + ordinary generation TP2 requires four GPUs")
    if not Path(model.path()).is_dir():
        pytest.skip(f"{model.pytest_id} checkpoint is not available at {model.path()}")
    block = model.tokens_per_block
    tokenizer = PromptTokenizer(model.path())
    prompts = build_prefix_pairs(
        tokenizer.encode,
        _REUSE_PAIRS,
        3 * block,
        tails=[sentence for sentence, _ in KNOWN_ANSWERS],
    )
    result = run_transfer_gate(
        model.path(),
        str(tmp_path / "reuse"),
        moe_backend=model.moe_backend or "CUTLASS",
        disable_finalize_fusion=model.disable_finalize_fusion,
        tokens_per_block=block,
        transfer_timeout_ms=30000,
        policy=policy or model.precision or EXACT_POLICY,
        prompts=prompts,
        reuse_hit_tokens=3 * block,
    )
    assert result["lifecycle"]["status"] == "passed"
    assert result["precision"]["status"] == "passed"
    print(f"context transfer with prefix reuse: {result}")
    if model.precision is None:
        return
    answers = [
        (2 * index + 1, KNOWN_ANSWERS[index % len(KNOWN_ANSWERS)][1])
        for index in range(_REUSE_PAIRS)
    ]
    for name in ("adp-control", "dkv"):
        run = torch.load(tmp_path / "reuse" / name / "outputs.pt", weights_only=True)
        assert_known_answers(run, answers, tokenizer.decode, name)


@pytest.mark.threadleak(enabled=False)
@pytest.mark.parametrize("model", MODELS, ids=MODEL_IDS)
def test_dkv_context_transfer_precision_and_timeouts(tmp_path: Path, model: DkvModel) -> None:
    """A DKV context worker hands the generation worker the same KV an ADP context worker would.

    The prompts are the real texts of the precision gate, so a transfer that delivered another
    request's KV would change the eight generated tokens. Timeouts and a cancellation race run
    between the first requests and the last two.
    """
    run_context_transfer_gate(tmp_path, model)
