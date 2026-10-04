# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Four-GPU context-first NIXL transfers with replicated DKV context state."""

from pathlib import Path

import pytest
import torch

from .dkv_models import MODEL_IDS, MODELS, DkvModel, PromptTokenizer
from .dkv_precision import (
    EXACT_POLICY,
    assert_known_answers,
    assess_prompt_separation,
    build_precision_prompts,
)
from .dkv_transfer_runner import run_transfer_gate


@pytest.mark.threadleak(enabled=False)
@pytest.mark.parametrize("model", MODELS, ids=MODEL_IDS)
def test_dkv_context_transfer_precision_and_timeouts(tmp_path: Path, model: DkvModel) -> None:
    """A DKV context worker hands the generation worker the same KV an ADP context worker would.

    The prompts are the real texts of the precision gate, so a transfer that delivered another
    request's KV would change the eight generated tokens. Timeouts and a cancellation race run
    between the first requests and the last two.
    """
    if torch.cuda.device_count() < 4:
        pytest.skip("DKV context TP2 + ordinary generation TP2 requires four GPUs")
    if not Path(model.path()).is_dir():
        pytest.skip(f"{model.pytest_id} checkpoint is not available at {model.path()}")
    tokenizer = PromptTokenizer(model.path())
    prompts, answers = build_precision_prompts(tokenizer.encode, model.tokens_per_block)
    policy = model.precision or EXACT_POLICY
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
