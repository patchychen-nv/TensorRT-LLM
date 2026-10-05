# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The transfer precision gate fails when one request hands generation corrupted KV."""

from pathlib import Path

import pytest
import torch

from .dkv_models import MODEL_IDS, MODELS, DkvModel, PromptTokenizer
from .dkv_precision import EXACT_POLICY, build_precision_prompts
from .dkv_transfer_runner import run_transfer_mutations

# The 130-token paragraph: the only prompt of its length, and one that the second context rank
# computes and sends.
_TARGET = 3

_MUTATIONS = {
    # The request's KV is gone.
    "zero": {"kind": "zero"},
    # The request is handed another request's KV, laid out as its own.
    "foreign": {"kind": "foreign"},
}


@pytest.mark.threadleak(enabled=False)
@pytest.mark.parametrize("model", MODELS, ids=MODEL_IDS)
def test_dkv_transfer_gate_fails_when_a_request_hands_over_corrupted_kv(
    tmp_path: Path, model: DkvModel
) -> None:
    """A gate that cannot fail proves nothing, so corrupt one request's KV and expect an objection.

    The context worker's sender overwrites the KV of a single prompt before the transport reads it,
    while the other thirteen prompts travel untouched. The gate must reject the run, and the
    requests that break the policy on their own must be that prompt and no other.
    """
    if torch.cuda.device_count() < 4:
        pytest.skip("DKV context TP2 + ordinary generation TP2 requires four GPUs")
    if not Path(model.path()).is_dir():
        pytest.skip(f"{model.pytest_id} checkpoint is not available at {model.path()}")
    prompts, _ = build_precision_prompts(
        PromptTokenizer(model.path()).encode, model.tokens_per_block
    )
    judgments = run_transfer_mutations(
        model.path(),
        str(tmp_path / "mutation"),
        _MUTATIONS,
        prompts=prompts,
        target_index=_TARGET,
        moe_backend=model.moe_backend or "CUTLASS",
        disable_finalize_fusion=model.disable_finalize_fusion,
        tokens_per_block=model.tokens_per_block,
        policy=model.precision or EXACT_POLICY,
    )
    for name, judgment in judgments.items():
        records = judgment["records"]
        # Both generation ranks receive their share of the request's KV, and both shares are hit.
        assert {record["peer_rank"] for record in records} == {0, 1}, (
            f"The {name} corruption did not reach both generation ranks: {records}"
        )
        assert all(record["corrupted_bytes"] > 0 for record in records), records
        assert judgment["gate_error"] is not None, (
            f"The precision gate accepted a request whose KV was corrupted ({name})"
        )
        assert judgment["failing_requests"] == [_TARGET], (
            f"After the {name} corruption the failing requests were {judgment['failing_requests']}"
            f", not just prompt {_TARGET}: {judgment['gate_error']}"
        )
