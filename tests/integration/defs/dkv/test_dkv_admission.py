# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise replicated admission with real attention-DP workers."""

from pathlib import Path

import pytest
import torch

from tensorrt_llm import SamplingParams
from tensorrt_llm.scheduling_params import SchedulingParams

from .dkv_models import MODEL_IDS, MODELS, DkvModel, make_dkv_llm

# A DKV worker logs this at startup, so seeing it proves the worker's warnings reach the capture
# and the absence of the violation messages below is evidence.
_WORKER_CANARY = "DKV duplicates the global KV workload"


@pytest.mark.threadleak(enabled=False)
@pytest.mark.parametrize(
    "dkv_enabled,enable_block_reuse",
    [(False, False), (True, False), (True, True)],
    ids=["adp", "dkv", "dkv-reuse"],
)
@pytest.mark.parametrize("model", MODELS, ids=MODEL_IDS)
def test_dkv_replicated_admission(
    monkeypatch, capfd, dkv_enabled: bool, enable_block_reuse: bool, model: DkvModel
) -> None:
    """Replicated state stays aligned through admission, retirement, and warm-prefix reuse.

    The dual ledger is off here so the single-ledger fallback keeps GPU coverage; the lifetime gate
    runs with it on.
    """
    if torch.cuda.device_count() < 2:
        pytest.skip("DKV admission needs two GPUs")
    if not Path(model.path()).is_dir():
        pytest.skip(f"{model.pytest_id} checkpoint is not available at {model.path()}")
    monkeypatch.setenv("TRTLLM_DKV_DEBUG", "1")
    monkeypatch.setenv("TRTLLM_DKV_DUAL_LEDGER", "0")
    monkeypatch.setenv("TLLM_LOG_LEVEL", "WARNING")
    monkeypatch.delenv("TLLM_WORKER_USE_SINGLE_PROCESS", raising=False)
    block = model.tokens_per_block
    sampling = SamplingParams(max_tokens=1, temperature=0, ignore_eos=True)
    placements = [
        SchedulingParams(attention_dp_rank=rank, attention_dp_relax=False) for rank in (0, 1, 0, 1)
    ]
    with make_dkv_llm(
        model.path(),
        dkv=dkv_enabled,
        tokens_per_block=block,
        moe_config=model.moe_config(),
        reuse=enable_block_reuse,
        # The reuse variant checks hit counts, not outputs, so partial-block matches stay enabled.
        kv_cache={"enable_partial_reuse": True} if enable_block_reuse else None,
        enable_iter_perf_stats=False,
    ) as llm:
        for iteration in range(8):
            # With reuse the prompts repeat and cover three blocks, so two full blocks can match
            # whatever the model's cache layout asks of a reusable prefix; without it every prompt
            # is new, so nothing can match.
            prompts = (
                [[1] + [42 + index] * (3 * block - 1) for index in range(4)]
                if enable_block_reuse
                else [[1] + [42 + iteration * 4 + index] * 63 for index in range(4)]
            )
            outputs = llm.generate(
                prompts,
                sampling_params=sampling,
                scheduling_params=placements,
                use_tqdm=False,
            )
            assert len(outputs) == len(prompts)
            for output in outputs:
                assert len(output.outputs) == 1
                assert len(output.outputs[0].token_ids) == 1
                if enable_block_reuse and iteration > 0:
                    assert output.cached_tokens > 0

    captured = capfd.readouterr()
    logs = captured.out + captured.err
    assert (_WORKER_CANARY in logs) == dkv_enabled
    assert "DKV invariant violation" not in logs
    assert "exceeds expected_num_active_requests" not in logs
