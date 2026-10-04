# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise replicated admission with real attention-DP workers."""

import pytest
import torch

from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm.llmapi import KvCacheConfig
from tensorrt_llm.llmapi.llm_args import BlockReuseConfig, DkvConfig
from tensorrt_llm.scheduling_params import SchedulingParams

from ..conftest import llm_models_root


@pytest.mark.post_merge
@pytest.mark.threadleak(enabled=False)
@pytest.mark.parametrize(
    "dkv_enabled,enable_block_reuse",
    [(False, False), (True, False), (True, True)],
    ids=["adp", "dkv", "dkv-reuse"],
)
@pytest.mark.parametrize(
    "model_name",
    ["llama-models-v2/TinyLlama-1.1B-Chat-v1.0", "DeepSeek-V3-Lite/bf16"],
    ids=["tiny-llama", "deepseek-v3-lite"],
)
def test_dkv_replicated_admission(
    monkeypatch, capfd, dkv_enabled: bool, enable_block_reuse: bool, model_name: str
) -> None:
    """C0-C3 remain aligned through admission, retirement, and warm-prefix reuse."""
    if torch.cuda.device_count() < 2:
        pytest.skip("DKV admission needs two GPUs")
    monkeypatch.setenv("TRTLLM_DKV_DEBUG", "1")
    monkeypatch.setenv("TRTLLM_DKV_DUAL_LEDGER", "0")
    monkeypatch.delenv("TLLM_WORKER_USE_SINGLE_PROCESS", raising=False)
    kv_config = KvCacheConfig(
        use_kv_cache_manager_v2=True,
        enable_block_reuse=enable_block_reuse,
        block_reuse_config=BlockReuseConfig(policy="per_request"),
        free_gpu_memory_fraction=0.5,
        max_tokens=4096,
    )
    sampling = SamplingParams(max_tokens=1, temperature=0, ignore_eos=True)
    placements = [
        SchedulingParams(attention_dp_rank=rank, attention_dp_relax=False) for rank in (0, 1, 0, 1)
    ]
    with LLM(
        model=f"{llm_models_root()}/{model_name}",
        tensor_parallel_size=2,
        moe_expert_parallel_size=2,
        enable_attention_dp=True,
        disable_overlap_scheduler=True,
        enable_chunked_prefill=False,
        enable_autotuner=False,
        cuda_graph_config=None,
        max_batch_size=2,
        max_num_tokens=256,
        max_seq_len=256,
        num_postprocess_workers=0,
        dkv_config=DkvConfig() if dkv_enabled else None,
        kv_cache_config=kv_config,
    ) as llm:
        for iteration in range(8):
            prompts = (
                [[1] + [42 + index] * 127 for index in range(4)]
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

    if dkv_enabled:
        captured = capfd.readouterr()
        logs = captured.out + captured.err
        assert "DKV invariant violation" not in logs
        assert "exceeds expected_num_active_requests" not in logs
