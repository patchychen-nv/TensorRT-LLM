# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A DKV context server behind the disaggregated trtllm-serve front end answers completions."""

from pathlib import Path

import pytest
import torch

from tensorrt_llm.llmapi import SamplingParams

from ..accuracy.test_disaggregated_serving import launch_disaggregated_llm
from .dkv_models import TINY_LLAMA

# A DKV context worker logs this at startup, so finding it proves the context server ran DKV.
_WORKER_MARKER = "DKV duplicates the global KV workload"
_PROMPTS = (
    "The capital of France is",
    "Two plus two equals",
    "The chemical symbol for gold is",
    "The largest planet in our solar system is",
)


@pytest.mark.threadleak(enabled=False)
def test_dkv_context_serve_answers_completions() -> None:
    """Completions reach a DKV context server, continue on a generation server and come back.

    TinyLlama keeps this smoke test free of restricted weights; the model's outputs are not judged.
    """
    if torch.cuda.device_count() < 4:
        pytest.skip("DKV context TP2 + ordinary generation TP2 requires four GPUs")
    model_path = TINY_LLAMA.path()
    if not Path(model_path).is_dir():
        pytest.skip(f"The checkpoint is not available at {model_path}")
    kv_cache_config = {
        "enable_block_reuse": False,
        "enable_swa_scratch_reuse": False,
        "use_kv_cache_manager_v2": True,
        "block_reuse_config": {"policy": "per_request"},
    }
    transceiver_config = {"backend": "NIXL", "transceiver_runtime": "PYTHON"}
    ctx_server_config = {
        "tensor_parallel_size": 2,
        "enable_attention_dp": True,
        "disable_overlap_scheduler": True,
        "enable_chunked_prefill": False,
        "cuda_graph_config": None,
        "dkv_config": {},
        "kv_cache_config": kv_cache_config,
        "cache_transceiver_config": transceiver_config,
    }
    gen_server_config = {
        "tensor_parallel_size": 2,
        "disable_overlap_scheduler": True,
        "cuda_graph_config": None,
        "kv_cache_config": kv_cache_config,
        "cache_transceiver_config": transceiver_config,
    }
    server_config = {
        "hostname": "localhost",
        "backend": "pytorch",
        "context_servers": {"num_instances": 1},
        "generation_servers": {"num_instances": 1},
    }
    with launch_disaggregated_llm(
        server_config,
        ctx_server_config,
        gen_server_config,
        model_path,
        extra_env={"TRTLLM_DKV_DEBUG": "1", "TLLM_LOG_LEVEL": "WARNING"},
        request_timeout_s=120,
        request_max_retries=0,
        assert_worker_log_contains=_WORKER_MARKER,
    ) as llm:
        sampling = SamplingParams(max_tokens=8)
        futures = [llm.generate_async(prompt, sampling) for prompt in _PROMPTS]
        for prompt, future in zip(_PROMPTS, futures, strict=True):
            output = future.result()
            assert len(output.outputs) == 1
            assert output.outputs[0].text, f"No completion came back for {prompt!r}"
