# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Models, tokenizers and the shared LLM factory of the DKV integration tests."""

import os
from dataclasses import dataclass
from pathlib import Path

try:
    from .dkv_precision import EXACT_POLICY, NOISY_POLICY, PrecisionPolicy
except ImportError:  # Imported as a plain module by the subprocess runners.
    from dkv_precision import EXACT_POLICY, NOISY_POLICY, PrecisionPolicy


@dataclass(frozen=True)
class DkvModel:
    """One model the DKV suites run on, and what each suite may assert about its outputs.

    ``precision`` is None for a model that is only used for lifetime and functional checks: its
    outputs are not a trustworthy oracle for the numerical gate.
    """

    pytest_id: str
    relative_path: str
    tokens_per_block: int
    moe_backend: str | None = None
    disable_finalize_fusion: bool = False
    precision: PrecisionPolicy | None = None
    path_env: str | None = None
    # Host cache bytes per rank that hold every block the host-tier test spills: 96 prompts of three
    # blocks, several times over.
    host_cache_bytes: int = 2 << 30
    # Whether the pressure bursts are known to make the scheduler release and restart a context.
    # They shrink the pool to 2100 tokens: an uncompressed KV cache is exhausted by that, but
    # DeepSeek-V4's compressed one is not, and its pool never drops below a fixed minimum.
    stalls_under_pressure: bool = False

    def path(self) -> str:
        """Resolve the checkpoint: an explicit environment override, else the models root."""
        override = os.environ.get(self.path_env) if self.path_env else None
        if override:
            return override
        from ..conftest import llm_models_root

        return str(Path(llm_models_root()) / self.relative_path)

    def moe_config(self):
        from tensorrt_llm.llmapi.llm_args import MoeConfig

        if self.moe_backend is None:
            return MoeConfig()
        return MoeConfig(
            backend=self.moe_backend, disable_finalize_fusion=self.disable_finalize_fusion
        )


# Llama does not run its MLP as attention-DP expects, so TinyLlama stays out of the numerical gate.
TINY_LLAMA = DkvModel(
    "tiny-llama", "llama-models-v2/TinyLlama-1.1B-Chat-v1.0", 32, stalls_under_pressure=True
)
# The CUTLASS MoE backend with the finalize fusion off is the deterministic configuration.
DEEPSEEK_V3_LITE = DkvModel(
    "deepseek-v3-lite",
    "DeepSeek-V3-Lite/bf16",
    32,
    moe_backend="CUTLASS",
    disable_finalize_fusion=True,
    precision=EXACT_POLICY,
)
# V4 needs the TRTLLM MoE backend and 128-token blocks; its attention-DP runs do not reproduce.
DEEPSEEK_V4 = DkvModel(
    "deepseek-v4",
    "DeepSeek-V4-Flash",
    128,
    moe_backend="TRTLLM",
    precision=NOISY_POLICY,
    path_env="DKV_DEEPSEEK_V4_PATH",
    # Its blocks span several cache groups; 2 GiB dropped part of the spilled prefixes.
    host_cache_bytes=64 << 30,
)
MODELS = (TINY_LLAMA, DEEPSEEK_V3_LITE, DEEPSEEK_V4)
MODEL_IDS = [model.pytest_id for model in MODELS]
PRECISION_MODELS = [model for model in MODELS if model.precision is not None]


def dkv_worker_env(*, measurement: bool = False) -> dict[str, str]:
    """The DKV switches a test turns on, as ``env_overrides`` for the LLM.

    Ranks that were launched through MPI do not see the environment a test sets, so the LLM carries
    these to every worker.
    """
    env = {"TRTLLM_DKV_DEBUG": "1", "TRTLLM_DKV_DUAL_LEDGER": "1"}
    if measurement:
        # Each rank's own pool snapshot rides on the iteration stats.
        env["TRTLLM_DKV_MEASUREMENT"] = "1"
    return env


def available_gpus() -> int:
    """GPUs a test may use: this node's, or the whole job's when ranks were launched with MPI."""
    import torch

    from tensorrt_llm.llmapi.mpi_session import get_mpi_world_size

    world_size = get_mpi_world_size()
    return world_size if world_size > 1 else torch.cuda.device_count()


class PromptTokenizer:
    """Tokenizer access that also works for checkpoints whose config transformers cannot build."""

    def __init__(self, model_path: str) -> None:
        self._hf = None
        self._fast = None
        try:
            from transformers import AutoTokenizer

            self._hf = AutoTokenizer.from_pretrained(model_path)
        # A model type transformers does not know fails differently across its versions: with a
        # KeyError or ValueError for the lookup, or an AttributeError when the generic config
        # standardizes the RoPE parameters.
        except (ValueError, KeyError, OSError, AttributeError):
            from tokenizers import Tokenizer

            self._fast = Tokenizer.from_file(str(Path(model_path) / "tokenizer.json"))

    def encode(self, text: str, special: bool = False) -> list[int]:
        """Token ids of ``text``; ``special`` adds the model's leading special tokens."""
        if self._hf is not None:
            return list(self._hf.encode(text, add_special_tokens=special))
        return list(self._fast.encode(text, add_special_tokens=special).ids)

    def decode(self, ids: list[int]) -> str:
        if self._hf is not None:
            return self._hf.decode(ids)
        return self._fast.decode(ids)


def make_dkv_llm(
    model: str,
    *,
    dkv: bool,
    tokens_per_block: int,
    moe_config=None,
    group_size: int = 2,
    attention_dp: bool = True,
    reuse: bool = False,
    gather_logits: bool = False,
    chunked_prefill: bool = False,
    cuda_graph: bool = False,
    autotuner: bool = False,
    kv_cache: dict | None = None,
    **llm_kwargs,
):
    """Build the LLM every DKV suite runs on, so a flag is set in one place.

    The defaults are the replicated-prefill configuration: V2 KV manager, per-request reuse policy,
    non-overlapped scheduling, no chunked prefill, no CUDA graphs, no autotuner. ``kv_cache``
    overrides ``KvCacheConfig`` fields and ``llm_kwargs`` override the LLM arguments.
    """
    from tensorrt_llm import LLM
    from tensorrt_llm.llmapi import KvCacheConfig
    from tensorrt_llm.llmapi.llm_args import BlockReuseConfig, DkvConfig, MoeConfig

    kv_options = dict(
        use_kv_cache_manager_v2=True,
        enable_block_reuse=reuse,
        enable_partial_reuse=False,
        enable_swa_scratch_reuse=False,
        block_reuse_config=BlockReuseConfig(policy="per_request"),
        free_gpu_memory_fraction=0.5,
        max_tokens=4096,
        host_cache_size=0,
        iteration_stats_interval=1,
        tokens_per_block=tokens_per_block,
    )
    kv_options.update(kv_cache or {})
    options = dict(
        model=model,
        tensor_parallel_size=group_size,
        moe_expert_parallel_size=group_size,
        enable_attention_dp=attention_dp,
        disable_overlap_scheduler=True,
        enable_chunked_prefill=chunked_prefill,
        enable_autotuner=autotuner,
        max_batch_size=2,
        max_num_tokens=512,
        max_seq_len=512,
        num_postprocess_workers=0,
        enable_iter_perf_stats=True,
        gather_generation_logits=gather_logits,
        dkv_config=DkvConfig() if dkv else None,
        moe_config=moe_config or MoeConfig(),
        kv_cache_config=KvCacheConfig(**kv_options),
    )
    if not cuda_graph:
        options["cuda_graph_config"] = None
    options.update(llm_kwargs)
    return LLM(**options)
