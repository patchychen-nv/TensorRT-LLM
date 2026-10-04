# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Two-rank aggregate lifetime and sequential ADP/DKV precision gates."""

import time
from collections import defaultdict
from pathlib import Path

import pytest
import torch

from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm.llmapi import KvCacheConfig
from tensorrt_llm.llmapi.llm_args import BlockReuseConfig, DkvConfig, MoeConfig
from tensorrt_llm.scheduling_params import SchedulingParams

from .dkv_precision import compare_precision_runs, validate_adp_control, validate_precision_inputs

_MODEL_NAMES = ["llama-models-v2/TinyLlama-1.1B-Chat-v1.0", "DeepSeek-V3-Lite/bf16"]


def _llm(
    model_path: str,
    *,
    dkv_enabled: bool,
    logits: bool,
    moe_config: MoeConfig | None = None,
    tokens_per_block: int = 32,
) -> LLM:
    kv_cache_config = KvCacheConfig(
        use_kv_cache_manager_v2=True,
        enable_block_reuse=False,
        enable_swa_scratch_reuse=False,
        block_reuse_config=BlockReuseConfig(policy="per_request"),
        free_gpu_memory_fraction=0.5,
        max_tokens=4096,
        host_cache_size=0,
        iteration_stats_interval=1,
        tokens_per_block=tokens_per_block,
    )
    validate_precision_inputs(
        enable_block_reuse=kv_cache_config.enable_block_reuse,
        tokens_per_block=kv_cache_config.tokens_per_block,
    )
    return LLM(
        model=model_path,
        tensor_parallel_size=2,
        moe_expert_parallel_size=2,
        enable_attention_dp=True,
        disable_overlap_scheduler=True,
        enable_chunked_prefill=False,
        enable_autotuner=False,
        cuda_graph_config=None,
        max_batch_size=2,
        max_num_tokens=512,
        max_seq_len=512,
        num_postprocess_workers=0,
        enable_iter_perf_stats=True,
        gather_generation_logits=logits,
        dkv_config=DkvConfig() if dkv_enabled else None,
        moe_config=moe_config or MoeConfig(),
        kv_cache_config=kv_cache_config,
    )


def _context_rows(stats: list[dict]) -> list[dict]:
    return [
        row
        for row in stats
        if row.get("inflightBatchingStats", {}).get("numContextRequests", 0) > 0
    ]


def _drain_stats(llm: LLM, stats: list[dict], expected_contexts: int) -> None:
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        stats.extend(llm.get_stats(timeout=0.2))
        contexts = sum(
            row["inflightBatchingStats"]["numContextRequests"] for row in _context_rows(stats)
        )
        assert contexts <= expected_contexts, (
            "Attention-DP stats counted a replicated request twice"
        )
        if contexts == expected_contexts:
            return
    pytest.fail(f"Iteration stats did not account for {expected_contexts} sequential requests")


def run_aggregate_soak(
    model_path: str,
    *,
    iterations: int = 1000,
    moe_config: MoeConfig | None = None,
    tokens_per_block: int = 32,
) -> dict:
    """Exercise one real prefill per iteration and verify pages return after every request."""
    if iterations < 1000:
        raise ValueError("The aggregate lifetime gate requires at least 1000 forward iterations")
    sampling = SamplingParams(max_tokens=1, temperature=0, ignore_eos=True)
    stats: list[dict] = []
    request_ids: set[int] = set()
    with _llm(
        model_path,
        dkv_enabled=True,
        logits=False,
        moe_config=moe_config,
        tokens_per_block=tokens_per_block,
    ) as llm:
        initial_stats = llm.get_stats(timeout=5)
        initial_free_pages = [
            row["kvCacheStats"]["freeNumBlocks"] for row in initial_stats if "kvCacheStats" in row
        ]
        assert initial_free_pages, "Missing resident-dummy KV baseline before first request"
        assert len(set(initial_free_pages)) == 1
        for index in range(iterations + 1):
            # Four consecutive requests on one rank exercise a resident idle-rank dummy.
            placement = SchedulingParams(
                attention_dp_rank=(index // 4) % 2, attention_dp_relax=False
            )
            output = llm.generate(
                [1] + [42 + index % 64] * 255,
                sampling_params=sampling,
                scheduling_params=placement,
                use_tqdm=False,
            )
            assert output.request_id not in request_ids
            request_ids.add(output.request_id)
            assert len(output.outputs) == 1
            assert len(output.outputs[0].token_ids) == 1
            if index % 32 == 0:
                stats.extend(llm.get_stats(timeout=0.2))
        _drain_stats(llm, stats, iterations + 1)

    context_rows = _context_rows(stats)
    by_iteration: dict[int, list[dict]] = defaultdict(list)
    for row in context_rows:
        by_iteration[row["iter"]].append(row)
    assert len(by_iteration) == iterations + 1
    assert all(len(rows) == 1 for rows in by_iteration.values())
    assert {row["attentionDpRank"] for row in context_rows} == {0, 1}
    assert all(row["inflightBatchingStats"]["numContextRequests"] == 1 for row in context_rows)
    assert all(row["inflightBatchingStats"]["numCtxTokens"] == 256 for row in context_rows)
    free_pages = [row["kvCacheStats"]["freeNumBlocks"] for row in context_rows]
    baseline = initial_free_pages[0]
    assert all(free == baseline for free in free_pages), (
        f"KV pages did not return to baseline {baseline}: {free_pages}"
    )
    return {
        "requests": len(request_ids),
        "forward_iterations": len(by_iteration),
        "baseline_free_blocks": baseline,
        "final_free_blocks": free_pages[-1],
        "stats": stats,
    }


def run_precision_sample(
    model_path: str,
    *,
    dkv_enabled: bool,
    prompts: list[list[int]],
    moe_config: MoeConfig | None = None,
    tokens_per_block: int = 32,
) -> dict:
    """Run pinned requests one at a time; each process can persist the returned CPU logits."""
    sampling = SamplingParams(
        max_tokens=1, temperature=0, ignore_eos=True, return_generation_logits=True
    )
    tokens: list[list[int]] = []
    logits: list[torch.Tensor] = []
    with _llm(
        model_path,
        dkv_enabled=dkv_enabled,
        logits=True,
        moe_config=moe_config,
        tokens_per_block=tokens_per_block,
    ) as llm:
        for index, prompt in enumerate(prompts):
            output = llm.generate(
                prompt,
                sampling_params=sampling,
                scheduling_params=SchedulingParams(
                    attention_dp_rank=index % 2, attention_dp_relax=False
                ),
                use_tqdm=False,
            )
            assert len(output.outputs) == 1
            result = output.outputs[0]
            assert len(result.token_ids) == 1
            assert isinstance(result.generation_logits, torch.Tensor), "Generation logits missing"
            values = result.generation_logits.detach().float().cpu().clone()
            assert values.numel() > 0, "Generation logits are empty"
            assert torch.isfinite(values).all()
            tokens.append(list(result.token_ids))
            logits.append(values)
    return {"tokens": tokens, "logits": logits}


@pytest.fixture
def dkv_gate_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    if torch.cuda.device_count() < 2:
        pytest.skip("DKV aggregate gates require two GPUs")
    monkeypatch.setenv("TRTLLM_DKV_DEBUG", "1")
    monkeypatch.setenv("TRTLLM_DKV_DUAL_LEDGER", "1")
    monkeypatch.delenv("TLLM_WORKER_USE_SINGLE_PROCESS", raising=False)


@pytest.mark.post_merge
@pytest.mark.threadleak(enabled=False)
@pytest.mark.parametrize("model_name", _MODEL_NAMES, ids=["tiny-llama", "deepseek-v3-lite"])
def test_dkv_aggregate_1000_steps(dkv_gate_environment, capfd, model_name: str) -> None:
    from ..conftest import llm_models_root

    result = run_aggregate_soak(str(Path(llm_models_root()) / model_name))
    assert result["forward_iterations"] >= 1000
    captured = capfd.readouterr()
    logs = captured.out + captured.err
    assert "DKV invariant violation" not in logs
    assert "No free slots" not in logs
    assert "exceeds expected_num_active_requests" not in logs


@pytest.mark.post_merge
@pytest.mark.threadleak(enabled=False)
@pytest.mark.parametrize("model_name", _MODEL_NAMES, ids=["tiny-llama", "deepseek-v3-lite"])
def test_dkv_sequential_precision(dkv_gate_environment, model_name: str) -> None:
    from ..conftest import llm_models_root

    model_path = str(Path(llm_models_root()) / model_name)
    prompts = [[1] + [42 + index] * (63 if index < 4 else 255) for index in range(8)]
    control = run_precision_sample(model_path, dkv_enabled=False, prompts=prompts)
    replay = run_precision_sample(model_path, dkv_enabled=False, prompts=prompts)
    validate_adp_control(control, replay)
    dkv = run_precision_sample(model_path, dkv_enabled=True, prompts=prompts)
    compare_precision_runs(control, replay, dkv)
