# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A host cache tier under DKV: blocks spill to host and onboard again, identically on every rank."""

from pathlib import Path

import pytest
import torch

from tensorrt_llm import SamplingParams
from tensorrt_llm.scheduling_params import SchedulingParams

from .dkv_models import MODEL_IDS, MODELS, DkvModel, make_dkv_llm
from .dkv_stats import latest_free_pages_by_rank

_REQUESTS = 96
# Tokens of the GPU pool per 32-token block size; the prompts need many times more than this.
_GPU_POOL_TOKENS = 2048
_TIER_FIELDS = {
    "offload": "iterOffloadBlocks",
    "onboard": "iterOnboardBlocks",
    "host_dropped": "iterHostDroppedBlocks",
}


def _tier_totals(rows: list[dict]) -> dict[str, int]:
    """Blocks moved between the tiers, summed over the iteration windows of rank 0's statistics."""
    totals = dict.fromkeys(_TIER_FIELDS, 0)
    for row in rows:
        for window in row.get("kvCacheIterationStats", {}).values():
            for name, field in _TIER_FIELDS.items():
                totals[name] += window.get(field, 0)
    return totals


@pytest.mark.threadleak(enabled=False)
@pytest.mark.parametrize("model", MODELS, ids=MODEL_IDS)
def test_dkv_host_tier_spills_and_onboards_replicated_prefixes(
    monkeypatch, model: DkvModel
) -> None:
    """Prefixes evicted from the GPU pool survive in the host tier and hit again on the other rank."""
    if torch.cuda.device_count() < 2:
        pytest.skip("The DKV host-tier test needs two GPUs")
    if not Path(model.path()).is_dir():
        pytest.skip(f"{model.pytest_id} checkpoint is not available at {model.path()}")
    monkeypatch.setenv("TRTLLM_DKV_DEBUG", "1")
    monkeypatch.setenv("TRTLLM_DKV_DUAL_LEDGER", "1")
    # Each rank's own pool snapshot rides on the iteration stats; kvCacheStats is rank 0's copy.
    monkeypatch.setenv("TRTLLM_DKV_MEASUREMENT", "1")
    monkeypatch.delenv("TLLM_WORKER_USE_SINGLE_PROCESS", raising=False)
    block = model.tokens_per_block
    prompts = [[1] + [100 + index] * (3 * block - 1) for index in range(_REQUESTS)]
    sampling = SamplingParams(max_tokens=1, temperature=0, ignore_eos=True)

    def pinned(rank: int) -> SchedulingParams:
        return SchedulingParams(attention_dp_rank=rank % 2, attention_dp_relax=False)

    rows: list[dict] = []
    with make_dkv_llm(
        model.path(),
        dkv=True,
        tokens_per_block=block,
        moe_config=model.moe_config(),
        reuse=True,
        max_num_tokens=4 * block,
        max_seq_len=4 * block,
        kv_cache={
            "max_tokens": _GPU_POOL_TOKENS * block // 32,
            "host_cache_size": model.host_cache_bytes,
            # Hit counts are the measurement here; no output is validated.
            "enable_partial_reuse": True,
        },
    ) as llm:
        for index, prompt in enumerate(prompts):
            output = llm.generate(
                prompt,
                sampling_params=sampling,
                scheduling_params=pinned(index),
                use_tqdm=False,
            )
            assert len(output.outputs[0].token_ids) == 1
            if index % 8 == 0:
                rows.extend(llm.get_stats(timeout=0.2))

        def replay(indices: range) -> list[int]:
            """Run earlier prompts again, each on the rank that did not compute it."""
            cached = []
            for index in indices:
                output = llm.generate(
                    prompts[index],
                    sampling_params=sampling,
                    scheduling_params=pinned(index + 1),
                    use_tqdm=False,
                )
                assert len(output.outputs[0].token_ids) == 1
                cached.append(output.cached_tokens)
            return cached

        recent = replay(range(_REQUESTS - 4, _REQUESTS))
        evicted = replay(range(8))
        rows.extend(llm.get_stats(timeout=5))

    totals = _tier_totals(rows)
    assert totals["offload"] > 0, f"Nothing spilled to the host tier: {totals}"
    assert totals["onboard"] > 0, f"Nothing came back from the host tier: {totals}"
    assert totals["host_dropped"] == 0, f"The host tier was too small for the prompts: {totals}"
    # The replica of the other rank holds the metadata, whether the block is on the GPU or on the host.
    assert all(cached > 0 for cached in recent), recent
    assert all(cached > 0 for cached in evicted), evicted
    free = latest_free_pages_by_rank(rows)
    assert set(free) == {0, 1} and len(set(free.values())) == 1, (
        f"The replicas' pools of all tiers differ: {free}"
    )
