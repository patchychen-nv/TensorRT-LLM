# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reuse workloads against ADP and DKV: replicated metadata makes every repeated prefix a hit."""

from pathlib import Path

import pytest
import torch

from tensorrt_llm import SamplingParams
from tensorrt_llm.scheduling_params import SchedulingParams

from .dkv_models import TINY_LLAMA, DkvModel, make_dkv_llm
from .dkv_workloads import multi_turn_workload, run_workload, zipf_prefix_workload

_VOCAB = 30000
_ZIPF_REQUESTS = 24
_PREFIXES = 3


def _run(model: DkvModel, *, dkv: bool, alternate_ranks: bool, concurrency: int = 1) -> dict:
    """Serve both workloads and count the prompt tokens answered from cache.

    The router places a request itself unless ``alternate_ranks`` pins it to rank
    ``(index + turn) % 2``: consecutive requests alternate, and so do the turns of a conversation,
    so a repeated prefix lands on a rank that mostly did not compute it.
    """
    block = model.tokens_per_block
    zipf = zipf_prefix_workload(
        _ZIPF_REQUESTS,
        prefixes=_PREFIXES,
        alpha=1.2,
        prefix_tokens=2 * block,
        suffix_tokens=8,
        vocab=_VOCAB,
    )
    conversations = multi_turn_workload(
        sessions=2, turns=3, first_tokens=2 * block, turn_tokens=16, vocab=_VOCAB, seed=5
    )
    sampling = SamplingParams(max_tokens=1, temperature=0, ignore_eos=True)
    with make_dkv_llm(
        model.path(),
        dkv=dkv,
        tokens_per_block=block,
        moe_config=model.moe_config(),
        reuse=True,
        # Hit counts are the measurement here; no output is validated.
        kv_cache={"enable_partial_reuse": True},
    ) as llm:

        def submit(request):
            placement = (
                SchedulingParams(
                    attention_dp_rank=(request.index + request.turn) % 2, attention_dp_relax=False
                )
                if alternate_ranks
                else None
            )
            future = llm.generate_async(
                request.prompt, sampling_params=sampling, scheduling_params=placement
            )
            return lambda: future.result(timeout=300).cached_tokens

        zipf_results = run_workload(submit, zipf, concurrency=concurrency)
        conversation_results = run_workload(submit, conversations, concurrency=concurrency)
    return {
        "zipf_cached": sum(result.cached_tokens for result in zipf_results),
        "conversation_cached": sum(result.cached_tokens for result in conversation_results),
        "answered": len(zipf_results) + len(conversation_results),
    }


@pytest.mark.threadleak(enabled=False)
def test_dkv_workload_reuse_smoke(monkeypatch) -> None:
    """Under DKV every repeated prefix hits on whichever rank serves it; ADP only hits where it ran."""
    if torch.cuda.device_count() < 2:
        pytest.skip("The workload smoke test needs two GPUs")
    if not Path(TINY_LLAMA.path()).is_dir():
        pytest.skip(f"The checkpoint is not available at {TINY_LLAMA.path()}")
    monkeypatch.setenv("TRTLLM_DKV_DEBUG", "1")
    monkeypatch.delenv("TLLM_WORKER_USE_SINGLE_PROCESS", raising=False)
    # Closed loop: every request but the first of each prefix finds its two prefix blocks.
    repeated_prefix_tokens = (_ZIPF_REQUESTS - _PREFIXES) * 2 * TINY_LLAMA.tokens_per_block
    expected_answers = _ZIPF_REQUESTS + 6
    for alternate_ranks in (False, True):
        adp = _run(TINY_LLAMA, dkv=False, alternate_ranks=alternate_ranks)
        dkv = _run(TINY_LLAMA, dkv=True, alternate_ranks=alternate_ranks)
        assert adp["answered"] == dkv["answered"] == expected_answers
        assert dkv["zipf_cached"] >= repeated_prefix_tokens
        assert dkv["zipf_cached"] >= adp["zipf_cached"]
        assert dkv["conversation_cached"] >= adp["conversation_cached"]
        assert dkv["conversation_cached"] > 0
        if alternate_ranks:
            assert dkv["zipf_cached"] > adp["zipf_cached"]
            assert dkv["conversation_cached"] > adp["conversation_cached"]
