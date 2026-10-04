# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Concurrent bursts through the real executor: one answer per request and no leaked KV pages."""

import time
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch

from tensorrt_llm import SamplingParams
from tensorrt_llm.scheduling_params import SchedulingParams

from .dkv_models import MODEL_IDS, MODELS, DkvModel, PromptTokenizer, make_dkv_llm
from .dkv_precision import build_burst_prompts
from .dkv_stats import latest_free_pages_by_rank


@dataclass(frozen=True)
class _Burst:
    """One executor configuration and the mix of prompt lengths that is thrown at it."""

    pytest_id: str
    group_size: int = 2
    chunked: bool = False
    graphs: bool = False
    max_batch_size: int = 4
    max_num_tokens: int = 1024
    max_seq_len: int = 1536
    lengths: tuple[int, ...] = (64, 256, 512, 1000)
    requests: int = 32
    # KV pool size in tokens; None leaves room for every request.
    kv_tokens: int | None = None


_BURSTS = (
    # Several requests per rank at once, so a rank's budget runs out while others keep admitting.
    _Burst("g2"),
    # Four ranks: every replica runs the lifecycle of the whole group's requests.
    _Burst("g4", group_size=4),
    # Prompts longer than the per-iteration token budget, so their context is chunked.
    _Burst("chunked-g2", chunked=True, max_num_tokens=512, lengths=(700, 1100, 1400)),
    # A pool that holds about one and a half prompts: started contexts block each other, and the
    # scheduler has to release one of them and restart it.
    _Burst(
        "pressure-g2",
        chunked=True,
        max_num_tokens=512,
        lengths=(700, 1100, 1400),
        kv_tokens=2100,
    ),
    # CUDA graphs and the autotuner on a replicated batch whose local view changes every iteration.
    _Burst("graphs-g2", graphs=True, max_seq_len=1024, lengths=(64, 256, 512)),
)


def _context_rows(rows: list[dict]) -> list[dict]:
    return [
        row for row in rows if row.get("inflightBatchingStats", {}).get("numContextRequests", 0) > 0
    ]


def _collect(llm, rows: list[dict], *, until, timeout: float = 30.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        rows.extend(llm.get_stats(timeout=0.2))
        if until(rows):
            return
    pytest.fail("Iteration stats did not account for the submitted requests in time")


def _probe_free_pages(llm, prompt: list[int], sampling, group_size: int, seen: list[dict]):
    """Run one short request on every rank and return each rank's free pages once they are done.

    ``seen`` holds the iteration rows read so far, so only the rows of the probes are used.
    """
    last_iteration = max((row["iter"] for row in seen), default=-1)
    for rank in range(group_size):
        probe = llm.generate(
            prompt[:64],
            sampling_params=sampling,
            scheduling_params=SchedulingParams(attention_dp_rank=rank, attention_dp_relax=False),
            use_tqdm=False,
        )
        assert len(probe.outputs[0].token_ids) == 1
    after: list[dict] = []
    _collect(
        llm,
        after,
        until=lambda rows: sum(
            row["inflightBatchingStats"]["numContextRequests"]
            for row in _context_rows([row for row in rows if row["iter"] > last_iteration])
        )
        >= group_size,
    )
    return latest_free_pages_by_rank([row for row in after if row["iter"] > last_iteration])


@pytest.mark.threadleak(enabled=False)
@pytest.mark.parametrize("burst", _BURSTS, ids=[burst.pytest_id for burst in _BURSTS])
@pytest.mark.parametrize("model", MODELS, ids=MODEL_IDS)
def test_dkv_concurrent_burst_answers_every_request_once_and_leaks_nothing(
    monkeypatch, capfd, model: DkvModel, burst: _Burst
) -> None:
    if torch.cuda.device_count() < burst.group_size:
        pytest.skip(f"The {burst.pytest_id} burst needs {burst.group_size} GPUs")
    if not Path(model.path()).is_dir():
        pytest.skip(f"{model.pytest_id} checkpoint is not available at {model.path()}")
    monkeypatch.setenv("TRTLLM_DKV_DEBUG", "1")
    monkeypatch.setenv("TRTLLM_DKV_DUAL_LEDGER", "1")
    # Each rank's own pool snapshot rides on the iteration stats; kvCacheStats is rank 0's copy.
    monkeypatch.setenv("TRTLLM_DKV_MEASUREMENT", "1")
    monkeypatch.setenv("TLLM_LOG_LEVEL", "WARNING")
    monkeypatch.delenv("TLLM_WORKER_USE_SINGLE_PROCESS", raising=False)
    tokenizer = PromptTokenizer(model.path())
    prompts = build_burst_prompts(tokenizer.encode, burst.requests, burst.lengths)
    sampling = SamplingParams(max_tokens=1, temperature=0, ignore_eos=True)
    with make_dkv_llm(
        model.path(),
        dkv=True,
        tokens_per_block=model.tokens_per_block,
        moe_config=model.moe_config(),
        group_size=burst.group_size,
        chunked_prefill=burst.chunked,
        cuda_graph=burst.graphs,
        autotuner=burst.graphs,
        max_batch_size=burst.max_batch_size,
        max_num_tokens=burst.max_num_tokens,
        max_seq_len=burst.max_seq_len,
        kv_cache={"max_tokens": burst.kv_tokens or 65536},
    ) as llm:
        # The idle pools after one probe per rank are the state every replica must return to. The
        # startup row and the first iteration both carry iteration number 0, so the startup rows
        # are discarded rather than used to tell the probe's rows apart.
        llm.get_stats(timeout=5)
        baseline = _probe_free_pages(llm, prompts[0], sampling, burst.group_size, [])
        assert set(baseline) == set(range(burst.group_size)), (
            f"Pool snapshots came from ranks {sorted(baseline)}"
        )
        assert len(set(baseline.values())) == 1, f"Replicas start with different pools: {baseline}"
        futures = [llm.generate_async(prompt, sampling_params=sampling) for prompt in prompts]
        outputs = [future.result(timeout=900) for future in futures]
        assert len({output.request_id for output in outputs}) == len(prompts)
        for output in outputs:
            assert len(output.outputs) == 1 and len(output.outputs[0].token_ids) == 1
            assert str(output.outputs[0].finish_reason).endswith("length")

        rows: list[dict] = []
        if not burst.chunked:
            _collect(
                llm,
                rows,
                until=lambda rows: sum(
                    row["inflightBatchingStats"]["numContextRequests"]
                    for row in _context_rows(rows)
                )
                >= len(prompts),
            )
            counted = sum(
                row["inflightBatchingStats"]["numContextRequests"] for row in _context_rows(rows)
            )
            assert counted == len(prompts), "A replicated request was counted more than once"
        else:
            rows.extend(llm.get_stats(timeout=2))
        served = {row["attentionDpRank"] for row in _context_rows(rows)}
        assert served == set(range(burst.group_size)), f"Only ranks {sorted(served)} served work"

        # Probe every rank once more, then require every replica to be back at the baseline.
        free = _probe_free_pages(llm, prompts[0], sampling, burst.group_size, rows)
        assert free == baseline, (
            f"KV pages leaked after the burst: baseline {baseline}, free {free}"
        )
