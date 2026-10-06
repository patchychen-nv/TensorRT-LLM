# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Two-rank aggregate lifetime and sequential ADP/DKV precision gates."""

import time
from collections import defaultdict
from pathlib import Path

import pytest
import torch

from tensorrt_llm import SamplingParams
from tensorrt_llm.scheduling_params import SchedulingParams

from .dkv_models import (
    MODEL_IDS,
    MODELS,
    PRECISION_MODELS,
    DkvModel,
    PromptTokenizer,
    available_gpus,
    dkv_worker_env,
    make_dkv_llm,
)
from .dkv_precision import (
    PrecisionRun,
    assert_known_answers,
    assess_prompt_separation,
    build_precision_prompts,
    compare_precision_runs,
    precision_placement,
    validate_precision_inputs,
)
from .dkv_stats import latest_snapshots, pool_free_pages

# The attention-DP group sizes the gates run at; eight ranks take two four-GPU nodes.
_GROUP_SIZES = (2, 8)
_GROUP_IDS = [f"g{size}" for size in _GROUP_SIZES]


def _llm(
    model: DkvModel,
    *,
    dkv_enabled: bool,
    logits: bool,
    group_size: int = 2,
    measurement: bool = False,
    **llm_kwargs,
):
    validate_precision_inputs(enable_block_reuse=False, tokens_per_block=model.tokens_per_block)
    return make_dkv_llm(
        model.path(),
        dkv=dkv_enabled,
        tokens_per_block=model.tokens_per_block,
        moe_config=model.moe_config(),
        group_size=group_size,
        gather_logits=logits,
        env_overrides=dkv_worker_env(measurement=measurement),
        **llm_kwargs,
    )


def _context_rows(stats: list[dict]) -> list[dict]:
    return [
        row
        for row in stats
        if row.get("inflightBatchingStats", {}).get("numContextRequests", 0) > 0
    ]


def _drain_stats(llm, stats: list[dict], expected_contexts: int) -> None:
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


def _assert_replicas_agree_on_free_pages(
    stats: list[dict], *, expected_iterations: int, group_size: int
) -> None:
    """Compare each rank's own free pages, which the kvCacheStats of an attention-DP row cannot."""
    by_iteration: dict[int, dict[int, int]] = defaultdict(dict)
    for row in stats:
        snapshot = row.get("dkvMeasurement")
        if snapshot is not None:
            by_iteration[row["iter"]][snapshot["rank"]] = pool_free_pages(snapshot)
    group = {rank for free in by_iteration.values() for rank in free}
    assert group == set(range(group_size)), (
        f"Per-rank KV snapshots were exported for ranks {sorted(group)}"
    )
    complete = {iteration: free for iteration, free in by_iteration.items() if set(free) == group}
    assert len(complete) >= expected_iterations, (
        f"Only {len(complete)} of {expected_iterations} iterations reported every rank's pools"
    )
    for iteration, free in complete.items():
        assert len(set(free.values())) == 1, (
            f"Replicas disagree on free pages at {iteration}: {free}"
        )


def run_aggregate_soak(model: DkvModel, *, iterations: int = 1000, group_size: int = 2) -> dict:
    """Exercise one real prefill per iteration and verify pages return after every request."""
    if iterations < 1000:
        raise ValueError("The aggregate lifetime gate requires at least 1000 forward iterations")
    sampling = SamplingParams(max_tokens=1, temperature=0, ignore_eos=True)
    stats: list[dict] = []
    request_ids: set[int] = set()
    # Per-rank pool snapshots let the soak compare the replicas' own free pages.
    with _llm(
        model, dkv_enabled=True, logits=False, group_size=group_size, measurement=True
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
                attention_dp_rank=(index // 4) % group_size, attention_dp_relax=False
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
    assert {row["attentionDpRank"] for row in context_rows} == set(range(group_size))
    assert all(row["inflightBatchingStats"]["numContextRequests"] == 1 for row in context_rows)
    assert all(row["inflightBatchingStats"]["numCtxTokens"] == 256 for row in context_rows)
    free_pages = [row["kvCacheStats"]["freeNumBlocks"] for row in context_rows]
    baseline = initial_free_pages[0]
    assert all(free == baseline for free in free_pages), (
        f"KV pages did not return to baseline {baseline}: {free_pages}"
    )
    _assert_replicas_agree_on_free_pages(
        stats, expected_iterations=iterations + 1, group_size=group_size
    )
    return {
        "requests": len(request_ids),
        "forward_iterations": len(by_iteration),
        "baseline_free_blocks": baseline,
        "final_free_blocks": free_pages[-1],
        # What the data plane of each rank moved, when the layout has one.
        "data_plane": {
            snapshot["rank"]: snapshot.get("data_plane") for snapshot in latest_snapshots(stats)
        },
        "stats": stats,
    }


def run_precision_passes(
    model: DkvModel,
    *,
    dkv_enabled: bool,
    prompts: list[list[int]],
    passes: int,
    group_size: int = 2,
    **llm_kwargs,
) -> list[PrecisionRun]:
    """Run the pinned prompts one at a time, ``passes`` times in one executor.

    Repeating inside one executor keeps the ADP control and replay under identical initialization,
    so any difference between them is run-to-run noise of the model itself. ``llm_kwargs`` are
    further arguments of ``make_dkv_llm``, such as the chunked prefill.
    """
    sampling = SamplingParams(
        max_tokens=1, temperature=0, ignore_eos=True, return_generation_logits=True
    )
    placement = precision_placement(group_size)
    runs: list[PrecisionRun] = []
    with _llm(
        model, dkv_enabled=dkv_enabled, logits=True, group_size=group_size, **llm_kwargs
    ) as llm:
        for _ in range(passes):
            tokens: list[list[int]] = []
            logits: list[torch.Tensor] = []
            for index, prompt in enumerate(prompts):
                output = llm.generate(
                    prompt,
                    sampling_params=sampling,
                    scheduling_params=SchedulingParams(
                        attention_dp_rank=placement[index], attention_dp_relax=False
                    ),
                    use_tqdm=False,
                )
                assert len(output.outputs) == 1
                result = output.outputs[0]
                assert len(result.token_ids) == 1
                assert isinstance(result.generation_logits, torch.Tensor), (
                    "Generation logits missing"
                )
                values = result.generation_logits.detach().float().cpu().clone()
                assert values.numel() > 0, "Generation logits are empty"
                assert torch.isfinite(values).all()
                tokens.append(list(result.token_ids))
                logits.append(values)
            runs.append({"tokens": tokens, "logits": logits})
    return runs


@pytest.fixture
def dkv_gate_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    if torch.cuda.device_count() < 2:
        pytest.skip("DKV aggregate gates require two GPUs")
    # The LLM also hands these to ranks that MPI launched; setting them here is what puts them
    # back after the test.
    for key, value in dkv_worker_env().items():
        monkeypatch.setenv(key, value)
    monkeypatch.delenv("TLLM_WORKER_USE_SINGLE_PROCESS", raising=False)


def _require_checkpoint(model: DkvModel) -> None:
    if not Path(model.path()).is_dir():
        pytest.skip(f"{model.pytest_id} checkpoint is not available at {model.path()}")


def _require_group(group_size: int) -> None:
    if available_gpus() < group_size:
        pytest.skip(f"A group of {group_size} ranks needs {group_size} GPUs")


@pytest.mark.threadleak(enabled=False)
@pytest.mark.parametrize("group_size", _GROUP_SIZES, ids=_GROUP_IDS)
@pytest.mark.parametrize("model", MODELS, ids=MODEL_IDS)
def test_dkv_aggregate_1000_steps(
    dkv_gate_environment, monkeypatch, capfd, model: DkvModel, group_size: int
) -> None:
    _require_checkpoint(model)
    _require_group(group_size)
    for key, value in dkv_worker_env(measurement=True).items():
        monkeypatch.setenv(key, value)
    result = run_aggregate_soak(model, group_size=group_size)
    assert result["forward_iterations"] >= 1000
    captured = capfd.readouterr()
    logs = captured.out + captured.err
    assert "DKV invariant violation" not in logs
    assert "No free slots" not in logs
    assert "exceeds expected_num_active_requests" not in logs


@pytest.mark.threadleak(enabled=False)
@pytest.mark.parametrize("group_size", _GROUP_SIZES, ids=_GROUP_IDS)
@pytest.mark.parametrize("model", PRECISION_MODELS, ids=[m.pytest_id for m in PRECISION_MODELS])
def test_dkv_sequential_precision(dkv_gate_environment, model: DkvModel, group_size: int) -> None:
    """DKV must match the ADP control on prompts that tell a wrong KV from a right one.

    The prompts are real text of different lengths spread over the ranks, plus four sentences with
    known answers. The ADP control must itself answer them, and different prompts must be far enough
    apart that a request reading another request's KV could not pass.
    """
    _require_checkpoint(model)
    _require_group(group_size)
    tokenizer = PromptTokenizer(model.path())
    prompts, answers = build_precision_prompts(tokenizer.encode, model.tokens_per_block)
    control, replay = run_precision_passes(
        model, dkv_enabled=False, prompts=prompts, passes=2, group_size=group_size
    )
    (dkv,) = run_precision_passes(
        model, dkv_enabled=True, prompts=prompts, passes=1, group_size=group_size
    )

    policy = model.precision
    result = compare_precision_runs(control, replay, dkv, policy)
    assess_prompt_separation(control, result["mean_adp_self_tv"], policy)
    assert_known_answers(control, answers, tokenizer.decode, "ADP control")
    assert_known_answers(dkv, answers, tokenizer.decode, "DKV")
    print(f"DKV precision {model.pytest_id}: {result}")
