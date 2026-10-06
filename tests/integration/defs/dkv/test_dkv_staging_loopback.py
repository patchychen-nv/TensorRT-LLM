# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4 on the staging area of the layer-split layout, with every rank owning every layer.

``TRTLLM_DKV_STAGING_LOOPBACK=1`` makes each rank of a replicated DKV group run the attention of a
layer on the staging area: the cached pages of the layer are copied from the rank's own cache manager
into its slot before the layer, and the pages the new tokens wrote are copied back after it. Nothing
crosses a rank, so these tests judge the staging path alone, against the same group without it.
"""

from pathlib import Path

import pytest
import torch

from tensorrt_llm import SamplingParams
from tensorrt_llm.scheduling_params import SchedulingParams

from .dkv_models import (
    DEEPSEEK_V4,
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
    build_burst_prompts,
    build_precision_prompts,
    compare_precision_runs,
)
from .test_dkv_concurrency import _BURSTS, run_burst
from .test_dkv_gate import run_aggregate_soak, run_precision_passes

_GROUP_SIZE = 2


def _require(model: DkvModel) -> None:
    if available_gpus() < _GROUP_SIZE:
        pytest.skip(f"The staging loopback gates need {_GROUP_SIZE} GPUs")
    if not Path(model.path()).is_dir():
        pytest.skip(f"{model.pytest_id} checkpoint is not available at {model.path()}")


def _enable_staging(monkeypatch: pytest.MonkeyPatch, fill: str = "") -> None:
    """Switch the staging loopback on for the LLMs this test builds from now on."""
    monkeypatch.setenv("TRTLLM_DKV_STAGING_LOOPBACK", "1")
    if fill:
        monkeypatch.setenv("TRTLLM_DKV_STAGING_FILL", fill)
    # The ranks may have been launched with the environment of an earlier LLM.
    monkeypatch.delenv("TLLM_WORKER_USE_SINGLE_PROCESS", raising=False)


@pytest.fixture(autouse=True)
def _no_staging_unless_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in ("TRTLLM_DKV_STAGING_LOOPBACK", "TRTLLM_DKV_STAGING_FILL"):
        monkeypatch.delenv(key, raising=False)
    for key, value in dkv_worker_env().items():
        monkeypatch.setenv(key, value)
    monkeypatch.delenv("TLLM_WORKER_USE_SINGLE_PROCESS", raising=False)


@pytest.mark.threadleak(enabled=False)
@pytest.mark.parametrize("fill", ["", "nan"], ids=["staging", "staging-nan"])
def test_the_staged_path_answers_like_the_cache_manager(
    monkeypatch: pytest.MonkeyPatch, fill: str
) -> None:
    """The staging loopback answers the discriminating prompts like the group without it.

    The two are compared within the run-to-run noise of the model. With ``fill="nan"`` the slots
    are overwritten with NaN before every layer's fetch, so a page the layer reads without it having
    been fetched or written would turn the logits into NaN, which the run rejects.
    """
    model = DEEPSEEK_V4
    _require(model)
    tokenizer = PromptTokenizer(model.path())
    prompts, answers = build_precision_prompts(tokenizer.encode, model.tokens_per_block)
    control, replay = run_precision_passes(
        model, dkv_enabled=True, prompts=prompts, passes=2, group_size=_GROUP_SIZE
    )
    _enable_staging(monkeypatch, fill)
    (staged,) = run_precision_passes(
        model, dkv_enabled=True, prompts=prompts, passes=1, group_size=_GROUP_SIZE
    )

    result = compare_precision_runs(control, replay, staged, model.precision)
    assess_prompt_separation(control, result["mean_adp_self_tv"], model.precision)
    assert_known_answers(control, answers, tokenizer.decode, "DKV control")
    assert_known_answers(staged, answers, tokenizer.decode, "DKV staging loopback")
    print(f"staging loopback precision ({fill or 'no fill'}): {result}")


def _prefix_hit_pass(model: DkvModel) -> tuple[PrecisionRun, list[int]]:
    """A prompt and then the same prompt with one token more, both on rank 0.

    The second request finds the whole first prompt in the cache, so its attention reads cached pages
    that another request wrote. Returns the logits of both requests and the tokens each took from
    the cache.
    """
    block = model.tokens_per_block
    tokenizer = PromptTokenizer(model.path())
    # The cache holds the end of a stored sequence, so the prompt ends where a block does.
    (first,) = build_burst_prompts(tokenizer.encode, 1, [2 * block])
    second = first + tokenizer.encode(" and", False)[:1]
    sampling = SamplingParams(
        max_tokens=1, temperature=0, ignore_eos=True, return_generation_logits=True
    )
    placement = SchedulingParams(attention_dp_rank=0, attention_dp_relax=False)
    tokens: list[list[int]] = []
    logits: list[torch.Tensor] = []
    cached: list[int] = []
    with make_dkv_llm(
        model.path(),
        dkv=True,
        tokens_per_block=block,
        moe_config=model.moe_config(),
        group_size=_GROUP_SIZE,
        reuse=True,
        kv_cache={"enable_partial_reuse": False},
        gather_logits=True,
        env_overrides=dkv_worker_env(),
    ) as llm:
        for prompt in (first, second):
            output = llm.generate(
                prompt, sampling_params=sampling, scheduling_params=placement, use_tqdm=False
            )
            result = output.outputs[0]
            assert len(result.token_ids) == 1
            tokens.append(list(result.token_ids))
            logits.append(result.generation_logits.detach().float().cpu().clone())
            cached.append(output.cached_tokens)
    return {"tokens": tokens, "logits": logits}, cached


@pytest.mark.threadleak(enabled=False)
def test_the_staged_path_reads_the_prefix_another_request_cached_on_its_rank(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A request that computes one token on top of 256 cached ones answers like it does without staging."""
    model = DEEPSEEK_V4
    _require(model)
    control, control_cached = _prefix_hit_pass(model)
    replay, replay_cached = _prefix_hit_pass(model)
    _enable_staging(monkeypatch)
    staged, staged_cached = _prefix_hit_pass(model)

    block = model.tokens_per_block
    for cached in (control_cached, replay_cached, staged_cached):
        assert cached[0] == 0 and cached[1] >= 2 * block, f"No prefix hit: cached tokens {cached}"
    result = compare_precision_runs(control, replay, staged, model.precision)
    print(f"staging loopback prefix hit: {result}")


@pytest.mark.threadleak(enabled=False)
def test_the_staged_path_keeps_the_aggregate_lifetime_of_a_request(
    monkeypatch: pytest.MonkeyPatch, capfd
) -> None:
    """One prefill per iteration for 1000 iterations: every request returns its pages."""
    model = DEEPSEEK_V4
    _require(model)
    _enable_staging(monkeypatch)
    for key, value in dkv_worker_env(measurement=True).items():
        monkeypatch.setenv(key, value)
    result = run_aggregate_soak(model, group_size=_GROUP_SIZE)
    assert result["forward_iterations"] >= 1000
    logs = "".join(capfd.readouterr())
    assert "DKV invariant violation" not in logs
    assert "StagingOverflow" not in logs


@pytest.mark.threadleak(enabled=False)
@pytest.mark.parametrize(
    "burst",
    [burst for burst in _BURSTS if burst.pytest_id in ("g2", "chunked-g2", "pressure-g2")],
    ids=lambda burst: burst.pytest_id,
)
def test_the_staged_path_answers_every_request_of_a_burst_once_and_leaks_nothing(
    monkeypatch: pytest.MonkeyPatch, capfd, burst
) -> None:
    """The bursts of the replicated lifecycle with the staging loopback on.

    Concurrent requests share the slots of every iteration, chunked prompts stage a history, and a
    starved pool restarts contexts.
    """
    model = DEEPSEEK_V4
    _require(model)
    _enable_staging(monkeypatch)
    run_burst(monkeypatch, capfd, model, burst)
