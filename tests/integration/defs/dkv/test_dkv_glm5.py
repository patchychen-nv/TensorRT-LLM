# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DKV on GLM-5.3, a DSA (DeepSeek sparse attention) model, with a group of four ranks.

GLM-5.3 runs on ``DSACacheManagerV2``: one KV life cycle that holds, per layer, the MLA latent KV and,
on the "full" indexer layers, the indexer K cache. Both DKV layouts are judged against an
attention-DP group of four that runs the same prompts the same way: one at a time, and in chunks of
one block for the chunked case, where every chunk of the layer split fetches the KV and the indexer
K that the chunks before it wrote back to the owners of the layers.

The checkpoint is read from ``DKV_GLM5_PATH``, else from ``GLM-5.3`` under the models root. Every
build loads about 700 GB, so the attention-DP runs are shared by the tests of the module.
"""

import dataclasses
from pathlib import Path

import pytest

from .dkv_models import GLM_5, PromptTokenizer, available_gpus, dkv_worker_env
from .dkv_precision import (
    MEAN_EXTRA_SIGMAS,
    PrecisionPolicy,
    assert_known_answers,
    assess_prompt_separation,
    build_precision_prompts,
    compare_precision_runs,
)
from .test_dkv_gate import run_precision_passes

_GROUP_SIZE = 4
# One block of tokens per iteration: the longer prompts run in up to six chunks.
_CHUNKED = {"chunked_prefill": True, "max_num_tokens": GLM_5.tokens_per_block}
_DKV_SWITCHES = (
    "DKV_TEST_KV_LAYOUT",
    "TRTLLM_DKV_STAGING_FILL",
    "TRTLLM_DKV_STAGING_LOOPBACK",
    "TRTLLM_DKV_FAULT",
    "TRTLLM_DKV_DEBUG",
)


def _require() -> None:
    if available_gpus() < _GROUP_SIZE:
        pytest.skip(f"GLM-5.3 needs {_GROUP_SIZE} GPUs")
    if not Path(GLM_5.path()).is_dir():
        pytest.skip(f"GLM-5.3 checkpoint is not available at {GLM_5.path()}")


def _precision() -> PrecisionPolicy:
    """The policy of GLM-5.3, with the mean of the extra distance judged by its spread.

    The gates compare 14 requests, whose mean scatters by about half of what the policy allows.
    """
    return dataclasses.replace(GLM_5.precision, mean_extra_sigmas=MEAN_EXTRA_SIGMAS)


def _clear_dkv_switches(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in _DKV_SWITCHES:
        monkeypatch.delenv(key, raising=False)
    for key, value in dkv_worker_env().items():
        monkeypatch.setenv(key, value)
    # The ranks may have been launched with the environment of an earlier LLM.
    monkeypatch.delenv("TLLM_WORKER_USE_SINGLE_PROCESS", raising=False)


@pytest.fixture(autouse=True)
def _glm5_group(monkeypatch: pytest.MonkeyPatch) -> None:
    _require()
    _clear_dkv_switches(monkeypatch)


@pytest.fixture(scope="module")
def prompts() -> tuple[PromptTokenizer, list[list[int]], list]:
    _require()
    tokenizer = PromptTokenizer(GLM_5.path())
    prompt_ids, answers = build_precision_prompts(tokenizer.encode, GLM_5.tokens_per_block)
    return tokenizer, prompt_ids, answers


def _attention_dp_runs(prompt_ids: list[list[int]], **llm_kwargs):
    with pytest.MonkeyPatch.context() as monkeypatch:
        _clear_dkv_switches(monkeypatch)
        return run_precision_passes(
            GLM_5,
            dkv_enabled=False,
            prompts=prompt_ids,
            passes=2,
            group_size=_GROUP_SIZE,
            **llm_kwargs,
        )


@pytest.fixture(scope="module")
def control(prompts):
    """Two passes of an attention-DP group over the prompts: the reference and its noise."""
    return _attention_dp_runs(prompts[1])


@pytest.fixture(scope="module")
def chunked_control(prompts):
    """The same, with the prompts chunked as the chunked layer split chunks them."""
    return _attention_dp_runs(prompts[1], **_CHUNKED)


def _dkv_run(monkeypatch: pytest.MonkeyPatch, layout: str, prompt_ids, **llm_kwargs):
    monkeypatch.setenv("DKV_TEST_KV_LAYOUT", layout)
    (run,) = run_precision_passes(
        GLM_5,
        dkv_enabled=True,
        prompts=prompt_ids,
        passes=1,
        group_size=_GROUP_SIZE,
        **llm_kwargs,
    )
    return run


@pytest.mark.threadleak(enabled=False)
def test_the_attention_dp_group_answers_the_known_prompts_alike_twice(prompts, control) -> None:
    """The control of the DKV runs: an attention-DP group of four that answers what it must.

    Its two passes measure the run-to-run noise that the DKV comparisons are judged against, and
    the prompts must be far enough apart that a request reading another request's KV is noticed.
    """
    tokenizer, _, answers = prompts
    reference, replay = control
    result = compare_precision_runs(reference, replay, replay, _precision())
    assess_prompt_separation(reference, result["mean_adp_self_tv"], GLM_5.precision)
    assert_known_answers(reference, answers, tokenizer.decode, "ADP control")
    assert_known_answers(replay, answers, tokenizer.decode, "ADP replay")
    print(f"GLM-5.3 attention-DP noise: {result}")


@pytest.mark.threadleak(enabled=False)
@pytest.mark.parametrize("checksums", [True, False], ids=["checksums-nan", "no-checksums"])
def test_the_layer_split_group_answers_like_attention_dp(
    monkeypatch: pytest.MonkeyPatch, prompts, control, checksums: bool
) -> None:
    """The layer split of four ranks gets the answers of attention DP.

    Every rank keeps the KV and the indexer K of its layers and computes its requests in the
    working cache.
    With checksums every message is checked at both ends and on the pages it went to, and the
    working pages are overwritten with NaN before every fetch, so a page the attention reads
    without it having been fetched or written turns the logits into NaN. Without them the data
    plane runs as in production.
    """
    tokenizer, prompt_ids, answers = prompts
    if checksums:
        monkeypatch.setenv("TRTLLM_DKV_STAGING_FILL", "nan")
    else:
        monkeypatch.setenv("TRTLLM_DKV_DEBUG", "0")
    split = _dkv_run(monkeypatch, "layer_split", prompt_ids)
    result = compare_precision_runs(*control, split, _precision())
    assert_known_answers(split, answers, tokenizer.decode, "DKV layer split")
    mode = "checksums, NaN fill" if checksums else "no checksums"
    print(f"GLM-5.3 DKV layer split precision ({mode}): {result}")


@pytest.mark.threadleak(enabled=False)
def test_the_layer_split_answers_chunked_prompts_like_attention_dp(
    monkeypatch: pytest.MonkeyPatch, prompts, chunked_control
) -> None:
    """The layer split gets the answers of attention DP for prompts chunked one block at a time.

    Every chunk fetches the latent KV and the indexer K of its history from the owners of the
    layers, which the chunks before it wrote back; checksums and NaN fill are on.
    """
    tokenizer, prompt_ids, answers = prompts
    monkeypatch.setenv("TRTLLM_DKV_STAGING_FILL", "nan")
    split = _dkv_run(monkeypatch, "layer_split", prompt_ids, **_CHUNKED)
    result = compare_precision_runs(*chunked_control, split, _precision())
    assert_known_answers(split, answers, tokenizer.decode, "DKV layer split, chunked")
    print(f"GLM-5.3 DKV layer split precision, chunked prompts: {result}")


@pytest.mark.threadleak(enabled=False)
def test_the_replicated_group_answers_like_attention_dp(
    monkeypatch: pytest.MonkeyPatch, prompts, control
) -> None:
    tokenizer, prompt_ids, answers = prompts
    dkv = _dkv_run(monkeypatch, "replicated", prompt_ids)
    result = compare_precision_runs(*control, dkv, _precision())
    assert_known_answers(dkv, answers, tokenizer.decode, "DKV replicated")
    print(f"GLM-5.3 DKV replicated precision: {result}")
