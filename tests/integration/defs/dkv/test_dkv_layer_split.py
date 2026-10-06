# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4 on the layer-split layout: each rank keeps the KV of the layers it owns only.

With ``kv_layout="layer_split"`` the cache manager of a rank holds the pages of its own layers. The
rank that computes a request fetches the cached pages of every other layer from the rank that owns
it before the layer runs, and writes the pages the new tokens touch back after it. These tests run
the suites of the replicated layout on it and judge it against the replicated group, within the
run-to-run noise of the model.
"""

from pathlib import Path

import pytest

from .dkv_models import DEEPSEEK_V4, DkvModel, PromptTokenizer, available_gpus, dkv_worker_env
from .dkv_precision import (
    assert_known_answers,
    assess_prompt_separation,
    build_precision_prompts,
    compare_precision_runs,
)
from .test_dkv_concurrency import _BURSTS, run_burst
from .test_dkv_gate import run_aggregate_soak, run_precision_passes

_GROUP_SIZE = 2


def _require(model: DkvModel) -> None:
    if available_gpus() < _GROUP_SIZE:
        pytest.skip(f"The layer-split gates need {_GROUP_SIZE} GPUs")
    if not Path(model.path()).is_dir():
        pytest.skip(f"{model.pytest_id} checkpoint is not available at {model.path()}")


def _layer_split(monkeypatch: pytest.MonkeyPatch, fill: str = "") -> None:
    """Build the DKV LLMs of this test from now on with the layer-split layout."""
    monkeypatch.setenv("DKV_TEST_KV_LAYOUT", "layer_split")
    if fill:
        monkeypatch.setenv("TRTLLM_DKV_STAGING_FILL", fill)
    # The ranks may have been launched with the environment of an earlier LLM.
    monkeypatch.delenv("TLLM_WORKER_USE_SINGLE_PROCESS", raising=False)


@pytest.fixture(autouse=True)
def _replicated_unless_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in ("DKV_TEST_KV_LAYOUT", "TRTLLM_DKV_STAGING_FILL", "TRTLLM_DKV_STAGING_LOOPBACK"):
        monkeypatch.delenv(key, raising=False)
    for key, value in dkv_worker_env().items():
        monkeypatch.setenv(key, value)
    monkeypatch.delenv("TLLM_WORKER_USE_SINGLE_PROCESS", raising=False)


@pytest.mark.threadleak(enabled=False)
@pytest.mark.parametrize("fill", ["", "nan"], ids=["layer-split", "layer-split-nan"])
def test_the_layer_split_answers_like_the_replicated_group(
    monkeypatch: pytest.MonkeyPatch, fill: str
) -> None:
    """The discriminating prompts get the answers of the replicated group.

    The two are compared within the run-to-run noise of the model. With ``fill="nan"`` the slots
    are overwritten with NaN before every layer's fetch, so a page that the layer reads without it
    having been fetched or written turns the logits into NaN, which the run rejects.
    """
    model = DEEPSEEK_V4
    _require(model)
    tokenizer = PromptTokenizer(model.path())
    prompts, answers = build_precision_prompts(tokenizer.encode, model.tokens_per_block)
    control, replay = run_precision_passes(
        model, dkv_enabled=True, prompts=prompts, passes=2, group_size=_GROUP_SIZE
    )
    _layer_split(monkeypatch, fill)
    (split,) = run_precision_passes(
        model, dkv_enabled=True, prompts=prompts, passes=1, group_size=_GROUP_SIZE
    )

    result = compare_precision_runs(control, replay, split, model.precision)
    assess_prompt_separation(control, result["mean_adp_self_tv"], model.precision)
    assert_known_answers(control, answers, tokenizer.decode, "DKV replicated")
    assert_known_answers(split, answers, tokenizer.decode, "DKV layer split")
    print(f"layer split precision ({fill or 'no fill'}): {result}")


@pytest.mark.threadleak(enabled=False)
def test_the_layer_split_keeps_the_aggregate_lifetime_of_a_request(
    monkeypatch: pytest.MonkeyPatch, capfd
) -> None:
    """One prefill per iteration for 1000 iterations: every request returns its pages."""
    model = DEEPSEEK_V4
    _require(model)
    _layer_split(monkeypatch)
    for key, value in dkv_worker_env(measurement=True).items():
        monkeypatch.setenv(key, value)
    result = run_aggregate_soak(model, group_size=_GROUP_SIZE)
    assert result["forward_iterations"] >= 1000
    # The layout ran: every rank computed requests whose KV went to the owners of the layers, and
    # owned layers whose KV came from the other rank.
    data_plane = result["data_plane"]
    assert set(data_plane) == set(range(_GROUP_SIZE)), data_plane
    for rank, counters in data_plane.items():
        assert counters is not None, f"rank {rank} exported no data plane counters"
        assert counters["bytes_sent"] > 0 and counters["bytes_received"] > 0, (rank, counters)
        assert counters["iterations"] >= 1000, (rank, counters)
    logs = "".join(capfd.readouterr())
    assert "DKV invariant violation" not in logs
    assert "StagingOverflow" not in logs


@pytest.mark.threadleak(enabled=False)
@pytest.mark.parametrize(
    "burst",
    [burst for burst in _BURSTS if burst.pytest_id in ("g2", "chunked-g2", "pressure-g2")],
    ids=lambda burst: burst.pytest_id,
)
def test_the_layer_split_answers_every_request_of_a_burst_once_and_leaks_nothing(
    monkeypatch: pytest.MonkeyPatch, capfd, burst
) -> None:
    """The bursts of the replicated lifecycle on the layer split.

    Concurrent requests share the slots of every iteration, chunked prompts fetch a history from the
    owners of the layers, and a starved pool restarts contexts.
    """
    model = DEEPSEEK_V4
    _require(model)
    _layer_split(monkeypatch)
    run_burst(monkeypatch, capfd, model, burst)
