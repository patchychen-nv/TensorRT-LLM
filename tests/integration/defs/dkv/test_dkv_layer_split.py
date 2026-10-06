# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4 on the layer-split layout: each rank keeps the KV of the layers it owns only.

With ``kv_layout="layer_split"`` the cache manager of a rank holds the pages of its own layers. The
rank that computes a request fetches the cached pages of every other layer from the rank that owns
it before the layer runs, and writes the pages the new tokens touch back after it. These tests run
the suites of the replicated layout on it and judge it against the replicated group, within the
run-to-run noise of the model.
"""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest
import torch

from tensorrt_llm import SamplingParams
from tensorrt_llm._torch.pyexecutor.dkv import compute_ownership
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
    failing_requests,
)
from .dkv_workloads import multi_turn_workload, run_workload
from .test_dkv_concurrency import _BURSTS, _Burst, run_burst
from .test_dkv_gate import run_aggregate_soak, run_precision_passes
from .test_dkv_host_tier import run_host_tier_case

_GROUP_SIZE = 2
# The pairs of requests of the cross-rank reuse gate. A request has one compared position, and the
# mean of the extra distance over the requests must have a spread well inside the threshold, which
# six requests did not have.
_CROSS_RANK_PAIRS = 12
# The conversations of the soak, which is also the number of requests it keeps in flight.
_CONVERSATIONS = 4
# How long a group that is expected to die may take to say so, and to be killed after it did.
_GROUP_TIMEOUT_S = 1200
_GROUP_GRACE_S = 20


def _require(model: DkvModel, group_size: int = _GROUP_SIZE) -> None:
    if available_gpus() < group_size:
        pytest.skip(f"The layer-split gates need {group_size} GPUs")
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
    for key in (
        "DKV_TEST_KV_LAYOUT",
        "TRTLLM_DKV_STAGING_FILL",
        "TRTLLM_DKV_STAGING_LOOPBACK",
        "TRTLLM_DKV_FAULT",
    ):
        monkeypatch.delenv(key, raising=False)
    for key, value in dkv_worker_env().items():
        monkeypatch.setenv(key, value)
    monkeypatch.delenv("TLLM_WORKER_USE_SINGLE_PROCESS", raising=False)


@pytest.mark.threadleak(enabled=False)
@pytest.mark.parametrize(
    ("fill", "group_size"),
    [("", 2), ("nan", 2), ("nan", 4)],
    ids=["layer-split", "layer-split-nan", "layer-split-nan-g4"],
)
def test_the_layer_split_answers_like_the_replicated_group(
    monkeypatch: pytest.MonkeyPatch, fill: str, group_size: int
) -> None:
    """The discriminating prompts get the answers of the replicated group.

    The two are compared within the run-to-run noise of the model. With ``fill="nan"`` the slots
    are overwritten with NaN before every layer's fetch, so a page that the layer reads without it
    having been fetched or written turns the logits into NaN, which the run rejects.
    """
    model = DEEPSEEK_V4
    _require(model, group_size)
    tokenizer = PromptTokenizer(model.path())
    prompts, answers = build_precision_prompts(tokenizer.encode, model.tokens_per_block)
    control, replay = run_precision_passes(
        model, dkv_enabled=True, prompts=prompts, passes=2, group_size=group_size
    )
    _layer_split(monkeypatch, fill)
    (split,) = run_precision_passes(
        model, dkv_enabled=True, prompts=prompts, passes=1, group_size=group_size
    )

    result = compare_precision_runs(control, replay, split, model.precision)
    assess_prompt_separation(control, result["mean_adp_self_tv"], model.precision)
    assert_known_answers(control, answers, tokenizer.decode, "DKV replicated")
    assert_known_answers(split, answers, tokenizer.decode, "DKV layer split")
    print(f"layer split precision ({fill or 'no fill'}): {result}")


@pytest.mark.threadleak(enabled=False)
def test_the_layer_split_answers_chunked_prompts_like_the_replicated_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every chunk of a prompt fetches what the chunks before it wrote back.

    The token budget of an iteration is one block, so the longer of the discriminating prompts run
    in two or three chunks, with boundaries inside a block and on one. The layer split is judged
    against the replicated group that chunks the same way, with the slots overwritten with NaN
    before every fetch.
    """
    model = DEEPSEEK_V4
    _require(model)
    tokenizer = PromptTokenizer(model.path())
    prompts, answers = build_precision_prompts(tokenizer.encode, model.tokens_per_block)
    chunked = {"chunked_prefill": True, "max_num_tokens": model.tokens_per_block}
    control, replay = run_precision_passes(
        model, dkv_enabled=True, prompts=prompts, passes=2, group_size=_GROUP_SIZE, **chunked
    )
    _layer_split(monkeypatch, "nan")
    (split,) = run_precision_passes(
        model, dkv_enabled=True, prompts=prompts, passes=1, group_size=_GROUP_SIZE, **chunked
    )

    result = compare_precision_runs(control, replay, split, model.precision)
    assess_prompt_separation(control, result["mean_adp_self_tv"], model.precision)
    assert_known_answers(split, answers, tokenizer.decode, "DKV layer split, chunked")
    print(f"layer split precision, chunked prompts: {result}")


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


@pytest.mark.threadleak(enabled=False)
def test_the_layer_split_serves_a_burst_through_a_staging_area_of_one_request(
    monkeypatch: pytest.MonkeyPatch, capfd
) -> None:
    """The scheduler keeps what a rank attends to within the staging area.

    The area holds one request of ``max_seq_len`` and no more. Chunked prompts of several requests
    per rank would attend to more than that together, so the scheduler has to hold requests back,
    and a request that overflowed the area would fail the plan.
    """
    model = DEEPSEEK_V4
    _require(model)
    _layer_split(monkeypatch)
    burst = _Burst("staging-g2", chunked=True, max_num_tokens=512, lengths=(700, 1100, 1400))
    monkeypatch.setenv("TRTLLM_DKV_STAGING_TOKENS", str(burst.max_seq_len))
    run_burst(monkeypatch, capfd, model, burst)


def _cross_rank_pass(
    model: DkvModel, *, reuse: bool, group_size: int
) -> tuple[PrecisionRun, list[int]]:
    """A prompt on rank 0, then the same prompt with one token more on the last rank, per pair.

    With reuse the second request of a pair finds the whole first prompt in the cache although
    another rank computed it, so its attention reads cached pages that the owners of the layers
    keep. The noise of the model is the size of the effect that the gate looks for in a pair, so the
    gate compares the requests of all pairs. Returns their logits and the tokens each took from the
    cache.
    """
    block = model.tokens_per_block
    tokenizer = PromptTokenizer(model.path())
    # The cache holds the end of a stored sequence, so the prompt ends where a block does.
    firsts = build_burst_prompts(tokenizer.encode, _CROSS_RANK_PAIRS, [3 * block])
    suffix = tokenizer.encode(" and", False)[:1]
    prompts = [
        (prompt, rank)
        for first in firsts
        for prompt, rank in ((first, 0), (first + suffix, group_size - 1))
    ]
    sampling = SamplingParams(
        max_tokens=1, temperature=0, ignore_eos=True, return_generation_logits=True
    )
    tokens: list[list[int]] = []
    logits: list[torch.Tensor] = []
    cached: list[int] = []
    with make_dkv_llm(
        model.path(),
        dkv=True,
        tokens_per_block=block,
        moe_config=model.moe_config(),
        group_size=group_size,
        reuse=reuse,
        kv_cache={"enable_partial_reuse": False},
        gather_logits=True,
        env_overrides=dkv_worker_env(),
    ) as llm:
        for prompt, rank in prompts:
            output = llm.generate(
                prompt,
                sampling_params=sampling,
                scheduling_params=SchedulingParams(
                    attention_dp_rank=rank, attention_dp_relax=False
                ),
                use_tqdm=False,
            )
            result = output.outputs[0]
            assert len(result.token_ids) == 1
            tokens.append(list(result.token_ids))
            logits.append(result.generation_logits.detach().float().cpu().clone())
            cached.append(output.cached_tokens)
    return {"tokens": tokens, "logits": logits}, cached


@pytest.mark.threadleak(enabled=False)
@pytest.mark.parametrize("group_size", [2, 4, 8], ids=["g2", "g4", "g8"])
def test_a_request_reads_the_prefix_that_a_request_of_another_rank_cached(
    monkeypatch: pytest.MonkeyPatch, group_size: int
) -> None:
    """The answer with the prefix reused from another rank is the answer without reuse."""
    model = DEEPSEEK_V4
    _require(model, group_size)
    _layer_split(monkeypatch)
    control, control_cached = _cross_rank_pass(model, reuse=False, group_size=group_size)
    replay, _ = _cross_rank_pass(model, reuse=False, group_size=group_size)
    reused, cached = _cross_rank_pass(model, reuse=True, group_size=group_size)

    block = model.tokens_per_block
    assert control_cached == [0] * (2 * _CROSS_RANK_PAIRS), (
        f"Reuse is off, yet tokens came from the cache: {control_cached}"
    )
    assert cached[0::2] == [0] * _CROSS_RANK_PAIRS and all(
        value >= 3 * block for value in cached[1::2]
    ), f"No prefix hit across ranks: {cached}"
    result = compare_precision_runs(control, replay, reused, model.precision)
    print(f"layer split, prefix reused across ranks: {result}")


@pytest.mark.threadleak(enabled=False)
def test_the_replicated_layout_fails_the_comparison_of_the_prefix_reused_across_ranks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The negative control: without the layer split the second rank reads pages it never wrote."""
    model = DEEPSEEK_V4
    _require(model)
    control, _ = _cross_rank_pass(model, reuse=False, group_size=_GROUP_SIZE)
    replay, _ = _cross_rank_pass(model, reuse=False, group_size=_GROUP_SIZE)
    reused, cached = _cross_rank_pass(model, reuse=True, group_size=_GROUP_SIZE)
    assert all(value >= 3 * model.tokens_per_block for value in cached[1::2]), (
        f"No prefix hit across ranks: {cached}"
    )
    failing = failing_requests(control, replay, reused, model.precision)
    print(
        f"replicated layout, prefix reused across ranks: failing requests {failing} of {len(cached)}"
    )
    with pytest.raises(AssertionError):
        compare_precision_runs(control, replay, reused, model.precision)


@pytest.mark.threadleak(enabled=False)
def test_the_layer_split_serves_conversations_across_ranks_with_reuse(
    monkeypatch: pytest.MonkeyPatch, capfd
) -> None:
    """Every turn of a conversation reads the history that a request of another rank cached.

    The turns of a conversation alternate over the ranks and each of them adds a block, so every
    turn hits the whole prompt of the turn before, whose pages the owners of the layers keep. The
    slots are overwritten with NaN before every fetch, the checksums compare every message, and the
    logits of every request must be finite. The cache of a rank holds the end of a stored sequence
    in a few pages of state, and the requests in flight need some of them too, so the sessions are
    as many as the requests in flight: the end that a turn needs is then among the latest ones.
    """
    model = DEEPSEEK_V4
    _require(model)
    _layer_split(monkeypatch, "nan")
    block = model.tokens_per_block
    conversations = multi_turn_workload(
        sessions=_CONVERSATIONS,
        turns=6,
        first_tokens=2 * block,
        turn_tokens=block,
        vocab=30000,
        seed=5,
    )
    sampling = SamplingParams(
        max_tokens=1, temperature=0, ignore_eos=True, return_generation_logits=True
    )
    with make_dkv_llm(
        model.path(),
        dkv=True,
        tokens_per_block=block,
        moe_config=model.moe_config(),
        group_size=_GROUP_SIZE,
        reuse=True,
        gather_logits=True,
        max_num_tokens=1024,
        max_seq_len=1024,
        kv_cache={"enable_partial_reuse": False, "max_tokens": 16384},
        env_overrides=dkv_worker_env(),
    ) as llm:

        def submit(request):
            future = llm.generate_async(
                request.prompt,
                sampling_params=sampling,
                scheduling_params=SchedulingParams(
                    attention_dp_rank=(request.index + request.turn) % _GROUP_SIZE,
                    attention_dp_relax=False,
                ),
            )

            def wait() -> int:
                output = future.result(timeout=300)
                logits = output.outputs[0].generation_logits
                assert torch.isfinite(logits).all(), f"request {request.index} has NaN logits"
                return output.cached_tokens

            return wait

        results = run_workload(submit, conversations, concurrency=_CONVERSATIONS)

    assert len(results) == len(conversations)
    for result in results:
        if result.turn:
            # The whole prompt of the turn before, which ended where a block does.
            assert result.cached_tokens >= result.prompt_tokens - block, result
        else:
            assert result.cached_tokens == 0, result
    logs = "".join(capfd.readouterr())
    assert "DKV invariant violation" not in logs
    assert "StagingOverflow" not in logs


@pytest.mark.threadleak(enabled=False)
def test_the_layer_split_spills_prefixes_to_the_host_tier_and_onboards_them_again(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Blocks that leave the GPU pool of a layer-split group are kept by its host tier.

    Every rank spills and onboards the layers it holds, and a prefix that a request of the other
    rank computed hits again after it left the GPU.
    """
    model = DEEPSEEK_V4
    _require(model)
    _layer_split(monkeypatch)
    run_host_tier_case(monkeypatch, model)


def _failure_text(error: BaseException | None) -> str:
    """The messages of an error and of the errors that caused it.

    The proxy of the executor reports a failure of a worker as a generic error, and the message of
    the worker is that of its cause.
    """
    texts = []
    while error is not None:
        texts.append(str(error))
        error = error.__cause__ or error.__context__
    return "\n".join(texts)


@pytest.mark.threadleak(enabled=False)
@pytest.mark.parametrize("kind", ["flip", "zero"])
def test_a_writeback_that_arrives_damaged_at_its_owner_fails_the_group_and_is_named(
    monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    """The checksums see a damaged writeback, whose output nothing else would reveal.

    The first layer of rank 1 is written back to it by the requests that compute on rank 0. Their
    messages are damaged on arrival at every iteration, so the group fails at the first forward pass,
    the one of the memory profiling at startup, with an error that names the message.
    """
    model = DEEPSEEK_V4
    _require(model)
    layers = json.loads((Path(model.path()) / "config.json").read_text())["num_hidden_layers"]
    layer = compute_ownership(layers, _GROUP_SIZE).index(1)
    _layer_split(monkeypatch)
    monkeypatch.setenv("TRTLLM_DKV_FAULT", f"*:{layer}:writeback:{kind}:1")
    tokenizer = PromptTokenizer(model.path())
    prompt = tokenizer.encode("The capital of France is", True)
    with pytest.raises(Exception) as caught:
        with make_dkv_llm(
            model.path(),
            dkv=True,
            tokens_per_block=model.tokens_per_block,
            moe_config=model.moe_config(),
            group_size=_GROUP_SIZE,
            env_overrides=dkv_worker_env(),
        ) as llm:
            llm.generate(
                prompt,
                sampling_params=SamplingParams(max_tokens=1, temperature=0),
                scheduling_params=SchedulingParams(attention_dp_rank=0, attention_dp_relax=False),
                use_tqdm=False,
            )
    message = _failure_text(caught.value)
    print(f"the group failed with: {message}")
    assert f"W({layer})/" in message and "checksums differ" in message, message
    assert "owner rank 1, compute rank 0" in message, message


def _run_group_until_it_names_its_failure(options: dict, log: Path, marker: str) -> str:
    """The output of a DKV group that is expected to die, run in a process of its own.

    The group stops every rank when the data plane finds a damaged message, and its client then
    waits for an answer for ever, so the process is killed shortly after it wrote ``marker`` (or
    when it ends or runs out of time).
    """
    child = Path(__file__).with_name("dkv_group_child.py")
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("SLURM_", "PMI_", "PMIX_", "OMPI_"))
    }
    env["PYTHONPATH"] = os.pathsep.join([str(child.parent), env.get("PYTHONPATH", "")])
    env.pop("TLLM_WORKER_USE_SINGLE_PROCESS", None)
    with log.open("w") as stream:
        process = subprocess.Popen(
            [sys.executable, str(child), json.dumps(options)],
            stdout=stream,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )
    deadline = time.monotonic() + _GROUP_TIMEOUT_S
    named_at = None
    try:
        while process.poll() is None and time.monotonic() < deadline:
            if named_at is None and marker in log.read_text(errors="replace"):
                named_at = time.monotonic()
            if named_at is not None and time.monotonic() - named_at > _GROUP_GRACE_S:
                break
            time.sleep(2)
    finally:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=60)
    return log.read_text(errors="replace")


@pytest.mark.threadleak(enabled=False)
@pytest.mark.parametrize("kind", ["flip", "zero"])
def test_a_fetch_that_arrives_damaged_at_the_compute_rank_fails_the_group_and_is_named(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, kind: str
) -> None:
    """The checksums see a fetch that arrives damaged, and name the message.

    The cached pages of a chunked prompt come from the owner of the layer. The first chunk of a
    request has nothing to fetch, so the group starts. The second chunk, on rank 0, fetches the
    first layer of rank 1, and the fault damages every such message on arrival. The group stops
    while it serves the request, and its client does not hear of it, so the group runs in a
    process of its own and the test reads what its ranks wrote.
    """
    model = DEEPSEEK_V4
    _require(model)
    layers = json.loads((Path(model.path()) / "config.json").read_text())["num_hidden_layers"]
    layer = compute_ownership(layers, _GROUP_SIZE).index(1)
    _layer_split(monkeypatch)
    monkeypatch.setenv("TRTLLM_DKV_FAULT", f"*:{layer}:fetch:{kind}:0")
    tokenizer = PromptTokenizer(model.path())
    (prompt,) = build_burst_prompts(tokenizer.encode, 1, [1100])
    options = {
        "model": model.path(),
        "tokens_per_block": model.tokens_per_block,
        "moe_backend": model.moe_backend or "CUTLASS",
        "disable_finalize_fusion": model.disable_finalize_fusion,
        "group_size": _GROUP_SIZE,
        "max_seq_len": 1536,
        "prompt": prompt,
    }
    output = _run_group_until_it_names_its_failure(
        options, tmp_path / "group.log", "checksums differ"
    )
    assert "DKV_GROUP_CHILD_DONE" not in output, "the group served the damaged fetch"
    assert f"F({layer})/" in output and "checksums differ" in output, output[-4000:]
    assert "owner rank 1, compute rank 0" in output, output[-4000:]
