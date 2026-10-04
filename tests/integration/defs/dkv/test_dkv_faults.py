# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Two-GPU aggregate cancellation and recoverable rank-local sampler errors."""

import json
import os
import time
from collections import Counter
from pathlib import Path

import pytest
import torch

from tensorrt_llm import SamplingParams
from tensorrt_llm._torch.pyexecutor.llm_request import FinishReason
from tensorrt_llm.llmapi.llm_args import MoeConfig
from tensorrt_llm.scheduling_params import SchedulingParams

from .dkv_fault_probe import FAULT_MESSAGE, FAULT_TOKEN
from .test_dkv_gate import _llm


def _wait(path: Path) -> None:
    deadline = time.monotonic() + 60
    while not path.exists():
        if time.monotonic() > deadline:
            pytest.fail(f"Worker handshake missing: {path.name}")
        time.sleep(0.01)


def _events(trace_dir: Path) -> list[dict]:
    return [
        json.loads(line)
        for path in sorted(trace_dir.glob("rank-*.jsonl"))
        for line in path.read_text().splitlines()
        if line
    ]


def run_aggregate_faults(
    model_path: str,
    trace_dir: str,
    *,
    moe_config: MoeConfig | None = None,
    tokens_per_block: int = 32,
) -> dict:
    """Require actual cancellation, one owner error, symmetric frees and subsequent inference."""
    traces = Path(trace_dir)
    sampling = SamplingParams(max_tokens=1, temperature=0, ignore_eos=True)
    pinned = SchedulingParams(attention_dp_rank=1, attention_dp_relax=False)
    with _llm(
        model_path,
        dkv_enabled=True,
        logits=False,
        moe_config=moe_config,
        tokens_per_block=tokens_per_block,
    ) as llm:
        try:
            for rank in range(2):
                _wait(traces / f"fetch-ready-{rank}")
            # Fill this rank's forward budget so cancellation reaches an unfinished request.
            blocker = llm.generate_async(
                [1] + [40] * 383,
                sampling_params=sampling,
                scheduling_params=pinned,
            )
            cancelled = llm.generate_async(
                [1] + [42] * 255, sampling_params=sampling, scheduling_params=pinned
            )
            cancelled.abort()
            _wait(traces / "cancel-queued")
        finally:
            # Release even on assertion failure so LLM shutdown can join its worker threads.
            (traces / "release-fetch").touch()

        cancel_id = int((traces / "cancel-queued").read_text())
        for rank in range(2):
            _wait(traces / f"freed-{cancel_id}-{rank}")
        assert len(blocker.result(timeout=60).outputs[0].token_ids) == 1

        failed = llm.generate_async(
            [1] + [FAULT_TOKEN] * 255, sampling_params=sampling, scheduling_params=pinned
        )
        with pytest.raises(RuntimeError, match=FAULT_MESSAGE):
            failed.result(timeout=60)
        for rank in (0, 1, 0, 1):
            recovered = llm.generate_async(
                [1] + [43 + rank] * 255,
                sampling_params=sampling,
                scheduling_params=SchedulingParams(
                    attention_dp_rank=rank, attention_dp_relax=False
                ),
            ).result(timeout=60)
            assert len(recovered.outputs[0].token_ids) == 1

    events = _events(traces)
    baselines = [event for event in events if event["event"] == "baseline"]
    assert {event["rank"] for event in baselines} == {0, 1}
    baseline = baselines[0]["free_blocks"]
    assert all(event["free_blocks"] == baseline for event in baselines)
    injected = [event for event in events if event["event"] == "injected"]
    assert len(injected) == 1 and injected[0]["rank"] == 1
    assert len(injected[0]["request_ids"]) == 1
    fault_id = injected[0]["request_ids"][0]
    cancellations = [event for event in events if event["event"] == "cancel_applied"]
    assert len(cancellations) == 2
    assert {event["rank"] for event in cancellations} == {0, 1}
    assert all(event["accepted"] for event in cancellations)
    cancel_id = cancellations[0]["request_id"]
    assert {event["request_id"] for event in cancellations} == {cancel_id}

    responses = [event for event in events if event["event"] == "response"]
    assert len(responses) == 7, (
        f"Expected blocker, cancellation, fault and four recovery replies: {responses}"
    )
    assert all(count == 1 for count in Counter(event["request_id"] for event in responses).values())
    errors = [event for event in responses if event["error"] is not None]
    assert len(errors) == 1
    assert errors[0]["request_id"] == fault_id and errors[0]["rank"] == 1
    assert FAULT_MESSAGE in errors[0]["error"]

    frees = [event for event in events if event["event"] == "free"]
    freed_by_rank = [
        [(event["request_id"], event["iteration"]) for event in frees if event["rank"] == rank]
        for rank in range(2)
    ]
    assert len(freed_by_rank[0]) == 7
    assert freed_by_rank[0] == freed_by_rank[1], "Ranks freed different requests or iterations"
    last_free_by_iteration = {(event["rank"], event["iteration"]): event for event in frees}
    assert all(event["free_blocks"] == baseline for event in last_free_by_iteration.values()), (
        "Request KV pages leaked after the iteration's final free"
    )
    owners = {event["request_id"]: event["owner"] for event in frees}
    assert all(event["rank"] == owners[event["request_id"]] for event in responses)
    assert all(
        FinishReason.CANCELLED.value in event["finish_reasons"]
        for event in frees
        if event["request_id"] == cancel_id
    ), "The cancellation must reach native request state before termination"
    return {
        "status": "passed",
        "fault_request_id": fault_id,
        "cancel_request_id": cancel_id,
        "unique_raw_responses": len(responses),
        "recovery_requests": 4,
        "baseline_free_blocks": baseline,
        "free_sequence_by_rank": freed_by_rank,
    }


@pytest.mark.post_merge
@pytest.mark.threadleak(enabled=False)
def test_dkv_aggregate_sampler_failure_and_cancel(monkeypatch, tmp_path: Path) -> None:
    from ..conftest import llm_models_root

    if torch.cuda.device_count() < 2:
        pytest.skip("DKV aggregate fault gate requires two GPUs")
    hook = tmp_path / "worker-hook"
    hook.mkdir()
    traces = tmp_path / "traces"
    traces.mkdir()
    (hook / "sitecustomize.py").write_text(
        "import os\n"
        "from dkv_fault_probe import install_import_hook\n"
        "install_import_hook(os.environ['DKV_FAULT_TRACE_DIR'])\n"
    )
    pythonpath = os.pathsep.join(
        [str(hook), str(Path(__file__).parent), os.environ.get("PYTHONPATH", "")]
    )
    monkeypatch.setenv("PYTHONPATH", pythonpath)
    monkeypatch.setenv("DKV_FAULT_TRACE_DIR", str(traces))
    monkeypatch.setenv("TRTLLM_DKV_DEBUG", "1")
    monkeypatch.delenv("TLLM_WORKER_USE_SINGLE_PROCESS", raising=False)
    run_aggregate_faults(
        str(Path(llm_models_root()) / "llama-models-v2/TinyLlama-1.1B-Chat-v1.0"), str(traces)
    )
