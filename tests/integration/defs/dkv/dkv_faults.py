# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Aggregate cancellation and recoverable rank-local sampler errors across the DKV group.

The scenario watches the executors through a probe that a sitecustomize hook installs in every
worker. A process spawns MPI workers with its environment only for the first LLM it builds, so the
scenario runs in a fresh process of its own (``run_fault_gate_in_fresh_process``), also runnable as a
script.
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import pytest

from tensorrt_llm import SamplingParams
from tensorrt_llm._torch.pyexecutor.llm_request import FinishReason
from tensorrt_llm.scheduling_params import SchedulingParams

if __package__:
    from .dkv_fault_probe import FAULT_MESSAGE, FAULT_TOKEN
    from .dkv_models import MODELS, DkvModel, dkv_worker_env, make_dkv_llm
else:
    from dkv_fault_probe import FAULT_MESSAGE, FAULT_TOKEN
    from dkv_models import MODELS, DkvModel, dkv_worker_env, make_dkv_llm

# Building and tearing down a large model's workers takes most of this.
_RUN_TIMEOUT_S = 1800


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
    model: DkvModel, trace_dir: str, group_size: int = 2, model_path: str | None = None
) -> dict:
    """Require actual cancellation, one owner error, symmetric frees and subsequent inference.

    ``model_path`` is the checkpoint to load; it defaults to the model's own location.
    """
    traces = Path(trace_dir)
    sampling = SamplingParams(max_tokens=1, temperature=0, ignore_eos=True)
    pinned = SchedulingParams(attention_dp_rank=1, attention_dp_relax=False)
    # Four requests after the fault; a group of more than four ranks serves one on every rank.
    recovery_ranks = tuple(range(group_size)) * (4 // group_size or 1)
    with make_dkv_llm(
        model_path or model.path(),
        dkv=True,
        tokens_per_block=model.tokens_per_block,
        moe_config=model.moe_config(),
        group_size=group_size,
        env_overrides=dkv_worker_env(),
    ) as llm:
        try:
            for rank in range(group_size):
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
        for rank in range(group_size):
            _wait(traces / f"freed-{cancel_id}-{rank}")
        assert len(blocker.result(timeout=60).outputs[0].token_ids) == 1

        failed = llm.generate_async(
            [1] + [FAULT_TOKEN] * 255, sampling_params=sampling, scheduling_params=pinned
        )
        with pytest.raises(RuntimeError, match=FAULT_MESSAGE):
            failed.result(timeout=60)
        for rank in recovery_ranks:
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
    assert {event["rank"] for event in baselines} == set(range(group_size))
    baseline = baselines[0]["free_blocks"]
    assert all(event["free_blocks"] == baseline for event in baselines)
    injected = [event for event in events if event["event"] == "injected"]
    assert len(injected) == 1 and injected[0]["rank"] == 1
    assert len(injected[0]["request_ids"]) == 1
    fault_id = injected[0]["request_ids"][0]
    cancellations = [event for event in events if event["event"] == "cancel_applied"]
    assert len(cancellations) == group_size
    assert {event["rank"] for event in cancellations} == set(range(group_size))
    assert all(event["accepted"] for event in cancellations)
    cancel_id = cancellations[0]["request_id"]
    assert {event["request_id"] for event in cancellations} == {cancel_id}

    responses = [event for event in events if event["event"] == "response"]
    expected_responses = 3 + len(recovery_ranks)
    assert len(responses) == expected_responses, (
        f"Expected blocker, cancellation, fault and {len(recovery_ranks)} recovery replies: "
        f"{responses}"
    )
    assert all(count == 1 for count in Counter(event["request_id"] for event in responses).values())
    errors = [event for event in responses if event["error"] is not None]
    assert len(errors) == 1
    assert errors[0]["request_id"] == fault_id and errors[0]["rank"] == 1
    assert FAULT_MESSAGE in errors[0]["error"]

    frees = [event for event in events if event["event"] == "free"]
    freed_by_rank = [
        [(event["request_id"], event["iteration"]) for event in frees if event["rank"] == rank]
        for rank in range(group_size)
    ]
    assert len(freed_by_rank[0]) == expected_responses
    assert all(freed == freed_by_rank[0] for freed in freed_by_rank), (
        "Ranks freed different requests or iterations"
    )
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
        "recovery_requests": len(recovery_ranks),
        "baseline_free_blocks": baseline,
        "free_sequence_by_rank": freed_by_rank,
    }


def run_fault_gate_in_fresh_process(model: DkvModel, work_dir: Path, group_size: int) -> dict:
    """Run ``run_aggregate_faults`` in a new process whose workers load the probe hook.

    Returns the scenario's result; a failure raises with the location of the logs.
    """
    work_dir.mkdir(parents=True)
    hook = work_dir / "hook"
    hook.mkdir()
    traces = work_dir / "traces"
    traces.mkdir()
    (hook / "sitecustomize.py").write_text(
        "import os\n"
        "from dkv_fault_probe import install_import_hook\n"
        "install_import_hook(os.environ['DKV_FAULT_TRACE_DIR'])\n"
    )
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("SLURM_", "PMI_", "PMIX_", "OMPI_"))
    }
    env.pop("TLLM_WORKER_USE_SINGLE_PROCESS", None)
    env.update(
        {
            "OMPI_ALLOW_RUN_AS_ROOT": "1",
            "OMPI_ALLOW_RUN_AS_ROOT_CONFIRM": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": os.pathsep.join(
                [str(hook), str(Path(__file__).parent), os.environ.get("PYTHONPATH", "")]
            ),
            "DKV_FAULT_TRACE_DIR": str(traces),
            **dkv_worker_env(),
        }
    )
    result_path = work_dir / "result.json"
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--model-id",
        model.pytest_id,
        "--model-path",
        model.path(),
        "--group-size",
        str(group_size),
        "--trace-dir",
        str(traces),
        "--result",
        str(result_path),
    ]
    with (work_dir / "run.log").open("w") as log:
        process = subprocess.Popen(
            command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
        try:
            process.wait(timeout=_RUN_TIMEOUT_S)
        finally:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    if process.returncode != 0 or not result_path.exists():
        raise RuntimeError(f"The DKV fault gate failed ({process.returncode}); logs: {work_dir}")
    return json.loads(result_path.read_text())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", required=True, choices=[model.pytest_id for model in MODELS])
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--group-size", type=int, default=2)
    parser.add_argument("--trace-dir", required=True)
    parser.add_argument("--result", required=True)
    args = parser.parse_args()
    model = next(model for model in MODELS if model.pytest_id == args.model_id)
    result = run_aggregate_faults(model, args.trace_dir, args.group_size, args.model_path)
    Path(args.result).write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
