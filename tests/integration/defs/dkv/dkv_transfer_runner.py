# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Four-GPU DKV context to ordinary generation gate, also runnable outside pytest."""

import argparse
import copy
import json
import os
import pickle
import signal
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

if __package__:
    from .dkv_precision import (
        compare_precision_runs,
        validate_adp_control,
        validate_precision_inputs,
    )
else:
    from dkv_precision import (
        compare_precision_runs,
        validate_adp_control,
        validate_precision_inputs,
    )


def _wait(path: Path, timeout: float = 120) -> None:
    deadline = time.monotonic() + timeout
    while not path.exists():
        if time.monotonic() >= deadline:
            raise TimeoutError(f"DKV transfer gate did not produce {path.name}")
        time.sleep(0.01)


def _save_message(path: Path, message) -> None:
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        pickle.dump(message, stream)
    temporary.replace(path)


def _load_message(path: Path):
    with path.open("rb") as stream:
        return pickle.load(stream)


def _llm(options: dict, *, context: bool):
    from tensorrt_llm import LLM
    from tensorrt_llm.llmapi import KvCacheConfig
    from tensorrt_llm.llmapi.llm_args import (
        BlockReuseConfig,
        CacheTransceiverConfig,
        DkvConfig,
        MoeConfig,
    )

    kv_cache_config = KvCacheConfig(
        use_kv_cache_manager_v2=True,
        enable_block_reuse=False,
        enable_swa_scratch_reuse=False,
        block_reuse_config=BlockReuseConfig(policy="per_request"),
        free_gpu_memory_fraction=0.5,
        max_tokens=4096,
        host_cache_size=0,
        tokens_per_block=options["tokens_per_block"],
    )
    validate_precision_inputs(
        enable_block_reuse=kv_cache_config.enable_block_reuse,
        tokens_per_block=kv_cache_config.tokens_per_block,
    )
    return LLM(
        model=options["model"],
        tensor_parallel_size=2,
        moe_expert_parallel_size=2,
        enable_attention_dp=context,
        disable_overlap_scheduler=True,
        enable_chunked_prefill=False,
        enable_autotuner=False,
        cuda_graph_config=None,
        max_batch_size=2,
        max_num_tokens=512,
        max_seq_len=512,
        num_postprocess_workers=0,
        gather_generation_logits=True,
        dkv_config=DkvConfig() if context and options["dkv"] else None,
        moe_config=MoeConfig(backend=options["moe_backend"]),
        cache_transceiver_config=CacheTransceiverConfig(
            backend="NIXL",
            transceiver_runtime="PYTHON",
            kv_transfer_timeout_ms=options["transfer_timeout_ms"],
            kv_transfer_sender_future_timeout_ms=100,
        ),
        kv_cache_config=kv_cache_config,
    )


def _generation_worker(options: dict, directory: Path) -> None:
    import torch

    from tensorrt_llm import SamplingParams

    with _llm(options, context=False) as llm:
        (directory / "generation-ready").touch()
        index = 0
        deadline = time.monotonic() + 900
        while not (directory / "stop-generation").exists():
            message = directory / f"request-{index}.pkl"
            if not message.exists():
                if time.monotonic() > deadline:
                    raise TimeoutError("Context worker stopped submitting generation requests")
                time.sleep(0.01)
                continue
            prompt, params = _load_message(message)
            params.request_type = "generation_only"
            output = llm.generate_async(
                prompt,
                sampling_params=SamplingParams(
                    max_tokens=8,
                    temperature=0,
                    ignore_eos=True,
                    return_generation_logits=True,
                ),
                disaggregated_params=params,
            ).result(timeout=120)
            assert len(output.outputs) == 1
            result = output.outputs[0]
            assert len(result.token_ids) == 8, "Generation must consume KV and decode seven tokens"
            assert isinstance(result.generation_logits, torch.Tensor)
            values = result.generation_logits.detach().float().cpu().clone()
            assert values.ndim == 2 and values.shape[0] == 8 and values.shape[1] > 1000
            assert torch.isfinite(values).all()
            _save_message(directory / f"response-{index}.pkl", (list(result.token_ids), values))
            index += 1
            deadline = time.monotonic() + 900


def _context_worker(options: dict, directory: Path) -> None:
    import torch
    from dkv_transfer_probe import CANCEL_TOKEN

    from tensorrt_llm import DisaggregatedParams, SamplingParams
    from tensorrt_llm.scheduling_params import SchedulingParams

    traces = directory / "traces"
    sampling = SamplingParams(
        max_tokens=1, temperature=0, ignore_eos=True, return_generation_logits=True
    )
    outputs = {"tokens": [], "logits": []}
    context_ids = []
    generation_index = 0
    timeout_ids = []

    with _llm(options, context=True) as llm:
        _wait(directory / "generation-ready", timeout=600)

        def submit(prompt, rank):
            return llm.generate_async(
                prompt,
                sampling_params=sampling,
                scheduling_params=SchedulingParams(
                    attention_dp_rank=rank, attention_dp_relax=False
                ),
                disaggregated_params=DisaggregatedParams(request_type="context_only"),
            )

        def wait_for_free(request_id):
            if options["dkv"]:
                for rank in range(2):
                    _wait(traces / f"freed-{request_id}-{rank}")

        def complete(prompt, rank):
            nonlocal generation_index
            output = submit(prompt, rank).result(timeout=120)
            params = copy.copy(output.disaggregated_params)
            assert params is not None and params.ctx_request_id is not None
            context_ids.append(params.ctx_request_id)
            assert params.first_gen_logits, "Context must publish its first-token full logits"
            params.first_gen_logits = [value.cpu().clone() for value in params.first_gen_logits]
            _save_message(directory / f"request-{generation_index}.pkl", (prompt, params))
            response = directory / f"response-{generation_index}.pkl"
            _wait(response)
            tokens, logits = _load_message(response)
            assert tokens[0] == params.first_gen_tokens[0]
            outputs["tokens"].append(tokens)
            outputs["logits"].append(logits)
            wait_for_free(params.ctx_request_id)
            generation_index += 1

        for index in range(4):
            complete([1] + [42 + index] * 255, index % 2)

        if options["dkv"]:
            # The context reply is deliberately delivered without forwarding its KV metadata.
            timed_out = submit([1] + [29002] * 255, 0).result(timeout=120)
            timeout_id = timed_out.disaggregated_params.ctx_request_id
            context_ids.append(timeout_id)
            timeout_ids.append(timeout_id)
            wait_for_free(timeout_id)

            pending = submit([1] + [CANCEL_TOKEN] * 255, 1)
            try:
                _wait(traces / "race-response-ready")
                pending.abort()
                _wait(traces / "cancel-queued")
            finally:
                (traces / "release-race-response").touch()
            cancelled = pending.result(timeout=120)
            cancel_id = cancelled.disaggregated_params.ctx_request_id
            assert cancel_id == int((traces / "cancel-queued").read_text())
            context_ids.append(cancel_id)
            timeout_ids.append(cancel_id)
            wait_for_free(cancel_id)

        for rank in range(2):
            complete([1] + [56 + rank] * 255, rank)

    torch.save(outputs, directory / "outputs.pt")
    (directory / "context-result.json").write_text(
        json.dumps(
            {"context_ids": context_ids, "timeout_ids": timeout_ids, "generated": generation_index},
            indent=2,
        )
    )
    (directory / "stop-generation").touch()


def validate_transfer_traces(directory: Path) -> dict:
    """Require one physical sender, two lifecycle replicas, and synchronized leak-free cleanup."""
    expected = json.loads((directory / "context-result.json").read_text())
    options = json.loads((directory / "options.json").read_text())
    events = [
        json.loads(line)
        for path in sorted((directory / "traces").glob("rank-*.jsonl"))
        for line in path.read_text().splitlines()
        if line
    ]
    by_kind = {
        kind: [row for row in events if row["event"] == kind]
        for kind in {
            "baseline",
            "release_index",
            "start",
            "send",
            "observed",
            "commit",
            "free",
            "response",
            "cancel_attempt",
        }
    }
    baselines = by_kind["baseline"]
    assert len(baselines) == 2 and {row["rank"] for row in baselines} == {0, 1}
    assert baselines[0]["free_pages"] == baselines[1]["free_pages"]
    baseline = baselines[0]
    expected_ids = set(expected["context_ids"])
    assert len(expected_ids) == 8
    for kind in ("release_index", "start", "commit", "free"):
        for rank in range(2):
            counts = Counter(row["request_id"] for row in by_kind[kind] if row["rank"] == rank)
            assert counts == Counter({request_id: 1 for request_id in expected_ids}), (
                kind,
                rank,
                counts,
            )
    for kind in ("send", "observed", "response"):
        counts = Counter(row["request_id"] for row in by_kind[kind])
        assert counts == Counter({request_id: 1 for request_id in expected_ids}), (kind, counts)
    owners = {row["request_id"]: row["owner"] for row in by_kind["start"]}
    assert all(row["error"] is None for row in by_kind["response"])
    for kind in ("send", "observed", "response"):
        assert all(row["rank"] == owners[row["request_id"]] for row in by_kind[kind]), kind
    for request_id in expected_ids:
        observed = next(row for row in by_kind["observed"] if row["request_id"] == request_id)
        commits = [row for row in by_kind["commit"] if row["request_id"] == request_id]
        frees = [row for row in by_kind["free"] if row["request_id"] == request_id]
        assert len({(row["control"], row["iteration"], row["outcome"]) for row in commits}) == 1
        assert len({(row["control"], row["iteration"]) for row in frees}) == 1
        expected_control = observed["control"] + (0 if observed["in_control"] else 1)
        assert commits[0]["control"] == expected_control
        assert all(row["control"] == expected_control and row["in_control"] for row in frees)
        assert all(row["iteration"] == commits[0]["iteration"] for row in frees)
        assert all(row["window"] == "control" for row in frees)
        assert all(row["free_pages"] == baseline["free_pages"] for row in frees)
        assert all(row["index_used"] == baseline["index_used"] for row in frees)
        outcome = "timed_out" if request_id in expected["timeout_ids"] else "completed"
        assert observed["outcome"] == outcome
        assert all(row["outcome"] == outcome for row in commits)
        if outcome == "timed_out":
            assert observed["elapsed_ms"] >= options["transfer_timeout_ms"]
        for rank in range(2):
            local = [
                row for row in events if row.get("request_id") == request_id and row["rank"] == rank
            ]
            positions = {row["event"]: index for index, row in enumerate(local)}
            assert (
                positions["release_index"]
                < positions["start"]
                < positions["commit"]
                < positions["free"]
            )
            if rank == owners[request_id]:
                assert (
                    positions["start"]
                    < positions["send"]
                    < positions["observed"]
                    < positions["commit"]
                )
            for row in local:
                if row["event"] in ("release_index", "start"):
                    assert row["index_used"] == baseline["index_used"]
                    current = [page for level in row["free_pages"] for page in level]
                    total = [page for level in baseline["free_pages"] for page in level]
                    assert all(
                        free <= initial for free, initial in zip(current, total, strict=True)
                    )
                    assert any(free < initial for free, initial in zip(current, total, strict=True))
    cancel_id = expected["timeout_ids"][-1]
    attempts = [row for row in by_kind["cancel_attempt"] if row["request_id"] == cancel_id]
    assert len(attempts) == 2 and {row["rank"] for row in attempts} == {0, 1}
    assert all(not row["accepted"] for row in attempts), "In-flight public abort freed a KV replica"
    return {
        "status": "passed",
        "context_requests": len(expected_ids),
        "generation_requests": expected["generated"],
        "unique_raw_responses": len(by_kind["response"]),
        "timeout_ids": expected["timeout_ids"],
        "cancel_request_id": cancel_id,
        "baseline_free_pages": baseline["free_pages"],
        "baseline_index_used": baseline["index_used"],
        "replicated_frees": len(by_kind["free"]),
        "tail_observations": sum(not row["in_control"] for row in by_kind["observed"]),
        "control_observations": sum(row["in_control"] for row in by_kind["observed"]),
    }


def _run_pair(options: dict, directory: Path) -> None:
    directory.mkdir(parents=True)
    options_path = directory / "options.json"
    options_path.write_text(json.dumps(options))
    hook = directory / "hook"
    hook.mkdir()
    (hook / "sitecustomize.py").write_text(
        "import os\n"
        "if os.environ.get('DKV_TRANSFER_TRACE_DIR'):\n"
        "    from dkv_transfer_probe import install_import_hook\n"
        "    install_import_hook(os.environ['DKV_TRANSFER_TRACE_DIR'])\n"
    )
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0,1,2,3").split(",")
    if len(visible) < 4:
        raise RuntimeError("DKV context TP2 + generation TP2 requires four visible GPUs")
    processes = []
    logs = []
    try:
        for role, devices in (("generation", visible[2:4]), ("context", visible[:2])):
            env = {
                key: value
                for key, value in os.environ.items()
                if not key.startswith(("SLURM_", "PMI_", "PMIX_", "OMPI_"))
            }
            env["CUDA_VISIBLE_DEVICES"] = ",".join(devices)
            env["OMPI_ALLOW_RUN_AS_ROOT"] = "1"
            env["OMPI_ALLOW_RUN_AS_ROOT_CONFIRM"] = "1"
            env["PYTHONDONTWRITEBYTECODE"] = "1"
            env["PYTHONPATH"] = os.pathsep.join(
                [str(hook), str(Path(__file__).parent), env.get("PYTHONPATH", "")]
            )
            env["TRTLLM_DKV_DEBUG"] = "1"
            env["UCX_MM_ERROR_HANDLING"] = "y"
            env["UCX_TLS"] = "cuda_copy,cuda_ipc,sm,self,tcp"
            env.pop("TLLM_WORKER_USE_SINGLE_PROCESS", None)
            env.pop("DKV_TRANSFER_TRACE_DIR", None)
            if role == "context" and options["dkv"]:
                env["DKV_TRANSFER_TRACE_DIR"] = str(directory / "traces")
            log = (directory / f"{role}.log").open("w")
            logs.append(log)
            processes.append(
                subprocess.Popen(
                    [
                        sys.executable,
                        str(Path(__file__).resolve()),
                        "--role",
                        role,
                        "--options",
                        str(options_path),
                        "--work-dir",
                        str(directory),
                    ],
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            )
        deadline = time.monotonic() + 900
        while any(process.poll() is None for process in processes):
            failed = [
                process.returncode for process in processes if process.poll() not in (None, 0)
            ]
            if failed:
                raise RuntimeError(f"DKV transfer worker failed ({failed}); logs: {directory}")
            if time.monotonic() > deadline:
                raise TimeoutError(f"DKV transfer pair exceeded 900 seconds; logs: {directory}")
            time.sleep(0.2)
        assert all(process.returncode == 0 for process in processes)
    finally:
        for process in processes:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        for process in processes:
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=20)
            finally:
                # A failed frontend can leave MPI children in its original process group.
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        for log in logs:
            log.close()


def run_transfer_gate(
    model: str,
    work_dir: str,
    *,
    mode: str = "precision",
    moe_backend: str = "CUTLASS",
    tokens_per_block: int = 32,
    transfer_timeout_ms: int = 15000,
) -> dict:
    """Run real NIXL transfers; precision requires an independently repeated ADP control."""
    import torch

    directory = Path(work_dir).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    options = {
        "model": model,
        "moe_backend": moe_backend,
        "tokens_per_block": tokens_per_block,
        "transfer_timeout_ms": transfer_timeout_ms,
    }
    runs = [("adp-control", False), ("adp-replay", False)] if mode == "precision" else []
    runs.append(("dkv", True))
    for name, enabled in runs:
        if enabled and mode == "precision":
            validate_adp_control(
                torch.load(directory / "adp-control" / "outputs.pt", weights_only=True),
                torch.load(directory / "adp-replay" / "outputs.pt", weights_only=True),
            )
        _run_pair({**options, "dkv": enabled}, directory / name)
    summary = {"lifecycle": validate_transfer_traces(directory / "dkv"), "precision": "not_run"}
    if mode == "precision":
        control, replay, actual = [
            torch.load(directory / name / "outputs.pt", weights_only=True) for name, _ in runs
        ]
        assert len(control["logits"]) == len(replay["logits"]) == len(actual["logits"]) == 6
        comparison = compare_precision_runs(control, replay, actual)
        summary["precision"] = {
            **comparison,
            "status": "passed",
            "tokens_per_request": 8,
            "bitwise": comparison["bitwise_adp_control"],
        }
    (directory / "result.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model")
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--mode", choices=("precision", "lifecycle"), default="precision")
    parser.add_argument("--moe-backend", default="CUTLASS")
    parser.add_argument("--tokens-per-block", type=int, default=32)
    parser.add_argument("--transfer-timeout-ms", type=int, default=15000)
    parser.add_argument("--role", choices=("context", "generation"))
    parser.add_argument("--options")
    args = parser.parse_args()
    if args.role:
        options = json.loads(Path(args.options).read_text())
        worker = _context_worker if args.role == "context" else _generation_worker
        worker(options, Path(args.work_dir))
    else:
        if not args.model:
            parser.error("--model is required for a gate run")
        print(
            json.dumps(
                run_transfer_gate(
                    args.model,
                    args.work_dir,
                    mode=args.mode,
                    moe_backend=args.moe_backend,
                    tokens_per_block=args.tokens_per_block,
                    transfer_timeout_ms=args.transfer_timeout_ms,
                ),
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
