# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DKV context to ordinary generation gate, also runnable outside pytest.

By default a context group and a generation group of two ranks each share one four-GPU node and
the runner starts both. Larger groups, or groups on different nodes, are started by the job: it
runs ``--prepare-phase`` once per phase (``adp-control``, ``adp-replay``, ``dkv``, or a mutated
``dkv``), starts every rank of each group with ``--role`` after exporting what ``--role-env``
prints, and judges the finished phases with ``--judge`` or ``--judge-mutation``.
"""

import argparse
import copy
import json
import os
import pickle
import shlex
import signal
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

if __package__:
    from .dkv_models import make_dkv_llm
    from .dkv_precision import (
        EXACT_POLICY,
        NOISY_POLICY,
        PrecisionPolicy,
        compare_precision_runs,
        failing_requests,
        validate_adp_control,
        validate_precision_inputs,
    )
else:
    from dkv_models import make_dkv_llm
    from dkv_precision import (
        EXACT_POLICY,
        NOISY_POLICY,
        PrecisionPolicy,
        compare_precision_runs,
        failing_requests,
        validate_adp_control,
        validate_precision_inputs,
    )

# The two requests that follow the timeout and cancellation injections prove the pool still works.
_TAIL_REQUESTS = 2
# Loading a large model twice, once per worker pair, takes most of this.
_PAIR_TIMEOUT_S = 1800


def _repeated_token_prompts() -> list[list[int]]:
    """Four requests before the injections and two after them, each of one repeated token."""
    return [[1] + [42 + index] * 255 for index in range(4)] + [
        [1] + [56 + rank] * 255 for rank in range(_TAIL_REQUESTS)
    ]


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


def _save_text(path: Path, text: str) -> None:
    """Write a file so that a process polling for it never sees it half written."""
    temporary = path.with_suffix(".tmp")
    temporary.write_text(text)
    temporary.replace(path)


def _load_message(path: Path):
    with path.open("rb") as stream:
        return pickle.load(stream)


def _group_size(options: dict, *, context: bool) -> int:
    """Ranks of the context (attention-DP) group or of the generation group."""
    return options.get("ctx_group_size", 2) if context else options.get("gen_group_size", 2)


def _llm(options: dict, *, context: bool):
    from tensorrt_llm.llmapi.llm_args import CacheTransceiverConfig, MoeConfig

    validate_precision_inputs(
        enable_block_reuse=False, tokens_per_block=options["tokens_per_block"]
    )
    return make_dkv_llm(
        options["model"],
        dkv=context and options["dkv"],
        tokens_per_block=options["tokens_per_block"],
        moe_config=MoeConfig(
            backend=options["moe_backend"],
            disable_finalize_fusion=options["disable_finalize_fusion"],
        ),
        group_size=_group_size(options, context=context),
        attention_dp=context,
        gather_logits=True,
        enable_iter_perf_stats=False,
        cache_transceiver_config=CacheTransceiverConfig(
            backend="NIXL",
            transceiver_runtime="PYTHON",
            kv_transfer_timeout_ms=options["transfer_timeout_ms"],
            kv_transfer_sender_future_timeout_ms=100,
        ),
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

    prompts = options["prompts"] or _repeated_token_prompts()
    group_size = _group_size(options, context=True)
    with _llm(options, context=True) as llm:
        _wait(directory / "generation-ready", timeout=1200)

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
                for rank in range(group_size):
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

        for index, prompt in enumerate(prompts[:-_TAIL_REQUESTS]):
            complete(prompt, index % group_size)

        if options["dkv"] and options.get("injections", True):
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

        for rank, prompt in enumerate(prompts[-_TAIL_REQUESTS:]):
            complete(prompt, rank)

    torch.save(outputs, directory / "outputs.pt")
    (directory / "context-result.json").write_text(
        json.dumps(
            {"context_ids": context_ids, "timeout_ids": timeout_ids, "generated": generation_index},
            indent=2,
        )
    )
    (directory / "stop-generation").touch()


def validate_transfer_traces(directory: Path) -> dict:
    """Require one physical sender, a lifecycle replica per rank, and synchronized leak-free cleanup."""
    expected = json.loads((directory / "context-result.json").read_text())
    options = json.loads((directory / "options.json").read_text())
    group_size = _group_size(options, context=True)
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
    assert len(baselines) == group_size
    assert {row["rank"] for row in baselines} == set(range(group_size))
    assert all(row["free_pages"] == baselines[0]["free_pages"] for row in baselines)
    baseline = baselines[0]
    expected_ids = set(expected["context_ids"])
    # Every prompt completes through generation; the timeout and the cancellation add one each.
    prompt_count = len(options["prompts"] or _repeated_token_prompts())
    assert expected["generated"] == prompt_count
    assert len(expected["timeout_ids"]) == 2
    assert len(expected_ids) == prompt_count + len(expected["timeout_ids"])
    for kind in ("release_index", "start", "commit", "free"):
        for rank in range(group_size):
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
        for rank in range(group_size):
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
    assert len(attempts) == group_size
    assert {row["rank"] for row in attempts} == set(range(group_size))
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


def _prepare_pair(options: dict, directory: Path) -> None:
    """Write the options of one run and the sitecustomize hook that its context workers load.

    The options come last: other processes of a multi-node run wait for them.
    """
    directory.mkdir(parents=True)
    hook = directory / "hook"
    hook.mkdir()
    (hook / "sitecustomize.py").write_text(
        "import os\n"
        "if os.environ.get('DKV_TRANSFER_TRACE_DIR'):\n"
        "    from dkv_transfer_probe import install_import_hook\n"
        "    install_import_hook(os.environ['DKV_TRANSFER_TRACE_DIR'])\n"
        "if os.environ.get('DKV_KV_MUTATION'):\n"
        "    from dkv_mutation_probe import install_mutation_hook\n"
        "    install_mutation_hook(os.environ['DKV_KV_MUTATION'])\n"
    )
    _save_text(directory / "options.json", json.dumps(options))


def _role_env(role: str, options: dict, directory: Path) -> dict[str, str]:
    """The variables a role's processes need on top of the inherited environment."""
    env = {
        "OMPI_ALLOW_RUN_AS_ROOT": "1",
        "OMPI_ALLOW_RUN_AS_ROOT_CONFIRM": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": os.pathsep.join(
            [str(directory / "hook"), str(Path(__file__).parent), os.environ.get("PYTHONPATH", "")]
        ),
        "TRTLLM_DKV_DEBUG": "1",
    }
    if role == "context" and options["dkv"]:
        env["DKV_TRANSFER_TRACE_DIR"] = str(directory / "traces")
        if options.get("mutation"):
            env["DKV_KV_MUTATION"] = json.dumps(options["mutation"])
    return env


def _run_pair(options: dict, directory: Path) -> None:
    _prepare_pair(options, directory)
    options_path = directory / "options.json"
    ctx_size = _group_size(options, context=True)
    gen_size = _group_size(options, context=False)
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0,1,2,3").split(",")
    if len(visible) < ctx_size + gen_size:
        raise RuntimeError(
            f"DKV context TP{ctx_size} + generation TP{gen_size} requires "
            f"{ctx_size + gen_size} visible GPUs"
        )
    processes = []
    logs = []
    try:
        for role, devices in (
            ("generation", visible[ctx_size : ctx_size + gen_size]),
            ("context", visible[:ctx_size]),
        ):
            env = {
                key: value
                for key, value in os.environ.items()
                if not key.startswith(("SLURM_", "PMI_", "PMIX_", "OMPI_"))
            }
            env["CUDA_VISIBLE_DEVICES"] = ",".join(devices)
            env["UCX_MM_ERROR_HANDLING"] = "y"
            env["UCX_TLS"] = "cuda_copy,cuda_ipc,sm,self,tcp"
            env.pop("TLLM_WORKER_USE_SINGLE_PROCESS", None)
            env.pop("DKV_TRANSFER_TRACE_DIR", None)
            env.pop("DKV_KV_MUTATION", None)
            env.update(_role_env(role, options, directory))
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
        deadline = time.monotonic() + _PAIR_TIMEOUT_S
        while any(process.poll() is None for process in processes):
            failed = [
                process.returncode for process in processes if process.poll() not in (None, 0)
            ]
            if failed:
                raise RuntimeError(f"DKV transfer worker failed ({failed}); logs: {directory}")
            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"DKV transfer pair exceeded {_PAIR_TIMEOUT_S} seconds; logs: {directory}"
                )
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


def _transfer_options(
    model: str,
    *,
    moe_backend: str,
    disable_finalize_fusion: bool,
    tokens_per_block: int,
    transfer_timeout_ms: int,
    prompts: list[list[int]] | None,
    ctx_group_size: int = 2,
    gen_group_size: int = 2,
) -> dict:
    return {
        "model": model,
        "moe_backend": moe_backend,
        "disable_finalize_fusion": disable_finalize_fusion,
        "tokens_per_block": tokens_per_block,
        "transfer_timeout_ms": transfer_timeout_ms,
        "prompts": prompts,
        "ctx_group_size": ctx_group_size,
        "gen_group_size": gen_group_size,
    }


def _load_outputs(directory: Path, name: str) -> dict:
    import torch

    return torch.load(directory / name / "outputs.pt", weights_only=True)


def judge_transfer_gate(directory: Path, *, mode: str, policy: PrecisionPolicy) -> dict:
    """Judge the finished runs of a work directory.

    The lifecycle traces are checked and, for precision, DKV is compared with the two ADP runs.
    """
    summary = {"lifecycle": validate_transfer_traces(directory / "dkv"), "precision": "not_run"}
    if mode == "precision":
        control, replay, actual = [
            _load_outputs(directory, name) for name in ("adp-control", "adp-replay", "dkv")
        ]
        options = json.loads((directory / "dkv" / "options.json").read_text())
        expected = len(options["prompts"] or _repeated_token_prompts())
        assert len(control["logits"]) == len(replay["logits"]) == len(actual["logits"]) == expected
        comparison = compare_precision_runs(control, replay, actual, policy)
        summary["precision"] = {
            **comparison,
            "status": "passed",
            "tokens_per_request": 8,
            "bitwise": comparison.get("bitwise_adp_control", False),
        }
    (directory / "result.json").write_text(json.dumps(summary, indent=2))
    return summary


def run_transfer_gate(
    model: str,
    work_dir: str,
    *,
    mode: str = "precision",
    moe_backend: str = "CUTLASS",
    disable_finalize_fusion: bool = False,
    tokens_per_block: int = 32,
    transfer_timeout_ms: int = 15000,
    policy: PrecisionPolicy = EXACT_POLICY,
    prompts: list[list[int]] | None = None,
    ctx_group_size: int = 2,
    gen_group_size: int = 2,
) -> dict:
    """Run real NIXL transfers; precision requires an independently repeated ADP control.

    ``prompts`` replaces the default repeated-token requests; the last two run after the timeout
    and cancellation injections. ``policy`` decides how closely DKV must match the ADP control.
    """
    directory = Path(work_dir).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    options = _transfer_options(
        model,
        moe_backend=moe_backend,
        disable_finalize_fusion=disable_finalize_fusion,
        tokens_per_block=tokens_per_block,
        transfer_timeout_ms=transfer_timeout_ms,
        prompts=prompts,
        ctx_group_size=ctx_group_size,
        gen_group_size=gen_group_size,
    )
    runs = [("adp-control", False), ("adp-replay", False)] if mode == "precision" else []
    runs.append(("dkv", True))
    for name, enabled in runs:
        if enabled and mode == "precision":
            validate_adp_control(
                _load_outputs(directory, "adp-control"),
                _load_outputs(directory, "adp-replay"),
                policy,
            )
        _run_pair({**options, "dkv": enabled}, directory / name)
    return judge_transfer_gate(directory, mode=mode, policy=policy)


def _mutation_spec(
    prompts: list[list[int]], target_index: int, spec: dict, record_dir: Path
) -> dict:
    """The complete spec for ``dkv_mutation_probe``: ``spec`` aimed at one prompt by its length."""
    lengths = [len(prompt) for prompt in prompts]
    if lengths.count(lengths[target_index]) != 1:
        raise ValueError("The corrupted prompt must be the only one of its length")
    return {**spec, "prompt_len": lengths[target_index], "record_dir": str(record_dir)}


def _mutation_records(run_directory: Path) -> list[dict]:
    return [
        json.loads(line)
        for path in sorted((run_directory / "mutations").glob("mutations-*.jsonl"))
        for line in path.read_text().splitlines()
        if line
    ]


def judge_mutation(directory: Path, name: str, policy: PrecisionPolicy) -> dict:
    """What the precision gate says about the mutated run ``name`` of a work directory.

    ``gate_error`` is the gate's failure message, or None when the gate accepted the run;
    ``failing_requests`` lists the requests that break the policy on their own; ``records`` are the
    corruptions the context workers applied.
    """
    control = _load_outputs(directory, "adp-control")
    replay = _load_outputs(directory, "adp-replay")
    mutated = _load_outputs(directory, name)
    try:
        compare_precision_runs(control, replay, mutated, policy)
    except AssertionError as error:
        gate_error = str(error)
    else:
        gate_error = None
    return {
        "gate_error": gate_error,
        "failing_requests": failing_requests(control, replay, mutated, policy),
        "records": _mutation_records(directory / name),
    }


def run_transfer_mutations(
    model: str,
    work_dir: str,
    mutations: dict[str, dict],
    *,
    prompts: list[list[int]],
    target_index: int,
    moe_backend: str = "CUTLASS",
    disable_finalize_fusion: bool = False,
    tokens_per_block: int = 32,
    transfer_timeout_ms: int = 30000,
    policy: PrecisionPolicy = EXACT_POLICY,
) -> dict[str, dict]:
    """Corrupt the KV of one prompt on its way to generation and judge what the gate makes of it.

    The ADP control runs twice, then each entry of ``mutations`` (run name -> ``kind`` and optional
    ``fraction`` for ``dkv_mutation_probe``) runs the DKV context worker with the KV of
    ``prompts[target_index]`` corrupted. The mutated runs skip the timeout and cancellation
    injections, because they test the precision gate and not the lifecycle. Returns ``judge_mutation``
    for every run name.
    """
    directory = Path(work_dir).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    options = {
        **_transfer_options(
            model,
            moe_backend=moe_backend,
            disable_finalize_fusion=disable_finalize_fusion,
            tokens_per_block=tokens_per_block,
            transfer_timeout_ms=transfer_timeout_ms,
            prompts=prompts,
        ),
        "injections": False,
    }
    for name in ("adp-control", "adp-replay"):
        _run_pair({**options, "dkv": False}, directory / name)
    validate_adp_control(
        _load_outputs(directory, "adp-control"), _load_outputs(directory, "adp-replay"), policy
    )
    judgments = {}
    for name, spec in mutations.items():
        mutation = _mutation_spec(prompts, target_index, spec, directory / name / "mutations")
        _run_pair({**options, "dkv": True, "mutation": mutation}, directory / name)
        judgments[name] = judge_mutation(directory, name, policy)
    (directory / "mutation-result.json").write_text(
        json.dumps({"target_index": target_index, "runs": judgments}, indent=2)
    )
    return judgments


def _natural_prompts(model: str, tokens_per_block: int) -> list[list[int]]:
    if __package__:
        from .dkv_models import PromptTokenizer
        from .dkv_precision import build_precision_prompts
    else:
        from dkv_models import PromptTokenizer
        from dkv_precision import build_precision_prompts

    prompts, _ = build_precision_prompts(PromptTokenizer(model).encode, tokens_per_block)
    return prompts


def _parse_mutations(items: list[str]) -> dict[str, dict]:
    """Run name -> spec for ``KIND[:FRACTION]`` command-line items."""
    mutations = {}
    for item in items:
        kind, _, fraction = item.partition(":")
        mutations[item.replace(":", "-")] = {
            "kind": kind,
            "fraction": float(fraction) if fraction else 1.0,
        }
    return mutations


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model")
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--mode", choices=("precision", "lifecycle"), default="precision")
    parser.add_argument("--moe-backend", default="CUTLASS")
    parser.add_argument("--disable-finalize-fusion", action="store_true")
    parser.add_argument("--tokens-per-block", type=int, default=32)
    parser.add_argument("--transfer-timeout-ms", type=int, default=15000)
    parser.add_argument("--ctx-group-size", type=int, default=2, help="ranks of the context group")
    parser.add_argument(
        "--gen-group-size", type=int, default=2, help="ranks of the generation group"
    )
    parser.add_argument(
        "--policy",
        choices=("exact", "noisy"),
        default="exact",
        help="noisy judges DKV against the run-to-run noise of the ADP control",
    )
    parser.add_argument(
        "--natural-prompts",
        action="store_true",
        help="use the discriminating text prompts instead of repeated tokens",
    )
    parser.add_argument(
        "--mutate",
        action="append",
        metavar="KIND[:FRACTION]",
        help="corrupt the KV of one prompt (zero or foreign, optionally only a leading fraction of "
        "its regions) and report what the gate makes of it; repeat for several runs. Needs "
        "--natural-prompts",
    )
    parser.add_argument(
        "--mutation-target",
        type=int,
        default=3,
        help="index of the prompt whose KV --mutate corrupts",
    )
    # The context and generation groups of a multi-node run are started by the job, one launch per
    # group and phase. WORK_DIR is the run's directory for --prepare-phase, --judge and
    # --judge-mutation, and the phase's own directory for --role and --role-env.
    parser.add_argument(
        "--prepare-phase",
        metavar="NAME",
        help="write the options and the hook of the phase WORK_DIR/NAME and exit",
    )
    parser.add_argument("--dkv", action="store_true", help="--prepare-phase: a DKV context group")
    parser.add_argument(
        "--role-env",
        choices=("context", "generation"),
        help="print the export lines that role needs in the phase directory WORK_DIR and exit",
    )
    parser.add_argument(
        "--judge", action="store_true", help="judge the finished phases in WORK_DIR and exit"
    )
    parser.add_argument(
        "--judge-mutation",
        action="append",
        metavar="NAME",
        help="judge the finished mutated phase WORK_DIR/NAME and exit",
    )
    parser.add_argument("--role", choices=("context", "generation"))
    parser.add_argument("--options")
    args = parser.parse_args()
    work_dir = Path(args.work_dir)
    policy = NOISY_POLICY if args.policy == "noisy" else EXACT_POLICY
    if args.role:
        worker = _context_worker if args.role == "context" else _generation_worker
        worker(json.loads(Path(args.options).read_text()), work_dir)
        return
    if args.role_env:
        options = json.loads((work_dir / "options.json").read_text())
        for key, value in _role_env(args.role_env, options, work_dir).items():
            print(f"export {key}={shlex.quote(value)}")
        return
    if args.judge:
        print(json.dumps(judge_transfer_gate(work_dir, mode=args.mode, policy=policy), indent=2))
        return
    if args.judge_mutation:
        judgments = {name: judge_mutation(work_dir, name, policy) for name in args.judge_mutation}
        print(json.dumps(judgments, indent=2))
        return
    if not args.model:
        parser.error("--model is required for a gate run")
    prompts = _natural_prompts(args.model, args.tokens_per_block) if args.natural_prompts else None
    if args.mutate and prompts is None:
        parser.error("--mutate needs --natural-prompts")
    if args.prepare_phase:
        options = {
            **_transfer_options(
                args.model,
                moe_backend=args.moe_backend,
                disable_finalize_fusion=args.disable_finalize_fusion,
                tokens_per_block=args.tokens_per_block,
                transfer_timeout_ms=args.transfer_timeout_ms,
                prompts=prompts,
                ctx_group_size=args.ctx_group_size,
                gen_group_size=args.gen_group_size,
            ),
            "dkv": args.dkv,
        }
        if args.mutate:
            if len(args.mutate) != 1:
                parser.error("a prepared phase takes one --mutate")
            (spec,) = _parse_mutations(args.mutate).values()
            options["mutation"] = _mutation_spec(
                prompts, args.mutation_target, spec, work_dir / args.prepare_phase / "mutations"
            )
            options["injections"] = False
        _prepare_pair(options, work_dir / args.prepare_phase)
        return
    if args.mutate:
        judgments = run_transfer_mutations(
            args.model,
            args.work_dir,
            _parse_mutations(args.mutate),
            prompts=prompts,
            target_index=args.mutation_target,
            moe_backend=args.moe_backend,
            disable_finalize_fusion=args.disable_finalize_fusion,
            tokens_per_block=args.tokens_per_block,
            transfer_timeout_ms=args.transfer_timeout_ms,
            policy=policy,
        )
        print(json.dumps(judgments, indent=2))
        return
    print(
        json.dumps(
            run_transfer_gate(
                args.model,
                args.work_dir,
                mode=args.mode,
                moe_backend=args.moe_backend,
                disable_finalize_fusion=args.disable_finalize_fusion,
                tokens_per_block=args.tokens_per_block,
                transfer_timeout_ms=args.transfer_timeout_ms,
                policy=policy,
                prompts=prompts,
                ctx_group_size=args.ctx_group_size,
                gen_group_size=args.gen_group_size,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
