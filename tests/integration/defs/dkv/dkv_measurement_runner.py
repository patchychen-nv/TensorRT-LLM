# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Two-rank metadata reuse experiments with explicit Option B validity controls."""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def _snapshots(directory: Path, *, baseline: bool = False) -> list[dict]:
    snapshots = {}
    for path in directory.glob("rank-*.jsonl"):
        contents = path.read_text()
        complete = contents.rsplit("\n", 1)[0]
        rows = [json.loads(line) for line in complete.splitlines()]
        if rows:
            row = rows[0] if baseline else rows[-1]
            rank = row["snapshot"]["rank"]
            assert rank not in snapshots, "Multiple final executors recorded the same rank"
            snapshots[rank] = row["snapshot"]
    return [snapshots[rank] for rank in sorted(snapshots)]


def _wait_snapshots(directory: Path, counts: list[int]) -> list[dict]:
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        baseline = _snapshots(directory, baseline=True)
        current = _snapshots(directory)
        if len(baseline) == len(current) == 2:
            actual = [
                row["counters"]["request_count"] - start["counters"]["request_count"]
                for row, start in zip(current, baseline, strict=True)
            ]
            assert all(a <= b for a, b in zip(actual, counts, strict=True)), actual
            if actual == counts:
                return current
        time.sleep(0.02)
    raise TimeoutError(f"Missing rank measurement counters {counts}: {directory}")


def _worker(options: dict, directory: Path) -> None:
    from tensorrt_llm import LLM, SamplingParams
    from tensorrt_llm._torch.pyexecutor.dkv_metrics import build_dkv_measurement_report
    from tensorrt_llm.llmapi import KvCacheConfig
    from tensorrt_llm.llmapi.llm_args import BlockReuseConfig, DkvConfig, MoeConfig
    from tensorrt_llm.scheduling_params import SchedulingParams

    block = options["tokens_per_block"]
    snapshots_dir = directory / "traces"
    events: list[dict] = []
    stats: list[dict] = []
    counts = [0, 0]
    phases: dict[str, dict] = {}
    request_ids = set()
    sampling = SamplingParams(max_tokens=1, temperature=0, ignore_eos=True)

    with LLM(
        model=options["model"],
        tensor_parallel_size=2,
        moe_expert_parallel_size=2,
        enable_attention_dp=True,
        disable_overlap_scheduler=True,
        enable_chunked_prefill=False,
        enable_autotuner=False,
        cuda_graph_config=None,
        max_batch_size=2,
        max_num_tokens=4 * block,
        max_seq_len=4 * block,
        num_postprocess_workers=0,
        enable_iter_perf_stats=True,
        dkv_config=DkvConfig() if options["dkv"] else None,
        moe_config=MoeConfig(backend=options["moe_backend"]),
        kv_cache_config=KvCacheConfig(
            use_kv_cache_manager_v2=True,
            enable_block_reuse=True,
            enable_swa_scratch_reuse=False,
            block_reuse_config=BlockReuseConfig(policy="per_request"),
            event_buffer_max_size=32768,
            free_gpu_memory_fraction=0.5,
            max_tokens=options["cache_blocks"] * block,
            host_cache_size=0,
            iteration_stats_interval=1,
            tokens_per_block=block,
        ),
    ) as llm:
        _wait_snapshots(snapshots_dir, counts)

        def drain() -> None:
            deadline = time.monotonic() + 15
            quiet_since = time.monotonic()
            while time.monotonic() < deadline:
                latest = [event for event in llm.get_kv_cache_events(0.2) if event]
                events.extend(latest)
                latest_stats = llm.get_stats(timeout=0.2)
                stats.extend(latest_stats)
                for name, rows in (("events", latest), ("iteration-stats", latest_stats)):
                    if rows:
                        with (directory / f"{name}.jsonl").open("a") as stream:
                            stream.writelines(json.dumps(row) + "\n" for row in rows)
                if latest:
                    quiet_since = time.monotonic()
                elif time.monotonic() - quiet_since >= 0.5:
                    return
            raise TimeoutError("KV event stream did not become idle after sequential requests")

        drain()

        def submit(prompt: list[int], rank: int, salt: str) -> None:
            output = llm.generate_async(
                prompt,
                sampling_params=sampling,
                scheduling_params=SchedulingParams(
                    attention_dp_rank=rank, attention_dp_relax=False
                ),
                cache_salt=salt,
            ).result(timeout=120)
            assert output.request_id not in request_ids
            request_ids.add(output.request_id)
            counts[rank] += 1

        submit([200], 0, "measurement-prime")
        _wait_snapshots(snapshots_dir, counts)
        drain()
        initial = _snapshots(snapshots_dir)
        initial_event_count = len(events)

        def phase(name: str, requests: list[tuple[list[int], int, str]]) -> None:
            before = _snapshots(snapshots_dir)
            event_start = len(events)
            for prompt, rank, salt in requests:
                submit(prompt, rank, salt)
            _wait_snapshots(snapshots_dir, counts)
            drain()
            after = _snapshots(snapshots_dir)
            (directory / f"{name}-snapshots.json").write_text(
                json.dumps({"before": before, "after": after}, indent=2)
            )
            report = build_dkv_measurement_report(
                after,
                events,
                dkv_enabled=options["dkv"],
                baseline_snapshots=before,
                baseline_event_count=event_start,
            )
            phases[name] = {
                "baseline_snapshots": before,
                "rank_snapshots": after,
                "event_start": event_start,
                "event_end": len(events),
                "report": report,
            }
            (directory / "partial-result.json").write_text(json.dumps(phases, indent=2))

        prefix = [42] * (2 * block)
        phase("warm", [(prefix + [100], 0, "shared")])
        phase("cross_rank", [(prefix + [101], 1, "shared")])
        phase("local_repeat", [(prefix + [102], 1, "shared")])
        if options["pressure"]:
            pressure = [
                ([1000 + index] * (2 * block) + [200], index % 2, f"pressure-{index}")
                for index in range(options["cache_blocks"] * 4)
            ]
            phase("pressure", pressure)
            phase("after_pressure", [(prefix + [103], 0, "shared")])
        final = _snapshots(snapshots_dir)
        total = build_dkv_measurement_report(
            final,
            events,
            dkv_enabled=options["dkv"],
            baseline_snapshots=initial,
            baseline_event_count=initial_event_count,
        )

    result = {
        "options": options,
        "request_count": len(request_ids),
        "priming_request_count": 1,
        "phases": phases,
        "report": total,
        "events": events,
        "iteration_stats": stats,
        "output_correctness_validated": False,
        "interpretation": "Prefix metadata and logical storage only; no replica KV accuracy claim",
    }
    (directory / "result.json").write_text(json.dumps(result, indent=2))


def _run_session(options: dict, directory: Path) -> dict:
    directory.mkdir(parents=True)
    options_path = directory / "options.json"
    options_path.write_text(json.dumps(options))
    hook = directory / "hook"
    hook.mkdir()
    (hook / "sitecustomize.py").write_text(
        "import os\n"
        "if os.environ.get('DKV_MEASUREMENT_TRACE_DIR'):\n"
        "    from dkv_measurement_probe import install_import_hook\n"
        "    install_import_hook(os.environ['DKV_MEASUREMENT_TRACE_DIR'])\n"
    )
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("SLURM_", "PMI_", "PMIX_", "OMPI_"))
    }
    env.update(
        {
            "OMPI_ALLOW_RUN_AS_ROOT": "1",
            "OMPI_ALLOW_RUN_AS_ROOT_CONFIRM": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "TRTLLM_DKV_DEBUG": "1",
            "TRTLLM_DKV_MEASUREMENT": "1",
            "TRTLLM_KV_FRESH_PAGE_FILL": "zero",
            "DKV_MEASUREMENT_TRACE_DIR": str(directory / "traces"),
        }
    )
    env["PYTHONPATH"] = os.pathsep.join(
        [str(hook), str(Path(__file__).parent), env.get("PYTHONPATH", "")]
    )
    env.pop("TLLM_WORKER_USE_SINGLE_PROCESS", None)
    with (directory / "worker.log").open("w") as log:
        process = subprocess.Popen(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker-options",
                str(options_path),
                "--work-dir",
                str(directory),
            ],
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            code = process.wait(timeout=1800)
            if code != 0:
                raise RuntimeError(f"Measurement worker failed ({code}); logs: {directory}")
        finally:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=20)
            finally:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
    return json.loads((directory / "result.json").read_text())


def run_measurement_gate(
    model: str,
    work_dir: str,
    *,
    moe_backend: str = "CUTLASS",
    tokens_per_block: int = 32,
    cache_blocks: int = 32,
    pressure: bool = True,
) -> dict:
    """Compare actual ADP/DKV metadata reuse, accepting only unpolluted hit-rate intervals."""
    if cache_blocks < 16 or tokens_per_block < 1:
        raise ValueError("Use at least 16 cache blocks and a positive block size")
    directory = Path(work_dir).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    options = {
        "model": model,
        "moe_backend": moe_backend,
        "tokens_per_block": tokens_per_block,
        "cache_blocks": cache_blocks,
        "pressure": pressure,
    }
    runs = {
        name: _run_session({**options, "dkv": enabled}, directory / name)
        for name, enabled in (("adp", False), ("dkv", True))
    }
    for name, run in runs.items():
        for phase_name in ("warm", "cross_rank", "local_repeat"):
            report = run["phases"][phase_name]["report"]
            assert report["storage"]["complete"], (name, phase_name, report)
            assert report["validity"]["prefix_hit_rate_proxy_valid"], (name, phase_name, report)
            assert report["validity"]["timing_perturbed_by_fresh_page_fill"]
            assert not report["validity"]["output_correctness_validated"]
            assert report["global"]["request_count"] == 1
        if pressure:
            report = run["phases"]["pressure"]["report"]
            assert report["storage"]["complete"]
            assert report["storage"]["removed_block_copies"] > 0
            assert report["storage"]["last_tier_capacity_dropped_pages"] > 0
            assert not report["validity"]["prefix_hit_rate_proxy_valid"]
    adp = runs["adp"]["phases"]["cross_rank"]["report"]["global"]
    dkv = runs["dkv"]["phases"]["cross_rank"]["report"]["global"]
    assert adp["matched_prefix_tokens"] == 0
    assert dkv["matched_prefix_tokens"] == 2 * tokens_per_block
    assert adp["scheduled_context_tokens"] == 2 * tokens_per_block + 1
    assert dkv["scheduled_context_tokens"] == 1
    summary = {
        "status": "passed",
        "cross_rank_prefix_tokens": {
            "adp": adp["matched_prefix_tokens"],
            "dkv": dkv["matched_prefix_tokens"],
        },
        "cross_rank_scheduled_context_tokens": {
            "adp": adp["scheduled_context_tokens"],
            "dkv": dkv["scheduled_context_tokens"],
        },
        "pressure_invalidated": pressure,
        "fresh_page_fill": "zero",
        "timing_valid": False,
        "output_correctness_validated": False,
        "runs": {
            name: {
                key: value for key, value in run.items() if key not in ("events", "iteration_stats")
            }
            for name, run in runs.items()
        },
    }
    (directory / "result.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model")
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--moe-backend", default="CUTLASS")
    parser.add_argument("--tokens-per-block", type=int, default=32)
    parser.add_argument("--cache-blocks", type=int, default=32)
    parser.add_argument("--skip-pressure", action="store_true")
    parser.add_argument("--worker-options")
    args = parser.parse_args()
    if args.worker_options:
        _worker(json.loads(Path(args.worker_options).read_text()), Path(args.work_dir))
    else:
        if not args.model:
            parser.error("--model is required")
        run_measurement_gate(
            args.model,
            args.work_dir,
            moe_backend=args.moe_backend,
            tokens_per_block=args.tokens_per_block,
            cache_blocks=args.cache_blocks,
            pressure=not args.skip_pressure,
        )


if __name__ == "__main__":
    main()
