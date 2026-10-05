# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Phase 6 on the replicated-lifecycle DKV: one workload, several placements of its requests.

On the same deterministic workload the runner compares

* ``adp``: attention DP with the default router,
* ``adp_kv:<beta>``: attention DP with the KV-aware router of load-balance weight ``beta``,
* ``dkv``: the replicated lifecycle, whose router is always the default one.

Every mode runs in its own process, because the DKV switches are read when the workers start.
``run`` starts one worker process per mode on this node. ``worker`` runs one mode in the current
process; a group that spans nodes is driven by starting it on every task of the job behind
``trtllm-llmapi-launch``. ``report``, ``summary`` and ``check`` read the results.

The results are hit rates, the context tokens that still had to be computed, the load of every rank
and the logical duplicate storage, each with the validity controls of
``build_dkv_measurement_report``. Timing is perturbed by the DKV debug checks and the fresh-page
fill, so no latency or throughput conclusion can be drawn from a run, and a replicated page that a
rank did not compute holds no valid KV, so no output is checked either.

DeepSeek-V4 reuses a prefix only where a stored sequence ended: a shared prefix followed by a
different suffix never hits, while a conversation's history and a prefix that was sent once on its
own do. Use ``--workload chat``, or ``--workload zipf --prime``, on that model.
"""

import argparse
import dataclasses
import json
import os
import signal
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

if __package__:
    from . import dkv_phase6_report as tables
    from .dkv_stats import latest_snapshots
    from .dkv_workloads import (
        WorkloadRequest,
        chat_workload,
        history_hit_ceiling,
        prefix_bodies,
        run_workload,
        summarize_results,
        warmup_prompts,
        zipf_prefix_workload,
    )
else:
    import dkv_phase6_report as tables
    from dkv_stats import latest_snapshots
    from dkv_workloads import (
        WorkloadRequest,
        chat_workload,
        history_hit_ceiling,
        prefix_bodies,
        run_workload,
        summarize_results,
        warmup_prompts,
        zipf_prefix_workload,
    )

_WARMUP_SALT = "phase6-warmup"
_POLL_S = 0.05


@dataclass(frozen=True)
class Mode:
    """How the requests of a run are placed."""

    name: str
    dkv: bool
    beta: float | None


def parse_mode(spec: str) -> Mode:
    """``dkv``, ``adp`` or ``adp_kv:<beta>``."""
    if spec == "dkv":
        return Mode("dkv", True, None)
    if spec == "adp":
        return Mode("adp", False, None)
    if spec.startswith("adp_kv:"):
        beta = float(spec.split(":", 1)[1])
        return Mode(f"adp_kv_b{beta:g}", False, beta)
    raise ValueError(f"unknown mode {spec!r}: use dkv, adp or adp_kv:<beta>")


@dataclass(frozen=True)
class Phase6Options:
    """Everything one worker needs; it is also written next to the results of the worker."""

    model: str
    mode: str
    group: int = 4
    moe_backend: str | None = None
    tokens_per_block: int = 128
    vocab: int = 0
    workload: Literal["zipf", "chat"] = "zipf"
    prime: bool = False
    requests: int = 192
    prefixes: int = 16
    alpha: float = 1.0
    prefix_blocks: int = 8
    suffix_tokens: int = 128
    sessions: int = 48
    turns_min: int = 3
    turns_max: int = 6
    first_tokens: str = "512,1024,2048"
    turn_tokens: str = "64,128,256"
    interleave: Literal["rounds", "random"] = "rounds"
    warmup_tokens: int = 1024
    seed: int = 0
    concurrency: int = 16
    warmup: int = 8
    phases: int = 2
    max_batch_size: int = 8
    max_num_tokens: int = 4096
    max_seq_len: int = 4096
    kv_max_tokens: int = 4194304
    kv_quota_gib: float | None = None
    event_buffer: int = 262144
    adp_reference: str | None = None
    capacity_tolerance_pages: int = 0
    request_timeout: int = 900
    debug: str = "1"


_OPTION_NAMES = {field.name for field in dataclasses.fields(Phase6Options)}


def _lengths(text: str) -> list[int]:
    return [int(value) for value in text.split(",")]


def build_requests(options: Phase6Options, vocab: int) -> list[WorkloadRequest]:
    """The measured requests of a run."""
    if options.workload == "chat":
        return chat_workload(
            sessions=options.sessions,
            turns=(options.turns_min, options.turns_max),
            first_tokens=_lengths(options.first_tokens),
            turn_tokens=_lengths(options.turn_tokens),
            vocab=vocab,
            seed=options.seed,
            interleave=options.interleave,
        )
    return zipf_prefix_workload(
        options.requests,
        prefixes=options.prefixes,
        alpha=options.alpha,
        prefix_tokens=options.prefix_blocks * options.tokens_per_block,
        suffix_tokens=options.suffix_tokens,
        vocab=vocab,
        seed=options.seed,
    )


def priming_prompts(options: Phase6Options, requests: Sequence[WorkloadRequest]) -> list[list[int]]:
    """The shared prefixes to send alone once before the interval, or none."""
    if options.workload != "zipf" or not options.prime:
        return []
    return prefix_bodies(requests, options.prefix_blocks * options.tokens_per_block)


def hit_ceiling(options: Phase6Options, requests: Sequence[WorkloadRequest]) -> int:
    """The prompt tokens a cache that keeps everything and is visible to every rank can serve.

    A conversation reuses its previous prompt. A shared prefix is reused by every request once it
    was primed, and by every request but the first of its prefix otherwise.
    """
    if options.workload == "chat":
        return history_hit_ceiling(requests)
    prefix_tokens = options.prefix_blocks * options.tokens_per_block
    if options.prime:
        return len(requests) * prefix_tokens
    return (len(requests) - len({request.session for request in requests})) * prefix_tokens


def kv_cache_options(options: Phase6Options) -> dict[str, int | bool | None]:
    """The ``KvCacheConfig`` fields of a run: partial reuse, events, and the capacity limit.

    An explicit quota sets ``max_gpu_total_bytes`` and drops the token limit, so the bytes of every
    rank are what the pools are cut from; otherwise ``max_tokens`` limits them.
    """
    fields: dict[str, int | bool | None] = {
        "enable_partial_reuse": True,
        "event_buffer_max_size": options.event_buffer,
    }
    if options.kv_quota_gib:
        fields["max_tokens"] = None
        fields["max_gpu_total_bytes"] = int(options.kv_quota_gib * (1 << 30))
    else:
        fields["max_tokens"] = options.kv_max_tokens
    return fields


def _dkv_models():
    """The shared LLM factory, imported on demand: it needs torch, the report commands do not."""
    if __package__:
        from . import dkv_models
    else:
        import dkv_models
    return dkv_models


def worker_env(debug: str) -> dict[str, str]:
    """The DKV switches of the workers, which a group launched through MPI only gets this way."""
    env = _dkv_models().dkv_worker_env(measurement=True)
    env["TRTLLM_DKV_DEBUG"] = debug
    env["TRTLLM_KV_FRESH_PAGE_FILL"] = "zero"
    return env


class _Streams(Protocol):
    def get_kv_cache_events(self, timeout: float) -> list[dict]: ...

    def get_stats(self, timeout: float) -> list[dict]: ...


class Observer:
    """Collects the KV events and iteration-stats rows of a running LLM and waits for them."""

    def __init__(
        self,
        llm: _Streams,
        group: int,
        directory: Path,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.events: list[dict] = []
        self.stats: list[dict] = []
        self._llm = llm
        self._group = group
        self._directory = directory
        self._clock = clock
        self._sleep = sleep

    def pull(self) -> list[dict]:
        """Collect what the runtime produced since the last pull; returns the new events."""
        fresh = [event for event in self._llm.get_kv_cache_events(_POLL_S) if event]
        rows = self._llm.get_stats(timeout=_POLL_S)
        self.events.extend(fresh)
        self.stats.extend(rows)
        for name, values in (("events", fresh), ("iteration-stats", rows)):
            if values:
                with (self._directory / f"{name}.jsonl").open("a") as stream:
                    stream.writelines(json.dumps(value) + "\n" for value in values)
        return fresh

    def drain(self, *, quiet_s: float = 1.0, deadline_s: float = 30.0) -> None:
        """Wait until no KV event has arrived for ``quiet_s`` seconds."""
        deadline = self._clock() + deadline_s
        quiet_since = self._clock()
        while self._clock() < deadline:
            if self.pull():
                quiet_since = self._clock()
            elif self._clock() - quiet_since >= quiet_s:
                return
        raise TimeoutError("The KV event stream did not become idle")

    def settle(self, expected: int, *, deadline_s: float = 120.0) -> list[dict]:
        """Wait until the ranks have counted ``expected`` requests and the events are quiet.

        Returns the newest snapshot of every rank. The counters of the last request reach the stats
        one iteration late, so a snapshot taken right after the last answer would miss it.
        """
        deadline = self._clock() + deadline_s
        seen = None
        while self._clock() < deadline:
            self.pull()
            snapshots = latest_snapshots(self.stats)
            if len(snapshots) == self._group:
                seen = sum(snapshot["counters"]["request_count"] for snapshot in snapshots)
                if seen > expected:
                    raise RuntimeError(f"The ranks counted {seen} requests, {expected} were sent")
                if seen == expected:
                    self.drain()
                    return latest_snapshots(self.stats)
            self._sleep(0.02)
        raise TimeoutError(f"The rank counters reached {seen} of {expected} requests")


def _await_cached_tokens(future, observer: Observer, timeout_s: float) -> int:
    """Wait for a request while collecting the streams, so that neither buffer overflows."""
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            return future.result(timeout=0.25).cached_tokens
        except TimeoutError:
            observer.pull()
            if time.monotonic() > deadline:
                raise


def _load_reference(options: Phase6Options, mode: Mode) -> list[dict] | None:
    if options.adp_reference is None:
        return None
    if not mode.dkv:
        raise ValueError("--adp-reference only applies to the dkv mode")
    result = json.loads(Path(options.adp_reference).read_text())
    return result["report"]["capacity_by_rank"]


def run_mode(options: Phase6Options, directory: Path) -> dict:
    """Serve the workload under one mode; writes ``result.json`` and the raw streams.

    Returns the result. Runs in the current process, which must have started every rank (the
    process itself, or ``trtllm-llmapi-launch`` for a group that spans nodes).
    """
    from tensorrt_llm import SamplingParams
    from tensorrt_llm._torch.pyexecutor.dkv_metrics import build_dkv_measurement_report
    from tensorrt_llm.llmapi.llm_args import AttentionDpConfig, MoeConfig

    if options.warmup < 1:
        raise ValueError("at least one warm-up request is needed to see the first statistics")
    directory.mkdir(parents=True, exist_ok=True)
    mode = parse_mode(options.mode)
    vocab = (
        options.vocab or json.loads((Path(options.model) / "config.json").read_text())["vocab_size"]
    )
    requests = build_requests(options, vocab)
    primers = priming_prompts(options, requests)
    warm_length = (
        options.prefix_blocks * options.tokens_per_block + options.suffix_tokens
        if options.workload == "zipf"
        else options.warmup_tokens
    )
    warm = warmup_prompts(count=options.warmup, tokens=warm_length, vocab=vocab, seed=options.seed)
    longest = max(len(request.prompt) for request in requests)
    if longest + 1 > options.max_seq_len or longest > options.max_num_tokens:
        raise ValueError(f"max_seq_len and max_num_tokens must hold the longest prompt ({longest})")
    reference = _load_reference(options, mode)
    per_session: dict[int, int] = {}
    for request in requests:
        per_session[request.session] = per_session.get(request.session, 0) + 1

    llm_kwargs: dict = {
        "max_batch_size": options.max_batch_size,
        "max_num_tokens": options.max_num_tokens,
        "max_seq_len": options.max_seq_len,
        "max_stats_len": -1,
        "env_overrides": worker_env(options.debug),
    }
    if mode.beta is not None:
        llm_kwargs["attention_dp_config"] = AttentionDpConfig(
            enable_kv_cache_aware_routing=True, kv_cache_routing_load_balance_weight=mode.beta
        )
    moe_config = MoeConfig(backend=options.moe_backend) if options.moe_backend else None
    sampling = SamplingParams(max_tokens=1, temperature=0, ignore_eos=True)
    started = time.monotonic()
    print(
        f"WORKER_START mode={mode.name} group={options.group} requests={len(requests)}", flush=True
    )

    with _dkv_models().make_dkv_llm(
        options.model,
        dkv=mode.dkv,
        tokens_per_block=options.tokens_per_block,
        moe_config=moe_config,
        group_size=options.group,
        reuse=True,
        kv_cache=kv_cache_options(options),
        **llm_kwargs,
    ) as llm:
        build_seconds = time.monotonic() - started
        print(f"LLM_READY after {build_seconds:.0f} s", flush=True)
        try:
            capacity = llm.get_kv_cache_capacity()
        except (RuntimeError, AttributeError, NotImplementedError) as error:
            capacity = {"unavailable": repr(error)}
        observer = Observer(llm, options.group, directory)

        def report_of(after: list[dict], before: list[dict], events: int, stats: int) -> dict:
            try:
                return build_dkv_measurement_report(
                    after,
                    observer.events,
                    dkv_enabled=mode.dkv,
                    baseline_snapshots=before,
                    baseline_event_count=events,
                    adp_capacity_reference=reference,
                    capacity_tolerance_pages=options.capacity_tolerance_pages,
                    iteration_stats=observer.stats[stats:],
                )
            except ValueError as error:
                return {"report_error": str(error)}

        def start_all(prompts: Sequence[Sequence[int]], salt: str | None) -> None:
            for begin in range(0, len(prompts), options.concurrency):
                futures = [
                    llm.generate_async(prompt, sampling_params=sampling, cache_salt=salt)
                    for prompt in prompts[begin : begin + options.concurrency]
                ]
                for future in futures:
                    _await_cached_tokens(future, observer, options.request_timeout)

        def submit(request: WorkloadRequest) -> Callable[[], int]:
            future = llm.generate_async(request.prompt, sampling_params=sampling)
            return lambda: _await_cached_tokens(future, observer, options.request_timeout)

        # A shared prefix is primed by sending it alone, which is what DeepSeek-V4 can reuse.
        start_all(warm, _WARMUP_SALT)
        start_all(primers, None)
        prelude = len(warm) + len(primers)
        initial = observer.settle(prelude)
        initial_events, initial_stats = len(observer.events), len(observer.stats)
        print(f"WARMUP_DONE {len(warm)} warm-up and {len(primers)} priming requests", flush=True)

        size = -(-len(requests) // options.phases)
        phases: dict[str, dict] = {}
        completed = prelude
        served: dict[int, int] = {}
        all_results = []
        for number, begin in enumerate(range(0, len(requests), size), start=1):
            part = requests[begin : begin + size]
            before = latest_snapshots(observer.stats)
            event_start, stats_start = len(observer.events), len(observer.stats)
            began = time.monotonic()
            results = run_workload(
                submit, part, concurrency=options.concurrency, completed_turns=served
            )
            seconds = time.monotonic() - began
            for result in results:
                served[result.session] = max(served.get(result.session, 0), result.turn + 1)
            completed += len(part)
            after = observer.settle(completed)
            all_results.extend(results)
            name = f"phase{number}"
            phases[name] = {
                "requests": len(part),
                "seconds": seconds,
                "event_start": event_start,
                "event_end": len(observer.events),
                "client": summarize_results(results),
                "report": report_of(after, before, event_start, stats_start),
            }
            (directory / f"{name}-snapshots.json").write_text(
                json.dumps({"before": before, "after": after}, indent=1)
            )
            counters = phases[name]["report"].get("global", {})
            print(
                f"PHASE_DONE {name} {len(part)} requests {seconds:.0f} s "
                f"hit={counters.get('token_prefix_hit_rate')} "
                f"scheduled={counters.get('scheduled_context_tokens')}",
                flush=True,
            )
        total = report_of(latest_snapshots(observer.stats), initial, initial_events, initial_stats)

    result = {
        "schema_version": 1,
        "options": dataclasses.asdict(options),
        "mode": dataclasses.asdict(mode),
        "kv_cache_capacity": capacity,
        "build_seconds": build_seconds,
        "wall_seconds": time.monotonic() - started,
        "workload": {
            "kind": options.workload,
            "requests": len(requests),
            "sessions": len(per_session),
            "requests_per_session": [per_session[index] for index in sorted(per_session)],
            "prompt_tokens_min": min(len(request.prompt) for request in requests),
            "prompt_tokens_max": longest,
            "primed_prefixes": len(primers),
            "hit_ceiling_tokens": hit_ceiling(options, requests),
        },
        "phases": phases,
        "report": total,
        "client": summarize_results(all_results),
        "requests": [
            {
                "index": r.index,
                "session": r.session,
                "turn": r.turn,
                "prompt_tokens": r.prompt_tokens,
                "cached_tokens": r.cached_tokens,
                "latency_s": r.latency_s,
            }
            for r in all_results
        ],
        "output_correctness_validated": False,
        "interpretation": "Prefix metadata, load and logical storage only; cross-rank reuse "
        "outputs are wrong by design",
    }
    (directory / "result.json").write_text(json.dumps(result, indent=1))
    print("WORKER_DONE", flush=True)
    return result


def run_modes(
    options: Phase6Options,
    modes: Sequence[str],
    out: Path,
    *,
    gpus: str | None = None,
    timeout_s: int = 3000,
) -> dict[str, int]:
    """Run each mode in a worker process of its own, one after the other; returns exit codes.

    The results of a mode are written to ``out/<mode name>``.
    """
    vocab = (
        options.vocab or json.loads((Path(options.model) / "config.json").read_text())["vocab_size"]
    )
    options = dataclasses.replace(options, vocab=vocab)
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("SLURM_", "PMI_", "PMIX_", "OMPI_"))
    }
    env.update(
        OMPI_ALLOW_RUN_AS_ROOT="1",
        OMPI_ALLOW_RUN_AS_ROOT_CONFIRM="1",
        PYTHONDONTWRITEBYTECODE="1",
        CUDA_VISIBLE_DEVICES=gpus or ",".join(str(index) for index in range(options.group)),
    )
    env.update(worker_env(options.debug))
    env["PYTHONPATH"] = os.pathsep.join([str(Path(__file__).parent), env.get("PYTHONPATH", "")])
    env.pop("TLLM_WORKER_USE_SINGLE_PROCESS", None)
    codes: dict[str, int] = {}
    for spec in modes:
        mode = parse_mode(spec)
        directory = out / mode.name
        directory.mkdir(parents=True, exist_ok=True)
        mode_options = dataclasses.replace(options, mode=spec)
        options_path = directory / "options.json"
        options_path.write_text(json.dumps(dataclasses.asdict(mode_options)))
        print(f"SESSION_START {mode.name} log={directory / 'worker.log'}", flush=True)
        began = time.monotonic()
        with (directory / "worker.log").open("w") as log:
            process = subprocess.Popen(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "worker",
                    "--options",
                    str(options_path),
                    "--out",
                    str(directory),
                ],
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                code = process.wait(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                code = 124
            finally:
                _kill_group(process)
        codes[mode.name] = code
        print(
            f"SESSION_END {mode.name} rc={code} seconds={time.monotonic() - began:.0f}", flush=True
        )
    return codes


def _kill_group(process: subprocess.Popen) -> None:
    """End the worker's whole process group, which also holds the ranks it spawned."""
    for signum in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(process.pid, signum)
        except ProcessLookupError:
            break
        time.sleep(3)
    process.wait(timeout=20)


def _add_options(parser: argparse.ArgumentParser, *, model_required: bool = True) -> None:
    defaults = Phase6Options(model="", mode="")
    parser.add_argument("--model", required=model_required)
    parser.add_argument("--group", type=int, default=defaults.group)
    parser.add_argument("--moe-backend", default=None)
    parser.add_argument("--tokens-per-block", type=int, default=defaults.tokens_per_block)
    parser.add_argument("--vocab", type=int, default=0, help="0 reads it from the checkpoint")
    parser.add_argument("--workload", choices=("zipf", "chat"), default=defaults.workload)
    parser.add_argument("--prime", action="store_true", help="zipf: send each prefix alone first")
    parser.add_argument("--requests", type=int, default=defaults.requests, help="zipf: requests")
    parser.add_argument("--prefixes", type=int, default=defaults.prefixes, help="zipf: prefixes")
    parser.add_argument("--alpha", type=float, default=defaults.alpha, help="zipf: popularity")
    parser.add_argument("--prefix-blocks", type=int, default=defaults.prefix_blocks)
    parser.add_argument("--suffix-tokens", type=int, default=defaults.suffix_tokens)
    parser.add_argument("--sessions", type=int, default=defaults.sessions, help="chat: sessions")
    parser.add_argument("--turns-min", type=int, default=defaults.turns_min)
    parser.add_argument("--turns-max", type=int, default=defaults.turns_max)
    parser.add_argument("--first-tokens", default=defaults.first_tokens, help="chat: lengths")
    parser.add_argument("--turn-tokens", default=defaults.turn_tokens, help="chat: growth")
    parser.add_argument("--interleave", choices=("rounds", "random"), default=defaults.interleave)
    parser.add_argument("--warmup-tokens", type=int, default=defaults.warmup_tokens)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--concurrency", type=int, default=defaults.concurrency)
    parser.add_argument("--warmup", type=int, default=defaults.warmup)
    parser.add_argument("--phases", type=int, default=defaults.phases)
    parser.add_argument("--max-batch-size", type=int, default=defaults.max_batch_size)
    parser.add_argument("--max-num-tokens", type=int, default=defaults.max_num_tokens)
    parser.add_argument("--max-seq-len", type=int, default=defaults.max_seq_len)
    parser.add_argument("--kv-max-tokens", type=int, default=defaults.kv_max_tokens)
    parser.add_argument("--kv-quota-gib", type=float, default=None, help="max_gpu_total_bytes")
    parser.add_argument("--event-buffer", type=int, default=defaults.event_buffer)
    parser.add_argument("--adp-reference", default=None, help="result.json of an adp run")
    parser.add_argument("--capacity-tolerance-pages", type=int, default=0)
    parser.add_argument("--request-timeout", type=int, default=defaults.request_timeout)
    parser.add_argument("--debug", default=defaults.debug, help="TRTLLM_DKV_DEBUG of the workers")


def _options_of(args: argparse.Namespace, mode: str) -> Phase6Options:
    return Phase6Options(
        **{key: value for key, value in vars(args).items() if key in _OPTION_NAMES - {"mode"}},
        mode=mode,
    )


def _parse_groups(specs: Sequence[str]) -> dict[str, list[dict]]:
    groups = {}
    for spec in specs:
        label, directories = spec.split("=", 1)
        groups[label] = tables.load_results(Path(path) for path in directories.split(","))
    return groups


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="one worker process per mode, on this node")
    _add_options(run)
    run.add_argument("--modes", default="adp,dkv", help="comma separated: dkv, adp, adp_kv:<beta>")
    run.add_argument("--out", required=True)
    run.add_argument("--gpus", default=None, help="CUDA_VISIBLE_DEVICES; default 0..group-1")
    run.add_argument("--timeout", type=int, default=3000, help="seconds allowed to one mode")
    work = commands.add_parser("worker", help="one mode in this process")
    work.add_argument("--options", help="options.json written by run; replaces the options below")
    work.add_argument("--mode")
    work.add_argument("--out", required=True)
    _add_options(work, model_required=False)
    report = commands.add_parser("report", help="tables of the results in the directories")
    report.add_argument("directories", nargs="+")
    summary = commands.add_parser("summary", help="a column per label: LABEL=DIR[,DIR...]")
    summary.add_argument("groups", nargs="+")
    check = commands.add_parser("check", help="list the runs and flag the incomplete ones")
    check.add_argument("directories", nargs="+")
    args = parser.parse_args(argv)

    if args.command == "run":
        try:
            for spec in args.modes.split(","):
                parse_mode(spec)
        except ValueError as error:
            parser.error(str(error))
        codes = run_modes(
            _options_of(args, mode=""),
            args.modes.split(","),
            Path(args.out),
            gpus=args.gpus,
            timeout_s=args.timeout,
        )
        failed = [name for name, code in codes.items() if code]
        print("PHASE6_DONE " + json.dumps({"failed": failed}), flush=True)
        return 1 if failed else 0
    if args.command == "worker":
        if args.options:
            options = Phase6Options(**json.loads(Path(args.options).read_text()))
        elif args.mode and args.model:
            options = _options_of(args, mode=args.mode)
        else:
            parser.error("worker needs --options, or --model and --mode")
        run_mode(options, Path(args.out))
        return 0
    if args.command == "report":
        print(
            tables.comparison_tables(tables.load_results(Path(path) for path in args.directories))
        )
        return 0
    if args.command == "summary":
        print(tables.seed_summary(_parse_groups(args.groups)))
        return 0
    lines = tables.check_runs(tables.load_results(Path(path) for path in args.directories))
    print("\n".join(lines))
    return 1 if any("INCOMPLETE" in line for line in lines) else 0


if __name__ == "__main__":
    sys.exit(main())
