# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Observe actual DKV transfers inside MPI workers without changing transport results."""

import importlib.abc
import importlib.machinery
import json
import os
import sys
import time
from functools import wraps
from pathlib import Path

CANCEL_TOKEN = 29003
_TARGET_MODULE = "tensorrt_llm._torch.pyexecutor.py_executor"


def _wait(path: Path) -> None:
    deadline = time.monotonic() + 60
    while not path.exists():
        if time.monotonic() > deadline:
            raise TimeoutError(f"DKV transfer handshake timed out: {path.name}")
        time.sleep(0.01)


def install_pyexecutor_hooks(executor_class: type, trace_dir: str) -> None:
    """Trace physical sends and the lifecycle of every rank, with one bounded abort handshake."""
    if getattr(executor_class, "_dkv_transfer_probe_installed", False):
        return
    executor_class._dkv_transfer_probe_installed = True
    target = Path(trace_dir)
    target.mkdir(parents=True, exist_ok=True)
    initialized: set[int] = set()
    race_ids: set[int] = set()

    def business(executor) -> bool:
        return (
            executor.dkv_enabled
            and not executor.is_warmup
            and not executor.kv_cache_manager.is_estimating_kv_cache
            and executor.model_engine._warmup_timer.purpose == "final_executor"
        )

    def record(executor, event: str, **values) -> None:
        row = {
            "event": event,
            "rank": executor.dist.tp_rank,
            "iteration": executor.iter_counter,
            "control": getattr(executor, "_transfer_probe_control", 0),
            "in_control": getattr(executor, "_transfer_probe_in_control", False),
            **values,
        }
        with (target / f"rank-{executor.dist.tp_rank}-{os.getpid()}.jsonl").open("a") as stream:
            stream.write(json.dumps(row) + "\n")

    def setup(executor) -> None:
        manager = executor.kv_cache_manager
        used, pages = manager.get_dkv_control_digest()
        record(executor, "baseline", index_used=used, free_pages=pages)
        original_release = manager.release_index_slot

        @wraps(original_release)
        def release(request_id):
            result = original_release(request_id)
            used, pages = manager.get_dkv_control_digest()
            record(
                executor,
                "release_index",
                request_id=request_id,
                index_used=used,
                free_pages=pages,
            )
            return result

        manager.release_index_slot = release
        transfers = executor.async_transfer_manager
        original_start = transfers.start_transfer

        @wraps(original_start)
        def start(request):
            result = original_start(request)
            used, pages = manager.get_dkv_control_digest()
            record(
                executor,
                "start",
                request_id=request.py_request_id,
                owner=request.py_dkv_compute_rank,
                index_used=used,
                free_pages=pages,
            )
            return result

        transfers.start_transfer = start
        transceiver = executor.kv_cache_transceiver
        assert not transceiver._ctx_need_tp_sync
        assert not transceiver._ctx_need_pp_sync
        original_send = transceiver.respond_and_send_async

        @wraps(original_send)
        def send(request):
            if request.get_tokens(0)[1] == CANCEL_TOKEN:
                race_ids.add(request.py_request_id)
            result = original_send(request)
            record(
                executor,
                "send",
                request_id=request.py_request_id,
                owner=request.py_dkv_compute_rank,
            )
            return result

        transceiver.respond_and_send_async = send
        coordinator = executor.disagg
        original_stage = coordinator._stage_dkv_transfer_event

        @wraps(original_stage)
        def stage(request, **kwargs):
            present = request.py_request_id in coordinator._dkv_transfer_events
            result = original_stage(request, **kwargs)
            if not present:
                event = coordinator._dkv_transfer_events[request.py_request_id]
                record(
                    executor,
                    "observed",
                    request_id=event.request_id,
                    owner=event.compute_rank,
                    outcome=event.outcome,
                    elapsed_ms=(time.monotonic() - request.py_kv_transfer_start_time) * 1000,
                )
            return result

        coordinator._stage_dkv_transfer_event = stage
        original_transfer_release = coordinator._release_dkv_transfer

        # A request is committed when its transfer ends on this rank: with every rank sending its
        # layers, that is once all of them have reported, and the event carries the merged outcome.
        @wraps(original_transfer_release)
        def commit(request, event):
            record(
                executor,
                "commit",
                request_id=event.request_id,
                owner=event.compute_rank,
                outcome=event.outcome,
            )
            return original_transfer_release(request, event)

        coordinator._release_dkv_transfer = commit

    original_fetch = executor_class._fetch_new_requests

    @wraps(original_fetch)
    def fetch(executor, *args, **kwargs):
        if business(executor) and id(executor) not in initialized:
            initialized.add(id(executor))
            setup(executor)
        return original_fetch(executor, *args, **kwargs)

    original_control = executor_class._sync_dkv_control

    @wraps(original_control)
    def control(executor, *args, **kwargs):
        if not business(executor):
            return original_control(executor, *args, **kwargs)
        executor._transfer_probe_control = getattr(executor, "_transfer_probe_control", 0) + 1
        executor._transfer_probe_in_control = True
        try:
            return original_control(executor, *args, **kwargs)
        finally:
            executor._transfer_probe_in_control = False

    original_cancel = executor_class.cancel_request

    @wraps(original_cancel)
    def cancel(executor, request_id):
        result = original_cancel(executor, request_id)
        if business(executor):
            record(executor, "cancel_queued", request_id=request_id)
            (target / "cancel-queued").write_text(str(request_id))
        return result

    original_try_cancel = executor_class._try_cancel_request

    @wraps(original_try_cancel)
    def try_cancel(executor, request):
        result = original_try_cancel(executor, request)
        if business(executor):
            # Idle polling can revisit a deferred cancellation many times.
            marker = target / f"cancel-tried-{request.py_request_id}-{executor.dist.tp_rank}"
            if not marker.exists():
                record(
                    executor, "cancel_attempt", request_id=request.py_request_id, accepted=result
                )
                marker.touch()
        return result

    original_enqueue = executor_class._enqueue_responses

    @wraps(original_enqueue)
    def enqueue(executor, responses):
        responses = list(responses)
        if business(executor):
            if any(request_id in race_ids for request_id, _ in responses):
                (target / "race-response-ready").touch()
                _wait(target / "release-race-response")
            for request_id, response in responses:
                record(executor, "response", request_id=request_id, error=response.error_msg)
        return original_enqueue(executor, responses)

    original_free = executor_class._free_request_resources

    @wraps(original_free)
    def free(executor, request):
        result = original_free(executor, request)
        if business(executor) and not request.is_dummy:
            used, pages = executor.kv_cache_manager.get_dkv_control_digest()
            record(
                executor,
                "free",
                request_id=request.py_request_id,
                owner=request.py_dkv_compute_rank,
                index_used=used,
                free_pages=pages,
                window=executor._dkv_commit_reason,
            )
            (target / f"freed-{request.py_request_id}-{executor.dist.tp_rank}").touch()
        return result

    executor_class._fetch_new_requests = fetch
    executor_class._sync_dkv_control = control
    executor_class.cancel_request = cancel
    executor_class._try_cancel_request = try_cancel
    executor_class._enqueue_responses = enqueue
    executor_class._free_request_resources = free


class _ProbeLoader(importlib.abc.Loader):
    def __init__(self, wrapped: importlib.abc.Loader, trace_dir: str) -> None:
        self.wrapped = wrapped
        self.trace_dir = trace_dir

    def create_module(self, spec):
        return self.wrapped.create_module(spec)

    def exec_module(self, module) -> None:
        self.wrapped.exec_module(module)
        install_pyexecutor_hooks(module.PyExecutor, self.trace_dir)


class _ProbeFinder(importlib.abc.MetaPathFinder):
    def __init__(self, trace_dir: str) -> None:
        self.trace_dir = trace_dir

    def find_spec(self, fullname, path, target=None):
        if fullname != _TARGET_MODULE:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
        if spec is not None:
            spec.loader = _ProbeLoader(spec.loader, self.trace_dir)
        return spec


def install_import_hook(trace_dir: str) -> None:
    """Install before TensorRT-LLM imports in each context MPI worker."""
    sys.meta_path.insert(0, _ProbeFinder(trace_dir))
