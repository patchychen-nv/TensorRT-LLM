# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test-only MPI worker instrumentation for the aggregate DKV fault gate."""

import importlib.abc
import importlib.machinery
import json
import os
import sys
import time
from functools import wraps
from pathlib import Path
from unittest.mock import patch

FAULT_TOKEN = 29001
FAULT_MESSAGE = "DKV gate injected rank-one sampler failure"
_TARGET_MODULE = "tensorrt_llm._torch.pyexecutor.py_executor"


def _wait_for(path: Path) -> None:
    deadline = time.monotonic() + 60
    while not path.exists():
        if time.monotonic() > deadline:
            raise TimeoutError(f"DKV fault gate handshake timed out: {path.name}")
        time.sleep(0.01)


def install_pyexecutor_hooks(executor_class: type, trace_dir: str) -> None:
    """Inject one sampler failure and observe frees before proxy response deduplication."""
    if getattr(executor_class, "_dkv_fault_probe_installed", False):
        return
    executor_class._dkv_fault_probe_installed = True
    target = Path(trace_dir)
    target.mkdir(parents=True, exist_ok=True)
    fetch_seen: set[int] = set()
    failure_seen: set[int] = set()

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
            **values,
        }
        with (target / f"rank-{executor.dist.tp_rank}-{os.getpid()}.jsonl").open("a") as stream:
            stream.write(json.dumps(row) + "\n")

    original_fetch = executor_class._fetch_new_requests

    @wraps(original_fetch)
    def fetch(executor, *args, **kwargs):
        if business(executor) and id(executor) not in fetch_seen:
            fetch_seen.add(id(executor))
            stats = executor.kv_cache_manager.get_kv_cache_stats()
            record(executor, "baseline", free_blocks=stats.free_num_blocks)
            (target / f"fetch-ready-{executor.dist.tp_rank}").touch()
            _wait_for(target / "release-fetch")
        return original_fetch(executor, *args, **kwargs)

    original_cancel = executor_class.cancel_request

    @wraps(original_cancel)
    def cancel(executor, request_id: int) -> None:
        original_cancel(executor, request_id)
        if business(executor):
            record(executor, "cancel_queued", request_id=request_id)
            (target / "cancel-queued").write_text(str(request_id))

    original_try_cancel = executor_class._try_cancel_request

    @wraps(original_try_cancel)
    def try_cancel(executor, request) -> bool:
        result = original_try_cancel(executor, request)
        if business(executor):
            record(executor, "cancel_applied", request_id=request.py_request_id, accepted=result)
        return result

    original_sample = executor_class._sample_async

    @wraps(original_sample)
    def sample(executor, batch, *args, **kwargs):
        local = [
            request
            for request in batch.all_requests()
            if request.py_dkv_is_local and not request.is_dummy
        ]
        inject = (
            business(executor)
            and executor.dist.tp_rank == 1
            and id(executor) not in failure_seen
            and any(request.get_tokens(0)[-1] == FAULT_TOKEN for request in local)
        )
        if inject:
            failure_seen.add(id(executor))
            record(executor, "injected", request_ids=[request.py_request_id for request in local])
            with patch.object(
                executor.sampler, "sample_async", side_effect=RuntimeError(FAULT_MESSAGE)
            ):
                return original_sample(executor, batch, *args, **kwargs)
        return original_sample(executor, batch, *args, **kwargs)

    original_enqueue = executor_class._enqueue_responses

    @wraps(original_enqueue)
    def enqueue(executor, responses):
        responses = list(responses)
        if business(executor):
            for request_id, response in responses:
                record(executor, "response", request_id=request_id, error=response.error_msg)
        return original_enqueue(executor, responses)

    original_free = executor_class._free_request_resources

    @wraps(original_free)
    def free(executor, request) -> None:
        original_free(executor, request)
        if business(executor) and not request.is_dummy:
            stats = executor.kv_cache_manager.get_kv_cache_stats()
            record(
                executor,
                "free",
                request_id=request.py_request_id,
                owner=request.py_dkv_compute_rank,
                free_blocks=stats.free_num_blocks,
                finish_reasons=[reason.value for reason in request.finish_reasons],
            )
            (target / f"freed-{request.py_request_id}-{executor.dist.tp_rank}").touch()

    executor_class._fetch_new_requests = fetch
    executor_class.cancel_request = cancel
    executor_class._try_cancel_request = try_cancel
    executor_class._sample_async = sample
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
    """Install before TensorRT-LLM imports in each spawned MPI worker process."""
    sys.meta_path.insert(0, _ProbeFinder(trace_dir))
