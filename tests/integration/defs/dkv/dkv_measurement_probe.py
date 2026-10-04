# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Persist opt-in manager counters without changing worker scheduling or collectives."""

import importlib.abc
import importlib.machinery
import json
import os
import sys
from functools import wraps
from pathlib import Path

_TARGET_MODULE = "tensorrt_llm._torch.pyexecutor.py_executor"


def install_pyexecutor_hooks(executor_class: type, trace_dir: str) -> None:
    """Record the final executor baseline and snapshots after its normal statistics pass."""
    if getattr(executor_class, "_dkv_measurement_probe_installed", False):
        return
    executor_class._dkv_measurement_probe_installed = True
    directory = Path(trace_dir)
    directory.mkdir(parents=True, exist_ok=True)
    previous: dict[int, dict] = {}

    def record(executor) -> None:
        if (
            executor.is_warmup
            or executor.kv_cache_manager.is_estimating_kv_cache
            or executor.model_engine._warmup_timer.purpose != "final_executor"
        ):
            return
        snapshot = executor.kv_cache_manager.get_dkv_measurement_snapshot()
        if snapshot is None:
            raise RuntimeError("Measurement requires TRTLLM_DKV_MEASUREMENT=1")
        key = id(executor)
        if snapshot == previous.get(key):
            return
        row = {
            "baseline": key not in previous,
            "iteration": executor.iter_counter,
            "snapshot": snapshot,
        }
        previous[key] = snapshot
        path = directory / f"rank-{executor.dist.tp_rank}-{os.getpid()}.jsonl"
        with path.open("a") as stream:
            stream.write(json.dumps(row) + "\n")

    original_fetch = executor_class._fetch_new_requests

    @wraps(original_fetch)
    def fetch(executor, *args, **kwargs):
        if id(executor) not in previous:
            record(executor)
        return original_fetch(executor, *args, **kwargs)

    executor_class._fetch_new_requests = fetch

    original_stats = executor_class._process_iter_stats

    @wraps(original_stats)
    def stats(executor, *args, **kwargs):
        result = original_stats(executor, *args, **kwargs)
        record(executor)
        return result

    executor_class._process_iter_stats = stats


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
    """Install before TensorRT-LLM imports in each MPI worker."""
    sys.meta_path.insert(0, _ProbeFinder(trace_dir))
