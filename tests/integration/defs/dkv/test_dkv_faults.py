# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Aggregate cancellation and recoverable rank-local sampler errors across the DKV group."""

import os
from pathlib import Path

import pytest

from tensorrt_llm.llmapi.mpi_session import get_mpi_world_size

from .dkv_faults import run_aggregate_faults, run_fault_gate_in_fresh_process
from .dkv_models import MODEL_IDS, MODELS, DkvModel, available_gpus, dkv_worker_env


@pytest.mark.threadleak(enabled=False)
@pytest.mark.parametrize("group_size", (2, 8), ids=("g2", "g8"))
@pytest.mark.parametrize("model", MODELS, ids=MODEL_IDS)
def test_dkv_aggregate_sampler_failure_and_cancel(
    monkeypatch, tmp_path: Path, model: DkvModel, group_size: int
) -> None:
    if available_gpus() < group_size:
        pytest.skip(f"The fault gate of {group_size} ranks needs {group_size} GPUs")
    if not Path(model.path()).is_dir():
        pytest.skip(f"{model.pytest_id} checkpoint is not available at {model.path()}")
    if get_mpi_world_size() > 1:
        # Ranks that MPI launched load no hook that a test writes, so the job has to provide it:
        # its PYTHONPATH carries a sitecustomize hook that installs the probe for the directory
        # named by DKV_FAULT_TRACE_DIR.
        if not os.environ.get("DKV_FAULT_TRACE_DIR"):
            pytest.skip("MPI-launched ranks need the fault probe installed by the job")
        traces = Path(os.environ["DKV_FAULT_TRACE_DIR"])
        # The probe keeps writing to this directory, so an earlier test of the job leaves its
        # traces and handshakes in it.
        for leftover in traces.iterdir():
            leftover.unlink()
        for key, value in dkv_worker_env().items():
            monkeypatch.setenv(key, value)
        result = run_aggregate_faults(model, str(traces), group_size)
    else:
        result = run_fault_gate_in_fresh_process(model, tmp_path / "faults", group_size)
    assert result["status"] == "passed"
