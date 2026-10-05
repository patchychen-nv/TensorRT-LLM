# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Observe metadata reuse without making DKV output-accuracy assertions."""

from pathlib import Path

import pytest
import torch

from .dkv_measurement_runner import run_measurement_gate


@pytest.mark.threadleak(enabled=False)
def test_dkv_prefix_measurement_validity(tmp_path: Path) -> None:
    from ..conftest import llm_models_root

    if torch.cuda.device_count() < 2:
        pytest.skip("DKV measurement requires two GPUs")
    result = run_measurement_gate(
        str(Path(llm_models_root()) / "llama-models-v2/TinyLlama-1.1B-Chat-v1.0"),
        str(tmp_path / "measurement"),
    )
    assert result["status"] == "passed"
    assert result["pressure_invalidated"]
    assert not result["timing_valid"]
    assert not result["output_correctness_validated"]
