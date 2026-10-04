# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Four-GPU context-first NIXL transfers with replicated DKV context state."""

from pathlib import Path

import pytest
import torch

from .dkv_transfer_runner import run_transfer_gate


@pytest.mark.post_merge
@pytest.mark.threadleak(enabled=False)
@pytest.mark.timeout(3000)
def test_dkv_context_transfer_precision_and_timeouts(tmp_path: Path) -> None:
    from ..conftest import llm_models_root

    if torch.cuda.device_count() < 4:
        pytest.skip("DKV context TP2 + ordinary generation TP2 requires four GPUs")
    result = run_transfer_gate(
        str(Path(llm_models_root()) / "llama-models-v2/TinyLlama-1.1B-Chat-v1.0"),
        str(tmp_path / "transfer"),
    )
    assert result["lifecycle"]["status"] == "passed"
    assert result["precision"]["status"] == "passed"
