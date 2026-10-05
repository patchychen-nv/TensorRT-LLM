# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Every placement of a reuse workload is measured, and the replicated lifecycle reaches the ceiling."""

import dataclasses
import json
from pathlib import Path

import pytest
import torch

from .dkv_models import TINY_LLAMA
from .dkv_phase6_report import check_runs, load_results
from .dkv_phase6_runner import Phase6Options, run_modes

_MODES = ["adp", "adp_kv:1", "dkv"]
_SMALL = dict(
    group=2,
    tokens_per_block=TINY_LLAMA.tokens_per_block,
    requests=24,
    prefixes=3,
    prefix_blocks=3,
    suffix_tokens=16,
    sessions=12,
    turns_min=2,
    turns_max=3,
    first_tokens="96,160",
    turn_tokens="32,64",
    warmup_tokens=96,
    concurrency=4,
    warmup=2,
    max_batch_size=4,
    max_num_tokens=512,
    max_seq_len=512,
    kv_max_tokens=65536,
    event_buffer=65536,
)


@pytest.mark.threadleak(enabled=False)
@pytest.mark.parametrize(
    "workload",
    [
        dict(workload="chat", interleave="random"),
        dict(workload="zipf", prime=True),
    ],
    ids=["chat", "primed-zipf"],
)
def test_dkv_phase6_modes_on_tinyllama(tmp_path: Path, workload: dict) -> None:
    """DKV serves what a cache every rank can read can serve; ADP cannot, and nothing is evicted."""
    if torch.cuda.device_count() < 2:
        pytest.skip("The Phase 6 runner needs two GPUs")
    if not Path(TINY_LLAMA.path()).is_dir():
        pytest.skip(f"The checkpoint is not available at {TINY_LLAMA.path()}")
    options = Phase6Options(model=TINY_LLAMA.path(), mode="", **_SMALL, **workload)
    codes = run_modes(options, _MODES, tmp_path, timeout_s=1500)
    assert codes == {"adp": 0, "adp_kv_b1": 0, "dkv": 0}

    results = {result["mode"]["name"]: result for result in load_results([tmp_path])}
    assert not any("INCOMPLETE" in line for line in check_runs(list(results.values())))
    ceiling = results["dkv"]["workload"]["hit_ceiling_tokens"]
    matched = {
        name: result["report"]["global"]["matched_prefix_tokens"]
        for name, result in results.items()
    }
    assert matched["dkv"] == ceiling > 0
    assert matched["adp"] < ceiling
    assert matched["adp_kv_b1"] <= ceiling
    for result in results.values():
        assert result["report"]["storage"]["last_tier_capacity_dropped_pages"] == 0
        assert not result["output_correctness_validated"]
    # Replication stores every block on both ranks.
    assert results["dkv"]["report"]["storage"]["duplicate_storage_ratio"] == pytest.approx(0.5)

    # The options of a run are on disk next to its results, so it can be repeated.
    saved = json.loads((tmp_path / "dkv" / "options.json").read_text())
    assert Phase6Options(**saved) == dataclasses.replace(options, mode="dkv", vocab=saved["vocab"])
