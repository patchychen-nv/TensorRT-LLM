# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The pieces of the transfer gate runner that need no GPU: option plumbing and judging."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch

_DKV_DIR = Path(__file__).resolve().parents[3] / "integration/defs/dkv"
sys.path.insert(0, str(_DKV_DIR))
try:
    _SPEC = importlib.util.spec_from_file_location(
        "dkv_transfer_runner_under_test", _DKV_DIR / "dkv_transfer_runner.py"
    )
    assert _SPEC is not None and _SPEC.loader is not None
    _RUNNER = importlib.util.module_from_spec(_SPEC)
    _SPEC.loader.exec_module(_RUNNER)
finally:
    sys.path.remove(str(_DKV_DIR))

pytestmark = pytest.mark.cpu_only

_VOCABULARY = 32
_REQUESTS = 6


def _options(**changes) -> dict:
    options = _RUNNER._transfer_options(
        "model",
        moe_backend="CUTLASS",
        disable_finalize_fusion=False,
        tokens_per_block=32,
        transfer_timeout_ms=1000,
        prompts=[[1] * (10 + index) for index in range(_REQUESTS)],
    )
    return {**options, "dkv": True, **changes}


def test_group_sizes_default_to_two_ranks_and_follow_the_options() -> None:
    assert _RUNNER._group_size({}, context=True) == 2
    assert _RUNNER._group_size({}, context=False) == 2
    options = {"ctx_group_size": 8, "gen_group_size": 4}
    assert _RUNNER._group_size(options, context=True) == 8
    assert _RUNNER._group_size(options, context=False) == 4


def test_a_mutation_spec_aims_at_a_prompt_by_its_unique_length(tmp_path: Path) -> None:
    prompts = [[1] * 10, [1] * 20, [1] * 20]
    spec = _RUNNER._mutation_spec(prompts, 0, {"kind": "zero"}, tmp_path)
    assert spec == {"kind": "zero", "prompt_len": 10, "record_dir": str(tmp_path)}
    with pytest.raises(ValueError, match="only one of its length"):
        _RUNNER._mutation_spec(prompts, 1, {"kind": "zero"}, tmp_path)


def test_mutations_are_parsed_from_the_command_line() -> None:
    parsed = _RUNNER._parse_mutations(["zero", "foreign:0.25"])
    assert parsed == {
        "zero": {"kind": "zero", "fraction": 1.0},
        "foreign-0.25": {"kind": "foreign", "fraction": 0.25},
    }


def test_only_a_dkv_context_group_gets_the_probe_and_the_mutation(tmp_path: Path) -> None:
    mutation = {"kind": "zero", "prompt_len": 10, "record_dir": str(tmp_path / "records")}
    options = _options(mutation=mutation)
    context = _RUNNER._role_env("context", options, tmp_path)
    assert context["DKV_TRANSFER_TRACE_DIR"] == str(tmp_path / "traces")
    assert json.loads(context["DKV_KV_MUTATION"]) == mutation
    assert context["PYTHONPATH"].startswith(str(tmp_path / "hook"))
    generation = _RUNNER._role_env("generation", options, tmp_path)
    assert "DKV_TRANSFER_TRACE_DIR" not in generation
    assert "DKV_KV_MUTATION" not in generation
    control = _RUNNER._role_env("context", _options(dkv=False), tmp_path)
    assert "DKV_TRANSFER_TRACE_DIR" not in control


def test_a_prepared_run_directory_is_complete_before_its_options_appear(tmp_path: Path) -> None:
    run = tmp_path / "dkv"
    _RUNNER._prepare_pair(_options(), run)
    hook = (run / "hook" / "sitecustomize.py").read_text()
    assert "install_import_hook" in hook and "install_mutation_hook" in hook
    assert json.loads((run / "options.json").read_text())["tokens_per_block"] == 32
    assert not list(run.glob("*.tmp"))


def _logits(seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(1, _VOCABULARY, generator=generator) * 5


def _save_run(root: Path, name: str, logits: list[torch.Tensor]) -> None:
    (root / name).mkdir(parents=True)
    tokens = [[int(row.argmax())] for row in logits]
    torch.save({"tokens": tokens, "logits": logits}, root / name / "outputs.pt")


def _write_records(root: Path, name: str) -> None:
    (root / name / "mutations").mkdir(parents=True)
    row = {"kind": "zero", "peer_rank": 0, "corrupted_regions": 3, "corrupted_bytes": 96}
    (root / name / "mutations" / "mutations-1.jsonl").write_text(json.dumps(row) + "\n")


def test_a_corrupted_request_is_reported_by_the_judge(tmp_path: Path) -> None:
    clean = [_logits(seed) for seed in range(_REQUESTS)]
    _save_run(tmp_path, "adp-control", clean)
    _save_run(tmp_path, "adp-replay", clean)
    corrupted = [value.clone() for value in clean]
    corrupted[2] = _logits(99)
    _save_run(tmp_path, "mutated", corrupted)
    _write_records(tmp_path, "mutated")
    judgment = _RUNNER.judge_mutation(tmp_path, "mutated", _RUNNER.EXACT_POLICY)
    assert judgment["failing_requests"] == [2]
    assert judgment["gate_error"] is not None
    assert [record["corrupted_bytes"] for record in judgment["records"]] == [96]


def test_a_clean_run_is_accepted_by_the_judge(tmp_path: Path) -> None:
    clean = [_logits(seed) for seed in range(_REQUESTS)]
    for name in ("adp-control", "adp-replay", "same"):
        _save_run(tmp_path, name, clean)
    (tmp_path / "same" / "mutations").mkdir()
    judgment = _RUNNER.judge_mutation(tmp_path, "same", _RUNNER.EXACT_POLICY)
    assert judgment == {"gate_error": None, "failing_requests": [], "records": []}
