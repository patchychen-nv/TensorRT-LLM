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
    parsed = _RUNNER._parse_mutations(["zero", "foreign:0.25", "zero:1.0:3"])
    assert parsed == {
        "zero": {"kind": "zero", "fraction": 1.0},
        "foreign-0.25": {"kind": "foreign", "fraction": 0.25},
        "zero-1.0-owner3": {"kind": "zero", "fraction": 1.0, "owners": [3]},
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


def _save_run(
    root: Path, name: str, logits: list[torch.Tensor], cached: list[int] | None = None
) -> None:
    (root / name).mkdir(parents=True, exist_ok=True)
    tokens = [[int(row.argmax())] for row in logits]
    outputs = {"tokens": tokens, "logits": logits}
    if cached is not None:
        outputs["cached"] = cached
    torch.save(outputs, root / name / "outputs.pt")


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
    assert judgment["failing_requests"] == judgment["far_requests"] == [2]
    assert judgment["gate_error"] is not None
    assert [record["corrupted_bytes"] for record in judgment["records"]] == [96]


@pytest.mark.parametrize("policy_name", ["EXACT_POLICY", "NOISY_POLICY"])
def test_a_request_whose_logits_turned_into_nan_is_reported_by_the_judge(
    tmp_path: Path, policy_name: str
) -> None:
    """Corrupted KV may leave NaN in the logits, which is the corruption noticed."""
    clean = [_logits(seed) for seed in range(_REQUESTS)]
    _save_run(tmp_path, "adp-control", clean)
    _save_run(tmp_path, "adp-replay", clean)
    corrupted = [value.clone() for value in clean]
    corrupted[2] = torch.full_like(clean[2], float("nan"))
    _save_run(tmp_path, "mutated", corrupted)
    _write_records(tmp_path, "mutated")
    judgment = _RUNNER.judge_mutation(tmp_path, "mutated", getattr(_RUNNER, policy_name))
    assert judgment["failing_requests"] == judgment["far_requests"] == [2]
    assert judgment["gate_error"] is not None


def test_a_clean_run_is_accepted_by_the_judge(tmp_path: Path) -> None:
    clean = [_logits(seed) for seed in range(_REQUESTS)]
    for name in ("adp-control", "adp-replay", "same"):
        _save_run(tmp_path, name, clean)
    (tmp_path / "same" / "mutations").mkdir()
    judgment = _RUNNER.judge_mutation(tmp_path, "same", _RUNNER.EXACT_POLICY)
    assert judgment == {
        "gate_error": None,
        "failing_requests": [],
        "far_requests": [],
        "records": [],
    }


# The lifecycle traces of the context group: what each rank records for each request.

_BASELINE_PAGES = [[10, 20], [5]]
_IN_FLIGHT_PAGES = [[9, 20], [5]]
_TIMEOUT_MS = 1000
# What the second prompt of a pair has to find in the cache of a run that reuses prefixes.
_HIT_TOKENS = 384
_CONTROL, _ITERATION = 5, 7
# Four requests before the injections, then the timeout and the cancellation, then two more.
_CONTEXT_IDS = [100, 101, 102, 103, 104, 105, 106, 107]
_TIMEOUT_IDS = [104, 105]


def _row(event: str, rank: int, **values) -> dict:
    return {
        "event": event,
        "rank": rank,
        "iteration": _ITERATION,
        "control": _CONTROL,
        "in_control": False,
        **values,
    }


def _rank_rows(
    rank: int,
    layout: str,
    group_size: int,
    observed_by_rank: dict,
    *,
    context_ids: list[int] = _CONTEXT_IDS,
    timeout_ids: list[int] = _TIMEOUT_IDS,
    freed_pages: list[list[int]] = _BASELINE_PAGES,
) -> list[dict]:
    """The rows one rank writes for every request, in the order the executor records them."""
    rows = [_row("baseline", rank, index_used=3, free_pages=_BASELINE_PAGES)]
    for request_id in context_ids:
        owner = request_id % group_size
        sends = layout == "layer_split" or rank == owner
        timed_out = request_id in timeout_ids
        outcome = observed_by_rank.get(
            (request_id, rank), "timed_out" if timed_out else "completed"
        )
        committed = "timed_out" if timed_out else "completed"
        held = {"request_id": request_id, "index_used": 3, "free_pages": _IN_FLIGHT_PAGES}
        rows.append(_row("release_index", rank, **held))
        rows.append(_row("start", rank, owner=owner, **held))
        if sends:
            rows.append(_row("send", rank, request_id=request_id, owner=owner))
            rows.append(
                _row(
                    "observed",
                    rank,
                    request_id=request_id,
                    owner=owner,
                    outcome=outcome,
                    elapsed_ms=_TIMEOUT_MS + 10 if outcome == "timed_out" else 12.0,
                    # A report reaches the others in the next control exchange, or in the one it
                    # was made in; the request is committed in the exchange of the last report.
                    control=_CONTROL - 1 + rank % 2,
                    in_control=bool(rank % 2),
                )
            )
        rows.append(_row("commit", rank, request_id=request_id, owner=owner, outcome=committed))
        rows.append(
            _row(
                "free",
                rank,
                request_id=request_id,
                owner=owner,
                index_used=3,
                free_pages=freed_pages,
                window="control",
                in_control=True,
            )
        )
        if rank == owner:
            rows.append(_row("response", rank, request_id=request_id, error=None))
    if timeout_ids:
        rows.append(_row("cancel_attempt", rank, request_id=timeout_ids[-1], accepted=False))
    return rows


def _write_traces(
    directory: Path,
    layout: str,
    *,
    group_size: int = 2,
    observed_by_rank: dict | None = None,
    edit=None,
    injections: bool = True,
    reuse: bool = False,
) -> None:
    """A work directory whose context ranks traced a run; ``edit`` may change the rows first.

    A run without injections has no timeout and no cancellation. A run that reuses prefixes ends
    its requests with some pages in the cache, so fewer than at the start are free.
    """
    context_ids = _CONTEXT_IDS if injections else [i for i in _CONTEXT_IDS if i not in _TIMEOUT_IDS]
    timeout_ids = _TIMEOUT_IDS if injections else []
    rows = {
        rank: _rank_rows(
            rank,
            layout,
            group_size,
            observed_by_rank or {},
            context_ids=context_ids,
            timeout_ids=timeout_ids,
            freed_pages=_IN_FLIGHT_PAGES if reuse else _BASELINE_PAGES,
        )
        for rank in range(group_size)
    }
    if edit is not None:
        edit(rows)
    (directory / "traces").mkdir(parents=True)
    for rank, rank_rows in rows.items():
        text = "".join(json.dumps(row) + "\n" for row in rank_rows)
        (directory / "traces" / f"rank-{rank}-1.jsonl").write_text(text)
    (directory / "context-result.json").write_text(
        json.dumps({"context_ids": context_ids, "timeout_ids": timeout_ids, "generated": 6})
    )
    options = _options(
        ctx_group_size=group_size,
        layout=layout,
        transfer_timeout_ms=_TIMEOUT_MS,
        injections=injections,
        reuse=reuse,
        reuse_hit_tokens=_HIT_TOKENS,
    )
    options["prompts"] = None
    (directory / "options.json").write_text(json.dumps(options))


def test_the_traces_of_a_replicated_group_have_one_sender_per_request(tmp_path: Path) -> None:
    _write_traces(tmp_path, "replicated")
    summary = _RUNNER.validate_transfer_traces(tmp_path)
    assert summary["layout"] == "replicated"
    assert summary["sends"] == len(_CONTEXT_IDS)
    assert summary["unique_raw_responses"] == len(_CONTEXT_IDS)


@pytest.mark.parametrize("group_size", [2, 4])
def test_the_traces_of_a_layer_split_group_have_a_sender_on_every_rank(
    tmp_path: Path, group_size: int
) -> None:
    _write_traces(tmp_path, "layer_split", group_size=group_size)
    summary = _RUNNER.validate_transfer_traces(tmp_path)
    assert summary["layout"] == "layer_split"
    assert summary["sends"] == group_size * len(_CONTEXT_IDS)
    # One response per request, from the rank that computed it.
    assert summary["unique_raw_responses"] == len(_CONTEXT_IDS)


def test_a_rank_that_has_not_timed_out_yet_may_report_the_cancellation_of_its_send(
    tmp_path: Path,
) -> None:
    observed = {(104, 1): "failed", (105, 0): "failed"}
    _write_traces(tmp_path, "layer_split", observed_by_rank=observed)
    assert _RUNNER.validate_transfer_traces(tmp_path)["layout"] == "layer_split"


def _drop(event: str, rank: int, request_id: int):
    def edit(rows: dict) -> None:
        rows[rank] = [
            row
            for row in rows[rank]
            if not (row["event"] == event and row.get("request_id") == request_id)
        ]

    return edit


def _change(event: str, in_file_of: int, request_id: int, **values):
    """Edit the row of ``event`` for ``request_id`` in the trace file of rank ``in_file_of``."""

    def edit(rows: dict) -> None:
        for row in rows[in_file_of]:
            if row["event"] == event and row.get("request_id") == request_id:
                row.update(values)

    return edit


def _add_send(rank: int, request_id: int, owner: int):
    def edit(rows: dict) -> None:
        rows[rank].append(_row("send", rank, request_id=request_id, owner=owner))

    return edit


@pytest.mark.parametrize(
    ("layout", "edit", "observed"),
    [
        # A layer-split rank that does not send its layers, or does not see its send end.
        ("layer_split", _drop("send", 1, 101), {}),
        ("layer_split", _drop("observed", 0, 102), {}),
        # Two replicas that both send the same request.
        ("replicated", _add_send(0, 101, 1), {}),
        # The response of a request comes from the rank that computed it, and from nobody else.
        ("layer_split", _change("response", 1, 101, rank=0), {}),
        ("layer_split", _drop("response", 1, 101), {}),
        # Every rank commits the same outcome.
        ("layer_split", _change("commit", 1, 103, outcome="failed"), {}),
        # A commit before the last report arrived.
        ("layer_split", _change("commit", 0, 103, control=_CONTROL - 1), {}),
        # A timeout that no rank reported as one: nobody had run out of time when it was cancelled.
        ("layer_split", None, {(104, 0): "failed", (104, 1): "failed"}),
    ],
)
def test_a_trace_that_breaks_the_lifecycle_of_the_layout_is_rejected(
    tmp_path: Path, layout: str, edit, observed: dict
) -> None:
    _write_traces(tmp_path, layout, observed_by_rank=observed, edit=edit)
    with pytest.raises(AssertionError):
        _RUNNER.validate_transfer_traces(tmp_path)


def test_a_replicated_trace_is_not_a_layer_split_trace(tmp_path: Path) -> None:
    """The ranks that do not own a request send nothing in a replicated group."""
    _write_traces(tmp_path, "replicated")
    options = json.loads((tmp_path / "options.json").read_text())
    (tmp_path / "options.json").write_text(json.dumps({**options, "layout": "layer_split"}))
    with pytest.raises(AssertionError):
        _RUNNER.validate_transfer_traces(tmp_path)


@pytest.mark.parametrize("layout", ["replicated", "layer_split"])
def test_a_run_without_injections_has_no_timeout_and_no_cancellation(
    tmp_path: Path, layout: str
) -> None:
    _write_traces(tmp_path, layout, injections=False)
    summary = _RUNNER.validate_transfer_traces(tmp_path)
    assert summary["timeout_ids"] == [] and summary["cancel_request_id"] is None
    assert summary["unique_raw_responses"] == summary["context_requests"] == 6


def test_a_run_that_injects_nothing_may_not_time_out(tmp_path: Path) -> None:
    _write_traces(tmp_path, "layer_split", injections=False)
    result = json.loads((tmp_path / "context-result.json").read_text())
    result["timeout_ids"] = [100]
    (tmp_path / "context-result.json").write_text(json.dumps(result))
    with pytest.raises(AssertionError):
        _RUNNER.validate_transfer_traces(tmp_path)


@pytest.mark.parametrize("layout", ["replicated", "layer_split"])
def test_a_run_that_reuses_prefixes_ends_its_requests_with_pages_in_the_cache(
    tmp_path: Path, layout: str
) -> None:
    _write_traces(tmp_path, layout, injections=False, reuse=True)
    assert _RUNNER.validate_transfer_traces(tmp_path)["layout"] == layout


def test_without_reuse_every_request_brings_the_pools_back_to_the_start(tmp_path: Path) -> None:
    """The pages that a run with reuse keeps in its cache are pages that a run without reuse leaked."""
    _write_traces(tmp_path, "layer_split", injections=False, reuse=True)
    options = json.loads((tmp_path / "options.json").read_text())
    (tmp_path / "options.json").write_text(json.dumps({**options, "reuse": False}))
    with pytest.raises(AssertionError):
        _RUNNER.validate_transfer_traces(tmp_path)


def test_a_cache_cannot_give_a_request_more_free_pages_than_the_pools_started_with(
    tmp_path: Path,
) -> None:
    more = _change("free", 0, 100, free_pages=[[11, 20], [5]])
    _write_traces(tmp_path, "layer_split", injections=False, reuse=True, edit=more)
    with pytest.raises(AssertionError):
        _RUNNER.validate_transfer_traces(tmp_path)


def test_only_the_second_prompt_of_a_pair_finds_the_prefix_in_the_cache() -> None:
    summary = _RUNNER.check_prefix_hits([0, 384, 0, 384, 0, 512], 384)
    assert summary["minimum_hit"] == 384
    for cached in ([0, 384, 128, 384], [0, 100, 0, 384], [0, 384, 0], [0, 0, 0, 0], []):
        with pytest.raises(AssertionError):
            _RUNNER.check_prefix_hits(cached, 384)


def test_a_reuse_run_compares_the_pairs_of_an_even_number_of_prompts(tmp_path: Path) -> None:
    for prompts in (None, [], [[1], [1, 2], [3]]):
        with pytest.raises(ValueError, match="pairs"):
            _RUNNER.run_transfer_gate("model", str(tmp_path), prompts=prompts, reuse_hit_tokens=384)
    with pytest.raises(ValueError, match="pairs"):
        _RUNNER.run_transfer_gate(
            "model", str(tmp_path), mode="lifecycle", prompts=[[1], [1, 2]], reuse_hit_tokens=384
        )


def _judge_a_reuse_run(tmp_path: Path, *, damaged: list[int], cached: list[int]) -> dict:
    """Three pairs: the controls are clean, and the requests in ``damaged`` differ in the DKV run."""
    _write_traces(tmp_path / "dkv", "layer_split", injections=False, reuse=True)
    clean = [_logits(seed) for seed in range(6)]
    for name in ("adp-control", "adp-replay"):
        _save_run(tmp_path, name, clean)
    actual = [
        _logits(100 + index) if index in damaged else value for index, value in enumerate(clean)
    ]
    _save_run(tmp_path, "dkv", actual, cached)
    return _RUNNER.judge_transfer_gate(tmp_path, mode="precision", policy=_RUNNER.EXACT_POLICY)


def test_a_reuse_run_is_judged_on_the_second_prompt_of_each_pair(tmp_path: Path) -> None:
    # The first prompt of a pair only fills the cache: its answer is not what the gate looks at.
    summary = _judge_a_reuse_run(tmp_path, damaged=[0, 2, 4], cached=[0, 384, 0, 384, 0, 384])
    assert summary["reuse"]["minimum_hit"] == _HIT_TOKENS
    assert summary["precision"]["status"] == "passed" and summary["precision"]["requests"] == 3


def test_a_reuse_run_fails_when_the_prompt_that_hits_the_cache_is_answered_wrongly(
    tmp_path: Path,
) -> None:
    with pytest.raises(AssertionError):
        _judge_a_reuse_run(tmp_path, damaged=[3], cached=[0, 384, 0, 384, 0, 384])


def test_a_reuse_run_fails_when_a_second_prompt_found_nothing_in_the_cache(tmp_path: Path) -> None:
    with pytest.raises(AssertionError, match="No prefix hit"):
        _judge_a_reuse_run(tmp_path, damaged=[], cached=[0, 384, 0, 0, 0, 384])
