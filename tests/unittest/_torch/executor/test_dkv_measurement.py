# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence
from copy import deepcopy
from types import SimpleNamespace

import pytest

from tensorrt_llm._torch.pyexecutor.dkv_metrics import (
    DkvMeasurementCounters,
    build_dkv_measurement_report,
)
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import (
    KVCacheManagerV2,
    _dkv_group_size_of,
    _dkv_measurement_of,
    _measured_dkv_context_operation,
)
from tensorrt_llm.runtime.kv_cache_manager_v2 import KVCacheEventManager

pytestmark = pytest.mark.cpu_only


def _record(
    counters: DkvMeasurementCounters,
    *,
    operation: str = "prepare_context",
    first: bool = True,
    match: int = 32,
    success: bool = True,
    raised: bool = False,
) -> None:
    counters.record_operation(
        operation,
        prompt_tokens=65,
        first_chunk=first,
        matched_tokens=match,
        success=success,
        raised=raised,
    )


def _snapshot(rank: int, *, dkv: bool = True, capacity: int = 8, held: int | None = None) -> dict:
    """A rank's snapshot; ``held`` pages (a DKV replica's resident dummies) are not usable."""
    held = (2 if dkv else 0) if held is None else held
    counters = DkvMeasurementCounters()
    _record(counters)
    counters.record_scheduled_context(33)
    return {
        "rank": rank,
        "group_size": 2,
        "dkv_enabled": dkv,
        "reuse_enabled": True,
        "tokens_per_block": 32,
        "cache_tiers": ["gpu"],
        "pools_by_level": [
            [
                {
                    "slot_sizes": [128],
                    "total": capacity,
                    "free": capacity - held,
                    "evictable": 0,
                    "available": capacity - held,
                }
            ]
        ],
        "fresh_page_fill": "none",
        "counters": counters.snapshot(),
        "last_tier_capacity_dropped_pages": 0,
        "gpu_offloaded_pages": 0,
        "iteration_stats_observations": 1,
        "reuse_resets": 0,
        "operation_generation": 1,
        "stats_generation": 1,
        "event_next_id": 2,
    }


def _event(rank: int, event_id: int, kind: str, hashes: Sequence[int] = ()) -> dict:
    data = {"type": kind}
    if kind == "stored":
        data["blocks"] = [{"block_hash": value} for value in hashes]
    elif kind == "removed":
        data["block_hashes"] = list(hashes)
    return {
        "attention_dp_rank": rank,
        "event_id": event_id,
        "window_size": 1024,
        "layer_group_id": 0,
        "hash_algo": "sha256",
        "data": data,
    }


def _events() -> list[dict]:
    return [
        event
        for rank in range(2)
        for event in (_event(rank, 0, "created"), _event(rank, 1, "stored", [11, 12]))
    ]


def _report(rows: list[dict] | None = None, events: list[dict] | None = None, **kwargs) -> dict:
    return build_dkv_measurement_report(
        rows if rows is not None else [_snapshot(0), _snapshot(1)],
        events if events is not None else _events(),
        dkv_enabled=True,
        **kwargs,
    )


def test_prepare_counts_first_success_and_retries_separately() -> None:
    counters = DkvMeasurementCounters()
    _record(counters, success=False)
    _record(counters)
    _record(counters, first=False)
    _record(counters, operation="resize_context", success=False)
    _record(counters, operation="resize_context", raised=True)
    actual = counters.snapshot()
    assert actual["request_count"] == 1
    assert actual["matched_prefix_tokens"] == 32
    assert actual["eligible_prefix_tokens"] == 64
    assert actual["prepare_attempts"] == 3
    assert actual["prepare_failures"] == 1
    assert actual["resize_failures"] == 1
    assert actual["resize_errors"] == 1


def test_match_reserves_final_prompt_token() -> None:
    counters = DkvMeasurementCounters()
    _record(counters, match=65)
    assert counters.snapshot()["matched_prefix_tokens"] == 64


def test_storage_delta_snapshot_does_not_consume_counters() -> None:
    counters = DkvMeasurementCounters()
    counters.record_storage_delta(3, 2)
    assert counters.snapshot() == counters.snapshot()
    counters.record_storage_delta(4, 1)
    assert counters.last_tier_capacity_dropped_pages == 7
    assert counters.gpu_offloaded_pages == 3
    assert counters.iteration_stats_observations == 2


@pytest.mark.parametrize("event_limit", [1, 16])
def test_native_event_flush_observer_survives_trim_and_empty_flush(event_limit: int) -> None:
    counters = DkvMeasurementCounters()
    gathered = []

    def gather(events: list) -> list:
        gathered.append(events)
        return [events]

    manager = KVCacheEventManager(
        event_limit,
        attention_dp_rank=0,
        attention_dp_gather=counters.wrap_event_gather(gather),
    )
    assert counters.event_next_id is None
    for _ in range(3):
        manager.add_created_event([16])
    counters.operation_generation += 1
    manager.flush_iteration_events()
    assert counters.event_next_id == 3
    assert counters.event_generation == 1
    assert len(gathered[-1]) == min(3, event_limit)
    assert [event.event_id for event in gathered[-1]] == list(range(3 - min(3, event_limit), 3))
    counters.operation_generation += 1
    manager.flush_iteration_events()
    assert gathered[-1] == []
    assert counters.event_next_id == 3
    assert counters.event_generation == 2


def test_failed_event_collective_does_not_publish_a_watermark() -> None:
    counters = DkvMeasurementCounters()

    def gather(events: list) -> list:
        raise RuntimeError("collective failed")

    with pytest.raises(RuntimeError, match="collective failed"):
        counters.wrap_event_gather(gather)([SimpleNamespace(event_id=2)])
    assert counters.event_next_id is None
    assert counters.event_generation == -1


def test_owner_aggregation_and_logical_residency() -> None:
    report = _report()
    assert report["global"]["request_count"] == 2
    assert report["global"]["matched_prefix_tokens"] == 64
    assert report["global"]["request_prefix_hit_rate"] == 1
    assert report["global"]["token_prefix_hit_rate"] == 0.5
    assert report["global"]["mean_matched_prefix_tokens"] == 32
    assert report["storage"]["resident_block_copies"] == 4
    assert report["storage"]["unique_resident_blocks"] == 2
    assert report["storage"]["duplicate_storage_ratio"] == 0.5
    assert report["validity"]["prefix_hit_rate_proxy_valid"]


def test_token_hit_rate_is_weighted_globally() -> None:
    rows = [_snapshot(0), _snapshot(1)]
    rows[0]["counters"]["eligible_prefix_tokens"] = 128
    report = _report(rows)
    assert report["global"]["token_prefix_hit_rate"] == pytest.approx(64 / 192)
    assert report["per_rank"][0]["token_prefix_hit_rate"] == 0.25


def test_local_load_variance_excludes_replica_counts() -> None:
    rows = [_snapshot(0), _snapshot(1)]
    rows[1]["counters"]["scheduled_context_tokens"] = 1
    load = _report(rows)["load"]
    assert load["interval_total_token_variance"] == 256
    assert load["scheduled_context_tokens_by_rank"] == [33, 1]
    assert load["per_iteration_token_variance"] is None


def _stats_row(iteration: int, rank: int, tokens: int) -> dict:
    return {
        "iter": iteration,
        "attentionDpRank": rank,
        "inflightBatchingStats": {"numCtxTokens": tokens},
    }


def test_load_imbalance_is_measured_inside_each_iteration_and_then_averaged() -> None:
    stats = [
        _stats_row(1, 0, 8),
        _stats_row(1, 1, 0),
        _stats_row(2, 0, 4),
        _stats_row(2, 1, 4),
        _stats_row(3, 0, 9),  # rank 1 reported nothing for this iteration
        {"iter": 4},  # a row without per-rank batching stats
    ]
    load = _report(iteration_stats=stats)["load"]
    assert load["iterations_measured"] == 2
    assert load["per_iteration_token_variance"] == 8
    assert load["per_iteration_token_imbalance"] == 4
    assert "numCtxTokens" in load["per_iteration_source"]
    # The same totals can hide a persistent imbalance that the interval variance reports as zero.
    assert load["interval_total_token_variance"] == 0


def test_load_imbalance_is_unknown_without_a_complete_iteration() -> None:
    load = _report(iteration_stats=[_stats_row(1, 0, 8)])["load"]
    assert load["per_iteration_token_variance"] is None
    assert load["iterations_measured"] is None


def test_report_uses_the_second_schema_and_names_its_scopes() -> None:
    report = _report()
    assert report["schema_version"] == 2
    assert "scheduled_context_token_rank_variance" not in report["global"]
    assert report["storage"]["tier_scope"] == {
        "tiers": ["gpu"],
        "hit_rate_includes_secondary_tiers": False,
    }
    assert "usable pages" in report["semantics"]["capacity"]
    assert "physical" in report["semantics"]["physical_vs_logical"]


def test_secondary_tiers_are_named_in_the_hit_rate_scope() -> None:
    rows = [_snapshot(0), _snapshot(1)]
    for row in rows:
        row["cache_tiers"] = ["gpu", "host"]
        row["pools_by_level"].append(row["pools_by_level"][0])
    scope = _report(rows)["storage"]["tier_scope"]
    assert scope["tiers"] == ["gpu", "host"]
    assert scope["hit_rate_includes_secondary_tiers"]


def _removal_events(removing_ranks: Sequence[int]) -> list[dict]:
    return [*_events(), *(_event(rank, 2, "removed", [11]) for rank in removing_ranks)]


@pytest.mark.parametrize("dkv", [True, False])
def test_removals_are_reported_physically_and_logically(dkv: bool) -> None:
    rows = [_snapshot(0, dkv=dkv), _snapshot(1, dkv=dkv)]
    for row in rows:
        row["event_next_id"] = 3
    report = build_dkv_measurement_report(rows, _removal_events([0, 1]), dkv_enabled=dkv)
    storage = report["storage"]
    assert storage["removed_block_copies"] == 2
    # Replicas remove the same block once each, so one logical removal; ADP ranks are independent.
    assert storage["removed_blocks_logical"] == (1 if dkv else 2)


def test_logical_counts_are_unknown_when_replicas_disagree() -> None:
    rows = [_snapshot(0), _snapshot(1)]
    for row in rows:
        row["event_next_id"] = 3
    rows[0]["last_tier_capacity_dropped_pages"] = 1
    storage = build_dkv_measurement_report(rows, _removal_events([0]), dkv_enabled=True)["storage"]
    assert storage["removed_blocks_logical"] is None
    assert storage["last_tier_capacity_dropped_pages_logical"] is None
    assert storage["last_tier_capacity_dropped_pages"] == 1


def test_unobserved_prefix_and_empty_residency_are_null() -> None:
    rows = [_snapshot(0), _snapshot(1)]
    for row in rows:
        row["counters"] = DkvMeasurementCounters().snapshot()
        row["event_next_id"] = 1
    report = _report(rows, [_event(0, 0, "created"), _event(1, 0, "created")])
    assert report["global"]["request_prefix_hit_rate"] is None
    assert report["global"]["token_prefix_hit_rate"] is None
    assert report["storage"]["duplicate_storage_ratio"] is None
    assert not report["validity"]["prefix_hit_rate_proxy_valid"]


@pytest.mark.parametrize(
    "fault", ["missing_rank", "gap", "duplicate", "tail", "missing_created", "mixed_hash"]
)
def test_event_loss_or_ambiguity_invalidates_residency(fault: str) -> None:
    events = _events()
    if fault == "missing_rank":
        events = events[:2]
    elif fault == "gap":
        events[1]["event_id"] = 2
    elif fault == "duplicate":
        events.append(deepcopy(events[-1]))
    elif fault == "tail":
        events.pop()
    elif fault == "missing_created":
        events.pop(0)
    else:
        events[-1]["hash_algo"] = "different"
    report = _report(events=events)
    assert not report["storage"]["complete"]
    assert report["storage"]["duplicate_storage_ratio"] is None
    assert report["storage"]["removed_block_copies"] is None
    assert not report["validity"]["prefix_hit_rate_proxy_valid"]


def test_removal_proxy_is_not_physical_drop_count() -> None:
    rows = [_snapshot(0), _snapshot(1)]
    rows[0]["event_next_id"] = 3
    rows[0]["last_tier_capacity_dropped_pages"] = 3
    events = [*_events(), _event(0, 2, "removed", [11])]
    report = _report(rows, events)
    assert report["storage"]["removed_block_copies"] == 1
    assert report["storage"]["last_tier_capacity_dropped_pages"] == 3
    assert report["storage"]["resident_block_copies"] == 3
    assert not report["validity"]["prefix_hit_rate_proxy_valid"]


def _equal_usable_reference() -> list[dict]:
    """ADP ranks with 3 usable pages each match one DKV replica of 6 usable pages (8 minus 2)."""
    return [_snapshot(0, dkv=False, capacity=3), _snapshot(1, dkv=False, capacity=3)]


def test_equal_capacity_counts_usable_pages_despite_removals() -> None:
    rows = [_snapshot(0), _snapshot(1)]
    rows[0]["last_tier_capacity_dropped_pages"] = 2
    report = _report(rows, adp_capacity_reference=_equal_usable_reference())
    assert report["validity"]["equal_capacity_verified"]
    assert report["validity"]["prefix_hit_rate_proxy_valid"]
    pool = report["capacity_comparison"]["pools"][0]
    assert pool["usable_gap_pages"] == 0
    # The replica's resident dummies make its pool larger than its usable capacity.
    assert pool["total_gap_pages"] == -2
    assert not report["validity"]["equal_total_pages"]


def test_equal_total_pages_do_not_prove_equal_capacity() -> None:
    reference = [_snapshot(0, dkv=False, capacity=4), _snapshot(1, dkv=False, capacity=4)]
    report = _report(adp_capacity_reference=reference)
    assert report["validity"]["equal_total_pages"]
    assert not report["validity"]["equal_capacity_verified"]
    assert report["capacity_comparison"]["pools"][0]["usable_gap_pages"] == 2


def test_declared_capacity_tolerance_accepts_a_known_gap() -> None:
    reference = [_snapshot(0, dkv=False, capacity=4), _snapshot(1, dkv=False, capacity=4)]
    assert _report(adp_capacity_reference=reference, capacity_tolerance_pages=2)["validity"][
        "equal_capacity_verified"
    ]
    assert not _report(adp_capacity_reference=reference, capacity_tolerance_pages=1)["validity"][
        "equal_capacity_verified"
    ]


def test_per_rank_pool_size_differences_between_modes_are_flagged() -> None:
    comparison = _report(adp_capacity_reference=_equal_usable_reference())["capacity_comparison"]
    pool = comparison["pools"][0]
    assert not pool["same_pages_per_rank"]
    assert pool["dkv_replica_total_pages"] == 8
    assert pool["adp_total_pages_by_rank"] == [3, 3]


@pytest.mark.parametrize(
    "fault", ["capacity", "slot_size", "tier", "block_size", "rank", "pool_count"]
)
def test_equal_capacity_rejects_incomparable_layouts(fault: str) -> None:
    reference = _equal_usable_reference()
    if fault == "capacity":
        reference[0]["pools_by_level"][0][0]["total"] = 8
        reference[0]["pools_by_level"][0][0]["available"] = 8
    elif fault == "slot_size":
        reference[0]["pools_by_level"][0][0]["slot_sizes"] = [64]
    elif fault == "tier":
        reference[0]["cache_tiers"] = ["host"]
    elif fault == "block_size":
        reference[0]["tokens_per_block"] = 64
    elif fault == "rank":
        reference[1]["rank"] = 0
    else:
        reference[0]["pools_by_level"][0].append(reference[0]["pools_by_level"][0][0])
    assert not _report(adp_capacity_reference=reference)["validity"]["equal_capacity_verified"]


def test_interval_preserves_residency_and_subtracts_counters() -> None:
    before = [_snapshot(0), _snapshot(1)]
    after = deepcopy(before)
    for row in after:
        row["counters"]["request_count"] += 1
        row["iteration_stats_observations"] += 1
    report = _report(after, baseline_snapshots=before, baseline_event_count=4)
    assert report["global"]["request_count"] == 2
    assert report["global"]["matched_prefix_tokens"] == 0
    assert report["storage"]["resident_block_copies"] == 4
    assert report["validity"]["prefix_hit_rate_proxy_valid"]


def test_future_events_do_not_cross_snapshot_watermark() -> None:
    events = [*_events(), _event(0, 2, "stored", [13]), _event(0, 3, "removed", [11])]
    report = _report(events=events)
    assert report["storage"]["resident_block_copies"] == 4
    assert report["storage"]["removed_block_copies"] == 0


def test_event_offset_cannot_hide_removals_without_counter_baseline() -> None:
    with pytest.raises(ValueError, match="matching counter snapshots"):
        _report(baseline_event_count=3)


def test_stale_native_stats_invalid_even_with_verified_equal_capacity() -> None:
    rows = [_snapshot(0), _snapshot(1)]
    rows[1]["operation_generation"] += 1
    report = _report(rows, adp_capacity_reference=_equal_usable_reference())
    assert report["validity"]["equal_capacity_verified"]
    assert not report["validity"]["prefix_hit_rate_proxy_valid"]


@pytest.mark.parametrize("start", [None, -1, 3])
def test_invalid_start_watermark_without_removals_is_not_complete(start: int | None) -> None:
    before = [_snapshot(0), _snapshot(1)]
    after = deepcopy(before)
    before[0]["event_next_id"] = start
    for row in after:
        row["counters"]["request_count"] += 1
        row["iteration_stats_observations"] += 1
    report = _report(after, baseline_snapshots=before)
    assert not report["storage"]["complete"]
    assert not report["validity"]["prefix_hit_rate_proxy_valid"]


@pytest.mark.parametrize(
    "fault",
    ["reuse_off", "prepare_failure", "resize_error", "unknown_drop", "no_stats_progress", "reset"],
)
def test_invalid_measurement_controls(fault: str) -> None:
    before = [_snapshot(0), _snapshot(1)]
    after = deepcopy(before)
    for row in after:
        row["counters"]["request_count"] += 1
        row["iteration_stats_observations"] += 1
    if fault == "reuse_off":
        after[0]["reuse_enabled"] = False
    elif fault == "prepare_failure":
        after[0]["counters"]["prepare_failures"] += 1
    elif fault == "resize_error":
        after[0]["counters"]["resize_errors"] += 1
    elif fault == "unknown_drop":
        after[0]["last_tier_capacity_dropped_pages"] = None
    elif fault == "no_stats_progress":
        after[0]["iteration_stats_observations"] -= 1
    else:
        after[0]["reuse_resets"] += 1
    report = _report(after, baseline_snapshots=before, baseline_event_count=4)
    assert not report["validity"]["prefix_hit_rate_proxy_valid"]


@pytest.mark.parametrize(
    "fault", ["missing_rank", "duplicate_rank", "mode", "counter_reset", "capacity_change"]
)
def test_bad_snapshot_sets_fail_explicitly(fault: str) -> None:
    before = [_snapshot(0), _snapshot(1)]
    rows = deepcopy(before)
    if fault == "missing_rank":
        rows.pop()
    elif fault == "duplicate_rank":
        rows[1]["rank"] = 0
    elif fault == "mode":
        rows[0]["dkv_enabled"] = False
    elif fault == "counter_reset":
        rows[0]["counters"]["request_count"] = 0
    else:
        rows[0]["pools_by_level"][0][0]["total"] += 1
    with pytest.raises(ValueError):
        _report(rows, baseline_snapshots=before)


def test_page_fill_marks_timing_and_never_asserts_output_correctness() -> None:
    rows = [_snapshot(0), _snapshot(1)]
    rows[0]["fresh_page_fill"] = "zero"
    validity = _report(rows)["validity"]
    assert validity["timing_perturbed_by_fresh_page_fill"]
    assert not validity["output_correctness_validated"]


def _manager() -> KVCacheManagerV2:
    manager = KVCacheManagerV2.__new__(KVCacheManagerV2)
    manager._dkv_measurement = DkvMeasurementCounters()
    manager.is_draft = False
    manager.dkv_group_size = 2
    manager.mapping = SimpleNamespace(tp_rank=0)
    manager.kv_cache_map = {1: SimpleNamespace(num_committed_tokens=32)}
    return manager


def _request(owner: int = 0, dummy: bool = False) -> SimpleNamespace:
    return SimpleNamespace(
        py_request_id=1,
        prompt_len=65,
        is_first_context_chunk=True,
        py_dkv_measurement_prepared=False,
        py_dkv_compute_rank=owner,
        is_dummy_request=dummy,
    )


def test_manager_hook_deduplicates_request_repreparation_without_retaining_ids() -> None:
    manager = _manager()
    request = _request()

    def prepare_context(req: SimpleNamespace) -> bool:
        return True

    hooked = _measured_dkv_context_operation(manager, prepare_context)
    assert hooked(request)
    assert hooked(request)
    assert request.py_dkv_measurement_prepared
    counters = manager._dkv_measurement.snapshot()
    assert counters["request_count"] == 1
    assert counters["prepare_attempts"] == 2


@pytest.mark.parametrize("owner,dummy", [(1, False), (0, True)])
def test_manager_hook_excludes_replica_and_dummy_work(owner: int, dummy: bool) -> None:
    manager = _manager()

    def prepare_context(req: SimpleNamespace) -> bool:
        return True

    hooked = _measured_dkv_context_operation(manager, prepare_context)
    assert hooked(_request(owner, dummy))
    assert manager._dkv_measurement.snapshot()["prepare_attempts"] == 0


def test_manager_hook_preserves_failure_and_exception_contract() -> None:
    manager = _manager()
    request = _request()

    def prepare_context(req: SimpleNamespace) -> bool:
        return False

    assert not _measured_dkv_context_operation(manager, prepare_context)(request)
    assert not request.py_dkv_measurement_prepared

    def resize_context(req: SimpleNamespace) -> bool:
        raise RuntimeError("allocation failed")

    with pytest.raises(RuntimeError, match="allocation failed"):
        _measured_dkv_context_operation(manager, resize_context)(request)
    assert manager._dkv_measurement.snapshot()["resize_errors"] == 1


def test_hooks_wrap_only_the_instance_that_enables_them() -> None:
    manager = _manager()
    other = KVCacheManagerV2.__new__(KVCacheManagerV2)
    manager._dkv_trace_enabled = False
    manager._install_dkv_hooks()
    assert "prepare_context" in vars(manager)
    assert "prepare_context" not in vars(other)
    # The class keeps the plain function, so borrowed methods run without any hook.
    assert not hasattr(KVCacheManagerV2.prepare_context, "__wrapped__")


def test_dkv_state_reads_as_off_for_partial_managers_and_spec_doubles() -> None:
    from unittest.mock import Mock

    for double in (
        Mock(spec=KVCacheManagerV2),
        SimpleNamespace(),
        KVCacheManagerV2.__new__(KVCacheManagerV2),
    ):
        assert _dkv_group_size_of(double) is None
        assert _dkv_measurement_of(double) is None
    manager = _manager()
    assert _dkv_group_size_of(manager) == 2
    assert _dkv_measurement_of(manager) is manager._dkv_measurement
