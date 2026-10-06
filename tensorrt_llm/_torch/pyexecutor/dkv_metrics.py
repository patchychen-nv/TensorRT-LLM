# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Opt-in replicated-KV experiment counters and offline measurement reports.

These reports are internal experiment artifacts, not telemetry or public iteration
stats. Event residency describes committed logical blocks across all cache tiers;
it does not assert that a replica contains valid KV tensor data.
"""

from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from statistics import pvariance


@dataclass
class _Counters:
    request_count: int = 0
    prompt_tokens: int = 0
    eligible_prefix_tokens: int = 0
    matched_prefix_tokens: int = 0
    requests_with_prefix_hit: int = 0
    prepare_attempts: int = 0
    prepare_failures: int = 0
    prepare_errors: int = 0
    resize_attempts: int = 0
    resize_failures: int = 0
    resize_errors: int = 0
    scheduled_context_tokens: int = 0
    scheduled_context_chunks: int = 0


class DkvMeasurementCounters:
    """Executor-thread counters; each logical request is charged to its owner.

    Prefix lengths are observed at the first successful context preparation, before
    forward. A scheduler retry neither contributes another request nor another hit.
    Operation failures count attempts, including retries, rather than failed requests.
    """

    def __init__(self) -> None:
        self._counters = _Counters()
        self.last_tier_capacity_dropped_pages = 0
        self.gpu_offloaded_pages = 0
        self.iteration_stats_observations = 0
        self.reuse_resets = 0
        self.operation_generation = 0
        self.stats_generation = -1
        self.event_next_id: int | None = None
        self.event_generation = -1

    def record_operation(
        self,
        operation: str,
        *,
        prompt_tokens: int,
        first_chunk: bool,
        matched_tokens: int,
        success: bool,
        raised: bool,
    ) -> None:
        """Observe one owner-local prepare/resize outcome without retaining a request."""
        if operation not in ("prepare_context", "resize_context"):
            raise ValueError(f"Unsupported measurement operation: {operation}")
        if min(prompt_tokens, matched_tokens) < 0:
            raise ValueError("Token counts must be nonnegative")
        kind = "prepare" if operation == "prepare_context" else "resize"
        field = f"{kind}_attempts"
        setattr(self._counters, field, getattr(self._counters, field) + 1)
        if raised or not success:
            field = f"{kind}_errors" if raised else f"{kind}_failures"
            setattr(self._counters, field, getattr(self._counters, field) + 1)
            return
        if kind != "prepare" or not first_chunk:
            return
        eligible = max(prompt_tokens - 1, 0)
        matched = min(matched_tokens, eligible)
        self._counters.request_count += 1
        self._counters.prompt_tokens += prompt_tokens
        self._counters.eligible_prefix_tokens += eligible
        self._counters.matched_prefix_tokens += matched
        self._counters.requests_with_prefix_hit += int(matched > 0)

    def record_scheduled_context(self, tokens: int) -> None:
        """Count an owner-local scheduled chunk, excluding dummy work."""
        if tokens < 0:
            raise ValueError("Scheduled context tokens must be nonnegative")
        self._counters.scheduled_context_tokens += tokens
        self._counters.scheduled_context_chunks += 1

    def snapshot(self) -> dict[str, int]:
        """Copy monotonic counters without consuming executor iteration statistics."""
        return asdict(self._counters)

    def record_storage_delta(self, dropped_pages: int, offloaded_pages: int) -> None:
        """Reuse the executor's consumed raw deltas, without resetting stats again."""
        self.last_tier_capacity_dropped_pages += dropped_pages
        self.gpu_offloaded_pages += offloaded_pages
        self.iteration_stats_observations += 1
        self.stats_generation = self.operation_generation

    def wrap_event_gather(self, gather: Callable[[list], list]) -> Callable[[list], list]:
        """Observe the existing ADP event gather without consuming or adding collectives.

        Both V2 backends trim local events to their newest suffix before this
        callback. Its maximum ID therefore includes any discarded prefix; the
        report detects that loss by checking the complete ID sequence.
        """

        def observed_gather(events: list) -> list:
            result = gather(events)
            self.event_next_id = max(
                [self.event_next_id or 0, *(event.event_id + 1 for event in events)]
            )
            self.event_generation = self.operation_generation
            return result

        return observed_gather


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _prefix_metrics(counters: dict[str, int]) -> dict:
    return {
        **counters,
        "mean_matched_prefix_tokens": _ratio(
            counters["matched_prefix_tokens"], counters["request_count"]
        ),
        "request_prefix_hit_rate": _ratio(
            counters["requests_with_prefix_hit"], counters["request_count"]
        ),
        "token_prefix_hit_rate": _ratio(
            counters["matched_prefix_tokens"], counters["eligible_prefix_tokens"]
        ),
    }


def _pool_signature(pool: dict) -> tuple:
    """A pool of a replicated manager is told by its slot sizes; a pool named by the semantic key of
    its life cycle (page counts fixed per life cycle) by that key, since the slot sizes of the same
    life cycle differ between ranks that hold different layers."""
    if "key" in pool:
        return (tuple(pool["key"]), pool["total"])
    return (tuple(pool["slot_sizes"]), pool["total"])


def _capacity_signature(snapshot: dict) -> tuple:
    return (
        snapshot["tokens_per_block"],
        tuple(snapshot["cache_tiers"]),
        tuple(
            tuple(_pool_signature(pool) for pool in level) for level in snapshot["pools_by_level"]
        ),
    )


def _pool_values(row: dict, key: str) -> list[list[int]]:
    return [[pool[key] for pool in level] for level in row["pools_by_level"]]


def _capacity_comparison(
    dkv_snapshots: Sequence[dict], adp_snapshots: Sequence[dict], tolerance_pages: int
) -> dict | None:
    """Compare the pages a request can use in each mode, or None when the layouts differ.

    One DKV replica holds the whole workload, so it is compared with the pages of all ADP ranks
    together. Usable pages are the free and evictable pages of an idle pool: a replica's resident
    forward dummies and a guard page are subtracted in this way, while a pool's total would count
    them as capacity.
    """
    if not adp_snapshots or len(adp_snapshots) != len(dkv_snapshots):
        return None
    if sorted(row["rank"] for row in adp_snapshots) != list(range(len(dkv_snapshots))):
        return None
    target = dkv_snapshots[0]
    if any(
        not row["dkv_enabled"] or _capacity_signature(row) != _capacity_signature(target)
        for row in dkv_snapshots
    ):
        return None
    if any(
        row["dkv_enabled"]
        or row["cache_tiers"] != target["cache_tiers"]
        or row["tokens_per_block"] != target["tokens_per_block"]
        or len(row["pools_by_level"]) != len(target["pools_by_level"])
        for row in adp_snapshots
    ):
        return None
    levels = []
    for level_id, pools in enumerate(target["pools_by_level"]):
        if any(len(row["pools_by_level"][level_id]) != len(pools) for row in adp_snapshots):
            return None
        for pool_id, pool in enumerate(pools):
            peers = [row["pools_by_level"][level_id][pool_id] for row in adp_snapshots]
            if any(peer["slot_sizes"] != pool["slot_sizes"] for peer in peers):
                return None
            levels.append(
                {
                    "level": level_id,
                    "pool": pool_id,
                    "dkv_replica_total_pages": pool["total"],
                    "adp_total_pages_by_rank": [peer["total"] for peer in peers],
                    "dkv_replica_usable_pages": pool["available"],
                    "adp_usable_pages_by_rank": [peer["available"] for peer in peers],
                    "usable_gap_pages": sum(peer["available"] for peer in peers)
                    - pool["available"],
                    "total_gap_pages": sum(peer["total"] for peer in peers) - pool["total"],
                    "same_pages_per_rank": all(peer["total"] == pool["total"] for peer in peers),
                }
            )
    return {
        "basis": "usable (free plus evictable) pages of an idle pool, summed over the ADP ranks",
        "tolerance_pages": tolerance_pages,
        "pools": levels,
        "equal_usable_capacity": all(
            abs(row["usable_gap_pages"]) <= tolerance_pages for row in levels
        ),
        "equal_total_capacity": all(row["total_gap_pages"] == 0 for row in levels),
    }


def _event_measurement(
    snapshots: Sequence[dict], events: Sequence[dict], baseline: Sequence[dict] | None
) -> dict:
    residents: list[set[tuple]] = [set() for _ in snapshots]
    last_ids = [-1] * len(snapshots)
    created = [False] * len(snapshots)
    removed = [0] * len(snapshots)
    complete = [True] * len(snapshots)
    hash_algorithms: set[str | None] = set()
    for event in events:
        rank = event.get("attention_dp_rank")
        if type(rank) is not int or not 0 <= rank < len(snapshots):
            raise ValueError("Every measurement event must identify its attention-DP rank")
        event_id = event["event_id"]
        watermark = snapshots[rank].get("event_next_id")
        if type(watermark) is not int:
            complete[rank] = False
            continue
        if event_id >= watermark:
            continue
        if event_id != last_ids[rank] + 1:
            complete[rank] = False
        last_ids[rank] = event_id
        data = event["data"]
        kind = data["type"]
        group = (event.get("layer_group_id"), event["window_size"])
        if kind == "created":
            created[rank] = True
        elif kind == "stored":
            hash_algorithms.add(event.get("hash_algo"))
            for block in data["blocks"]:
                residents[rank].add((*group, block["block_hash"]))
        elif kind == "removed":
            for block_hash in data["block_hashes"]:
                residents[rank].discard((*group, block_hash))
                start = baseline[rank].get("event_next_id") if baseline is not None else 0
                if type(start) is not int:
                    complete[rank] = False
                elif event_id >= start:
                    removed[rank] += 1
        elif kind != "updated":
            complete[rank] = False
    for rank, row in enumerate(snapshots):
        watermark = row.get("event_next_id")
        complete[rank] &= (
            created[rank] and type(watermark) is int and last_ids[rank] + 1 == watermark
        )
        if baseline is not None:
            start = baseline[rank].get("event_next_id")
            complete[rank] &= (
                type(start) is int and type(watermark) is int and 0 <= start <= watermark
            )
    observable = all(complete) and len(hash_algorithms) <= 1
    copies = sum(map(len, residents))
    unique = len(set().union(*residents))
    return {
        "complete_by_rank": complete,
        "complete": observable,
        "resident_block_copies": copies if observable else None,
        "unique_resident_blocks": unique if observable else None,
        "duplicate_storage_ratio": _ratio(copies - unique, copies) if observable else None,
        "resident_blocks_by_rank": list(map(len, residents)) if observable else None,
        "removed_blocks_by_rank": removed if observable else None,
        "removed_block_copies": sum(removed) if observable else None,
        "hash_algorithms": sorted(str(value) for value in hash_algorithms),
        "scope": "committed logical blocks, keyed by layer group/window/hash, across all cache tiers",
    }


def _logical_counts(event_stats: dict, dkv_enabled: bool) -> dict:
    """Per-replica (logical) counts next to the physical sums.

    Under DKV every rank holds a full copy, so the logical count is one rank's value when all ranks
    agree and unknown when they do not. Under ADP each rank holds different data and the logical
    count is the sum.
    """
    logical = {}
    for name in ("removed_blocks", "last_tier_capacity_dropped_pages"):
        by_rank = event_stats.get(f"{name}_by_rank")
        if by_rank is None:
            logical[f"{name}_logical"] = None
        elif dkv_enabled:
            logical[f"{name}_logical"] = by_rank[0] if len(set(by_rank)) == 1 else None
        else:
            logical[f"{name}_logical"] = sum(by_rank)
    return logical


def _iteration_load(iteration_stats: Sequence[dict] | None, group_size: int) -> dict:
    """Cross-rank imbalance of the context tokens scheduled in each iteration, averaged."""
    empty = {
        "per_iteration_token_variance": None,
        "per_iteration_token_imbalance": None,
        "iterations_measured": None,
        "per_iteration_source": None,
    }
    if iteration_stats is None:
        return empty
    by_iteration: dict[int, dict[int, int]] = {}
    for row in iteration_stats:
        rank = row.get("attentionDpRank")
        batching = row.get("inflightBatchingStats")
        if type(rank) is not int or batching is None:
            continue
        by_iteration.setdefault(row["iter"], {})[rank] = batching["numCtxTokens"]
    complete = [
        [loads[rank] for rank in range(group_size)]
        for loads in by_iteration.values()
        if set(loads) == set(range(group_size))
    ]
    if not complete:
        return empty
    return {
        "per_iteration_token_variance": sum(pvariance(loads) for loads in complete) / len(complete),
        "per_iteration_token_imbalance": sum(max(loads) - min(loads) for loads in complete)
        / len(complete),
        "iterations_measured": len(complete),
        "per_iteration_source": "iteration stats numCtxTokens of every rank in each iteration",
    }


def build_dkv_measurement_report(
    rank_snapshots: Sequence[dict],
    events: Sequence[dict],
    *,
    dkv_enabled: bool,
    baseline_snapshots: Sequence[dict] | None = None,
    baseline_event_count: int = 0,
    adp_capacity_reference: Sequence[dict] | None = None,
    capacity_tolerance_pages: int = 0,
    iteration_stats: Sequence[dict] | None = None,
) -> dict:
    """Build an interval report from owner counters and complete rank event streams.

    Pass all events from startup so resident hashes and missing events can be
    checked. ``baseline_event_count`` and ``baseline_snapshots`` select the
    measurement interval without discarding earlier residency. Event boundaries
    use the per-rank snapshot watermarks; the legacy list count only validates
    that a supplied boundary is within the input. Equal-capacity
    validation compares the usable pages of each measured ADP pool, summed over
    the ranks, with one DKV replica, including secondary tiers; the gap is
    reported and may not exceed ``capacity_tolerance_pages``. ``iteration_stats``
    are the exported per-rank rows of the interval and give the per-iteration
    load imbalance. No missing measurement becomes a zero.
    """
    rows = sorted(rank_snapshots, key=lambda row: row["rank"])
    if not rows or [row["rank"] for row in rows] != list(range(len(rows))):
        raise ValueError("Exactly one snapshot for each rank is required")
    if any(row["group_size"] != len(rows) for row in rows):
        raise ValueError("The snapshot set does not cover the entire attention-DP group")
    if any(row["dkv_enabled"] != dkv_enabled for row in rows):
        raise ValueError("Snapshot mode does not match the experiment mode")
    if not 0 <= baseline_event_count <= len(events):
        raise ValueError("Invalid event interval start")
    if baseline_event_count and baseline_snapshots is None:
        raise ValueError("An event interval requires matching counter snapshots")
    baseline = (
        sorted(baseline_snapshots, key=lambda row: row["rank"])
        if baseline_snapshots is not None
        else None
    )
    if baseline is not None and [row["rank"] for row in baseline] != list(range(len(rows))):
        raise ValueError("The interval baseline must contain every rank")
    per_rank = []
    for rank, row in enumerate(rows):
        counters = dict(row["counters"])
        if baseline is not None:
            if _capacity_signature(row) != _capacity_signature(baseline[rank]):
                raise ValueError("Capacity changed within the measurement interval")
            counters = {
                key: value - baseline[rank]["counters"][key] for key, value in counters.items()
            }
        if any(value < 0 for value in counters.values()):
            raise ValueError("Measurement counters reset within the interval")
        per_rank.append({"rank": rank, **_prefix_metrics(counters)})
    global_counters = {key: sum(row[key] for row in per_rank) for key in rows[0]["counters"]}
    event_stats = _event_measurement(rows, events, baseline)
    storage_counters: dict[str, list[int] | None] = {}
    for name in ("last_tier_capacity_dropped_pages", "gpu_offloaded_pages"):
        values = []
        for rank, row in enumerate(rows):
            value = row[name]
            before = baseline[rank][name] if baseline is not None else 0
            if value is None or before is None:
                values = None
                break
            if value < before:
                raise ValueError("Storage counters reset within the interval")
            values.append(value - before)
        storage_counters[name] = values
        event_stats[f"{name}_by_rank"] = values
        event_stats[name] = sum(values) if values is not None else None
    event_stats.update(_logical_counts(event_stats, dkv_enabled))
    event_stats["tier_scope"] = {
        "tiers": list(rows[0]["cache_tiers"]),
        "hit_rate_includes_secondary_tiers": any(name != "gpu" for name in rows[0]["cache_tiers"]),
    }
    event_stats["eviction_count_source"] = (
        "last_tier_capacity_dropped_pages: consumed V2 iter_host_dropped_blocks; "
        "direct physical LRU victims at the last tier, including GPU-only pools; "
        "excludes indirect radix GC, quota changes and cache resets"
    )
    capacity_comparison = (
        _capacity_comparison(rows, adp_capacity_reference, capacity_tolerance_pages)
        if adp_capacity_reference is not None and dkv_enabled
        else None
    )
    matched_capacity = (
        capacity_comparison is not None and capacity_comparison["equal_usable_capacity"]
    )
    reuse = all(row["reuse_enabled"] for row in rows)
    no_failures = not any(
        global_counters[key]
        for key in ("prepare_failures", "prepare_errors", "resize_failures", "resize_errors")
    )
    no_removals = event_stats["complete"] and event_stats["removed_block_copies"] == 0
    storage_stats_advanced = all(
        row["iteration_stats_observations"]
        > (baseline[rank]["iteration_stats_observations"] if baseline is not None else 0)
        and row["stats_generation"] == row["operation_generation"]
        for rank, row in enumerate(rows)
    )
    baseline_stats_complete = baseline is None or all(
        row["stats_generation"] == row["operation_generation"] for row in baseline
    )
    drop_observed = (
        storage_counters["last_tier_capacity_dropped_pages"] is not None
        and storage_stats_advanced
        and baseline_stats_complete
    )
    no_drops = drop_observed and event_stats["last_tier_capacity_dropped_pages"] == 0
    has_observations = global_counters["request_count"] > 0
    reasons = []
    if not reuse:
        reasons.append("reuse_disabled")
    if not has_observations:
        reasons.append("no_successfully_prepared_requests")
    if not event_stats["complete"]:
        reasons.append("incomplete_event_observation")
    if not no_failures:
        reasons.append("prepare_or_resize_failure")
    if not drop_observed:
        reasons.append("capacity_drop_observation_unavailable")
    if any(
        row["reuse_resets"] != (baseline[rank]["reuse_resets"] if baseline is not None else 0)
        for rank, row in enumerate(rows)
    ):
        reasons.append("cache_reset_within_interval")
    if not (no_removals and no_drops) and not matched_capacity:
        reasons.append("removals_without_verified_equal_capacity")
    if dkv_enabled and any(
        _capacity_signature(row) != _capacity_signature(rows[0]) for row in rows
    ):
        reasons.append("dkv_replica_capacity_mismatch")
    return {
        "schema_version": 2,
        "mode": "dkv_replicated" if dkv_enabled else "adp",
        "global": _prefix_metrics(global_counters),
        "load": {
            "scheduled_context_tokens_by_rank": [
                row["scheduled_context_tokens"] for row in per_rank
            ],
            "request_count_by_rank": [row["request_count"] for row in per_rank],
            "interval_total_token_variance": pvariance(
                row["scheduled_context_tokens"] for row in per_rank
            ),
            "interval_total_request_variance": pvariance(row["request_count"] for row in per_rank),
            **_iteration_load(iteration_stats, len(rows)),
        },
        "capacity_comparison": capacity_comparison,
        "per_rank": per_rank,
        "storage": event_stats,
        "capacity_by_rank": [
            {key: value for key, value in row.items() if key != "counters"} for row in rows
        ],
        "validity": {
            "prefix_hit_rate_proxy_valid": not reasons,
            "invalid_reasons": reasons,
            "no_committed_block_removals": no_removals,
            "no_capacity_drops": no_drops,
            "equal_capacity_verified": matched_capacity,
            "equal_total_pages": (
                capacity_comparison["equal_total_capacity"]
                if capacity_comparison is not None
                else None
            ),
            "timing_perturbed_by_fresh_page_fill": any(
                row["fresh_page_fill"] != "none" for row in rows
            ),
            "output_correctness_validated": False,
        },
        "semantics": {
            "prefix": "first successful prepare per owner request; match capped at prompt_tokens - 1",
            "request_hit_rate": "requests with a nonzero match / successfully prepared requests",
            "token_hit_rate": "matched prefix tokens / eligible prompt prefix tokens",
            "load": (
                "scheduled owner context tokens, including retries; excludes dummy and replica "
                "work. per_iteration_* statistics are taken across ranks within each iteration and "
                "then averaged; interval_total_* statistics are across ranks of the interval totals"
            ),
            "physical_vs_logical": (
                "removed_block_copies and the dropped-page counters are physical, summed over the "
                "ranks; the *_logical counts are per replica under DKV (equal on every rank, or "
                "null when the replicas disagree) and the rank sum under ADP"
            ),
            "hit_rate_scope": (
                "matched prefix tokens include every cache tier in storage.tier_scope.tiers; "
                "no per-tier split is measured"
            ),
            "duplicate_storage_ratio": "(resident rank copies - distinct blocks) / resident rank copies",
            "eviction_proxy": (
                "removals invalidate the no-eviction proxy conservatively; "
                "not a physical page eviction count"
            ),
            "capacity": (
                "Replicated KV: unique capacity equals one replica; compare the usable pages of "
                "every ADP pool summed across ranks (capacity_comparison)"
            ),
            "interval": (
                "fixed capacity, no explicit cache reset; rank snapshots taken "
                "after the interval's iteration stats were collected"
            ),
        },
    }
