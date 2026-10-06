# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""``KVCacheManagerConfig.lifecycle_slot_counts`` on the native storage manager.

Managers that hold different layer subsets have different slot bytes. Given the same page counts
per life cycle they must nevertheless account for every life cycle identically, which is what the
group-wide ``lifecycle_slot_counts`` of a layer-split run relies on: the same free pages, the same
failures, the same reuse and the same evictions on every rank.
"""

import copy
import random
from collections.abc import Callable, Sequence
from dataclasses import replace
from typing import NamedTuple, Optional

import pytest
import torch
from _torch.executor.dkv_test_utils import LockstepTpGroup
from _torch.executor.lifecycle_test_configs import (
    TOKENS_PER_BLOCK,
    tier_bytes,
    typical_batch,
    v4_like_config,
)

from tensorrt_llm._torch.pyexecutor.kv_cache.lifecycle_slot_counts import (
    LifecycleKey,
    lifecycle_layouts,
    slots_for_batch,
    slots_from_constraints,
    solve_lifecycle_slot_counts,
)
from tensorrt_llm.runtime.kv_cache_manager_v2 import (
    AttentionLayerConfig,
    BatchDesc,
    BufferConfig,
    CacheLevel,
    CudaStream,
    DataRole,
    GpuCacheTierConfig,
    HostCacheTierConfig,
    KVCacheDesc,
    KVCacheEventManager,
    KVCacheManager,
    KVCacheManagerConfig,
    KVCacheRemovedData,
    KVCacheStoredData,
    KVCacheUpdatedData,
    LayerId,
    ReuseScope,
    SsmLayerConfig,
    _introspection,
    _KVCache,
)

MIB = 1 << 20
GPU_LEVEL = CacheLevel(0)
HOST_LEVEL = CacheLevel(1)

SWA = LifecycleKey(False, 128, 0, False)
FULL = LifecycleKey(False, 0, 0, False)
STATE = LifecycleKey(False, 8, 0, False)

# Two ranks of one model: rank A holds a window-only layer and five compressed sparse layers, rank
# B six compressed sparse layers. Window 128 and window 8 have equal slot bytes on rank B only.
KINDS_A = ["swa"] + ["csa"] * 5
KINDS_B = ["csa"] * 6

# Pages per life cycle and cache tier, the same for both ranks. Small enough that a handful of
# requests run both tiers out of pages, and above the minimums of the life cycles.
HOT_PAGES = {SWA: 40, FULL: 60, STATE: 14}
HOST_PAGES = {SWA: 50, FULL: 80, STATE: 20}


def rows_for(config: KVCacheManagerConfig, hot: dict, host: Optional[dict]) -> list[list[int]]:
    """The counts of every tier in the life cycle order of ``config``'s own layers."""
    keys = [layout.key for layout in lifecycle_layouts(config)]
    rows = [[hot[key] for key in keys]]
    if host is not None:
        rows.append([host[key] for key in keys])
    return rows


def fixed_config(kinds: Sequence[str], *, with_host: bool = True, **kwargs) -> KVCacheManagerConfig:
    config = v4_like_config(kinds, host_quota=16 * MIB if with_host else None, **kwargs)
    return replace(
        config, lifecycle_slot_counts=rows_for(config, HOT_PAGES, HOST_PAGES if with_host else None)
    )


def totals_by_key(manager: KVCacheManager, config: KVCacheManagerConfig) -> dict:
    """(tier, life cycle key) -> (total, free, evictable) pages of the life cycle's pool group."""
    keys = {layout.key: index for index, layout in enumerate(lifecycle_layouts(config))}
    out = {}
    for level in range(len(config.cache_tiers)):
        statistics = manager.get_storage_statistics(CacheLevel(level))
        pool_groups = manager.get_life_cycle_pool_group_indices(CacheLevel(level))
        for key, life_cycle in keys.items():
            stats = statistics[pool_groups[life_cycle]]
            out[(level, key)] = (stats.total, stats.free, stats.evictable)
    return out


def test_the_field_is_part_of_the_config_surface() -> None:
    config = v4_like_config(KINDS_A)
    assert config.lifecycle_slot_counts is None
    assert "lifecycle_slot_counts" in KVCacheManagerConfig.__dataclass_fields__

    counts = [[50, 60, 70], [80, 90, 100]]
    config.lifecycle_slot_counts = counts
    assert config.lifecycle_slot_counts == counts
    assert copy.copy(config).lifecycle_slot_counts == counts
    assert copy.deepcopy(config).lifecycle_slot_counts == counts
    # dataclasses.replace rebuilds a config from its fields and must not drop the counts.
    assert replace(config, layers=config.layers).lifecycle_slot_counts == counts
    assert replace(config, lifecycle_slot_counts=[[1, 2, 3], [4, 5, 6]]).lifecycle_slot_counts == [
        [1, 2, 3],
        [4, 5, 6],
    ]
    # None is the way back to the byte-quota sizing; the constructor takes it, as replace() needs.
    cleared = replace(config, lifecycle_slot_counts=None)
    assert cleared.lifecycle_slot_counts is None
    assert replace(cleared, layers=cleared.layers).lifecycle_slot_counts is None
    assert config.lifecycle_slot_counts == counts


@pytest.mark.parametrize(
    "counts,message",
    [
        ([[50, 60, 70]], "one row per cache tier"),
        ([[50, 60, 70], [80, 90, 100], [1, 2, 3]], "one row per cache tier"),
        ([], "one row per cache tier"),
        ([[50, 60, 70], [80, 90]], "same length"),
        ([[], []], "non-empty"),
        ([[50, 0, 70], [80, 90, 100]], "positive"),
        ([[50, 60, 70], [80, 90, -1]], "positive"),
        ([[50, 60, 70], [80, 90, 1 << 31]], "positive"),
    ],
)
def test_invalid_counts_are_rejected_when_the_config_is_built(
    counts: list[list[int]], message: str
) -> None:
    config = v4_like_config(KINDS_A)
    with pytest.raises(ValueError, match=message):
        config.lifecycle_slot_counts = counts
        config.validate()
    with pytest.raises(ValueError, match=message):
        replace(config, lifecycle_slot_counts=counts)


def test_counts_exclude_an_initial_pool_ratio() -> None:
    config = v4_like_config(KINDS_A)
    config.initial_pool_ratio = [0.2, 0.3, 0.5]
    with pytest.raises(ValueError, match="mutually exclusive"):
        replace(config, lifecycle_slot_counts=[[50, 60, 70], [80, 90, 100]])


@pytest.mark.parametrize("counts", [[[50, 60], [80, 90]], [[50, 60, 70, 80], [80, 90, 100, 110]]])
def test_the_storage_manager_rejects_counts_for_another_number_of_life_cycles(
    counts: list[list[int]],
) -> None:
    config = v4_like_config(KINDS_A, lifecycle_slot_counts=counts)
    with pytest.raises(ValueError, match="number of life cycles"):
        KVCacheManager(config)


@pytest.mark.parametrize("kinds", [KINDS_A, KINDS_B])
@pytest.mark.parametrize("with_host", [False, True])
def test_every_life_cycle_gets_its_own_pool_group_and_exactly_its_count(
    kinds: list[str], with_host: bool
) -> None:
    config = fixed_config(kinds, with_host=with_host)
    manager = KVCacheManager(config)
    try:
        keys = [layout.key for layout in lifecycle_layouts(config)]
        for level in range(len(config.cache_tiers)):
            assert manager.get_life_cycle_pool_group_indices(CacheLevel(level)) == [0, 1, 2]
            pages = HOT_PAGES if level == 0 else HOST_PAGES
            assert [stats.total for stats in manager.get_storage_statistics(CacheLevel(level))] == [
                pages[key] for key in keys
            ]
        descs = manager.pool_group_descs
        assert [desc.num_slots for desc in descs] == [HOT_PAGES[key] for key in keys]
        for index, (desc, layout) in enumerate(zip(descs, lifecycle_layouts(config))):
            assert [pool.slot_bytes for pool in desc.pools] == list(layout.pool_slot_bytes)
            assert [variant.layer_group_id for variant in desc.slot_desc.variants] == [index]
    finally:
        manager.shutdown()


def test_without_counts_equal_slot_bytes_share_a_pool_group_on_one_rank_only() -> None:
    # The byte-quota sizing groups life cycles by slot bytes, so the two ranks lay their pools out
    # differently and no byte quotas can give both the same pages per life cycle.
    configs = [v4_like_config(kinds) for kinds in (KINDS_A, KINDS_B)]
    managers = [KVCacheManager(config) for config in configs]
    try:
        groupings = []
        for manager, config in zip(managers, configs):
            pool_groups = manager.get_life_cycle_pool_group_indices(GPU_LEVEL)
            groupings.append(
                {
                    layout.key: pool_groups[life_cycle]
                    for life_cycle, layout in enumerate(lifecycle_layouts(config))
                }
            )
        assert len(set(groupings[0].values())) == 3
        assert groupings[1][SWA] == groupings[1][STATE] != groupings[1][FULL]
    finally:
        for manager in managers:
            manager.shutdown()


def test_the_pools_follow_the_counts_not_the_quotas() -> None:
    totals = []
    for gpu_mib, host_mib in ((4, 4), (64, 1), (512, 128)):
        config = v4_like_config(KINDS_A, gpu_quota=gpu_mib * MIB, host_quota=host_mib * MIB)
        config = replace(config, lifecycle_slot_counts=rows_for(config, HOT_PAGES, HOST_PAGES))
        manager = KVCacheManager(config)
        try:
            totals.append({key: value[0] for key, value in totals_by_key(manager, config).items()})
        finally:
            manager.shutdown()
    assert totals[0] == totals[1] == totals[2]
    assert totals[0][(0, SWA)] == HOT_PAGES[SWA]
    assert totals[0][(1, STATE)] == HOST_PAGES[STATE]


def test_hot_counts_below_the_constraint_minimum_are_raised_to_it() -> None:
    constraints = [typical_batch(8, 96), BatchDesc([KVCacheDesc(capacity=120, history_length=119)])]
    config = v4_like_config(KINDS_A, constraints=constraints, max_util_for_resume=0.9)
    keys = [layout.key for layout in lifecycle_layouts(config)]
    minimum = slots_from_constraints(keys, constraints, TOKENS_PER_BLOCK, 0.9)
    assert minimum[1] > 1
    config = replace(config, lifecycle_slot_counts=[[1] * len(keys), [1] * len(keys)])
    manager = KVCacheManager(config)
    try:
        assert [s.total for s in manager.get_storage_statistics(GPU_LEVEL)] == minimum
        # The cold tiers only have the structural floor of one page.
        assert [s.total for s in manager.get_storage_statistics(HOST_LEVEL)] == [1] * len(keys)
    finally:
        manager.shutdown()
    # Counts above the minimum are taken as they are.
    config = replace(config, lifecycle_slot_counts=[[m + 3 for m in minimum], [5] * len(keys)])
    manager = KVCacheManager(config)
    try:
        assert [s.total for s in manager.get_storage_statistics(GPU_LEVEL)] == [
            m + 3 for m in minimum
        ]
    finally:
        manager.shutdown()


def _hybrid_config(**kwargs) -> KVCacheManagerConfig:
    layers = [
        AttentionLayerConfig(LayerId(0), [BufferConfig(DataRole("key"), 1024)], 64, 4),
        AttentionLayerConfig(LayerId(1), [BufferConfig(DataRole("key"), 2048)]),
        SsmLayerConfig(LayerId(2), [BufferConfig(DataRole("state"), 4096)]),
    ]
    return KVCacheManagerConfig(
        tokens_per_block=8,
        cache_tiers=[GpuCacheTierConfig(quota=16 * MIB), HostCacheTierConfig(quota=16 * MIB)],
        layers=layers,
        commit_min_snapshot=True,
        enable_partial_reuse=False,
        **kwargs,
    )


def _override_config(**kwargs) -> KVCacheManagerConfig:
    """Buffers that hold sub-blocks of 4 and 2 tokens in blocks of 8, next to a plain one."""
    layers = [
        AttentionLayerConfig(
            LayerId(0),
            [BufferConfig(DataRole("key"), 1024, 4), BufferConfig(DataRole("value"), 1024)],
            64,
        ),
        AttentionLayerConfig(LayerId(1), [BufferConfig(DataRole("key"), 2048, 2)]),
        AttentionLayerConfig(LayerId(2), [BufferConfig(DataRole("key"), 512)], 64),
    ]
    return KVCacheManagerConfig(
        tokens_per_block=8,
        cache_tiers=[GpuCacheTierConfig(quota=16 * MIB), HostCacheTierConfig(quota=16 * MIB)],
        layers=layers,
        **kwargs,
    )


@pytest.mark.parametrize("kind", ["v4_like", "hybrid", "buffer_override"])
def test_the_python_page_arithmetic_matches_the_native_storage_manager(kind: str) -> None:
    """Page needs, minimum counts and slot bytes of the solver equal what the native core derives."""
    generator = random.Random(11)
    constraints = [typical_batch(3, 60), BatchDesc([KVCacheDesc(capacity=90, history_length=40)])]
    if kind == "hybrid":
        config = _hybrid_config(constraints=constraints)
    elif kind == "buffer_override":
        config = _override_config(constraints=constraints)
    else:
        config = v4_like_config(KINDS_A, constraints=constraints)
    layouts = lifecycle_layouts(config)
    keys = [layout.key for layout in layouts]
    config = replace(config, lifecycle_slot_counts=[[1] * len(keys), [1] * len(keys)])
    manager = KVCacheManager(config)
    try:
        # The minimum counts: nothing but the floors survives counts of one page.
        floors = slots_from_constraints(keys, constraints, config.tokens_per_block, 0.97)
        assert [s.total for s in manager.get_storage_statistics(GPU_LEVEL)] == floors
        # One pool group per life cycle, with the slot bytes the solver derives from the layers.
        for desc, layout in zip(manager.pool_group_descs, layouts):
            assert [pool.slot_bytes for pool in desc.pools] == list(layout.pool_slot_bytes)
        # Pages occupied by arbitrary batches.
        for _ in range(60):
            kv_caches = []
            for _ in range(generator.randrange(1, 6)):
                capacity = generator.randrange(1, 300)
                kv_caches.append(KVCacheDesc(capacity, generator.randrange(0, capacity + 1)))
            batch = BatchDesc(kv_caches, generator.choice([0, 0, 16, 40]))
            native = _introspection.compute_slots_for_batch(manager, batch, config.tokens_per_block)
            assert native == slots_for_batch(keys, batch, config.tokens_per_block), batch
    finally:
        manager.shutdown()


def test_the_group_solution_sizes_the_pools_of_every_rank_alike() -> None:
    """The counts the group agrees on, in each rank's life cycle order, size its native pools."""
    configs = [
        v4_like_config(
            kinds,
            gpu_quota=gpu * MIB,
            host_quota=host * MIB,
            state_window_first=reordered,
            typical_step=typical_batch(4, 96),
        )
        for kinds, gpu, host, reordered in (
            (KINDS_A, 24, 32, False),
            (KINDS_B, 16, 64, True),
        )
    ]
    group = LockstepTpGroup(len(configs))
    rows = group.run(
        lambda dist: solve_lifecycle_slot_counts(configs[dist.tp_rank], allgather=dist.tp_allgather)
    )
    alone = [solve_lifecycle_slot_counts(config) for config in configs]
    # The ranks could not hold the same counts on their own.
    assert alone[0] != rows[0] or alone[1] != rows[1]

    managers = [
        KVCacheManager(replace(config, lifecycle_slot_counts=row))
        for config, row in zip(configs, rows)
    ]
    try:
        seen = [
            {key: stats[0] for key, stats in totals_by_key(manager, config).items()}
            for manager, config in zip(managers, configs)
        ]
        assert seen[0] == seen[1]
        for manager, config, row in zip(managers, configs, rows):
            # The solver's byte accounting is the native one, and every rank stays in its budget.
            for level, tier in enumerate(config.cache_tiers):
                assert manager.get_quota(CacheLevel(level)) == tier_bytes(config, level, row[level])
                assert manager.get_quota(CacheLevel(level)) <= tier.quota
    finally:
        for manager in managers:
            manager.shutdown()


@pytest.mark.parametrize("with_host", [False, True])
def test_the_pool_layout_is_never_rebalanced(with_host: bool) -> None:
    config = fixed_config(KINDS_B, with_host=with_host)
    manager = KVCacheManager(config)
    stream = torch.cuda.Stream()
    try:
        before = totals_by_key(manager, config)
        quota = manager.get_quota(GPU_LEVEL)
        _introspection.force_rebalance_precondition(manager)
        assert manager.need_adjustment is False

        cache = manager.create_kv_cache(ReuseScope(), list(range(40)))
        assert cache.resume(CudaStream(stream.cuda_stream))
        assert cache.resize(40)
        # Nothing to do, so the active cache does not have to be suspended first.
        manager.adjust()
        assert manager.need_adjustment is False
        # Closing the cache runs the tuner statistics update.
        _introspection.set_num_sampled_kv_caches(manager, 5000)
        stream.synchronize()
        cache.close()
        assert manager.need_adjustment is False

        assert manager.resize(GPU_LEVEL, 256 * MIB) is False
        if with_host:
            assert manager.resize(HOST_LEVEL, 256 * MIB) is False
        assert manager.get_quota(GPU_LEVEL) == quota
        after = totals_by_key(manager, config)
        assert {key: value[0] for key, value in after.items()} == {
            key: value[0] for key, value in before.items()
        }
    finally:
        manager.shutdown()


@pytest.mark.parametrize("with_host", [False, True])
def test_the_sequence_length_clamp_does_not_depend_on_the_layer_subset(with_host: bool) -> None:
    """The longest sequence a batch fits follows from the page counts and the life cycles alone."""
    managers = [
        KVCacheManager(fixed_config(kinds, with_host=with_host)) for kinds in (KINDS_A, KINDS_B)
    ]
    try:
        clamps = {}
        for batch_size in (1, 2, 5, 12):
            for upper_bound in (16, 160, 512, 4096, 1 << 20):
                clamped = [
                    manager.clamp_max_seq_len_for_mem(batch_size, upper_bound)
                    for manager in managers
                ]
                assert clamped[0] == clamped[1], (batch_size, upper_bound)
                clamps[(batch_size, upper_bound)] = clamped[0]
        # The pools are small enough to bind: the clamp is neither the bound nor zero everywhere.
        assert 0 < clamps[(1, 1 << 20)] < 1 << 20
        assert clamps[(12, 1 << 20)] < clamps[(1, 1 << 20)]
    finally:
        for manager in managers:
            manager.shutdown()


def test_the_default_layout_is_still_rebalanced() -> None:
    manager = KVCacheManager(v4_like_config(KINDS_A))
    try:
        assert manager.need_adjustment is False
        _introspection.force_rebalance_precondition(manager)
        assert manager.need_adjustment is True
    finally:
        manager.shutdown()


class Op(NamedTuple):
    name: str
    request: int
    args: tuple


class RankedManager:
    """One rank: its manager, its events and its requests."""

    def __init__(self, name: str, config: KVCacheManagerConfig) -> None:
        self.name = name
        self.config = config
        self.keys = {layout.key: index for index, layout in enumerate(lifecycle_layouts(config))}
        self.events = KVCacheEventManager(
            1 << 20,
            window_size=1 << 20,
            window_size_by_layer_group={
                index: key.window_size or (1 << 20) for key, index in self.keys.items()
            },
        )
        self.manager = KVCacheManager(config, event_manager=self.events)
        self.stream = torch.cuda.Stream()
        self.caches: dict[int, _KVCache] = {}

    def drain_events(self) -> list:
        self.events.flush_iteration_events()
        records = []
        for event in self.events.get_latest_events(0):
            data = event.data
            if isinstance(data, KVCacheStoredData):
                records += [
                    ("stored", event.window_size, str(b.block_hash), b.cache_level, b.priority)
                    for b in data.blocks
                ]
            elif isinstance(data, KVCacheRemovedData):
                records += [("removed", event.window_size, str(h)) for h in data.block_hashes]
            elif isinstance(data, KVCacheUpdatedData):
                level = data.cache_level
                diff = None if level is None else (level.old_value, level.new_value)
                records.append(("updated", event.window_size, str(data.block_hash), diff))
        return sorted(records, key=repr)

    def probe(self, prompts: list[list[int]]) -> list:
        """Per prompt and life cycle: the reusable tokens and which blocks still hold a page."""
        out = []
        for prompt in prompts:
            row = []
            for key in sorted(self.keys):
                tokens, pages = _introspection.reuse_match_pages(
                    self.manager, ReuseScope(), prompt, self.keys[key], True
                )
                row.append((key, tokens, [None if page is None else page[1] for page in pages]))
            out.append(row)
        return out

    def request_state(self, request: int) -> tuple:
        cache = self.caches[request]
        return (
            cache.capacity,
            cache.history_length,
            cache.num_committed_tokens,
            cache.is_active,
            cache.status,
        )

    def shutdown(self) -> None:
        for cache in self.caches.values():
            cache.close()
        self.stream.synchronize()
        self.manager.shutdown()


class Request:
    """What the driver tracks of a request, identically for every rank."""

    def __init__(self, prompt: list[int]) -> None:
        self.prompt = prompt
        self.tokens = list(prompt)
        self.reused = 0
        self.prefilled = False
        self.active = False
        self.committing = True


def run_random_sequence(ranks: list[RankedManager], seed: int, num_steps: int) -> dict:
    """Drive every rank through the same random sequence and compare them after every step."""
    generator = random.Random(seed)
    next_token = [1000]

    def fresh(count: int) -> list[int]:
        start = next_token[0]
        next_token[0] += count
        return list(range(start, start + count))

    # A few base prompts whose prefixes the requests share, mostly cut inside a block so that
    # partial reuse has something to copy.
    bases = [fresh(120) for _ in range(4)]
    prompts_seen: list[list[int]] = []
    requests: dict[int, Request] = {}
    outcomes = {
        "create": 0,
        "resume_failed": 0,
        "resize_failed": 0,
        "reused": 0,
        "partial_reused": 0,
        "closed": 0,
        "stored": 0,
        "removed": 0,
        "moved": 0,
    }
    # The events of the final clean-up remove every block that is left and say nothing about
    # evictions under pressure, so they are not tallied.
    tally_events = True
    next_id = 0

    def compare(step: int, op: Op, results: list, *, probe: bool) -> None:
        context = f"seed {seed} step {step} {op}"
        assert all(result == results[0] for result in results), (context, results)
        reference = ranks[0]
        for rank in ranks[1:]:
            assert totals_by_key(rank.manager, rank.config) == totals_by_key(
                reference.manager, reference.config
            ), context
            for request in reference.caches:
                assert rank.request_state(request) == reference.request_state(request), (
                    context,
                    request,
                )
        events = [rank.drain_events() for rank in ranks]
        assert all(entry == events[0] for entry in events), (context, events)
        if tally_events:
            for record in events[0]:
                if record[0] == "updated":
                    outcomes["moved"] += record[3] is not None
                else:
                    outcomes[record[0]] += 1
        if probe:
            patterns = [rank.probe(prompts_seen[-30:]) for rank in ranks]
            assert all(entry == patterns[0] for entry in patterns), context

    def apply(step: int, op: Op, action: Callable[[RankedManager], object]) -> list:
        results = [action(rank) for rank in ranks]
        compare(step, op, results, probe=step % 4 == 0 or step >= num_steps)
        return results

    for step in range(num_steps):
        live = sorted(requests)
        choice = generator.random()
        if not live or (choice < 0.18 and len(live) < 7):
            base = generator.choice(bases)
            cut = generator.randrange(5, len(base))
            prompt = base[:cut] + fresh(generator.randrange(0, 40))
            request_id = next_id
            next_id += 1
            op = Op("create", request_id, (len(prompt),))
            prompts_seen.append(prompt)
            request = Request(prompt)
            requests[request_id] = request

            def create(rank: RankedManager, rid=request_id, prompt=prompt) -> int:
                rank.caches[rid] = rank.manager.create_kv_cache(ReuseScope(), prompt)
                return rank.caches[rid].num_committed_tokens

            reused = apply(step, op, create)[0]
            request.reused = reused
            outcomes["create"] += 1
            outcomes["reused"] += reused > 0
            outcomes["partial_reused"] += reused % TOKENS_PER_BLOCK != 0
            continue

        request_id = generator.choice(live)
        request = requests[request_id]
        action = generator.random()
        if not request.active:
            if action < 0.8:
                op = Op("resume", request_id, ())
                ok = apply(
                    step,
                    op,
                    lambda rank, rid=request_id: rank.caches[rid].resume(
                        CudaStream(rank.stream.cuda_stream)
                    ),
                )[0]
                request.active = bool(ok)
                outcomes["resume_failed"] += not ok
            else:
                op = Op("close", request_id, ())
                apply(step, op, lambda rank, rid=request_id: rank.caches.pop(rid).close())
                del requests[request_id]
                outcomes["closed"] += 1
            continue

        if not request.prefilled:
            op = Op("prefill", request_id, (len(request.prompt),))

            def prefill(rank: RankedManager, rid=request_id, req=request) -> bool:
                cache = rank.caches[rid]
                if not cache.resize(len(req.prompt)):
                    return False
                if len(req.prompt) > req.reused:
                    cache.commit(req.prompt[req.reused :])
                return True

            ok = apply(step, op, prefill)[0]
            request.prefilled = bool(ok)
            outcomes["resize_failed"] += not ok
            if not ok and generator.random() < 0.5:
                op = Op("suspend", request_id, ())
                apply(step, op, lambda rank, rid=request_id: rank.caches[rid].suspend())
                request.active = False
            continue

        if action < 0.45:
            op = Op("decode", request_id, (1,))
            request.tokens.append(fresh(1)[0])

            def decode(rank: RankedManager, rid=request_id, req=request) -> bool:
                cache = rank.caches[rid]
                if len(req.tokens) > cache.capacity:
                    if not cache.resize(cache.capacity + TOKENS_PER_BLOCK):
                        return False
                if req.committing:
                    cache.commit(req.tokens[cache.history_length :])
                return True

            ok = apply(step, op, decode)[0]
            outcomes["resize_failed"] += not ok
            if not ok:
                request.tokens.pop()
        elif action < 0.55 and request.committing:
            op = Op("stop_committing", request_id, ())
            apply(step, op, lambda rank, rid=request_id: rank.caches[rid].stop_committing())
            request.committing = False
        elif action < 0.7:
            op = Op("suspend", request_id, ())
            apply(step, op, lambda rank, rid=request_id: rank.caches[rid].suspend())
            request.active = False
        elif action < 0.78:
            op = Op("shrink", request_id, ())

            def shrink(rank: RankedManager, rid=request_id, req=request) -> bool:
                cache = rank.caches[rid]
                return cache.resize(max(len(req.tokens), cache.capacity - TOKENS_PER_BLOCK))

            apply(step, op, shrink)
        else:
            op = Op("close", request_id, ())
            apply(step, op, lambda rank, rid=request_id: rank.caches.pop(rid).close())
            del requests[request_id]
            outcomes["closed"] += 1

    # Close what is left and compare once more with everything released.
    tally_events = False
    for request_id in sorted(requests):
        op = Op("close", request_id, ())
        apply(num_steps, op, lambda rank, rid=request_id: rank.caches.pop(rid).close())
    apply(num_steps + 1, Op("clear", -1, ()), lambda rank: rank.manager.clear_reusable_blocks())
    for rank in ranks:
        for (level, key), (total, free, evictable) in totals_by_key(
            rank.manager, rank.config
        ).items():
            assert (free, evictable) == (total, 0), (rank.name, level, key)
    return outcomes


# Layer subsets of one model as the ranks of a group hold them: five against six compressed sparse
# layers, and three ranks whose other layers differ as well.
RANK_SUBSETS = [
    pytest.param([["csa"] * 5, ["csa"] * 6], id="5-vs-6-csa"),
    pytest.param([KINDS_A, KINDS_B, ["csa"] * 4 + ["hca"] * 2], id="three-ranks"),
]


def make_ranks(
    subsets: Sequence[Sequence[str]], *, with_host: bool, state_window_first: bool
) -> list[RankedManager]:
    """One manager per layer subset; the second one may number its life cycles differently.

    A resume is refused above half utilization of any pool group, so that the sequences exercise
    this decision as well as the failure of resizes.
    """
    return [
        RankedManager(
            f"rank{index}",
            fixed_config(
                kinds,
                with_host=with_host,
                state_window_first=state_window_first and index == 1,
                max_util_for_resume=0.5,
            ),
        )
        for index, kinds in enumerate(subsets)
    ]


@pytest.mark.parametrize("with_host", [False, True])
@pytest.mark.parametrize("state_window_first", [False, True])
@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("subsets", RANK_SUBSETS)
def test_ranks_with_different_layer_subsets_account_for_pages_identically(
    subsets: list[list[str]], seed: int, state_window_first: bool, with_host: bool
) -> None:
    """Free pages, request state, reuse, evictions and failures match step by step.

    The ranks differ in their slot bytes, so some life cycles share a slot size on one rank and
    not on another, and with ``state_window_first`` the second rank numbers its life cycles
    differently. Neither the pool layouts nor the life cycle ids agree; the page counts do.
    """
    ranks = make_ranks(subsets, with_host=with_host, state_window_first=state_window_first)
    try:
        outcomes = run_random_sequence(ranks, seed, 260)
    finally:
        for rank in ranks:
            rank.shutdown()
    # The sequence is not trivial: it reuses prefixes, partly inside a block, runs the pools out of
    # pages and evicts.
    assert outcomes["reused"] > 0, outcomes
    assert outcomes["partial_reused"] > 0, outcomes
    assert outcomes["resize_failed"] > 0, outcomes
    assert outcomes["resume_failed"] > 0, outcomes
    assert outcomes["removed"] > 0, outcomes
    if with_host:
        assert outcomes["moved"] > 0, outcomes


def test_the_comparison_notices_ranks_that_account_differently() -> None:
    """A rank that refuses resumes at a lower utilization than the other one is flagged."""
    configs = [fixed_config(KINDS_A), fixed_config(KINDS_B, max_util_for_resume=0.3)]
    ranks = [RankedManager(name, config) for name, config in zip("AB", configs)]
    try:
        with pytest.raises(AssertionError, match="resume"):
            run_random_sequence(ranks, 0, 260)
    finally:
        for rank in ranks:
            rank.shutdown()
