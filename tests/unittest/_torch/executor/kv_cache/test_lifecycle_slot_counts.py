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
"""The group-wide page counts of ``lifecycle_slot_counts`` and the manager hook that applies them."""

import random
from collections.abc import Sequence
from dataclasses import replace
from functools import partial

import pytest
from _torch.executor.dkv_test_utils import LockstepTpGroup
from _torch.executor.lifecycle_test_configs import (
    STATE_WINDOW,
    SWA_WINDOW,
    TOKENS_PER_BLOCK,
    tier_bytes,
    typical_batch,
    v4_like_config,
)

from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import (
    KVCacheManagerV2,
    _without_host_tier,
)
from tensorrt_llm._torch.pyexecutor.kv_cache.lifecycle_slot_counts import (
    MAX_SLOT_COUNT,
    LifecycleKey,
    _stale_range,
    gpu_granularity,
    lifecycle_layouts,
    lifecycle_page_need,
    slots_for_batch,
    slots_from_constraints,
    solve_lifecycle_slot_counts,
)
from tensorrt_llm.runtime.kv_cache_manager_v2 import (
    AttentionLayerConfig,
    AttnLifeCycle,
    BatchDesc,
    BufferConfig,
    DataRole,
    GpuCacheTierConfig,
    HostCacheTierConfig,
    KVCacheDesc,
    KVCacheManagerConfig,
    LayerId,
    SsmLayerConfig,
)

pytestmark = pytest.mark.cpu_only

MIB = 1 << 20

SWA = LifecycleKey(False, SWA_WINDOW, 0, False)
FULL = LifecycleKey(False, 0, 0, False)
STATE = LifecycleKey(False, STATE_WINDOW, 0, False)

# Three layer subsets of one model, as three ranks would hold them. Rank 0 has one window-only
# layer in front of five compressed sparse layers, rank 1 six of the latter, rank 2 four of them
# and two heavily compressed layers.
SUBSET_KINDS = (["swa"] + ["csa"] * 5, ["csa"] * 6, ["csa"] * 4 + ["hca"] * 2)


def _by_key(config: KVCacheManagerConfig, rows: Sequence[Sequence[int]]) -> dict:
    keys = [layout.key for layout in lifecycle_layouts(config)]
    return {key: tuple(row[index] for row in rows) for index, key in enumerate(keys)}


def test_life_cycle_keys_and_slot_bytes_follow_the_layer_subset() -> None:
    rank0, rank1, rank2 = (lifecycle_layouts(v4_like_config(kinds)) for kinds in SUBSET_KINDS)

    # Ids are assigned in the order layers first use a life cycle: window 128, none, window 8.
    assert [layout.key for layout in rank0] == [SWA, FULL, STATE]
    assert [layout.key for layout in rank1] == [SWA, FULL, STATE]
    assert [layout.key for layout in rank2] == [SWA, FULL, STATE]

    # Six window-128 buffers of 2 KiB; the full-attention life cycle has two pool sizes.
    assert [layout.pool_slot_bytes for layout in rank0] == [(12288,), (10240, 2560), (10240,)]
    assert [layout.pool_slot_bytes for layout in rank1] == [(12288,), (12288, 3072), (12288,)]
    # Two heavily compressed layers add a 1 KiB state buffer to window 128 and 512 B of compressed
    # KV to the full-attention pool that holds the 512 B indexer buffers.
    assert [layout.pool_slot_bytes for layout in rank2] == [
        (12288, 2048),
        (8192, 3072),
        (8192,),
    ]
    # Rank 1 has equal slot bytes for window 128 and window 8, which makes the byte-quota sizing
    # merge them into one pool group there and not on rank 0.
    assert rank1[0].slot_bytes == rank1[2].slot_bytes
    assert rank0[0].slot_bytes != rank0[2].slot_bytes

    reordered = lifecycle_layouts(v4_like_config(SUBSET_KINDS[1], state_window_first=True))
    assert [layout.key for layout in reordered] == [STATE, SWA, FULL]
    assert {layout.key: layout.pool_slot_bytes for layout in reordered} == {
        layout.key: layout.pool_slot_bytes for layout in rank1
    }


def test_life_cycle_keys_match_the_native_life_cycles() -> None:
    layers = [
        AttentionLayerConfig(
            LayerId(0),
            [BufferConfig(DataRole("key"), 4096, None, True)],
            sliding_window_size=64,
            num_sink_tokens=9,
        ),
        AttentionLayerConfig(LayerId(1), [BufferConfig(DataRole("key"), 4096)]),
        AttentionLayerConfig(LayerId(2), [BufferConfig(DataRole("key"), 4096)], 48),
        SsmLayerConfig(LayerId(3), [BufferConfig(DataRole("state"), 8192)]),
    ]
    config = KVCacheManagerConfig(
        tokens_per_block=8,
        cache_tiers=[GpuCacheTierConfig(quota=4 * MIB), HostCacheTierConfig(quota=4 * MIB)],
        layers=layers,
        commit_min_snapshot=True,
    )
    keys = [layout.key for layout in lifecycle_layouts(config)]
    assert keys == [
        LifecycleKey(False, 64, 2, True),
        LifecycleKey(False, 0, 0, False),
        LifecycleKey(False, 48, 0, False),
        LifecycleKey(True, 0, 0, False),
    ]
    native = AttnLifeCycle.make(64, 9, 8, True)
    assert (keys[0].window_size, keys[0].num_sink_blocks, keys[0].is_sparse) == (
        native.window_size,
        native.num_sink_blocks,
        native.is_sparse,
    )


@pytest.mark.parametrize("tokens_per_block", [4, 8, 32])
@pytest.mark.parametrize("window,sink_tokens", [(None, 0), (8, 0), (128, 0), (128, 33), (50, 9)])
def test_stale_ranges_match_the_native_ones(
    tokens_per_block: int, window: int | None, sink_tokens: int
) -> None:
    native = AttnLifeCycle.make(window, sink_tokens, tokens_per_block, False)
    key = LifecycleKey(False, window or 0, -(-sink_tokens // tokens_per_block), False)
    assert key.num_sink_blocks == native.num_sink_blocks
    for history_length in range(0, 400):
        stale = native.get_stale_range(history_length, tokens_per_block)
        beg, end = _stale_range(key, history_length, tokens_per_block)
        assert (beg, max(beg, end)) == (stale.beg, max(stale.beg, stale.end)), history_length


def test_batch_slots_match_hand_counted_pages() -> None:
    keys = [SWA, FULL, STATE]
    # Four requests 96 tokens in, with room for the next token: 13 blocks each. Window 128 and
    # the full-attention life cycle keep all of them, window 8 keeps the last two.
    assert slots_for_batch(keys, typical_batch(4, 96), TOKENS_PER_BLOCK) == [52, 52, 8]
    # A request 200 tokens in has outgrown window 128: its first 9 blocks are stale.
    batch = BatchDesc([KVCacheDesc(capacity=201, history_length=200)])
    assert slots_for_batch(keys, batch, TOKENS_PER_BLOCK) == [26 - 9, 26, 2]
    # Two requests that share a 40 token system prompt count its 5 blocks once.
    batch = BatchDesc([KVCacheDesc(48, 0), KVCacheDesc(48, 0)], system_prompt_length=40)
    assert slots_for_batch([FULL], batch, TOKENS_PER_BLOCK) == [5 + 2 * 1]
    # A recurrent state is one page per request.
    assert slots_for_batch([LifecycleKey(True, 0, 0, False)], typical_batch(3, 50), 8) == [3]


def test_constraints_set_the_minimum_counts() -> None:
    keys = [SWA, FULL, STATE]
    # Without constraints only the structural floor remains: a full window plus the in-flight
    # block, and the largest such floor for every full-attention life cycle.
    assert slots_from_constraints(keys, [], TOKENS_PER_BLOCK, 0.97) == [17, 17, 2]
    # A constraint adds its pages divided by the utilization that a resume may reach.
    batch = typical_batch(8, 96)
    assert slots_from_constraints(keys, [batch], TOKENS_PER_BLOCK, 1.0) == [104, 104, 16]
    scaled = slots_from_constraints(keys, [batch], TOKENS_PER_BLOCK, 0.5)
    assert scaled == [208, 208, 32]
    with pytest.raises(ValueError, match="max_util_for_resume"):
        slots_from_constraints(keys, [batch], TOKENS_PER_BLOCK, 0.0)


def test_page_need_prefers_the_typical_step_then_constraints_then_a_default() -> None:
    keys = [SWA, FULL, STATE]
    step = typical_batch(4, 96)
    constraint = typical_batch(2, 40)
    with_step = v4_like_config(SUBSET_KINDS[0], typical_step=step, constraints=[constraint])
    assert lifecycle_page_need(with_step, keys) == [52, 52, 8]
    constraints_only = v4_like_config(SUBSET_KINDS[0], constraints=[constraint])
    assert lifecycle_page_need(constraints_only, keys) == slots_from_constraints(
        keys, [constraint], TOKENS_PER_BLOCK, 0.97
    )
    # One request with 2048 tokens of history: 257 blocks, of which window 128 keeps 17 and
    # window 8 keeps 2.
    assert lifecycle_page_need(v4_like_config(SUBSET_KINDS[0]), keys) == [17, 257, 2]


def test_gpu_granularity_grows_with_the_quota() -> None:
    gib = 1 << 30
    assert [gpu_granularity(quota) for quota in (MIB, 512 * MIB, gib - 1)] == [2 * MIB] * 3
    assert gpu_granularity(gib) == 2 * MIB
    assert gpu_granularity(2 * gib) == 4 * MIB
    assert gpu_granularity(7 * gib) == 8 * MIB
    assert gpu_granularity(16 * gib) == 32 * MIB
    assert gpu_granularity(1000 * gib) == 32 * MIB


@pytest.mark.parametrize("kinds", SUBSET_KINDS)
@pytest.mark.parametrize("gpu_mib,host_mib", [(10, 4), (16, 64), (64, 16), (512, 2048)])
def test_counts_fit_the_budget_and_cannot_grow(
    kinds: list[str], gpu_mib: int, host_mib: int
) -> None:
    config = v4_like_config(
        kinds, gpu_quota=gpu_mib * MIB, host_quota=host_mib * MIB, typical_step=typical_batch(4, 96)
    )
    rows = solve_lifecycle_slot_counts(config)
    keys = [layout.key for layout in lifecycle_layouts(config)]
    need = lifecycle_page_need(config, keys)
    floors = slots_from_constraints(keys, [], TOKENS_PER_BLOCK, 0.97)

    for level, row in enumerate(rows):
        budget = config.cache_tiers[level].quota
        minimum = floors if level == 0 else [1] * len(keys)
        assert tier_bytes(config, level, row) <= budget
        assert all(count >= low for count, low in zip(row, minimum))

        # The counts follow the need of the life cycles, with the minimums as floors, and are the
        # largest that fit: one more page of the most needed life cycle does not.
        reference = max(need)
        largest = max(row[index] for index, count in enumerate(need) if count == reference)

        def counts_at(pages: int) -> list[int]:
            return [max(low, pages * count // reference) for low, count in zip(minimum, need)]

        assert row == counts_at(largest)
        assert tier_bytes(config, level, counts_at(largest + 1)) > budget


def test_a_larger_budget_never_gives_fewer_pages() -> None:
    generator = random.Random(7)
    for _ in range(40):
        kinds = generator.choice(SUBSET_KINDS)
        small = generator.randrange(4, 200) * MIB
        host = generator.randrange(1, 100) * MIB
        steps = typical_batch(generator.randrange(1, 9), generator.randrange(8, 300))
        before = solve_lifecycle_slot_counts(
            v4_like_config(kinds, gpu_quota=small, host_quota=host, typical_step=steps)
        )
        after = solve_lifecycle_slot_counts(
            v4_like_config(
                kinds,
                gpu_quota=small + generator.randrange(0, 100) * MIB,
                host_quota=host + generator.randrange(0, 100) * MIB,
                typical_step=steps,
            )
        )
        for earlier, later in zip(before, after):
            assert all(low <= high for low, high in zip(earlier, later))


def test_the_minimums_win_when_the_budget_is_too_small() -> None:
    config = v4_like_config(
        SUBSET_KINDS[1], gpu_quota=2 * MIB, host_quota=4096, typical_step=typical_batch(4, 96)
    )
    assert solve_lifecycle_slot_counts(config) == [[17, 17, 2], [1, 1, 1]]


def test_cold_pages_can_have_the_size_a_codec_gives_them() -> None:
    config = v4_like_config(SUBSET_KINDS[0], host_quota=8 * MIB, typical_step=typical_batch(4, 96))
    hot, same_size = solve_lifecycle_slot_counts(config)
    _, smaller = solve_lifecycle_slot_counts(config, cold_page_bytes=[3072, 2560, 2560])
    _, larger = solve_lifecycle_slot_counts(config, cold_page_bytes=[24576, 20480, 20480])
    assert all(small >= same for small, same in zip(smaller, same_size))
    assert all(large <= same for large, same in zip(larger, same_size))
    assert smaller != same_size != larger
    assert tier_bytes(config, 1, same_size) <= 8 * MIB
    with pytest.raises(ValueError, match="one entry per life cycle"):
        solve_lifecycle_slot_counts(config, cold_page_bytes=[4096])
    assert hot == solve_lifecycle_slot_counts(config)[0]


def test_counts_are_capped_at_a_32_bit_page_index() -> None:
    config = v4_like_config(
        SUBSET_KINDS[1],
        gpu_quota=1 << 50,
        host_quota=1 << 50,
        typical_step=typical_batch(1, 8),
    )
    for row in solve_lifecycle_slot_counts(config):
        assert max(row) == MAX_SLOT_COUNT
        assert all(count <= MAX_SLOT_COUNT for count in row)


def test_scratch_reuse_is_not_supported() -> None:
    from tensorrt_llm.runtime.kv_cache_manager_v2 import SwaScratchReuseConfig

    config = v4_like_config(SUBSET_KINDS[1])
    config.swa_scratch_reuse = SwaScratchReuseConfig(0)
    with pytest.raises(ValueError, match="swa_scratch_reuse"):
        solve_lifecycle_slot_counts(config)


def _group_configs() -> list[KVCacheManagerConfig]:
    gpu_mib = (16, 8, 24)
    host_mib = (32, 64, 16)
    return [
        v4_like_config(
            kinds,
            gpu_quota=gpu * MIB,
            host_quota=host * MIB,
            # Rank 1 mentions window 8 first, so its life cycle ids differ from the others'.
            state_window_first=rank == 1,
            typical_step=typical_batch(4, 96),
        )
        for rank, (kinds, gpu, host) in enumerate(zip(SUBSET_KINDS, gpu_mib, host_mib))
    ]


def test_group_counts_are_the_minimum_over_the_ranks_by_life_cycle() -> None:
    configs = _group_configs()
    alone = [_by_key(config, solve_lifecycle_slot_counts(config)) for config in configs]
    # The budgets and slot bytes differ per rank, so the ranks would not agree without the group.
    assert len({tuple(sorted(local.items())) for local in alone}) == len(configs)

    group = LockstepTpGroup(len(configs))
    rows = group.run(
        lambda dist: solve_lifecycle_slot_counts(configs[dist.tp_rank], allgather=dist.tp_allgather)
    )

    agreed = [_by_key(config, row) for config, row in zip(configs, rows)]
    assert all(result == agreed[0] for result in agreed)
    for key, counts in agreed[0].items():
        assert counts == tuple(min(local[key][level] for local in alone) for level in range(2))
    # The ranks that bind are not the same in every tier, and every rank fits what it was given.
    hot_binding = {
        rank
        for rank, local in enumerate(alone)
        if all(local[key][0] == agreed[0][key][0] for key in agreed[0])
    }
    host_binding = {
        rank
        for rank, local in enumerate(alone)
        if all(local[key][1] == agreed[0][key][1] for key in agreed[0])
    }
    assert hot_binding and host_binding and hot_binding != host_binding
    for config, row in zip(configs, rows):
        for level, counts in enumerate(row):
            assert tier_bytes(config, level, counts) <= config.cache_tiers[level].quota
    # Rank 1 lists its life cycles in another order, so its row is another permutation.
    assert rows[1] != rows[0]
    assert [layout.key for layout in lifecycle_layouts(configs[1])][0] == STATE


def test_group_rejects_ranks_with_different_life_cycles() -> None:
    configs = _group_configs()
    # A rank that holds only heavily compressed layers has no window-8 life cycle.
    configs[2] = v4_like_config(["hca"] * 3, gpu_quota=24 * MIB, typical_step=typical_batch(4, 96))
    group = LockstepTpGroup(len(configs))
    with pytest.raises(ValueError, match="same life cycles"):
        group.run(
            lambda dist: solve_lifecycle_slot_counts(
                configs[dist.tp_rank], allgather=dist.tp_allgather
            )
        )


def test_group_rejects_ranks_with_different_workloads_or_tiers() -> None:
    configs = _group_configs()
    configs[1] = v4_like_config(
        SUBSET_KINDS[1], gpu_quota=8 * MIB, host_quota=64 * MIB, typical_step=typical_batch(5, 96)
    )
    group = LockstepTpGroup(len(configs))
    with pytest.raises(ValueError, match="same page need"):
        group.run(
            lambda dist: solve_lifecycle_slot_counts(
                configs[dist.tp_rank], allgather=dist.tp_allgather
            )
        )

    configs = _group_configs()
    configs[2] = replace(configs[2], cache_tiers=configs[2].cache_tiers[:1])
    group = LockstepTpGroup(len(configs))
    with pytest.raises(ValueError, match="cache tiers"):
        group.run(
            lambda dist: solve_lifecycle_slot_counts(
                configs[dist.tp_rank], allgather=dist.tp_allgather
            )
        )


class _Manager:
    """The part of KVCacheManagerV2 the config hooks use."""

    _apply_lifecycle_slot_counts = KVCacheManagerV2._apply_lifecycle_slot_counts


def test_the_manager_hook_assigns_rows_or_asks_a_function() -> None:
    config = v4_like_config(SUBSET_KINDS[1], typical_step=typical_batch(4, 96))
    manager = _Manager()
    assert manager._apply_lifecycle_slot_counts(config, None) is config
    assert config.lifecycle_slot_counts is None

    fixed = manager._apply_lifecycle_slot_counts(config, [(50, 60, 70), (80, 90, 100)])
    assert fixed.lifecycle_slot_counts == [[50, 60, 70], [80, 90, 100]]
    assert config.lifecycle_slot_counts is None

    seen = []

    def solve(final: KVCacheManagerConfig) -> list[list[int]]:
        seen.append(final)
        return solve_lifecycle_slot_counts(final)

    solved = manager._apply_lifecycle_slot_counts(config, solve)
    assert len(seen) == 1 and seen[0] is config
    assert solved.lifecycle_slot_counts == solve_lifecycle_slot_counts(config)
    # What replace() rebuilds from a config keeps the field.
    assert (
        replace(solved, layers=solved.layers).lifecycle_slot_counts == solved.lifecycle_slot_counts
    )

    partial_solve = partial(solve_lifecycle_slot_counts, cold_page_bytes=[4096, 4096, 4096])
    assert manager._apply_lifecycle_slot_counts(config, partial_solve).lifecycle_slot_counts

    with pytest.raises(ValueError, match="one row per cache tier"):
        manager._apply_lifecycle_slot_counts(config, [[50, 60, 70]])
    with pytest.raises(ValueError, match="exclusive|mutually"):
        manager._apply_lifecycle_slot_counts(
            replace(config, initial_pool_ratio=[0.2, 0.3, 0.5]), [[50, 60, 70], [80, 90, 100]]
        )


def test_dropping_the_host_tier_drops_its_row_of_counts() -> None:
    config = v4_like_config(SUBSET_KINDS[1], lifecycle_slot_counts=[[50, 60, 70], [80, 90, 100]])
    assert [type(tier) for tier in config.cache_tiers] == [GpuCacheTierConfig, HostCacheTierConfig]
    stripped = _without_host_tier(config)
    assert [type(tier) for tier in stripped.cache_tiers] == [GpuCacheTierConfig]
    assert stripped.lifecycle_slot_counts == [[50, 60, 70]]
    # Without counts the config only loses the tier, and a config without a host tier is unchanged.
    plain = _without_host_tier(v4_like_config(SUBSET_KINDS[1]))
    assert plain.lifecycle_slot_counts is None
    assert len(plain.cache_tiers) == 1
    assert len(_without_host_tier(plain).cache_tiers) == 1
