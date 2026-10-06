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
"""Page counts per life cycle for managers whose layer subsets differ.

``KVCacheManagerConfig.lifecycle_slot_counts`` fixes the number of pages of every life cycle in
every cache tier. Ranks that hold different layer subsets have different slot bytes and so cannot
agree on counts by agreeing on byte quotas; this module computes counts every rank can use.

Each rank derives, for its own layer subset, the life cycles (identified by the semantic key
``(window, sink blocks, sparse)``, which does not depend on which layers the rank holds), the
relative page need of every life cycle and the largest page counts its byte budget holds. The
group then takes the minimum of the counts per life cycle and tier, so that every rank configures
the same count for the same life cycle.

The page need and the minimum slot counts repeat what the C++ storage manager derives from the
config (``computeSlotsForBatch``, ``computeSlotsFromConstraints`` and the cache tier
granularity), because the storage manager keeps those private and needs a live manager. The
parity with the native results is covered by ``test_dkv_lifecycle_slot_counts.py``.
"""

import math
import struct
from dataclasses import dataclass
from typing import Any, Callable, NamedTuple, Optional, Sequence

from tensorrt_llm.logger import logger
from tensorrt_llm.runtime.kv_cache_manager_v2 import (
    AttentionLayerConfig,
    BatchDesc,
    DiskCacheTierConfig,
    GpuCacheTierConfig,
    HostCacheTierConfig,
    KVCacheDesc,
    SsmLayerConfig,
)

# Page indices are 32-bit throughout the storage layer.
MAX_SLOT_COUNT = (1 << 31) - 1

# One request with an average history, the workload assumed when neither a typical step nor
# constraints describe it.
_FALLBACK_CAPACITY = 2049
_FALLBACK_HISTORY_LENGTH = 2048

_GPU_PHYS_MEM_BASE = 2 << 20
_HOST_GRANULARITY = 4096
_DISK_GRANULARITY = 2 << 20


class LifecycleKey(NamedTuple):
    """Semantic identity of a life cycle, the same on every rank whatever layers it holds.

    Life cycle ids are assigned in the order the layers of a rank first mention each life cycle,
    so they differ between ranks that hold different layer subsets; this key does not.
    """

    is_ssm: bool
    window_size: int  # 0: no sliding window
    num_sink_blocks: int
    is_sparse: bool


@dataclass(frozen=True)
class LifecycleLayout:
    """A life cycle of one rank: its key and the byte size of each hot pool slot.

    ``pool_slot_bytes`` has one entry per coalesced pool of the life cycle. A slot is one page of
    the life cycle and holds one sub-page of every buffer of every layer of the life cycle.
    """

    key: LifecycleKey
    pool_slot_bytes: tuple[int, ...]

    @property
    def slot_bytes(self) -> int:
        return sum(self.pool_slot_bytes)


def _div_up(value: int, divisor: int) -> int:
    return -(-value // divisor)


def _round_up(value: int, granularity: int) -> int:
    return _div_up(value, granularity) * granularity


def _c_div(numerator: int, denominator: int) -> int:
    """Integer division that truncates toward zero, as C++ does."""
    quotient = abs(numerator) // abs(denominator)
    return quotient if (numerator >= 0) == (denominator > 0) else -quotient


def lifecycle_layouts(config: Any) -> list[LifecycleLayout]:
    """Life cycles of ``config`` in life cycle id order, which is the order layers first use them."""
    tokens_per_block = int(config.tokens_per_block)
    keys: list[LifecycleKey] = []
    sizes_by_key: dict[LifecycleKey, dict[int, int]] = {}
    for layer in config.layers:
        if isinstance(layer, SsmLayerConfig):
            key = LifecycleKey(True, 0, 0, False)
        elif isinstance(layer, AttentionLayerConfig):
            window = layer.sliding_window_size
            sink_tokens = layer.num_sink_tokens or 0
            is_sparse = bool(layer.buffers) and bool(layer.buffers[0].is_sparse)
            key = LifecycleKey(
                False,
                int(window) if window else 0,
                _div_up(int(sink_tokens), tokens_per_block),
                is_sparse,
            )
        else:
            raise ValueError(f"Unsupported layer config {type(layer).__name__}")
        if key not in sizes_by_key:
            keys.append(key)
            sizes_by_key[key] = {}
        # Buffers of one size share a pool; a pool slot holds one sub-page per such buffer.
        for buffer in layer.buffers:
            override = buffer.tokens_per_block_override or tokens_per_block
            expanded_size = int(buffer.size) * (tokens_per_block // override)
            counts = sizes_by_key[key]
            counts[expanded_size] = counts.get(expanded_size, 0) + 1
    return [
        LifecycleLayout(
            key,
            tuple(
                sorted((size * count for size, count in sizes_by_key[key].items()), reverse=True)
            ),
        )
        for key in keys
    ]


def _stale_range(key: LifecycleKey, history_length: int, tokens_per_block: int) -> tuple[int, int]:
    """Blocks of a request with ``history_length`` tokens that no longer hold live pages."""
    if key.is_ssm:
        return 0, history_length // tokens_per_block
    start = min(_div_up(history_length, tokens_per_block), key.num_sink_blocks)
    if key.window_size == 0:
        return start, start
    # The in-flight token at position history_length keeps the window one token longer.
    window_start = _c_div(history_length + 1 - key.window_size, tokens_per_block)
    return start, max(start, window_start)


def _length(block_range: tuple[int, int]) -> int:
    return max(0, block_range[1] - block_range[0])


def _intersect(first: tuple[int, int], second: tuple[int, int]) -> tuple[int, int]:
    return max(first[0], second[0]), min(first[1], second[1])


def slots_for_batch(
    keys: Sequence[LifecycleKey], batch: BatchDesc, tokens_per_block: int
) -> list[int]:
    """Pages of every life cycle that ``batch`` occupies, without sliding-window scratch reuse."""
    system_blocks = batch.system_prompt_length // tokens_per_block
    system_range = (0, system_blocks)
    caches: Sequence[KVCacheDesc] = batch.kv_caches
    slots = []
    for key in keys:
        if key.is_ssm:
            slots.append(len(caches))
            continue
        shared = system_range
        for cache in caches:
            shared = _intersect(shared, _stale_range(key, cache.history_length, tokens_per_block))
        count = system_blocks - _length(shared)
        for cache in caches:
            stale = _stale_range(key, cache.history_length, tokens_per_block)
            non_stale = _div_up(cache.capacity, tokens_per_block) - _length(stale)
            non_stale_system = system_blocks - _length(_intersect(stale, system_range))
            count += max(0, non_stale - non_stale_system)
        slots.append(count)
    return slots


def _float32(value: float) -> float:
    return struct.unpack("f", struct.pack("f", value))[0]


def slots_from_constraints(
    keys: Sequence[LifecycleKey],
    constraints: Sequence[BatchDesc],
    tokens_per_block: int,
    max_util_for_resume: float,
) -> list[int]:
    """Pages every life cycle needs so that all ``constraints`` batches can run.

    This is also the hot tier's minimum slot count of a life cycle: the structural floor (a full
    window plus the in-flight block, shared by all full-attention life cycles) and, for every
    constraint, its pages divided by ``max_util_for_resume`` because a resume above that
    utilization fails.
    """
    max_util = _float32(max_util_for_resume)
    if not 0.0 < max_util <= 1.0:
        raise ValueError("max_util_for_resume must be in (0, 1]")

    def window_floor(key: LifecycleKey) -> int:
        return (
            key.num_sink_blocks + (key.window_size + tokens_per_block - 2) // tokens_per_block + 1
        )

    full_attention_floor = max(
        [1] + [window_floor(key) for key in keys if not key.is_ssm and key.window_size]
    )
    slots = []
    for key in keys:
        if key.is_ssm:
            slots.append(1)
        elif key.window_size:
            slots.append(window_floor(key))
        else:
            slots.append(full_attention_floor)
    for batch in constraints:
        for index, count in enumerate(slots_for_batch(keys, batch, tokens_per_block)):
            slots[index] = max(slots[index], math.ceil(count / max_util))
    return slots


def lifecycle_page_need(config: Any, keys: Sequence[LifecycleKey]) -> list[int]:
    """Relative page need of every life cycle: the pages of the typical step.

    The typical step wins, then the constraints, then a single request with 2048 tokens of history.
    Pages are counted per life cycle, not in bytes, so the result is the same on every rank.
    """
    tokens_per_block = int(config.tokens_per_block)
    if config.typical_step is not None:
        return slots_for_batch(keys, config.typical_step, tokens_per_block)
    if config.constraints:
        return slots_from_constraints(
            keys, config.constraints, tokens_per_block, config.max_util_for_resume
        )
    fallback = BatchDesc([KVCacheDesc(_FALLBACK_CAPACITY, _FALLBACK_HISTORY_LENGTH)])
    return slots_for_batch(keys, fallback, tokens_per_block)


def gpu_granularity(quota: int) -> int:
    """Physical allocation chunk of a GPU tier: 2 MiB, growing to 32 MiB with the quota."""
    ratio = quota // ((2 << 20) * 512)
    exponent = 0 if ratio == 0 else min(4, ratio.bit_length() - 1)
    return _GPU_PHYS_MEM_BASE << exponent


def _tier_granularity(tier: Any, gpu_chunk: int) -> int:
    if isinstance(tier, GpuCacheTierConfig):
        return gpu_chunk
    if isinstance(tier, HostCacheTierConfig):
        return _HOST_GRANULARITY
    if isinstance(tier, DiskCacheTierConfig):
        return _DISK_GRANULARITY
    raise ValueError(f"Unsupported cache tier {type(tier).__name__}")


def _largest_scale(
    cost: Callable[[int], int],
    budget: int,
    scale_cap: int,
) -> int:
    """Largest scale in ``[0, scale_cap]`` whose cost fits ``budget``; 0 if none does."""
    if cost(0) > budget:
        return 0
    low = 0
    high = 1
    while high <= scale_cap and cost(high) <= budget:
        low = high
        high *= 2
    high = min(high, scale_cap + 1)
    # cost(low) <= budget, and cost(high) > budget or high is past the cap.
    while low + 1 < high:
        middle = (low + high) // 2
        if cost(middle) <= budget:
            low = middle
        else:
            high = middle
    return low


def _tier_slot_counts(
    need: Sequence[int],
    floors: Sequence[int],
    page_pool_bytes: Sequence[Sequence[int]],
    granularity: int,
    budget: int,
    tier_name: str,
) -> list[int]:
    """Largest page counts, proportional to ``need``, that fit ``budget`` in one tier.

    The counts of a scale ``m`` are ``max(floor, m * need / max(need))`` for every life cycle, so
    the life cycle with the largest need holds ``m`` pages and the others hold their share of
    that. The floors win over the proportion, and when even the floors do not fit the budget they
    are returned anyway.
    """
    reference = max(max(need), 1)

    def counts_at(scale: int) -> list[int]:
        return [
            min(MAX_SLOT_COUNT, max(floor, scale * count // reference))
            for floor, count in zip(floors, need)
        ]

    def cost(scale: int) -> int:
        return sum(
            _round_up(count * pool_bytes, granularity)
            for count, pools in zip(counts_at(scale), page_pool_bytes)
            for pool_bytes in pools
        )

    if cost(0) > budget:
        logger.warning(
            f"lifecycle_slot_counts: the {budget} byte budget of {tier_name} cannot hold the "
            f"minimum page counts ({cost(0)} bytes); using the minimum"
        )
    return counts_at(_largest_scale(cost, budget, MAX_SLOT_COUNT))


def _local_slot_counts(
    config: Any,
    layouts: Sequence[LifecycleLayout],
    cold_page_bytes: Optional[Sequence[int]],
) -> tuple[list[int], list[int], list[list[int]]]:
    """(need, hot floors, counts per tier) of this rank, in local life cycle order."""
    keys = [layout.key for layout in layouts]
    tokens_per_block = int(config.tokens_per_block)
    if config.swa_scratch_reuse is not None:
        raise ValueError("lifecycle_slot_counts is not supported with swa_scratch_reuse")
    need = lifecycle_page_need(config, keys)
    floors = slots_from_constraints(
        keys, config.constraints, tokens_per_block, config.max_util_for_resume
    )
    cold_bytes = list(cold_page_bytes) if cold_page_bytes is not None else None
    if cold_bytes is not None and len(cold_bytes) != len(layouts):
        raise ValueError("cold_page_bytes must have one entry per life cycle")

    hot_tier = config.cache_tiers[0]
    gpu_chunk = gpu_granularity(int(hot_tier.quota))
    rows = []
    for level, tier in enumerate(config.cache_tiers):
        if level == 0:
            tier_floors = floors
            pools = [layout.pool_slot_bytes for layout in layouts]
        else:
            tier_floors = [1] * len(layouts)
            pools = [
                (cold_bytes[index] if cold_bytes is not None else layout.slot_bytes,)
                for index, layout in enumerate(layouts)
            ]
        rows.append(
            _tier_slot_counts(
                need,
                tier_floors,
                pools,
                _tier_granularity(tier, gpu_chunk),
                int(tier.quota),
                f"cache tier {level}",
            )
        )
    return need, floors, rows


def solve_lifecycle_slot_counts(
    config: Any,
    *,
    allgather: Optional[Callable[[object], Sequence[object]]] = None,
    cold_page_bytes: Optional[Sequence[int]] = None,
) -> list[list[int]]:
    """Page counts for ``config.lifecycle_slot_counts`` that every rank of a group agrees on.

    ``config`` is the final cache config of this rank: its layers (after zero-size buffers are
    removed), cache tiers (whose quotas are the byte budgets, already net of everything else that
    uses the device), typical step, constraints and ``max_util_for_resume``. The result has one
    row per cache tier and one count per life cycle in life cycle id order, ready to be assigned
    to ``config.lifecycle_slot_counts``.

    ``allgather`` is the group's object allgather (``Distributed.allgather``); without it the
    rank is alone. The counts are reduced with the minimum over the group, so no rank is asked
    for more pages than its budget holds. The group must hold the same life cycles and derive the
    same page need and minimum counts and the same kinds of cache tiers; a mismatch raises
    ``ValueError`` on every rank.

    ``cold_page_bytes`` is the size of a cold page of every life cycle in life cycle id order. It
    defaults to the hot slot bytes, which is what the default cold-page codec stores.
    """
    layouts = lifecycle_layouts(config)
    need, floors, rows = _local_slot_counts(config, layouts, cold_page_bytes)

    canonical = sorted(range(len(layouts)), key=lambda index: layouts[index].key)
    signature = (
        tuple(layouts[index].key for index in canonical),
        tuple(need[index] for index in canonical),
        tuple(floors[index] for index in canonical),
        tuple(type(tier).__name__ for tier in config.cache_tiers),
    )
    counts = tuple(tuple(row[index] for index in canonical) for row in rows)

    if allgather is not None:
        gathered = list(allgather((signature, counts)))
        if any(entry[0] != signature for entry in gathered):
            raise ValueError(
                "lifecycle_slot_counts needs every rank to hold the same life cycles and cache "
                "tiers and to derive the same page need and minimum counts from the same "
                "constraints; "
                f"got {[entry[0] for entry in gathered]}"
            )
        counts = tuple(
            tuple(min(column) for column in zip(*(entry[1][level] for entry in gathered)))
            for level in range(len(rows))
        )

    local_position = {index: position for position, index in enumerate(canonical)}
    return [[int(row[local_position[index]]) for index in range(len(layouts))] for row in counts]
