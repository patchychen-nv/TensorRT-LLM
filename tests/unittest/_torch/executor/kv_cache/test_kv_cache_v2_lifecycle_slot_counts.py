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
"""``KVCacheManagerV2`` with fixed page counts per life cycle, on a real native storage manager.

The layers below have four attention windows over a 256 token context: layers 0 and 2 see the
whole context, layer 1 a window of 128 tokens and layer 3 a window of 64 tokens. Windows 128 and 64
have equal slot bytes, so the byte-quota sizing puts them in one pool group while fixed counts give
every life cycle its own.
"""

from functools import partial

import pytest
import torch

from tensorrt_llm._torch.pyexecutor.kv_cache import kv_cache_manager_v2 as kv_cache_v2_module
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import (
    KVCacheManagerV2,
    _KVCacheManagerInitStatus,
)
from tensorrt_llm._torch.pyexecutor.kv_cache.lifecycle_slot_counts import (
    lifecycle_layouts,
    solve_lifecycle_slot_counts,
)
from tensorrt_llm.bindings import DataType
from tensorrt_llm.bindings.internal.batch_manager import CacheType
from tensorrt_llm.llmapi.llm_args import KvCacheConfig
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.runtime.kv_cache_manager_v2 import CacheLevel

MAX_SEQ_LEN = 256
TOKENS_PER_BLOCK = 8
WINDOWS = [MAX_SEQ_LEN, 128, MAX_SEQ_LEN, 64]
HOT_PAGES = [90, 40, 24]  # no window, window 128, window 64: the order the layers use them
HOST_PAGES = [120, 60, 30]


@pytest.fixture(autouse=True)
def cuda_cleanup():
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    torch.cuda.init()
    yield
    torch.cuda.empty_cache()


def make_manager(host: bool = True, **kwargs) -> KVCacheManagerV2:
    return KVCacheManagerV2(
        KvCacheConfig(
            max_gpu_total_bytes=16 << 20,
            host_cache_size=(16 << 20) if host else 0,
            max_attention_window=WINDOWS,
            enable_block_reuse=False,
        ),
        CacheType.SELF,
        num_layers=len(WINDOWS),
        num_kv_heads=2,
        head_dim=64,
        tokens_per_block=TOKENS_PER_BLOCK,
        max_seq_len=MAX_SEQ_LEN,
        max_batch_size=1,
        mapping=Mapping(world_size=1, rank=0, tp_size=1, pp_size=1),
        dtype=DataType.HALF,
        vocab_size=16,
        **kwargs,
    )


def pages(manager: KVCacheManagerV2, level: int) -> list[int]:
    return [stats.total for stats in manager.impl.get_storage_statistics(CacheLevel(level))]


def test_without_counts_the_pools_are_sized_by_bytes_and_merged() -> None:
    manager = make_manager()
    try:
        assert manager.kv_cache_manager_py_config.lifecycle_slot_counts is None
        # Windows 128 and 64 have equal slot bytes and share a pool group; the groups are ordered by
        # slot bytes, so the life cycle without a window, whose slots are twice as large, is last.
        assert manager.impl.get_life_cycle_pool_group_indices(CacheLevel(0)) == [1, 0, 0]
    finally:
        manager.shutdown()


def test_explicit_counts_give_every_life_cycle_its_own_pool_group() -> None:
    rows = [HOT_PAGES, HOST_PAGES]
    manager = make_manager(lifecycle_slot_counts=rows)
    try:
        assert manager.kv_cache_manager_py_config.lifecycle_slot_counts == rows
        assert pages(manager, 0) == HOT_PAGES
        assert pages(manager, 1) == HOST_PAGES
        assert manager.can_evict
        assert manager.impl.get_life_cycle_pool_group_indices(CacheLevel(0)) == [0, 1, 2]
        assert manager.impl.get_life_cycle_pool_group_indices(CacheLevel(1)) == [0, 1, 2]
    finally:
        manager.shutdown()


def test_a_function_computes_the_counts_from_the_final_config() -> None:
    seen = []

    def solve(config):
        seen.append(config)
        return solve_lifecycle_slot_counts(config)

    manager = make_manager(lifecycle_slot_counts=solve)
    try:
        # The function sees the config after zero-size buffers are removed and before the counts.
        assert len(seen) == 1 and seen[0].lifecycle_slot_counts is None
        config = manager.kv_cache_manager_py_config
        assert config.lifecycle_slot_counts == solve_lifecycle_slot_counts(config)
        assert pages(manager, 0) == config.lifecycle_slot_counts[0]
        assert pages(manager, 1) == config.lifecycle_slot_counts[1]
        keys = [layout.key for layout in lifecycle_layouts(config)]
        assert [key.window_size for key in keys] == [0, 128, 64]
    finally:
        manager.shutdown()

    partial_solve = partial(solve_lifecycle_slot_counts, cold_page_bytes=[4096, 4096, 4096])
    manager = make_manager(lifecycle_slot_counts=partial_solve)
    try:
        assert pages(manager, 1) != config.lifecycle_slot_counts[1]
    finally:
        manager.shutdown()


def test_counts_without_a_host_tier_have_one_row() -> None:
    manager = make_manager(host=False, lifecycle_slot_counts=[HOT_PAGES])
    try:
        assert manager.kv_cache_manager_py_config.lifecycle_slot_counts == [HOT_PAGES]
        assert not manager.can_evict
        assert pages(manager, 0) == HOT_PAGES
    finally:
        manager.shutdown()


def test_a_failed_host_tier_takes_its_row_of_counts_with_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every rank rebuilds the manager without the host tier when one rank cannot register it."""
    calls = []

    def sync(status, mapping):
        calls.append(status)
        return _KVCacheManagerInitStatus.USE_NO_HOST if len(calls) == 1 else status

    monkeypatch.setattr(kv_cache_v2_module, "_sync_kv_cache_manager_init_status", sync)
    manager = make_manager(lifecycle_slot_counts=[HOT_PAGES, HOST_PAGES])
    try:
        assert len(calls) == 2
        config = manager.kv_cache_manager_py_config
        assert len(config.cache_tiers) == 1
        assert config.lifecycle_slot_counts == [HOT_PAGES]
        assert not manager.can_evict
        assert pages(manager, 0) == HOT_PAGES
    finally:
        manager.shutdown()


def test_a_function_that_returns_the_wrong_number_of_rows_is_rejected() -> None:
    with pytest.raises(ValueError, match="one row per cache tier"):
        make_manager(lifecycle_slot_counts=lambda config: [HOT_PAGES])


def test_counts_and_a_pool_ratio_exclude_each_other() -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        KVCacheManagerV2(
            KvCacheConfig(
                max_gpu_total_bytes=16 << 20,
                host_cache_size=16 << 20,
                max_attention_window=WINDOWS,
                pool_ratio=[0.5, 0.3, 0.2],
            ),
            CacheType.SELF,
            num_layers=len(WINDOWS),
            num_kv_heads=2,
            head_dim=64,
            tokens_per_block=TOKENS_PER_BLOCK,
            max_seq_len=MAX_SEQ_LEN,
            max_batch_size=1,
            mapping=Mapping(world_size=1, rank=0, tp_size=1, pp_size=1),
            dtype=DataType.HALF,
            vocab_size=16,
            lifecycle_slot_counts=[HOT_PAGES, HOST_PAGES],
        )
