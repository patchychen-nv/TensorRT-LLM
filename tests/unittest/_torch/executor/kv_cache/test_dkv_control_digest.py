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

from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest

from tensorrt_llm._torch.attention.backends.sparse.deepseek_v4.cache_manager import (
    DeepseekV4CacheManager,
)
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
from tensorrt_llm.bindings.internal.batch_manager.kv_cache_manager_v2_utils import IndexMapper
from tensorrt_llm.runtime.kv_cache_manager_v2 import CacheLevel

pytestmark = pytest.mark.cpu_only


def _manager(
    free_pages: tuple[tuple[int, ...], ...],
    manager_type: type[KVCacheManagerV2] = KVCacheManagerV2,
) -> KVCacheManagerV2:
    manager = object.__new__(manager_type)
    manager.impl = SimpleNamespace(
        cache_tier_list=[None] * len(free_pages),
        get_storage_statistics=Mock(
            side_effect=lambda level: [SimpleNamespace(free=free) for free in free_pages[level]]
        ),
    )
    manager.index_mapper = IndexMapper(8, 1)
    return manager


@pytest.mark.parametrize("manager_type", [KVCacheManagerV2, DeepseekV4CacheManager])
def test_digest_reads_every_cache_level_and_pool_group(
    manager_type: type[KVCacheManagerV2],
) -> None:
    free_pages = ((11, 7, 0), (29, 13), (0, 3))
    manager = _manager(free_pages, manager_type)
    manager.index_mapper.add_new_sequence(5)
    manager.index_mapper.add_new_sequence(6)

    assert manager.get_dkv_control_digest() == (2, free_pages)
    assert manager.impl.get_storage_statistics.call_args_list == [
        call(CacheLevel(0)),
        call(CacheLevel(1)),
        call(CacheLevel(2)),
    ]


def test_digest_distinguishes_cache_level_and_pool_group_boundaries() -> None:
    first = _manager(((2, 3), (4,)))
    second = _manager(((2,), (3, 4)))
    assert first.get_dkv_control_digest() != second.get_dkv_control_digest()


def test_digest_counts_index_leases_without_request_or_physical_slot_ids() -> None:
    first = _manager(((4, 5),))
    second = _manager(((4, 5),))
    first.index_mapper.add_new_sequence(11)
    second.index_mapper.add_new_sequence(91)
    second.index_mapper.add_new_sequence(92)
    second.index_mapper.remove_sequence(91)

    assert first.get_dkv_control_digest() == second.get_dkv_control_digest()
    second.index_mapper.add_new_sequence(93)
    assert first.get_dkv_control_digest() != second.get_dkv_control_digest()


def test_digest_is_an_immutable_snapshot_of_current_free_counts() -> None:
    manager = _manager(((9, 8),))
    counts = [SimpleNamespace(free=9), SimpleNamespace(free=8)]
    manager.impl.get_storage_statistics.side_effect = None
    manager.impl.get_storage_statistics.return_value = counts
    before = manager.get_dkv_control_digest()
    counts[0].free = 7

    assert before == (0, ((9, 8),))
    assert manager.get_dkv_control_digest() == (0, ((7, 8),))
    assert hash(before) == hash((0, ((9, 8),)))


def test_digest_ignores_per_request_state_and_does_not_drain_statistics() -> None:
    manager = _manager(((5,),))
    manager.kv_cache_map = Mock()
    manager.get_dkv_state_fingerprint = Mock(side_effect=AssertionError("debug-only traversal"))
    manager.impl.get_and_reset_iteration_peak_block_stats_by_level = Mock(
        side_effect=AssertionError("drains counters")
    )

    assert manager.get_dkv_control_digest() == (0, ((5,),))
    manager.kv_cache_map.assert_not_called()
    manager.get_dkv_state_fingerprint.assert_not_called()
    manager.impl.get_and_reset_iteration_peak_block_stats_by_level.assert_not_called()
