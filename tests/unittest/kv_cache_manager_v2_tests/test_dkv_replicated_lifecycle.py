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

from collections.abc import Callable

import pytest
import torch

from tensorrt_llm.runtime.kv_cache_manager_v2 import (
    AttentionLayerConfig,
    BufferConfig,
    CacheLevel,
    CudaStream,
    DataRole,
    GpuCacheTierConfig,
    HostCacheTierConfig,
    KVCacheManager,
    KVCacheManagerConfig,
    LayerId,
    ReuseScope,
    _KVCache,
)


@pytest.mark.parametrize("replicas", [2, 4])
@pytest.mark.parametrize("with_host", [False, True])
def test_replicated_kv_lifecycle_keeps_logical_state_and_free_counts(
    replicas: int, with_host: bool
) -> None:
    """Exercise the selected C++ or Python core without comparing physical slots."""
    streams = [torch.cuda.Stream() for _ in range(replicas)]
    tiers = [GpuCacheTierConfig(quota=4 << 20)]
    if with_host:
        tiers.append(HostCacheTierConfig(quota=4 << 20))
    config = KVCacheManagerConfig(
        tokens_per_block=4,
        cache_tiers=tiers,
        layers=[
            AttentionLayerConfig(
                layer_id=LayerId(0),
                buffers=[BufferConfig(DataRole("key"), 1024)],
            ),
            AttentionLayerConfig(
                layer_id=LayerId(1),
                buffers=[BufferConfig(DataRole("key"), 2048)],
                sliding_window_size=8,
            ),
        ],
    )
    managers = [KVCacheManager(config) for _ in range(replicas)]
    live_caches: list[_KVCache] = []

    def check_storage() -> None:
        snapshots = [
            tuple(
                tuple(
                    (item.total, item.free, item.evictable)
                    for item in manager.get_storage_statistics(CacheLevel(level))
                )
                for level in range(len(tiers))
            )
            for manager in managers
        ]
        assert all(snapshot == snapshots[0] for snapshot in snapshots)

    def apply(caches: list[_KVCache], operation: Callable[[_KVCache], object]) -> list[object]:
        results = [operation(cache) for cache in caches]
        assert all(result == results[0] for result in results)
        states = [
            (cache.capacity, cache.history_length, cache.num_committed_tokens, cache.is_active)
            for cache in caches
        ]
        assert all(state == states[0] for state in states)
        check_storage()
        return results

    try:
        for iteration, prompt in enumerate(
            [list(range(24)), list(range(24)) + list(range(100, 108))]
        ):
            caches = [manager.create_kv_cache(ReuseScope(), prompt) for manager in managers]
            live_caches.extend(caches)
            reused = [cache.num_committed_tokens for cache in caches]
            assert all(length == reused[0] for length in reused)
            assert reused[0] == (0 if iteration == 0 else 24)
            resumed = [
                cache.resume(CudaStream(stream.cuda_stream))
                for cache, stream in zip(caches, streams)
            ]
            assert all(resumed)
            assert all(apply(caches, lambda cache: cache.resize(len(prompt))))
            apply(caches, lambda cache: cache.commit(prompt[reused[0] :]))
            apply(caches, lambda cache: cache.stop_committing())
            # A failed allocation must roll back identically on every replica.
            assert not any(apply(caches, lambda cache: cache.resize(1 << 16)))
            apply(caches, lambda cache: cache.suspend())
            resumed = [
                cache.resume(CudaStream(stream.cuda_stream))
                for cache, stream in zip(caches, streams)
            ]
            assert all(resumed)
            check_storage()
            for cache in caches:
                cache.close()
                live_caches.remove(cache)
            check_storage()
    finally:
        for cache in live_caches:
            cache.close()
        for stream in streams:
            stream.synchronize()
        for manager in managers:
            manager.shutdown()
