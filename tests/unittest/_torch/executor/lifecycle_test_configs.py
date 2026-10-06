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
"""Cache configs of DeepSeek-V4-like layer subsets for the tests of ``lifecycle_slot_counts``.

A V4 model layer becomes one to three attention layer configs, one per life cycle it touches:

* ``swa``: a sliding-window layer, one config of window 128.
* ``hca``: a heavily compressed layer, a window-128 config (latent and state) and a full-attention
  config (compressed KV).
* ``csa``: a compressed sparse layer, a window-128 config (latent), a full-attention config
  (compressed KV and indexer KV) and a window-8 config (state).

Every rank of a group holds a different subset of the layers, so the slot bytes of the same life
cycle differ between ranks and, with equal buffer sizes, life cycles that are distinct on one rank
can have equal slot bytes on another. The default sizes make window 128 and window 8 equal on a
rank whose count of ``csa`` layers equals its count of window-128 configs.
"""

import math
from collections.abc import Sequence
from typing import Optional

from tensorrt_llm._torch.pyexecutor.kv_cache.lifecycle_slot_counts import (
    gpu_granularity,
    lifecycle_layouts,
)
from tensorrt_llm.runtime.kv_cache_manager_v2 import (
    AttentionLayerConfig,
    BatchDesc,
    BufferConfig,
    DataRole,
    GpuCacheTierConfig,
    HostCacheTierConfig,
    KVCacheDesc,
    KVCacheManagerConfig,
    LayerId,
)

SWA_WINDOW = 128
STATE_WINDOW = 8
TOKENS_PER_BLOCK = 8

# Buffer bytes of one block of one layer.
SWA_BYTES = 2048
HCA_STATE_BYTES = 1024
HCA_COMPRESS_BYTES = 512
CSA_COMPRESS_BYTES = 2048
CSA_INDEXER_BYTES = 512
CSA_STATE_BYTES = 2048


def _config(layer_id: int, window: Optional[int], buffers: Sequence[tuple[str, int]]):
    return AttentionLayerConfig(
        layer_id=LayerId(layer_id),
        buffers=[BufferConfig(DataRole(role), size) for role, size in buffers],
        sliding_window_size=window,
    )


def v4_like_layers(
    kinds: Sequence[str], *, state_window_first: bool = False
) -> list[AttentionLayerConfig]:
    """Attention layer configs of a rank that holds the model layers ``kinds``.

    ``state_window_first`` moves the first window-8 config in front of everything, so the life
    cycles first appear as (window 8, window 128, no window) instead of (window 128, no window,
    window 8) and a rank's life cycle ids no longer match the ids of a rank that does not.
    """
    specs: list[tuple[Optional[int], list[tuple[str, int]]]] = []
    for kind in kinds:
        if kind == "swa":
            specs.append((SWA_WINDOW, [("swa", SWA_BYTES)]))
        elif kind == "hca":
            specs.append((SWA_WINDOW, [("swa", SWA_BYTES), ("state", HCA_STATE_BYTES)]))
            specs.append((None, [("compress", HCA_COMPRESS_BYTES)]))
        elif kind == "csa":
            specs.append((SWA_WINDOW, [("swa", SWA_BYTES)]))
            specs.append((None, [("compress", CSA_COMPRESS_BYTES), ("indexer", CSA_INDEXER_BYTES)]))
            specs.append((STATE_WINDOW, [("state", CSA_STATE_BYTES)]))
        else:
            raise ValueError(f"unknown layer kind {kind!r}")
    if state_window_first:
        first = next(index for index, (window, _) in enumerate(specs) if window == STATE_WINDOW)
        specs.insert(0, specs.pop(first))
    return [_config(layer_id, window, buffers) for layer_id, (window, buffers) in enumerate(specs)]


def v4_like_config(
    kinds: Sequence[str],
    *,
    gpu_quota: int = 16 << 20,
    host_quota: Optional[int] = 16 << 20,
    state_window_first: bool = False,
    typical_step: Optional[BatchDesc] = None,
    constraints: Optional[list[BatchDesc]] = None,
    lifecycle_slot_counts: Optional[list[list[int]]] = None,
    enable_partial_reuse: bool = True,
    max_util_for_resume: float = 0.97,
) -> KVCacheManagerConfig:
    tiers = [GpuCacheTierConfig(quota=gpu_quota)]
    if host_quota is not None:
        tiers.append(HostCacheTierConfig(quota=host_quota))
    return KVCacheManagerConfig(
        tokens_per_block=TOKENS_PER_BLOCK,
        cache_tiers=tiers,
        layers=v4_like_layers(kinds, state_window_first=state_window_first),
        typical_step=typical_step,
        constraints=constraints or [],
        lifecycle_slot_counts=lifecycle_slot_counts,
        enable_partial_reuse=enable_partial_reuse,
        max_util_for_resume=max_util_for_resume,
    )


def typical_batch(num_requests: int = 4, length: int = 96) -> BatchDesc:
    """A batch of ``num_requests`` requests that are ``length`` tokens into their generation."""
    return BatchDesc([KVCacheDesc(capacity=length + 1, history_length=length)] * num_requests)


def tier_bytes(config: KVCacheManagerConfig, level: int, counts: Sequence[int]) -> int:
    """Bytes the pools of one cache tier take for ``counts`` pages per life cycle.

    Every pool rounds its size up to the allocation granularity of the tier. A cold page holds all
    pools of its life cycle, so the cold tiers have one pool per life cycle.
    """
    tier = config.cache_tiers[level]
    if isinstance(tier, GpuCacheTierConfig):
        granularity = gpu_granularity(config.cache_tiers[0].quota)
    elif isinstance(tier, HostCacheTierConfig):
        granularity = 4096
    else:
        raise ValueError(f"unexpected cache tier {type(tier).__name__}")
    total = 0
    for layout, count in zip(lifecycle_layouts(config), counts):
        pools = layout.pool_slot_bytes if level == 0 else (layout.slot_bytes,)
        for pool_bytes in pools:
            total += math.ceil(count * pool_bytes / granularity) * granularity
    return total
