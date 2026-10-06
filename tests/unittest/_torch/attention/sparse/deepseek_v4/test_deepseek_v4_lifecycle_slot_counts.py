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
"""A ``DeepseekV4CacheManager`` with fixed page counts per life cycle.

The V4 manager rebuilds the layer configs of the base manager in ``_build_cache_config``, so the
counts must be applied to the layers it ends up with and not to the ones the base manager started
from.
"""

import pytest
from utils.util import skip_pre_blackwell

from tensorrt_llm._torch.attention.backends.sparse.deepseek_v4 import DeepseekV4CacheManager
from tensorrt_llm._torch.pyexecutor.kv_cache.lifecycle_slot_counts import (
    lifecycle_layouts,
    solve_lifecycle_slot_counts,
)
from tensorrt_llm.bindings import DataType
from tensorrt_llm.bindings.internal.batch_manager import CacheType as CacheTypeCpp
from tensorrt_llm.llmapi.llm_args import DeepSeekV4SparseAttentionConfig, KvCacheConfig
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.runtime.kv_cache_manager_v2 import CacheLevel

pytestmark = [skip_pre_blackwell, pytest.mark.skip_less_device_memory(80000)]

# One compressed sparse layer and one heavily compressed layer.
COMPRESS_RATIOS = [4, 128]
TOKENS_PER_BLOCK = 256
MAX_SEQ_LEN = 1024


def make_manager(*, enable_swa_scratch_reuse: bool = False, **kwargs) -> DeepseekV4CacheManager:
    return DeepseekV4CacheManager(
        kv_cache_config=KvCacheConfig(
            enable_block_reuse=False,
            max_tokens=2 * MAX_SEQ_LEN,
            event_buffer_max_size=0,
            enable_swa_scratch_reuse=enable_swa_scratch_reuse,
            host_cache_size=64 << 20,
        ),
        kv_cache_type=CacheTypeCpp.SELFKONLY,
        num_layers=len(COMPRESS_RATIOS),
        num_kv_heads=1,
        head_dim=512,
        tokens_per_block=TOKENS_PER_BLOCK,
        max_seq_len=MAX_SEQ_LEN,
        max_batch_size=2,
        max_input_len=MAX_SEQ_LEN,
        mapping=Mapping(world_size=1, rank=0, tp_size=1, pp_size=1),
        dtype=DataType.BF16,
        compressor_dtype=DataType.FLOAT,
        vocab_size=129280,
        max_num_tokens=2 * (MAX_SEQ_LEN + 1),
        sparse_attn_config=DeepSeekV4SparseAttentionConfig(
            index_head_dim=128, window_size=128, compress_ratios=COMPRESS_RATIOS
        ),
        **kwargs,
    )


def test_the_counts_apply_to_the_layers_the_v4_manager_builds() -> None:
    manager = make_manager(lifecycle_slot_counts=solve_lifecycle_slot_counts)
    try:
        config = manager.kv_cache_manager_py_config
        keys = [layout.key for layout in lifecycle_layouts(config)]
        # The window of the latent cache, the full-attention compressed cache and the state of the
        # compressor of the sparse layer: life cycles that only the V4 layer configs contain.
        assert {0, 8, 128} <= {key.window_size for key in keys}

        rows = config.lifecycle_slot_counts
        assert rows is not None and len(rows) == len(config.cache_tiers) == 2
        assert rows == solve_lifecycle_slot_counts(config)
        for level, row in enumerate(rows):
            assert len(row) == len(keys)
            assert manager.impl.get_life_cycle_pool_group_indices(CacheLevel(level)) == list(
                range(len(keys))
            )
            stats = manager.impl.get_storage_statistics(CacheLevel(level))
            assert [stat.total for stat in stats] == row
    finally:
        manager.shutdown()


def test_the_v4_manager_without_counts_is_sized_by_bytes() -> None:
    manager = make_manager()
    try:
        assert manager.kv_cache_manager_py_config.lifecycle_slot_counts is None
    finally:
        manager.shutdown()


def test_the_solve_rejects_a_v4_manager_with_swa_scratch_reuse() -> None:
    with pytest.raises(ValueError, match="swa_scratch_reuse"):
        make_manager(
            enable_swa_scratch_reuse=True, lifecycle_slot_counts=solve_lifecycle_slot_counts
        )
