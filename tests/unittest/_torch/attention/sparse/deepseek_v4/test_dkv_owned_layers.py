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
"""A DeepSeek-V4 cache manager that stores the KV of the layers a rank owns (layer-split layout)."""

from collections.abc import Iterator

import pytest
from utils.util import skip_pre_blackwell

from tensorrt_llm._torch.attention.backends.sparse.deepseek_v4.cache_manager import (
    DeepseekV4CacheManager,
)
from tensorrt_llm._torch.attention.backends.sparse.deepseek_v4.params import DeepseekV4AttentionType
from tensorrt_llm.bindings import DataType
from tensorrt_llm.bindings.internal.batch_manager import CacheType
from tensorrt_llm.llmapi.llm_args import DeepSeekV4SparseAttentionConfig, KvCacheConfig
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.runtime.kv_cache_manager_v2 import BAD_PAGE_INDEX

from .test_deepseek_v4_cache_manager import TestDeepseekV4CacheManager as _ManagerFactory

pytestmark = skip_pre_blackwell

_RATIOS = [1, 4, 128, 4, 128]
_TOKENS = 700


def _manager(**kwargs) -> DeepseekV4CacheManager:
    return DeepseekV4CacheManager(
        kv_cache_config=KvCacheConfig(
            enable_block_reuse=False,
            enable_swa_scratch_reuse=False,
            max_tokens=2048,
            event_buffer_max_size=0,
        ),
        kv_cache_type=CacheType.SELFKONLY,
        num_layers=len(_RATIOS),
        num_kv_heads=1,
        head_dim=512,
        tokens_per_block=128,
        max_seq_len=2048,
        max_batch_size=2,
        max_input_len=2048,
        mapping=Mapping(world_size=1, rank=0, tp_size=1, pp_size=1),
        dtype=DataType.FP8,
        compressor_dtype=DataType.FLOAT,
        vocab_size=129280,
        max_num_tokens=2048,
        sparse_attn_config=DeepSeekV4SparseAttentionConfig(
            index_head_dim=128, window_size=128, compress_ratios=_RATIOS, indexer_k_dtype="fp8"
        ),
        **kwargs,
    )


@pytest.fixture
def managers() -> Iterator[tuple[DeepseekV4CacheManager, DeepseekV4CacheManager]]:
    everything, owned = _manager(), _manager(owned_layers=[2, 3, 4])
    try:
        yield everything, owned
    finally:
        everything.shutdown()
        owned.shutdown()


def _pages(manager: DeepseekV4CacheManager, layer: int, role) -> int:
    indices = manager.get_cache_indices(0, layer, role)
    return sum(index != BAD_PAGE_INDEX for index in indices)


def test_the_manager_holds_the_layers_it_owns_and_no_others(managers) -> None:
    everything, owned = managers
    assert everything.pp_layers == list(range(len(_RATIOS)))
    assert owned.pp_layers == [2, 3, 4]
    assert owned.num_local_layers == 3
    assert owned.layer_offsets == {2: 0, 3: 1, 4: 2}
    assert (
        owned.get_buffers(3, DeepseekV4AttentionType.COMPRESS).shape[1:]
        == (everything.get_buffers(3, DeepseekV4AttentionType.COMPRESS).shape[1:])
    )
    with pytest.raises(KeyError):
        owned.get_buffers(1, DeepseekV4AttentionType.SWA)
    # The tables the attention op takes cover the owned layers only.
    assert owned.kv_cache_pool_pointers.shape[0] == 3
    assert everything.kv_cache_pool_pointers.shape[0] == len(_RATIOS)


def test_a_request_takes_the_pages_of_the_owned_layers_it_takes_in_a_full_manager(managers) -> None:
    everything, owned = managers
    factory = _ManagerFactory()
    requests = []
    try:
        for manager in managers:
            request = factory._create_request(0, _TOKENS)
            requests.append(request)
            assert manager.prepare_context(request)
            assert manager.resize_context(request, _TOKENS)
        for layer in (2, 3, 4):
            for role in (
                DeepseekV4AttentionType.SWA,
                DeepseekV4AttentionType.COMPRESS,
                DeepseekV4AttentionType.COMPRESSOR_KV,
            ):
                assert _pages(owned, layer, role) == _pages(everything, layer, role), (layer, role)
    finally:
        for manager, request in zip(managers, requests):
            manager.free_resources(request)


def test_layers_that_are_not_distinct_or_not_in_the_model_are_rejected() -> None:
    with pytest.raises(ValueError, match="distinct layers of the model"):
        _manager(owned_layers=[2, 2, 3])
    with pytest.raises(ValueError, match="distinct layers of the model"):
        _manager(owned_layers=[2, 7])
    with pytest.raises(ValueError, match="at least one layer"):
        _manager(owned_layers=[])
