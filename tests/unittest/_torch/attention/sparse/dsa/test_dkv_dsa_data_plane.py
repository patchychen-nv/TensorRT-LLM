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
"""The DKV data plane of the layer split on the DSA cache manager of GLM-5.3, on a GPU.

The working cache is a second ``DSACacheManagerV2``, and the streamer runs a group of one rank that
owns every layer, so each step is a copy between the cache of the rank and its working cache. The
geometry is GLM-5.3's with ten layers: the MLA latent KV on every layer and the indexer K cache on
the "full" indexer layers (0, 1, 2 and 6). The pages are checked where the attention of DSA reads
them: the latent KV in the primary pool at the base page index of a block, the indexer K in the
indexer pool of the layer at the base page index times the indexer page scale.
"""

from collections.abc import Iterator
from types import SimpleNamespace

import pytest
import torch
from utils.util import skip_pre_blackwell

from tensorrt_llm._torch.attention.backends.sparse.dsa import DSACacheManagerV2
from tensorrt_llm._torch.pyexecutor.dkv_plan import PlanRequest
from tensorrt_llm._torch.pyexecutor.dkv_streamer import DkvStreamer
from tensorrt_llm._torch.pyexecutor.dkv_working import (
    DkvWorkingSession,
    fetch_blocks,
    manager_buffers,
    staged_blocks,
    working_slot_counts,
    writeback_blocks,
)
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import Role
from tensorrt_llm.bindings import DataType
from tensorrt_llm.bindings.internal.batch_manager import CacheType
from tensorrt_llm.llmapi.llm_args import DeepSeekSparseAttentionConfig, KvCacheConfig
from tensorrt_llm.mapping import Mapping

pytestmark = skip_pre_blackwell

_CONFIG = SimpleNamespace(
    architectures=["GlmMoeDsaForCausalLM"],
    num_hidden_layers=10,
    kv_lora_rank=512,
    qk_rope_head_dim=64,
    index_n_heads=32,
    index_head_dim=128,
    index_topk=2048,
    index_topk_freq=4,
    index_skip_topk_offset=3,
    index_topk_pattern=None,
    indexer_rope_interleave=True,
)
_LAYERS = range(_CONFIG.num_hidden_layers)
_FULL_INDEXER_LAYERS = (0, 1, 2, 6)
_BLOCK = 64
_REQUEST = 1
_HISTORY, _CHUNK = 300, 400  # the cache holds 700 tokens, the first 300 of them already computed
_KEY, _INDEX_KEY = str(Role.KEY), str(Role.INDEX_KEY)


def _manager(dtype, **kwargs) -> DSACacheManagerV2:
    return DSACacheManagerV2(
        KvCacheConfig(
            enable_block_reuse=False,
            max_tokens=4096,
            event_buffer_max_size=0,
            host_cache_size=0,
        ),
        CacheType.SELFKONLY,
        num_layers=len(_LAYERS),
        num_kv_heads=1,
        head_dim=_CONFIG.kv_lora_rank + _CONFIG.qk_rope_head_dim,
        tokens_per_block=_BLOCK,
        max_seq_len=4096,
        max_batch_size=2,
        mapping=Mapping(world_size=1, rank=0, tp_size=1, pp_size=1),
        dtype=dtype,
        max_num_tokens=2048,
        sparse_attention_config=DeepSeekSparseAttentionConfig(
            index_n_heads=_CONFIG.index_n_heads,
            index_head_dim=_CONFIG.index_head_dim,
            index_topk=_CONFIG.index_topk,
            indexer_rope_interleave=_CONFIG.indexer_rope_interleave,
            indexer_k_dtype="fp8",
        ),
        pretrained_config=_CONFIG,
        **kwargs,
    )


@pytest.fixture(params=[DataType.BF16, DataType.FP8], ids=["bf16 kv", "fp8 kv"])
def caches(request) -> Iterator[tuple[DSACacheManagerV2, DSACacheManagerV2]]:
    """The cache of a rank that owns every layer, with the request's 700 tokens, and a working
    cache."""
    owner = _manager(request.param)
    working = _manager(
        request.param,
        lifecycle_slot_counts=lambda config: working_slot_counts(
            config, max_batch_size=2, max_num_tokens=2048, staging_tokens=4096
        ),
    )
    try:
        assert owner.dkv_open_cache(_REQUEST, _HISTORY, _HISTORY + _CHUNK)
        yield owner, working
    finally:
        owner.dkv_close_cache(_REQUEST)
        working.shutdown()
        owner.shutdown()


def _roles(layer: int) -> tuple[str, ...]:
    return (_KEY, _INDEX_KEY) if layer in _FULL_INDEXER_LAYERS else (_KEY,)


def _pages(manager: DSACacheManagerV2, layer: int, role: str) -> dict[int, torch.Tensor]:
    """The pages of the request by block, where the attention of DSA reads them."""
    base = manager.get_batch_cache_indices([_REQUEST], raw_indices=True)[0]
    if role == _KEY:
        pool = manager.get_unique_primary_pool()
        local_layer = manager.layer_offsets[layer]
        return {
            block: pool[index, local_layer].view(torch.uint8).reshape(-1)
            for block, index in enumerate(base)
            if index >= 0
        }
    pool = manager.get_indexer_k_cache_buffers(layer)
    scale = manager.indexer_k_cache_page_scale
    return {
        block: pool[index * scale].view(torch.uint8).reshape(-1)
        for block, index in enumerate(base)
        if index >= 0
    }


def _pattern(layer: int, role: str, block: int, generation: int, size: int) -> torch.Tensor:
    salt = layer * 97 + (role == _INDEX_KEY) * 13 + block * 7 + generation * 1009
    values = (torch.arange(size, dtype=torch.int64, device="cuda") * 7 + salt * 31) % 253
    return values.to(torch.uint8)


def _key(manager: DSACacheManagerV2, layer: int, role: str):
    return manager_buffers(manager)[layer, role].buffer.key


def test_the_working_cache_lists_the_latent_kv_of_every_layer_and_the_indexer_k_of_full_layers(
    caches,
) -> None:
    _, working = caches
    names = sorted(manager_buffers(working))
    assert names == sorted((layer, role) for layer in _LAYERS for role in _roles(layer))
    # One life cycle holds both: the whole history of every layer.
    assert len({entry.buffer.key for entry in manager_buffers(working).values()}) == 1


@pytest.mark.parametrize("fill", ["", "nan"])
def test_the_streamer_fetches_the_cached_pages_and_writes_back_the_new_ones(caches, fill) -> None:
    owner, working = caches
    # Every page of the request in the owner's cache gets a content of its own.
    for layer in _LAYERS:
        for role in _roles(layer):
            for block, page in _pages(owner, layer, role).items():
                page.copy_(_pattern(layer, role, block, 0, page.numel()))
    torch.cuda.synchronize()

    streamer = DkvStreamer(owner, working, None, (0,) * len(_LAYERS), 0, group_size=1, fill=fill)
    session = DkvWorkingSession(working, streamer)
    streamer.set_plan(streamer.plan_for([PlanRequest(_REQUEST, 0, _HISTORY, _CHUNK)]))
    session.begin_batch([_REQUEST], [_HISTORY], [_CHUNK])
    fetched_bytes = written_bytes = 0
    try:
        for layer in _LAYERS:
            streamer.on_layer(layer)
            # The forward pass waits for the fetch of the layer; so does this check.
            torch.cuda.current_stream().synchronize()
            for role in _roles(layer):
                key = _key(working, layer, role)
                pages = _pages(working, layer, role)
                first, count = staged_blocks(key, _HISTORY, _CHUNK, _BLOCK)
                assert sorted(pages) == list(range(first, first + count)), (layer, role)
                first, count = fetch_blocks(key, _HISTORY, _CHUNK, _BLOCK)
                for block, page in pages.items():
                    if first <= block < first + count:
                        expected = _pattern(layer, role, block, 0, page.numel())
                        assert torch.equal(page, expected), ("fetch", layer, role, block)
                        fetched_bytes += page.numel()
                    elif fill:
                        # What was not fetched holds the fill, not what the pages held before.
                        assert (page == 0xFF).all(), ("fill", layer, role, block)
                # The attention writes the pages of the new tokens, on the stream of the forward.
                first, count = writeback_blocks(key, _HISTORY, _CHUNK, _BLOCK)
                for block in range(first, first + count):
                    page = pages[block]
                    page.copy_(_pattern(layer, role, block, 1, page.numel()))
                    written_bytes += page.numel()
        streamer.end_forward()
        streamer.drain(60.0)
    finally:
        session.end()

    stats = streamer.stats
    assert stats.bytes_local_fetch == fetched_bytes > 0
    assert stats.bytes_local_writeback == written_bytes > 0
    for layer in _LAYERS:
        for role in _roles(layer):
            first, count = writeback_blocks(_key(owner, layer, role), _HISTORY, _CHUNK, _BLOCK)
            for block, page in _pages(owner, layer, role).items():
                generation = int(first <= block < first + count)
                expected = _pattern(layer, role, block, generation, page.numel())
                assert torch.equal(page, expected), ("write back", layer, role, block)
