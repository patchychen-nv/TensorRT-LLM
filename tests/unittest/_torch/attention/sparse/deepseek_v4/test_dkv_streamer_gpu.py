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
"""The page copies of the DKV data plane and its streamer on real cache managers, on a GPU.

The streamer runs a group of one rank: the rank owns every layer, so each step is a copy between
the cache manager of the rank and its working cache. Both are DeepSeek-V4 cache managers, and the
pages are checked where the attention of DeepSeek-V4 reads them (``get_buffers`` at the indices of
``get_cache_indices``), not where the data plane computed them to be.
"""

from collections.abc import Iterator

import numpy as np
import pytest
import torch
from utils.util import skip_pre_blackwell

from tensorrt_llm._torch.attention.backends.sparse.deepseek_v4.cache_manager import (
    DeepseekV4CacheManager,
)
from tensorrt_llm._torch.attention.backends.sparse.deepseek_v4.params import (
    DeepseekV4AttentionType,
    compress_ratio_has_attention,
)
from tensorrt_llm._torch.pyexecutor.dkv_plan import PlanRequest
from tensorrt_llm._torch.pyexecutor.dkv_staging import DkvPageCopier
from tensorrt_llm._torch.pyexecutor.dkv_streamer import DkvStreamer
from tensorrt_llm._torch.pyexecutor.dkv_working import (
    DkvWorkingSession,
    fetch_blocks,
    manager_buffers,
    staged_blocks,
    working_slot_counts,
    writeback_blocks,
)
from tensorrt_llm.bindings import DataType
from tensorrt_llm.bindings.internal.batch_manager import CacheType
from tensorrt_llm.llmapi.llm_args import DeepSeekV4SparseAttentionConfig, KvCacheConfig
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.runtime.kv_cache_manager_v2 import BAD_PAGE_INDEX

pytestmark = skip_pre_blackwell

# A layer of each type, and two layers of each compressed type.
_RATIOS = [1, 4, 128, 4, 128]
_BLOCK = 128
_REQUEST = 1
_HISTORY, _CHUNK = 300, 400  # the cache holds 700 tokens, the first 300 of them already computed

_VARIANTS = {
    "params": [(DataType.FP8, "fp8"), (DataType.FP8, "fp4"), (DataType.BF16, "fp8")],
    "ids": ["fp8 kv, fp8 indexer", "fp8 kv, fp4 indexer", "bf16 kv, fp8 indexer"],
}


def _manager(dtype, indexer: str, **kwargs) -> DeepseekV4CacheManager:
    return DeepseekV4CacheManager(
        kv_cache_config=KvCacheConfig(
            enable_block_reuse=False,
            enable_swa_scratch_reuse=False,
            max_tokens=4096,
            event_buffer_max_size=0,
            host_cache_size=0,
        ),
        kv_cache_type=CacheType.SELFKONLY,
        num_layers=len(_RATIOS),
        num_kv_heads=1,
        head_dim=512,
        tokens_per_block=_BLOCK,
        max_seq_len=4096,
        max_batch_size=2,
        max_input_len=4096,
        mapping=Mapping(world_size=1, rank=0, tp_size=1, pp_size=1),
        dtype=dtype,
        compressor_dtype=DataType.FLOAT,
        vocab_size=129280,
        max_num_tokens=2048,
        sparse_attn_config=DeepSeekV4SparseAttentionConfig(
            index_head_dim=128, window_size=128, compress_ratios=_RATIOS, indexer_k_dtype=indexer
        ),
        **kwargs,
    )


@pytest.fixture(**_VARIANTS)
def caches(request) -> Iterator[tuple[DeepseekV4CacheManager, DeepseekV4CacheManager]]:
    """The cache of a rank that owns every layer, with the request's 700 tokens, and a working
    cache."""
    dtype, indexer = request.param
    owner = _manager(dtype, indexer)
    working = _manager(
        dtype,
        indexer,
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


def _attention_types(layer: int):
    return [
        attn_type
        for attn_type in DeepseekV4AttentionType
        if compress_ratio_has_attention(_RATIOS[layer], attn_type)
    ]


def _pages(manager, layer: int, attn_type) -> dict[int, torch.Tensor]:
    """The pages of the request by block, where the attention reads them."""
    buffer = manager.get_buffers(layer, attn_type)
    return {
        block: buffer[index].view(torch.uint8).reshape(-1)
        for block, index in enumerate(manager.get_cache_indices(_REQUEST, layer, attn_type))
        if index != BAD_PAGE_INDEX
    }


def _pattern(layer: int, attn_type, block: int, generation: int, size: int) -> torch.Tensor:
    salt = layer * 97 + list(DeepseekV4AttentionType).index(attn_type) * 13 + block * 7
    salt += generation * 1009
    values = (torch.arange(size, dtype=torch.int64, device="cuda") * 7 + salt * 31) % 253
    return values.to(torch.uint8)


def _key(manager, layer: int, attn_type):
    return manager_buffers(manager)[layer, str(attn_type.role)].buffer.key


@pytest.mark.parametrize("fill", ["", "nan"])
def test_the_streamer_fetches_the_cached_pages_and_writes_back_the_new_ones(caches, fill) -> None:
    owner, working = caches
    layers = range(len(_RATIOS))
    # Every page of the request in the owner's cache gets a content of its own.
    for layer in layers:
        for attn_type in _attention_types(layer):
            for block, page in _pages(owner, layer, attn_type).items():
                page.copy_(_pattern(layer, attn_type, block, 0, page.numel()))
    torch.cuda.synchronize()

    streamer = DkvStreamer(owner, working, None, (0,) * len(_RATIOS), 0, group_size=1, fill=fill)
    session = DkvWorkingSession(working, streamer)
    streamer.set_plan(streamer.plan_for([PlanRequest(_REQUEST, 0, _HISTORY, _CHUNK)]))
    session.begin_batch([_REQUEST], [_HISTORY], [_CHUNK])
    fetched_bytes = written_bytes = 0
    try:
        for layer in layers:
            streamer.on_layer(layer)
            # The forward pass waits for the fetch of the layer; so does this check.
            torch.cuda.current_stream().synchronize()
            for attn_type in _attention_types(layer):
                key = _key(working, layer, attn_type)
                pages = _pages(working, layer, attn_type)
                first, count = staged_blocks(key, _HISTORY, _CHUNK, _BLOCK)
                assert sorted(pages) == list(range(first, first + count)), (layer, attn_type)
                first, count = fetch_blocks(key, _HISTORY, _CHUNK, _BLOCK)
                for block, page in pages.items():
                    if first <= block < first + count:
                        expected = _pattern(layer, attn_type, block, 0, page.numel())
                        assert torch.equal(page, expected), ("fetch", layer, attn_type, block)
                    elif fill:
                        # What was not fetched holds the fill, not what the pages held before.
                        assert (page == 0xFF).all(), ("fill", layer, attn_type, block)
                    fetched_bytes += page.numel() * (first <= block < first + count)
                # The attention writes the pages of the new tokens, on the stream of the forward.
                first, count = writeback_blocks(key, _HISTORY, _CHUNK, _BLOCK)
                for block in range(first, first + count):
                    page = pages[block]
                    page.copy_(_pattern(layer, attn_type, block, 1, page.numel()))
                    written_bytes += page.numel()
        streamer.end_forward()
        streamer.drain(60.0)
    finally:
        session.end()

    stats = streamer.stats
    assert stats.bytes_local_fetch == fetched_bytes > 0
    assert stats.bytes_local_writeback == written_bytes > 0
    assert stats.messages_sent == stats.messages_received == 0
    for layer in layers:
        for attn_type in _attention_types(layer):
            first, count = writeback_blocks(_key(owner, layer, attn_type), _HISTORY, _CHUNK, _BLOCK)
            for block, page in _pages(owner, layer, attn_type).items():
                generation = int(first <= block < first + count)
                expected = _pattern(layer, attn_type, block, generation, page.numel())
                assert torch.equal(page, expected), ("write back", layer, attn_type, block)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float8_e4m3fn])
def test_the_nan_fill_reads_as_nan_in_every_float_format_of_a_cache(dtype) -> None:
    page = torch.full((64,), 0xFF, dtype=torch.uint8, device="cuda")
    assert torch.isnan(page.view(dtype)).all()


# The page copies


def test_a_copy_of_more_pages_than_one_launch_holds_is_split_and_any_stream_works() -> None:
    copier = DkvPageCopier()
    pages, page_bytes = 3 * DkvPageCopier.MAX_TASKS_PER_CALL + 17, 4096
    source = torch.randint(0, 255, (pages, page_bytes), dtype=torch.uint8, device="cuda")
    target = torch.zeros_like(source)
    order = torch.randperm(pages).tolist()
    pairs = [
        (target.data_ptr() + order[page] * page_bytes, source.data_ptr() + page * page_bytes)
        for page in range(pages)
    ]
    stream = torch.cuda.Stream()
    copier.copy(pairs, page_bytes, stream.cuda_stream)
    stream.synchronize()
    assert torch.equal(target[order], source)
    # An unaligned size is rejected before anything is enqueued.
    with pytest.raises(ValueError, match="multiple of 16"):
        copier.copy(pairs[:1], 100, stream.cuda_stream)
    with pytest.raises(ValueError, match="multiple of 16"):
        copier.copy(pairs[:1], 0, stream.cuda_stream)


def test_copying_by_address_arrays_moves_the_same_pages_as_the_task_list(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    copier = DkvPageCopier()
    pages, page_bytes = 2 * DkvPageCopier.MAX_TASKS_PER_CALL + 5, 4096
    source = torch.randint(0, 255, (pages, page_bytes), dtype=torch.uint8, device="cuda")
    order = torch.randperm(pages).tolist()
    sources = source.data_ptr() + np.arange(pages, dtype=np.int64) * page_bytes
    stream = torch.cuda.Stream()

    def copied() -> torch.Tensor:
        target = torch.zeros_like(source)
        destinations = target.data_ptr() + np.asarray(order, dtype=np.int64) * page_bytes
        copier.copy_addresses(destinations, sources, page_bytes, stream.cuda_stream)
        stream.synchronize()
        return target

    assert torch.equal(copied()[order], source)
    # The same through the per-page task list of an older native build.
    monkeypatch.setattr(copier, "_copy_addresses", None)
    assert torch.equal(copied()[order], source)
    with pytest.raises(ValueError, match="multiple of 16"):
        copier.copy_addresses(sources[:1], sources[:1], 100, stream.cuda_stream)
    with pytest.raises(ValueError, match="destination addresses"):
        copier.copy_addresses(sources[:2], sources[:1], page_bytes, stream.cuda_stream)


def test_copying_runs_follows_the_page_indices_of_a_side_and_names_a_missing_page(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    copier = DkvPageCopier()
    pages, page_bytes = 300, 2048
    source = torch.randint(0, 255, (pages, page_bytes), dtype=torch.uint8, device="cuda")
    order = np.random.default_rng(3).permutation(pages).astype(np.int32)
    stream = torch.cuda.Stream()

    def copied(indexed_side: str) -> torch.Tensor:
        target = torch.zeros_like(source)
        if indexed_side == "destination":
            # Page i of the source goes to page order[i] of the target.
            runs = [
                page_bytes,
                pages,
                target.data_ptr(),
                page_bytes,
                order.ctypes.data,
                source.data_ptr(),
                page_bytes,
                0,
            ]
        else:
            # Page i of the target comes from page order[i] of the source.
            runs = [
                page_bytes,
                pages,
                target.data_ptr(),
                page_bytes,
                0,
                source.data_ptr(),
                page_bytes,
                order.ctypes.data,
            ]
        copier.copy_runs(runs, stream.cuda_stream)
        stream.synchronize()
        return target

    permutation = torch.from_numpy(order.astype(np.int64)).cuda()
    assert torch.equal(copied("destination")[permutation], source)
    assert torch.equal(copied("source"), source[permutation])
    # A block without a page is reported.
    bad = order.copy()
    bad[7] = -1
    missing = [
        page_bytes,
        pages,
        source.data_ptr(),
        page_bytes,
        bad.ctypes.data,
        source.data_ptr(),
        page_bytes,
        0,
    ]
    with pytest.raises(RuntimeError, match="without a page"):
        copier.copy_runs(missing, stream.cuda_stream)
    with pytest.raises(ValueError, match="integers"):
        copier.copy_runs([page_bytes, pages, 0], stream.cuda_stream)
    # The same through the per-page task list of an older native build.
    monkeypatch.setattr(copier, "_copy_runs", None)
    assert torch.equal(copied("destination")[permutation], source)
    with pytest.raises(RuntimeError, match="without a page"):
        copier.copy_runs(missing, stream.cuda_stream)
