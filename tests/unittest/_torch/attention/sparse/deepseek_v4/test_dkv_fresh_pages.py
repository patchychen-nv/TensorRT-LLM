# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Diagnostic zero-fill coverage on native V4 pools and CUDA buffer views."""

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import pytest
import torch
from utils.util import skip_pre_blackwell

from tensorrt_llm._torch.attention.backends.sparse.deepseek_v4.cache_manager import (
    DeepseekV4CacheManager,
)
from tensorrt_llm._torch.attention.backends.sparse.deepseek_v4.params import DeepseekV4AttentionType
from tensorrt_llm.bindings import DataType
from tensorrt_llm.runtime.kv_cache_manager_v2 import BAD_PAGE_INDEX

from .test_deepseek_v4_cache_manager import TestDeepseekV4CacheManager as _ManagerFactory


@dataclass
class _CacheView:
    pages: dict[int, list[int]]
    num_committed_tokens: int = 0

    @property
    def num_blocks(self) -> int:
        return len(self.pages[0])

    def get_base_page_indices(self, pool_id: int) -> np.ndarray:
        return np.asarray(self.pages[pool_id], dtype=np.int32)

    def get_scratch_desc(self, pool_id: int) -> None:
        return None


@pytest.fixture(params=[DataType.FP8, DataType.NVFP4])
def native_pools(request) -> Iterator[tuple[DeepseekV4CacheManager, _CacheView]]:
    factory = _ManagerFactory()
    manager, _ = factory._create_deepseek_v4_cache_manager(
        tokens_per_block=128,
        max_batch_size=1,
        max_seq_len=1024,
        compress_ratios=[4, 4],
        dtype=request.param,
        compressor_dtype=DataType.FLOAT,
        indexer_k_dtype="fp4",
        enable_swa_scratch_reuse=False,
    )
    req = factory._create_request(1, 512)
    manager._fresh_page_fill = None
    original = None
    try:
        assert manager.prepare_context(req)
        assert manager.resize_context(req, 512)
        original = manager.kv_cache_map[1]
        pages = {
            pool: original.get_base_page_indices(pool).tolist() for pool in range(manager.num_pools)
        }
        assert all(len(value) >= 4 for value in pages.values())
        view = _CacheView(pages)
        manager.kv_cache_map[1] = view
        manager._fresh_page_fill = 0.0
        manager._fresh_pages_filled = {}
        yield manager, view
    finally:
        if original is not None:
            manager.kv_cache_map[1] = original
        if 1 in manager.kv_cache_map:
            manager.free_resources(req)
        manager.shutdown()


def _views(manager: DeepseekV4CacheManager) -> list[tuple[torch.Tensor, list[int]]]:
    views = []
    roles = set()
    for layer, attn in manager._layer_attn_to_layer_id:
        roles.add(attn)
        indices = manager.get_cache_indices(1, layer, attn)
        views.append((manager.get_buffers(layer, attn), indices))
        if manager._use_nvfp4_compress and attn == DeepseekV4AttentionType.COMPRESS:
            views.append((manager.get_compress_scale_buffers(layer), indices))
    assert roles == set(DeepseekV4AttentionType)
    return views


def _assert_fill(
    manager: DeepseekV4CacheManager, cache: _CacheView, fresh_ordinals: list[int]
) -> None:
    views = _views(manager)
    target_ranges = set()
    protected_ranges = set()
    for buffer, indices in views:
        assert buffer.is_contiguous()
        expansion = len(indices) // cache.num_blocks
        row_bytes = buffer.stride(0) * buffer.element_size()
        for ordinal in range(cache.num_blocks):
            for offset in range(expansion):
                page = indices[ordinal * expansion + offset]
                if page == BAD_PAGE_INDEX:
                    continue
                assert 0 <= page < buffer.shape[0]
                region = (
                    buffer.data_ptr() + page * row_bytes,
                    buffer.data_ptr() + (page + 1) * row_bytes,
                )
                if ordinal in fresh_ordinals:
                    target_ranges.add(region)
                else:
                    protected_ranges.add(region)
    assert all(
        end <= protected_start or start >= protected_end
        for start, end in target_ranges
        for protected_start, protected_end in protected_ranges
    ), "A fresh role's write must not overlap any role's existing logical blocks"
    expected = []
    for buffer, _ in views:
        value = buffer.view(torch.uint8).clone()
        # SHARED views can overlap with different base pointers. Apply the
        # union of expected physical byte writes to every intersecting view.
        base = buffer.data_ptr()
        for start, end in target_ranges:
            begin = max(start - base, 0)
            stop = min(end - base, value.numel())
            if begin < stop:
                value.reshape(-1)[begin:stop].zero_()
        expected.append(value)
    manager._fill_fresh_kv_pages(1)
    for (buffer, _), value in zip(views, expected, strict=True):
        assert torch.equal(buffer.view(torch.uint8), value)


@skip_pre_blackwell
@pytest.mark.parametrize("committed_tokens", [0, 129])
def test_v4_fresh_zero_covers_all_roles_and_protects_partial_prefix(
    native_pools: tuple[DeepseekV4CacheManager, _CacheView], committed_tokens: int
) -> None:
    manager, cache = native_pools
    for buffer, _ in _views(manager):
        buffer.view(torch.uint8).fill_(17)
    cache.num_committed_tokens = committed_tokens
    protected = (committed_tokens + 127) // 128
    _assert_fill(manager, cache, list(range(protected, cache.num_blocks)))


@skip_pre_blackwell
def test_v4_fresh_zero_preserves_relocated_ordinals_and_fills_growth(
    native_pools: tuple[DeepseekV4CacheManager, _CacheView],
) -> None:
    manager, cache = native_pools
    original_pages = cache.pages
    cache.pages = {pool: pages[:3] for pool, pages in original_pages.items()}
    manager._fill_fresh_kv_pages(1)
    for buffer, _ in _views(manager):
        buffer.view(torch.uint8).fill_(17)
    cache.pages = {pool: list(reversed(pages)) for pool, pages in cache.pages.items()}
    _assert_fill(manager, cache, [])
    cache.pages = {pool: pages + [original_pages[pool][3]] for pool, pages in cache.pages.items()}
    _assert_fill(manager, cache, [3])
