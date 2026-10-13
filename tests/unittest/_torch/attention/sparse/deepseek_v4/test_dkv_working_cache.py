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
"""The working cache of the DKV layer split on DeepSeek-V4 cache managers.

The working cache is a second DeepSeek-V4 cache manager that covers every layer. The data plane
describes its buffers, the blocks it moves and where their pages are from what any V2 manager
publishes; these tests hold that description against what the DeepSeek-V4 manager itself says.
"""

import functools
from collections.abc import Iterator

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
from tensorrt_llm._torch.pyexecutor.dkv_working import (
    DkvWorkingSession,
    WorkingCacheOverflow,
    base_page_indices,
    buffers_by_layer,
    manager_buffers,
    page_affine,
    staged_blocks,
    working_cache_bytes,
    working_slot_counts,
)
from tensorrt_llm._torch.pyexecutor.kv_cache.lifecycle_slot_counts import (
    lifecycle_key,
    lifecycle_layouts,
)
from tensorrt_llm.bindings import DataType
from tensorrt_llm.bindings.internal.batch_manager import CacheType
from tensorrt_llm.llmapi.llm_args import DeepSeekV4SparseAttentionConfig, KvCacheConfig
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.runtime.kv_cache_manager_v2 import BAD_PAGE_INDEX

pytestmark = skip_pre_blackwell

_RATIOS = [1, 4, 128, 4, 128]
_BLOCK = 128
_MAX_BATCH = 2
_MAX_NUM_TOKENS = 2048
_STAGING_TOKENS = 4096
# (cached tokens, new tokens) of a request; covers empty, partial and full blocks and histories
# below and past every window.
_SHAPES = [(0, 1), (0, 300), (100, 28), (127, 1), (700, 300), (1024, 128), (1500, 2048)]


def _manager(**kwargs) -> DeepseekV4CacheManager:
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
        max_batch_size=_MAX_BATCH,
        max_input_len=4096,
        mapping=Mapping(world_size=1, rank=0, tp_size=1, pp_size=1),
        dtype=DataType.FP8,
        compressor_dtype=DataType.FLOAT,
        vocab_size=129280,
        max_num_tokens=_MAX_NUM_TOKENS,
        sparse_attn_config=DeepSeekV4SparseAttentionConfig(
            index_head_dim=128, window_size=128, compress_ratios=_RATIOS, indexer_k_dtype="fp8"
        ),
        **kwargs,
    )


def _working_counts():
    return functools.partial(
        working_slot_counts,
        max_batch_size=_MAX_BATCH,
        max_num_tokens=_MAX_NUM_TOKENS,
        staging_tokens=_STAGING_TOKENS,
    )


@pytest.fixture(scope="module")
def working() -> Iterator[DeepseekV4CacheManager]:
    manager = _manager(lifecycle_slot_counts=_working_counts())
    try:
        yield manager
    finally:
        manager.shutdown()


@pytest.fixture(scope="module")
def owned() -> Iterator[DeepseekV4CacheManager]:
    manager = _manager(owned_layers=[2, 3, 4])
    try:
        yield manager
    finally:
        manager.shutdown()


def _attention_types(layer: int):
    return [
        attn_type
        for attn_type in DeepseekV4AttentionType
        if compress_ratio_has_attention(_RATIOS[layer], attn_type)
    ]


def test_the_buffers_of_a_layer_are_the_attention_types_the_manager_gives_it(working) -> None:
    """The data plane moves every buffer the attention of a layer reads and nothing else."""
    buffers = buffers_by_layer(entry.buffer for entry in manager_buffers(working).values())
    assert sorted(buffers) == list(range(len(_RATIOS)))
    for layer, layer_buffers in buffers.items():
        roles = sorted(buffer.role for buffer in layer_buffers)
        assert roles == sorted(str(attn_type.role) for attn_type in _attention_types(layer)), layer


def test_the_layer_group_of_a_buffer_is_its_life_cycle(working) -> None:
    """The page counts of a manager are given per life cycle, and the data plane finds the count
    of a buffer by its layer group."""
    config = working.kv_cache_manager_py_config
    keys = [layout.key for layout in lifecycle_layouts(config)]
    for layer in config.layers:
        group = working.layer_to_pool_mapping_dict[layer.layer_id]
        assert group == keys.index(lifecycle_key(layer, _BLOCK)), layer.layer_id


@pytest.mark.parametrize(("history", "chunk"), _SHAPES)
def test_a_working_cache_has_a_page_for_every_staged_block_and_no_other(
    working, history, chunk
) -> None:
    session = DkvWorkingSession(working)
    session.begin([(7, history, chunk)])
    try:
        for entry in manager_buffers(working).values():
            affine = page_affine(working, entry.layer_id, entry.role)
            indices = base_page_indices(working, 7, affine.layer_group)
            with_page = [block for block, index in enumerate(indices) if index != BAD_PAGE_INDEX]
            first, count = staged_blocks(entry.buffer.key, history, chunk, _BLOCK)
            assert with_page == list(range(first, first + count)), entry.buffer
    finally:
        session.end()


def _check_affine_matches_the_attention_view(manager, request_id: int) -> int:
    """Compare every page of every buffer of the request; return how many were compared."""
    compared = 0
    entries = manager_buffers(manager)
    for layer in manager.pp_layers:
        for attn_type in _attention_types(layer):
            entry = entries[layer, str(attn_type.role)]
            affine = page_affine(manager, entry.layer_id, entry.role)
            base = base_page_indices(manager, request_id, affine.layer_group)
            view = manager.get_buffers(layer, attn_type)
            view_page_bytes = view.stride(0) * view.element_size()
            assert view_page_bytes == affine.page_bytes, (layer, attn_type)
            kernel_indices = manager.get_cache_indices(request_id, layer, attn_type)
            for block, index in enumerate(kernel_indices):
                if index == BAD_PAGE_INDEX:
                    assert base[block] == BAD_PAGE_INDEX, (layer, attn_type, block)
                    continue
                expected = view.data_ptr() + index * view_page_bytes
                assert affine.base + int(base[block]) * affine.step == expected, (
                    layer,
                    attn_type,
                    block,
                )
                compared += 1
    return compared


def test_the_page_affine_finds_the_pages_the_attention_reads_in_the_working_cache(working) -> None:
    session = DkvWorkingSession(working)
    session.begin([(7, 700, 300), (8, 1024, 128)])
    try:
        assert _check_affine_matches_the_attention_view(working, 7) > 0
        assert _check_affine_matches_the_attention_view(working, 8) > 0
    finally:
        session.end()


def test_the_page_affine_finds_the_pages_of_a_manager_that_holds_some_layers(owned) -> None:
    # The owner's cache: the request is placed the way the executor's life cycle places it.
    assert owned.dkv_open_cache(5, 700, 1000)
    try:
        assert _check_affine_matches_the_attention_view(owned, 5) > 0
    finally:
        owned.dkv_close_cache(5)


def test_the_attention_tables_of_the_working_cache_are_built_by_the_manager_itself(working) -> None:
    session = DkvWorkingSession(working)
    requests = [(7, 700, 300), (8, 0, 129)]
    session.begin(requests)
    try:
        request_ids = [request_id for request_id, _, _ in requests]
        working.compute_sliding_block_tables(request_ids, len(request_ids))
        offsets = torch.empty(
            (working.num_local_layers, len(request_ids), 2, working.max_blocks_per_seq),
            dtype=torch.int32,
            device="cuda",
        )
        working.copy_batch_block_offsets(
            offsets, request_ids, 1, len(request_ids), len(request_ids)
        )
        offsets = offsets.cpu()
        for row, request_id in enumerate(request_ids):
            for layer in range(len(_RATIOS)):
                expected = working.get_cache_indices(request_id, layer, DeepseekV4AttentionType.SWA)
                local = working.layer_offsets[layer]
                actual = offsets[local, row, 0, : len(expected)].tolist()
                assert actual == list(expected), (request_id, layer)
        for ratio in (4, 128):
            table = torch.full(
                (len(request_ids), working.max_blocks_per_seq), -7, dtype=torch.int32, device="cuda"
            )
            working.copy_batch_compress_block_tables(
                table, request_ids, ratio, 1, len(request_ids), len(request_ids)
            )
            layer = _RATIOS.index(ratio)
            for row, request_id in enumerate(request_ids):
                expected = working.get_cache_indices(
                    request_id, layer, DeepseekV4AttentionType.COMPRESS
                )
                assert table[row, : len(expected)].tolist() == list(expected), (request_id, ratio)
    finally:
        session.end()


def test_the_working_cache_takes_any_admitted_batch_and_gives_it_back(working) -> None:
    free_slots = working.index_mapper.num_free_slots()
    # The largest batch the scheduler admits: every new token, every staged token.
    chunk = _MAX_NUM_TOKENS // _MAX_BATCH
    history = _STAGING_TOKENS // _MAX_BATCH - chunk
    session = DkvWorkingSession(working)
    session.begin([(10 + index, history, chunk) for index in range(_MAX_BATCH)])
    assert session.request_ids == (10, 11)
    session.end()
    assert not working.kv_cache_map
    assert working.index_mapper.num_free_slots() == free_slots
    # A batch past the budget does not fit and leaves nothing open: every request has nearly
    # max_seq_len tokens, together more than the staged tokens.
    with pytest.raises(WorkingCacheOverflow):
        session.begin([(20 + index, 4000, 1) for index in range(_MAX_BATCH)])
    assert session.request_ids == ()
    assert not working.kv_cache_map
    assert working.index_mapper.num_free_slots() == free_slots


def test_the_working_cache_reserves_what_its_page_counts_name() -> None:
    torch.cuda.synchronize()
    free_before, _ = torch.cuda.mem_get_info()
    manager = _manager(lifecycle_slot_counts=_working_counts())
    try:
        torch.cuda.synchronize()
        free_after, _ = torch.cuda.mem_get_info()
        config = manager.kv_cache_manager_py_config
        expected = working_cache_bytes(config)
        reserved = free_before - free_after
        print(
            f"working cache: {reserved / (1 << 20):.1f} MiB reserved, {expected / (1 << 20):.1f} "
            f"MiB in pages, counts {list(config.lifecycle_slot_counts[0])}"
        )
        # The pools round up to the granularity of their allocations, by far less than a page each.
        assert expected <= reserved <= expected + len(lifecycle_layouts(config)) * (8 << 20)
    finally:
        manager.shutdown()
