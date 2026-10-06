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
"""The layer-split staging pool and page copier against a real DeepSeek-V4 cache manager."""

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

import pytest
import torch
from utils.util import skip_pre_blackwell

from tensorrt_llm._torch.attention.backends.sparse.deepseek_v4.cache_manager import (
    DeepseekV4CacheManager,
)
from tensorrt_llm._torch.attention.backends.sparse.deepseek_v4.params import DeepseekV4AttentionType
from tensorrt_llm._torch.pyexecutor.dkv_staging import (
    BAD_PAGE_INDEX,
    DkvPageCopier,
    StagingComponent,
    StagingGeometry,
    StagingKind,
    StagingLayout,
    StagingPool,
    pool_page_addresses,
)
from tensorrt_llm.bindings import DataType

from .test_deepseek_v4_cache_manager import TestDeepseekV4CacheManager as _ManagerFactory

pytestmark = skip_pre_blackwell

# A layer of each type, and two layers of each compressed type so that the rings hold more than
# one layer of a kind.
_RATIOS = [1, 4, 128, 4, 128]
_REQUEST = 1
_HISTORY, _CHUNK = 300, 400  # the cache holds 700 tokens, the first 300 of them already computed


@dataclass
class _Case:
    manager: DeepseekV4CacheManager
    layout: StagingLayout
    pool: StagingPool


_VARIANTS = {
    "params": [(DataType.FP8, "fp8"), (DataType.FP8, "fp4"), (DataType.BF16, "fp8")],
    "ids": ["fp8 kv, fp8 indexer", "fp8 kv, fp4 indexer", "bf16 kv, fp8 indexer"],
}


@contextmanager
def _staged_case(dtype, indexer: str, ring_depth: int) -> Iterator[_Case]:
    factory = _ManagerFactory()
    manager, _ = factory._create_deepseek_v4_cache_manager(
        tokens_per_block=128,
        max_batch_size=2,
        max_seq_len=2048,
        compress_ratios=_RATIOS,
        dtype=dtype,
        compressor_dtype=DataType.FLOAT,
        indexer_k_dtype=indexer,
        enable_swa_scratch_reuse=False,
    )
    req = factory._create_request(_REQUEST, _HISTORY + _CHUNK)
    try:
        assert manager.prepare_context(req)
        assert manager.resize_context(req, _HISTORY + _CHUNK)
        geometry = StagingGeometry.from_cache_manager(
            manager, max_staging_tokens=8192, ring_depth=ring_depth
        )
        layout = StagingLayout(geometry)
        yield _Case(manager, layout, StagingPool(layout))
    finally:
        manager.free_resources(req)
        manager.shutdown()


@pytest.fixture(**_VARIANTS)
def case(request) -> Iterator[_Case]:
    """A slot for every layer: the copies of most tests are not ordered by layer, so two layers
    must not share the rows a test compares."""
    with _staged_case(*request.param, ring_depth=len(_RATIOS)) as value:
        yield value


def _blocks() -> int:
    return -(-(_HISTORY + _CHUNK) // 128)


def _pattern(salt: int, size: int) -> torch.Tensor:
    values = (torch.arange(size, dtype=torch.int64, device="cuda") * 7 + salt * 31) % 253
    return values.to(torch.uint8)


def _pool_page(manager, layer, attention_type, index) -> torch.Tensor:
    return manager.get_buffers(layer, attention_type)[index].view(torch.uint8).reshape(-1)


def _staged(case: _Case):
    """Every page of the request that is staged: (component, layer, block, pool page index,
    staging row of the component's ring, pool address, staging address)."""
    layout, pool, manager = case.layout, case.pool, case.manager
    num_blocks = _blocks()
    for kind in layout.kinds:
        span = layout.request_spans(kind, [(_HISTORY, _CHUNK)])[0]
        for layer in layout.layers_of(kind):
            table = layout.block_table(kind, layer, span, num_blocks)
            for component in (c for c in layout.components if c.kind is kind):
                indices = manager.get_cache_indices(_REQUEST, layer, component.attention_type)
                # The manager's list covers every block a request may have; the first ones are its.
                assert len(indices) >= num_blocks
                addresses = pool_page_addresses(manager, layer, component.attention_type, indices)
                page_bytes = layout.page_bytes(component)
                base = pool.layer_pointer(layer, component)
                region = pool.buffer.data_ptr() + layout.region_offset(component)
                for block in range(num_blocks):
                    if table[block] == BAD_PAGE_INDEX:
                        continue
                    assert addresses[block] is not None
                    staging = base + table[block] * page_bytes
                    row = (staging - region) // page_bytes
                    yield component, layer, block, indices[block], row, addresses[block], staging


def test_page_sizes_and_slots_follow_the_cache_manager(case: _Case) -> None:
    layout, manager = case.layout, case.manager
    assert layout.kinds == tuple(StagingKind)
    for component in layout.components:
        for layer in layout.layers_of(component.kind):
            buffer = manager.get_buffers(layer, component.attention_type)
            assert (
                layout.page_bytes(component)
                == manager._get_attn_bytes_per_block(component.attention_type, layer)
                == buffer.stride(0) * buffer.element_size()
            )


def test_the_pool_is_aligned_and_holds_every_ring(case: _Case) -> None:
    pool, layout = case.pool, case.layout
    assert pool.buffer.numel() == layout.total_bytes == pool.bytes_reserved
    for component in layout.components:
        region = pool.region(component)
        assert region.data_ptr() % 256 == 0
        assert region.numel() == layout.region_bytes(component)
        pages = pool.pages(component)
        assert pages.shape == (
            layout.geometry.ring_depth * layout.slot_pages(component.kind),
            layout.page_bytes(component),
        )
    # A layer's pointer is its ring (the sliding-window kinds), or its own slot of it.
    compress = StagingComponent(StagingKind.COMPRESS_R4, DeepseekV4AttentionType.COMPRESS)
    first, second = layout.layers_of(StagingKind.COMPRESS_R4)
    assert pool.layer_pointer(second, compress) - pool.layer_pointer(
        first, compress
    ) == layout.slot_bytes(compress)
    swa = StagingComponent(StagingKind.SWA, DeepseekV4AttentionType.SWA)
    assert pool.layer_pointer(0, swa) == pool.layer_pointer(4, swa)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float8_e4m3fn])
def test_the_nan_fill_reads_as_nan_in_every_staged_float_format(case: _Case, dtype) -> None:
    case.pool.fill("zero")
    assert not case.pool.buffer.any()
    case.pool.fill("nan")
    for component in case.layout.components:
        region = case.pool.region(component)
        usable = region.numel() // dtype.itemsize * dtype.itemsize
        assert torch.isnan(region[:usable].view(dtype)).all()
    with pytest.raises(ValueError, match="unknown staging fill"):
        case.pool.fill("one")


def test_gathering_copies_exactly_the_staged_pages_and_scattering_restores_them(
    case: _Case,
) -> None:
    layout, pool, manager = case.layout, case.pool, case.manager
    copier = DkvPageCopier()
    staged = list(_staged(case))
    assert staged
    # Give every page of the request in the cache manager a content of its own.
    for salt, (component, layer, block, index, *_rest) in enumerate(staged):
        page = _pool_page(manager, layer, component.attention_type, index)
        page.copy_(_pattern(salt, page.numel()))
    torch.cuda.synchronize()

    pool.fill("nan")
    stream = torch.cuda.Stream()
    for component in layout.components:
        rows = [item for item in staged if item[0] == component]
        copier.gather(
            [item[5] for item in rows],
            [item[6] for item in rows],
            layout.page_bytes(component),
            stream.cuda_stream,
        )
    stream.synchronize()

    expected_rows = {component: set() for component in layout.components}
    for salt, (component, layer, block, index, row, _pool_address, _staging) in enumerate(staged):
        expected = _pattern(salt, layout.page_bytes(component))
        assert torch.equal(pool.pages(component)[row], expected), (component, layer, block)
        expected_rows[component].add(row)
    for component in layout.components:
        untouched = [
            row
            for row in range(pool.pages(component).shape[0])
            if row not in expected_rows[component]
        ]
        assert (pool.pages(component)[untouched] == 0xFF).all(), component

    # Scatter other content from staging and see that only the staged pages of the cache change.
    snapshot: dict[int, torch.Tensor] = {}
    for component in layout.components:
        for layer in layout.layers_of(component.kind):
            buffer = manager.get_buffers(layer, component.attention_type)
            snapshot.setdefault(buffer.data_ptr(), buffer.view(torch.uint8).clone())
    for salt, (component, _layer, _block, _index, row, _pool, _staging) in enumerate(staged):
        pool.pages(component)[row].copy_(_pattern(salt + 1000, layout.page_bytes(component)))
    torch.cuda.synchronize()
    for component in layout.components:
        rows = [item for item in staged if item[0] == component]
        copier.scatter(
            [item[6] for item in rows],
            [item[5] for item in rows],
            layout.page_bytes(component),
            stream.cuda_stream,
        )
    stream.synchronize()
    for salt, (component, layer, block, index, *_rest) in enumerate(staged):
        page = _pool_page(manager, layer, component.attention_type, index)
        assert torch.equal(page, _pattern(salt + 1000, page.numel())), (component, layer, block)
    scattered = {item[5] for item in staged}
    for component in layout.components:
        for layer in layout.layers_of(component.kind):
            buffer = manager.get_buffers(layer, component.attention_type)
            page_bytes = layout.page_bytes(component)
            before = snapshot[buffer.data_ptr()]
            untouched = [
                row
                for row in range(buffer.shape[0])
                if buffer.data_ptr() + row * page_bytes not in scattered
            ]
            assert torch.equal(buffer.view(torch.uint8)[untouched], before[untouched]), (
                component,
                layer,
            )


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


def test_a_block_without_a_page_has_no_address(case: _Case) -> None:
    manager = case.manager
    indices = manager.get_cache_indices(_REQUEST, 0, case.layout.components[0].attention_type)
    addresses = pool_page_addresses(manager, 0, case.layout.components[0].attention_type, indices)
    assert all(address is not None for address in addresses[: _blocks()])
    assert pool_page_addresses(
        manager, 0, case.layout.components[0].attention_type, [BAD_PAGE_INDEX, indices[0]]
    ) == [None, addresses[0]]


def test_a_fill_covers_the_slots_of_one_layer(case: _Case) -> None:
    layout, pool = case.layout, case.pool
    pool.fill("zero")
    pool.fill_layer(2, "nan")
    for component in layout.components:
        pages = pool.pages(component)
        slot_pages = layout.slot_pages(component.kind)
        for layer in layout.layers_of(component.kind):
            slot = layout.slot_of(layer, component.kind)
            rows = pages[slot * slot_pages : (slot + 1) * slot_pages]
            assert (rows == (0xFF if layer == 2 else 0)).all(), (component, layer)
    with pytest.raises(ValueError, match="unknown staging fill"):
        pool.fill_layer(2, "one")
