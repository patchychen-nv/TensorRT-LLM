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
    STAGING_COMPONENTS,
    DkvPageCopier,
    DkvStagedKvView,
    StagingComponent,
    StagingGeometry,
    StagingKind,
    StagingLayout,
    StagingOverflow,
    StagingPool,
    pool_page_addresses,
)
from tensorrt_llm._torch.pyexecutor.dkv_streamer import LoopbackStreamer
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


@pytest.fixture(**_VARIANTS)
def ringed(request) -> Iterator[_Case]:
    """The default ring of two slots, which the layers of a kind take turns in."""
    with _staged_case(*request.param, ring_depth=2) as value:
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


# ---- the view the attention backend sees ---------------------------------------------------------


@pytest.fixture
def view(case: _Case) -> DkvStagedKvView:
    return DkvStagedKvView(case.manager, case.pool)


def test_the_view_covers_every_layer_with_the_staging_pointers(case: _Case, view) -> None:
    manager, layout, pool = case.manager, case.layout, case.pool
    assert view.pp_layers == list(range(len(_RATIOS))) == list(manager.pp_layers)
    assert view.layer_offsets == dict(manager.layer_offsets)
    assert view.num_local_layers == manager.num_local_layers
    swa = StagingComponent(StagingKind.SWA, DeepseekV4AttentionType.SWA)
    assert view.swa_pool_ptr == pool.buffer.data_ptr() + layout.region_offset(swa)
    assert set(view.compress_pool_ptrs) == {4, 128}
    assert view.kv_cache_pool_pointers.shape == manager.kv_cache_pool_pointers.shape
    assert (view.kv_cache_pool_pointers[:, 0] == view.swa_pool_ptr).all()
    assert torch.equal(view.kv_cache_pool_mapping, manager.kv_cache_pool_mapping)
    # What the view does not define is the manager's.
    assert view.tokens_per_block == manager.tokens_per_block
    assert view.max_blocks_per_seq == manager.max_blocks_per_seq
    assert view.dtype == manager.dtype


def test_view_buffers_have_the_rows_of_the_managers_and_start_at_the_staging_pages(
    case: _Case, view
) -> None:
    manager, layout, pool = case.manager, case.layout, case.pool
    for component in layout.components:
        for layer in layout.layers_of(component.kind):
            real = manager.get_buffers(layer, component.attention_type)
            staged = view.get_buffers(layer, component.attention_type)
            assert staged.dtype == real.dtype
            assert staged.shape[1:] == real.shape[1:]
            assert staged.stride(0) * staged.element_size() == layout.page_bytes(component)
            assert staged.data_ptr() == pool.layer_pointer(layer, component)
            assert staged.shape[0] * layout.page_bytes(component) <= layout.region_bytes(component)
    indexer = view.get_indexer_k_cache_buffers(1)
    assert indexer.dtype == torch.uint8
    assert indexer.shape[1:] == manager.get_indexer_k_cache_buffers(1).shape[1:]


def test_view_tables_are_the_identity_maps_onto_the_slots_of_the_staged_requests(
    case: _Case, view
) -> None:
    layout = case.layout
    requests = [(300, 400), (0, 50), (1000, 1)]
    view.begin_staged_batch([11, 12, 13], [h for h, _ in requests], [c for _, c in requests], 3)
    view.compute_sliding_block_tables([11, 12, 13], 3)
    max_blocks = case.manager.max_blocks_per_seq
    sliding = torch.empty(len(_RATIOS), 5, 4, max_blocks, dtype=torch.int32, device="cuda")
    view.copy_batch_sliding_block_tables(sliding, [11, 12, 13], 3, 3)
    for layer in range(len(_RATIOS)):
        for index, attention_type in enumerate(DkvStagedKvView._SLIDING):
            component = [
                c
                for c in layout.components
                if c.attention_type is attention_type and layer in layout.layers_of(c.kind)
            ]
            table = sliding[layer, index].cpu()
            if not component:
                assert (table == BAD_PAGE_INDEX).all()
                continue
            kind = component[0].kind
            spans = layout.request_spans(kind, requests)
            for row, span in enumerate(spans):
                expected = layout.block_table(kind, layer, span, max_blocks)
                assert table[row].tolist() == expected, (layer, attention_type, row)
            assert (table[3] == BAD_PAGE_INDEX).all()  # the unused row of the batch
    # The generic block offsets are the SWA tables, for every layer.
    offsets = torch.empty(len(_RATIOS), 4, 2, max_blocks, dtype=torch.int32, device="cuda")
    view.copy_batch_block_offsets(offsets, [11, 12, 13], 1, 3, 3)
    assert torch.equal(offsets[:, :, 0], sliding[:, DeepseekV4AttentionType.SWA.value])
    assert (offsets[:, :, 1] == BAD_PAGE_INDEX).all()
    # The shared tables of the compressed kinds do not depend on the layer.
    compress = torch.empty(4, max_blocks, dtype=torch.int32, device="cuda")
    for ratio, kind in ((4, StagingKind.COMPRESS_R4), (128, StagingKind.COMPRESS_R128)):
        view.copy_batch_compress_block_tables(compress, [11, 12, 13], ratio, 1, 3, 3)
        for row, span in enumerate(layout.request_spans(kind, requests)):
            layer = layout.layers_of(kind)[0]
            assert compress[row].cpu().tolist() == layout.block_table(kind, layer, span, max_blocks)
    host = torch.full((4, max_blocks), 7, dtype=torch.int32)
    view.copy_batch_indexer_compress_block_tables(host, [11, 12, 13], 1, 3, 3)
    for row, span in enumerate(layout.request_spans(StagingKind.INDEXER_COMPRESS, requests)):
        layer = layout.layers_of(StagingKind.INDEXER_COMPRESS)[0]
        assert host[row].tolist() == layout.block_table(
            StagingKind.INDEXER_COMPRESS, layer, span, max_blocks
        )
    assert (host[3] == 7).all()  # rows beyond the batch are not written


def test_the_view_needs_a_batch_before_it_builds_tables(case: _Case, view) -> None:
    with pytest.raises(RuntimeError, match="begin_staged_batch has not been called"):
        view.compute_sliding_block_tables([1], 1)
    with pytest.raises(ValueError, match="differ in length"):
        view.begin_staged_batch([1, 2], [0], [10], 1)
    with pytest.raises(StagingOverflow):
        view.begin_staged_batch(list(range(100)), [0] * 100, [4096] * 100, 100)


# ---- the streamer that moves the pages between the cache manager and the staging area ------------


def _components_of(layout: StagingLayout, layer: int) -> list[StagingComponent]:
    return [c for c in layout.components if layer in layout.layers_of(c.kind)]


def _salt(layer: int, component: StagingComponent, block: int, generation: int) -> int:
    role = STAGING_COMPONENTS.index(component)
    return (((generation * len(_RATIOS) + layer) * len(STAGING_COMPONENTS)) + role) * 16 + block


def _cache_pages(manager, layer: int, component: StagingComponent) -> dict[int, torch.Tensor]:
    """The pages of the request in the cache manager by block."""
    indices = manager.get_cache_indices(_REQUEST, layer, component.attention_type)
    return {
        block: _pool_page(manager, layer, component.attention_type, index)
        for block, index in enumerate(indices[: _blocks()])
        if index != BAD_PAGE_INDEX
    }


def _staged_page(view: DkvStagedKvView, layer: int, component: StagingComponent, index: int):
    return view.get_buffers(layer, component.attention_type)[index].view(torch.uint8).reshape(-1)


@pytest.mark.parametrize("fill", ["", "nan"])
def test_the_streamer_fetches_the_cached_pages_and_writes_back_the_new_ones(
    ringed: _Case, fill: str
) -> None:
    manager, layout, pool = ringed.manager, ringed.layout, ringed.pool
    view = DkvStagedKvView(manager, pool)
    streamer = LoopbackStreamer(manager, view, fill=fill)
    view.dkv_streamer = streamer
    layers = range(len(_RATIOS))
    # Every page of the request in the cache manager gets a content of its own.
    for layer in layers:
        for component in _components_of(layout, layer):
            for block, page in _cache_pages(manager, layer, component).items():
                page.copy_(_pattern(_salt(layer, component, block, 0), page.numel()))
    torch.cuda.synchronize()

    view.begin_staged_batch([_REQUEST], [_HISTORY], [_CHUNK], 1)
    fetched_bytes = written_bytes = 0
    for layer in layers:
        streamer.on_layer(layer)
        for component in _components_of(layout, layer):
            kind = component.kind
            span = layout.request_spans(kind, [(_HISTORY, _CHUNK)])[0]
            table = layout.block_table(kind, layer, span, _blocks())
            first, count = layout.fetch_range(kind, _HISTORY, _CHUNK)
            for block in range(span.first_block, span.first_block + span.num_pages):
                page = _staged_page(view, layer, component, table[block])
                if first <= block < first + count:
                    expected = _pattern(_salt(layer, component, block, 0), page.numel())
                    assert torch.equal(page, expected), ("fetch", component, layer, block)
                elif fill:
                    # What was not fetched is what the fill left, not what an earlier layer of the
                    # ring held.
                    assert (page == 0xFF).all(), ("fill", component, layer, block)
            fetched_bytes += count * layout.page_bytes(component)
            # The attention writes the pages of the new tokens.
            first, count = layout.writeback_range(kind, _HISTORY, _CHUNK)
            for block in range(first, first + count):
                page = _staged_page(view, layer, component, table[block])
                page.copy_(_pattern(_salt(layer, component, block, 1), page.numel()))
            written_bytes += count * layout.page_bytes(component)
    streamer.end_forward()
    torch.cuda.synchronize()

    assert streamer.bytes_fetched == fetched_bytes
    assert streamer.bytes_written_back == written_bytes
    for layer in layers:
        for component in _components_of(layout, layer):
            first, count = layout.writeback_range(component.kind, _HISTORY, _CHUNK)
            for block, page in _cache_pages(manager, layer, component).items():
                generation = int(first <= block < first + count)
                expected = _pattern(_salt(layer, component, block, generation), page.numel())
                assert torch.equal(page, expected), ("write back", component, layer, block)


def test_the_streamer_skips_a_request_the_cache_manager_does_not_hold(ringed: _Case) -> None:
    # A warm-up or dummy request has no pages to move.
    view = DkvStagedKvView(ringed.manager, ringed.pool)
    streamer = LoopbackStreamer(ringed.manager, view)
    view.dkv_streamer = streamer
    view.begin_staged_batch([_REQUEST + 100], [0], [64], 1)
    for layer in range(len(_RATIOS)):
        streamer.on_layer(layer)
    streamer.end_forward()
    assert streamer.bytes_fetched == streamer.bytes_written_back == 0


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
