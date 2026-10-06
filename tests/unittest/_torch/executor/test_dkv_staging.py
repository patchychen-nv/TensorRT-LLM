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
"""The layout arithmetic of the layer-split staging area: sizes, slots, spans and block tables."""

import types
from dataclasses import replace

import pytest

from tensorrt_llm._torch.attention.backends.sparse.deepseek_v4.cache_manager import (
    _get_attn_bytes_per_token,
)
from tensorrt_llm._torch.attention.backends.sparse.deepseek_v4.params import DeepseekV4AttentionType
from tensorrt_llm._torch.pyexecutor.dkv_staging import (
    BAD_PAGE_INDEX,
    REGION_ALIGNMENT,
    STAGING_COMPONENTS,
    RequestSpan,
    StagingComponent,
    StagingGeometry,
    StagingKind,
    StagingLayout,
    StagingOverflow,
)
from tensorrt_llm.bindings import DataType

pytestmark = pytest.mark.cpu_only

_TYPES = DeepseekV4AttentionType


def _pro_ratios() -> tuple[int, ...]:
    """DeepSeek-V4-Pro: two SWA-only layers, then CSA and HCA layers alternating (43 layers)."""
    return (1, 1) + tuple(4 if layer % 2 == 0 else 128 for layer in range(41))


def _geometry(**changes) -> StagingGeometry:
    values = dict(
        compress_ratios=_pro_ratios(),
        tokens_per_block=128,
        head_dim=512,
        index_head_dim=128,
        has_fp8_kv_cache=True,
        indexer_k_dtype="fp8",
        window_size=128,
        max_num_tokens=4096,
        max_staging_tokens=131072 + 4096,
        max_batch_size=8,
        ring_depth=2,
    )
    values.update(changes)
    return StagingGeometry(**values)


def _component(kind: StagingKind, attention_type) -> StagingComponent:
    return StagingComponent(kind, attention_type)


def test_the_layout_has_a_region_for_every_cache_role_of_the_model() -> None:
    layout = StagingLayout(_geometry())
    assert layout.kinds == tuple(StagingKind)
    assert len(layout.components) == len(STAGING_COMPONENTS) == 10
    assert [c.attention_type for c in layout.components if c.kind is StagingKind.STATE_CSA] == [
        _TYPES.COMPRESSOR_KV,
        _TYPES.COMPRESSOR_SCORE,
    ]


def test_a_model_without_hca_layers_has_no_hca_storage() -> None:
    layout = StagingLayout(_geometry(compress_ratios=(1, 4, 4, 1, 4)))
    assert StagingKind.STATE_HCA not in layout.kinds
    assert StagingKind.COMPRESS_R128 not in layout.kinds
    assert not [c for c in layout.components if c.kind is StagingKind.STATE_HCA]


@pytest.mark.parametrize(
    ("fp8", "indexer", "expected"),
    [
        (True, "fp8", (65536, 16384, 512, 4224, 524288, 262144, 131072)),
        (True, "fp4", (65536, 16384, 512, 2176, 524288, 262144, 131072)),
        (False, "fp8", (131072, 32768, 1024, 4224, 524288, 262144, 131072)),
        (False, "fp4", (131072, 32768, 1024, 2176, 524288, 262144, 131072)),
    ],
    ids=[
        "fp8 kv, fp8 indexer",
        "fp8 kv, fp4 indexer",
        "bf16 kv, fp8 indexer",
        "bf16 kv, fp4 indexer",
    ],
)
def test_page_sizes_are_those_of_the_cache_manager(fp8, indexer, expected) -> None:
    layout = StagingLayout(_geometry(has_fp8_kv_cache=fp8, indexer_k_dtype=indexer))
    swa, c4, c128, indexer_compress, csa, hca, indexer_state = expected
    assert layout.page_bytes(_component(StagingKind.SWA, _TYPES.SWA)) == swa
    assert layout.page_bytes(_component(StagingKind.COMPRESS_R4, _TYPES.COMPRESS)) == c4
    assert layout.page_bytes(_component(StagingKind.COMPRESS_R128, _TYPES.COMPRESS)) == c128
    assert (
        layout.page_bytes(_component(StagingKind.INDEXER_COMPRESS, _TYPES.INDEXER_COMPRESS))
        == indexer_compress
    )
    for attention_type in (_TYPES.COMPRESSOR_KV, _TYPES.COMPRESSOR_SCORE):
        assert layout.page_bytes(_component(StagingKind.STATE_CSA, attention_type)) == csa
        assert layout.page_bytes(_component(StagingKind.STATE_HCA, attention_type)) == hca
    for attention_type in (_TYPES.INDEXER_COMPRESSOR_KV, _TYPES.INDEXER_COMPRESSOR_SCORE):
        assert layout.page_bytes(_component(StagingKind.STATE_INDEXER, attention_type)) == (
            indexer_state
        )


@pytest.mark.parametrize("tokens_per_block", [128, 256])
@pytest.mark.parametrize("fp8", [True, False])
@pytest.mark.parametrize("indexer", ["fp8", "fp4"])
def test_a_page_holds_the_per_token_bytes_of_the_cache_manager_for_every_block_token(
    tokens_per_block, fp8, indexer
) -> None:
    """The per-token estimate the pool sizes are derived from, restated per block of tokens."""
    layout = StagingLayout(
        _geometry(tokens_per_block=tokens_per_block, has_fp8_kv_cache=fp8, indexer_k_dtype=indexer)
    )
    for component in layout.components:
        ratio = component.kind.compress_ratio or 1
        per_token = _get_attn_bytes_per_token(
            512, 128, ratio, component.attention_type, fp8, indexer_k_dtype=indexer
        )
        assert layout.page_bytes(component) == per_token * tokens_per_block


def test_layers_have_the_kinds_their_compress_ratio_gives_them() -> None:
    layout = StagingLayout(_geometry())
    assert layout.layers_of(StagingKind.SWA) == tuple(range(43))
    assert layout.layers_of(StagingKind.COMPRESS_R4) == tuple(range(2, 43, 2))
    assert layout.layers_of(StagingKind.STATE_INDEXER) == layout.layers_of(StagingKind.COMPRESS_R4)
    assert layout.layers_of(StagingKind.COMPRESS_R128) == tuple(range(3, 42, 2))
    assert len(layout.layers_of(StagingKind.STATE_CSA)) == 21
    assert len(layout.layers_of(StagingKind.STATE_HCA)) == 20


@pytest.mark.parametrize("ring_depth", [1, 2, 3])
def test_a_layer_always_takes_the_same_slot_and_neighbours_of_a_kind_take_different_ones(
    ring_depth: int,
) -> None:
    layout = StagingLayout(_geometry(ring_depth=ring_depth))
    for kind in layout.kinds:
        layers = layout.layers_of(kind)
        slots = [layout.slot_of(layer, kind) for layer in layers]
        assert slots == [index % ring_depth for index, _ in enumerate(layers)]
        # The layers that are in flight together are consecutive layers of the kind, and each has
        # a slot of its own as long as there are no more of them than slots.
        for window in zip(*[slots[offset:] for offset in range(ring_depth)]):
            assert len(set(window)) == ring_depth


def test_a_layer_without_the_kind_has_no_slot() -> None:
    layout = StagingLayout(_geometry())
    with pytest.raises(ValueError, match="layer 0 has no compress_r4 storage"):
        layout.slot_of(0, StagingKind.COMPRESS_R4)


@pytest.mark.parametrize(
    ("kind", "history", "chunk", "first", "pages"),
    [
        (StagingKind.SWA, 0, 1, 0, 1),
        (StagingKind.SWA, 0, 4096, 0, 32),
        # The window of the first new token reaches back into the block before.
        (StagingKind.SWA, 128, 1, 0, 2),
        (StagingKind.SWA, 254, 1, 0, 2),
        (StagingKind.SWA, 255, 1, 1, 1),
        (StagingKind.SWA, 1000, 500, 6, 6),
        (StagingKind.STATE_CSA, 1000, 500, 7, 5),
        (StagingKind.STATE_INDEXER, 1000, 500, 7, 5),
        (StagingKind.STATE_HCA, 1000, 500, 6, 6),
        (StagingKind.STATE_CSA, 127, 1, 0, 1),
        (StagingKind.STATE_CSA, 135, 1, 1, 1),
        (StagingKind.STATE_CSA, 134, 1, 0, 2),
        (StagingKind.COMPRESS_R4, 1000, 500, 0, 12),
        (StagingKind.COMPRESS_R128, 0, 1, 0, 1),
        (StagingKind.INDEXER_COMPRESS, 1023, 2, 0, 9),
    ],
)
def test_block_ranges(kind, history, chunk, first, pages) -> None:
    assert StagingLayout(_geometry()).block_range(kind, history, chunk) == (first, pages)


def test_block_ranges_need_a_chunk() -> None:
    layout = StagingLayout(_geometry())
    with pytest.raises(ValueError, match="chunk >= 1"):
        layout.block_range(StagingKind.SWA, 10, 0)
    with pytest.raises(ValueError, match="history >= 0"):
        layout.block_range(StagingKind.SWA, -1, 1)


def test_requests_take_their_pages_one_after_the_other() -> None:
    layout = StagingLayout(_geometry())
    spans = layout.request_spans(StagingKind.COMPRESS_R4, [(0, 300), (1000, 10), (128, 128)])
    assert spans == (RequestSpan(0, 3, 0), RequestSpan(0, 8, 3), RequestSpan(0, 2, 11))
    swa = layout.request_spans(StagingKind.SWA, [(0, 300), (1000, 10)])
    assert [span.page_offset for span in swa] == [0, 3]


def test_requests_that_do_not_fit_a_slot_overflow_it() -> None:
    layout = StagingLayout(_geometry(max_staging_tokens=1024, max_num_tokens=512, ring_depth=2))
    limit = layout.slot_pages(StagingKind.COMPRESS_R4)
    assert limit == 8 + (1024 - 8) // 128
    layout.request_spans(StagingKind.COMPRESS_R4, [(0, 128)] * limit)
    with pytest.raises(StagingOverflow, match=f"need {limit + 1} compress_r4 pages"):
        layout.request_spans(StagingKind.COMPRESS_R4, [(0, 128)] * (limit + 1))


def test_block_tables_are_the_identity_onto_a_slot_and_invalid_elsewhere() -> None:
    layout = StagingLayout(_geometry())
    span = RequestSpan(first_block=3, num_pages=4, page_offset=10)
    # The compressed kinds are addressed from the layer's slot, so the table does not move.
    assert layout.block_table(StagingKind.COMPRESS_R4, 4, span, 9) == (
        [BAD_PAGE_INDEX] * 3 + [10, 11, 12, 13] + [BAD_PAGE_INDEX] * 2
    )
    # The sliding-window kinds fold the slot into the page index: layer 2 is the first SWA layer
    # after two others, so it takes slot 0 and layer 3 slot 1.
    slot_pages = layout.slot_pages(StagingKind.SWA)
    assert layout.block_table(StagingKind.SWA, 2, span, 8)[3] == 10
    assert layout.block_table(StagingKind.SWA, 3, span, 8)[3] == slot_pages + 10
    assert layout.block_table(StagingKind.SWA, 3, span, 8)[0] == BAD_PAGE_INDEX


def test_regions_are_aligned_ordered_and_do_not_overlap() -> None:
    layout = StagingLayout(_geometry())
    end = 0
    for component in layout.components:
        offset = layout.region_offset(component)
        assert offset % REGION_ALIGNMENT == 0 and offset >= end
        assert layout.region_bytes(component) == layout.geometry.ring_depth * layout.slot_bytes(
            component
        )
        assert layout.slot_bytes(component) == layout.slot_pages(
            component.kind
        ) * layout.page_bytes(component)
        end = offset + layout.region_bytes(component)
    assert layout.total_bytes >= end
    assert layout.total_bytes - end < REGION_ALIGNMENT


def test_compressed_kinds_are_addressed_from_their_slot_and_the_others_from_their_ring() -> None:
    layout = StagingLayout(_geometry())
    compress = _component(StagingKind.COMPRESS_R4, _TYPES.COMPRESS)
    first, second = layout.layers_of(StagingKind.COMPRESS_R4)[:2]
    assert layout.layer_base_offset(first, compress) == layout.region_offset(compress)
    assert layout.layer_base_offset(second, compress) == layout.region_offset(
        compress
    ) + layout.slot_bytes(compress)
    # The difference to the ring base is a whole number of compressed tokens, which the attention
    # kernels assert.
    row_bytes = layout.page_bytes(compress) // (128 // 4)
    assert (layout.layer_base_offset(second, compress) - layout.region_offset(compress)) % (
        row_bytes
    ) == 0
    swa = _component(StagingKind.SWA, _TYPES.SWA)
    assert layout.layer_base_offset(5, swa) == layout.region_offset(swa)
    assert layout.slot_page_offset(5, StagingKind.SWA) == layout.slot_pages(StagingKind.SWA)
    assert layout.slot_page_offset(4, StagingKind.SWA) == 0
    assert layout.slot_page_offset(5, StagingKind.COMPRESS_R128) == 0


def test_the_staging_area_of_deepseek_v4_pro_is_a_few_hundred_megabytes() -> None:
    layout = StagingLayout(_geometry())
    assert 100 << 20 < layout.total_bytes < 512 << 20
    bigger_ring = StagingLayout(_geometry(ring_depth=3))
    assert bigger_ring.total_bytes > layout.total_bytes


def _best_pages_per_request(layout: StagingLayout, kind: StagingKind, chunks: range, histories):
    return {
        chunk: max(layout.num_pages(kind, history, chunk) for history in histories)
        for chunk in chunks
    }


@pytest.mark.parametrize(
    ("kind", "window"),
    [
        (StagingKind.SWA, 8),
        (StagingKind.SWA, 11),
        (StagingKind.SWA, 16),
        (StagingKind.STATE_CSA, 8),
    ],
)
@pytest.mark.parametrize("batch", [1, 2, 3])
def test_a_slot_holds_the_pages_of_any_batch_of_a_windowed_kind(kind, window, batch) -> None:
    """Exhaustively: no split of the chunk budget over the requests, and no history, takes more
    pages than the slot has, and the slot is larger than it needs to be by no more than the
    rounding of a page per request."""
    tokens = 24
    layout = StagingLayout(
        StagingGeometry(
            compress_ratios=(4, 4),
            tokens_per_block=8,
            head_dim=512,
            index_head_dim=128,
            has_fp8_kv_cache=True,
            indexer_k_dtype="fp8",
            window_size=window,
            max_num_tokens=tokens,
            max_staging_tokens=4 * tokens,
            max_batch_size=batch,
        )
    )
    histories = range(0, 6 * 8)
    best = _best_pages_per_request(layout, kind, range(1, tokens + 1), histories)
    # Knapsack: the most pages ``batch`` requests with at most ``tokens`` new tokens can take.
    most = {0: 0}
    for _ in range(batch):
        step = dict(most)
        for used, pages in most.items():
            for chunk, extra in best.items():
                if used + chunk <= tokens:
                    step[used + chunk] = max(step.get(used + chunk, 0), pages + extra)
        most = step
    needed = max(most.values())
    assert layout.slot_pages(kind) >= needed
    assert layout.slot_pages(kind) - needed <= batch + 1


@pytest.mark.parametrize("batch", [1, 2, 4])
@pytest.mark.parametrize("budget", [4, 9, 32, 70])
def test_a_slot_holds_the_pages_of_any_batch_of_a_history_kind(batch, budget) -> None:
    layout = StagingLayout(
        StagingGeometry(
            compress_ratios=(1, 4),
            tokens_per_block=8,
            head_dim=512,
            index_head_dim=128,
            has_fp8_kv_cache=True,
            indexer_k_dtype="fp8",
            window_size=8,
            max_num_tokens=min(budget, 4),
            max_staging_tokens=budget,
            max_batch_size=batch,
        )
    )
    kind = StagingKind.COMPRESS_R4
    # A request of ``x`` cached and new tokens takes ``ceil(x / 8)`` pages.
    most = {0: 0}
    for _ in range(batch):
        step = dict(most)
        for used, pages in most.items():
            for tokens in range(1, budget - used + 1):
                step[used + tokens] = max(
                    step.get(used + tokens, 0), pages + layout.num_pages(kind, 0, tokens)
                )
        most = step
    assert layout.slot_pages(kind) == max(most.values())


def test_the_geometry_is_validated() -> None:
    with pytest.raises(ValueError, match="ring_depth"):
        _geometry(ring_depth=0)
    with pytest.raises(ValueError, match="must be positive"):
        _geometry(max_batch_size=0)
    with pytest.raises(ValueError, match="must hold one full chunk"):
        _geometry(max_staging_tokens=100, max_num_tokens=200)
    with pytest.raises(ValueError, match="Unsupported compress ratios"):
        _geometry(compress_ratios=(1, 2))
    with pytest.raises(ValueError, match="multiple of 4"):
        _geometry(compress_ratios=(1, 4), tokens_per_block=6)


def test_the_geometry_is_read_off_a_cache_manager() -> None:
    manager = types.SimpleNamespace(
        _compress_ratios=list(_pro_ratios()) + [1],
        num_layers=43,
        tokens_per_block=128,
        head_dim=512,
        index_head_dim=128,
        dtype=DataType.FP8,
        _indexer_k_dtype="fp4",
        _swa_window_size=128,
        max_num_tokens=4096,
        max_batch_size=8,
        _max_draft_len=0,
    )
    geometry = StagingGeometry.from_cache_manager(manager, max_staging_tokens=8192, ring_depth=3)
    assert geometry == replace(
        _geometry(),
        indexer_k_dtype="fp4",
        max_staging_tokens=8192,
        ring_depth=3,
    )
    bf16 = StagingGeometry.from_cache_manager(
        types.SimpleNamespace(**{**vars(manager), "dtype": DataType.BF16}), max_staging_tokens=8192
    )
    assert not bf16.has_fp8_kv_cache


@pytest.mark.parametrize(
    ("kind", "history", "chunk", "fetch", "writeback"),
    [
        (StagingKind.SWA, 0, 400, (0, 0), (0, 4)),
        (StagingKind.SWA, 300, 400, (1, 2), (2, 4)),
        (StagingKind.COMPRESS_R4, 300, 400, (0, 3), (2, 4)),
        # A history that ends on a block boundary has no partly cached block.
        (StagingKind.SWA, 256, 128, (1, 1), (2, 1)),
        (StagingKind.COMPRESS_R128, 256, 128, (0, 2), (2, 1)),
        (StagingKind.STATE_CSA, 300, 1, (2, 1), (2, 1)),
        (StagingKind.STATE_CSA, 255, 2, (1, 1), (1, 2)),
    ],
)
def test_what_is_fetched_and_what_is_written_back(kind, history, chunk, fetch, writeback) -> None:
    layout = StagingLayout(_geometry())
    assert layout.fetch_range(kind, history, chunk) == fetch
    assert layout.writeback_range(kind, history, chunk) == writeback


def test_every_staged_page_is_fetched_or_written_back_and_nothing_else() -> None:
    layout = StagingLayout(_geometry())
    for kind in StagingKind:
        for history in range(0, 700, 37):
            for chunk in (1, 2, 127, 128, 129, 400):
                first, pages = layout.block_range(kind, history, chunk)
                staged = set(range(first, first + pages))
                fetched = set(range(*_blocks(layout.fetch_range(kind, history, chunk))))
                written = set(range(*_blocks(layout.writeback_range(kind, history, chunk))))
                assert fetched | written == staged, (kind, history, chunk)
                assert fetched <= staged and written <= staged
                # Only blocks with cached tokens are fetched; only blocks with new tokens return.
                assert all(block * 128 < history for block in fetched)
                assert all((block + 1) * 128 > history for block in written)


def _blocks(span: tuple[int, int]) -> tuple[int, int]:
    first, count = span
    return first, first + count
