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
"""DKV layer-split staging: the buffers a compute rank runs a layer's attention on.

Under the ``layer_split`` layout a rank stores the KV of only the layers it owns. To run the
attention of layer ``l`` the compute rank needs that layer's cached KV next to the rows it is about
to produce, so it keeps a *staging* area: for every kind of KV storage a ring of ``ring_depth``
slots, each large enough for the pages of all requests of one iteration. A layer always uses the
same slot of the ring of each kind it has (``slot_of``), so the pointers the attention backend bakes
at construction stay valid. The slot holds the pages of the requests back to back and the block
table of a slot is the identity map onto them, offset by where the request's pages start.

This module holds the arithmetic (``StagingLayout``, pure functions of the model geometry), the
buffers (``StagingPool``), the cache manager the attention backend sees when the layers are staged
(``DkvStagedKvView``) and the copies between staging pages and the pages of the real cache manager
(``DkvPageCopier``). It does not decide when anything is copied.
"""

import enum
from collections.abc import Sequence
from dataclasses import dataclass

import torch

from tensorrt_llm._torch.attention.backends.sparse.deepseek_v4.cache_manager import get_token_bytes
from tensorrt_llm._torch.attention.backends.sparse.deepseek_v4.params import (
    DeepseekV4AttentionType,
    is_overlap_compressor,
)
from tensorrt_llm._utils import (
    TensorWrapper,
    convert_to_torch_tensor,
    get_size_in_bytes,
    prefer_pinned,
)
from tensorrt_llm.bindings import DataType

# The index the cache manager gives a block that holds no page.
BAD_PAGE_INDEX = -1

# Each region of the pool starts on a multiple of this many bytes. The attention kernels need 16;
# a larger step keeps the regions on separate cache lines of the memory system.
REGION_ALIGNMENT = 256


class StagingKind(enum.Enum):
    """A kind of KV storage that is staged. Layers of one kind share one ring of slots."""

    SWA = "swa"
    COMPRESS_R4 = "compress_r4"
    COMPRESS_R128 = "compress_r128"
    INDEXER_COMPRESS = "indexer_compress"
    STATE_CSA = "state_csa"
    STATE_HCA = "state_hca"
    STATE_INDEXER = "state_indexer"

    @property
    def shared_page_index(self) -> bool:
        """Whether every layer indexes pages with the same page indices.

        The compressed caches are indexed alike by all layers that have them, and a layer's own
        region of the slot is selected by the base pointer. The sliding-window caches fold the
        layer into the page index instead and share one base pointer.
        """
        return self in (
            StagingKind.COMPRESS_R4,
            StagingKind.COMPRESS_R128,
            StagingKind.INDEXER_COMPRESS,
        )

    @property
    def windowed(self) -> bool:
        """Whether only the last tokens of the history are kept (a sliding window)."""
        return not self.shared_page_index

    @property
    def compress_ratio(self) -> int | None:
        """The compress ratio of the layers that have this kind; ``None``: every layer."""
        return {
            StagingKind.SWA: None,
            StagingKind.COMPRESS_R4: 4,
            StagingKind.COMPRESS_R128: 128,
            StagingKind.INDEXER_COMPRESS: 4,
            StagingKind.STATE_CSA: 4,
            StagingKind.STATE_HCA: 128,
            StagingKind.STATE_INDEXER: 4,
        }[self]

    @property
    def attention_types(self) -> tuple[DeepseekV4AttentionType, ...]:
        """The cache roles a layer holds for this kind, in the order their regions are laid out."""
        types = DeepseekV4AttentionType
        return {
            StagingKind.SWA: (types.SWA,),
            StagingKind.COMPRESS_R4: (types.COMPRESS,),
            StagingKind.COMPRESS_R128: (types.COMPRESS,),
            StagingKind.INDEXER_COMPRESS: (types.INDEXER_COMPRESS,),
            StagingKind.STATE_CSA: (types.COMPRESSOR_KV, types.COMPRESSOR_SCORE),
            StagingKind.STATE_HCA: (types.COMPRESSOR_KV, types.COMPRESSOR_SCORE),
            StagingKind.STATE_INDEXER: (
                types.INDEXER_COMPRESSOR_KV,
                types.INDEXER_COMPRESSOR_SCORE,
            ),
        }[self]


@dataclass(frozen=True)
class StagingComponent:
    """One buffer of the staging area: a cache role of a kind. A ring has one region per role."""

    kind: StagingKind
    attention_type: DeepseekV4AttentionType


STAGING_COMPONENTS: tuple[StagingComponent, ...] = tuple(
    StagingComponent(kind, attention_type)
    for kind in StagingKind
    for attention_type in kind.attention_types
)


@dataclass(frozen=True)
class StagingGeometry:
    """What the layout depends on; everything is known when the executor is built.

    ``compress_ratios`` has one entry per model layer: 1 for a layer with sliding-window attention
    only, 4 for a CSA layer, 128 for an HCA layer. ``max_num_tokens`` bounds the tokens one compute
    rank schedules in an iteration and ``max_staging_tokens`` the sum over its requests of the
    cached and the new tokens; the scheduler enforces both before it admits a request.
    """

    compress_ratios: tuple[int, ...]
    tokens_per_block: int
    head_dim: int
    index_head_dim: int
    has_fp8_kv_cache: bool
    indexer_k_dtype: str
    window_size: int
    max_num_tokens: int
    max_staging_tokens: int
    max_batch_size: int
    ring_depth: int = 2
    max_draft_len: int = 0

    def __post_init__(self) -> None:
        if self.ring_depth < 1:
            raise ValueError(f"ring_depth must be at least 1, got {self.ring_depth}")
        if min(self.tokens_per_block, self.max_num_tokens, self.max_batch_size) < 1:
            raise ValueError("tokens_per_block, max_num_tokens and max_batch_size must be positive")
        if self.max_staging_tokens < self.max_num_tokens:
            raise ValueError(
                f"max_staging_tokens ({self.max_staging_tokens}) must hold one full chunk "
                f"({self.max_num_tokens} tokens)"
            )
        unsupported = set(self.compress_ratios) - {1, 4, 128}
        if unsupported:
            raise ValueError(f"Unsupported compress ratios {sorted(unsupported)}")
        for ratio in set(self.compress_ratios) - {1}:
            if self.tokens_per_block % ratio:
                raise ValueError(
                    f"tokens_per_block ({self.tokens_per_block}) must be a multiple of {ratio}"
                )

    @classmethod
    def from_cache_manager(
        cls, manager, *, max_staging_tokens: int, ring_depth: int = 2
    ) -> "StagingGeometry":
        """Read the geometry off a DeepSeek-V4 cache manager."""
        return cls(
            compress_ratios=tuple(manager._compress_ratios[: manager.num_layers]),
            tokens_per_block=manager.tokens_per_block,
            head_dim=manager.head_dim,
            index_head_dim=manager.index_head_dim,
            has_fp8_kv_cache=manager.dtype == DataType.FP8,
            indexer_k_dtype=manager._indexer_k_dtype,
            window_size=manager._swa_window_size,
            max_num_tokens=manager.max_num_tokens,
            max_staging_tokens=max_staging_tokens,
            max_batch_size=manager.max_batch_size,
            ring_depth=ring_depth,
            max_draft_len=manager._max_draft_len,
        )


@dataclass(frozen=True)
class RequestSpan:
    """The staged pages of one request in one kind.

    Block ``b`` of the request (the tokens ``b * tokens_per_block`` and up) is staged for ``b`` in
    ``first_block <= b < first_block + num_pages``; its page is ``page_offset + b - first_block``
    of the slot.
    """

    first_block: int
    num_pages: int
    page_offset: int


class StagingOverflow(ValueError):
    """The requests of an iteration do not fit the slots of a kind."""


class StagingLayout:
    """Sizes, offsets and block tables of the staging area. A pure function of the geometry."""

    def __init__(self, geometry: StagingGeometry) -> None:
        self.geometry = geometry
        self._layers: dict[StagingKind, tuple[int, ...]] = {}
        for kind in StagingKind:
            ratio = kind.compress_ratio
            self._layers[kind] = tuple(
                layer
                for layer, layer_ratio in enumerate(geometry.compress_ratios)
                if ratio is None or layer_ratio == ratio
            )
        self.kinds = tuple(kind for kind in StagingKind if self._layers[kind])
        self.components = tuple(
            component for component in STAGING_COMPONENTS if self._layers[component.kind]
        )
        self._region_offsets: dict[StagingComponent, int] = {}
        offset = 0
        for component in self.components:
            self._region_offsets[component] = offset
            offset += _align_up(self.region_bytes(component), REGION_ALIGNMENT)
        self.total_bytes = offset

    # ---- layers and slots -----------------------------------------------------------------

    def layers_of(self, kind: StagingKind) -> tuple[int, ...]:
        """The model layers that have KV of this kind, in increasing order."""
        return self._layers[kind]

    def slot_of(self, layer: int, kind: StagingKind) -> int:
        """The ring slot layer ``layer`` uses for this kind.

        A layer is fetched ``ring_depth - 1`` layers ahead of its attention, so the slot it takes
        was last used by a layer that has finished by then.
        """
        layers = self._layers[kind]
        try:
            return layers.index(layer) % self.geometry.ring_depth
        except ValueError:
            raise ValueError(f"layer {layer} has no {kind.value} storage") from None

    # ---- sizes ----------------------------------------------------------------------------

    def page_bytes(self, component: StagingComponent) -> int:
        """The bytes of one page of the component; equal to the cache manager's page."""
        geometry = self.geometry
        kind, attention_type = component.kind, component.attention_type
        ratio = kind.compress_ratio or 1
        token_bytes = get_token_bytes(
            geometry.head_dim,
            geometry.index_head_dim,
            ratio,
            attention_type,
            geometry.has_fp8_kv_cache,
            indexer_k_dtype=geometry.indexer_k_dtype,
        )
        rows = geometry.tokens_per_block
        if attention_type in (
            DeepseekV4AttentionType.COMPRESS,
            DeepseekV4AttentionType.INDEXER_COMPRESS,
        ):
            rows //= ratio
        return token_bytes * rows

    def window(self, kind: StagingKind) -> int | None:
        """The window of a windowed kind in tokens, ``None`` for a kind that keeps the history."""
        geometry = self.geometry
        if not kind.windowed:
            return None
        if kind is StagingKind.SWA:
            return geometry.window_size + geometry.max_draft_len
        factor = 2 if is_overlap_compressor(kind.compress_ratio) else 1
        return factor * kind.compress_ratio + geometry.max_draft_len

    def slot_pages(self, kind: StagingKind) -> int:
        """The pages of one slot: what the requests of any admitted iteration can take together."""
        geometry = self.geometry
        batch, block = geometry.max_batch_size, geometry.tokens_per_block
        window = self.window(kind)
        if window is None:
            budget = geometry.max_staging_tokens
            if budget <= batch:
                return budget
            return batch + (budget - batch) // block
        # A request touches the blocks of its window and its new tokens: an interval of at most
        # ``chunk + window - 1`` tokens, which spans at most ``ceil((chunk + window - 2) / block) + 1``
        # blocks. Summed over ``n`` requests whose chunks add up to the chunk budget this is at most
        # ``2n + (budget + n * (window - 3)) // block``, largest at one end of the range of ``n``.
        return max(
            2 * n + max(0, (geometry.max_num_tokens + n * (window - 3)) // block)
            for n in {1, min(batch, geometry.max_num_tokens)}
        )

    def slot_bytes(self, component: StagingComponent) -> int:
        return self.slot_pages(component.kind) * self.page_bytes(component)

    def region_bytes(self, component: StagingComponent) -> int:
        """The bytes of the component's whole ring."""
        return self.geometry.ring_depth * self.slot_bytes(component)

    def region_offset(self, component: StagingComponent) -> int:
        """Where the component's ring starts in the pool."""
        return self._region_offsets[component]

    def layer_base_offset(self, layer: int, component: StagingComponent) -> int:
        """The byte offset in the pool of the pages layer ``layer`` is addressed from.

        Sliding-window kinds put the slot into the page index (see ``slot_page_offset``) and are
        addressed from the start of their ring; the compressed kinds are addressed from the start
        of the layer's own slot.
        """
        offset = self.region_offset(component)
        if component.kind.shared_page_index:
            offset += self.slot_of(layer, component.kind) * self.slot_bytes(component)
        return offset

    def slot_page_offset(self, layer: int, kind: StagingKind) -> int:
        """The pages to add to the page index of a sliding-window kind for ``layer``'s slot."""
        if kind.shared_page_index:
            return 0
        return self.slot_of(layer, kind) * self.slot_pages(kind)

    # ---- pages of a request ---------------------------------------------------------------

    def block_range(self, kind: StagingKind, history: int, chunk: int) -> tuple[int, int]:
        """The first block and the number of blocks staged for a request.

        ``history`` tokens are cached and ``chunk`` new ones are computed. A kind that keeps the
        history stages every block up to the last new token; a windowed kind stages the blocks
        from the oldest token the first new token still attends to.
        """
        if history < 0 or chunk < 1:
            raise ValueError(f"need history >= 0 and chunk >= 1, got {history} and {chunk}")
        block = self.geometry.tokens_per_block
        last = (history + chunk - 1) // block
        window = self.window(kind)
        first = 0 if window is None else max(0, history + 1 - window) // block
        return first, last - first + 1

    def num_pages(self, kind: StagingKind, history: int, chunk: int) -> int:
        return self.block_range(kind, history, chunk)[1]

    def fetch_range(self, kind: StagingKind, history: int, chunk: int) -> tuple[int, int]:
        """The staged blocks that hold cached tokens, as ``(first block, number of blocks)``.

        These are what has to be brought into the slot before the layer runs; the blocks past the
        last cached token start out empty. A block with only some of its tokens cached counts.
        """
        first, pages = self.block_range(kind, history, chunk)
        cached_blocks = -(-history // self.geometry.tokens_per_block)
        return first, max(0, min(first + pages, cached_blocks) - first)

    def writeback_range(self, kind: StagingKind, history: int, chunk: int) -> tuple[int, int]:
        """The staged blocks the new tokens write to, as ``(first block, number of blocks)``.

        They are sent back after the layer. A block that held cached tokens as well as new ones is
        sent whole; the cached part is unchanged, so overwriting it is harmless.
        """
        first, pages = self.block_range(kind, history, chunk)
        first_new = max(first, history // self.geometry.tokens_per_block)
        return first_new, first + pages - first_new

    def request_spans(
        self, kind: StagingKind, requests: Sequence[tuple[int, int]]
    ) -> tuple[RequestSpan, ...]:
        """Place the requests, each given as ``(history, chunk)``, one after the other in a slot.

        Raises:
            StagingOverflow: The requests need more pages than a slot has.
        """
        spans = []
        offset = 0
        for history, chunk in requests:
            first, pages = self.block_range(kind, history, chunk)
            spans.append(RequestSpan(first, pages, offset))
            offset += pages
        if offset > self.slot_pages(kind):
            raise StagingOverflow(
                f"{len(requests)} requests need {offset} {kind.value} pages, "
                f"a slot has {self.slot_pages(kind)}"
            )
        return tuple(spans)

    def block_table(
        self, kind: StagingKind, layer: int, span: RequestSpan, num_blocks: int
    ) -> list[int]:
        """The page index of each of the first ``num_blocks`` blocks of a request for ``layer``.

        A block that is not staged has no page, as an expired block has none in the cache manager.
        """
        slot_offset = self.slot_page_offset(layer, kind)
        end = span.first_block + span.num_pages
        return [
            slot_offset + span.page_offset + block - span.first_block
            if span.first_block <= block < end
            else BAD_PAGE_INDEX
            for block in range(num_blocks)
        ]


class StagingPool:
    """The staging buffers of one compute rank: one allocation holding every ring of a layout."""

    def __init__(self, layout: StagingLayout, device: torch.device | str | int = "cuda") -> None:
        self.layout = layout
        self.buffer = torch.empty(layout.total_bytes, dtype=torch.uint8, device=device)
        if self.buffer.data_ptr() % REGION_ALIGNMENT:
            raise RuntimeError(
                f"The staging buffer starts at {self.buffer.data_ptr():#x}, which is not "
                f"{REGION_ALIGNMENT}-byte aligned"
            )

    @property
    def bytes_reserved(self) -> int:
        return self.layout.total_bytes

    def region(self, component: StagingComponent) -> torch.Tensor:
        """The ring of ``component`` as bytes."""
        begin = self.layout.region_offset(component)
        return self.buffer[begin : begin + self.layout.region_bytes(component)]

    def pages(self, component: StagingComponent) -> torch.Tensor:
        """The ring of ``component`` as ``[ring_depth * slot_pages, page_bytes]`` bytes."""
        return self.region(component).view(-1, self.layout.page_bytes(component))

    def layer_pointer(self, layer: int, component: StagingComponent) -> int:
        """The address the pages of ``layer`` are indexed from (see ``layer_base_offset``)."""
        return self.buffer.data_ptr() + self.layout.layer_base_offset(layer, component)

    def slot(self, layer: int, component: StagingComponent) -> torch.Tensor:
        """The bytes of the slot of the component's ring that ``layer`` uses."""
        layout = self.layout
        begin = layout.region_offset(component) + layout.slot_of(
            layer, component.kind
        ) * layout.slot_bytes(component)
        return self.buffer[begin : begin + layout.slot_bytes(component)]

    def fill(self, mode: str) -> None:
        """Overwrite every byte: ``zero``, or ``nan``.

        The ``nan`` fill makes a byte pattern that reads as a NaN in the float32, bfloat16 and
        float8 formats the staged storages use, so an output that depends on a staging page that
        was never fetched or written turns into NaN.
        """
        self.buffer.fill_(_fill_byte(mode))

    def fill_layer(self, layer: int, mode: str) -> None:
        """Overwrite the slots ``layer`` uses, whatever an earlier layer left in them."""
        value = _fill_byte(mode)
        for component in self.layout.components:
            if layer in self.layout.layers_of(component.kind):
                self.slot(layer, component).fill_(value)


class DkvStagedKvView:
    """The cache manager as the attention backend sees it when the layers are staged.

    The backend reads every layer's KV through ``metadata.kv_cache_manager``: the buffers of a
    layer, the per-layer block tables and the base pointers it bakes into its metadata. This view
    answers all of that from the staging area: the buffers are views of the rings, the base
    pointers are those of the rings, and the block tables are the identity maps onto the pages the
    requests of the iteration occupy in a slot. Everything else (sizes, dtypes, the capacity the
    metadata is allocated for, ...) is the real manager's, so the backend does not know the
    difference.

    The view covers every layer of the model, whether or not the real manager holds it, which is
    what the layer-split layout needs. It supports the FP8 and BF16 KV layouts of DeepSeek-V4 on
    GPUs whose attention op reads the KV directly; the footer-scale and NVFP4 layouts are not
    staged.
    """

    # The block tables and the buffers of the real manager have these cache roles per layer.
    _SLIDING = (
        DeepseekV4AttentionType.SWA,
        DeepseekV4AttentionType.COMPRESSOR_KV,
        DeepseekV4AttentionType.COMPRESSOR_SCORE,
        DeepseekV4AttentionType.INDEXER_COMPRESSOR_KV,
        DeepseekV4AttentionType.INDEXER_COMPRESSOR_SCORE,
    )

    def __init__(self, manager, pool: StagingPool) -> None:
        if getattr(manager, "use_fp8_ds_mla", False) or getattr(
            manager, "_use_nvfp4_compress", False
        ):
            raise NotImplementedError(
                "The footer-scale and NVFP4 KV layouts are not supported by the staged view"
            )
        self._manager = manager
        self.pool = pool
        self.layout = pool.layout
        num_layers = len(self.layout.geometry.compress_ratios)
        self.num_layers = num_layers
        self.pp_layers = list(range(num_layers))
        self.num_local_layers = num_layers
        self.layer_offsets = {layer: layer for layer in range(num_layers)}
        self.max_attention_window_vec = list(manager.max_attention_window_vec)
        if len(self.max_attention_window_vec) != num_layers:
            raise NotImplementedError(
                "The staged view needs the attention windows of every layer; the manager has "
                f"{len(self.max_attention_window_vec)} of {num_layers}"
            )
        component = {(c.kind, c.attention_type): c for c in self.layout.components}
        swa = component[(StagingKind.SWA, DeepseekV4AttentionType.SWA)]
        self.swa_pool_ptr = pool.buffer.data_ptr() + self.layout.region_offset(swa)
        self.compress_pool_ptrs = {
            ratio: pool.buffer.data_ptr()
            + self.layout.region_offset(component[(kind, DeepseekV4AttentionType.COMPRESS)])
            for ratio, kind in ((4, StagingKind.COMPRESS_R4), (128, StagingKind.COMPRESS_R128))
            if kind in self.layout.kinds
        }
        self.compress_scale_pool_ptrs: dict[int, int] = {}
        # What the attention op sees of the sliding-window pool: one virtual pool per layer, all
        # at the base of the SWA ring (the slot of a layer is in its page indices).
        self.kv_cache_pool_pointers = torch.tensor(
            [[self.swa_pool_ptr, 0] for _ in range(num_layers)],
            dtype=torch.int64,
            device="cpu",
            pin_memory=prefer_pinned(),
        )
        self.kv_cache_pool_mapping = torch.tensor(
            [[layer, 0] for layer in range(num_layers)],
            dtype=torch.int32,
            device="cpu",
            pin_memory=prefer_pinned(),
        )
        self.num_attention_op_pools = num_layers
        # The staged kind of every (layer, sliding-window cache role), which does not change.
        self._sliding_kinds: list[list[StagingKind | None]] = [
            [None] * len(self._SLIDING) for _ in range(num_layers)
        ]
        for index, attention_type in enumerate(self._SLIDING):
            for kind in self.layout.kinds:
                if attention_type in kind.attention_types:
                    for layer in self.layout.layers_of(kind):
                        self._sliding_kinds[layer][index] = kind
        self._buffers: dict[tuple[int, DeepseekV4AttentionType], torch.Tensor] = {}
        self._tables: dict[tuple[StagingKind, int], torch.Tensor] | None = None
        self._width = 0
        self._num_requests = 0
        self._sliding: torch.Tensor | None = None
        # Moves the pages between the real manager and the staging area; none: the content of the
        # staging area is not maintained (warm-up and profiling forwards).
        self.dkv_streamer = None

    def __getattr__(self, name: str):
        # Only called for what the view does not define itself.
        if name == "_manager":
            raise AttributeError(name)
        return getattr(self._manager, name)

    # ---- buffers --------------------------------------------------------------------------

    def _component(self, layer: int, attention_type: DeepseekV4AttentionType) -> StagingComponent:
        for component in self.layout.components:
            if component.attention_type is attention_type and layer in self.layout.layers_of(
                component.kind
            ):
                return component
        raise KeyError(f"layer {layer} has no {attention_type.name} storage")

    def get_buffers(self, layer_idx: int, attn_type: DeepseekV4AttentionType) -> torch.Tensor:
        """The staged pages of a layer's cache role, shaped ``[pages, rows per page, row size]``.

        The pages are indexed like the cache manager's: by the entries of the block tables this
        view builds. The tensor starts at the layer's base address, so a table entry is a number of
        pages from there.
        """
        key = (layer_idx, attn_type)
        if key not in self._buffers:
            manager = self._manager
            component = self._component(layer_idx, attn_type)
            layout = self.layout
            page_bytes = layout.page_bytes(component)
            rows = layout.geometry.tokens_per_block
            if attn_type in (
                DeepseekV4AttentionType.COMPRESS,
                DeepseekV4AttentionType.INDEXER_COMPRESS,
            ):
                rows //= component.kind.compress_ratio
            row_bytes = page_bytes // rows
            base = self.pool.layer_pointer(layer_idx, component)
            region_end = (
                self.pool.buffer.data_ptr()
                + layout.region_offset(component)
                + layout.region_bytes(component)
            )
            pages = (region_end - base) // page_bytes
            if attn_type is DeepseekV4AttentionType.INDEXER_COMPRESS:
                dtype = manager._indexer_dtype
            elif attn_type in (
                DeepseekV4AttentionType.COMPRESSOR_KV,
                DeepseekV4AttentionType.COMPRESSOR_SCORE,
                DeepseekV4AttentionType.INDEXER_COMPRESSOR_KV,
                DeepseekV4AttentionType.INDEXER_COMPRESSOR_SCORE,
            ):
                dtype = manager._compressor_dtype
            else:
                dtype = manager.dtype
            if attn_type is DeepseekV4AttentionType.INDEXER_COMPRESS:
                row_elements = manager._indexer_data_size + manager._indexer_scale_size
            else:
                row_elements = _elements_per_row(row_bytes, dtype)
            self._buffers[key] = convert_to_torch_tensor(
                TensorWrapper(base, dtype, (pages, rows, row_elements))
            )
        return self._buffers[key]

    def get_indexer_k_cache_buffers(self, layer_idx: int) -> torch.Tensor:
        buffer = self.get_buffers(layer_idx, DeepseekV4AttentionType.INDEXER_COMPRESS).unsqueeze(2)
        return buffer.view(torch.uint8)

    def get_compress_scale_buffers(self, layer_idx: int) -> torch.Tensor:
        raise RuntimeError("COMPRESS block scales exist only for the NVFP4 KV cache.")

    def get_compress_pool_buffers(self, compress_ratio: int):
        raise RuntimeError("COMPRESS NVFP4 pools are not staged.")

    # ---- block tables ---------------------------------------------------------------------

    def begin_staged_batch(
        self,
        request_ids: Sequence[int],
        history_tokens: Sequence[int],
        new_tokens: Sequence[int],
        num_contexts: int,
    ) -> None:
        """Place the requests of the iteration in the slots; the tables follow from this.

        ``history_tokens`` are the tokens each request already has cached and ``new_tokens`` the
        ones the iteration computes, in the order of the batch. Raises ``StagingOverflow`` when the
        requests need more pages than a slot has.
        """
        if not (len(request_ids) == len(history_tokens) == len(new_tokens)):
            raise ValueError("request_ids, history_tokens and new_tokens differ in length")
        requests = list(zip(history_tokens, new_tokens))
        layout = self.layout
        spans = {kind: layout.request_spans(kind, requests) for kind in layout.kinds}
        self._width = max(
            (
                span.first_block + span.num_pages
                for kind_spans in spans.values()
                for span in kind_spans
            ),
            default=0,
        )
        max_blocks = self._manager.max_blocks_per_seq
        if self._width > max_blocks:
            raise StagingOverflow(
                f"a request spans {self._width} blocks, the block tables have {max_blocks}"
            )
        tables: dict[tuple[StagingKind, int], torch.Tensor] = {}
        for kind in layout.kinds:
            slots = (0,) if kind.shared_page_index else range(layout.geometry.ring_depth)
            for slot in slots:
                slot_offset = 0 if kind.shared_page_index else slot * layout.slot_pages(kind)
                table = torch.full((len(requests), self._width), BAD_PAGE_INDEX, dtype=torch.int32)
                for row, span in enumerate(spans[kind]):
                    first = slot_offset + span.page_offset
                    table[row, span.first_block : span.first_block + span.num_pages] = torch.arange(
                        first, first + span.num_pages, dtype=torch.int32
                    )
                tables[kind, slot] = table
        self._tables = tables
        self._sliding = None
        self._num_requests = len(requests)
        if self.dkv_streamer is not None:
            self.dkv_streamer.begin_iteration(request_ids, history_tokens, new_tokens, spans)

    def _table(self, kind: StagingKind, layer: int) -> torch.Tensor:
        if self._tables is None:
            raise RuntimeError("No batch is staged: begin_staged_batch has not been called")
        slot = 0 if kind.shared_page_index else self.layout.slot_of(layer, kind)
        return self._tables[kind, slot]

    def compute_sliding_block_tables(self, request_ids: Sequence[int], num_contexts: int) -> None:
        """Build the per-layer block tables of the sliding-window cache roles for the batch."""
        if self._tables is None:
            raise RuntimeError("No batch is staged: begin_staged_batch has not been called")
        tables = torch.full(
            (self.num_layers, len(self._SLIDING), self._num_requests, self._width),
            BAD_PAGE_INDEX,
            dtype=torch.int32,
        )
        for layer in range(self.num_layers):
            for index, kind in enumerate(self._sliding_kinds[layer]):
                if kind is not None:
                    tables[layer, index] = self._table(kind, layer)
        self._sliding = tables

    def copy_batch_sliding_block_tables(
        self, dst_tensor: torch.Tensor, request_ids: Sequence[int], num_contexts: int, num_seqs: int
    ) -> None:
        assert dst_tensor.is_cuda, "copy_batch_sliding_block_tables expects a CUDA destination"
        dst_tensor.fill_(BAD_PAGE_INDEX)
        if self._width:
            dst_tensor[:, :, :num_seqs, : self._width].copy_(
                self._sliding[:, :, :num_seqs], non_blocking=True
            )

    def copy_batch_block_offsets(
        self,
        dst_tensor: torch.Tensor,
        request_ids: Sequence[int],
        beam_width: int,
        num_contexts: int,
        num_seqs: int,
        max_blocks: int | None = None,
    ) -> None:
        """For the attention op: the sliding-window (SWA) table of every layer."""
        assert beam_width == 1, "DSV4 only supports beam width 1 now"
        assert dst_tensor.is_cuda, "copy_batch_block_offsets expects a CUDA destination"
        dst_tensor.fill_(BAD_PAGE_INDEX)
        if self._width:
            dst_tensor[:, :num_seqs, 0, : self._width].copy_(
                self._sliding[:, DeepseekV4AttentionType.SWA.value, :num_seqs], non_blocking=True
            )

    def copy_batch_compress_block_tables(
        self,
        dst_tensor: torch.Tensor,
        request_ids: Sequence[int],
        compress_ratio: int,
        beam_width: int,
        num_contexts: int,
        num_seqs: int,
    ) -> None:
        assert beam_width == 1, "DSV4 only supports beam width 1 now"
        kind = {4: StagingKind.COMPRESS_R4, 128: StagingKind.COMPRESS_R128}[compress_ratio]
        dst_tensor[:num_seqs].fill_(BAD_PAGE_INDEX)
        if self._width:
            dst_tensor[:num_seqs, : self._width].copy_(self._table(kind, 0), non_blocking=True)

    def copy_batch_indexer_compress_block_tables(
        self,
        host_block_table: torch.Tensor,
        request_ids: Sequence[int],
        beam_width: int,
        num_contexts: int,
        num_seqs: int,
    ) -> None:
        assert beam_width == 1, "DSV4 only supports beam width 1 now"
        host_block_table[:num_seqs].fill_(BAD_PAGE_INDEX)
        if self._width:
            host_block_table[:num_seqs, : self._width] = self._table(
                StagingKind.INDEXER_COMPRESS, 0
            )


def _elements_per_row(row_bytes: int, dtype) -> int:
    """How many elements of a bindings ``DataType`` a row of ``row_bytes`` bytes holds."""
    return row_bytes // get_size_in_bytes(1, dtype)


def pool_page_addresses(
    manager, layer: int, attention_type: DeepseekV4AttentionType, page_indices: Sequence[int]
) -> list[int | None]:
    """The device address of each page a cache manager holds for ``layer``'s ``attention_type``.

    ``page_indices`` are the entries of the block table the manager hands to the attention kernels
    (``get_cache_indices``); a block without a page maps to ``None``.
    """
    buffer = manager.get_buffers(layer, attention_type)
    page_bytes = buffer.stride(0) * buffer.element_size()
    base = buffer.data_ptr()
    return [
        None if index == BAD_PAGE_INDEX else base + index * page_bytes for index in page_indices
    ]


class DkvPageCopier:
    """Copies pages between the staging area and the pages of the cache manager, on a stream."""

    # The copy kernel receives its tasks by value in its launch arguments.
    MAX_TASKS_PER_CALL = 256

    def __init__(self) -> None:
        from tensorrt_llm.bindings.internal.batch_manager import kv_cache_manager_v2_utils

        self._utils = kv_cache_manager_v2_utils

    def copy(self, pairs: Sequence[tuple[int, int]], num_bytes: int, stream: int) -> None:
        """Copy ``num_bytes`` from the source to the destination address of every pair.

        Args:
            pairs: ``(destination address, source address)`` of device memory.
            num_bytes: The size of every copy, a positive multiple of 16.
            stream: The CUDA stream handle the copies are enqueued on.
        """
        if num_bytes <= 0 or num_bytes % 16:
            raise ValueError(f"num_bytes must be a positive multiple of 16, got {num_bytes}")
        for begin in range(0, len(pairs), self.MAX_TASKS_PER_CALL):
            tasks = [
                self._utils.MemToMemTask(destination, source)
                for destination, source in pairs[begin : begin + self.MAX_TASKS_PER_CALL]
            ]
            result = self._utils.copy_device_to_device(tasks, num_bytes, stream)
            if result != 0:
                raise RuntimeError(f"copy_device_to_device failed with CUDA error {result}")

    def gather(
        self,
        pool_addresses: Sequence[int],
        staging_addresses: Sequence[int],
        page_bytes: int,
        stream: int,
    ) -> None:
        """Copy pages of the cache manager into staging pages."""
        self.copy(list(zip(staging_addresses, pool_addresses)), page_bytes, stream)

    def scatter(
        self,
        staging_addresses: Sequence[int],
        pool_addresses: Sequence[int],
        page_bytes: int,
        stream: int,
    ) -> None:
        """Copy staging pages into pages of the cache manager."""
        self.copy(list(zip(pool_addresses, staging_addresses)), page_bytes, stream)


def _align_up(value: int, alignment: int) -> int:
    return -(-value // alignment) * alignment


def _fill_byte(mode: str) -> int:
    try:
        return {"zero": 0x00, "nan": 0xFF}[mode]
    except KeyError:
        raise ValueError(f"unknown staging fill {mode!r}: use 'zero' or 'nan'") from None
