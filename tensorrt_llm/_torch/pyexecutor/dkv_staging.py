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
buffers (``StagingPool``) and the copies between staging pages and the pages of the real cache
manager (``DkvPageCopier``). It does not decide when anything is copied.
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
