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
"""Moves the KV of a layer between the cache manager and the staging area around its attention.

The attention backend calls ``on_layer(l)`` at the top of every layer and ``end_forward()`` after
the last one. ``LoopbackStreamer`` serves a rank that owns every layer: the cached pages of the
layer are copied into the slot before its attention runs and the pages the new tokens wrote are
copied back after it, all on the stream of the forward pass, so the copies are ordered with the
kernels and need no events.
"""

from collections.abc import Sequence

import torch

from .dkv_staging import (
    BAD_PAGE_INDEX,
    DkvPageCopier,
    DkvStagedKvView,
    RequestSpan,
    StagingKind,
    pool_page_addresses,
)


class LoopbackStreamer:
    """Fetches and writes back the pages of one layer at a time, between the staging area and the
    cache manager of the same rank."""

    def __init__(
        self,
        manager,
        view: DkvStagedKvView,
        copier: DkvPageCopier | None = None,
        *,
        fill: str = "",
    ) -> None:
        """``fill`` (``"zero"`` or ``"nan"``, empty: none) overwrites the slots of a layer before
        its pages are fetched."""
        self._manager = manager
        self._fill = fill
        self._view = view
        self._layout = view.layout
        self._pool = view.pool
        self._copier = copier or DkvPageCopier()
        self._request_ids: list[int] = []
        self._history: list[int] = []
        self._new: list[int] = []
        self._spans: dict[StagingKind, tuple[RequestSpan, ...]] = {}
        self._written_layer: int | None = None
        self.bytes_fetched = 0
        self.bytes_written_back = 0

    def begin_iteration(
        self,
        request_ids: Sequence[int],
        history: Sequence[int],
        new: Sequence[int],
        spans: dict[StagingKind, tuple[RequestSpan, ...]],
    ) -> None:
        """Remember the staged batch; its pages are copied layer by layer."""
        self._request_ids = list(request_ids)
        self._history = list(history)
        self._new = list(new)
        self._spans = spans
        self._written_layer = None

    def on_layer(self, layer: int) -> None:
        """Called at the top of a layer: return the previous layer's pages, bring this layer's in."""
        if self._written_layer is not None:
            self._write_back(self._written_layer)
        if self._fill:
            self._pool.fill_layer(layer, self._fill)
        self._fetch(layer)
        self._written_layer = layer

    def end_forward(self) -> None:
        """Called after the last layer: return its pages."""
        if self._written_layer is not None:
            self._write_back(self._written_layer)
            self._written_layer = None

    def _pages(self, layer: int, fetch: bool) -> dict:
        """The (pool address, staging address) pairs of every component of ``layer``."""
        layout, pool, manager = self._layout, self._pool, self._manager
        stream_pairs: dict = {}
        for kind in layout.kinds:
            if layer not in layout.layers_of(kind):
                continue
            components = [c for c in layout.components if c.kind is kind]
            for index, request_id in enumerate(self._request_ids):
                if request_id not in manager.kv_cache_map:
                    continue
                history, new = self._history[index], self._new[index]
                span = self._spans[kind][index]
                first, count = (layout.fetch_range if fetch else layout.writeback_range)(
                    kind, history, new
                )
                if not count:
                    continue
                table = layout.block_table(kind, layer, span, first + count)
                for component in components:
                    indices = manager.get_cache_indices(request_id, layer, component.attention_type)
                    addresses = pool_page_addresses(
                        manager, layer, component.attention_type, indices
                    )
                    base = pool.layer_pointer(layer, component)
                    page_bytes = layout.page_bytes(component)
                    pairs = stream_pairs.setdefault(component, [])
                    for block in range(first, first + count):
                        if (
                            table[block] == BAD_PAGE_INDEX
                            or block >= len(addresses)
                            or addresses[block] is None
                        ):
                            continue
                        pairs.append((addresses[block], base + table[block] * page_bytes))
        return stream_pairs

    def _fetch(self, layer: int) -> None:
        stream = torch.cuda.current_stream().cuda_stream
        for component, pairs in self._pages(layer, fetch=True).items():
            if pairs:
                page_bytes = self._layout.page_bytes(component)
                self._copier.gather(
                    [pool for pool, _ in pairs],
                    [staging for _, staging in pairs],
                    page_bytes,
                    stream,
                )
                self.bytes_fetched += page_bytes * len(pairs)

    def _write_back(self, layer: int) -> None:
        stream = torch.cuda.current_stream().cuda_stream
        for component, pairs in self._pages(layer, fetch=False).items():
            if pairs:
                page_bytes = self._layout.page_bytes(component)
                self._copier.scatter(
                    [staging for _, staging in pairs],
                    [pool for pool, _ in pairs],
                    page_bytes,
                    stream,
                )
                self.bytes_written_back += page_bytes * len(pairs)
