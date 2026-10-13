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
"""DKV layer split: the working cache the attention reads, and what the data plane moves.

Under ``kv_layout="layer_split"`` the cache manager of a rank stores the KV of the layers the rank
owns, while every rank runs the attention of every layer for the requests it computes. Those
requests read and write their KV in the *working cache*: a second instance of the model's own cache
manager class that covers every layer and holds pages only for the requests one rank computes in
one iteration. The attention metadata reads it through the interfaces it uses with any cache
manager (pool base addresses, page index converters, the base page indices of the requests), so
nothing on this side knows the model.

Around each layer the data plane copies that layer's pages between the owner's cache and the working
cache of the rank that computes. Which pages, how many bytes and where they are follows from what
every V2 cache manager publishes, again without knowing the model:

* the buffers of a model layer (``manager_buffers``): a V2 layer id, a data role and the bytes of
  one block, named by ``(model layer, role)`` because V2 layer ids differ between managers that
  hold different layers;
* the life cycle of each buffer, whose stale-block rule (``stale_range``, the rule the V2 manager
  applies itself) says which blocks of a request hold pages while it computes (``staged_blocks``),
  which of those hold cached tokens to fetch (``fetch_blocks``) and which the new tokens write
  (``writeback_blocks``);
* the page geometry (``page_affine``): the page of base index ``p`` of a buffer is at
  ``base + p * step``.
"""

import math
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from tensorrt_llm.runtime.kv_cache_manager_v2 import PageIndexMode

from .dkv_plan import PageCost
from .dkv_types import LifecycleKey
from .kv_cache.lifecycle_slot_counts import lifecycle_key, lifecycle_layouts, stale_range

#: Unless the user sets the staged tokens, the pages of the life cycles that keep the whole history,
#: the only ones that grow with the staged tokens, take at most this many bytes in the working cache
#: and the message buffers of a rank (see ``WorkingCacheSizing``).
DEFAULT_STAGING_HISTORY_BYTES = 1 << 30


class WorkingCacheOverflow(RuntimeError):
    """The requests of an iteration need more pages than the working cache has."""


@dataclass(frozen=True)
class DkvBuffer:
    """A KV buffer of one model layer, named the same way on every rank.

    ``page_bytes`` is the size of one block of the buffer; ``key`` is the life cycle that decides
    which blocks of a request hold a page.
    """

    layer: int
    role: str
    key: LifecycleKey
    page_bytes: int

    @property
    def name(self) -> tuple[int, str]:
        return self.layer, self.role


@dataclass(frozen=True)
class ManagerBuffer:
    """A ``DkvBuffer`` as one cache manager holds it: under which V2 layer id and data role."""

    buffer: DkvBuffer
    layer_id: Any
    role: Any


def manager_buffers(manager) -> dict[tuple[int, str], ManagerBuffer]:
    """Every buffer of the layers ``manager`` holds, by ``(model layer, role)``.

    The roles the manager excludes from transfers (``get_disagg_ignored_roles``, such as padding)
    are left out.
    """
    config = manager.kv_cache_manager_py_config
    tokens_per_block = int(config.tokens_per_block)
    ignored = set(manager.get_disagg_ignored_roles())
    buffers: dict[tuple[int, str], ManagerBuffer] = {}
    for layer in config.layers:
        key = lifecycle_key(layer, tokens_per_block)
        for buffer_config in layer.buffers:
            role = buffer_config.role
            if role in ignored:
                continue
            override = buffer_config.tokens_per_block_override or tokens_per_block
            page_bytes = int(buffer_config.size) * (tokens_per_block // override)
            buffer = DkvBuffer(
                manager.dkv_model_layer_of(layer.layer_id, role), str(role), key, page_bytes
            )
            if buffer.name in buffers:
                raise ValueError(f"two buffers of model layer {buffer.layer} have role {role}")
            buffers[buffer.name] = ManagerBuffer(buffer, layer.layer_id, role)
    return buffers


def buffers_by_layer(buffers: Iterable[DkvBuffer]) -> dict[int, tuple[DkvBuffer, ...]]:
    """The buffers of every model layer, ordered by role so that every rank lists them alike."""
    by_layer: dict[int, list[DkvBuffer]] = defaultdict(list)
    for buffer in buffers:
        by_layer[buffer.layer].append(buffer)
    return {
        layer: tuple(sorted(layer_buffers, key=lambda buffer: buffer.role))
        for layer, layer_buffers in sorted(by_layer.items())
    }


# ---- which blocks move ----------------------------------------------------------------------------


def _check_staged(key: LifecycleKey) -> None:
    if key.is_ssm:
        raise NotImplementedError("recurrent-state life cycles are not staged by the layer split")
    if key.num_sink_blocks:
        raise NotImplementedError("attention sinks are not staged by the layer split")


def _blocks(tokens: int, tokens_per_block: int) -> int:
    return -(-tokens // tokens_per_block)


def staged_blocks(
    key: LifecycleKey, history: int, chunk: int, tokens_per_block: int
) -> tuple[int, int]:
    """``(first block, count)`` of the blocks a request holds pages for while it computes.

    ``history`` tokens are cached and ``chunk`` new ones are computed: the blocks up to the last new
    token, from the first block the life cycle still keeps at that history.
    """
    _check_staged(key)
    if history < 0 or chunk < 1:
        raise ValueError(f"need history >= 0 and chunk >= 1, got {history} and {chunk}")
    first = stale_range(key, history, tokens_per_block)[1]
    return first, _blocks(history + chunk, tokens_per_block) - first


def fetch_blocks(
    key: LifecycleKey, history: int, chunk: int, tokens_per_block: int
) -> tuple[int, int]:
    """``(first block, count)`` of the staged blocks that hold cached tokens.

    A block with only some of its tokens cached counts: it has to be brought in whole.
    """
    first, count = staged_blocks(key, history, chunk, tokens_per_block)
    cached_end = _blocks(history, tokens_per_block)
    return first, max(0, min(first + count, cached_end) - first)


def writeback_blocks(
    key: LifecycleKey, history: int, chunk: int, tokens_per_block: int
) -> tuple[int, int]:
    """``(first block, count)`` of the staged blocks the new tokens write and the cache keeps.

    A block that held cached tokens as well as new ones goes back whole: the cached part is
    unchanged. A block the life cycle drops once the chunk is done does not go back.
    """
    first, count = staged_blocks(key, history, chunk, tokens_per_block)
    end = first + count
    first_written = max(
        first,
        history // tokens_per_block,
        stale_range(key, history + chunk, tokens_per_block)[1],
    )
    return first_written, max(0, end - first_written)


class BufferCostModel:
    """The plan's cost model (``StagingCostModel``) over the buffers of every model layer."""

    def __init__(self, buffers: Iterable[DkvBuffer], tokens_per_block: int) -> None:
        self._by_layer = buffers_by_layer(buffers)
        self._buffers = {
            buffer.name: buffer
            for layer_buffers in self._by_layer.values()
            for buffer in layer_buffers
        }
        self.tokens_per_block = tokens_per_block

    def buffers_of(self, layer: int) -> tuple[str, ...]:
        return tuple(buffer.role for buffer in self._by_layer.get(layer, ()))

    def buffer(self, layer: int, role: str) -> DkvBuffer:
        return self._buffers[layer, role]

    def page_cost(self, layer: int, buffer: str, history: int, chunk: int) -> PageCost:
        spec = self._buffers[layer, buffer]
        return PageCost(
            spec.page_bytes,
            fetch_blocks(spec.key, history, chunk, self.tokens_per_block)[1],
            writeback_blocks(spec.key, history, chunk, self.tokens_per_block)[1],
        )


# ---- where the pages are --------------------------------------------------------------------------


@dataclass(frozen=True)
class PageAffine:
    """Where a manager keeps the pages of one buffer.

    Block ``b`` of a request is at ``base + indices[b] * step``, where ``indices`` are the request's
    base page indices of ``layer_group``; each page is ``page_bytes`` long.
    """

    layer_group: int
    base: int
    step: int
    page_bytes: int


def page_affine(manager, layer_id: Any, role: Any) -> PageAffine:
    """The ``PageAffine`` of the buffer ``(layer_id, role)`` of a V2 cache manager.

    A base page index names a slot of the buffer's pool, and a slot holds one block of every buffer
    of the pool at a fixed offset: the page is at the pool base plus that offset plus the index
    times the slot size, which is the converter's scale times the page stride.
    """
    impl = manager.impl
    converter = impl.get_page_index_converter(layer_id, role)
    if converter.expansion != 1:
        raise NotImplementedError("a buffer whose block spans several pages is not staged yet")
    stride = int(impl.get_page_stride(layer_id, role))
    return PageAffine(
        layer_group=int(manager.layer_to_pool_mapping_dict[layer_id]),
        base=int(impl.get_mem_pool_base_address(layer_id, role, PageIndexMode.SHARED)),
        step=int(converter.scale) * stride,
        page_bytes=stride,
    )


def base_page_indices(manager, request_id: int, layer_group: int) -> np.ndarray:
    """The base page index of every block of a request in a layer group (``-1``: no page).

    A read-only view of the table the request's cache keeps, valid until its next lifecycle
    operation.
    """
    return np.frombuffer(
        manager.kv_cache_map[request_id].get_base_page_indices(layer_group), dtype=np.int32
    )


# ---- the working cache ----------------------------------------------------------------------------


def working_slot_counts(
    config: Any, *, max_batch_size: int, max_num_tokens: int, staging_tokens: int
) -> list[list[int]]:
    """Pages of every life cycle of a working cache, as ``lifecycle_slot_counts`` rows.

    Enough for any batch the scheduler admits to one rank: at most ``max_batch_size`` requests
    whose new tokens add up to at most ``max_num_tokens`` and whose cached and new tokens add up to
    at most ``staging_tokens``. A full-attention life cycle holds the blocks of all those tokens,
    plus a partial block per request. A sliding-window one holds per request the blocks of its
    window and of its new tokens, at most ``(chunk + window) / block + 2``. The working cache has
    only the GPU tier, so there is one row.
    """
    tokens_per_block = int(config.tokens_per_block)
    if len(config.cache_tiers) != 1:
        raise ValueError("the working cache has only the GPU tier")
    counts = []
    for layout in lifecycle_layouts(config):
        key = layout.key
        _check_staged(key)
        if key.window_size:
            pages = 2 * max_batch_size + math.ceil(
                (max_num_tokens + max_batch_size * key.window_size) / tokens_per_block
            )
        else:
            pages = max_batch_size + math.ceil(staging_tokens / tokens_per_block)
        counts.append(pages)
    return [counts]


def working_cache_bytes(config: Any) -> int:
    """The bytes the pages of a manager with fixed page counts take on the GPU."""
    counts = config.lifecycle_slot_counts
    if counts is None:
        raise ValueError("the manager has no fixed page counts")
    return sum(
        count * layout.slot_bytes for count, layout in zip(counts[0], lifecycle_layouts(config))
    )


class WorkingCacheSizing:
    """The page counts of a working cache and the tokens one rank may stage in an iteration.

    It is passed as ``lifecycle_slot_counts`` to the constructor of the working cache, which calls
    it with the final cache config of all layers. Unless ``staging_tokens`` is given, the staged
    tokens are every request of a batch at ``max_seq_len``, unless the pages of the life cycles
    that keep the whole history would then take more than ``history_bytes`` in the working cache
    and in the ``arena_copies`` message buffers that hold one layer's pages each. They never go
    below a request of ``max_seq_len`` next to a full chunk, as far as the batch holds that much,
    nor below one full chunk. After the call ``staging_tokens`` holds the result, which the
    scheduler enforces.
    """

    def __init__(
        self,
        *,
        max_batch_size: int,
        max_num_tokens: int,
        max_seq_len: int,
        staging_tokens: int | None = None,
        history_bytes: int = DEFAULT_STAGING_HISTORY_BYTES,
        arena_copies: int = 0,
    ) -> None:
        self.max_batch_size = max_batch_size
        self.max_num_tokens = max_num_tokens
        self.max_seq_len = max_seq_len
        self.staging_tokens = staging_tokens
        self.history_bytes = history_bytes
        self.arena_copies = arena_copies

    def default_staging_tokens(self, config: Any) -> int:
        """The staged tokens when none were given, for the layers of ``config``."""
        tokens_per_block = int(config.tokens_per_block)
        full_batch = self.max_batch_size * self.max_seq_len
        least = max(min(full_batch, self.max_seq_len + self.max_num_tokens), self.max_num_tokens)
        # Bytes per token of the full-history buffers: in the working cache all layers, in a
        # message buffer the largest layer.
        cache_bytes = 0
        largest_layer = 0
        for layer in config.layers:
            key = lifecycle_key(layer, tokens_per_block)
            if key.window_size or key.is_ssm:
                continue
            layer_bytes = sum(int(buffer.size) for buffer in layer.buffers)
            cache_bytes += layer_bytes
            largest_layer = max(largest_layer, layer_bytes)
        block_bytes = cache_bytes + self.arena_copies * largest_layer
        if block_bytes == 0:
            return full_batch
        within = self.history_bytes // block_bytes * tokens_per_block
        return max(least, min(full_batch, within))

    def __call__(self, config: Any) -> list[list[int]]:
        if self.staging_tokens is None:
            self.staging_tokens = self.default_staging_tokens(config)
        return working_slot_counts(
            config,
            max_batch_size=self.max_batch_size,
            max_num_tokens=self.max_num_tokens,
            staging_tokens=self.staging_tokens,
        )


class DkvWorkingSession:
    """The caches of the requests one rank computes in an iteration, in the working cache.

    ``begin`` opens a cache for every request with the tokens it has cached and the tokens it
    computes, so that the cache holds a page for every block the attention reads or writes;
    ``end`` releases them all once the data plane is done with the iteration. ``begin_batch`` is
    what the attention metadata calls before every forward pass (through the manager's
    ``begin_staged_batch``): it closes what an earlier pass left open, opens the batch and hands it
    to the data plane.
    """

    def __init__(self, manager, streamer=None) -> None:
        self.manager = manager
        self.streamer = streamer
        self._open: list[int] = []

    @property
    def request_ids(self) -> tuple[int, ...]:
        return tuple(self._open)

    def begin_batch(
        self, request_ids: Sequence[int], history: Sequence[int], new: Sequence[int]
    ) -> None:
        """Open the batch of a forward pass and tell the data plane its requests are placed."""
        if not (len(request_ids) == len(history) == len(new)):
            raise ValueError("request_ids, history and new differ in length")
        self.end()
        self.begin(list(zip(request_ids, history, new)))
        if self.streamer is not None:
            self.streamer.begin_iteration(request_ids, history, new)

    def begin(self, requests: Sequence[tuple[int, int, int]]) -> None:
        """Open a cache for every ``(request id, cached tokens, new tokens)``.

        Raises:
            WorkingCacheOverflow: The pages of a request do not fit; nothing stays open.
        """
        if self._open:
            raise RuntimeError("the working caches of the previous batch are still open")
        for request_id, history, chunk in requests:
            if not self.manager.dkv_open_cache(request_id, history, history + chunk):
                self.end()
                raise WorkingCacheOverflow(
                    f"request {request_id} ({history} cached and {chunk} new tokens) does not fit "
                    "the working cache"
                )
            self._open.append(request_id)

    def end(self) -> None:
        """Release the caches of the batch."""
        for request_id in self._open:
            self.manager.dkv_close_cache(request_id)
        self._open.clear()
