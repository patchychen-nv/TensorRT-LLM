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
"""What the DKV layer split derives from any V2 cache manager, on cache configs without a GPU.

The buffers of every layer, the blocks a request moves, the address of a page and the size of the
working cache come from the layer configs and the address arithmetic a V2 manager publishes. The
numbers here are computed by hand; ``test_dkv_working_cache`` holds the same derivation against a
real DeepSeek-V4 manager on a GPU.
"""

from types import SimpleNamespace

import pytest

from tensorrt_llm._torch.pyexecutor.dkv_types import LifecycleKey
from tensorrt_llm._torch.pyexecutor.dkv_working import (
    DEFAULT_STAGING_HISTORY_BYTES,
    BufferCostModel,
    DkvBuffer,
    DkvWorkingSession,
    PageAffine,
    WorkingCacheOverflow,
    WorkingCacheSizing,
    buffers_by_layer,
    fetch_blocks,
    manager_buffers,
    page_affine,
    staged_blocks,
    working_cache_bytes,
    working_slot_counts,
    writeback_blocks,
)
from tensorrt_llm.runtime.kv_cache_manager_v2 import (
    AttentionLayerConfig,
    BufferConfig,
    PageIndexMode,
    SsmLayerConfig,
)

pytestmark = pytest.mark.cpu_only

_BLOCK = 128
_FULL_HISTORY = LifecycleKey(is_ssm=False, window_size=0, num_sink_blocks=0, is_sparse=False)
_ONE_BLOCK_WINDOW = _FULL_HISTORY._replace(window_size=_BLOCK)
_EIGHT_TOKEN_WINDOW = _FULL_HISTORY._replace(window_size=8)


# Which blocks move


@pytest.mark.parametrize(
    ("key", "history", "chunk", "staged", "fetched", "written"),
    [
        # Everything cached is fetched and every block the chunk touches goes back.
        (_FULL_HISTORY, 0, 300, (0, 3), (0, 0), (0, 3)),
        (_FULL_HISTORY, 127, 1, (0, 1), (0, 1), (0, 1)),
        (_FULL_HISTORY, 700, 300, (0, 8), (0, 6), (5, 3)),
        (_FULL_HISTORY, 1024, 128, (0, 9), (0, 8), (8, 1)),
        # The in-flight token keeps the window one token longer, and a block that leaves the
        # window during the chunk does not go back.
        (_ONE_BLOCK_WINDOW, 0, 300, (0, 3), (0, 0), (1, 2)),
        (_ONE_BLOCK_WINDOW, 127, 1, (0, 1), (0, 1), (0, 1)),
        (_ONE_BLOCK_WINDOW, 700, 300, (4, 4), (4, 2), (6, 2)),
        (_ONE_BLOCK_WINDOW, 1024, 128, (7, 2), (7, 1), (8, 1)),
        (_ONE_BLOCK_WINDOW, 300, 400, (1, 5), (1, 2), (4, 2)),
        # A history that ends on a block boundary has no partly cached block.
        (_ONE_BLOCK_WINDOW, 256, 128, (1, 2), (1, 1), (2, 1)),
        (_FULL_HISTORY, 256, 128, (0, 3), (0, 2), (2, 1)),
        # A window of a few tokens, as a compressor state keeps.
        (_EIGHT_TOKEN_WINDOW, 300, 1, (2, 1), (2, 1), (2, 1)),
        (_EIGHT_TOKEN_WINDOW, 255, 2, (1, 2), (1, 1), (1, 2)),
        # A long chunk leaves one block of a window behind, and every block of the history.
        (_ONE_BLOCK_WINDOW, 0, 8192, (0, 64), (0, 0), (63, 1)),
        (_EIGHT_TOKEN_WINDOW, 0, 8192, (0, 64), (0, 0), (63, 1)),
        (_FULL_HISTORY, 0, 8192, (0, 64), (0, 0), (0, 64)),
    ],
)
def test_the_blocks_a_request_stages_fetches_and_writes_back(
    key, history, chunk, staged, fetched, written
) -> None:
    """``(first block, count)`` computed by hand for blocks of 128 tokens."""
    assert staged_blocks(key, history, chunk, _BLOCK) == staged
    assert fetch_blocks(key, history, chunk, _BLOCK) == fetched
    assert writeback_blocks(key, history, chunk, _BLOCK) == written


_KEYS = {
    "whole_history": _FULL_HISTORY,
    "window_of_8": _EIGHT_TOKEN_WINDOW,
    "window_of_a_block": _ONE_BLOCK_WINDOW,
    "window_of_300": _FULL_HISTORY._replace(window_size=300),
}


def _blocks(span: tuple[int, int]) -> set[int]:
    first, count = span
    return set(range(first, first + count))


@pytest.mark.parametrize("key", list(_KEYS.values()), ids=list(_KEYS))
def test_every_staged_block_is_fetched_or_written_back_unless_it_leaves_the_window(key) -> None:
    for history in range(0, 700, 37):
        for chunk in (1, 2, 127, 128, 129, 400):
            staged = _blocks(staged_blocks(key, history, chunk, _BLOCK))
            fetched = _blocks(fetch_blocks(key, history, chunk, _BLOCK))
            written = _blocks(writeback_blocks(key, history, chunk, _BLOCK))
            assert fetched <= staged and written <= staged
            if not key.window_size:
                assert fetched | written == staged, (history, chunk)
            # Only blocks with cached tokens are fetched; only blocks with new tokens return.
            assert all(block * _BLOCK < history for block in fetched)
            assert all((block + 1) * _BLOCK > history for block in written)
            # What stays out of both is released by the cache manager after the chunk.
            alive_from = max(0, history + chunk + 1 - key.window_size) // _BLOCK
            assert all(block < alive_from for block in staged - fetched - written)


@pytest.mark.parametrize("key", list(_KEYS.values()), ids=list(_KEYS))
@pytest.mark.parametrize(
    "chunks", [(128,) * 12, (1,) * 300, (300, 7, 129, 1, 512, 64), (500, 500, 500), (5000, 3, 5000)]
)
def test_every_block_a_chunk_fetches_was_written_back_by_an_earlier_one(key, chunks) -> None:
    written: set[int] = set()
    history = 0
    for chunk in chunks:
        fetched = _blocks(fetch_blocks(key, history, chunk, _BLOCK))
        assert fetched <= written, (history, chunk, sorted(fetched - written))
        written |= _blocks(writeback_blocks(key, history, chunk, _BLOCK))
        history += chunk


@pytest.mark.parametrize(
    "key", [_FULL_HISTORY._replace(is_ssm=True), _ONE_BLOCK_WINDOW._replace(num_sink_blocks=1)]
)
def test_recurrent_states_and_attention_sinks_are_not_staged(key) -> None:
    with pytest.raises(NotImplementedError):
        staged_blocks(key, 700, 300, _BLOCK)


@pytest.mark.parametrize(("history", "chunk"), [(-1, 5), (0, 0)])
def test_a_request_stages_a_chunk_after_its_history(history: int, chunk: int) -> None:
    with pytest.raises(ValueError, match="need history >= 0 and chunk >= 1"):
        staged_blocks(_FULL_HISTORY, history, chunk, _BLOCK)


# The buffers of the layers a manager holds


def _manager(layers, *, ignored=(), model_layer=None):
    """A V2 manager as ``manager_buffers`` reads it: layer configs and the model layer of each."""
    return SimpleNamespace(
        kv_cache_manager_py_config=SimpleNamespace(layers=layers, tokens_per_block=_BLOCK),
        get_disagg_ignored_roles=lambda: ignored,
        dkv_model_layer_of=model_layer or (lambda layer_id, role: layer_id),
    )


def test_the_buffers_of_a_manager_are_named_by_model_layer_and_role() -> None:
    layers = [
        AttentionLayerConfig(
            layer_id=0,
            buffers=[BufferConfig(role="kv", size=1024), BufferConfig(role="padding", size=16)],
            sliding_window_size=None,
        ),
        # A buffer with smaller blocks of its own holds two of them per block of the manager.
        AttentionLayerConfig(
            layer_id=1,
            buffers=[BufferConfig(role="kv", size=96, tokens_per_block_override=64)],
            sliding_window_size=256,
        ),
    ]
    # Two V2 layers that are the buffers of one model layer, as the DeepSeek-V4 manager has them.
    buffers = manager_buffers(
        _manager(layers, ignored=("padding",), model_layer=lambda layer_id, role: 7 + layer_id)
    )
    window = _FULL_HISTORY._replace(window_size=256)
    assert {name: entry.buffer for name, entry in buffers.items()} == {
        (7, "kv"): DkvBuffer(7, "kv", _FULL_HISTORY, 1024),
        (8, "kv"): DkvBuffer(8, "kv", window, 192),
    }
    assert [(entry.layer_id, entry.role) for entry in buffers.values()] == [(0, "kv"), (1, "kv")]


def test_two_buffers_of_a_model_layer_with_one_role_are_refused() -> None:
    layers = [
        AttentionLayerConfig(layer_id=layer_id, buffers=[BufferConfig(role="kv", size=64)])
        for layer_id in range(2)
    ]
    with pytest.raises(ValueError, match="two buffers of model layer 3 have role kv"):
        manager_buffers(_manager(layers, model_layer=lambda layer_id, role: 3))


def test_every_rank_lists_the_buffers_of_a_layer_by_role() -> None:
    state, kv, compressed = (
        DkvBuffer(2, "state", _ONE_BLOCK_WINDOW, 64),
        DkvBuffer(2, "kv", _ONE_BLOCK_WINDOW, 512),
        DkvBuffer(0, "compressed", _FULL_HISTORY, 32),
    )
    assert buffers_by_layer([state, kv, compressed]) == {0: (compressed,), 2: (kv, state)}


def test_the_cost_model_gives_the_pages_of_the_block_rules() -> None:
    swa = DkvBuffer(0, "swa", _ONE_BLOCK_WINDOW, 512)
    compressed = DkvBuffer(0, "compressed", _FULL_HISTORY, 32)
    costs = BufferCostModel([swa, compressed], _BLOCK)
    assert costs.buffers_of(0) == ("compressed", "swa")
    assert costs.buffer(0, "swa") is swa
    cost = costs.page_cost(0, "swa", 700, 300)
    assert (cost.page_bytes, cost.fetch_pages, cost.writeback_pages) == (512, 2, 2)
    cost = costs.page_cost(0, "compressed", 700, 300)
    assert (cost.page_bytes, cost.fetch_pages, cost.writeback_pages) == (32, 6, 3)


def test_layers_share_a_class_when_their_buffers_have_the_same_life_cycles_and_pages() -> None:
    """The plan asks for the costs of a request once per class of layers."""

    def layer(index: int, window=_ONE_BLOCK_WINDOW, page_bytes: int = 512) -> list[DkvBuffer]:
        return [
            DkvBuffer(index, "swa", window, page_bytes),
            DkvBuffer(index, "compressed", _FULL_HISTORY, 32),
        ]

    costs = BufferCostModel(
        layer(0) + layer(1) + layer(2, window=_FULL_HISTORY) + layer(3, page_bytes=1024), _BLOCK
    )
    assert costs.layer_class(0) == costs.layer_class(1)
    assert len({costs.layer_class(index) for index in (0, 2, 3)}) == 3
    for history, chunk in ((0, 300), (700, 300), (5000, 1)):
        for role in costs.buffers_of(0):
            assert costs.page_cost(0, role, history, chunk) == costs.page_cost(
                1, role, history, chunk
            )


# Where a page is


def _addressed(converter):
    asked = []

    class Impl:
        def get_page_index_converter(self, layer_id, role):
            return converter

        def get_page_stride(self, layer_id, role):
            return 256

        def get_mem_pool_base_address(self, layer_id, role, mode):
            asked.append(mode)
            return 0x10000 + 512

    manager = SimpleNamespace(impl=Impl(), layer_to_pool_mapping_dict={4: 2})
    return manager, asked


def test_a_page_is_at_the_shared_base_plus_the_index_times_the_slot() -> None:
    """A slot of the pool holds three pages of this size; the buffer's page is the third."""
    manager, asked = _addressed(SimpleNamespace(expansion=1, scale=3))
    assert page_affine(manager, 4, "kv") == PageAffine(
        layer_group=2, base=0x10000 + 512, step=768, page_bytes=256
    )
    assert asked == [PageIndexMode.SHARED]


def test_a_block_that_spans_several_pages_is_not_staged_yet() -> None:
    manager, _ = _addressed(SimpleNamespace(expansion=2, scale=1))
    with pytest.raises(NotImplementedError, match="spans several pages"):
        page_affine(manager, 4, "kv")


# The size of the working cache


def _config(*, tiers: int = 1):
    """Two layers that keep the whole history, 1024 and 512 + 256 bytes per block, and a
    sliding-window layer of 2048 bytes per block."""
    return SimpleNamespace(
        tokens_per_block=_BLOCK,
        cache_tiers=[object()] * tiers,
        layers=[
            AttentionLayerConfig(layer_id=0, buffers=[BufferConfig(role="kv", size=1024)]),
            AttentionLayerConfig(
                layer_id=1,
                buffers=[BufferConfig(role="kv", size=512), BufferConfig(role="index", size=256)],
            ),
            AttentionLayerConfig(
                layer_id=2,
                buffers=[BufferConfig(role="swa", size=2048)],
                sliding_window_size=_BLOCK,
            ),
        ],
    )


@pytest.mark.parametrize(
    ("batch", "max_seq_len", "max_num_tokens", "history_bytes", "arena_copies", "expected"),
    [
        # Every request of the batch at max_seq_len.
        (4, 1000, 300, DEFAULT_STAGING_HISTORY_BYTES, 0, 4000),
        # A batch of one request.
        (1, 1000, 300, DEFAULT_STAGING_HISTORY_BYTES, 0, 1000),
        # At least one full chunk.
        (1, 100, 300, DEFAULT_STAGING_HISTORY_BYTES, 0, 300),
        # As many blocks as the budget holds: 1792 bytes a block in the cache, then another 1024
        # for each message buffer, which holds the largest layer.
        (8, 4096, 512, 1792 * 60, 0, 60 * 128),
        (8, 4096, 512, 3840 * 60, 2, 60 * 128),
        # But never less than a request of max_seq_len next to a full chunk.
        (8, 4096, 512, 3840 * 28, 2, 4096 + 512),
    ],
)
def test_the_default_staged_tokens(
    batch, max_seq_len, max_num_tokens, history_bytes, arena_copies, expected
) -> None:
    sizing = WorkingCacheSizing(
        max_batch_size=batch,
        max_num_tokens=max_num_tokens,
        max_seq_len=max_seq_len,
        history_bytes=history_bytes,
        arena_copies=arena_copies,
    )
    assert sizing.default_staging_tokens(_config()) == expected


def test_a_working_cache_of_sliding_windows_only_stages_the_whole_batch() -> None:
    config = _config()
    config.layers = config.layers[2:]
    sizing = WorkingCacheSizing(
        max_batch_size=8, max_num_tokens=512, max_seq_len=1 << 20, history_bytes=1
    )
    assert sizing.default_staging_tokens(config) == 8 << 20


def test_the_page_counts_hold_any_batch_the_scheduler_admits() -> None:
    """Life cycles in the order the layers first use them: the whole history, then the window."""
    # The whole history: a partial block for each of the 4 requests and the 32 blocks of 4000
    # staged tokens. The window: two partial blocks for each request and the 7 blocks of 300 new
    # tokens next to four windows of 128 tokens.
    assert working_slot_counts(
        _config(), max_batch_size=4, max_num_tokens=300, staging_tokens=4000
    ) == [[36, 15]]
    sizing = WorkingCacheSizing(max_batch_size=4, max_num_tokens=300, max_seq_len=1000)
    assert sizing(_config()) == [[36, 15]]
    assert sizing.staging_tokens == 4000
    given = WorkingCacheSizing(
        max_batch_size=4, max_num_tokens=300, max_seq_len=1000, staging_tokens=1280
    )
    assert given(_config()) == [[14, 15]] and given.staging_tokens == 1280


def test_the_pages_of_a_working_cache_take_their_counts_times_the_slots() -> None:
    """A slot of the life cycle that keeps the whole history holds the 1024 + 512 + 256 bytes of a
    block of its two layers, one of the window the 2048 of its layer."""
    config = _config()
    config.lifecycle_slot_counts = [[36, 15]]
    assert working_cache_bytes(config) == 36 * 1792 + 15 * 2048
    config.lifecycle_slot_counts = None
    with pytest.raises(ValueError, match="no fixed page counts"):
        working_cache_bytes(config)


def test_a_working_cache_has_one_tier_and_no_recurrent_state() -> None:
    with pytest.raises(ValueError, match="only the GPU tier"):
        working_slot_counts(
            _config(tiers=2), max_batch_size=1, max_num_tokens=8, staging_tokens=128
        )
    config = _config()
    config.layers = [SsmLayerConfig(layer_id=0, buffers=[BufferConfig(role="state", size=64)])]
    with pytest.raises(NotImplementedError, match="recurrent-state"):
        working_slot_counts(config, max_batch_size=1, max_num_tokens=8, staging_tokens=128)


# The caches of one iteration


class _Manager:
    """The working cache as the session drives it; ``fits`` names the requests that fit."""

    def __init__(self, fits=None) -> None:
        self.fits = fits
        self.calls: list[tuple] = []

    def dkv_open_cache(self, request_id: int, history: int, capacity: int) -> bool:
        self.calls.append(("open", request_id, history, capacity))
        return self.fits is None or request_id in self.fits

    def dkv_close_cache(self, request_id: int) -> None:
        self.calls.append(("close", request_id))


def test_a_session_opens_the_requests_of_a_batch_and_closes_them() -> None:
    manager = _Manager()
    session = DkvWorkingSession(manager)
    session.begin([(3, 700, 300), (1, 0, 128)])
    assert session.request_ids == (3, 1)
    # The capacity of a request covers its cached and its new tokens.
    assert manager.calls == [("open", 3, 700, 1000), ("open", 1, 0, 128)]
    with pytest.raises(RuntimeError, match="still open"):
        session.begin([(4, 0, 1)])
    session.end()
    assert manager.calls[2:] == [("close", 3), ("close", 1)]
    assert session.request_ids == ()


def test_a_batch_that_does_not_fit_leaves_nothing_open() -> None:
    manager = _Manager(fits={3})
    session = DkvWorkingSession(manager)
    with pytest.raises(WorkingCacheOverflow, match="request 1"):
        session.begin([(3, 700, 300), (1, 0, 128), (2, 0, 1)])
    assert manager.calls == [("open", 3, 700, 1000), ("open", 1, 0, 128), ("close", 3)]
    assert session.request_ids == ()


def test_the_batch_of_a_forward_pass_replaces_the_last_one_and_goes_to_the_data_plane() -> None:
    manager = _Manager()
    begun = []
    session = DkvWorkingSession(
        manager, SimpleNamespace(begin_iteration=lambda *batch: begun.append(batch))
    )
    session.begin_batch([5], [0], [64])
    session.begin_batch([6, 7], [128, 0], [1, 9])
    assert manager.calls == [
        ("open", 5, 0, 64),
        ("close", 5),
        ("open", 6, 128, 129),
        ("open", 7, 0, 9),
    ]
    assert begun == [([5], [0], [64]), ([6, 7], [128, 0], [1, 9])]
    with pytest.raises(ValueError, match="differ in length"):
        session.begin_batch([8], [0, 1], [2])
