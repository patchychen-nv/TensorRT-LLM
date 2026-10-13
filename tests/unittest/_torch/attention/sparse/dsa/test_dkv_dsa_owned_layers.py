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
"""The DSA cache manager of GLM-5.3 holding the layers one rank of a DKV group owns (layer split).

A group of ranks is simulated in one process: one manager per rank, each built with the layers the
DKV ownership table gives that rank, and with page counts that the group agrees on through
``solve_lifecycle_slot_counts`` over an allgather that returns the payloads of every rank. The
geometry is GLM-5.3's: 78 MLA layers of 576 latent elements per token, and an indexer K cache on
the "full" indexer layers only (every fourth layer from layer 2, plus layers 0 and 1).
"""

import functools
from collections.abc import Iterator
from types import SimpleNamespace

import pytest
from utils.util import skip_pre_blackwell

from tensorrt_llm._torch.attention.backends.sparse.dsa import DSACacheManagerV2
from tensorrt_llm._torch.pyexecutor.dkv import compute_ownership, owned_layers, validate_ownership
from tensorrt_llm._torch.pyexecutor.dkv_types import LifecycleKey
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import Role
from tensorrt_llm._torch.pyexecutor.kv_cache.lifecycle_slot_counts import (
    solve_lifecycle_slot_counts,
)
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest
from tensorrt_llm.bindings import DataType, SamplingConfig
from tensorrt_llm.bindings.internal.batch_manager import CacheType
from tensorrt_llm.llmapi.llm_args import DeepSeekSparseAttentionConfig, KvCacheConfig
from tensorrt_llm.mapping import Mapping

pytestmark = skip_pre_blackwell

# The fields of GLM-5.3's config.json that decide the cache layout.
GLM5_CONFIG = SimpleNamespace(
    architectures=["GlmMoeDsaForCausalLM"],
    num_hidden_layers=78,
    kv_lora_rank=512,
    qk_rope_head_dim=64,
    index_n_heads=32,
    index_head_dim=128,
    index_topk=2048,
    index_topk_freq=4,
    index_skip_topk_offset=3,
    index_topk_pattern=None,
    indexer_rope_interleave=True,
)
NUM_LAYERS = GLM5_CONFIG.num_hidden_layers
HEAD_DIM = GLM5_CONFIG.kv_lora_rank + GLM5_CONFIG.qk_rope_head_dim
FULL_INDEXER_LAYERS = [0, 1, 2] + list(range(6, NUM_LAYERS, 4))
TOKENS_PER_BLOCK = 64
PROMPT_TOKENS = 700
GROUP_SIZE = 4


def _sparse_config() -> DeepSeekSparseAttentionConfig:
    return DeepSeekSparseAttentionConfig(
        index_n_heads=GLM5_CONFIG.index_n_heads,
        index_head_dim=GLM5_CONFIG.index_head_dim,
        index_topk=GLM5_CONFIG.index_topk,
        indexer_rope_interleave=GLM5_CONFIG.indexer_rope_interleave,
        indexer_k_dtype="fp8",
    )


def _manager(**kwargs) -> DSACacheManagerV2:
    return DSACacheManagerV2(
        KvCacheConfig(
            enable_block_reuse=False,
            max_tokens=4096,
            event_buffer_max_size=0,
            host_cache_size=0,
        ),
        CacheType.SELFKONLY,
        num_layers=NUM_LAYERS,
        num_kv_heads=1,
        head_dim=HEAD_DIM,
        tokens_per_block=TOKENS_PER_BLOCK,
        max_seq_len=4096,
        max_batch_size=4,
        mapping=Mapping(world_size=1, rank=0, tp_size=1, pp_size=1),
        dtype=DataType.BF16,
        max_num_tokens=2048,
        sparse_attention_config=_sparse_config(),
        pretrained_config=GLM5_CONFIG,
        **kwargs,
    )


def _request(request_id: int, prompt_tokens: int) -> LlmRequest:
    return LlmRequest(
        request_id=request_id,
        max_new_tokens=1,
        input_tokens=list(range(prompt_tokens)),
        sampling_config=SamplingConfig(),
        is_streaming=False,
    )


class _GroupAllgather:
    """An allgather over the ranks of a group whose managers are built one after the other.

    The first round records what each rank sends, alone. The second round answers every rank
    with what all ranks sent, in rank order, which is what a real allgather returns.
    """

    def __init__(self) -> None:
        self.sent: dict[int, object] = {}

    def for_rank(self, rank: int, *, recording: bool):
        def allgather(payload):
            if recording:
                self.sent[rank] = payload
                return [payload]
            return [self.sent[peer] for peer in sorted(self.sent)]

        return allgather


@pytest.fixture(scope="module")
def group() -> Iterator[tuple[DSACacheManagerV2, list[DSACacheManagerV2]]]:
    owners = compute_ownership(NUM_LAYERS, GROUP_SIZE)
    exchange = _GroupAllgather()
    for rank in range(GROUP_SIZE):
        _manager(
            owned_layers=owned_layers(owners, rank),
            lifecycle_slot_counts=functools.partial(
                solve_lifecycle_slot_counts, allgather=exchange.for_rank(rank, recording=True)
            ),
        ).shutdown()
    ranks = [
        _manager(
            owned_layers=owned_layers(owners, rank),
            lifecycle_slot_counts=functools.partial(
                solve_lifecycle_slot_counts, allgather=exchange.for_rank(rank, recording=False)
            ),
        )
        for rank in range(GROUP_SIZE)
    ]
    everything = _manager()
    try:
        yield everything, ranks
    finally:
        everything.shutdown()
        for manager in ranks:
            manager.shutdown()


def test_every_rank_owns_a_contiguous_range_with_an_indexer_layer() -> None:
    owners = compute_ownership(NUM_LAYERS, GROUP_SIZE)
    # DSA keeps the latent KV and the indexer K of every layer for the whole history, in one life
    # cycle, so every rank holds every life cycle whatever layers it owns.
    validate_ownership(owners, [{LifecycleKey(False, 0, 0, False)}] * NUM_LAYERS, GROUP_SIZE)
    for rank in range(GROUP_SIZE):
        layers = owned_layers(owners, rank)
        assert layers == tuple(range(layers[0], layers[-1] + 1))
        assert set(layers) & set(FULL_INDEXER_LAYERS), rank


def test_a_rank_holds_its_layers_and_the_indexer_caches_among_them(group) -> None:
    everything, ranks = group
    owners = compute_ownership(NUM_LAYERS, GROUP_SIZE)
    assert everything.indexer_k_cache_local_layer_mask == [
        layer in FULL_INDEXER_LAYERS for layer in range(NUM_LAYERS)
    ]
    held = []
    for rank, manager in enumerate(ranks):
        layers = list(owned_layers(owners, rank))
        assert manager.pp_layers == layers
        assert manager.num_local_layers == len(layers)
        assert manager.indexer_k_cache_local_layer_mask == [
            layer in FULL_INDEXER_LAYERS for layer in layers
        ]
        for layer in layers:
            if layer in FULL_INDEXER_LAYERS:
                assert (
                    manager.get_indexer_k_cache_buffers(layer).shape[1:]
                    == everything.get_indexer_k_cache_buffers(layer).shape[1:]
                )
            else:
                with pytest.raises(AssertionError, match="shared-indexer layer"):
                    manager.get_indexer_k_cache_buffers(layer)
        # The attention op reads one pool of the latent KV of the local layers.
        pool = manager.get_unique_primary_pool()
        assert pool.shape[1] == len(layers)
        assert pool.shape[2:] == everything.get_unique_primary_pool().shape[2:]
        held.extend(layers)
    assert sorted(held) == list(range(NUM_LAYERS))


def test_the_ranks_configure_the_same_page_counts(group) -> None:
    _, ranks = group
    counts = {tuple(manager.blocks_in_primary_pool for manager in ranks)}
    assert len(counts) == 1 and next(iter(counts))[0] > 0
    # The ranks hold different numbers of layers, so the agreed page count is what a rank's
    # budget allows; its bytes differ.
    assert len({manager.num_local_layers for manager in ranks}) > 1


def test_a_request_takes_the_same_pages_on_every_rank(group) -> None:
    """The replicated life cycle places a request at the same pages of every rank.

    Under the layer split a rank moves the pages of its layers to the compute rank by page index,
    so the indices must agree, for the latent KV and for the indexer K cache.
    """
    _, ranks = group
    requests = []
    try:
        for manager in ranks:
            for request_id, prompt in ((7, PROMPT_TOKENS), (8, 3 * TOKENS_PER_BLOCK + 1)):
                request = _request(request_id, prompt)
                requests.append((manager, request))
                assert manager.prepare_context(request)
                assert manager.resize_context(request, prompt)
        for request_id, prompt in ((7, PROMPT_TOKENS), (8, 3 * TOKENS_PER_BLOCK + 1)):
            base = {
                tuple(manager.get_batch_cache_indices([request_id], raw_indices=True)[0])
                for manager in ranks
            }
            assert len(base) == 1, request_id
            assert sum(1 for page in next(iter(base)) if page >= 0) == -(
                -prompt // TOKENS_PER_BLOCK
            )
        # The indexer K pages of every layer follow the base page by one uniform scale, so the
        # pages a rank sends are found from the base indices alone.
        for manager in ranks:
            for local_layer, layer in enumerate(manager.pp_layers):
                if layer in FULL_INDEXER_LAYERS:
                    converter = manager.impl.get_page_index_converter(local_layer, Role.INDEX_KEY)
                    assert int(converter.scale) == manager.indexer_k_cache_page_scale
    finally:
        for manager, request in requests:
            manager.free_resources(request)
