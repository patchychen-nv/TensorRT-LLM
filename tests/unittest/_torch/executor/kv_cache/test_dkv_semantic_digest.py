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
"""The DKV digest and fingerprints of managers that hold different layers, on real native managers.

Under the layer-split layout every rank holds only the layers it owns, so the life cycle ids, the
pool group numbers and the slot bytes of the same life cycle differ from rank to rank. The pages of
a life cycle are the same on every rank (the page counts are fixed per life cycle), and the digest
and the fingerprints name them by the semantic key of the life cycle.

The managers below have the windows of layers 0 to 3 of one model (no window, 128, no window, 64
tokens over a 256 token context), the first with all four layers and the second with the layers in
another order and without the third, so its life cycle ids are not those of the first.
"""

import pytest
import torch
from _torch.executor.dkv_test_utils import make_request

from tensorrt_llm._torch.pyexecutor.dkv import DkvControlDigest, DkvControlPayload, sync_dkv_control
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
from tensorrt_llm._torch.pyexecutor.kv_cache.lifecycle_slot_counts import (
    LifecycleKey,
    lifecycle_layouts,
)
from tensorrt_llm.bindings import DataType
from tensorrt_llm.bindings.internal.batch_manager import CacheType
from tensorrt_llm.llmapi.llm_args import KvCacheConfig
from tensorrt_llm.mapping import Mapping

MAX_SEQ_LEN = 256
TOKENS_PER_BLOCK = 8
FULL, WINDOW_128, WINDOW_64 = (LifecycleKey(False, window, 0, False) for window in (0, 128, 64))
# Hot and host pages of every life cycle.
PAGES = {FULL: (90, 120), WINDOW_128: (40, 60), WINDOW_64: (24, 30)}


@pytest.fixture(autouse=True)
def cuda_cleanup():
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    torch.cuda.init()
    yield
    torch.cuda.empty_cache()


def counts_of(config) -> list[list[int]]:
    """The rows of page counts of a config, in the life cycle order of its layers."""
    layouts = lifecycle_layouts(config)
    return [[PAGES[layout.key][level] for layout in layouts] for level in range(2)]


def make_manager(windows: list[int], **kwargs) -> KVCacheManagerV2:
    return KVCacheManagerV2(
        KvCacheConfig(
            max_gpu_total_bytes=16 << 20,
            host_cache_size=16 << 20,
            max_attention_window=windows,
            enable_block_reuse=False,
        ),
        CacheType.SELF,
        num_layers=len(windows),
        num_kv_heads=2,
        head_dim=64,
        tokens_per_block=TOKENS_PER_BLOCK,
        max_seq_len=MAX_SEQ_LEN,
        max_batch_size=2,
        mapping=Mapping(world_size=1, rank=0, tp_size=1, pp_size=1),
        dtype=DataType.HALF,
        vocab_size=16,
        **kwargs,
    )


@pytest.fixture
def managers():
    first = make_manager([MAX_SEQ_LEN, 128, MAX_SEQ_LEN, 64], lifecycle_slot_counts=counts_of)
    second = make_manager([128, MAX_SEQ_LEN, 64], lifecycle_slot_counts=counts_of)
    try:
        yield first, second
    finally:
        first.shutdown()
        second.shutdown()


def occupy(manager: KVCacheManagerV2, request_id: int, tokens: int) -> None:
    request = make_request(request_id, prompt_len=tokens)
    assert manager.prepare_context(request)
    request.context_chunk_size = tokens
    assert manager.resize_context(request, tokens)


def test_the_managers_do_not_number_their_life_cycles_alike(managers) -> None:
    first, second = managers
    assert [layout.key for layout in lifecycle_layouts(first.kv_cache_manager_py_config)] == [
        FULL,
        WINDOW_128,
        WINDOW_64,
    ]
    assert [layout.key for layout in lifecycle_layouts(second.kv_cache_manager_py_config)] == [
        WINDOW_128,
        FULL,
        WINDOW_64,
    ]
    # Their pool groups hold different layers, so the bytes of a slot differ too.
    assert [pool.slot_sizes for pool in first.impl.get_storage_statistics(0)] != [
        pool.slot_sizes for pool in second.impl.get_storage_statistics(0)
    ]


def test_the_digest_and_the_fingerprints_do_not_depend_on_the_layers_a_manager_holds(
    managers,
) -> None:
    first, second = managers
    used_at_start = first.get_dkv_control_digest()[0]
    for step in range(3):
        assert first.get_dkv_control_digest() == second.get_dkv_control_digest()
        assert first.get_dkv_config_fingerprint() == second.get_dkv_config_fingerprint()
        assert first.get_dkv_state_fingerprint() == second.get_dkv_state_fingerprint()
        # The same requests, up to a window and a half long, in the same order.
        for manager in managers:
            occupy(manager, step, 40 + 8 * step)
    used, free_pages = first.get_dkv_control_digest()
    assert used == used_at_start + 3
    # Every life cycle holds pages now, in the order of the keys: no window, 64, 128.
    assert all(free < total for free, total in zip(free_pages[0], (90, 24, 40)))


def test_the_digest_lists_the_life_cycles_in_the_order_of_their_keys(managers) -> None:
    first, _ = managers
    _, free_pages = first.get_dkv_control_digest()
    assert free_pages == ((90, 24, 40), (120, 30, 60))


def test_the_config_fingerprint_names_no_layer_quota_or_pool_group(managers) -> None:
    fingerprint = managers[0].get_dkv_config_fingerprint()
    assert not [record for record in fingerprint if record[0] == "layer"]
    assert [record for record in fingerprint if record[0] == "pool"] == [
        ("pool", 0, FULL, 90),
        ("pool", 0, WINDOW_64, 24),
        ("pool", 0, WINDOW_128, 40),
        ("pool", 1, FULL, 120),
        ("pool", 1, WINDOW_64, 30),
        ("pool", 1, WINDOW_128, 60),
    ]


def test_a_manager_that_holds_a_page_the_other_does_not_fails_the_control_check(managers) -> None:
    first, second = managers
    occupy(first, 0, 40)

    def payload(manager: KVCacheManagerV2) -> DkvControlPayload:
        used, free_pages = manager.get_dkv_control_digest()
        return DkvControlPayload(
            DkvControlDigest(
                iter_counter=7,
                free_pages=free_pages,
                index_mapper_used=used,
                active_request_count=len(manager.kv_cache_map),
            )
        )

    class TwoRanks:
        tp_size = 2

        def tp_allgather(self, value):
            return [payload(first), payload(second)]

    with pytest.raises(RuntimeError, match="capacity digest differs"):
        sync_dkv_control(TwoRanks(), payload(first))
    # What differs is a single life cycle: the digests agree again once both hold the request.
    occupy(second, 0, 40)
    assert first.get_dkv_control_digest() == second.get_dkv_control_digest()


def test_without_fixed_counts_the_records_keep_the_pool_group_numbers() -> None:
    manager = make_manager([MAX_SEQ_LEN, 128, MAX_SEQ_LEN, 64])
    try:
        pools = [record for record in manager.get_dkv_state_fingerprint() if record[0] == "pool"]
        assert pools and all(type(record[2]) is int for record in pools)
        assert [record for record in manager.get_dkv_config_fingerprint() if record[0] == "layer"]
    finally:
        manager.shutdown()
