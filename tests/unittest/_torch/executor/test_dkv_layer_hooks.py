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
"""How a forward pass tells the working cache of the DKV layer split what to stage, without any
model knowing about it.

The model runner opens the pages of the batch before it prepares the attention metadata, and the
attention of every layer announces the layer through its metadata before it touches the KV cache.
Both reach the data plane only through the cache manager the metadata reads.
"""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.attention.backends.interface import AttentionMetadata
from tensorrt_llm._torch.metadata import KVCacheParams
from tensorrt_llm._torch.pyexecutor.engine.runners.decoder.runner import DecoderRunner
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2

pytestmark = pytest.mark.cpu_only


def _working_cache(calls: list) -> KVCacheManagerV2:
    """A V2 manager that is the working cache of a rank: it has a session and a data plane."""
    manager = KVCacheManagerV2.__new__(KVCacheManagerV2)
    manager.dkv_working_session = SimpleNamespace(
        begin_batch=lambda *batch: calls.append(("batch", *batch))
    )
    manager.dkv_streamer = SimpleNamespace(on_layer=lambda layer: calls.append(("layer", layer)))
    return manager


def _metadata(manager) -> AttentionMetadata:
    """A pass of one context request with 300 cached and 128 new tokens and one decoding request
    with 99 cached tokens."""
    metadata = AttentionMetadata(
        max_num_requests=2,
        max_num_tokens=256,
        kv_cache_manager=manager,
        request_ids=[7, 3],
        kv_cache_params=KVCacheParams(use_cache=True, num_cached_tokens_per_seq=[300, 99]),
    )
    metadata.seq_lens = torch.tensor([128, 1], dtype=torch.int32)
    metadata.num_contexts = 1
    return metadata


def test_the_attention_of_a_layer_announces_it_to_the_working_cache() -> None:
    calls: list = []
    metadata = _metadata(_working_cache(calls))
    for layer in (0, 1, 2):
        metadata.begin_layer(layer)
    assert calls == [("layer", 0), ("layer", 1), ("layer", 2)]


@pytest.mark.parametrize(
    "manager",
    [None, SimpleNamespace(), KVCacheManagerV2.__new__(KVCacheManagerV2)],
    ids=["no manager", "a manager without the hook", "a manager that stages nothing"],
)
def test_announcing_a_layer_does_nothing_without_a_working_cache(manager) -> None:
    _metadata(manager).begin_layer(4)


def test_the_runner_opens_the_batch_of_a_staged_pass_before_the_metadata_is_prepared() -> None:
    calls: list = []
    metadata = _metadata(_working_cache(calls))
    DecoderRunner._begin_staged_batch(SimpleNamespace(dkv_staging=True), metadata)
    assert calls == [("batch", [7, 3], [300, 99], [128, 1])]
    # Without staging the runner leaves the manager alone.
    calls.clear()
    DecoderRunner._begin_staged_batch(SimpleNamespace(dkv_staging=False), metadata)
    assert calls == []
