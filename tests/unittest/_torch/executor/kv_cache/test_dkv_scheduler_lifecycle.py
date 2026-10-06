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
"""Single-GPU scheduler contracts against the selected Python or C++ KV core."""

from collections.abc import Iterator
from contextlib import contextmanager

import pytest
import torch
from _torch.executor.dkv_test_utils import make_request

from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest
from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests
from tensorrt_llm._torch.pyexecutor.scheduler.scheduler_v2 import KVCacheV2Scheduler
from tensorrt_llm.bindings import DataType
from tensorrt_llm.bindings.internal.batch_manager import CacheType
from tensorrt_llm.llmapi.llm_args import CapacitySchedulerPolicy, KvCacheConfig
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.runtime.kv_cache_manager_v2 import CacheLevel

_TOKENS_PER_BLOCK = 4
_PAGE_BYTES = 2 << 20


@contextmanager
def _manager(num_pages: int) -> Iterator[KVCacheManagerV2]:
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    torch.cuda.init()
    manager = KVCacheManagerV2(
        KvCacheConfig(
            enable_block_reuse=False,
            enable_swa_scratch_reuse=False,
            max_gpu_total_bytes=num_pages * _PAGE_BYTES,
            host_cache_size=0,
            max_util_for_resume=1.0,
        ),
        CacheType.SELF,
        num_layers=1,
        num_kv_heads=128,
        head_dim=1024,
        tokens_per_block=_TOKENS_PER_BLOCK,
        max_seq_len=24,
        max_batch_size=2,
        max_num_tokens=8,
        mapping=Mapping(),
        dtype=DataType.HALF,
        disable_overlap_scheduler=True,
        dkv_group_size=2,
    )
    try:
        statistics = manager.impl.get_storage_statistics(CacheLevel(0))
        assert len(statistics) == 1
        assert statistics[0].total == num_pages
        yield manager
    finally:
        torch.cuda.current_stream().synchronize()
        manager.shutdown()


def _scheduler(manager: KVCacheManagerV2) -> KVCacheV2Scheduler:
    return KVCacheV2Scheduler(
        max_batch_size=2,
        max_num_tokens=8,
        kv_cache_manager=manager,
        scheduler_policy=CapacitySchedulerPolicy.MAX_UTILIZATION,
        ctx_chunk_config=(None, _TOKENS_PER_BLOCK),
        enable_recompute_pause=False,
        dkv_group_size=2,
    )


def _advance_context(manager: KVCacheManagerV2, request: LlmRequest) -> None:
    batch = ScheduledRequests()
    batch.append_context_request(request)
    request.move_to_next_context_chunk()
    manager.update_context_resources(batch)


def _start_context(manager: KVCacheManagerV2, request: LlmRequest, chunk_size: int) -> None:
    assert manager.prepare_context(request)
    request.context_chunk_size = chunk_size
    assert manager.resize_context(request, chunk_size)
    _advance_context(manager, request)


def _free_pages(manager: KVCacheManagerV2) -> int:
    return manager.impl.get_storage_statistics(CacheLevel(0))[0].free


def test_dkv_attended_cap_rolls_back_and_resumes_real_continuation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TRTLLM_DKV_DUAL_LEDGER", "1")
    with _manager(num_pages=16) as manager:
        scheduler = _scheduler(manager)
        leader = make_request(1, prompt_len=4)
        continuation = make_request(2, prompt_len=12)
        try:
            _start_context(manager, continuation, chunk_size=8)
            cache = manager.kv_cache_map[continuation.py_request_id]
            assert (cache.capacity, cache.history_length, cache.is_active) == (8, 8, True)
            free_before = _free_pages(manager)
            manager.fp8_ctx_mla_kv_len_cap = 12

            output = scheduler.schedule_request([leader, continuation], set())

            assert output.context_requests == [leader]
            assert manager.kv_cache_map[continuation.py_request_id] is cache
            assert (cache.capacity, cache.history_length, cache.is_active) == (8, 8, False)
            assert continuation.context_current_position == 8
            assert continuation.prepopulated_prompt_len == 0
            assert not continuation.is_first_context_chunk
            assert continuation.py_ctx_pre_resize_cap is None
            manager.free_resources(leader)
            assert _free_pages(manager) == free_before

            retry = scheduler.schedule_request([continuation], set())

            assert retry.context_requests == [continuation]
            assert continuation.context_chunk_size == 4
            assert (cache.capacity, cache.history_length, cache.is_active) == (12, 8, True)
            _advance_context(manager, continuation)
            assert continuation.context_current_position == 12
            assert cache.history_length == 12
        finally:
            manager.free_resources(leader)
            manager.free_resources(continuation)


def test_dkv_stall_releases_real_kv_and_rewinds_native_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TRTLLM_DKV_DUAL_LEDGER", "1")
    with _manager(num_pages=6) as manager:
        scheduler = _scheduler(manager)
        first = make_request(1, compute_rank=0, prompt_len=20)
        last = make_request(2, compute_rank=1, prompt_len=20)
        try:
            for request in (first, last):
                _start_context(manager, request, chunk_size=12)
            assert _free_pages(manager) == 0
            free_indices = manager.index_mapper.num_free_slots()

            stalled = scheduler.schedule_request([first, last], set())

            assert stalled.context_requests == []
            assert list(manager.kv_cache_map) == [first.py_request_id]
            assert manager.index_mapper.num_free_slots() == free_indices + 1
            assert _free_pages(manager) == 3
            assert last.context_current_position == 0
            assert last.prepopulated_prompt_len == 0
            assert last.context_chunk_size == last.prompt_len
            assert last.estimated_reusable_tokens == 0
            assert last.py_ctx_pre_resize_cap is None
            assert last.is_first_context_chunk

            retry = scheduler.schedule_request([first, last], set())

            assert retry.context_requests == [first]
            assert first.context_chunk_size == 8
            _advance_context(manager, first)
            assert first.context_current_position == first.prompt_len
            assert manager.kv_cache_map[first.py_request_id].history_length == first.prompt_len
            manager.free_resources(first)

            replay = scheduler.schedule_request([last], set())

            assert replay.context_requests == [last]
            _advance_context(manager, last)
            assert last.context_current_position == 8
            assert manager.kv_cache_map[last.py_request_id].history_length == 8
        finally:
            manager.free_resources(first)
            manager.free_resources(last)
