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

"""Local forward views, replicated resource lifecycles and DKV slot headroom."""

from types import SimpleNamespace

import pytest
import torch
from _torch.executor.dkv_test_utils import make_request

from tensorrt_llm import Mapping
from tensorrt_llm._torch.pyexecutor._util import (
    compute_max_num_sequences,
    create_torch_sampler_args,
    resolve_max_num_sequences,
    validate_seq_slot_pool_covers_admission,
)
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest, LlmRequestState
from tensorrt_llm._torch.pyexecutor.resource_manager import (
    BaseResourceManager,
    ResourceManager,
    ResourceManagerType,
)
from tensorrt_llm._torch.pyexecutor.sampler import TorchSampler
from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests
from tensorrt_llm._torch.pyexecutor.seq_slot_manager import SeqSlotManager


def _batch(requests: list[LlmRequest]) -> ScheduledRequests:
    batch = ScheduledRequests()
    batch.context_requests_last_chunk = requests
    return batch


def _ids(requests: list[LlmRequest]) -> list[int]:
    return [request.py_request_id for request in requests]


class _LifecycleRecorder(BaseResourceManager):
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def get_max_resource_count(self) -> int:
        return 16

    def get_needed_resource_to_completion(self, request: LlmRequest) -> int:
        return 1

    def prepare_resources(self, batch: ScheduledRequests) -> None:
        self.calls.append(("prepare", batch))

    def update_resources(self, batch: ScheduledRequests, *metadata: object) -> None:
        self.calls.append(("update", batch, *metadata))

    def free_resources(self, request: LlmRequest) -> None:
        self.calls.append(("free", request))


class _ForwardRecorder(_LifecycleRecorder):
    dkv_scope = "forward"


@pytest.mark.cpu_only
def test_local_view_preserves_order_classification_and_request_identity() -> None:
    requests = [make_request(index, compute_rank=index % 2) for index in range(8)]
    batch = ScheduledRequests()
    batch.context_requests_chunking = requests[:3]
    batch.context_requests_last_chunk = requests[3:6]
    batch.generation_requests = requests[6:]
    batch.paused_requests = requests[:1]
    batch.recompute_paused_requests = requests[1:2]
    batch.added_inflight_req_ids = [request.py_request_id for request in requests]
    # Cursor-dependent classifications can change after scheduling. The view
    # must retain the scheduler's buckets even when those predicates change.
    assert requests[0].is_last_context_chunk
    requests[4].context_chunk_size = 1
    assert not requests[4].is_last_context_chunk

    local = batch.local_view()

    assert _ids(local.context_requests_chunking) == [0, 2]
    assert _ids(local.context_requests_last_chunk) == [4]
    assert _ids(local.generation_requests) == [6]
    assert local.context_requests_chunking[0] is requests[0]
    assert local.context_requests_last_chunk[0] is requests[4]
    assert local.encoder_requests == []
    assert local.scheduled_mm_encoder_items is None
    assert local.paused_requests == []
    assert local.recompute_paused_requests == []
    assert local.added_inflight_req_ids == []
    local.generation_requests.append(make_request(100, is_dummy=True))
    local.context_requests_chunking.clear()
    assert batch.context_requests_chunking == requests[:3]
    assert batch.generation_requests == requests[6:]
    assert batch.added_inflight_req_ids == list(range(8))


@pytest.mark.cpu_only
def test_local_view_accepts_an_explicit_rank_predicate() -> None:
    batch = _batch([make_request(index, compute_rank=index % 2) for index in range(6)])
    local = batch.local_view(lambda request: request.py_dkv_compute_rank == 1)
    assert _ids(local.context_requests_last_chunk) == [1, 3, 5]
    assert all(not request.py_dkv_is_local for request in local.all_requests())


@pytest.mark.cpu_only
@pytest.mark.parametrize("multimodal", [False, True])
def test_local_view_rejects_unsupported_encoder_work(multimodal: bool) -> None:
    batch = ScheduledRequests()
    if multimodal:
        batch.scheduled_mm_encoder_items = {1: [0]}
    else:
        batch.encoder_requests = [make_request(1)]
    with pytest.raises(ValueError, match="encoder requests"):
        batch.local_view()


@pytest.mark.cpu_only
@pytest.mark.parametrize("use_local_view", [False, True])
def test_resource_scope_selects_batch_without_changing_metadata_or_lifecycle(
    use_local_view: bool,
) -> None:
    lifecycle, forward = _LifecycleRecorder(), _ForwardRecorder()
    resources = ResourceManager(
        {
            ResourceManagerType.KV_CACHE_MANAGER: lifecycle,
            ResourceManagerType.SEQ_SLOT_MANAGER: forward,
        }
    )
    batch = _batch([make_request(1), make_request(2, compute_rank=1)])
    local = batch.local_view() if use_local_view else None
    metadata = object()

    resources.prepare_resources(batch, forward_batch=local)
    resources.update_resources(batch, metadata, 2.0, forward_batch=local)
    for request in batch.all_requests():
        resources.free_resources(request)

    assert lifecycle.calls == [
        ("prepare", batch),
        ("update", batch, metadata, 2.0),
        ("free", batch.all_requests()[0]),
        ("free", batch.all_requests()[1]),
    ]
    assert forward.calls == [
        ("prepare", local if use_local_view else batch),
        ("update", local if use_local_view else batch),
        ("free", batch.all_requests()[0]),
        ("free", batch.all_requests()[1]),
    ]


@pytest.mark.cpu_only
def test_only_local_requests_receive_sequence_slots_and_remote_free_is_harmless() -> None:
    slots = SeqSlotManager(2)
    lifecycle = _LifecycleRecorder()
    resources = ResourceManager(
        {
            ResourceManagerType.SEQ_SLOT_MANAGER: slots,
            ResourceManagerType.KV_CACHE_MANAGER: lifecycle,
        }
    )
    requests = [make_request(index, compute_rank=index % 2) for index in range(4)]
    batch = _batch(requests)
    resources.prepare_resources(batch, forward_batch=batch.local_view())
    assert {request.py_seq_slot for request in requests[::2]} == {0, 1}
    assert all(request.py_seq_slot is None for request in requests[1::2])
    for remote in requests[1::2]:
        resources.free_resources(remote)
    assert slots.slot_manager.slot_mapping.keys() == {0, 2}
    for local in requests[::2]:
        resources.free_resources(local)
    assert slots.slot_manager.free_slots == {0, 1}
    assert lifecycle.calls[0] == ("prepare", batch)


@pytest.mark.cpu_only
@pytest.mark.parametrize("batch_size", [1, 2, 8])
@pytest.mark.parametrize("group_size", [2, 4, 8])
def test_slot_pool_reserves_one_local_dummy_and_fits_global_kv_admission(
    batch_size: int, group_size: int
) -> None:
    mapping = Mapping(world_size=group_size, tp_size=group_size, enable_attention_dp=True)
    capacity = compute_max_num_sequences(mapping, batch_size, True, dkv_enabled=True)
    assert capacity == batch_size + 1
    assert compute_max_num_sequences(mapping, batch_size, True) == batch_size
    validate_seq_slot_pool_covers_admission(
        capacity, SimpleNamespace(max_admissible_sequences=group_size * batch_size)
    )
    slots = SeqSlotManager(capacity)
    dummy = make_request(1000, is_dummy=True)
    dummy_batch = ScheduledRequests()
    dummy_batch.generation_requests = [dummy]
    slots.prepare_resources(dummy_batch)
    reserved = dummy.py_seq_slot
    for iteration in range(3):
        requests = [make_request(1 + iteration * batch_size + index) for index in range(batch_size)]
        slots.prepare_resources(_batch(requests))
        assert len(slots.slot_manager.slot_mapping) == capacity
        assert dummy.py_seq_slot == reserved
        assert reserved not in {request.py_seq_slot for request in requests}
        for request in requests:
            slots.free_resources(request)
        slots.prepare_resources(dummy_batch)
        assert slots.slot_manager.slot_mapping == {dummy.py_request_id: reserved}


@pytest.mark.cpu_only
@pytest.mark.parametrize("published", [None, 3])
def test_slot_resolution_adds_dummy_only_at_the_capacity_source(published: int | None) -> None:
    engine = SimpleNamespace(max_num_seq_slots=published, _enable_overlap_headroom=False)
    llm_args = SimpleNamespace(disable_overlap_scheduler=True, dkv_config=object())
    mapping = Mapping(world_size=2, tp_size=2, enable_attention_dp=True)
    assert resolve_max_num_sequences(engine, mapping, 2, llm_args) == 3
    assert resolve_max_num_sequences(engine, mapping, 2, llm_args, max_num_sequences=3) == 3


@pytest.mark.parametrize("batch_size", [1, 8])
def test_real_torch_sampler_initializes_every_dkv_slot_including_dummy(batch_size: int) -> None:
    mapping = Mapping(world_size=2, tp_size=2, enable_attention_dp=True)
    capacity = compute_max_num_sequences(mapping, batch_size, True, dkv_enabled=True)
    engine = SimpleNamespace(max_num_seq_slots=capacity)
    llm_args = SimpleNamespace(disable_overlap_scheduler=True, dkv_config=object())
    resolved = resolve_max_num_sequences(engine, mapping, batch_size, llm_args)
    slots = SeqSlotManager(resolved)
    requests = [make_request(index + 1) for index in range(batch_size)]
    batch = _batch(requests)
    dummy = make_request(1000, is_dummy=True)
    dummy.state = LlmRequestState.GENERATION_IN_PROGRESS
    batch.generation_requests = [dummy]
    dummy_batch = ScheduledRequests()
    dummy_batch.generation_requests = [dummy]
    slots.prepare_resources(dummy_batch)
    slots.prepare_resources(batch)
    assert {request.py_seq_slot for request in batch.all_requests()} == set(range(capacity))
    assert max(request.py_seq_slot for request in requests) == capacity - 1

    sampler = TorchSampler(
        create_torch_sampler_args(
            max_seq_len=32,
            speculative_config=None,
            max_beam_width=1,
            disable_overlap_scheduler=True,
            enable_async_worker=False,
            enable_speculative_beam_history_d2h=False,
            max_num_sequences=resolved,
        )
    )
    dummy_before = (tuple(dummy.get_tokens(0)), dummy.state, dummy.py_seq_slot)
    for _ in range(4):
        sampler.setup_sampler_step(batch)
        assert (tuple(dummy.get_tokens(0)), dummy.state, dummy.py_seq_slot) == dummy_before
    torch.cuda.synchronize()

    lengths = sampler._finish_reasons_handler.store.max_lengths_cuda.cpu().tolist()
    for request in batch.all_requests():
        assert lengths[request.py_seq_slot] == request.py_prompt_len + request.max_new_tokens
    assert sampler.max_num_sequences == capacity
    assert sampler.store.new_tokens.shape[1] == capacity + 1
    assert sampler._finish_reasons_handler.store.finish_reasons_cuda.shape[1] == capacity
    assert len(sampler._pending_steps) == capacity
    assert len(sampler._penalty_handler._slots) == capacity
    assert len(sampler._seed_manager._slot_owner) == capacity
    assert sampler.dummy_slot_row == capacity
    assert dummy.py_seq_slot != sampler.dummy_slot_row
