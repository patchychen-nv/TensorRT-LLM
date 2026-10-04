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

import hashlib
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from _torch.executor.dkv_test_utils import LockstepDistributed, LockstepTpGroup

from tensorrt_llm._torch.attention.backends.sparse.deepseek_v4.cache_manager import (
    DeepseekV4CacheManager,
)
from tensorrt_llm._torch.distributed.communicator import ReduceOp
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import (
    CUDA_GRAPH_DUMMY_REQUEST_ID,
    KVCacheManagerV2,
)
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest, SamplingConfig
from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests
from tensorrt_llm.bindings import DataType
from tensorrt_llm.llmapi.llm_args import KvCacheConfig
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.runtime.kv_cache_manager_v2 import (
    AttentionLayerConfig,
    BufferConfig,
    CacheTier,
    DataRole,
    GpuCacheTierConfig,
    LayerId,
)

pytestmark = pytest.mark.cpu_only


def _manager(group_size: int | None) -> KVCacheManagerV2:
    manager = object.__new__(KVCacheManagerV2)
    manager.dkv_group_size = group_size
    manager.max_batch_size = 2
    manager.max_num_tokens = 100
    manager.max_seq_len = 4096
    manager.tokens_per_block = 64
    manager._has_cp_helix = False
    manager._generation_kv_capacity_headroom = 1
    manager.enable_swa_scratch_reuse = False
    manager._get_runtime_cache_size_layer_components = lambda: ([10, 10, 20], [128, 128, None])
    manager._guard_page_value = None
    manager.is_disagg = False
    manager.mapping = Mapping()
    manager._dkv_trace = []
    return manager


@pytest.mark.parametrize("group_size", [None, 2, 4, 8])
@pytest.mark.parametrize("is_disagg", [False, True])
@pytest.mark.parametrize("guard", [None, 0])
def test_sequence_capacity_reserves_global_admission_and_dummies(
    group_size: int | None, is_disagg: bool, guard: int | None
) -> None:
    manager = _manager(group_size)
    manager.is_disagg = is_disagg
    manager._guard_page_value = guard
    admission, indices = manager._get_sequence_capacities(1, True)
    assert manager._get_max_num_sequences() == 2 * (group_size or 1)
    assert admission == 2 * (group_size or 1) * (2 if is_disagg else 1)
    assert indices == admission + 1 + (group_size or 0) + (guard is not None)


def test_non_dkv_overlap_retains_extra_sequence_leases() -> None:
    manager = _manager(None)
    assert manager._get_sequence_capacities(1, False) == (4, 5)
    manager.mapping = Mapping(world_size=2, pp_size=2)
    assert manager._get_max_num_sequences() == 4
    assert manager._get_sequence_capacities(1, False) == (4, 5)


@pytest.mark.parametrize("group_size", [None, 2, 4, 8])
def test_global_typical_context_keeps_local_startup_feasibility(group_size: int | None) -> None:
    manager = _manager(group_size)
    manager.num_extra_kv_tokens = 0
    manager.max_draft_len = 0
    manager.max_cuda_graph_batch_size = None
    manager.num_local_layers = 1
    manager.max_attention_window_vec = [128]
    manager._ledger_tokens_per_block = 64
    manager.reuse_match_backoff = 0
    manager.enable_stats = False
    manager._get_buffer_roles_for_layer = lambda layer: [DataRole("key")]
    manager.get_layer_bytes_per_token = lambda local_layer_idx, data_role: 10
    config = manager._build_base_config(
        KvCacheConfig(avg_seq_len=256, enable_block_reuse=False),
        tokens_per_block=64,
        cache_tiers=[GpuCacheTierConfig(quota=4 << 20)],
    )
    assert len(config.typical_step.kv_caches) == 2 * (group_size or 1)
    assert config.typical_step.kv_caches[0].capacity == 100 * (group_size or 1)
    assert len(config.constraints[-1].kv_caches) == 1
    assert config.constraints[-1].kv_caches[0].capacity == 100
    assert config.constraints[-1].kv_caches[0].history_length == 0


@pytest.mark.parametrize("group_size", [None, 2, 4, 8])
@pytest.mark.parametrize("tokens", [1, 100, 300, 1000])
def test_swa_quota_accounts_for_global_context_tokens_and_round_trips(
    group_size: int | None, tokens: int
) -> None:
    manager = _manager(group_size)
    replicas = group_size or 1
    fixed_window_bytes = replicas * 2 * 2 * 192 * 10
    context_swa_bytes = min(tokens, replicas * 100) * 20
    quota = tokens * 20 + context_swa_bytes + fixed_window_bytes
    assert manager._get_quota_from_max_tokens(tokens) == quota
    assert manager._get_max_tokens_from_quota(quota) == tokens
    assert manager._get_max_tokens_from_quota(fixed_window_bytes - 1) == 0
    assert manager.max_num_tokens == 100
    assert manager.max_batch_size == 2


@pytest.mark.parametrize("group_size", [None, 2, 4, 8])
@pytest.mark.parametrize("tokens", [1, 1024, 4096, 16384])
@pytest.mark.parametrize("context_limit", [None, 1024])
def test_v4_quota_round_trip_preserves_local_forward_capacity(
    group_size: int | None, tokens: int, context_limit: int | None
) -> None:
    manager = object.__new__(DeepseekV4CacheManager)
    manager.dkv_group_size = group_size
    manager.pp_layers = [0, 1]
    manager._compress_ratios = [4, 4]
    manager.dtype = DataType.BF16
    manager._use_nvfp4_compress = False
    manager.head_dim = 576
    manager.index_head_dim = 128
    manager._indexer_k_dtype = "fp8"
    manager.use_fp8_ds_mla = False
    manager._swa_window_size = 128
    manager._max_draft_len = 0
    manager._max_num_tokens = context_limit
    manager.tokens_per_block = 128
    manager.max_batch_size = 2
    manager.enable_swa_scratch_reuse = False

    quota = manager._get_quota_from_max_tokens(tokens)
    assert manager._get_max_tokens_from_quota(quota) == tokens
    assert manager._get_max_tokens_from_quota(manager._get_quota_from_max_tokens(0) - 1) == 0
    if group_size is not None:
        manager.dkv_group_size = None
        assert quota > manager._get_quota_from_max_tokens(tokens)
    assert manager.max_batch_size == 2
    assert manager._max_num_tokens == context_limit


@pytest.mark.parametrize("group_size", [None, 2])
def test_all_swa_infinite_tokens_synchronize_dkv_byte_quota(group_size: int | None) -> None:
    manager = _manager(group_size)
    manager._get_max_tokens_from_quota = Mock(return_value=float("inf"))
    dist = Mock()
    dist.allreduce.side_effect = [float("inf"), 4096]
    assert manager._sync_device_quota(8192, 0.9, dist) == (8192 if group_size is None else 4096)
    assert dist.allreduce.call_count == (1 if group_size is None else 2)
    for call in dist.allreduce.call_args_list:
        assert call.kwargs == {"op": ReduceOp.MIN}


def test_finite_capacity_sync_never_increases_local_quota() -> None:
    manager = _manager(2)
    manager._get_max_tokens_from_quota = Mock(return_value=20)
    manager._get_quota_from_max_tokens = Mock(return_value=9000)
    dist = Mock()
    dist.allreduce.side_effect = [10, 8192]
    assert manager._sync_device_quota(8192, 0.9, dist) == 8192


@pytest.mark.parametrize("local_quota", [6000, 8000])
def test_zero_token_capacity_still_unifies_device_bytes(local_quota: int) -> None:
    manager = _manager(2)
    manager._get_max_tokens_from_quota = Mock(return_value=0)
    manager._get_quota_from_max_tokens = Mock(return_value=7000)
    dist = Mock()
    dist.allreduce.side_effect = [0, 6000]
    assert manager._sync_device_quota(local_quota, 1.0, dist) == 6000
    assert isinstance(dist.allreduce.call_args_list[0].args[0], float)


@pytest.mark.parametrize("all_swa", [False, True])
def test_device_quota_collectives_agree_at_zero_and_infinite_capacity(all_swa: bool) -> None:
    group = LockstepTpGroup(2)
    local_quotas = [100000, 90000] if all_swa else [15000, 20000]

    def synchronize(dist: LockstepDistributed) -> int:
        manager = _manager(2)
        if all_swa:
            manager._get_runtime_cache_size_layer_components = lambda: ([10], [128])
        # DKV excludes PP and CP, so its world and attention-DP groups coincide.
        communicator = SimpleNamespace(allreduce=dist.tp_allreduce)
        return manager._sync_device_quota(local_quotas[dist.tp_rank], 1.0, communicator)

    assert group.run(synchronize) == [min(local_quotas)] * 2
    assert all(len(trace) == 2 for trace in group.traces)
    assert all(
        entry[2:] == ("tp_allreduce", ReduceOp.MIN) for trace in group.traces for entry in trace
    )


def _request(request_id: int = 17) -> LlmRequest:
    return LlmRequest(
        request_id=request_id,
        max_new_tokens=1,
        input_tokens=[1, 2],
        sampling_config=SamplingConfig(beam_width=1),
        is_streaming=False,
    )


def test_generation_guard_precedes_allocation_or_resource_mutation() -> None:
    manager = _manager(2)
    manager.kv_cache_map = {}
    request = _request()
    with pytest.raises(RuntimeError, match="not supported with dkv_config yet"):
        manager.try_allocate_generation(request)
    batch = ScheduledRequests()
    batch.generation_requests = [request]
    with pytest.raises(RuntimeError, match="not supported with dkv_config yet"):
        manager.update_resources(batch)
    request.is_dummy_request = True
    assert not manager.try_allocate_generation(request)
    manager.dkv_group_size = None
    request.is_dummy_request = False
    assert not manager.try_allocate_generation(request)


def test_generation_dummy_can_still_initialize_kv() -> None:
    manager = _manager(2)
    manager.num_extra_kv_tokens = 0
    manager._stream = SimpleNamespace(cuda_stream=1)
    cache = SimpleNamespace(
        num_committed_tokens=0,
        capacity=0,
        resume=Mock(return_value=True),
        stop_committing=Mock(),
    )

    def resize(capacity: int, history_length: int | None = None) -> bool:
        cache.capacity = capacity
        return True

    cache.resize = resize
    manager._create_kv_cache = Mock(return_value=cache)
    requests = manager.add_dummy_requests(
        [CUDA_GRAPH_DUMMY_REQUEST_ID], token_nums=[2], is_gen=True
    )
    assert len(requests) == 1
    assert requests[0].py_request_id == CUDA_GRAPH_DUMMY_REQUEST_ID
    assert requests[0].is_dummy_request
    assert cache.capacity == 3
    cache.stop_committing.assert_called_once()


def test_debug_trace_records_failed_admission_and_drains() -> None:
    manager = _manager(2)
    manager._dkv_trace_enabled = True
    manager._install_dkv_hooks()
    manager.prepare_context_cache = Mock(return_value=None)
    request = _request()
    assert not manager.prepare_context(request)
    trace = manager.consume_dkv_trace()
    assert trace == [
        ("call", "prepare_context", ((17, 0, False),), ()),
        ("return", "prepare_context", False),
    ]
    assert manager.consume_dkv_trace() == []
    manager._dkv_trace_enabled = False
    assert not manager.prepare_context(request)
    assert manager.consume_dkv_trace() == []


def test_debug_trace_preserves_generation_guard_errors() -> None:
    manager = _manager(2)
    manager._dkv_trace_enabled = True
    manager._install_dkv_hooks()
    with pytest.raises(RuntimeError, match="not supported with dkv_config yet"):
        manager.try_allocate_generation(_request())
    assert manager.consume_dkv_trace()[-1] == ("raised", "try_allocate_generation", None)


@pytest.mark.parametrize("resize_succeeds", [False, True])
def test_context_rollback_exposes_success_and_failure_to_c3(resize_succeeds: bool) -> None:
    manager = _manager(2)
    manager._dkv_trace_enabled = True
    manager._install_dkv_hooks()
    manager._connector_reservations_enabled = lambda: False
    cache = SimpleNamespace(
        capacity=128,
        history_length=64,
        is_active=True,
        resize=Mock(return_value=resize_succeeds),
        suspend=Mock(),
    )
    request = _request()
    request.py_ctx_pre_resize_cap = 64
    manager.kv_cache_map = {request.py_request_id: cache}

    if resize_succeeds:
        assert manager.revert_allocate_context(request)
        cache.suspend.assert_called_once_with()
    else:
        with pytest.raises(RuntimeError, match="Failed to revert KV cache capacity"):
            manager.revert_allocate_context(request)
        cache.suspend.assert_not_called()

    cache.resize.assert_called_once_with(64, 64)
    assert request.py_ctx_pre_resize_cap is None
    trace = manager.consume_dkv_trace()
    assert trace[0] == ("call", "revert_allocate_context", ((17, 0, False),), ())
    assert trace[-1] == (
        "return" if resize_succeeds else "raised",
        "revert_allocate_context",
        True if resize_succeeds else None,
    )
    assert manager.consume_dkv_trace() == []


def _fingerprint_manager() -> KVCacheManagerV2:
    manager = _manager(2)
    manager.dtype = DataType.BF16
    manager.max_admissible_sequences = 4
    manager.enable_block_reuse = True
    manager.block_reuse_policy = "per_request"
    manager.index_mapper = SimpleNamespace(size=lambda: 1, num_free_slots=lambda: 6)
    manager._early_freed_index_requests = set()
    manager.kv_cache_map = {
        -100: SimpleNamespace(capacity=2, history_length=0, num_committed_tokens=0, is_active=True),
    }
    manager.kv_cache_manager_py_config = SimpleNamespace(
        tokens_per_block=64,
        max_util_for_resume=0.9,
        layers=[
            AttentionLayerConfig(
                LayerId(0), [BufferConfig(DataRole("key"), 512)], sliding_window_size=128
            ),
        ],
    )
    manager.impl = SimpleNamespace(
        cache_tier_list=[CacheTier.GPU_MEM],
        get_quota=lambda level: 8192,
        get_life_cycle_pool_group_indices=lambda level: [0],
        get_storage_statistics=lambda level: [
            SimpleNamespace(slot_sizes=[512], total=16, free=15, evictable=0)
        ],
    )
    return manager


def test_fingerprints_include_dummies_and_ignore_physical_slot_ids() -> None:
    left = _fingerprint_manager()
    right = _fingerprint_manager()
    left.kv_cache_map[-100].slot_id = 7
    right.kv_cache_map[-100].slot_id = 11
    assert left.get_dkv_config_fingerprint() == right.get_dkv_config_fingerprint()
    assert left.get_dkv_state_fingerprint() == right.get_dkv_state_fingerprint()
    original = hashlib.sha256(repr(left.get_dkv_state_fingerprint()).encode()).hexdigest()
    right.kv_cache_map[-100].capacity += 1
    changed = hashlib.sha256(repr(right.get_dkv_state_fingerprint()).encode()).hexdigest()
    assert original != changed
    right._early_freed_index_requests.add(-100)
    assert left.get_dkv_state_fingerprint() != right.get_dkv_state_fingerprint()
    right.max_admissible_sequences += 1
    assert left.get_dkv_config_fingerprint() != right.get_dkv_config_fingerprint()
