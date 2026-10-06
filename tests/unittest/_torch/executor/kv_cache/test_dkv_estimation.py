# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DKV construction and capacity synchronization without GPU allocations."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from _torch.executor.dkv_test_utils import LockstepDistributed, LockstepTpGroup

from tensorrt_llm._torch.distributed.communicator import Distributed, ReduceOp
from tensorrt_llm._torch.pyexecutor._util import (
    KvCacheCreator,
    _create_kv_cache_manager,
    create_py_executor_instance,
)
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
from tensorrt_llm._torch.pyexecutor.resource_manager import ResourceManagerType
from tensorrt_llm.llmapi.llm_args import DkvConfig, KvCacheConfig
from tensorrt_llm.mapping import Mapping

pytestmark = pytest.mark.cpu_only


def _make_creator(
    dkv_enabled: bool, kv_layout: str = "replicated", rank: int = 0, num_layers: int = 43
) -> KvCacheCreator:
    config = KvCacheConfig(use_kv_cache_manager_v2=True)
    model_config = SimpleNamespace(
        is_generation=True,
        is_encoder_decoder=False,
        pretrained_config=SimpleNamespace(num_hidden_layers=num_layers),
    )
    engine = SimpleNamespace(
        model=SimpleNamespace(model_config=model_config),
        kv_cache_manager_key=ResourceManagerType.KV_CACHE_MANAGER,
    )
    args = SimpleNamespace(
        dkv_config=DkvConfig(kv_layout=kv_layout) if dkv_enabled else None,
        kv_cache_config=config,
        cache_transceiver_config=None,
        disable_overlap_scheduler=True,
        enable_iter_perf_stats=False,
        return_perf_metrics=False,
        kv_cache_compression_config=None,
    )
    with patch.object(
        KvCacheCreator, "_get_model_kv_cache_manager_cls", return_value=KVCacheManagerV2
    ):
        return KvCacheCreator(
            model_engine=engine,
            draft_model_engine=None,
            mapping=Mapping(world_size=4, rank=rank, tp_size=4, enable_attention_dp=True),
            net_max_seq_len=1024,
            kv_connector_manager=None,
            max_num_tokens=256,
            max_beam_width=1,
            tokens_per_block=32,
            max_seq_len=1024,
            max_batch_size=2,
            kv_cache_config=config,
            llm_args=args,
            speculative_config=None,
            sparse_attention_config=None,
            profiling_stage_data=None,
            is_disagg=False,
        )


@pytest.mark.parametrize("dkv_enabled", [False, True])
@pytest.mark.parametrize("estimating", [False, True])
def test_creator_propagates_group_size(dkv_enabled: bool, estimating: bool) -> None:
    creator = _make_creator(dkv_enabled)
    manager = SimpleNamespace(max_seq_len=1024)
    with (
        patch.object(creator, "_get_model_kv_cache_manager_cls", return_value=KVCacheManagerV2),
        patch(
            "tensorrt_llm._torch.pyexecutor._util._create_kv_cache_manager", return_value=manager
        ) as create,
    ):
        assert creator._create_kv_cache_manager(creator._model_engine, estimating) is manager

    assert create.call_args.kwargs["dkv_group_size"] == (4 if dkv_enabled else None)
    assert create.call_args.kwargs["estimating_kv_cache"] is estimating
    # The replicated layout stores every layer on every rank and syncs byte quotas.
    assert "dkv_owned_layers" not in create.call_args.kwargs
    assert "dkv_lifecycle_slot_counts" not in create.call_args.kwargs


@pytest.mark.parametrize("estimating", [False, True])
@pytest.mark.parametrize("rank", range(4))
def test_creator_gives_a_layer_split_rank_the_layers_it_owns_and_the_page_counts(
    rank: int, estimating: bool
) -> None:
    creator = _make_creator(True, "layer_split", rank=rank, num_layers=43)
    manager = SimpleNamespace(max_seq_len=1024)
    dist = LockstepDistributed(LockstepTpGroup(4), rank)
    with (
        patch.object(creator, "_get_model_kv_cache_manager_cls", return_value=KVCacheManagerV2),
        patch(
            "tensorrt_llm._torch.pyexecutor._util._create_kv_cache_manager", return_value=manager
        ) as create,
        patch.object(Distributed, "get", return_value=dist),
    ):
        assert creator._create_kv_cache_manager(creator._model_engine, estimating) is manager

    owned = create.call_args.kwargs["dkv_owned_layers"]
    # 43 layers on 4 ranks: the first three ranks own 11 layers and the last one 10, in order.
    assert list(owned) == list(range(11 * rank, 11 * rank + (10 if rank == 3 else 11)))
    counts = create.call_args.kwargs["dkv_lifecycle_slot_counts"]
    assert counts.func.__name__ == "solve_lifecycle_slot_counts"
    assert counts.keywords["allgather"] == dist.tp_allgather


@pytest.mark.parametrize("dkv_group_size", [None, 4])
@pytest.mark.parametrize("mla", [False, True])
def test_factory_propagates_group_size(dkv_group_size: int | None, mla: bool) -> None:
    recorded = {}

    class RecordingManager(KVCacheManagerV2):
        def __init__(self, *args: object, **kwargs: object) -> None:
            recorded.update(kwargs)

    pretrained = SimpleNamespace(
        hidden_size=1024,
        num_attention_heads=8,
        num_key_value_heads=8,
        num_hidden_layers=2,
        vocab_size=32000,
    )
    if mla:
        pretrained.kv_lora_rank = 512
        pretrained.qk_rope_head_dim = 64
    _create_kv_cache_manager(
        model_engine=None,
        model_config=SimpleNamespace(pretrained_config=pretrained, quant_config=None),
        kv_cache_manager_cls=RecordingManager,
        mapping=Mapping(world_size=4, tp_size=4, enable_attention_dp=True),
        kv_cache_config=KvCacheConfig(),
        tokens_per_block=32,
        max_seq_len=1024,
        max_batch_size=2,
        spec_config=None,
        sparse_attention_config=None,
        max_num_tokens=256,
        max_beam_width=1,
        kv_connector_manager=None,
        dtype=torch.bfloat16,
        dkv_group_size=dkv_group_size,
    )
    assert recorded.get("dkv_group_size") == dkv_group_size


@pytest.mark.parametrize("dkv_enabled", [False, True])
def test_static_cache_cost_uses_group_size(dkv_enabled: bool) -> None:
    creator = _make_creator(dkv_enabled)
    model_config = SimpleNamespace(pretrained_config=SimpleNamespace())
    with patch.object(KVCacheManagerV2, "get_cache_size_per_token", return_value=(2, 3)) as cost:
        creator._per_manager_cache_cost(KVCacheManagerV2, model_config)

    assert cost.call_args.kwargs.get("dkv_group_size") == (4 if dkv_enabled else None)


@pytest.mark.parametrize("cap", [None, 0, 8192])
def test_non_dkv_cap_does_not_communicate(cap: int | None) -> None:
    creator = _make_creator(False)
    creator._fp8_ctx_mla_kv_len_cap = cap
    with patch.object(Distributed, "get") as get_dist:
        assert creator._get_synchronized_ctx_mla_kv_len_cap() == cap
    get_dist.assert_not_called()


@pytest.mark.parametrize("local_caps", [(None, None), (8192, 4096), (None, 4096), (0, None)])
def test_dkv_cap_reduces_all_ranks_even_without_local_cap(
    local_caps: tuple[int | None, ...],
) -> None:
    group = LockstepTpGroup(len(local_caps))
    creators = []
    for dist, local_cap in zip(group.ranks, local_caps):
        creator = _make_creator(True)
        creator._mapping = dist.mapping
        creator._dkv_group_size = group.size
        creator._fp8_ctx_mla_kv_len_cap = local_cap
        creators.append(creator)

    with patch.object(Distributed, "get", side_effect=lambda mapping: group.ranks[mapping.tp_rank]):
        results = group.run(
            lambda dist: creators[dist.tp_rank]._get_synchronized_ctx_mla_kv_len_cap()
        )

    finite_caps = [cap for cap in local_caps if cap is not None]
    expected = min(finite_caps) if finite_caps else None
    assert results == [expected] * group.size
    assert all(len(trace) == 1 for trace in group.traces)
    assert all(trace[0][2:] == ("tp_allreduce", ReduceOp.MIN) for trace in group.traces)


@pytest.mark.parametrize("estimating", [False, True])
def test_build_managers_only_synchronizes_final_cap(estimating: bool) -> None:
    creator = _make_creator(True)
    manager = SimpleNamespace()
    resources = {}
    with (
        patch.object(creator, "_create_kv_cache_manager", return_value=manager),
        patch.object(creator, "_get_synchronized_ctx_mla_kv_len_cap", return_value=4096) as sync,
    ):
        creator.build_managers(resources, estimating_kv_cache=estimating)

    assert resources[ResourceManagerType.KV_CACHE_MANAGER] is manager
    if estimating:
        sync.assert_not_called()
        assert not hasattr(manager, "fp8_ctx_mla_kv_len_cap")
    else:
        sync.assert_called_once_with()
        assert manager.fp8_ctx_mla_kv_len_cap == 4096


@pytest.mark.parametrize(
    ("dkv_enabled", "batch_size", "expected_capacity"),
    [(False, 1, 2), (True, 1, 1), (False, 2, 2), (True, 2, 2)],
)
def test_dkv_scheduler_does_not_reserve_an_adp_dummy(
    dkv_enabled: bool, batch_size: int, expected_capacity: int
) -> None:
    class StopAtScheduler(Exception):
        pass

    manager = object.__new__(KVCacheManagerV2)
    args = SimpleNamespace(
        dkv_config=DkvConfig() if dkv_enabled else None,
        extra_resource_managers={},
    )
    with (
        patch("tensorrt_llm._torch.pyexecutor._util.set_low_latency_dispatch"),
        patch(
            "tensorrt_llm._torch.pyexecutor._util.KVCacheV2Scheduler",
            side_effect=StopAtScheduler,
        ) as scheduler,
        pytest.raises(StopAtScheduler),
    ):
        create_py_executor_instance(
            dist=None,
            resources={ResourceManagerType.KV_CACHE_MANAGER: manager},
            mapping=Mapping(world_size=4, tp_size=4, enable_attention_dp=True),
            llm_args=args,
            ctx_chunk_config=None,
            model_engine=SimpleNamespace(spec_config=None),
            start_worker=False,
            sampler=None,
            drafter=None,
            max_batch_size=batch_size,
            max_num_tokens=256,
            max_num_sequences=4,
        )
    assert scheduler.call_args.kwargs["scheduler_capacity"] == expected_capacity
