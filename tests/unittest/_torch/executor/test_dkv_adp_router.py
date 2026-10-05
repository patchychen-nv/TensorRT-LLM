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
"""DKV token-load routing with real request bindings and exchanged rank payloads."""

import pytest
from dkv_test_utils import LockstepTpGroup, make_request

from tensorrt_llm._torch.pyexecutor.executor_request_queue import RequestQueueItem
from tensorrt_llm._torch.pyexecutor.llm_request import ExecutorRequest, LlmRequestState
from tensorrt_llm._torch.pyexecutor.scheduler.adp_router import (
    ADPRouter,
    DefaultADPRouter,
    KVCacheAwareADPRouter,
    RankIterStatsPayload,
)
from tensorrt_llm.bindings.executor import LoraConfig
from tensorrt_llm.scheduling_params import SchedulingParams

pytestmark = pytest.mark.cpu_only


class _PrefixCache:
    def __init__(self, *, reuse: bool = True, matches: dict[int, int] | None = None) -> None:
        self.enable_block_reuse = reuse
        self.matches = matches or {}
        self.calls: list[tuple[list[int], int | None, str | None]] = []

    def probe_prefix_match_length(
        self, tokens: list[int], lora_task_id: int | None, *, cache_salt: str | None
    ) -> int:
        assert self.enable_block_reuse, "Reuse-disabled routing must not probe the cache"
        self.calls.append((tokens, lora_task_id, cache_salt))
        return self.matches.get(len(tokens), 0)


def _item(
    request_id: int,
    prompt_len: int,
    *,
    target_rank: int | None = None,
    lora_task_id: int | None = None,
    cache_salt: str | None = None,
) -> RequestQueueItem:
    request = ExecutorRequest(
        input_token_ids=list(range(prompt_len)),
        max_tokens=1,
        lora_config=LoraConfig(lora_task_id) if lora_task_id is not None else None,
        cache_salt=cache_salt,
    )
    request.py_scheduling_params = SchedulingParams(
        attention_dp_rank=target_rank, attention_dp_relax=False
    )
    return RequestQueueItem(request_id, request)


@pytest.mark.parametrize("reuse", [False, True])
def test_dkv_routes_by_new_tokens_with_distinct_rank_payloads(reuse: bool) -> None:
    group = LockstepTpGroup(2)

    def run_rank(dist):
        cache = _PrefixCache(reuse=reuse, matches={99: 99, 29: 20})
        router = ADPRouter.create(
            dist, has_seq_slot_headroom=False, kv_cache_manager=cache, dkv_enabled=True
        )
        assert isinstance(router, DefaultADPRouter)
        assert router.kv_cache_manager is cache
        active = [
            make_request(1, compute_rank=0, local_rank=dist.tp_rank, prompt_len=100),
            make_request(2, compute_rank=1, local_rank=dist.tp_rank, prompt_len=20),
        ]
        active[0].cached_tokens = 99 if reuse else 0
        direct_state = router.create_rank_state(active, [])
        assert direct_state.num_active_requests == 1
        states = router.gather_all_rank_states(
            active,
            iter_stats_payload=RankIterStatsPayload(
                has_iter_stats=1, num_ctx_tokens=dist.tp_rank + 7
            ),
        )
        assert [state.num_active_requests for state in states] == [1, 1]
        assert [state.num_active_tokens for state in states] == ([1, 20] if reuse else [100, 20])
        assert [state.iter_stats.num_ctx_tokens for state in states] == [7, 8]
        items = [_item(11, 100), _item(12, 50), _item(13, 30), _item(14, 10)]
        routed, expected = router.route_requests(states, items, max_num_active_requests=4)
        assert expected == 3
        assert router.dkv_prefix_matches == (
            [(11, 99), (12, 0), (13, 20), (14, 0)]
            if reuse
            else [(11, 0), (12, 0), (13, 0), (14, 0)]
        )
        assert len(cache.calls) == (4 if reuse else 0)
        assert [item.id for item in items] == [11, 12, 13, 14]
        router.route_requests(states, [], max_num_active_requests=4)
        assert router.dkv_prefix_matches == []
        return {rank: [item.id for item in assigned] for rank, assigned in routed.items()}

    result = group.run(run_rank)
    expected_routes = {0: [12, 11], 1: [13, 14]} if reuse else {0: [12, 14], 1: [11, 13]}
    assert result == [expected_routes, expected_routes]
    # RankState, including per-rank iteration stats, is the only router collective.
    assert [len(trace) for trace in group.traces] == [1, 1]


@pytest.mark.parametrize("cached", [0, 149])
def test_dkv_explicit_placement_charges_its_new_tokens(cached: int) -> None:
    group = LockstepTpGroup(2)

    def run_rank(dist):
        cache = _PrefixCache(matches={149: cached})
        router = ADPRouter.create(
            dist, has_seq_slot_headroom=False, kv_cache_manager=cache, dkv_enabled=True
        )
        active = [
            make_request(1, compute_rank=0, local_rank=dist.tp_rank, prompt_len=1),
            make_request(2, compute_rank=1, local_rank=dist.tp_rank, prompt_len=1),
        ]
        states = router.gather_all_rank_states(active)
        items = [_item(11, 150, target_rank=0), _item(12, 20), _item(13, 10)]
        routed, expected = router.route_requests(states, items, max_num_active_requests=4)
        assert expected == 3
        return {rank: [item.id for item in assigned] for rank, assigned in routed.items()}

    result = group.run(run_rank)
    expected_routes = {0: [11, 13], 1: [12]} if cached else {0: [11], 1: [12, 13]}
    assert result == [expected_routes, expected_routes]


def test_dkv_local_load_clips_cached_tokens_and_filters_retiring_requests() -> None:
    group = LockstepTpGroup(2)

    def run_rank(dist):
        router = ADPRouter.create(dist, has_seq_slot_headroom=True, dkv_enabled=True)
        active = [
            make_request(1, compute_rank=0, local_rank=dist.tp_rank, prompt_len=8),
            make_request(2, compute_rank=1, local_rank=dist.tp_rank, prompt_len=30),
            make_request(3, compute_rank=1, local_rank=dist.tp_rank, prompt_len=100),
        ]
        active[0].cached_tokens = 9
        active[1].cached_tokens = 10
        active[2].state = LlmRequestState.GENERATION_TO_COMPLETE
        states = router.gather_all_rank_states(active)
        return [
            (state.num_active_requests, state.num_active_tokens, state.num_retiring_requests)
            for state in states
        ]

    assert group.run(run_rank) == [[(1, 0, 0), (1, 20, 1)]] * 2


@pytest.mark.parametrize("dkv", [False, True])
def test_shared_probe_preserves_lora_salt_and_last_token_contract(dkv: bool) -> None:
    group = LockstepTpGroup(2)

    def run_rank(dist):
        cache = _PrefixCache(matches={7: 4 if dkv else 2 + dist.tp_rank})
        router = (
            ADPRouter.create(
                dist, has_seq_slot_headroom=False, kv_cache_manager=cache, dkv_enabled=True
            )
            if dkv
            else KVCacheAwareADPRouter(dist, kv_cache_manager=cache)
        )
        items = [_item(23, 8, lora_task_id=42, cache_salt="tenant-a"), _item(17, 1)]
        states = router.gather_all_rank_states([])
        if dkv:
            router.route_requests(states, items, max_num_active_requests=4)
            assert router.dkv_prefix_matches == [(23, 4), (17, 0)]
        else:
            router.gather_prefix_matches(items)
            assert router._all_ranks_prefix_matches == [{23: 2, 17: 0}, {23: 3, 17: 0}]
        assert cache.calls == [(list(range(7)), 42, "tenant-a"), ([], None, None)]

    group.run(run_rank)
    assert [len(trace) for trace in group.traces] == ([1, 1] if dkv else [2, 2])


def test_non_dkv_default_preserves_full_prompt_load_and_does_not_probe() -> None:
    group = LockstepTpGroup(2)

    def run_rank(dist):
        cache = _PrefixCache(matches={99: 99})
        router = DefaultADPRouter(dist, kv_cache_manager=cache)
        active = [make_request(dist.tp_rank + 1, prompt_len=100 if dist.tp_rank == 0 else 20)]
        active[0].cached_tokens = 99 if dist.tp_rank == 0 else 10
        states = router.gather_all_rank_states(active)
        assert [state.num_active_tokens for state in states] == [100, 20]
        routed, _ = router.route_requests(states, [_item(11, 100)], max_num_active_requests=4)
        assert [item.id for item in routed[1]] == [11]
        assert cache.calls == []

    group.run(run_rank)
