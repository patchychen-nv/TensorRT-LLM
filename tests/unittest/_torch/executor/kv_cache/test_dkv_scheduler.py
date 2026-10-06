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

"""DKV scheduling with native requests and deterministic, capacity-limited KV pages."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
from _torch.executor.dkv_test_utils import make_request

from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import (
    BlockReusePolicy,
    _settle_context_cursor,
)
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest, LlmRequestState
from tensorrt_llm._torch.pyexecutor.scheduler.scheduler_v2 import (
    BudgetTracker,
    DkvBudgetTracker,
    KVCacheV2Scheduler,
    ScheduleAction,
    _RecomputePauseState,
)
from tensorrt_llm.llmapi.llm_args import CapacitySchedulerPolicy

pytestmark = pytest.mark.cpu_only


class _PageManager:
    """A single bounded page pool; no rank-local input influences admission."""

    def __init__(
        self,
        *,
        pages: int = 4096,
        block_size: int = 64,
        prefix: dict[int, int] | None = None,
        cap: int | None = None,
        tp_rank: int = 0,
    ) -> None:
        self.pages = pages
        self.tokens_per_block = block_size
        self.tp_rank = tp_rank
        self.prefix = prefix or {}
        self.fp8_ctx_mla_kv_len_cap = cap
        self.kv_cache_map: dict[int, SimpleNamespace] = {}
        self.trace: list[tuple] = []
        self.fail_prepare: set[int] = set()
        self.fail_resize: set[int] = set()
        self.kv_connector_manager = None
        self.enable_joint_kv_cache_reuse = False
        self.enable_block_reuse = bool(prefix)
        self.block_reuse_policy = BlockReusePolicy.PER_REQUEST
        self._has_cp_helix = False
        # With a tier below the GPU the scheduler suspends requests instead of preempting them.
        self.has_cache_tier_below_gpu = True

    @property
    def used_pages(self) -> int:
        return sum(self._pages(cache.capacity) for cache in self.kv_cache_map.values())

    def _pages(self, tokens: int) -> int:
        return (tokens + self.tokens_per_block - 1) // self.tokens_per_block

    def prepare_context(self, req: LlmRequest) -> bool:
        success = req.py_request_id not in self.fail_prepare
        self.trace.append(("prepare", req.py_request_id, success))
        if not success:
            return False
        if req.py_request_id not in self.kv_cache_map:
            reuse = self.prefix.get(req.py_request_id, 0)
            self.kv_cache_map[req.py_request_id] = SimpleNamespace(capacity=0, is_active=True)
            _settle_context_cursor(req, reuse, self.tokens_per_block)
        self.kv_cache_map[req.py_request_id].is_active = True
        return True

    def resize_context(self, req: LlmRequest, tokens: int) -> bool:
        cache = self.kv_cache_map[req.py_request_id]
        capacity = max(cache.capacity, req.context_current_position + tokens)
        needed = self._pages(capacity) - self._pages(cache.capacity)
        success = (
            req.py_request_id not in self.fail_resize and self.used_pages + needed <= self.pages
        )
        self.trace.append(("resize", req.py_request_id, tokens, success))
        if success:
            req.py_ctx_pre_resize_cap = cache.capacity if capacity > cache.capacity else None
            cache.capacity = capacity
        return success

    def free_resources(self, req: LlmRequest) -> None:
        self.trace.append(("free", req.py_request_id))
        self.kv_cache_map.pop(req.py_request_id, None)

    def revert_allocate_context(self, req: LlmRequest) -> bool:
        self.trace.append(("revert", req.py_request_id))
        if req.py_ctx_pre_resize_cap is None:
            return True
        cache = self.kv_cache_map[req.py_request_id]
        cache.capacity = req.py_ctx_pre_resize_cap
        cache.is_active = False
        req.py_ctx_pre_resize_cap = None
        return True

    def is_request_active(self, request_id: int) -> bool:
        return self.kv_cache_map[request_id].is_active


def _scheduler(
    manager: _PageManager,
    *,
    group_size: int | None = 2,
    batch_size: int = 8,
    tokens: int | None = 1024,
    chunked: bool = False,
    **kwargs,
) -> KVCacheV2Scheduler:
    with patch(
        "tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2.KVCacheManagerV2",
        _PageManager,
    ):
        return KVCacheV2Scheduler(
            max_batch_size=batch_size,
            max_num_tokens=tokens,
            kv_cache_manager=manager,
            scheduler_policy=CapacitySchedulerPolicy.MAX_UTILIZATION,
            ctx_chunk_config=(None, manager.tokens_per_block) if chunked else None,
            dkv_group_size=group_size,
            **kwargs,
        )


@pytest.fixture
def dual_ledger(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TRTLLM_DKV_DUAL_LEDGER", "1")


def _ids(requests: list[LlmRequest]) -> list[int]:
    return [req.py_request_id for req in requests]


def _contexts(specs: list[tuple[int, int, int]], local_rank: int = 0) -> list[LlmRequest]:
    return [
        make_request(request_id, compute_rank=rank, local_rank=local_rank, prompt_len=length)
        for request_id, rank, length in specs
    ]


def test_budget_views_stop_only_one_rank_and_share_peft() -> None:
    peft = SimpleNamespace(max_device_pages=3, determine_num_pages=lambda req: 2)
    budget = DkvBudgetTracker(256, 2, 2, peft)
    a, b = SimpleNamespace(lora_task_id=1), SimpleNamespace(lora_task_id=2)
    rank0, rank1 = budget.view(0), budget.view(1)
    assert rank0.stop_action is ScheduleAction.STOP_RANK
    assert BudgetTracker.stop_action is ScheduleAction.STOP
    assert rank0.peft_pages_needed(a) == 2
    rank0.commit(a, 256, 2)
    assert rank1.peft_pages_needed(a) == 0
    assert rank1.peft_pages_needed(b) is None
    rank1.commit(b, 128, 0)
    assert budget.rank_tokens == [256, 128]
    assert budget.rank_requests == [1, 1]
    assert rank0.remaining_tokens == 0
    assert not rank0.can_fit_tokens(1)
    assert rank1.can_fit_tokens(128)
    budget.stop_rank(0)
    assert rank0.requests_full and not budget.all_ranks_full
    budget.stop_rank(1)
    assert budget.all_ranks_full


def test_unlimited_budget_and_preclaimed_peft_are_shared() -> None:
    peft = SimpleNamespace(max_device_pages=1, determine_num_pages=lambda req: 1)
    budget = DkvBudgetTracker(None, 1, 2, peft)
    req = SimpleNamespace(lora_task_id=42)
    budget.view(0).pre_claim_peft(req)
    assert budget.view(1).peft_pages_needed(req) == 0
    assert budget.view(0).remaining_tokens is None
    assert budget.view(0).can_fit_tokens(10**9)
    budget.view(0).commit(req, 10**9, 0)
    assert budget.view(0).requests_full
    assert not budget.all_ranks_full


@pytest.mark.parametrize("rank", [-1, 2, None])
def test_budget_rejects_invalid_compute_rank(rank) -> None:
    with pytest.raises(ValueError, match="Invalid DKV compute rank"):
        DkvBudgetTracker(32, 2, 2).view(rank)


@pytest.mark.parametrize("batch_size", [1, 2])
def test_full_context_uses_independent_token_and_request_budgets(dual_ledger, batch_size) -> None:
    manager = _PageManager()
    scheduler = _scheduler(manager, batch_size=batch_size, tokens=1024, scheduler_capacity=1)
    requests = _contexts([(0, 0, 600), (1, 1, 600)])
    assert _ids(scheduler.schedule_request(requests, set()).context_requests) == [0, 1]


def test_design_example_stops_rank_without_preparing_later_request(dual_ledger) -> None:
    manager = _PageManager()
    scheduler = _scheduler(manager, tokens=256)
    requests = _contexts([(0, 0, 256), (1, 1, 128), (2, 0, 256), (3, 0, 64), (4, 1, 64)])
    output = scheduler.schedule_request(requests, set())
    assert _ids(output.context_requests) == [0, 1, 4]
    assert [item[1] for item in manager.trace if item[0] == "prepare"] == [0, 1, 2, 4]
    assert requests[2].context_current_position == 0
    assert requests[2].py_request_id not in manager.kv_cache_map


@pytest.mark.parametrize("prefix_aware", [False, True])
def test_nonchunked_compute_failure_stops_rank_but_prepare_failure_stops_globally(
    dual_ledger, prefix_aware
) -> None:
    specs = [(0, 0, 192), (1, 0, 128), (2, 1, 128)]
    manager = _PageManager()
    scheduler = _scheduler(manager, tokens=256, enable_prefix_aware_scheduling=prefix_aware)
    assert _ids(scheduler.schedule_request(_contexts(specs), set()).context_requests) == [0, 2]
    manager = _PageManager()
    manager.fail_prepare.add(1)
    scheduler = _scheduler(manager, tokens=512, enable_prefix_aware_scheduling=prefix_aware)
    assert _ids(scheduler.schedule_request(_contexts(specs), set()).context_requests) == [0]
    assert not any(item[1] == 2 for item in manager.trace)


@pytest.mark.parametrize("failure", ["prepare", "resize"])
def test_chunked_kv_failure_skips_without_stopping_other_ranks(dual_ledger, failure) -> None:
    manager = _PageManager()
    getattr(manager, "fail_" + failure).add(0)
    scheduler = _scheduler(manager, tokens=128, chunked=True)
    requests = _contexts([(0, 0, 256), (1, 1, 256), (2, 0, 64)])
    assert _ids(scheduler.schedule_request(requests, set()).context_requests) == [1, 2]
    assert 0 not in manager.kv_cache_map
    assert requests[0].context_current_position == 0


def test_nonchunked_resize_failure_skips_later_requests(dual_ledger) -> None:
    manager = _PageManager()
    manager.fail_resize.add(0)
    output = _scheduler(manager).schedule_request(_contexts([(0, 0, 64), (1, 1, 64)]), set())
    assert _ids(output.context_requests) == [1]


def test_chunk_budget_precheck_uses_each_rank(dual_ledger) -> None:
    manager = _PageManager()
    scheduler = _scheduler(manager, tokens=128, chunked=True)
    requests = _contexts([(0, 0, 128), (1, 0, 128), (2, 1, 128)])
    assert _ids(scheduler.schedule_request(requests, set()).context_requests) == [0, 2]
    assert not any(item[1] == 1 for item in manager.trace)


def test_preparation_credits_cached_prefix_before_charging(dual_ledger) -> None:
    manager = _PageManager(prefix={0: 512, 1: 512})
    scheduler = _scheduler(manager, tokens=128)
    requests = _contexts([(0, 0, 576), (1, 1, 576)])
    output = scheduler.schedule_request(requests, set())
    assert _ids(output.context_requests) == [0, 1]
    assert [req.context_chunk_size for req in requests] == [64, 64]
    assert [req.context_current_position for req in requests] == [512, 512]


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize(
    "state",
    [
        LlmRequestState.ENCODER_INIT,
        LlmRequestState.DISAGG_GENERATION_INIT,
        LlmRequestState.DISAGG_GENERATION_TRANS_IN_PROGRESS,
        LlmRequestState.DISAGG_GENERATION_TRANS_COMPLETE,
        LlmRequestState.GENERATION_IN_PROGRESS,
    ],
)
def test_dkv_guard_runs_with_dual_ledger_off_or_on(monkeypatch, enabled, state) -> None:
    monkeypatch.setenv("TRTLLM_DKV_DUAL_LEDGER", str(int(enabled)))
    manager = _PageManager()
    scheduler = _scheduler(manager)
    request = make_request(0)
    request.state = state
    with pytest.raises(RuntimeError, match="DKV prefill-only"):
        scheduler.schedule_request([request], {0})
    assert manager.trace == []


@pytest.mark.parametrize(
    "state",
    [
        LlmRequestState.GENERATION_TO_COMPLETE,
        LlmRequestState.GENERATION_COMPLETE,
        LlmRequestState.DISAGG_CONTEXT_TRANS_IN_PROGRESS,
    ],
)
def test_retiring_and_context_transfer_requests_are_not_decode(dual_ledger, state) -> None:
    manager = _PageManager()
    request = make_request(0)
    request.state = state
    output = _scheduler(manager).schedule_request([request], set())
    assert not output.context_requests and not output.generation_requests
    assert manager.trace == []


def test_generation_dummy_is_exempt_from_guard_and_dual_budget(dual_ledger) -> None:
    manager = _PageManager()
    scheduler = _scheduler(manager, batch_size=1)
    dummy = make_request(0, compute_rank=-1, is_dummy=True)
    dummy.state = LlmRequestState.GENERATION_IN_PROGRESS
    with patch.object(scheduler, "_try_allocate_generation", return_value=True):
        output = scheduler.schedule_request(
            [dummy, make_request(1), make_request(2, compute_rank=1)], set()
        )
    assert _ids(output.generation_requests) == [0]
    assert _ids(output.context_requests) == [1, 2]


def test_switch_defaults_on_with_explicit_off_fallback_and_plain_adp_unchanged(monkeypatch) -> None:
    monkeypatch.delenv("TRTLLM_DKV_DUAL_LEDGER", raising=False)
    scheduler = _scheduler(_PageManager(), tokens=1024)
    assert scheduler.dkv_dual_ledger_enabled
    assert _ids(
        scheduler.schedule_request(_contexts([(0, 0, 600), (1, 1, 600)]), set()).context_requests
    ) == [0, 1]
    monkeypatch.setenv("TRTLLM_DKV_DUAL_LEDGER", "0")
    scheduler = _scheduler(_PageManager(), tokens=1024)
    assert not scheduler.dkv_dual_ledger_enabled
    assert _ids(
        scheduler.schedule_request(_contexts([(0, 0, 600), (1, 1, 600)]), set()).context_requests
    ) == [0]
    monkeypatch.setenv("TRTLLM_DKV_DUAL_LEDGER", "1")
    assert not _scheduler(_PageManager(), group_size=None).dkv_dual_ledger_enabled


def test_generation_and_encoder_budget_failures_use_view_stop_action(dual_ledger) -> None:
    scheduler = _scheduler(_PageManager(), cross_kv_cache_manager=object())
    request = make_request(0)
    rank_view = DkvBudgetTracker(0, 1, 2).view(0)
    encoder = SimpleNamespace(encoder_output_len=1, py_request_id=0)
    assert scheduler._try_schedule_encoder(encoder, rank_view)[0] is ScheduleAction.STOP_RANK
    request.state = LlmRequestState.GENERATION_IN_PROGRESS
    result = scheduler._try_schedule_generation(
        request, rank_view, [request], 0, 1, _RecomputePauseState(1), [], [], set(), 0
    )
    assert result[0] is ScheduleAction.STOP_RANK


@pytest.mark.parametrize("cap", [0, 128])
def test_attended_kv_cap_is_per_rank_and_keeps_first_request(dual_ledger, cap) -> None:
    manager = _PageManager(cap=cap)
    scheduler = _scheduler(manager)
    requests = _contexts([(0, 0, 128), (1, 0, 64), (2, 1, 128), (3, 0, 64)])
    assert _ids(scheduler.schedule_request(requests, set()).context_requests) == [0, 2]
    assert 1 not in manager.kv_cache_map
    assert not any(item[1] == 3 for item in manager.trace)
    assert requests[1].context_current_position == 0


def test_attended_cap_rejects_cached_first_chunk_and_releases_prefix(dual_ledger) -> None:
    manager = _PageManager(prefix={1: 128}, cap=192)
    scheduler = _scheduler(manager)
    requests = _contexts([(0, 0, 128), (1, 0, 192), (2, 1, 64)])
    assert _ids(scheduler.schedule_request(requests, set()).context_requests) == [0, 2]
    rejected = requests[1]
    assert rejected.is_first_context_chunk
    assert rejected.context_current_position == 0
    assert rejected.context_remaining_length == 192
    assert rejected.context_chunk_size == 192
    assert rejected.estimated_reusable_tokens == 0
    assert rejected.py_ctx_pre_resize_cap is None
    assert ("free", 1) in manager.trace


def test_attended_cap_reverts_continuation_without_rewinding_executed_history(dual_ledger) -> None:
    manager = _PageManager(cap=128)
    scheduler = _scheduler(manager, tokens=256, chunked=True)
    requests = _contexts([(0, 0, 128), (1, 0, 256), (2, 1, 128)])
    continuation = requests[1]
    manager.prepare_context(continuation)
    manager.resize_context(continuation, 64)
    continuation.context_current_position = 64
    manager.trace.clear()
    assert not continuation.is_first_context_chunk
    assert _ids(scheduler.schedule_request(requests, set()).context_requests) == [0, 2]
    assert ("revert", 1) in manager.trace and ("free", 1) not in manager.trace
    assert continuation.context_current_position == 64
    assert manager.kv_cache_map[1].capacity == 64


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("suspended", [False, True])
def test_empty_prefill_recovers_by_rewinding_last_started_request(
    monkeypatch, enabled, suspended
) -> None:
    monkeypatch.setenv("TRTLLM_DKV_DUAL_LEDGER", str(int(enabled)))
    manager = _PageManager(pages=2)
    scheduler = _scheduler(manager, tokens=64, chunked=True)
    requests = _contexts([(5, 0, 128), (2, 1, 128)])
    for request in requests:
        manager.prepare_context(request)
        manager.resize_context(request, 64)
        request.context_current_position = 64
        if suspended:
            manager.kv_cache_map[request.py_request_id].is_active = False
            manager.fail_prepare.add(request.py_request_id)
    manager.trace.clear()
    output = scheduler.schedule_request(requests, set())
    assert output.context_requests == []
    assert output.recompute_paused_requests == []
    assert manager.trace[-1] == ("free", 2)
    assert requests[1].context_current_position == 0
    assert requests[1].context_chunk_size == 128
    assert requests[0].context_current_position == 64
    manager.fail_prepare.clear()
    assert _ids(scheduler.schedule_request(requests, set()).context_requests) == [5]


def test_repeated_stall_recoveries_are_counted_and_warned_on_each_doubling() -> None:
    manager = _PageManager(pages=2)
    scheduler = _scheduler(manager, tokens=64, chunked=True)
    requests = _contexts([(5, 0, 128), (2, 1, 128)])

    def stall_again() -> None:
        for request in requests:
            manager.fail_prepare.discard(request.py_request_id)
            if request.py_request_id not in manager.kv_cache_map:
                manager.prepare_context(request)
                manager.resize_context(request, 64)
                request.context_current_position = 64
            manager.kv_cache_map[request.py_request_id].is_active = False
            manager.fail_prepare.add(request.py_request_id)
        assert scheduler.schedule_request(requests, set()).context_requests == []

    with patch("tensorrt_llm._torch.pyexecutor.scheduler.scheduler_v2.logger.warning") as warning:
        for _ in range(4):
            stall_again()
        assert scheduler._dkv_stall_recoveries == {2: 4}
        messages = [call.args[0] for call in warning.call_args_list]
        assert [message.split(":")[0] for message in messages] == [
            "DKV stall recovery #1",
            "DKV stall recovery #2",
            "DKV stall recovery #4",
        ]
        assert all("request 2 released its KV cache" in message for message in messages)
        # A request that left the active set no longer holds a counter.
        requests.pop()
        manager.fail_prepare.discard(5)
        manager.kv_cache_map[5].is_active = False
        manager.fail_prepare.add(5)
        scheduler.schedule_request(requests, set())
    assert scheduler._dkv_stall_recoveries == {5: 1}


@pytest.mark.parametrize("waiting", ["inflight", "connector", "transfer", "unstarted"])
def test_empty_batch_does_not_rewind_asynchronous_or_unstarted_work(dual_ledger, waiting) -> None:
    manager = _PageManager(pages=1)
    scheduler = _scheduler(manager, tokens=64, chunked=True)
    request = make_request(0, prompt_len=128)
    manager.prepare_context(request)
    manager.resize_context(request, 64)
    request.context_current_position = 64
    inflight = {0} if waiting == "inflight" else set()
    if waiting == "connector":
        manager.kv_connector_manager = SimpleNamespace(has_pending_load=lambda req: True)
    elif waiting == "transfer":
        request.state = LlmRequestState.DISAGG_CONTEXT_TRANS_IN_PROGRESS
    elif waiting == "unstarted":
        _settle_context_cursor(request, 64, manager.tokens_per_block)
        assert request.is_first_context_chunk
        inflight = {0}
    manager.trace.clear()
    scheduler.schedule_request([request], inflight)
    assert ("free", 0) not in manager.trace


@pytest.mark.parametrize("group_size", [2, 4, 8])
@pytest.mark.parametrize("chunked", [False, True])
def test_replica_replay_matches_independent_adp_views(dual_ledger, group_size, chunked) -> None:
    specs = [(index, index % group_size, 64 + (index % 3) * 64) for index in range(group_size * 4)]
    traces = []
    outputs = []
    for tp_rank in range(group_size):
        manager = _PageManager(tp_rank=tp_rank)
        scheduler = _scheduler(
            manager, group_size=group_size, batch_size=2, tokens=256, chunked=chunked
        )
        requests = _contexts(specs, tp_rank)
        rounds = []
        while requests:
            scheduled = scheduler.schedule_request(requests, set()).context_requests
            assert scheduled
            rounds.append(
                [
                    (req.py_request_id, req.context_current_position, req.context_chunk_size)
                    for req in scheduled
                ]
            )
            for req in scheduled:
                req.context_current_position += req.context_chunk_size
                if req.context_remaining_length == 0:
                    manager.free_resources(req)
                    requests.remove(req)
            rounds[-1].append(("pages", manager.used_pages))
        traces.append(manager.trace)
        outputs.append(rounds)
    assert all(trace == traces[0] for trace in traces)
    assert all(output == outputs[0] for output in outputs)
    local_rounds = []
    for rank in range(group_size):
        manager = _PageManager(tp_rank=rank)
        scheduler = _scheduler(manager, group_size=None, batch_size=2, tokens=256, chunked=chunked)
        requests = _contexts([spec for spec in specs if spec[1] == rank], rank)
        rounds = []
        while requests:
            scheduled = scheduler.schedule_request(requests, set()).context_requests
            assert scheduled
            rounds.append(
                [
                    (req.py_request_id, req.context_current_position, req.context_chunk_size)
                    for req in scheduled
                ]
            )
            for req in scheduled:
                req.context_current_position += req.context_chunk_size
                if req.context_remaining_length == 0:
                    manager.free_resources(req)
                    requests.remove(req)
        local_rounds.append(rounds)
    for iteration, global_round in enumerate(outputs[0]):
        global_round = global_round[:-1]
        views = []
        for rank in range(group_size):
            expected = [entry for entry in global_round if entry[0] % group_size == rank]
            actual = local_rounds[rank][iteration] if iteration < len(local_rounds[rank]) else []
            assert actual == expected
            local_ids = {entry[0] for entry in actual}
            assert not any(local_ids & previous for previous in views)
            views.append(local_ids)
        assert set().union(*views) == {entry[0] for entry in global_round}
    assert max(map(len, local_rounds)) == len(outputs[0])
