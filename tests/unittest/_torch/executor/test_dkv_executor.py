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
"""DKV admission, replicated activation and startup invariants."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
from dkv_test_utils import LockstepTpGroup, make_request

from tensorrt_llm._torch.pyexecutor.dkv import DkvControlResult, DkvInvariantChecker
from tensorrt_llm._torch.pyexecutor.executor_request_queue import RequestQueueItem
from tensorrt_llm._torch.pyexecutor.llm_request import ExecutorRequest, LlmRequest, SamplingConfig
from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor
from tensorrt_llm._torch.pyexecutor.request_utils import merge_requests_to_llm_requests
from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests, SchedulerOutput
from tensorrt_llm.bindings.internal.batch_manager import LlmRequestType
from tensorrt_llm.disaggregated_params import DisaggregatedParams
from tensorrt_llm.llmapi import DisaggScheduleStyle

pytestmark = pytest.mark.cpu_only


def _queue_item(request_id: int, *, children: list[int] | None = None) -> RequestQueueItem:
    request = ExecutorRequest(
        input_token_ids=[1, 2, 3],
        max_tokens=1,
        sampling_config=SamplingConfig(beam_width=1, num_return_sequences=1 + len(children or [])),
    )
    return RequestQueueItem(request_id, request, child_req_ids=children)


def _fetch_executor(dist, items, *, dkv_enabled: bool = True, routes=None) -> PyExecutor:
    executor = PyExecutor.__new__(PyExecutor)
    executor.dist = dist
    executor.dkv_enabled = dkv_enabled
    executor.enable_attention_dp = True
    executor.enable_iter_perf_stats = False
    executor._enable_non_overlap_adp_forward_intent = True
    executor.num_fetch_requests = 0
    executor.num_fetch_requests_cur_rank = 0
    executor.max_num_active_requests = 8
    executor.iter_counter = 0
    executor._fetch_and_enqueue_requests = Mock()
    executor._pop_from_waiting_queue = Mock(return_value=items)
    executor._should_exclude_last_generation_logits = Mock(return_value=False)
    executor.kv_cache_manager = SimpleNamespace(consume_dkv_trace=Mock(return_value=[]))
    if routes is None:
        routes = {1: [items[0]], 0: [items[1]]}

    def gather_states(active, *, iter_stats_payload=None):
        return dist.tp_allgather(SimpleNamespace(num_active_requests=0, num_retiring_requests=0))

    executor.adp_router = SimpleNamespace(
        gather_all_rank_states=gather_states,
        route_requests=lambda states, new, capacity: (routes, 1),
        needs_prefix_matches=False,
        dkv_prefix_matches=[(item.id, 0) for item in items],
    )
    executor._dkv_invariant_checker = (
        DkvInvariantChecker(dist, enabled=True) if dkv_enabled else None
    )
    return executor


@pytest.mark.parametrize("dkv_enabled", [False, True])
def test_fetch_activates_replicas_and_preserves_queue_order(dkv_enabled: bool) -> None:
    group = LockstepTpGroup(2)

    def fetch(dist):
        executor = _fetch_executor(
            dist, [_queue_item(10), _queue_item(20)], dkv_enabled=dkv_enabled
        )
        requests = executor._fetch_new_requests(None, [])
        return [
            (request.py_request_id, request.py_dkv_compute_rank, request.py_dkv_is_local)
            for request in requests
        ]

    results = group.run(fetch)
    if dkv_enabled:
        assert results == [[(10, 1, False), (20, 0, True)], [(10, 1, True), (20, 0, False)]]
    else:
        assert [[item[0] for item in result] for result in results] == [[20], [10]]
        assert all(item[1] is None for result in results for item in result)
    assert group.traces[0] == group.traces[1]


def test_fetch_preserves_fifo_despite_rank_local_routing_reorder() -> None:
    def fetch(dist):
        items = [_queue_item(request_id) for request_id in [40, 10, 30, 20]]
        routes = {1: [items[3], items[0]], 0: [items[2], items[1]]}
        executor = _fetch_executor(dist, items, routes=routes)
        requests = executor._fetch_new_requests(None, [])
        executor.kv_cache_manager.consume_dkv_trace.assert_not_called()
        return [(request.py_request_id, request.py_dkv_compute_rank) for request in requests]

    assert LockstepTpGroup(2).run(fetch) == [[(40, 1), (10, 0), (30, 0), (20, 1)]] * 2


def test_queue_tag_propagates_to_parent_and_children() -> None:
    item = _queue_item(10, children=[11])
    item.dkv_compute_rank = 1
    requests = merge_requests_to_llm_requests([item], exclude_last_generation_logits=False)
    assert [request.py_request_id for request in requests] == [10, 11]
    assert [request.py_dkv_compute_rank for request in requests] == [1, 1]


def test_queue_tag_preserves_existing_positional_arguments() -> None:
    item = RequestQueueItem(10, None, [11], True, dkv_compute_rank=1)
    assert item.child_req_ids == [11]
    assert item.is_canceled_request is True
    assert item.dkv_compute_rank == 1


@pytest.mark.parametrize(
    "route_error", ["missing", "duplicate", "foreign_identity", "invalid_rank", "bool_rank"]
)
def test_fetch_rejects_invalid_routing(route_error: str) -> None:
    def fetch(dist):
        items = [_queue_item(10), _queue_item(20)]
        routes = {0: [items[0]], 1: [items[1]]}
        if route_error == "missing":
            routes[1] = []
        elif route_error == "duplicate":
            routes[1] = [items[0]]
        elif route_error == "foreign_identity":
            routes[1] = [_queue_item(20)]
        elif route_error == "bool_rank":
            routes = {0: [items[0]], True: [items[1]]}
        else:
            routes = {0: [items[0]], 1: [], 2: [items[1]]}
        executor = _fetch_executor(dist, items, routes=routes)
        executor._fetch_new_requests(None, [])

    with pytest.raises(RuntimeError, match="DKV routing|Invalid DKV compute rank"):
        LockstepTpGroup(2).run(fetch)


def test_fetch_rejects_lost_compute_rank() -> None:
    def lose_labels(items, **kwargs):
        requests = merge_requests_to_llm_requests(items, exclude_last_generation_logits=False)
        requests[0].py_dkv_compute_rank = None
        return requests

    def fetch(dist):
        executor = _fetch_executor(dist, [_queue_item(10), _queue_item(20)])
        executor._fetch_new_requests(None, [])

    with patch(
        "tensorrt_llm._torch.pyexecutor.py_executor.merge_requests", side_effect=lose_labels
    ):
        with pytest.raises(RuntimeError, match="has no compute rank"):
            LockstepTpGroup(2).run(fetch)


def test_different_rank_assignments_are_detected() -> None:
    def fetch(dist):
        items = [_queue_item(10), _queue_item(20)]
        routes = {0: [items[dist.tp_rank]], 1: [items[1 - dist.tp_rank]]}
        executor = _fetch_executor(dist, items, routes=routes)
        executor._fetch_new_requests(None, [])

    with pytest.raises(RuntimeError, match="global request order"):
        LockstepTpGroup(2).run(fetch)


@pytest.mark.parametrize("mismatch", ["prefix_length", "request_order"])
def test_prefix_probe_divergence_is_detected(mismatch: str) -> None:
    def fetch(dist):
        executor = _fetch_executor(dist, [_queue_item(10), _queue_item(20)])
        if dist.tp_rank:
            if mismatch == "prefix_length":
                executor.adp_router.dkv_prefix_matches[0] = (10, 2)
            else:
                executor.adp_router.dkv_prefix_matches.reverse()
        executor._fetch_new_requests(None, [])

    with pytest.raises(RuntimeError, match="prefix probes"):
        LockstepTpGroup(2).run(fetch)


@pytest.mark.parametrize("different_capacity", [False, True])
def test_startup_capacity_is_checked(monkeypatch, different_capacity: bool) -> None:
    monkeypatch.setenv("TRTLLM_DKV_DEBUG", "1")

    def initialize(dist):
        executor = PyExecutor.__new__(PyExecutor)
        executor.dist = dist
        capacity = 8 + (dist.tp_rank if different_capacity else 0)
        executor.kv_cache_manager = SimpleNamespace(
            get_dkv_config_fingerprint=lambda: [("max_admissible_sequences", capacity)],
            get_dkv_startup_settings=lambda: {},
        )
        executor.scheduler = SimpleNamespace(dkv_dual_ledger_enabled=False)
        executor._initialize_dkv_invariant_checker()
        return executor._dkv_invariant_checker.enabled

    if different_capacity:
        with pytest.raises(RuntimeError, match="startup configuration"):
            LockstepTpGroup(2).run(initialize)
    else:
        assert LockstepTpGroup(2).run(initialize) == [True, True]


@pytest.mark.parametrize("debug", ["0", "1"])
def test_different_startup_switches_are_detected_without_waiting_for_a_check(
    monkeypatch, debug: str
) -> None:
    monkeypatch.setenv("TRTLLM_DKV_DEBUG", debug)

    def initialize(dist):
        executor = PyExecutor.__new__(PyExecutor)
        executor.dist = dist
        executor.kv_cache_manager = SimpleNamespace(
            get_dkv_config_fingerprint=lambda: [], get_dkv_startup_settings=lambda: {}
        )
        executor.scheduler = SimpleNamespace(dkv_dual_ledger_enabled=bool(dist.tp_rank))
        executor._initialize_dkv_invariant_checker()

    group = LockstepTpGroup(2)
    with pytest.raises(RuntimeError, match="startup settings differ.*dual_ledger"):
        group.run(initialize)
    assert [len(trace) for trace in group.traces] == [1, 1]


def test_startup_settings_cover_every_process_level_switch(monkeypatch) -> None:
    monkeypatch.setenv("TLLM_METRICS_ALL_RANKS", "1")
    recorded = {}

    class _RecordingChecker:
        enabled = False

        def __init__(self, dist, enabled=None, startup_settings=None) -> None:
            recorded.update(startup_settings)

        def check_many(self, iter_counter, items_by_tag) -> None:
            pass

    monkeypatch.setattr(
        "tensorrt_llm._torch.pyexecutor.py_executor.DkvInvariantChecker", _RecordingChecker
    )
    executor = PyExecutor.__new__(PyExecutor)
    executor.dist = None
    executor.kv_cache_manager = SimpleNamespace(
        get_dkv_config_fingerprint=lambda: [],
        get_dkv_startup_settings=lambda: {"kv_cache_backend": "py"},
    )
    executor.scheduler = SimpleNamespace(dkv_dual_ledger_enabled=True)
    executor._initialize_dkv_invariant_checker()
    assert recorded == {"dual_ledger": True, "metrics_all_ranks": True, "kv_cache_backend": "py"}


@pytest.mark.parametrize("layer_split", [False, True])
def test_layer_split_ranks_must_agree_on_the_ownership_table(layer_split: bool) -> None:
    recorded = []

    def initialize(dist):
        executor = PyExecutor.__new__(PyExecutor)
        executor.dist = dist
        executor.dkv_layer_split = layer_split
        # The ranks disagree about how many layers the model has.
        executor.kv_cache_manager = SimpleNamespace(
            num_layers=43 + dist.tp_rank,
            get_dkv_config_fingerprint=lambda: [],
            get_dkv_startup_settings=lambda: {},
        )
        executor.scheduler = SimpleNamespace(dkv_dual_ledger_enabled=True)
        executor._initialize_dkv_invariant_checker()
        recorded.append(dist.tp_rank)

    if layer_split:
        with pytest.raises(RuntimeError, match="startup settings differ.*layer_ownership"):
            LockstepTpGroup(2).run(initialize)
    else:
        # The replicated layout has no table to compare.
        LockstepTpGroup(2).run(initialize)
        assert sorted(recorded) == [0, 1]


def test_layer_split_records_the_ownership_table_in_the_startup_settings(monkeypatch) -> None:
    from tensorrt_llm._torch.pyexecutor.dkv import compute_ownership, ownership_fingerprint

    recorded = {}

    class _RecordingChecker:
        enabled = False

        def __init__(self, dist, enabled=None, startup_settings=None) -> None:
            recorded.update(startup_settings)

        def check_many(self, iter_counter, items_by_tag) -> None:
            pass

    monkeypatch.setattr(
        "tensorrt_llm._torch.pyexecutor.py_executor.DkvInvariantChecker", _RecordingChecker
    )
    executor = PyExecutor.__new__(PyExecutor)
    executor.dist = SimpleNamespace(mapping=SimpleNamespace(tp_size=4))
    executor.dkv_layer_split = True
    executor.kv_cache_manager = SimpleNamespace(
        num_layers=43,
        get_dkv_config_fingerprint=lambda: [],
        get_dkv_startup_settings=lambda: {},
    )
    executor.scheduler = SimpleNamespace(dkv_dual_ledger_enabled=True)
    executor._initialize_dkv_invariant_checker()
    assert recorded["layer_ownership"] == ownership_fingerprint(compute_ownership(43, 4))


def _staging_executor(
    max_batch_size: int = 4, max_seq_len: int = 1000, max_num_tokens: int = 300
) -> PyExecutor:
    executor = PyExecutor.__new__(PyExecutor)
    executor.kv_cache_manager = SimpleNamespace(
        max_batch_size=max_batch_size, max_seq_len=max_seq_len, max_num_tokens=max_num_tokens
    )
    return executor


@pytest.mark.parametrize(
    "shape,staging_tokens",
    [
        # A request of max_seq_len next to a full chunk.
        ((4, 1000, 300), 1300),
        # Not more than every request of the batch at max_seq_len.
        ((1, 1000, 300), 1000),
        # At least one full chunk.
        ((1, 100, 300), 300),
    ],
)
def test_the_default_staging_area_holds_a_long_request_next_to_a_chunk(
    monkeypatch, shape, staging_tokens
) -> None:
    monkeypatch.delenv("TRTLLM_DKV_STAGING_TOKENS", raising=False)
    assert _staging_executor(*shape)._dkv_staging_settings()["staging_tokens"] == staging_tokens


def test_a_staging_area_below_one_request_of_max_seq_len_is_rejected(monkeypatch) -> None:
    executor = _staging_executor()
    monkeypatch.setenv("TRTLLM_DKV_STAGING_TOKENS", "1000")
    assert executor._dkv_staging_settings()["staging_tokens"] == 1000
    monkeypatch.setenv("TRTLLM_DKV_STAGING_TOKENS", "999")
    with pytest.raises(ValueError, match="one request of max_seq_len"):
        executor._dkv_staging_settings()


@pytest.mark.parametrize("checks", [False, True])
def test_the_plan_is_fingerprinted_only_when_the_checks_are_on(monkeypatch, checks: bool) -> None:
    fingerprinted = []
    checked = []
    monkeypatch.setattr(
        "tensorrt_llm._torch.pyexecutor.py_executor.plan_fingerprint",
        lambda plan: fingerprinted.append(plan) or "fingerprint",
    )
    executor = PyExecutor.__new__(PyExecutor)
    executor._dkv_invariant_checker = SimpleNamespace(
        enabled=checks, check=lambda *args: checked.append(args)
    )
    executor.dkv_streamer = SimpleNamespace(plan_for=lambda requests: "plan")
    executor._dkv_forward_dummies = []
    executor.iter_counter = 3
    batch = SimpleNamespace(context_requests=[], generation_requests=[])
    assert executor._dkv_plan(batch) == "plan"
    assert fingerprinted == (["plan"] if checks else [])
    assert checked == ([(3, "data plane plan", "fingerprint")] if checks else [])


def test_the_staged_view_publishes_its_token_budget_to_the_scheduler() -> None:
    executor = _staging_executor()
    executor.model_engine = SimpleNamespace()
    executor._install_dkv_staged_view("view", {"staging_tokens": 1300})
    assert executor.kv_cache_manager.dkv_staged_view == "view"
    assert executor.kv_cache_manager.dkv_max_staging_tokens == 1300


def _scheduled_executor(dist, *, trace, requests, checker_enabled=True) -> PyExecutor:
    executor = PyExecutor.__new__(PyExecutor)
    executor.dist = dist
    executor.dkv_enabled = True
    executor.iter_counter = 3
    executor.is_shutdown = False
    executor.enable_iter_perf_stats = False
    executor.drafter = None
    executor.model_engine = SimpleNamespace(is_spec_decode=False)
    executor.kv_cache_transceiver = None
    executor.active_requests = requests
    executor._disagg_coordinator = Mock()
    executor._release_unused_connector_reservations = Mock()
    executor._poll_encoder_steps = Mock()
    executor._fetch_and_activate_new_requests = Mock(return_value=[])
    executor._prefetch_for_context_requests = Mock()
    executor.kv_cache_manager = SimpleNamespace(consume_dkv_trace=Mock(return_value=trace))
    executor._dkv_invariant_checker = DkvInvariantChecker(dist, enabled=checker_enabled)
    scheduled = ScheduledRequests()
    scheduled.reset_context_requests(requests)
    executor._schedule = Mock(return_value=(scheduled, [], len(requests)))
    return executor


@pytest.mark.parametrize("mismatch", [None, "request_order", "chunk", "kv_order", "kv_result"])
def test_scheduling_decisions_and_ordered_kv_trace_are_checked(mismatch: str | None) -> None:
    def schedule(dist):
        requests = [make_request(8, compute_rank=1), make_request(3, compute_rank=0)]
        for request in requests:
            request.context_chunk_size = 4
        trace = [("prepare", 8, 0), ("resize", 8, 4, True), ("revert_allocate_context", 3)]
        if dist.tp_rank:
            if mismatch == "request_order":
                requests.reverse()
            elif mismatch == "chunk":
                requests[0].context_chunk_size = 5
            elif mismatch == "kv_order":
                trace.reverse()
            elif mismatch == "kv_result":
                trace[1] = ("resize", 8, 4, False)
        executor = _scheduled_executor(dist, trace=trace, requests=requests)
        batch, _ = executor._prepare_and_schedule_batch()
        executor.kv_cache_manager.consume_dkv_trace.assert_called_once_with()
        return [request.py_request_id for request in batch.all_requests()]

    if mismatch is None:
        assert LockstepTpGroup(2).run(schedule) == [[8, 3]] * 2
    else:
        with pytest.raises(RuntimeError, match="scheduling decisions"):
            LockstepTpGroup(2).run(schedule)


def test_kv_trace_is_drained_when_debug_is_disabled() -> None:
    def schedule(dist):
        executor = _scheduled_executor(dist, trace=[], requests=[], checker_enabled=False)
        executor._prepare_and_schedule_batch()
        executor.kv_cache_manager.consume_dkv_trace.assert_called_once_with()

    group = LockstepTpGroup(2)
    group.run(schedule)
    assert [len(trace) for trace in group.traces] == [1, 1]


@pytest.mark.parametrize("dkv_enabled,dual_ledger", [(False, False), (True, False), (True, True)])
def test_global_context_cap_is_preserved_until_dual_ledger_enabled(
    dkv_enabled: bool, dual_ledger: bool
) -> None:
    executor = PyExecutor.__new__(PyExecutor)
    executor.dkv_enabled = dkv_enabled
    requests = [make_request(1), make_request(2, compute_rank=1)]
    for request in requests:
        request.context_chunk_size = 8
    executor.active_requests = requests
    executor.inflight_req_ids = set()
    executor.kv_cache_manager = SimpleNamespace()
    executor.scheduler = SimpleNamespace(
        dkv_dual_ledger_enabled=dual_ledger,
        schedule_request=Mock(return_value=SchedulerOutput([], requests, [], [], [], 2)),
    )
    executor.is_encoder_decoder = False
    executor.enable_attention_dp = True
    executor.attention_dp_enable_balance = False
    executor._cap_context_by_total_kv_len = Mock(return_value=requests[:1])
    batch, _, _ = executor._schedule()
    if dual_ledger:
        executor._cap_context_by_total_kv_len.assert_not_called()
        assert batch.context_requests == requests
    else:
        executor._cap_context_by_total_kv_len.assert_called_once_with(requests)
        assert batch.context_requests == requests[:1]


def _request_executor(*, dkv_enabled: bool = True) -> PyExecutor:
    executor = PyExecutor.__new__(PyExecutor)
    executor.dkv_enabled = dkv_enabled
    executor.kv_cache_transceiver = None
    executor.max_beam_width = 1
    executor._validate_token_id_range = Mock()
    executor._validate_request_budget = Mock()
    executor.sampler = SimpleNamespace(validate_request=Mock())
    return executor


def _request(**kwargs) -> LlmRequest:
    parameters = dict(
        request_id=10,
        input_tokens=[1, 2, 3],
        max_new_tokens=1,
        sampling_config=SamplingConfig(beam_width=1, num_return_sequences=1),
        is_streaming=False,
    )
    parameters.update(kwargs)
    request = LlmRequest(**parameters)
    request.py_disaggregated_params = None
    return request


@pytest.mark.parametrize(
    "request_type", ["decode", "generation_only", "multiple", "beam", "multimodal", "gen_first"]
)
def test_request_guard_rejects_unsupported_requests(request_type: str) -> None:
    request = _request()
    if request_type == "decode":
        request.max_new_tokens = 2
    elif request_type == "generation_only":
        request = _request(llm_request_type=LlmRequestType.LLMREQUEST_TYPE_GENERATION_ONLY)
    elif request_type == "multiple":
        request = _request(sampling_config=SamplingConfig(beam_width=1, num_return_sequences=2))
    elif request_type == "beam":
        request = _request(sampling_config=SamplingConfig(beam_width=2, num_return_sequences=1))
    elif request_type == "multimodal":
        request.py_multimodal_data = {"image": {}}
    else:
        request.py_disaggregated_params = DisaggregatedParams(
            schedule_style=DisaggScheduleStyle.GENERATION_FIRST
        )
    with pytest.raises(ValueError, match="not supported with dkv_config yet"):
        _request_executor()._validate_request(request)


@pytest.mark.parametrize(
    "field",
    [
        "multimodal_embedding",
        "multimodal_hashes",
        "multimodal_positions",
        "multimodal_lengths",
        "mrope_rotary_cos_sin",
        "mrope_position_deltas",
    ],
)
def test_request_guard_rejects_preprocessed_multimodal_fields(field: str) -> None:
    values = {
        "multimodal_embedding": torch.zeros(1, 1),
        "multimodal_hashes": [[0] * 8],
        "multimodal_positions": [0],
        "multimodal_lengths": [1],
        "mrope_rotary_cos_sin": torch.zeros(1, 1),
        "mrope_position_deltas": 0,
    }
    request = _request(**{field: values[field]})
    with pytest.raises(ValueError, match="multimodal inputs is not supported"):
        _request_executor()._validate_request(request)


@pytest.mark.parametrize(
    "field",
    ["multimodal_embedding_handles", "mrope_position_ids_handle", "mrope_position_deltas_handle"],
)
def test_request_guard_rejects_remote_multimodal_handles(field: str) -> None:
    request = _request()
    values = (
        {"multimodal_embedding_handles": [{}], "multimodal_hashes": [[0] * 8]}
        if field == "multimodal_embedding_handles"
        else {field: {}}
    )
    request.py_disaggregated_params = DisaggregatedParams(**values)
    with pytest.raises(ValueError, match="multimodal inputs is not supported"):
        _request_executor()._validate_request(request)


@pytest.mark.parametrize("dkv_enabled", [False, True])
def test_request_guard_accepts_one_token_and_preserves_non_dkv(dkv_enabled: bool) -> None:
    request = _request(max_new_tokens=1 if dkv_enabled else 8)
    executor = _request_executor(dkv_enabled=dkv_enabled)
    executor._validate_request(request)
    executor.sampler.validate_request.assert_called_once_with(request)


@pytest.mark.parametrize("max_new_tokens", [1, 8])
def test_request_guard_checks_context_only_request_budget(max_new_tokens: int) -> None:
    request = _request(
        llm_request_type=LlmRequestType.LLMREQUEST_TYPE_CONTEXT_ONLY, max_new_tokens=max_new_tokens
    )
    executor = _request_executor()
    if max_new_tokens == 1:
        executor._validate_request(request)
    else:
        with pytest.raises(ValueError, match="not supported with dkv_config yet"):
            executor._validate_request(request)


def test_dkv_skips_rank_local_dummy_allocation() -> None:
    executor = PyExecutor.__new__(PyExecutor)
    executor.dkv_enabled = True
    executor._pad_attention_dp_dummy_request()
    executor._pad_empty_attention_dp_batch(ScheduledRequests())


def test_dkv_rejects_rank_local_balancing() -> None:
    executor = PyExecutor.__new__(PyExecutor)
    executor.dkv_enabled = True
    with pytest.raises(ValueError, match="not supported with dkv_config yet"):
        executor._balance_adp_requests([], [])


@pytest.mark.parametrize(
    "dkv_enabled, manager, expected",
    [
        (False, SimpleNamespace(get_dkv_measurement_snapshot=lambda: {"rank": 1}), {"rank": 1}),
        (True, SimpleNamespace(get_dkv_measurement_snapshot=lambda: {"rank": 1}), {"rank": 1}),
        (False, SimpleNamespace(get_dkv_measurement_snapshot=lambda: None), None),
        (True, SimpleNamespace(get_dkv_measurement_snapshot=lambda: None), None),
        (True, SimpleNamespace(), None),
        (False, None, None),
        (True, Mock(), None),
    ],
    ids=[
        "adp-control-collected",
        "dkv-collected",
        "adp-not-collected",
        "dkv-not-collected",
        "no-accessor",
        "no-cache-manager",
        "fabricating-double",
    ],
)
def test_measurement_snapshot_is_exported_whenever_the_manager_collects_a_dict(
    dkv_enabled: bool, manager, expected
) -> None:
    executor = PyExecutor.__new__(PyExecutor)
    executor.dkv_enabled = dkv_enabled
    executor.kv_cache_manager = manager
    assert executor._dkv_measurement_snapshot() == expected


def test_measurement_snapshot_carries_what_the_data_plane_of_the_layer_split_moved() -> None:
    from tensorrt_llm._torch.pyexecutor.dkv_streamer import DataPlaneStats

    executor = PyExecutor.__new__(PyExecutor)
    executor.dkv_enabled = True
    executor.kv_cache_manager = SimpleNamespace(get_dkv_measurement_snapshot=lambda: {"rank": 1})
    executor.dkv_streamer = SimpleNamespace(
        stats=DataPlaneStats(
            iterations=3,
            messages_sent=4,
            bytes_sent=4096,
            bytes_local=512,
            bytes_local_fetch=384,
            bytes_local_writeback=128,
            hook_seconds=0.25,
            compile_seconds=0.05,
            copy_seconds=0.03,
            transport_seconds=0.02,
            fetch_wait_seconds=0.0625,
            drain_seconds=0.125,
        )
    )
    snapshot = executor._dkv_measurement_snapshot()
    assert snapshot["rank"] == 1
    assert snapshot["data_plane"] == {
        "iterations": 3,
        "messages_sent": 4,
        "messages_received": 0,
        "bytes_sent": 4096,
        "bytes_received": 0,
        "local_copies": 0,
        "bytes_local": 512,
        "bytes_local_fetch": 384,
        "bytes_local_writeback": 128,
        "hook_seconds": 0.25,
        "compile_seconds": 0.05,
        "copy_seconds": 0.03,
        "transport_seconds": 0.02,
        "fetch_wait_seconds": 0.0625,
        "drain_seconds": 0.125,
    }


def _control_timing_executor() -> PyExecutor:
    executor = PyExecutor.__new__(PyExecutor)
    executor.dkv_enabled = True
    executor.dist = None
    executor.iter_counter = 0
    executor.active_requests = []
    executor._dkv_commit_reason = None
    executor._dkv_fatal_messages = []
    executor._dkv_freed_request_ids = []
    executor._dkv_sampler_errors = []
    executor._pending_transfer_responses = []
    executor._pending_response_terminations = []
    executor.kv_cache_transceiver = None
    executor.kv_cache_manager = SimpleNamespace(
        get_dkv_control_digest=lambda: (0, ((100,),)),
        get_dkv_config_fingerprint=lambda: [],
        get_dkv_startup_settings=lambda: {},
        get_dkv_measurement_snapshot=lambda: {"rank": 0},
    )
    executor.scheduler = SimpleNamespace(dkv_dual_ledger_enabled=False)
    with patch(
        "tensorrt_llm._torch.pyexecutor.py_executor.DkvInvariantChecker",
        return_value=Mock(enabled=False),
    ):
        executor._initialize_dkv_invariant_checker()
    return executor


@pytest.mark.parametrize("measurement", [None, "0", "1"])
def test_control_plane_timing_is_opt_in_and_cumulative(
    monkeypatch: pytest.MonkeyPatch, measurement: str | None
) -> None:
    if measurement is None:
        monkeypatch.delenv("TRTLLM_DKV_MEASUREMENT", raising=False)
    else:
        monkeypatch.setenv("TRTLLM_DKV_MEASUREMENT", measurement)
    executor = _control_timing_executor()
    with (
        patch(
            "tensorrt_llm._torch.pyexecutor.py_executor.time.perf_counter",
            side_effect=[10.0, 10.25, 20.0, 20.75, 30.0, 30.5, 40.0, 41.0],
        ) as timer,
        patch(
            "tensorrt_llm._torch.pyexecutor.py_executor.sync_dkv_control",
            return_value=DkvControlResult(fatal_messages=(), has_pending_responses=False),
        ) as control,
        patch(
            "tensorrt_llm._torch.pyexecutor.py_executor.sync_dkv_sample_results", return_value=[]
        ) as sample,
    ):
        for _ in range(2):
            executor._sync_dkv_control()
            executor._sync_dkv_samples(ScheduledRequests(), ScheduledRequests())
    assert control.call_count == 2
    assert sample.call_count == 2
    snapshot = executor._dkv_measurement_snapshot()
    if measurement == "1":
        assert timer.call_count == 8
        assert snapshot["control_plane"] == {
            "control_seconds": 0.75,
            "control_calls": 2,
            "sample_seconds": 1.75,
            "sample_calls": 2,
        }
        executor._dkv_control_plane_stats.control_calls += 1
        assert snapshot["control_plane"]["control_calls"] == 2
    else:
        timer.assert_not_called()
        assert snapshot == {"rank": 0}


@pytest.mark.parametrize("phase", ["control", "sample"])
def test_control_plane_timing_records_failed_collectives(
    monkeypatch: pytest.MonkeyPatch, phase: str
) -> None:
    monkeypatch.setenv("TRTLLM_DKV_MEASUREMENT", "1")
    executor = _control_timing_executor()
    collective = "sync_dkv_control" if phase == "control" else "sync_dkv_sample_results"
    with (
        patch(
            "tensorrt_llm._torch.pyexecutor.py_executor.time.perf_counter",
            side_effect=[2.0, 2.125],
        ),
        patch(
            f"tensorrt_llm._torch.pyexecutor.py_executor.{collective}",
            side_effect=RuntimeError("collective failed"),
        ),
        pytest.raises(RuntimeError, match="collective failed"),
    ):
        if phase == "control":
            executor._sync_dkv_control()
        else:
            executor._sync_dkv_samples(ScheduledRequests(), ScheduledRequests())
    expected = {
        "control_seconds": 0.0,
        "control_calls": 0,
        "sample_seconds": 0.0,
        "sample_calls": 0,
    }
    expected[f"{phase}_seconds"] = 0.125
    expected[f"{phase}_calls"] = 1
    assert executor._dkv_measurement_snapshot()["control_plane"] == expected
