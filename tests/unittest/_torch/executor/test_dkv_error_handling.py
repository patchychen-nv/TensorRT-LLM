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
"""DKV error ownership, synchronized retirement and fatal queue draining."""

import queue
from collections import deque
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from dkv_test_utils import LockstepTpGroup, make_request

from tensorrt_llm._torch.pyexecutor.dkv import sync_dkv_sample_results
from tensorrt_llm._torch.pyexecutor.executor_request_queue import RequestQueueItem
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequestState
from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor
from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests

pytestmark = pytest.mark.cpu_only


class _WaitingQueue:
    def __init__(self, items=()) -> None:
        self.items = deque(items)

    def __bool__(self) -> bool:
        return bool(self.items)

    def pop_request(self):
        return self.items.popleft()


def _executor(dist, requests, *, debug: bool = True) -> PyExecutor:
    executor = PyExecutor.__new__(PyExecutor)
    executor.dkv_enabled = True
    executor.dist = dist
    executor.enable_attention_dp = True
    executor.gather_all_responses = False
    executor.active_requests = list(requests)
    executor._pending_transfer_responses = []
    executor._pending_response_terminations = []
    executor._fatal_error = None
    executor._dkv_fatal_messages = []
    executor._dkv_sampler_errors = []
    executor._error_budget = Mock(consume=Mock(return_value=False), budget=1.0)
    executor._dkv_invariant_checker = SimpleNamespace(enabled=debug)
    executor.is_shutdown = False
    executor.waiting_queue = _WaitingQueue()
    executor.executor_request_queue = Mock(get_request_queue=Mock(return_value=queue.Queue()))
    executor.published = []
    executor.releases = []
    executor.commit_reasons = []

    @contextmanager
    def commit_window(reason):
        executor.commit_reasons.append(reason)
        try:
            yield
        finally:
            executor.commit_reasons.pop()

    def terminate(request):
        assert executor.commit_reasons, "request freed outside a commit window"
        executor.releases.append((request.py_request_id, executor.commit_reasons[-1]))

    def enqueue(responses):
        records = [
            (request_id, dist.tp_rank, response.error_msg) for request_id, response in responses
        ]
        executor.published.extend(dist.tp_allgather(records))

    executor._dkv_commit_window = commit_window
    executor._terminate_request = Mock(side_effect=terminate)
    executor._enqueue_responses = Mock(side_effect=enqueue)
    return executor


def _flush(executor):
    with executor._dkv_commit_window("control"):
        executor._flush_pending_transfer_responses()


def test_synchronized_error_has_one_owner_and_one_release_per_replica() -> None:
    def fail(dist):
        request = make_request(8, compute_rank=1, local_rank=dist.tp_rank)
        unaffected = make_request(9, compute_rank=0, local_rank=dist.tp_rank)
        executor = _executor(dist, [request, unaffected])
        executor._handle_errors("sampling failed", requests=[request, request], charge_budget=False)
        executor._handle_errors("sampling failed", requests=[request], charge_budget=False)
        assert executor.active_requests == [unaffected]
        assert request.state == LlmRequestState.GENERATION_COMPLETE
        assert executor._pending_response_terminations == [request]
        assert executor.releases == []
        executor._enqueue_responses.assert_not_called()
        executor._error_budget.consume.assert_not_called()
        _flush(executor)
        assert executor.releases == [(8, "control")]
        assert executor.published == [[], [(8, 1, "sampling failed")]]
        assert executor._pending_transfer_responses == []
        assert executor._pending_response_terminations == []

    group = LockstepTpGroup(2)
    group.run(fail)
    assert group.traces[0] == group.traces[1]


def test_rank_one_sampler_error_charges_only_origin_and_retires_in_lockstep() -> None:
    def sample(dist):
        request = make_request(8, compute_rank=1, local_rank=dist.tp_rank)
        request.context_chunk_size = request.orig_prompt_len
        executor = _executor(dist, [request])
        executor.sampler = Mock(sample_async=Mock(side_effect=RuntimeError("sampler failure")))
        local = ScheduledRequests()
        if request.py_dkv_is_local:
            local.reset_context_requests([request])
        else:
            dummy = make_request(0, local_rank=dist.tp_rank, is_dummy=True)
            dummy.state = LlmRequestState.GENERATION_IN_PROGRESS
            local.generation_requests = [dummy]
        assert executor._sample_async(local, {"logits": object()}) is None
        request.context_current_position = request.orig_prompt_len
        request.state = LlmRequestState.GENERATION_IN_PROGRESS
        executor._update_requests(None)
        assert executor.releases == []
        errors = sync_dkv_sample_results(
            dist, [request], local.all_requests(), executor._dkv_sampler_errors
        )
        assert len(errors) == 1
        assert errors[0].request_ids == (8,)
        executor._handle_errors(errors[0].message, requests=[request], charge_budget=False)
        assert executor.releases == []
        _flush(executor)
        assert executor.releases == [(8, "control")]
        assert executor.published == [[], [(8, 1, "sampler failure")]]
        assert executor._error_budget.consume.call_count == dist.tp_rank
        assert executor._fatal_error is None

    group = LockstepTpGroup(2)
    with (
        patch("tensorrt_llm._torch.pyexecutor.py_executor.HandleLogits"),
        patch("tensorrt_llm._torch.pyexecutor.py_executor.HandleAdditionalOutputs"),
    ):
        group.run(sample)
    assert group.traces[0] == group.traces[1]


def test_invalid_activation_responds_only_on_compute_rank() -> None:
    def activate(dist):
        request = make_request(8, compute_rank=1, local_rank=dist.tp_rank)
        executor = _executor(dist, [])
        executor._fetch_new_requests = Mock(return_value=[request])
        executor._validate_request = Mock(side_effect=ValueError("invalid prompt"))
        assert executor._fetch_and_activate_new_requests() == []
        assert executor.active_requests == []
        executor._error_budget.consume.assert_not_called()
        _flush(executor)
        assert executor.releases == [(8, "control")]
        assert executor.published == [[], [(8, 1, "invalid prompt")]]

    LockstepTpGroup(2).run(activate)


@pytest.mark.parametrize("dummy", [False, True])
def test_untagged_and_dummy_requests_do_not_generate_errors(dummy: bool) -> None:
    def fail(dist):
        request = make_request(8, local_rank=dist.tp_rank, is_dummy=dummy)
        request.py_dkv_compute_rank = None
        executor = _executor(dist, [request], debug=dummy)
        executor._handle_errors("failed", requests=[request], charge_budget=False)
        assert executor._pending_transfer_responses == []
        _flush(executor)
        assert executor.published == [[], []]
        assert executor.releases == [(8, "control")]

    LockstepTpGroup(2).run(fail)


def test_missing_compute_tag_is_a_debug_invariant() -> None:
    def fail(dist):
        request = make_request(8, local_rank=dist.tp_rank)
        request.py_dkv_compute_rank = None
        executor = _executor(dist, [request])
        with pytest.raises(RuntimeError, match="no compute-rank tag"):
            executor._handle_errors("failed", requests=[request], charge_budget=False)
        assert executor.active_requests == [request]
        assert request.state == LlmRequestState.CONTEXT_INIT
        assert executor.releases == []
        executor._enqueue_responses.assert_not_called()

    LockstepTpGroup(2).run(fail)


def test_unsynchronized_error_cannot_charge_or_mutate_dkv() -> None:
    def fail(dist):
        request = make_request(8, local_rank=dist.tp_rank)
        executor = _executor(dist, [request])
        with pytest.raises(RuntimeError, match="must be synchronized"):
            executor._handle_errors("rank-local error", requests=[request])
        executor._error_budget.consume.assert_not_called()
        executor._enqueue_responses.assert_not_called()
        assert executor.active_requests == [request]
        assert request.state == LlmRequestState.CONTEXT_INIT
        assert executor.releases == []

    LockstepTpGroup(2).run(fail)


def test_aligned_fatal_drains_queues_without_duplicate_responses_or_frees() -> None:
    def fail(dist):
        pending = make_request(8, compute_rank=1, local_rank=dist.tp_rank)
        active = make_request(9, compute_rank=0, local_rank=dist.tp_rank)
        executor = _executor(dist, [pending, active])
        executor.gather_all_responses = True
        executor._handle_errors("first failure", requests=[pending], charge_budget=False)

        def item(request_id):
            return RequestQueueItem(request_id, request=SimpleNamespace(client_id=request_id))

        executor.waiting_queue = _WaitingQueue([item(8), item(9), item(90), item(90)])
        raw_queue = executor.executor_request_queue.get_request_queue()
        for request_id in [8, 9, 90, 91]:
            raw_queue.put(item(request_id))
        executor._fatal_error = RuntimeError("fatal sampler failure")
        executor._handle_errors(
            "fatal sampler failure", charge_budget=False, fatal_is_collective_aligned=True
        )
        assert executor.is_shutdown
        assert executor.active_requests == []
        assert not executor.waiting_queue
        assert raw_queue.empty()
        assert executor.releases == [(8, "fatal"), (9, "fatal")]
        executor._error_budget.consume.assert_not_called()
        executor.executor_request_queue.enqueue_shutdown_request.assert_called_once_with()
        assert executor._enqueue_responses.call_count == 2
        records = [record for rank_records in executor.published for record in rank_records]
        assert sorted((request_id, rank) for request_id, rank, _ in records) == [
            (8, 1),
            (9, 0),
            (90, 0),
            (91, 0),
        ]
        assert executor._pending_transfer_responses == []
        assert executor._pending_response_terminations == []

    group = LockstepTpGroup(2)
    group.run(fail)
    assert group.traces[0] == group.traces[1]
