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
"""Only a DKV request's compute rank creates its response; all replicas retire."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from dkv_test_utils import LockstepTpGroup, make_request

from tensorrt_llm._torch.pyexecutor.llm_request import FinishReason, LlmRequestState
from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor

pytestmark = pytest.mark.cpu_only


def _executor(dist, requests, *, dkv_enabled: bool = True) -> PyExecutor:
    executor = PyExecutor.__new__(PyExecutor)
    executor.dkv_enabled = dkv_enabled
    executor._dkv_commit_reason = None
    executor.dist = dist
    executor.active_requests = requests
    executor.iter_counter = 2
    executor.stream_interval = 1
    executor.force_terminate_ctx_for_partial_reuse = False
    executor.perf_manager = Mock()
    executor._disagg_coordinator = Mock()
    executor._maybe_attach_ctx_usage = Mock()
    executor._enqueue_responses = Mock()
    executor._terminate_request = Mock()
    return executor


@pytest.mark.parametrize(
    "finish_reason", [FinishReason.END_ID, FinishReason.LENGTH, FinishReason.STOP_WORDS]
)
def test_compute_rank_one_emits_once_and_both_replicas_terminate(finish_reason) -> None:
    group = LockstepTpGroup(2)

    def respond(dist):
        request = make_request(8, compute_rank=1, local_rank=dist.tp_rank)
        prompt = request.get_tokens(0)
        if request.py_dkv_is_local:
            request.add_new_token(42, 0)
            request.py_decoding_iter = 1
        request.finish_by_reason(finish_reason)
        request.create_response = Mock(wraps=request.create_response)
        executor = _executor(dist, [request])

        def enqueue(responses):
            assert executor._terminate_request.call_count == 0
            emitted = [(request_id, dist.tp_rank) for request_id, _ in responses]
            return dist.tp_allgather(emitted)

        executor._enqueue_responses.side_effect = enqueue
        assert executor._handle_responses() == [request]
        assert executor.active_requests == []
        executor._enqueue_responses.assert_called_once()
        executor._terminate_request.assert_called_once_with(request)
        if request.py_dkv_is_local:
            request.create_response.assert_called_once_with(False, dist.rank)
        else:
            request.create_response.assert_not_called()
            executor.perf_manager.append_step_metrics.assert_not_called()
            assert request.get_tokens(0) == prompt
            assert request.py_decoding_iter == 0
        return [request_id for request_id, _ in executor._enqueue_responses.call_args.args[0]]

    assert group.run(respond) == [[], [8]]
    assert group.traces[0] == group.traces[1]


def test_remote_unfinished_request_stays_active_without_response() -> None:
    request = make_request(8, compute_rank=1)
    request.create_response = Mock(side_effect=AssertionError("remote result creation"))
    executor = _executor(SimpleNamespace(rank=0), [request])
    assert executor._handle_responses() == []
    assert executor.active_requests == [request]
    executor._enqueue_responses.assert_called_once_with([])
    executor._terminate_request.assert_not_called()
    request.create_response.assert_not_called()


def test_remote_transmitting_request_never_needs_context_phase_params() -> None:
    request = make_request(8, compute_rank=1)
    request.state = LlmRequestState.DISAGG_CONTEXT_TRANS_IN_PROGRESS
    request.create_response = Mock(side_effect=AssertionError("missing context_phase_params"))
    executor = _executor(SimpleNamespace(rank=0), [request])
    assert executor._handle_responses() == []
    executor._enqueue_responses.assert_called_once_with([])
    executor._terminate_request.assert_not_called()
    request.create_response.assert_not_called()


def test_non_dkv_response_ignores_local_dkv_label() -> None:
    request = make_request(8, compute_rank=1)
    request.add_new_token(42, 0)
    request.py_decoding_iter = 1
    request.finish_by_reason(FinishReason.LENGTH)
    request.create_response = Mock(wraps=request.create_response)
    executor = _executor(SimpleNamespace(rank=0), [request], dkv_enabled=False)
    assert executor._handle_responses() == [request]
    request.create_response.assert_called_once_with(False, 0)
    assert len(executor._enqueue_responses.call_args.args[0]) == 1
