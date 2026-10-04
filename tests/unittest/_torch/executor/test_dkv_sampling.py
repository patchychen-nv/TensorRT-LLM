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
"""DKV idle sampling and deferred, request-scoped sampler failures."""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from dkv_test_utils import make_request

from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequestState
from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor
from tensorrt_llm._torch.pyexecutor.sampler import SampleState
from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests

pytestmark = pytest.mark.cpu_only


def _executor(*, dkv_enabled: bool = True) -> PyExecutor:
    executor = PyExecutor.__new__(PyExecutor)
    executor.dkv_enabled = dkv_enabled
    executor.sampler = Mock()
    executor._dkv_sampler_errors = []
    executor._dkv_fatal_messages = []
    executor._fatal_error = None
    executor._error_budget = Mock(consume=Mock(return_value=False), budget=1.0)
    executor._handle_errors = Mock()
    return executor


def test_dummy_only_rank_skips_sampling_for_repeated_iterations() -> None:
    executor = _executor()
    dummy = make_request(0, is_dummy=True)
    dummy.state = LlmRequestState.GENERATION_IN_PROGRESS
    batch = ScheduledRequests()
    batch.generation_requests = [dummy]
    tokens = dummy.get_tokens(0)
    resource_manager = Mock()

    for _ in range(5):
        sample_state = executor._sample_async(batch, {"logits": object()})
        assert sample_state is None
        executor._update_requests(sample_state, resource_manager)
        assert dummy.get_tokens(0) == tokens
        assert dummy.state == LlmRequestState.GENERATION_IN_PROGRESS

    assert executor.sampler.mock_calls == []
    assert resource_manager.mock_calls == []
    executor._handle_errors.assert_not_called()
    assert executor._dkv_sampler_errors == []


@pytest.mark.parametrize("dkv_enabled", [False, True])
def test_update_none_preserves_non_dkv_error_handling(dkv_enabled: bool) -> None:
    executor = _executor(dkv_enabled=dkv_enabled)
    executor.sampler.update_requests.side_effect = AttributeError("missing sample state")
    executor._update_requests(None)
    if dkv_enabled:
        executor.sampler.update_requests.assert_not_called()
        executor._handle_errors.assert_not_called()
    else:
        executor._handle_errors.assert_called_once_with("missing sample state")


@pytest.mark.parametrize("stage", ["sample", "update"])
@pytest.mark.parametrize("dkv_enabled", [False, True])
def test_sampler_failure_is_staged_without_local_termination(stage: str, dkv_enabled: bool) -> None:
    executor = _executor(dkv_enabled=dkv_enabled)
    local = make_request(8, compute_rank=0)
    remote = make_request(9, compute_rank=1)
    dummy = make_request(0, is_dummy=True)
    requests = [local, remote, dummy]
    states = [request.state for request in requests]
    resource_manager = Mock()
    if stage == "sample":
        batch = ScheduledRequests()
        for request in requests:
            request.context_chunk_size = request.orig_prompt_len
        batch.reset_context_requests(requests)
        executor.sampler.sample_async.side_effect = RuntimeError("sampler failed")
        with (
            patch("tensorrt_llm._torch.pyexecutor.py_executor.HandleLogits"),
            patch("tensorrt_llm._torch.pyexecutor.py_executor.HandleAdditionalOutputs"),
        ):
            assert executor._sample_async(batch, {"logits": object()}) is None
    else:
        executor.sampler.update_requests.side_effect = RuntimeError("sampler failed")
        executor._update_requests(SampleState(requests=requests), resource_manager)

    if dkv_enabled:
        assert executor._dkv_sampler_errors == [("sampler failed", (8,))]
        executor._error_budget.consume.assert_called_once_with("sampler failed")
        executor._handle_errors.assert_not_called()
    else:
        assert executor._dkv_sampler_errors == []
        executor._handle_errors.assert_called_once_with("sampler failed")
    assert [request.state for request in requests] == states
    assert resource_manager.mock_calls == []


@pytest.mark.parametrize("dkv_enabled", [False, True])
def test_setup_failure_is_staged_before_sampling_only_for_dkv(dkv_enabled: bool) -> None:
    executor = _executor(dkv_enabled=dkv_enabled)
    executor.sampler.setup_sampler_step.side_effect = RuntimeError("setup failed")
    request = make_request(8)
    request.context_chunk_size = request.orig_prompt_len
    batch = ScheduledRequests()
    batch.reset_context_requests([request])
    executor._setup_sampler_step(batch)
    if dkv_enabled:
        assert executor._dkv_sampler_errors == [("setup failed", (8,))]
        assert executor._sample_async(batch, {"logits": object()}) is None
        executor._update_requests(None)
        executor.sampler.sample_async.assert_not_called()
        executor.sampler.update_requests.assert_not_called()
        executor._error_budget.consume.assert_called_once_with("setup failed")
        executor._handle_errors.assert_not_called()
    else:
        executor._handle_errors.assert_called_once_with("setup failed")


def test_fatal_sampler_error_waits_for_scontrol_without_setting_executor_fatal() -> None:
    executor = _executor()
    executor._error_budget.consume.return_value = True
    request = make_request(8)
    state_before = request.state
    executor.sampler.update_requests.side_effect = RuntimeError("fatal sampling")
    executor._update_requests(SampleState(requests=[request]))
    assert executor._dkv_sampler_errors == [("fatal sampling", (8,))]
    assert executor._dkv_fatal_messages == ["fatal sampling"]
    assert executor._fatal_error is None
    assert request.state == state_before
    executor._error_budget.consume.assert_called_once_with("fatal sampling")
    executor._handle_errors.assert_not_called()


@pytest.mark.parametrize("dkv_enabled", [False, True])
def test_forward_failure_raises_only_for_dkv(dkv_enabled: bool) -> None:
    executor = _executor(dkv_enabled=dkv_enabled)
    executor.iter_counter = 0
    executor._iter_adp_dummy_ctx_tokens = 0
    executor._iter_adp_dummy_gen_tokens = 0
    executor.model_engine = SimpleNamespace(
        route_capture=None, forward=Mock(side_effect=RuntimeError("forward failed"))
    )
    executor.execution_stream = Mock()
    executor.resource_manager = Mock()
    executor._attach_encoder_output_to_execution_stream = Mock()
    batch = ScheduledRequests()
    request = make_request(8)
    request.context_chunk_size = request.orig_prompt_len
    batch.reset_context_requests([request])

    with (
        patch("tensorrt_llm._torch.pyexecutor.py_executor.torch.cuda.current_stream"),
        patch(
            "tensorrt_llm._torch.pyexecutor.py_executor.torch.cuda.stream",
            return_value=nullcontext(),
        ),
    ):
        if dkv_enabled:
            with pytest.raises(RuntimeError, match="forward failed"):
                executor._forward_step(batch)
            executor._handle_errors.assert_not_called()
        else:
            assert executor._forward_step(batch) is None
            executor._handle_errors.assert_called_once_with("forward failed")
    executor.model_engine.forward.assert_called_once()
    assert executor.resource_manager.mock_calls == []
