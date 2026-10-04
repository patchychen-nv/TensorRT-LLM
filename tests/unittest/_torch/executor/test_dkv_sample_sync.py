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
"""S-sample completion replication using native requests and real rank payloads."""

from dataclasses import replace

import pytest
from dkv_test_utils import LockstepTpGroup, make_request

from tensorrt_llm._torch.pyexecutor.dkv import (
    DkvSamplePayload,
    DkvSamplerError,
    digest_global_batch,
    sync_dkv_sample_results,
)
from tensorrt_llm._torch.pyexecutor.llm_request import (
    FinishReason,
    LlmRequest,
    LlmRequestState,
    SamplingConfig,
)
from tensorrt_llm.bindings.internal.batch_manager import LlmRequestType

pytestmark = pytest.mark.cpu_only


def _request(request_id: int, compute_rank: int, rank: int, *, context_only: bool = False):
    request = LlmRequest(
        request_id=request_id,
        input_tokens=list(range(8)),
        max_new_tokens=1,
        sampling_config=SamplingConfig(1),
        is_streaming=False,
        llm_request_type=(
            LlmRequestType.LLMREQUEST_TYPE_CONTEXT_ONLY
            if context_only
            else LlmRequestType.LLMREQUEST_TYPE_CONTEXT_AND_GENERATION
        ),
    )
    request.py_dkv_compute_rank = compute_rank
    request.py_dkv_is_local = compute_rank == rank
    return request


def _advance_context(request: LlmRequest, chunk_size: int = 8) -> None:
    request.context_chunk_size = chunk_size
    request.move_to_next_context_chunk()
    if request.context_remaining_length == 0:
        request.state = LlmRequestState.GENERATION_IN_PROGRESS


def test_native_finish_reasons_are_read_only_and_do_not_create_a_response() -> None:
    request = make_request(7)
    assert request.finish_reasons == [FinishReason.NOT_FINISHED]
    request.set_finished_reason(FinishReason.END_ID, 0)
    snapshot = request.finish_reasons
    assert snapshot == [FinishReason.END_ID]
    snapshot[0] = FinishReason.LENGTH
    assert request.finish_reasons == [FinishReason.END_ID]
    assert request.get_num_tokens(0) == 8
    assert request.state == LlmRequestState.CONTEXT_INIT
    with pytest.raises(AttributeError):
        request.finish_reasons = [FinishReason.LENGTH]


@pytest.mark.parametrize("context_only", [False, True])
@pytest.mark.parametrize(
    "reason", [FinishReason.END_ID, FinishReason.LENGTH, FinishReason.STOP_WORDS]
)
def test_completion_replicates_state_and_reason_without_tokens(reason, context_only: bool) -> None:
    group = LockstepTpGroup(2)

    def run_rank(dist):
        requests = [_request(7, 1, dist.tp_rank, context_only=context_only)]
        request = requests[0]
        _advance_context(request)
        local_requests = requests if request.py_dkv_is_local else []
        if request.py_dkv_is_local:
            request.add_new_token(42, 0)
            request.finish_by(reason, 0)
        sync_dkv_sample_results(dist, requests, local_requests)
        assert request.state == LlmRequestState.GENERATION_COMPLETE
        assert request.finish_reasons == [reason]
        assert request.is_finished
        assert not request.is_context_finished
        assert request.is_finished_due_to_length == (reason == FinishReason.LENGTH)
        assert not request.is_finished_due_to_cancellation
        # These are the coordinator's unchanged KV-send conditions.
        should_send = request.is_context_only_request and (
            request.is_context_finished or request.is_finished_due_to_length
        )
        assert should_send == (context_only and reason == FinishReason.LENGTH)
        assert request.get_num_tokens(0) == (9 if request.py_dkv_is_local else 8)

    group.run(run_rank)
    assert [len(trace) for trace in group.traces] == [1, 1]


@pytest.mark.parametrize("group_size", [3, 8])
def test_intermediate_chunks_and_idle_dummy_stay_unchanged_across_rounds(group_size: int) -> None:
    group = LockstepTpGroup(group_size)

    def run_rank(dist):
        request = _request(7, 0, dist.tp_rank)
        dummy = make_request(100, local_rank=dist.tp_rank, is_dummy=True, prompt_len=1)
        dummy_before = (dummy.state, dummy.get_num_tokens(0), dummy.context_current_position)
        for _ in range(4):
            _advance_context(request, 2)
            if request.py_dkv_is_local and request.context_remaining_length == 0:
                request.add_new_token(42, 0)
                request.finish_by(FinishReason.LENGTH, 0)
            local_requests = [request] if request.py_dkv_is_local else [dummy]
            sync_dkv_sample_results(dist, [request], local_requests)
            assert (
                dummy.state,
                dummy.get_num_tokens(0),
                dummy.context_current_position,
            ) == dummy_before
            if request.context_remaining_length:
                assert request.state == LlmRequestState.CONTEXT_INIT
                assert request.finish_reasons == [FinishReason.NOT_FINISHED]
            else:
                assert request.state == LlmRequestState.GENERATION_COMPLETE
                assert request.finish_reasons == [FinishReason.LENGTH]
        return request.get_num_tokens(0)

    assert group.run(run_rank) == [9] + [8] * (group_size - 1)
    assert [len(trace) for trace in group.traces] == [4] * group_size


@pytest.mark.parametrize("group_size", [3, 8])
def test_all_compute_ranks_publish_their_own_completions(group_size: int) -> None:
    group = LockstepTpGroup(group_size)
    kinds = [FinishReason.END_ID, FinishReason.LENGTH, FinishReason.STOP_WORDS]
    reasons = [kinds[rank % len(kinds)] for rank in range(group_size)]

    def run_rank(dist):
        requests = [_request(7 + rank, rank, dist.tp_rank) for rank in range(group.size)]
        for request in requests:
            _advance_context(request)
            if request.py_dkv_is_local:
                request.finish_by_reason(reasons[dist.tp_rank])
        local_requests = [request for request in requests if request.py_dkv_is_local]
        sync_dkv_sample_results(dist, requests, local_requests)
        assert [request.finish_reasons[0] for request in requests] == reasons
        assert all(request.is_finished for request in requests)

    group.run(run_rank)
    assert [len(trace) for trace in group.traces] == [1] * group_size


@pytest.mark.parametrize(
    "fault, diagnostic",
    [
        ("missing", "missing=\\[7\\]"),
        ("duplicate", "duplicate completion"),
        ("wrong_owner", "unowned 8"),
        ("unknown", "unowned 99"),
        ("state", "invalid completed state"),
        ("reason", "invalid single-beam finish reason"),
        ("multi_beam", "invalid single-beam finish reason"),
    ],
)
def test_corrupt_completion_fails_on_all_ranks_before_mutation(fault: str, diagnostic: str) -> None:
    group = LockstepTpGroup(2)

    def run_rank(dist):
        requests = [_request(7, 0, dist.tp_rank), _request(8, 1, dist.tp_rank)]
        for request in requests:
            _advance_context(request)
            if request.py_dkv_is_local:
                request.finish_by_reason(FinishReason.LENGTH)
        original_allgather = dist.tp_allgather

        def exchange(payload: DkvSamplePayload):
            if dist.tp_rank == 0:
                completion = payload.completions[0]
                if fault == "missing":
                    completions = ()
                elif fault == "duplicate":
                    completions = (completion, completion)
                elif fault == "wrong_owner":
                    completions = (replace(completion, request_id=8),)
                elif fault == "unknown":
                    completions = (replace(completion, request_id=99),)
                elif fault == "state":
                    completions = (replace(completion, state=LlmRequestState.CONTEXT_INIT.value),)
                elif fault == "reason":
                    completions = (
                        replace(completion, finish_reasons=(FinishReason.NOT_FINISHED.value,)),
                    )
                else:
                    completions = (replace(completion, finish_reasons=(1, 1)),)
                payload = replace(payload, completions=completions)
            return original_allgather(payload)

        dist.tp_allgather = exchange
        local_requests = [request for request in requests if request.py_dkv_is_local]
        with pytest.raises(RuntimeError, match=diagnostic):
            sync_dkv_sample_results(dist, requests, local_requests)
        remote = requests[1 - dist.tp_rank]
        assert remote.state == LlmRequestState.GENERATION_IN_PROGRESS
        assert remote.finish_reasons == [FinishReason.NOT_FINISHED]

    group.run(run_rank)


@pytest.mark.parametrize("invalid_local_batch", ["missing", "remote", "duplicate"])
def test_local_batch_validation_is_exchanged_before_raising(invalid_local_batch: str) -> None:
    group = LockstepTpGroup(2)

    def run_rank(dist):
        requests = [_request(7, 0, dist.tp_rank), _request(8, 1, dist.tp_rank)]
        for request in requests:
            _advance_context(request, 2)
        local_requests = [requests[dist.tp_rank]]
        if dist.tp_rank == 0:
            local_requests = {
                "missing": [],
                "remote": [requests[1]],
                "duplicate": [requests[0], requests[0]],
            }[invalid_local_batch]
        with pytest.raises(RuntimeError, match="S-sample invariant violation"):
            sync_dkv_sample_results(dist, requests, local_requests)

    group.run(run_rank)
    assert [len(trace) for trace in group.traces] == [1, 1]


@pytest.mark.parametrize("partially_finished", [False, True])
def test_sampler_error_preserves_healthy_completions(partially_finished: bool) -> None:
    group = LockstepTpGroup(2)

    def run_rank(dist):
        requests = [_request(7, 0, dist.tp_rank), _request(8, 1, dist.tp_rank)]
        for request in requests:
            _advance_context(request)
        if dist.tp_rank == 0:
            requests[0].finish_by_reason(FinishReason.LENGTH)
        elif partially_finished:
            requests[1].finish_by_reason(FinishReason.LENGTH)
        errors = [("sampler failed", (8,))] if dist.tp_rank else []
        result = sync_dkv_sample_results(dist, requests, [requests[dist.tp_rank]], errors)
        assert result == (DkvSamplerError("sampler failed", (8,)),)
        assert requests[0].state == LlmRequestState.GENERATION_COMPLETE
        assert requests[0].finish_reasons == [FinishReason.LENGTH]
        if dist.tp_rank == 0 or not partially_finished:
            assert requests[1].state == LlmRequestState.GENERATION_IN_PROGRESS
            assert requests[1].finish_reasons == [FinishReason.NOT_FINISHED]

    group.run(run_rank)
    assert [len(trace) for trace in group.traces] == [1, 1]


@pytest.mark.parametrize("group_size", [3, 8])
def test_sampler_errors_keep_rank_and_event_order_with_overlapping_failure_sets(
    group_size: int,
) -> None:
    group = LockstepTpGroup(group_size)

    def run_rank(dist):
        requests = [_request(7 + rank, rank, dist.tp_rank) for rank in range(group.size)]
        for request in requests:
            _advance_context(request)
        request_id = 7 + dist.tp_rank
        errors = [(f"rank {dist.tp_rank} first", (request_id,))]
        if dist.tp_rank == 1:
            errors.append(("rank 1 second", (request_id,)))
        result = sync_dkv_sample_results(dist, requests, [requests[dist.tp_rank]], errors)
        assert all(request.state == LlmRequestState.GENERATION_IN_PROGRESS for request in requests)
        return result

    expected = []
    for rank in range(group_size):
        expected.append(DkvSamplerError(f"rank {rank} first", (7 + rank,)))
        if rank == 1:
            expected.append(DkvSamplerError("rank 1 second", (8,)))
    assert group.run(run_rank) == [tuple(expected)] * group.size


def test_sampler_error_on_intermediate_chunk_needs_no_completion() -> None:
    group = LockstepTpGroup(2)

    def run_rank(dist):
        request = _request(7, 0, dist.tp_rank)
        _advance_context(request, 2)
        errors = [("intermediate chunk failed", (7,))] if dist.tp_rank == 0 else []
        result = sync_dkv_sample_results(
            dist, [request], [request] if request.py_dkv_is_local else [], errors
        )
        assert result == (DkvSamplerError("intermediate chunk failed", (7,)),)
        assert request.state == LlmRequestState.CONTEXT_INIT

    group.run(run_rank)


@pytest.mark.parametrize(
    "error, diagnostic",
    [
        (DkvSamplerError("wrong owner", (8,)), "sampler error for unowned 8"),
        (DkvSamplerError("unknown", (99,)), "sampler error for unowned 99"),
        (DkvSamplerError("duplicate", (7, 7)), "duplicate sampler error request IDs"),
        (DkvSamplerError("invalid ID", ("7",)), "invalid sampler error"),
        (DkvSamplerError("invalid IDs", [7]), "invalid sampler error"),
        (DkvSamplerError(42, (7,)), "invalid sampler error"),
        (None, "invalid sampler error"),
    ],
)
def test_invalid_sampler_error_fails_before_mutating_healthy_replicas(
    error, diagnostic: str
) -> None:
    group = LockstepTpGroup(2)

    def run_rank(dist):
        requests = [_request(7, 0, dist.tp_rank), _request(8, 1, dist.tp_rank)]
        for request in requests:
            _advance_context(request)
            if request.py_dkv_is_local:
                request.finish_by_reason(FinishReason.LENGTH)
        original_allgather = dist.tp_allgather

        def exchange(payload):
            if dist.tp_rank == 0:
                payload = replace(payload, errors=(error,))
            return original_allgather(payload)

        dist.tp_allgather = exchange
        with pytest.raises(RuntimeError, match=diagnostic):
            sync_dkv_sample_results(dist, requests, [requests[dist.tp_rank]])
        assert requests[1 - dist.tp_rank].state == LlmRequestState.GENERATION_IN_PROGRESS

    group.run(run_rank)
    assert [len(trace) for trace in group.traces] == [1, 1]


def test_failed_request_completion_metadata_still_requires_validation() -> None:
    group = LockstepTpGroup(2)

    def run_rank(dist):
        request = _request(7, 0, dist.tp_rank)
        _advance_context(request)
        if request.py_dkv_is_local:
            request.finish_by_reason(FinishReason.LENGTH)
        original_allgather = dist.tp_allgather

        def exchange(payload):
            if dist.tp_rank == 0:
                payload = replace(
                    payload,
                    completions=(replace(payload.completions[0], finish_reasons=(9999,)),),
                )
            return original_allgather(payload)

        dist.tp_allgather = exchange
        errors = [("partially failed", (7,))] if dist.tp_rank == 0 else []
        with pytest.raises(RuntimeError, match="unknown finish reason"):
            sync_dkv_sample_results(
                dist, [request], [request] if request.py_dkv_is_local else [], errors
            )

    group.run(run_rank)


def test_stale_remote_finish_reason_is_rejected_on_all_ranks() -> None:
    group = LockstepTpGroup(2)

    def run_rank(dist):
        request = _request(7, 0, dist.tp_rank)
        _advance_context(request)
        if request.py_dkv_is_local:
            request.finish_by_reason(FinishReason.LENGTH)
        else:
            request.set_finished_reason(FinishReason.END_ID, 0)
        with pytest.raises(RuntimeError, match="non-local request 7 already has a finish reason"):
            sync_dkv_sample_results(dist, [request], [request] if request.py_dkv_is_local else [])

    group.run(run_rank)


def test_global_batch_digest_covers_identity_owner_position_and_order_but_not_dummies() -> None:
    first, second = _request(7, 0, 0), _request(8, 1, 0)
    for request in (first, second):
        _advance_context(request, 2)
    dummy = make_request(100, is_dummy=True)
    reference = digest_global_batch([first, second])
    assert reference[0] == 2
    assert digest_global_batch([first, dummy, second]) == reference
    assert digest_global_batch([second, first]) != reference
    assert digest_global_batch([first]) != reference
    _advance_context(second, 2)
    assert digest_global_batch([first, second]) != reference
    assert digest_global_batch([])[0] == 0


@pytest.mark.parametrize("group_size", [3, 8])
@pytest.mark.parametrize("divergence", ["missing", "position", "owner", "order"])
def test_diverged_global_batch_is_reported_by_every_rank_in_one_exchange(
    divergence: str, group_size: int
) -> None:
    group = LockstepTpGroup(group_size)
    last = group_size - 1

    def run_rank(dist):
        requests = [_request(7, 0, dist.tp_rank), _request(8, 1, dist.tp_rank)]
        for request in requests:
            _advance_context(request, 2)
        local_requests = [request for request in requests if request.py_dkv_is_local]
        global_requests = list(requests)
        if dist.tp_rank == last:
            if divergence == "missing":
                global_requests = requests[:1]
            elif divergence == "position":
                _advance_context(requests[0], 2)
            elif divergence == "owner":
                requests[0].py_dkv_compute_rank = last
            else:
                global_requests.reverse()
        with pytest.raises(
            RuntimeError, match=f"rank {last}: global batch differs from rank 0"
        ) as caught:
            sync_dkv_sample_results(dist, global_requests, local_requests)
        for other in range(1, last):
            assert f"rank {other}: global batch differs" not in str(caught.value)
        return str(caught.value).count("global batch differs")

    assert group.run(run_rank) == [1] * group_size
    assert [len(trace) for trace in group.traces] == [1] * group_size


@pytest.mark.parametrize("digest", [(1,), (1, 2, 3), "digest", (True, 1), (-1, 1), [1, 2], None])
def test_invalid_batch_digest_fails_on_every_rank(digest) -> None:
    group = LockstepTpGroup(2)

    def run_rank(dist):
        request = _request(7, 0, dist.tp_rank)
        _advance_context(request, 2)
        original_allgather = dist.tp_allgather

        def exchange(payload):
            if dist.tp_rank == 1:
                payload = replace(payload, batch_digest=digest)
            return original_allgather(payload)

        dist.tp_allgather = exchange
        with pytest.raises(RuntimeError, match="rank 1: invalid global batch digest"):
            sync_dkv_sample_results(dist, [request], [request] if request.py_dkv_is_local else [])

    group.run(run_rank)
    assert [len(trace) for trace in group.traces] == [1, 1]
