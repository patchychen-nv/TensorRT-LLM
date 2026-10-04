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
"""Self-tests for strict DKV collectives and real-request fixtures."""

import threading

import numpy as np
import pytest
from dkv_test_utils import LockstepDistributed, LockstepTpGroup, make_request
from utils.collectives import ThreadSafeDistributed, run_concurrent

from tensorrt_llm._torch.distributed.communicator import ReduceOp
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest
from tensorrt_llm._torch.pyexecutor.scheduler.adp_router import DefaultADPRouter

pytestmark = pytest.mark.cpu_only


def test_collectives_preserve_each_rank_payload() -> None:
    group = LockstepTpGroup(3)

    def exchange(dist: LockstepDistributed) -> tuple[object, ...]:
        rank = dist.tp_rank
        gathered = dist.tp_allgather({"rank": rank})
        integers = dist.tp_allgather_int64([rank, rank + 10])
        root_result = dist.tp_gather(rank, root=1)
        broadcast = dist.broadcast({"rank": rank}, root=2)
        broadcast_int = dist.broadcast_int64([rank], root=1)
        return gathered, integers, root_result, broadcast, broadcast_int

    for rank, result in enumerate(group.run(exchange)):
        gathered, integers, root_result, broadcast, broadcast_int = result
        assert gathered == [{"rank": index} for index in range(3)]
        np.testing.assert_array_equal(integers, [[0, 10], [1, 11], [2, 12]])
        assert root_result == ([0, 1, 2] if rank == 1 else None)
        assert broadcast == {"rank": 2}
        np.testing.assert_array_equal(broadcast_int, [1])
    assert all(trace == group.traces[0] for trace in group.traces)


@pytest.mark.parametrize(
    ("operation", "expected"),
    [
        (ReduceOp.SUM, 6),
        (ReduceOp.PRODUCT, 6),
        (ReduceOp.MIN, 1),
        (ReduceOp.MAX, 3),
        (ReduceOp.BAND, 0),
        (ReduceOp.BOR, 3),
        (ReduceOp.BXOR, 0),
    ],
)
def test_reduce_operations(operation: ReduceOp, expected: int) -> None:
    results = LockstepTpGroup(3).run(lambda dist: dist.tp_allreduce(dist.tp_rank + 1, operation))
    assert results == [expected] * 3


def test_extra_collective_fails_and_joins_threads() -> None:
    before = set(threading.enumerate())
    group = LockstepTpGroup(2)

    def exchange(dist: LockstepDistributed) -> None:
        dist.tp_allgather(dist.tp_rank)
        if dist.tp_rank:
            dist.tp_allgather("extra")

    with pytest.raises(AssertionError, match="finished|unmatched collective"):
        group.run(exchange)
    assert set(threading.enumerate()) == before


@pytest.mark.parametrize("different_method", [False, True])
def test_collective_callsite_and_kind_must_match(different_method: bool) -> None:
    group = LockstepTpGroup(2)

    def exchange(dist: LockstepDistributed) -> None:
        if dist.tp_rank == 0:
            dist.tp_allgather(0)
        elif different_method:
            dist.broadcast(1)
        else:
            dist.tp_allgather(1)

    with pytest.raises(AssertionError, match="Collective sequence mismatch"):
        group.run(exchange)


def test_skipped_collective_times_out_with_trace() -> None:
    group = LockstepTpGroup(2, timeout=0.02)
    before = set(threading.enumerate())
    timed_out = threading.Event()

    def exchange(dist: LockstepDistributed) -> None:
        if dist.tp_rank:
            assert timed_out.wait(timeout=1)
        try:
            dist.tp_allgather(dist.tp_rank)
        finally:
            if dist.tp_rank == 0:
                timed_out.set()

    with pytest.raises(AssertionError, match="Collective timeout.*\n.*rank0:.*\n.*rank1:"):
        group.run(exchange)
    assert set(threading.enumerate()) == before


def test_unwired_collective_is_explicit_failure() -> None:
    with pytest.raises(AssertionError, match="unwired collective: cp_allgather"):
        LockstepTpGroup(1).ranks[0].cp_allgather(0)


def test_worker_failure_unblocks_peers_and_preserves_original_error() -> None:
    before = set(threading.enumerate())

    def fail(dist: LockstepDistributed) -> None:
        if dist.tp_rank:
            raise RuntimeError("rank1 source failure")
        dist.tp_allgather(0)

    with pytest.raises(RuntimeError, match="rank1 source failure"):
        LockstepTpGroup(2).run(fail)
    assert set(threading.enumerate()) == before


def test_integer_collective_shape_must_match() -> None:
    with pytest.raises(AssertionError, match="Collective sequence mismatch"):
        LockstepTpGroup(2).run(lambda dist: dist.tp_allgather_int64([0] * (dist.tp_rank + 1)))


def test_shared_transfer_harness_keeps_real_rank_payloads() -> None:
    shared = {"barrier": threading.Barrier(2), "lock": threading.Lock()}
    ranks = [ThreadSafeDistributed(rank, 2, 2, 1, rank, 0, shared) for rank in range(2)]
    assert run_concurrent(ranks, lambda dist: dist.tp_allgather(dist.rank)) == [[0, 1], [0, 1]]


def test_shared_transfer_harness_independent_tp_pp_cp_groups() -> None:
    shared = {
        "barrier": threading.Barrier(8),
        "lock": threading.Lock(),
        "pp_barriers": [threading.Barrier(2) for _ in range(4)],
        "tp_barriers": [threading.Barrier(2) for _ in range(4)],
    }
    ranks = [
        ThreadSafeDistributed(
            rank,
            8,
            2,
            2,
            (rank // 2) % 2,
            rank // 4,
            shared,
            cp_rank=rank % 2,
            cp_size=2,
        )
        for rank in range(8)
    ]

    def exchange(dist: ThreadSafeDistributed) -> tuple[list[object], list[object]]:
        for _ in range((dist.rank // 2) % 2 + 1):
            pp = dist.pp_allgather(dist.rank)
        for _ in range(dist.rank // 4 + 1):
            tp = dist.tp_allgather(dist.rank)
        return pp, tp

    for rank, (pp, tp) in enumerate(run_concurrent(ranks, exchange)):
        assert pp == [rank % 4, rank % 4 + 4]
        assert tp == [(rank // 4) * 4 + rank % 2, (rank // 4) * 4 + rank % 2 + 2]


def test_request_factory_sets_concrete_tags() -> None:
    request = make_request(7, compute_rank=1, local_rank=0, prompt_len=5, is_dummy=True)
    assert isinstance(request, LlmRequest)
    assert request.py_request_id == 7
    assert request.py_orig_prompt_len == 5
    assert request.py_dkv_compute_rank == 1
    assert request.py_dkv_is_local is False
    assert request.is_attention_dp_dummy is True


def test_dkv_router_counts_only_local_replicated_requests() -> None:
    group = LockstepTpGroup(2)

    def gather(dist: LockstepDistributed) -> object:
        requests = [
            make_request(index, compute_rank=owner, local_rank=dist.tp_rank, prompt_len=length)
            for index, owner, length in [(0, 0, 5), (1, 1, 7), (2, 1, 11)]
        ]
        router = DefaultADPRouter(dist=dist)
        router.dkv_enabled = True
        return router.gather_all_rank_states(requests)

    for states in group.run(gather):
        assert [state.num_active_requests for state in states] == [1, 2]
        assert [state.num_active_tokens for state in states] == [5, 18]
