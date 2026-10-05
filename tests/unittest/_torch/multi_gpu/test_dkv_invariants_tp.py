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
"""Exercise DKV agreement and negative injections over a real two-rank MPI group."""

import pickle
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import cloudpickle
import pytest
import torch
from mpi4py import MPI
from mpi4py.futures import MPIPoolExecutor

import tensorrt_llm
from tensorrt_llm import Mapping
from tensorrt_llm._torch.distributed.communicator import MPIDist
from tensorrt_llm._torch.pyexecutor.dkv import DkvInvariantChecker
from tensorrt_llm._torch.pyexecutor.executor_request_queue import RequestQueueItem
from tensorrt_llm._torch.pyexecutor.llm_request import ExecutorRequest, SamplingConfig
from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor
from tensorrt_llm._torch.pyexecutor.scheduler import ADPRouter, ScheduledRequests

cloudpickle.register_pickle_by_value(sys.modules[__name__])
MPI.pickle.__init__(cloudpickle.dumps, cloudpickle.loads, pickle.HIGHEST_PROTOCOL)
pytestmark = pytest.mark.threadleak(enabled=False)


def _check_rank(scenario: str) -> bool:
    rank = tensorrt_llm.mpi_rank()
    torch.cuda.set_device(rank)
    dist = MPIDist(Mapping(world_size=2, rank=rank, tp_size=2))
    if scenario == "enable_flags":
        with pytest.raises(RuntimeError, match="enable flags differ"):
            DkvInvariantChecker(dist, enabled=bool(rank))
        return True
    checker = DkvInvariantChecker(dist, enabled=True)
    checker.check_many(
        0,
        {
            "startup configuration": [("slots", 32), ("mapper", 10)],
            "global request order": [(7, 0), (8, 1)],
        },
    )
    for iteration in range(3):
        requests = [(7, 0), (8, 1), (-1, 0)]
        if scenario == "request_order" and rank:
            requests.reverse()
        if scenario == "request_order":
            with pytest.raises(RuntimeError, match="DKV invariant violation"):
                checker.check(iteration, "global request order", requests)
        else:
            checker.check(iteration, "global request order", requests)
    return True


@pytest.mark.parametrize("scenario", ["consistent", "enable_flags", "request_order"])
@pytest.mark.parametrize("mpi_pool_executor", [2], indirect=True)
def test_dkv_checker_two_rank_mpi(mpi_pool_executor: MPIPoolExecutor, scenario: str) -> None:
    if torch.cuda.device_count() < 2:
        pytest.skip("Requires two GPUs")
    results = mpi_pool_executor.map(_check_rank, [scenario] * 2, timeout=60)
    assert list(results) == [True, True]


def _check_executor_rank(scenario: str) -> bool:
    rank = tensorrt_llm.mpi_rank()
    torch.cuda.set_device(rank)
    dist = MPIDist(Mapping(world_size=2, rank=rank, tp_size=2, enable_attention_dp=True))
    executor = PyExecutor.__new__(PyExecutor)
    executor.dist = dist
    executor.dkv_enabled = True
    executor.enable_attention_dp = True
    executor.enable_iter_perf_stats = False
    executor._enable_non_overlap_adp_forward_intent = True
    executor.num_fetch_requests = 0
    executor.num_fetch_requests_cur_rank = 0
    executor.max_num_active_requests = 8
    executor.is_shutdown = False
    executor.drafter = None
    executor.model_engine = SimpleNamespace(is_spec_decode=False)
    executor.kv_cache_transceiver = None
    executor.scheduler = SimpleNamespace(dkv_dual_ledger_enabled=False)
    executor._disagg_coordinator = Mock()
    executor._release_unused_connector_reservations = Mock()
    executor._poll_encoder_steps = Mock()
    executor._prefetch_for_context_requests = Mock()
    executor._fetch_and_enqueue_requests = Mock()
    executor._should_exclude_last_generation_logits = Mock(return_value=False)
    executor.kv_cache_manager = SimpleNamespace(
        enable_block_reuse=True,
        probe_prefix_match_length=lambda *args, **kwargs: 1
        + (rank if scenario == "prefix_length" else 0),
        get_dkv_config_fingerprint=lambda: [("slots", 32)],
    )
    executor.adp_router = ADPRouter.create(
        dist, False, kv_cache_manager=executor.kv_cache_manager, dkv_enabled=True
    )
    executor._dkv_invariant_checker = DkvInvariantChecker(dist, enabled=True)
    executor._dkv_invariant_checker.check_many(
        -1,
        {
            "startup configuration": {
                "dual_ledger_enabled": executor.scheduler.dkv_dual_ledger_enabled
            }
        },
    )

    def fetch_and_activate():
        executor.active_requests = executor._fetch_new_requests(None, [])
        return executor.active_requests

    def schedule():
        scheduled = ScheduledRequests()
        for request in executor.active_requests:
            request.context_chunk_size = request.orig_prompt_len
        scheduled.reset_context_requests(executor.active_requests)
        return scheduled, [], scheduled.batch_size

    executor._fetch_and_activate_new_requests = fetch_and_activate
    executor._schedule = schedule
    for iteration in range(3):
        executor.iter_counter = iteration
        executor.active_requests = []
        items = [
            RequestQueueItem(
                iteration * 10 + index,
                ExecutorRequest(
                    input_token_ids=list(range(length)),
                    max_tokens=1,
                    sampling_config=SamplingConfig(1),
                ),
            )
            for index, length in enumerate([4, 8, 6])
        ]
        executor._pop_from_waiting_queue = Mock(return_value=items)
        trace = [("prepare", item.id, 1) for item in items]
        if scenario == "kv_order" and rank:
            trace.reverse()
        executor.kv_cache_manager.consume_dkv_trace = Mock(return_value=trace)
        if scenario == "consistent":
            scheduled, _ = executor._prepare_and_schedule_batch()
            assert [request.py_request_id for request in scheduled.context_requests] == [
                item.id for item in items
            ]
            executor.kv_cache_manager.consume_dkv_trace.assert_called_once_with()
        else:
            checkpoint = "prefix probes" if scenario == "prefix_length" else "scheduling decisions"
            with pytest.raises(RuntimeError, match=checkpoint):
                executor._prepare_and_schedule_batch()
    return True


@pytest.mark.parametrize("scenario", ["consistent", "prefix_length", "kv_order"])
@pytest.mark.parametrize("mpi_pool_executor", [2], indirect=True)
def test_dkv_executor_scheduling_checks_two_rank_mpi(
    mpi_pool_executor: MPIPoolExecutor, scenario: str
) -> None:
    if torch.cuda.device_count() < 2:
        pytest.skip("Requires two GPUs")
    results = mpi_pool_executor.map(_check_executor_rank, [scenario] * 2, timeout=60)
    assert list(results) == [True, True]
