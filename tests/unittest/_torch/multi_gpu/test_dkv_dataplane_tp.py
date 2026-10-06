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
"""The data plane of the layer-split layout between real ranks, without a model.

Every rank holds a DeepSeek-V4 cache manager that stores only the layers it owns, a staging area and
a ``DkvStreamer`` that moves pages over NCCL. Instead of the attention of a layer, a kernel reads
the pages the streamer fetched into the slot of the layer, checks them against a pattern that
depends on the request, the layer, the cache role and the block, and writes the pattern into the
pages the new tokens touch. Prompts are prefilled in chunks over several iterations, so every
iteration but the first fetches what an earlier one wrote back. After the last iteration the pages
in the cache manager of the owner of each layer must hold the pattern.
"""

import pickle
import sys
from dataclasses import dataclass, field

import cloudpickle
import pytest
import torch
from mpi4py import MPI
from mpi4py.futures import MPIPoolExecutor
from utils.util import skip_pre_blackwell

import tensorrt_llm
from tensorrt_llm import Mapping
from tensorrt_llm._torch.attention.backends.sparse.deepseek_v4.cache_manager import (
    DeepseekV4CacheManager,
)
from tensorrt_llm._torch.pyexecutor.dkv import compute_ownership, owned_layers
from tensorrt_llm._torch.pyexecutor.dkv_plan import PlanRequest
from tensorrt_llm._torch.pyexecutor.dkv_staging import (
    BAD_PAGE_INDEX,
    DkvStagedKvView,
    StagingGeometry,
    StagingLayout,
    StagingPool,
)
from tensorrt_llm._torch.pyexecutor.dkv_streamer import DkvStreamer
from tensorrt_llm._torch.pyexecutor.dkv_transport import nccl_p2p_transport
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest, LlmRequestState, SamplingConfig
from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests
from tensorrt_llm.bindings import DataType
from tensorrt_llm.bindings.internal.batch_manager import CacheType
from tensorrt_llm.llmapi.llm_args import DeepSeekV4SparseAttentionConfig, KvCacheConfig

cloudpickle.register_pickle_by_value(sys.modules[__name__])
MPI.pickle.__init__(cloudpickle.dumps, cloudpickle.loads, pickle.HIGHEST_PROTOCOL)
pytestmark = [pytest.mark.threadleak(enabled=False), skip_pre_blackwell]

_RATIOS = [4, 128, 1] * 4
_CHUNK = 128
_TOKENS_PER_BLOCK = 128
_TIMEOUT = 120.0


@dataclass(frozen=True)
class Scenario:
    """Requests that prefill in chunks of ``_CHUNK`` tokens on the ranks of a group."""

    group_size: int
    ring_depth: int
    prompts: dict[int, int]
    ranks: dict[int, int]
    arrival: dict[int, int] = field(default_factory=dict)
    # The fault: ``(rank, n)`` flips a byte of the n-th message that the rank receives.
    flip: tuple[int, int] = (-1, -1)

    def batch(self, iteration: int) -> list[PlanRequest]:
        """The global scheduled batch of an iteration, with a dummy for every idle rank."""
        requests = []
        for request_id, prompt in self.prompts.items():
            start = self.arrival.get(request_id, 0)
            progress = min(prompt, max(0, iteration - start) * _CHUNK)
            if iteration >= start and progress < prompt:
                chunk = min(_CHUNK, prompt - progress)
                requests.append(PlanRequest(request_id, self.ranks[request_id], progress, chunk))
        busy = {request.compute_rank for request in requests}
        for rank in range(self.group_size):
            if rank not in busy:
                requests.append(PlanRequest(1000 + rank, rank, 0, 1, is_dummy=True))
        return requests

    @property
    def iterations(self) -> int:
        return max(
            self.arrival.get(request_id, 0) + -(-prompt // _CHUNK)
            for request_id, prompt in self.prompts.items()
        )


def _pattern(request_id: int, layer: int, index: int, blocks: list[int], nbytes: int):
    """The bytes of the pages of ``blocks``, one row per block, on the device."""
    seeds = torch.tensor(
        [(request_id * 131 + layer * 31 + index * 7 + block * 13) % 251 + 1 for block in blocks],
        dtype=torch.int64,
        device="cuda",
    )
    positions = torch.arange(nbytes, dtype=torch.int64, device="cuda")
    return ((positions[None, :] * 7 + seeds[:, None]) % 251).to(torch.uint8)


def _manager(owned: tuple[int, ...], rank: int, group_size: int) -> DeepseekV4CacheManager:
    return DeepseekV4CacheManager(
        kv_cache_config=KvCacheConfig(
            enable_block_reuse=False,
            enable_swa_scratch_reuse=False,
            max_tokens=16384,
            event_buffer_max_size=0,
        ),
        kv_cache_type=CacheType.SELFKONLY,
        num_layers=len(_RATIOS),
        num_kv_heads=1,
        head_dim=512,
        tokens_per_block=_TOKENS_PER_BLOCK,
        max_seq_len=4096,
        max_batch_size=8,
        max_input_len=4096,
        mapping=Mapping(
            world_size=group_size, rank=rank, tp_size=group_size, enable_attention_dp=True
        ),
        dtype=DataType.FP8,
        compressor_dtype=DataType.FLOAT,
        vocab_size=129280,
        max_num_tokens=1024,
        sparse_attn_config=DeepSeekV4SparseAttentionConfig(
            index_head_dim=128, window_size=128, compress_ratios=_RATIOS, indexer_k_dtype="fp8"
        ),
        owned_layers=owned,
    )


def _staged_pages(pool, layout, layer, component, span):
    return pool.slot(layer, component).view(-1, layout.page_bytes(component))[
        span.page_offset : span.page_offset + span.num_pages
    ]


def _attention(layer, local, spans, layout, pool, checks):
    """Stand-in for the attention of a layer: check what was fetched, write the new pages."""
    for index, request in enumerate(local):
        history, chunk = request.context_current_position, request.context_chunk_size
        for kind in layout.kinds:
            if layer not in layout.layers_of(kind):
                continue
            span = spans[kind][index]
            for component in layout.components_of(kind):
                slot = layout.components.index(component)
                nbytes = layout.page_bytes(component)
                pages = _staged_pages(pool, layout, layer, component, span)
                first, count = layout.fetch_range(kind, history, chunk)
                if count:
                    blocks = list(range(first, first + count))
                    rows = pages[first - span.first_block : first - span.first_block + count]
                    wrong = (rows != _pattern(request.request_id, layer, slot, blocks, nbytes)).any(
                        dim=1
                    )
                    checks.append((layer, request.request_id, component, blocks, wrong))
                first, count = layout.writeback_range(kind, history, chunk)
                if count:
                    blocks = list(range(first, first + count))
                    pages[first - span.first_block : first - span.first_block + count] = _pattern(
                        request.request_id, layer, slot, blocks, nbytes
                    )


def _record_writes(written, batch, layout, manager) -> None:
    """Note the pages that the iteration writes back into the cache manager of this rank."""
    for request in batch:
        history, chunk = request.context_current_position, request.context_chunk_size
        for kind in layout.kinds:
            first, count = layout.writeback_range(kind, history, chunk)
            for component in layout.components_of(kind):
                for layer in layout.layers_of(kind):
                    if (layer, component.attention_type) in manager._layer_attn_to_layer_id:
                        key = (request.request_id, layer, component.attention_type)
                        written.setdefault(key, set()).update(range(first, first + count))


class _FlippingTransport:
    """A transport that flips a byte of one of the messages it receives."""

    def __init__(self, transport, number: int) -> None:
        self._transport = transport
        self._number = number
        self._received = 0

    def send(self, buffer, peer, stream) -> None:
        self._transport.send(buffer, peer, stream)

    def recv(self, buffer, peer, stream) -> None:
        self._transport.recv(buffer, peer, stream)
        if self._received == self._number:
            with torch.cuda.stream(stream):
                buffer[buffer.numel() // 2].bitwise_xor_(0xFF)
        self._received += 1


def _probe_rank(scenario: Scenario) -> dict:
    rank = tensorrt_llm.mpi_rank()
    group_size = scenario.group_size
    torch.cuda.set_device(rank % torch.cuda.device_count())
    owners = compute_ownership(len(_RATIOS), group_size)
    transport = nccl_p2p_transport(group_size, rank)
    manager = _manager(owned_layers(owners, rank), rank, group_size)
    geometry = StagingGeometry.from_cache_manager(
        manager, max_staging_tokens=8192, ring_depth=scenario.ring_depth
    )
    layout = StagingLayout(geometry)
    pool = StagingPool(layout)
    view = DkvStagedKvView(manager, pool)
    data_stream = torch.cuda.Stream()
    transport.warmup(data_stream)
    flipping = scenario.flip[0] == rank
    streamer = DkvStreamer(
        manager,
        view,
        _FlippingTransport(transport, scenario.flip[1]) if flipping else transport,
        owners,
        rank,
        group_size=group_size,
        data_stream=data_stream,
    )
    view.dkv_streamer = streamer
    requests = {
        request_id: LlmRequest(
            request_id=request_id,
            max_new_tokens=16,
            input_tokens=list(range(prompt)),
            sampling_config=SamplingConfig(),
            is_streaming=False,
        )
        for request_id, prompt in scenario.prompts.items()
    }
    errors: list[str] = []
    written: dict[tuple[int, int, object], set[int]] = {}
    try:
        for iteration in range(scenario.iterations):
            batch = scenario.batch(iteration)
            real = [request for request in batch if not request.is_dummy]
            scheduled = ScheduledRequests()
            for plan_request in real:
                request = requests[plan_request.request_id]
                if request.py_request_id not in manager.kv_cache_map:
                    assert manager.prepare_context(request)
                request.context_chunk_size = plan_request.context_chunk_size
                assert manager.resize_context(request, plan_request.context_chunk_size)
                scheduled.append_context_request(request)
            manager.prepare_resources(scheduled)
            local = [request for request in real if request.compute_rank == rank]
            streamer.set_plan(streamer.plan_for(batch))
            view.begin_staged_batch(
                [r.request_id for r in local],
                [r.context_current_position for r in local],
                [r.context_chunk_size for r in local],
                len(local),
            )
            placement = [(r.context_current_position, r.context_chunk_size) for r in local]
            spans = {kind: layout.request_spans(kind, placement) for kind in layout.kinds}
            checks: list = []
            for layer in range(len(_RATIOS)):
                streamer.on_layer(layer)
                _attention(layer, local, spans, layout, pool, checks)
            _record_writes(written, real, layout, manager)
            streamer.end_forward()
            streamer.drain(_TIMEOUT)
            torch.cuda.synchronize()
            for layer, request_id, component, blocks, wrong in checks:
                for block, bad in zip(blocks, wrong.tolist()):
                    if bad:
                        errors.append(
                            f"rank {rank}, iteration {iteration}: block {block} of request "
                            f"{request_id} in layer {layer} ({component.attention_type.name}) was "
                            "not fetched"
                        )
            for plan_request in real:
                request = requests[plan_request.request_id]
                request.move_to_next_context_chunk()
                if request.context_remaining_length == 0:
                    request.add_new_token(0, 0)
                    request.state = LlmRequestState.GENERATION_IN_PROGRESS
            manager.update_context_resources(scheduled)
        errors.extend(_check_owner_pages(rank, layout, manager, written))
        stats = streamer.stats
        return {
            "errors": errors,
            "bytes_sent": stats.bytes_sent,
            "bytes_received": stats.bytes_received,
            "bytes_local": stats.bytes_local,
            "iterations": stats.iterations,
        }
    finally:
        for request in requests.values():
            if request.py_request_id in manager.kv_cache_map:
                manager.free_resources(request)
        manager.shutdown()


def _check_owner_pages(rank, layout, manager, written) -> list[str]:
    problems = []
    for (request_id, layer, attention_type), blocks in sorted(
        written.items(), key=lambda item: (item[0][0], item[0][1], item[0][2].value)
    ):
        if (layer, attention_type) not in manager._layer_attn_to_layer_id:
            continue
        component = next(
            c
            for c in layout.components
            if c.attention_type is attention_type and layer in layout.layers_of(c.kind)
        )
        slot = layout.components.index(component)
        indices = manager.get_cache_indices(request_id, layer, attention_type)
        buffer = manager.get_buffers(layer, attention_type)
        for block in sorted(blocks):
            if block >= len(indices) or indices[block] == BAD_PAGE_INDEX:
                continue
            page = buffer[indices[block]].view(torch.uint8).reshape(-1)
            expected = _pattern(request_id, layer, slot, [block], layout.page_bytes(component))[0]
            if not torch.equal(page, expected):
                problems.append(
                    f"rank {rank}: block {block} of request {request_id} in layer {layer} "
                    f"({attention_type.name}) was not written back"
                )
    return problems


def _spread(group_size: int, prompts: dict[int, int]) -> dict[int, int]:
    return {request_id: request_id % group_size for request_id in prompts}


_PROMPTS = {1: 300, 2: 700, 3: 150, 4: 1000, 5: 90}


@pytest.mark.parametrize("ring_depth", [1, 2, 3])
@pytest.mark.parametrize("placement", ["spread", "one_rank"])
@pytest.mark.parametrize("mpi_pool_executor", [2, 4], indirect=True)
def test_the_pages_of_every_layer_reach_the_compute_rank_and_come_back(
    mpi_pool_executor: MPIPoolExecutor, placement: str, ring_depth: int
) -> None:
    group_size = mpi_pool_executor.num_workers
    if torch.cuda.device_count() < group_size:
        pytest.skip(f"Requires {group_size} GPUs")
    ranks = _spread(group_size, _PROMPTS) if placement == "spread" else {rid: 0 for rid in _PROMPTS}
    scenario = Scenario(group_size, ring_depth, _PROMPTS, ranks, arrival={4: 1, 5: 2})
    results = list(mpi_pool_executor.map(_probe_rank, [scenario] * group_size, timeout=900))
    for rank, result in enumerate(results):
        assert result["errors"] == [], f"rank {rank}"
        assert result["iterations"] == scenario.iterations
    assert sum(r["bytes_sent"] for r in results) == sum(r["bytes_received"] for r in results)
    assert sum(r["bytes_sent"] for r in results) > 0


@pytest.mark.parametrize("number", [3, 25])
@pytest.mark.parametrize("mpi_pool_executor", [2], indirect=True)
def test_a_byte_flipped_in_a_received_message_is_noticed(
    mpi_pool_executor: MPIPoolExecutor, number: int
) -> None:
    if torch.cuda.device_count() < 2:
        pytest.skip("Requires two GPUs")
    scenario = Scenario(2, 2, {1: 400, 2: 520}, {1: 0, 2: 1}, flip=(0, number))
    results = list(mpi_pool_executor.map(_probe_rank, [scenario] * 2, timeout=900))
    assert any(result["errors"] for result in results), "the flipped byte went unnoticed"
