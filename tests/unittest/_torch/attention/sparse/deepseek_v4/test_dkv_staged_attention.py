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
"""A/B of the DeepSeek-V4 attention layers on the cache manager against its staged view.

Five attention layers (one with sliding-window attention only, two CSA and two HCA layers), each
with weights of its own, run a schedule of iterations on two cache managers: a prefill, a chunked
prefill on a cached prefix next to a decode step (a mixed batch) and a decode step of two requests.
The layers read and write the pages of the first manager directly. The second manager is behind a
``DkvStagedKvView``: a ``LoopbackStreamer`` copies the cached pages of a layer into its slot of the
staging area before the layer runs and the pages the new tokens wrote back after it, so the kernels
only ever see the staging area. The outputs of every layer and the pages of both managers must be
the same after every iteration. The first manager is run twice, to know how much the layers differ
from themselves.

Three more checks show that the comparison is sensitive and that the kernels use the staging area
alone. Bytes flipped in the staging area after the fetch must change the result, for every kind of
storage. And while a layer runs, the pages it has in the cache manager are poisoned: the result
must not change, and the kernels must not write them.
"""

import weakref
from dataclasses import dataclass
from functools import lru_cache

import pytest
import torch
from utils.util import skip_blackwell_geforce, skip_pre_blackwell

from tensorrt_llm._torch.attention.backends.sparse.deepseek_v4 import (
    DeepseekV4CacheManager,
    DeepseekV4TrtllmAttentionMetadata,
)
from tensorrt_llm._torch.attention.backends.sparse.deepseek_v4.params import DeepseekV4AttentionType
from tensorrt_llm._torch.configs.deepseekv4 import DeepseekV4Config
from tensorrt_llm._torch.metadata import KVCacheParams
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_deepseekv4 import DeepseekV4Attention
from tensorrt_llm._torch.pyexecutor.dkv_staging import (
    BAD_PAGE_INDEX,
    DkvStagedKvView,
    StagingGeometry,
    StagingKind,
    StagingLayout,
    StagingPool,
    attention_types,
)
from tensorrt_llm._torch.pyexecutor.dkv_streamer import LoopbackStreamer
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest, LlmRequestState, SamplingConfig
from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests
from tensorrt_llm._torch.utils import AuxStreamType, model_extra_attrs
from tensorrt_llm.bindings import DataType
from tensorrt_llm.bindings.internal.batch_manager import CacheType
from tensorrt_llm.llmapi.llm_args import DeepSeekV4SparseAttentionConfig, KvCacheConfig
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantAlgo

pytestmark = [skip_pre_blackwell, skip_blackwell_geforce]

_RATIOS = [1, 4, 128, 4, 128]
_CSA_RATIO = 4
_TOKENS_PER_BLOCK = 128
_MAX_SEQ_LEN = 4096
_MAX_BATCH_SIZE = 2
_MAX_NUM_TOKENS = 4096
_VOCAB_SIZE = 129280
_HIDDEN_SIZE = 4096
_MODEL = {
    "architectures": ["DeepseekV4ForCausalLM"],
    "model_type": "deepseek_v4",
    "hidden_size": _HIDDEN_SIZE,
    "num_attention_heads": 64,
    "num_key_value_heads": 1,
    "qk_nope_head_dim": 448,
    "qk_rope_head_dim": 64,
    "v_head_dim": 512,
    "q_lora_rank": 1024,
    "kv_lora_rank": 448,
    "o_groups": 8,
    "o_lora_rank": 1024,
    "max_position_embeddings": 65536,
    "rms_norm_eps": 1e-6,
    "dtype": "bfloat16",
    "vocab_size": _VOCAB_SIZE,
    "num_hidden_layers": len(_RATIOS),
    "compress_rope_theta": 40000.0,
    "rope_theta": 10000.0,
    "rope_scaling": {
        "type": "yarn",
        "factor": 4.0,
        "original_max_position_embeddings": 65536,
        "beta_fast": 32,
        "beta_slow": 1,
    },
}


@dataclass(frozen=True)
class _Step:
    """One iteration: ``(request, chunk)`` of the context requests, then the decoding requests."""

    contexts: tuple[tuple[int, int], ...] = ()
    generations: tuple[int, ...] = ()


@dataclass(frozen=True)
class _Workload:
    """The prompts of two requests and the iterations that compute them."""

    prompts: tuple[int, int]
    schedule: tuple[_Step, ...]


# Request 1 is prefilled in two chunks, the first next to the whole prompt of request 0; the second
# computes on 300 cached tokens, with request 0 decoding beside it. Then both decode.
#
# The prompt of 1500 tokens has 375 compressed tokens, fewer than the 512 an indexer selects, so it
# selects all of them. Every layer then reproduces itself bit for bit.
_SELECTS_ALL = _Workload(
    (300, 1500),
    (
        _Step(contexts=((0, 300), (1, 300))),
        _Step(contexts=((1, 1200),), generations=(0,)),
        _Step(generations=(0, 1)),
    ),
)
# The prompt of 2800 tokens has 700, so the indexer has to choose among them. The layers with an
# indexer then differ from run to run, by about a unit in the last place of the output.
_SELECTS_SOME = _Workload(
    (300, 2800),
    (
        _Step(contexts=((0, 300), (1, 300))),
        _Step(contexts=((1, 2500),), generations=(0,)),
        _Step(generations=(0, 1)),
    ),
)


@dataclass
class _Model:
    config: ModelConfig
    sparse: DeepSeekV4SparseAttentionConfig
    layers: list[DeepseekV4Attention]
    fp8_kv: bool


@lru_cache(maxsize=None)
def _model(indexer: str, fp8_kv: bool) -> _Model:
    config = DeepseekV4Config(**_MODEL)
    config.dtype = torch.bfloat16
    config.mapping = Mapping(world_size=1, tp_size=1, rank=0)
    config.tie_word_embeddings = False
    sparse = DeepSeekV4SparseAttentionConfig(
        index_n_heads=64,
        index_head_dim=128,
        window_size=128,
        compress_ratios=_RATIOS,
        index_topk=512,
        indexer_k_dtype=indexer,
        skip_indexer_for_short_seqs=False,
    )
    config.sparse_attention_config = sparse
    model_config = ModelConfig(
        pretrained_config=config,
        sparse_attention_config=sparse,
        attn_backend="TRTLLM",
        quant_config=QuantConfig(kv_cache_quant_algo=QuantAlgo.FP8 if fp8_kv else None),
    )
    model_config.extra_attrs["kv_cache_dtype"] = "fp8" if fp8_kv else "auto"
    streams = [torch.cuda.Stream() for _ in range(4)]
    aux_streams = {
        AuxStreamType.Attention: streams[0],
        AuxStreamType.MoeShared: streams[0],
        AuxStreamType.MoeChunkingOverlap: streams[1],
        AuxStreamType.MoeBalancer: streams[2],
        AuxStreamType.MoeOutputMemset: streams[3],
        AuxStreamType.MlaCompressor: streams[1],
        AuxStreamType.MlaIndexer: streams[2],
        AuxStreamType.MlaIndexerAux: streams[3],
    }
    layers = [
        DeepseekV4Attention(model_config, layer_idx=layer, aux_stream_dict=aux_streams).to("cuda")
        for layer in range(len(_RATIOS))
    ]
    # The weights are not loaded from a checkpoint: give every layer finite weights of its own.
    generator = torch.Generator(device="cuda").manual_seed(2718)
    for layer in layers:
        for name, parameter in layer.named_parameters():
            assert parameter.is_floating_point(), name
            if "norm" in name and name.endswith(".weight"):
                parameter.data.fill_(1.0)
            elif "norm" in name and name.endswith(".bias"):
                parameter.data.zero_()
            else:
                parameter.data.normal_(0.0, 0.02, generator=generator)
    return _Model(model_config, sparse, layers, fp8_kv)


def _create_manager(model: _Model) -> DeepseekV4CacheManager:
    return DeepseekV4CacheManager(
        kv_cache_config=KvCacheConfig(
            dtype="fp8" if model.fp8_kv else "auto",
            enable_block_reuse=False,
            enable_swa_scratch_reuse=False,
            max_tokens=_MAX_SEQ_LEN * _MAX_BATCH_SIZE,
            event_buffer_max_size=0,
        ),
        kv_cache_type=CacheType.SELFKONLY,
        num_layers=len(_RATIOS),
        num_kv_heads=1,
        head_dim=512,
        tokens_per_block=_TOKENS_PER_BLOCK,
        max_seq_len=_MAX_SEQ_LEN,
        max_batch_size=_MAX_BATCH_SIZE,
        mapping=Mapping(world_size=1, tp_size=1, rank=0),
        dtype=DataType.FP8 if model.fp8_kv else DataType.BF16,
        compressor_dtype=DataType.FLOAT,
        vocab_size=_VOCAB_SIZE,
        max_num_tokens=_MAX_NUM_TOKENS,
        sparse_attn_config=model.sparse,
        model_config=model.config,
    )


@dataclass
class _Result:
    outputs: list[list[torch.Tensor]]  # [iteration][layer]
    pages: list[dict[tuple, torch.Tensor]]  # [iteration] {(request, layer, role, block): bytes}


class _Flow:
    """The schedule on one cache manager, directly or through a staged view of it."""

    def __init__(self, model: _Model, workload: _Workload, *, make_streamer=None) -> None:
        """``make_streamer(manager, view)`` returns the streamer that stages the layers; ``None``:
        the layers use the pages of the manager."""
        self.model = model
        self.workload = workload
        self.manager = _create_manager(model)
        # The pages start the same in every run, so that the bytes a request does not write compare.
        for layer, role in self.manager._layer_attn_to_layer_id:
            self.manager.get_buffers(layer, role).view(torch.uint8).zero_()
        self.view = None
        self.streamer = None
        if make_streamer is not None:
            geometry = StagingGeometry.from_cache_manager(self.manager, max_staging_tokens=8192)
            self.view = DkvStagedKvView(self.manager, StagingPool(StagingLayout(geometry)))
            self.streamer = make_streamer(self.manager, self.view)
            self.view.dkv_streamer = self.streamer

    def run(self) -> _Result:
        manager = self.manager
        requests = [
            LlmRequest(
                request_id=index,
                max_new_tokens=16,
                input_tokens=list(range(prompt)),
                sampling_config=SamplingConfig(),
                is_streaming=False,
            )
            for index, prompt in enumerate(self.workload.prompts)
        ]
        cached = [0] * len(requests)
        result = _Result([], [])
        try:
            for iteration, step in enumerate(self.workload.schedule):
                batch = ScheduledRequests()
                order = [index for index, _ in step.contexts] + list(step.generations)
                for index, chunk in step.contexts:
                    request = requests[index]
                    if request.py_request_id not in manager.kv_cache_map:
                        assert manager.prepare_context(request)
                    request.context_chunk_size = chunk
                    assert manager.resize_context(request, chunk)
                    batch.append_context_request(request)
                for index in step.generations:
                    assert manager.try_allocate_generation(requests[index])
                    batch.generation_requests.append(requests[index])
                manager.prepare_resources(batch)
                new_tokens = [chunk for _, chunk in step.contexts] + [1] * len(step.generations)
                history = [cached[index] for index in order]
                result.outputs.append(self._forward(iteration, step, history, new_tokens))
                # What the executor does when the iteration is over.
                for index, chunk in step.contexts:
                    request = requests[index]
                    request.move_to_next_context_chunk()
                    cached[index] += chunk
                    if request.context_remaining_length == 0:
                        request.add_new_token(0, 0)
                        request.state = LlmRequestState.GENERATION_IN_PROGRESS
                manager.update_context_resources(batch)
                for index in step.generations:
                    requests[index].add_new_token(0, 0)
                    cached[index] += 1
                manager.update_resources(batch)
                result.pages.append(self._pages(requests, cached))
        finally:
            for request in requests:
                if request.py_request_id in manager.kv_cache_map:
                    manager.free_resources(request)
            manager.shutdown()
        return result

    def _forward(self, iteration, step, history, new_tokens) -> list[torch.Tensor]:
        model = self.model
        order = [index for index, _ in step.contexts] + list(step.generations)
        metadata = DeepseekV4TrtllmAttentionMetadata(
            seq_lens=torch.tensor(new_tokens, dtype=torch.int32),
            num_contexts=len(step.contexts),
            max_num_requests=len(order),
            kv_cache_params=KVCacheParams(use_cache=True, num_cached_tokens_per_seq=history),
            kv_cache_manager=self.view if self.view is not None else self.manager,
            request_ids=order,
            # The executor passes the tokens of the chunk for a context request.
            prompt_lens=[chunk for _, chunk in step.contexts]
            + [self.workload.prompts[index] for index in step.generations],
            max_num_tokens=8192,
            mapping=Mapping(world_size=1, tp_size=1, rank=0),
            sparse_attention_config=model.sparse,
        )
        positions = torch.cat(
            [torch.arange(past, past + new) for past, new in zip(history, new_tokens)]
        )
        position_ids = positions.unsqueeze(0).to(torch.int32).cuda()
        extra_attrs = model.config.extra_attrs
        extra_attrs["attention_metadata"] = weakref.ref(metadata)
        outputs = []
        with torch.inference_mode(), model_extra_attrs(extra_attrs):
            metadata.prepare()
            for layer_index, layer in enumerate(model.layers):
                generator = torch.Generator(device="cuda").manual_seed(
                    1000 * iteration + layer_index
                )
                hidden = torch.randn(
                    sum(new_tokens),
                    _HIDDEN_SIZE,
                    generator=generator,
                    dtype=torch.bfloat16,
                    device="cuda",
                )
                output = layer(
                    position_ids=position_ids, hidden_states=hidden, attn_metadata=metadata
                )
                outputs.append(output.detach().clone())
            if self.streamer is not None:
                self.streamer.end_forward()
        torch.cuda.synchronize()
        return outputs

    def _pages(self, requests, cached: list[int]) -> dict[tuple, torch.Tensor]:
        """The pages of the requests, up to the rows that hold a token of the sequence.

        The rows of the last page behind the end of the sequence hold whatever the kernels left
        there, which depends on memory they do not own, so they are not compared. Neither are the
        pages that have left the window of their kind: the cache manager releases them before
        anything reads them, and the staged path does not send them back.
        """
        manager = self.manager
        layout = StagingLayout(StagingGeometry.from_cache_manager(manager, max_staging_tokens=8192))
        pages = {}
        for request in requests:
            request_id = request.py_request_id
            if request_id not in manager.kv_cache_map:
                continue
            for layer, role in manager._layer_attn_to_layer_id:
                ratio = _RATIOS[layer]
                compressed = role in (
                    DeepseekV4AttentionType.COMPRESS,
                    DeepseekV4AttentionType.INDEXER_COMPRESS,
                )
                rows = _TOKENS_PER_BLOCK // ratio if compressed else _TOKENS_PER_BLOCK
                kind = next(
                    component.kind
                    for component in layout.components
                    if component.attention_type is role
                    and layer in layout.layers_of(component.kind)
                )
                window = layout.window(kind)
                first_alive = (
                    0
                    if window is None
                    else max(0, cached[request_id] + 1 - window) // _TOKENS_PER_BLOCK
                )
                indices = manager.get_cache_indices(request_id, layer, role)
                buffer = manager.get_buffers(layer, role)
                for block, index in enumerate(indices):
                    if index == BAD_PAGE_INDEX or block < first_alive:
                        continue
                    tokens = min(
                        _TOKENS_PER_BLOCK, max(0, cached[request_id] - block * _TOKENS_PER_BLOCK)
                    )
                    valid_rows = tokens // ratio if compressed else tokens
                    page = buffer[index].view(torch.uint8).reshape(-1)
                    pages[request_id, layer, role, block] = page[
                        : page.numel() // rows * valid_rows
                    ].clone()
        return pages


# ---- streamers that misbehave on purpose -----------------------------------------------------


def _loopback(manager, view):
    return LoopbackStreamer(manager, view, fill="zero")


def _nan_loopback(manager, view):
    return LoopbackStreamer(manager, view, fill="nan")


class _FlipAfterFetch(LoopbackStreamer):
    """A streamer whose fetch is followed by ``corrupt(streamer, layer)``, as a bad copy would be."""

    def __init__(self, manager, view, corrupt) -> None:
        super().__init__(manager, view, fill="zero")
        self._corrupt = corrupt

    def _fetch(self, layer: int) -> None:
        super()._fetch(layer)
        self._corrupt(self, layer)


def _flip_kind(kind: StagingKind):
    """Invert every byte of the slots the layers of ``kind`` have in the staging area."""

    def corrupt(streamer: LoopbackStreamer, layer: int) -> None:
        layout = streamer._layout
        for component in layout.components:
            if component.kind is kind and layer in layout.layers_of(kind):
                streamer._pool.slot(layer, component).bitwise_xor_(0xFF)

    return corrupt


def _flip_one_bit(layer: int):
    """Flip a bit of the exponent of one value in the sliding-window page that holds the last
    cached token of the first request of the batch, in the slot of ``layer``."""

    def corrupt(streamer: LoopbackStreamer, current: int) -> None:
        history = streamer._history[0]
        if current != layer or not history:
            return
        layout = streamer._layout
        component = [c for c in layout.components if c.kind is StagingKind.SWA][0]
        block = (history - 1) // _TOKENS_PER_BLOCK
        span = streamer._spans[StagingKind.SWA][0]
        page = layout.block_table(StagingKind.SWA, layer, span, block + 1)[block]
        row_bytes = layout.page_bytes(component) // _TOKENS_PER_BLOCK
        row = (history - 1) % _TOKENS_PER_BLOCK
        streamer._pool.pages(component)[page, row * row_bytes + 1] ^= 0x40

    return corrupt


class _PoisonPages(LoopbackStreamer):
    """A streamer that poisons the pages of the layer in the cache manager while the layer runs.

    The pages are saved after the fetch and overwritten with NaN bytes. Before the pages are
    written back they must still be poisoned, which they are if no kernel wrote them, and they are
    restored.
    """

    def __init__(self, manager, view) -> None:
        super().__init__(manager, view, fill="zero")
        self._saved: list[tuple[torch.Tensor, torch.Tensor]] = []
        self.poisoned_pages = 0

    def _layer_pages(self, layer: int) -> list[torch.Tensor]:
        manager, layout = self._manager, self._layout
        pages = []
        for component in layout.components:
            if layer not in layout.layers_of(component.kind):
                continue
            for request_id in self._request_ids:
                if request_id not in manager.kv_cache_map:
                    continue
                indices = manager.get_cache_indices(request_id, layer, component.attention_type)
                buffer = manager.get_buffers(layer, component.attention_type)
                pages += [
                    buffer[index].view(torch.uint8).reshape(-1)
                    for index in indices
                    if index != BAD_PAGE_INDEX
                ]
        return pages

    def _fetch(self, layer: int) -> None:
        super()._fetch(layer)
        pages = self._layer_pages(layer)
        self._saved = [(page, page.clone()) for page in pages]
        for page in pages:
            page.fill_(0xFF)
        self.poisoned_pages += len(pages)

    def _write_back(self, layer: int) -> None:
        for page, saved in self._saved:
            assert (page == 0xFF).all(), f"a kernel wrote a page of layer {layer} of the manager"
            page.copy_(saved)
        self._saved = []
        super()._write_back(layer)


# ---- comparing runs ---------------------------------------------------------------------------


@lru_cache(maxsize=None)
def _reference(indexer: str, fp8_kv: bool, workload: _Workload) -> tuple[_Result, _Result]:
    """The schedule on the cache manager twice."""
    model = _model(indexer, fp8_kv)
    return _Flow(model, workload).run(), _Flow(model, workload).run()


def _difference(a: torch.Tensor, b: torch.Tensor) -> float:
    """The largest absolute difference of two tensors, infinite when either has a NaN."""
    difference = (a.float() - b.float()).abs()
    return float("inf") if difference.isnan().any() else difference.max().item()


def _differences_by_layer(left: _Result, right: _Result) -> list[float]:
    """For every layer, the largest difference of its outputs over the iterations."""
    return [
        max(_difference(a[layer], b[layer]) for a, b in zip(left.outputs, right.outputs))
        for layer in range(len(_RATIOS))
    ]


def _allowed_differences(
    reference: tuple[_Result, _Result], noise_factor: float = 4.0
) -> list[float]:
    """How far a run may be from the reference, for every layer.

    A layer that reproduces itself bit for bit has to be reproduced bit for bit. The layers with an
    indexer (CSA) do not reproduce, and no single pair of runs says how much: they may be as far
    from the reference as the largest difference between the two runs of the reference, times a
    factor.
    """
    noise = _differences_by_layer(*reference)
    pooled = noise_factor * max(noise)
    return [
        pooled if ratio == _CSA_RATIO or layer_noise > 0 else 0.0
        for ratio, layer_noise in zip(_RATIOS, noise)
    ]


def _differs(reference: tuple[_Result, _Result], other: _Result) -> bool:
    """Whether some layer of ``other`` is further from the reference than it may be."""
    difference = _differences_by_layer(reference[0], other)
    return any(d > allowed for d, allowed in zip(difference, _allowed_differences(reference)))


def _page_differences(left: _Result, right: _Result, skip_roles=()) -> list[str]:
    """What differs between the pages of two runs: a missing page, or the bytes of a page."""
    found = []
    for iteration, (pages_a, pages_b) in enumerate(zip(left.pages, right.pages)):
        for key in sorted(pages_a.keys() ^ pages_b.keys(), key=str):
            found.append(f"iteration {iteration}: only one run has the page {key}")
        for key, page in pages_a.items():
            if key not in pages_b or key[2] in skip_roles:
                continue
            different = (page != pages_b[key]).nonzero().flatten()
            if different.numel():
                request, layer, role, block = key
                found.append(
                    f"iteration {iteration}: request {request} layer {layer} {role.name} block "
                    f"{block}: {different.numel()} of {page.numel()} bytes differ, "
                    f"from byte {different[0].item()} to {different[-1].item()}"
                )
    return found


def _pages_differ(left: _Result, right: _Result, skip_roles=()) -> bool:
    return bool(_page_differences(left, right, skip_roles))


def _assert_outputs_agree(reference: tuple[_Result, _Result], other: _Result) -> None:
    """Every layer of ``other`` is as close to the reference as it may be."""
    noise = _differences_by_layer(*reference)
    difference = _differences_by_layer(reference[0], other)
    allowed = _allowed_differences(reference)
    print(f"the reference differs from itself by {noise} per layer")
    print(f"the other run differs from the reference by {difference}, may by {allowed}")
    for layer, (actual, limit) in enumerate(zip(difference, allowed)):
        assert actual <= limit, f"layer {layer} is {actual} from the reference, may be {limit}"


def _assert_same(reference: tuple[_Result, _Result], other: _Result) -> None:
    """The outputs of ``other`` are those of the reference, and so are its pages if the reference
    reproduces them."""
    first, second = reference
    _assert_outputs_agree(reference, other)
    pages_reproduce = not _pages_differ(first, second)
    print(f"pages of the reference reproduce: {pages_reproduce}")
    if pages_reproduce:
        differences = _page_differences(first, other)
        assert not differences, f"{len(differences)} pages differ: {differences[:12]}"


_VARIANTS = pytest.mark.parametrize(
    ("indexer", "fp8_kv", "workload"),
    [
        (indexer, fp8_kv, workload)
        for workload in (_SELECTS_ALL, _SELECTS_SOME)
        for indexer, fp8_kv in (("fp8", True), ("fp4", True), ("fp8", False), ("fp4", False))
    ],
    ids=[
        f"{kv} kv, {indexer} indexer, indexer selects {selects}"
        for selects in ("all", "some")
        for kv, indexer in (("fp8", "fp8"), ("fp8", "fp4"), ("bf16", "fp8"), ("bf16", "fp4"))
    ],
)


@_VARIANTS
def test_the_staged_view_gives_the_outputs_and_pages_of_the_cache_manager(
    indexer: str, fp8_kv: bool, workload: _Workload
) -> None:
    reference = _reference(indexer, fp8_kv, workload)
    staged = _Flow(_model(indexer, fp8_kv), workload, make_streamer=_loopback).run()
    _assert_same(reference, staged)


@_VARIANTS
def test_what_the_staging_area_held_before_the_fetch_does_not_matter(
    indexer: str, fp8_kv: bool, workload: _Workload
) -> None:
    reference = _reference(indexer, fp8_kv, workload)
    filled = _Flow(_model(indexer, fp8_kv), workload, make_streamer=_nan_loopback).run()
    for step in filled.outputs:
        for output in step:
            assert not output.isnan().any()
    _assert_outputs_agree(reference, filled)


@_VARIANTS
def test_the_kernels_neither_read_nor_write_the_pages_of_the_cache_manager(
    indexer: str, fp8_kv: bool, workload: _Workload
) -> None:
    reference = _reference(indexer, fp8_kv, workload)
    flow = _Flow(_model(indexer, fp8_kv), workload, make_streamer=_PoisonPages)
    poisoned = flow.run()
    assert flow.streamer.poisoned_pages > 0
    _assert_same(reference, poisoned)


def test_the_layers_reproduce_themselves_when_the_indexer_selects_everything() -> None:
    noise = _differences_by_layer(*_reference("fp8", True, _SELECTS_ALL))
    assert noise == [0.0] * len(_RATIOS)


def _roles(kind: StagingKind) -> tuple:
    return attention_types(kind)


@pytest.mark.parametrize("kind", list(StagingKind), ids=lambda kind: kind.label)
def test_a_corruption_of_every_kind_of_storage_in_the_staging_area_is_seen(
    kind: StagingKind,
) -> None:
    reference = _reference("fp8", True, _SELECTS_SOME)
    corrupted = _Flow(
        _model("fp8", True),
        _SELECTS_SOME,
        make_streamer=lambda manager, view: _FlipAfterFetch(manager, view, _flip_kind(kind)),
    ).run()
    # The corruption shows in the outputs, or, for what only feeds the kernels that fill the other
    # storages, in the pages of the other storages.
    seen = _differs(reference, corrupted)
    if not seen and not _pages_differ(*reference):
        seen = _pages_differ(reference[0], corrupted, skip_roles=_roles(kind))
    assert seen


def test_a_flipped_bit_in_the_last_cached_token_of_a_sliding_window_page_changes_the_output() -> (
    None
):
    reference = _reference("fp8", True, _SELECTS_ALL)
    corrupted = _Flow(
        _model("fp8", True),
        _SELECTS_ALL,
        make_streamer=lambda manager, view: _FlipAfterFetch(manager, view, _flip_one_bit(layer=0)),
    ).run()
    assert _differs(reference, corrupted)
