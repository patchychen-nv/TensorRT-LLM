# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tensorrt_llm.disaggregated_params import DisaggregatedParams, DisaggScheduleStyle
from tensorrt_llm.llmapi._dkv import _validate_dkv_request
from tensorrt_llm.llmapi.llm import BaseLLM, PreprocessedInputs
from tensorrt_llm.sampling_params import SamplingParams

pytestmark = pytest.mark.cpu_only


@pytest.mark.parametrize(
    "overrides, feature",
    [
        ({"max_tokens": None}, "max_tokens"),
        ({"max_tokens": 0}, "max_tokens"),
        ({"max_tokens": 2}, "max_tokens"),
        ({"is_generation_only": True}, "generation-only"),
        ({"n": 2}, "n > 1"),
        ({"best_of": 2}, "best_of > 1"),
        ({"has_multimodal_input": True}, "multimodal"),
        ({"is_generation_first": True}, "GENERATION_FIRST"),
    ],
)
def test_dkv_request_rejects_unsupported_input(overrides, feature):
    kwargs = {"max_tokens": 1, **overrides}
    with pytest.raises(ValueError, match=feature + ".*not supported with dkv_config yet"):
        _validate_dkv_request(**kwargs)


@pytest.mark.parametrize("best_of", [None, 1])
def test_dkv_request_accepts_single_prefill(best_of):
    _validate_dkv_request(max_tokens=1, best_of=best_of)


class _ArgumentsReached(Exception):
    """Stop the front door before executor submission."""


def _llm_stub(dkv_enabled=True):
    return SimpleNamespace(
        _encode_only=False,
        _executor=Mock(is_shutdown=Mock(return_value=False)),
        args=SimpleNamespace(
            dkv_config=object() if dkv_enabled else None, return_perf_metrics=False
        ),
        _prepare_sampling_params=lambda params: params,
        _configure_bart_decoder_prefix=Mock(),
        _preprocess=Mock(side_effect=AssertionError("Unexpected preprocessing")),
        _check_arguments=Mock(side_effect=_ArgumentsReached),
    )


def test_front_door_normalizes_context_only_before_dkv_validation():
    llm = _llm_stub()
    sampling = SamplingParams(max_tokens=128)
    with pytest.raises(_ArgumentsReached):
        BaseLLM.generate_async(
            llm,
            PreprocessedInputs(prompt_token_ids=[1, 2]),
            sampling_params=sampling,
            disaggregated_params=DisaggregatedParams(request_type="context_only"),
        )
    assert sampling.max_tokens == 1
    llm._executor.generate_async.assert_not_called()


def test_front_door_rejects_aggregate_decode_before_submission():
    llm = _llm_stub()
    with pytest.raises(ValueError, match="max_tokens.*not supported with dkv_config yet"):
        BaseLLM.generate_async(llm, [1, 2], sampling_params=SamplingParams(max_tokens=2))
    llm._preprocess.assert_not_called()
    llm._executor.generate_async.assert_not_called()


@pytest.mark.parametrize(
    "inputs, disagg",
    [
        ({"prompt": "image", "multi_modal_data": {"image": [object()]}}, None),
        (PreprocessedInputs(prompt_token_ids=[1], multimodal_params=object()), None),
        (
            [1],
            DisaggregatedParams(request_type="context_only", multimodal_embedding_handles=[{}]),
        ),
        ([1], DisaggregatedParams(request_type="generation_only")),
        (
            [1],
            DisaggregatedParams(
                request_type="context_only", schedule_style=DisaggScheduleStyle.GENERATION_FIRST
            ),
        ),
    ],
)
def test_front_door_rejects_before_preprocessing_or_submission(inputs, disagg):
    llm = _llm_stub()
    with pytest.raises(ValueError, match="not supported with dkv_config yet"):
        BaseLLM.generate_async(
            llm, inputs, sampling_params=SamplingParams(max_tokens=1), disaggregated_params=disagg
        )
    llm._preprocess.assert_not_called()
    llm._executor.generate_async.assert_not_called()


@pytest.mark.parametrize("has_dkv_field", [False, True])
def test_non_dkv_front_door_preserves_generation_request(has_dkv_field):
    llm = _llm_stub(dkv_enabled=False)
    if not has_dkv_field:
        del llm.args.dkv_config
    sampling = SamplingParams(max_tokens=128)
    with pytest.raises(_ArgumentsReached):
        BaseLLM.generate_async(
            llm, PreprocessedInputs(prompt_token_ids=[1, 2]), sampling_params=sampling
        )
    assert sampling.max_tokens == 128
