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
"""Layer ownership of the layer-split layout: the table, and the rule that every rank covers
every KV life cycle of the model."""

import types

import pytest

from tensorrt_llm._torch.attention.backends.sparse.deepseek_v4.cache_manager import (
    DeepseekV4CacheManager,
)
from tensorrt_llm._torch.pyexecutor.dkv import (
    compute_ownership,
    owned_layers,
    ownership_fingerprint,
    validate_ownership,
)
from tensorrt_llm._torch.pyexecutor.dkv_types import LifecycleKey

pytestmark = pytest.mark.cpu_only

_SWA_ONLY, _CSA, _HCA = 1, 4, 128


def _alternating(num_layers: int) -> list[int]:
    """Two layers without compression, then CSA and HCA layers alternating, as DeepSeek-V4-Pro does
    (its 43 layers hold 2 SWA-only, 21 CSA and 20 HCA layers)."""
    return [_SWA_ONLY, _SWA_ONLY] + [
        _CSA if layer % 2 == 0 else _HCA for layer in range(num_layers - 2)
    ]


_V4_PRO = _alternating(43)
_V4_TEST_CONFIG = _alternating(61)


def _key(window: int, *, sparse: bool = False) -> LifecycleKey:
    """The key of a life cycle of a window of ``window`` tokens (0: the whole history)."""
    return LifecycleKey(False, window, 0, sparse)


def _life_cycle_keys(ratios: list[int], *, offload: bool = False) -> list[frozenset]:
    """The keys of the real V4 cache manager method, on a stand-in that has only what it reads."""
    manager = types.SimpleNamespace(
        num_layers=len(ratios),
        _compress_ratios=list(ratios),
        _swa_window_size=128,
        _max_draft_len=0,
        _enable_kv_cache_offload=offload,
    )
    manager._get_window_size = types.MethodType(DeepseekV4CacheManager._get_window_size, manager)
    return DeepseekV4CacheManager.get_layer_life_cycle_keys(manager)


@pytest.mark.parametrize(
    ("num_layers", "group_size"),
    [(43, 1), (43, 2), (43, 4), (43, 8), (43, 16), (43, 21), (43, 22), (61, 4), (61, 8), (8, 8)],
)
def test_layers_split_into_contiguous_ranges_that_differ_by_at_most_one(
    num_layers: int, group_size: int
) -> None:
    owners = compute_ownership(num_layers, group_size)
    assert len(owners) == num_layers
    assert list(owners) == sorted(owners)
    counts = [len(owned_layers(owners, rank)) for rank in range(group_size)]
    assert sum(counts) == num_layers
    assert max(counts) - min(counts) <= 1
    # The remainder goes to the lowest ranks.
    assert counts == sorted(counts, reverse=True)
    assert counts[0] == -(-num_layers // group_size)


def test_the_table_of_four_ranks_over_forty_three_layers_is_the_documented_one() -> None:
    owners = compute_ownership(43, 4)
    assert [len(owned_layers(owners, rank)) for rank in range(4)] == [11, 11, 11, 10]
    assert owned_layers(owners, 0) == tuple(range(0, 11))
    assert owned_layers(owners, 3) == tuple(range(33, 43))


@pytest.mark.parametrize(("num_layers", "group_size"), [(0, 4), (4, 0), (-1, 2), (4, -2)])
def test_a_table_needs_layers_and_ranks(num_layers: int, group_size: int) -> None:
    with pytest.raises(ValueError, match="must be positive"):
        compute_ownership(num_layers, group_size)


def test_ranks_without_layers_are_left_in_the_table() -> None:
    owners = compute_ownership(2, 4)
    assert owners == (0, 1)
    assert owned_layers(owners, 3) == ()


def test_the_fingerprint_follows_the_table() -> None:
    assert ownership_fingerprint(compute_ownership(43, 4)) == ownership_fingerprint(
        list(compute_ownership(43, 4))
    )
    assert ownership_fingerprint(compute_ownership(43, 4)) != ownership_fingerprint(
        compute_ownership(43, 8)
    )
    assert ownership_fingerprint(compute_ownership(43, 4)) != ownership_fingerprint(
        compute_ownership(44, 4)
    )


def test_life_cycle_keys_follow_the_layer_types() -> None:
    swa, state_csa, history = _key(128), _key(8), _key(0)
    keys = _life_cycle_keys([_SWA_ONLY, _CSA, _HCA])
    assert keys[0] == {swa}
    assert keys[1] == {swa, state_csa, history}
    # The state window of an HCA layer is the same 128 tokens as the SWA window.
    assert keys[2] == {swa, history}


def test_offloaded_csa_history_is_a_sparse_life_cycle_of_its_own() -> None:
    swa, state_csa = _key(128), _key(8)
    keys = _life_cycle_keys([_CSA, _HCA], offload=True)
    assert keys[0] == {swa, state_csa, _key(0, sparse=True), _key(0)}
    assert keys[1] == {swa, _key(0)}


def _valid_by_construction(ratios: list[int], group_size: int) -> bool:
    """The rule restated on the layer types: only a CSA layer has the 8-token state life cycle, and
    every layer that is not SWA-only has the history one, so each rank needs a CSA layer."""
    owners = compute_ownership(len(ratios), group_size)
    return all(
        any(ratios[layer] == _CSA for layer in owned_layers(owners, rank))
        for rank in range(group_size)
    )


@pytest.mark.parametrize("ratios", [_V4_PRO, _V4_TEST_CONFIG], ids=["43 layers", "61 layers"])
@pytest.mark.parametrize("group_size", [1, 2, 4, 8, 16, 21, 22, 30, 31])
def test_every_rank_must_be_able_to_hold_every_life_cycle(
    ratios: list[int], group_size: int
) -> None:
    owners = compute_ownership(len(ratios), group_size)
    keys = _life_cycle_keys(ratios)
    if _valid_by_construction(ratios, group_size):
        validate_ownership(owners, keys, group_size)
    else:
        with pytest.raises(ValueError, match="not supported with dkv_config layer_split yet"):
            validate_ownership(owners, keys, group_size)


def test_the_documented_limits_of_deepseek_v4_pro() -> None:
    keys = _life_cycle_keys(_V4_PRO)
    for group_size in (2, 4, 8, 21):
        validate_ownership(compute_ownership(43, group_size), keys, group_size)
    with pytest.raises(ValueError, match=r"rank 0 would own layers \[0, 1\]"):
        validate_ownership(compute_ownership(43, 22), keys, 22)


def test_the_error_names_the_missing_life_cycles() -> None:
    keys = _life_cycle_keys(_V4_PRO)
    with pytest.raises(ValueError) as caught:
        validate_ownership(compute_ownership(43, 22), keys, 22)
    message = str(caught.value)
    assert "window_size=0" in message and "window_size=8" in message
    assert "window_size=128" not in message
    assert "smaller attention-DP group" in message


def test_a_rank_without_layers_is_rejected() -> None:
    keys = _life_cycle_keys([_CSA] * 3)
    with pytest.raises(ValueError, match=r"rank 3 would own layers \[\]"):
        validate_ownership(compute_ownership(3, 4), keys, 4)


def test_a_model_without_compression_is_covered_by_any_split() -> None:
    keys = _life_cycle_keys([_SWA_ONLY] * 8)
    for group_size in (1, 2, 8):
        validate_ownership(compute_ownership(8, group_size), keys, group_size)


@pytest.mark.parametrize(
    ("owners", "message"),
    [
        ((0, 1, 1), "has 3 layers, the model 4"),
        ((0, 0, 1, 2), r"Ranks \[2\] own layers but the group has 2 ranks"),
        ((0, 0, 1, -1), r"Ranks \[-1\] own layers"),
    ],
)
def test_a_table_that_does_not_fit_the_model_or_the_group_is_rejected(
    owners: tuple[int, ...], message: str
) -> None:
    keys = _life_cycle_keys([_CSA] * 4)
    with pytest.raises(ValueError, match=message):
        validate_ownership(owners, keys, 2)
