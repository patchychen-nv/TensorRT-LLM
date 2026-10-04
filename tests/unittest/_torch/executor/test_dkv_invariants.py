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
"""DKV checker catches configuration, ordering and state divergence."""

import pytest
from dkv_test_utils import LockstepDistributed, LockstepTpGroup

from tensorrt_llm._torch.pyexecutor.dkv import DkvInvariantChecker

pytestmark = pytest.mark.cpu_only


def test_checker_batches_tags_in_one_collective() -> None:
    group = LockstepTpGroup(2)

    def check(dist: LockstepDistributed) -> None:
        checker = DkvInvariantChecker(dist, enabled=True)
        checker.check_many(
            3, {"startup configuration": [("pool", 32)], "global request order": [(8, 1), (2, 0)]}
        )

    group.run(check)
    assert [len(trace) for trace in group.traces] == [2, 2]


def test_disabled_checker_only_exchanges_enable_flag() -> None:
    group = LockstepTpGroup(2)

    def check(dist: LockstepDistributed) -> None:
        checker = DkvInvariantChecker(dist, enabled=False)
        checker.check(1, "global request order", [(dist.tp_rank,)])

    group.run(check)
    assert [len(trace) for trace in group.traces] == [1, 1]


def test_inconsistent_debug_flags_fail_during_construction() -> None:
    group = LockstepTpGroup(2)

    def check(dist: LockstepDistributed) -> None:
        with pytest.raises(RuntimeError, match="enable flags differ"):
            DkvInvariantChecker(dist, enabled=bool(dist.tp_rank))

    group.run(check)


@pytest.mark.parametrize("enabled", ["0", "1"])
def test_debug_environment_is_resolved_before_agreement(
    monkeypatch: pytest.MonkeyPatch, enabled: str
) -> None:
    monkeypatch.setenv("TRTLLM_DKV_DEBUG", enabled)
    assert (
        LockstepTpGroup(2).run(lambda dist: DkvInvariantChecker(dist).enabled)
        == [enabled == "1"] * 2
    )


_FIRST_DIFFERENCE = {
    "order": "tag global request order, record 0: (3, 0) != (-1, 0)",
    "value": "tag global request order, record 0: (3, 0) != (3, 1)",
    "tag": "tags ['global request order'] != ['scheduling decisions']",
    "iteration": "iteration 5 != 6",
    "dummy": "tag global request order, 3 records != 2 records",
}


@pytest.mark.parametrize("mismatch", list(_FIRST_DIFFERENCE))
def test_checker_reports_ordered_rank_payloads(mismatch: str) -> None:
    group = LockstepTpGroup(2)

    def check(dist: LockstepDistributed) -> None:
        checker = DkvInvariantChecker(dist, enabled=True)
        iteration = 5
        tag = "global request order"
        values = [(3, 0), (9, 1), (-1, 0)]
        if dist.tp_rank:
            if mismatch == "order":
                values.reverse()
            elif mismatch == "value":
                values[0] = (3, 1)
            elif mismatch == "tag":
                tag = "scheduling decisions"
            elif mismatch == "iteration":
                iteration += 1
            else:
                values.pop()
        with pytest.raises(RuntimeError, match="DKV invariant violation") as error:
            checker.check(iteration, tag, values)
        assert "rank0:" in str(error.value)
        assert "rank1:" in str(error.value)
        assert "(3, 0)" in str(error.value)
        assert f"rank1 differs from rank0: {_FIRST_DIFFERENCE[mismatch]}" in str(error.value)

    group.run(check)
    assert [len(trace) for trace in group.traces] == [3, 3]


def test_checker_report_bounds_each_rank_payload() -> None:
    group = LockstepTpGroup(2)

    def check(dist: LockstepDistributed) -> None:
        checker = DkvInvariantChecker(dist, enabled=True)
        records = [(index, dist.tp_rank) for index in range(5000)]
        with pytest.raises(RuntimeError, match="DKV invariant violation") as error:
            checker.check(1, "scheduling decisions", records)
        assert len(str(error.value)) < 12000
        assert "characters)" in str(error.value)
        assert "record 0: (0, 0) != (0, 1)" in str(error.value)

    group.run(check)


@pytest.mark.parametrize("enabled", [False, True])
def test_startup_settings_that_differ_fail_on_every_rank_in_one_collective(enabled: bool) -> None:
    group = LockstepTpGroup(2)

    def check(dist: LockstepDistributed) -> None:
        settings = {"dual_ledger": bool(dist.tp_rank), "metrics_all_ranks": False}
        with pytest.raises(RuntimeError, match="startup settings differ") as error:
            DkvInvariantChecker(dist, enabled=enabled, startup_settings=settings)
        assert "rank 0: {'dual_ledger': False" in str(error.value)
        assert "rank 1: {'dual_ledger': True" in str(error.value)

    group.run(check)
    assert [len(trace) for trace in group.traces] == [1, 1]


def test_equal_startup_settings_are_accepted_regardless_of_key_order() -> None:
    group = LockstepTpGroup(2)

    def check(dist: LockstepDistributed) -> bool:
        settings = {"a": 1, "b": 2} if dist.tp_rank else {"b": 2, "a": 1}
        return DkvInvariantChecker(dist, enabled=False, startup_settings=settings).enabled

    assert group.run(check) == [False, False]
    assert [len(trace) for trace in group.traces] == [1, 1]
