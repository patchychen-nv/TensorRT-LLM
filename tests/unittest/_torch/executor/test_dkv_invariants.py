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
        checker.check_many(3, {"C0": [("pool", 32)], "C1": [(8, 1), (2, 0)]})

    group.run(check)
    assert [len(trace) for trace in group.traces] == [2, 2]


def test_disabled_checker_only_exchanges_enable_flag() -> None:
    group = LockstepTpGroup(2)

    def check(dist: LockstepDistributed) -> None:
        checker = DkvInvariantChecker(dist, enabled=False)
        checker.check(1, "C1", [(dist.tp_rank,)])

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


@pytest.mark.parametrize("mismatch", ["order", "value", "tag", "iteration", "dummy"])
def test_checker_reports_ordered_rank_payloads(mismatch: str) -> None:
    group = LockstepTpGroup(2)

    def check(dist: LockstepDistributed) -> None:
        checker = DkvInvariantChecker(dist, enabled=True)
        iteration = 5
        tag = "C1"
        values = [(3, 0), (9, 1), (-1, 0)]
        if dist.tp_rank:
            if mismatch == "order":
                values.reverse()
            elif mismatch == "value":
                values[0] = (3, 1)
            elif mismatch == "tag":
                tag = "C3"
            elif mismatch == "iteration":
                iteration += 1
            else:
                values.pop()
        with pytest.raises(RuntimeError, match="DKV invariant violation") as error:
            checker.check(iteration, tag, values)
        assert "rank0:" in str(error.value)
        assert "rank1:" in str(error.value)
        assert "(3, 0)" in str(error.value)

    group.run(check)
    assert [len(trace) for trace in group.traces] == [3, 3]
