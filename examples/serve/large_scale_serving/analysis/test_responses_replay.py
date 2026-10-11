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
"""The replay invariant, as a test.

Runs `responses_replay` over the recorded frames and asserts the property from
`SPEC_responses_streaming_fix.md`: an `output_text.done` payload is exactly the
concatenation of the `output_text.delta` payloads already streamed for that
item.

Against the pre-fix source these tests fail; that is what makes them worth
having. `RESPONSES_REPLAY_SOURCE` points the module under test at a specific
`responses_utils.py` -- leave it unset to test the working tree.

    PY=/code/tensorrt_llm/.venv-3.12/bin/python3
    # the tree as it stands (the AFTER gate)
    $PY -m pytest examples/serve/large_scale_serving/analysis/test_responses_replay.py -v
    # a pinned snapshot, e.g. to confirm the tests do fail before the fix
    RESPONSES_REPLAY_SOURCE=examples/serve/large_scale_serving/analysis/data/responses_utils_BEFORE.py \
        $PY -m pytest examples/serve/large_scale_serving/analysis/test_responses_replay.py -v

This lives beside the harness rather than in `tests/unittest/` because it needs
the recorded corpus in `data/`, and because it loads `responses_utils` through
the harness's stub loader so it runs in a checkout with no compiled bindings.
The unit tests the spec asks for belong in
`tests/unittest/llmapi/apps/test_responses_streaming_tool_calls.py`.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import responses_replay as rr  # noqa: E402

DATA = Path(__file__).resolve().parent / "data"
ROOT = Path(__file__).resolve().parents[4]
FRAMES = DATA / "responses_replay_frames.json"
MULTICALL = DATA / "responses_replay_multicall.jsonl"

# The reference record: raw-2779107.jsonl line 5, 8 frames, 2 calls.
REFERENCE_TEXT_LEN = 116
REFERENCE_CALLS = 2


@pytest.fixture(scope="session")
def responses_utils():
    module, provenance = rr.load_responses_utils(
        ROOT, os.environ.get("RESPONSES_REPLAY_SOURCE") or None
    )
    rr.verify_provenance(provenance)
    return module


@pytest.fixture(scope="session")
def reference_record():
    return rr.load_records([str(FRAMES)])[0]


@pytest.fixture(scope="session")
def multicall_records():
    if not MULTICALL.exists():
        pytest.skip(f"{MULTICALL} not present")
    return rr.load_records([str(MULTICALL)], min_calls=2)


def _violations(responses_utils, record, mode="stream"):
    if mode == "stream":
        return rr.check_invariant(rr.replay_stream(responses_utils, record))
    out = []
    for replay in rr.replay_truncations(responses_utils, record):
        out.extend(rr.check_invariant(replay))
    return out


def test_harness_loads_real_source(responses_utils):
    """Guard against a clean report produced by a stub.

    Every assertion below passes vacuously if `responses_utils` is a stub
    module, so check first that the real one is executing.
    """
    assert responses_utils.__file__.endswith(".py")
    assert hasattr(responses_utils, "_generate_streaming_event")
    assert hasattr(responses_utils, "_close_open_item")


def test_reference_record_emits_something(responses_utils, reference_record):
    """An empty stream satisfies the invariant and proves nothing."""
    replay = rr.replay_stream(responses_utils, reference_record)
    assert replay.error is None, replay.error
    assert replay.text_deltas(), "no output_text.delta events were emitted"
    assert replay.text_dones(), "no output_text.done events were emitted"


def test_no_markup_in_any_done_payload(responses_utils, reference_record):
    """Spec Verification, point 1: no prefix may produce markup in a done."""
    replay = rr.replay_stream(responses_utils, reference_record)
    leaking = [
        done
        for done in replay.text_dones()
        if any(token in done.payload for token in rr.MARKUP_TOKENS)
    ]
    assert not leaking, (
        f"{len(leaking)} done payload(s) carry tool-call markup, at prefixes "
        f"{[d.prefix for d in leaking]}; first is "
        f"{leaking[0].payload[:200]!r}"
    )


def test_reference_record_final_state(responses_utils, reference_record):
    """Spec Verification, point 1: both calls, and exactly 116 chars of text."""
    replay = rr.replay_stream(responses_utils, reference_record)
    dones = replay.text_dones()
    assert dones, "no output_text.done event was emitted"
    assert len(dones[-1].payload) == REFERENCE_TEXT_LEN
    assert len(replay.tool_calls()) == REFERENCE_CALLS


def test_invariant_on_reference_record(responses_utils, reference_record):
    """Spec Verification, point 2, on the record the defect was found in."""
    violations = _violations(responses_utils, reference_record)
    assert not violations, _describe(violations)


def test_invariant_on_reference_record_truncated(responses_utils, reference_record):
    """Edge case 6: the same record, with the stream ending at every frame."""
    violations = _violations(responses_utils, reference_record, mode="truncate")
    assert not violations, _describe(violations)


def test_invariant_on_all_multicall_records(responses_utils, multicall_records):
    """Spec Verification, point 3, restricted to the records that trip it."""
    violations: list[rr.Violation] = []
    for record in multicall_records:
        violations.extend(_violations(responses_utils, record))
    assert not violations, _describe(violations)


def test_tool_calls_still_recovered(responses_utils, multicall_records):
    """A fix that dropped the calls would satisfy the invariant too."""
    missing = []
    for record in multicall_records:
        replay = rr.replay_stream(responses_utils, record)
        if not replay.tool_calls():
            missing.append(record.record_id)
    assert not missing, f"no tool calls surfaced for records {missing}"


def _describe(violations: list[rr.Violation], limit: int = 3) -> str:
    lines = [f"{len(violations)} invariant violation(s)"]
    for violation in violations[:limit]:
        lines.append(
            f"  record={violation.record_id} mode={violation.mode} "
            f"prefix={violation.prefix} "
            f"streamed={len(violation.streamed)} chars, "
            f"done={len(violation.done)} chars, "
            f"+{violation.extra_chars} never streamed "
            f"({violation.markup_chars} of them markup)"
        )
        lines.append("  " + violation.diff().replace("\n", "\n  "))
    return "\n".join(lines)
