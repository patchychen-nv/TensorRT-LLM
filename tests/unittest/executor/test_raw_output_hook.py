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
"""Unit tests for the raw model-output dump.

The dump exists to be read next to a request trace while debugging a parser, so
what these pin down is mostly what a reader of the two files together relies on:
that the frames arrive separated, that the id joining them is the one the client
saw, and above all that the hook changed nothing on its way past.
"""

import glob
import json
import os

import pytest

from tensorrt_llm.executor.postprocessor_hook import (
    PostProcessorHookAction,
    apply_post_processor_hook,
    load_post_processor_hook,
)
from tensorrt_llm.executor.raw_output_hook import (
    RAW_OUTPUT_DIR_ENV,
    RawOutputDump,
    raw_output_dir_from_env,
)

pytestmark = pytest.mark.cpu_only

HOOK_PATH = "tensorrt_llm.executor.raw_output_hook.RawOutputDump"


class FakeOutput:
    """The parts of ``CompletionOutput`` the hook and its applier touch."""

    def __init__(self, index=0):
        self.index = index
        self.text = ""
        self._last_text_len = 0
        self.token_ids = []
        self._last_token_ids_len = 0
        self.logprobs = None
        self.finish_reason = None
        self.stop_reason = None

    @property
    def text_diff(self):
        return self.text[self._last_text_len :]

    @property
    def token_ids_diff(self):
        return self.token_ids[self._last_token_ids_len :]

    def append(self, text):
        """Advance the detok watermark the way the executor does."""
        self._last_text_len = len(self.text)
        self.text += text


class FakeResult:
    def __init__(self, request_id=7, outputs=None):
        self.id = request_id
        self.outputs = outputs if outputs is not None else [FakeOutput()]
        self._done = False
        self._aborted = False


def drive(hook, result, deltas, streaming=True):
    """Feed ``deltas`` through the hook one chunk at a time, as the engine does.

    Returns the text each output ended up with, which is the assertion that
    matters most here: an observer that alters what is served is worse than no
    observer at all.
    """
    for i, delta in enumerate(deltas):
        for output in result.outputs:
            output.append(delta)
        result._done = i == len(deltas) - 1
        apply_post_processor_hook(hook, result, streaming=streaming)
    return [output.text for output in result.outputs]


def records_in(directory):
    lines = []
    for path in sorted(glob.glob(os.path.join(directory, "*", "raw-*.jsonl"))):
        with open(path, encoding="utf-8") as handle:
            lines += [json.loads(line) for line in handle if line.strip()]
    return lines


@pytest.fixture
def dump_dir(tmp_path, monkeypatch):
    directory = tmp_path / "raw"
    monkeypatch.setenv(RAW_OUTPUT_DIR_ENV, str(directory))
    return directory


def test_the_hook_does_not_change_what_is_served(dump_dir):
    """The whole licence to run this in production.

    Driven through the real applier rather than called directly, because it is
    the applier that rebuilds ``output.text`` from the verdict -- a hook that
    returned the wrong thing would show up here and nowhere else.
    """
    hook = RawOutputDump()
    result = FakeResult()
    deltas = ["<think>", "weighing ", "it</think>", "the answer"]

    (text,) = drive(hook, result, deltas)

    assert text == "".join(deltas)


def test_it_is_inert_without_the_directory(tmp_path, monkeypatch):
    """Unset means off, including for the passthrough.

    Worth its own test because the env var is read once in the constructor: a
    hook built before the variable is set must still serve traffic correctly,
    just silently.
    """
    monkeypatch.delenv(RAW_OUTPUT_DIR_ENV, raising=False)
    assert raw_output_dir_from_env() is None

    hook = RawOutputDump()
    result = FakeResult()
    (text,) = drive(hook, result, ["alpha", "beta"])

    assert text == "alphabeta"
    assert not list(tmp_path.rglob("*.jsonl"))
    # Off, not failing quietly. Without the early return the recording path
    # still runs and still writes nothing -- it throws on the missing directory
    # and the guard swallows it -- so "no files appeared" alone cannot tell the
    # two apart. Every deployment that leaves this disabled is the one paying
    # for the difference.
    assert hook._dropped == 0


def test_streaming_frames_are_kept_apart(dump_dir):
    """The reason this file exists rather than a single joined string.

    The split here is the one that breaks incremental tool parsing: the marker
    and the JSON that follows it arrive in different chunks. A record that
    joined them first could not tell that apart from the model emitting it
    whole.
    """
    hook = RawOutputDump()
    deltas = ['<tool_call>{"na', 'me": "f"}', "</tool_call>"]

    drive(hook, FakeResult(request_id=99), deltas)

    (record,) = records_in(dump_dir)
    assert record["frames"] == deltas, "the chunk boundaries were lost"
    assert record["text"] == "".join(deltas)
    assert record["streaming"] is True
    assert record["request_id"] == "99"


def test_empty_chunks_do_not_become_frames(dump_dir):
    """A chunk that added no text is not a seam.

    The terminating call routinely carries no new text, and a blank between two
    real frames reads as a boundary that was never there.
    """
    hook = RawOutputDump()

    drive(hook, FakeResult(), ["one", "", "two", ""])

    (record,) = records_in(dump_dir)
    assert record["frames"] == ["one", "two"]


def test_each_output_of_one_request_keeps_its_own_frames(dump_dir):
    """``is_final`` is request-level, so n>1 finishes every output at once.

    Driven over two chunks on purpose. With one chunk a hook that keyed on the
    request alone still produces one record per output -- the final call pops
    the shared entry and the next output starts a fresh one -- so a single-chunk
    version of this test passes against the bug it is named for. The damage is
    interleaving, and interleaving needs a second chunk to show up.
    """
    hook = RawOutputDump()
    outputs = [FakeOutput(0), FakeOutput(1)]
    result = FakeResult(request_id=5, outputs=outputs)

    for chunk_index, (first, second) in enumerate([("a1", "b1"), ("a2", "b2")]):
        outputs[0].append(first)
        outputs[1].append(second)
        result._done = chunk_index == 1
        apply_post_processor_hook(hook, result, streaming=True)

    records = sorted(records_in(dump_dir), key=lambda r: r["output_index"])
    assert [r["output_index"] for r in records] == [0, 1]
    assert all(r["request_id"] == "5" for r in records)
    assert records[0]["frames"] == ["a1", "a2"], "output 1's frames leaked in"
    assert records[1]["frames"] == ["b1", "b2"], "output 0's frames leaked in"


def test_a_non_streaming_request_records_its_one_chunk(dump_dir):
    hook = RawOutputDump()

    drive(hook, FakeResult(), ["the whole answer"], streaming=False)

    (record,) = records_in(dump_dir)
    assert record["streaming"] is False
    assert record["frames"] == ["the whole answer"]
    assert record["text"] == "the whole answer"


def test_nothing_is_written_until_the_output_finishes(dump_dir):
    """One line per sequence, not one per chunk.

    The bound that keeps this affordable on a fleet: a 2,000-token answer costs
    one record, not two thousand.
    """
    hook = RawOutputDump()
    result = FakeResult()

    for delta in ["still ", "going"]:
        result.outputs[0].append(delta)
        apply_post_processor_hook(hook, result, streaming=True)

    assert records_in(dump_dir) == []

    result._done = True
    apply_post_processor_hook(hook, result, streaming=True)
    assert len(records_in(dump_dir)) == 1


def test_an_unwritable_directory_costs_the_record_not_the_response(tmp_path, monkeypatch):
    """The one place this deliberately parts company with the hook contract.

    The seam fails closed so a hook that vets output cannot have its failures
    serve un-vetted text. This one vets nothing, so a directory it cannot write
    must cost the record and not the response.
    """
    blocked = tmp_path / "wall"
    blocked.write_text("not a directory")
    monkeypatch.setenv(RAW_OUTPUT_DIR_ENV, str(blocked / "raw"))

    hook = RawOutputDump()
    result = FakeResult()

    (text,) = drive(hook, result, ["served ", "anyway"])

    assert text == "served anyway"
    assert hook._dropped > 0, "the write failed silently without being counted"


def test_an_unexpected_failure_also_costs_only_the_record(dump_dir, monkeypatch):
    """The outer guard, which the unwritable-directory case does not reach.

    That one is caught where it happens, by the ``OSError`` handler around the
    write, so it says nothing about what a fault anywhere else in the recording
    path would do -- and it is the faults nobody predicted that decide whether
    this is safe to leave switched on. Injected rather than provoked for the
    same reason: a failure mode that can be named has usually been handled.
    """
    hook = RawOutputDump()

    def explode(*_args, **_kwargs):
        raise RuntimeError("something nobody thought of")

    monkeypatch.setattr(hook, "_write", explode)
    result = FakeResult()

    (text,) = drive(hook, result, ["served ", "anyway"])

    assert text == "served anyway"
    assert hook._dropped > 0


def test_an_aborted_output_is_still_recorded(dump_dir):
    """Truncated output is evidence, not a failure to discard.

    A stream cut mid-tool-call is one of the shapes worth looking at, and the
    flag is what separates it from a model that simply stopped.
    """
    hook = RawOutputDump()
    result = FakeResult()
    result.outputs[0].append("<tool_call>{")
    result._aborted = True
    result._done = True

    apply_post_processor_hook(hook, result, streaming=True)

    (record,) = records_in(dump_dir)
    assert record["aborted"] is True
    assert record["text"] == "<tool_call>{"


def test_it_loads_through_the_documented_import_path(dump_dir):
    """--post_processor_hook takes this exact string; a rename would break it."""
    hook = load_post_processor_hook(HOOK_PATH)

    assert isinstance(hook, RawOutputDump)
    verdict = hook(
        type(
            "Chunk",
            (),
            {
                "request_id": 1,
                "output_index": 0,
                "text_diff": "x",
                "text": "x",
                "token_ids_diff": [],
                "is_final": True,
                "aborted": False,
                "streaming": False,
            },
        )()
    )
    assert verdict.action is PostProcessorHookAction.EMIT
    assert verdict.text == "x"


def test_records_carry_the_pid_so_workers_do_not_splice(dump_dir):
    """One hook instance per process, and they share the hour bucket."""
    RawOutputDump()
    drive(RawOutputDump(), FakeResult(), ["x"])

    paths = glob.glob(os.path.join(str(dump_dir), "*", "raw-*.jsonl"))
    assert len(paths) == 1
    assert os.path.basename(paths[0]) == f"raw-{os.getpid()}.jsonl"


def test_a_request_ended_by_an_error_keeps_its_partial_output_and_releases_it(dump_dir):
    """An executor error ends a request with no final chunk.

    What it generated before the error is written, marked partial, and its state
    is released instead of living as long as the worker.
    """
    hook = RawOutputDump()
    result = FakeResult(request_id=7)
    for delta in ["<tool_call>", '{"a"']:
        result.outputs[0].append(delta)
        apply_post_processor_hook(hook, result, streaming=True)
    assert records_in(dump_dir) == []

    hook.on_request_end(7, "error")

    assert hook._pending == {}
    (record,) = records_in(dump_dir)
    assert record["partial"] is True
    assert record["end_reason"] == "error"
    assert record["aborted"] is True
    assert record["frames"] == ["<tool_call>", '{"a"']
    assert record["text"] == '<tool_call>{"a"'


def test_ending_one_request_leaves_the_others_alone(dump_dir):
    hook = RawOutputDump()
    first, second = FakeResult(request_id=7), FakeResult(request_id=8)
    for result in (first, second):
        result.outputs[0].append("partial")
        apply_post_processor_hook(hook, result, streaming=True)

    hook.on_request_end(7, "error")
    assert list(hook._pending) == [(8, 0)]

    second.outputs[0].append(" and done")
    second._done = True
    apply_post_processor_hook(hook, second, streaming=True)
    by_id = {record["request_id"]: record for record in records_in(dump_dir)}
    assert by_id["7"]["partial"] is True
    assert "partial" not in by_id["8"]
    assert by_id["8"]["text"] == "partial and done"


def test_a_request_that_failed_before_any_text_writes_nothing(dump_dir):
    hook = RawOutputDump()
    result = FakeResult(request_id=7)
    apply_post_processor_hook(hook, result, streaming=True)
    assert (7, 0) in hook._pending

    hook.on_request_end(7, "error")

    assert hook._pending == {}
    assert records_in(dump_dir) == []


def test_failed_requests_do_not_accumulate(dump_dir):
    """A thousand requests that each fail after some output leave nothing behind."""
    hook = RawOutputDump()
    for request_id in range(1000):
        result = FakeResult(request_id=request_id)
        result.outputs[0].append("x" * 1024)
        apply_post_processor_hook(hook, result, streaming=True)
        hook.on_request_end(request_id, "error")

    assert hook._pending == {}
    assert len(records_in(dump_dir)) == 1000


def test_ending_an_unknown_request_is_harmless(dump_dir):
    hook = RawOutputDump()
    hook.on_request_end(12345, "error")
    assert hook._pending == {}
    assert records_in(dump_dir) == []
