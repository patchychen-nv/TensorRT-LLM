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
"""A silent Responses stream repeats its opening response.in_progress.

A tool call is sent only once it is complete. While a long one is generated
the client receives nothing, and clients drop a stream that stays silent for
long: Kernel Factory's side disconnects after about 120 s, cutting the call.
"""

import json

import pytest

from tensorrt_llm.serve import responses_utils
from tensorrt_llm.serve.openai_protocol import ResponsesRequest
from tensorrt_llm.serve.responses_utils import (
    RESPONSES_STREAM_KEEPALIVE_ENV,
    ResponsesStreamingProcessor,
    StreamKeepalive,
    responses_stream_keepalive_interval,
    stamp_sse_sequence_number,
)

pytestmark = pytest.mark.cpu_only


class FakeClock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def _initial_frames():
    request = ResponsesRequest(model="test-model", input="hi", stream=True)
    processor = ResponsesStreamingProcessor(
        request=request,
        sampling_params=request.to_sampling_params(),
        model_name="test-model",
        use_harmony=False,
    )
    return processor.get_initial_responses()


def _payload(frame):
    return json.loads(frame.split("data: ", 1)[1].split("\n", 1)[0])


class TestInterval:
    @pytest.mark.parametrize(
        "raw, expected",
        [(None, 15.0), ("", 15.0), ("2.5", 2.5), ("0", 0.0), ("-3", 0.0)],
    )
    def test_read_from_the_environment(self, monkeypatch, raw, expected):
        if raw is None:
            monkeypatch.delenv(RESPONSES_STREAM_KEEPALIVE_ENV, raising=False)
        else:
            monkeypatch.setenv(RESPONSES_STREAM_KEEPALIVE_ENV, raw)
        assert responses_stream_keepalive_interval() == expected

    def test_a_value_that_is_not_a_number_falls_back_with_a_warning(self, monkeypatch):
        warnings = []
        monkeypatch.setattr(
            responses_utils.logger, "warning_once", lambda msg, key=None: warnings.append(msg)
        )
        monkeypatch.setenv(RESPONSES_STREAM_KEEPALIVE_ENV, "soon")
        assert responses_stream_keepalive_interval() == 15.0
        assert len(warnings) == 1 and "'soon'" in warnings[0]


class TestStreamKeepalive:
    def test_the_repeated_frame_is_the_opening_in_progress(self):
        frames = _initial_frames()
        keepalive = StreamKeepalive(frames, 15.0)
        assert keepalive.frame == frames[1]
        assert keepalive.frame.startswith("event: response.in_progress\n")

    def test_nothing_before_the_interval_then_the_frame(self):
        clock = FakeClock()
        keepalive = StreamKeepalive(_initial_frames(), 15.0, clock)
        clock.now += 14.9
        assert keepalive.step(False) is None
        clock.now += 0.2
        assert keepalive.step(False) == keepalive.frame

    def test_each_silent_interval_gets_one_keepalive(self):
        clock = FakeClock()
        keepalive = StreamKeepalive(_initial_frames(), 15.0, clock)
        sent = []
        for _ in range(60):  # 60 silent steps, one per second
            clock.now += 1.0
            sent.append(keepalive.step(False) is not None)
        assert sum(sent) == 4  # at 15, 30, 45 and 60 s

    def test_frames_reset_the_silence(self):
        clock = FakeClock()
        keepalive = StreamKeepalive(_initial_frames(), 15.0, clock)
        for _ in range(10):
            clock.now += 10.0
            assert keepalive.step(True) is None  # a frame every 10 s: never silent
        clock.now += 10.0
        assert keepalive.step(False) is None
        clock.now += 5.0
        assert keepalive.step(False) is not None

    @pytest.mark.parametrize("interval", [0.0, -1.0])
    def test_turned_off(self, interval):
        clock = FakeClock()
        keepalive = StreamKeepalive(_initial_frames(), interval, clock)
        clock.now += 3600.0
        assert keepalive.step(False) is None

    def test_a_stream_that_opens_without_in_progress_is_left_alone(self):
        clock = FakeClock()
        keepalive = StreamKeepalive(["event: response.created\ndata: {}\n\n"], 15.0, clock)
        clock.now += 60.0
        assert keepalive.step(False) is None


def test_the_egress_stream_stays_numbered_and_says_nothing_new():
    """The egress loop as the server runs it: stamp, step, stamp the keepalive."""
    clock = FakeClock()
    initial = _initial_frames()
    keepalive = StreamKeepalive(initial, 15.0, clock)
    # A reasoning delta, then 40 s of a tool call being generated (no frames),
    # then the call's events.
    steps = [['event: response.output_text.delta\ndata: {"sequence_number": 0}\n\n']]
    steps += [[] for _ in range(40)]
    steps += [['event: response.output_item.done\ndata: {"sequence_number": 0}\n\n']]
    out, sequence_number = [], 0
    for frame in initial:
        out.append(stamp_sse_sequence_number(frame, sequence_number))
        sequence_number += 1
    for frames in steps:
        clock.now += 1.0
        for frame in frames:
            out.append(stamp_sse_sequence_number(frame, sequence_number))
            sequence_number += 1
        keepalive_frame = keepalive.step(bool(frames))
        if keepalive_frame is not None:
            out.append(stamp_sse_sequence_number(keepalive_frame, sequence_number))
            sequence_number += 1

    assert [_payload(frame)["sequence_number"] for frame in out] == list(range(len(out)))
    repeats = [frame for frame in out[2:] if frame.startswith("event: response.in_progress\n")]
    assert len(repeats) == 2  # at 15 s and 30 s of the 40 s silence
    opening = _payload(initial[1])
    for frame in repeats:
        payload = _payload(frame)
        assert payload.pop("sequence_number") != opening["sequence_number"]
        assert payload == {k: v for k, v in opening.items() if k != "sequence_number"}
