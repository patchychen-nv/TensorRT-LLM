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
"""Unit tests for the request trace writer."""

import asyncio
import contextlib
import gc
import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest

from tensorrt_llm.serve.conversation_id import resolve_request_conversation_id
from tensorrt_llm.serve.openai_protocol import ChatCompletionRequest, ResponsesRequest
from tensorrt_llm.serve.request_trace import (
    _WRITER_QUEUE_SIZE,
    REQUEST_TRACE_DIR_ENV,
    WRITER_SIDECAR_DIR,
    RequestTraceWriter,
    _join_frames,
    brief_validation_errors,
    find_tool_call_markup,
    is_internal_disagg_request,
    request_trace_dir_from_env,
    resolve_session_key,
    responses_outcome,
    sanitize_session_key,
)

pytestmark = pytest.mark.cpu_only


class FakeHeaders(dict):
    """Minimal stand-in for starlette's case-insensitive header mapping."""


class FakeState:
    """Stand-in for starlette's request.state (a plain attribute bag)."""


class FakeURL:
    def __init__(self, path):
        self.path = path


class FakeRequest:
    """Enough of starlette's Request for the trace hooks."""

    def __init__(
        self,
        body=None,
        headers=None,
        raw_body=b"",
        json_error=None,
        path="/v1/messages",
        arrival_time=None,
    ):
        self._body = body
        self._raw_body = raw_body
        self._json_error = json_error
        self.headers = FakeHeaders(headers or {})
        self.url = FakeURL(path)
        self.state = FakeState()
        if arrival_time is not None:
            # What ServerArrivalTimeMiddleware puts there before any handler.
            self.state.server_arrival_time = arrival_time

    async def json(self):
        if self._json_error is not None:
            raise self._json_error
        return self._body

    async def body(self):
        return self._raw_body


def read_lines(directory, session, kind):
    """Records of one kind for one session.

    The layout is ``<hour bucket>/<kind><writer suffix>.jsonl`` -- the session
    is a field on every record, not a directory, so it is filtered rather than
    looked up.
    """
    records = []
    for path in sorted(directory.glob(f"*/{kind}*.jsonl")):
        records += [json.loads(line) for line in path.read_text().splitlines() if line]
    return [record for record in records if record["session"] == session]


async def drain(writer):
    """Flush the queue by closing; the writer is single-use afterwards."""
    await writer.close()


async def _until(predicate, timeout=5.0):
    """Wait on the writer's own progress markers instead of sleeping blind."""
    deadline = asyncio.get_running_loop().time() + timeout
    while not predicate():
        assert asyncio.get_running_loop().time() < deadline, "condition never became true"
        await asyncio.sleep(0.01)


class TestSanitizeSessionKey:
    @pytest.mark.parametrize("value", [None, "", "   "])
    def test_missing_falls_back(self, value):
        assert sanitize_session_key(value) == "_no_session"

    @pytest.mark.parametrize("value", ["..", ".", "./../etc", ".hidden"])
    def test_traversal_and_hidden_rejected(self, value):
        # A leading dot is the only way a sanitized leaf can still escape or
        # hide, so it is rejected outright rather than patched up.
        assert sanitize_session_key(value) == "_no_session"

    def test_separators_replaced(self):
        assert sanitize_session_key("a/b\\c") == "a_b_c"
        assert "/" not in sanitize_session_key("../../root")

    def test_null_byte_replaced(self):
        assert sanitize_session_key("sess\x00id") == "sess_id"

    def test_length_bounded(self):
        assert len(sanitize_session_key("x" * 500)) == 128

    def test_ordinary_value_preserved(self):
        assert sanitize_session_key("sess_ab12-cd.3") == "sess_ab12-cd.3"


class TestResolveSessionKey:
    def test_header_wins(self):
        headers = {"x-claude-code-session-id": "sess_hdr"}
        body = {"client_metadata": {"session_id": "sess_body"}}
        assert resolve_session_key(headers, body) == "sess_hdr"

    def test_header_priority_order(self):
        # x-claude-code-session-id precedes x-session-id in the shared tuple.
        headers = {"x-session-id": "generic", "x-claude-code-session-id": "claude"}
        assert resolve_session_key(headers, None) == "claude"

    def test_body_fallback(self):
        body = {"client_metadata": {"session_id": "sess_body"}}
        assert resolve_session_key({}, body) == "sess_body"

    def test_body_field_order_matches_routing(self):
        # Same order as CONVERSATION_ID_BODY_FIELDS, so the trace key equals the routing key:
        # prompt_cache_key, then client_metadata.thread_id, then client_metadata.session_id.
        body = {
            "prompt_cache_key": "thread-1",
            "client_metadata": {"session_id": "run-1", "thread_id": "thread-2"},
        }
        assert resolve_session_key({}, body) == "thread-1"
        assert (
            resolve_session_key(
                {}, {"client_metadata": {"session_id": "run-1", "thread_id": "thread-2"}}
            )
            == "thread-2"
        )

    @pytest.mark.parametrize("body", [None, {}, "not-a-dict", {"client_metadata": 7}])
    def test_no_session(self, body):
        assert resolve_session_key({}, body) == "_no_session"

    def test_conversation_params_beat_a_header_as_they_do_for_the_router(self):
        """The router reads body ``conversation_params`` before any header.

        The trace used to start at the headers, so a client that sent both had
        its turns filed under one id while the router pinned them by another.
        """
        headers = {"x-session-id": "sess_hdr"}
        body = {
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "conversation_params": {"conversation_id": " conv_body "},
        }
        routed = resolve_request_conversation_id(ChatCompletionRequest(**body), headers)

        assert routed == "conv_body"
        assert resolve_session_key(headers, body) == sanitize_session_key(routed)

    @pytest.mark.parametrize(
        "headers, fields",
        [
            ({"x-session-id": "hdr"}, {"conversation_params": {"conversation_id": "params"}}),
            ({"x-session-id": "hdr"}, {"prompt_cache_key": "pck"}),
            ({}, {"conversation_params": {"conversation_id": "params"}, "prompt_cache_key": "pck"}),
            ({}, {"client_metadata": {"session_id": "run", "thread_id": "thread"}}),
            (
                {"x-claude-code-session-id": "claude"},
                {
                    "conversation_params": {"conversation_id": "params"},
                    "client_metadata": {"thread_id": "thread"},
                },
            ),
            ({"x-session-id": "  "}, {"prompt_cache_key": "pck"}),
            ({}, {}),
        ],
    )
    def test_agrees_with_the_router_on_every_source_combination(self, headers, fields):
        body = {"model": "m", "input": "hi", **fields}
        routed = resolve_request_conversation_id(ResponsesRequest(**body), headers)

        assert resolve_session_key(headers, body) == sanitize_session_key(routed)

    @pytest.mark.asyncio
    async def test_request_line_carries_the_routed_id(self, tmp_path):
        writer = RequestTraceWriter(str(tmp_path))
        await writer.start()
        handle = await writer.on_request(
            FakeRequest(
                body={"conversation_params": {"conversation_id": "conv_body"}},
                headers={"x-session-id": "sess_hdr"},
            )
        )
        await drain(writer)

        assert handle.session == "conv_body"
        (record,) = read_lines(tmp_path, "conv_body", "requests")
        assert record["trace_id"] == handle.trace_id


class TestBriefValidationErrors:
    def test_keeps_only_safe_keys(self):
        errors = [
            {
                "loc": ("body", "client_metadata"),
                "type": "extra_forbidden",
                "msg": "Extra inputs are not permitted",
                "input": {"the": "whole body"},
                "handle": {"error": ValueError("live object")},
            }
        ]
        assert brief_validation_errors(errors) == [
            {
                "loc": ["body", "client_metadata"],
                "type": "extra_forbidden",
                "msg": "Extra inputs are not permitted",
            }
        ]

    def test_result_is_serializable(self):
        errors = [
            {"loc": ("body",), "type": "x", "msg": "y", "handle": {"error": ValueError("live")}}
        ]
        json.dumps(brief_validation_errors(errors))

    @pytest.mark.parametrize("errors", [[], None, ["not-a-mapping"]])
    def test_tolerates_junk(self, errors):
        assert brief_validation_errors(errors) == []


class TestIsInternalDisaggRequest:
    """The predicate that keeps orchestrator hops out of a worker's trace."""

    @pytest.mark.parametrize("request_type", ["context_only", "generation_only"])
    def test_orchestrator_hops(self, request_type):
        body = {"messages": [], "disaggregated_params": {"request_type": request_type}}
        assert is_internal_disagg_request(body) is True

    def test_context_and_generation_is_client_traffic(self):
        """The third legal value rides on requests one server answers end to end.

        The gRPC frontend falls back to it when the proto leaves request_type
        unset, and EPD multimodal sets it on the prefill+decode half. A check
        for the presence of disaggregated_params rather than for these two
        values would drop both.
        """
        body = {"disaggregated_params": {"request_type": "context_and_generation"}}
        assert is_internal_disagg_request(body) is False

    @pytest.mark.parametrize(
        "body",
        [
            {"messages": []},
            {"disaggregated_params": None},
            {"disaggregated_params": {}},
            {"disaggregated_params": {"request_type": None}},
            {"disaggregated_params": {"ctx_request_id": 7}},
            {"disaggregated_params": "context_only"},
            None,
            "not a mapping",
            [],
        ],
    )
    def test_anything_else_reads_as_client_traffic(self, body):
        assert is_internal_disagg_request(body) is False

    def test_unknown_request_type_fails_open(self):
        """A hop type this predicate has not heard of is recorded, not dropped."""
        body = {"disaggregated_params": {"request_type": "encoder_only"}}
        assert is_internal_disagg_request(body) is False


class TestInternalDisaggRequestsAreNotTraced:
    """A worker behind the orchestrator writes nothing, on either hook."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("request_type", ["context_only", "generation_only"])
    async def test_on_request_records_nothing(self, tmp_path, request_type):
        writer = RequestTraceWriter(str(tmp_path))
        await writer.start()
        request = FakeRequest(
            body={
                "model": "m",
                "messages": [{"role": "user", "content": "hi"}],
                "prompt_token_ids": [1, 2, 3],
                "disaggregated_params": {"request_type": request_type},
            },
            headers={"x-session-id": "sess_a"},
            path="/v1/chat/completions",
        )
        handle = await writer.on_request(request)
        await drain(writer)

        assert handle is None
        # No handle was stamped, so a nested handler cannot mistake this for a
        # request someone else already owns.
        assert getattr(request.state, "request_trace_handle", None) is None
        assert read_lines(tmp_path, "sess_a", "requests") == []
        assert read_lines(tmp_path, "_no_session", "requests") == []

    @pytest.mark.asyncio
    async def test_response_hooks_are_inert_for_the_dropped_handle(self, tmp_path):
        writer = RequestTraceWriter(str(tmp_path))
        await writer.start()
        request = FakeRequest(
            body={"disaggregated_params": {"request_type": "generation_only"}},
            path="/v1/chat/completions",
        )
        handle = await writer.on_request(request)

        async def upstream():
            yield "data: {}\n\n"

        stream = writer.wrap_stream(upstream(), handle)
        assert [chunk async for chunk in stream] == ["data: {}\n\n"]
        writer.on_response(handle, payload={"id": "x"}, status="completed")
        await drain(writer)

        assert read_lines(tmp_path, "_no_session", "responses") == []

    @pytest.mark.asyncio
    async def test_on_rejected_records_nothing(self, tmp_path):
        """Deliberate: a worker creates no trace files at all, not even rare ones."""
        writer = RequestTraceWriter(str(tmp_path))
        await writer.start()
        await writer.on_rejected(
            FakeRequest(
                body={"disaggregated_params": {"request_type": "context_only"}},
                path="/v1/chat/completions",
            ),
            [{"loc": ("body", "model"), "type": "missing", "msg": "Field required"}],
        )
        await drain(writer)

        assert read_lines(tmp_path, "_no_session", "requests") == []

    @pytest.mark.asyncio
    async def test_the_proxys_own_client_request_is_still_traced(self, tmp_path):
        """The same writer on the orchestrator keeps recording: no such field."""
        writer = RequestTraceWriter(str(tmp_path))
        await writer.start()
        handle = await writer.on_request(
            FakeRequest(
                body={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
                headers={"x-session-id": "sess_a"},
            )
        )
        await drain(writer)

        assert handle is not None
        (record,) = read_lines(tmp_path, "sess_a", "requests")
        assert record["body"]["messages"][0]["content"] == "hi"


class TestDisabledWriter:
    @pytest.mark.asyncio
    async def test_all_hooks_are_inert(self):
        writer = RequestTraceWriter(None)
        await writer.start()
        assert not writer.enabled
        assert await writer.on_request(FakeRequest(body={})) is None
        await writer.on_rejected(FakeRequest(body={}), [])
        writer.on_response(None)
        await writer.close()

    @pytest.mark.asyncio
    async def test_wrap_stream_returns_original(self):
        writer = RequestTraceWriter(None)
        await writer.start()

        async def source():
            yield "a"

        stream = source()
        assert writer.wrap_stream(stream, None) is stream
        await stream.aclose()

    def test_env_helper(self, monkeypatch):
        monkeypatch.delenv(REQUEST_TRACE_DIR_ENV, raising=False)
        assert request_trace_dir_from_env() is None
        monkeypatch.setenv(REQUEST_TRACE_DIR_ENV, "")
        assert request_trace_dir_from_env() is None
        monkeypatch.setenv(REQUEST_TRACE_DIR_ENV, "/tmp/traces")
        assert request_trace_dir_from_env() == "/tmp/traces"


class TestRequestRecords:
    @pytest.mark.asyncio
    async def test_route_derived_from_url(self, tmp_path):
        writer = RequestTraceWriter(str(tmp_path))
        await writer.start()
        handle = await writer.on_request(FakeRequest(body={}, path="/v1/responses"))
        await drain(writer)

        assert handle.route == "/v1/responses"
        (record,) = read_lines(tmp_path, "_no_session", "requests")
        assert record["route"] == "/v1/responses"

    @pytest.mark.asyncio
    async def test_accepted_request(self, tmp_path):
        writer = RequestTraceWriter(str(tmp_path))
        await writer.start()
        request = FakeRequest(
            body={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
            headers={"x-session-id": "sess_a", "authorization": "Bearer secret"},
        )
        handle = await writer.on_request(request)
        await drain(writer)

        assert handle.session == "sess_a"
        assert handle.trace_id.startswith("tr_")
        assert handle.route == "/v1/messages"
        # Part 3 reaches the context through the request, not through an
        # argument threaded down the handler.
        assert request.state.request_trace_handle is handle
        (record,) = read_lines(tmp_path, "sess_a", "requests")
        assert record["event"] == "request"
        assert record["status"] == "accepted"
        assert record["recorded_at"].endswith("+00:00")
        assert record["route"] == "/v1/messages"
        assert record["trace_id"] == handle.trace_id
        assert record["body"]["messages"][0]["content"] == "hi"

    @pytest.mark.asyncio
    async def test_arrival_time_recorded(self, tmp_path):
        writer = RequestTraceWriter(str(tmp_path))
        await writer.start()
        await writer.on_request(FakeRequest(body={}, arrival_time=12345.5))
        await drain(writer)

        (record,) = read_lines(tmp_path, "_no_session", "requests")
        assert record["server_arrival_time"] == 12345.5

    @pytest.mark.asyncio
    async def test_arrival_time_absent_is_null(self, tmp_path):
        # The middleware is installed unconditionally, but the field must not
        # be load-bearing: a request that reaches the hook without one still
        # produces a record.
        writer = RequestTraceWriter(str(tmp_path))
        await writer.start()
        await writer.on_request(FakeRequest(body={}))
        await drain(writer)

        (record,) = read_lines(tmp_path, "_no_session", "requests")
        assert record["server_arrival_time"] is None

    @pytest.mark.asyncio
    async def test_repeated_headers_all_kept(self, tmp_path):
        writer = RequestTraceWriter(str(tmp_path))
        await writer.start()

        class MultiHeaders(FakeHeaders):
            # HTTP allows repeats and starlette's Headers is a multidict; a
            # dict-shaped dump would silently keep only the last hop.
            def items(self):
                return [
                    ("x-forwarded-for", "10.0.0.1"),
                    ("x-forwarded-for", "10.0.0.2"),
                    ("user-agent", "claude-cli/2.1"),
                ]

        request = FakeRequest(body={})
        request.headers = MultiHeaders()
        await writer.on_request(request)
        await drain(writer)

        (record,) = read_lines(tmp_path, "_no_session", "requests")
        assert record["headers"] == [
            ["x-forwarded-for", "10.0.0.1"],
            ["x-forwarded-for", "10.0.0.2"],
            ["user-agent", "claude-cli/2.1"],
        ]

    # Credential-bearing names in the case a client might send them. The value
    # of each is unique so a leak anywhere in the file is caught, not just one
    # in the header list.
    CREDENTIALS = {
        "Authorization": "Bearer sk-live-a1",
        "Proxy-Authorization": "Basic cHJveHk6YjI=",
        "X-Api-Key": "sk-ant-c3",
        "api-key": "azure-d4",
        "Cookie": "session=e5",
        "set-cookie": "id=f6",
        "x-auth-token": "tok-g7",
        "X-Amz-Security-Token": "tok-h8",
        "x-client-secret": "sec-i9",
    }

    @pytest.mark.asyncio
    @pytest.mark.parametrize("hook", ["on_request", "on_rejected"])
    async def test_credentials_are_redacted_but_named(self, tmp_path, hook):
        writer = RequestTraceWriter(str(tmp_path))
        await writer.start()
        request = FakeRequest(
            body={},
            headers={
                **self.CREDENTIALS,
                "x-session-id": "s_creds",
                "user-agent": "claude-cli/2.1",
            },
        )
        if hook == "on_request":
            await writer.on_request(request)
        else:
            await writer.on_rejected(request, [])
        await drain(writer)

        (record,) = read_lines(tmp_path, "s_creds", "requests")
        dumped = dict(record["headers"])
        # Presence stays visible under the name the client used...
        for name in self.CREDENTIALS:
            assert dumped[name] == "[redacted]", name
        # ...while the headers the trace exists to keep are untouched.
        assert dumped["x-session-id"] == "s_creds"
        assert dumped["user-agent"] == "claude-cli/2.1"
        written = "".join(path.read_text() for path in tmp_path.glob("*/requests*.jsonl"))
        for value in self.CREDENTIALS.values():
            assert value not in written

    @pytest.mark.asyncio
    async def test_repeated_credential_headers_are_each_redacted(self, tmp_path):
        writer = RequestTraceWriter(str(tmp_path))
        await writer.start()

        class MultiHeaders(FakeHeaders):
            def items(self):
                return [("cookie", "a=1"), ("cookie", "b=2"), ("x-session-id", "s_multi")]

        request = FakeRequest(body={})
        request.headers = MultiHeaders()
        await writer.on_request(request)
        await drain(writer)

        (record,) = read_lines(tmp_path, "s_multi", "requests")
        assert record["headers"] == [
            ["cookie", "[redacted]"],
            ["cookie", "[redacted]"],
            ["x-session-id", "s_multi"],
        ]

    @pytest.mark.asyncio
    async def test_unparseable_body_kept_as_text(self, tmp_path):
        writer = RequestTraceWriter(str(tmp_path))
        await writer.start()
        request = FakeRequest(
            raw_body=b'{"model": "m",',
            json_error=ValueError("Expecting value"),
        )
        await writer.on_rejected(
            request, [{"loc": ("body",), "type": "json_invalid", "msg": "bad"}]
        )
        await drain(writer)

        (record,) = read_lines(tmp_path, "_no_session", "requests")
        assert record["status"] == "rejected_400"
        assert "server_arrival_time" in record
        assert record["body"] == '{"model": "m",'
        assert "ValueError" in record["body_parse_error"]
        assert record["validation_errors"][0]["type"] == "json_invalid"

    @pytest.mark.asyncio
    async def test_rejected_request_has_no_response_line(self, tmp_path):
        writer = RequestTraceWriter(str(tmp_path))
        await writer.start()
        await writer.on_rejected(FakeRequest(body={"a": 1}), [])
        await drain(writer)

        assert len(read_lines(tmp_path, "_no_session", "requests")) == 1
        assert read_lines(tmp_path, "_no_session", "responses") == []


class TestStreamingResponse:
    @pytest.mark.asyncio
    async def test_frames_recorded_verbatim(self, tmp_path):
        writer = RequestTraceWriter(str(tmp_path))
        await writer.start()
        handle = await writer.on_request(FakeRequest(body={}, headers={"x-session-id": "s1"}))
        handle.set_ids(client_id=184)

        async def source():
            yield "event: message_start\ndata: {}\n\n"
            yield "event: message_stop\ndata: {}\n\n"

        seen = [chunk async for chunk in writer.wrap_stream(source(), handle)]
        await drain(writer)

        assert seen == [
            "event: message_start\ndata: {}\n\n",
            "event: message_stop\ndata: {}\n\n",
        ]
        (record,) = read_lines(tmp_path, "s1", "responses")
        assert record["status"] == "completed"
        assert record["client_id"] == 184
        assert record["response"]["kind"] == "sse_text"
        assert record["response"]["body"] == "".join(seen)

    @pytest.mark.asyncio
    async def test_client_disconnect_keeps_partial_frames(self, tmp_path):
        writer = RequestTraceWriter(str(tmp_path))
        await writer.start()
        handle = await writer.on_request(FakeRequest(body={}, headers={"x-session-id": "s2"}))

        async def source():
            yield "frame-0"
            yield "frame-1"
            yield "frame-2"

        stream = writer.wrap_stream(source(), handle)
        assert await stream.__anext__() == "frame-0"
        assert await stream.__anext__() == "frame-1"
        # What starlette does when the client hangs up mid-stream.
        await stream.aclose()
        await drain(writer)

        (record,) = read_lines(tmp_path, "s2", "responses")
        assert record["status"] == "client_disconnected"
        assert record["response"]["body"] == "frame-0frame-1"

    @pytest.mark.asyncio
    async def test_upstream_error_is_recorded_and_reraised(self, tmp_path):
        writer = RequestTraceWriter(str(tmp_path))
        await writer.start()
        handle = await writer.on_request(FakeRequest(body={}, headers={"x-session-id": "s3"}))

        async def source():
            yield "frame-0"
            raise RuntimeError("engine died")

        with pytest.raises(RuntimeError, match="engine died"):
            async for _ in writer.wrap_stream(source(), handle):
                pass
        await drain(writer)

        (record,) = read_lines(tmp_path, "s3", "responses")
        assert record["status"] == "error"
        assert record["response"]["body"] == "frame-0"

    @pytest.mark.asyncio
    async def test_bytes_frames_coerced(self, tmp_path):
        writer = RequestTraceWriter(str(tmp_path))
        await writer.start()
        handle = await writer.on_request(FakeRequest(body={}))

        async def source():
            yield b"raw-bytes"

        async for _ in writer.wrap_stream(source(), handle):
            pass
        await drain(writer)

        (record,) = read_lines(tmp_path, "_no_session", "responses")
        assert record["response"]["body"] == "raw-bytes"

    @pytest.mark.asyncio
    async def test_multibyte_character_split_across_reads(self, tmp_path):
        """A read that ends mid-character must not decode to replacement chars.

        ``iter_any()`` splits wherever the socket did, and that can fall inside
        a multi-byte character. Decoded per read, the one character becomes
        U+FFFD on both sides of the seam -- and the mangled frame still parses
        as JSON, so the trace keeps corrupted text with nothing looking wrong.
        """
        writer = RequestTraceWriter(str(tmp_path))
        await writer.start()
        handle = await writer.on_request(FakeRequest(body={}))
        frame = 'data: {"choices":[{"delta":{"content":"\u4f60\u597d"}}]}\n\n'.encode()
        cut = frame.index("\u4f60".encode()) + 1

        async def source():
            yield frame[:cut]
            yield frame[cut:]

        async for _ in writer.wrap_stream(source(), handle):
            pass
        await drain(writer)

        (record,) = read_lines(tmp_path, "_no_session", "responses")
        assert record["response"]["body"] == frame.decode()

    @pytest.mark.asyncio
    async def test_cancellation_is_recorded_as_a_disconnect(self, tmp_path):
        """Starlette cancels the scope the body iterator runs in on hangup.

        Delivered while this generator is suspended it arrives as
        CancelledError, not GeneratorExit, and used to fall through to the
        catch-all and be recorded as "error" -- the same bucket a genuine
        server fault lands in, which is where a census of truncated streams
        goes looking for one.
        """
        writer = RequestTraceWriter(str(tmp_path))
        await writer.start()
        handle = await writer.on_request(FakeRequest(body={}, headers={"x-session-id": "s5"}))

        async def source():
            yield "frame-0"
            raise asyncio.CancelledError

        with pytest.raises(asyncio.CancelledError):
            async for _ in writer.wrap_stream(source(), handle):
                pass
        await drain(writer)

        (record,) = read_lines(tmp_path, "s5", "responses")
        assert record["status"] == "client_disconnected"
        assert record["response"]["body"] == "frame-0"

    @pytest.mark.asyncio
    async def test_producer_reported_cause_is_recorded(self, tmp_path):
        """The status says a stream failed; only the producer knows how."""
        writer = RequestTraceWriter(str(tmp_path))
        await writer.start()
        handle = await writer.on_request(FakeRequest(body={}, headers={"x-session-id": "s6"}))

        async def source():
            yield "frame-0"
            writer.note_stream_termination(handle, "internal_error", "ValueError: x")
            raise ValueError("x")

        with pytest.raises(ValueError):
            async for _ in writer.wrap_stream(source(), handle):
                pass
        await drain(writer)

        (record,) = read_lines(tmp_path, "s6", "responses")
        assert record["status"] == "error"
        assert record["termination"] == {"cause": "internal_error", "detail": "ValueError: x"}

    @pytest.mark.asyncio
    async def test_no_cause_leaves_the_field_off(self, tmp_path):
        writer = RequestTraceWriter(str(tmp_path))
        await writer.start()
        handle = await writer.on_request(FakeRequest(body={}, headers={"x-session-id": "s7"}))

        async def source():
            yield "frame-0"

        async for _ in writer.wrap_stream(source(), handle):
            pass
        await drain(writer)

        (record,) = read_lines(tmp_path, "s7", "responses")
        assert "termination" not in record

    def test_note_stream_termination_tolerates_an_absent_handle(self):
        """Producers call this from an exception handler, tracing off or on."""
        RequestTraceWriter(None).note_stream_termination(None, "internal_error", "x")

    def test_join_frames_keeps_str_and_bytes_in_order(self):
        assert _join_frames([b"a", "b", b"c"]) == "abc"


@contextlib.contextmanager
def cyclic_gc_off():
    """Both servers run with gc.disable(); a record must not wait on a collection."""
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        yield
    finally:
        if was_enabled:
            gc.enable()


class TestStreamNeverStarted:
    """An accepted streaming request whose body is never iterated.

    An async generator that is never started runs none of its body, so the
    ``finally`` that writes the response line never runs either. Starlette does
    exactly that when the client is already gone as the response begins: under
    ASGI 2.4 ``send`` raises, the response raises ClientDisconnect, and the
    body iterator is dropped untouched -- an accepted request with no terminal
    record.
    """

    @staticmethod
    async def _accepted(writer, session):
        return await writer.on_request(FakeRequest(body={}, headers={"x-session-id": session}))

    @staticmethod
    async def _source():
        yield "frame-0"

    @pytest.mark.asyncio
    async def test_dropped_without_being_started(self, tmp_path):
        writer = RequestTraceWriter(str(tmp_path))
        await writer.start()
        handle = await self._accepted(writer, "s_dropped")

        with cyclic_gc_off():
            stream = writer.wrap_stream(self._source(), handle)
            del stream
        await drain(writer)

        (record,) = read_lines(tmp_path, "s_dropped", "responses")
        assert record["status"] == "client_disconnected"
        assert record["termination"]["cause"] == "not_started"
        assert record["response"]["body"] == ""
        assert record["trace_id"] == handle.trace_id

    @pytest.mark.asyncio
    async def test_closed_before_the_first_chunk_is_recorded_once(self, tmp_path):
        writer = RequestTraceWriter(str(tmp_path))
        await writer.start()
        handle = await self._accepted(writer, "s_closed")

        with cyclic_gc_off():
            stream = writer.wrap_stream(self._source(), handle)
            await stream.aclose()
            del stream
        await drain(writer)

        (record,) = read_lines(tmp_path, "s_closed", "responses")
        assert record["status"] == "client_disconnected"

    @pytest.mark.asyncio
    async def test_starlette_dropping_an_unstarted_body(self, tmp_path):
        """The real path: a 2.4 server's send raises, the body is never touched."""
        from starlette.responses import StreamingResponse

        writer = RequestTraceWriter(str(tmp_path))
        await writer.start()
        handle = await self._accepted(writer, "s_gone")
        iterated = []

        async def source():
            iterated.append(True)
            yield "data: {}\n\n"

        async def receive():
            return {"type": "http.disconnect"}

        async def send(message):
            raise OSError("client already gone")

        async def serve(response):
            scope = {"type": "http", "asgi": {"version": "3.0", "spec_version": "2.4"}}
            try:
                await response(scope, receive, send)
            except Exception as error:  # noqa: BLE001 - reported, then dropped with its frames
                return type(error).__name__
            return None

        with cyclic_gc_off():
            response = StreamingResponse(
                writer.wrap_stream(source(), handle), media_type="text/event-stream"
            )
            assert await serve(response) == "ClientDisconnect"
            del response
        await drain(writer)

        assert not iterated
        (record,) = read_lines(tmp_path, "s_gone", "responses")
        assert record["status"] == "client_disconnected"
        assert record["termination"]["cause"] == "not_started"

    @pytest.mark.asyncio
    async def test_a_terminal_already_recorded_is_not_overwritten(self, tmp_path):
        """Exactly once: a server that closed the trace itself keeps its status."""
        writer = RequestTraceWriter(str(tmp_path))
        await writer.start()
        handle = await self._accepted(writer, "s_first")

        with cyclic_gc_off():
            stream = writer.wrap_stream(self._source(), handle)
            writer.on_response(handle, payload={"error": "boom"}, status="error")
            del stream
        await drain(writer)

        (record,) = read_lines(tmp_path, "s_first", "responses")
        assert record["status"] == "error"
        assert "termination" not in record

    @pytest.mark.asyncio
    async def test_a_started_stream_records_only_its_own_outcome(self, tmp_path):
        writer = RequestTraceWriter(str(tmp_path))
        await writer.start()
        handle = await self._accepted(writer, "s_ran")

        with cyclic_gc_off():
            stream = writer.wrap_stream(self._source(), handle)
            assert [chunk async for chunk in stream] == ["frame-0"]
            del stream
        await drain(writer)

        (record,) = read_lines(tmp_path, "s_ran", "responses")
        assert record["status"] == "completed"
        assert record["response"]["body"] == "frame-0"
        assert "termination" not in record


class TestNonStreamingResponse:
    @pytest.mark.asyncio
    async def test_json_payload(self, tmp_path):
        writer = RequestTraceWriter(str(tmp_path))
        await writer.start()
        handle = await writer.on_request(FakeRequest(body={}, headers={"x-session-id": "s4"}))
        handle.set_ids(disagg_request_id=99, ctx_request_id=7)
        writer.on_response(handle, payload={"id": "msg_1", "content": []})
        await drain(writer)

        (record,) = read_lines(tmp_path, "s4", "responses")
        assert record["response"] == {"kind": "json", "body": {"id": "msg_1", "content": []}}
        assert record["disagg_request_id"] == 99
        assert record["ctx_request_id"] == 7
        assert record["client_id"] is None

    @pytest.mark.asyncio
    async def test_written_at_most_once(self, tmp_path):
        writer = RequestTraceWriter(str(tmp_path))
        await writer.start()
        handle = await writer.on_request(FakeRequest(body={}))
        writer.on_response(handle, payload={"first": True})
        writer.on_response(handle, payload={"second": True})
        await drain(writer)

        records = read_lines(tmp_path, "_no_session", "responses")
        assert len(records) == 1
        assert records[0]["response"]["body"] == {"first": True}

    @pytest.mark.asyncio
    async def test_an_error_answer_keeps_its_http_status(self, tmp_path):
        writer = RequestTraceWriter(str(tmp_path))
        await writer.start()
        handle = await writer.on_request(FakeRequest(body={}, headers={"x-session-id": "h1"}))
        writer.on_response(
            handle, payload={"detail": "engine overloaded"}, status="error", http_status=503
        )
        await drain(writer)

        (record,) = read_lines(tmp_path, "h1", "responses")
        assert record["http_status"] == 503
        assert record["response"] == {"kind": "json", "body": {"detail": "engine overloaded"}}

    @pytest.mark.asyncio
    async def test_a_reply_without_an_http_status_records_none(self, tmp_path):
        writer = RequestTraceWriter(str(tmp_path))
        await writer.start()
        handle = await writer.on_request(FakeRequest(body={}, headers={"x-session-id": "h2"}))
        writer.on_response(handle, payload={"id": "msg_1"})
        await drain(writer)

        (record,) = read_lines(tmp_path, "h2", "responses")
        assert "http_status" not in record

    @pytest.mark.asyncio
    async def test_a_plain_text_answer_is_recorded_as_text(self, tmp_path):
        writer = RequestTraceWriter(str(tmp_path))
        await writer.start()
        handle = await writer.on_request(FakeRequest(body={}, headers={"x-session-id": "h3"}))
        writer.on_response(handle, payload="Internal Server Error", status="error", http_status=500)
        await drain(writer)

        (record,) = read_lines(tmp_path, "h3", "responses")
        assert record["response"] == {"kind": "text", "body": "Internal Server Error"}


def _sse(event_type, data):
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"


_BLED_EXEC = {
    "type": "custom_tool_call",
    "id": "ctc_1",
    "call_id": "call_1",
    "name": "exec",
    "input": '// @exec: {"max_output_tokens</arg_key><arg_value>20000, "yield_time_ms": 30000}',
}
_CLEAN_CALL = {
    "type": "function_call",
    "id": "fc_2",
    "call_id": "call_2",
    "name": "exec",
    "arguments": '{"input": "ls"}',
}


class TestToolCallMarkup:
    """Calls whose arguments still hold tool-call markup are flagged, not changed."""

    def test_a_streamed_call_with_markup_is_flagged_once(self):
        text = "".join(
            [
                _sse("response.output_item.done", {"item": _BLED_EXEC}),
                _sse("response.output_item.done", {"item": _CLEAN_CALL}),
                _sse("response.completed", {"response": {"output": [_BLED_EXEC, _CLEAN_CALL]}}),
            ]
        )
        assert find_tool_call_markup(text) == [
            {"call_id": "call_1", "name": "exec", "markers": ["</arg_key>", "<arg_value>"]}
        ]

    def test_markup_outside_tool_calls_is_not_flagged(self):
        """A model can write about the format in its reasoning or its answer."""
        message = {
            "type": "message",
            "content": [{"type": "output_text", "text": "Calls look like <arg_key>k</arg_key>."}],
        }
        text = _sse("response.completed", {"response": {"output": [message, _CLEAN_CALL]}})
        assert find_tool_call_markup(text) == []

    def test_a_json_reply_is_checked_too(self):
        payload = {"output": [_CLEAN_CALL, _BLED_EXEC]}
        (flagged,) = find_tool_call_markup(None, payload)
        assert flagged["call_id"] == "call_1"

    @pytest.mark.asyncio
    async def test_the_response_record_carries_the_flag(self, tmp_path):
        writer = RequestTraceWriter(str(tmp_path))
        await writer.start()
        handle = await writer.on_request(FakeRequest(body={}, headers={"x-session-id": "s9"}))

        async def source():
            yield _sse("response.output_item.done", {"item": _BLED_EXEC})
            yield _sse("response.completed", {"response": {"output": [_BLED_EXEC]}})

        [chunk async for chunk in writer.wrap_stream(source(), handle)]
        await drain(writer)

        (record,) = read_lines(tmp_path, "s9", "responses")
        assert record["tool_call_markup"][0]["call_id"] == "call_1"

    @pytest.mark.asyncio
    async def test_a_clean_response_record_has_no_flag(self, tmp_path):
        writer = RequestTraceWriter(str(tmp_path))
        await writer.start()
        handle = await writer.on_request(FakeRequest(body={}, headers={"x-session-id": "s10"}))
        writer.on_response(handle, payload={"output": [_CLEAN_CALL]})
        await drain(writer)

        (record,) = read_lines(tmp_path, "s10", "responses")
        assert "tool_call_markup" not in record


def _responses_object(status, **fields):
    return {"object": "response", "id": "resp_1", "status": status, "output": [], **fields}


class TestResponsesOutcome:
    """The record says how the generation ended, not just how the transport did."""

    @pytest.mark.asyncio
    async def test_a_stream_cut_by_the_token_budget_reads_incomplete(self, tmp_path):
        """The transport completed; the generation did not."""
        writer = RequestTraceWriter(str(tmp_path))
        await writer.start()
        handle = await writer.on_request(FakeRequest(body={}, headers={"x-session-id": "o1"}))
        incomplete = _responses_object(
            "incomplete", incomplete_details={"reason": "max_output_tokens"}
        )

        async def source():
            yield _sse("response.created", {"response": _responses_object("in_progress")})
            yield _sse("response.incomplete", {"response": incomplete})

        [chunk async for chunk in writer.wrap_stream(source(), handle)]
        await drain(writer)

        (record,) = read_lines(tmp_path, "o1", "responses")
        assert record["status"] == "completed"
        assert record["response_status"] == "incomplete"
        assert record["incomplete_reason"] == "max_output_tokens"
        assert "error_code" not in record

    def test_a_finished_stream_reads_completed(self):
        text = _sse("response.created", {"response": _responses_object("in_progress")}) + _sse(
            "response.completed", {"response": _responses_object("completed")}
        )
        assert responses_outcome(text) == {"response_status": "completed"}

    def test_a_failed_stream_carries_its_error_code(self):
        failed = _responses_object(
            "failed", error={"code": "context_length_exceeded", "message": "too long"}
        )
        text = _sse("error", {"code": "context_length_exceeded"}) + _sse(
            "response.failed", {"response": failed}
        )
        assert responses_outcome(text) == {
            "response_status": "failed",
            "error_code": "context_length_exceeded",
        }

    def test_a_stream_cut_before_its_terminal_event_states_nothing(self):
        text = _sse("response.created", {"response": _responses_object("in_progress")}) + _sse(
            "response.output_text.delta", {"delta": "partial"}
        )
        assert responses_outcome(text) == {}

    @pytest.mark.asyncio
    async def test_a_json_responses_reply_is_read_too(self, tmp_path):
        writer = RequestTraceWriter(str(tmp_path))
        await writer.start()
        handle = await writer.on_request(FakeRequest(body={}, headers={"x-session-id": "o2"}))
        writer.on_response(
            handle,
            payload=_responses_object(
                "incomplete", incomplete_details={"reason": "max_output_tokens"}
            ),
        )
        await drain(writer)

        (record,) = read_lines(tmp_path, "o2", "responses")
        assert record["response_status"] == "incomplete"
        assert record["incomplete_reason"] == "max_output_tokens"

    @pytest.mark.parametrize(
        "payload",
        [
            {"id": "chatcmpl-1", "object": "chat.completion", "choices": []},
            {"error": {"code": "context_length_exceeded"}},
            None,
        ],
    )
    def test_other_replies_state_nothing(self, payload):
        assert responses_outcome(None, payload) == {}


class TestWriterMechanics:
    @pytest.mark.asyncio
    async def test_sessions_share_one_bucket_and_stay_separable(self, tmp_path):
        """Session count is unbounded, so it is a field, not a directory."""
        writer = RequestTraceWriter(str(tmp_path))
        await writer.start()
        for session in ("s_a", "s_b", "s_a"):
            await writer.on_request(FakeRequest(body={}, headers={"x-session-id": session}))
        await drain(writer)

        assert len(read_lines(tmp_path, "s_a", "requests")) == 2
        assert len(read_lines(tmp_path, "s_b", "requests")) == 1
        # The writer's own accounting lives beside the buckets, not in them.
        buckets = [p for p in tmp_path.iterdir() if p.is_dir() and p.name != WRITER_SIDECAR_DIR]
        assert len(buckets) == 1

    @pytest.mark.asyncio
    async def test_writer_suffix_applied(self, tmp_path):
        writer = RequestTraceWriter(str(tmp_path), writer_suffix="-1234")
        await writer.start()
        await writer.on_request(FakeRequest(body={}, headers={"x-session-id": "s"}))
        await drain(writer)

        (path,) = tmp_path.glob("*/requests-1234.jsonl")
        assert path.parent.name == datetime.now(timezone.utc).strftime("%Y-%m-%dT%H")

    @pytest.mark.asyncio
    async def test_unserializable_record_dropped_not_raised(self, tmp_path):
        writer = RequestTraceWriter(str(tmp_path))
        await writer.start()
        handle = await writer.on_request(FakeRequest(body={}))
        # object() has no JSON form; the batch must survive it.
        writer.on_response(handle, payload=object())
        await drain(writer)

        assert read_lines(tmp_path, "_no_session", "responses") == []
        assert writer.dropped_records == 1

    @pytest.mark.asyncio
    async def test_unwritable_directory_disables_quietly(self, tmp_path):
        blocker = tmp_path / "traces"
        blocker.write_text("not a directory")
        writer = RequestTraceWriter(str(blocker))
        await writer.start()
        assert not writer.enabled
        assert await writer.on_request(FakeRequest(body={})) is None
        await writer.close()

    @pytest.mark.asyncio
    async def test_full_queue_drops_without_raising(self, tmp_path):
        writer = RequestTraceWriter(str(tmp_path))
        await writer.start()
        # Stall the drain so put_nowait hits the bound, then overfill it.
        # Sized off the constant: a hardcoded count silently stops testing the
        # drop path the moment the queue is resized.
        writer._task.cancel()
        for _ in range(_WRITER_QUEUE_SIZE + 10):
            writer._submit("s", "requests", {"ok": True})
        assert writer.dropped_records == 10
        writer._task = None


class TestWriterResilience:
    """One poisoned record or one failed batch must never kill the drain task.

    The failure mode being guarded: the task dies, ``enabled`` stays True, the
    counters stay still, and every record from that moment on queues toward a
    drain that no longer exists -- the trace silently stops. UnicodeEncodeError
    out of ``file.write`` (a ValueError the write-side clause did not catch)
    did exactly that once.
    """

    @pytest.mark.asyncio
    async def test_unpaired_surrogate_does_not_kill_the_writer(self, tmp_path):
        writer = RequestTraceWriter(str(tmp_path))
        await writer.start()
        handle = await writer.on_request(FakeRequest(body={}, headers={"x-session-id": "s_bad"}))
        # Half a surrogate pair: json.dumps(ensure_ascii=False) emits it
        # happily, and only the UTF-8 encode inside file.write() objects.
        writer.on_response(handle, payload={"text": "cut \ud800 here"})
        # Sanitation happens while the batch is prepared, so this doubles as
        # "the drain task got past the record that used to kill it".
        await _until(lambda: writer.sanitized_records == 1)
        assert not writer._task.done()

        # The definitive proof of survival: a later record still lands.
        later = await writer.on_request(FakeRequest(body={}, headers={"x-session-id": "s_good"}))
        assert later is not None
        await drain(writer)

        assert writer.sanitized_records == 1
        assert writer.dropped_records == 0
        assert writer.last_write_at is not None
        (bad,) = read_lines(tmp_path, "s_bad", "responses")
        # Lossless: the record is written with \u escapes, which JSON reads back
        # as the very code point the client sent.
        assert bad["response"]["body"]["text"] == "cut \ud800 here"
        (good,) = read_lines(tmp_path, "s_good", "requests")
        assert good["status"] == "accepted"

    @pytest.mark.asyncio
    async def test_healthy_batchmates_survive_a_poisoned_record(self, tmp_path):
        writer = RequestTraceWriter(str(tmp_path))
        await writer.start()
        # No await between submits: both records sit on the queue before the
        # drain task wakes, so they flush as one batch -- the shape in which a
        # single poisoned record used to take its healthy batchmates with it.
        writer._submit("2026-01-01T00", "requests", {"session": "s_ok", "text": "fine"})
        writer._submit("2026-01-01T00", "requests", {"session": "s_poison", "text": "\udfff"})
        await drain(writer)

        (ok,) = read_lines(tmp_path, "s_ok", "requests")
        assert ok["text"] == "fine"
        (poisoned,) = read_lines(tmp_path, "s_poison", "requests")
        assert poisoned["text"] == "\udfff"
        assert writer.sanitized_records == 1
        assert writer.dropped_records == 0

    @pytest.mark.asyncio
    async def test_only_the_unencodable_record_is_escaped(self, tmp_path):
        """Escaping is per record: a batchmate keeps its text as raw UTF-8.

        And nothing is added to the escaped record -- it reads back identical
        to what was submitted, so no consumer has a second shape to handle.
        """
        writer = RequestTraceWriter(str(tmp_path))
        await writer.start()
        clean = {"session": "s_clean", "text": "你好"}
        poisoned = {"session": "s_lone", "text": "你 \ud83d"}
        writer._submit("2026-01-01T00", "requests", clean)
        writer._submit("2026-01-01T00", "requests", poisoned)
        await drain(writer)

        (path,) = tmp_path.glob("2026-01-01T00/requests*.jsonl")
        clean_line, poisoned_line = path.read_text(encoding="utf-8").splitlines()
        assert "你好" in clean_line
        assert poisoned_line.isascii()
        assert json.loads(clean_line) == clean
        assert json.loads(poisoned_line) == poisoned

    @pytest.mark.asyncio
    async def test_unexpected_write_failure_does_not_kill_the_writer(self, tmp_path, monkeypatch):
        """UnicodeEncodeError was the instance; the guarantee is broader.

        Whatever one batch's write raises, the batch is counted as dropped and
        the loop keeps draining.
        """
        writer = RequestTraceWriter(str(tmp_path))
        await writer.start()
        real_write = RequestTraceWriter._write_groups
        failures = iter([RuntimeError("neither an OSError nor a serialization error")])

        def flaky(groups):
            error = next(failures, None)
            if error is not None:
                raise error
            real_write(writer, groups)

        monkeypatch.setattr(writer, "_write_groups", flaky)

        await writer.on_request(FakeRequest(body={}, headers={"x-session-id": "s_lost"}))
        await _until(lambda: writer._write_error_count == 1)
        assert not writer._task.done()
        assert writer.last_write_at is None  # nothing has landed yet

        await writer.on_request(FakeRequest(body={}, headers={"x-session-id": "s_kept"}))
        await drain(writer)

        assert writer.dropped_records == 1  # the failed batch is accounted for
        assert writer.last_write_at is not None  # the later flush did land
        assert read_lines(tmp_path, "s_lost", "requests") == []
        (kept,) = read_lines(tmp_path, "s_kept", "requests")
        assert kept["status"] == "accepted"

    @pytest.mark.asyncio
    async def test_partial_batch_failure_counts_only_the_unwritten_groups(self, tmp_path):
        """A poisoned second group must not drag the first group into the drop count.

        _write_groups appends one file per (bucket, kind) group. When a later
        group cannot be written, the groups before it are already durably on
        disk, so only the unwritten remainder is genuinely lost. The old handler
        charged the WHOLE batch to dropped_records -- overcounting the loss and
        contradicting the file sitting on disk.
        """
        writer = RequestTraceWriter(str(tmp_path))
        await writer.start()
        # Two groups in one batch: distinct hour buckets so _write_groups makes
        # two separate appends, processed in submit order. Block the SECOND
        # bucket by planting a file where its directory must go, so its mkdir
        # raises FileExistsError (an OSError) only AFTER the first group's append
        # has already returned.
        good_bucket = "2026-01-01T00"
        bad_bucket = "2026-01-01T01"
        (tmp_path / bad_bucket).write_text("a file, not a directory")
        # No await between submits, so both land in one drained batch -- the
        # shape a partial failure needs (one write call spanning both groups).
        writer._submit(good_bucket, "requests", {"session": "s_ok", "text": "landed"})
        writer._submit(bad_bucket, "requests", {"session": "s_lost", "text": "never"})
        await drain(writer)

        # The first group is on disk; only the second group's single line is lost.
        (ok,) = read_lines(tmp_path, "s_ok", "requests")
        assert ok["text"] == "landed"
        assert writer.dropped_records == 1
        assert read_lines(tmp_path, "s_lost", "requests") == []
        # The failure was still logged/accounted, just not overcounted.
        assert writer._write_error_count == 1

    @pytest.mark.asyncio
    async def test_last_write_at_tracks_successful_flushes(self, tmp_path):
        writer = RequestTraceWriter(str(tmp_path))
        await writer.start()
        assert writer.last_write_at is None
        await writer.on_request(FakeRequest(body={}))
        await drain(writer)

        # Same ISO form as every recorded_at, so a monitor comparing the two
        # needs no conversion.
        assert writer.last_write_at is not None
        assert writer.last_write_at.endswith("+00:00")


class TestHalfWrittenLineRecovery:
    """An append must not weld a new record onto an orphaned fragment.

    A write cut short (crash, quota) leaves the file ending mid-line; gluing
    the next record onto it corrupts that record too -- two lost where one
    already was. ``_write_groups`` is exercised directly so the test does not
    depend on the wall-clock hour bucket.
    """

    BUCKET = "2026-01-01T00"

    def _path(self, tmp_path):
        return tmp_path / self.BUCKET / "requests-w.jsonl"

    def test_append_isolates_a_half_written_line(self, tmp_path):
        writer = RequestTraceWriter(str(tmp_path), writer_suffix="-w")
        path = self._path(tmp_path)
        path.parent.mkdir()
        path.write_text('{"cut": "mid')  # what an interrupted write leaves
        writer._write_groups({(self.BUCKET, "requests"): ['{"ok":1}\n']})

        lines = path.read_text().splitlines()
        assert lines == ['{"cut": "mid', '{"ok":1}']
        # The fragment is one bad line a JSONL reader skips; the record after
        # it parses untouched.
        assert json.loads(lines[1]) == {"ok": 1}

    @pytest.mark.parametrize("existing", ["", '{"whole":1}\n'])
    def test_no_spurious_newline_when_none_is_missing(self, tmp_path, existing):
        writer = RequestTraceWriter(str(tmp_path), writer_suffix="-w")
        path = self._path(tmp_path)
        path.parent.mkdir()
        path.write_text(existing)
        writer._write_groups({(self.BUCKET, "requests"): ['{"ok":1}\n']})

        assert path.read_text() == existing + '{"ok":1}\n'

    def test_first_write_needs_no_repair(self, tmp_path):
        writer = RequestTraceWriter(str(tmp_path), writer_suffix="-w")
        writer._write_groups({(self.BUCKET, "requests"): ['{"ok":1}\n']})

        assert self._path(tmp_path).read_text() == '{"ok":1}\n'


class TestNestedHandlerOwnership:
    """Only the handler the client called may record the response.

    /v1/messages converts and then calls the chat handler with the same Request.
    Both would wrap their own generator, and the inner one is drained by the
    outer, so it reaches its ``finally`` first and claims the record -- storing
    the OpenAI frames the Anthropic route exists to convert away. Handing the
    inner handler None is what stops that, and ``response_written`` cannot: it
    enforces exactly-once, not which one wins.
    """

    @pytest.mark.asyncio
    async def test_reentry_gets_no_handle(self, tmp_path):
        writer = RequestTraceWriter(str(tmp_path))
        await writer.start()
        request = FakeRequest(body={}, headers={"x-session-id": "s"})

        outer = await writer.on_request(request)
        inner = await writer.on_request(request)
        await drain(writer)

        assert outer is not None
        assert inner is None
        assert len(read_lines(tmp_path, "s", "requests")) == 1

    @pytest.mark.asyncio
    async def test_inner_wrap_is_inert_and_outer_keeps_client_frames(self, tmp_path):
        writer = RequestTraceWriter(str(tmp_path))
        await writer.start()
        request = FakeRequest(body={}, headers={"x-session-id": "s"}, path="/v1/messages")
        outer = await writer.on_request(request)  # anthropic_messages
        inner = await writer.on_request(request)  # openai_chat, nested

        async def openai_frames():
            yield "openai-frame"

        async def reframed(source):
            async for _ in source:
                yield "anthropic-frame"

        # Both layers wrap, exactly as the two handlers do.
        teed_inner = writer.wrap_stream(openai_frames(), inner)
        teed_outer = writer.wrap_stream(reframed(teed_inner), outer)
        assert [chunk async for chunk in teed_outer] == ["anthropic-frame"]
        await drain(writer)

        (record,) = read_lines(tmp_path, "s", "responses")
        assert record["response"]["body"] == "anthropic-frame"

    @pytest.mark.asyncio
    async def test_engine_ids_reach_the_handle_from_the_request(self, tmp_path):
        # The chat handler holds the promise even as the inner half of an
        # Anthropic turn -- precisely when it has no handle of its own -- so the
        # id has to be set through the request.
        writer = RequestTraceWriter(str(tmp_path))
        await writer.start()
        request = FakeRequest(body={}, headers={"x-session-id": "s"})
        outer = await writer.on_request(request)
        assert await writer.on_request(request) is None

        request.state.request_trace_handle.set_ids(client_id=184)
        writer.on_response(outer, payload={})
        await drain(writer)

        (record,) = read_lines(tmp_path, "s", "responses")
        assert record["client_id"] == 184


class TestNonStreamingOwnership:
    """The nested handler must stay silent on the non-streaming path too."""

    @pytest.mark.asyncio
    async def test_inner_handler_records_nothing(self, tmp_path):
        # openai_chat reached via /v1/messages returns its own JSON response and
        # calls on_response with the handle it was given -- which is None.
        writer = RequestTraceWriter(str(tmp_path))
        await writer.start()
        request = FakeRequest(body={}, headers={"x-session-id": "s"}, path="/v1/messages")
        outer = await writer.on_request(request)
        inner = await writer.on_request(request)

        writer.on_response(inner, payload={"openai": "shape"})
        writer.on_response(outer, payload={"anthropic": "shape"})
        await drain(writer)

        (record,) = read_lines(tmp_path, "s", "responses")
        assert record["response"]["body"] == {"anthropic": "shape"}

    @pytest.mark.asyncio
    async def test_conversion_failure_is_recorded_with_the_upstream_body(self, tmp_path):
        # convert_chat_response raises on a tool call whose arguments are not a
        # JSON object; the client only sees a 500, so the trace is the only
        # place that sample survives.
        writer = RequestTraceWriter(str(tmp_path))
        await writer.start()
        handle = await writer.on_request(FakeRequest(body={}, headers={"x-session-id": "s"}))

        writer.on_response(
            handle,
            payload={"upstream_body": '{"choices":[{"message":{"tool_calls":[]}}]}'},
            status="conversion_error",
        )
        await drain(writer)

        (record,) = read_lines(tmp_path, "s", "responses")
        assert record["status"] == "conversion_error"
        assert "tool_calls" in record["response"]["body"]["upstream_body"]


class TestAnthropicRouteEndToEnd:
    """The /v1/messages chain with tracing actually on.

    ``anthropic_messages`` converts, hands the request to ``openai_chat``, takes
    the ``body_iterator`` off the StreamingResponse that comes back, reframes it
    and returns a *new* StreamingResponse. Two response objects for one client
    request, and only the second one's frames are what the client reads.
    """

    @staticmethod
    def _client(tmp_path, openai_response):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from tensorrt_llm.serve.openai_server import OpenAIServer

        app = FastAPI()
        server = object.__new__(OpenAIServer)
        server.model = "test-model"
        server._request_trace = RequestTraceWriter(str(tmp_path))
        server.openai_chat = AsyncMock(return_value=openai_response)
        app.add_api_route("/v1/messages", server.anthropic_messages, methods=["POST"])
        return TestClient(app), server

    @staticmethod
    def _openai_stream():
        from starlette.responses import StreamingResponse

        async def source():
            yield (
                'data: {"id":"chatcmpl-1","object":"chat.completion.chunk",'
                '"created":0,"model":"test-model","choices":[{"index":0,'
                '"delta":{"role":"assistant"}}]}\n\n'
            )
            yield (
                'data: {"id":"chatcmpl-1","object":"chat.completion.chunk",'
                '"created":0,"model":"test-model","choices":[{"index":0,'
                '"delta":{"content":"hello"}}]}\n\n'
            )
            yield (
                'data: {"id":"chatcmpl-1","object":"chat.completion.chunk",'
                '"created":0,"model":"test-model","choices":[{"index":0,'
                '"delta":{},"finish_reason":"stop"}]}\n\n'
            )
            yield "data: [DONE]\n\n"

        return StreamingResponse(source(), media_type="text/event-stream")

    @pytest.mark.asyncio
    async def test_streaming_client_output_is_unchanged_and_traced(self, tmp_path):
        client, server = self._client(tmp_path, self._openai_stream())
        await server._request_trace.start()

        response = client.post(
            "/v1/messages",
            json={
                "model": "test-model",
                "max_tokens": 16,
                "stream": True,
                "messages": [{"role": "user", "content": "hi"}],
            },
            headers={"x-session-id": "sess_e2e"},
        )
        await server._request_trace.close()

        # The client still gets a correct Anthropic event stream.
        assert response.status_code == 200
        assert response.text.startswith("event: message_start\n")
        assert '"text":"hello"' in response.text
        assert response.text.rstrip().endswith('event: message_stop\ndata: {"type":"message_stop"}')

        # And the trace holds exactly those frames -- not the OpenAI ones the
        # inner handler produced.
        (record,) = read_lines(tmp_path, "sess_e2e", "responses")
        assert record["status"] == "completed"
        assert record["response"]["body"] == response.text
        assert "chatcmpl-" not in record["response"]["body"]

        (request_record,) = read_lines(tmp_path, "sess_e2e", "requests")
        assert request_record["route"] == "/v1/messages"
        assert request_record["body"]["messages"][0]["content"] == "hi"
        assert request_record["trace_id"] == record["trace_id"]

    @pytest.mark.asyncio
    async def test_disabled_writer_leaves_the_stream_alone(self, tmp_path):
        client, server = self._client(tmp_path, self._openai_stream())
        server._request_trace = RequestTraceWriter(None)
        await server._request_trace.start()

        response = client.post(
            "/v1/messages",
            json={
                "model": "test-model",
                "max_tokens": 16,
                "stream": True,
                "messages": [{"role": "user", "content": "hi"}],
            },
        )

        assert response.status_code == 200
        assert '"text":"hello"' in response.text
        assert not list(tmp_path.iterdir())


class TestRejectedRequestEndToEnd:
    """A body FastAPI refuses never reaches a handler; the trace is its only copy."""

    @staticmethod
    def _client(tmp_path):
        from fastapi import FastAPI
        from fastapi.exceptions import RequestValidationError
        from fastapi.responses import JSONResponse
        from fastapi.testclient import TestClient

        from tensorrt_llm.serve.openai_protocol import ChatCompletionRequest

        writer = RequestTraceWriter(str(tmp_path))
        app = FastAPI()

        @app.exception_handler(RequestValidationError)
        async def handler(request, exc):
            # The same call openai_server makes at the top of its own handler.
            await writer.on_rejected(request, exc.errors())
            return JSONResponse(status_code=400, content={"error": str(exc)})

        @app.post("/v1/chat/completions")
        async def chat(request: ChatCompletionRequest):
            return {"ok": True}

        return TestClient(app), writer

    @pytest.mark.asyncio
    async def test_unknown_field_is_recorded_with_the_body(self, tmp_path):
        # extra="forbid" on ChatCompletionRequest: an agent client that starts
        # sending a new field gets every turn rejected, and this is where the
        # field name and the body it came in are visible.
        client, writer = self._client(tmp_path)
        await writer.start()

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "m",
                "messages": [{"role": "user", "content": "hi"}],
                "brand_new_field": {"nested": 1},
            },
            headers={"x-session-id": "sess_400"},
        )
        await writer.close()

        assert response.status_code == 400
        (record,) = read_lines(tmp_path, "sess_400", "requests")
        assert record["status"] == "rejected_400"
        assert record["route"] == "/v1/chat/completions"
        assert record["body"]["brand_new_field"] == {"nested": 1}
        assert record["validation_errors"][0]["loc"][-1] == "brand_new_field"
        assert record["validation_errors"][0]["type"] == "extra_forbidden"
        # No response line: the request never produced one.
        assert read_lines(tmp_path, "sess_400", "responses") == []

    @pytest.mark.asyncio
    async def test_malformed_json_body_is_kept_as_text(self, tmp_path):
        client, writer = self._client(tmp_path)
        await writer.start()

        response = client.post(
            "/v1/chat/completions",
            content=b'{"model": "m", "messages": [',
            headers={"content-type": "application/json"},
        )
        await writer.close()

        assert response.status_code == 400
        (record,) = read_lines(tmp_path, "_no_session", "requests")
        assert record["status"] == "rejected_400"
        assert record["body"] == '{"model": "m", "messages": ['
        assert record["body_parse_error"]


class TestDisaggIdHandoff:
    """``disagg_request_id`` reaches the handle across the handler split.

    On /v1/messages the id is minted inside the wrapped chat entry point, which
    by the ownership rule holds no handle, while the handle that gets written
    belongs to the outer Anthropic handler. The two are locals of different
    functions; the Request they share is the only thing both can see.
    """

    @staticmethod
    def _hooks(request, disagg_request_id):
        from types import SimpleNamespace

        return SimpleNamespace(raw_req=request, disagg_request_id=disagg_request_id)

    @pytest.mark.asyncio
    async def test_id_set_from_the_half_without_a_handle(self, tmp_path):
        from tensorrt_llm.serve.openai_disagg_server import _set_disagg_ids

        writer = RequestTraceWriter(str(tmp_path))
        await writer.start()
        request = FakeRequest(body={}, headers={"x-session-id": "s"}, path="/v1/messages")

        outer = await writer.on_request(request)  # anthropic_messages
        inner = await writer.on_request(request)  # the wrapped entry point
        assert inner is None  # the half that has hooks

        # What the wrapper does: it holds no handle, only hooks.
        _set_disagg_ids(self._hooks(request, 8812345))

        writer.on_response(outer, payload={"ok": True})
        await drain(writer)

        (record,) = read_lines(tmp_path, "s", "responses")
        assert record["disagg_request_id"] == 8812345
        assert record["client_id"] is None  # no engine on a proxy

    @pytest.mark.asyncio
    async def test_id_set_when_the_wrapper_owns_the_request(self, tmp_path):
        # /v1/completions and /v1/chat/completions route straight to the
        # wrapper, so it is the creator and the same code path applies.
        from tensorrt_llm.serve.openai_disagg_server import _set_disagg_ids

        writer = RequestTraceWriter(str(tmp_path))
        await writer.start()
        request = FakeRequest(body={}, headers={"x-session-id": "s"}, path="/v1/chat/completions")
        handle = await writer.on_request(request)

        _set_disagg_ids(self._hooks(request, 991))

        writer.on_response(handle, payload={"ok": True})
        await drain(writer)

        (record,) = read_lines(tmp_path, "s", "responses")
        assert record["disagg_request_id"] == 991

    def test_no_handle_is_a_no_op(self):
        # Tracing off: the request carries nothing and this must not raise.
        from tensorrt_llm.serve.openai_disagg_server import _set_disagg_ids

        _set_disagg_ids(self._hooks(FakeRequest(body={}), 7))

    @pytest.mark.asyncio
    async def test_none_id_leaves_the_field_null(self, tmp_path):
        # A deployment that never allocates one (or a failure before it does).
        from tensorrt_llm.serve.openai_disagg_server import _set_disagg_ids

        writer = RequestTraceWriter(str(tmp_path))
        await writer.start()
        request = FakeRequest(body={}, headers={"x-session-id": "s"})
        handle = await writer.on_request(request)

        _set_disagg_ids(self._hooks(request, None))

        writer.on_response(handle, payload={})
        await drain(writer)

        (record,) = read_lines(tmp_path, "s", "responses")
        assert record["disagg_request_id"] is None


class TestDisaggNonStreamingShape:
    """Same wrapper, two routes, two recorded shapes.

    On the disaggregated server ``_wrap_entry_point`` serves /v1/completions and
    /v1/chat/completions directly and is *also* called by anthropic_messages.
    It only ever holds the OpenAI-shaped response, so on the Anthropic route its
    own on_response is a no-op (handle is None) and the record has to be written
    after convert_chat_response, from the handler the client actually called.
    """

    OPENAI_BODY = {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 0,
        "model": "test-model",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "hello"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
    }

    @staticmethod
    def _server(tmp_path, openai_response):
        from types import SimpleNamespace
        from unittest.mock import Mock

        from tensorrt_llm.serve.openai_disagg_server import OpenAIDisaggServer

        server = object.__new__(OpenAIDisaggServer)
        server._request_trace = RequestTraceWriter(str(tmp_path))
        server._service = SimpleNamespace(openai_chat_completion=object())
        server._wrap_entry_point = Mock(return_value=AsyncMock(return_value=openai_response))
        return server

    @staticmethod
    def _anthropic_request():
        from tensorrt_llm.serve.anthropic_protocol import AnthropicMessagesRequest

        return AnthropicMessagesRequest(
            model="test-model", max_tokens=16, messages=[{"role": "user", "content": "hi"}]
        )

    @pytest.mark.asyncio
    async def test_anthropic_route_records_the_anthropic_shape(self, tmp_path):
        from fastapi.responses import JSONResponse

        server = self._server(tmp_path, JSONResponse(content=self.OPENAI_BODY))
        await server._request_trace.start()
        raw = FakeRequest(body={}, headers={"x-session-id": "s"}, path="/v1/messages")

        response = await server.anthropic_messages(self._anthropic_request(), raw)
        await server._request_trace.close()

        assert response.status_code == 200
        (record,) = read_lines(tmp_path, "s", "responses")
        body = record["response"]["body"]
        # Anthropic shape: content blocks and stop_reason...
        assert body["content"] == [{"type": "text", "text": "hello"}]
        assert body["stop_reason"] == "end_turn"
        # ...not the OpenAI one the wrapper handed back.
        assert "choices" not in body
        assert body["id"] != "chatcmpl-1"

    @pytest.mark.asyncio
    async def test_conversion_failure_keeps_the_upstream_body(self, tmp_path):
        from fastapi.responses import JSONResponse

        # convert_chat_response raises when a tool call's arguments are not a
        # JSON object; the client only gets a 500, so this is the only copy.
        broken = {
            **self.OPENAI_BODY,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "get_weather", "arguments": "not json at all"},
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
        }
        server = self._server(tmp_path, JSONResponse(content=broken))
        await server._request_trace.start()
        raw = FakeRequest(body={}, headers={"x-session-id": "s"}, path="/v1/messages")

        response = await server.anthropic_messages(self._anthropic_request(), raw)
        await server._request_trace.close()

        assert response.status_code == 500
        (record,) = read_lines(tmp_path, "s", "responses")
        assert record["status"] == "conversion_error"
        assert "not json at all" in record["response"]["body"]["upstream_body"]

    @pytest.mark.asyncio
    async def test_chat_route_records_the_openai_shape(self, tmp_path):
        # The other half of the contrast: the same wrapper, reached directly, is
        # the outermost handler and so records what *its* client receives.
        from types import SimpleNamespace

        from tensorrt_llm.serve.openai_protocol import ChatCompletionRequest, ChatCompletionResponse

        server = self._server(tmp_path, None)
        await server._request_trace.start()
        server._perf_metrics_collector = SimpleNamespace(
            total_requests=SimpleNamespace(inc=lambda: None),
            stream_requests=SimpleNamespace(inc=lambda: None),
            nonstream_requests=SimpleNamespace(inc=lambda: None),
            total_responses=SimpleNamespace(inc=lambda: None),
            queue_latency_seconds=SimpleNamespace(observe=lambda _v: None),
        )
        server._collect_perf_metrics = False
        server._allow_request_chat_template = True
        server._extract_conversation_id = lambda req, raw: None

        chat_response = ChatCompletionResponse(**self.OPENAI_BODY)
        entry_point = AsyncMock(return_value=chat_response)
        # The real factory this time, not the Mock the other tests install.
        from tensorrt_llm.serve.openai_disagg_server import OpenAIDisaggServer

        wrapper = OpenAIDisaggServer._wrap_entry_point(server, entry_point, ChatCompletionRequest)

        raw = FakeRequest(
            body={}, headers={"x-session-id": "s"}, path="/v1/chat/completions", arrival_time=1.0
        )
        req = ChatCompletionRequest(
            model="test-model", messages=[{"role": "user", "content": "hi"}]
        )
        await wrapper(req, raw)
        await server._request_trace.close()

        (record,) = read_lines(tmp_path, "s", "responses")
        body = record["response"]["body"]
        assert body["id"] == "chatcmpl-1"
        assert body["choices"][0]["message"]["content"] == "hello"
        assert "content" not in body or not isinstance(body.get("content"), list)
