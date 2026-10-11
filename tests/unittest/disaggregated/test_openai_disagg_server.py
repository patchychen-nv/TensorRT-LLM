# Copyright (c) 2026, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import asyncio
import gc
import json
import weakref
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import aiohttp
import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from starlette.datastructures import Headers

from tensorrt_llm.executor.utils import context_length_exceeded_message
from tensorrt_llm.llmapi.disagg_utils import ServerRole, extract_disagg_cfg
from tensorrt_llm.serve import openai_disagg_server
from tensorrt_llm.serve.anthropic_protocol import AnthropicMessagesRequest
from tensorrt_llm.serve.openai_disagg_server import (
    OpenAIDisaggServer,
    _DownstreamClientDisconnected,
)
from tensorrt_llm.serve.openai_protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatMessage,
    CompletionRequest,
    ConversationParams,
    DisaggregatedParams,
    ResponsesRequest,
    ResponsesResponse,
    UsageInfo,
)
from tensorrt_llm.serve.request_trace import RequestTraceWriter
from tensorrt_llm.serve.responses_utils import ServerArrivalTimeMiddleware

pytestmark = pytest.mark.cpu_only


def _raw_request(headers: dict[str, str]):
    return SimpleNamespace(headers=Headers(headers=headers))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("config_kwargs", "expected_timeout"),
    [({}, 10), ({"server_keep_alive_timeout": 3600}, 3600)],
)
async def test_server_keep_alive_timeout_is_passed_to_uvicorn(
    monkeypatch, config_kwargs, expected_timeout
):
    config = extract_disagg_cfg(
        context_servers={"num_instances": 0},
        generation_servers={"num_instances": 0},
        **config_kwargs,
    )
    server = object.__new__(OpenAIDisaggServer)
    server._config = config
    server.app = object()

    uvicorn_config = object()
    config_factory = Mock(return_value=uvicorn_config)
    uvicorn_server = SimpleNamespace(serve=AsyncMock())
    server_factory = Mock(return_value=uvicorn_server)
    monkeypatch.setattr(openai_disagg_server.uvicorn, "Config", config_factory)
    monkeypatch.setattr(openai_disagg_server.uvicorn, "Server", server_factory)

    await server(host="localhost", port=8000)

    assert config_factory.call_args.kwargs["timeout_keep_alive"] == expected_timeout
    server_factory.assert_called_once_with(uvicorn_config)
    uvicorn_server.serve.assert_awaited_once_with(sockets=None)


@pytest.mark.asyncio
async def test_http_cluster_storage_request_is_proxied_to_coordinator():
    payload = b'{"key":"worker","value":"ready"}'

    async def receive():
        return {"type": "http.request", "body": payload, "more_body": False}

    request = Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/set",
            "query_string": b"source=worker",
            "headers": [(b"content-type", b"application/json")],
        },
        receive,
    )
    server = OpenAIDisaggServer.__new__(OpenAIDisaggServer)
    server._coordinator = SimpleNamespace(
        proxy_cluster_storage_request=AsyncMock(
            return_value=(b'{"result":true}', 200, "application/json")
        )
    )

    response = await server._proxy_cluster_storage_request(request)

    server._coordinator.proxy_cluster_storage_request.assert_awaited_once_with(
        "POST", "/set", [("source", "worker")], payload, "application/json"
    )
    assert response.status_code == 200
    assert response.body == b'{"result":true}'


def test_create_client_does_not_register_with_server_metrics_collector():
    server = OpenAIDisaggServer.__new__(OpenAIDisaggServer)
    server._coordinator = SimpleNamespace(get_disagg_request_id=AsyncMock(return_value=1))
    server._req_timeout_secs = 30
    server._collect_perf_metrics = True
    server._config = SimpleNamespace(internal_request_auth_key="key")
    server._perf_metrics_collector = SimpleNamespace()

    with patch("tensorrt_llm.serve.openai_disagg_server.OpenAIHttpClient") as mock_client:
        client = server._create_client(SimpleNamespace(), ServerRole.GENERATION, max_retries=2)

    assert client is mock_client.return_value
    mock_client.assert_called_once()


def test_extract_conversation_id_from_headers():
    cases = [
        ({"X-Session-ID": "session-id"}, "session-id"),
        ({"X-Correlation-ID": "correlation-id"}, "correlation-id"),
        ({"x-session-affinity": "session-affinity"}, "session-affinity"),
        ({"x-multi-turn-session-id": "multi-turn-session-id"}, "multi-turn-session-id"),
        (
            {
                "X-Correlation-ID": "correlation-id",
                "X-Session-ID": "session-id",
                "x-session-affinity": "session-affinity",
                "x-multi-turn-session-id": "multi-turn-session-id",
            },
            "session-id",
        ),
        (
            {
                "x-session-affinity": "session-affinity",
                "x-multi-turn-session-id": "multi-turn-session-id",
            },
            "session-affinity",
        ),
        (
            {
                "X-Session-ID": "",
                "X-Correlation-ID": "correlation-id",
            },
            "correlation-id",
        ),
    ]

    for headers, expected_conversation_id in cases:
        request = CompletionRequest(model="test-model", prompt="hello")

        OpenAIDisaggServer._extract_conversation_id(request, _raw_request(headers))

        assert request.disaggregated_params is None
        assert request.conversation_params.conversation_id == expected_conversation_id


def test_extract_conversation_id_ignores_empty_headers():
    request = CompletionRequest(model="test-model", prompt="hello")

    OpenAIDisaggServer._extract_conversation_id(
        request,
        _raw_request(
            {
                "X-Session-ID": "",
                "X-Correlation-ID": " ",
                "x-session-affinity": "",
                "x-multi-turn-session-id": " ",
            }
        ),
    )

    assert request.disaggregated_params is None
    assert request.conversation_params is None


def test_extract_conversation_id_preserves_body_conversation_params():
    request = CompletionRequest(
        model="test-model",
        prompt="hello",
        conversation_params=ConversationParams(conversation_id="body-id"),
        disaggregated_params=DisaggregatedParams(request_type="context_only"),
    )

    OpenAIDisaggServer._extract_conversation_id(
        request,
        _raw_request({"X-Session-ID": "header-id"}),
    )

    assert request.conversation_params.conversation_id == "body-id"


def test_extract_conversation_id_populates_conversation_params_with_existing_disaggregated_params():
    request = CompletionRequest(
        model="test-model",
        prompt="hello",
        disaggregated_params=DisaggregatedParams(request_type="context_only"),
    )

    OpenAIDisaggServer._extract_conversation_id(
        request,
        _raw_request({"x-multi-turn-session-id": "multi-turn-session-id"}),
    )

    assert request.conversation_params.conversation_id == "multi-turn-session-id"


class TestClientDisconnectWatch:
    """A client that hangs up mid-pipeline must take the pipeline down with it.

    The pre-response window is the proxy's blind spot: the pinned
    fastapi/starlette request_response runs a handler to completion no matter
    what the socket does, so a context-first request whose client timed out
    kept its whole prefill grinding on the context worker -- and the retry
    storm stacked one giant prefill per abandoned attempt. These tests drive
    _serve_until_client_disconnect (and the wrapper around it) at the seam
    below FastAPI, where no engine and no worker are needed.
    """

    @staticmethod
    def _server():
        return OpenAIDisaggServer.__new__(OpenAIDisaggServer)

    @staticmethod
    def _raw(is_disconnected):
        return SimpleNamespace(is_disconnected=is_disconnected)

    @pytest.mark.asyncio
    async def test_disconnect_mid_pipeline_cancels_it_and_raises(self):
        """The disconnect cancels the entry task; its finalizers run first."""
        server = self._server()
        started = asyncio.Event()
        finalized = asyncio.Event()

        async def entry(req, hooks):
            started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                # Stands in for the pipeline's real cleanup: the client's
                # finalize (router settle + upstream socket close) runs under
                # this very cancellation, so it must be given the chance to.
                finalized.set()
                raise

        with pytest.raises(_DownstreamClientDisconnected):
            await server._serve_until_client_disconnect(
                entry, "req", "hooks", self._raw(AsyncMock(return_value=True))
            )

        assert started.is_set()
        # The settlement was awaited before the disconnect was reported.
        assert finalized.is_set()

    @pytest.mark.asyncio
    async def test_pipeline_completing_first_is_returned_unchanged(self):
        """No disconnect: the watch is invisible and the result flows through."""
        server = self._server()
        polls = AsyncMock(return_value=False)

        async def entry(req, hooks):
            return "response"

        result = await server._serve_until_client_disconnect(
            entry, "req", "hooks", self._raw(polls)
        )

        assert result == "response"

    @pytest.mark.asyncio
    async def test_pipeline_error_propagates_unchanged(self):
        """An entry failure keeps its meaning; the watch adds nothing to it."""
        server = self._server()

        async def entry(req, hooks):
            raise ValueError("upstream rejected")

        with pytest.raises(ValueError, match="upstream rejected"):
            await server._serve_until_client_disconnect(
                entry, "req", "hooks", self._raw(AsyncMock(return_value=False))
            )

    @pytest.mark.asyncio
    @pytest.mark.parametrize("watch", ["healthy", "broken"])
    async def test_a_failed_pipeline_is_freed_without_the_cyclic_gc(self, watch):
        """A failure must not leave a frame -> task -> error -> traceback cycle.

        The handler re-raises the entry task's error from its own frame, so that
        error's traceback holds the frame. A frame that still held the task --
        or asyncio.wait's set with the task in it -- would make the task, its
        error and the request reachable only through a cycle. The serving
        processes ran with the cyclic GC off, so each failed request leaked.
        Both ways the handler can re-raise are covered: straight from the
        finished task, and after a broken watch.
        """
        server = self._server()
        entry_tasks = []
        # A broken watch must fail first, so the handler takes its
        # serve-to-completion path and re-raises from `await entry_task`.
        delay = 0 if watch == "healthy" else 0.05

        async def entry(req, hooks):
            entry_tasks.append(weakref.ref(asyncio.current_task()))
            await asyncio.sleep(delay)
            raise ValueError("upstream rejected")

        polls = (
            AsyncMock(return_value=False)
            if watch == "healthy"
            else AsyncMock(side_effect=RuntimeError("watch broke"))
        )
        gc.collect()
        gc.disable()
        try:
            try:
                await server._serve_until_client_disconnect(entry, "req", "hooks", self._raw(polls))
            except ValueError:
                pass
            # The loop handle that woke this task after `await entry_task`
            # holds the task as its argument until the loop moves on.
            await asyncio.sleep(0)
            assert entry_tasks[0]() is None, "the failed entry task outlived its request"
        finally:
            gc.enable()

    @pytest.mark.asyncio
    async def test_cancel_losing_the_race_returns_the_finished_response(self):
        """A pipeline that completes during the cancel is a response, not an abort.

        Whatever it produced is handed back: uvicorn discards the send and a
        streaming body is torn down by StreamingResponse's own disconnect
        handling, both of which finalize -- reporting a disconnect instead
        would leave a never-consumed generator and its routing entry behind.
        """
        server = self._server()

        async def entry(req, hooks):
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                # The pipeline finished on its own while the cancel unwound --
                # e.g. the gen generator was already built and returned without
                # another suspension point.
                return "finished-during-unwind"

        result = await server._serve_until_client_disconnect(
            entry, "req", "hooks", self._raw(AsyncMock(return_value=True))
        )

        assert result == "finished-during-unwind"

    @pytest.mark.asyncio
    async def test_watch_failure_degrades_to_serving_not_cancelling(self):
        """A broken watcher says nothing about the client; the request survives."""
        server = self._server()

        async def entry(req, hooks):
            await asyncio.sleep(0)
            return "served"

        result = await server._serve_until_client_disconnect(
            entry, "req", "hooks", self._raw(AsyncMock(side_effect=RuntimeError("watch broke")))
        )

        assert result == "served"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("outcome", ["response", "rejection"])
    async def test_a_pipeline_settling_at_once_does_not_strand_the_handler(self, outcome):
        """Reaping the watch must survive Starlette's own cancel scope.

        A real Request's is_disconnected() reads receive() inside an anyio
        cancel scope that it cancels itself, and the watch is inside that read
        from its first poll. A pipeline that settles in its first step -- a 400
        raised before any upstream I/O, or any answer that needs no await --
        has the handler cancel the watch while that scope's cancellation is
        still in flight. The two merge into one CancelledError, the scope
        swallows it as its own, and the watch polls on until the client gives
        up and leaves: the handler sits on its answer all that time.
        """
        server = self._server()
        body_sent = False
        never = asyncio.Event()

        async def receive():
            nonlocal body_sent
            if not body_sent:
                body_sent = True
                return {"type": "http.request", "body": b"{}", "more_body": False}
            # What uvicorn does once the body is read: block until the client
            # leaves or the response completes.
            await never.wait()
            return {"type": "http.disconnect"}

        raw_req = Request(
            {"type": "http", "method": "POST", "path": "/v1/chat/completions", "headers": []},
            receive,
        )
        # FastAPI reads the body before the handler runs.
        await raw_req.body()

        async def entry(req, hooks):
            if outcome == "rejection":
                raise ValueError("rejected before any upstream I/O")
            return "response"

        serving = asyncio.create_task(
            server._serve_until_client_disconnect(entry, "req", "hooks", raw_req)
        )
        # The watch polls once a second; a handler still waiting after two is
        # waiting on a watch nothing will stop.
        done, _ = await asyncio.wait({serving}, timeout=2)
        if not done:
            serving.cancel()
            await asyncio.wait({serving})
        assert serving in done, "the handler is still waiting on the disconnect watch"
        if outcome == "rejection":
            with pytest.raises(ValueError, match="rejected before any upstream I/O"):
                serving.result()
        else:
            assert serving.result() == "response"

    @pytest.mark.asyncio
    async def test_wrapper_answers_a_disconnect_with_499_and_traces_it(self):
        """End of the proxy's story: a finalized abort, an honest status, a trace terminal."""
        server = self._server()
        counter = lambda: SimpleNamespace(inc=Mock())  # noqa: E731
        server._perf_metrics_collector = SimpleNamespace(
            total_requests=counter(),
            stream_requests=counter(),
            nonstream_requests=counter(),
            total_responses=counter(),
            queue_latency_seconds=SimpleNamespace(observe=Mock()),
        )
        server._request_trace = SimpleNamespace(
            on_request=AsyncMock(return_value=None),
            on_response=Mock(),
            wrap_stream=lambda stream, handle: stream,
        )
        server._allow_request_chat_template = False
        server._collect_perf_metrics = False

        cancelled = asyncio.Event()

        async def entry(req, hooks):
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelled.set()
                raise

        raw_req = SimpleNamespace(
            state=SimpleNamespace(server_arrival_time=0.0),
            headers=Headers({}),
            is_disconnected=AsyncMock(return_value=True),
            client=("10.0.0.1", 40000),
        )
        request = CompletionRequest(model="m", prompt="hi", stream=False)

        wrapper = server._wrap_entry_point(entry, CompletionRequest)
        response = await wrapper(request, raw_req)

        assert cancelled.is_set()
        assert response.status_code == 499
        server._request_trace.on_response.assert_called_once_with(
            None, payload=None, status="client_disconnected"
        )


def test_disagg_config_allows_request_chat_template_opt_in():
    config = extract_disagg_cfg(
        context_servers={"num_instances": 0},
        generation_servers={"num_instances": 0},
        allow_request_chat_template=True,
    )

    assert config.allow_request_chat_template is True


@pytest.mark.parametrize("value", ["false", "true", 0, 1, None])
def test_disagg_config_rejects_non_bool_request_chat_template_opt_in(value):
    with pytest.raises(ValueError, match="allow_request_chat_template must be a boolean"):
        extract_disagg_cfg(
            context_servers={"num_instances": 0},
            generation_servers={"num_instances": 0},
            allow_request_chat_template=value,
        )


# ---------------------------------------------------------------------------
# Trace terminals: every accepted request ends in exactly one response line
# ---------------------------------------------------------------------------

_CHAT_ROUTE = "/v1/chat/completions"
_MESSAGES_ROUTE = "/v1/messages"
_RESPONSES_ROUTE = "/v1/responses"


def _upstream_error(status: int, message: str) -> aiohttp.ClientResponseError:
    """The error OpenAIHttpClient raises when a worker answers with an HTTP error.

    Built the way the client builds it: the reason phrase, then the worker's
    own error body.
    """
    body = json.dumps({"object": "error", "message": message, "code": status})
    return aiohttp.ClientResponseError(Mock(), (), status=status, message=f"Error: {body}")


def _route_body(route: str, stream: bool = False, **overrides) -> dict:
    body = {"model": "m", "messages": [{"role": "user", "content": "hello"}], "stream": stream}
    if route == _MESSAGES_ROUTE:
        body["max_tokens"] = 16
    body.update(overrides)
    return body


def _chat_completion() -> ChatCompletionResponse:
    return ChatCompletionResponse(
        model="m",
        choices=[
            ChatCompletionResponseChoice(
                index=0,
                message=ChatMessage(role="assistant", content="hi"),
                finish_reason="stop",
            )
        ],
        usage=UsageInfo(prompt_tokens=1, completion_tokens=1, total_tokens=2),
    )


def _traced_server(entry_point):
    """A disagg front end over a fake pipeline, its trace captured in a list.

    The writer is real -- on_request/on_response run their own logic, the
    exactly-once guard included -- with only the queue handoff replaced, so the
    records can be asserted without a drain task or a trace directory.
    """
    server = OpenAIDisaggServer.__new__(OpenAIDisaggServer)
    counter = lambda: SimpleNamespace(inc=Mock())  # noqa: E731
    server._perf_metrics_collector = SimpleNamespace(
        total_requests=counter(),
        stream_requests=counter(),
        nonstream_requests=counter(),
        total_responses=counter(),
        http_exceptions=counter(),
        internal_errors=counter(),
        queue_latency_seconds=SimpleNamespace(observe=Mock()),
    )
    server._allow_request_chat_template = False
    server._collect_perf_metrics = False
    # /v1/messages forwards into the chat pipeline through this attribute.
    server._service = SimpleNamespace(openai_chat_completion=entry_point)
    writer = RequestTraceWriter("unused-trace-dir")
    writer._task = object()  # enabled, without a running drain task
    records = []
    writer._submit = lambda bucket, kind, record: records.append((kind, record))
    server._request_trace = writer
    return server, records


def _route_client(server, entry_point, route=_CHAT_ROUTE, request_type=ChatCompletionRequest):
    app = FastAPI()
    # RawRequestResponseHooks reads the arrival stamp this middleware sets.
    app.add_middleware(ServerArrivalTimeMiddleware)
    app.add_api_route(route, server._wrap_entry_point(entry_point, request_type), methods=["POST"])
    app.add_api_route(_MESSAGES_ROUTE, server.anthropic_messages, methods=["POST"])
    return TestClient(app)


def _only_terminal(records, status: str) -> dict:
    """Assert one accepted request line and exactly one terminal line, joined."""
    requests = [record for kind, record in records if kind == "requests"]
    terminals = [record for kind, record in records if kind == "responses"]
    assert [record["status"] for record in requests] == ["accepted"]
    assert [record["status"] for record in terminals] == [status]
    # The two lines must join, or the terminal explains nothing.
    assert terminals[0]["trace_id"] == requests[0]["trace_id"]
    return terminals[0]


class TestEveryAcceptedRequestEndsInOneTraceTerminal:
    """The trace writes ``accepted`` at handler entry; every exit must close it.

    The front end is the only process writing traces in a disaggregated
    deployment, and the exits that left without a terminal line were exactly
    the failures: a worker's 4xx/5xx, an internal 500, a chat-template
    rejection, a shutdown cancel -- and on /v1/messages also the conversion
    rejection and the pre-response hangup. Each of those read in the trace as a
    request still in flight. The client-disconnect exit already wrote its
    terminal; these hold every other exit to the same standard, through both
    the OpenAI wrapper and the Anthropic adapter that forwards into it.
    """

    _FAILURES = {
        # A worker's verdict on the request itself (a context overflow).
        "upstream_400": (
            lambda: _upstream_error(400, "prompt is too long"),
            400,
            "rejected_400",
            "prompt is too long",
        ),
        "upstream_503": (
            lambda: _upstream_error(503, "engine overloaded"),
            503,
            "error",
            "engine overloaded",
        ),
        # A context overflow keeps its machine-readable code, so the wrapper
        # answers with the worker's error envelope instead of raising.
        "upstream_context_length": (
            lambda: _upstream_error(400, context_length_exceeded_message(8, 9)),
            400,
            "rejected_400",
            "maximum context length is 8 tokens",
        ),
        # No worker involved: the orchestrator itself failed.
        "internal": (
            lambda: RuntimeError("Cluster is not ready"),
            500,
            "error",
            "Cluster is not ready",
        ),
    }

    @pytest.mark.parametrize("failure", sorted(_FAILURES))
    @pytest.mark.parametrize("stream", [False, True])
    @pytest.mark.parametrize("route", [_CHAT_ROUTE, _MESSAGES_ROUTE])
    def test_failure_before_streaming_is_recorded_once(self, route, stream, failure):
        make_error, http_status, trace_status, message = self._FAILURES[failure]

        async def entry(req, hooks):
            # The orchestrator allocates the id before it reaches a worker.
            hooks.on_disagg_request_id(77)
            raise make_error()

        server, records = _traced_server(entry)

        response = _route_client(server, entry).post(route, json=_route_body(route, stream))

        assert response.status_code == http_status
        terminal = _only_terminal(records, trace_status)
        # The record holds the body the client received, not a summary of it.
        assert terminal["response"] == {"kind": "json", "body": response.json()}
        assert message in json.dumps(terminal["response"]["body"])
        # The HTTP status survives where the trace status collapses 5xx into
        # "error": a worker's 503 and the orchestrator's own 500 stay apart.
        assert terminal["http_status"] == http_status
        # What joins the record to the workers' logs of the failed attempt.
        assert terminal["disagg_request_id"] == 77

    def test_chat_template_rejection_is_recorded_once(self):
        entry = AsyncMock()
        server, records = _traced_server(entry)

        response = _route_client(server, entry).post(
            _CHAT_ROUTE, json=_route_body(_CHAT_ROUTE, chat_template="{{ messages }}")
        )

        assert response.status_code == 400
        entry.assert_not_called()
        terminal = _only_terminal(records, "rejected_400")
        assert terminal["response"]["body"] == response.json()
        assert "chat_template" in terminal["response"]["body"]["detail"]
        assert terminal["http_status"] == 400

    def test_anthropic_conversion_rejection_is_recorded_once(self):
        entry = AsyncMock()
        server, records = _traced_server(entry)
        document = {
            "type": "document",
            "source": {"type": "base64", "data": "QUJD", "media_type": "application/pdf"},
        }

        response = _route_client(server, entry).post(
            _MESSAGES_ROUTE,
            json=_route_body(_MESSAGES_ROUTE, messages=[{"role": "user", "content": [document]}]),
        )

        assert response.status_code == 400
        entry.assert_not_called()
        terminal = _only_terminal(records, "rejected_400")
        assert terminal["response"]["body"] == response.json()
        assert "not supported" in terminal["response"]["body"]["error"]["message"]
        assert terminal["http_status"] == 400

    def test_anthropic_pre_response_disconnect_is_recorded_once(self):
        """The wrapped chat entry point answers 499 but owns no trace handle.

        The adapter holds the handle, so the adapter has to record the
        hangup -- under the same status the OpenAI routes use for it.
        """

        async def entry(req, hooks):
            await asyncio.Event().wait()

        server, records = _traced_server(entry)
        # The client is already gone the first time the watch looks.
        server._watch_client_disconnect = AsyncMock(return_value=None)

        response = _route_client(server, entry).post(
            _MESSAGES_ROUTE, json=_route_body(_MESSAGES_ROUTE)
        )

        assert response.status_code == 499
        _only_terminal(records, "client_disconnected")

    def test_an_executor_error_is_recorded_as_the_null_200_it_answers(self):
        """The executor-error exit signals shutdown and returns nothing.

        FastAPI answers that with a 200 and a JSON null; the record says so,
        with the exception kept beside the body that does not mention it.
        """

        class ExecutorError(RuntimeError):
            pass

        async def entry(req, hooks):
            raise ExecutorError("executor died")

        server, records = _traced_server(entry)
        with (
            patch.object(openai_disagg_server, "CppExecutorError", ExecutorError),
            patch.object(openai_disagg_server.signal, "raise_signal") as raise_signal,
        ):
            response = _route_client(server, entry).post(_CHAT_ROUTE, json=_route_body(_CHAT_ROUTE))

        raise_signal.assert_called_once()
        assert response.status_code == 200
        assert response.json() is None
        terminal = _only_terminal(records, "error")
        assert terminal["response"] == {"kind": "json", "body": None}
        assert terminal["http_status"] == 200
        assert terminal["termination"] == {
            "cause": "internal_error",
            "detail": "ExecutorError: executor died",
        }

    @pytest.mark.asyncio
    async def test_an_unanticipated_anthropic_failure_is_recorded_as_the_plain_500(self):
        """An exception nobody handles is answered by the framework's text 500."""
        server, records = _traced_server(AsyncMock())
        server._serve_anthropic_messages = AsyncMock(side_effect=RuntimeError("boom"))
        body = _route_body(_MESSAGES_ROUTE)
        raw_req = SimpleNamespace(
            state=SimpleNamespace(server_arrival_time=0.0),
            headers=Headers({}),
            url=SimpleNamespace(path=_MESSAGES_ROUTE),
            json=AsyncMock(return_value=body),
            client=("10.0.0.1", 40000),
        )

        with pytest.raises(RuntimeError, match="boom"):
            await server.anthropic_messages(AnthropicMessagesRequest(**body), raw_req)

        terminal = _only_terminal(records, "error")
        assert terminal["response"] == {"kind": "text", "body": "Internal Server Error"}
        assert terminal["http_status"] == 500
        assert terminal["termination"] == {
            "cause": "internal_error",
            "detail": "RuntimeError: boom",
        }

    @pytest.mark.asyncio
    @pytest.mark.parametrize("route", [_CHAT_ROUTE, _MESSAGES_ROUTE])
    async def test_shutdown_cancel_is_recorded_once(self, route):
        """A handler cancelled mid-pipeline (shutdown) still closes its trace."""
        entered = asyncio.Event()

        async def entry(req, hooks):
            hooks.on_disagg_request_id(77)
            entered.set()
            await asyncio.Event().wait()

        server, records = _traced_server(entry)
        body = _route_body(route)
        raw_req = SimpleNamespace(
            state=SimpleNamespace(server_arrival_time=0.0),
            headers=Headers({}),
            url=SimpleNamespace(path=route),
            json=AsyncMock(return_value=body),
            is_disconnected=AsyncMock(return_value=False),
            client=("10.0.0.1", 40000),
        )
        if route == _CHAT_ROUTE:
            wrapper = server._wrap_entry_point(entry, ChatCompletionRequest)
            handler = wrapper(ChatCompletionRequest(**body), raw_req)
        else:
            handler = server.anthropic_messages(AnthropicMessagesRequest(**body), raw_req)

        task = asyncio.create_task(handler)
        await asyncio.wait_for(entered.wait(), timeout=5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        terminal = _only_terminal(records, "cancelled")
        assert terminal["disagg_request_id"] == 77

    @pytest.mark.parametrize("route", [_CHAT_ROUTE, _MESSAGES_ROUTE])
    def test_success_is_still_recorded_once(self, route):
        async def entry(req, hooks):
            return _chat_completion()

        server, records = _traced_server(entry)

        response = _route_client(server, entry).post(route, json=_route_body(route))

        assert response.status_code == 200
        _only_terminal(records, "completed")

    def test_stream_is_still_recorded_once_by_the_stream_wrapper(self):
        async def entry(req, hooks):
            async def body():
                yield b'data: {"choices":[]}\n\n'
                yield b"data: [DONE]\n\n"

            return body()

        server, records = _traced_server(entry)

        response = _route_client(server, entry).post(
            _CHAT_ROUTE, json=_route_body(_CHAT_ROUTE, stream=True)
        )

        assert response.status_code == 200
        terminal = _only_terminal(records, "completed")
        assert terminal["response"]["kind"] == "sse_text"


class TestStreamedResponsesContextOverflow:
    """A streamed Responses request too long for the context fails in-stream.

    OpenAI answers one with ``response.failed`` whose ``error.code`` is
    ``context_length_exceeded``, and clients act on that event: Codex reads it
    as a full context window and compacts before its next turn, while a 400
    reads as an invalid request and is not recovered from.
    """

    @staticmethod
    def _post(stream: bool):
        async def entry(req, hooks):
            hooks.on_disagg_request_id(77)
            raise _upstream_error(400, context_length_exceeded_message(8, 9))

        server, records = _traced_server(entry)
        client = _route_client(server, entry, route=_RESPONSES_ROUTE, request_type=ResponsesRequest)
        response = client.post(
            _RESPONSES_ROUTE, json={"model": "m", "input": "hello", "stream": stream}
        )
        return response, records

    def test_the_stream_ends_in_response_failed_with_the_context_length_code(self):
        response, records = self._post(stream=True)

        assert response.status_code == 200
        events = [
            json.loads(line[len("data: ") :])
            for line in response.text.splitlines()
            if line.startswith("data: ")
        ]
        assert [event["type"] for event in events] == [
            "response.created",
            "response.in_progress",
            "response.failed",
        ]
        failed = events[-1]["response"]
        assert failed["error"]["code"] == "context_length_exceeded"
        assert "maximum context length is 8 tokens" in failed["error"]["message"]
        # One terminal line, saying the generation failed and why.
        terminal = _only_terminal(records, "completed")
        assert terminal["response_status"] == "failed"
        assert terminal["error_code"] == "context_length_exceeded"
        assert terminal["disagg_request_id"] == 77

    def test_a_non_streamed_request_keeps_the_400(self):
        response, records = self._post(stream=False)

        assert response.status_code == 400
        assert response.json()["code"] == "context_length_exceeded"
        _only_terminal(records, "rejected_400")


def test_json_responses_body_keeps_its_wire_field_names():
    """`schema` must not reach the client spelled `schema_`.

    pydantic cannot hold a field called `schema`, so the json_schema text
    format declares `schema_` with `schema` as its alias. Re-serialising the
    worker's response without `by_alias` hands the client a field name the API
    does not have, and the trace records the same wrong body.
    """
    response_object = ResponsesResponse(
        model="m",
        output=[],
        parallel_tool_calls=False,
        temperature=1.0,
        tool_choice="auto",
        tools=[],
        top_p=1.0,
        background=False,
        service_tier="auto",
        status="completed",
        top_logprobs=0,
        truncation="disabled",
        text={
            "format": {
                "type": "json_schema",
                "name": "structured_output",
                "schema": {"type": "object"},
                "strict": True,
            }
        },
    )

    async def entry(req, hooks):
        return response_object

    server, records = _traced_server(entry)
    client = _route_client(server, entry, route=_RESPONSES_ROUTE, request_type=ResponsesRequest)

    response = client.post(_RESPONSES_ROUTE, json={"model": "m", "input": "hi"})

    assert response.status_code == 200
    text_format = response.json()["text"]["format"]
    assert text_format.get("schema") == {"type": "object"}
    assert "schema_" not in text_format
    terminal = _only_terminal(records, "completed")
    assert terminal["response"]["body"]["text"]["format"].get("schema") == {"type": "object"}
