# Copyright (c) 2025-2026, NVIDIA CORPORATION.
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
import json
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import aiohttp
import pytest

from tensorrt_llm.llmapi.disagg_utils import ServerRole
from tensorrt_llm.serve.disagg_auth import INTERNAL_DISAGG_AUTH_HEADER
from tensorrt_llm.serve.openai_client import OpenAIHttpClient
from tensorrt_llm.serve.openai_protocol import (  # noqa: F401
    CompletionRequest,
    CompletionResponse,
    CompletionResponseChoice,
    DisaggregatedParams,
    ResponsesRequest,
    UsageInfo,
)
from tensorrt_llm.serve.perf_metrics import _PERF_METRICS_HEADER_BUDGET_BYTES, SSE_METRICS_EVENT
from tensorrt_llm.serve.responses_utils import ResponseHooks
from tensorrt_llm.serve.router import LoadBalancingRouter, Router

pytestmark = pytest.mark.cpu_only


def _reset_prometheus_registry():
    from prometheus_client.registry import REGISTRY

    REGISTRY._names_to_collectors = {}
    REGISTRY._collector_to_names = {}


@pytest.fixture
def mock_router():
    """Create a mock router."""
    router = AsyncMock(spec=Router)
    router.servers = ["localhost:8000", "localhost:8001"]
    router.get_next_server = AsyncMock(return_value=("localhost:8000", None))
    router.finish_request = AsyncMock()
    return router


@pytest.fixture
def mock_session():
    """Create a mock aiohttp session."""
    return AsyncMock(spec=aiohttp.ClientSession)


@pytest.fixture
def openai_client(mock_router, mock_session):
    """Create an OpenAIHttpClient instance."""
    # uninitialize the prometheus metrics collector or it will raise a duplicate metric error
    _reset_prometheus_registry()
    return OpenAIHttpClient(
        router=mock_router,
        role=ServerRole.CONTEXT,
        timeout_secs=180,
        max_retries=2,
        retry_interval_sec=1,
        session=mock_session,
    )


@pytest.fixture
def completion_request():
    """Create a sample non-streaming CompletionRequest."""
    return CompletionRequest(
        model="test-model",
        prompt="Hello, world!",
        stream=False,
        disaggregated_params=DisaggregatedParams(
            request_type="generation_only", first_gen_tokens=[123], ctx_request_id=123
        ),
    )


@pytest.fixture
def streaming_completion_request():
    """Create a sample streaming CompletionRequest."""
    return CompletionRequest(
        model="test-model",
        prompt="Hello, world!",
        stream=True,
        disaggregated_params=DisaggregatedParams(
            request_type="generation_only", first_gen_tokens=[456], ctx_request_id=456
        ),
    )


class TestOpenAIHttpClient:
    """Test OpenAIHttpClient main functionality."""

    def dummy_response(self):
        return CompletionResponse(
            id="test-123",
            object="text_completion",
            created=1234567890,
            model="test-model",
            usage=UsageInfo(prompt_tokens=10, completion_tokens=10),
            choices=[CompletionResponseChoice(index=0, text="Hello!")],
        )

    def test_initialization(self, mock_router, mock_session):
        """Test client initialization."""
        client = OpenAIHttpClient(
            router=mock_router,
            role=ServerRole.GENERATION,
            timeout_secs=300,
            max_retries=5,
            session=mock_session,
        )
        assert client._router == mock_router
        assert client._role == ServerRole.GENERATION
        assert client._session == mock_session
        assert client._max_retries == 5

    @pytest.mark.asyncio
    async def test_internal_client_accepts_perf_metrics_header_size(self, mock_router):
        with (
            patch("tensorrt_llm.serve.openai_client.ClientMetricsCollector"),
            patch("tensorrt_llm.serve.openai_client.aiohttp.ClientSession") as session,
        ):
            OpenAIHttpClient(router=mock_router, role=ServerRole.GENERATION)

        assert session.call_args.kwargs["max_field_size"] == _PERF_METRICS_HEADER_BUDGET_BYTES

    @pytest.mark.asyncio
    async def test_generation_request_with_opaque_state_is_signed(self, mock_router, mock_session):
        """Opaque state forwarded to generation workers gets internal auth."""
        _reset_prometheus_registry()
        client = OpenAIHttpClient(
            router=mock_router,
            role=ServerRole.GENERATION,
            timeout_secs=300,
            max_retries=0,
            session=mock_session,
            internal_disagg_auth_key="secret",
        )
        mock_response = self.dummy_response()
        mock_http_response = AsyncMock()
        mock_http_response.status = 200
        mock_http_response.headers = {"Content-Type": "application/json"}
        mock_http_response.json = AsyncMock(return_value=mock_response.model_dump())
        mock_http_response.__aenter__ = AsyncMock(return_value=mock_http_response)
        mock_http_response.__aexit__ = AsyncMock()
        mock_session.post.return_value = mock_http_response

        request = CompletionRequest(
            model="test-model",
            prompt="Hello, world!",
            stream=False,
            disaggregated_params=DisaggregatedParams(
                request_type="generation_only",
                encoded_opaque_state="b3BhcXVl",
            ),
        )

        await client.send_request(request)

        headers = mock_session.post.call_args.kwargs["headers"]
        assert headers[INTERNAL_DISAGG_AUTH_HEADER].startswith("sha256=")

    @pytest.mark.asyncio
    async def test_generation_request_with_opaque_state_without_key_warns(
        self, mock_router, mock_session
    ):
        """Opaque state without auth key emits a transitional warning."""
        _reset_prometheus_registry()
        client = OpenAIHttpClient(
            router=mock_router,
            role=ServerRole.GENERATION,
            timeout_secs=300,
            max_retries=0,
            session=mock_session,
        )
        mock_response = self.dummy_response()
        mock_http_response = AsyncMock()
        mock_http_response.status = 200
        mock_http_response.headers = {"Content-Type": "application/json"}
        mock_http_response.json = AsyncMock(return_value=mock_response.model_dump())
        mock_http_response.__aenter__ = AsyncMock(return_value=mock_http_response)
        mock_http_response.__aexit__ = AsyncMock()
        mock_session.post.return_value = mock_http_response
        request = CompletionRequest(
            model="test-model",
            prompt="Hello, world!",
            stream=False,
            disaggregated_params=DisaggregatedParams(
                request_type="generation_only",
                encoded_opaque_state="b3BhcXVl",
            ),
        )

        warning_message = "In a future release the requirement to use internal_request_auth_key"
        with pytest.warns(FutureWarning, match=warning_message):
            await client.send_request(request)

        headers = mock_session.post.call_args.kwargs["headers"]
        assert INTERNAL_DISAGG_AUTH_HEADER not in headers

    @pytest.mark.asyncio
    async def test_non_streaming_completion_request(
        self, openai_client, completion_request, mock_session, mock_router
    ):
        """Test non-streaming completion request end-to-end."""
        mock_response = self.dummy_response()

        # Mock HTTP response
        mock_http_response = AsyncMock()
        mock_http_response.status = 200
        mock_http_response.headers = {"Content-Type": "application/json"}
        mock_http_response.json = AsyncMock(return_value=mock_response.model_dump())
        mock_http_response.raise_for_status = Mock()
        mock_http_response.__aenter__ = AsyncMock(return_value=mock_http_response)
        mock_http_response.__aexit__ = AsyncMock()

        mock_session.post.return_value = mock_http_response

        # Send request
        response = await openai_client.send_request(completion_request)

        # Assertions
        assert isinstance(response, CompletionResponse)
        assert response.model == "test-model"
        mock_session.post.assert_called_once()
        mock_router.finish_request.assert_called_once_with(
            completion_request, mock_session, success=True
        )

    @pytest.mark.asyncio
    async def test_streaming_completion_request(
        self, openai_client, streaming_completion_request, mock_session, mock_router
    ):
        """Test streaming completion request end-to-end."""
        # Mock HTTP streaming response
        mock_http_response = AsyncMock()
        mock_http_response.status = 200
        mock_http_response.headers = {"Content-Type": "text/event-stream"}

        dummy_data = [
            b'data: "Hello"\n\n',
            b'data: "world"\n\n',
            b'data: "!"\n\n',
        ]

        async def mock_iter_any():
            for data in dummy_data:
                yield data

        mock_http_response.content = AsyncMock()
        mock_http_response.content.iter_any = mock_iter_any
        mock_http_response.__aenter__ = AsyncMock(return_value=mock_http_response)
        mock_http_response.__aexit__ = AsyncMock()

        mock_session.post.return_value = mock_http_response

        # Send streaming request
        response_generator = await openai_client.send_request(streaming_completion_request)

        # Consume the generator
        chunks = []
        async for chunk in response_generator:
            chunks.append(chunk)

        # Assertions
        assert len(chunks) == 3
        for i, chunk in enumerate(chunks):
            assert chunk == dummy_data[i]
        mock_session.post.assert_called_once()
        mock_router.finish_request.assert_called_once_with(
            streaming_completion_request, mock_session, success=True
        )

    @pytest.mark.asyncio
    async def test_streaming_perf_metrics_preserve_sse_event_boundaries(
        self, openai_client, streaming_completion_request, mock_session, mock_router
    ):
        openai_client._request_perf_metrics = True
        mock_http_response = AsyncMock()
        mock_http_response.status = 200
        mock_http_response.headers = {"Content-Type": "text/event-stream"}

        response_data = b'data: {"choices":[{"text":"Hello"}]}\n\n'
        done_data = b"data: [DONE]\n\n"
        metrics_data = (
            f'event: {SSE_METRICS_EVENT}\ndata: {{"Server-Timing":"server_ttft;dur=1.0"}}\n\n'
        ).encode()
        marker_split = len("event: trtllm")

        async def mock_iter_any():
            yield response_data
            yield done_data + metrics_data[:marker_split]
            yield metrics_data[marker_split:]

        mock_http_response.content = AsyncMock()
        mock_http_response.content.iter_any = mock_iter_any
        mock_http_response.__aenter__ = AsyncMock(return_value=mock_http_response)
        mock_http_response.__aexit__ = AsyncMock()
        mock_session.post.return_value = mock_http_response
        hooks = MagicMock(spec=ResponseHooks)

        response_generator = await openai_client.send_request(
            streaming_completion_request, hooks=hooks
        )
        chunks = [chunk async for chunk in response_generator]

        assert chunks == [response_data, done_data]
        hooks.on_first_token.assert_called_once_with("localhost:8000", streaming_completion_request)
        hooks.on_perf_metrics.assert_called_once()
        hooks.on_resp_done.assert_called_once_with(
            "localhost:8000", streaming_completion_request, None
        )
        mock_router.finish_request.assert_called_once_with(
            streaming_completion_request, mock_session, success=True
        )

    @pytest.mark.asyncio
    async def test_malformed_streaming_metrics_do_not_fail_request(
        self, openai_client, streaming_completion_request, mock_session, mock_router
    ):
        openai_client._request_perf_metrics = True
        mock_http_response = AsyncMock()
        mock_http_response.status = 200
        mock_http_response.headers = {"Content-Type": "text/event-stream"}

        response_data = b'data: "Hello"\n\ndata: [DONE]\n\n'
        metrics_data = f"event: {SSE_METRICS_EVENT}\ndata: not-json\n\n".encode()

        async def mock_iter_any():
            yield b""
            yield response_data
            yield metrics_data

        mock_http_response.content = AsyncMock()
        mock_http_response.content.iter_any = mock_iter_any
        mock_http_response.__aenter__ = AsyncMock(return_value=mock_http_response)
        mock_http_response.__aexit__ = AsyncMock()
        mock_session.post.return_value = mock_http_response
        hooks = MagicMock(spec=ResponseHooks)

        response_generator = await openai_client.send_request(
            streaming_completion_request, hooks=hooks
        )
        chunks = [chunk async for chunk in response_generator]

        assert b"".join(chunks) == response_data
        hooks.on_first_token.assert_called_once_with("localhost:8000", streaming_completion_request)
        hooks.on_perf_metrics.assert_not_called()
        hooks.on_resp_done.assert_called_once_with(
            "localhost:8000", streaming_completion_request, None
        )
        mock_router.finish_request.assert_called_once_with(
            streaming_completion_request, mock_session, success=True
        )

    @pytest.mark.asyncio
    async def test_request_with_custom_server(
        self, openai_client, completion_request, mock_session, mock_router
    ):
        """Test sending request to a specific server."""
        custom_server = "localhost:9000"
        mock_response = self.dummy_response()

        mock_http_response = AsyncMock()
        mock_http_response.status = 200
        mock_http_response.headers = {"Content-Type": "application/json"}
        mock_http_response.json = AsyncMock(return_value=mock_response.model_dump())
        mock_http_response.__aenter__ = AsyncMock(return_value=mock_http_response)
        mock_http_response.__aexit__ = AsyncMock()

        mock_session.post.return_value = mock_http_response

        await openai_client.send_request(completion_request, server=custom_server)

        # Verify custom server was used in URL
        call_args = mock_session.post.call_args[0][0]
        assert custom_server in call_args
        # Router should not be called when server is specified
        mock_router.get_next_server.assert_not_called()

    @pytest.mark.asyncio
    async def test_request_error_handling(
        self, openai_client, completion_request, mock_session, mock_router
    ):
        """Test error handling when request fails."""
        mock_session.post.side_effect = aiohttp.ClientError("Connection failed")

        with pytest.raises(aiohttp.ClientError):
            await openai_client.send_request(completion_request)

        # Should finish request on error with success=False so the router
        # doesn't record routed-block cache state for a request that didn't complete.
        mock_router.finish_request.assert_called_once_with(
            completion_request, mock_session, success=False
        )

    @pytest.mark.asyncio
    async def test_request_with_retry(
        self, openai_client, completion_request, mock_session, mock_router
    ):
        """Test retry mechanism on transient failures."""
        mock_response = self.dummy_response()

        mock_http_response = AsyncMock()
        mock_http_response.status = 200
        mock_http_response.headers = {"Content-Type": "application/json"}
        mock_http_response.json = AsyncMock(return_value=mock_response.model_dump())
        mock_http_response.__aenter__ = AsyncMock(return_value=mock_http_response)
        mock_http_response.__aexit__ = AsyncMock()

        # First attempt fails, second succeeds
        mock_session.post.side_effect = [
            aiohttp.ClientError("Temporary failure"),
            mock_http_response,
        ]

        with patch("asyncio.sleep", new_callable=AsyncMock):
            response = await openai_client.send_request(completion_request)

        assert isinstance(response, CompletionResponse)
        assert mock_session.post.call_count == 2  # Initial + 1 retry

    @pytest.mark.asyncio
    async def test_max_retries_exceeded(
        self, openai_client, completion_request, mock_session, mock_router
    ):
        """Test that request fails after max retries."""
        mock_session.post.side_effect = aiohttp.ClientError("Connection failed")

        with patch("asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(aiohttp.ClientError):
                await openai_client.send_request(completion_request)

        # Should try max_retries + 1 times
        assert mock_session.post.call_count == openai_client._max_retries + 1
        mock_router.finish_request.assert_called_once()

    @pytest.mark.asyncio
    async def test_invalid_request_type(self, openai_client):
        """Test handling of invalid request type."""
        with pytest.raises(ValueError, match="Invalid request type"):
            await openai_client.send_request("invalid_request")

    def test_generation_request_with_ctx_info_endpoint_is_signed(self, mock_router, mock_session):
        _reset_prometheus_registry()
        client = OpenAIHttpClient(
            router=mock_router,
            role=ServerRole.GENERATION,
            session=mock_session,
            internal_disagg_auth_key="secret",
        )
        request = CompletionRequest(
            model="test-model",
            prompt="Hello, world!",
            disaggregated_params=DisaggregatedParams(
                request_type="generation_only",
                ctx_request_id=1,
                disagg_request_id=2,
                ctx_info_endpoint="tcp://10.0.0.1:5000",
            ),
        )

        headers = client._get_request_headers(request)

        assert headers is not None
        assert INTERNAL_DISAGG_AUTH_HEADER in headers

    def test_generation_request_with_ctx_info_endpoint_without_key_warns(
        self, mock_router, mock_session
    ):
        _reset_prometheus_registry()
        client = OpenAIHttpClient(
            router=mock_router,
            role=ServerRole.GENERATION,
            session=mock_session,
        )
        request = CompletionRequest(
            model="test-model",
            prompt="Hello, world!",
            disaggregated_params=DisaggregatedParams(
                request_type="generation_only",
                ctx_request_id=1,
                disagg_request_id=2,
                ctx_info_endpoint="tcp://10.0.0.1:5000",
            ),
        )

        warning_message = "In a future release the requirement to use internal_request_auth_key"
        with pytest.warns(FutureWarning, match=warning_message):
            headers = client._get_request_headers(request)

        assert INTERNAL_DISAGG_AUTH_HEADER not in headers


class TestHttpErrorBodyPreservation:
    """Test that HTTP 4xx/5xx errors include the response body (TRTLLM-11123)."""

    def _mock_http_error(self, status, body):
        r = AsyncMock()
        r.status = status
        r.reason = "Bad Request" if status == 400 else "Internal Server Error"
        r.text = AsyncMock(return_value=body)
        r.headers = {"Content-Type": "application/json"}
        r.request_info = MagicMock()
        r.history = ()
        r.__aenter__ = AsyncMock(return_value=r)
        r.__aexit__ = AsyncMock(return_value=False)
        return r

    def _make_client(self, session, **kwargs):
        from prometheus_client.registry import REGISTRY

        REGISTRY._names_to_collectors = {}
        REGISTRY._collector_to_names = {}
        router = AsyncMock(spec=Router)
        router.servers = ["localhost:8000"]
        router.get_next_server = AsyncMock(return_value=("localhost:8000", None))
        router.finish_request = AsyncMock()
        return OpenAIHttpClient(
            router=router,
            role=ServerRole.CONTEXT,
            timeout_secs=10,
            max_retries=0,
            session=session,
            **kwargs,
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "status,body",
        [
            (400, '{"error":"missing field X"}'),
            (500, "internal failure detail"),
        ],
    )
    async def test_error_body_in_exception(self, status, body):
        session = AsyncMock(spec=aiohttp.ClientSession)
        session.post.return_value = self._mock_http_error(status, body)
        client = self._make_client(session)
        req = CompletionRequest(
            model="m",
            prompt="hi",
            stream=False,
            disaggregated_params=DisaggregatedParams(request_type="context_only", ctx_request_id=1),
        )
        with pytest.raises(aiohttp.ClientResponseError) as exc_info:
            await client.send_request(req)
        assert body[:20] in str(exc_info.value.message)


class TestDisaggIdRegenOnRetry:
    """Test that disagg_request_id is regenerated on retry (TRTLLM-11123)."""

    def _ok_response(self):
        return CompletionResponse(
            model="m",
            usage=UsageInfo(prompt_tokens=1, completion_tokens=1),
            choices=[CompletionResponseChoice(index=0, text="ok")],
        ).model_dump()

    def _mock_http_ok(self, json_val):
        r = AsyncMock()
        r.status = 200
        r.headers = {"Content-Type": "application/json"}
        r.json = AsyncMock(return_value=json_val)
        r.__aenter__ = AsyncMock(return_value=r)
        r.__aexit__ = AsyncMock()
        return r

    def _make_client(self, session, role=ServerRole.CONTEXT, **kwargs):
        from prometheus_client.registry import REGISTRY

        REGISTRY._names_to_collectors = {}
        REGISTRY._collector_to_names = {}
        router = AsyncMock(spec=Router)
        router.servers = ["localhost:8000"]
        router.get_next_server = AsyncMock(return_value=("localhost:8000", None))
        router.finish_request = AsyncMock()
        return OpenAIHttpClient(
            router=router,
            role=role,
            timeout_secs=10,
            max_retries=2,
            retry_interval_sec=0,
            session=session,
            **kwargs,
        )

    def _mock_sse_ok(self):
        r = AsyncMock()
        r.status = 200
        r.headers = {"Content-Type": "text/event-stream"}

        async def iter_any():
            yield b'data: {"choices":[]}\n\n'
            yield b"data: [DONE]\n\n"

        r.content = AsyncMock()
        r.content.iter_any = iter_any
        r.__aenter__ = AsyncMock(return_value=r)
        r.__aexit__ = AsyncMock()
        return r

    @pytest.mark.asyncio
    @pytest.mark.parametrize("stream", [False, True])
    async def test_generation_retry_keeps_the_kv_handoff_id(self, stream):
        """A retried generation request must ask for the KV its context phase made.

        The context worker registered its KV-transfer session under the
        context request's disagg_request_id (native/transfer.py TxSession,
        keyed in Sender.setup_session), and the orchestrator copies that id
        onto the generation request (openai_disagg_service._get_gen_request).
        The generation worker keys its receive session on the disagg_request_id
        it is sent (RxSession.disagg_request_id prefers it over ctx_request_id)
        and asks the context worker for KV under that key. So the id on the
        wire is the handoff key: re-minting it on a retry -- a stale keep-alive
        socket alone earns up to five -- sends the generation worker after KV
        that nobody will ever send, and the request sits in transfer until the
        receive timeout instead of generating.
        """
        session = AsyncMock(spec=aiohttp.ClientSession)
        ids = iter(range(1000, 2000))

        async def next_id():
            return next(ids)

        client = self._make_client(session, role=ServerRole.GENERATION, disagg_id_generator=next_id)
        answer = self._mock_sse_ok() if stream else self._mock_http_ok(self._ok_response())
        session.post.side_effect = [ConnectionResetError(), answer]
        req = CompletionRequest(
            model="m",
            prompt=[1, 2, 3],
            stream=stream,
            disaggregated_params=DisaggregatedParams(
                request_type="generation_only",
                first_gen_tokens=[7],
                ctx_request_id=42,
                disagg_request_id=42,
            ),
        )
        hooks = MagicMock(spec=ResponseHooks)

        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = await client.send_request(req, hooks=hooks)
            if stream:
                _ = [chunk async for chunk in result]

        wire_ids = [
            json.loads(call.kwargs["data"])["disaggregated_params"]["disagg_request_id"]
            for call in session.post.call_args_list
        ]
        # Both attempts ask for the KV under the key the context side used.
        assert wire_ids == [42, 42]
        hooks.on_disagg_request_id.assert_not_called()

    @pytest.mark.asyncio
    async def test_retry_regenerates_disagg_id(self):
        session = AsyncMock(spec=aiohttp.ClientSession)
        ids = iter(range(1000, 2000))

        async def next_id():
            return next(ids)

        client = self._make_client(session, disagg_id_generator=next_id)

        session.post.side_effect = [
            aiohttp.ClientError("transient"),
            self._mock_http_ok(self._ok_response()),
        ]
        req = CompletionRequest(
            model="m",
            prompt="hi",
            stream=False,
            disaggregated_params=DisaggregatedParams(
                request_type="context_only", disagg_request_id=42
            ),
        )

        with patch("asyncio.sleep", new_callable=AsyncMock):
            resp = await client.send_request(req)

        assert req.disaggregated_params.disagg_request_id != 42
        assert isinstance(resp, CompletionResponse)

    @pytest.mark.asyncio
    async def test_no_generator_keeps_original_id(self):
        session = AsyncMock(spec=aiohttp.ClientSession)
        client = self._make_client(session)  # no disagg_id_generator

        session.post.side_effect = [
            aiohttp.ClientError("transient"),
            self._mock_http_ok(self._ok_response()),
        ]
        req = CompletionRequest(
            model="m",
            prompt="hi",
            stream=False,
            disaggregated_params=DisaggregatedParams(
                request_type="context_only", disagg_request_id=42
            ),
        )

        with patch("asyncio.sleep", new_callable=AsyncMock):
            await client.send_request(req)

        assert req.disaggregated_params.disagg_request_id == 42


class TestSelectiveTransientTcpRetry:
    """Selective retry budget for transient TCP race symptoms.

    ServerDisconnectedError and ConnectionResetError (which include
    aiohttp.ClientConnectionResetError via MRO) get an extended retry budget
    of up to 5 attempts; all other client errors keep the original
    max_retries fail-fast behaviour.
    """

    def _ok_response(self):
        return CompletionResponse(
            id="cmpl-1",
            object="text_completion",
            created=0,
            model="m",
            choices=[CompletionResponseChoice(index=0, text="ok", finish_reason="stop")],
            usage=UsageInfo(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        )

    def _mock_http_ok(self, body):
        r = AsyncMock()
        r.status = 200
        r.headers = {"Content-Type": "application/json"}
        r.json = AsyncMock(return_value=body.model_dump())
        r.__aenter__ = AsyncMock(return_value=r)
        r.__aexit__ = AsyncMock()
        return r

    def _make_client(self, session, max_retries=1):
        from prometheus_client.registry import REGISTRY

        REGISTRY._names_to_collectors = {}
        REGISTRY._collector_to_names = {}
        router = AsyncMock(spec=Router)
        router.servers = ["localhost:8000"]
        router.get_next_server = AsyncMock(return_value=("localhost:8000", None))
        router.finish_request = AsyncMock()
        return OpenAIHttpClient(
            router=router,
            role=ServerRole.CONTEXT,
            timeout_secs=10,
            max_retries=max_retries,
            retry_interval_sec=0,
            session=session,
        )

    def _make_request(self):
        return CompletionRequest(
            model="m",
            prompt="hi",
            stream=False,
            disaggregated_params=DisaggregatedParams(
                request_type="context_only", disagg_request_id=1
            ),
        )

    @pytest.mark.asyncio
    async def test_server_disconnected_gets_extra_retries(self):
        """ServerDisconnectedError: even with max_retries=1, retry up to 5."""
        session = AsyncMock(spec=aiohttp.ClientSession)
        client = self._make_client(session, max_retries=1)

        # 4 disconnect failures then success on the 5th attempt
        session.post.side_effect = [
            aiohttp.ServerDisconnectedError(),
            aiohttp.ServerDisconnectedError(),
            aiohttp.ServerDisconnectedError(),
            aiohttp.ServerDisconnectedError(),
            self._mock_http_ok(self._ok_response()),
        ]

        with patch("asyncio.sleep", new_callable=AsyncMock):
            await client.send_request(self._make_request())

        # 1 original + 4 retries = 5 total attempts (extra budget kicked in)
        assert session.post.call_count == 5

    @pytest.mark.asyncio
    async def test_connection_reset_gets_extra_retries(self):
        """ConnectionResetError: same extra budget as ServerDisconnectedError."""
        session = AsyncMock(spec=aiohttp.ClientSession)
        client = self._make_client(session, max_retries=1)

        session.post.side_effect = [
            ConnectionResetError(),
            ConnectionResetError(),
            self._mock_http_ok(self._ok_response()),
        ]

        with patch("asyncio.sleep", new_callable=AsyncMock):
            await client.send_request(self._make_request())

        # 1 original + 2 retries = 3 total attempts (within extra budget)
        assert session.post.call_count == 3

    @pytest.mark.asyncio
    async def test_other_client_error_keeps_fail_fast(self):
        """Generic aiohttp.ClientError still respects max_retries (=1)."""
        session = AsyncMock(spec=aiohttp.ClientSession)
        client = self._make_client(session, max_retries=1)

        session.post.side_effect = aiohttp.ClientError("transient non-tcp")

        with patch("asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(aiohttp.ClientError):
                await client.send_request(self._make_request())

        # Original + 1 retry = 2 attempts, NOT promoted to 5
        assert session.post.call_count == 2

    @pytest.mark.asyncio
    async def test_max_retries_zero_still_gets_transient_tcp_budget(self):
        """Even when max_retries=0, transient TCP races still retry up to 5."""
        session = AsyncMock(spec=aiohttp.ClientSession)
        client = self._make_client(session, max_retries=0)

        session.post.side_effect = [
            aiohttp.ServerDisconnectedError(),
            self._mock_http_ok(self._ok_response()),
        ]

        with patch("asyncio.sleep", new_callable=AsyncMock):
            await client.send_request(self._make_request())

        assert session.post.call_count == 2

    @pytest.mark.asyncio
    async def test_transient_tcp_capped_at_5_when_max_retries_smaller(self):
        """If transient TCP keeps failing, give up after the extended budget."""
        session = AsyncMock(spec=aiohttp.ClientSession)
        client = self._make_client(session, max_retries=1)

        # Always raise — must give up after extended (1 + 5) = 6 attempts
        session.post.side_effect = aiohttp.ServerDisconnectedError()

        with patch("asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(aiohttp.ServerDisconnectedError):
                await client.send_request(self._make_request())

        # 1 original + 5 retries
        assert session.post.call_count == 6


class TestWorkerRejectionIsNotRetried:
    """A worker's 4xx is a verdict on the request, so it is sent exactly once.

    Retries go back to the same server with the same body, and a 400 there --
    a context overflow, a malformed tool -- comes back identically, after the
    worker has re-rendered a prompt that can run to hundreds of thousands of
    characters. The caller already relays the worker's status and reason to
    the client, so the second attempt bought nothing but load. 408 and 429 are
    the 4xx that say "try again", and 5xx and connection failures keep their
    retries.
    """

    def _http_error(self, status):
        r = AsyncMock()
        r.status = status
        r.reason = "Error"
        r.headers = {"Content-Type": "application/json"}
        r.text = AsyncMock(
            return_value=json.dumps({"object": "error", "message": "rejected", "code": status})
        )
        r.request_info = MagicMock()
        r.history = ()
        r.__aenter__ = AsyncMock(return_value=r)
        r.__aexit__ = AsyncMock(return_value=False)
        return r

    def _make_client(self, session):
        _reset_prometheus_registry()
        router = AsyncMock(spec=Router)
        router.servers = ["localhost:8000"]
        router.get_next_server = AsyncMock(return_value=("localhost:8000", None))
        router.finish_request = AsyncMock()
        # The production default: one retry.
        client = OpenAIHttpClient(
            router=router,
            role=ServerRole.CONTEXT,
            timeout_secs=10,
            max_retries=1,
            retry_interval_sec=0,
            session=session,
        )
        return client, router

    async def _send(self, client, stream):
        request = CompletionRequest(
            model="m",
            prompt="hi",
            stream=stream,
            disaggregated_params=DisaggregatedParams(
                request_type="context_only", disagg_request_id=1
            ),
        )
        result = await client.send_request(request)
        if stream:
            # A streaming request meets the upstream status on first read.
            _ = [chunk async for chunk in result]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("stream", [False, True])
    @pytest.mark.parametrize("status", [400, 404, 413, 422])
    async def test_a_rejection_is_sent_once(self, status, stream):
        session = AsyncMock(spec=aiohttp.ClientSession)
        session.post.return_value = self._http_error(status)
        client, router = self._make_client(session)

        with patch("asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(aiohttp.ClientResponseError) as exc:
                await self._send(client, stream)

        assert exc.value.status == status
        assert session.post.call_count == 1
        # Settled once, as a failure, like any other failed request.
        router.finish_request.assert_called_once()
        assert router.finish_request.call_args.kwargs.get("success") is False

    @pytest.mark.asyncio
    @pytest.mark.parametrize("stream", [False, True])
    @pytest.mark.parametrize("status", [408, 429, 500, 502, 503])
    async def test_a_retryable_status_is_still_retried(self, status, stream):
        session = AsyncMock(spec=aiohttp.ClientSession)
        session.post.return_value = self._http_error(status)
        client, router = self._make_client(session)

        with patch("asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(aiohttp.ClientResponseError) as exc:
                await self._send(client, stream)

        assert exc.value.status == status
        # Original + max_retries=1.
        assert session.post.call_count == 2
        router.finish_request.assert_called_once()

    @pytest.mark.asyncio
    async def test_a_connection_error_is_still_retried(self):
        session = AsyncMock(spec=aiohttp.ClientSession)
        session.post.side_effect = aiohttp.ClientConnectionError("refused")
        client, _ = self._make_client(session)

        with patch("asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(aiohttp.ClientConnectionError):
                await self._send(client, stream=False)

        assert session.post.call_count == 2


class TestStreamingTimeoutBudget:
    """What bounds a streaming request to a worker.

    A stream is bounded by silence, not by how long it runs. Bounding it by
    total elapsed time cuts healthy long generations: the worker is still
    emitting, the budget expires anyway, and the SSE body ends without a
    terminator. Downstream that is close to invisible -- the client already
    holds a 200 and the proxy cannot retry, having already yielded -- so it is
    asserted here rather than left to a deployment to discover.
    """

    @staticmethod
    def _http_response(content_type, chunks=()):
        response = AsyncMock()
        response.status = 200
        response.headers = {"Content-Type": content_type}
        # A non-streaming response is validated into a CompletionResponse, so
        # an empty body fails before the assertion under test is reached.
        response.json = AsyncMock(
            return_value=CompletionResponse(
                id="test-123",
                object="text_completion",
                created=1234567890,
                model="test-model",
                usage=UsageInfo(prompt_tokens=1, completion_tokens=1),
                choices=[CompletionResponseChoice(index=0, text="hi")],
            ).model_dump()
        )

        async def iter_any():
            for chunk in chunks:
                yield chunk

        response.content = AsyncMock()
        response.content.iter_any = iter_any
        response.__aenter__ = AsyncMock(return_value=response)
        response.__aexit__ = AsyncMock()
        return response

    @pytest.mark.asyncio
    async def test_a_stream_is_bounded_by_silence_not_by_elapsed_time(
        self, openai_client, streaming_completion_request, mock_session
    ):
        mock_session.post.return_value = self._http_response(
            "text/event-stream", [b'data: "hi"\n\n']
        )

        generator = await openai_client.send_request(streaming_completion_request)
        async for _ in generator:
            pass

        timeout = mock_session.post.call_args.kwargs["timeout"]
        # No ceiling on the generation itself: a worker that keeps producing
        # tokens for longer than the budget is healthy, not stuck.
        assert timeout.total is None
        # A worker that goes quiet for the budget is stuck, and still fails.
        assert timeout.sock_read == 180

    @pytest.mark.asyncio
    async def test_a_non_streaming_request_keeps_its_total_budget(
        self, openai_client, completion_request, mock_session
    ):
        mock_session.post.return_value = self._http_response("application/json")

        await openai_client.send_request(completion_request)

        timeout = mock_session.post.call_args.kwargs["timeout"]
        # Nothing streams back here, so elapsed time is the whole of the
        # request and remains the right thing to bound.
        assert timeout.total == 180


class TestStreamingErrorPathFinalization:
    """Every exit of a streaming request finalizes the router exactly once.

    _send_request hands the streaming generator back to the caller unstarted,
    so for a streaming request the client's own except never runs -- the
    returned generator is the only thing left to clean up. Two of its exits used
    to skip finalization entirely: a response that is not an event-stream (a 400
    with a JSON body surfaced as a bare ``AssertionError: Response is not
    streaming``, thrown ahead of the finalizing ``finally``) and a transport
    failure that exhausts retries before any response object exists. Both left
    the router load count and the routing entry leaked, with zero finish calls.
    The error also has to carry the upstream status and body so the retry
    decision upstream can tell a deterministic 400 from a transport flake.
    """

    def _mock_non_sse_response(self, status, body):
        r = AsyncMock()
        r.status = status
        r.reason = "Bad Request" if status == 400 else "OK"
        r.headers = {"Content-Type": "application/json"}
        r.text = AsyncMock(return_value=body)
        r.request_info = MagicMock()
        r.history = ()
        r.__aenter__ = AsyncMock(return_value=r)
        r.__aexit__ = AsyncMock(return_value=False)
        return r

    def _mock_sse_response(self, chunks):
        r = AsyncMock()
        r.status = 200
        r.headers = {"Content-Type": "text/event-stream"}

        async def iter_any():
            for c in chunks:
                yield c

        r.content = AsyncMock()
        r.content.iter_any = iter_any
        r.__aenter__ = AsyncMock(return_value=r)
        r.__aexit__ = AsyncMock()
        return r

    def _make_client(self, session, router=None, max_retries=0):
        _reset_prometheus_registry()
        if router is None:
            router = AsyncMock(spec=Router)
            router.servers = ["localhost:8000"]
            router.get_next_server = AsyncMock(return_value=("localhost:8000", None))
            router.finish_request = AsyncMock()
        return OpenAIHttpClient(
            router=router,
            role=ServerRole.CONTEXT,
            timeout_secs=10,
            max_retries=max_retries,
            retry_interval_sec=0,
            session=session,
        ), router

    def _stream_request(self):
        return CompletionRequest(
            model="m",
            prompt="hi",
            stream=True,
            disaggregated_params=DisaggregatedParams(request_type="context_only", ctx_request_id=1),
        )

    @pytest.mark.asyncio
    async def test_stream_non_sse_400_carries_status_and_body_and_finalizes_once(self):
        """(a) 400 + JSON body on the stream path: error is classifiable, finalize once."""
        session = AsyncMock(spec=aiohttp.ClientSession)
        session.post.return_value = self._mock_non_sse_response(
            400, '{"error":"deterministic bad request field X"}'
        )
        client, router = self._make_client(session)

        gen = await client.send_request(self._stream_request())
        with pytest.raises(aiohttp.ClientResponseError) as exc:
            async for _ in gen:
                pass

        # The status and body survive -- a bare AssertionError carried neither.
        assert exc.value.status == 400
        assert "deterministic bad request field X" in str(exc.value.message)
        # Exactly one finalize, as a failure. Not zero (the leak), not two.
        router.finish_request.assert_called_once()
        assert router.finish_request.call_args.kwargs.get("success") is False

    @pytest.mark.asyncio
    async def test_stream_non_sse_400_leaves_no_leak_in_real_router(self):
        """(a) The leak itself: a real LoadBalancingRouter ends with no active count / route."""
        _reset_prometheus_registry()
        router = LoadBalancingRouter(server_role=ServerRole.CONTEXT, servers=["localhost:8000"])
        session = AsyncMock(spec=aiohttp.ClientSession)
        session.post.return_value = self._mock_non_sse_response(400, '{"error":"bad"}')
        client, _ = self._make_client(session, router=router)

        gen = await client.send_request(self._stream_request())
        with pytest.raises(aiohttp.ClientResponseError):
            async for _ in gen:
                pass

        assert router._server_state["localhost:8000"]._num_active_requests == 0
        assert router._req_routing_table == {}

    @pytest.mark.asyncio
    async def test_stream_transport_exhaustion_before_response_finalizes_once(self):
        """(b) Connector dies before any response object exists: finalize once, no leak."""
        session = AsyncMock(spec=aiohttp.ClientSession)
        session.post.side_effect = aiohttp.ClientConnectionError("connector exhausted")
        client, router = self._make_client(session)

        gen = await client.send_request(self._stream_request())
        with pytest.raises(aiohttp.ClientError):
            async for _ in gen:
                pass

        router.finish_request.assert_called_once()
        assert router.finish_request.call_args.kwargs.get("success") is False

    @pytest.mark.asyncio
    async def test_stream_transport_exhaustion_leaves_no_leak_in_real_router(self):
        """(b) Same, against a real router: nothing left routed."""
        _reset_prometheus_registry()
        router = LoadBalancingRouter(server_role=ServerRole.CONTEXT, servers=["localhost:8000"])
        session = AsyncMock(spec=aiohttp.ClientSession)
        session.post.side_effect = aiohttp.ClientConnectionError("connector exhausted")
        client, _ = self._make_client(session, router=router)

        gen = await client.send_request(self._stream_request())
        with pytest.raises(aiohttp.ClientError):
            async for _ in gen:
                pass

        assert router._server_state["localhost:8000"]._num_active_requests == 0
        assert router._req_routing_table == {}

    @pytest.mark.asyncio
    async def test_happy_sse_control_finalizes_exactly_once(self):
        """(c) Control: a good stream still finalizes exactly once, as success."""
        session = AsyncMock(spec=aiohttp.ClientSession)
        session.post.return_value = self._mock_sse_response(
            [b'data: "hi"\n\n', b"data: [DONE]\n\n"]
        )
        client, router = self._make_client(session)

        gen = await client.send_request(self._stream_request())
        seen = [c async for c in gen]

        assert seen == [b'data: "hi"\n\n', b"data: [DONE]\n\n"]
        router.finish_request.assert_called_once()
        assert router.finish_request.call_args.kwargs.get("success") is True

    @pytest.mark.asyncio
    async def test_nonstream_400_control_finalizes_exactly_once(self):
        """(c) Control: the non-streaming 400 path was already correct; keep it so."""
        session = AsyncMock(spec=aiohttp.ClientSession)
        session.post.return_value = self._mock_non_sse_response(400, '{"error":"nonstream bad"}')
        client, router = self._make_client(session)
        req = CompletionRequest(
            model="m",
            prompt="hi",
            stream=False,
            disaggregated_params=DisaggregatedParams(request_type="context_only", ctx_request_id=1),
        )

        with pytest.raises(aiohttp.ClientResponseError) as exc:
            await client.send_request(req)

        assert exc.value.status == 400
        assert "nonstream bad" in str(exc.value.message)
        router.finish_request.assert_called_once()
        assert router.finish_request.call_args.kwargs.get("success") is False


class TestClientDisconnectAbort:
    """A cancelled in-flight request must abort upstream work, settled exactly once.

    The production shape: a client hangs up while its 650K-token prefill is
    still grinding on a context worker. The orchestrator turns that into a
    cancellation of the coroutine awaiting the ctx POST (see
    openai_disagg_server._serve_until_client_disconnect). For the abort to
    reach the worker's engine, the cancellation must CLOSE the upstream socket
    -- the worker's own await_disconnected poller is watching for exactly that
    -- not park the connection back in the pool; and the router/metrics
    settlement must run exactly once, the same guarantee the streaming paths
    got in the finalize-once rework.
    """

    def _make_client(self, session, role=ServerRole.CONTEXT, **kwargs):
        _reset_prometheus_registry()
        router = AsyncMock(spec=Router)
        router.servers = ["localhost:8000"]
        router.get_next_server = AsyncMock(return_value=("localhost:8000", None))
        router.finish_request = AsyncMock()
        client = OpenAIHttpClient(
            router=router,
            role=role,
            timeout_secs=30,
            max_retries=0,
            retry_interval_sec=0,
            session=session,
            **kwargs,
        )
        return client, router

    def _ctx_request(self):
        return CompletionRequest(
            model="m",
            prompt="hi",
            stream=False,
            disaggregated_params=DisaggregatedParams(
                request_type="context_only", disagg_request_id=7
            ),
        )

    @pytest.mark.asyncio
    async def test_cancel_mid_ctx_post_closes_the_socket_not_the_pool(self):
        """(a) The cancellation reaches the transport.

        A real aiohttp session against a real localhost server that never
        answers -- the exact posture of a context worker mid-prefill. The
        server observing EOF on its connection is the proof that matters: a
        connection released to the keep-alive pool stays open and would never
        EOF here, so the worker's disconnect poller would never fire and the
        prefill would grind on.
        """
        request_seen = asyncio.Event()
        peer_gone = asyncio.Event()

        async def never_answer(reader, writer):
            head = b""
            while b"\r\n\r\n" not in head:
                chunk = await reader.read(65536)
                if not chunk:
                    break
                head += chunk
            request_seen.set()
            # Drain until EOF. Only a closed client connection EOFs; a pooled
            # one idles open, and this wait times the test out instead.
            while await reader.read(65536):
                pass
            peer_gone.set()
            writer.close()

        server = await asyncio.start_server(never_answer, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        session = aiohttp.ClientSession()
        try:
            client, router = self._make_client(session)
            task = asyncio.create_task(
                client.send_request(self._ctx_request(), server=f"127.0.0.1:{port}")
            )
            await asyncio.wait_for(request_seen.wait(), timeout=5)

            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

            # The transport-level fact under test: the worker saw its client vanish.
            await asyncio.wait_for(peer_gone.wait(), timeout=5)
            # And nothing was handed back to the connection pool.
            assert not any(session.connector._conns.values())
            # Settled exactly once, as a failure.
            router.finish_request.assert_called_once()
            assert router.finish_request.call_args.kwargs.get("success") is False
        finally:
            await session.close()
            server.close()
            await server.wait_closed()

    @pytest.mark.asyncio
    async def test_cancel_mid_ctx_post_finalizes_exactly_once(self):
        """(a) Settlement under cancellation, at the mock seam.

        The POST never produces a response (its __aenter__ hangs, as a worker
        mid-prefill does). Before the rework this exit finalized zero times --
        the failure settle lived in an `except Exception` that CancelledError
        walks straight past -- leaking the router load count and routing entry
        of every disconnected retry.
        """
        session = AsyncMock(spec=aiohttp.ClientSession)
        entered = asyncio.Event()
        hang_forever = asyncio.Event()

        class _HangingPost:
            async def __aenter__(self):
                entered.set()
                await hang_forever.wait()

            async def __aexit__(self, *exc_info):
                return False

        session.post = Mock(return_value=_HangingPost())
        client, router = self._make_client(session)

        task = asyncio.create_task(client.send_request(self._ctx_request()))
        await asyncio.wait_for(entered.wait(), timeout=5)

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        router.finish_request.assert_called_once()
        assert router.finish_request.call_args.kwargs.get("success") is False

    @pytest.mark.asyncio
    async def test_cancel_landing_during_the_success_settle_starts_no_second(self):
        """(a) The exactly-once guard composes with cancellation.

        A cancel can land while the inline success finish is itself in flight.
        The settled flag flips before that await, so the finally must not start
        a second, contradictory (success=False) settlement -- one finish per
        request, even one cut short, beats a double-decrement of router load.
        """
        session = AsyncMock(spec=aiohttp.ClientSession)
        ok = AsyncMock()
        ok.status = 200
        ok.headers = {"Content-Type": "application/json"}
        ok.json = AsyncMock(
            return_value=CompletionResponse(
                model="m",
                usage=UsageInfo(prompt_tokens=1, completion_tokens=1),
                choices=[CompletionResponseChoice(index=0, text="ok")],
            ).model_dump()
        )
        ok.__aenter__ = AsyncMock(return_value=ok)
        ok.__aexit__ = AsyncMock()
        session.post.return_value = ok

        client, router = self._make_client(session)
        settle_started = asyncio.Event()
        never = asyncio.Event()

        async def slow_finish(*args, **kwargs):
            settle_started.set()
            await never.wait()

        router.finish_request = AsyncMock(side_effect=slow_finish)

        task = asyncio.create_task(client.send_request(self._ctx_request()))
        await asyncio.wait_for(settle_started.wait(), timeout=5)

        task.cancel()
        # A cancellation escaping the async-generator boundary is converted by
        # asyncio into clean completion of the outer coroutine, so the task may
        # return rather than raise. Either way is fine; what must hold is that
        # the settled flag suppressed a second, contradictory finish.
        try:
            await task
        except asyncio.CancelledError:
            pass

        assert router.finish_request.call_count == 1

    @pytest.mark.asyncio
    async def test_gen_stream_cancel_mid_generation_behavior_is_pinned(self):
        """(b) Disconnect during generation streaming: unchanged, and pinned.

        Mid-stream cancellation already worked before this change --
        StreamingResponse's disconnect handling cancels the consumer, the
        cancellation unwinds _response_generator, and its finally settles the
        request. Pinned here including its quirk: the settle reports
        success=True (CancelledError skips both of the generator's except
        clauses), which the router's routed-block accounting has always been
        fed on a client hangup. Changing that is a router-semantics decision,
        not a side effect this fix is allowed to smuggle in.
        """
        session = AsyncMock(spec=aiohttp.ClientSession)
        sse = AsyncMock()
        sse.status = 200
        sse.headers = {"Content-Type": "text/event-stream"}
        stall = asyncio.Event()

        async def iter_any():
            yield b'data: "x"\n\n'
            await stall.wait()

        sse.content = AsyncMock()
        sse.content.iter_any = iter_any
        sse.__aenter__ = AsyncMock(return_value=sse)
        sse.__aexit__ = AsyncMock()
        session.post.return_value = sse

        client, router = self._make_client(session, role=ServerRole.GENERATION)
        request = CompletionRequest(
            model="m",
            prompt="hi",
            stream=True,
            disaggregated_params=DisaggregatedParams(
                request_type="generation_only", first_gen_tokens=[1], ctx_request_id=7
            ),
        )

        gen = await client.send_request(request)
        got_first = asyncio.Event()

        async def consume():
            async for _chunk in gen:
                got_first.set()

        consumer = asyncio.create_task(consume())
        await asyncio.wait_for(got_first.wait(), timeout=5)

        consumer.cancel()
        # As on the non-streaming path, the cancel is converted to clean
        # completion at the async-generator boundary; the invariant is the
        # settle, not the propagation shape.
        try:
            await consumer
        except asyncio.CancelledError:
            pass

        router.finish_request.assert_called_once()
        assert router.finish_request.call_args.kwargs.get("success") is True

    @pytest.mark.asyncio
    async def test_no_disconnect_changes_nothing(self):
        """(c) Control: an undisturbed non-streaming request settles once, as success."""
        session = AsyncMock(spec=aiohttp.ClientSession)
        ok = AsyncMock()
        ok.status = 200
        ok.headers = {"Content-Type": "application/json"}
        ok.json = AsyncMock(
            return_value=CompletionResponse(
                model="m",
                usage=UsageInfo(prompt_tokens=1, completion_tokens=1),
                choices=[CompletionResponseChoice(index=0, text="ok")],
            ).model_dump()
        )
        ok.__aenter__ = AsyncMock(return_value=ok)
        ok.__aexit__ = AsyncMock()
        session.post.return_value = ok
        client, router = self._make_client(session)

        response = await client.send_request(self._ctx_request())

        assert isinstance(response, CompletionResponse)
        router.finish_request.assert_called_once()
        assert router.finish_request.call_args.kwargs.get("success") is True


def test_a_forwarded_request_keeps_the_field_names_it_arrived_with():
    """`schema` must not reach a worker spelled `schema_`.

    pydantic cannot hold a field called `schema` -- it shadows
    `BaseModel.schema` -- so the model declares `schema_` with `schema` as its
    alias. Serialising without `by_alias` sends the internal name, every
    member of the format union fails to validate on the worker, and the
    request comes back 400. That was 43 of 195 Responses requests in one
    campaign round: every structured-output call the agents made.

    Only disaggregated serving re-serialises a request, so only it is
    affected.
    """
    _reset_prometheus_registry()
    session = AsyncMock(spec=aiohttp.ClientSession)
    # Constructed for its metrics-registration side effect only.
    _ = OpenAIHttpClient(router=AsyncMock(spec=Router), role=ServerRole.CONTEXT, session=session)

    request = ResponsesRequest.model_validate(
        {
            "model": "m",
            "input": "hi",
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "structured_output",
                    "schema": {"type": "object"},
                    "strict": True,
                }
            },
        }
    )

    body = request.model_dump_json(exclude_unset=True, by_alias=True)

    assert '"schema"' in body
    assert '"schema_"' not in body
