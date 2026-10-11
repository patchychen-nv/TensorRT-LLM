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

# yapf: disable
import asyncio
import json
import os
import traceback
from abc import ABC, abstractmethod
from typing import (
    Any,
    AsyncGenerator,
    Awaitable,
    Callable,
    List,
    Optional,
    Protocol,
    Tuple,
    Type,
    TypeVar,
)

import aiohttp
from pydantic import BaseModel

from tensorrt_llm.llmapi.disagg_utils import ServerRole
from tensorrt_llm.logger import logger
from tensorrt_llm.serve.disagg_auth import (
    build_internal_disagg_auth_headers,
    request_requires_internal_disagg_auth,
)
from tensorrt_llm.serve.openai_protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    CompletionRequest,
    CompletionResponse,
    ResponsesRequest,
    ResponsesResponse,
    UCompletionRequest,
    UCompletionResponse,
)
from tensorrt_llm.serve.perf_metrics import (
    _PERF_METRICS_HEADER_BUDGET_BYTES,
    RETURN_METRICS_HEADER,
    SSE_METRICS_EVENT,
    ClientMetricsCollector,
    build_metrics_record_from_headers,
)
from tensorrt_llm.serve.responses_utils import (
    ResponseHooks,
    UCompletionResponseOrGenerator,
    get_steady_clock_now_in_seconds,
)
from tensorrt_llm.serve.router import Router

# yapf: enable

# msgspec msgpack is an opt-in transport for the orchestrator->worker request
# body (alternative to the JSON path). Enable with TRTLLM_SERVE_ENABLE_MSGSPEC=1;
# the orchestrator encodes the forwarded body as msgpack and flags it with the
# X-TRTLLM-Msgpack header (Content-Type stays application/json so FastAPI still
# routes it through Request.json()). Fails loudly at import if msgspec is missing.
_MSGSPEC_ENABLED = os.getenv("TRTLLM_SERVE_ENABLE_MSGSPEC", "0") == "1"
if _MSGSPEC_ENABLED:
    try:
        import msgspec
    except ImportError as exc:
        raise ImportError(
            "TRTLLM_SERVE_ENABLE_MSGSPEC=1 requires the msgspec package "
            "(listed in requirements.txt)."
        ) from exc
    _msgpack_encoder = msgspec.msgpack.Encoder()


# post_json's request and response are ordinary pydantic models, but not the
# UCompletion* union: it carries auxiliary payloads such as count-tokens. The
# protocol names the one method actually called, and the TypeVar ties the
# returned object to the type the caller asked for, so a mismatch is a type
# error rather than an Any that silently propagates.
class _SerializableRequest(Protocol):
    def model_dump_json(self, *, exclude_unset: bool = ...) -> str: ...

    def model_dump(self, *, mode: str = ..., exclude_unset: bool = ...) -> dict: ...


_ResponseT = TypeVar("_ResponseT", bound=BaseModel)


def _metrics_phase(role: ServerRole) -> str:
    return "ctx" if role is ServerRole.CONTEXT else "gen"


# The 4xx statuses that mean "try again later" rather than "this request is wrong".
_RETRYABLE_CLIENT_ERROR_STATUSES = frozenset((408, 429))


def _is_worker_rejection(error: Exception) -> bool:
    """Whether a worker answered with a verdict on the request itself.

    Such an answer is deterministic: a retry goes back to the same server with
    the same body and gets the same 4xx -- after the worker has re-rendered a
    prompt that can run to hundreds of thousands of characters. The caller
    relays the status and reason to the client either way.
    """
    return (
        isinstance(error, aiohttp.ClientResponseError)
        and 400 <= error.status < 500
        and error.status not in _RETRYABLE_CLIENT_ERROR_STATUSES
    )


class OpenAIClient(ABC):
    async def send_request(
        self,
        request: UCompletionRequest,
        server: Optional[str] = None,
        hooks: Optional[ResponseHooks] = None,
        req_id: Optional[int] = None,
    ) -> UCompletionResponseOrGenerator:
        if isinstance(request, CompletionRequest):
            return await self._send_request(
                "v1/completions", request, CompletionResponse, server, hooks, req_id
            )
        elif isinstance(request, ChatCompletionRequest):
            return await self._send_request(
                "v1/chat/completions",
                request,
                ChatCompletionResponse,
                server,
                hooks,
                req_id,
            )
        elif isinstance(request, ResponsesRequest):
            return await self._send_request(
                "v1/responses", request, ResponsesResponse, server, hooks, req_id
            )
        else:
            raise ValueError(f"Invalid request type: {type(request)}")

    @abstractmethod
    async def _send_request(
        self,
        endpoint: str,
        request: UCompletionRequest,
        response_type: Type[UCompletionResponse],
        server: Optional[str] = None,
        hooks: Optional[ResponseHooks] = None,
        req_id: Optional[int] = None,
    ) -> UCompletionResponseOrGenerator:
        """Send a request to the server and return the response and the body generator.

        The request is finished (in routers) when the generator is exhausted or there is an error.
        """
        ...

    @abstractmethod
    async def check_ready(self) -> Tuple[List[str], List[str]]:
        """Return the list of ready servers and the list of unready servers."""
        ...

    @abstractmethod
    async def post_json(
        self,
        endpoint: str,
        request: _SerializableRequest,
        response_type: Type[_ResponseT],
        server: str,
    ) -> _ResponseT:
        """Post a non-streaming auxiliary request to a selected worker.

        Unlike _send_request this carries no disaggregation state and does not
        touch the router's in-flight accounting: it is for side endpoints such
        as token counting, which need a worker's tokenizer but no KV transfer.
        """
        ...

    @abstractmethod
    async def get_json(
        self,
        endpoint: str,
        response_type: Type[_ResponseT],
        server: str,
    ) -> _ResponseT:
        """Read a worker's bodyless GET endpoint.

        The counterpart to post_json for endpoints that describe the worker
        rather than ask it to do anything -- /v1/models being the one that
        matters. Same exemption from routing state, and no request body to
        serialise.
        """
        ...

    async def shutdown(self) -> None: ...

    @abstractmethod
    async def _finish_request(
        self,
        request: UCompletionRequest,
        success: bool = True,
        req_id: Optional[int] = None,
    ) -> None:
        """Finish the request in the router.

        ``success`` lets the router distinguish completed vs failed requests
        (e.g. for routed-block tracking).
        """
        ...


class OpenAIHttpClient(OpenAIClient):
    def __init__(
        self,
        router: Router,
        role: ServerRole,
        timeout_secs: int = 180,
        max_retries: int = 1,
        retry_interval_sec: int = 1,
        session: Optional[aiohttp.ClientSession] = None,
        disagg_id_generator: Optional[Callable[[], Awaitable[int]]] = None,
        request_perf_metrics: bool = False,
        internal_disagg_auth_key: Optional[str] = None,
    ):
        self._router = router
        self._role = role
        self._timeout_secs = timeout_secs
        self._metrics_collector = ClientMetricsCollector(role)
        self._session = session or aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(
                limit=0,
                limit_per_host=0,
                force_close=False,
                # Set keepalive_timeout below the server-side keepalive timeout to avoid reusing stale connections.
                keepalive_timeout=1,
            ),
            timeout=aiohttp.ClientTimeout(total=timeout_secs),
            max_field_size=_PERF_METRICS_HEADER_BUDGET_BYTES,
        )
        self._max_retries = max_retries
        self._retry_interval_sec = retry_interval_sec
        self._disagg_id_generator = disagg_id_generator
        self._request_perf_metrics = request_perf_metrics
        self._internal_disagg_auth_key = internal_disagg_auth_key

    def _get_request_headers(self, request: UCompletionRequest) -> dict[str, str]:
        if self._role != ServerRole.GENERATION:
            return {}
        if not request_requires_internal_disagg_auth(request):
            return {}
        return build_internal_disagg_auth_headers(self._internal_disagg_auth_key, request)

    async def post_json(
        self,
        endpoint: str,
        request: _SerializableRequest,
        response_type: Type[_ResponseT],
        server: str,
    ) -> _ResponseT:
        server_url = server if server.startswith("http") else f"http://{server}"
        url = f"{server_url.rstrip('/')}/{endpoint}"
        # A count-tokens body carries the whole conversation, so it is as large
        # as a chat body and serialising it inline costs the orchestrator's
        # event loop the same way. The worker decodes either form: _MsgspecRoute
        # is the app's route_class, so it covers every route.
        if _MSGSPEC_ENABLED:
            body = _msgpack_encoder.encode(
                request.model_dump(mode="json", exclude_unset=True, by_alias=True)
            )
            headers = {"Content-Type": "application/json", "X-TRTLLM-Msgpack": "1"}
        else:
            body = request.model_dump_json(exclude_unset=True, by_alias=True)
            headers = {"Content-Type": "application/json"}
        async with self._session.post(url, data=body, headers=headers) as response:
            if response.status >= 400:
                error_body = await response.text()
                raise aiohttp.ClientResponseError(
                    response.request_info,
                    response.history,
                    status=response.status,
                    message=f"{response.reason}: {error_body[:2048]}",
                    headers=response.headers,
                )
            return response_type(**await response.json())

    async def get_json(
        self,
        endpoint: str,
        response_type: Type[_ResponseT],
        server: str,
    ) -> _ResponseT:
        server_url = server if server.startswith("http") else f"http://{server}"
        url = f"{server_url.rstrip('/')}/{endpoint}"
        async with self._session.get(url) as response:
            if response.status >= 400:
                error_body = await response.text()
                raise aiohttp.ClientResponseError(
                    response.request_info,
                    response.history,
                    status=response.status,
                    message=f"{response.reason}: {error_body[:2048]}",
                    headers=response.headers,
                )
            return response_type(**await response.json())

    async def _send_request(
        self,
        endpoint: str,
        request: UCompletionRequest,
        response_type: Type[UCompletionResponse],
        server: Optional[str] = None,
        hooks: Optional[ResponseHooks] = None,
        req_id: Optional[int] = None,
    ) -> UCompletionResponseOrGenerator:
        if server is None:
            if req_id is None:
                server, _ = await self._router.get_next_server(request)
            else:
                server, _ = await self._router.get_next_server(request, req_id=req_id)
        url = f"http://{server}/{endpoint}"
        # disaggregated_params is None when conditional_disagg bypasses ctx.
        _dp = request.disaggregated_params
        _ctx_rid = _dp.ctx_request_id if _dp is not None else None
        logger.debug(f"Sending {self._role} request {_ctx_rid} to {url}")
        resp_generator = self._post_with_retry(server, url, request, hooks, req_id)
        try:
            self._metrics_collector.total_requests.inc()
            if request.stream:
                # return the response generator, the request is not done yet
                return resp_generator
            else:
                # consume the generator to get the response and return it directly when it's not streaming
                response = None
                async for resp_json in resp_generator:
                    response = response_type(**resp_json)
                    if hooks:
                        if self._role == ServerRole.CONTEXT:
                            hooks.on_ctx_resp(server, response)
                        else:
                            hooks.on_first_token(server, request)
                            hooks.on_resp_done(server, request, response)
                return response
        except (Exception, asyncio.CancelledError):
            # The generator's `finally` owns the request's settlement (router
            # bookkeeping + error counter), for non-streaming exits too.
            # Closing the generator here runs that `finally` NOW when the
            # failure was raised in THIS frame (a response that does not
            # validate, a hook that raises) and the generator is still
            # suspended at its yield; when the failure came from inside the
            # generator - including asyncio.CancelledError, which is how a
            # downstream client disconnect arrives once the orchestrator
            # cancels the pipeline mid-prefill - the frame already unwound and
            # settled itself, and aclose() is a no-op. Either way the request
            # is settled exactly once, and the exception (cancellation
            # included) propagates unchanged.
            await resp_generator.aclose()
            raise

    async def _post_with_retry(
        self,
        server: str,
        url: str,
        request: UCompletionRequest,
        hooks: Optional[ResponseHooks] = None,
        req_id: Optional[int] = None,
    ) -> AsyncGenerator[Any, None]:
        is_stream = request.stream
        # aiohttp's `total` budget covers reading the response body, so on a
        # streaming request it bounds the whole generation instead of catching a
        # stalled worker: a stream that is still emitting tokens is cut at
        # `timeout_secs`, mid-SSE, with no terminator. Long agent turns hit this
        # routinely -- four ~190s generations died at ~1127 events each while a
        # 180s one finished -- and the failure is nearly invisible downstream:
        # the client already holds a 200, the body just stops, and this client
        # cannot retry because it has already yielded.
        #
        # Inactivity is the bound that means something for a stream, so
        # `sock_read` replaces `total`. A worker that goes quiet for
        # `timeout_secs` still fails; one that keeps sending never does.
        request_timeout = (
            aiohttp.ClientTimeout(
                total=None,
                connect=self._timeout_secs,
                sock_read=self._timeout_secs,
            )
            if is_stream
            else aiohttp.ClientTimeout(total=self._timeout_secs)
        )
        # Loop range must cover the transient-TCP extended budget (up to 5)
        # so the conditional raise inside the except block can actually decide
        # to keep retrying.  Non-transient errors still raise on the first
        # attempt that reaches self._max_retries.
        _TRANSIENT_TCP_BUDGET = 5
        loop_max = max(self._max_retries, _TRANSIENT_TCP_BUDGET) + 1
        # This generator owns the request's finalization, streaming and
        # non-streaming alike -- every exit here must settle the request exactly
        # once. For a streaming request, _send_request hands the generator back
        # to the caller unstarted (see its `if request.stream` branch), so once
        # returned no caller frame is left to run the router/bookkeeping cleanup
        # on error. Draining `_response_generator` hands that duty to its
        # `finally` (which runs on a clean stream and on a mid-stream failure
        # alike); until that handoff the duty is ours, and the `finally` below
        # covers the two exits that used to leak silently: a response that is not
        # an event-stream, and a transport failure that exhausts retries before
        # any response arrives. A non-streaming request settles inline on
        # success -- with `request_settled` flipped BEFORE the settling await, so
        # a cancellation landing during that very await cannot trigger a second,
        # contradictory settlement -- and via the `finally` on every other exit,
        # including asyncio.CancelledError: that is the shape a downstream
        # client disconnect takes once the orchestrator cancels the pipeline
        # mid-prefill, and the _send_request-side `except Exception` that used
        # to own the non-streaming failure settle never saw it.
        stream_finalize_delegated = False
        request_settled = False
        try:
            for attempt in range(loop_max):
                # Regenerate disagg_request_id on retry to avoid ID collision on
                # workers -- for a context-only request alone. The orchestrator
                # reads a context request's id back off its response, so a new
                # one carries through to the generation phase. A generation
                # request's id is the key the context worker registered its KV
                # transfer under; a new one sends the generation worker after
                # KV that nobody will ever send.
                if attempt > 0 and self._disagg_id_generator is not None:
                    dp = getattr(request, "disaggregated_params", None)
                    if (
                        dp is not None
                        and dp.request_type == "context_only"
                        and getattr(dp, "disagg_request_id", None) is not None
                    ):
                        dp.disagg_request_id = await self._disagg_id_generator()
                        if hooks:
                            hooks.on_disagg_request_id(dp.disagg_request_id)
                # Serialize once on the orchestrator's single event-loop thread.
                if _MSGSPEC_ENABLED:
                    # msgspec msgpack: encode the request dict to msgpack bytes. Keep
                    # Content-Type application/json so FastAPI still routes the body
                    # through Request.json() (it only does that for json/+json content
                    # subtypes); the X-TRTLLM-Msgpack header tells the worker's
                    # Request.json() to decode with msgspec instead of stdlib json.
                    body = _msgpack_encoder.encode(
                        request.model_dump(mode="json", exclude_unset=True, by_alias=True)
                    )
                    headers = {"Content-Type": "application/json", "X-TRTLLM-Msgpack": "1"}
                else:
                    body = request.model_dump_json(exclude_unset=True, by_alias=True)
                    headers = {"Content-Type": "application/json"}
                if self._request_perf_metrics:
                    headers[RETURN_METRICS_HEADER] = "1"
                headers.update(self._get_request_headers(request))
                try:
                    lines_yielded = 0
                    start_time = get_steady_clock_now_in_seconds()
                    async with self._session.post(
                        url,
                        data=body,
                        headers=headers,
                        timeout=request_timeout,
                    ) as http_response:
                        content_type = http_response.headers.get("Content-Type", "")
                        if self._request_perf_metrics:
                            role = _metrics_phase(self._role)
                            disagg_params = getattr(request, "disaggregated_params", None)
                            request_id = ""
                            if disagg_params is not None:
                                request_id = str(
                                    disagg_params.disagg_request_id
                                    or disagg_params.ctx_request_id
                                    or ""
                                )
                            response_metrics = build_metrics_record_from_headers(
                                http_response.headers,
                                role,
                                request_id=request_id,
                            )
                            if hooks and response_metrics:
                                hooks.on_perf_metrics(
                                    server,
                                    role,
                                    response_metrics,
                                )
                        if is_stream and "text/event-stream" not in content_type:
                            # A streaming request answered with anything but an
                            # event-stream is a hard upstream rejection -- a 4xx/5xx
                            # with a JSON error body -- not a stream that lost its
                            # framing. Surface the status and a bounded slice of the
                            # body (same shape as the non-streaming error path below)
                            # so the callsite that decides retries can tell a
                            # deterministic 400 from a transport flake. The bare
                            # AssertionError this replaces carried neither, and it
                            # fired from inside _response_generator ahead of that
                            # generator's finalizing `finally`, leaking the request;
                            # raised here, ownership stays with the `finally` below.
                            error_body = await http_response.text()
                            raise aiohttp.ClientResponseError(
                                http_response.request_info,
                                http_response.history,
                                status=http_response.status,
                                message=f"{http_response.reason}: {error_body[:2048]}",
                                headers=http_response.headers,
                            )
                        if not is_stream and "text/event-stream" in content_type:
                            raise ValueError(
                                "Received an event-stream although request stream was False"
                            )
                        if is_stream:
                            # do NOT return generator directly here or the response will go
                            # out of scope and get destroyed
                            #
                            # Draining _response_generator hands finalization to its
                            # `finally`; record the handoff so the `finally` below
                            # does not finalize the same request a second time.
                            stream_finalize_delegated = True
                            async for line in self._response_generator(
                                request, http_response, start_time, server, hooks, req_id
                            ):
                                lines_yielded += 1
                                yield line
                            # don't finish the request here since the response generator is not done yet
                        else:
                            if http_response.status >= 400:
                                error_body = await http_response.text()
                                raise aiohttp.ClientResponseError(
                                    http_response.request_info,
                                    http_response.history,
                                    status=http_response.status,
                                    message=f"{http_response.reason}: {error_body[:2048]}",
                                    headers=http_response.headers,
                                )
                            response_dict = await http_response.json()
                            # yield here since python forbids return statements in async generators
                            yield response_dict
                            # Settle inline after the successful response. The
                            # flag flips before the await: should a cancellation
                            # land while the finish call is in flight, the
                            # `finally` must not start a second settlement --
                            # one finish per request, even one cut short, beats
                            # a double-decrement of the router's load counts.
                            request_settled = True
                            await self._finish_request(request, req_id=req_id)
                            self._metrics_collector.complete_latency_seconds.observe(
                                get_steady_clock_now_in_seconds() - start_time
                            )
                    break  # break and skip retries if the whole response is processed without exception
                except (aiohttp.ClientError, OSError) as e:
                    if lines_yielded > 0:
                        logger.error(
                            f"Client error to {url}: {e} - cannot retry since {lines_yielded} lines were yielded",
                            traceback.format_exc(),
                        )
                        raise
                    if _is_worker_rejection(e):
                        raise
                    # Selective retry budget: ServerDisconnectedError and
                    # ConnectionResetError are transient TCP races (typically at
                    # burst start when client keepalive vs server keepalive race).
                    # Give them an extended retry budget while preserving the
                    # original fail-fast for genuine upstream errors.
                    is_transient_tcp = isinstance(
                        e,
                        (aiohttp.ServerDisconnectedError, ConnectionResetError),
                    )
                    effective_max = self._max_retries
                    if is_transient_tcp:
                        effective_max = max(self._max_retries, _TRANSIENT_TCP_BUDGET)
                    if attempt >= effective_max:
                        logger.error(
                            f"Client error to {url}: {e} - last retry {attempt} of {effective_max}"
                            "failed",
                            traceback.format_exc(),
                        )
                        raise
                    logger.error(
                        f"{self._role} client error to {url}: {e} - retry {attempt} of {effective_max}",
                        traceback.format_exc(),
                    )
                    await asyncio.sleep(self._retry_interval_sec)
                    self._metrics_collector.retry_requests.inc()
                except Exception as e:
                    logger.error(
                        f"Unexpected error while processing {self._role} request to {url}: {e}"
                    )
                    raise
        finally:
            # Reached on every exit of this generator. A streaming request that
            # delegated to _response_generator (clean stream or mid-stream
            # failure) is already settled there; a non-streaming request that
            # reached its inline settle above is too. What is left is a request
            # that exited with nothing having settled it:
            #   - streaming and undelegated: it failed before the stream began
            #     (a non-event-stream response, or a transport failure that
            #     exhausted retries) and raised out of the loop still routed;
            #   - non-streaming and unsettled: it failed or was cancelled before
            #     the inline settle. Cancellation is the load-bearing case: a
            #     downstream client disconnect becomes a cancel of the in-flight
            #     POST (see openai_disagg_server's disconnect watch), aiohttp's
            #     BaseException cleanup closes the worker connection rather than
            #     pooling it, and that socket close is what lets the worker's
            #     own disconnect poller abort the engine-side prefill. It also
            #     covers a caller that abandons the generator after an error in
            #     its own frame -- _send_request closes us deterministically for
            #     that.
            # Settle it once, as a failure, so the router load count and routing
            # entry do not leak.
            undelegated_stream = is_stream and not stream_finalize_delegated
            unsettled_nonstream = not is_stream and not request_settled
            if undelegated_stream or unsettled_nonstream:
                self._metrics_collector.error_requests.inc()
                await self._finish_request(request, success=False, req_id=req_id)

    async def _response_generator(
        self,
        request: UCompletionRequest,
        http_response: aiohttp.ClientResponse,
        start_time: float,
        server: str,
        hooks: Optional[ResponseHooks] = None,
        req_id: Optional[int] = None,
    ) -> AsyncGenerator[Any, None]:
        # _post_with_retry has already confirmed this is a streaming request
        # answered with an event-stream and owns the failure path until it
        # delegates here. The content-type assertion that used to stand at the
        # top of this generator is gone: it raised a diagnostic-free
        # AssertionError *before* the try below, so this finalizing `finally`
        # never ran and the request leaked. Nothing runs ahead of the try now,
        # so once the caller delegates, this `finally` is guaranteed to settle
        # the request exactly once.
        success = True
        try:
            last_token_time = start_time
            chunk_count = 0
            marker = f"event: {SSE_METRICS_EVENT}\n".encode()
            event_delimiter = b"\n\n"
            pending = b""
            metrics_event = b""
            async for chunk in http_response.content.iter_any():
                now_time = get_steady_clock_now_in_seconds()
                if chunk:
                    if chunk_count == 0 and hooks:
                        hooks.on_first_token(server, request)
                    latency = now_time - last_token_time
                    metric = (
                        self._metrics_collector.first_token_latency_seconds
                        if chunk_count == 0
                        else self._metrics_collector.per_token_latency_seconds
                    )
                    metric.observe(latency)
                    chunk_count += 1
                    last_token_time = now_time

                if not self._request_perf_metrics:
                    yield chunk
                    await asyncio.sleep(0)
                    continue

                if metrics_event:
                    metrics_event += chunk
                    continue
                pending += chunk
                marker_index = pending.find(marker)
                if marker_index >= 0:
                    if marker_index:
                        yield pending[:marker_index]
                    metrics_event = pending[marker_index:]
                    pending = b""
                    continue
                event_end = pending.rfind(event_delimiter)
                if event_end >= 0:
                    emit_size = event_end + len(event_delimiter)
                    yield pending[:emit_size]
                    pending = pending[emit_size:]
                await asyncio.sleep(0)

            if pending:
                yield pending
            if hooks and metrics_event:
                data_prefix = b"data: "
                data_index = metrics_event.find(data_prefix)
                if data_index >= 0:
                    data = metrics_event[data_index + len(data_prefix) :].split(b"\n", 1)[0]
                    try:
                        headers = json.loads(data)
                        metrics = build_metrics_record_from_headers(
                            headers, _metrics_phase(self._role)
                        )
                    except (TypeError, ValueError) as error:
                        logger.warning(f"Ignoring malformed perf metrics event: {error}")
                        metrics = None
                    if metrics:
                        hooks.on_perf_metrics(server, _metrics_phase(self._role), metrics)

            if hooks:
                hooks.on_resp_done(server, request, None)
            self._metrics_collector.complete_latency_seconds.observe(
                get_steady_clock_now_in_seconds() - start_time
            )
        except aiohttp.ClientError as e:
            # a client error is expected when the response stream is done if the connector has close=True
            logger.error(f"{self._role} client {server} error: {e}")
            self._metrics_collector.error_requests.inc()
            success = False
            raise
        except Exception:
            self._metrics_collector.error_requests.inc()
            success = False
            raise
        finally:
            # finish the request after streaming response is done or error is raised
            await self._finish_request(request, success=success, req_id=req_id)

    async def _finish_request(
        self,
        request: UCompletionRequest,
        success: bool = True,
        req_id: Optional[int] = None,
    ) -> None:
        self._metrics_collector.completed_requests.inc()
        if req_id is None:
            await self._router.finish_request(request, self._session, success=success)
        else:
            await self._router.finish_request(
                request, self._session, success=success, req_id=req_id
            )

    async def shutdown(self) -> None:
        await self._session.close()

    async def check_ready(self) -> Tuple[List[str], List[str]]:
        ready_servers, unready_servers = await OpenAIHttpClient.check_ready_for_servers(
            self._session, self._router.servers
        )
        if ready_servers:
            await self._router.prepare_servers(ready_servers)
        return ready_servers, unready_servers

    @staticmethod
    async def check_ready_for_servers(
        session: aiohttp.ClientSession, servers: List[str]
    ) -> Tuple[List[str], List[str]]:
        async def check_server_ready(server: str) -> bool:
            try:
                url = (
                    f"{server}/health"
                    if server.startswith("http://")
                    else f"http://{server}/health"
                )
                async with session.get(url) as response:
                    return response.status == 200
            except Exception:
                return False

        servers_ready = await asyncio.gather(*[check_server_ready(server) for server in servers])
        return [server for server, ready in zip(servers, servers_ready) if ready], [
            server for server, ready in zip(servers, servers_ready) if not ready
        ]
