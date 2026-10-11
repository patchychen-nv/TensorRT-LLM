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
#!/usr/bin/env python

# yapf: disable
import asyncio
import functools
import json
import signal
import socket
import traceback
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Callable, List, Optional

import aiohttp
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import ValidationError

from tensorrt_llm.executor import CppExecutorError
from tensorrt_llm.executor.executor import CppExecutorError
from tensorrt_llm.executor.utils import (CONTEXT_LENGTH_EXCEEDED_CODE,
                                         is_context_length_exceeded_message)
from tensorrt_llm.llmapi import tracing
from tensorrt_llm.llmapi.disagg_utils import (DisaggServerConfig,
                                              MetadataServerConfig, ServerRole)
from tensorrt_llm.logger import logger
from tensorrt_llm.serve.anthropic_adapter import (AnthropicRequestError,
                                                  AnthropicResponseError,
                                                  anthropic_error_response,
                                                  convert_anthropic_request,
                                                  convert_chat_response,
                                                  reframe_openai_stream)
from tensorrt_llm.serve.anthropic_protocol import (AnthropicCountTokensRequest,
                                                   AnthropicMessagesRequest)
from tensorrt_llm.serve.cluster_storage import (
    HttpClusterStorageServer, create_cluster_storage,
    validate_http_cluster_storage_scope)
from tensorrt_llm.serve.conversation_id import resolve_request_conversation_id
from tensorrt_llm.serve.disagg_coordinator import (CoordinatorClient,
                                                   DisaggCoordinatorService)
from tensorrt_llm.serve.openai_client import OpenAIClient, OpenAIHttpClient
from tensorrt_llm.serve.openai_disagg_service import (
    OpenAIDisaggregatedService, ResponseHooks)
from tensorrt_llm.serve.openai_protocol import (
    ChatCompletionRequest, ChatCompletionResponse, CompletionRequest,
    ErrorResponse, ResponsesRequest, UCompletionRequest, UCompletionResponse,
    ensure_request_chat_template_allowed)
from tensorrt_llm.serve.perf_metrics import (DisaggPerfMetricsCollector,
                                             PerfMetricsJsonlWriter,
                                             PerfMetricsMiddleware,
                                             combine_disagg_metrics)
from tensorrt_llm.serve.request_trace import (RequestTraceHandle,
                                              RequestTraceWriter,
                                              request_trace_dir_from_env)
from tensorrt_llm.serve.responses_utils import (RelayedResponseSnapshot,
                                                ServerArrivalTimeMiddleware,
                                                context_length_exceeded_stream,
                                                get_steady_clock_now_in_seconds,
                                                guard_responses_stream)
from tensorrt_llm.serve.router import Router
from tensorrt_llm.version import __version__ as VERSION

# yapf: enale
_LOG_CONTROL_CHARACTERS = {
    code: f"\\x{code:02x}"
    for code in (*range(32), 127)
}

# nginx's "client closed request": what a request whose client hung up before
# the response is answered with.
_CLIENT_CLOSED_REQUEST = 499


def _error_type(status_code: int) -> str:
    """The error ``type`` the OpenAI and Anthropic shapes share for a status."""
    return "invalid_request_error" if 400 <= status_code < 500 else "api_error"


class RawRequestResponseHooks(ResponseHooks):
    def __init__(self, raw_req: Request, queue_latency_metric,
                 collect_perf_metrics: bool):
        self.raw_req = raw_req
        self.queue_latency_metric = queue_latency_metric
        self.collect_perf_metrics = collect_perf_metrics
        self.ctx_server = ""
        self.gen_server = ""
        self.request_id = ""
        self.disagg_request_id = None
        self.request_arrival_time = raw_req.state.server_arrival_time
        self.server_first_token_time = 0
        self.ctx_dispatch_time = 0
        self.ctx_metrics = None
        self.gen_metrics = None

    def on_req_begin(self, request: UCompletionRequest):
        params = request.disaggregated_params
        if params is not None:
            self.disagg_request_id = params.disagg_request_id
            request_id = params.disagg_request_id or params.ctx_request_id
            self.request_id = str(request_id or "")
        self.queue_latency_metric.observe(
            get_steady_clock_now_in_seconds() - self.request_arrival_time)

    def on_disagg_request_id(self, disagg_request_id: int):
        self.disagg_request_id = disagg_request_id
        self.request_id = str(disagg_request_id)

    def on_ctx_dispatch(self, request: UCompletionRequest):
        self.ctx_dispatch_time = get_steady_clock_now_in_seconds()

    def on_perf_metrics(self, server: str, role: str, metrics: dict):
        if role == "ctx":
            self.ctx_server = server
            self.ctx_metrics = metrics
        elif role == "gen":
            self.gen_server = server
            self.gen_metrics = metrics

    def on_ctx_resp(self, ctx_server: str, response: UCompletionResponse):
        self.ctx_server = ctx_server

    def on_first_token(
            self, gen_server: str, request: UCompletionRequest,
            response: UCompletionResponse = None):
        self.gen_server = gen_server
        self.server_first_token_time = get_steady_clock_now_in_seconds()

    def on_resp_done(
            self, gen_server: str, request: UCompletionRequest,
            response: UCompletionResponse = None):
        self.gen_server = gen_server
        if not self.collect_perf_metrics:
            return
        disagg_phase = {
            "ctx_server": self.ctx_server,
            "gen_server": self.gen_server,
            "timing_metrics": {
                "arrival_time": self.request_arrival_time,
                "last_token_time": get_steady_clock_now_in_seconds(),
                "server_arrival_time": self.request_arrival_time,
                "ctx_dispatch_time": self.ctx_dispatch_time or None,
                "server_first_token_time": self.server_first_token_time or None,
            },
        }
        self.raw_req.state.perf_metrics_records.append(
            combine_disagg_metrics(
                self.request_id,
                disagg_phase,
                self.ctx_metrics,
                self.gen_metrics,
                disagg_request_id=self.disagg_request_id,
            ))


def _upstream_error_message(error: aiohttp.ClientResponseError) -> str:
    """Recover the worker's own error text without leaking the worker.

    str(ClientResponseError) embeds the full upstream URL, so returning it
    verbatim tells every client the internal host and port of a context worker.
    The worker is itself an Anthropic-shaped server, so its body is usually an
    error envelope; lifting the inner message out also avoids handing the
    client an envelope nested inside an envelope, which no SDK will unwrap.
    """
    raw = error.message or ""
    start = raw.find("{")
    if start != -1:
        try:
            payload = json.loads(raw[start:])
        except ValueError:
            payload = None
        if isinstance(payload, dict):
            inner = payload.get("error")
            if isinstance(inner, dict) and inner.get("message"):
                return str(inner["message"])
            if payload.get("message"):
                return str(payload["message"])
    # Not a body this server recognises. Report the status only: anything else
    # in `raw` may still carry the upstream URL.
    return f"context worker rejected the request with status {error.status}"


def _set_disagg_ids(hooks: "RawRequestResponseHooks") -> None:
    """Copy the join key onto whatever handle this request carries.

    Reached through the request rather than a handle the caller holds. On the
    Anthropic route the id is minted inside the wrapped chat entry point, which
    by the ownership rule is the half that owns no handle, while the handle that
    will be written belongs to the outer handler. Both halves see the same
    Request, so that is where they meet.

    Safe to call as soon as the entry point returns: the orchestrator allocates
    the id and finishes the context phase before handing back the generator
    (openai_disagg_service.py:155 and :185), so a streaming request already has
    its final id here.
    """
    handle = getattr(hooks.raw_req.state, "request_trace_handle", None)
    if handle is not None:
        handle.set_ids(disagg_request_id=hooks.disagg_request_id)


async def _iterate(frames: List[bytes]) -> AsyncGenerator[bytes, None]:
    """Serve frames already built in full as a response body."""
    for frame in frames:
        yield frame


class _DownstreamClientDisconnected(Exception):
    """The downstream client hung up while its request was still being served.

    Internal control flow only: raised by the disconnect watch so "client gone"
    becomes a normal, fully finalized exit of the request pipeline instead of a
    response nobody reads -- and, before this existed, instead of a 650K-token
    prefill grinding on a context worker for a client that left minutes ago.
    """


class OpenAIDisaggServer:
    def __init__(self,
                 config: DisaggServerConfig,
                 req_timeout_secs: int = 180,
                 server_start_timeout_secs: int = 180,
                 metadata_server_cfg: Optional[MetadataServerConfig] = None,
                 metrics_interval_secs: int = 0,
                 coordinator_url: Optional[str] = None):
        self._config = config
        self._req_timeout_secs = req_timeout_secs
        self._server_start_timeout_secs = server_start_timeout_secs
        self._metadata_server_cfg = metadata_server_cfg
        self._metrics_interval_secs = metrics_interval_secs
        self._allow_request_chat_template = getattr(
            config, "allow_request_chat_template", False)
        # When set, this is a forked worker: routing/readiness are delegated to
        # the coordinator at coordinator_url (CoordinatorClient). Otherwise this
        # process owns the routers + cluster state (DisaggCoordinatorService).
        self._coordinator_url = coordinator_url

        self._perf_metrics_collector = DisaggPerfMetricsCollector(
            config.perf_metrics_max_requests)
        self._expose_perf_metrics = config.return_perf_metrics
        self._collect_perf_metrics = (
            config.return_perf_metrics
            or config.perf_metrics_output_dir is not None)
        self._perf_metrics_writer = PerfMetricsJsonlWriter(
            config.perf_metrics_output_dir, "disagg")
        # The frontend is the only place a disaggregated deployment sees what the
        # client actually sent: the workers get a rewritten body with no client
        # headers, and the generation worker gets token ids rather than messages.
        self._request_trace = RequestTraceWriter(request_trace_dir_from_env())

        self._disagg_cluster_storage = None
        if config.disagg_cluster_config:
            validate_http_cluster_storage_scope(
                config.disagg_cluster_config.cluster_uri, config.hostname)
            self._disagg_cluster_storage = create_cluster_storage(
                config.disagg_cluster_config.cluster_uri,
                config.disagg_cluster_config.cluster_name)
        # The server doesn't build routers -- the coordinator object does:
        # DisaggCoordinatorService (owner) or CoordinatorClient (delegating). The
        # server just reads .ctx_router / .gen_router off whichever it holds.
        if self._coordinator_url:
            self._coordinator = CoordinatorClient(
                self._coordinator_url, self._config, metadata_server_cfg,
                request_timeout_s=self._req_timeout_secs,
                startup_timeout_s=self._server_start_timeout_secs)
        else:
            self._coordinator = DisaggCoordinatorService(
                self._config, self._create_client,
                metadata_config=self._metadata_server_cfg,
                server_preparation_func=self._sync_server_clock,
                server_start_timeout_secs=self._server_start_timeout_secs)
        self._ctx_router = self._coordinator.ctx_router
        self._gen_router = self._coordinator.gen_router

        self._service = OpenAIDisaggregatedService(
            self._config, self._coordinator, self._create_client,
            req_timeout_secs=self._req_timeout_secs)

        try:
            otlp_cfg = config.otlp_config
            if otlp_cfg and otlp_cfg.otlp_traces_endpoint:
                tracing.init_tracer("trt.llm", otlp_cfg.otlp_traces_endpoint)
                logger.info(
                    f"Initialized OTLP tracer successfully, endpoint: {otlp_cfg.otlp_traces_endpoint}"
                )
        except Exception as e:
            logger.error(f"Failed to initialize OTLP tracer: {e}")


        @asynccontextmanager
        async def lifespan(app) -> None:
            # The cluster manager (via setup) owns server preparation + monitoring.
            await self._perf_metrics_writer.start()
            await self._request_trace.start()
            await self._service.setup()
            yield
            await self._service.teardown()
            await self._request_trace.close()
            await self._perf_metrics_writer.close()

        self.app = FastAPI(lifespan=lifespan)

        if self._collect_perf_metrics:
            self.app.add_middleware(
                PerfMetricsMiddleware,
                expose_headers=self._expose_perf_metrics,
                writer=self._perf_metrics_writer)
        self.app.add_middleware(ServerArrivalTimeMiddleware)

        # Log request-body validation failures so a client/server schema mismatch
        # shows up server-side. Throttled (first, then every 1000th) to avoid
        # flooding the event loop when every request fails identically.
        self._val_err_n = 0
        @self.app.exception_handler(RequestValidationError)
        async def validation_exception_handler(request: Request, exc):
            await self._request_trace.on_rejected(request, exc.errors())
            self._perf_metrics_collector.validation_exceptions.inc()
            # Anthropic clients parse the error envelope, so every /v1/messages
            # route must fail in that shape rather than the generic one below.
            if request.url.path.startswith("/v1/messages"):
                return anthropic_error_response(str(exc),
                                                "invalid_request_error", 400)
            self._val_err_n += 1
            if self._val_err_n == 1 or self._val_err_n % 1000 == 0:
                try:
                    errs = exc.errors()
                    # Compact: [{loc, type, msg}] -- drops the (large) echoed input.
                    brief = [{"loc": e.get("loc"), "type": e.get("type"),
                              "msg": e.get("msg")} for e in errs][:8]
                except Exception:  # noqa: BLE001
                    brief = str(exc)[:500]
                method = request.method.translate(_LOG_CONTROL_CHARACTERS)
                path = request.url.path.translate(_LOG_CONTROL_CHARACTERS)
                logger.warning(
                    f"[validation] {method} {path} 400 "
                    f"(n={self._val_err_n}): {brief}")
            return JSONResponse(status_code=400, content={"error": str(exc)})

        self.register_routes()

    def _create_client(self, router: Router, role: ServerRole, max_retries: int = 1) -> OpenAIClient:
        async def disagg_id_generator():
            return await self._coordinator.get_disagg_request_id()
        client = OpenAIHttpClient(
            router, role, self._req_timeout_secs, max_retries,
            disagg_id_generator=disagg_id_generator,
            request_perf_metrics=self._collect_perf_metrics,
            internal_disagg_auth_key=self._config.internal_request_auth_key)
        return client

    def register_routes(self):
        # The disagg service owns only the request-serving endpoints (/v1/*) and
        # perf metrics. Readiness / cluster topology are the coordinator's state,
        # so /health and /cluster_info hook straight to self._coordinator.
        self.app.add_api_route("/v1/completions", self._wrap_entry_point(self._service.openai_completion, CompletionRequest), methods=["POST"])
        self.app.add_api_route("/v1/chat/completions", self._wrap_entry_point(self._service.openai_chat_completion, ChatCompletionRequest), methods=["POST"])
        self.app.add_api_route("/v1/messages", self.anthropic_messages, methods=["POST"])
        self.app.add_api_route("/v1/messages/count_tokens", self.anthropic_count_tokens, methods=["POST"])
        self.app.add_api_route("/v1/responses", self._wrap_entry_point(self._service.openai_responses, ResponsesRequest), methods=["POST"])
        # GET, and forwarded to a worker rather than answered from config:
        # the orchestrator has no model of its own and DisaggServerConfig
        # carries no model name. Clients that do model discovery -- Codex
        # among them -- got a 404 here and had to be told the name out of
        # band.
        self.app.add_api_route("/v1/models", self.get_model, methods=["GET"])
        self.app.add_api_route("/health", self.health, methods=["GET"])
        self.app.add_api_route("/cluster_info", self.cluster_info, methods=["GET"])
        self.app.add_api_route("/version", self.version, methods=["GET"])
        # import prometheus_client lazily to break the `set_prometheus_multiproc_dir`
        from prometheus_client import (CollectorRegistry, make_asgi_app,
                                       multiprocess)
        registry = CollectorRegistry()
        multiprocess.MultiProcessCollector(registry)
        self.app.mount("/prometheus/metrics", make_asgi_app(registry=registry))
        # Single-process (local coordinator): mount the in-process HTTP cluster
        # storage routes on this app. In worker mode the coordinator is remote and
        # owns those routes (CoordinatorClient has no cluster_storage).
        cluster_storage = getattr(self._coordinator, "cluster_storage", None)
        if isinstance(cluster_storage, HttpClusterStorageServer):
            cluster_storage.add_routes(self.app)
        elif (isinstance(self._coordinator, CoordinatorClient)
              and isinstance(self._disagg_cluster_storage,
                             HttpClusterStorageServer)):
            # Keep the configured public cluster_uri valid in fleet mode while
            # the coordinator remains the sole owner of the HTTP storage state.
            for path, method in (("/set", "POST"), ("/get", "GET"),
                                 ("/delete", "DELETE"), ("/expire", "GET"),
                                 ("/get_prefix", "GET")):
                self.app.add_api_route(path,
                                       self._proxy_cluster_storage_request,
                                       methods=[method])

    async def _proxy_cluster_storage_request(self,
                                             raw_req: Request) -> Response:
        try:
            body, status, content_type = (
                await self._coordinator.proxy_cluster_storage_request(
                    raw_req.method, raw_req.url.path,
                    list(raw_req.query_params.multi_items()),
                    await raw_req.body(), raw_req.headers.get("Content-Type")))
        except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as e:
            logger.warning(f"Failed to proxy cluster storage request: {e}")
            return JSONResponse(status_code=502,
                                content={"error": "coordinator unavailable"})
        headers = {"Content-Type": content_type} if content_type else None
        return Response(content=body, status_code=status, headers=headers)

    @staticmethod
    def _extract_conversation_id(req: UCompletionRequest, raw_req: Request):
        """Populate conversation_params.conversation_id from headers or known body fields.

        Body ``conversation_params.conversation_id`` is canonical. Headers are
        used when the body does not provide one, and the client-native body
        fields (Codex's ``prompt_cache_key`` / ``client_metadata.session_id``)
        when neither does.
        """
        resolve_request_conversation_id(req, raw_req.headers)

    async def _watch_client_disconnect(self, raw_req: Request,
                                       stop: asyncio.Event) -> None:
        """Return when the downstream client is gone, or once ``stop`` is set.

        Nothing in the pinned stack asks the transport on our behalf: FastAPI's
        request_response runs the handler to completion whatever the socket
        does, and uvicorn merely flags cycle.disconnected for whoever polls.
        The aggregated server polls per engine promise
        (openai_server.await_disconnected, same 1 Hz cadence); this proxy holds
        no promise, so the poll lives here and its completion means "abort the
        pipeline".

        ``stop`` is checked after every poll because cancelling this task is
        not enough on its own. Starlette's is_disconnected() reads receive()
        inside an anyio cancel scope that it cancels itself; a Task.cancel()
        that lands while that scope's cancellation is in flight is merged into
        it, and the scope swallows the one resulting CancelledError as its
        own. A pipeline that settles within a poll -- a rejection raised before
        any upstream I/O does, every time -- hit exactly that window, and the
        handler reaping this watch then sat on its answer until the client gave
        up and disconnected.
        """
        while not await raw_req.is_disconnected():
            if stop.is_set():
                return
            await asyncio.sleep(1)

    async def _serve_until_client_disconnect(
            self, entry_point: Callable, req: UCompletionRequest,
            hooks: ResponseHooks, raw_req: Request):
        """Run the entry point, aborting the whole pipeline if the client leaves.

        The pre-response window is the orchestrator's blind spot: a
        context-first disagg request spends its entire prefill -- minutes, for
        a cold 100-700K-token conversation -- inside `await entry_point(...)`,
        blocked on the context worker's non-streaming POST, and a client that
        times out and hangs up during it used to change nothing. Once a
        streaming response is returned, StreamingResponse's own
        listen_for_disconnect owns the job; that is why the watch is scoped to
        this call and cancelled on the way out instead of polling receive()
        concurrently with it.

        Cancelling entry_task is what actually aborts the upstream work: the
        cancellation lands in the in-flight ctx/gen POST, aiohttp's
        BaseException cleanup closes that connection (it is never returned to
        the pool), the worker sees its client vanish, and the worker's own
        await_disconnected poller aborts the engine request. The pipeline's
        finalizers -- router load counts, routing entries, client metrics --
        run under this single plain-asyncio cancel, so they complete; the
        settlement is awaited before the disconnect is reported.
        """
        entry_task = asyncio.create_task(entry_point(req, hooks))
        stop_watch = asyncio.Event()
        watch_task = asyncio.create_task(
            self._watch_client_disconnect(raw_req, stop_watch))
        try:
            done, _ = await asyncio.wait({entry_task, watch_task},
                                         return_when=asyncio.FIRST_COMPLETED)
            if entry_task in done:
                # Covers the race where the client vanished in the same tick:
                # the response already exists, so hand it back -- uvicorn
                # discards the send, and a streaming body is torn down by
                # StreamingResponse's own disconnect handling.
                return entry_task.result()
            if watch_task.exception() is not None:
                # The watch itself broke, which says nothing about the client.
                # Degrading to the old serve-to-completion behavior beats
                # cancelling a healthy request over a watcher bug.
                logger.error(
                    f"Disconnect watch failed; serving to completion: "
                    f"{watch_task.exception()!r}")
                return await entry_task
            entry_task.cancel()
            # Wait for the pipeline to settle before answering: its finalizers
            # close the upstream sockets (the abort signal the workers act on)
            # and release the router bookkeeping. Bounded: a cancelled aiohttp
            # await raises immediately and the routers' finish paths carry
            # their own timeouts.
            await asyncio.wait({entry_task})
            if entry_task.cancelled():
                raise _DownstreamClientDisconnected()
            # The cancel arrived after the pipeline finished on its own; fall
            # through to the result -- or to its genuine error -- exactly as if
            # the race had gone the other way.
            return entry_task.result()
        except asyncio.CancelledError:
            # This handler itself is being cancelled (shutdown): take the
            # pipeline down with it and let the cancellation keep unwinding.
            entry_task.cancel()
            raise
        finally:
            # Both signals: the cancel ends a watch that is sleeping between
            # polls at once, and the stop flag ends one whose cancel Starlette
            # swallowed mid-poll (see _watch_client_disconnect).
            stop_watch.set()
            watch_task.cancel()
            # Reap the watcher so a failure in it is consumed here rather than
            # logged at GC as "Task exception was never retrieved".
            try:
                await watch_task
            except (asyncio.CancelledError, Exception):
                pass
            # A failed pipeline is re-raised from this frame, so the error's
            # traceback holds the frame. If the frame still held the entry
            # task (directly or in asyncio.wait's sets), the task's stored
            # error would close a frame -> task -> error -> traceback -> frame
            # cycle that only the cyclic GC frees, with the request attached.
            entry_task = watch_task = done = _ = None

    def _wrap_entry_point(self, entry_point: Callable, request_type: type = UCompletionRequest) -> Callable:
        # Bind the concrete request model per route so FastAPI validates against it.
        # The bare Union UCompletionRequest (no discriminator) makes Pydantic try
        # CompletionRequest first and 400 every chat body, so override the wrapper's
        # annotation with request_type (as openai_server.py does).
        @tracing.trace_span("disaggregated_request")
        async def wrapper(req: request_type, raw_req: Request) -> Response:
            trace_handle = await self._request_trace.on_request(raw_req)
            # Unset until the request is past its local checks; the failure
            # exits below stamp the trace with its ids only once it exists.
            hooks: Optional[RawRequestResponseHooks] = None
            try:
                self._perf_metrics_collector.total_requests.inc()
                if req.stream:
                    self._perf_metrics_collector.stream_requests.inc()
                else:
                    self._perf_metrics_collector.nonstream_requests.inc()
                try:
                    ensure_request_chat_template_allowed(
                        req, self._allow_request_chat_template)
                except ValueError as e:
                    raise HTTPException(status_code=400, detail=str(e)) from e
                self._extract_conversation_id(req, raw_req)
                hooks = RawRequestResponseHooks(
                    raw_req, self._perf_metrics_collector.queue_latency_seconds,
                    self._collect_perf_metrics)
                # Raced against the client's own departure: with a context-first
                # disagg request this await holds the entire prefill, and a
                # client that hangs up during it must take the context-side
                # engine request down with it (see _serve_until_client_disconnect).
                response_or_generator = await self._serve_until_client_disconnect(
                    entry_point, req, hooks, raw_req)
                self._perf_metrics_collector.total_responses.inc()
                _set_disagg_ids(hooks)
                if req.stream:
                    stream = response_or_generator
                    if isinstance(req, ResponsesRequest):
                        # Only the Responses protocol: a stream that stops
                        # before response.completed loses everything it
                        # produced, because that event is the only place the
                        # full text is repeated. The other protocols carry
                        # their content entirely in deltas and end on a
                        # sentinel, so a truncation there is already visible.
                        #
                        # The relay never assembles the response, but the
                        # worker's opening `response.created` carries the
                        # whole snapshot, which is all `response.failed`
                        # needs. A cut stream therefore ends as the worker
                        # ends one of its own: `error`, then
                        # `response.failed`. Failures that never reached a
                        # worker have no snapshot and end on the bare
                        # `error`. Either way the events forwarded are
                        # counted, so the sequence numbers are exact.
                        snapshot = RelayedResponseSnapshot()
                        stream = guard_responses_stream(
                            snapshot.observe(stream),
                            snapshot.failed_events,
                            on_termination=functools.partial(
                                self._request_trace.note_stream_termination,
                                trace_handle),
                        )
                    return StreamingResponse(
                        content=self._request_trace.wrap_stream(
                            stream, trace_handle),
                        media_type="text/event-stream")
                # by_alias: the wire name, not the python one -- `schema` for
                # a json_schema text format, which pydantic only lets a model
                # hold as `schema_`. The worker sent `schema`; re-serialising
                # under the internal name handed the client a field the API
                # does not have, and recorded the same wrong body.
                payload = response_or_generator.model_dump(by_alias=True)
                self._request_trace.on_response(trace_handle, payload=payload)
                return JSONResponse(content=payload)
            except _DownstreamClientDisconnected:
                # A finalized abort, not an error: the pipeline was cancelled,
                # its settlement awaited, and there is nobody left to answer.
                # The trace terminal reuses the vocabulary wrap_stream already
                # records for mid-stream hangups, so a pre-response hangup is
                # searchable under the same status.
                _set_disagg_ids(hooks)
                self._request_trace.on_response(
                    trace_handle, payload=None, status="client_disconnected")
                logger.info(
                    f"{raw_req.client} disconnected before the response; "
                    f"aborted disagg request {hooks.disagg_request_id}")
                # 499 is nginx's "client closed request". Nothing reaches the
                # wire -- uvicorn discards sends once the client is gone -- but
                # middleware and access logs see an honest status instead of a
                # fabricated success or a 500.
                return Response(status_code=_CLIENT_CLOSED_REQUEST)
            except asyncio.CancelledError:
                # The handler itself is being cancelled -- shutdown; a client
                # hangup arrives above as _DownstreamClientDisconnected. Closed
                # in the trace so the request does not read as still in
                # flight, and the cancellation keeps unwinding.
                if hooks is not None:
                    _set_disagg_ids(hooks)
                self._trace_cancelled(trace_handle)
                raise
            except Exception as e:
                # The ids join this record to the workers' logs of the
                # attempt; on /v1/messages they land on the adapter's handle.
                if hooks is not None:
                    _set_disagg_ids(hooks)
                try:
                    # Usually raises; returns a Response for worker errors that
                    # carry a machine-readable code (context_length_exceeded).
                    error_response = self._handle_exception(e)
                except HTTPException as http_error:
                    # Every failure the client is answered for leaves here: a
                    # worker's 4xx/5xx, an internal error, the chat-template
                    # rejection above. FastAPI answers an HTTPException with
                    # {"detail": ...}; that body is what the trace keeps.
                    self._trace_failure(trace_handle, http_error.status_code,
                                        {"detail": http_error.detail})
                    raise
                if error_response is not None:
                    error_body = json.loads(error_response.body)
                    message = str(error_body.get("message", ""))
                    if (isinstance(req, ResponsesRequest) and req.stream
                            and error_body.get("code")
                            == CONTEXT_LENGTH_EXCEEDED_CODE):
                        # A streamed Responses request is failed in-stream, as
                        # OpenAI fails one: the client acts on the terminal
                        # event's code, where a 400 reads as an invalid request
                        # (see context_length_exceeded_stream).
                        return StreamingResponse(
                            content=self._request_trace.wrap_stream(
                                _iterate(
                                    context_length_exceeded_stream(
                                        req, message)), trace_handle),
                            media_type="text/event-stream")
                    # The worker's error envelope, relayed with its code, is
                    # the body the client reads.
                    self._trace_failure(trace_handle,
                                        error_response.status_code, error_body)
                    return error_response
                # Reached only for CppExecutorError, after SIGINT was raised to
                # take the server down. Nothing is returned, which FastAPI
                # answers with a 200 and a JSON null.
                self._request_trace.note_stream_termination(
                    trace_handle, "internal_error", f"{type(e).__name__}: {e}")
                self._request_trace.on_response(trace_handle,
                                                payload=None,
                                                status="error",
                                                http_status=200)
        return wrapper

    def _trace_failure(self,
                       trace_handle: Optional[RequestTraceHandle],
                       status_code: int,
                       body: Any,
                       *,
                       cause: Optional[str] = None,
                       detail: str = "") -> None:
        """Close the trace of a request answered with an error.

        ``body`` is exactly what the client receives -- FastAPI's
        ``{"detail": ...}`` for an HTTPException, an error envelope a route
        builds itself, the framework's plain-text 500 -- and ``status_code``
        is its HTTP status, recorded as ``http_status`` so a worker's 503 stays
        distinguishable from a 500 raised here. ``status`` uses the aggregated
        server's vocabulary: ``rejected_<code>`` below 500, ``error`` from 500
        up. ``cause`` and ``detail`` record what the body does not say, such as
        the exception behind a bare 500. on_response is exactly-once per
        handle, so a later record for the same request is a no-op.
        """
        if cause is not None:
            self._request_trace.note_stream_termination(
                trace_handle, cause, detail)
        self._request_trace.on_response(
            trace_handle,
            payload=body,
            status=(f"rejected_{status_code}"
                    if status_code < 500 else "error"),
            http_status=status_code,
        )

    def _trace_cancelled(self,
                         trace_handle: Optional[RequestTraceHandle]) -> None:
        """Close the trace of a handler cancelled before it answered."""
        self._request_trace.on_response(
            trace_handle,
            payload={
                "error": {
                    "type": "cancelled",
                    "message": "request handler cancelled before a response "
                    "was sent",
                }
            },
            status="cancelled",
        )

    async def anthropic_messages(self, request: AnthropicMessagesRequest,
                                 raw_request: Request) -> Response:
        """Serve Anthropic Messages through the disaggregated chat pipeline.

        Owns the request's trace. The wrapped chat entry point is handed no
        handle on this route, so every exit has to be closed on this side: the
        deliberate answers in _serve_anthropic_messages, and here the exits
        that raise -- a shutdown cancel, or a failure nobody anticipated, which
        FastAPI answers with a bare 500.
        """
        # Before the conversion, and before the wrapped chat entry point hooks
        # the same request again and is handed None for it.
        trace_handle = await self._request_trace.on_request(raw_request)
        try:
            return await self._serve_anthropic_messages(request, raw_request,
                                                        trace_handle)
        except asyncio.CancelledError:
            self._trace_cancelled(trace_handle)
            raise
        except Exception as e:
            # Unanticipated, so FastAPI's error middleware answers it with its
            # plain-text 500.
            self._trace_failure(trace_handle,
                                500,
                                "Internal Server Error",
                                cause="internal_error",
                                detail=f"{type(e).__name__}: {e}")
            raise

    async def _serve_anthropic_messages(
            self, request: AnthropicMessagesRequest, raw_request: Request,
            trace_handle: Optional[RequestTraceHandle]) -> Response:
        """The adapter itself; every answer it returns closes the trace."""

        def _reject(message: str, status_code: int) -> Response:
            # Every deliberate error answer settles the trace on its way out,
            # with the body the client reads.
            response = anthropic_error_response(message,
                                                _error_type(status_code),
                                                status_code)
            self._trace_failure(trace_handle, status_code,
                                json.loads(response.body))
            return response

        try:
            chat_request = convert_anthropic_request(request)
        except (AnthropicRequestError, ValidationError) as e:
            return _reject(str(e), 400)

        try:
            openai_response = await self._wrap_entry_point(
                self._service.openai_chat_completion)(chat_request, raw_request)
        except HTTPException as e:
            return _reject(str(e.detail), e.status_code)

        if isinstance(openai_response, StreamingResponse):
            return StreamingResponse(
                content=self._request_trace.wrap_stream(
                    reframe_openai_stream(openai_response.body_iterator,
                                          model=request.model), trace_handle),
                media_type="text/event-stream",
            )

        status = getattr(openai_response, "status_code", 500)
        if status == _CLIENT_CLOSED_REQUEST:
            # The wrapped entry point's pre-response hangup. It recorded
            # nothing, holding no handle here, so the hangup is recorded under
            # the status the OpenAI routes give it. Nobody is left to read an
            # error envelope; the bare 499 goes back as it came.
            self._request_trace.on_response(trace_handle,
                                            payload=None,
                                            status="client_disconnected")
            return openai_response
        if status != 200:
            try:
                payload = json.loads(openai_response.body)
                message = (payload.get("message") or payload.get("detail")
                           or payload.get("error") or json.dumps(payload))
            except (json.JSONDecodeError, AttributeError, TypeError):
                message = "Internal server error"
            return _reject(str(message), status)

        try:
            chat_response = ChatCompletionResponse(
                **json.loads(openai_response.body))
            anthropic_response = convert_chat_response(chat_response)
        except (AnthropicResponseError, ValidationError, json.JSONDecodeError):
            logger.error(
                "Invalid response from OpenAI chat pipeline:\n"
                f"{traceback.format_exc()}")
            # The upstream body rather than the 500 the client sees: a tool call
            # whose arguments are not a JSON object fails the conversion here and
            # exists nowhere else.
            self._request_trace.on_response(
                trace_handle,
                payload={
                    "upstream_body":
                    openai_response.body.decode("utf-8", "replace")
                },
                status="conversion_error")
            return anthropic_error_response("Internal server error",
                                            "api_error", 500)
        # The wrapped chat entry point holds no handle on this route, so its own
        # on_response was a no-op and this is the only place the Anthropic-shaped
        # body -- the one the client reads -- gets recorded.
        payload = anthropic_response.model_dump(exclude_none=True)
        self._request_trace.on_response(trace_handle, payload=payload)
        return JSONResponse(content=payload)

    async def get_model(self) -> Response:
        """List the served model, asking a context worker for the name.

        Error shape is OpenAI's, not Anthropic's, because this is an OpenAI
        route -- a client that model-discovers here parses OpenAI errors.
        """
        try:
            response = await self._service.get_model()
        except aiohttp.ClientResponseError as error:
            return JSONResponse(
                status_code=error.status or 500,
                content={
                    "object": "error",
                    "message": _upstream_error_message(error),
                    "type": ("invalid_request_error"
                             if 400 <= (error.status or 500) < 500 else "api_error"),
                    "code": error.status or 500,
                },
            )
        except (RuntimeError, ValueError) as error:
            # No worker to ask yet. 503 says "retry", which is true: the answer
            # exists as soon as a context worker registers.
            return JSONResponse(
                status_code=503,
                content={
                    "object": "error",
                    "message": str(error),
                    "type": "api_error",
                    "code": 503,
                },
            )
        return JSONResponse(content=response.model_dump())

    async def anthropic_count_tokens(
            self, request: AnthropicCountTokensRequest) -> Response:
        """Count Anthropic input tokens on a context worker."""
        try:
            response = await self._service.anthropic_count_tokens(request)
        except aiohttp.ClientResponseError as error:
            error_type = ("invalid_request_error"
                          if 400 <= error.status < 500 else "api_error")
            return anthropic_error_response(_upstream_error_message(error),
                                            error_type, error.status or 500)
        except (RuntimeError, ValueError) as error:
            # Raised when the cluster is not ready or has no context worker;
            # 503 tells the caller to retry rather than to fix the request.
            return anthropic_error_response(str(error), "api_error", 503)
        return JSONResponse(content=response.model_dump())

    def _handle_exception(self, exception):
        if isinstance(exception, CppExecutorError):
            logger.error("CppExecutorError: ", traceback.format_exc())
            signal.raise_signal(signal.SIGINT)
        elif isinstance(exception, aiohttp.ClientResponseError):
            self._perf_metrics_collector.http_exceptions.inc()
            status = exception.status or 502
            logger.error(
                f"Upstream HTTP error {status} {exception.message}: ",
                traceback.format_exc())
            # exception.message is the worker's raw response body (up to 2048
            # chars, built in openai_client.post_json/_send_request). Forwarding
            # it verbatim publishes the worker's URL and internals to every
            # caller, so it goes through the same unwrapping the Anthropic
            # route uses. This branch is shared with /v1/completions and
            # /v1/chat/completions, so those get the same treatment.
            message = _upstream_error_message(exception)
            if is_context_length_exceeded_message(message):
                # The worker tagged this rejection with the machine-readable
                # code "context_length_exceeded" (openai_server.py,
                # create_error_response). HTTPException would flatten it to
                # {"detail": message}, dropping the code, so re-emit the
                # worker's error envelope. Detection is by message text, the
                # same way the worker itself detects it: only the string is
                # guaranteed to survive the hops.
                return JSONResponse(status_code=status,
                                    content=ErrorResponse(
                                        message=message,
                                        type="BadRequestError",
                                        code=CONTEXT_LENGTH_EXCEEDED_CODE,
                                    ).model_dump())
            raise HTTPException(status_code=status,
                                detail=message) from exception
        elif isinstance(exception, HTTPException):
            self._perf_metrics_collector.http_exceptions.inc()
            logger.error(f"HTTPException {exception.status_code} {exception.detail}: ", traceback.format_exc())
            raise exception
        elif (isinstance(exception, aiohttp.ClientResponseError)
              and 400 <= exception.status < 500):
            # A worker rejected the request itself - an unsupported tool, a bad
            # parameter. That verdict is about the client's request and would
            # be identical on any worker, so relaying it as a 500 both blames
            # the server for the client's input and throws away the one thing
            # that makes a 4xx useful: the reason. Clients retry a 500, which
            # cannot succeed, and Codex spends its retry budget before
            # reporting a generic failure.
            self._perf_metrics_collector.http_exceptions.inc()
            logger.error(
                f"Worker rejected the request with {exception.status}: {exception.message}"
            )
            raise HTTPException(status_code=exception.status,
                                detail=_upstream_error_message(exception))
        else:
            self._perf_metrics_collector.internal_errors.inc()
            logger.error("Internal server error: ", traceback.format_exc())
            raise HTTPException(status_code=500, detail=f"Internal server error {str(exception)}")


    async def health(self) -> Response:
        if not await self._coordinator.is_ready():
            return Response(status_code=503)
        return Response(status_code=200)

    async def cluster_info(self) -> JSONResponse:
        return JSONResponse(content=await self._coordinator.cluster_info())

    async def version(self) -> JSONResponse:
        return JSONResponse(content={"version": VERSION})

    async def __call__(self, host: str, port: int, sockets: list[socket.socket] | None = None):
        keep_alive_timeout = self._config.server_keep_alive_timeout
        config = uvicorn.Config(self.app,
                                host=host,
                                port=port,
                                log_level=logger.level,
                                timeout_keep_alive=keep_alive_timeout)
        await uvicorn.Server(config).serve(sockets=sockets)

    async def _sync_server_clock(self, server: str):
        """ Sync the ctx/gen server's steady clock with the disagg-server's steady clock (in case NTP service is not running). """
        async def query_steady_clock_offset(session: aiohttp.ClientSession, server_url: str) -> tuple[Optional[float], Optional[float]]:
            try:
                originate_ts = get_steady_clock_now_in_seconds()
                async with session.get(server_url) as response:
                    destination_ts = get_steady_clock_now_in_seconds()
                    if response.status == 200:
                        response_content = await response.json()
                        # Compute the steady clock timestamp difference using the NTP clock synchronization algorithm. https://en.wikipedia.org/wiki/Network_Time_Protocol#Clock_synchronization_algorithm
                        receive_ts = response_content['receive_ts']
                        transmit_ts = response_content['transmit_ts']
                        delay = (destination_ts - originate_ts) - (transmit_ts - receive_ts)
                        offset = ((receive_ts - originate_ts) + (transmit_ts - destination_ts)) / 2
                        return delay, offset
                    else:
                        return None, None
            except Exception:
                return None, None

        async def set_steady_clock_offset(session: aiohttp.ClientSession, server_url: str, offset: float) -> None:
            payload = {"offset": offset}
            async with session.post(server_url, json=payload) as response:
                if response.status != 200:
                    logger.warning(f"Cannot set disagg server steady clock offset for server {server_url}, the perf metrics timestamps could be mis-aligned")

        async def align_steady_clock_offset(session: aiohttp.ClientSession, server_url: str) -> None:
            delay, offset = await query_steady_clock_offset(session, server_url)
            if delay is None or offset is None:
                logger.warning(f"Unable to measure steady clock offset for {server_url}; skipping adjustment")
                return
            logger.info(f'Server: {server_url}, delay: {delay} second, offset: {offset} second')
            # Negate the offset so that worker servers can adjust their steady clock by adding the new offset
            await set_steady_clock_offset(session, server_url, -offset)

        server_scheme = "http://" if not server.startswith("http://") else ""
        server_url = f"{server_scheme}{server}/steady_clock_offset"

        try:
            async with aiohttp.ClientSession(
                connector=aiohttp.TCPConnector(limit=0, limit_per_host=0, force_close=True),
                timeout=aiohttp.ClientTimeout(total=self._req_timeout_secs)) as session:
                await align_steady_clock_offset(session, server_url)
        except (aiohttp.ClientError, OSError) as e:
            logger.warning(f"Unable to align steady clock offset for {server_url}: {e}; skipping adjustment")
