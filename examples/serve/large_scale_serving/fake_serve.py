#!/usr/bin/env python3
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
"""A stand-in for trtllm-serve that answers 200 to everything.

It serves the routes an OpenAIServer and an OpenAIDisaggServer expose, with
bodies shaped like the real ones, so that a gateway's watchdog, a router's
registration and a real client all get something they can parse -- with no
model, no GPU and no engine.

Bodies matter because "200 with an empty body" is not enough for the clients
that actually drive this fleet. A Responses API client asks for SSE and parses
an event sequence; handing it a JSON object makes it hang or error, which looks
like a fleet bug rather than the stub being a stub. So `stream: true` gets a
real event sequence on the streaming routes, and everything else gets a
well-formed object.

Nothing here ever fails: no route 404s, no request 500s, and an unknown path
still answers 200. That is the point -- it isolates whatever you are testing
from the backend.

Every request is printed as it arrives -- time, peer, method, endpoint -- so
that pointing something at this and watching the log tells you what it calls,
in what order, and from where.

Usage:

    ./fake_serve.py --port 8000
    ./fake_serve.py --port 8000 --model GLM-5.2-NVFP4 --tokens 40
    ./fake_serve.py --port 8000 -q          # silent, for load tests

    # Point a gateway at it, or talk to it directly:
    curl localhost:8000/health -i
    curl localhost:8000/v1/responses -d '{"stream": true, "input": "hi"}'
"""

from __future__ import annotations

import argparse
import json
import random
import string
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Iterator, Optional, Tuple

# Routes that stream when the request body says `stream: true`, mapped to the
# event dialect each one speaks. Everything else answers with a single object.
STREAMING_ROUTES = {
    "/v1/responses": "responses",
    "/v1/chat/completions": "chat",
    "/v1/completions": "completions",
    "/v1/messages": "anthropic",
}

ARGS: argparse.Namespace  # set in main()

# Serialises the arrival lines. Without it two threads can interleave mid-line
# and the output stops being greppable under any real concurrency.
_LOG_LOCK = threading.Lock()


def _uid(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def _now() -> int:
    return int(time.time())


def _reply_text() -> str:
    """The text the stub pretends the model produced."""
    if ARGS.reply:
        return ARGS.reply
    # Deterministic length, arbitrary content: callers that measure token
    # counts get a stable number, callers that print it get something legible.
    words = ["ok"] + [
        "".join(random.choices(string.ascii_lowercase, k=4)) for _ in range(max(0, ARGS.tokens - 1))
    ]
    return " ".join(words)


def _usage(out_tokens: int) -> Dict[str, int]:
    return {
        "prompt_tokens": 8,
        "completion_tokens": out_tokens,
        "total_tokens": 8 + out_tokens,
    }


# --------------------------------------------------------------------------
# Non-streaming bodies
# --------------------------------------------------------------------------


def _message_item(text: str) -> Dict[str, Any]:
    return {
        "id": _uid("msg"),
        "type": "message",
        "status": "completed",
        "role": "assistant",
        "content": [{"type": "output_text", "text": text, "annotations": []}],
    }


def _responses_body(req: Dict[str, Any], text: str) -> Dict[str, Any]:
    return {
        "id": _uid("resp"),
        "object": "response",
        "created_at": _now(),
        "status": "completed",
        "model": req.get("model") or ARGS.model,
        "output": [_message_item(text)],
        "parallel_tool_calls": True,
        "tool_choice": req.get("tool_choice", "auto"),
        "tools": req.get("tools", []),
        "usage": {
            "input_tokens": 8,
            "output_tokens": len(text.split()),
            "total_tokens": 8 + len(text.split()),
        },
    }


def _chat_body(req: Dict[str, Any], text: str) -> Dict[str, Any]:
    return {
        "id": _uid("chatcmpl"),
        "object": "chat.completion",
        "created": _now(),
        "model": req.get("model") or ARGS.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "logprobs": None,
                "finish_reason": "stop",
            }
        ],
        "usage": _usage(len(text.split())),
    }


def _completions_body(req: Dict[str, Any], text: str) -> Dict[str, Any]:
    return {
        "id": _uid("cmpl"),
        "object": "text_completion",
        "created": _now(),
        "model": req.get("model") or ARGS.model,
        "choices": [{"index": 0, "text": text, "logprobs": None, "finish_reason": "stop"}],
        "usage": _usage(len(text.split())),
    }


def _anthropic_body(req: Dict[str, Any], text: str) -> Dict[str, Any]:
    return {
        "id": _uid("msg"),
        "type": "message",
        "role": "assistant",
        "model": req.get("model") or ARGS.model,
        "content": [{"type": "text", "text": text}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 8, "output_tokens": len(text.split())},
    }


# --------------------------------------------------------------------------
# Streaming event sequences
# --------------------------------------------------------------------------


def _responses_events(req: Dict[str, Any], text: str) -> Iterator[Tuple[Optional[str], Any]]:
    """The Responses API sequence, in the order responses_utils.py emits it."""
    seq = 0

    def ev(kind: str, **fields) -> Tuple[str, Dict[str, Any]]:
        nonlocal seq
        payload = {"type": kind, "sequence_number": seq, **fields}
        seq += 1
        return kind, payload

    body = _responses_body(req, text)
    item_id = body["output"][0]["id"]

    in_progress = dict(body, status="in_progress", output=[], usage=None)
    yield ev("response.created", response=dict(in_progress, status="queued"))
    yield ev("response.in_progress", response=in_progress)

    opening = {
        "id": item_id,
        "type": "message",
        "status": "in_progress",
        "role": "assistant",
        "content": [],
    }
    yield ev("response.output_item.added", output_index=0, item=opening)
    yield ev(
        "response.content_part.added",
        item_id=item_id,
        output_index=0,
        content_index=0,
        part={"type": "output_text", "text": "", "annotations": []},
    )

    for token in text.split():
        yield ev(
            "response.output_text.delta",
            item_id=item_id,
            output_index=0,
            content_index=0,
            delta=token + " ",
        )

    yield ev(
        "response.output_text.done", item_id=item_id, output_index=0, content_index=0, text=text
    )
    yield ev(
        "response.content_part.done",
        item_id=item_id,
        output_index=0,
        content_index=0,
        part={"type": "output_text", "text": text, "annotations": []},
    )
    yield ev("response.output_item.done", output_index=0, item=body["output"][0])
    yield ev("response.completed", response=body)


def _chat_events(req: Dict[str, Any], text: str) -> Iterator[Tuple[Optional[str], Any]]:
    """Chat Completions chunks: no event names, terminated by [DONE]."""
    base = {
        "id": _uid("chatcmpl"),
        "object": "chat.completion.chunk",
        "created": _now(),
        "model": req.get("model") or ARGS.model,
    }

    def chunk(delta: Dict[str, Any], finish: Optional[str] = None):
        return None, dict(
            base, choices=[{"index": 0, "delta": delta, "logprobs": None, "finish_reason": finish}]
        )

    yield chunk({"role": "assistant", "content": ""})
    for token in text.split():
        yield chunk({"content": token + " "})
    yield chunk({}, finish="stop")
    yield None, "[DONE]"


def _completions_events(req: Dict[str, Any], text: str) -> Iterator[Tuple[Optional[str], Any]]:
    base = {
        "id": _uid("cmpl"),
        "object": "text_completion",
        "created": _now(),
        "model": req.get("model") or ARGS.model,
    }
    for token in text.split():
        yield (
            None,
            dict(
                base,
                choices=[
                    {"index": 0, "text": token + " ", "logprobs": None, "finish_reason": None}
                ],
            ),
        )
    yield (
        None,
        dict(base, choices=[{"index": 0, "text": "", "logprobs": None, "finish_reason": "stop"}]),
    )
    yield None, "[DONE]"


def _anthropic_events(req: Dict[str, Any], text: str) -> Iterator[Tuple[Optional[str], Any]]:
    msg = _anthropic_body(req, text)
    start = dict(msg, content=[], stop_reason=None)
    yield "message_start", {"type": "message_start", "message": start}
    yield (
        "content_block_start",
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": ""},
        },
    )
    for token in text.split():
        yield (
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": token + " "},
            },
        )
    yield "content_block_stop", {"type": "content_block_stop", "index": 0}
    yield (
        "message_delta",
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
            "usage": {"output_tokens": len(text.split())},
        },
    )
    yield "message_stop", {"type": "message_stop"}


STREAM_BUILDERS = {
    "responses": _responses_events,
    "chat": _chat_events,
    "completions": _completions_events,
    "anthropic": _anthropic_events,
}

BODY_BUILDERS = {
    "/v1/responses": _responses_body,
    "/v1/chat/completions": _chat_body,
    "/v1/completions": _completions_body,
    "/v1/messages": _anthropic_body,
}


# --------------------------------------------------------------------------
# HTTP
# --------------------------------------------------------------------------


class Handler(BaseHTTPRequestHandler):
    # Keep-alive, so a client that reuses connections is not paying for a new
    # one per request while measuring something else.
    protocol_version = "HTTP/1.1"
    server_version = "trtllm-fake-serve/1.0"

    # -- plumbing ----------------------------------------------------------

    def log_message(self, fmt: str, *args) -> None:
        # Silenced: BaseHTTPRequestHandler logs from send_response, i.e. once
        # the reply is on its way. For an SSE route that is after the whole
        # stream has drained, which is no use for watching traffic arrive.
        # _announce() below prints on arrival instead.
        pass

    def _announce(self, note: str = "") -> None:
        """Print the request the moment it arrives, before any work."""
        if ARGS.quiet:
            return
        host, port = self.client_address[:2]
        peer = f"{host}:{port}"
        # A gateway or router proxies on the client's behalf, so the socket
        # address is the proxy's. Name the original too when it says who it is.
        forwarded = self.headers.get("X-Forwarded-For")
        if forwarded:
            peer += f" <- {forwarded.split(',')[0].strip()}"
        now = time.time()
        stamp = time.strftime("%H:%M:%S", time.localtime(now))
        line = (
            f"{stamp}.{int(now % 1 * 1000):03d}  {peer:<28}  "
            f"{self.command:<6} {self._path()}{note}\n"
        )
        # One write of one whole line, under the lock, then flushed: a reader
        # tailing the log sees each request as it lands rather than when the
        # block buffer happens to fill.
        with _LOG_LOCK:
            sys.stdout.write(line)
            sys.stdout.flush()

    def _read_body(self) -> Dict[str, Any]:
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            length = 0
        if length <= 0:
            return {}
        try:
            return json.loads(self.rfile.read(length) or b"{}")
        except (ValueError, OSError):
            # A body we cannot parse is still a request we answer 200 to.
            return {}

    def _send_json(self, obj: Any, status: int = 200) -> None:
        blob = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(blob)))
        self.end_headers()
        self.wfile.write(blob)

    def _send_empty(self, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _send_text(self, text: str, ctype: str = "text/plain") -> None:
        blob = text.encode()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(blob)))
        self.end_headers()
        self.wfile.write(blob)

    def _send_sse(self, events: Iterator[Tuple[Optional[str], Any]]) -> None:
        """Stream events as SSE.

        Chunked rather than close-delimited: HTTP/1.1 forbids a body with
        neither a length nor chunking, and closing per stream would make every
        request pay for a new connection.
        """
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

        def write(payload: bytes) -> None:
            self.wfile.write(b"%x\r\n%s\r\n" % (len(payload), payload))
            self.wfile.flush()

        try:
            for name, data in events:
                blob = data if isinstance(data, str) else json.dumps(data)
                frame = ""
                if name:
                    frame += f"event: {name}\n"
                frame += f"data: {blob}\n\n"
                write(frame.encode())
                if ARGS.delay:
                    time.sleep(ARGS.delay)
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            # The client hung up mid-stream. Normal when a caller cancels.
            pass

    # -- routing -----------------------------------------------------------

    def _path(self) -> str:
        return self.path.split("?", 1)[0].rstrip("/") or "/"

    def do_GET(self) -> None:  # noqa: N802
        self._announce()
        path = self._path()

        if path in ("/health", "/health_generate"):
            # The real server answers 200 with no body.
            return self._send_empty()
        if path == "/version":
            return self._send_json({"version": ARGS.version})
        if path == "/v1/models":
            return self._send_json(
                {
                    "object": "list",
                    "data": [
                        {
                            "id": ARGS.model,
                            "object": "model",
                            "created": _now(),
                            "owned_by": "tensorrt_llm",
                            "max_model_len": 131072,
                        }
                    ],
                }
            )
        if path == "/metrics":
            return self._send_json([])
        if path == "/cluster_info":
            return self._send_json({"cluster": [], "version": ARGS.version})
        if path == "/server_info":
            return self._send_json(
                {
                    "model": ARGS.model,
                    "version": ARGS.version,
                    "backend": "pytorch",
                }
            )
        if path == "/steady_clock_offset":
            return self._send_json({"offset": 0.0, "delay": 0.0})
        if path in ("/kv_cache_events", "/v1/data_transceiver_state", "/energy_metrics"):
            return self._send_json([])
        if path.startswith("/v1/responses/"):
            return self._send_json(_responses_body({}, _reply_text()))

        # Anything unrecognised is still a 200.
        return self._send_json({})

    def do_POST(self) -> None:  # noqa: N802
        self._announce()
        path = self._path()
        req = self._read_body()
        text = _reply_text()

        dialect = STREAMING_ROUTES.get(path)
        if dialect and req.get("stream"):
            return self._send_sse(STREAM_BUILDERS[dialect](req, text))

        builder = BODY_BUILDERS.get(path)
        if builder:
            return self._send_json(builder(req, text))

        if path == "/v1/messages/count_tokens":
            return self._send_json({"input_tokens": 8})
        if path == "/v1/embeddings":
            return self._send_json(
                {
                    "object": "list",
                    "model": req.get("model") or ARGS.model,
                    "data": [{"object": "embedding", "index": 0, "embedding": [0.0] * 8}],
                    "usage": {"prompt_tokens": 8, "total_tokens": 8},
                }
            )
        if path == "/_internal/tokenize":
            return self._send_json({"tokens": list(range(8))})

        return self._send_json({})

    # The remaining verbs exist so that nothing a caller tries can 501.
    def do_PUT(self) -> None:  # noqa: N802
        self._announce()
        self._read_body()
        self._send_json({})

    def do_DELETE(self) -> None:  # noqa: N802
        self._announce()
        self._send_json({})

    def do_HEAD(self) -> None:  # noqa: N802
        self._announce()
        self._send_empty()


class Server(ThreadingHTTPServer):
    daemon_threads = True
    # socketserver defaults this to 5, which is the length of the kernel's
    # accept queue. Past that the kernel resets connections rather than
    # queueing them, so a caller ramping concurrency sees ECONNRESET and reads
    # it as the thing under test failing. A stub that exists to never fail has
    # to be able to accept the arrivals.
    request_queue_size = 1024
    # Rebind immediately after a restart instead of waiting out TIME_WAIT.
    allow_reuse_address = True


def main() -> int:
    global ARGS
    p = argparse.ArgumentParser(
        description="A trtllm-serve stand-in that answers 200 to everything."
    )
    p.add_argument("--host", default="0.0.0.0")  # nosec B104
    p.add_argument("--port", type=int, default=8000)
    p.add_argument(
        "--model", default="fake-model", help="reported by /v1/models and echoed into responses"
    )
    p.add_argument("--version", default="1.3.0rc26")
    p.add_argument(
        "--tokens", type=int, default=16, help="how many whitespace tokens the reply carries"
    )
    p.add_argument("--reply", default=None, help="fixed reply text, overriding --tokens")
    p.add_argument("--delay", type=float, default=0.0, help="seconds between streamed events")
    p.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="do not print arriving requests. Printing is on by "
        "default; turn it off for a load test, where writing "
        "a line per request becomes the bottleneck",
    )
    ARGS = p.parse_args()

    server = Server((ARGS.host, ARGS.port), Handler)
    print(
        f"fake trtllm-serve on http://{ARGS.host}:{ARGS.port}  "
        f"model={ARGS.model}  threads={threading.active_count()}",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping", flush=True)
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
