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
"""The smallest thing `gateway.py` will route to: a fake fleet member.

This exists so the handover tests can run a REAL gateway on a laptop or a CI
box -- no SLURM, no GPUs, no model. It implements exactly the three things the
gateway asks of a backend and nothing else:

1. **Discovery.** The gateway reads `<fleet-dir>/*.json`; each file is one
   backend (`Fleet.read_registrations` / `Fleet.discover`). This process does
   not write that file itself -- the test harness does, because the registration
   is written by the serving job's *controller*, and the scenario that matters
   most for the supervisor (`revive_dead_backends`) is precisely "controller
   alive and writing heartbeats, server dead". Keeping the two apart is what
   lets the harness stage that.

2. **Health.** `probe()` opens a connection and does `GET /v1/models`, then
   looks for `" 200 "` in the status line. Nothing else counts -- deliberately,
   see the docstring on `gateway.probe`. `--mode-file` flips this answer at
   runtime between 200 (healthy) and 503 (present but not serving), which is
   how a test makes a backend sick without killing the process that heartbeats
   for it.

3. **Proxying.** Every other path is relayed verbatim by `Gateway.proxy`. The
   reply carries `X-Backend-Id`, which survives the relay untouched, so a
   client can tell which backend served it. `X-Test-Delay: <seconds>` makes a
   request stay in flight for a controlled length of time, which is how the
   drain window is exercised. `X-Test-Reset-After: <bytes>` cuts the reply off
   mid-body with a connection reset.

Run standalone:

    python3 fake_backend.py --job-id job1 --port 20001 --mode-file /tmp/m
"""

import argparse
import errno
import json
import os
import socket
import struct
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# Must match `gateway.PROBE_PATH` (gateway.py:2253). The harness asserts this
# at import time rather than importing gateway, so a change there shows up as a
# clear message instead of every backend silently reading as dead.
PROBE_PATH = "/v1/models"

# Not in `gateway.STRIP_REQUEST_HEADERS`, so it reaches the backend verbatim.
DELAY_HEADER = "x-test-delay"

# Also forwarded verbatim. `X-Test-Reset-After: <n>` sends the response head and
# the first n body bytes, then resets the connection: a backend that dies after
# the gateway has already started relaying its answer.
RESET_HEADER = "x-test-reset-after"

# How long the half-sent response sits before the reset, so the gateway has
# relayed the head by the time it lands -- the failure under test is the one
# that comes after the head, not before.
RESET_GRACE_SECONDS = 0.3

# gateway.DEFAULT_KEY_SOURCES[0] (gateway.py:1019).
CONVERSATION_HEADER = "x-conversation-id"


class FakeBackend(ThreadingHTTPServer):
    """Threaded on purpose: a delayed request must not block the health probe."""

    daemon_threads = True
    # Not SO_REUSEPORT. A backend has one owner; only the gateway ever shares a
    # port, and letting a stray second backend silently join this one would
    # produce exactly the confusing split the handover tests are trying to
    # measure.
    allow_reuse_address = True

    def __init__(self, addr, handler, job_id, mode_file, access_log, probe_delay=0.0):
        super().__init__(addr, handler)
        self.job_id = job_id
        self.mode_file = mode_file
        self.access_log = access_log
        self.probe_delay = probe_delay
        self.log_lock = threading.Lock()
        self.served = 0

    def mode(self):
        """Either healthy or sick, re-read per probe so a test can flip it live."""
        if not self.mode_file:
            return "healthy"
        try:
            with open(self.mode_file) as handle:
                return handle.read().strip() or "healthy"
        except OSError:
            return "healthy"

    def record(self, line):
        if not self.access_log:
            return
        with self.log_lock:
            try:
                with open(self.access_log, "a") as handle:
                    handle.write(line + "\n")
            except OSError:
                pass


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "fake-backend/1"

    def log_message(self, fmt, *args):
        # The default writes to stderr per request, which at the rates these
        # tests drive would dwarf the gateway's own log. `record()` below keeps
        # the part a post-mortem actually wants.
        return

    def do_GET(self):
        self._serve("GET")

    def do_POST(self):
        self._serve("POST")

    def _reply(self, status, payload, extra=()):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        for name, value in extra:
            self.send_header(name, value)
        # The gateway sends `Connection: close` upstream and relays the reply to
        # EOF for anything that is not SSE, so saying so here keeps the framing
        # unambiguous at both ends.
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(payload)
        self.close_connection = True

    def _serve(self, method):
        try:
            length = int(self.headers.get("content-length") or 0)
        except ValueError:
            length = 0
        body = self.rfile.read(length) if length > 0 else b""

        if self.path == PROBE_PATH:
            # A real /v1/models on a loaded node is not instant, and the delay
            # is load-bearing for the tests: it is what makes a successor's
            # "bound but not yet probed" window wide enough to observe rather
            # than sample. See the comment in test_handover_serves_every_request.
            if self.server.probe_delay > 0:
                time.sleep(self.server.probe_delay)
            if self.server.mode() == "healthy":
                self._reply(200, json.dumps({"object": "list", "data": [{"id": "fake"}]}).encode())
            else:
                # Present, listening, and not serving. `probe()` reads this as
                # "dead" (no " 200 " in the status line) and takes the backend
                # out of rotation immediately -- the behaviour a test needs to
                # stage an unhealthy fleet without killing the process.
                self._reply(503, json.dumps({"error": "not serving"}).encode())
            return

        try:
            delay = float(self.headers.get(DELAY_HEADER) or 0.0)
        except ValueError:
            delay = 0.0
        convo = self.headers.get(CONVERSATION_HEADER) or ""
        seq = self.headers.get("x-test-seq") or ""
        # Logged BEFORE the delay, not after. The drain tests need to know when
        # a long request reached a backend while it is still running -- a line
        # written on completion tells them only after the fact, which is exactly
        # too late to time a handover against.
        self._record("start", method, seq, delay)
        if delay > 0:
            time.sleep(delay)
        self._record("end", method, seq, delay)

        payload = json.dumps(
            {
                "backend": self.server.job_id,
                "pid": os.getpid(),
                "convo": convo,
                "seq": seq,
                "request_bytes": len(body),
                "gateway_user": self.headers.get("x-gateway-user") or "",
            }
        ).encode()
        self.server.served += 1
        try:
            reset_after = int(self.headers.get(RESET_HEADER) or -1)
        except ValueError:
            reset_after = -1
        if reset_after >= 0:
            self._reply_then_reset(payload, reset_after)
            return
        self._reply(200, payload, extra=[("X-Backend-Id", self.server.job_id)])

    def _reply_then_reset(self, payload, sent):
        """Promise the whole payload, deliver `sent` bytes of it, then reset."""
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("X-Backend-Id", self.server.job_id)
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(payload[:sent])
        time.sleep(RESET_GRACE_SECONDS)
        # An RST, not a FIN: SO_LINGER 0 makes close() abort the connection, so
        # the gateway's next read raises instead of reading a clean end of body.
        # rfile holds a reference to the socket, and close() only reaches the
        # kernel once the last one is released.
        self.connection.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
        self.rfile.close()
        self.connection.close()
        self.close_connection = True

    def _record(self, kind, method, seq, delay):
        self.server.record(
            "%s\t%.6f\t%s\t%s\t%s\t%s\t%.3f"
            % (kind, time.time(), self.server.job_id, method, self.path, seq or "-", delay)
        )


def wait_listening(host, port, deadline):
    """Block until something answers on host:port, or the deadline passes."""
    while time.time() < deadline:
        sock = socket.socket()
        sock.settimeout(0.25)
        try:
            sock.connect((host, port))
            return True
        except OSError as exc:
            if exc.errno not in (errno.ECONNREFUSED, errno.EAGAIN, errno.ETIMEDOUT):
                pass
        finally:
            sock.close()
        time.sleep(0.02)
    return False


def main(argv):
    parser = argparse.ArgumentParser(description="fake fleet member for the gateway tests")
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--mode-file", default="", help="file holding 'healthy' or 'sick'")
    parser.add_argument("--access-log", default="", help="append one line per relayed request")
    parser.add_argument(
        "--probe-delay",
        type=float,
        default=0.0,
        help="seconds to hold %s before answering it" % PROBE_PATH,
    )
    args = parser.parse_args(argv)

    server = FakeBackend(
        (args.host, args.port),
        Handler,
        args.job_id,
        args.mode_file,
        args.access_log,
        args.probe_delay,
    )
    # The harness waits on the port rather than on this line, but a human
    # reading the log wants to know which id went where.
    sys.stdout.write("fake backend %s listening on %s:%d\n" % (args.job_id, args.host, args.port))
    sys.stdout.flush()
    try:
        server.serve_forever(poll_interval=0.05)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
