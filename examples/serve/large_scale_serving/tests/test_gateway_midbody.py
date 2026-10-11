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
"""What the gateway does when a relayed body that is not SSE fails partway.

Once the backend's head has been written to the client, the gateway's two
remedies for a failed backend are both off the table: a retry and a 502 would
each put a second status line into a response the client is already reading.
The non-SSE relay used to let a mid-body read error escape to the caller tagged
`upstream_read`, which the retry test (`side == "upstream"`) skips, so the
caller fell through to writing exactly that 502 -- after the backend's 200 and
part of its body. All that is left is to end the connection short of the length
the head promised, which the client can recognise, and to log it as what it
was: a response the backend cut off, not one it never gave.
"""

import importlib.util
import re
import socket
from types import SimpleNamespace

import pytest
from fake_backend import RESET_HEADER
from handover_harness import API_USER, CONVERSATION_HEADER, GATEWAY_PY, Scenario, control, wait_for

SPEC = importlib.util.spec_from_file_location("gw_midbody_under_test", GATEWAY_PY)
gw = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gw)

PIN_PREFIX = "hdr:"
HEAD = b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 64"
BACKEND = SimpleNamespace(url="http://10.0.0.7:8200", job_id="job7")


@pytest.fixture
def make_scenario():
    created = []

    def factory(*args, **kwargs):
        scenario = Scenario(*args, **kwargs)
        created.append(scenario)
        return scenario

    yield factory
    for scenario in reversed(created):
        scenario.close()


def pin(port, convo, job_id):
    status, data, _ = control(
        port,
        "POST",
        "/_gateway/pin",
        {"conversation": PIN_PREFIX + convo, "backend": job_id},
    )
    assert status == 200, data


def read_until_closed(port, convo, extra_headers=(), timeout=10.0):
    """Send one request; return every byte the gateway wrote before it closed.

    Raw rather than the harness `request()`, which stops reading at
    Content-Length -- and what matters here is whatever follows the head.
    """
    body = b'{"model":"m","messages":[]}'
    lines = [
        "POST /v1/messages HTTP/1.1",
        "Host: 127.0.0.1:%d" % port,
        "x-api-key: %s" % API_USER,
        "%s: %s" % (CONVERSATION_HEADER, convo),
        "Content-Length: %d" % len(body),
        "Content-Type: application/json",
        "Connection: close",
    ]
    lines.extend("%s: %s" % header for header in extra_headers)
    received = b""
    with socket.create_connection(("127.0.0.1", port), timeout=timeout) as sock:
        sock.sendall(("\r\n".join(lines) + "\r\n\r\n").encode("latin-1") + body)
        while True:
            try:
                chunk = sock.recv(65536)
            except ConnectionResetError:
                break
            except TimeoutError:
                pytest.fail(
                    "the gateway neither finished nor closed the connection within %.0fs; "
                    "received %r" % (timeout, received)
                )
            if not chunk:
                break
            received += chunk
    return received


def test_a_backend_dying_mid_body_ends_the_connection_with_one_status_line(make_scenario):
    """The backend promises a JSON body, sends 12 bytes of it, and resets."""
    scenario = make_scenario("midbody-reset", backends=2)
    scenario.start()
    scenario.wait_health_status("ok", timeout=25.0)
    dying, bystander = scenario.fleet.backends

    convo = "cut-short"
    pin(scenario.port, convo, dying.job_id)
    sent = 12
    raw = read_until_closed(scenario.port, convo, extra_headers=[(RESET_HEADER, sent)])

    statuses = re.findall(rb"HTTP/1\.[01] \d{3} ", raw)
    assert len(statuses) == 1, "expected one status line, got %d: %r" % (len(statuses), raw)
    head, _, body = raw.partition(b"\r\n\r\n")
    assert head.startswith(b"HTTP/1.1 200 "), raw
    assert ("X-Backend-Id: %s" % dying.job_id).encode() in head, raw
    assert len(body) == sent, (
        "the client should get the %d bytes the backend sent, then the close: %r" % (sent, raw)
    )

    # Logged as a response the backend cut off -- not 502, which says it never
    # answered -- and attributed to the backend that did it.
    access_line = r"(\S+) POST /v1/messages .*convo=%s%s " % (PIN_PREFIX, convo)
    access = wait_for(
        "the access line for the cut-short request",
        lambda: re.search(access_line, scenario.gateway.log_text()),
        timeout=10.0,
    )
    assert access.group(1) == "200!", access.group(0)
    log = scenario.gateway.log_text()
    assert "upstream %s failed mid-body side=upstream_read" % dying.url in log, log
    # And not retried: the bystander never saw the request.
    posts = [
        row for row in scenario.backend_access_log(bystander.job_id) if row["method"] == "POST"
    ]
    assert not posts, posts


class ScriptedUpstream:
    """A backend reader that hands out `chunks` in order, then raises `then`."""

    def __init__(self, chunks, then):
        self.chunks = list(chunks)
        self.then = then

    async def read(self, size):
        if self.chunks:
            return self.chunks.pop(0)
        raise self.then


class RecordingClient:
    """The client's StreamWriter, keeping every write; drain fails past `fail_after`."""

    def __init__(self, fail_after=None):
        self.writes = []
        self.fail_after = fail_after
        self.closed = False

    def write(self, data):
        self.writes.append(bytes(data))

    async def drain(self):
        if self.fail_after is not None and len(self.writes) > self.fail_after:
            raise ConnectionResetError(104, "Connection reset by peer")

    def close(self):
        self.closed = True

    async def wait_closed(self):
        return None


@pytest.mark.asyncio
async def test_relay_closes_the_client_when_the_backend_fails_mid_body():
    first = HEAD + b"\r\n\r\n" + b'{"partial":'
    upstream = ScriptedUpstream([first], then=ConnectionResetError(104, "reset"))
    client = RecordingClient()

    status = await gw.Gateway.__new__(gw.Gateway).relay_response(
        BACKEND, upstream, client, gw.RequestTrace()
    )

    assert status == "200!"
    assert client.writes == [first], "nothing may follow what the backend sent"
    assert client.closed


@pytest.mark.asyncio
async def test_relay_records_a_client_gone_mid_body_as_undelivered():
    """The other side of the same loop: the client, not the backend, went away.

    Same rule -- the head is out, so no 502 -- and the status says the response
    could not be delivered rather than blaming the backend for it.
    """
    first = HEAD + b"\r\n\r\n" + b'{"a":'
    upstream = ScriptedUpstream(
        [first, b'"rest"}'], then=AssertionError("relay kept reading after the client failed")
    )
    client = RecordingClient(fail_after=1)

    status = await gw.Gateway.__new__(gw.Gateway).relay_response(
        BACKEND, upstream, client, gw.RequestTrace()
    )

    assert status == "200?"
    assert client.writes == [first, b'"rest"}']
