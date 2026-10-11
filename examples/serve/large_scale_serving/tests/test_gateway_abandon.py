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
"""What the gateway does when the client gives up before the backend answers.

The failure these guard against, measured on the fill-first benchmark fleet: a
saturated backend's first byte is minutes away, the Codex client times out at
120s and hangs up, and its retry carries the same conversation key -- so the
pin sends it straight back into the same queue, behind the corpse of the
request it just abandoned. Four timeouts later the campaign is dead.

Two halves. The watcher: a buffered request leaves the client socket idle, so
its EOF is the abandonment signal, available exactly in the window where the
gateway is otherwise blind (blocked on the backend's first byte). The unpin:
zero events delivered is the timeout signature, and only then is the pin
dropped -- a stream that died mid-flight was being served, and re-homing those
would shed load the backend was carrying fine.
"""

import socket
import time

import pytest
from handover_harness import API_USER, CONVERSATION_HEADER, Scenario, control, request

PIN_PREFIX = "hdr:"


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
    return data


def unpin_reporting_previous(port, convo):
    """Drop the pin by hand; the reply names what was pinned, or null."""
    status, data, _ = control(
        port,
        "POST",
        "/_gateway/pin",
        {"conversation": PIN_PREFIX + convo, "backend": None},
    )
    assert status == 200, data
    return data.get("unpinned")


def abandon(port, convo, delay_s, wait_before_close=0.5):
    """Send a request whose backend will stall, then hang up mid-wait.

    A raw socket rather than the harness client, because the harness waits for
    the response and the whole point here is not to.
    """
    body = b'{"model":"m","messages":[]}'
    head = (
        "POST /v1/messages HTTP/1.1\r\n"
        "Host: t\r\n"
        f"x-api-key: {API_USER}\r\n"
        f"{CONVERSATION_HEADER}: {convo}\r\n"
        f"x-test-delay: {delay_s}\r\n"
        f"Content-Length: {len(body)}\r\n"
        "Content-Type: application/json\r\n"
        "\r\n"
    ).encode()
    sock = socket.create_connection(("127.0.0.1", port), timeout=5.0)
    try:
        sock.sendall(head + body)
        time.sleep(wait_before_close)
    finally:
        sock.close()


def test_abandoned_wait_drops_the_pin_so_the_retry_can_go_elsewhere(make_scenario):
    """The retry after a client timeout must not be sent back into the queue.

    Verified through the pin table rather than through placement: with two idle
    backends the post-abandon placement could legitimately land on either, so
    "did the retry go elsewhere" is a coin flip -- but "is the pin gone" is
    not. The pin table is read through GET /_gateway/fleet, so a missing entry
    there says the abandon handler already removed it.
    """
    scenario = make_scenario("abandon-unpin", backends=2)
    scenario.start()
    scenario.wait_health_status("ok", timeout=25.0)
    slow = scenario.fleet.backends[0]

    convo = "abandoner"
    pin(scenario.port, convo, slow.job_id)

    # Stall the backend for far longer than the client is willing to wait.
    abandon(scenario.port, convo, delay_s=6.0, wait_before_close=0.5)

    # The watcher fires on the close; give the handler a beat to run. Poll the
    # pin table read-only. Probing with unpin-then-repin raced the handler: an
    # unpin landing first left the handler nothing to remove, and the repin that
    # followed restored the very pin under test - a flaky failure on a correct
    # gateway (about one run in five).
    key = PIN_PREFIX + convo
    deadline = time.time() + 5.0
    was = slow.job_id
    while time.time() < deadline:
        status, data, _ = control(scenario.port, "GET", "/_gateway/fleet")
        assert status == 200, data
        was = data["routing"]["manual_pins"].get(key)
        if was is None:
            break
        time.sleep(0.2)
    assert was is None, (
        "the abandoned wait left the pin on %s; a retry would queue behind "
        "the request the client just gave up on" % was
    )

    # The conversation is not poisoned: a fresh request on it still answers.
    after = request(
        scenario.port,
        "POST",
        "/v1/messages",
        headers=[("x-api-key", API_USER), (CONVERSATION_HEADER, convo)],
        body=b'{"model":"m","messages":[]}',
    )
    assert after.status == 200, after


def test_a_completed_request_keeps_its_pin(make_scenario):
    """The watcher must not read a normal close-after-response as abandonment.

    Every harness client closes its socket once it has read the response; if
    that close raced the relay's completion into a ClientAbandoned, every
    request would unpin its conversation and the cache-affinity routing would
    be silently gone. The relay's verdict wins the race by construction; this
    holds it there.
    """
    scenario = make_scenario("abandon-keeps", backends=2)
    scenario.start()
    scenario.wait_health_status("ok", timeout=25.0)
    chosen = scenario.fleet.backends[1]

    convo = "finisher"
    pin(scenario.port, convo, chosen.job_id)

    first = request(
        scenario.port,
        "POST",
        "/v1/messages",
        headers=[("x-api-key", API_USER), (CONVERSATION_HEADER, convo)],
        body=b'{"model":"m","messages":[]}',
    )
    assert first.status == 200, first
    assert first.backend == chosen.job_id

    # Long enough for a mistaken abandon handler to have run.
    time.sleep(1.0)
    assert unpin_reporting_previous(scenario.port, convo) == chosen.job_id, (
        "a completed request lost its pin; normal closes are being read as abandonment"
    )


def test_abandonment_does_not_disturb_other_conversations(make_scenario):
    """The unpin is scoped to the conversation that left, and only that one."""
    scenario = make_scenario("abandon-scoped", backends=2)
    scenario.start()
    scenario.wait_health_status("ok", timeout=25.0)
    _slow, other = scenario.fleet.backends[0], scenario.fleet.backends[1]

    bystander = "bystander"
    pin(scenario.port, bystander, other.job_id)

    abandon(scenario.port, "leaver", delay_s=6.0, wait_before_close=0.5)
    time.sleep(1.0)

    assert unpin_reporting_previous(scenario.port, bystander) == other.job_id, (
        "an unrelated conversation lost its pin to someone else's abandonment"
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
