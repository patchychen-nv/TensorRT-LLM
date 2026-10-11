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
"""A backend that drops connection attempts is given up on within the connect timeout.

A listener whose accept queue is full drops further SYNs on Linux rather than refusing
them, so a connect to it hangs while the kernel retries -- for about two minutes by
default. That is how a saturated serving frontend looks from the gateway, and it is
longer than a Codex client waits for its first event.
"""

import asyncio
import socket
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import gateway  # noqa: E402

pytestmark = pytest.mark.skipif(
    not sys.platform.startswith("linux"),
    reason="relies on Linux dropping SYNs to a full accept queue",
)


@pytest.fixture
def full_listener():
    """A listening socket whose accept queue is already full."""
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen(0)
    fillers = []
    for _ in range(2):
        filler = socket.socket()
        filler.setblocking(False)
        filler.connect_ex(server.getsockname())
        fillers.append(filler)
    time.sleep(0.2)  # let the first filler's handshake complete and occupy the queue
    yield server.getsockname()
    for sock in fillers + [server]:
        sock.close()


def _gateway(timeout):
    gw = gateway.Gateway.__new__(gateway.Gateway)
    gw.fleet = SimpleNamespace(args=SimpleNamespace(upstream_connect_timeout=timeout))
    return gw


def test_a_backend_dropping_connections_fails_within_the_connect_timeout(full_listener):
    host, port = full_listener
    backend = SimpleNamespace(host=host, port=port, url=f"http://{host}:{port}", job_id="hole")

    started = time.monotonic()
    with pytest.raises(ConnectionError, match="upstream connect timed out after 0.5s"):
        asyncio.run(
            _gateway(0.5).proxy(
                backend, "POST", "/v1/responses", [], b"", b"{}", None, None, "u", None
            )
        )
    # Well inside the client's patience, and far from the kernel's two minutes.
    assert time.monotonic() - started < 5


def test_the_connect_timeout_is_an_option_with_a_default():
    args = gateway.parse_args(
        ["--fleet-dir", "/tmp/unused", "--users", "/tmp/unused-users.txt", "--no-relay"]
    )
    assert args.upstream_connect_timeout == 10.0
