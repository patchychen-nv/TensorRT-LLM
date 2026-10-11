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
"""Machinery for the gateway hot-handover tests. No assertions live here.

Everything in this file exists to make one sentence testable: *during a
handover a client with a steady stream of requests sees zero refused
connections and zero failed in-flight requests*. That sentence is only worth
anything if the thing under test is a real gateway process, so nothing here
mocks the gateway -- it starts `gateway.py` with `subprocess`, lets the real
handover spawn a real successor, and measures from outside with raw sockets.

Three pieces are worth reading before the tests:

`lease_port`
    Port choice is a correctness problem here, not a convenience one. The
    gateway binds with `SO_REUSEPORT` (work item E), and a `SO_REUSEPORT` bind
    *succeeds* against a port somebody else already holds the same way. So the
    usual "bind it, close it, assume it is free" check does not answer the
    question: a test could bind on top of a real gateway, have the kernel
    deal it half the connections, and measure a fleet that is not its own.
    Defence is in three layers -- an advisory `flock` per port so parallel test
    runs cannot choose the same one, a refused `connect()` so nothing is
    already listening, an exclusive (non-reuse) bind so nothing holds it
    without listening -- and then, because all three are checks-of-the-moment,
    a fourth that runs continuously: every `/_gateway/health` reply carries
    `deployment`, which is the basename of `--fleet-dir`, and the harness gives
    each run a unique one. A foreign gateway sharing the port is therefore not
    a silent wrong answer, it is an assertion failure naming the intruder.

`RequestStream`
    A client is only useful here if it can tell three failures apart:
    *refused* (the connection never came up -- the handover left the port
    unbound), *lost in flight* (the connection came up and then died with no
    usable reply -- the old process was killed instead of drained), and
    *served, badly* (an HTTP error, e.g. the 503 a successor serves in the
    window after it binds but before its first health probe). `http.client`
    collapses the first two. So the stream speaks HTTP over a raw socket and
    records which syscall failed.

`require_cli` / `require_endpoint`
    Work items A-E are not merged yet, so these tests are expected to fail.
    They should fail saying *which clause of the contract is missing*, not with
    an argparse traceback or a 404 from deep inside a wait loop.
"""

import errno
import fcntl
import itertools
import json
import os
import random
import re
import shutil
import signal
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
GATEWAY_DIR = os.path.dirname(TESTS_DIR)
FAKE_BACKEND_PY = os.path.join(TESTS_DIR, "fake_backend.py")

# Normally the gateway next door. `GW_HANDOVER_GATEWAY` points the suite at a
# candidate build instead -- useful while a work item is in flight, and it is
# how this harness was validated against a reference implementation before the
# real one existed.
GATEWAY_PY = os.environ.get("GW_HANDOVER_GATEWAY") or os.path.join(GATEWAY_DIR, "gateway.py")

# The gateway is stdlib-only and must stay runnable under whatever python3 is
# around, so the tests launch it with the same interpreter that runs them.
PYTHON = sys.executable or "python3"

API_USER = "handover-test-user"
CONVERSATION_HEADER = "x-conversation-id"
DELAY_HEADER = "x-test-delay"
PROBE_PATH = "/v1/models"

# Set to "1" to turn "the contract is not implemented yet" from a failure into
# a skip. Off by default and it should stay off in CI: this suite is the thing
# that decides whether the feature ships, and a suite that skips itself when
# the feature is missing cannot do that job.
SKIP_IF_UNIMPLEMENTED = os.environ.get("GW_HANDOVER_SKIP_IF_UNIMPLEMENTED") == "1"


class HarnessError(AssertionError):
    """Something the harness itself needed did not happen in time."""


# ---------------------------------------------------------------------------
# Contract surface
# ---------------------------------------------------------------------------
_CLI_CACHE = {}

CONTRACT_OWNER = {
    "--router-only": "contract 'CLI surface (exact)' / work item A (supervisor lock)",
    "--supervisor-lock": "contract 'CLI surface (exact)' / work item A (supervisor lock)",
    "--handover-drain-deadline": "contract 'CLI surface (exact)' / work item B (graceful drain)",
    "--handover-ready-timeout": "contract 'CLI surface (exact)' / work item C (orchestration)",
    "--reuse-port": "contract 'CLI surface (exact)' / work item E (reuse_port)",
    "--no-reuse-port": "contract 'CLI surface (exact)' / work item E (reuse_port)",
}


def gateway_cli_options():
    """Every option `gateway.py` actually defines, from its usage block.

    The usage block is used rather than the option list below it because it is
    generated from the parser and contains nothing else; the help bodies quote
    option names in prose ("--yaml is required unless --no-relay is given"),
    and a scan of those would report options that do not exist.
    """
    if "options" in _CLI_CACHE:
        return _CLI_CACHE["options"]
    proc = subprocess.run(
        [PYTHON, GATEWAY_PY, "--help"], capture_output=True, text=True, timeout=60
    )
    if proc.returncode != 0:
        raise HarnessError(
            "`%s %s --help` exited %d; the gateway does not even parse:\n%s\n%s"
            % (PYTHON, GATEWAY_PY, proc.returncode, proc.stdout[-2000:], proc.stderr[-2000:])
        )
    usage = proc.stdout.split("\n\n", 1)[0]
    options = set(re.findall(r"--[a-z0-9][a-z0-9-]*", usage))
    _CLI_CACHE["options"] = options
    return options


def require_cli(*names):
    """Fail with the contract clause, not with an argparse traceback."""
    missing = [name for name in names if name not in gateway_cli_options()]
    if not missing:
        return
    import pytest

    detail = "\n".join(
        "  %s  --  not implemented; %s" % (name, CONTRACT_OWNER.get(name, "see CONTRACT.md"))
        for name in missing
    )
    message = (
        "gateway.py does not implement the handover CLI surface yet.\n"
        "Missing options:\n%s\n"
        "Implemented options: %s\n"
        "This test is written against CONTRACT.md and is expected to fail until "
        "work items A-E land." % (detail, " ".join(sorted(gateway_cli_options())))
    )
    if SKIP_IF_UNIMPLEMENTED:
        pytest.skip(message)
    pytest.fail(message, pytrace=False)


def require_endpoint(status, method, path, clause):
    """Turn a 404/405 on a contract endpoint into a sentence about the contract."""
    if status not in (404, 405):
        return
    import pytest

    message = (
        "%s %s answered %d.\nCONTRACT '%s' requires this endpoint.\n"
        "Expected to fail until work item C lands." % (method, path, status, clause)
    )
    if SKIP_IF_UNIMPLEMENTED:
        pytest.skip(message)
    pytest.fail(message, pytrace=False)


def check_probe_path_unchanged():
    """The fake backend hardcodes the probe path; make drift loud, not silent."""
    with open(GATEWAY_PY) as handle:
        source = handle.read()
    if 'PROBE_PATH = "%s"' % PROBE_PATH not in source:
        raise HarnessError(
            "gateway.py no longer defines PROBE_PATH = %r. fake_backend.py answers that "
            "path and nothing else, so every fake backend would read as dead. Update "
            "PROBE_PATH in fake_backend.py and handover_harness.py together." % PROBE_PATH
        )


# ---------------------------------------------------------------------------
# Ports
# ---------------------------------------------------------------------------
def _port_band():
    """A band of ports the kernel will not hand out to an outgoing connection.

    Landing inside `ip_local_port_range` means an ephemeral socket somewhere on
    the machine can take the port between the check and the bind, which reads
    exactly like the bug these tests exist to catch.
    """
    low, high = 32768, 60999
    try:
        with open("/proc/sys/net/ipv4/ip_local_port_range") as handle:
            low, high = (int(x) for x in handle.read().split()[:2])
    except (OSError, ValueError):
        pass
    if low > 21000:
        return 20000, min(low - 1, 30000)
    if high < 64000:
        return high + 1, 65000
    raise HarnessError(
        "no usable port band outside ip_local_port_range=%d-%d; set a narrower range "
        "or run the tests in a namespace that has one" % (low, high)
    )


class PortLease:
    """One port, held for the life of the test by an advisory lock."""

    def __init__(self, port, fd, path):
        self.port = port
        self._fd = fd
        self._path = path

    def release(self):
        if self._fd is None:
            return
        try:
            fcntl.flock(self._fd, fcntl.LOCK_UN)
        except OSError:
            pass
        os.close(self._fd)
        self._fd = None

    def __repr__(self):
        return "PortLease(%d)" % self.port


# One empty file per port ever used, deliberately never unlinked. Removing a
# lock file that another run still holds an flock on would break the mutual
# exclusion outright: that run keeps its lock on the now-unlinked inode, and the
# next run creates a *different* inode at the same path and locks it happily.
# The files are zero bytes and bounded by the port band, so leaving them is the
# cheap side of the trade.
_LEASE_DIR = os.path.join(tempfile.gettempdir(), "trtllm-gw-handover-ports")


def lease_port(rng):
    """Reserve one port. See the module docstring for why this is three checks."""
    os.makedirs(_LEASE_DIR, exist_ok=True)
    low, high = _port_band()
    for _ in range(400):
        port = rng.randrange(low, high)
        path = os.path.join(_LEASE_DIR, "%d.lock" % port)
        fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            os.close(fd)
            continue  # another test run owns it
        if _something_is_listening(port) or not _can_bind_exclusively(port):
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)
            continue
        return PortLease(port, fd, path)
    raise HarnessError("could not find a free port in %d-%d after 400 tries" % (low, high))


def _something_is_listening(port):
    sock = socket.socket()
    sock.settimeout(0.2)
    try:
        return sock.connect_ex(("127.0.0.1", port)) == 0
    finally:
        sock.close()


def _can_bind_exclusively(port):
    """Bind without SO_REUSEADDR or SO_REUSEPORT.

    Deliberately without them: a reuse bind is exactly the thing that would
    succeed on top of a real gateway. This one fails if anybody holds the
    address at all, which is the question being asked.
    """
    sock = socket.socket()
    try:
        sock.bind(("127.0.0.1", port))
        return True
    except OSError:
        return False
    finally:
        sock.close()


# ---------------------------------------------------------------------------
# Raw HTTP client
# ---------------------------------------------------------------------------
# `kind` values. The split is the point of the whole file: `refused` means the
# handover left the port unbound, `lost` means a connection was accepted and
# then abandoned, `http_NNN` means the gateway answered and the answer was bad.
REFUSED_KINDS = ("refused", "connect_timeout")
LOST_KINDS = ("send_failed", "no_response", "truncated_head", "truncated_body", "read_timeout")


class Outcome:
    """One request, with enough detail to name what broke and when."""

    __slots__ = (
        "seq",
        "kind",
        "status",
        "backend",
        "convo",
        "delay",
        "t_start",
        "t_connected",
        "t_head",
        "t_done",
        "error",
        "body",
    )

    def __init__(self, seq, convo="", delay=0.0):
        self.seq = seq
        self.kind = "pending"
        self.status = None
        self.backend = None
        self.convo = convo
        self.delay = delay
        self.t_start = time.time()
        self.t_connected = None
        self.t_head = None
        self.t_done = None
        self.error = ""
        self.body = b""

    @property
    def ok(self):
        return self.kind == "ok"

    def __repr__(self):
        return "Outcome(seq=%s kind=%s status=%s backend=%s t_start=%.3f err=%s)" % (
            self.seq,
            self.kind,
            self.status,
            self.backend,
            self.t_start,
            self.error,
        )


def _describe(exc):
    name = errno.errorcode.get(getattr(exc, "errno", None), "")
    return "%s(%s)%s" % (type(exc).__name__, exc, " " + name if name else "")


def request(
    port,
    method="POST",
    path="/v1/messages",
    body=b"",
    headers=(),
    seq=0,
    convo="",
    delay=0.0,
    connect_timeout=5.0,
    read_timeout=30.0,
):
    """One HTTP request over one fresh connection, classified by what failed.

    A fresh connection per request is not a simplification: `Gateway.handle`
    closes the client socket in a `finally` after every request, so the gateway
    never offers keep-alive. It also makes "refused" directly observable --
    every request is its own bind-and-accept test of the listening socket.
    """
    out = Outcome(seq, convo, delay)
    sock = socket.socket()
    sock.settimeout(connect_timeout)
    try:
        try:
            sock.connect(("127.0.0.1", port))
        except TimeoutError as exc:
            out.kind, out.error = "connect_timeout", _describe(exc)
            return out
        except OSError as exc:
            out.kind, out.error = "refused", _describe(exc)
            return out
        out.t_connected = time.time()

        lines = ["%s %s HTTP/1.1" % (method, path), "Host: 127.0.0.1:%d" % port]
        lines.extend("%s: %s" % (name, value) for name, value in headers)
        if body or method == "POST":
            lines.append("Content-Length: %d" % len(body))
            lines.append("Content-Type: application/json")
        lines.append("Connection: close")
        wire = ("\r\n".join(lines) + "\r\n\r\n").encode("latin-1") + body

        sock.settimeout(read_timeout)
        try:
            sock.sendall(wire)
        except OSError as exc:
            out.kind, out.error = "send_failed", _describe(exc)
            return out

        buf = b""
        try:
            while b"\r\n\r\n" not in buf:
                chunk = sock.recv(65536)
                if not chunk:
                    out.kind = "no_response" if not buf else "truncated_head"
                    out.error = "eof after %d head bytes" % len(buf)
                    return out
                buf += chunk
        except TimeoutError as exc:
            out.kind, out.error = "read_timeout", _describe(exc)
            return out
        except OSError as exc:
            out.kind = "no_response" if not buf else "truncated_head"
            out.error = _describe(exc)
            return out

        out.t_head = time.time()
        head, _, rest = buf.partition(b"\r\n\r\n")
        text = head.decode("latin-1")
        first, _, tail = text.partition("\r\n")
        parts = first.split(" ")
        if len(parts) < 2 or not parts[1].isdigit():
            out.kind, out.error = "truncated_head", "unparsable status line %r" % first
            return out
        out.status = int(parts[1])
        reply_headers = {}
        for line in tail.split("\r\n"):
            name, sep, value = line.partition(":")
            if sep:
                reply_headers[name.strip().lower()] = value.strip()
        out.backend = reply_headers.get("x-backend-id")

        length = reply_headers.get("content-length")
        try:
            want = int(length) if length is not None else None
        except ValueError:
            want = None
        try:
            while want is None or len(rest) < want:
                chunk = sock.recv(65536)
                if not chunk:
                    if want is not None and len(rest) < want:
                        out.kind = "truncated_body"
                        out.error = "eof at %d of %s body bytes" % (len(rest), want)
                        out.body = rest
                        return out
                    break
                rest += chunk
        except TimeoutError as exc:
            out.kind, out.error = "read_timeout", _describe(exc)
            return out
        except OSError as exc:
            out.kind, out.error = "truncated_body", _describe(exc)
            return out

        out.body = rest[:want] if want is not None else rest
        out.t_done = time.time()
        out.kind = "ok" if out.status == 200 else "http_%d" % out.status
        return out
    finally:
        try:
            sock.close()
        except OSError:
            pass


def json_body(out):
    try:
        return json.loads(out.body)
    except (ValueError, TypeError):
        return None


def control(port, method, path, payload=None, timeout=15.0, authenticated=True):
    """Call a `/_gateway/...` endpoint. Returns (status, decoded-or-raw, Outcome).

    The key is sent even for endpoints that do not check one. `/_gateway/health`
    and `/_gateway/start_server` are unauthenticated today (gateway.py:1698 and
    gateway.py:1714) while `/_gateway/route` and friends are not, and CONTRACT
    only says "same x-api-key scheme as the existing endpoints" -- which is two
    different schemes. Sending it satisfies both readings.
    """
    body = json.dumps(payload or {}).encode() if method == "POST" else b""
    headers = [("x-api-key", API_USER)] if authenticated else []
    out = request(
        port,
        method=method,
        path=path,
        body=body,
        headers=headers,
        connect_timeout=timeout,
        read_timeout=timeout,
    )
    if not out.status:
        return None, out.error or out.kind, out
    return out.status, json_body(out) if out.body else {}, out


# ---------------------------------------------------------------------------
# Waiting, bounded, always
# ---------------------------------------------------------------------------
def wait_for(what, predicate, timeout, interval=0.05):
    """Poll until true, then return it. Never waits forever, on purpose."""
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        last = predicate()
        if last:
            return last
        time.sleep(interval)
    raise HarnessError(
        "timed out after %.1fs waiting for %s (last value: %r)" % (timeout, what, last)
    )


# ---------------------------------------------------------------------------
# The fake fleet
# ---------------------------------------------------------------------------
class Backend:
    """A fake fleet member plus the registration record that advertises it."""

    def __init__(
        self,
        root,
        job_id,
        lease,
        healthy=True,
        state="running",
        probe_delay=0.0,
        fleet_dir=None,
        pool=None,
    ):
        self.job_id = job_id
        self.lease = lease
        self.port = lease.port
        self.probe_delay = probe_delay
        # Where the registration goes, when it is not the fleet's own directory:
        # a member of an extra pool (another model's fleet) registers into that
        # pool's directory, exactly as serve.sh gives each cluster_name/model.name
        # pair its own.
        self.fleet_dir = fleet_dir
        self.pool = pool
        self.url = "http://127.0.0.1:%d" % lease.port
        self.run_dir = os.path.join(root, "runs", job_id)
        self.mode_file = os.path.join(root, "%s.mode" % job_id)
        self.access_log = os.path.join(root, "%s.access" % job_id)
        self.log_path = os.path.join(root, "%s.log" % job_id)
        self.state = state
        self.registered = True
        self.proc = None
        os.makedirs(self.run_dir, exist_ok=True)
        self.set_healthy(healthy)

    def set_healthy(self, healthy):
        with open(self.mode_file, "w") as handle:
            handle.write("healthy" if healthy else "sick")

    def record(self, now):
        """The registration record `Fleet.discover` reads.

        Fields and why each one is here:
          job_id     the dict key, and the filename (`<job_id>.json`)
          url        must match `^http://host:port$` (Backend.__init__)
          heartbeat  `now - heartbeat > --stale-after` retires the backend
          end_time   0 means "unknown"; a real timestamp keeps `accepting()`
                     from skipping this backend for being too close to its own
                     wall clock (--new-conversation-margin, default 1800s)
          state      `attempt_failed()` reads " exited with status " out of it,
                     which is what makes `revive_dead_backends` act
          run_dir    handed to `serve.sh restart <run_dir>`
        """
        return {
            "job_id": self.job_id,
            "url": self.url,
            "run_dir": self.run_dir,
            "state": self.state,
            "end_time": now + 86400,
            "heartbeat": now,
        }

    def start(self):
        log = open(self.log_path, "wb")
        self.proc = subprocess.Popen(
            [
                PYTHON,
                FAKE_BACKEND_PY,
                "--job-id",
                self.job_id,
                "--port",
                str(self.port),
                "--mode-file",
                self.mode_file,
                "--access-log",
                self.access_log,
                "--probe-delay",
                str(self.probe_delay),
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        log.close()
        wait_for(
            "fake backend %s to listen on %d" % (self.job_id, self.port),
            lambda: _something_is_listening(self.port),
            timeout=20.0,
        )

    def stop(self):
        if self.proc is None:
            return
        _terminate(self.proc)
        self.proc = None


def _terminate(proc, grace=4.0):
    if proc.poll() is not None:
        return proc.returncode
    try:
        proc.terminate()
    except OSError:
        pass
    try:
        return proc.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        pass
    try:
        proc.kill()
    except OSError:
        pass
    try:
        return proc.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        return None


SERVE_SH_STUB = '''#!/usr/bin/env python3
"""Stands in for serve.sh so a supervisor lifecycle action is observable.

`revive_dead_backends` is the only lifecycle action a `--no-relay` gateway with
no `--fleet-config` can take, and it takes it by running
`<--serve-sh> restart <run_dir>`. That is the proxy the lock tests use for
"this process supervised": no SLURM, no scheduler, just a line in a file.

os.getppid() is the gateway that ran us -- `run_serve_sh` uses
`asyncio.create_subprocess_exec`, so this is a direct child of that process and
the ppid attributes the action unambiguously to one of the two gateways.
"""
import os
import sys
import time

with open(%(log)r, "a") as handle:
    handle.write("%%d\\t%%.6f\\t%%s\\n" %% (os.getppid(), time.time(), " ".join(sys.argv[1:])))
print("ok")
'''


class Fleet:
    """A fleet directory, its users file, and the heartbeats that keep it alive.

    `pools` maps an extra pool's name to how many fake backends it gets.
    Each pool registers into a directory of its own (`pool_dirs[name]`), and its
    job ids are built from the pool's name -- `kimik3job1` -- so which pool
    served a request can be read straight off X-Backend-Id.
    """

    def __init__(self, root, rng, backends=2, probe_delay=0.0, pools=None):
        self.root = root
        self.rng = rng
        self.dir = os.path.join(root, "fleet-%d-%d" % (os.getpid(), rng.randrange(1 << 20)))
        # `/_gateway/health` reports basename(--fleet-dir) as "deployment". It
        # is unique per run, which is what turns "a foreign gateway is sharing
        # our SO_REUSEPORT port" from an invisible wrong answer into an
        # assertion failure.
        self.deployment = os.path.basename(self.dir)
        self.users_file = os.path.join(root, "users.txt")
        self.router_state = os.path.join(root, "router_state.json")
        self.serve_sh = os.path.join(root, "fake_serve.py")
        self.serve_sh_log = os.path.join(root, "serve_sh_calls.log")
        self.supervisor_lock = os.path.join(self.dir, ".supervisor.lock")
        os.makedirs(self.dir, exist_ok=True)
        with open(self.users_file, "w") as handle:
            handle.write("# gateway handover test\n%s\n" % API_USER)
        with open(self.serve_sh, "w") as handle:
            handle.write(SERVE_SH_STUB % {"log": self.serve_sh_log})
        os.chmod(self.serve_sh, 0o755)
        open(self.serve_sh_log, "w").close()

        self.backends = []
        for index in range(backends):
            self.backends.append(
                Backend(root, "fakejob%d" % (index + 1), lease_port(rng), probe_delay=probe_delay)
            )
        self.pool_dirs = {}
        for name, count in (pools or {}).items():
            pool_dir = os.path.join(root, "pool-%s-%d" % (name, rng.randrange(1 << 20)))
            os.makedirs(pool_dir, exist_ok=True)
            self.pool_dirs[name] = pool_dir
            stem = re.sub(r"[^A-Za-z0-9]", "", name)
            for index in range(count):
                self.backends.append(
                    Backend(
                        root,
                        "%sjob%d" % (stem, index + 1),
                        lease_port(rng),
                        probe_delay=probe_delay,
                        fleet_dir=pool_dir,
                        pool=name,
                    )
                )
        self._stop = threading.Event()
        self._beat = None

    def members(self, pool=None):
        """The fake backends of one pool; None is the fleet's own directory."""
        return [backend for backend in self.backends if backend.pool == pool]

    # -- registration ----------------------------------------------------
    def write_registrations(self):
        now = time.time()
        for backend in self.backends:
            path = os.path.join(backend.fleet_dir or self.dir, "%s.json" % backend.job_id)
            if not backend.registered:
                if os.path.exists(path):
                    os.remove(path)
                continue
            # Atomic, and the temp name must not end in .json: discovery globs
            # `*.json`, and a half-written file that matched would make the
            # backend vanish from one sweep and come back on the next -- which
            # the gateway would (correctly) report as a backend going away.
            tmp = path + ".tmp"
            with open(tmp, "w") as handle:
                json.dump(backend.record(now), handle)
            os.replace(tmp, path)

    def start_heartbeats(self, interval=1.0):
        self.write_registrations()

        def beat():
            while not self._stop.wait(interval):
                try:
                    self.write_registrations()
                except OSError:
                    pass

        self._beat = threading.Thread(target=beat, name="fleet-heartbeat", daemon=True)
        self._beat.start()

    def start_backends(self):
        for backend in self.backends:
            backend.start()

    def serve_sh_calls(self):
        """(ppid, timestamp, argv) for every supervisor action taken so far."""
        calls = []
        try:
            with open(self.serve_sh_log) as handle:
                for line in handle:
                    parts = line.rstrip("\n").split("\t")
                    if len(parts) == 3 and parts[0].isdigit():
                        calls.append((int(parts[0]), float(parts[1]), parts[2]))
        except OSError:
            pass
        return calls

    def stop(self):
        self._stop.set()
        if self._beat is not None:
            self._beat.join(timeout=3.0)
        for backend in self.backends:
            backend.stop()
            backend.lease.release()


# ---------------------------------------------------------------------------
# The gateway under test
# ---------------------------------------------------------------------------
class GatewayProc:
    """One real `gateway.py` process."""

    def __init__(self, fleet, lease, name="A", extra_args=(), log_level="INFO"):
        self.fleet = fleet
        self.lease = lease
        self.port = lease.port
        self.name = name
        self.log_path = os.path.join(fleet.root, "gateway-%s.log" % name)
        self.argv = [
            PYTHON,
            GATEWAY_PY,
            "--fleet-dir",
            fleet.dir,
            "--users",
            fleet.users_file,
            "--no-relay",
            "--host",
            # 127.0.0.1, never 0.0.0.0. A production gateway binds the wildcard;
            # binding the loopback address puts this test in a different
            # SO_REUSEPORT group, so the two can never silently share a port.
            "127.0.0.1",
            "--port",
            str(self.port),
            "--serve-sh",
            fleet.serve_sh,
            "--router-state",
            fleet.router_state,
            # Fast enough that a test does not spend its life waiting for a
            # sweep; slow enough that the loops are not the load.
            "--discover-interval",
            "0.5",
            "--health-interval",
            "0.5",
            "--supervisor-interval",
            "0.5",
            "--probe-timeout",
            "2",
            "--log-level",
            log_level,
        ]
        self.argv.extend(extra_args)
        self.proc = None
        self.exit_seen_at = None

    @property
    def pid(self):
        return self.proc.pid if self.proc else None

    def start(self):
        log = open(self.log_path, "wb")
        self.proc = subprocess.Popen(self.argv, stdout=log, stderr=subprocess.STDOUT)
        log.close()
        return self

    def alive(self):
        return self.proc is not None and self.proc.poll() is None

    def log_tail(self, lines=60):
        try:
            with open(self.log_path, errors="replace") as handle:
                return "".join(handle.readlines()[-lines:])
        except OSError:
            return "(no log)"

    def log_text(self):
        try:
            with open(self.log_path, errors="replace") as handle:
                return handle.read()
        except OSError:
            return ""

    def wait_serving(self, timeout=30.0):
        """Up, listening, and with a backend elected. Anything less is not ready."""

        def probe():
            if not self.alive():
                raise HarnessError(
                    "gateway %s exited with %s before it served anything.\nargv: %s\nlog:\n%s"
                    % (self.name, self.proc.returncode, " ".join(self.argv), self.log_tail())
                )
            status, payload, _ = control(self.port, "GET", "/_gateway/health", timeout=3.0)
            if status != 200 or not isinstance(payload, dict):
                return None
            if payload.get("deployment") != self.fleet.deployment:
                raise HarnessError(
                    "port %d is being answered by a gateway that is not ours: deployment=%r, "
                    "expected %r. SO_REUSEPORT means a foreign process can share this port; "
                    "pick another one."
                    % (self.port, payload.get("deployment"), self.fleet.deployment)
                )
            return payload if payload.get("status") == "ok" else None

        return wait_for("gateway %s to elect a backend" % self.name, probe, timeout)

    def stop(self):
        if self.proc is None:
            return None
        code = _terminate(self.proc)
        self.proc = None
        return code

    def wait_exit(self, timeout):
        """Return (returncode, wall-clock time the exit was observed)."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                self.exit_seen_at = time.time()
                return self.proc.returncode, self.exit_seen_at
            time.sleep(0.01)
        return None, None


def pid_alive(pid):
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def kill_pid(pid):
    if not pid:
        return
    for sig in (signal.SIGTERM, signal.SIGKILL):
        if not pid_alive(pid):
            return
        try:
            os.kill(pid, sig)
        except OSError:
            return
        for _ in range(50):
            if not pid_alive(pid):
                return
            time.sleep(0.02)


def sweep_orphans(marker, keep=()):
    """Kill anything whose argv still mentions this run's directory.

    The successor is spawned with `start_new_session=True` (contract, item C),
    so it is not in this process's group and a group kill cannot reach it. A
    failed handover that leaves one behind would otherwise hold the port and
    keep heart-beating at the fleet directory for as long as the machine is up.
    The marker is a unique temp path, so this can only ever match this run.
    """
    killed = []
    keep = set(keep) | {os.getpid()}
    try:
        entries = os.listdir("/proc")
    except OSError:
        return killed
    for entry in entries:
        if not entry.isdigit():
            continue
        pid = int(entry)
        if pid in keep:
            continue
        try:
            with open("/proc/%d/cmdline" % pid, "rb") as handle:
                raw = handle.read()
        except OSError:
            continue
        cmd = raw.decode("utf-8", "replace")
        if marker not in cmd:
            continue
        killed.append((pid, cmd.replace("\0", " ").strip()))
        kill_pid(pid)
    return killed


# ---------------------------------------------------------------------------
# The client
# ---------------------------------------------------------------------------
class RequestStream:
    """A steady concurrent stream of requests, recording every outcome."""

    def __init__(self, port, workers=10, pace=0.004, slow_every=8, slow_delay=0.5):
        self.port = port
        self.workers = workers
        self.pace = pace
        self.slow_every = slow_every
        self.slow_delay = slow_delay
        self.results = []
        self.marks = []  # (timestamp, label) -- the timeline a failure is read against
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._threads = []
        self._seq = itertools.count(1)
        self.started_at = None
        self.stopped_at = None

    def mark(self, label):
        with self._lock:
            self.marks.append((time.time(), label))

    def _record(self, out):
        with self._lock:
            self.results.append(out)

    def _worker(self, index):
        while not self._stop.is_set():
            seq = next(self._seq)
            convo = "stream-%d-%d" % (index, seq % 12)
            delay = self.slow_delay if seq % self.slow_every == 0 else 0.0
            payload = json.dumps(
                {
                    "model": "fake",
                    "max_tokens": 4,
                    "messages": [{"role": "user", "content": "ping %d" % seq}],
                }
            ).encode()
            headers = [
                ("x-api-key", API_USER),
                (CONVERSATION_HEADER, convo),
                ("x-test-seq", str(seq)),
            ]
            if delay:
                headers.append((DELAY_HEADER, "%.3f" % delay))
            out = request(
                self.port,
                body=payload,
                headers=headers,
                seq=seq,
                convo=convo,
                delay=delay,
                connect_timeout=5.0,
                # Comfortably past the slowest deliberate delay, so a timeout
                # here means something really stopped answering.
                read_timeout=20.0,
            )
            self._record(out)
            if self.pace:
                time.sleep(self.pace)

    def start(self):
        self.started_at = time.time()
        for index in range(self.workers):
            thread = threading.Thread(
                target=self._worker, args=(index,), name="stream-%d" % index, daemon=True
            )
            thread.start()
            self._threads.append(thread)
        return self

    def stop(self, join_timeout=25.0):
        self._stop.set()
        for thread in self._threads:
            thread.join(timeout=join_timeout)
        self.stopped_at = time.time()
        self._threads = []
        return self.results

    # -- analysis ---------------------------------------------------------
    def snapshot(self):
        with self._lock:
            return list(self.results)

    def refused(self):
        return [r for r in self.snapshot() if r.kind in REFUSED_KINDS]

    def lost(self):
        return [r for r in self.snapshot() if r.kind in LOST_KINDS]

    def errored(self):
        return [r for r in self.snapshot() if r.kind.startswith("http_")]

    def served(self):
        return [r for r in self.snapshot() if r.ok]

    def by_kind(self):
        counts = {}
        for out in self.snapshot():
            counts[out.kind] = counts.get(out.kind, 0) + 1
        return counts

    def by_backend(self):
        counts = {}
        for out in self.served():
            counts[out.backend] = counts.get(out.backend, 0) + 1
        return counts


class HealthProbe:
    """A second, low-rate stream that answers "which process is on the port?".

    Nothing in a proxied reply names the gateway generation that relayed it --
    the request line and headers go upstream verbatim and the reply comes back
    verbatim, by design. `/_gateway/health` does carry `uptime_s`, and the
    successor's is near zero while the predecessor's is however long the test
    has been running, so the pair separates cleanly as long as the predecessor
    has been up for a few seconds before the handover. That is the only
    generation signal the contract's surface offers; see the report.
    """

    def __init__(self, port, deployment, interval=0.02):
        self.port = port
        self.deployment = deployment
        self.interval = interval
        self.samples = []
        # The Outcome of every probe, so a refused or half-dead control
        # connection counts towards the zero-loss property too. The control
        # plane runs on the same listening socket as the traffic; a handover
        # that drops one drops the other.
        self.outcomes = []
        self._stop = threading.Event()
        self._thread = None
        self._lock = threading.Lock()

    def _loop(self):
        while not self._stop.is_set():
            status, payload, out = control(self.port, "GET", "/_gateway/health", timeout=5.0)
            sample = {
                "t": time.time(),
                "kind": out.kind,
                "status": status,
                "uptime_s": payload.get("uptime_s") if isinstance(payload, dict) else None,
                "deployment": payload.get("deployment") if isinstance(payload, dict) else None,
                "active": payload.get("active") if isinstance(payload, dict) else None,
            }
            with self._lock:
                self.samples.append(sample)
                self.outcomes.append(out)
            self._stop.wait(self.interval)

    def refused(self):
        with self._lock:
            return [out for out in self.outcomes if out.kind in REFUSED_KINDS]

    def lost(self):
        with self._lock:
            return [out for out in self.outcomes if out.kind in LOST_KINDS]

    def errored(self):
        with self._lock:
            return [out for out in self.outcomes if out.kind.startswith("http_")]

    def start(self):
        self._thread = threading.Thread(target=self._loop, name="health-probe", daemon=True)
        self._thread.start()
        return self

    def stop(self, join_timeout=15.0):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=join_timeout)
        return self.snapshot()

    def snapshot(self):
        with self._lock:
            return list(self.samples)

    def generations(self, boundary):
        """Label each sample A or B by the process start time its uptime implies.

        `boundary` is a wall-clock time after the predecessor started and at or
        before the successor started -- the handover request, in practice.
        """
        labelled = []
        for sample in self.snapshot():
            if sample["uptime_s"] is None:
                labelled.append((sample["t"], None, sample))
                continue
            implied_start = sample["t"] - sample["uptime_s"]
            labelled.append((sample["t"], "A" if implied_start < boundary - 2.0 else "B", sample))
        return labelled


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def timeline(stream, events, limit=12):
    """Render failures against the events they need to be read against."""
    lines = ["timeline (t=0 is the first request):"]
    base = stream.started_at or time.time()
    for when, label in sorted(list(events) + stream.marks):
        lines.append("  %+8.3fs  %s" % (when - base, label))
    bad = [r for r in stream.snapshot() if not r.ok]
    if bad:
        lines.append("first %d failures:" % min(limit, len(bad)))
        for out in sorted(bad, key=lambda r: r.t_start)[:limit]:
            lines.append(
                "  %+8.3fs  seq=%-6d kind=%-14s status=%-5s backend=%-9s %s"
                % (
                    out.t_start - base,
                    out.seq,
                    out.kind,
                    out.status,
                    out.backend,
                    out.error,
                )
            )
    return "\n".join(lines)


def stream_report(stream, events=()):
    served = stream.served()
    lines = [
        "requests: %d total, %d served, %d refused, %d lost in flight, %d http errors"
        % (
            len(stream.snapshot()),
            len(served),
            len(stream.refused()),
            len(stream.lost()),
            len(stream.errored()),
        ),
        "outcomes by kind: %s" % sorted(stream.by_kind().items()),
        "served by backend: %s" % sorted(stream.by_backend().items()),
        timeline(stream, events),
    ]
    return "\n".join(lines)


def make_root(prefix="gw-handover-"):
    return tempfile.mkdtemp(prefix=prefix)


def drop_root(root):
    shutil.rmtree(root, ignore_errors=True)


def new_rng():
    """Seeded off the pid and the clock so parallel runs diverge immediately."""
    return random.Random((os.getpid() << 20) ^ time.time_ns())


# ---------------------------------------------------------------------------
# One test's worth of world
# ---------------------------------------------------------------------------
class Scenario:
    """A fake fleet, a real gateway, and a guarantee that nothing is left running.

    Teardown is the part worth reading. The successor is spawned with
    `start_new_session=True`, so it is neither a child of this process nor a
    member of any group this process can signal -- a failed handover leaves a
    process holding the port and heart-beating at a temp directory forever. So
    teardown kills by pid where it knows one, and then sweeps `/proc` for
    anything whose argv still names this run's directory.
    """

    def __init__(
        self, name, backends=2, extra_args=(), log_level="INFO", probe_delay=0.0, pools=None
    ):
        self.name = name
        self.rng = new_rng()
        self.root = make_root("gw-handover-%s-" % name)
        self.fleet = Fleet(
            self.root, self.rng, backends=backends, probe_delay=probe_delay, pools=pools
        )
        self.lease = lease_port(self.rng)
        self.port = self.lease.port
        self.gateway = GatewayProc(
            self.fleet, self.lease, name="A", extra_args=extra_args, log_level=log_level
        )
        self.successor_pid = None
        self.extra_gateways = []
        self.notes = []

    # -- lifecycle --------------------------------------------------------
    def start(self, wait=True, serving_timeout=30.0):
        self.fleet.start_backends()
        self.fleet.start_heartbeats()
        self.gateway.start()
        if wait:
            self.gateway.wait_serving(timeout=serving_timeout)
        return self

    def add_gateway(self, name, extra_args=(), log_level="INFO", lease=None):
        lease = lease or lease_port(self.rng)
        proc = GatewayProc(self.fleet, lease, name=name, extra_args=extra_args, log_level=log_level)
        self.extra_gateways.append(proc)
        return proc

    def note(self, text):
        self.notes.append(text)

    def close(self):
        # Diagnostics first: pytest hides captured stdout on a pass and shows it
        # on a failure, which is exactly when these are worth having.
        print(self.diagnostics())
        if self.successor_pid:
            kill_pid(self.successor_pid)
        for proc in [self.gateway] + self.extra_gateways:
            proc.stop()
        self.fleet.stop()
        self.lease.release()
        for proc in self.extra_gateways:
            proc.lease.release()
        stragglers = sweep_orphans(self.root)
        if stragglers:
            print("WARNING: killed %d orphaned process(es): %s" % (len(stragglers), stragglers))
        drop_root(self.root)

    # -- observation ------------------------------------------------------
    def diagnostics(self, lines=40):
        chunks = [
            "--- scenario %s (port %d, deployment %s) ---"
            % (self.name, self.port, self.fleet.deployment)
        ]
        for text in self.notes:
            chunks.append("note: %s" % text)
        for proc in [self.gateway] + self.extra_gateways:
            chunks.append(
                "--- gateway %s pid=%s alive=%s rc=%s ---"
                % (
                    proc.name,
                    proc.pid,
                    proc.alive(),
                    None if proc.proc is None else proc.proc.returncode,
                )
            )
            chunks.append(proc.log_tail(lines))
        calls = self.fleet.serve_sh_calls()
        chunks.append("serve.sh calls (ppid, t, argv): %s" % (calls or "none"))
        return "\n".join(chunks)

    # -- actions ----------------------------------------------------------
    def start_handover(self, timeout=20.0):
        """POST /_gateway/handover, remembering the successor so teardown can kill it."""
        return self.start_handover_with({}, timeout=timeout)

    def start_handover_with(self, body, timeout=20.0):
        """start_handover(), with a request body."""
        status, payload, out = control(
            self.port, "POST", "/_gateway/handover", body, timeout=timeout
        )
        require_endpoint(status, "POST", "/_gateway/handover", "HTTP surface (exact)")
        if status is None:
            raise HarnessError(
                "POST /_gateway/handover did not answer: %s (%s)" % (out.kind, out.error)
            )
        if isinstance(payload, dict):
            pid = payload.get("successor_pid")
            if isinstance(pid, int) and pid > 0:
                self.successor_pid = pid
        return status, payload

    def handover_state(self, timeout=10.0):
        """GET /_gateway/handover.

        Note for whoever reads a failure here: after step 4 the successor is
        bound to the same port, so this request may be answered by EITHER
        process and a `phase` of "idle" may simply mean the successor replied.
        The tests never gate on this field for that reason -- they gate on
        process state -- but it is worth recording.
        """
        status, payload, _ = control(self.port, "GET", "/_gateway/handover", timeout=timeout)
        return status, payload

    def set_fleet_health(self, healthy):
        for backend in self.fleet.backends:
            backend.set_healthy(healthy)

    def wait_health_status(self, want, timeout=25.0):
        def probe():
            status, payload, _ = control(self.port, "GET", "/_gateway/health", timeout=3.0)
            if status == 200 and isinstance(payload, dict) and payload.get("status") == want:
                return payload
            return None

        return wait_for("/_gateway/health to report %r" % want, probe, timeout)

    def backend_access_log(self, job_id):
        """Dicts of {kind, t, job, method, path, seq, delay} for relayed requests.

        `kind` is "start" or "end". The start line is what the drain tests time
        against: it is written before the fake backend's artificial delay, so a
        long request is observable while it is still in flight.
        """
        backend = next(b for b in self.fleet.backends if b.job_id == job_id)
        rows = []
        try:
            with open(backend.access_log) as handle:
                for line in handle:
                    parts = line.rstrip("\n").split("\t")
                    if len(parts) != 7 or parts[0] not in ("start", "end"):
                        continue
                    try:
                        rows.append(
                            {
                                "kind": parts[0],
                                "t": float(parts[1]),
                                "job": parts[2],
                                "method": parts[3],
                                "path": parts[4],
                                "seq": parts[5],
                                "delay": float(parts[6]),
                            }
                        )
                    except ValueError:
                        continue
        except OSError:
            pass
        return rows

    def all_backend_requests(self):
        rows = []
        for backend in self.fleet.backends:
            rows.extend(self.backend_access_log(backend.job_id))
        return sorted(rows, key=lambda row: row["t"])

    def slow_request_started(self, at_least):
        """Start lines for requests the backend was told to hold `at_least` seconds."""
        return [
            row
            for row in self.all_backend_requests()
            if row["kind"] == "start" and row["delay"] >= at_least - 0.01
        ]


def send_and_reset(port, path="/v1/messages", headers=(), body=b"", linger_after=0.3):
    """Send a request, then kill the client socket with an RST mid-flight.

    This is the case the integrator's correction is about: `Server.wait_closed()`
    is keyed on `Server._active_count`, which drops when the *transport* dies --
    so after this RST a `wait_closed()`-only drain declares the process idle
    while the handler is still relaying. Production does this 15,785 times; a
    drain that cannot survive it is not a drain.

    Returns the moment the reset was sent.
    """
    sock = socket.socket()
    sock.settimeout(10.0)
    sock.connect(("127.0.0.1", port))
    lines = ["POST %s HTTP/1.1" % path, "Host: 127.0.0.1:%d" % port]
    lines.extend("%s: %s" % (name, value) for name, value in headers)
    lines.append("Content-Length: %d" % len(body))
    lines.append("Content-Type: application/json")
    lines.append("Connection: close")
    sock.sendall(("\r\n".join(lines) + "\r\n\r\n").encode("latin-1") + body)
    time.sleep(linger_after)
    # SO_LINGER with a zero timeout turns close() into an RST rather than a FIN,
    # which is what a client process being killed actually produces.
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
    sock.close()
    return time.time()
