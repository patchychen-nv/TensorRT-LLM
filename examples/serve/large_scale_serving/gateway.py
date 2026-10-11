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
"""Stable front door for the Anthropic-compatibility server.

A serving job lives for hours and lands on whatever node the scheduler gives
it, so its URL changes every time it is rescheduled. Agent CLIs need one
address that never changes. This gateway holds that address and forwards to
whichever backend is currently healthy, so `ANTHROPIC_BASE_URL` is written once
and never edited again.

    python3 gateway.py --fleet-dir /shared/fleet --users gateway_users.txt --no-relay

Backends announce themselves by dropping one JSON file per job into
`--fleet-dir` (see `discover` for the fields). Nothing else couples the gateway
to the jobs: it never talks to the launcher on the request path, so an
unhealthy or missing registry only costs routing, never correctness.

Relay mode (`--yaml` without `--no-relay`) additionally submits the successor
job before the current one hits its wall clock and reclaims the predecessor
once the successor has proven itself. That half drives the serving jobs through
a launcher script exposing `submit`, `restart` and `quit` subcommands, passed
with `--serve-sh`; `serve.sh` beside this file provides them and is the
default, so relay mode needs no extra argument here. On a fleet where several
instances serve at once, `--relay-per-instance` rolls each one against its own
wall clock through fleetctl instead of running a single lineage.

Nothing schedules the handover: `elect` simply prefers the healthy backend
with the furthest end time, so a successor wins the moment it passes /health,
and `--lead-time` decides only how early it is submitted.

Standard library only, on purpose: the gateway has to outlive every serving
job, so it runs outside the TRT-LLM container on whatever long-lived host is
available. Requiring httpx or uvicorn there would mean a venv, which means
outbound network -- one more thing that host has to provide.

Reading the access log
----------------------
Every line carries `rid=rNNNNNN`; grepping one id gives that request end to
end, which is the only way to read the log at all once requests overlap. The
status field is the backend's own status plus at most two suffixes:

    200     relayed intact
    200!    the backend's response ended early: a stream with no terminal
            event got one appended, any other body was cut off by closing
            the connection
    200?    the ending could not be delivered; the client had already gone
    502     the BACKEND failed -- it never answered, or answered unusably

The distinction between `200?` and `502` is the point. Both used to be logged
as 502, so a client that hung up mid-stream was indistinguishable from a
serving job that had fallen over, and the counts blamed the backend for both.
`side=` on each warning names which end of the proxy actually broke.

Pools: more than one fleet directory
------------------------------------
`--fleet-dir` is the default pool. Each `--pool NAME=FLEET_DIR[,FLEET_CONFIG]`
adds another fleet directory -- a second model's fleet, say -- and the same
specs are read, one per line, from `pools.conf` beside the users file (see
`--pools-file`). A pool decides two things and only two: where its backends
are discovered, and what supervises them -- revive, recovery, walltime relay
and drain act on a pool's jobs through that pool's own fleet config, and a
pool with none is never acted on. Routing sees pools in one place only: a
new conversation may be placed on any backend of any pool, whatever model a
request names, but once served it is only re-placed (a lost backend, an
abandoned request, a retry) inside the pool it started in -- pools are models,
and a conversation must not continue on a model that did not write its history
(`--pool-affinity`, on by default; see Router). With no extra pool configured
nothing here changes -- not a log line, not a status field.
"""

import argparse
import asyncio
import collections
import errno
import fcntl
import glob
import hashlib
import importlib.util
import itertools
import json
import logging
import math
import os
import re
import signal
import sys
import threading
import time

LOG = logging.getLogger("gateway")

# Hop-by-hop request headers plus the ones this gateway owns. Request framing
# headers (content-length, transfer-encoding, te, trailer) deliberately survive:
# request bodies are relayed byte for byte. SSE response framing is normalized
# separately so the gateway can append a valid terminal error event.
STRIP_REQUEST_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "upgrade",
    "host",
    "x-api-key",
    "authorization",
    "accept-encoding",
    # Owned by the gateway: it is the only identity the backend sees, so a
    # client copy must not ride along beside the one written below.
    "x-gateway-user",
}

MAX_HEAD_BYTES = 64 * 1024
RELAY_CHUNK = 64 * 1024
PENDING_VISIBILITY_GRACE = 60

# The pool `--fleet-dir` describes. Reserved: no extra pool may take the name,
# so a status report or a control request can always say "default" and mean it.
DEFAULT_POOL = "default"

# Identity used when a client sends no key at all. It is an ordinary allowlist
# entry, not a bypass: keyless requests are refused unless this name is listed
# in the users file, and the access log then attributes them all to one shared
# identity.
ANONYMOUS_USER = "anonymous"

# Anthropic-shaped so a client's error handling reports something meaningful
# instead of a bare transport failure.
ERROR_BODIES = {
    401: (
        "authentication_error",
        "unknown api key; ask the gateway owner to add your username to the users file",
    ),
    404: ("not_found_error", "no such gateway endpoint"),
    405: ("invalid_request_error", "this endpoint only accepts POST"),
    502: ("api_error", "backend refused the connection"),
    503: (
        "overloaded_error",
        "no healthy backend right now; the serving job is rotating, retry shortly",
    ),
}

ERROR_REASONS = {
    401: "Unauthorized",
    404: "Not Found",
    405: "Method Not Allowed",
    502: "Bad Gateway",
    503: "Service Unavailable",
}

SSE_ROTATED = (
    b"event: error\n"
    b'data: {"type":"error","error":{"type":"overloaded_error",'
    b'"message":"backend rotated mid-stream; the response is '
    b'incomplete, please resend"}}\n\n'
)


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------
# Monotonic rather than random: request ids exist to be read in a log file, and
# increasing ones also show how many requests the gateway has taken since it
# started. Wrapped so the field stays a fixed width in the access log.
_REQUEST_IDS = itertools.count(1)


def next_request_id():
    return "r%06d" % (next(_REQUEST_IDS) % 1000000)


class ClientAbandoned(ConnectionError):
    """The client hung up while its request was still waiting on the backend.

    Raised by the watcher in `proxy`, not by an I/O call: nothing on the
    client socket fails when the client leaves quietly, so without the watcher
    the gateway learns of the departure only at the first write -- which for a
    request stuck in a saturated backend's queue is minutes away. By then the
    abandoned request has held its queue slot the whole time and every retry
    the client sent has stacked up behind it.
    """


class SideError(OSError):
    """An I/O failure tagged with the side of the proxy that produced it.

    The relay reads from the backend and writes to the client inside a single
    try block. An untagged OSError from one side is indistinguishable from the
    other, and the handlers used to report every one of them as "upstream
    failed" -- which was simply wrong whenever the client was the end that went
    away. A post-mortem then blames the serving job for a client disconnect.

    Deliberately an OSError subclass rather than a new branch of the exception
    hierarchy: every handler on the request path already catches OSError, so
    adding the tag can never turn a handled failure into an escaped one. The
    tag is purely additive, and code that does not care reads it as before.
    """

    def __init__(self, side, exc):
        super().__init__("%s %s: %s" % (side, type(exc).__name__, exc))
        self.side = side
        self.cause = exc


def error_side(exc, default="unknown"):
    """Name the side an exception came from, for exceptions that carry no tag.

    The default is "unknown" rather than a plausible guess on purpose. Guessing
    is what produced the misleading logs this machinery replaces: an untagged
    failure reported as the likelier side reads exactly like a measured one,
    and there is no way to tell them apart afterwards. A caller that can prove
    the side from its position in the code passes it explicitly.
    """
    return getattr(exc, "side", default)


async def read_upstream(reader, size):
    """Read from the backend, tagging failures with the side."""
    try:
        return await reader.read(size)
    except OSError as exc:
        raise SideError("upstream_read", exc) from exc


async def read_body(reader, rest, headers, limit):
    """Read the whole request body, when it is bounded and worth buffering.

    Returns (body, remainder). A body of None means it was not buffered -- no
    content-length, chunked, or over the limit -- and the caller must fall back
    to pumping `remainder` and whatever follows it.

    Buffering exists only so the conversation can be identified before a
    backend is chosen. It is capped because the alternative, holding an
    unbounded client upload in memory, trades a routing optimisation for an
    availability risk.
    """
    if header_value(headers, "transfer-encoding"):
        return None, rest
    raw = header_value(headers, "content-length")
    if raw is None:
        return None, rest
    try:
        length = int(raw)
    except ValueError:
        return None, rest
    if length <= 0:
        return b"", rest
    if length > limit:
        return None, rest
    chunks = [rest[:length]]
    have = len(chunks[0])
    leftover = rest[length:]
    while have < length:
        chunk = await read_upstream(reader, min(RELAY_CHUNK, length - have))
        if not chunk:
            # The client stopped mid-body. Give back what arrived and let the
            # backend reject the short request on its own terms, rather than
            # inventing an error for a conversation we cannot even name.
            return None, b"".join(chunks)
        chunks.append(chunk)
        have += len(chunk)
    return b"".join(chunks), leftover


async def write_client(writer, data):
    """Write to the downstream client, tagging failures with the side."""
    try:
        writer.write(data)
        await writer.drain()
    except OSError as exc:
        raise SideError("client_write", exc) from exc


class RequestTrace:
    """Per-request diagnostic state, populated as the request progresses.

    Failure logs used to name only the exception. That is not enough to analyse
    anything after the fact: warnings from concurrent requests interleave with
    nothing to correlate them, the side that failed is unrecorded, and both the
    backend's own status code and the stream's progress are discarded at the
    moment they become interesting. Every failure path emits `detail()`, so a
    single grep on the id reconstructs one request end to end.
    """

    __slots__ = ("rid", "upstream_status", "tracker", "request_body_error", "conversation", "pool")

    def __init__(self):
        self.rid = next_request_id()
        self.upstream_status = None
        self.tracker = None
        self.request_body_error = None
        self.conversation = None
        # The extra pool the serving backend registered in, so the access log
        # says which fleet answered. None for the default pool, so a
        # default-pool line reads exactly as it did before pools existed.
        self.pool = None

    def detail(self):
        parts = ["rid=%s" % self.rid]
        if self.pool is not None:
            parts.append("pool=%s" % self.pool)
        if self.conversation is not None:
            # Which conversation a failure belongs to is the first thing wanted
            # when one agent session misbehaves and five others are fine.
            parts.append("convo=%s" % short_convo(self.conversation))
        if self.upstream_status is not None:
            parts.append("upstream_status=%s" % self.upstream_status)
        if self.tracker is not None:
            parts.append(self.tracker.progress())
        if self.request_body_error is not None:
            # A client that died mid-upload leaves the backend holding a
            # truncated request, and the backend's reaction to that reads like
            # a server bug unless this says otherwise.
            parts.append("request_body=%s" % self.request_body_error)
        return " ".join(parts)


# ---------------------------------------------------------------------------
# Fleet state
# ---------------------------------------------------------------------------
def finite_float(value, field, source=""):
    """Coerce to float, rejecting NaN and infinity.

    float() happily accepts "nan" and "inf", and NaN then poisons every
    comparison it touches: `now - nan > stale_after` is False, so a dead
    backend with a NaN heartbeat never goes stale and is never retired, while
    a NaN end_time makes the max() in the election return an arbitrary winner.
    Both fail silently, which is why they are rejected at the boundary.
    """
    number = float(value or 0)
    if not math.isfinite(number):
        raise ValueError("%s is not finite (%r)%s" % (field, value, source))
    return number


class Backend:
    """One serving job, as seen through its registration file."""

    def __init__(self, record, pool=DEFAULT_POOL):
        self.job_id = str(record["job_id"])
        # Which pool's directory it registered in, and so which fleet config
        # supervises it. Decided by the directory, never by anything the record
        # says about itself: a record that could name its own pool could hand
        # its lifecycle to another fleet's supervisor. Routing never reads it.
        self.pool = pool
        self.url = record["url"].rstrip("/")
        self.run_dir = record.get("run_dir", "")
        self.state = record.get("state", "")
        self.end_time = finite_float(record.get("end_time"), "end_time")
        self.heartbeat = finite_float(record.get("heartbeat"), "heartbeat")
        self.healthy = False
        self.timeouts = 0  # consecutive probe timeouts
        self.healthy_since = 0.0
        # Timestamps of recent connection refusals on REAL requests, which is
        # the same evidence probe() acts on ("dead") arriving from a much
        # denser sampler. The probe asks once every few seconds; traffic asks
        # hundreds of times. When a backend died here the request stream
        # reported 192 refusals inside one minute while the probe had taken
        # about seven samples, and every one of those 192 was a 502 somebody
        # had to retry.
        self.refusals = []
        # Probing resolves the URL once; every request reuses host/port.
        match = re.match(r"^http://([^:/]+):(\d+)$", self.url)
        if not match:
            raise ValueError("unusable url %r" % self.url)
        self.host = match.group(1)
        self.port = int(match.group(2))

    def refresh(self, record):
        """Take every mutable field, not just state and heartbeat.

        A registration file is rewritten as the job learns about itself. Two
        fields in particular arrive late, so latching the first value read is
        not enough: `end_time`, absent until the job resolves its wall clock -
        and `end_time == 0` makes the supervisor skip relay entirely, so the
        fleet would expire with no successor queued, which is the one failure
        the relay exists to prevent. And `url`, which changes when a failed
        successor is restarted in place and binds a different port, leaving the
        gateway proxying a dead address while a fresh heartbeat keeps the entry
        from ever being retired.
        """
        self.state = record.get("state", self.state)
        self.heartbeat = finite_float(record.get("heartbeat"), "heartbeat")
        self.end_time = finite_float(record.get("end_time") or self.end_time, "end_time")
        self.run_dir = record.get("run_dir", self.run_dir)
        # Normalised the same way as in __init__, so a rewrite that differs
        # only by a trailing slash does not read as a move.
        url = (record.get("url") or "").rstrip("/")
        if url and url != self.url:
            match = re.match(r"^http://([^:/]+):(\d+)$", url)
            if not match:
                LOG.warning("ignoring unusable url %r for %s", url, self.job_id)
            else:
                LOG.info("%s moved from %s to %s", self.job_id, self.url, url)
                self.url = url
                self.host = match.group(1)
                self.port = int(match.group(2))


# ---------------------------------------------------------------------------
# Pools: fleet directories beyond --fleet-dir
# ---------------------------------------------------------------------------
PoolSpec = collections.namedtuple("PoolSpec", "name fleet_dir fleet_config")

# The name tags log lines, keys the status report, and is what an HTTP
# registration says to land in a pool's directory, so it gets the same
# whitelist a job id does.
_POOL_NAME = re.compile(r"\A[A-Za-z0-9][A-Za-z0-9._-]{0,63}\Z")


def parse_pool_spec(text):
    """`NAME=FLEET_DIR[,FLEET_CONFIG]` as a PoolSpec. Raises ValueError.

    FLEET_CONFIG is kept exactly as written, like --fleet-config: it is handed
    to fleetctl, which may run on another host where this one's idea of a
    relative path means nothing.
    """
    text = (text or "").strip()
    name, sep, rest = text.partition("=")
    name = name.strip()
    if not sep or not name:
        raise ValueError("pool spec %r is not NAME=FLEET_DIR[,FLEET_CONFIG]" % text)
    if not _POOL_NAME.match(name):
        raise ValueError(
            "pool name %r must be 1-64 chars of [A-Za-z0-9._-] and start alphanumeric" % name
        )
    if name == DEFAULT_POOL:
        raise ValueError("pool name %r is reserved for --fleet-dir" % name)
    items = [item.strip() for item in rest.split(",")]
    if not items[0]:
        raise ValueError("pool %s names no fleet directory" % name)
    if len(items) > 2:
        raise ValueError("pool %s: %r has more than NAME=FLEET_DIR[,FLEET_CONFIG]" % (name, text))
    fleet_config = items[1] if len(items) == 2 else ""
    if len(items) == 2 and not fleet_config:
        raise ValueError("pool %s: empty fleet config in %r" % (name, text))
    return PoolSpec(name, os.path.abspath(items[0]), fleet_config)


def read_pools_file(path):
    """Pool specs from a file, one per line, `#` comments. [] if there is no file.

    Any other failure raises, deliberately: a gateway that started without a
    pool it was told about would silently drop that pool's backends -- out of
    routing, and out of supervision. Refusing to start is safe -- a handover
    successor that exits is aborted, and its predecessor keeps serving.
    """
    try:
        with open(path) as handle:
            lines = handle.read().splitlines()
    except FileNotFoundError:
        return []
    specs = []
    for number, line in enumerate(lines, 1):
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        try:
            specs.append(parse_pool_spec(line))
        except ValueError as exc:
            raise ValueError("%s:%d: %s" % (path, number, exc)) from None
    return specs


def combine_pools(flags, from_file, default_fleet_dir):
    """The configured pools, checked as a set. Raises ValueError.

    A pool may be stated both by flag and in the file -- the usual way to move
    one from the file a live gateway picked it up from into the job script for
    its next start -- but only if both say the same thing.
    """
    pools = collections.OrderedDict()
    for spec in list(flags) + list(from_file):
        known = pools.get(spec.name)
        if known is not None and known != spec:
            raise ValueError(
                "pool %s is defined twice, differently: %s / %s" % (spec.name, known, spec)
            )
        pools[spec.name] = spec
    owners = {os.path.abspath(default_fleet_dir): DEFAULT_POOL}
    for spec in pools.values():
        # One directory, one pool: a backend's pool is read off the directory
        # its registration is in, so a shared one would make it both.
        if spec.fleet_dir in owners:
            raise ValueError(
                "pool %s shares fleet dir %s with pool %s"
                % (spec.name, spec.fleet_dir, owners[spec.fleet_dir])
            )
        owners[spec.fleet_dir] = spec.name
    return list(pools.values())


class Pool:
    """An extra pool: its configuration, and the lifecycle state kept per pool.

    The state attributes have the names and meanings of the Fleet attributes
    the default pool has always kept them in. PoolView relies on that to run
    one lifecycle implementation over either. None of it is routing state:
    new conversations are placed across every directory, and only the Router
    keeps a placed conversation inside its pool.
    """

    def __init__(self, spec):
        self.name = spec.name
        self.fleet_dir = spec.fleet_dir
        self.fleet_config = spec.fleet_config
        self.active = None
        self.pending = None
        self.ever_active = False
        self.stopped = False
        self.last_submit = 0.0
        self.seen_running = set()
        self.superseded = set()
        self.draining = {}
        self.relaying = {}
        self.lost = {}
        self.recovered = {}
        self.last_recovery = 0.0


def pool_tag(fleet):
    """Log-line prefix naming the pool a lifecycle message is about.

    Empty for the default pool -- and for anything that predates pools, which
    is what the getattr is for -- so its lines read exactly as they always did.
    """
    return getattr(fleet, "pool_tag", "")


def extra_pool(backend):
    """The pool a backend registered in, for a log line -- None for the default."""
    return None if backend.pool == DEFAULT_POOL else backend.pool


def decode_json_object(data):
    """A request body as a dict, or None for anything that is not a JSON object."""
    if not data:
        return None
    try:
        payload = json.loads(data)
    except (ValueError, RecursionError):
        return None
    return payload if isinstance(payload, dict) else None


class Fleet:
    """Everything the request path and the supervisor share."""

    def __init__(self, args):
        self.args = args
        self.backends = {}  # job_id -> Backend
        self.active = None  # job_id currently taking new requests
        self.draining = {}  # job_id -> reclaim deadline (unix ts)
        # A default dict on purpose. Requests outlive their backend's entry --
        # discovery can retire a job while its streams are still draining -- and
        # the release below runs in a finally that also closes the client
        # socket. A KeyError there would leak the connection, so counting must
        # not be able to raise.
        self.inflight = collections.defaultdict(int)
        self.users = set()
        self.users_mtime = 0.0
        # (job_id, submitted_at), retained through registration until healthy.
        self.pending = None
        self.ever_active = False  # distinguish recovery from initial startup
        # Set by stop_server, cleared by start_server. Separate from
        # ever_active because elect() overwrites that one.
        self.stopped = False
        self.last_submit = 0.0
        self.seen_running = set()
        # Replaced, but not yet cleared for reclaim: draining ends in a `quit`,
        # so it waits until the successor has proven itself.
        self.superseded = set()
        # instance label -> when its successor was last submitted. Per label so
        # that one instance rolling never delays another.
        self.relaying = {}
        # job_id -> (restarts attempted, last attempt). Kept on the fleet
        # rather than the Backend so a job that re-registers cannot reset its
        # own budget by being rediscovered.
        self.revived = {}
        # job_id -> (run_dir, retired_at) for backends that left the table.
        # revive_dead_backends cannot see these: a preempted deployment takes
        # its controller with it, and the controller is what deletes the
        # registration. Recovery therefore needs its own record of what used
        # to be here, kept until the job is confirmed gone from the scheduler
        # or the instance comes back.
        self.lost = {}
        # job_id -> (recoveries attempted, last attempt), same reasoning as
        # `revived`: a job that reappears and vanishes again must not get a
        # fresh budget just for having been rediscovered.
        self.recovered = {}
        self.last_recovery = 0.0
        # Extra pools -- fleet directories beyond --fleet-dir -- in the order
        # configured. The default pool is not in here: its lifecycle state is
        # this object's own attributes above, exactly as before pools existed,
        # and each Pool carries the same attributes for itself.
        self.pools = collections.OrderedDict(
            (spec.name, Pool(spec)) for spec in getattr(args, "pools", None) or ()
        )
        # The default pool's log prefix. A PoolView carries its own.
        self.pool_tag = ""
        # (job id, pool) pairs already reported as registered in two pools, so
        # a stray copy of a registration says so once rather than every sweep.
        self._collisions = set()
        # Lifecycle authority, which is not routing authority: during a handover
        # two gateways both route, and only the one holding this may act on the
        # fleet. It lives here rather than in main_async because "everything the
        # request path and the supervisor share" is what this class is, and both
        # halves need to be able to ask whether this process is the supervisor.
        # Nothing is opened until the first acquire, so building a Fleet still
        # touches no files.
        self.supervisor_lock = SupervisorLock(
            getattr(args, "supervisor_lock", None)
            or os.path.join(args.fleet_dir, ".supervisor.lock")
        )
        self.started = time.time()
        # job id -> [(host, port), ...] to copy that backend's requests to.
        # A target is not a fleet member and never becomes one: it is not
        # discovered, not probed, not routed to, and its answers are read only
        # to be thrown away. The point is to put a real workload in front of a
        # server whose behaviour is being compared, without that server being
        # able to affect the answer any caller receives. Several targets per
        # backend, because the comparison is usually between candidates -- two
        # builds shadowing one live instance -- and running them one at a time
        # is comparing against different traffic.
        self.mirror_stats = collections.Counter()
        # The same counters, split by target. With one target the totals were
        # the story; with two, "failed: 50" does not say which one, and the
        # answer decides whether an experiment's numbers are worth keeping.
        self.mirror_target_stats = collections.defaultdict(collections.Counter)
        # Consecutive failures per target. A mirror is added to measure
        # something and then forgotten about; without this, a target that goes
        # away leaves the gateway attempting a connection per request for as
        # long as nobody notices, and the only trace is a counter nobody is
        # watching. Reset by the first copy that succeeds.
        self.mirror_misses = collections.Counter()
        # Bounds the copies, because they are the half of this that can hurt.
        # A mirror target that stops reading would otherwise accumulate one
        # pending connection per request until the gateway runs out of them,
        # and the traffic being mirrored is the traffic that matters.
        self.mirror_slots = asyncio.Semaphore(max(1, getattr(args, "mirror_concurrency", 64)))
        self.router = Router(
            args.sticky_ttl,
            args.sticky_capacity,
            policy=args.route_policy,
            state_path=args.router_state,
            key_sources=args.key_sources or DEFAULT_KEY_SOURCES,
            pool_affinity=getattr(args, "pool_affinity", True),
        )
        # Loaded before the saved state is restored, so a policy that came from
        # a file is a legitimate thing to have been using before the restart.
        # Otherwise every rotation would silently demote a custom policy back
        # to the default, which is the sort of thing nobody notices until the
        # placement they tuned for stops happening.
        policy_dir = getattr(args, "policy_dir", None)
        self.router.policies = PolicyDir(policy_dir) if policy_dir else None
        if self.router.policies is not None:
            self.router.policies.reload()
        self.router.load()
        if getattr(args, "route_policy", "least_conversations") not in known_policies(self.router):
            LOG.warning(
                "--route-policy %r is not a built-in and no such file is in "
                "--policy-dir; placement will use least_conversations until "
                "it appears",
                getattr(args, "route_policy", None),
            )

    # -- users ------------------------------------------------------------
    def reload_users(self):
        path = self.args.users
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            if self.users:
                LOG.warning(
                    "users file disappeared: %s (keeping %d entries)", path, len(self.users)
                )
            return
        if mtime == self.users_mtime:
            return
        names = set()
        with open(path) as handle:
            for line in handle:
                line = line.split("#", 1)[0].strip()
                if line:
                    names.add(line)
        self.users_mtime = mtime
        if names != self.users:
            LOG.info("users reloaded: %d entries", len(names))
        self.users = names

    # -- discovery --------------------------------------------------------
    def read_registrations(self):
        """Read the registration directory. Blocking, and nothing else.

        Split out of discover() so the caller can put it on a thread: the
        directory lives on the shared filesystem every serving job writes to,
        and at 46 backends one sweep costs ~45 ms there against ~1 ms locally.
        Spent on the event loop that is 45 ms of added tail latency on every
        stream in flight. Only the reads move -- applying the result stays on
        the loop, because the backend table is read by the request path and a
        dict that changes size mid-iteration raises.

        Every pool's directory, the default first: (path, record, pool).
        """
        records = []
        for pool, fleet_dir in self.pool_dirs():
            for path in glob.glob(os.path.join(fleet_dir, "*.json")):
                try:
                    with open(path) as handle:
                        records.append((path, json.load(handle), pool))
                except (OSError, ValueError):
                    # Mid-rename or truncated. The writer replaces the file
                    # atomically, so the next sweep gets a whole one. Stay
                    # quiet: this is expected and would otherwise log on
                    # every sweep.
                    continue
        return records

    # -- pools ------------------------------------------------------------
    def pool_names(self):
        return [DEFAULT_POOL] + list(self.pools)

    def pool_dirs(self):
        """(pool, fleet dir) for every pool, the default first."""
        return [(DEFAULT_POOL, self.args.fleet_dir)] + [
            (pool.name, pool.fleet_dir) for pool in self.pools.values()
        ]

    def pool_dir(self, name):
        return self.args.fleet_dir if name == DEFAULT_POOL else self.pools[name].fleet_dir

    def known_pool(self, name):
        return name == DEFAULT_POOL or name in self.pools

    def state_for(self, name):
        """Where a pool's lifecycle state lives: an extra pool's own, or here."""
        return self if name == DEFAULT_POOL else self.pools.get(name, self)

    @staticmethod
    def tag_for(name):
        return "" if name == DEFAULT_POOL else "[%s] " % name

    def view(self, name):
        return PoolView(self, name)

    def lifecycle_view(self):
        """What the default pool's lifecycle endpoints act on.

        The Fleet itself while it is the only pool, which is exactly what they
        always acted on; with extra pools, the default pool's view, so that
        stop_server cannot release jobs another pool's fleet config owns.
        """
        return self.view(DEFAULT_POOL) if self.pools else self

    def supervised_views(self):
        """Every pool that lifecycle actions may touch.

        The default pool always, and each extra pool with a fleet config --
        the only thing that may act on that pool's jobs.
        """
        return [self.view(DEFAULT_POOL)] + [
            self.view(pool.name) for pool in self.pools.values() if pool.fleet_config
        ]

    def pool_actives(self):
        return [(DEFAULT_POOL, self.active)] + [
            (pool.name, pool.active) for pool in self.pools.values()
        ]

    def overall_active(self):
        """The backend one election over every pool would pick.

        Each pool elects its own `active` for its lifecycle -- whose successor
        supersedes whom is a question inside one fleet. Routing is one pool, so
        what /health reports is the longest-lived of those winners, which is
        exactly what a single election over every backend picks. With no extra
        pool it is simply `active`.
        """
        if not self.pools:
            return self.active
        winners = [job for _, job in self.pool_actives() if job in self.backends]
        if not winners:
            return None
        return max(winners, key=lambda job: self.backends[job].end_time)

    def pool_report(self, accepting, conversations):
        """Per-pool counts for /_gateway/fleet."""
        report = {}
        for name in self.pool_names():
            pool = self.pools.get(name)
            members = {j: b for j, b in self.backends.items() if b.pool == name}
            report[name] = {
                "fleet_dir": pool.fleet_dir if pool else self.args.fleet_dir,
                "fleet_config": (
                    pool.fleet_config if pool else getattr(self.args, "fleet_config", "")
                )
                or None,
                # The default pool is always supervised; recovery inside it
                # still needs --fleet-config, as it always has.
                "supervised": bool(pool.fleet_config) if pool else True,
                "active": pool.active if pool else self.active,
                "backends": len(members),
                "healthy": sum(1 for b in members.values() if b.healthy),
                "accepting": sorted(j for j in members if j in accepting),
                "serving": sorted(j for j, b in members.items() if b.healthy),
                "inflight": sum(self.inflight.get(j, 0) for j in members),
                "conversations": sum(conversations.get(j, 0) for j in members),
            }
        return report

    def discover(self, records=None):
        """Rebuild the backend table from the registration directory.

        Each serving job owns exactly one file named after its scheduler job id,
        so there is never more than one writer per file and the gateway never
        has to coordinate with anybody. The union is just the directory listing.

        `records` is what read_registrations() returned, for callers that did
        the reads elsewhere; omitting it reads them here, which is what the
        synchronous start-up path and the tests want.
        """
        now = time.time()
        seen = set()
        if records is None:
            records = self.read_registrations()
        for item in records:
            path, record = item[0], item[1]
            pool = item[2] if len(item) > 2 else DEFAULT_POOL
            # A registration has to be a JSON object. Valid JSON that is not one
            # (a bare array, string or number) would otherwise reach .get() and
            # raise AttributeError, which is not a coercion error and escapes
            # the guard below -- aborting the sweep before the retirement pass.
            if not isinstance(record, dict):
                LOG.warning("ignoring %s: registration is not a JSON object", path)
                continue
            try:
                job_id = str(record.get("job_id", ""))
                heartbeat = finite_float(record.get("heartbeat"), "heartbeat")
            except (ValueError, TypeError) as exc:
                # One unusable field must not cost every other backend its
                # sweep, so the failure stays local to this file.
                LOG.warning("ignoring %s: %s", path, exc)
                continue
            if not job_id:
                continue
            known = self.backends.get(job_id)
            if known is not None and known.pool != pool:
                # Job ids are unique per scheduler, so this is a registration
                # copied or misdirected into a second pool's directory. The
                # pool that has the backend keeps it: flapping between two
                # would hand its lifecycle to each pool's supervisor in turn.
                if (job_id, pool) not in self._collisions:
                    self._collisions.add((job_id, pool))
                    LOG.warning(
                        "ignoring %s: job %s is already registered in pool %s",
                        path,
                        job_id,
                        known.pool,
                    )
                continue
            if record.get("manual") and job_id in self.backends:
                # Nothing writes heartbeats for a backend registered by hand,
                # so the staleness rule below would retire it the moment it
                # first went unhealthy -- and then never re-add it, because
                # the re-add path also refuses a stale record. The probe is
                # already the authority on whether a backend can serve; for
                # these it is the only authority.
                pass
            elif now - heartbeat > self.args.stale_after:
                # A stale heartbeat is a hint, not a verdict. The file lives on
                # a shared filesystem that every serving job writes to, so one
                # stall there makes every backend look stale in the same sweep
                # -- and that is exactly what happened under load: all four went
                # `gone: no heartbeat for 30s` together and all four came back
                # five seconds later, having answered /health throughout. The
                # gateway served 503 to 43 requests for a fleet that was fine.
                #
                # The probe is the authority on whether a backend can serve, so
                # a backend that is still passing it keeps its place. One that
                # is not gets retired as before, and a backend whose file is
                # gone entirely never reaches this branch.
                existing = self.backends.get(job_id)
                if existing is None or not existing.healthy:
                    continue
                LOG.debug(
                    "%s heartbeat is %ds stale but it is still passing probes; keeping",
                    job_id,
                    int(now - heartbeat),
                )
            seen.add(job_id)
            if job_id in self.backends:
                try:
                    self.backends[job_id].refresh(record)
                except (ValueError, TypeError, AttributeError) as exc:
                    # Same exposure as construction: keep a bad rewrite local to
                    # its own file and hold the last good view of the backend.
                    LOG.warning("ignoring refresh from %s: %s", path, exc)
            else:
                try:
                    self.backends[job_id] = Backend(record, pool)
                # AttributeError covers a non-string url, which reaches
                # .rstrip() in __init__. One unusable record must cost only
                # itself, never the rest of the sweep.
                except (KeyError, ValueError, TypeError, AttributeError) as exc:
                    LOG.warning("ignoring %s: %s", path, exc)
                    continue
                self.inflight.setdefault(job_id, 0)
                # It came back on its own -- SLURM requeued it, or an operator
                # brought it up. Either way nothing needs recovering.
                self.state_for(pool).lost.pop(job_id, None)
                LOG.info(
                    "%sbackend appeared: %s at %s (ends %s)",
                    self.tag_for(pool),
                    job_id,
                    self.backends[job_id].url,
                    fmt_time(self.backends[job_id].end_time),
                )

        for job_id in [j for j in self.backends if j not in seen]:
            backend = self.backends[job_id]
            # Each pool's own records: recovering this one is its pool's
            # business, under that pool's fleet config.
            state = self.state_for(backend.pool)
            LOG.info(
                "%sbackend gone: %s (no heartbeat for %ds)",
                self.tag_for(backend.pool),
                job_id,
                self.args.stale_after,
            )
            # Remember it before dropping it. Every way a node is preempted
            # ends here -- SIGTERM and "allocation gone" both delete the
            # registration through clear_fleet, and SIGKILL leaves one that
            # goes stale -- so this is the single place that sees all three,
            # and the only place still holding the run_dir.
            state.lost.setdefault(job_id, (backend.run_dir, now))
            self.backends.pop(job_id, None)
            state.draining.pop(job_id, None)
            state.superseded.discard(job_id)
            # Keep the counter while anything is still streaming off this
            # backend; a later sweep collects it once the count reaches zero.
            if not self.inflight.get(job_id):
                self.inflight.pop(job_id, None)

        # Discovery, probing and supervision run on independent timers, so
        # retiring a backend has to clear the pointer to it here rather than
        # waiting for the next election. Otherwise `active` names a job that is
        # no longer in the table, and everything that dereferences it raises.
        if self.active is not None and self.active not in self.backends:
            LOG.warning("active backend %s retired; serving 503", self.active)
            self.active = None
        for pool in self.pools.values():
            if pool.active is not None and pool.active not in self.backends:
                LOG.warning("[%s] active backend %s retired; serving 503", pool.name, pool.active)
                pool.active = None

    # -- routable sets ----------------------------------------------------
    def serving(self):
        """Every backend that can still answer, including ones being drained.

        A conversation pinned to a draining backend keeps going there. Moving
        it would throw away the KV cache the pin exists to preserve, and the
        drain already guarantees the backend outlives the requests on it.
        """
        return {j for j, b in self.backends.items() if b.healthy}

    @property
    def mirrors(self):
        """Delegated to the router so that --router-state carries it.

        The mirror table is gateway state that a successor has to inherit, and
        the router's state file is exactly the bag of such things; keeping a
        second copy on the Fleet would mean a second thing to remember to
        persist, which is how this one came to be dropped by a handover in the
        first place.
        """
        return self.router.mirrors

    def accepting(self):
        """Backends eligible for a NEW conversation, with their load.

        Returns job_id -> (conversations, inflight) so the router can order
        them without reaching back into the fleet.

        Excluded: anything draining, superseded, or too close to its own end
        time to see a conversation through. That last one is what replaces the
        single-active election -- instead of one backend holding all traffic
        until it is replaced, an ageing backend simply stops being offered new
        work and empties out on its own.

        Every pool's backends alike: a new conversation may go to any pool
        (Router.route keeps an existing one inside its pool). Only the drain
        is read per pool, because each pool's supervisor keeps its own.
        """
        now = time.time()
        horizon = self.args.new_conversation_margin
        eligible = {}
        for job_id, backend in self.backends.items():
            if not backend.healthy:
                continue
            # `superseded` is deliberately not consulted. It means "a
            # longer-lived backend exists", which is a statement about which
            # allocation to reclaim first -- not about whether this one can
            # take work. Excluding it here collapsed a fleet of N healthy
            # instances down to one accepting instance the moment a successor
            # was elected, because every other instance is superseded by
            # definition. Only `draining`, which means "this one is going
            # away", keeps new conversations out.
            if job_id in self.state_for(backend.pool).draining:
                continue
            if job_id in self.router.paused:
                continue
            remaining = backend.end_time - now if backend.end_time else 0.0
            if backend.end_time and remaining < horizon:
                continue
            eligible[job_id] = (0, self.inflight.get(job_id, 0), remaining)
        # Every backend is ageing out at once: rather than serve 503, offer the
        # longest-lived one. A conversation started here may be cut short, which
        # is strictly better than refusing to start it.
        if not eligible:
            # A hand-set pause is an instruction, so it survives this fallback;
            # ageing out is not, so it does not.
            healthy = [
                j for j, b in self.backends.items() if b.healthy and j not in self.router.paused
            ]
            if healthy:
                best = max(healthy, key=lambda j: self.backends[j].end_time)
                eligible[best] = (0, self.inflight.get(best, 0), self.backends[best].end_time - now)
        return eligible

    # -- election ---------------------------------------------------------
    def elect(self):
        """Run the election in every pool, each over its own backends only.

        The election is lifecycle state, not routing: it decides which job
        supersedes which, and so what a pool's supervisor drains and quits.
        One election across pools would let a K3 job that outlives every GLM
        job become GLM's `active` -- and mark GLM's own active superseded,
        which ends in a drain and a `quit`.
        """
        if not getattr(self, "pools", None):
            return self._elect_one()
        for name in self.pool_names():
            self.view(name)._elect_one()

    def _elect_one(self):
        """Pick the healthy backend that will live the longest.

        Choosing by end time is what makes relay work without anybody
        orchestrating it: a freshly started job outlives the one it replaces,
        so the moment it passes /health it wins the election on its own.
        """
        tag = pool_tag(self)
        candidates = [j for j, b in self.backends.items() if b.healthy]
        if self.pending and self.pending[0] in candidates:
            LOG.info("%ssuccessor %s is healthy", tag, self.pending[0])
            self.pending = None
        winner = max(candidates, key=lambda j: self.backends[j].end_time) if candidates else None
        if winner == self.active:
            return
        previous = self.active
        self.active = winner
        if winner is None:
            LOG.warning("%sno healthy backend; serving 503", tag)
        else:
            self.ever_active = True
            LOG.info("%sactive backend -> %s (%s)", tag, winner, self.backends[winner].url)
            # Won the election back: whatever replaced it is gone or sicker, so
            # it is no longer a candidate for reclaim.
            self.superseded.discard(winner)
            if winner in self.draining:
                self.draining.pop(winner, None)
                LOG.info("%scancelled drain of re-elected backend %s", tag, winner)

        # Only a forward handover marks the predecessor. Falling back to an
        # older job after the active backend fails is reversible: the newer job
        # may merely be restarting and must remain eligible to win back routing.
        #
        # And only within one instance. The election ranks every backend by end
        # time, which across a multi-instance fleet says nothing about
        # replacement: i02 submitted an hour after i00 outlives it and would
        # "supersede" it here, though the two serve side by side and neither is
        # replacing anything. Acting on that drained every instance but the
        # longest-lived one, which is why relay had to be switched off
        # fleet-wide. Comparing labels costs nothing in the single-instance
        # case -- consecutive jobs of one instance share a label, so the
        # condition is simply true.
        if winner is not None and previous and previous in self.backends:
            same_instance = instance_label(self.backends[winner].run_dir) == instance_label(
                self.backends[previous].run_dir
            )
            if not same_instance:
                LOG.info(
                    "%s%s outlives %s but is a different instance; neither supersedes the other",
                    tag,
                    winner,
                    previous,
                )
            elif self.backends[winner].end_time > self.backends[previous].end_time:
                self.superseded.add(previous)
                LOG.info("%ssuperseded %s; reclaim held until %s is stable", tag, previous, winner)
            else:
                LOG.warning(
                    "%sfailed back from %s to older backend %s; keeping %s available for recovery",
                    tag,
                    previous,
                    winner,
                    previous,
                )


class PoolView:
    """One pool of the fleet, shaped like the Fleet itself.

    The lifecycle code -- election, revive, recovery, relay, drain, reclaim and
    the start/stop endpoints -- was written against a Fleet, and every line of
    it is right for a single pool. Rather than teach each function about
    pools, each is handed one of these: `backends` holds only this pool's
    members, `args` names this pool's fleet config, and the per-pool state
    lives on the pool. For the default pool that state is the Fleet's own
    attributes, so its supervision reads and writes exactly what it always did.

    `inflight` and `revived` stay shared: both are keyed by job id, and no job
    id is in two pools (discovery refuses the second).
    """

    STATE = (
        "active",
        "pending",
        "ever_active",
        "stopped",
        "last_submit",
        "seen_running",
        "superseded",
        "draining",
        "relaying",
        "lost",
        "recovered",
        "last_recovery",
    )

    def __init__(self, fleet, name):
        pool = fleet.pools.get(name)
        args = fleet.args
        if pool is not None:
            # A copy, so the pool's fleet config is what run_fleetctl passes
            # and the default pool's stays the default pool's.
            args = argparse.Namespace(**vars(fleet.args))
            args.fleet_dir = pool.fleet_dir
            args.fleet_config = pool.fleet_config
            args.yaml = ""
        set_ = object.__setattr__
        set_(self, "fleet", fleet)
        set_(self, "name", name)
        set_(self, "args", args)
        set_(self, "state", pool if pool is not None else fleet)
        # The single-lineage relay resubmits --yaml, one serve.sh deployment;
        # it is the default pool's alone. An extra pool rolls per instance.
        set_(self, "single_lineage", pool is None)
        set_(self, "pool_tag", fleet.tag_for(name))

    @property
    def backends(self):
        return {j: b for j, b in self.fleet.backends.items() if b.pool == self.name}

    @property
    def inflight(self):
        return self.fleet.inflight

    @property
    def revived(self):
        return self.fleet.revived

    def __getattr__(self, attr):
        if attr in PoolView.STATE:
            return getattr(self.state, attr)
        raise AttributeError("a pool view has no %r" % attr)

    def __setattr__(self, attr, value):
        if attr not in PoolView.STATE:
            raise AttributeError("%r is not per-pool state" % attr)
        setattr(self.state, attr, value)

    # The election, over this pool's backends and this pool's state.
    _elect_one = Fleet._elect_one


def fmt_time(ts):
    if not ts:
        return "unknown"
    return time.strftime("%H:%M:%S", time.localtime(ts))


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------
async def read_head(reader):
    """Read up to and including the blank line ending an HTTP head."""
    buf = b""
    while b"\r\n\r\n" not in buf:
        chunk = await reader.read(8192)
        if not chunk:
            return None, b""
        buf += chunk
        if len(buf) > MAX_HEAD_BYTES:
            raise ValueError("head exceeds %d bytes" % MAX_HEAD_BYTES)
    head, _, rest = buf.partition(b"\r\n\r\n")
    return head, rest


def parse_request_head(head):
    lines = head.decode("latin-1").split("\r\n")
    parts = lines[0].split(" ")
    if len(parts) != 3:
        raise ValueError("malformed request line: %r" % lines[0])
    headers = []
    for line in lines[1:]:
        if not line:
            continue
        name, sep, value = line.partition(":")
        if not sep:
            raise ValueError("malformed header: %r" % line)
        headers.append((name.strip(), value.strip()))
    return parts[0], parts[1], headers


def header_value(headers, name):
    name = name.lower()
    for key, value in headers:
        if key.lower() == name:
            return value
    return None


def extract_key(headers):
    key = header_value(headers, "x-api-key")
    if key:
        return key.strip()
    auth = header_value(headers, "authorization") or ""
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return None


def build_response(status, reason, body, extra_headers=()):
    head = [
        "HTTP/1.1 %d %s" % (status, reason),
        "Content-Type: application/json",
        "Content-Length: %d" % len(body),
        "Connection: close",
    ]
    head.extend(extra_headers)
    return ("\r\n".join(head) + "\r\n\r\n").encode("latin-1") + body


def error_response(status, retry_after=None):
    kind, message = ERROR_BODIES[status]
    body = json.dumps({"type": "error", "error": {"type": kind, "message": message}}).encode()
    extra = ["Retry-After: %d" % retry_after] if retry_after else []
    return build_response(status, ERROR_REASONS[status], body, extra)


def json_response(payload, status=200, reason="OK"):
    body = json.dumps(payload, indent=2).encode()
    return build_response(status, reason, body)


def parse_response_head(head):
    lines = head.decode("latin-1").split("\r\n")
    if not lines or not lines[0].startswith("HTTP/"):
        raise ValueError("malformed upstream status line")
    headers = []
    for line in lines[1:]:
        if not line:
            continue
        name, sep, value = line.partition(":")
        if not sep:
            raise ValueError("malformed upstream header: %r" % line)
        headers.append((name.strip(), value.strip()))
    return lines[0], headers


def rewrite_sse_head(status_line, headers):
    """Make downstream SSE framing independent from the upstream framing."""
    owned = {"connection", "content-length", "keep-alive", "trailer", "transfer-encoding"}
    lines = [status_line]
    lines.extend("%s: %s" % (name, value) for name, value in headers if name.lower() not in owned)
    lines.extend(("Transfer-Encoding: chunked", "Connection: close"))
    return ("\r\n".join(lines) + "\r\n\r\n").encode("latin-1")


def chunk_frame(payload):
    return b"%x\r\n%s\r\n" % (len(payload), payload)


# How each dialect says "that is the whole response".
#
# Anthropic closes with `message_stop`. Chat Completions closes with a `[DONE]`
# data line and no event name, handled separately below. The Responses API
# closes with `response.completed` and never sends `[DONE]`, so a Responses
# stream that finished cleanly used to fall through every check here and be
# reported as truncated -- the gateway then injected an error into a response
# that was complete, and a client reading that error gives up on the endpoint.
_TERMINAL_EVENTS = frozenset(
    (
        b"message_stop",
        b"response.completed",
    )
)

# Terminal too, but terminal as a failure: the response is over and did not
# succeed. Distinguished from the set above so the gateway does not report a
# server-side refusal as a healthy completion.
_FAILURE_EVENTS = frozenset(
    (
        b"error",
        b"response.failed",
        b"response.incomplete",
    )
)


class SseTracker:
    """Recognize a terminal event across arbitrary transport reads.

    Two stream dialects reach this gateway. Anthropic ends a stream with a
    named `event: message_stop`; OpenAI's chat-completions stream has no event
    names at all and ends with a bare `data: [DONE]` sentinel. Recognising only
    the Anthropic form makes every OpenAI stream look truncated, and the caller
    then appends an Anthropic-shaped error event to a stream that had in fact
    completed normally -- which is worse than the missing terminal it was meant
    to repair, because it corrupts a good response.
    """

    def __init__(self):
        self.buffer = bytearray()
        self.current_event = None
        self.saw_stop = False
        self.saw_error = False
        # Progress counters. A truncated stream is only diagnosable if the log
        # says how far it got: "died before the first event" and "died after
        # 400 events" have entirely different causes, and the exception text
        # alone cannot tell them apart. These are counted on the way past, so
        # they cost one add per line already being parsed.
        self.bytes = 0
        self.events = 0
        self.last_event = None
        self.last_data = b""

    @property
    def terminal(self):
        return self.saw_stop or self.saw_error

    def progress(self):
        return "bytes=%d events=%d last_event=%s" % (
            self.bytes,
            self.events,
            self.last_event.decode("latin-1", "replace") if self.last_event else "-",
        )

    def preview(self):
        """Last data line, truncated. Response content, so debug level only."""
        return self.last_data.decode("latin-1", "replace")

    def feed(self, payload):
        self.bytes += len(payload)
        self.buffer.extend(payload)
        while True:
            newline = self.buffer.find(b"\n")
            if newline < 0:
                return
            line = bytes(self.buffer[:newline]).rstrip(b"\r")
            del self.buffer[: newline + 1]
            if not line:
                if self.current_event in _TERMINAL_EVENTS:
                    self.saw_stop = True
                elif self.current_event in _FAILURE_EVENTS:
                    self.saw_error = True
                self.current_event = None
                continue
            name, sep, value = line.partition(b":")
            if not sep:
                continue
            if name == b"event":
                self.current_event = value.lstrip(b" ")
                self.last_event = self.current_event
            elif name == b"data":
                # Both dialects carry exactly one data line per event, so this
                # counts events without needing a separate state machine.
                self.events += 1
                self.last_data = value.strip()[:120]
                if value.strip() == b"[DONE]":
                    # OpenAI's terminal sentinel. It carries no event name, so
                    # it has to be matched on the data line itself.
                    self.saw_stop = True


class BufferedUpstream:
    """Expose bytes already read with the response head before the socket."""

    def __init__(self, reader, initial):
        self.reader = reader
        self.buffer = bytearray(initial)

    async def read(self, size):
        if self.buffer:
            data = bytes(self.buffer[:size])
            del self.buffer[:size]
            return data
        return await read_upstream(self.reader, size)

    async def read_exact(self, size):
        parts = []
        remaining = size
        while remaining:
            data = await self.read(remaining)
            if not data:
                return b"".join(parts), False
            parts.append(data)
            remaining -= len(data)
        return b"".join(parts), True

    async def read_line(self):
        while True:
            newline = self.buffer.find(b"\r\n")
            if newline >= 0:
                line = bytes(self.buffer[:newline])
                del self.buffer[: newline + 2]
                return line
            if len(self.buffer) > MAX_HEAD_BYTES:
                raise ValueError("upstream framing line is too long")
            data = await read_upstream(self.reader, 8192)
            if not data:
                return None
            self.buffer.extend(data)


async def emit_sse_payload(writer, tracker, payload):
    if not payload:
        return
    tracker.feed(payload)
    await write_client(writer, chunk_frame(payload))


async def relay_chunked_sse(source, writer, tracker):
    while True:
        line = await source.read_line()
        if line is None:
            return False
        try:
            size = int(line.split(b";", 1)[0].strip(), 16)
        except ValueError:
            LOG.warning("invalid upstream chunk size: %r", line)
            return False
        if size < 0:
            return False
        if size == 0:
            # Consume trailers, but do not forward them: rewrite_sse_head removes
            # Trailer and the gateway owns the downstream terminal chunk.
            while True:
                trailer = await source.read_line()
                if trailer is None:
                    return False
                if not trailer:
                    return True

        remaining = size
        while remaining:
            payload = await source.read(min(RELAY_CHUNK, remaining))
            if not payload:
                return False
            remaining -= len(payload)
            await emit_sse_payload(writer, tracker, payload)
        ending, complete = await source.read_exact(2)
        if not complete or ending != b"\r\n":
            return False


async def relay_sized_sse(source, writer, tracker, size):
    remaining = size
    while remaining:
        payload = await source.read(min(RELAY_CHUNK, remaining))
        if not payload:
            return False
        remaining -= len(payload)
        await emit_sse_payload(writer, tracker, payload)
    return True


async def relay_close_delimited_sse(source, writer, tracker):
    while True:
        payload = await source.read(RELAY_CHUNK)
        if not payload:
            return True
        await emit_sse_payload(writer, tracker, payload)


# ---------------------------------------------------------------------------
# Request path
# ---------------------------------------------------------------------------

# --------------------------------------------------------------------------
# conversation routing
# --------------------------------------------------------------------------

# Checked in order. A header wins because it is the only source a client can
# set deliberately; everything below is inferred from what clients already send.
_CONVERSATION_HEADERS = ("x-conversation-id", "x-session-id", "session-id", "thread-id")

# Where each API surface keeps its conversation identity. Measured against
# 2501 captured requests rather than guessed:
#   /v1/responses         prompt_cache_key, on 58% of them, one value per
#                         session and about 120 requests per value.
#   /v1/chat/completions  client_metadata.session_id -- nested, which is why a
#                         scan of top-level body keys finds nothing and makes
#                         this surface look unroutable.
# The order requests are asked for their identity. Each entry is one of
#   header:<name>       a request header
#   body:<a.b.c>        a body field, dotted for nesting
#   prefix              hash the opening of the conversation
# Configurable because a client that appears later should not need a code
# change to be routed, and because the right order is a property of the fleet's
# traffic rather than of the gateway.
DEFAULT_KEY_SOURCES = (
    "header:x-conversation-id",
    "header:x-session-id",
    "header:session-id",
    "header:thread-id",
    "body:prompt_cache_key",
    "body:previous_response_id",
    "body:conversation_id",
    "body:client_metadata.session_id",
    "body:client_metadata.thread_id",
    "body:metadata.session_id",
    "body:metadata.conversation_id",
    "prefix",
)

POLICIES = ("least_conversations", "least_inflight", "round_robin", "longest_lived")


class PolicyDir:
    """Custom routing policies, loaded from .py files in a directory.

    A directory rather than an HTTP endpoint, deliberately. A policy is code,
    and this gateway authenticates with a username that its own users file
    describes as guessable -- so taking code over HTTP would make "knows a
    colleague's name" enough to run anything on this node. Writing a file to
    the shared filesystem is a far higher bar, and it is the same bar the
    fleet directory already sets for registering a backend.

    A file defines `select(accepting)` and is named by the policy it provides:
    `prefill_aware.py` supplies `prefill_aware`. `accepting` maps job id to
    `(conversations, inflight, remaining_seconds)`; return one of its keys.

    Custom policies run on the request path, so failure is contained rather
    than trusted away: anything that raises, returns a job that is not on
    offer, or takes the process down a path the built-ins would not is counted
    against the policy, and after `strikes` it is dropped and placement falls
    back to least_conversations. Editing the file clears the count -- the
    intended way to fix a policy is to fix it, not to restart the gateway.
    """

    strikes = 3

    def __init__(self, path, strikes=None):
        self.path = path
        self.policies = {}
        self.mtimes = {}
        self.failures = collections.Counter()
        self.disabled = set()
        if strikes is not None:
            self.strikes = strikes

    def names(self):
        return tuple(sorted(n for n in self.policies if n not in self.disabled))

    def reload(self):
        """Pick up new, changed and deleted policy files. Never raises."""
        if not self.path:
            return
        try:
            paths = sorted(glob.glob(os.path.join(self.path, "*.py")))
        except OSError as exc:
            LOG.warning("policy dir unreadable: %s", exc)
            return
        seen = set()
        for path in paths:
            name = os.path.splitext(os.path.basename(path))[0]
            if name.startswith("_"):
                continue
            seen.add(name)
            try:
                mtime = os.path.getmtime(path)
            except OSError:
                continue
            if self.mtimes.get(name) == mtime:
                continue
            self.mtimes[name] = mtime
            # An edit is a retraction of the last verdict on this policy.
            self.failures.pop(name, None)
            self.disabled.discard(name)
            fn = self._load(path, name)
            if fn is None:
                self.policies.pop(name, None)
                continue
            self.policies[name] = fn
            LOG.info("routing policy loaded: %s (%s)", name, path)
        for name in [n for n in self.policies if n not in seen]:
            self.policies.pop(name, None)
            self.mtimes.pop(name, None)
            self.failures.pop(name, None)
            self.disabled.discard(name)
            LOG.info("routing policy withdrawn: %s", name)

    def _load(self, path, name):
        """Import one file. A broken policy costs itself and nothing else."""
        if name in POLICIES:
            LOG.warning("policy %s shadows a built-in; ignoring %s", name, path)
            return None
        try:
            spec = importlib.util.spec_from_file_location("kfpolicy_" + name, path)
            if spec is None or spec.loader is None:
                raise ImportError("no loader for %s" % path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        except Exception as exc:  # noqa: BLE001 - arbitrary user code
            LOG.warning("policy %s failed to load: %s: %s", name, type(exc).__name__, exc)
            return None
        fn = getattr(module, "select", None)
        if not callable(fn):
            LOG.warning("policy %s defines no select(accepting); ignoring", name)
            return None
        return fn

    def run(self, name, accepting):
        """Ask a custom policy to place a conversation, or return None.

        None means "the caller should use the built-in default" and is the
        answer for every way this can go wrong, so the request path has one
        branch instead of a taxonomy of failures.
        """
        fn = self.policies.get(name)
        if fn is None or name in self.disabled:
            return None
        try:
            picked = fn(dict(accepting))
        except Exception as exc:  # noqa: BLE001 - arbitrary user code
            return self._strike(name, "%s: %s" % (type(exc).__name__, exc))
        if picked not in accepting:
            return self._strike(name, "returned %r, which is not on offer" % (picked,))
        return picked

    def _strike(self, name, why):
        self.failures[name] += 1
        count = self.failures[name]
        if count >= self.strikes:
            self.disabled.add(name)
            LOG.error(
                "routing policy %s disabled after %d failures (%s); "
                "placement falls back to least_conversations until the "
                "file is edited",
                name,
                count,
                why,
            )
        else:
            LOG.warning("routing policy %s failed (%d/%d): %s", name, count, self.strikes, why)
        return None


SAFE_JOB_ID = re.compile(r"\A[A-Za-z0-9][A-Za-z0-9._-]{0,63}\Z")


def registration_path(fleet_dir, job_id):
    """Where a backend's registration file goes, or None if the id is unsafe.

    The id becomes a filename, so it decides where this writes. `../../x` or an
    absolute path would put a file wherever the gateway can reach, and this is
    reachable by anyone the users file admits -- which that file says is anyone
    who can guess a colleague's name. The check is a whitelist because a
    blacklist of traversal spellings is a game nobody wins.
    """
    if not isinstance(job_id, str) or not SAFE_JOB_ID.match(job_id):
        return None
    path = os.path.join(fleet_dir, job_id + ".json")
    # Belt and braces: even a whitelist-passing id must not escape the dir.
    if os.path.dirname(os.path.abspath(path)) != os.path.abspath(fleet_dir):
        return None
    return path


def known_policies(router):
    """Built-ins plus whatever the policy directory currently offers."""
    custom = getattr(router, "policies", None)
    return tuple(POLICIES) + (custom.names() if custom is not None else ())


def conversation_prefix(payload):
    """Hashable bytes for the part of a conversation that does not change.

    A last resort, used only when a client sends no identity of its own. Agent
    turns replay the whole history, so the opening of that history is stable
    for the life of the session -- and it is also exactly the prefix whose KV
    cache the routing exists to reuse.

    Only the first two entries are hashed. Including more would fold in text
    that grows every turn, and the key would then change under the session it
    is supposed to pin.
    """
    parts = []
    instructions = payload.get("instructions")
    if isinstance(instructions, str) and instructions:
        parts.append(instructions)
    for field in ("input", "messages"):
        value = payload.get(field)
        if isinstance(value, str) and value:
            parts.append(value)
            break
        if isinstance(value, list) and value:
            try:
                parts.append(json.dumps(value[:2], sort_keys=True, separators=(",", ":")))
            except (TypeError, ValueError):
                pass
            break
    if not parts:
        return None
    return "\0".join(parts).encode("utf-8", "replace")


def dig(payload, path):
    """Follow a dotted path into a decoded body, returning a string or None."""
    node = payload
    for part in path.split("."):
        if not isinstance(node, dict):
            return None
        node = node.get(part)
    return node if isinstance(node, str) and node else None


def conversation_key(headers, body, sources=DEFAULT_KEY_SOURCES):
    """Identify the conversation a request belongs to, or None.

    Sources are tried in order and the first that answers wins, so the caller
    controls precedence without this function knowing which client is which.
    """
    payload = None
    decoded = False
    for source in sources:
        if source.startswith("header:"):
            value = header_value(headers, source[7:])
            if value and value.strip():
                return "hdr:" + value.strip()
            continue
        if not body:
            continue
        if not decoded:
            decoded = True
            try:
                payload = json.loads(body)
            except (ValueError, UnicodeDecodeError):
                payload = None
            if not isinstance(payload, dict):
                payload = None
        if payload is None:
            continue
        if source.startswith("body:"):
            path = source[5:]
            value = dig(payload, path)
            if value:
                return "%s:%s" % (path, value)
        elif source == "prefix":
            prefix = conversation_prefix(payload)
            if prefix:
                return "prefix:" + hashlib.sha256(prefix).hexdigest()[:32]
    return None


def short_convo(key):
    """Shorten a conversation key for a log line without losing what it names.

    Truncating the head is worse than useless here: the source prefix is the
    fixed part and the identifier is the tail, so `key[:28]` renders every
    conversation from one client identically -- which is exactly the case the
    log line exists to tell apart. Keep both ends.
    """
    if key is None:
        return "-"
    if len(key) <= 44:
        return key
    return key[:20] + ".." + key[-20:]


class Router:
    """Pins a conversation to a backend and keeps it there.

    Affinity is not about correctness -- any healthy backend can answer any
    request. It is about the prefix KV cache: an agent turn resends the whole
    conversation, so a turn that lands where the previous one landed reuses
    that cache and one that does not pays to rebuild it.

    So a broken pin is a slowdown, never a wrong answer. That is what lets
    every failure here resolve by re-pinning rather than by refusing to serve.

    The exception is a gateway fronting several models, one per pool. A
    conversation that moves to another pool's backend continues on a
    different model: that model reads the first one's output as its own
    history and copies its habits, and the recorded trajectory mixes two
    models. So each conversation also remembers the pool it was first placed
    in (`families`), longer than its pin. With `pool_affinity` it is only
    ever re-placed inside that pool; while no backend of it is accepting, the
    request gets no backend rather than another model's.
    """

    # How long a conversation's pool is remembered after its last request.
    # Longer than the pin: a pin lapses after a quiet spell and is simply
    # rebuilt, but the conversation must come back to the same model even
    # after an agent's longest wait. Campaigns run for at most four hours.
    FAMILY_TTL = 6 * 3600.0

    def __init__(
        self,
        ttl,
        capacity,
        policy="least_conversations",
        state_path=None,
        key_sources=DEFAULT_KEY_SOURCES,
        pool_affinity=True,
    ):
        self.ttl = ttl
        self.capacity = capacity
        self.policy = policy
        self.key_sources = list(key_sources)
        self.state_path = state_path
        self.pool_affinity = pool_affinity
        self.pins = collections.OrderedDict()  # key -> (job_id, last_seen)
        # key -> (pool, last_seen): the pool a conversation was first placed
        # in. Kept through unpin, which is the point: a pin is dropped when a
        # client gives up on a backend, and the retry must still find the
        # same model.
        self.families = collections.OrderedDict()
        # Placements refused because no backend of the conversation's pool
        # was accepting, and when that was last logged, per pool.
        self.family_waits = 0
        self._family_wait_logged = {}
        # Set by hand and authoritative: never expired, never displaced by a
        # placement decision. Only a backend that has actually gone away can
        # override one, because the alternative is refusing to serve.
        self.manual = {}  # key -> job_id
        # Which backends are being copied elsewhere, job id -> [(host, port)].
        # Kept here, with the pins, because this file is the set of things a
        # successor has to inherit -- and a handover that silently ends a
        # running mirror experiment is a handover that did not keep its
        # promise. Found the hard way: the second live handover dropped a
        # mirror that had been running for an hour, with no message anywhere.
        self.mirrors = {}
        self.paused = set()  # job ids held out of new placement by hand
        self.hits = 0
        self.misses = 0
        self.rehomed = 0
        self.dirty = False
        self._rr = 0
        # Conversations per backend, maintained incrementally. Rebuilding it by
        # walking `pins` on every request made placement cost O(pins) rather
        # than O(backends), and `pins` is bounded by --sticky-capacity, not by
        # the fleet size: at the 20000 default that was 2.2 ms of the 3.7 ms a
        # request spent routing. Every mutation of `pins` must go through
        # _set_pin/_drop_pin so this stays in step.
        self._tally = collections.Counter()
        self._last_expire = 0.0
        # Writes are serialised and ordered, not just made atomic one at a
        # time. There are two writers inside one process now -- the periodic
        # save and flush_now -- and os.replace makes the loser of a race
        # invisible rather than obviously wrong. See write_snapshot.
        self._write_lock = threading.Lock()
        self._snapshots = itertools.count(1)
        self._written_seq = 0

    # -- persistence ------------------------------------------------------
    def load(self):
        """Restore pins written by a previous process.

        Without this, restarting the gateway silently discards every pin, and
        each conversation in flight rebuilds a prefix cache it already had. On
        a fleet whose gateway lives inside a four-hour scheduler job, that is
        not a rare event.
        """
        if not self.state_path or not os.path.exists(self.state_path):
            return
        state = self._read_state()
        if state is None:
            return
        if not isinstance(state, dict):
            LOG.warning("ignoring router state %s: not an object", self.state_path)
            return
        now = time.time()
        restored = 0
        for key, value in (state.get("pins") or {}).items():
            try:
                job_id, last_seen = str(value[0]), float(value[1])
            except (TypeError, ValueError, IndexError):
                continue
            if now - last_seen <= self.ttl:
                self.pins[key] = (job_id, last_seen)
                restored += 1
        # Absent from a file written before conversations remembered their
        # pool, which is every predecessor in the first handover after the
        # upgrade; pinned conversations then learn their pool on their next
        # request (route()).
        for key, value in (state.get("families") or {}).items():
            try:
                pool, last_seen = str(value[0]), float(value[1])
            except (TypeError, ValueError, IndexError):
                continue
            if now - last_seen <= self.FAMILY_TTL:
                self.families[key] = (pool, last_seen)
        self.manual = {
            str(k): str(v) for k, v in (state.get("manual") or {}).items() if isinstance(v, str)
        }
        self.paused = {str(j) for j in (state.get("paused") or [])}
        restored_mirrors = {}
        for job, entry in (state.get("mirrors") or {}).items():
            # Two shapes on disk. A file written before mirrors went
            # multi-target holds ["host", port]; one written after holds
            # [["host", port], ...]. The predecessor in a live handover is by
            # definition running the older code, so the old shape is not a
            # legacy to migrate once -- it arrives at every upgrade.
            if isinstance(entry, (list, tuple)) and entry and isinstance(entry[0], (list, tuple)):
                candidates = entry
            else:
                candidates = [entry]
            targets = []
            for target in candidates:
                # A malformed entry drops that one target rather than the whole
                # restore: everything else in this file is still worth having.
                try:
                    host, port = str(target[0]), int(target[1])
                except (TypeError, ValueError, IndexError):
                    LOG.warning("ignoring unusable mirror entry for %s: %r", job, target)
                    continue
                if host and 0 < port < 65536 and (host, port) not in targets:
                    targets.append((host, port))
            if targets:
                restored_mirrors[str(job)] = targets
        self.mirrors = restored_mirrors
        if state.get("policy") in known_policies(self):
            self.policy = state["policy"]
        sources = state.get("key_sources")
        if isinstance(sources, list) and sources:
            self.key_sources = [str(x) for x in sources]
        # load() is the one path that fills `pins` without going through
        # _set_pin, and it runs after __init__, so the tally has to be rebuilt
        # here. Skipping this leaves every count at zero while the table is
        # full, and least_conversations then reads the whole fleet as idle and
        # piles every new conversation onto whichever job id sorts first.
        self._rebuild_tally()
        LOG.info(
            "router state restored: %d pins, %d manual, %d paused, policy=%s",
            restored,
            len(self.manual),
            len(self.paused),
            self.policy,
        )
        if self.families:
            LOG.info("router state restored: %d conversations' pools", len(self.families))

    def _read_state(self):
        """Decode the state file, or None if it cannot be read right now.

        The bytes are read in one go and parsed afterwards, rather than parsed
        off the handle, so what gets decoded is one consistent set of bytes
        however the file behaves while we are looking at it.

        Read twice before giving up, because the process most likely to call
        this is a handover successor loading while its predecessor is replacing
        this very path. os.replace is atomic, so the content is never half a
        table -- but on the shared filesystem the fleet dir lives on, a client
        holding a cached dentry for the inode that just got replaced gets
        ESTALE instead, which is indistinguishable here from a corrupt file.
        One collision is bad luck; two in a row is a real problem, and the
        caller degrades to an empty table either way rather than failing to
        start. No sleep between the attempts: this runs before the event loop
        does, and a startup stall is the one thing a handover cannot afford.
        """
        failure = None
        for _ in range(2):
            try:
                with open(self.state_path) as handle:
                    text = handle.read()
                if not text or text.isspace():
                    # Never produced by write_snapshot, which only ever renames
                    # a finished file into place -- so an empty one means
                    # something else truncated it, and restoring nothing is the
                    # right reading of it.
                    raise ValueError("file is empty")
                return json.loads(text)
            except (OSError, ValueError) as exc:
                failure = exc
        LOG.warning("ignoring unreadable router state %s: %s", self.state_path, failure)
        return None

    def snapshot(self, force=False):
        """Copy out the state to persist, or None if there is nothing to write.

        Every container here is copied rather than referenced, because the
        write runs on a thread and json.dump would otherwise iterate a table
        the request path is still inserting into. Cheap enough to stay on the
        event loop: at the 20000-pin default it is one dict comprehension.

        `force` is for flush_now. The question a handover asks is "is the file
        current", not "is it newer than the last one written", and those differ
        exactly when the table has not changed since the last periodic save --
        which is also when `dirty` would skip the write and leave the successor
        reading whatever happens to be on disk.
        """
        if not self.state_path or not (self.dirty or force):
            return None
        # Cleared at snapshot time, not after the write: this snapshot is what
        # the write will persist, so a mutation arriving while it is in flight
        # belongs to the next flush and has to re-dirty the table itself.
        self.dirty = False
        state = {
            # Popped by write_snapshot before the dump. It orders two writes
            # against each other inside this process and would mean nothing to
            # anyone reading the file back, so it does not belong on disk.
            "_seq": next(self._snapshots),
            "version": 1,
            "saved_at": time.time(),
            # Always the multi-target shape, [[host, port], ...]. load()
            # accepts both, so a successor on either version reads this.
            "mirrors": {job: [list(t) for t in targets] for job, targets in self.mirrors.items()},
            "policy": self.policy,
            "key_sources": list(self.key_sources),
            "manual": dict(self.manual),
            "paused": sorted(self.paused),
            "pins": {k: [j, t] for k, (j, t) in self.pins.items()},
        }
        if self.families:
            # Ignored by a predecessor's code, which reads only the keys it
            # knows. Left out when empty, as on any single-pool gateway.
            state["families"] = {k: [p, t] for k, (p, t) in self.families.items()}
        return state

    def _fsync_directory(self):
        """Make the rename durable as well as the bytes. Best effort.

        os.replace is atomic for a reader the instant it returns, but the
        directory entry it created only survives the node going down once the
        directory itself is synced. Best effort on purpose: the successor this
        is for reads through the page cache on the same node either way, and
        some shared filesystems refuse an fsync on a directory handle -- taking
        the flush down over that would cost the pins it exists to protect.
        """
        parent = os.path.dirname(self.state_path) or "."
        try:
            handle = os.open(parent, os.O_RDONLY)
        except OSError as exc:
            LOG.debug("could not open %s to fsync it: %s", parent, exc)
            return
        try:
            os.fsync(handle)
        except OSError as exc:
            LOG.debug("could not fsync %s: %s", parent, exc)
        finally:
            os.close(handle)

    def write_snapshot(self, state, fsync=False):
        """Write a snapshot out. Blocking, and safe to run on a thread.

        `fsync` is for handover step 2, where the successor is spawned as soon
        as this returns and so "written" has to mean on the disk rather than in
        this process's buffers. The periodic save leaves it off: an fsync per
        flush onto the shared filesystem costs more than what it protects,
        which is at most one interval of pins and a few re-homed conversations.
        """
        if not state:
            return
        seq = state.pop("_seq", 0)
        # One temp file per snapshot, not one per process. The pid was unique
        # enough while router_state_loop was the only writer; flush_now is a
        # second one in the same process, and two threads sharing a temp path
        # interleave their json.dump into it -- after which os.replace installs
        # the tear, atomically. That is the bug we already had to take back out
        # of the fleet-sync tool, for exactly this reason.
        tmp = "%s.tmp.%d.%d" % (self.state_path, os.getpid(), seq)
        try:
            # Serialised, and ordered. Exclusion alone is not enough: a
            # periodic write whose snapshot predates flush_now's must not be
            # allowed to land after it, or the successor boots from precisely
            # the stale table flush_now was called to get rid of.
            with self._write_lock:
                if seq and seq < self._written_seq:
                    return
                with open(tmp, "w") as handle:
                    json.dump(state, handle)
                    if fsync:
                        handle.flush()
                        os.fsync(handle.fileno())
                os.replace(tmp, self.state_path)
                self._written_seq = max(self._written_seq, seq)
                if fsync:
                    self._fsync_directory()
        except OSError as exc:
            # Losing the snapshot costs cache warmth on the next restart and
            # nothing else, so it must never take the gateway down with it.
            # Put the flag back, or an unlucky write leaves the on-disk copy
            # behind for as long as nothing else happens to dirty the table.
            LOG.warning("could not save router state: %s", exc)
            self.dirty = True
            if os.path.exists(tmp):
                try:
                    os.unlink(tmp)
                except OSError:
                    pass

    def save(self):
        self.write_snapshot(self.snapshot())

    async def flush_now(self):
        """Persist the pin table right now, changed or not, and wait for it.

        Handover step 2, run before the successor is spawned. The successor's
        entire claim to cache affinity is that it boots from a current table,
        and the periodic save is up to thirty seconds behind -- which on an
        agent fleet is every conversation opened in that window, each of them
        arriving at the successor with a cache it will have to rebuild.
        Returns True if something was written.

        Awaitable rather than blocking because the caller is the request path:
        the handover POST is served by the same event loop that is relaying
        every in-flight stream. Snapshot on the loop, write on a thread, same
        split as router_state_loop -- the snapshot has to be taken here or the
        table changes under the thread. That does not buy a free flush: the
        json encode holds the GIL and still costs the loop what the periodic
        save has always cost it (6 ms at the 20000-pin default, measured). What
        it does buy is the half with no upper bound -- the write, the fsync and
        the replace onto a shared filesystem that can stall for seconds --
        happening somewhere other than under the streams being handed over.

        Deliberately not gated on the supervisor lock, unlike the periodic
        save: this runs at step 2, where the caller is still the holder, and a
        flush that quietly did nothing would hand the successor a stale table
        with nothing anywhere to say so.
        """
        state = self.snapshot(force=True)
        if state is None:
            return False
        await in_thread(lambda: self.write_snapshot(state, fsync=True))
        return True

    # -- pin table bookkeeping --------------------------------------------
    # `pins` and `_tally` are one structure in two parts. These three are the
    # only places that write to it.
    def _rebuild_tally(self):
        self._tally = collections.Counter(job_id for job_id, _ in self.pins.values())

    def _set_pin(self, key, job_id, now):
        previous = self.pins.get(key)
        if previous is not None:
            self._tally[previous[0]] -= 1
        self.pins[key] = (job_id, now)
        self.pins.move_to_end(key)
        self._tally[job_id] += 1

    def _drop_pin(self, key):
        dropped = self.pins.pop(key, None)
        if dropped is not None:
            self._tally[dropped[0]] -= 1
        return dropped

    def _expire(self, now):
        """Reclaim pins the TTL has passed, at most once a second.

        This used to run per request, which made every request pay for the
        whole table. It can be amortised because it is only reclaiming memory:
        route() expires the one key it was asked about exactly, so a caller
        never observes a stale pin no matter how long this waits.
        """
        if now - self._last_expire < 1.0:
            return
        self._last_expire = now
        for key in [k for k, (_, seen) in self.pins.items() if now - seen > self.ttl]:
            self._drop_pin(key)
        for key in [k for k, (_, seen) in self.families.items() if now - seen > self.FAMILY_TTL]:
            del self.families[key]

    def _note_family(self, key, pool, now):
        """Remember (or refresh) the pool a conversation is served from."""
        if pool is None:
            return
        previous = self.families.get(key)
        self.families[key] = (pool, now)
        self.families.move_to_end(key)
        if previous is None or previous[0] != pool:
            self.dirty = True
        while len(self.families) > self.capacity:
            self.families.popitem(last=False)

    def _family_of(self, key, now):
        family = self.families.get(key)
        if family is None or now - family[1] > self.FAMILY_TTL:
            return None
        return family[0]

    def _trim(self):
        # Runs after the insert, not before it. Trimming first leaves room for
        # exactly one more and the table settles one over capacity forever.
        while len(self.pins) > self.capacity:
            key, (job_id, _) = self.pins.popitem(last=False)
            self._tally[job_id] -= 1

    def route(self, key, accepting, serving, now=None, pool_of=None):
        """Return the job id to use, pinning the conversation on first sight.

        `accepting` are the backends taking new conversations; `serving` also
        includes those that are draining, because a conversation already pinned
        to one should stay there while it lasts rather than lose its cache to a
        handover it did not need.

        `pool_of` maps job id to pool. With it, a conversation that has been
        served before is re-placed only inside its pool (see the class
        docstring); without it, or with `pool_affinity` off, placement spans
        every pool.
        """
        now = time.time() if now is None else now
        self._expire(now)
        pool_of = pool_of or {}
        if key is None:
            return self.select(accepting) if accepting else None
        # A hand-placed pin outranks everything, including the load picture --
        # that is the point of setting one. It still yields to a backend that
        # has actually gone away, because refusing to serve would be worse.
        manual = self.manual.get(key)
        if manual is not None:
            if manual in serving:
                self.hits += 1
                self._note_family(key, pool_of.get(manual), now)
                return manual
            LOG.warning("manual pin %s -> %s is not serving; falling back", key[:40], manual)
        pinned = self.pins.get(key)
        if pinned is not None and now - pinned[1] > self.ttl:
            # The amortised sweep in _expire may not have reached this key yet.
            # Expiring the one key we were asked about is O(1) and keeps the
            # observable behaviour identical to the old full scan per request.
            self._drop_pin(key)
            pinned = None
        if pinned is not None:
            job_id = pinned[0]
            if job_id in serving:
                self._set_pin(key, job_id, now)
                self.hits += 1
                self._note_family(key, pool_of.get(job_id), now)
                return job_id
            # The backend it was pinned to is gone. Re-pin rather than fail:
            # the conversation loses its cache, which is the whole cost.
            self.rehomed += 1
            LOG.info("conversation %s lost backend %s; re-homing", key[:40], job_id)
        else:
            self.misses += 1
        family = None
        if self.pool_affinity:
            family = self._family_of(key, now)
            if family is None and pinned is not None:
                # Pinned before its pool was recorded -- every conversation
                # restored from a predecessor that did not record pools. Its
                # backend says which pool it belongs to, registered or not.
                family = pool_of.get(pinned[0])
        if family is not None:
            accepting = {j: v for j, v in accepting.items() if pool_of.get(j) == family}
            if not accepting:
                # Another pool's backend would continue this conversation on a
                # different model. No backend is the lesser harm: the client
                # retries, and the pool's backends come back.
                self.family_waits += 1
                # Every conversation of the pool retries while it is out, so
                # this is reported per pool at most every 30 s, with the count.
                if now - self._family_wait_logged.get(family, 0.0) >= 30.0:
                    self._family_wait_logged[family] = now
                    LOG.warning(
                        "no backend of pool %s is accepting; conversation %s and the pool's "
                        "others wait rather than move to another pool (%d held so far)",
                        family,
                        key[:40],
                        self.family_waits,
                    )
                return None
        if not accepting:
            return None
        job_id = self.select(accepting)
        self._set_pin(key, job_id, now)
        self._trim()
        self._note_family(key, pool_of.get(job_id), now)
        self.dirty = True
        return job_id

    @staticmethod
    def _stat(stats, index):
        return stats[index] if len(stats) > index else 0

    def select(self, accepting):
        """Place a new conversation according to the current policy.

        `accepting` maps job id to (conversations, inflight, remaining_seconds).

        least_conversations is the default because conversation count is what
        predicts future load: in-flight counts only what is happening this
        instant, and an agent spends most of a turn thinking rather than
        streaming, so a backend holding five idle sessions reads as empty.
        """
        jobs = sorted(accepting)
        custom = getattr(self, "policies", None)
        if custom is not None and self.policy not in POLICIES:
            picked = custom.run(self.policy, accepting)
            if picked is not None:
                return picked
            # run() has already logged why. Fall through rather than fail the
            # placement: a bad policy should cost its own behaviour, not the
            # conversation that happened to arrive while it was loaded.
        if self.policy == "least_inflight":
            return min(
                jobs, key=lambda j: (self._stat(accepting[j], 1), self._stat(accepting[j], 0), j)
            )
        if self.policy == "round_robin":
            job = jobs[self._rr % len(jobs)]
            self._rr += 1
            return job
        if self.policy == "longest_lived":
            # Ties broken by load, so a fleet of equally long-lived backends
            # still balances instead of piling onto whichever sorts first.
            return max(
                jobs, key=lambda j: (self._stat(accepting[j], 2), -self._stat(accepting[j], 0))
            )
        return min(
            jobs, key=lambda j: (self._stat(accepting[j], 0), self._stat(accepting[j], 1), j)
        )

    # -- manual control ---------------------------------------------------
    def pin(self, key, job_id):
        self.manual[key] = job_id
        self._set_pin(key, job_id, time.time())
        self.dirty = True

    def unpin(self, key):
        removed = self.manual.pop(key, None)
        self._drop_pin(key)
        self.dirty = True
        return removed

    def counts(self, job_ids):
        # O(backends), not O(pins) -- see _tally in __init__.
        return {job_id: self._tally.get(job_id, 0) for job_id in job_ids}


class InflightCounter:
    """How many requests this PROCESS is currently serving.

    `Fleet.inflight` already counts requests, but per backend, and it is the
    wrong instrument for a drain: it is keyed by job id, it is only maintained
    around `proxy`, and every early return above that -- an unparsable head, a
    401, a 503 with no backend, anything under `/_gateway/` -- never touches
    it. A handover has to answer a different question, "does this process
    still owe anybody a response", and that one has to count all of them.

    One module-level instance guards every concurrent handler at once, which
    is safe because `__enter__`/`__exit__` only add and subtract; there is no
    per-use state to collide.

    No lock, deliberately. Every mutation happens on the event loop thread, so
    `+= 1` is already exact -- `in_thread` work never reaches here. An
    `asyncio.Lock` would be worse than useless: acquiring it is an `await`, and
    that would open a suspension point between the increment and the guarded
    block where a cancellation could slip in and leak a count forever.
    """

    def __init__(self):
        self.count = 0
        self._idle_waiters = []

    def __enter__(self):
        self.count += 1
        return self

    def __exit__(self, exc_type, exc, tb):
        # Synchronous, so it runs on the cancellation path too -- which is the
        # entire point. This gateway sees a steady trickle of "Task was
        # destroyed but it is pending!" from clients that hang up mid-stream,
        # and a counter those could escape would never fall back to zero, so
        # every drain would sit out its full deadline and then report phantom
        # abandoned requests.
        self.count -= 1
        if self.count == 0:
            for waiter in self._idle_waiters:
                if not waiter.done():
                    waiter.set_result(None)
            self._idle_waiters.clear()
        return False

    async def wait_idle(self):
        """Block until this process owes nobody a response."""
        if self.count == 0:
            return
        # A future made here, on the loop that is actually draining, rather
        # than an asyncio.Event built in __init__. INFLIGHT is a module global
        # constructed at import time, and an asyncio primitive binds to the
        # first loop that awaits it and then refuses every other one with
        # "bound to a different event loop" -- measured. One gateway process
        # only ever runs one loop, but a test that calls asyncio.run() twice
        # would trip over it, and that is precisely what a drain test does.
        waiter = asyncio.get_running_loop().create_future()
        self._idle_waiters.append(waiter)
        try:
            await waiter
        finally:
            # The deadline expiring cancels this; leaving a dead future behind
            # would make the next zero-crossing call set_result on it.
            if waiter in self._idle_waiters:
                self._idle_waiters.remove(waiter)


# Process-level, not per-Gateway: the drain and the handover endpoints ask
# about the process, and there is exactly one Gateway in it anyway.
INFLIGHT = InflightCounter()
# ---------------------------------------------------------------------------
# Hot handover
# ---------------------------------------------------------------------------
# Replacing the gateway process without refusing a connection. The successor
# binds the same port with SO_REUSEPORT while this process is still serving on
# it, proves itself, and only then does this process stop accepting.
#
# Phases are exactly the ones the design contract names; nothing else may
# appear in a status answer, because that answer is what a script driving a
# rollout branches on.
HANDOVER_PHASES = ("idle", "spawning", "waiting_ready", "draining", "done", "aborted")

# The phases a new handover may be started from. "aborted" is one of them on
# purpose: an abort leaves this process fully in service, so a second attempt
# once the reason has been fixed is an ordinary request, not a retry of
# something still running. "done" is not: this process is on its way out and
# the successor is the one to ask.
HANDOVER_RESTARTABLE = ("idle", "aborted")

HANDOVER_POLL_INTERVAL = 1.0
# How long the successor gets to answer SIGTERM before SIGKILL. It has its own
# drain to run and almost nothing to drain, so this is short.
HANDOVER_TERM_GRACE = 10.0
HANDOVER_KILL_GRACE = 10.0

# 202 and 409 are not errors, so they are not in ERROR_REASONS -- which
# error_response() indexes in step with ERROR_BODIES, and neither has a body
# for these. Kept here rather than widening that pair.
HANDOVER_REASONS = {
    202: "Accepted",
    400: "Bad Request",
    409: "Conflict",
    503: "Service Unavailable",
}


def dial_host(host):
    """The address to connect to for a listener bound to `host`.

    A wildcard bind has no address to dial back: 0.0.0.0 as a *destination*
    only works by accident, and on some stacks not at all. The successor binds
    exactly what this process bound, so the question is only ever "what reaches
    our own port from here".
    """
    if host in ("", "0.0.0.0", "*"):
        return "127.0.0.1"
    if host in ("::", "[::]"):
        return "::1"
    return host


def successor_argv():
    """The command line for the next generation: this one, plus --router-only.

    Absolute, because the successor is started with start_new_session and will
    outlive the process whose relative paths made sense. --router-only is
    appended unconditionally -- argparse's store_true makes a second copy a
    no-op, which is what a handover *out of* a --router-only generation needs,
    and every generation after the first is one of those.
    """
    return [sys.executable, os.path.abspath(sys.argv[0])] + list(sys.argv[1:]) + ["--router-only"]


def pid_file_path(args):
    """Where this generation announces itself: `<GW_DIR>/gateway.pid`.

    GW_DIR is not passed to this process. gateway.sbatch keeps the users file
    in it (`--users "$GW_OWN/users.txt"`), and --users is both required and
    always inside that directory, so it is the handle on GW_DIR that already
    exists. Deriving the path from it keeps the two in step without adding a
    CLI knob the contract does not list.
    """
    return os.path.join(os.path.dirname(os.path.abspath(args.users)), "gateway.pid")


def write_pid_file(args):
    """Announce this generation's pid, atomically. Returns the path, or None.

    Atomic because the job script reads this while a successor is writing it,
    and a reader that catches a half-written file does not get an error -- it
    gets a shorter number, which is a perfectly valid pid belonging to somebody
    else. The temporary name carries the pid for the reason Router.save's does:
    two generations overlap by design during a handover, and one shared
    temporary name is one of them clobbering the other's write.
    """
    path = pid_file_path(args)
    tmp = "%s.tmp.%d" % (path, os.getpid())
    try:
        with open(tmp, "w") as handle:
            handle.write("%d\n" % os.getpid())
            handle.flush()
            # The reader is a shell script on a different clock; a pid sitting
            # in the page cache is not an announcement.
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except OSError as exc:
        # Advisory. Losing it costs the job script its handle on the current
        # generation, which is a problem for the job script -- not a reason for
        # a gateway that is otherwise serving fine to refuse to start.
        LOG.warning("could not write pid file %s: %s", path, exc)
        try:
            os.unlink(tmp)
        except OSError:
            pass
        return None
    LOG.info("pid file: %s (pid %d)", path, os.getpid())
    return path


async def successor_ready(host, port, pid, timeout):
    """Is the gateway that answered on `port` the successor, and is it ready?

    The pid check is not belt and braces. The successor binds with
    SO_REUSEPORT, so for the length of the handover there are two listeners on
    this port and the kernel gives each new connection to whichever one the
    4-tuple hashes to -- which means roughly half of these polls reach the
    *outgoing* gateway. That one is ready by definition and would cheerfully
    answer 200 to its own question, and taking that as proof would close the
    outgoing listener while the successor was still loading: precisely the
    outage this sequence exists to prevent. Every new connection re-hashes, so
    the poll that reaches the successor arrives within a few attempts.

    The answer is the successor's ready report when it is the successor and
    ready -- truthy, so a caller that only asks "is it" can treat it as a
    bool -- and False otherwise.
    """
    writer = None
    raw = b""
    try:
        reader, writer = await asyncio.wait_for(asyncio.open_connection(host, port), timeout)
        writer.write(
            b"GET /_gateway/ready HTTP/1.1\r\nHost: %s\r\nConnection: close\r\n\r\n"
            % host.encode("latin-1")
        )
        await writer.drain()
        # Read to EOF rather than one chunk: the answer is small, but "small"
        # is not "one segment", and a truncated body reads as a pid mismatch,
        # which would stall the handover until the timeout for no reason.
        while len(raw) < MAX_HEAD_BYTES:
            chunk = await asyncio.wait_for(reader.read(8192), timeout)
            if not chunk:
                break
            raw += chunk
    except (asyncio.TimeoutError, ConnectionError, OSError):
        return False
    finally:
        if writer is not None:
            await close(writer)
    head, _, body = raw.partition(b"\r\n\r\n")
    status_line = head.split(b"\r\n")[0] if head else b""
    if b" 200 " not in status_line:
        return False
    try:
        report = json.loads(body)
        answered_by = report.get("pid")
    except (ValueError, AttributeError):
        return False
    return report if answered_by == pid else False


async def terminate_successor(proc):
    """Stop a successor that never became ready. Does not return until it is gone.

    SIGTERM first, so its own drain runs and the handful of connections the
    kernel already handed it finish instead of being cut; SIGKILL only once it
    has had its grace period. The signals go to the process *group*: the
    successor leads one of its own (start_new_session), so the group is the way
    to reach anything it spawned -- serve.sh, fleetctl, an sbatch -- which a
    bare proc.terminate() would orphan.
    """
    if proc.returncode is not None:
        return proc.returncode
    escalation = ((signal.SIGTERM, HANDOVER_TERM_GRACE), (signal.SIGKILL, HANDOVER_KILL_GRACE))
    for sig, grace in escalation:
        try:
            os.killpg(os.getpgid(proc.pid), sig)
        except (ProcessLookupError, PermissionError, OSError) as exc:
            # Racing its exit, or a group this process may not signal. Fall
            # back to the child itself: it is the one holding the port, which
            # is the part that must not survive.
            LOG.debug("could not signal successor group: %s", exc)
            try:
                proc.send_signal(sig)
            except ProcessLookupError:
                pass
        try:
            # wait() also reaps it. An unreaped successor is a zombie holding
            # nothing, but it is also the only evidence left that the abort
            # finished, and the status endpoint would keep reporting a pid that
            # no longer means anything.
            return await asyncio.wait_for(proc.wait(), grace)
        except asyncio.TimeoutError:
            LOG.warning("successor %d ignored %s; escalating", proc.pid, sig.name)
    # SIGKILL cannot be ignored, so getting here means the child is stuck in
    # uninterruptible sleep and there is nothing further to try. Say so loudly:
    # it may still hold a share of the port.
    LOG.error("successor %d survived SIGKILL; it may still be sharing the port", proc.pid)
    return None


class Handover:
    """Hands this process's port to a fresh one without refusing a connection.

    The eight steps of the design contract, with the ordering property that
    makes the whole thing tractable stated once, here: **this process does not
    close its listener until step 6**, which is after the successor has proved
    itself on the port. Every failure before that point is therefore a no-op
    for traffic -- the successor is killed and this process carries on with a
    listener it never stopped using, so there is nothing to reopen and no
    window in which nobody is accepting. Nothing below may move the close()
    earlier, and nothing before it may become irreversible.

    Steps 2, 7 and 8 are not implemented here. The pin-table flush belongs to
    the router, the supervisor lock to the supervisor and the drain to the
    request path; this class calls them and owns the order they happen in.
    They are resolved by name at call time rather than assumed, so this
    sequence can land and be exercised before they do: a missing collaborator
    degrades one step and says so, instead of turning the whole handover into
    an AttributeError halfway through.

    One trap, because it is invisible and it is fatal. asyncio's subprocess
    transport kills the child it is holding when the transport closes, and the
    transport closes on any orderly interpreter shutdown -- so from step 6,
    when the successor is the only listener left, this process exiting
    *politely* takes the gateway down with it. `_run` leaves through os._exit
    for that reason and swallows a cancel during the drain for the same one.
    Anything else added here that can end this process between step 6 and the
    exit has to go the same way.
    """

    def __init__(self, fleet):
        self.fleet = fleet
        self.server = None
        self.phase = "idle"
        self.successor = None
        self.successor_pid = None
        self.started_at = None
        self.detail = "no handover has been started"
        # Held so the task is not collected mid-handover; see _run().
        self.task = None
        # Pools the operator retired with this handover. Any other pool this
        # process serves, the successor has to serve too; see _wait_ready().
        self.dropping = ()

    def attach(self, server):
        """Take the listener. Called once the bind has succeeded, never before."""
        self.server = server

    def listening(self):
        # Server.close() drops the socket list, so this is the question
        # "is the listener still open" and not "was it ever opened".
        return self.server is not None and bool(self.server.sockets)

    def inflight(self):
        """Requests this process is still serving, from the process-level counter.

        The same number the drain works from, deliberately: a status answer
        that counts differently from the thing deciding when to stop waiting is
        worse than no status answer at all.
        """
        counter = globals().get("INFLIGHT")
        if counter is None:
            # Not merged yet. The per-backend table the request path already
            # keeps is the nearest thing this file has on its own, and it
            # misses every request that never reached a backend -- which is
            # the gap the process-level counter exists to close.
            return sum(self.fleet.inflight.values())
        return counter.count

    def ready(self):
        """(ready, why): can this process take traffic right now?

        The contract's three conditions, cheapest first. "Fleet dir read" is
        checked as "the directory is still there" plus the healthy-backend test
        below, which cannot pass unless a sweep has read a registration out of
        it and a probe has answered.
        """
        if not self.listening():
            return False, "not listening"
        if self.phase in ("draining", "done"):
            return False, "handed over; no longer taking new traffic"
        fleet_dir = self.fleet.args.fleet_dir
        if not os.path.isdir(fleet_dir):
            return False, "fleet dir %s is not a readable directory" % fleet_dir
        healthy = self.fleet.serving()
        if not healthy:
            return False, "no healthy backend"
        return True, "%d healthy backend(s)" % len(healthy)

    def lifecycle_refusal(self):
        """Why this process must not act on the fleet's lifecycle, or None.

        `start_server` and `stop_server` submit and cancel jobs -- the same
        actions `supervise` takes, reached through a different door. `supervise`
        is gated on the supervisor lock; these two were not, and a handover is
        what makes that gap load-bearing: through steps 4 to 6 both generations
        are accepting HTTP, and the kernel decides which of them gets the POST.
        A `stop_server` landing on the *successor* would release every serving
        job and clear `superseded` while the outgoing process was still the
        supervisor -- and that one would see an empty fleet and resubmit. Two
        supervisors acting at once is the single thing the lock exists to
        prevent, so these two have to answer to it as well.

        The two conditions are different failures, and the messages say which.
        Without the lock this process is not the supervisor and was never
        entitled to act. With the lock but mid-handover it is about to stop
        being one: it would take the action, record `stopped` on itself alone,
        and hand the fleet to a successor that knows nothing about it and
        resubmits within the minute.

        A deployment with no supervisor lock at all -- the class missing, not
        the lock unheld -- keeps today's behaviour. There is only one gateway
        in that world, so there is nothing to be exclusive about.
        """
        lock = getattr(self.fleet, "supervisor_lock", None)
        if lock is not None and not lock.held:
            return {
                "error": "this gateway is not the fleet's supervisor",
                "detail": "lifecycle actions belong to whichever generation holds %s"
                % getattr(lock, "path", "the supervisor lock"),
                "phase": self.phase,
            }
        if self.phase not in HANDOVER_RESTARTABLE:
            return {
                "error": "a handover is in progress on this gateway",
                "detail": self.detail,
                "phase": self.phase,
            }
        return None

    def status(self):
        """Exactly the five fields the contract names, and no others."""
        return {
            "phase": self.phase,
            "successor_pid": self.successor_pid,
            "inflight": self.inflight(),
            "started_at": self.started_at,
            "detail": self.detail,
        }

    # -- steps 1-3, on the request's own task ------------------------------
    async def start(self, drop_pools=()):
        """Steps 1 to 3. Returns (http status, body).

        The spawn happens here rather than on the background task because the
        answer has to carry the successor's pid, and a 503 has to mean "the
        spawn failed" rather than "ask again later". Everything after the spawn
        can be answered for asynchronously, and is.

        `drop_pools` names pools this process serves that the successor may
        leave out: retiring a pool, on purpose. Any other pool the successor
        lacks aborts the handover -- see _wait_ready().
        """
        if self.phase not in HANDOVER_RESTARTABLE:
            return 409, {
                "status": "already_running",
                "successor_pid": self.successor_pid,
                "phase": self.phase,
                "detail": self.detail,
            }
        unknown = sorted(set(drop_pools) - set(self.fleet.pools))
        if unknown:
            return 400, {
                "status": "bad_request",
                "detail": "drop_pools names pool(s) this gateway does not serve: %s"
                % ", ".join(unknown),
                "pools": list(self.fleet.pools),
            }
        self.dropping = tuple(drop_pools)
        # The test above and the claim below are one block on purpose: there is
        # no await between them, so on a single-threaded loop a second POST
        # arriving mid-handover cannot pass the test. An await here would
        # reintroduce exactly the double spawn this prevents -- two successors
        # racing for one port, one of which nobody holds a handle on.
        self.phase = "spawning"
        self.started_at = time.time()
        self.successor = None
        self.successor_pid = None
        self.detail = "flushing the pin table"
        LOG.info("handover requested; flushing the pin table before spawning")

        # -- step 2 --------------------------------------------------------
        # Before the spawn, and on the event loop rather than a thread, so
        # "before" is a fact rather than a scheduling accident: the successor
        # reads this file as it boots, and a pin table one flush behind
        # re-homes every conversation that moved since the last periodic save.
        # Awaited to completion here rather than alongside the spawn: "before"
        # is the guarantee, and a flush racing the successor's startup read is
        # not one.
        await self._flush_router()

        # -- step 3 --------------------------------------------------------
        try:
            self.successor = await self._spawn()
        except OSError as exc:
            self._set("aborted", "could not spawn a successor: %s" % exc)
            LOG.error("handover: %s", self.detail)
            return 503, {"status": "spawn_failed", "successor_pid": None, "detail": self.detail}
        self.successor_pid = self.successor.pid
        self._set(
            "waiting_ready",
            "successor %d spawned; waiting for it to report ready" % self.successor_pid,
        )
        LOG.info("handover: successor is pid %d", self.successor_pid)
        self.task = asyncio.ensure_future(self._run())
        return 202, {"status": "started", "successor_pid": self.successor_pid}

    async def _spawn(self):
        argv = successor_argv()
        LOG.info("handover: spawning %s", " ".join(argv))
        # start_new_session because the successor has to outlive this process:
        # it is still serving after step 8 exits, and under SLURM it must not
        # be in a process group the scheduler tears down with its parent.
        #
        # stdout and stderr are deliberately inherited rather than piped. A
        # pipe nobody reads fills at 64 KiB and blocks the successor on its
        # next log line -- an outage with no symptom, from a gateway that looks
        # alive in every way except that it never answers. The inherited
        # descriptors are the job's own output file and stay valid after this
        # process is gone. stdin is closed for the opposite reason: nothing
        # will ever write to it, and inheriting a terminal would make the
        # successor stop on SIGTTIN.
        return await asyncio.create_subprocess_exec(
            *argv,
            stdin=asyncio.subprocess.DEVNULL,
            start_new_session=True,
        )

    # -- steps 4-8, on a task of its own -----------------------------------
    async def _run(self):
        ready = False
        try:
            ready = await self._wait_ready()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            LOG.exception("handover: waiting for the successor failed")
            self.detail = "waiting for the successor failed: %s" % exc
        if not ready:
            await self._abort(self.detail)
            return

        # -- step 6: the point of no return --------------------------------
        # Everything above this line is reversible and everything below it is
        # not. The listener closes here and not one step earlier, which is the
        # entire reason the abort above is safe -- and the reason no failure
        # from here on may kill the successor: it is the only listener left.
        self._set("draining", "listener closed; successor %d owns the port" % self.successor_pid)
        LOG.info(
            "handover: closing the listener; every new connection now goes to %d",
            self.successor_pid,
        )
        self.server.close()

        # -- step 7 --------------------------------------------------------
        self._release_lock()

        # -- step 8 --------------------------------------------------------
        try:
            await self._drain()
        except asyncio.CancelledError:
            # Deliberately not re-raised, which nothing else in this file does.
            # A cancel here means the process is winding down some other way,
            # and winding down is the one thing that must not happen now: see
            # the exit below for what an orderly shutdown does to the
            # successor. Fall through to it instead.
            LOG.warning("handover: drain cancelled; leaving without finishing it")
        except Exception:
            # Exiting is still right: this process has no listener and no lock,
            # so staying up serves nobody. Say what went wrong and go.
            LOG.exception("handover: drain failed; exiting anyway")
        self._set("done", "handed over to %d" % self.successor_pid)
        LOG.info("handover complete; exiting 0")
        # os._exit, and not for speed. asyncio's subprocess transport kills the
        # child it is holding when it is closed, and it is closed on the way
        # out of asyncio.run() -- so an orderly exit here reaches interpreter
        # shutdown and SIGKILLs the successor, which by this point is the only
        # listener on the port. start_new_session does not help: the kill is a
        # direct one, aimed at a pid this process still holds a handle on.
        # Measured on 3.12.3: an ordinary exit leaves the successor dead, this
        # one leaves it serving.
        #
        # Nothing is lost by leaving this way. The listener is closed, the
        # drain is done, the supervisor lock is gone, and the pin table must
        # NOT be written from here -- after step 7 this process is no longer
        # the lock holder, and the lock holder is the only one allowed to
        # persist router state.
        for stream in (sys.stdout, sys.stderr):
            try:
                stream.flush()
            except (ValueError, OSError):
                pass
        logging.shutdown()
        os._exit(0)

    async def _wait_ready(self):
        """Steps 4 and 5: poll the successor until it reports ready, or give up."""
        args = self.fleet.args
        host = dial_host(args.host)
        limit = time.monotonic() + args.handover_ready_timeout
        while True:
            # Asked first, and every round: a successor that failed to bind --
            # the likeliest reason for one not to come up -- exits in under a
            # second, and waiting out a three minute timeout to report that
            # would be three minutes of a handover nobody can cancel.
            if self.successor.returncode is not None:
                self.detail = "successor exited with status %s before reporting ready" % (
                    self.successor.returncode,
                )
                return False
            report = await successor_ready(host, args.port, self.successor_pid, args.probe_timeout)
            if report:
                missing = self.missing_pools(report)
                if missing:
                    # Ready, and incomplete. From the moment the listener
                    # closes a missing pool's backends would get no traffic,
                    # their conversations would re-home and nothing would
                    # supervise them -- so this is refused outright rather
                    # than waited on: nothing about the successor will change
                    # by itself.
                    self.detail = (
                        "successor %d does not serve pool(s) %s, so their backends would drop "
                        "out of routing and supervision; restore them (--pool or %s) and hand "
                        'over again, or retire them on purpose with {"drop_pools": [...]}'
                        % (
                            self.successor_pid,
                            ", ".join(missing),
                            getattr(args, "pools_file", None) or "a pools file",
                        )
                    )
                    return False
                LOG.info("handover: successor %d reports ready", self.successor_pid)
                return True
            left = limit - time.monotonic()
            if left <= 0:
                self.detail = "successor %d did not report ready within %ds" % (
                    self.successor_pid,
                    args.handover_ready_timeout,
                )
                return False
            self.detail = "waiting for successor %d to report ready (%ds left)" % (
                self.successor_pid,
                round(left),
            )
            await asyncio.sleep(HANDOVER_POLL_INTERVAL)

    def missing_pools(self, report):
        """Pools this process serves that the successor's ready report lacks.

        Less the ones being retired with this handover. A successor that says
        nothing about pools -- one built before they existed -- serves none.
        """
        theirs = report.get("pools") if isinstance(report, dict) else None
        theirs = set(theirs) if isinstance(theirs, list) else set()
        return sorted(set(self.fleet.pools) - set(self.dropping) - theirs)

    async def _abort(self, why):
        """Kill the successor and stay in service.

        Safe only because step 6 has not run: this process still holds the
        listener it has held all along, so there is nothing to reopen and no
        moment when nobody was accepting. The one thing that must not survive
        this function is the successor -- it is bound to our port with
        SO_REUSEPORT, so a survivor keeps taking a share of every new
        connection while believing it is still starting up.
        """
        proc = self.successor
        self.successor = None
        if proc is not None:
            await terminate_successor(proc)
        # Put the pid file back. The successor wrote its own at startup and the
        # job script waits on whichever generation it names; leaving a dead pid
        # there ends the SLURM job while this gateway is still serving, which
        # turns a handover that failed safely into an outage.
        write_pid_file(self.fleet.args)
        self._set("aborted", why)
        LOG.warning("handover aborted: %s; still serving on the original listener", why)

    # -- collaborators owned by other work items ---------------------------
    async def _flush_router(self):
        """Step 2, via Router.flush_now()."""
        flush = getattr(self.fleet.router, "flush_now", None)
        if flush is None:
            LOG.error(
                "handover: Router.flush_now() is missing; the successor will boot from "
                "whatever the periodic save last wrote"
            )
            return
        try:
            # It is a coroutine, and the await is the entire point: without it
            # this line builds a coroutine object, drops it, and returns having
            # written nothing -- and the successor then boots from a stale pin
            # table with not one word in the log to say so.
            written = await flush()
        except OSError:
            # A pin table one flush behind costs cache warmth on whatever moved
            # recently, and nothing else. Not a reason to refuse a handover
            # that is otherwise fine.
            LOG.exception("handover: flushing the pin table failed; continuing")
            return
        if written is False:
            LOG.info("handover: --router-state is disabled; nothing to flush")

    def _release_lock(self):
        """Step 7, via the SupervisorLock on the fleet."""
        lock = getattr(self.fleet, "supervisor_lock", None)
        if lock is None:
            LOG.warning(
                "handover: no supervisor lock on the fleet; the successor will start "
                "supervising only once that work item lands"
            )
            return
        try:
            if not getattr(lock, "held", True):
                LOG.info("handover: supervisor lock was not held here; nothing to release")
                return
            lock.release()
        except OSError:
            # Nothing to do about it from here, and the listener is already
            # closed. flock is released by the kernel when this process exits,
            # so the successor gets it either way -- a few seconds later.
            LOG.exception("handover: releasing the supervisor lock failed")
            return
        LOG.info("handover: supervisor lock released; %d may take over", self.successor_pid)

    async def _drain(self):
        """Step 8, via the drain routine on the request path."""
        routine = globals().get("drain")
        if routine is None:
            # No local substitute is offered on purpose. wait_closed() looks
            # like one and is not: it is keyed on the transport, so it returns
            # the moment a client's socket dies and says "idle" with the
            # handler still running -- which for this gateway is the normal
            # case, not an edge one. Abandoning the requests loudly beats
            # abandoning them while reporting a clean drain.
            LOG.error(
                "handover: drain() is missing; step 8 cannot run and %d request(s) are "
                "being abandoned",
                self.inflight(),
            )
            return
        await routine(self.server, self.fleet.args.handover_drain_deadline)

    def _set(self, phase, detail):
        self.phase = phase
        self.detail = detail


def mirror_table(mirrors):
    """The mirror map as it is reported: job id -> ["host:port", ...].

    A list even when there is one entry. The value used to be a bare string,
    and a shape that changes with the count makes every reader carry both
    branches forever; the count is visible from the length.
    """
    return {job: ["%s:%d" % t for t in targets] for job, targets in sorted(mirrors.items())}


def _mirror_count(fleet, target, key):
    """One event, counted twice: in the totals and against its target.

    The totals answer "is mirroring working"; the per-target split answers
    "which one" -- a question that does not exist with one target and decides
    everything with two, because a comparison where one side silently received
    half the copies is not a comparison.
    """
    fleet.mirror_stats[key] += 1
    fleet.mirror_target_stats["%s:%d" % target][key] += 1


def _mirror_missed(fleet, target, why):
    """Count a failed copy, and stop mirroring to a target that has gone.

    A mirror is set up to answer a question and then left alone, so "keeps
    trying forever" is the default nobody chooses and everybody gets. The
    failures are already counted, but a counter is only a signal to whoever is
    looking at it, and the reason for adding a mirror is usually that nobody is
    looking yet.

    So a target that has missed this many copies in a row is dropped, loudly.
    Consecutive rather than cumulative: a target that answers at all is working,
    and a handful of failures in an hour of copies says nothing. Re-enable it
    with another POST once the target is back -- deliberately manual, since a
    mirror silently resuming is how it came to be pointed at something that no
    longer exists.
    """
    misses = fleet.mirror_misses[target] = fleet.mirror_misses[target] + 1
    if misses < fleet.args.mirror_max_misses:
        return
    host, port = target
    # Only the dead target is removed. A backend mirrored to two places where
    # one has gone must keep copying to the other -- the healthy half of an
    # experiment is precisely the half still worth having.
    stopped = []
    for job, targets in list(fleet.mirrors.items()):
        if target in targets:
            targets.remove(target)
            stopped.append(job)
            if not targets:
                fleet.mirrors.pop(job, None)
    fleet.mirror_misses.pop(target, None)
    _mirror_count(fleet, target, "disabled")
    # Or a successor would pick the dead target back up from the state file.
    fleet.router.dirty = True
    fleet.router.save()
    LOG.error(
        "mirror to %s:%d failed %d times in a row (%s); it is no longer being copied to. "
        "Backends affected: %s. POST /_gateway/mirror again once the target is back.",
        host,
        port,
        misses,
        why,
        ", ".join(sorted(stopped)) or "none",
    )


async def mirror_request(fleet, target, method, path, headers, body, user):
    """Send a copy of one request to a server the fleet does not own.

    Shadow traffic: the copy exists to make another server do the same work,
    so that its behaviour under a real load can be compared against the
    instance actually serving. Its answer is read to completion and dropped.
    Reading rather than hanging up is deliberate -- an early close invites the
    server to cancel the generation, and a generation it did not finish is not
    the measurement anybody wanted.

    Nothing here may reach the caller. Every failure is counted and swallowed:
    the request being mirrored is the one that matters, and a mirror that can
    turn a served request into an error is worse than no mirror.
    """
    if fleet.mirror_slots.locked():
        # Already at the ceiling. Dropping is the right answer rather than
        # queueing: a target that has stopped draining would otherwise build a
        # backlog of copies that are stale by the time they are sent.
        _mirror_count(fleet, target, "dropped")
        return
    host, port = target
    async with fleet.mirror_slots:
        up_reader = up_writer = None
        try:
            async with asyncio.timeout(fleet.args.mirror_timeout):
                # Connecting gets its own, much shorter deadline. Sharing the
                # overall one means a target that accepts nothing and refuses
                # nothing -- a dropped route rather than a closed port -- holds
                # a slot for the full timeout, and sixty-four of those hold
                # every slot for fifteen minutes at a time.
                up_reader, up_writer = await asyncio.wait_for(
                    asyncio.open_connection(host, port), fleet.args.mirror_connect_timeout
                )
                lines = [
                    "%s %s HTTP/1.1" % (method, path),
                    "Host: %s:%d" % (host, port),
                    "Connection: close",
                    "Accept-Encoding: identity",
                    "X-Gateway-User: %s" % user,
                    # So the far side can tell copied traffic from the real
                    # thing in its own logs. It has no effect here.
                    "X-Gateway-Mirror: 1",
                ]
                for name, value in headers:
                    if name.lower() in STRIP_REQUEST_HEADERS:
                        continue
                    lines.append("%s: %s" % (name, value))
                up_writer.write(("\r\n".join(lines) + "\r\n\r\n").encode("latin-1"))
                if body:
                    up_writer.write(body)
                await up_writer.drain()
                # The status is read before the body is drained, because
                # "delivered" and "accepted" are different questions and only
                # the second one says the mirror is working. A target that
                # answers 404 to everything completes the exchange exactly
                # like one that is serving: connection made, request written,
                # response read to EOF. Counting only that made a mirror that
                # had never processed a single request report 2959 sent.
                # The head, and then nothing. Measured against this fleet's
                # own target: a request closed immediately after its status
                # line still produced all 200 requested tokens, exactly as one
                # read to completion did. The server does not take the hang-up
                # as a cancellation, so draining the body buys no extra work
                # from it -- it only moves the bytes.
                #
                # And those bytes are the expensive part. The traffic being
                # copied here carries prompts with a median around 80k tokens,
                # so reading every response back to EOF would put a second
                # full response through the same event loop that is serving
                # the first, for nothing.
                head, _ = await read_head(up_reader)
                status = 0
                if head is not None:
                    try:
                        line, _ = parse_response_head(head)
                        status = int(line.split(" ")[1])
                    except (ValueError, IndexError):
                        status = 0
            if 200 <= status < 300:
                _mirror_count(fleet, target, "sent")
                fleet.mirror_misses.pop(target, None)
            elif status == 0:
                # Connected, wrote, and got nothing back that parses as a
                # response. Not a transport failure and not an answer either.
                _mirror_count(fleet, target, "no_response")
                _mirror_missed(fleet, target, "no parsable response")
            else:
                # A target rejecting every copy is as useless as one that is
                # down, so it counts against the same budget. Without this the
                # only symptom is a counter nobody is watching -- which is how
                # a mirror can run for an hour against a server that processed
                # none of it.
                _mirror_count(fleet, target, "rejected_%dxx" % (status // 100))
                _mirror_missed(fleet, target, "HTTP %d" % status)
        except asyncio.CancelledError:
            _mirror_count(fleet, target, "failed")
            raise
        except (OSError, ConnectionError, TimeoutError, asyncio.TimeoutError) as exc:
            _mirror_count(fleet, target, "failed")
            _mirror_missed(fleet, target, exc)
            # One line per distinct failure would be one line per request when
            # a target is down, so this is deliberately quiet; the counters in
            # /_gateway/fleet are the place to notice.
            LOG.debug("mirror to %s:%d failed: %s", host, port, exc)
        except Exception:  # noqa: BLE001 - a mirror may not take down a request
            _mirror_count(fleet, target, "failed")
            LOG.exception("mirror to %s:%d raised", host, port)
        finally:
            if up_writer is not None:
                await close(up_writer)


class Gateway:
    """Terminates client connections and forwards them to the active backend."""

    def __init__(self, fleet):
        self.fleet = fleet
        self.handover = Handover(fleet)

    async def handle(self, reader, writer):
        # A thin wrapper purely so the accounting cannot be escaped: `_handle`
        # returns from eight different places, and wrapping the body instead
        # would mean trusting every future early return to remember. Anything
        # that reaches the connection handler at all is in flight, including
        # the requests that never get as far as a backend.
        with INFLIGHT:
            await self._handle(reader, writer)

    async def _handle(self, reader, writer):
        peer = writer.get_extra_info("peername")
        started = time.time()
        trace = RequestTrace()
        try:
            head, rest = await read_head(reader)
            if head is None:
                return
            method, path, headers = parse_request_head(head)
        # OSError, not just ConnectionError: reader.read raises other OSError
        # subclasses (a socket timeout, for one). ConnectionError is itself an
        # OSError subclass, so this only widens. Letting one escape means the
        # task dies with an unretrieved exception and close(writer) never runs,
        # leaking the connection.
        except (ValueError, OSError) as exc:
            # INFO, not DEBUG: the gateway runs at INFO, so an unparsable
            # client request used to leave no trace whatsoever -- and that is
            # the single failure mode a client-integration bug presents as.
            LOG.info("dropped rid=%s from %s: %s", trace.rid, peer, exc)
            await close(writer)
            return

        if path.startswith("/_gateway/"):
            # `rest` has to travel with the request: read_head returns whatever
            # arrived past the head, and for a small control POST that is the
            # entire body. Dropping it leaves the control handler waiting for
            # bytes that were already consumed, which the client sees as a hang.
            await self.serve_introspection(method, path, headers, rest, reader, writer)
            return

        key = extract_key(headers) or ANONYMOUS_USER
        if key not in self.fleet.users:
            LOG.info("401 %s %s user=%r [rid=%s]", method, path, key, trace.rid)
            await respond(writer, error_response(401))
            return

        try:
            body, rest = await read_body(reader, rest, headers, self.fleet.args.max_body_buffer)
        except (ValueError, OSError) as exc:
            LOG.info("dropped rid=%s from %s reading body: %s", trace.rid, peer, exc)
            await close(writer)
            return

        convo = (
            conversation_key(headers, body, self.fleet.router.key_sources)
            if self.fleet.args.route_by_conversation
            else None
        )
        trace.conversation = convo
        # Resolved once. Discovery and election run on their own timers and may
        # move things while this request is in flight; everything below must
        # keep talking about the same backend, or the inflight count is
        # incremented on one and decremented on another.
        job_id = self.route(convo)
        backend = self.fleet.backends.get(job_id) if job_id else None
        if backend is None:
            LOG.info("503 %s %s user=%s (no backend) [rid=%s]", method, path, key, trace.rid)
            await respond(writer, error_response(503, retry_after=20))
            return
        trace.pool = extra_pool(backend)
        status = "-"
        tried = []
        # Closing the client socket sits in its own finally so that no amount of
        # bookkeeping trouble above can leak the connection.
        try:
            while True:
                # The attempt's own backend, because job_id moves when a retry
                # picks another one and the decrement below must land on the
                # counter the increment did. Getting this wrong leaks inflight
                # on one backend and underflows another, and the router places
                # on those numbers.
                here, here_backend = job_id, backend
                self.fleet.inflight[here] += 1
                try:
                    status = await self.proxy(
                        here_backend, method, path, headers, rest, body, reader, writer, key, trace
                    )
                    break
                # ValueError covers an unusable upstream head: read_head raises
                # it past MAX_HEAD_BYTES and parse_response_head on a malformed
                # one. Both happen before anything is written downstream, so
                # 502 is safe here -- without it the client just sees the
                # socket close with no status at all. Framing failures after
                # the head is out are caught in relay_response, which cannot
                # use this path.
                except ClientAbandoned:
                    # The client gave up waiting -- on this fleet that is the
                    # Codex 120s no-response timeout -- and its retry is
                    # already on its way with the same conversation key. Two
                    # consequences, both handled here and nowhere else:
                    #
                    # The upstream socket is already closed (proxy's finally),
                    # which arms the serving stack's own abort chain: the
                    # disagg proxy cancels its worker requests, and the
                    # worker's 1Hz disconnect poll aborts the engine request,
                    # queued or running. The abandoned request stops holding a
                    # queue slot for a response nobody can receive.
                    #
                    # The pin is dropped -- but only when nothing was
                    # delivered. Zero events is the timeout signature: the
                    # request never got its first byte, which is a statement
                    # about the backend's queue, so the retry must be free to
                    # land elsewhere. A stream that died with events already
                    # delivered is a different animal (mid-stream disconnect),
                    # and re-homing those would shed load the backend was in
                    # fact serving.
                    delivered = trace.tracker.events if trace.tracker else 0
                    if convo is not None and delivered == 0:
                        was = self.fleet.router.unpin(convo)
                        LOG.warning(
                            "client abandoned before first event; unpinned %s (was %s) [%s]",
                            short_convo(convo),
                            was,
                            trace.detail(),
                        )
                    else:
                        LOG.warning(
                            "client abandoned after %d event(s); pin kept [%s]",
                            delivered,
                            trace.detail(),
                        )
                    # 499, nginx's "client closed request". No response is
                    # written: there is no one to write it to.
                    status = "499"
                    break
                except (ConnectionError, OSError, ValueError) as exc:
                    # `side` distinguishes a backend that never answered from
                    # one whose response could not be handed to the client.
                    # Both used to be logged as "upstream failed" and counted
                    # as 502, which blamed the serving job for client-side
                    # disconnects. "upstream" is the justified default HERE and
                    # only here: every untagged failure that can reach this
                    # handler is backend-side (open_connection, the request
                    # head write, or read_head on the upstream reader). All
                    # client writes happen inside relay_response, which handles
                    # them itself.
                    side = error_side(exc, "upstream")
                    LOG.warning(
                        "upstream %s failed side=%s: %s [%s]",
                        here_backend.url,
                        side,
                        exc,
                        trace.detail(),
                    )
                    if side == "upstream" and is_refusal(exc):
                        note_refusal(
                            here_backend,
                            time.time(),
                            self.fleet.args.refusal_window,
                            self.fleet.args.refusals_before_drop,
                        )
                    # Retry only on the side that proves nothing reached the
                    # client. A client-side failure has already had bytes
                    # written to it, so re-sending would put a second response
                    # into the same stream.
                    nxt = None
                    if side == "upstream" and len(tried) < self.fleet.args.retry_upstream:
                        tried.append(here)
                        nxt = self.route(convo, exclude=tried)
                    if nxt is None:
                        status = "502"
                        await respond(writer, error_response(502))
                        break
                    job_id, backend = nxt, self.fleet.backends.get(nxt)
                    if backend is None:
                        # Retired between the route call and here. Nothing was
                        # written downstream, so this is still a clean 502.
                        status = "502"
                        await respond(writer, error_response(502))
                        break
                    LOG.info("retrying on %s after %s failed [%s]", job_id, here, trace.detail())
                    trace.pool = extra_pool(backend)
                finally:
                    self.fleet.inflight[here] -= 1
        finally:
            LOG.info(
                "%s %s %s user=%s backend=%s convo=%s %.1fs%s [%s]",
                status,
                method,
                path,
                key,
                job_id,
                short_convo(convo),
                time.time() - started,
                " retried=%d" % len(tried) if tried else "",
                trace.detail(),
            )
            await close(writer)

    async def serve_introspection(self, method, path, headers, rest, reader, writer):
        if path == "/_gateway/health":
            # Deliberately unauthenticated: whatever watches the gateway from
            # outside has no reason to hold an allowlist entry, and the answer
            # names no user and no request.
            # Over every pool: routing is one pool, so the gateway can serve
            # while any of them has a healthy backend. Without extra pools this
            # is `active`, exactly as it always was.
            active = self.fleet.overall_active()
            payload = {
                "status": "ok" if active is not None else "no_backend",
                # Which deployment is answering, so a caller about to stop a
                # serving job can check it has reached the right gateway.
                "deployment": os.path.basename(os.path.normpath(self.fleet.args.fleet_dir)),
                "active": active,
                "pending": self.fleet.pending[0] if self.fleet.pending else None,
                "uptime_s": round(time.time() - self.fleet.started),
            }
            if self.fleet.pools:
                # Whether each pool's fleet has anything healthy right now.
                payload["pools"] = {
                    name: {"status": "ok" if job else "no_backend", "active": job}
                    for name, job in self.fleet.pool_actives()
                }
            await respond(writer, json_response(payload))
            return
        if path in ("/_gateway/start_server", "/_gateway/stop_server"):
            if method != "POST":
                await respond(writer, error_response(405))
                return
            # These two submit and cancel serving jobs, which makes them
            # lifecycle actions arriving by a route that never consulted the
            # supervisor lock. Harmless while one gateway is running and not
            # harmless at all while two are: see lifecycle_refusal().
            refusal = self.handover.lifecycle_refusal()
            if refusal is not None:
                LOG.warning("refusing %s: %s", path, refusal["error"])
                await respond(writer, json_response(refusal, 409, "Conflict"))
                return
            # The default pool's jobs only: these predate pools, and a
            # `stop_server` releasing jobs another pool's fleet config owns
            # would be a surprise nobody could undo quickly.
            if path.endswith("/start_server"):
                status, payload = await start_server(self.fleet.lifecycle_view())
            else:
                status, payload = await stop_server(self.fleet.lifecycle_view())
            await respond(writer, json_response(payload, status, ERROR_REASONS.get(status, "OK")))
            return
        if path == "/_gateway/ready":
            # Unauthenticated for the same reason /_gateway/health is: the
            # answer names no user and no request. It is also what an outgoing
            # gateway polls on a successor that has not necessarily read the
            # users file yet, and gating it would make an empty allowlist -- a
            # routine state on a fresh deployment -- indistinguishable from a
            # successor that never came up.
            ready, why = self.handover.ready()
            status = 200 if ready else 503
            payload = {"ready": ready, "pid": os.getpid(), "detail": why}
            if self.fleet.pools:
                # What a predecessor checks before handing this process its
                # listener: a pool it serves and this one does not would drop
                # out of routing and supervision. See _wait_ready().
                payload["pools"] = list(self.fleet.pools)
            # The pid is load-bearing, not diagnostic: during a handover two
            # processes share this port, so the poller needs to know which one
            # answered. See successor_ready().
            await respond(writer, json_response(payload, status, ERROR_REASONS.get(status, "OK")))
            return
        if path == "/_gateway/handover":
            # Authenticated like the routing endpoints below, and for a
            # stronger version of the same reason: this one replaces the
            # process every other user's traffic is going through.
            if extract_key(headers) not in self.fleet.users:
                await respond(writer, error_response(401))
                return
            if method == "GET":
                await respond(writer, json_response(self.handover.status()))
                return
            if method != "POST":
                await respond(writer, error_response(405))
                return
            drop = []
            if self.fleet.pools:
                # Only a gateway with extra pools has any to drop, so only it
                # reads the body; without them the POST is handled exactly as
                # it always was. Read leniently: anything that is not a JSON
                # object asks for nothing, and only `drop_pools`, when
                # present, has to be well formed.
                try:
                    body, _ = await read_body(reader, rest, headers, 64 * 1024)
                except (ValueError, OSError):
                    body = None
                drop = (decode_json_object(body) or {}).get("drop_pools", [])
                if not isinstance(drop, list) or not all(isinstance(name, str) for name in drop):
                    await respond(
                        writer,
                        json_response(
                            {
                                "status": "bad_request",
                                "detail": "drop_pools must be a list of pool names",
                            },
                            400,
                            HANDOVER_REASONS[400],
                        ),
                    )
                    return
            status, payload = await self.handover.start(drop_pools=drop)
            reason = HANDOVER_REASONS.get(status, "OK")
            await respond(writer, json_response(payload, status, reason))
            return
        if path in (
            "/_gateway/route",
            "/_gateway/mirror",
            "/_gateway/pin",
            "/_gateway/drain",
            "/_gateway/backend",
            "/_gateway/backend/remove",
        ):
            # Authenticated: these change how every other user's traffic is
            # placed, which is not something an unlisted caller should reach.
            if extract_key(headers) not in self.fleet.users:
                await respond(writer, error_response(401))
                return
            if method != "POST":
                await respond(writer, error_response(405))
                return
            status, payload = await self.control(path, rest, reader, headers)
            await respond(writer, json_response(payload, status, ERROR_REASONS.get(status, "OK")))
            return
        if path == "/_gateway/fleet":
            if extract_key(headers) not in self.fleet.users:
                await respond(writer, error_response(401))
                return
            now = time.time()
            router = self.fleet.router
            accepting = self.fleet.accepting()
            conversations = router.counts(self.fleet.backends)
            mirrors = mirror_table(self.fleet.mirrors)
            backends = {}
            for job_id, b in sorted(self.fleet.backends.items()):
                state = self.fleet.state_for(b.pool)
                entry = {
                    "url": b.url,
                    "healthy": b.healthy,
                    "healthy_for_s": round(now - b.healthy_since) if b.healthy_since else None,
                    "probe_timeouts": b.timeouts,
                    "state": b.state,
                    "ends_at": fmt_time(b.end_time),
                    "ends_in_s": round(b.end_time - now),
                    "last_beat_s": round(now - b.heartbeat, 1),
                    "inflight": self.fleet.inflight.get(job_id, 0),
                    "conversations": conversations.get(job_id, 0),
                    "accepting": job_id in accepting,
                    "superseded": job_id in state.superseded,
                    "revived": self.fleet.revived.get(job_id, (0, 0.0))[0],
                    "draining": job_id in state.draining,
                }
                if self.fleet.pools:
                    # Which fleet directory it registered in -- and so which
                    # fleet config supervises it. Routing does not read it.
                    entry["pool"] = b.pool
                backends[job_id] = entry
            payload = {
                # The same answer /_gateway/health gives; each pool's own
                # election is under "pools".
                "active": self.fleet.overall_active(),
                "pending_successor": self.fleet.pending[0] if self.fleet.pending else None,
                # Which backends are being copied elsewhere, and how those
                # copies have fared. Here rather than in /_gateway/health
                # because a mirror is an operator's business, not a caller's.
                # mirroring values are lists of "host:port" -- one entry per
                # target -- and mirror_targets splits the counters the same
                # way, because totals across two targets answer neither
                # question anyone asks of them.
                "mirroring": mirrors,
                "mirror_stats": dict(self.fleet.mirror_stats),
                "mirror_targets": {t: dict(c) for t, c in self.fleet.mirror_target_stats.items()},
                "backends": backends,
                "routing": {
                    "enabled": self.fleet.args.route_by_conversation,
                    "policy": router.policy,
                    "key_sources": router.key_sources,
                    "manual_pins": router.manual,
                    "paused": sorted(router.paused),
                    "state_file": router.state_path,
                    "pinned": len(router.pins),
                    "hits": router.hits,
                    "misses": router.misses,
                    "rehomed": router.rehomed,
                    "accepting": sorted(accepting),
                    "serving": sorted(self.fleet.serving()),
                },
            }
            if self.fleet.pools:
                payload["pools"] = self.fleet.pool_report(accepting, conversations)
                payload["routing"].update(
                    pool_affinity=router.pool_affinity,
                    conversation_pools=len(router.families),
                    held_for_pool=router.family_waits,
                )
            await respond(writer, json_response(payload))
            return
        await respond(writer, error_response(404))

    def route(self, convo, exclude=()):
        """Choose the backend for this request.

        `exclude` names backends this request has already failed against. They
        are removed from `serving` as well as from the candidates, because a
        pinned conversation is only re-homed when its pin is absent from
        `serving` -- leaving it there would hand the retry straight back to the
        backend that just refused it, which is precisely the loop the retry
        exists to break.
        """
        accepting = self.fleet.accepting()
        serving = self.fleet.serving()
        if exclude:
            accepting = {j: v for j, v in accepting.items() if j not in exclude}
            serving = serving - set(exclude)
        conversations = self.fleet.router.counts(accepting)
        loads = {
            job_id: (conversations.get(job_id, 0), inflight, remaining)
            for job_id, (_, inflight, remaining) in accepting.items()
        }
        # Pools are only told to the router when there is more than one: a
        # single-pool gateway places and records exactly as it always has.
        pool_of = None
        if self.fleet.pools:
            pool_of = {job_id: backend.pool for job_id, backend in self.fleet.backends.items()}
        return self.fleet.router.route(convo, loads, serving, pool_of=pool_of)

    async def control(self, path, rest, reader, headers):
        """Change routing while the gateway keeps running.

        Restarting to change a policy would drop the pin table, so every
        conversation in flight would rebuild the prefix cache it already had.
        Making these changes hot is what keeps "adjust the policy" from meaning
        "throw away everyone's warm cache".
        """
        body, _ = await read_body(reader, rest, headers, 64 * 1024)
        try:
            request = json.loads(body) if body else {}
        except ValueError:
            return 400, {"error": "body is not JSON"}
        if not isinstance(request, dict):
            return 400, {"error": "body must be a JSON object"}
        router = self.fleet.router

        if path.endswith("/mirror"):
            job_id = request.get("job_id")
            if not isinstance(job_id, str) or not job_id:
                return 400, {"error": "job_id is required"}
            if "target" not in request:
                return 400, {"error": "target is required (an addr:port, or null to stop)"}
            target = request["target"]
            if target in (None, "", False):
                # No target names them all. That was the whole meaning of stop
                # when a backend had one mirror, and every existing caller --
                # curl by hand, the dashboard -- says it this way.
                gone = self.fleet.mirrors.pop(job_id, None)
                router.dirty = True
                router.save()
                LOG.info("mirror of %s stopped (was %s)", job_id, gone)
                return 200, {
                    "job_id": job_id,
                    "mirroring": [],
                    "was": ["%s:%d" % t for t in gone] if gone else None,
                    "mirrors": mirror_table(self.fleet.mirrors),
                }
            if not isinstance(target, str):
                return 400, {"error": "target must be a string addr:port"}
            host, _, port = target.rpartition(":")
            if not host or not port.isdigit() or not 0 < int(port) < 65536:
                return 400, {"error": "target must look like host:port", "target": target}
            if request.get("stop"):
                # Stop one of several. `stop` rides beside the target rather
                # than replacing the null convention above, so a caller who has
                # never heard of multiple targets loses nothing.
                targets = self.fleet.mirrors.get(job_id) or []
                pair = (host, int(port))
                if pair not in targets:
                    return 404, {
                        "error": "that target is not mirroring this backend",
                        "job_id": job_id,
                        "target": target,
                        "mirroring": ["%s:%d" % t for t in targets],
                    }
                targets.remove(pair)
                if not targets:
                    self.fleet.mirrors.pop(job_id, None)
                router.dirty = True
                router.save()
                LOG.info(
                    "mirror of %s to %s stopped; %d target(s) remain", job_id, target, len(targets)
                )
                return 200, {
                    "job_id": job_id,
                    "mirroring": ["%s:%d" % t for t in targets],
                    "was": target,
                    "mirrors": mirror_table(self.fleet.mirrors),
                }
            # Both ends are checked before the table is written, because every
            # way this can be wrong is otherwise silent. A mirror of a backend
            # that is not served copies nothing; a mirror to an address that
            # answers nothing burns one real request per copy until the
            # auto-disable budget runs out. That budget is a backstop for a
            # target that dies later, not a way to discover a typo -- it cost
            # 50 requests to find out that a target 404ed everything, which is
            # 50 more than asking once.
            #
            # `force` exists because "not up yet" is a legitimate order to give:
            # staging the mirror before starting the target is reasonable, and
            # refusing it outright would trade one silent failure for a
            # different one. It is opt-in so that the quiet default is the safe
            # one.
            source = self.fleet.backends.get(job_id)
            if source is None:
                return 404, {
                    "error": "no such backend",
                    "backend": job_id,
                    "known": sorted(self.fleet.backends),
                }
            if not isinstance(target, str):
                return 400, {"error": "target must be a string addr:port"}
            host, _, port = target.rpartition(":")
            if not host or not port.isdigit() or not 0 < int(port) < 65536:
                return 400, {"error": "target must look like host:port", "target": target}
            forced = bool(request.get("force"))
            if not source.healthy and not forced:
                return 409, {
                    "error": "backend is not serving; it would mirror nothing",
                    "backend": job_id,
                    "state": source.state,
                    "hint": 'pass "force": true to mirror it anyway',
                }
            reached = await probe_addr(host, int(port), self.fleet.args.probe_timeout)
            if reached != "ok" and not forced:
                return 502, {
                    "error": "target did not answer %s" % PROBE_PATH,
                    "target": target,
                    "probe": reached,
                    "means": {
                        "dead": "nothing is listening there, or it answered "
                        "something other than 200",
                        "timeout": "it accepted the connection but did not "
                        "answer in %.1fs" % self.fleet.args.probe_timeout,
                    }.get(reached, reached),
                    "hint": 'pass "force": true if the target is not up yet',
                }
            # Added beside whatever is already there, never in place of it.
            # Overwriting was the single-target behaviour, and keeping it
            # would make "add the second target" silently end the first
            # experiment -- the same POST that used to mean "start" has to
            # keep meaning "start", not "replace".
            targets = self.fleet.mirrors.setdefault(job_id, [])
            pair = (host, int(port))
            already = pair in targets
            if not already:
                targets.append(pair)
            # Persisted immediately rather than at the next periodic save: the
            # window between the two is exactly when a handover would hand the
            # successor a table without this in it.
            router.dirty = True
            router.save()
            LOG.info(
                "mirroring %s to %s:%s (%d target(s) total, source healthy=%s, "
                "target probe=%s%s); responses are read and discarded",
                job_id,
                host,
                port,
                len(targets),
                source.healthy,
                reached,
                ", forced" if forced else "",
            )
            return 200, {
                "job_id": job_id,
                "mirroring": ["%s:%d" % t for t in targets],
                # Both checks are reported rather than merely implied by the
                # 200, so that a caller who passed force can see what it
                # overrode instead of having to infer it.
                "source": {"healthy": source.healthy, "state": source.state},
                "target": {"probe": reached},
                "forced": forced,
                "already": already,
                "mirrors": mirror_table(self.fleet.mirrors),
                "stats": dict(self.fleet.mirror_stats),
            }

        if path.endswith("/route"):
            changed = {}
            if "policy" in request:
                policy = request["policy"]
                custom = getattr(router, "policies", None)
                if custom is not None:
                    custom.reload()
                if policy not in known_policies(router):
                    return 400, {
                        "error": "unknown policy %r" % policy,
                        "known": list(known_policies(router)),
                    }
                router.policy = policy
                changed["policy"] = policy
            if "enabled" in request:
                self.fleet.args.route_by_conversation = bool(request["enabled"])
                changed["enabled"] = self.fleet.args.route_by_conversation
            if "sticky_ttl" in request:
                try:
                    router.ttl = float(request["sticky_ttl"])
                except (TypeError, ValueError):
                    return 400, {"error": "sticky_ttl must be a number"}
                changed["sticky_ttl"] = router.ttl
            if "key_sources" in request:
                sources = request["key_sources"]
                if not isinstance(sources, list) or not sources:
                    return 400, {"error": "key_sources must be a non-empty list"}
                bad = [
                    x
                    for x in sources
                    if not isinstance(x, str)
                    or not (x.startswith(("header:", "body:")) or x == "prefix")
                ]
                if bad:
                    return 400, {
                        "error": "unusable key sources",
                        "sources": bad,
                        "expected": "header:<name>, body:<a.b.c>, or prefix",
                    }
                router.key_sources = sources
                changed["key_sources"] = sources
            if not changed:
                return 400, {
                    "error": "nothing to change",
                    "accepts": ["policy", "enabled", "sticky_ttl", "key_sources"],
                }
            router.dirty = True
            router.save()
            LOG.info("routing changed: %s", changed)
            return 200, {
                "changed": changed,
                "policy": router.policy,
                "enabled": self.fleet.args.route_by_conversation,
            }

        if path.endswith("/pin"):
            key = request.get("conversation")
            if not isinstance(key, str) or not key:
                return 400, {"error": "conversation is required"}
            job_id = request.get("backend")
            if job_id in (None, "", False):
                removed = router.unpin(key)
                router.save()
                LOG.info("unpinned %s (was %s)", key[:40], removed)
                return 200, {"conversation": key, "unpinned": removed}
            job_id = str(job_id)
            if job_id not in self.fleet.backends:
                return 404, {
                    "error": "no such backend",
                    "backend": job_id,
                    "known": sorted(self.fleet.backends),
                }
            router.pin(key, job_id)
            router.save()
            LOG.info("pinned %s -> %s", key[:40], job_id)
            return 200, {"conversation": key, "backend": job_id}

        if path.endswith("/backend"):
            # Registration is a file in the fleet directory, so this endpoint
            # writes one rather than keeping a second, HTTP-shaped list of
            # backends beside it. Two sources of truth for "what is in the
            # fleet" would disagree within one discovery sweep, and the sweep
            # would win: anything registered only in memory is removed five
            # seconds later for not being in the directory.
            job_id = request.get("job_id")
            url = request.get("url")
            # Which pool's directory the record is written into, and therefore
            # which fleet config supervises the backend. Routing does not care.
            # Absent is the default pool.
            pool = request.get("pool") or DEFAULT_POOL
            if not isinstance(pool, str) or not self.fleet.known_pool(pool):
                return 400, {
                    "error": "no such pool",
                    "pool": pool,
                    "known": self.fleet.pool_names(),
                }
            reg = registration_path(self.fleet.pool_dir(pool), job_id)
            if reg is None:
                return 400, {
                    "error": "job_id must be 1-64 chars of [A-Za-z0-9._-] and start alphanumeric",
                    "job_id": job_id,
                }
            if not isinstance(url, str) or not url.startswith(("http://", "https://")):
                return 400, {"error": "url is required and must be http(s)://", "url": url}
            other = self.fleet.backends.get(job_id)
            if other is not None and other.pool != pool:
                return 409, {
                    "error": "job %s is already registered in pool %s" % (job_id, other.pool),
                    "job_id": job_id,
                }
            record = {
                "job_id": job_id,
                "url": url.rstrip("/"),
                "run_dir": str(request.get("run_dir") or ""),
                "state": str(request.get("state") or "registered by hand"),
                "end_time": 0,
                "heartbeat": time.time(),
                # Nothing will write heartbeats for this one; see discover().
                "manual": True,
            }
            try:
                candidate = Backend(record, pool)
            except (KeyError, ValueError, TypeError, AttributeError) as exc:
                return 400, {"error": "unusable registration: %s" % exc}
            # Probe before writing. fleetctl once decided a deployment was ready
            # by reading a state string, killed the instance it was replacing,
            # and left a serving gap -- the lesson being that only an answer
            # from the address counts as evidence about the address. Pass
            # "probe": false to register something that is still loading.
            verdict = "skipped"
            if request.get("probe", True):
                verdict = await probe(candidate, self.fleet.args.probe_timeout)
                if verdict != "ok":
                    return 503, {
                        "error": "backend did not answer %s" % PROBE_PATH,
                        "verdict": verdict,
                        "url": record["url"],
                        "hint": 'pass "probe": false to register it anyway',
                    }
            existing = job_id in self.fleet.backends
            try:
                tmp = reg + ".tmp"
                with open(tmp, "w") as handle:
                    json.dump(record, handle)
                os.replace(tmp, reg)
            except OSError as exc:
                return 500, {"error": "could not write registration: %s" % exc}
            # Take effect now rather than at the next sweep, so the caller can
            # send the next request to a fleet that already contains this.
            self.fleet.discover()
            LOG.info(
                "%sbackend registered by hand: %s at %s (probe %s)",
                self.fleet.tag_for(pool),
                job_id,
                record["url"],
                verdict,
            )
            reply = {
                "registered": job_id,
                "url": record["url"],
                "probe": verdict,
                "replaced": existing,
                "backends": sorted(self.fleet.backends),
            }
            if self.fleet.pools:
                reply["pool"] = pool
            return 200, reply

        if path.endswith("/backend/remove"):
            job_id = request.get("job_id")
            # The pool it is registered in, unless the caller says otherwise:
            # a deregistration that does not name one must still find a
            # backend that lives in another pool's directory.
            known = self.fleet.backends.get(job_id) if isinstance(job_id, str) else None
            pool = request.get("pool") or (known.pool if known is not None else DEFAULT_POOL)
            if not isinstance(pool, str) or not self.fleet.known_pool(pool):
                return 400, {
                    "error": "no such pool",
                    "pool": pool,
                    "known": self.fleet.pool_names(),
                }
            reg = registration_path(self.fleet.pool_dir(pool), job_id)
            if reg is None:
                return 400, {"error": "job_id is not a usable name", "job_id": job_id}
            if not os.path.exists(reg) and job_id not in self.fleet.backends:
                return 404, {
                    "error": "no such backend",
                    "backend": job_id,
                    "known": sorted(self.fleet.backends),
                }
            # Held out of new placement first. Deleting the file alone works,
            # but every conversation pinned here migrates at once and rebuilds
            # a prefix cache it already had; pausing first lets the ones in
            # flight finish where their cache is. Pass "drain": false to take
            # it out immediately.
            drained = False
            if request.get("drain", True):
                router.paused.add(job_id)
                router.dirty = True
                router.save()
                drained = True
            still = router.counts([job_id]).get(job_id, 0)
            if drained and still and not request.get("force"):
                return 202, {
                    "draining": job_id,
                    "conversations_still_here": still,
                    "message": "held out of new placement; call again with "
                    '"force": true to remove it now, or wait for '
                    "these to finish",
                }
            try:
                if os.path.exists(reg):
                    os.remove(reg)
            except OSError as exc:
                return 500, {"error": "could not remove registration: %s" % exc}
            self.fleet.discover()
            router.paused.discard(job_id)
            router.dirty = True
            router.save()
            LOG.info(
                "backend deregistered by hand: %s (%d conversations were pinned)", job_id, still
            )
            return 200, {
                "removed": job_id,
                "conversations_moved": still,
                "backends": sorted(self.fleet.backends),
            }

        # /drain
        job_id = request.get("job_id")
        if not isinstance(job_id, str) or not job_id:
            return 400, {"error": "job_id is required"}
        if job_id not in self.fleet.backends:
            return 404, {
                "error": "no such backend",
                "backend": job_id,
                "known": sorted(self.fleet.backends),
            }
        accepting = request.get("accepting")
        if not isinstance(accepting, bool):
            return 400, {"error": "accepting must be true or false"}
        if accepting:
            router.paused.discard(job_id)
        else:
            router.paused.add(job_id)
        router.dirty = True
        router.save()
        LOG.info(
            "backend %s %s new conversations",
            job_id,
            "accepts" if accepting else "no longer accepts",
        )
        # Conversations already pinned here keep running: this holds a backend
        # out of new placement, it does not evict anybody.
        return 200, {
            "job_id": job_id,
            "accepting": accepting,
            "paused": sorted(router.paused),
            "conversations_still_here": router.counts([job_id])[job_id],
        }

    async def proxy(self, backend, method, path, headers, rest, body, reader, writer, user, trace):
        timeout = self.fleet.args.upstream_connect_timeout
        try:
            up_reader, up_writer = await asyncio.wait_for(
                asyncio.open_connection(backend.host, backend.port), timeout
            )
        except asyncio.TimeoutError as exc:
            # A backend whose listen queue is full drops the SYN instead of
            # refusing it, and the kernel keeps retrying for about two minutes:
            # past the client's own first-event timeout, so a retry on another
            # backend would only reach a client that had already left. As a
            # ConnectionError it takes the same retry path as a refusal.
            raise ConnectionError(f"upstream connect timed out after {timeout:g}s") from exc
        try:
            up_writer.write(self.upstream_head(backend, method, path, headers, user))
            pump = None
            if body is not None:
                # Already read in full, to identify the conversation. Forward it
                # in one piece; there is nothing left for a pump to carry.
                if body:
                    up_writer.write(body)
                await up_writer.drain()
                # Copy it, if this backend is being mirrored. After the real
                # request is on the wire, so the served request is never made
                # to wait for the copy, and as a task so a slow target cannot
                # hold up the response either. Only the buffered path can be
                # mirrored: a streamed body has already been handed to the
                # pump, and teeing it would mean holding an upload the gateway
                # deliberately refuses to hold.
                # One copy per target, each its own task. tuple() because the
                # list is shared state: an auto-disable between two of these
                # create_task calls mutates it, and iterating the live list
                # would then skip or double a target mid-request.
                for target in tuple(self.fleet.mirrors.get(backend.job_id) or ()):
                    asyncio.create_task(
                        mirror_request(self.fleet, target, method, path, headers, body, user)
                    )
            else:
                # Not buffered, so nothing here understands the body. The pump
                # runs until the client stops sending or the response finishes,
                # which serves content-length and chunked alike.
                if rest:
                    up_writer.write(rest)
                await up_writer.drain()
                # Counted rather than passed over in silence: a mirror that
                # quietly covers some of the traffic and not the rest is a
                # measurement nobody can interpret, and the reason -- chunked,
                # no content-length, or past --max-body-buffer -- is invisible
                # from the far side.
                if backend.job_id in self.fleet.mirrors:
                    self.fleet.mirror_stats["skipped_unbuffered"] += 1
                pump = asyncio.create_task(relay(reader, up_writer, trace))
            try:
                if pump is not None:
                    # A streamed request body: the client socket is in active
                    # use, so its EOF is not a signal -- half of these uploads
                    # end exactly that way.
                    return await self.relay_response(backend, up_reader, writer, trace)

                # Buffered request: the body has been read in full, so the
                # client has nothing left to send and a read on its socket can
                # only resolve when it disconnects. That makes the socket a
                # free abandonment signal for precisely the window where the
                # gateway is otherwise blind -- blocked on the backend's first
                # byte, which on a saturated instance is minutes away. Without
                # this, a client that times out and hangs up leaves its request
                # occupying the backend's queue until that first byte arrives,
                # and every retry it sends queues behind the abandoned one.
                #
                # Caveat, documented rather than solved: a client that
                # half-closes (shutdown(SHUT_WR) after the request, reading
                # the response on the other direction) reads as abandonment
                # here. None of this deployment's clients do that.
                relay_task = asyncio.create_task(
                    self.relay_response(backend, up_reader, writer, trace)
                )
                watcher = asyncio.create_task(reader.read(1))
                try:
                    done, _ = await asyncio.wait(
                        {relay_task, watcher}, return_when=asyncio.FIRST_COMPLETED
                    )
                    # When both finish in the same tick, the response verdict
                    # wins: a client that closed right after reading the whole
                    # response is a completed request, not an abandoned one.
                    if relay_task in done:
                        return relay_task.result()
                    stray = b""
                    try:
                        stray = watcher.result()
                    except OSError:
                        pass  # a reset is the same statement as a close
                    raise ClientAbandoned(
                        "client %s while %s had yet to answer"
                        % ("sent bytes mid-response" if stray else "hung up", backend.url)
                    )
                finally:
                    for task in (relay_task, watcher):
                        if not task.done():
                            task.cancel()
                            try:
                                await task
                            except (asyncio.CancelledError, OSError, ValueError):
                                pass
            finally:
                if pump is not None:
                    pump.cancel()
        finally:
            await close(up_writer)

    def upstream_head(self, backend, method, path, headers, user):
        # The request line is forwarded verbatim, which is what keeps the
        # gateway independent of the Anthropic surface the backend happens to
        # register. /v1/messages, /v1/messages/count_tokens and every Message
        # Batches route reach the backend without this file naming them, and a
        # route added later needs no change here. The only paths the gateway
        # claims for itself are under /_gateway/.
        lines = [
            "%s %s HTTP/1.1" % (method, path),
            "Host: %s:%d" % (backend.host, backend.port),
            # Close framing gives non-SSE responses an unambiguous EOF.
            # SSE responses are decoded and reframed below.
            "Connection: close",
            "Accept-Encoding: identity",
            "X-Gateway-User: %s" % user,
        ]
        for name, value in headers:
            if name.lower() in STRIP_REQUEST_HEADERS:
                continue
            lines.append("%s: %s" % (name, value))
        return ("\r\n".join(lines) + "\r\n\r\n").encode("latin-1")

    async def relay_response(self, backend, up_reader, writer, trace):
        head, rest = await read_head(up_reader)
        if head is None:
            raise ConnectionError("upstream closed before sending a response")
        status_line, headers = parse_response_head(head)
        parts = status_line.split(" ")
        if len(parts) < 2:
            raise ConnectionError("malformed upstream status line")
        status = parts[1]
        # Recorded before anything can fail downstream. The backend's own view
        # of the request is the first thing a post-mortem wants, and it used to
        # be discarded the moment the relay threw.
        trace.upstream_status = status
        content_type = header_value(headers, "content-type") or ""
        content_encoding = header_value(headers, "content-encoding") or ""
        transfer_encoding = header_value(headers, "transfer-encoding") or ""
        encodings = [
            value.strip().lower() for value in transfer_encoding.split(",") if value.strip()
        ]
        # Only SSE is reframed, and only SSE gets a terminal event appended.
        # Batch results come back as application/x-ndjson and count_tokens as
        # application/json, so both take the verbatim path below untouched.
        is_sse = "text/event-stream" in content_type.lower()
        supported_transfer = not encodings or encodings == ["chunked"]

        # Reframing an encoding the gateway does not decode would mix an
        # unencoded injected event into that stream. Accept-Encoding: identity
        # prevents content encoding for the normal server; retain raw relay as a
        # safe fallback for encoded bodies or unknown transfer codings.
        if not is_sse or content_encoding.lower() not in ("", "identity") or not supported_transfer:
            await write_client(writer, head + b"\r\n\r\n" + rest)
            relayed = len(rest)
            # The head is on the client's socket from here on, so a failure can
            # be neither retried nor answered with a 502: both put a second
            # status line into a response the client is partway through. A
            # body that is not SSE has no terminal event to append either, so
            # the ending left is closing the connection short of the length or
            # final chunk the head framed, which the client reads as a failure.
            try:
                while True:
                    chunk = await read_upstream(up_reader, RELAY_CHUNK)
                    if not chunk:
                        return status
                    await write_client(writer, chunk)
                    relayed += len(chunk)
            except OSError as exc:
                side = error_side(exc)
                if side == "client_write":
                    LOG.warning(
                        "could not deliver body side=%s after %d byte(s): %s [%s]",
                        side,
                        relayed,
                        exc,
                        trace.detail(),
                    )
                    return status + "?"
                LOG.warning(
                    "upstream %s failed mid-body side=%s after %d byte(s); "
                    "closing the client connection: %s [%s]",
                    backend.url,
                    side,
                    relayed,
                    exc,
                    trace.detail(),
                )
                await close(writer)
                return status + "!"

        await write_client(writer, rewrite_sse_head(status_line, headers))
        source = BufferedUpstream(up_reader, rest)
        tracker = SseTracker()
        trace.tracker = tracker
        # The head is already on the wire, so a framing failure from here on
        # cannot become a 502: writing error_response would put a second HTTP
        # head inside a response the client is already reading. Degrade to the
        # truncated-stream path instead and let the terminal-event injection
        # below give the client a valid end to the stream.
        clean_end = False
        try:
            if "chunked" in encodings:
                clean_end = await relay_chunked_sse(source, writer, tracker)
            else:
                content_length = header_value(headers, "content-length")
                if content_length is None:
                    clean_end = await relay_close_delimited_sse(source, writer, tracker)
                else:
                    try:
                        length = int(content_length)
                    except ValueError:
                        length = -1
                    if length < 0:
                        clean_end = False
                    else:
                        clean_end = await relay_sized_sse(source, writer, tracker, length)
        # The response head is already on the wire, so a failure from here on
        # cannot become a 502 -- that would put a second HTTP head inside a
        # body the client is reading. OSError covers an upstream that dies
        # mid-stream, which is the common case during a handover. Both degrade
        # to the truncated path so the terminal event below closes the stream.
        #
        # side= is the whole point of the tag: this block reads the backend AND
        # writes the client, so the old unconditional "upstream framing failed"
        # was a guess that happened to be wrong every time the client was the
        # end that left.
        except (ValueError, OSError) as exc:
            LOG.warning(
                "sse relay stopped side=%s: %s [%s]",
                error_side(exc),
                exc,
                trace.detail(),
            )
            LOG.debug("last data line before failure: %s", tracker.preview())
            clean_end = False

        # Deliberately its own try. Everything below writes to the client, so
        # it can raise -- and it used to raise straight past this function into
        # the caller's handler, which appended exactly the second HTTP head the
        # block above exists to prevent. That went unnoticed only because the
        # client socket was already broken in every observed case, so the stray
        # 502 head was discarded by the kernel instead of corrupting a stream
        # the client was still reading. It also mislabelled the request 502
        # when the backend had in fact answered it, which made the access log
        # blame the serving job for client-side disconnects.
        try:
            if not tracker.terminal:
                ending = "clean end" if clean_end else "truncated upstream framing"
                LOG.warning(
                    "stream reached %s without message_stop or [DONE]; injecting error [%s]",
                    ending,
                    trace.detail(),
                )
                await emit_sse_payload(writer, tracker, SSE_ROTATED)
                status += "!"
            await write_client(writer, b"0\r\n\r\n")
        except (ValueError, OSError) as exc:
            # The client is gone; there is nobody left to tell. Record that the
            # ending never landed and let the caller log a delivered-partially
            # status rather than a backend failure.
            LOG.warning(
                "could not deliver stream ending side=%s: %s [%s]",
                error_side(exc),
                exc,
                trace.detail(),
            )
            status += "?"
        return status


async def relay(reader, writer, trace=None):
    """Pump the client's request body into the backend."""
    try:
        while True:
            chunk = await reader.read(RELAY_CHUNK)
            if not chunk:
                break
            writer.write(chunk)
            await writer.drain()
    except asyncio.CancelledError:
        # Routine: the response finished first and proxy() cancels the pump.
        pass
    except (ConnectionError, OSError) as exc:
        # Not routine, and previously invisible -- this handler was a bare
        # `pass`. A client that dies mid-upload leaves the backend holding a
        # truncated request body, and the backend's reaction to that reads like
        # a server-side bug with nothing in the log to contradict it.
        if trace is not None:
            trace.request_body_error = "%s: %s" % (type(exc).__name__, exc)
        LOG.debug("request body pump failed: %s", exc)


async def respond(writer, payload):
    try:
        writer.write(payload)
        await writer.drain()
    except (ConnectionError, OSError):
        pass
    await close(writer)


async def close(writer):
    try:
        writer.close()
        await writer.wait_closed()
    except (ConnectionError, OSError):
        pass


# ---------------------------------------------------------------------------
# Background loops
# ---------------------------------------------------------------------------
PROBE_PATH = "/v1/models"


async def probe(backend, timeout):
    """GET the probe path, classified into three outcomes rather than a boolean.

    "dead" and "timeout" look the same to a boolean probe but mean opposite
    things. A refused connection or an unresolvable host says the process is
    gone -- unambiguous, act at once. A timeout usually says the server is too
    busy to answer a health check, and a server that busy is normally still
    generating tokens fine; taking it out of rotation would turn "slow" into
    "503" with nowhere better to send the traffic.

    The path is `/v1/models` rather than `/health`, and the difference is not
    cosmetic. `/health` answers 200 from anything at that address: on these
    GPU nodes a container registry listens on one of the ports a fleet might
    pick, and it returns a cheerful 200 OK to /health while 404ing every real
    request. The gateway elected it, called it healthy for half an hour, and
    404ed the traffic it sent there -- while the actual worker had died on
    "Address already in use" and nothing said so.

    `/v1/models` cannot be answered by accident. It is 404 until the OpenAI
    routes are mounted, which happens only once the deployment is genuinely
    able to serve, so it tests the two things that matter together: something
    is listening, and that something is ours and ready. It costs a static list
    and no engine work.
    """
    return await probe_addr(backend.host, backend.port, timeout)


async def probe_addr(host, port, timeout):
    """probe(), addressed directly rather than through a Backend record.

    Split out for the mirror control endpoint, which has to ask the same
    question about something that is not a fleet member and therefore has no
    Backend. Sharing the body rather than writing a second check keeps one
    definition of "reachable and ours" -- including the reason probe() uses
    PROBE_PATH and not /health, which applies to a mirror target unchanged and
    for the same reason: a registry container answering 200 to /health on the
    port a fleet member might have used is exactly the thing that looks
    healthy and serves nothing.
    """
    writer = None
    try:
        reader, writer = await asyncio.wait_for(asyncio.open_connection(host, port), timeout)
        writer.write(
            b"GET %s HTTP/1.1\r\nHost: %s\r\nConnection: close\r\n\r\n"
            % (PROBE_PATH.encode("latin-1"), host.encode("latin-1"))
        )
        await writer.drain()
        line = await asyncio.wait_for(reader.readline(), timeout)
        return "ok" if b" 200 " in line else "dead"
    except asyncio.TimeoutError:
        return "timeout"
    except (ConnectionError, OSError):
        return "dead"
    finally:
        if writer is not None:
            await close(writer)


def is_refusal(exc):
    """True when nothing was listening, as opposed to anything else.

    ECONNREFUSED only. A timeout is deliberately not counted: a backend too
    busy to accept a connection is usually still generating tokens, and that
    is the case the probe's 20-timeout threshold exists to protect. This adds
    a fast path for the unambiguous death, not a second way to evict a loaded
    backend.

    Unwraps SideError, which tags the side but keeps the original as `cause`.
    """
    inner = getattr(exc, "cause", exc)
    return (
        isinstance(inner, ConnectionRefusedError)
        or getattr(inner, "errno", None) == errno.ECONNREFUSED
    )


def note_refusal(backend, now, window, threshold):
    """Count a refusal against a backend; drop it once they cluster.

    Thresholded rather than acted on at the first one, because ECONNREFUSED is
    also what a healthy backend returns when its listen backlog overflows --
    evicting on a single refusal would turn a load spike into an outage. The
    gap between the two cases is wide enough that the threshold does not need
    to be delicate: a real death produced 192 refusals in a minute and 551 in
    four seconds, while backlog noise is a handful.

    Returns True if this call is what made the backend unhealthy.
    """
    cutoff = now - window
    backend.refusals = [t for t in backend.refusals if t > cutoff]
    backend.refusals.append(now)
    if not backend.healthy or len(backend.refusals) < threshold:
        return False
    LOG.warning(
        "backend %s refused %d connections in %.0fs; dropping it without waiting for the probe",
        backend.job_id,
        len(backend.refusals),
        window,
    )
    backend.healthy = False
    backend.healthy_since = 0.0
    backend.refusals = []
    return True


def apply_probe(backend, result, unhealthy_after):
    if result == "ok":
        backend.timeouts = 0
        # A backend that answers the probe is serving, whatever the request
        # stream said a moment ago: the refusals were either transient or
        # belong to an instance that has since come back.
        backend.refusals = []
        if not backend.healthy:
            backend.healthy = True
            backend.healthy_since = time.time()
            LOG.info("backend %s healthy", backend.job_id)
        return
    if result == "dead":
        if backend.healthy:
            LOG.warning("backend %s unreachable; dropping it now", backend.job_id)
        backend.timeouts = 0
        backend.healthy = False
        backend.healthy_since = 0.0
        return
    backend.timeouts += 1
    if backend.healthy and backend.timeouts >= unhealthy_after:
        LOG.warning(
            "backend %s timed out %d times in a row; marking unhealthy",
            backend.job_id,
            backend.timeouts,
        )
        backend.healthy = False
        backend.healthy_since = 0.0
    elif backend.healthy:
        LOG.info(
            "backend %s health probe timed out (%d/%d)",
            backend.job_id,
            backend.timeouts,
            unhealthy_after,
        )


async def in_thread(fn, *args):
    """Run blocking work on the default executor.

    asyncio.to_thread reads better but is 3.9+, and this file runs under
    whatever python3 the CPU node happens to have -- deliberately, so the
    gateway does not depend on the serving container it outlives.
    """
    return await asyncio.get_running_loop().run_in_executor(None, fn, *args)


def read_fleet_state(fleet):
    """The blocking half of one discovery sweep. Runs on a thread.

    reload_users() belongs here because it ends in a single rebind of
    fleet.users, which a reader either sees or does not; the registration
    records are returned rather than applied for the opposite reason.
    """
    fleet.reload_users()
    return fleet.read_registrations()


async def discovery_loop(fleet):
    while True:
        try:
            # Reads on a thread, mutation on the loop. The fleet directory is
            # on the shared filesystem every serving job writes to, and a stall
            # there used to land directly on the request path: 46 backends cost
            # ~45 ms a sweep there against ~1 ms locally.
            records = await in_thread(read_fleet_state, fleet)
            fleet.discover(records)
            if fleet.router.policies is not None:
                fleet.router.policies.reload()
        except Exception:
            LOG.exception("discovery failed")
        await asyncio.sleep(fleet.args.discover_interval)


# Latched, so a misconfiguration says so once instead of once per 30s forever.
_UNLOCKED_SAVE_WARNED = False


def may_save_router_state(fleet):
    """True when this process is the one allowed to persist the pin table.

    Only the supervisor-lock holder writes router state. A handover has two
    gateways alive at once, each holding a full pin table, and every save ends
    in an os.replace onto one path -- so whichever writes last wins and the
    other one's conversations lose their affinity with nothing logged and
    nothing to notice. Hanging the write off the lock makes "who owns the state
    file" the same question as "who owns the fleet", and that one is already
    answered exactly once by construction.
    """
    global _UNLOCKED_SAVE_WARNED
    lock = getattr(fleet, "supervisor_lock", None)
    if lock is None:
        lock = getattr(fleet, "lock", None)
    if lock is None:
        # No lock on the fleet means one gateway, which is how this file ran
        # for its whole life before handover. Keep saving -- silently dropping
        # persistence would be a worse regression than the race the lock
        # guards -- but say it once, because the other way to reach this line
        # is a lock that got renamed out from under the check.
        if not _UNLOCKED_SAVE_WARNED:
            _UNLOCKED_SAVE_WARNED = True
            LOG.warning(
                "no supervisor lock on the fleet; saving router state unguarded, "
                "which is last-writer-wins if two gateways ever overlap"
            )
        return True
    return bool(lock.held)


async def router_state_loop(fleet):
    """Snapshot the pin table periodically rather than on every request.

    Saving inline would put a filesystem write on the request path for a
    benefit that is entirely about surviving a restart. A periodic flush costs
    at most one interval's worth of pins, and losing those only means a few
    conversations re-home.
    """
    while True:
        await asyncio.sleep(30)
        try:
            if not may_save_router_state(fleet):
                # A gateway without the lock routes and pins exactly as usual,
                # it just does not own the file. `dirty` is left standing
                # rather than cleared, so everything pinned while waiting for
                # the lock lands in the first snapshot taken after it arrives.
                continue
            # Snapshot on the loop, write on a thread: at 20000 pins the write
            # is a 43 ms json.dump onto the shared filesystem, and the snapshot
            # is what makes it safe to leave the loop.
            state = fleet.router.snapshot()
            if state is not None:
                await in_thread(fleet.router.write_snapshot, state)
        except Exception:
            LOG.exception("router state save failed")


async def health_loop(fleet):
    while True:
        try:
            backends = list(fleet.backends.values())
            if backends:
                results = await asyncio.gather(
                    *[probe(b, fleet.args.probe_timeout) for b in backends],
                    return_exceptions=True,
                )
                for backend, result in zip(backends, results):
                    if not isinstance(result, str):
                        result = "timeout"
                    apply_probe(backend, result, fleet.args.unhealthy_after)
            fleet.elect()
        except Exception:
            LOG.exception("health loop failed")
        await asyncio.sleep(fleet.args.health_interval)


async def initial_health_pass(fleet):
    """One probe sweep, before the listener exists.

    Without SO_REUSEPORT this did not matter: a process that had not bound yet
    was simply not reachable, and by the time anything could ask it a question
    health_loop had long since run. With SO_REUSEPORT the kernel starts handing
    this process a share of new connections the instant it binds, so whatever it
    does not know at that moment is answered *wrongly* rather than late -- every
    backend still healthy=False, so the first requests get 503 "no backend" from
    a process sitting next to a perfectly healthy fleet. Measured over five
    handovers: 16 non-200s in 51408 requests, every one of them the successor's
    first request.

    Bounded, and its outcome is deliberately ignored. A first-ever gateway, or
    one whose fleet is genuinely down, has nothing to find here and must still
    reach its listener and answer 503 from there; refusing to listen until
    something is healthy would turn a degraded deployment into an unreachable
    one. The bound is belt and braces -- probe() already bounds itself to two
    probe_timeouts and they all run concurrently -- so it costs nothing on the
    path where the probes answer.
    """
    backends = list(fleet.backends.values())
    if not backends:
        return
    deadline = 2 * fleet.args.probe_timeout + 1.0
    try:
        results = await asyncio.wait_for(
            asyncio.gather(
                *[probe(b, fleet.args.probe_timeout) for b in backends],
                return_exceptions=True,
            ),
            deadline,
        )
    except asyncio.TimeoutError:
        LOG.warning("initial health pass unfinished after %.1fs; listening anyway", deadline)
        return
    for backend, result in zip(backends, results):
        if not isinstance(result, str):
            result = "timeout"
        apply_probe(backend, result, fleet.args.unhealthy_after)
    fleet.elect()


async def run_serve_sh(fleet, *serve_args):
    try:
        proc = await asyncio.create_subprocess_exec(
            fleet.args.serve_sh,
            *serve_args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
    except OSError as exc:
        # --no-relay skips the start-up check that this path exists, so a
        # missing or unexecutable serve.sh first shows up here. Returning like
        # any other failure keeps it local: raising would abandon the rest of
        # the sweep, and every backend after this one would go unexamined
        # while having already spent its retry budget.
        LOG.error("cannot run %s: %s", fleet.args.serve_sh, exc)
        return None, ""
    out, _ = await proc.communicate()
    return proc.returncode, out.decode(errors="replace").strip()


async def run_fleetctl(fleet, *fleetctl_args):
    """Invoke fleetctl, the only thing that knows the configured fleet."""
    try:
        proc = await asyncio.create_subprocess_exec(
            fleet.args.fleetctl,
            "--config",
            fleet.args.fleet_config,
            *fleetctl_args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
    except OSError as exc:
        LOG.error("cannot run %s: %s", fleet.args.fleetctl, exc)
        return None, ""
    out, _ = await proc.communicate()
    return proc.returncode, out.decode(errors="replace").strip()


SLURM_WRAPPER = ""


async def run_slurm_command(*command):
    """Ask the scheduler something, wherever the scheduler happens to be.

    `SLURM_WRAPPER` is empty when the gateway runs on the cluster, which is the
    case this started as. Off-cluster there is no squeue to exec, and the
    failure is silent in the worst way: the query returns "cannot tell", and
    every caller correctly declines to act on an answer it did not get.
    """
    if SLURM_WRAPPER:
        command = (SLURM_WRAPPER,) + command
    try:
        proc = await asyncio.create_subprocess_exec(
            *command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT
        )
    except OSError as exc:
        LOG.error("cannot run %s: %s", command[0], exc)
        return None, ""
    out, _ = await proc.communicate()
    return proc.returncode, out.decode(errors="replace").strip()


async def slurm_job_status(job_id):
    """Return (state, reason), ("GONE", ""), or None on query failure."""
    code, out = await run_slurm_command("squeue", "-h", "-j", job_id, "-o", "%T|%r")
    if code is not None and code != 0 and "invalid job id" in out.lower():
        return "GONE", ""
    if code is None or code != 0:
        LOG.warning("cannot query successor %s: %s", job_id, out)
        return None
    if not out:
        return "GONE", ""
    state, _, reason = out.splitlines()[0].partition("|")
    return state.strip().upper(), reason.strip()


# job id -> the PreemptTime last reported for it, so each notice is logged once.
_PREEMPTION_NOTED = {}

_SLURM_UNSET = ("None", "Unknown", "N/A", "(null)")


async def slurm_preemption_notice(job_id):
    """When Slurm signalled the job's current run for preemption, or None.

    A preempted job is not killed at once. Slurm sends SIGTERM, which ends the
    serving attempt, and takes the allocation back when the grace period runs
    out, so the allocation looks retained and idle in between. PreemptTime is
    when the signal went out; a notice earlier than the run's StartTime belongs
    to a run before a requeue, not to this one. None also when the scheduler
    cannot be asked, which leaves the caller to act as it would without the
    question.
    """
    code, out = await run_slurm_command("scontrol", "show", "job", str(job_id))
    if code != 0:
        return None
    fields = dict(re.findall(r"\b(PreemptTime|StartTime)=(\S+)", out))
    notice = fields.get("PreemptTime")
    if not notice or notice in _SLURM_UNSET:
        return None
    start = fields.get("StartTime")
    if start and start not in _SLURM_UNSET and notice < start:
        return None
    return notice


async def find_untracked_job(fleet):
    """Newest queued job for this deployment that the fleet has no handle on."""
    name = os.path.basename(os.path.normpath(fleet.args.fleet_dir))
    code, out = await run_slurm_command(
        "squeue", "--me", "--noheader", "--name", name, "--format", "%i", "--sort", "-V"
    )
    if code != 0:
        LOG.error("cannot reconcile %s by job name (rc=%s): %s", name, code, out)
        return None
    known = set(fleet.backends) | ({fleet.pending[0]} if fleet.pending else set())
    return next((job_id for job_id in out.split() if job_id not in known), None)


async def submit_successor(fleet, now, label):
    fleet.last_submit = now
    code, out = await run_serve_sh(fleet, "submit", "--yaml", fleet.args.yaml, "--label", label)
    match = re.search(r"Submitted batch job (\d+)", out)
    if code == 0 and match:
        fleet.pending = (match.group(1), now)
        LOG.info("successor submitted: job %s", match.group(1))
        return True
    LOG.error("submit failed (rc=%d): %s", code, out)
    job_id = await find_untracked_job(fleet)
    if job_id is None:
        return False
    fleet.pending = (job_id, now)
    LOG.warning("adopted %s: submit reported failure but the job exists", job_id)
    return True


async def release_job(fleet, job_id, run_dir):
    """Stop one serving job. `quit` needs a live control dir; else scancel."""
    if run_dir:
        code, out = await run_serve_sh(fleet, "quit", run_dir)
        if code == 0:
            LOG.info("released %s via quit (%s)", job_id, run_dir)
            return True
        LOG.warning("quit %s failed (rc=%s): %s; falling back to scancel", job_id, code, out)
    code, out = await run_slurm_command("scancel", job_id)
    if code == 0:
        LOG.info("released %s via scancel", job_id)
        return True
    LOG.error("scancel %s failed (rc=%s): %s", job_id, code, out)
    return False


async def start_server(fleet):
    """POST /_gateway/start_server: get a serving job going if there is none."""
    fleet.stopped = False
    if fleet.backends or fleet.pending:
        # Name the job being adopted: the one still loading if there is one,
        # else the one taking traffic, else whatever registered.
        if fleet.pending:
            job_id = fleet.pending[0]
        else:
            job_id = fleet.active or sorted(fleet.backends)[0]
        LOG.info("start_server: a serving job is already present (%s)", job_id)
        return 200, {"action": "adopted", "job_id": job_id, "active": fleet.active}
    if await submit_successor(fleet, time.time(), "start"):
        return 200, {
            "action": "submitted",
            "job_id": fleet.pending[0],
            "active": None,
            "message": "queued; poll /_gateway/health until status is ok",
        }
    return 503, {
        "action": "submit_failed",
        "job_id": None,
        "active": None,
        "message": "sbatch refused the job; the gateway log has its output",
    }


async def stop_server(fleet):
    """POST /_gateway/stop_server: release every serving job, keep the gateway."""
    targets = []
    if fleet.pending and fleet.pending[0] not in fleet.backends:
        # Submitted but never registered, so scancel is the only handle on it.
        targets.append((fleet.pending[0], ""))
    targets.extend((job_id, b.run_dir) for job_id, b in sorted(fleet.backends.items()))
    fleet.pending = None
    # The supervisor resubmits whenever a fleet that once had a backend goes
    # empty, which is exactly what is about to happen. Recorded in its own flag
    # rather than by clearing ever_active: the job being released stays
    # discoverable until its heartbeat goes stale, so it can win one more
    # election on the way down -- and elect() sets ever_active back to True.
    fleet.stopped = True

    inflight = sum(fleet.inflight.get(job_id, 0) for job_id, _ in targets)
    fleet.draining.clear()
    fleet.superseded.clear()
    # Stop routing before the servers go down: a clean 503 beats a truncated
    # response mid-stream.
    fleet.active = None

    outcomes = await asyncio.gather(
        *[release_job(fleet, job_id, run_dir) for job_id, run_dir in targets],
        return_exceptions=True,
    )
    released, failed = [], []
    for (job_id, _), outcome in zip(targets, outcomes):
        if isinstance(outcome, BaseException):
            LOG.error("releasing %s raised: %r", job_id, outcome)
        (released if outcome is True else failed).append(job_id)

    LOG.info(
        "stop_server: released %s (interrupted %d request(s))%s",
        ", ".join(released) or "nothing",
        inflight,
        "; FAILED on %s" % ", ".join(failed) if failed else "",
    )
    message = (
        "could not stop %s; check it with squeue and scancel it by hand" % ", ".join(failed)
        if failed
        else "POST /_gateway/start_server to bring a serving job back"
    )
    return (502 if failed else 200), {
        "action": "stopped",
        "released": released,
        "failed": failed,
        "interrupted_requests": inflight,
        "message": message,
    }


def attempt_failed(backend):
    """True when a deployment has declared its own server dead.

    serve.sh writes this state itself and keeps the allocation, waiting for a
    `restart` control file that nothing writes on its own. Read from the state
    rather than inferred from probes: a failed probe can be a network blip,
    while this string is the deployment saying so.
    """
    return " exited with status " in backend.state or backend.state.startswith("stopped;")


async def supervise_pending(fleet, now):
    """Keep a submitted successor tracked until it is actually healthy."""
    if not fleet.pending:
        return
    job_id, submitted_at = fleet.pending
    backend = fleet.backends.get(job_id)
    if backend is not None:
        if not backend.healthy and attempt_failed(backend):
            LOG.warning("successor %s failed to start; restarting its retained allocation", job_id)
            code, out = await run_serve_sh(fleet, "restart", backend.run_dir)
            if code == 0:
                fleet.pending = (job_id, now)
            else:
                LOG.error("restart %s failed (rc=%d): %s", job_id, code, out)
                cancel_code, cancel_out = await run_slurm_command("scancel", job_id)
                if cancel_code == 0:
                    fleet.pending = None
                else:
                    LOG.error("scancel %s failed (rc=%s): %s", job_id, cancel_code, cancel_out)
        return

    # sbatch can take a moment to publish a new job into squeue. Do not mistake
    # that visibility gap for an immediate terminal failure.
    if now - submitted_at < PENDING_VISIBILITY_GRACE:
        return
    status = await slurm_job_status(job_id)
    if status is None:
        return
    state, reason = status
    terminal = {
        "BOOT_FAIL",
        "CANCELLED",
        "DEADLINE",
        "FAILED",
        "NODE_FAIL",
        "OUT_OF_MEMORY",
        "TIMEOUT",
    }
    if state == "GONE" or state in terminal:
        LOG.warning("successor %s is no longer runnable (%s); retrying", job_id, state)
        fleet.pending = None
        return

    # Slurm reports RUNNING as soon as it hands the node over, which is before
    # the prolog finishes and therefore before the batch script exists to
    # register anything. Restart the grace clock instead of reading that as a
    # launcher failure: node setup here regularly outlasts
    # PENDING_VISIBILITY_GRACE, and cancelling mid-prolog throws away a
    # successor that was about to come up -- repeatedly, since every retry
    # meets the same prolog. A prolog cannot stall forever, because Slurm caps
    # it with PrologEpilogTimeout and fails the job into a terminal state the
    # branch above already handles.
    if state == "RUNNING" and "prolog" in reason.lower():
        LOG.info("successor %s is still in prolog; deferring its failure check", job_id)
        fleet.pending = (job_id, now)
        return

    if state == "RUNNING" and job_id not in fleet.seen_running:
        fleet.seen_running.add(job_id)
        fleet.pending = (job_id, now)
        return

    # A RUNNING job reaches its serving command and registers before it starts
    # loading the model. If it remains invisible here, the launcher failed
    # before that point. Held jobs cannot make progress either. Cancel before
    # retrying so a delayed job cannot later appear as a duplicate successor.
    should_cancel = state == "RUNNING" or "held" in reason.lower()
    if should_cancel:
        LOG.warning(
            "successor %s is %s without registering (%s); cancelling and retrying",
            job_id,
            state,
            reason or "no reason",
        )
        code, out = await run_slurm_command("scancel", job_id)
        if code == 0:
            fleet.pending = None
        else:
            LOG.error("scancel %s failed (rc=%s): %s", job_id, code, out)


# How often a gateway that does not hold the supervisor lock asks for it again.
# Far shorter than --supervisor-interval on purpose: this poll is the entire
# takeover mechanism, so it decides how long the fleet goes unsupervised across
# a handover, and what waits on it is a dead deployment sitting on eight idle
# nodes. It costs one open() and one flock() per second, and only while the
# lock is *not* held -- a gateway that is supervising sweeps at
# --supervisor-interval as before.
SUPERVISOR_LOCK_POLL = 1.0


class SupervisorLock:
    """The exclusive right to take lifecycle actions on the fleet.

    A handover runs two gateways at once and both of them run `supervisor_loop`.
    Nothing that loop does is idempotent: two supervisors would each submit an
    eight-node successor for the same dead instance, or both drop a `restart`
    control file into one run dir and race the controller reading it. So
    `supervise` happens only while this is held. Discovery, probing and routing
    are read-only and deliberately keep running in both processes -- the
    successor has to already have a current fleet view at the moment it is asked
    to take traffic, not start building one then.

    `fcntl.flock`, and deliberately **not** a lease with a TTL or a heartbeat.
    The point of a kernel lock is that the kernel is what releases it: a gateway
    that segfaults, is SIGKILLed, or is torn down with its job drops this the
    instant its last fd closes -- no cleanup path that has to have run, no clock
    that has to be right, no operator deciding whether a record is stale. A TTL
    reintroduces the exact production failure this fleet has already paid for
    once, a state file insisting a controller was alive after it was gone, and
    adds a window where the lease has expired but its holder has not noticed,
    which is precisely the window in which two supervisors act at the same time.
    Handover is same-node by construction, so this never has to mean anything
    across hosts and nothing here leans on the shared filesystem's lock manager.

    The file's *contents* are diagnostics and nothing more: who holds it, where,
    since when -- so a handover that never completes can be read out of a log
    instead of guessed at. They are only ever reported after flock has already
    proved that somebody holds it, and nothing decides anything from them.
    Reading a pid out of this file and acting on it is the same trap as the TTL.
    """

    def __init__(self, path):
        self.path = path
        # The flock lives on the open file description, so the fd and the lock
        # are one fact with one representation. There is no second flag to fall
        # out of sync with the kernel.
        self._fd = None
        # Set once this process has handed supervision over for good (handover
        # step 7). Both gateways poll for the lock a second apart, so without
        # this the one that just stepped down wins about half of those races and
        # takes back the authority it had just given away.
        self.retired = False
        # Last holder reported by acquire(), so a successor waiting minutes for
        # its predecessor logs the name of what it is waiting for once instead
        # of once a second.
        self._reported = None
        # Whether each active `with` block was the thing that took the lock, so
        # a nested one cannot release a lock it did not acquire.
        self._entered = []

    @property
    def held(self):
        return self._fd is not None

    def acquire(self):
        """Take the lock if it is free. Never blocks and never raises.

        Blocking would park the supervisor behind whatever the other gateway is
        doing, which for most of a handover is "still serving traffic". Raising
        would put a traceback in the log every second of a normal takeover.
        """
        if self._fd is not None:
            return True
        if self.retired:
            return False
        try:
            # O_CLOEXEC explicitly, even though CPython has defaulted to
            # non-inheritable fds since 3.4. The successor gateway is spawned by
            # this process, and an inherited fd would carry this lock into it on
            # the same open file description -- so the lock would outlive the
            # death of its holder, which is the one thing the kernel is here to
            # prevent, and the successor would end up blocked on a lock it was
            # itself holding open.
            fd = os.open(self.path, os.O_RDWR | os.O_CREAT | os.O_CLOEXEC, 0o644)
        except OSError as exc:
            # Not fatal and not silent: routing is unaffected, but nothing will
            # ever supervise until this works, and that has to be visible.
            LOG.warning("supervisor lock %s cannot be opened: %s", self.path, exc)
            return False
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self._report_contention(fd)
            os.close(fd)
            return False
        self._fd = fd
        self._reported = None
        LOG.info("supervisor lock acquired: %s by %s; supervising", self.path, self._stamp())
        return True

    def release(self, retire=True):
        """Give the lock up. Idempotent, and never raises.

        Releasing is a stand-down rather than a pause, so `retire` defaults on.
        Every explicit caller is on its way out -- handover step 7, or a
        graceful shutdown -- and both gateways poll for this lock a second
        apart, so a predecessor that released without retiring wins about half
        of those races and takes back the authority it had just handed over.
        That costs more than an untidy log: the successor is the one taking new
        traffic by then, but the predecessor would still be the lock holder, and
        the lock holder is what decides who persists the router state.

        The one release that is not a stand-down is `__exit__`'s, which passes
        retire=False because a `with` block ending is not the process ending.
        """
        if retire:
            self.retired = True
        fd = self._fd
        if fd is None:
            return False
        self._fd = None
        self._reported = None
        try:
            # Clear the record before dropping the lock, never after: after, the
            # thing being wiped would be the *next* holder's record. Empty
            # therefore means "released cleanly" and a leftover record means the
            # holder died still holding it -- which flock has already forgiven,
            # and which is the only difference between the two worth seeing.
            os.ftruncate(fd, 0)
            fcntl.flock(fd, fcntl.LOCK_UN)
        except OSError as exc:
            LOG.warning("supervisor lock %s: unlock failed: %s (closing anyway)", self.path, exc)
        finally:
            # The close is the release that actually counts: dropping the last
            # fd on the open file description frees the flock whatever went
            # wrong above, and it is the same path the kernel takes on our
            # behalf when the process dies.
            try:
                os.close(fd)
            except OSError:
                pass
        LOG.info(
            "supervisor lock released: %s by pid %d%s",
            self.path,
            os.getpid(),
            "; retired, it will not be taken back" if self.retired else "",
        )
        return True

    def holder(self):
        """How the current holder describes itself, or "" if nobody has said.

        Diagnostics only -- see the class docstring. A caller that needs to know
        whether the lock is free has to ask for the lock, because that is the
        only answer still true by the time it is acted on.
        """
        try:
            fd = os.open(self.path, os.O_RDONLY | os.O_CLOEXEC)
        except OSError:
            return ""
        try:
            return self._record(fd)
        finally:
            os.close(fd)

    def _stamp(self):
        """Write this process into the lock file and return what it wrote.

        Returned as well as written so the acquire log line and the file say the
        same words: grepping the log for what the lock file contains finds the
        moment it was taken.
        """
        now = time.time()
        record = "pid %d on %s since %s (%d)" % (
            os.getpid(),
            os.uname().nodename,
            fmt_time(now),
            now,
        )
        try:
            os.ftruncate(self._fd, 0)
            os.pwrite(self._fd, record.encode(), 0)
        except OSError as exc:
            # The lock is the flock; this is a label on it. Losing the label
            # costs the next debugging session a clue and costs correctness
            # nothing, so it must not cost us the lock we just took.
            LOG.warning("supervisor lock %s: could not record the holder: %s", self.path, exc)
        return record

    def _report_contention(self, fd):
        """Name the holder, once per holder, at INFO.

        Once per holder rather than once per attempt: this runs every second for
        as long as a successor waits for its predecessor to step down, and a
        handover that is stuck is diagnosed from the one line naming what it is
        stuck behind, not from a thousand copies of it.
        """
        holder = self._record(fd) or "an unidentified process"
        if holder == self._reported:
            return
        self._reported = holder
        LOG.info(
            "supervisor lock %s is held by %s; routing only, retrying every %.0fs",
            self.path,
            holder,
            SUPERVISOR_LOCK_POLL,
        )

    @staticmethod
    def _record(fd):
        try:
            return os.pread(fd, 4096, 0).decode("utf-8", "replace").strip()
        except OSError:
            return ""

    def __enter__(self):
        # `with lock:` means the body runs holding it or not at all -- running
        # lifecycle code because nobody checked a return value is the failure
        # this class exists to prevent. Callers that can carry on without it,
        # like supervisor_loop, call acquire() and read the answer.
        self._entered.append(self.held)
        if not self.acquire():
            self._entered.pop()
            raise BlockingIOError(
                "supervisor lock %s is held by %s"
                % (self.path, self.holder() or "an unidentified process")
            )
        return self

    def __exit__(self, exc_type, exc, tb):
        if not self._entered.pop():
            self.release(retire=False)
        return False


async def supervisor_loop(fleet):
    """Sweep the fleet's lifecycle, but only while holding the supervisor lock.

    Two gateways overlap for the length of a handover and both run this. Only
    one of them may act, so everything `supervise` does is behind the lock,
    while the loops either side of this one -- discovery, probing, routing --
    are read-only and run in both.

    A gateway that cannot get the lock keeps asking rather than giving up, and
    that polling *is* the takeover: the predecessor releases at handover step 7,
    or dies and the kernel releases on its behalf, and one poll later the
    successor is supervising. Nothing is handed over explicitly, so there is no
    message that can be lost and no state that can disagree.
    """
    lock = fleet.supervisor_lock
    if getattr(fleet.args, "router_only", False):
        LOG.info("--router-only: routing now, supervising once %s comes free", lock.path)
    while True:
        try:
            if not lock.held and not lock.retired:
                # On a thread, for the same reason a discovery sweep is: the
                # lock file sits in the fleet directory, on the shared
                # filesystem every serving job writes to, and an open() there
                # can stall long enough to be felt on the request path.
                await in_thread(lock.acquire)
            if lock.held:
                await supervise(fleet)
        except Exception:
            LOG.exception("supervisor failed")
        # Poll quickly while waiting for the lock, sweep at the configured
        # interval while holding it. A retired gateway asks for nothing and is
        # only still here to finish draining.
        delay = fleet.args.supervisor_interval
        if not lock.held and not lock.retired:
            delay = min(SUPERVISOR_LOCK_POLL, delay)
        await asyncio.sleep(delay)


async def revive_dead_backends(fleet, now):
    """Restart an in-service deployment that exited but kept its allocation.

    `supervise_pending` already does exactly this, but only for a successor the
    gateway itself submitted -- so a backend that served for hours and then had
    its server killed sat dead indefinitely, holding eight idle nodes, while
    the controller waited for a `restart` nobody was going to write. Twice in
    one night that cost tens of minutes of a quarter of the fleet.

    Restarting in place reuses the existing allocation and takes about ten
    seconds, against roughly fifteen minutes and eight more nodes to roll.

    Deliberately conservative, because this is the routing process taking a
    lifecycle action:
      - only on the deployment's own "exited" state, never on probe failure;
      - never while a roll is in flight (`draining`, plus `superseded` when
        relay is on), or the gateway would fight fleetctl over the same job;
      - only while the heartbeat is fresh, since a controller that is gone
        cannot see the file;
      - capped per job, so a deployment that cannot start is left alone and
        reported instead of being restarted forever.
    """
    if fleet.args.revive_limit <= 0:
        return
    tag = pool_tag(fleet)
    pending_job = fleet.pending[0] if fleet.pending else None
    for job_id, backend in sorted(fleet.backends.items()):
        if job_id == pending_job:
            continue  # supervise_pending owns this one
        # `superseded` only gates revive when a roll can actually happen. It
        # means "a longer-lived backend exists", which under --no-relay is true
        # of every instance but one and never clears: the promotion to
        # `draining` that would discard it is behind the same --no-relay
        # return. Consulting it there disabled revive fleet-wide -- 10 of 14
        # healthy backends were permanently disqualified, and three died
        # holding eight nodes each until a human rolled them by hand. This is
        # the same "superseded by definition" trap that `serving_candidates`
        # documents for routing.
        if job_id in fleet.draining or (not fleet.args.no_relay and job_id in fleet.superseded):
            continue
        if backend.healthy or not attempt_failed(backend):
            continue
        if now - backend.heartbeat > fleet.args.stale_after:
            continue  # controller is not there to act
        tries, last = fleet.revived.get(job_id, (0, 0.0))
        if now - last < fleet.args.revive_cooldown:
            continue
        # A preemption notice is what ended this attempt, and the scheduler
        # takes the allocation back when the grace period runs out. Reviving
        # there admits ten minutes of new requests that are then all cut
        # mid-stream together; Slurm requeues the job on its own afterwards.
        notice = await slurm_preemption_notice(job_id)
        if notice is not None:
            if _PREEMPTION_NOTED.get(job_id) != notice:
                _PREEMPTION_NOTED[job_id] = notice
                LOG.warning(
                    "%s%s exited after Slurm signalled it for preemption at %s; "
                    "not restarting an allocation the scheduler is taking back",
                    tag,
                    job_id,
                    notice,
                )
            continue
        if tries >= fleet.args.revive_limit:
            if tries == fleet.args.revive_limit:
                LOG.error(
                    "%s%s exited %d times; leaving it alone -- roll or investigate it by hand",
                    tag,
                    job_id,
                    tries,
                )
                fleet.revived[job_id] = (tries + 1, now)
            continue
        LOG.warning(
            "%s%s is not serving (%s); restarting its retained allocation (attempt %d of %d)",
            tag,
            job_id,
            backend.state,
            tries + 1,
            fleet.args.revive_limit,
        )
        fleet.revived[job_id] = (tries + 1, now)
        code, out = await run_serve_sh(fleet, "restart", backend.run_dir)
        if code != 0:
            LOG.error("%srestart %s failed (rc=%d): %s", tag, job_id, code, out)


def instance_label(run_dir):
    """serve.sh names a run <user>_<date>_<jobid>_<cfgname>_<label>."""
    base = os.path.basename(os.path.normpath(run_dir or ""))
    return base.rsplit("_", 1)[-1] if "_" in base else ""


async def relay_per_instance(fleet, now):
    """Roll each instance before its own wall clock, independently.

    The single-lineage relay above asks one question -- is `active` about to
    expire -- and submits one successor. That is right for a deployment with
    one instance and wrong for this fleet, where several serve at once and each
    has its own end time; the expiring one is usually not `active`.

    So the question is asked per instance label instead. `fleetctl up --only
    <label> --force` is what submits, rather than `serve.sh submit --yaml`:
    fleetctl is the only thing that knows the configured shape of a named
    instance, and --force is required because the instance still has a job --
    the overlap is the entire point. Retirement is not done here: marking the
    predecessor superseded lets the existing drain and reclaim take it, which
    already waits for the successor to hold up and for in-flight requests to
    finish.

    Submissions are rate-limited per label rather than tracked to completion. A
    submitted job takes minutes to register, so "is there a newer sibling yet"
    stays false for a while and would re-submit on every sweep; the cooldown is
    what makes this idempotent, and it is keyed per label so one instance
    rolling cannot block another.
    """
    tag = pool_tag(fleet)
    by_label = {}
    for job_id, backend in fleet.backends.items():
        label = instance_label(backend.run_dir)
        if not label:
            continue
        by_label.setdefault(label, []).append((job_id, backend))

    for label, members in sorted(by_label.items()):
        # Newest by end time is the one that will still be here; anything older
        # of the same label is what it replaces.
        members.sort(key=lambda kv: kv[1].end_time)
        newest_id, newest = members[-1]
        older = members[:-1]

        if older and newest.healthy:
            for job_id, _ in older:
                if job_id in fleet.superseded or job_id in fleet.draining:
                    continue
                fleet.superseded.add(job_id)
                LOG.info(
                    "%ssuperseded %s; %s is the newer %s and is serving",
                    tag,
                    job_id,
                    newest_id,
                    label,
                )

        if newest.end_time <= 0:
            continue  # no wall clock to relay against; recover_lost_backends still covers it
        remaining = newest.end_time - now
        if remaining >= fleet.args.lead_time:
            continue
        if len(members) > 1:
            continue  # a successor for this label already exists
        last = fleet.relaying.get(label, 0.0)
        if now - last < fleet.args.min_submit_interval:
            continue
        fleet.relaying[label] = now
        LOG.info(
            "%s%s (%s) ends in %ds; submitting its successor", tag, label, newest_id, int(remaining)
        )
        code, out = await run_fleetctl(fleet, "up", "--only", label, "--force")
        if code != 0:
            # `out` as-is, like the recovery path below: an undefined helper
            # here raised NameError exactly when a successor submit failed,
            # replacing fleetctl's reason with a traceback and skipping the
            # rest of that supervise tick.
            LOG.error(
                "%srelay submit for %s failed (rc=%s): %s", tag, label, code, out or "(no output)"
            )


async def recover_lost_backends(fleet, now):
    """Bring back instances the scheduler is not going to bring back itself.

    revive_dead_backends cannot cover preemption. It acts on the deployment's
    own "exited" state, written by a controller that is still running -- and
    preemption takes the controller with it. All three ways a node is lost end
    with the backend simply absent: SIGTERM and "allocation gone" delete the
    registration through clear_fleet, and SIGKILL leaves one that goes stale.
    Nothing in the table is left to iterate, so recovery works off `fleet.lost`
    instead.

    The hard part is not noticing, it is not fighting SLURM. `PreemptMode` here
    is REQUEUE, so a preempted job usually comes back on its own, keeping its
    job id -- and resubmitting on top of that both duplicates the instance and
    throws away the PreemptExemptTime already earned. So nothing is submitted
    until the scheduler says it has no record of the job at all, which is the
    one state that means nobody else is going to bring it back.

    The action is `fleetctl up`, not a direct submit, because fleetctl is what
    knows the configured fleet -- and it reconciles against squeue itself, so
    an instance the scheduler is still holding is skipped there too. That makes
    this idempotent: the worst a spurious call does is print what is already
    running.
    """
    if not fleet.args.fleet_config or fleet.args.recover_limit <= 0:
        return
    if now - fleet.last_recovery < fleet.args.recover_cooldown:
        return
    tag = pool_tag(fleet)
    due = [
        (j, rec)
        for j, rec in sorted(fleet.lost.items())
        if now - rec[1] >= fleet.args.recover_grace
    ]
    if not due:
        return

    orphaned = []
    for job_id, (run_dir, _) in due:
        status = await slurm_job_status(job_id)
        if status is None:
            continue  # cannot tell; ask again next sweep
        if status[0] != "GONE":
            continue  # requeued or still queued: not ours
        tries, _ = fleet.recovered.get(job_id, (0, 0.0))
        if tries >= fleet.args.recover_limit:
            if tries == fleet.args.recover_limit:
                LOG.error(
                    "%s%s never came back after %d recovery attempts; leaving it to an operator",
                    tag,
                    job_id,
                    tries,
                )
                fleet.recovered[job_id] = (tries + 1, now)
            continue
        orphaned.append((job_id, run_dir, tries))
    if not orphaned:
        return

    labels = sorted(
        {label for label in (instance_label(run_dir) for _, run_dir, _ in orphaned) if label}
    )
    LOG.warning(
        "%sscheduler has no record of %s (was %s); reconciling the fleet",
        tag,
        ", ".join(job_id for job_id, _, _ in orphaned),
        ", ".join(labels) if labels else "unlabelled",
    )
    # Counted before the call, so a fleetctl that cannot run still spends the
    # budget rather than being retried every cooldown forever.
    for job_id, _, tries in orphaned:
        fleet.recovered[job_id] = (tries + 1, now)
    fleet.last_recovery = now
    code, out = await run_fleetctl(fleet, "up")
    if code == 0:
        for job_id, _, _ in orphaned:
            fleet.lost.pop(job_id, None)
        LOG.info("%sfleetctl up: %s", tag, out or "(no output)")
    else:
        # Left in `lost` on purpose: a scheduler that was busy this minute may
        # not be the next, and the attempt counter bounds the retries.
        LOG.error("%sfleetctl up failed (rc=%s): %s", tag, code, out)


async def check_recovery(fleet):
    """check_recovery_pool() for each supervised pool, against its own fleet config."""
    if not getattr(fleet, "pools", None):
        await check_recovery_pool(fleet)
        return
    for view in fleet.supervised_views():
        await check_recovery_pool(view)


async def check_recovery_pool(fleet):
    """Prove at startup that recovery could actually run -- both halves of it.

    The failure this catches is quiet and slow: recovery has to reach the
    cluster's scheduler, and a gateway running anywhere else cannot. Pointed at
    a local fleetctl from off-cluster it execs fine and then fails on squeue --
    which nobody sees until the first preemption, hours later, by which time
    the fleet has been shrinking unattended. Two calls here turn that into a
    line in the startup log.

    Two, not one, because recovery is a query and an action and they travel by
    different routes. `fleetctl up` is the action; asking whether the job is
    really gone is the query, and it runs squeue directly rather than through
    fleetctl. Wrapping only the action leaves the gateway announcing that
    recovery is ready and then never recovering anything, because the query it
    gates on can never answer. That is the exact failure this probe existed to
    prevent, reintroduced one level down.

    A warning rather than a refusal: the scheduler being briefly unreachable is
    not a reason to decline to route traffic, and recovery retries anyway.
    """
    if not fleet.args.fleet_config:
        return
    tag = pool_tag(fleet)

    code, out = await run_fleetctl(fleet, "status")
    if code != 0:
        LOG.warning(
            "%spreemption recovery is configured but `%s --config %s status` failed (rc=%s): %s",
            tag,
            fleet.args.fleetctl,
            fleet.args.fleet_config,
            code,
            (out or "")[-300:],
        )
        LOG.warning(
            "nothing will bring a preempted instance back until this works. "
            "fleetctl runs squeue and sbatch, so it has to run on the cluster's "
            "login node -- from elsewhere, point --fleetctl at a wrapper that "
            "ssh-es there and give --fleet-config the path as that host sees it."
        )
        return

    code, out = await run_slurm_command("squeue", "--me", "--noheader", "--format", "%i")
    if code != 0:
        LOG.warning(
            "%spreemption recovery can submit but cannot ask whether a job is gone: "
            "squeue failed (rc=%s): %s",
            tag,
            code,
            (out or "")[-300:],
        )
        LOG.warning(
            "a lost instance is only replaced once the scheduler says it has no record "
            "of the job, so this leaves recovery permanently undecided -- it will never "
            "act, and never say why. Off-cluster, point --slurm-wrapper at slurm-remote "
            "with SLURM_SSH_HOST set."
        )
        return

    LOG.info(
        "%spreemption recovery ready: %s %s (scheduler reachable via %s)",
        tag,
        fleet.args.fleetctl,
        fleet.args.fleet_config,
        fleet.args.slurm_wrapper or "local squeue",
    )


async def supervise(fleet):
    """One lifecycle sweep, pool by pool.

    With no extra pool this is the sweep it always was, over the Fleet itself.
    With extra pools, each supervised pool is swept through its own PoolView:
    its own backends, its own fleet config, its own relay labels -- GLM's i00
    and K3's i00 are two instances, and one sweep over both would have the
    longer-lived one supersede, drain and quit the other. A pool with no fleet
    config is never swept at all.

    Each pool's sweep is contained: a failure acting on one fleet's jobs must
    not cost the other fleets their supervision.
    """
    if not getattr(fleet, "pools", None):
        await _supervise_pool(fleet)
        return
    for view in fleet.supervised_views():
        try:
            await _supervise_pool(view)
        except Exception:
            LOG.exception("%ssupervisor failed", pool_tag(view))


async def _supervise_pool(fleet):
    now = time.time()
    await supervise_pending(fleet, now)
    await revive_dead_backends(fleet, now)
    # Ahead of the --no-relay return below, deliberately. Relay is about a job
    # reaching its own wall clock, which a proxy-only gateway has no business
    # preempting; losing a node to the scheduler is the opposite situation --
    # nothing else is watching for it, and the fleet only shrinks until
    # somebody notices.
    await recover_lost_backends(fleet, now)

    # Relay: submit the next job early enough that it finishes loading weights
    # before this one hits the wall clock.
    if fleet.args.relay_per_instance:
        await relay_per_instance(fleet, now)

    tag = pool_tag(fleet)
    # The single-lineage relay below resubmits --yaml, a serve.sh deployment
    # file that only the default pool has.
    lineage = getattr(fleet, "single_lineage", True)
    backend = fleet.backends.get(fleet.active) if fleet.active else None
    if (
        lineage
        and backend is not None
        and not fleet.args.no_relay
        and not fleet.args.relay_per_instance
    ):
        remaining = backend.end_time - now
        # A submitted/loading successor is represented by `pending`. An older
        # job that took traffic and then failed remains discoverable so it can
        # recover, but must not block a replacement successor forever.
        successors = [
            j for j, candidate in fleet.backends.items() if j != fleet.active and candidate.healthy
        ]
        if backend.end_time <= 0:
            # Registration could not determine the wall clock. Routing still
            # works; relaying on `0 - now` would read as "already expired" and
            # submit a job every single sweep.
            LOG.warning("backend %s has no end time; relay disabled for it", fleet.active)
        elif remaining < fleet.args.lead_time and not successors and not fleet.pending:
            LOG.info("%s ends in %ds; submitting successor", fleet.active, int(remaining))
            await submit_successor(fleet, now, "relay")
    elif (
        lineage
        and not fleet.args.no_relay
        and fleet.ever_active
        and not fleet.stopped
        and not fleet.backends
        and not fleet.pending
        and now - fleet.last_submit >= fleet.args.min_submit_interval
    ):
        # Recovery cannot depend on a live active backend: a cancelled pending
        # job may disappear just as its predecessor reaches the wall clock.
        LOG.warning("fleet lost every backend; submitting recovery successor")
        await submit_successor(fleet, now, "recovery")

    # Reclaim, and the drain that prepares for it, are two halves of the same
    # authority, and --no-relay withholds it. Draining a backend that will
    # never be reclaimed only takes it out of rotation for nothing -- which is
    # exactly what it did to every instance but the longest-lived one once
    # routing began serving them all at once.
    # Per-instance relay needs the same authority: it marks predecessors
    # superseded and relies on the drain below to retire them.
    if fleet.args.no_relay and not fleet.args.relay_per_instance:
        return

    # Promote superseded backends to draining, but only once the successor has
    # held up. Handing over routing is reversible and happens the instant the
    # successor is healthy; releasing the predecessor's allocation is not, so it
    # waits. Without this, a successor that passes one probe and then dies takes
    # the predecessor down with it and leaves nothing serving until the next job
    # finishes loading.
    winner = fleet.backends.get(fleet.active) if fleet.active else None
    if winner is not None and fleet.superseded:
        stable_for = now - winner.healthy_since if winner.healthy_since else 0
        if winner.healthy and stable_for >= fleet.args.promote_after:
            for job_id in sorted(fleet.superseded):
                fleet.superseded.discard(job_id)
                if job_id == fleet.active:
                    LOG.warning("%srefusing to drain active backend %s", tag, job_id)
                    continue
                backend = fleet.backends.get(job_id)
                if backend is None:
                    continue
                deadline = backend.end_time - 60
                fleet.draining[job_id] = deadline
                LOG.info(
                    "%sdraining %s (%s stable %ds, inflight=%d, reclaim by %s)",
                    tag,
                    job_id,
                    fleet.active,
                    int(stable_for),
                    fleet.inflight.get(job_id, 0),
                    fmt_time(deadline),
                )

    # Reclaim: the drained job is already past being useful, and its allocation
    # is worth releasing a little early.
    for job_id, deadline in list(fleet.draining.items()):
        if job_id == fleet.active:
            LOG.warning("%scancelled stale drain of active backend %s", tag, job_id)
            fleet.draining.pop(job_id, None)
            continue
        backend = fleet.backends.get(job_id)
        if backend is None:
            fleet.draining.pop(job_id, None)
            continue
        inflight = fleet.inflight.get(job_id, 0)
        # A backend that never published an end_time yields deadline < 0, so a
        # bare `now <= deadline` is false immediately and the quit below kills
        # whatever is still streaming. Without a real deadline there is nothing
        # to time out against, so wait for the drain instead.
        if inflight and (deadline <= 0 or now <= deadline):
            continue
        why = "drained" if not inflight else "deadline"
        LOG.info("%sreclaiming %s (%s, inflight=%d)", tag, job_id, why, inflight)
        code, out = await run_serve_sh(fleet, "quit", backend.run_dir)
        if code != 0:
            LOG.error("%squit %s failed (rc=%d): %s", tag, job_id, code, out)
        fleet.draining.pop(job_id, None)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def parse_args(argv):
    parser = argparse.ArgumentParser(
        description="stable front door for the Anthropic-compatibility server"
    )
    parser.add_argument(
        "--fleet-dir", required=True, help="directory the serving jobs register into"
    )
    parser.add_argument(
        "--pool",
        dest="pools",
        action="append",
        default=None,
        metavar="NAME=FLEET_DIR[,FLEET_CONFIG]",
        help="add another fleet directory beside --fleet-dir; repeatable. Its backends are "
        "discovered, probed and routed exactly like the default fleet's -- every "
        "directory is one routing pool, whatever model a request names -- and "
        "supervised (revive, recovery, walltime relay, drain) through FLEET_CONFIG, the "
        "pool's own fleet.yaml. Without one the pool is routed but never supervised",
    )
    parser.add_argument(
        "--pools-file",
        default=None,
        help="more --pool specs, one per line, read at startup (default: pools.conf beside "
        "--users; an empty string disables). A handover successor inherits its "
        "predecessor's command line, so this file is how a pool is added to a running "
        "gateway: write it, then POST /_gateway/handover",
    )
    parser.add_argument("--users", required=True, help="allowlist, one username per line")
    parser.add_argument("--yaml", default="", help="deployment YAML the supervisor resubmits")
    parser.add_argument(
        "--serve-sh",
        default="",
        help="launcher script exposing submit/restart/quit (defaults to serve.sh next to this file)",
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8333)
    parser.add_argument(
        "--reuse-port",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="bind the listener with SO_REUSEPORT (default on). This is what "
        "lets a successor bind the port while the outgoing gateway is still "
        "holding it, which is the whole basis of a handover with no refused "
        "connection; --no-reuse-port restores the exclusive bind and with it "
        "the restart gap.",
    )
    parser.add_argument(
        "--relay-per-instance",
        action="store_true",
        help="roll each configured instance before its own wall clock, via "
        "fleetctl. Use instead of the single-lineage relay on a fleet where "
        "several instances serve at once",
    )
    parser.add_argument(
        "--lead-time",
        type=int,
        default=2700,
        help="seconds before the wall clock to submit the successor (default "
        "45min). Must exceed the successor's cold start, or the predecessor "
        "expires before its replacement serves and the relay buys nothing: "
        "measured 50min for GLM-5.3-FP8 with install_repo (30min of pip, then "
        "weights and CUDA graphs), so that deployment sets 3300",
    )
    parser.add_argument(
        "--stale-after",
        type=int,
        default=30,
        help="drop a backend after this long without a heartbeat",
    )
    parser.add_argument(
        "--min-submit-interval",
        type=int,
        default=300,
        help="minimum seconds between recovery submits (default 5min); start_server ignores it",
    )
    parser.add_argument(
        "--mirror-concurrency",
        type=int,
        default=64,
        help="most request copies in flight to mirror targets at once. Past "
        "this, copies are dropped rather than queued: a target that has "
        "stopped draining would otherwise build a backlog of copies that are "
        "stale by the time they are sent (default 64)",
    )
    parser.add_argument(
        "--mirror-timeout",
        type=float,
        default=900.0,
        help="seconds a single mirrored request may take, including reading "
        "and discarding its response. Generous, because the point is to make "
        "the target do the whole job -- a copy cut short measures nothing "
        "(default 900)",
    )
    parser.add_argument(
        "--mirror-connect-timeout",
        type=float,
        default=5.0,
        help="seconds to wait for a mirror target to accept a connection. "
        "Separate from --mirror-timeout because a target that drops packets "
        "rather than refusing them would otherwise hold a slot for the whole "
        "of it (default 5)",
    )
    parser.add_argument(
        "--mirror-max-misses",
        type=int,
        default=50,
        help="consecutive failed copies after which a target stops being "
        "mirrored to, loudly. Without it a target that goes away is retried "
        "once per request for as long as nobody notices (default 50)",
    )
    parser.add_argument("--discover-interval", type=float, default=5.0)
    parser.add_argument("--health-interval", type=float, default=5.0)
    parser.add_argument("--supervisor-interval", type=float, default=30.0)
    parser.add_argument("--probe-timeout", type=float, default=3.0)
    parser.add_argument(
        "--revive-limit",
        type=int,
        default=3,
        help="restart a deployment that exited but kept its allocation, at most "
        "this many times (0 disables)",
    )
    parser.add_argument(
        "--revive-cooldown",
        type=int,
        default=180,
        help="seconds to wait between restart attempts on the same job",
    )
    parser.add_argument(
        "--fleet-config",
        default="",
        help="fleet.yaml to reconcile against when an instance is lost to the "
        "scheduler; unset disables preemption recovery entirely",
    )
    parser.add_argument(
        "--fleetctl",
        default="",
        help="fleet launcher used for that reconciliation (defaults to fleetctl next to this file)",
    )
    parser.add_argument(
        "--slurm-wrapper",
        default="",
        help="run squeue and friends through this instead of directly; needed when the "
        "gateway is not on the cluster, since recovery gates on asking the scheduler "
        "whether a lost job still exists (see slurm-remote)",
    )
    parser.add_argument(
        "--recover-grace",
        type=int,
        default=120,
        help="seconds a backend must stay absent before the scheduler is asked "
        "whether anything still owns its job",
    )
    parser.add_argument(
        "--recover-cooldown",
        type=int,
        default=600,
        help="seconds between fleet reconciliations, whatever went missing",
    )
    parser.add_argument(
        "--recover-limit",
        type=int,
        default=3,
        help="reconcile on behalf of the same lost job at most this many times (0 disables)",
    )
    parser.add_argument(
        "--unhealthy-after",
        type=int,
        default=20,
        help="consecutive probe timeouts before a backend is taken out of rotation; a refused "
        "connection is acted on immediately regardless",
    )
    parser.add_argument(
        "--upstream-connect-timeout",
        type=float,
        default=10.0,
        help="seconds to wait for a backend to accept a connection before treating the "
        "attempt as failed and retrying elsewhere. A backend that drops packets rather than "
        "refusing them otherwise holds the request for the kernel's two-minute connect "
        "timeout, longer than a Codex client waits for its first event (default 10)",
    )
    parser.add_argument(
        "--retry-upstream",
        type=int,
        default=1,
        help="how many other backends to try when one fails before any byte has "
        "reached the client. 0 restores the old behaviour of returning 502 at once",
    )
    parser.add_argument(
        "--refusals-before-drop",
        type=int,
        default=3,
        help="connection refusals on real requests, within --refusal-window, that "
        "take a backend out of rotation without waiting for the probe",
    )
    parser.add_argument(
        "--refusal-window",
        type=float,
        default=5.0,
        help="seconds over which --refusals-before-drop is counted",
    )
    parser.add_argument(
        "--promote-after",
        type=float,
        default=180.0,
        help="seconds a successor must stay healthy before its predecessor may be reclaimed",
    )
    # --handover-drain-deadline belongs to the drain, not to this sequence:
    # the drain is what enforces it, and a second definition here would be a
    # second default to keep in step with it.
    parser.add_argument(
        "--handover-ready-timeout",
        type=int,
        default=180,
        # Long, deliberately: it covers a successor reading a fleet directory
        # on a shared filesystem and probing every backend in it. Spending it
        # costs nothing -- this gateway is still serving the whole time -- and
        # cutting a successor off just short of ready wastes the handover.
        help="seconds to wait for a spawned successor to report ready before the handover "
        "is abandoned and the successor killed (default 180)",
    )
    parser.add_argument(
        "--no-relay",
        action="store_true",
        help="proxy only; never submit a successor and never reclaim a drained job",
    )
    parser.add_argument(
        "--router-only",
        action="store_true",
        help="start routing without supervising, and keep asking for the supervisor "
        "lock until it is free. How a gateway spawned to replace a running one "
        "comes up: it serves traffic immediately and takes over the fleet's "
        "lifecycle the moment its predecessor steps down or dies",
    )
    parser.add_argument(
        "--supervisor-lock",
        default=None,
        # In the fleet directory because that is the thing being supervised, and
        # the one path every gateway for this fleet is already given. Dotted so
        # it stays out of the `*.json` glob discovery reads.
        help="file whose flock decides which gateway may act on the fleet "
        "(default <fleet-dir>/.supervisor.lock)",
    )
    parser.add_argument(
        "--sticky-ttl",
        type=float,
        default=1800.0,
        help="seconds of silence after which a conversation loses its pin (default 1800)",
    )
    parser.add_argument(
        "--sticky-capacity",
        type=int,
        default=20000,
        help="most conversations pinned at once; the oldest are dropped (default 20000)",
    )
    parser.add_argument(
        "--max-body-buffer",
        type=int,
        default=4 * 1024 * 1024,
        # Agent turns resend the whole history, so the request grows with the
        # conversation. 4 MiB covers a long one; past that the request still
        # goes through, it just loses its affinity.
        help="largest request body read whole to identify a conversation (default 4 MiB)",
    )
    parser.add_argument(
        "--new-conversation-margin",
        type=float,
        default=1800.0,
        # A conversation started on a backend with less time left than this
        # would be cut off partway. Better to start it somewhere it can finish.
        help="stop giving a backend new conversations this many seconds before it ends "
        "(default 1800)",
    )
    parser.add_argument(
        "--pool-affinity",
        action=argparse.BooleanOptionalAction,
        default=True,
        # Pools are models here (GLM, Kimi-K3). A conversation moved across
        # them continues on a model that did not write its history.
        help="re-place a conversation only inside the pool it was first served from; while "
        "none of that pool's backends is accepting, its requests get no backend. "
        "--no-pool-affinity lets it move to any pool (default on)",
    )
    parser.add_argument(
        "--route-policy",
        default="least_conversations",
        help="how a new conversation picks a backend (default least_conversations). "
        "One of %s, or the name of a file in --policy-dir." % ", ".join(POLICIES),
    )
    parser.add_argument(
        "--policy-dir",
        default=None,
        help="directory of custom routing policies. Each <name>.py defines "
        "select(accepting) and supplies the policy <name>; accepting maps job "
        "id to (conversations, inflight, remaining_seconds). Rescanned when a "
        "file changes, so a policy can be added or fixed without a restart.",
    )
    parser.add_argument(
        "--router-state",
        default=None,
        # Defaults next to the fleet directory below, so the pins live with the
        # deployment they describe rather than in whatever the cwd happens to be.
        help="file the pin table is saved to and restored from "
        "(default <fleet-dir>/../router_state.json; empty string disables)",
    )
    parser.add_argument(
        "--key-source",
        dest="key_sources",
        action="append",
        default=None,
        metavar="SOURCE",
        help="where to look for a conversation id, in order; repeatable. "
        "header:<name>, body:<a.b.c>, or prefix. Replaces the built-in chain.",
    )
    parser.add_argument(
        "--no-conversation-routing",
        dest="route_by_conversation",
        action="store_false",
        help="ignore conversation identity and balance every request independently",
    )
    parser.add_argument(
        "--handover-drain-deadline",
        type=float,
        default=120.0,
        help="seconds to let in-flight requests finish after SIGTERM/SIGINT "
        "before exiting anyway (default 120)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING"),
        # DEBUG additionally prints the last SSE data line before a failure,
        # which is response content -- useful when reproducing a truncation,
        # inappropriate as a standing default.
        help="DEBUG also logs response previews on failure (contains model output)",
    )
    args = parser.parse_args(argv)
    if args.router_state is None:
        args.router_state = os.path.join(
            os.path.dirname(os.path.normpath(args.fleet_dir)), "router_state.json"
        )
    elif not args.router_state:
        args.router_state = None
    if not args.supervisor_lock:
        # No way to switch this off. A gateway without it is a gateway that will
        # supervise beside another one, and the whole reason the lock exists is
        # that doing so costs eight nodes at a time.
        args.supervisor_lock = os.path.join(args.fleet_dir, ".supervisor.lock")
    if args.key_sources:
        bad = [
            x for x in args.key_sources if not (x.startswith(("header:", "body:")) or x == "prefix")
        ]
        if bad:
            parser.error(
                "unusable --key-source %s; expected header:<name>, "
                "body:<a.b.c> or prefix" % ", ".join(bad)
            )
    if not args.serve_sh:
        args.serve_sh = os.path.join(os.path.dirname(os.path.abspath(__file__)), "serve.sh")
    if not args.fleetctl:
        args.fleetctl = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fleetctl")
    if args.pools_file is None:
        # Beside the users file, the one path every generation is given that
        # sits in GW_DIR -- the same derivation as the pid file. A default
        # rather than an opt-in, because the generation that most needs it is
        # a handover successor, and that one cannot be given a new flag.
        args.pools_file = os.path.join(os.path.dirname(os.path.abspath(args.users)), "pools.conf")
    try:
        flags = [parse_pool_spec(text) for text in args.pools or ()]
        from_file = read_pools_file(args.pools_file) if args.pools_file else []
        args.pools = combine_pools(flags, from_file, args.fleet_dir)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    args.pools_from_file = [spec.name for spec in from_file]
    if args.fleet_config or any(spec.fleet_config for spec in args.pools):
        # --fleetctl is something this host executes, so it is checked here.
        # --fleet-config deliberately is not: it is passed *to* fleetctl, and
        # fleetctl has to run where the cluster's scheduler is. A gateway that
        # outlives the scheduler is by definition not there, so it points
        # --fleetctl at a wrapper that runs the real one over ssh -- and then
        # --fleet-config names a path on that host, which does not exist on
        # this one. Whether the pair actually works is a question only running
        # them can answer; check_recovery does that at startup.
        if not os.path.isfile(args.fleetctl):
            parser.error("--fleetctl %s does not exist" % args.fleetctl)
        if not os.access(args.fleetctl, os.X_OK):
            parser.error("--fleetctl %s is not executable" % args.fleetctl)
    # Checked whether or not recovery is configured: it is also what the
    # successor chain queries with, and a wrapper named by mistake should fail
    # at startup rather than at the first handover.
    if args.slurm_wrapper:
        if not os.path.isfile(args.slurm_wrapper):
            parser.error("--slurm-wrapper %s does not exist" % args.slurm_wrapper)
        if not os.access(args.slurm_wrapper, os.X_OK):
            parser.error("--slurm-wrapper %s is not executable" % args.slurm_wrapper)
    if not args.no_relay:
        if not args.yaml:
            parser.error("--yaml is required unless --no-relay is given")
        # Relay drives job lifecycles through the launcher script. That script
        # does not ship with this example, so refuse at startup rather than
        # discovering it hours later at the first handover, which is exactly
        # when there is no time to recover.
        if not os.path.isfile(args.serve_sh):
            parser.error(
                "launcher script %s does not exist; pass --serve-sh or run with --no-relay"
                % args.serve_sh
            )
    return args


async def drain(server, deadline):
    """Stop accepting, let the requests already in flight finish, count the losses.

    Returns the number of requests abandoned -- zero unless the deadline ran
    out. The figure comes from `INFLIGHT` rather than from the server, because
    once the deadline has expired the server will no longer say how many
    handlers it is still holding.

    Waiting on BOTH the server and `INFLIGHT` is not belt and braces; neither
    one is sufficient, and the reason was measured on the 3.12.3 interpreter
    this runs against rather than taken from the docs.

    `close()` behaves as advertised: it returns immediately and the listening
    socket starts refusing at once (ECONNREFUSED, not a hang), while
    `wait_closed()` blocks for every connection still attached -- all of them,
    not just the first to finish.

    But "attached" is the catch. `wait_closed()` is keyed on the server's
    active *transport* count, which drops the moment the client's socket dies,
    not when the handler finishes. That distinction is invisible in a normal
    HTTP server, where the handler finishes by answering the client. It is not
    invisible in a proxy: when an agent CLI is killed mid-stream, this gateway
    is still holding the backend connection, still reading from it, and still
    has a request to finish or abandon -- and `wait_closed()` has already
    returned. Measured: with six streams whose clients had been RST, it
    returned in 0.02s with all six handlers still running. Draining on that
    alone would call the process idle and exit while it was demonstrably not.

    So `INFLIGHT.wait_idle()` covers the work, and `wait_closed()` covers the
    two edges the counter cannot see: a connection accepted whose handler task
    has not started yet, and the transport teardown after the handler returns.

    The deadline is not paranoia either. A connection that was accepted but
    whose client then went quiet holds the drain open exactly as firmly as one
    still streaming tokens -- `_handle` blocks in `read_head` for a request
    that never arrives. Unbounded, one such client keeps the outgoing gateway
    alive forever and the handover never completes.
    """
    at_start = INFLIGHT.count
    server.close()
    LOG.info("listener closed; draining %d in-flight request(s), deadline %gs", at_start, deadline)
    started = time.time()

    async def finished():
        await server.wait_closed()
        await INFLIGHT.wait_idle()

    try:
        await asyncio.wait_for(finished(), deadline)
    except TimeoutError:
        abandoned = INFLIGHT.count
        # WARNING, not INFO: this is the one outcome where a client genuinely
        # lost a response, which is the thing the handover exists to prevent.
        # It has to be greppable afterwards, with the count, or "did anyone
        # actually get hurt" is unanswerable.
        LOG.warning(
            "drain deadline of %gs expired with %d request(s) still in flight; "
            "abandoning them and exiting",
            deadline,
            abandoned,
        )
        return abandoned
    LOG.info("drained %d request(s) in %.1fs", at_start, time.time() - started)
    return 0


def install_drain_signals(loop, request_drain):
    """Make SIGTERM and SIGINT start a drain instead of killing the process.

    Registered on the loop rather than with `signal.signal`: a loop signal
    handler runs as an ordinary loop callback, so it may touch asyncio objects.
    A `signal.signal` handler runs between bytecodes on whatever the
    interpreter was doing and may not.

    Note what this costs: once registered, SIGINT no longer raises
    KeyboardInterrupt. That is the point -- Ctrl-C on an interactive gateway
    should drain too -- but it does leave an operator who has decided the drain
    is taking too long with no lever short of SIGKILL. So the SECOND signal is
    deliberately not a drain request: it exits now and says how many requests
    that cost.
    """

    def on_signal(name):
        if request_drain.is_set():
            LOG.warning("%s during drain; exiting now, %d request(s) dropped", name, INFLIGHT.count)
            # os._exit, not sys.exit: this is a loop callback, so a SystemExit
            # raised here would be reported by the exception handler and
            # swallowed, and the process would carry on draining. Flush first,
            # because os._exit runs no atexit hooks -- and the line above is
            # the only explanation anyone will get.
            logging.shutdown()
            os._exit(1)
        LOG.info("%s received; draining %d in-flight request(s)", name, INFLIGHT.count)
        request_drain.set()

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, on_signal, signal.Signals(sig).name)


async def main_async(args):
    # A module global rather than something threaded through, because
    # slurm_job_status is handed a job id and nothing else, and the alternative
    # is passing a transport down four call sites that have no other reason to
    # know about one. Set once, at startup, before anything can query.
    global SLURM_WRAPPER
    SLURM_WRAPPER = args.slurm_wrapper

    fleet = Fleet(args)
    os.makedirs(args.fleet_dir, exist_ok=True)
    for pool in fleet.pools.values():
        # Not fatal: an extra pool whose directory cannot be made has no
        # backends, which costs only its own share of the fleet; every other
        # pool's backends serve as normal.
        try:
            os.makedirs(pool.fleet_dir, exist_ok=True)
        except OSError as exc:
            LOG.warning("[%s] cannot create fleet dir %s: %s", pool.name, pool.fleet_dir, exc)
    fleet.reload_users()
    if not fleet.users:
        LOG.warning("users file %s is empty; every request will get 401", args.users)
    fleet.discover()
    # Before the listener, not after: with reuse_port the bind is the moment
    # traffic can arrive, so anything this process has not learned by then it
    # answers wrongly. See initial_health_pass.
    await initial_health_pass(fleet)

    gateway = Gateway(fleet)
    server = await asyncio.start_server(
        gateway.handle, args.host, args.port, reuse_port=args.reuse_port
    )
    # The handover owns the listener from here: step 6 closes it and
    # /_gateway/ready answers on whether it is still open. Attached after the
    # bind rather than before, so a port that could not be taken leaves no
    # half-armed handover behind.
    gateway.handover.attach(server)
    # Written once the port is genuinely held, for the same reason: the pid
    # file is what the job script waits on, and it must never name a
    # generation that lost the bind race. Every generation writes it.
    write_pid_file(args)
    LOG.info("listening on %s:%d", args.host, args.port)
    LOG.info("fleet dir: %s", args.fleet_dir)
    for pool in fleet.pools.values():
        LOG.info(
            "pool %s: fleet dir %s, routed together with every other pool%s",
            pool.name,
            pool.fleet_dir,
            " (from %s)" % args.pools_file if pool.name in args.pools_from_file else "",
        )
        if pool.fleet_config:
            LOG.info("pool %s: supervised against %s", pool.name, pool.fleet_config)
        else:
            # Once, here, rather than every supervisor sweep: the pool is
            # skipped by every sweep, and saying so each time would bury the
            # log in a message that never changes.
            LOG.warning(
                "pool %s has no fleet config: its backends are routed but never revived, "
                "recovered or relayed",
                pool.name,
            )
    LOG.info(
        "relay: %s",
        "off" if args.no_relay else "lead time %ds from %s" % (args.lead_time, args.yaml),
    )
    # After the listener is up, so a slow or unreachable scheduler delays the
    # answer about recovery rather than the serving of traffic.
    await check_recovery(fleet)

    drain_requested = asyncio.Event()
    install_drain_signals(asyncio.get_running_loop(), drain_requested)

    background = asyncio.gather(
        discovery_loop(fleet),
        health_loop(fleet),
        supervisor_loop(fleet),
        router_state_loop(fleet),
    )
    signalled = asyncio.ensure_future(drain_requested.wait())
    # Deliberately not `async with server` any more. Its __aexit__ awaits
    # wait_closed() with no timeout, so a drain that hit its deadline would
    # hang there forever on the very connections the deadline just gave up on
    # -- measured, not inferred. The drain below closes the server itself, so
    # nothing is lost by dropping the context manager.
    await asyncio.wait({background, signalled}, return_when=asyncio.FIRST_COMPLETED)
    signalled.cancel()

    # Cancelled before the drain, not after. These loops are lifecycle
    # actions -- submitting successors, writing restart control files, reviving
    # backends -- and a gateway on its way out has no business taking any of
    # them; during a handover its replacement is already doing so. The requests
    # still draining are unaffected: each one resolved its backend when it
    # arrived (see `_handle`), so a fleet view that stops updating cannot move
    # anything out from under them.
    background.cancel()
    abandoned = await drain(server, args.handover_drain_deadline)
    # Re-raise whatever took a background loop down, if that is why we are
    # here rather than a signal. `cancel()` above did nothing to an
    # already-finished gather, so its exception is still there to collect; when
    # we got here on a signal this is just the cancellation coming back.
    try:
        await background
    except asyncio.CancelledError:
        pass
    return abandoned


def main():
    args = parse_args(sys.argv[1:])
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )
    try:
        asyncio.run(main_async(args))
    except KeyboardInterrupt:
        # Only reachable in the window before install_drain_signals runs --
        # reading the users file, the first fleet discovery, check_recovery.
        # After that SIGINT is a drain request and never becomes an exception,
        # so the normal Ctrl-C path is the clean return below.
        LOG.info("interrupted")
        return 0
    # Zero even when the deadline abandoned requests: a bounded loss is still a
    # completed handover, and drain() has already logged the count at WARNING.
    return 0


if __name__ == "__main__":
    sys.exit(main())
