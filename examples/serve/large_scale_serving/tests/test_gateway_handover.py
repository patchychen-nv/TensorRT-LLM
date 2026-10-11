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
"""End-to-end tests for the gateway hot handover (CONTRACT.md, work item F).

The property these exist to protect is one sentence:

    During a handover, a client holding a steady stream of requests sees
    **zero refused connections and zero failed in-flight requests**.

Everything else here is secondary, and the file is arranged so that the
secondary things cannot dilute it: `test_handover_serves_every_request` asserts
that sentence and nothing else, against a real `gateway.py` process that spawns
a real successor, measured by a real client on raw sockets.

Why a process test and not a unit test. Every mechanism in the contract is a
property of two processes and a kernel: `SO_REUSEPORT` splitting a listen
queue, `flock` released by process death, a listener closing while its handlers
are still running, a successor surviving `start_new_session=True`. A mock of
`gateway.py` would have to model all four, and the modelling is where the bugs
would be.

Running them::

    pytest examples/serve/large_scale_serving/tests/test_gateway_handover.py -v

No SLURM, no GPU, no network beyond loopback. Roughly two and a half minutes.
Every wait is bounded; nothing here can hang. `pytest --timeout=600` is a
reasonable belt-and-braces if pytest-timeout happens to be installed, but the
tests do not require it.

Until work items A-E land these fail, on purpose, naming the missing clause.
"""

import json
import os
import re
import threading
import time

import pytest
from handover_harness import (
    API_USER,
    CONVERSATION_HEADER,
    DELAY_HEADER,
    HealthProbe,
    RequestStream,
    Scenario,
    check_probe_path_unchanged,
    pid_alive,
    request,
    require_cli,
    send_and_reset,
    stream_report,
    wait_for,
)

# The exact sentence item B logs on deadline expiry, per the contract's
# "Confirmed integration seams". Matched loosely enough to survive a reworded
# suffix, strictly enough that a log line without a count does not pass.
DRAIN_EXPIRY = re.compile(
    r"drain deadline of\s*([0-9.]+)\s*s?\s*expired with\s*(\d+)\s*request", re.IGNORECASE
)

HANDOVER_ARGS = [
    "--handover-drain-deadline",
    "30",
    "--handover-ready-timeout",
    "45",
]


@pytest.fixture(autouse=True, scope="module")
def _probe_path_is_still_what_the_fakes_answer():
    check_probe_path_unchanged()


@pytest.fixture
def make_scenario():
    """Build scenarios and guarantee they are torn down, successor included."""
    created = []

    def factory(*args, **kwargs):
        scenario = Scenario(*args, **kwargs)
        created.append(scenario)
        return scenario

    yield factory
    for scenario in reversed(created):
        scenario.close()


def post(port, seq, convo, delay=0.0, read_timeout=30.0):
    """One data-plane request, shaped like the traffic this gateway carries."""
    headers = [("x-api-key", API_USER), (CONVERSATION_HEADER, convo), ("x-test-seq", str(seq))]
    if delay:
        headers.append((DELAY_HEADER, "%.3f" % delay))
    body = json.dumps(
        {"model": "fake", "max_tokens": 4, "messages": [{"role": "user", "content": "hi %d" % seq}]}
    ).encode()
    return request(
        port,
        body=body,
        headers=headers,
        seq=seq,
        convo=convo,
        delay=delay,
        read_timeout=read_timeout,
    )


# ---------------------------------------------------------------------------
# Does the harness itself work? This one passes against today's gateway.
# ---------------------------------------------------------------------------
def test_harness_self_check(make_scenario):
    """Prove the fake fleet satisfies gateway.py before trusting any verdict.

    A fake backend is only useful if `gateway.py` genuinely discovers it,
    genuinely calls it healthy, and genuinely proxies to it. If any of those is
    wrong, every other test in this file measures an empty fleet and reports
    "zero requests lost" while proving nothing. So this test asserts the four
    contracts the fakes have to satisfy, against an unmodified gateway with no
    handover flags at all -- it is expected to pass today.
    """
    scenario = make_scenario("selfcheck", backends=2)
    scenario.start()

    health = scenario.wait_health_status("ok", timeout=25.0)
    assert health["deployment"] == scenario.fleet.deployment
    assert health["active"] in {b.job_id for b in scenario.fleet.backends}, (
        "gateway elected %r, which is not one of the fake backends %s"
        % (health["active"], sorted(b.job_id for b in scenario.fleet.backends))
    )

    stream = RequestStream(scenario.port, workers=6).start()
    try:
        wait_for("120 served requests", lambda: len(stream.served()) >= 120, timeout=30.0)
    finally:
        stream.stop()

    report = stream_report(stream)
    assert not stream.refused(), "connections were refused with nothing going on:\n%s" % report
    assert not stream.lost(), "requests were lost with nothing going on:\n%s" % report
    assert not stream.errored(), "HTTP errors with nothing going on:\n%s" % report

    # Both fakes have to be reachable, or a "handover kept serving" verdict
    # could be one backend answering the whole time.
    assert len(stream.by_backend()) == 2, (
        "conversation routing only ever reached %s; the other fake backend is not being "
        "proxied to, so any later verdict about which backend served what is worthless:\n%s"
        % (sorted(stream.by_backend()), report)
    )

    # Affinity inside one process, which is the thing the handover has to carry
    # across two.
    first = post(scenario.port, 1, "selfcheck-convo")
    second = post(scenario.port, 2, "selfcheck-convo")
    assert first.ok and second.ok, (first, second)
    assert first.backend == second.backend, (
        "a conversation moved backends inside a single gateway process (%s -> %s); "
        "conversation routing is broken before the handover is even involved"
        % (first.backend, second.backend)
    )

    # The supervisor-action proxy the lock test depends on. A backend that is
    # unhealthy AND says it exited is what `revive_dead_backends` acts on, and
    # the action is `<--serve-sh> restart <run_dir>`.
    sick = scenario.fleet.backends[1]
    sick.set_healthy(False)
    sick.state = "server exited with status 1"
    calls = wait_for(
        "gateway to take a supervisor action (serve.sh restart)",
        lambda: [c for c in scenario.fleet.serve_sh_calls() if c[0] == scenario.gateway.pid],
        timeout=30.0,
    )
    assert any("restart" in argv for _, _, argv in calls), calls
    assert any(sick.run_dir in argv for _, _, argv in calls), (
        "restart was called for the wrong run dir: %s" % calls
    )


# ---------------------------------------------------------------------------
# The property that decides whether this ships.
# ---------------------------------------------------------------------------
def test_handover_serves_every_request(make_scenario):
    """Zero refused connections and zero failed in-flight requests, end to end."""
    require_cli("--handover-drain-deadline", "--handover-ready-timeout", "--router-only")
    # The fake backends hold the health probe for 200ms. That is not padding:
    # the window in which a successor is bound but has not yet probed anything
    # is otherwise one event-loop turn wide, and a stream sampling it every few
    # hundred microseconds only caught it in 3 runs out of 8 -- a detector that
    # misses a real 503-serving regression five times out of eight is not a
    # detector. With a 200ms probe the window is wide enough that a successor
    # which binds before it is ready is caught every time, and one that probes
    # before binding (the correct order) simply binds 200ms later and loses
    # nothing. A real /v1/models on a loaded GPU node is not instant either.
    scenario = make_scenario("zeroloss", backends=2, extra_args=HANDOVER_ARGS, probe_delay=0.2)
    scenario.start()
    scenario.wait_health_status("ok")

    # Two streams on the one listening socket. The data plane exercises the
    # proxy path; the health plane opens a connection every 20ms and is what
    # makes a narrow unbound-port window observable at all -- a refusal lasting
    # a few milliseconds is invisible to a stream whose requests each take
    # longer than that.
    stream = RequestStream(scenario.port, workers=8, pace=0.01).start()
    probe = HealthProbe(scenario.port, scenario.fleet.deployment, interval=0.02).start()
    events = []
    predecessor_pid = scenario.gateway.pid
    try:
        # A has to have been up for several seconds before the handover, for two
        # reasons: the baseline has to be big enough that "nothing broke" means
        # something, and `uptime_s` is what separates A from B in the health
        # samples.
        started = time.time()
        wait_for("200 baseline requests", lambda: len(stream.served()) >= 200, timeout=45.0)
        while time.time() - started < 5.0:
            time.sleep(0.2)
        baseline = len(stream.served())

        # Work that is deliberately still in flight when the handover starts:
        # the drain in step 8 is the only thing that can save these.
        long_results = []
        long_threads = []
        for index in range(4):

            def hold(index=index):
                long_results.append(
                    post(scenario.port, 90000 + index, "long-%d" % index, delay=3.0)
                )

            thread = threading.Thread(target=hold, daemon=True)
            thread.start()
            long_threads.append(thread)
        time.sleep(0.4)

        # Drop the pacing for the duration of the handover: the workers now
        # reconnect as fast as the gateway answers, roughly one new connection
        # every few hundred microseconds. On its own this was not enough (it
        # raised detection of a bound-before-ready successor from 3-in-10 to
        # 3-in-8, measured) -- the probe delay above is what actually makes that
        # deterministic -- but it is what covers the *unbound* window, which is
        # narrower still and has no equivalent knob. No backlog risk: eight
        # worker threads can have at most eight connections outstanding against
        # a listen backlog of 100.
        stream.pace = 0.0

        t_post = time.time()
        events.append((t_post, "POST /_gateway/handover"))
        status, payload = scenario.start_handover()
        assert status == 202, (
            "CONTRACT 'HTTP surface (exact)': POST /_gateway/handover must answer "
            '202 {"status":"started","successor_pid":N}; got %s %r' % (status, payload)
        )
        assert isinstance(payload, dict) and isinstance(payload.get("successor_pid"), int), (
            "CONTRACT 'HTTP surface (exact)': the 202 body must carry successor_pid; got %r"
            % (payload,)
        )
        successor_pid = payload["successor_pid"]
        assert successor_pid != scenario.gateway.pid, (
            "the handover reported the predecessor's own pid (%d) as the successor; "
            "no new process was started" % successor_pid
        )

        rc, t_exit = scenario.gateway.wait_exit(timeout=120.0)
        if rc is None:
            state = scenario.handover_state()
            pytest.fail(
                "the predecessor (pid %s) never exited 120s after POST /_gateway/handover.\n"
                "CONTRACT step 8: A drains in-flight requests (bounded by "
                "--handover-drain-deadline) then exits 0.\n"
                "GET /_gateway/handover said: %r\n%s\n%s"
                % (predecessor_pid, state, stream_report(stream, events), scenario.diagnostics()),
                pytrace=False,
            )
        events.append((t_exit, "predecessor pid %d exited rc=%s" % (predecessor_pid, rc)))

        # Keep the stream on the port well past the moment A left, so a
        # successor that binds and then falls over is caught rather than
        # celebrated.
        time.sleep(2.0)
        stream.pace = 0.01
        time.sleep(2.0)
        for thread in long_threads:
            thread.join(timeout=30.0)
    finally:
        stream.stop()
        probe.stop()

    report = stream_report(stream, events)

    # -- the property ----------------------------------------------------
    refused = stream.refused()
    assert not refused, (
        "%d connection(s) were REFUSED during the handover. CONTRACT: 'zero refused "
        "connections'. This is the port being unbound for a moment -- step 6 closed A's "
        "listener before B had bound it, or B died after binding.\n%s" % (len(refused), report)
    )
    lost = stream.lost()
    assert not lost, (
        "%d request(s) were accepted and then LOST in flight during the handover. "
        "CONTRACT: 'no dropped in-flight request'. A connection that is accepted and then "
        "dies without a usable reply means the predecessor was killed rather than drained "
        "(work item B), or the successor dropped work it had already accepted.\n%s"
        % (len(lost), report)
    )
    errored = stream.errored()
    assert not errored, (
        "%d request(s) got an HTTP error during the handover. The likely one is 503: the "
        "contract has B bind the port at step 4 and only then evaluate readiness at step 5, "
        "so any connection the kernel deals to B before its first health probe lands routes "
        "to an empty fleet. Fix by making the successor complete one health sweep before "
        "`asyncio.start_server`, the way `fleet.discover()` already runs before it.\n%s"
        % (len(errored), report)
    )
    # The control plane shares the listening socket, so it is part of the same
    # promise -- and at one connection every 20ms it is the more sensitive of
    # the two detectors for a short unbound-port window.
    assert not probe.refused(), (
        "%d /_gateway/health connection(s) were refused during the handover, out of %d. "
        "The port was unbound for a moment even though the data stream did not happen to "
        "hit it.\nfirst: %r\n%s"
        % (len(probe.refused()), len(probe.outcomes), probe.refused()[:3], report)
    )
    assert not probe.lost(), (
        "%d /_gateway/health connection(s) were accepted and then dropped during the "
        "handover.\nfirst: %r\n%s" % (len(probe.lost()), probe.lost()[:3], report)
    )
    for out in long_results:
        assert out.ok, (
            "a request that was in flight when the handover started did not finish: %r\n"
            "CONTRACT step 8: A drains in-flight requests before exiting.\n%s" % (out, report)
        )
    assert len(long_results) == 4, "only %d of 4 in-flight probes returned" % len(long_results)

    # -- the successor really took over ----------------------------------
    assert rc == 0, "CONTRACT step 8: A 'then exits 0'; it exited %s.\n%s" % (rc, report)
    assert pid_alive(successor_pid), (
        "the successor (pid %d) is not running after the handover completed.\n%s"
        % (successor_pid, report)
    )
    after = [r for r in stream.served() if r.t_start > t_exit]
    assert len(after) >= 20, (
        "only %d request(s) were served after the predecessor exited, so there is no "
        "evidence the successor is carrying traffic.\n%s" % (len(after), report)
    )
    before = [r for r in stream.served() if r.t_done and r.t_done < t_post]
    assert before, "no requests were served before the handover; the baseline is empty"
    assert baseline >= 300

    # -- generations, as far as the contract's surface exposes them -------
    samples = probe.snapshot()
    foreign = [s for s in samples if s["deployment"] not in (None, scenario.fleet.deployment)]
    assert not foreign, (
        "port %d was answered by a gateway that is not ours: %r. SO_REUSEPORT lets a "
        "foreign process join the listen group silently; this run's numbers cannot be "
        "trusted." % (scenario.port, foreign[:3])
    )
    labelled = probe.generations(boundary=t_post)
    saw_a = [s for t, gen, s in labelled if gen == "A" and t < t_post]
    saw_b = [s for t, gen, s in labelled if gen == "B" and t > t_exit]
    assert saw_a, "no health sample was attributable to the predecessor before the handover"
    assert saw_b, (
        "no health sample after the predecessor exited reports a fresh uptime, so the "
        "process now on the port is not a newly started successor.\n%s" % report
    )


# ---------------------------------------------------------------------------
# Pins have to survive the process boundary, or the handover costs every
# conversation its prefix cache.
# ---------------------------------------------------------------------------
def test_conversation_affinity_survives_handover(make_scenario):
    """A conversation pinned before the handover lands on the same backend after.

    This is what `--router-state` inheritance is for: contract step 2 flushes
    the pin table synchronously *before* B is spawned, and item D's
    `flush_now()` is what makes that write happen regardless of the `dirty`
    flag.
    """
    require_cli("--handover-drain-deadline", "--handover-ready-timeout", "--router-only")
    scenario = make_scenario("affinity", backends=2, extra_args=HANDOVER_ARGS)
    scenario.start()
    scenario.wait_health_status("ok")

    conversations = ["affinity-%02d" % index for index in range(10)]
    pinned = {}
    for index, convo in enumerate(conversations):
        out = post(scenario.port, index, convo)
        assert out.ok, "pre-handover request for %s failed: %r" % (convo, out)
        pinned[convo] = out.backend
    # Stable inside one process first; otherwise "it moved" after the handover
    # would not mean anything.
    for index, convo in enumerate(conversations):
        out = post(scenario.port, 100 + index, convo)
        assert out.ok and out.backend == pinned[convo], (
            "conversation %s moved (%s -> %s) without any handover"
            % (convo, pinned[convo], out.backend)
        )
    assert len(set(pinned.values())) >= 2, (
        "all %d conversations pinned to one backend (%s), so 'they stayed put' would be "
        "true of any implementation. The placement policy is not spreading them."
        % (len(pinned), sorted(set(pinned.values())))
    )

    t_post = time.time()
    status, payload = scenario.start_handover()
    assert status == 202, (status, payload)
    successor_pid = payload["successor_pid"]
    rc, t_exit = scenario.gateway.wait_exit(timeout=120.0)
    assert rc is not None, (
        "the predecessor never exited; see test_handover_serves_every_request for the "
        "contract clause.\n%s" % scenario.diagnostics()
    )
    assert pid_alive(successor_pid), "successor died during the handover"

    # Direct evidence for step 2: the state file was written by the handover
    # itself, not left over from a periodic save. `router_state_loop` only runs
    # every 30s, so nothing else could have touched it inside this test.
    state_path = scenario.fleet.router_state
    assert os.path.exists(state_path), (
        "the handover never wrote --router-state (%s).\nCONTRACT step 2: 'A flushes the "
        "router pin table to --router-state (synchronously)', performed by work item D's "
        "`flush_now()`. Note it is a coroutine: calling it without `await` is a silent "
        "no-op that looks exactly like this.\n%s" % (state_path, scenario.diagnostics())
    )
    written_at = os.path.getmtime(state_path)
    assert written_at >= t_post - 1.0, (
        "--router-state was last written %.1fs BEFORE the handover request, so step 2 did "
        "not flush; the successor booted from a stale table.\n%s"
        % (t_post - written_at, scenario.diagnostics())
    )

    # Re-queried in REVERSE order, which is the whole point. `least_conversations`
    # breaks ties by job id, so a successor that booted with an EMPTY pin table
    # re-pins the first conversation it sees onto the lowest-sorting backend --
    # and asking in the original order would reproduce the predecessor's
    # alternating assignment exactly, making this test pass against a gateway
    # that inherited nothing at all. (It did, until this comment existed.)
    moved = {}
    for index, convo in enumerate(reversed(conversations)):
        out = post(scenario.port, 200 + index, convo)
        assert out.ok, "post-handover request for %s failed: %r\n%s" % (
            convo,
            out,
            scenario.diagnostics(),
        )
        if out.backend != pinned[convo]:
            moved[convo] = (pinned[convo], out.backend)
    assert not moved, (
        "%d of %d conversations lost their pin across the handover: %r\n"
        "CONTRACT step 2: 'A flushes the router pin table to --router-state "
        "(synchronously) ... MUST precede step 3 so B boots with a current pin table', "
        "and work item D's `flush_now()` is what performs it. Either the flush did not "
        "happen before the successor was spawned, or the successor did not load the "
        "state file (%s).\n%s"
        % (len(moved), len(conversations), moved, state_path, scenario.diagnostics())
    )


# ---------------------------------------------------------------------------
# Abort: the successor never becomes ready.
# ---------------------------------------------------------------------------
def test_handover_aborts_when_successor_never_ready(make_scenario):
    """B cannot reach readiness -> B is killed and A is still in service.

    Readiness at step 5 is "listening + fleet dir read + >=1 healthy backend",
    so the only lever the contract offers for forcing a successor to never
    become ready is to leave no healthy backend. That necessarily makes A
    unhealthy too -- both processes read one fleet directory -- so the test
    restores the fleet afterwards and checks A is still the process serving it.
    """
    require_cli("--handover-ready-timeout", "--router-only")
    ready_timeout = 6
    scenario = make_scenario(
        "abort",
        backends=2,
        extra_args=[
            "--handover-drain-deadline",
            "30",
            "--handover-ready-timeout",
            str(ready_timeout),
        ],
    )
    scenario.start()
    scenario.wait_health_status("ok")
    original_pid = scenario.gateway.pid

    scenario.set_fleet_health(False)
    scenario.wait_health_status("no_backend", timeout=25.0)

    status, payload = scenario.start_handover()
    if status == 503:
        pytest.fail(
            "POST /_gateway/handover answered 503 with an unhealthy fleet. CONTRACT reserves "
            "503 for 'spawning failed', and step 5 is what handles a successor that cannot "
            "become ready -- refusing to start the handover at all skips the abort path "
            "entirely. body=%r" % (payload,),
            pytrace=False,
        )
    assert status == 202, (status, payload)
    successor_pid = payload["successor_pid"]
    assert successor_pid != original_pid

    wait_for(
        "the successor (pid %s) to be killed after --handover-ready-timeout=%ds"
        % (successor_pid, ready_timeout),
        lambda: not pid_alive(successor_pid),
        timeout=ready_timeout + 40.0,
    )
    assert scenario.gateway.alive(), (
        "CONTRACT step 5: on a ready-timeout A 'aborts the handover, kills B, and stays in "
        "service'. A exited with %s instead.\n%s"
        % (scenario.gateway.proc.returncode, scenario.diagnostics())
    )
    assert scenario.gateway.pid == original_pid

    # And it is still the *same* process actually serving, not a corpse holding
    # a socket.
    scenario.set_fleet_health(True)
    health = scenario.wait_health_status("ok", timeout=30.0)
    assert health["deployment"] == scenario.fleet.deployment
    survivors = [post(scenario.port, 300 + index, "post-abort-%d" % index) for index in range(12)]
    bad = [out for out in survivors if not out.ok]
    assert not bad, "after an aborted handover the gateway is not serving cleanly again: %r\n%s" % (
        bad[:4],
        scenario.diagnostics(),
    )
    assert scenario.gateway.alive() and scenario.gateway.pid == original_pid
    # uptime_s proves it is the original process and not a silently restarted one.
    assert health["uptime_s"] >= ready_timeout, (
        "the process answering on the port reports uptime %ss, less than the ready timeout "
        "it was supposed to have survived; this is not the original gateway" % health["uptime_s"]
    )


# ---------------------------------------------------------------------------
# Lock mutual exclusion.
# ---------------------------------------------------------------------------
def test_supervisor_lock_is_mutually_exclusive(make_scenario):
    """A second `--router-only` process must not supervise until the lock is free.

    The observable proxy for "performed a supervisor action", with no SLURM
    anywhere: `revive_dead_backends` acts on a backend that is unhealthy and
    whose registration says it exited, and the action it takes is
    `<--serve-sh> restart <run_dir>`. `--serve-sh` here is a stub that records
    `os.getppid()`, and because `run_serve_sh` uses
    `asyncio.create_subprocess_exec` that ppid is exactly the gateway process
    that acted. So the log answers "which of the two supervised" directly.
    """
    require_cli("--router-only", "--supervisor-lock")
    scenario = make_scenario("lock", backends=2)
    lock_path = scenario.fleet.supervisor_lock

    # One healthy backend so both processes can route, one that is alive,
    # heart-beating, and saying it died -- which is what revive acts on.
    sick = scenario.fleet.backends[1]
    sick.set_healthy(False)
    sick.state = "server exited with status 1"

    scenario.gateway.argv.extend(["--supervisor-lock", lock_path])
    scenario.start()
    scenario.wait_health_status("ok")
    first_pid = scenario.gateway.pid

    first_calls = wait_for(
        "the lock holder to take a supervisor action",
        lambda: [c for c in scenario.fleet.serve_sh_calls() if c[0] == first_pid],
        timeout=30.0,
    )
    assert any("restart" in argv for _, _, argv in first_calls), first_calls

    second = scenario.add_gateway(
        "B-router-only",
        extra_args=[
            "--router-only",
            "--supervisor-lock",
            lock_path,
            # A separate state file: a shared one cannot say which process wrote it.
            "--router-state",
            scenario.fleet.router_state + ".routeronly",
        ],
    )
    second.start()
    second.wait_serving(timeout=30.0)
    second_pid = second.pid
    assert second_pid != first_pid

    # Ten supervisor intervals (--supervisor-interval 0.5) with the lock held by
    # somebody else. A router-only process that is going to act would have acted.
    deadline = time.time() + 5.0
    while time.time() < deadline:
        trespass = [c for c in scenario.fleet.serve_sh_calls() if c[0] == second_pid]
        assert not trespass, (
            "the --router-only gateway (pid %d) took a supervisor action while pid %d held "
            "%s: %r\nCONTRACT 'Supervisor lock': 'supervisor_loop performs lifecycle actions "
            "ONLY while the lock is held'. Two gateways acting on one dead instance is the "
            "failure the lock exists to prevent.\n%s"
            % (second_pid, first_pid, lock_path, trespass, scenario.diagnostics())
        )
        time.sleep(0.2)
    # Recorded, not asserted. Item D gates `router_state_loop` on the lock, but
    # that loop sleeps a hardcoded 30s (gateway.py:2380), so "no file yet" is
    # also what an ungated implementation looks like this early. Making the
    # interval configurable would turn this into a real second assertion.
    scenario.note(
        "router-only process had written its own --router-state file after 5s unlocked: %s"
        % os.path.exists(scenario.fleet.router_state + ".routeronly")
    )

    # The kernel releases a flock when the holder dies -- SIGKILL rather than
    # SIGTERM precisely so nothing but process death is doing the releasing.
    scenario.gateway.proc.kill()
    scenario.gateway.proc.wait(timeout=10.0)

    took_over = wait_for(
        "the --router-only gateway (pid %d) to acquire the lock and supervise after pid %d "
        "died. CONTRACT work item A: 'A --router-only process must poll to acquire the lock "
        "so it takes over automatically when A releases it.'" % (second_pid, first_pid),
        lambda: [c for c in scenario.fleet.serve_sh_calls() if c[0] == second_pid],
        timeout=45.0,
    )
    assert any("restart" in argv for _, _, argv in took_over), took_over
    assert second.alive()


# ---------------------------------------------------------------------------
# Drain, both directions: it must give up on time, and it must not give up early.
# ---------------------------------------------------------------------------
def test_drain_deadline_expiry_abandons_and_exits(make_scenario):
    """A request that outlives the deadline is abandoned, the process still exits.

    CONTRACT item B: 'On deadline expiry, log how many requests were abandoned
    and exit anyway.'
    """
    require_cli("--handover-drain-deadline", "--handover-ready-timeout", "--router-only")
    deadline_s = 2
    hold_s = 12.0
    scenario = make_scenario(
        "drain-expiry",
        backends=2,
        extra_args=["--handover-drain-deadline", str(deadline_s), "--handover-ready-timeout", "45"],
    )
    scenario.start()
    scenario.wait_health_status("ok")

    stuck = []
    thread = threading.Thread(
        target=lambda: stuck.append(
            post(scenario.port, 1, "stuck", delay=hold_s, read_timeout=hold_s + 20.0)
        ),
        daemon=True,
    )
    thread.start()
    arrivals = wait_for(
        "the long request to reach a backend",
        lambda: scenario.slow_request_started(hold_s),
        timeout=20.0,
    )
    t_arrived = arrivals[0]["t"]

    status, payload = scenario.start_handover()
    assert status == 202, (status, payload)
    successor_pid = payload["successor_pid"]

    rc, t_exit = scenario.gateway.wait_exit(timeout=hold_s + 60.0)
    thread.join(timeout=hold_s + 25.0)
    assert rc is not None, (
        "the predecessor never exited at all, so the drain deadline did not bound "
        "anything.\n%s" % scenario.diagnostics()
    )
    assert rc == 0, (
        "CONTRACT item B: on deadline expiry the process must 'exit anyway' with 0; it "
        "exited %s.\n%s" % (rc, scenario.diagnostics())
    )
    assert t_exit < t_arrived + hold_s - 2.0, (
        "the predecessor waited %.1fs for a request that --handover-drain-deadline=%ds was "
        "supposed to abandon (the backend does not reply for %.0fs). The deadline is not "
        "bounding the drain.\n%s" % (t_exit - t_arrived, deadline_s, hold_s, scenario.diagnostics())
    )

    log = scenario.gateway.log_text()
    match = DRAIN_EXPIRY.search(log)
    assert match, (
        "no drain-expiry line in the predecessor's log. CONTRACT item B logs\n"
        "  'drain deadline of Ns expired with N request(s) still in flight'\n"
        "and the count is the point: without it an operator cannot tell a clean drain from "
        "an abandonment. Log tail:\n%s" % scenario.gateway.log_tail(80)
    )
    assert int(match.group(2)) >= 1, (
        "the drain-expiry line reports %s abandoned requests while one was demonstrably "
        "still in flight: %r" % (match.group(2), match.group(0))
    )
    scenario.note("abandoned client request came back as: %r" % (stuck[0] if stuck else None))
    assert pid_alive(successor_pid), "the successor is not running after the drain expired"


def test_drain_is_not_held_by_a_handler_whose_client_disconnected(make_scenario):
    """A dead client's handler must not hold the drain hostage.

    This test used to assert the opposite -- that after a client RST the
    predecessor stays alive until the backend finishes -- because the handler
    was blind to the client's death and drain-by-inflight-counter was the only
    honest signal. The abandonment watcher removed the blindness: a buffered
    request whose client hangs up is now detected at the disconnect, the
    upstream is closed (arming the serving stack's own abort chain), and the
    handler returns. Keeping the old assertion would pin the gateway to
    serving corpses.

    What this must still guarantee, and asserts:

    * the abandoned handler ends promptly, so the drain -- and with it the
      handover -- finishes long before the backend's stall would have, and
    * the predecessor still exits 0 without tripping its drain deadline.

    The live-client half of the drain contract -- no request whose client is
    still there may be dropped -- is `test_handover_serves_every_request`.
    """
    require_cli("--handover-drain-deadline", "--handover-ready-timeout", "--router-only")
    hold_s = 12.0
    scenario = make_scenario(
        "drain-rst",
        backends=2,
        extra_args=["--handover-drain-deadline", "30", "--handover-ready-timeout", "45"],
    )
    scenario.start()
    scenario.wait_health_status("ok")

    body = json.dumps({"model": "fake", "messages": [{"role": "user", "content": "rst"}]}).encode()
    t_reset = send_and_reset(
        scenario.port,
        headers=[
            ("x-api-key", API_USER),
            (CONVERSATION_HEADER, "rst-convo"),
            (DELAY_HEADER, "%.1f" % hold_s),
        ],
        body=body,
        linger_after=0.4,
    )
    arrivals = wait_for(
        "the reset request to reach a backend",
        lambda: scenario.slow_request_started(hold_s),
        timeout=20.0,
    )
    t_arrived = arrivals[0]["t"]
    expected_finish = t_arrived + hold_s
    assert t_reset < expected_finish - 1.0, (
        "the client reset after the handler had already finished; the test proved nothing"
    )

    status, payload = scenario.start_handover()
    assert status == 202, (status, payload)
    assert time.time() < expected_finish - 1.0, (
        "the handover started after the stalled backend had already replied; nothing "
        "abandoned was in flight to (not) wait for"
    )

    rc, t_exit = scenario.gateway.wait_exit(timeout=hold_s + 90.0)
    assert rc is not None, "the predecessor never exited.\n%s" % scenario.diagnostics()
    assert t_exit < expected_finish - 1.0, (
        "the predecessor exited %.2fs AFTER the stalled backend replied: the abandoned "
        "handler sat out the full %.0fs stall, so the client's disconnect was never "
        "noticed and the drain waited for a request nobody can receive.\n%s"
        % (t_exit - expected_finish, hold_s, scenario.diagnostics())
    )
    assert rc == 0, "the predecessor should exit 0; it exited %s" % rc
    assert not DRAIN_EXPIRY.search(scenario.gateway.log_text()), (
        "the drain hit its deadline over a handler whose client was already gone:\n%s"
        % scenario.gateway.log_tail(60)
    )
