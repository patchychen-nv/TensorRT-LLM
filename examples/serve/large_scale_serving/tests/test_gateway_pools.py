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
"""Pools: one gateway, several fleet directories, one routing pool.

The data-gen gateway fronts the GLM fleet. Kimi-K3 joins it as a second
*pool*: its backends register into a fleet directory of their own, and a
fleet config of its own supervises them. A new conversation may go to any
backend of any pool, whatever model a request names, and a conversation's pin
keeps it on one backend whichever pool that backend is in. Once served, a
conversation is only ever re-placed inside its own pool: pools are models, and
another pool's backend would continue it on a model that did not write its
history (`--pool-affinity`).

So the properties pinned here are, in this order: both pools' backends take
traffic; affinity spans pools, retries and re-placement stay inside one; the
status reports say which pool each backend came from; each pool is supervised
through its own fleet config, or not at all; and a handover keeps every pool,
refusing a successor that would silently drop one.

Real `gateway.py` processes against fake backends, like the handover tests.
"""

import json
import os
import time

import pytest
from handover_harness import (
    API_USER,
    CONVERSATION_HEADER,
    RequestStream,
    Scenario,
    control,
    gateway_cli_options,
    pid_alive,
    request,
    stream_report,
    wait_for,
)

K3 = "kimi-k3"
HANDOVER_ARGS = ["--handover-drain-deadline", "30", "--handover-ready-timeout", "45"]
# The router keys a header-borne conversation id as "hdr:<id>".
PIN_PREFIX = "hdr:"

FLEETCTL_STUB = '''#!/usr/bin/env python3
"""Stands in for fleetctl: records how the gateway called it, and nothing else."""
import os
import sys
import time

with open(%(log)r, "a") as handle:
    handle.write("%%d\\t%%.6f\\t%%s\\n" %% (os.getppid(), time.time(), "\\x1f".join(sys.argv[1:])))
print("ok")
'''


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


def require_pools(*options):
    """Fail naming the missing feature, not with an argparse exit deep in a wait."""
    missing = [name for name in ("--pool",) + options if name not in gateway_cli_options()]
    if missing:
        pytest.fail(
            "gateway.py does not implement pools: no %s option (has: %s)"
            % (", ".join(missing), " ".join(sorted(gateway_cli_options()))),
            pytrace=False,
        )


def pool_spec(scenario, name=K3, config=None):
    spec = "%s=%s" % (name, scenario.fleet.pool_dirs[name])
    return spec + ("," + config if config else "")


def pooled(make_scenario, label, glm=2, k3=2, config=None, extra_args=(), start=True):
    """A default pool of `glm` fake backends plus a kimi-k3 pool of `k3`."""
    require_pools()
    scenario = make_scenario(label, backends=glm, pools={K3: k3}, extra_args=extra_args)
    scenario.gateway.argv.extend(["--pool", pool_spec(scenario, config=config)])
    if start:
        scenario.start()
        wait_healthy(scenario)
    return scenario


def ids(scenario, pool=None):
    return {backend.job_id for backend in scenario.fleet.members(pool)}


def fleet_status(port):
    status, data, _ = control(port, "GET", "/_gateway/fleet")
    assert status == 200 and isinstance(data, dict), (status, data)
    return data


def wait_healthy(scenario, jobs=None):
    """Wait until every fake backend (or just `jobs`) is probed healthy.

    Placement is only deterministic once the router can see all of them.
    """
    want = set(jobs) if jobs is not None else {b.job_id for b in scenario.fleet.backends}

    def ready():
        backends = fleet_status(scenario.port).get("backends") or {}
        return all((backends.get(job) or {}).get("healthy") for job in want)

    wait_for("backends %s to be probed healthy" % sorted(want), ready, timeout=25.0)


def ask(port, path="/v1/responses", model=None, seq=0, convo=None):
    headers = [("x-api-key", API_USER), ("x-test-seq", str(seq))]
    if convo is not None:
        headers.append((CONVERSATION_HEADER, convo))
    payload = {"input": "turn %d" % seq}
    if model is not None:
        payload["model"] = model
    body = json.dumps(payload).encode()
    return request(port, path=path, body=body, headers=headers, seq=seq, convo=convo or "")


def pin(port, convo, job_id):
    status, data, _ = control(
        port, "POST", "/_gateway/pin", {"conversation": PIN_PREFIX + convo, "backend": job_id}
    )
    assert status == 200, data


def place(port, convos, seq0, model=None):
    """Open each conversation once; return conversation -> the backend that took it."""
    homes = {}
    for n, convo in enumerate(convos):
        out = ask(port, model=model, seq=seq0 + n, convo=convo)
        assert out.ok, out
        homes[convo] = out.backend
    return homes


def handover(scenario, body=None):
    status, payload = scenario.start_handover_with(body or {})
    assert status == 202, (status, payload)
    successor = payload["successor_pid"]
    rc, _ = scenario.gateway.wait_exit(timeout=120.0)
    assert rc == 0, "the predecessor did not hand over cleanly (rc=%s)\n%s" % (
        rc,
        scenario.diagnostics(),
    )
    assert pid_alive(successor), "the successor died during the handover"
    return successor


# ---------------------------------------------------------------------------
# One routing pool
# ---------------------------------------------------------------------------
def test_every_pools_backends_take_traffic(make_scenario):
    """Both fleets serve, and the model a request names does not choose between them."""
    scenario = pooled(make_scenario, "pool-shared")
    port = scenario.port
    every = ids(scenario) | ids(scenario, K3)

    # Placement is least_conversations with ties broken by job id, so eight
    # new conversations land two on each backend: fakejob1, fakejob2,
    # kimik3job1, kimik3job2, and round again. Alternating the body's model
    # means each model's conversations land in BOTH pools.
    k3_named = place(port, ["named-k3-%d" % n for n in range(0, 8, 2)], 100, model=K3)
    glm_named = place(port, ["named-glm-%d" % n for n in range(1, 8, 2)], 200, model="glm5.3")
    for named, homes in (("kimi-k3", k3_named), ("glm5.3", glm_named)):
        landed = set(homes.values())
        assert landed & ids(scenario) and landed & ids(scenario, K3), (
            "conversations naming model %s landed only on %s; routing must not depend "
            "on the model" % (named, sorted(landed))
        )

    # No path is special either: a /kimi-k3/ prefix is relayed as sent.
    out = ask(port, "/kimi-k3/v1/responses", seq=300)
    assert out.ok and out.backend in every, out
    paths = [row["path"] for row in scenario.backend_access_log(out.backend) if row["seq"] == "300"]
    assert paths and set(paths) == {"/kimi-k3/v1/responses"}, paths

    stream = RequestStream(port, workers=6).start()
    try:
        wait_for("150 served", lambda: len(stream.served()) >= 150, timeout=45.0)
    finally:
        stream.stop()
    report = stream_report(stream)
    assert not stream.refused() and not stream.lost() and not stream.errored(), report
    assert set(stream.by_backend()) == every, (
        "a shared stream was served by %s; every backend of both pools should take a share\n%s"
        % (sorted(stream.by_backend()), report)
    )

    # The access log says which pool answered, and only for an extra pool.
    log = scenario.gateway.log_text()
    k3_lines = [line for line in log.splitlines() if " backend=kimik3job" in line]
    glm_lines = [line for line in log.splitlines() if " backend=fakejob" in line]
    assert k3_lines and all("pool=kimi-k3" in line for line in k3_lines), k3_lines[:3]
    assert glm_lines and not any("pool=" in line for line in glm_lines), glm_lines[:3]


def test_affinity_holds_across_pools(make_scenario):
    """A conversation stays on its backend, whichever pool that is and whatever it names."""
    scenario = pooled(make_scenario, "pool-affinity")
    port = scenario.port
    convos = ["affinity-%02d" % n for n in range(8)]
    homes = place(port, convos, 100)
    assert set(homes.values()) == ids(scenario) | ids(scenario, K3), homes

    for turn, model in enumerate((K3, "glm5.3", None)):
        for n, convo in enumerate(convos):
            out = ask(port, model=model, seq=1000 * (turn + 1) + n, convo=convo)
            assert out.ok and out.backend == homes[convo], (
                "%s moved from %s to %s when its request named model %r"
                % (convo, homes[convo], out.backend, model)
            )
    routing = fleet_status(port)["routing"]
    assert routing["pinned"] == len(convos) and routing["rehomed"] == 0, routing

    # A hand-placed pin needs no pool and may point into either one.
    for convo, job_id in (("by-hand-k3", "kimik3job2"), ("by-hand-glm", "fakejob1")):
        pin(port, convo, job_id)
        for n in range(3):
            assert ask(port, model="glm5.3", seq=5000 + n, convo=convo).backend == job_id


def test_a_retry_stays_in_the_pool(make_scenario):
    """The backend refusing a request is routed around, but never into another model."""
    scenario = pooled(make_scenario, "pool-retry", glm=1, k3=1)
    port = scenario.port
    (k3,) = scenario.fleet.members(K3)

    pin(port, "staying", k3.job_id)
    first = ask(port, model=K3, seq=1, convo="staying")
    assert first.ok and first.backend == k3.job_id, first

    k3.stop()  # heartbeats continue: the gateway still believes in it
    out = ask(port, model=K3, seq=2, convo="staying")
    assert not out.ok and out.backend is None, (
        "a K3 conversation whose only K3 backend refused it must not be retried on GLM; "
        "got %r" % (out,)
    )


def test_without_pool_affinity_a_retry_can_cross_pools(make_scenario):
    """--no-pool-affinity: the refused request goes to whichever pool has room."""
    scenario = pooled(
        make_scenario, "pool-retry-any", glm=1, k3=1, extra_args=["--no-pool-affinity"]
    )
    port = scenario.port
    (k3,) = scenario.fleet.members(K3)
    (glm,) = scenario.fleet.members(None)

    pin(port, "crossing", k3.job_id)
    first = ask(port, model=K3, seq=1, convo="crossing")
    assert first.ok and first.backend == k3.job_id, first

    k3.stop()
    out = ask(port, model=K3, seq=2, convo="crossing")
    assert out.ok and out.backend == glm.job_id, out


# ---------------------------------------------------------------------------
# What an operator sees
# ---------------------------------------------------------------------------
def test_fleet_status_and_health_report_each_pool(make_scenario):
    scenario = pooled(make_scenario, "pool-status")
    port = scenario.port
    k3, glm = ids(scenario, K3), ids(scenario)
    homes = place(port, ["status-%d" % n for n in range(4)], 100)

    fleet = fleet_status(port)
    for job, entry in fleet["backends"].items():
        assert entry.get("pool") == (K3 if job in k3 else "default"), (job, entry)
    assert sorted(fleet["routing"]["accepting"]) == sorted(k3 | glm), fleet["routing"]
    pools = fleet.get("pools") or {}
    assert set(pools) == {"default", K3}, pools
    assert pools[K3]["backends"] == 2 and pools[K3]["healthy"] == 2, pools[K3]
    assert sorted(pools[K3]["serving"]) == sorted(k3)
    assert pools[K3]["conversations"] == 2 and pools["default"]["conversations"] == 2, pools
    assert pools[K3]["fleet_dir"] == scenario.fleet.pool_dirs[K3]
    assert pools[K3]["supervised"] is False and pools["default"]["supervised"] is True
    assert pools[K3]["active"] in k3 and pools["default"]["active"] in glm, pools

    status, health, _ = control(port, "GET", "/_gateway/health")
    assert status == 200 and health["status"] == "ok", health
    assert health["active"] in k3 | glm, health
    assert health["pools"][K3]["status"] == "ok" and health["pools"]["default"]["status"] == "ok"

    # With every GLM backend down the gateway still serves, and says which pool
    # is empty. New conversations go to K3; one GLM was serving waits for GLM.
    for backend in scenario.fleet.members(None):
        backend.set_healthy(False)

    def glm_gone():
        _, data, _ = control(port, "GET", "/_gateway/health")
        return data if data["pools"]["default"]["status"] == "no_backend" else None

    health = wait_for("the default pool to report no backend", glm_gone, timeout=25.0)
    assert health["status"] == "ok" and health["active"] in k3, health
    out = ask(port, model="glm5.3", seq=200, convo="status-new")
    assert out.ok and out.backend in k3, out
    on_glm = [convo for convo, home in homes.items() if home in glm]
    assert on_glm, homes
    held = ask(port, model="glm5.3", seq=201, convo=on_glm[0])
    assert held.status == 503 and held.backend is None, held
    routing = fleet_status(port)["routing"]
    assert routing["pool_affinity"] is True and routing["held_for_pool"] >= 1, routing


# ---------------------------------------------------------------------------
# Supervision is per pool
# ---------------------------------------------------------------------------
def test_an_unconfigured_pool_is_routed_but_never_supervised(make_scenario):
    """No fleet config means nothing may act on the pool's jobs -- said once."""
    scenario = pooled(make_scenario, "pool-unsupervised", start=False)
    sick_k3 = scenario.fleet.members(K3)[0]
    sick_k3.set_healthy(False)
    sick_k3.state = "server exited with status 1"
    sick_glm = scenario.fleet.members(None)[1]
    sick_glm.set_healthy(False)
    sick_glm.state = "server exited with status 1"
    scenario.start()
    scenario.wait_health_status("ok", timeout=25.0)

    # The default pool is supervised exactly as before...
    wait_for(
        "the default pool to revive its exited backend",
        lambda: [c for c in scenario.fleet.serve_sh_calls() if sick_glm.run_dir in c[2]],
        timeout=30.0,
    )
    # ...and the unconfigured pool is not, however many sweeps go by.
    time.sleep(3.0)
    touched = [c for c in scenario.fleet.serve_sh_calls() if sick_k3.run_dir in c[2]]
    assert not touched, "a pool with no fleet config was acted on: %r" % touched
    assert scenario.gateway.log_text().count("has no fleet config") == 1, (
        "the unsupervised pool should be reported exactly once:\n%s" % scenario.gateway.log_tail()
    )
    # Unsupervised is not unrouted: its healthy backend takes conversations.
    healthy_k3 = scenario.fleet.members(K3)[1].job_id
    wait_healthy(scenario, [healthy_k3, scenario.fleet.members(None)[0].job_id])
    homes = place(scenario.port, ["unsupervised-%d" % n for n in range(4)], 100)
    assert healthy_k3 in homes.values(), homes


def test_a_configured_pool_is_supervised_with_its_own_fleet_config(make_scenario):
    require_pools()
    scenario = make_scenario("pool-supervised", backends=2, pools={K3: 2})
    fleetctl_log = os.path.join(scenario.root, "fleetctl_calls.log")
    fleetctl = os.path.join(scenario.root, "fleetctl_stub.py")
    with open(fleetctl, "w") as handle:
        handle.write(FLEETCTL_STUB % {"log": fleetctl_log})
    os.chmod(fleetctl, 0o755)
    k3_config = os.path.join(scenario.root, "fleet_k3.yaml")
    with open(k3_config, "w") as handle:
        handle.write("defaults: {}\ninstances: []\n")
    scenario.gateway.argv.extend(
        ["--fleetctl", fleetctl, "--pool", pool_spec(scenario, config=k3_config)]
    )
    sick_k3 = scenario.fleet.members(K3)[1]
    sick_k3.set_healthy(False)
    sick_k3.state = "server exited with status 1"
    scenario.start()
    scenario.wait_health_status("ok", timeout=25.0)

    calls = wait_for(
        "the gateway to revive the pool's exited backend",
        lambda: [c for c in scenario.fleet.serve_sh_calls() if sick_k3.run_dir in c[2]],
        timeout=30.0,
    )
    assert calls[0][0] == scenario.gateway.pid and calls[0][2].startswith("restart"), calls

    def fleetctl_calls():
        try:
            with open(fleetctl_log) as handle:
                return [line.rstrip("\n").split("\t")[2].split("\x1f") for line in handle]
        except OSError:
            return []

    # The startup check proves the pool's own recovery could run.
    assert ["--config", k3_config, "status"] in fleetctl_calls(), fleetctl_calls()
    assert "has no fleet config" not in scenario.gateway.log_text()


# ---------------------------------------------------------------------------
# Handover
# ---------------------------------------------------------------------------
def test_a_handover_keeps_the_pools_and_their_pins(make_scenario):
    scenario = pooled(make_scenario, "pool-handover", extra_args=HANDOVER_ARGS)
    port = scenario.port
    convos = ["handover-%02d" % n for n in range(8)]
    homes = place(port, convos, 100)
    assert set(homes.values()) == ids(scenario) | ids(scenario, K3), homes

    handover(scenario)

    # Reversed, so a successor that inherited no pins would not reproduce the
    # predecessor's alternating placement by accident.
    moved = {}
    for n, convo in enumerate(reversed(convos)):
        got = ask(port, seq=200 + n, convo=convo).backend
        if got != homes[convo]:
            moved[convo] = (homes[convo], got)
    assert not moved, "pins lost across the handover: %r\n%s" % (moved, scenario.diagnostics())
    pools = {job: entry.get("pool") for job, entry in fleet_status(port)["backends"].items()}
    assert pools == {
        **{job: "default" for job in ids(scenario)},
        **{job: K3 for job in ids(scenario, K3)},
    }, pools


def test_a_handover_picks_up_pools_from_the_pools_file(make_scenario):
    """The production activation path.

    A live gateway's successor is spawned with its predecessor's own command
    line, so a pool cannot be added to a running deployment by flag. The
    pools file beside the users file is read by every generation at startup:
    write it, hand over, and the successor discovers the new fleet -- no gap,
    no restart of the job.
    """
    require_pools("--pools-file")
    scenario = make_scenario("pool-file", backends=2, pools={K3: 2}, extra_args=HANDOVER_ARGS)
    scenario.start()
    wait_healthy(scenario, ids(scenario))
    port = scenario.port

    # Before: the K3 fleet registers, but into a directory nobody reads.
    assert set(fleet_status(port)["backends"]) == ids(scenario)
    homes = place(port, ["before-%d" % n for n in range(4)], 100)
    assert set(homes.values()) == ids(scenario), homes

    pools_file = os.path.join(os.path.dirname(scenario.fleet.users_file), "pools.conf")
    with open(pools_file, "w") as handle:
        handle.write("# extra fleets, one --pool spec per line\n%s\n" % pool_spec(scenario))

    handover(scenario)
    wait_healthy(scenario)
    status = fleet_status(port)
    assert {job for job, e in status["backends"].items() if e.get("pool") == K3} == ids(
        scenario, K3
    ), status["backends"]
    homes = place(port, ["after-%d" % n for n in range(8)], 200)
    assert set(homes.values()) & ids(scenario, K3), (
        "after the handover new conversations still never reached the K3 fleet: %r" % homes
    )


def test_a_handover_never_drops_a_pool_by_accident(make_scenario):
    """A successor missing a pool would silently drop that fleet's backends.

    The successor's pools come from its predecessor's command line and from
    the pools file as it stands at the handover. Delete the file, or rename
    the pool in it by mistake, and the successor comes up without the K3
    fleet: its backends stop taking traffic, every conversation pinned there
    re-homes onto GLM, and nothing supervises those jobs any more. The
    predecessor knows which pools it serves, so it refuses to hand over to a
    successor that lacks one -- unless the operator retires it on purpose.
    """
    require_pools("--pools-file")
    scenario = make_scenario("pool-guard", backends=2, pools={K3: 2}, extra_args=HANDOVER_ARGS)
    pools_file = os.path.join(os.path.dirname(scenario.fleet.users_file), "pools.conf")
    with open(pools_file, "w") as handle:
        handle.write("%s\n" % pool_spec(scenario))
    scenario.start()
    wait_healthy(scenario)
    port = scenario.port
    k3_job = scenario.fleet.members(K3)[0].job_id
    original = scenario.gateway.pid

    os.unlink(pools_file)  # the mistake
    status, payload = scenario.start_handover()
    assert status == 202, (status, payload)
    successor = payload["successor_pid"]
    wait_for(
        "the successor without the pool to be refused and killed",
        lambda: not pid_alive(successor),
        timeout=60.0,
    )
    assert scenario.gateway.alive() and scenario.gateway.pid == original, (
        "the predecessor handed over to a successor that does not serve %s\n%s"
        % (K3, scenario.diagnostics())
    )
    _, state = scenario.handover_state()
    assert state["phase"] == "aborted" and K3 in state["detail"], state
    pin(port, "still-k3", k3_job)
    out = ask(port, seq=1, convo="still-k3")
    assert out.ok and out.backend == k3_job, "the predecessor must go on routing the pool: %r" % (
        out,
    )

    # Retiring the pool is allowed -- but only when the operator says so.
    handover(scenario, {"drop_pools": [K3]})
    wait_healthy(scenario, ids(scenario))
    assert set(fleet_status(port)["backends"]) == ids(scenario)
    homes = place(port, ["after-drop-%d" % n for n in range(4)], 100)
    assert set(homes.values()) == ids(scenario), homes
