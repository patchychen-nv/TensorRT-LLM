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
"""A second model's fleet has to be told apart from the first everywhere.

`model.name` is what separates them: it is in every run directory name, it
keys the registration directory the gateway reads, and it is the job name in
the queue. A K3 fleet beside the GLM one -- same cluster_name, same
trace.root -- must therefore come out as `..._kffleet_kimi-k3_<label>` run
directories registering into `_fleet/kffleet_kimi-k3/`, with its request
traces under a root of its own.

And fleetctl, which matches running jobs to configured instances, must not
mistake one model's `i00` for the other's: the gateway's preemption recovery
runs `fleetctl up` for each pool, and a conflation there means a lost instance
is never brought back because the other model's job "already has" its name.
"""

import importlib.machinery
import importlib.util
import os
import re
import subprocess
import sys

import pytest
import yaml
from launcher_stubs import (
    FLEETCTL,
    calls,
    copy_fleetctl,
    make_stub_bin,
    serve_sh,
    stub_env,
    write_deployment,
)

K3_REQUEST_ROOT_NAME = "trace_kimi_k3"


@pytest.fixture
def world(tmp_path):
    root = str(tmp_path)
    bin_dir, log = make_stub_bin(root)
    return root, stub_env(root, bin_dir, log), log


def k3_deployment(root, **server):
    return write_deployment(
        root,
        model_name="kimi-k3",
        server=dict({"capture": True, "install_repo": False}, **server),
        trace={"request_root": os.path.join(root, K3_REQUEST_ROOT_NAME)},
    )


def resolved(env, deployment):
    done = serve_sh(env, "resolve", "--yaml", deployment)
    assert done.returncode == 0, done.stdout + done.stderr
    values = {}
    for line in done.stdout.splitlines():
        key, sep, value = line.partition("=")
        if sep:
            values[key] = value.strip("'")
    return values


def test_a_kimi_k3_deployment_resolves_to_its_own_names(world):
    root, env, _ = world
    values = resolved(env, k3_deployment(root))
    assert values["CFG_NAME"] == "kffleet_kimi-k3"
    assert values["CFG_FLEET_DIR"] == os.path.join(root, "var", "_fleet", "kffleet_kimi-k3")
    assert values["CFG_REQUEST_TRACE_ROOT"] == os.path.join(root, K3_REQUEST_ROOT_NAME)


def test_the_queue_shows_which_model_a_job_serves(world):
    root, env, log = world
    done = serve_sh(env, "submit", "--yaml", k3_deployment(root), "--label", "k00")
    assert done.returncode == 0, done.stdout + done.stderr
    (argv,) = calls(log, "sbatch")
    assert argv[argv.index("--job-name") + 1] == "kffleet_kimi-k3"
    assert argv[argv.index("--output") + 1] == os.path.join(
        root, "var", "_sbatch_logs", "kffleet_kimi-k3-%j.out"
    )


def test_a_kimi_k3_run_carries_the_model_in_its_run_dir_registration_and_traces(world):
    """The controller end to end: run dir, registration file, request-trace root."""
    root, env, _ = world
    env = dict(env, SLURM_JOB_NODELIST="node1", SLURM_JOB_END_TIME="4102444800")
    deployment = k3_deployment(root)

    done = serve_sh(env, "run", "--yaml", deployment, "--label", "k00", timeout=180)
    output = done.stdout + done.stderr
    assert done.returncode == 0, output

    run_dir = re.search(r"^run dir: (\S+)$", done.stdout, re.M).group(1)
    assert re.fullmatch(r"tester_\d{6}_4242_kffleet_kimi-k3_k00", os.path.basename(run_dir)), (
        run_dir
    )
    fleet_file = re.search(r"^fleet:\s+(\S+)", done.stdout, re.M).group(1)
    assert fleet_file == os.path.join(root, "var", "_fleet", "kffleet_kimi-k3", "4242.json")

    traces = os.path.join(root, K3_REQUEST_ROOT_NAME, os.path.basename(run_dir), "attempt-001")
    assert os.path.isdir(traces), "request traces were not given the K3 root:\n%s" % output
    link = os.path.join(run_dir, "attempt-001", "request_trace")
    assert os.path.islink(link) and os.path.realpath(link) == os.path.realpath(traces)


def load_fleetctl():
    loader = importlib.machinery.SourceFileLoader("fleetctl_under_test", FLEETCTL)
    spec = importlib.util.spec_from_loader(loader.name, loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


def fleet_cfg(root, model, instances=("i00",)):
    return {
        "defaults": {
            "cluster_name": "kffleet",
            "model": {"name": model},
            "trace": {"root": os.path.join(root, "var")},
        },
        "instances": [{"name": name, "port": 8400 + n} for n, name in enumerate(instances)],
        "__path__": os.path.join(root, "fleet_%s.yaml" % model),
    }


def make_run_dir(root, job_id, model, label):
    path = os.path.join(
        root, "var", "2026-10", "06", "tester_100612_%s_kffleet_%s_%s" % (job_id, model, label)
    )
    os.makedirs(path, exist_ok=True)
    return path


def test_fleetctl_does_not_mistake_another_models_instance_for_its_own(tmp_path):
    root = str(tmp_path)
    fleetctl = load_fleetctl()
    glm, k3 = fleet_cfg(root, "glm5.3"), fleet_cfg(root, "kimi-k3")
    make_run_dir(root, "111", "glm5.3", "i00")
    make_run_dir(root, "222", "kimi-k3", "i00")

    assert fleetctl.instance_of("111", glm, ledger={}) == "i00"
    assert fleetctl.instance_of("222", k3, ledger={}) == "i00"
    assert fleetctl.instance_of("222", glm, ledger={}) is None, (
        "GLM's fleetctl claimed K3's job 222 as its own i00"
    )
    assert fleetctl.instance_of("111", k3, ledger={}) is None, (
        "K3's fleetctl claimed GLM's job 111 as its own i00"
    )


def test_recovery_brings_up_an_instance_whose_name_the_other_model_also_uses(world):
    """`fleetctl up` is what the gateway runs to recover a pool's lost instance."""
    root, env, _ = world
    for path in ("model", "repo"):
        os.makedirs(os.path.join(root, path), exist_ok=True)
    open(os.path.join(root, "image.sqsh"), "w").close()
    with open(os.path.join(root, "agg.yaml"), "w") as handle:
        yaml.safe_dump({"tensor_parallel_size": 1}, handle)
    k3 = {
        "defaults": {
            "cluster_name": "kffleet",
            "repo_dir": os.path.join(root, "repo"),
            "model": {"name": "kimi-k3", "path": os.path.join(root, "model"), "tool_parser": "p"},
            "slurm": {"account": "a", "partition": "p", "time": "01:00:00", "gpus_per_node": 1},
            "container": {"image": os.path.join(root, "image.sqsh"), "mounts": ["/x:/x"]},
            "trace": {"root": os.path.join(root, "var")},
        },
        "instances": [{"name": "i00", "port": 8400, "server": "agg.yaml"}],
    }
    config = os.path.join(root, "fleet_k3.yaml")
    with open(config, "w") as handle:
        yaml.safe_dump(k3, handle)
    # GLM's i00 is running; K3's i00 is not.
    make_run_dir(root, "111", "glm5.3", "i00")
    env = dict(env, STUB_SQUEUE_LINES="111|kffleet_glm5.3|RUNNING|1:00:00|node1\n")
    fleetctl = copy_fleetctl(root)

    done = subprocess.run(
        [sys.executable, fleetctl, "--config", config, "--dry-run", "up"],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert done.returncode == 0, done.stdout + done.stderr
    assert "already has a job" not in done.stdout, (
        "K3's i00 was skipped because GLM's i00 is running:\n%s" % done.stdout
    )
    assert "would run" in done.stdout, done.stdout


def test_a_pending_instance_is_not_submitted_twice_when_the_model_path_has_digits(world):
    """`up` recognises a job that has not started by the id it recorded at submit.

    serve.sh prints the model path before sbatch's own line, and a checkpoint
    directory such as hf-b2428a0_orig carries a run of digits. Recorded as the
    job id, it left the real job, still pending, matched to nothing, so the next
    `up` -- the gateway's preemption recovery runs the same command -- submitted
    the instance a second time.
    """
    root, env, log = world
    model = os.path.join(root, "models", "hf-b2428a0_orig")
    for path in (model, os.path.join(root, "repo")):
        os.makedirs(path, exist_ok=True)
    open(os.path.join(root, "image.sqsh"), "w").close()
    with open(os.path.join(root, "agg.yaml"), "w") as handle:
        yaml.safe_dump({"tensor_parallel_size": 1}, handle)
    k3 = {
        "defaults": {
            "cluster_name": "kffleet",
            "repo_dir": os.path.join(root, "repo"),
            "model": {"name": "kimi-k3", "path": model, "tool_parser": "p"},
            "slurm": {"account": "a", "partition": "p", "time": "01:00:00", "gpus_per_node": 1},
            "container": {"image": os.path.join(root, "image.sqsh"), "mounts": ["/x:/x"]},
            "trace": {"root": os.path.join(root, "var")},
        },
        "instances": [{"name": "i00", "port": 8400, "server": "agg.yaml"}],
    }
    config = os.path.join(root, "fleet_k3.yaml")
    with open(config, "w") as handle:
        yaml.safe_dump(k3, handle)
    fleetctl = copy_fleetctl(root)

    def up(squeue_lines):
        done = subprocess.run(
            [sys.executable, fleetctl, "--config", config, "up"],
            env=dict(env, STUB_SQUEUE_LINES=squeue_lines),
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert done.returncode == 0, done.stdout + done.stderr
        return done.stdout

    first = up("")
    assert "submitted job 4343" in first, first
    # The stub sbatch answers "Submitted batch job 4343"; that job is now pending.
    second = up("4343|kffleet_kimi-k3|PENDING|0:00|\n")
    assert "already has a job" in second, "i00 was submitted again:\n%s" % second
    assert len(calls(log, "sbatch")) == 1
