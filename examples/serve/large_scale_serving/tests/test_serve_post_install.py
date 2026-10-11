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
"""server.post_install: commands every node runs after the repo install.

Kimi-K3 needs packages the repo does not pin -- fla-core, einops -- and a
FlashInfer fork. The fork is the reason for the ordering: `pip install -e .`
reinstalls the pinned flashinfer on every attempt, so anything installed
before it is silently replaced. These drive `serve.sh launch` against stubbed
srun/ssh (see launcher_stubs.py) and read back what it ran, in what order.
"""

import os
import subprocess
import sys

import pytest
import yaml
from launcher_stubs import (
    POST_INSTALL_NAME,
    calls,
    copy_fleetctl,
    make_attempt,
    make_stub_bin,
    serve_sh,
    stub_env,
    write_deployment,
    write_engine_config,
)


@pytest.fixture
def world(tmp_path):
    root = str(tmp_path)
    bin_dir, log = make_stub_bin(root)
    return root, stub_env(root, bin_dir, log), log


def srun_steps(log):
    """Each srun as the step it is: install, post_install or serve."""
    steps = []
    for argv in calls(log, "srun"):
        if "bash" not in argv:
            continue
        payload = argv[argv.index("bash") :]
        script = payload[2] if len(payload) > 2 else ""
        if "pip install -e ." in script:
            steps.append("install")
        elif len(payload) > 3 and payload[3] == POST_INSTALL_NAME:
            steps.append("post_install")
        elif "trtllm-serve" in script:
            steps.append("serve")
        else:
            steps.append("other")
    return steps


def post_install_argv(log):
    for argv in calls(log, "srun"):
        if "bash" in argv:
            payload = argv[argv.index("bash") :]
            if len(payload) > 3 and payload[3] == POST_INSTALL_NAME:
                return argv
    return None


def test_the_commands_run_on_every_node_after_the_install(world):
    root, env, log = world
    marker = os.path.join(root, "ran.txt")
    commands = [
        "echo fla-core >> %s" % marker,
        "echo einops >> %s" % marker,
        # Whatever directory the commands start in, it is the checkout the
        # install step just installed -- the same one a relative wheel path
        # would be written against.
        'basename "$PWD" >> %s' % marker,
    ]
    deployment = write_deployment(root, server={"install_repo": True, "post_install": commands})
    attempt = make_attempt(root)

    done = serve_sh(env, "launch", "--yaml", deployment, "--attempt-dir", attempt)
    assert done.returncode == 0, done.stdout + done.stderr

    assert srun_steps(log) == ["install", "post_install", "serve"], (
        "post_install must run after `pip install -e .` (which reinstalls the pinned "
        "flashinfer) and before the workers start; sruns were %s" % srun_steps(log)
    )
    with open(marker) as handle:
        assert handle.read().split() == ["fla-core", "einops", "repo"]
    argv = post_install_argv(log)
    # Once per node, inside the job's container, the same placement as the install.
    assert "--ntasks-per-node" in argv and argv[argv.index("--ntasks-per-node") + 1] == "1"
    assert "--container-name" in argv
    payload = argv[argv.index("bash") :]
    assert payload[4:] == [os.path.join(root, "repo")] + commands

    with open(os.path.join(attempt, "post_install.log")) as handle:
        logged = handle.read()
    for command in commands:
        assert command in logged, "each command is logged as it runs:\n%s" % logged
    with open(os.path.join(attempt, "launch_cmd.sh")) as handle:
        recorded = handle.read()
    assert "# post_install:" in recorded and "fla-core" in recorded, recorded


def test_the_commands_run_even_without_a_repo_install(world):
    root, env, log = world
    marker = os.path.join(root, "ran.txt")
    deployment = write_deployment(
        root, server={"install_repo": False, "post_install": ["echo only >> %s" % marker]}
    )
    attempt = make_attempt(root)

    done = serve_sh(env, "launch", "--yaml", deployment, "--attempt-dir", attempt)
    assert done.returncode == 0, done.stdout + done.stderr

    assert srun_steps(log) == ["post_install", "serve"], srun_steps(log)
    with open(marker) as handle:
        assert handle.read().split() == ["only"]


def test_a_failing_command_fails_the_attempt_and_names_the_command(world):
    root, env, log = world
    marker = os.path.join(root, "ran.txt")
    commands = [
        "echo first >> %s" % marker,
        "test -e /no/such/flashinfer-fork.whl",
        "echo never >> %s" % marker,
    ]
    deployment = write_deployment(root, server={"install_repo": True, "post_install": commands})
    attempt = make_attempt(root)

    done = serve_sh(env, "launch", "--yaml", deployment, "--attempt-dir", attempt)
    output = done.stdout + done.stderr

    assert done.returncode != 0, "a failed post-install step must fail the launch:\n%s" % output
    assert "serve" not in srun_steps(log), "the workers started after a failed post_install"
    assert "post_install" in output and commands[1] in output, (
        "the failure has to name the command that failed:\n%s" % output
    )
    with open(marker) as handle:
        assert handle.read().split() == ["first"], "the commands after the failing one ran"


def test_no_commands_means_no_extra_step(world):
    root, env, log = world
    deployment = write_deployment(root, server={"install_repo": True})
    attempt = make_attempt(root)

    done = serve_sh(env, "launch", "--yaml", deployment, "--attempt-dir", attempt)
    assert done.returncode == 0, done.stdout + done.stderr
    assert srun_steps(log) == ["install", "serve"]
    assert not os.path.exists(os.path.join(attempt, "post_install.log"))


@pytest.mark.parametrize("bad", ["pip install einops", [""], [["nested"]], {"a": 1}])
def test_post_install_must_be_a_list_of_commands(world, bad):
    root, env, _ = world
    deployment = write_deployment(root, server={"post_install": bad})
    done = serve_sh(env, "resolve", "--yaml", deployment)
    assert done.returncode != 0
    assert "server.post_install" in done.stderr, done.stderr


def test_fleetctl_renders_the_commands_into_each_instance(world):
    """Fleet yaml -> fleetctl render -> serve.sh launch, the path production takes."""
    root, env, log = world
    marker = os.path.join(root, "ran.txt")
    commands = ["echo fla-core >> %s" % marker, "echo einops >> %s" % marker]
    write_engine_config(os.path.join(root, "agg.yaml"))
    fleet = {
        "defaults": {
            "cluster_name": "kffleet",
            "repo_dir": os.path.join(root, "repo"),
            "model": {"name": "glm5.3", "path": os.path.join(root, "model"), "tool_parser": "p"},
            "slurm": {"account": "a", "partition": "p", "time": "01:00:00", "gpus_per_node": 1},
            "container": {"image": os.path.join(root, "image.sqsh"), "mounts": ["/x:/x"]},
            "server": {"install_repo": True, "capture": False, "post_install": commands},
            "trace": {"root": os.path.join(root, "var")},
        },
        "instances": [{"name": "i00", "port": 8400, "server": "agg.yaml"}],
    }
    os.makedirs(os.path.join(root, "repo"), exist_ok=True)
    os.makedirs(os.path.join(root, "model"), exist_ok=True)
    fleet_yaml = os.path.join(root, "fleet.yaml")
    with open(fleet_yaml, "w") as handle:
        yaml.safe_dump(fleet, handle)
    fleetctl = copy_fleetctl(root)

    done = subprocess.run(
        [sys.executable, fleetctl, "--config", fleet_yaml, "render"],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert done.returncode == 0, done.stdout + done.stderr
    generated = os.path.join(root, "generated", "i00.yaml")
    with open(generated) as handle:
        rendered = yaml.safe_load(handle)
    assert rendered["server"]["post_install"] == commands

    resolved = serve_sh(env, "resolve", "--yaml", generated)
    assert resolved.returncode == 0, resolved.stderr
    assert "CFG_POST_INSTALL=(" in resolved.stdout, resolved.stdout

    attempt = make_attempt(root)
    done = serve_sh(env, "launch", "--yaml", generated, "--attempt-dir", attempt)
    assert done.returncode == 0, done.stdout + done.stderr
    assert srun_steps(log) == ["install", "post_install", "serve"]
    with open(marker) as handle:
        assert handle.read().split() == ["fla-core", "einops"]
