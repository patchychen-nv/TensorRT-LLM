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
"""A requeued job must not overwrite the record of the run before it.

Slurm requeues a job under the same job id, and the batch script starts a new
controller. Its run directory is named <user>_<MMDDHH>_<jobid>_<name>_<label>,
so a requeue within the same hour lands on the run directory of the run that
just died. The new run then started over at attempt-001 and truncated that
attempt's launcher, server and worker logs. On 10-09 that erased the only
server-side record of two lhr runs that ended in NODE_FAIL. The job's sbatch log
had the same problem one level up: Slurm truncates --output on every requeue
unless the job asks to append.

These drive the real controller (`serve.sh run`) under one job id against the
stubbed scheduler in launcher_stubs.py, with `date` pinned so every run lands
in the same hour -- the collision always happens, so nothing here can pass by
skipping it -- and `serve.sh submit` once.
"""

import hashlib
import os
import re

import pytest
from launcher_stubs import calls, make_stub_bin, serve_sh, stub_env, write_deployment

# serve.sh names the run dir from these three `date` formats; everything else
# (`date -d`, `date +%s`, ...) still goes to the real date.
DATE_STUB = r"""#!/usr/bin/env python3
import subprocess, sys
pinned = {"+%Y-%m": "2026-10", "+%d": "09", "+%m%d%H": "100908"}
if len(sys.argv) == 2 and sys.argv[1] in pinned:
    print(pinned[sys.argv[1]])
    sys.exit(0)
sys.exit(subprocess.call(["/bin/date"] + sys.argv[1:]))
"""
RUN_DIR_NAME = "tester_100908_4242_kffleet_glm5.3_i00"


@pytest.fixture
def world(tmp_path):
    root = str(tmp_path)
    bin_dir, log = make_stub_bin(root)
    date = os.path.join(bin_dir, "date")
    with open(date, "w") as handle:
        handle.write(DATE_STUB)
    os.chmod(date, 0o755)
    env = stub_env(root, bin_dir, log, SLURM_JOB_NODELIST="node1", SLURM_JOB_END_TIME="4102444800")
    return root, env, log


def run_controller(env, deployment):
    """One controller lifetime: start, launch attempt, see the job gone, exit."""
    if os.path.exists(env["STUB_SERVE_MARKER"]):
        # The stubbed queue reports the job running again: a requeue.
        os.remove(env["STUB_SERVE_MARKER"])
    done = serve_sh(env, "run", "--yaml", deployment, "--label", "i00", timeout=180)
    output = done.stdout + done.stderr
    assert done.returncode == 0, output
    run_dir = re.search(r"^run dir: (\S+)$", done.stdout, re.M).group(1)
    assert os.path.basename(run_dir) == RUN_DIR_NAME, run_dir
    return run_dir, done.stdout


def snapshot(directory):
    """sha256 of every file under directory, by relative path."""
    out = {}
    for base, _, names in os.walk(directory):
        for name in names:
            path = os.path.join(base, name)
            if os.path.islink(path):
                continue
            with open(path, "rb") as handle:
                out[os.path.relpath(path, directory)] = hashlib.sha256(handle.read()).hexdigest()
    return out


def mark(run_dir, attempt, tag):
    """What a dying run leaves behind that a successor must not touch."""
    with open(os.path.join(run_dir, attempt, "server.log"), "a") as handle:
        handle.write(f"{tag}: slurmstepd: *** STEP CANCELLED DUE TO NODE FAILURE ***\n")
    with open(os.path.join(run_dir, "run_metadata.txt"), "a") as handle:
        handle.write(f"marker={tag}\n")


def read(path):
    with open(path) as handle:
        return handle.read()


def test_two_requeues_in_one_hour_keep_every_earlier_run(world):
    root, env, _ = world
    deployment = write_deployment(root)

    run_dir, stdout = run_controller(env, deployment)
    assert "already holds" not in stdout
    mark(run_dir, "attempt-001", "run1")
    run1_attempt = snapshot(os.path.join(run_dir, "attempt-001"))
    run1_metadata = read(os.path.join(run_dir, "run_metadata.txt"))
    run1_deployment = read(os.path.join(run_dir, "deployment.yaml"))

    _, stdout = run_controller(env, deployment)
    assert "this run starts at attempt 2" in stdout, stdout
    mark(run_dir, "attempt-002", "run2")
    run2_attempt = snapshot(os.path.join(run_dir, "attempt-002"))
    run2_metadata = read(os.path.join(run_dir, "run_metadata.txt"))

    _, stdout = run_controller(env, deployment)
    assert "this run starts at attempt 3" in stdout, stdout

    attempts = sorted(n for n in os.listdir(run_dir) if n.startswith("attempt-"))
    assert attempts == ["attempt-001", "attempt-002", "attempt-003"]
    # Every byte the two dead runs left in their attempts is still there.
    assert snapshot(os.path.join(run_dir, "attempt-001")) == run1_attempt
    assert snapshot(os.path.join(run_dir, "attempt-002")) == run2_attempt
    # Each earlier run's metadata kept under its own name, none overwritten.
    assert read(os.path.join(run_dir, "run_metadata.txt.before-run-1")) == run1_metadata
    assert read(os.path.join(run_dir, "run_metadata.txt.before-run-2")) == run2_metadata
    assert read(os.path.join(run_dir, "deployment.yaml.before-run-1")) == run1_deployment
    assert os.path.isfile(os.path.join(run_dir, "deployment.yaml.before-run-2"))
    assert "marker=" not in read(os.path.join(run_dir, "run_metadata.txt"))
    assert (
        read(os.path.join(run_dir, "control", "current_attempt_dir"))
        .strip()
        .endswith("attempt-003")
    )


def test_an_interrupted_preservation_overwrites_nothing(world):
    """A controller that died after keeping only some files of the run before it."""
    root, env, _ = world
    deployment = write_deployment(root)
    run_dir, _ = run_controller(env, deployment)
    mark(run_dir, "attempt-001", "run1")
    run1_metadata = read(os.path.join(run_dir, "run_metadata.txt"))
    # The dead successor had moved deployment.yaml aside, and nothing else.
    os.rename(
        os.path.join(run_dir, "deployment.yaml"),
        os.path.join(run_dir, "deployment.yaml.before-run-1"),
    )
    kept_deployment = read(os.path.join(run_dir, "deployment.yaml.before-run-1"))

    _, stdout = run_controller(env, deployment)
    assert "this run starts at attempt 2" in stdout, stdout
    assert read(os.path.join(run_dir, "deployment.yaml.before-run-1")) == kept_deployment
    assert read(os.path.join(run_dir, "run_metadata.txt.before-run-1")) == run1_metadata
    assert not os.path.exists(os.path.join(run_dir, "deployment.yaml.before-run-2"))
    assert "WARNING: could not keep" not in stdout


def test_a_first_run_starts_at_attempt_one_and_keeps_nothing(world):
    root, env, _ = world
    run_dir, stdout = run_controller(env, write_deployment(root))
    assert "already holds" not in stdout
    assert sorted(n for n in os.listdir(run_dir) if n.startswith("attempt-")) == ["attempt-001"]
    assert not [n for n in os.listdir(run_dir) if ".before-run-" in n]


def test_submit_asks_slurm_to_append_to_the_job_log(world):
    root, env, log = world
    done = serve_sh(env, "submit", "--yaml", write_deployment(root))
    assert done.returncode == 0, done.stdout + done.stderr
    (argv,) = calls(log, "sbatch")
    assert "--open-mode" in argv and argv[argv.index("--open-mode") + 1] == "append", argv
