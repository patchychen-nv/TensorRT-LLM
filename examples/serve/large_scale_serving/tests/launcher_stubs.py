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
"""Just enough of Slurm for serve.sh and fleetctl to run on a laptop. No assertions here.

serve.sh drives a deployment through srun, ssh, squeue, scontrol and sbatch.
Each is replaced by a stub on PATH that records how it was called, so a test
can read off what the launcher *would* have run on the cluster -- in which
order, with which arguments -- and still exercise the real resolver, the real
launch script and the real controller loop.

One srun payload is executed rather than recorded: the server.post_install
step, recognised by the `post_install` name serve.sh gives its script (`$0`).
Running it is the only way to show that the commands run, in order, and that
a failing one stops the launch. The install and serve payloads are recorded
only -- one is `pip install -e .`, the other `trtllm-serve`.
"""

import json
import os
import shutil
import stat
import subprocess
import sys

import yaml

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
HERE = os.path.dirname(TESTS_DIR)
SERVE_SH = os.path.join(HERE, "serve.sh")
FLEETCTL = os.path.join(HERE, "fleetctl")

# The `$0` serve.sh passes to the post-install script inside srun.
POST_INSTALL_NAME = "post_install"

SRUN = r"""#!/usr/bin/env python3
import json, os, subprocess, sys
argv = sys.argv[1:]
with open(os.environ["STUB_LOG"], "a") as handle:
    handle.write(json.dumps({"cmd": "srun", "argv": argv}) + "\n")
if "bash" not in argv:
    sys.exit(0)
i = argv.index("bash")
if argv[i + 1 : i + 2] != ["-lc"] or len(argv) < i + 4:
    sys.exit(0)
script, name, rest = argv[i + 2], argv[i + 3], argv[i + 4 :]
if name == %(name)r:
    sys.exit(subprocess.call(["bash", "-c", script, name] + rest))
# The serve step: from here on the launch is past everything it sets up.
marker = os.environ.get("STUB_SERVE_MARKER")
if marker and "trtllm-serve" in script:
    open(marker, "w").close()
sys.exit(0)
"""

RECORDER = r"""#!/usr/bin/env python3
import json, os, sys
with open(os.environ["STUB_LOG"], "a") as handle:
    handle.write(json.dumps({"cmd": %(cmd)r, "argv": sys.argv[1:]}) + "\n")
sys.stdout.write(%(out)r)
"""

# Running until the launch has reached its serve step, then gone: what a
# controller sees when its allocation is taken away.
SQUEUE = r"""#!/usr/bin/env python3
import json, os, sys
with open(os.environ["STUB_LOG"], "a") as handle:
    handle.write(json.dumps({"cmd": "squeue", "argv": sys.argv[1:]}) + "\n")
lines = os.environ.get("STUB_SQUEUE_LINES")
if lines is not None:
    sys.stdout.write(lines)
    sys.exit(0)
marker = os.environ.get("STUB_SERVE_MARKER")
if not (marker and os.path.exists(marker)):
    print("RUNNING")
"""


def _write(path, text):
    with open(path, "w") as handle:
        handle.write(text)
    os.chmod(path, os.stat(path).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def make_stub_bin(root):
    """Create the stub directory; returns (bin_dir, log_path)."""
    bin_dir = os.path.join(root, "bin")
    os.makedirs(bin_dir, exist_ok=True)
    log = os.path.join(root, "stub_calls.jsonl")
    open(log, "w").close()
    # serve.sh runs its resolver as `python3 -`, and it needs PyYAML: point
    # python3 at the interpreter running the tests, which has it.
    _write(os.path.join(bin_dir, "python3"), '#!/bin/sh\nexec "%s" "$@"\n' % sys.executable)
    _write(os.path.join(bin_dir, "srun"), SRUN % {"name": POST_INSTALL_NAME})
    _write(os.path.join(bin_dir, "ssh"), RECORDER % {"cmd": "ssh", "out": ""})
    _write(
        os.path.join(bin_dir, "sbatch"),
        RECORDER % {"cmd": "sbatch", "out": "Submitted batch job 4343\n"},
    )
    _write(os.path.join(bin_dir, "scontrol"), RECORDER % {"cmd": "scontrol", "out": "node1\n"})
    _write(
        os.path.join(bin_dir, "getent"),
        RECORDER % {"cmd": "getent", "out": "10.0.0.7 STREAM node1\n"},
    )
    _write(os.path.join(bin_dir, "squeue"), SQUEUE)
    # The controller polls every 2s and cleanup_workers waits 5s between TERM
    # and KILL. Neither wait is what a test is about, so both shrink.
    real_sleep = shutil.which("sleep") or "/bin/sleep"
    _write(os.path.join(bin_dir, "sleep"), '#!/bin/sh\nexec "%s" 0.05\n' % real_sleep)
    return bin_dir, log


def stub_env(root, bin_dir, log, **extra):
    env = dict(os.environ)
    env["PATH"] = bin_dir + os.pathsep + env.get("PATH", "")
    env["STUB_LOG"] = log
    env["STUB_SERVE_MARKER"] = os.path.join(root, "serve_started")
    env["USER"] = "tester"
    env["SLURM_JOB_ID"] = "4242"
    env.pop("SLURM_JOB_NODELIST", None)
    env.pop("SLURM_JOB_END_TIME", None)
    env.update({key: str(value) for key, value in extra.items()})
    return env


def calls(log, cmd=None):
    with open(log) as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    return [row["argv"] for row in rows if cmd is None or row["cmd"] == cmd]


def write_engine_config(path, tp=1):
    with open(path, "w") as handle:
        yaml.safe_dump({"tensor_parallel_size": tp, "backend": "pytorch"}, handle)
    return path


def write_deployment(root, name="deployment.yaml", model_name="glm5.3", server=None, trace=None):
    """A single aggregated, one-node, one-GPU deployment serve.sh accepts."""
    os.makedirs(os.path.join(root, "repo"), exist_ok=True)
    os.makedirs(os.path.join(root, "model"), exist_ok=True)
    engine = write_engine_config(os.path.join(root, "engine.yaml"))
    cfg = {
        "cluster_name": "kffleet",
        "repo_dir": os.path.join(root, "repo"),
        "model": {
            "name": model_name,
            "path": os.path.join(root, "model"),
            "tool_parser": "fake_parser",
        },
        "slurm": {
            "account": "acct",
            "partition": "part",
            "time": "01:00:00",
            "gpus_per_node": 1,
        },
        "container": {"image": os.path.join(root, "image.sqsh"), "mounts": [root + ":" + root]},
        "server": dict({"config": engine, "port": 8400, "capture": False}, **(server or {})),
        "trace": dict({"root": os.path.join(root, "var")}, **(trace or {})),
    }
    path = os.path.join(root, name)
    with open(path, "w") as handle:
        yaml.safe_dump(cfg, handle, sort_keys=False)
    return path


def make_attempt(root, run_name="tester_100612_4242_kffleet_glm5.3_i00"):
    """The run directory a controller would have made: control/nodes and an attempt."""
    run_dir = os.path.join(root, "var", "2026-10", "06", run_name)
    os.makedirs(os.path.join(run_dir, "control"), exist_ok=True)
    with open(os.path.join(run_dir, "control", "nodes"), "w") as handle:
        handle.write("node1\n")
    attempt = os.path.join(run_dir, "attempt-001")
    os.makedirs(attempt, exist_ok=True)
    return attempt


def serve_sh(env, *argv, timeout=120):
    return subprocess.run(
        ["bash", SERVE_SH] + list(argv), env=env, capture_output=True, text=True, timeout=timeout
    )


def copy_fleetctl(root):
    """Fleetctl writes generated/ beside itself, so a test runs a private copy.

    serve.sh sits beside it because `submit` refuses to run without one, even
    under --dry-run.
    """
    target = os.path.join(root, "fleetctl")
    shutil.copy2(FLEETCTL, target)
    os.symlink(SERVE_SH, os.path.join(root, "serve.sh"))
    return target
