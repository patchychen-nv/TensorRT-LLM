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
"""No deployment turns off the cyclic GC in a process that parses requests.

Validating a Responses request leaves reference cycles behind: pydantic turns the
`Iterable[...]` fields of the input item types into ValidatorIterators that point back
at the dict holding them, once per union member it tries. Only the cyclic GC frees
them. With TRTLLM_SERVER_DISABLE_GC=1 (trtllm-serve) or TRTLLM_DISAGG_SERVER_DISABLE_GC=1
(the disaggregated frontend's fleet workers, whose code default is "1") every request
leaked its parsed input. On 10-09 that was 10-30 GiB/h per frontend and 7-21 GiB/h per
ctx/gen HTTP server, on course to run a ctx head node out of memory within a day.

serve.sh's ENV_DEFAULTS must therefore keep both on, and a deployment YAML must not
switch either off. The executor ranks (TRTLLM_WORKER_DISABLE_GC) never see a request
body and are not covered here.
"""

import ast
import glob
import os
import re

import pytest
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
SERVE_SH = os.path.join(HERE, os.pardir, "serve.sh")
DEPLOYMENTS = sorted(glob.glob(os.path.join(HERE, os.pardir, "deployments", "*.yaml")))
REQUEST_PARSING = ("TRTLLM_SERVER_DISABLE_GC", "TRTLLM_DISAGG_SERVER_DISABLE_GC")


def env_defaults():
    text = open(SERVE_SH).read()
    match = re.search(r"^ENV_DEFAULTS = (\{.*?^\})", text, re.M | re.S)
    assert match, "ENV_DEFAULTS not found in serve.sh"
    return ast.literal_eval(match.group(1))


def env_blocks(node, path=""):
    """Every `env:` mapping in a deployment, with where it sits."""
    if isinstance(node, dict):
        for key, value in node.items():
            here = f"{path}.{key}" if path else str(key)
            if key == "env" and isinstance(value, dict):
                yield here, value
            else:
                yield from env_blocks(value, here)
    elif isinstance(node, list):
        for index, value in enumerate(node):
            yield from env_blocks(value, f"{path}[{index}]")


def test_serve_sh_keeps_the_gc_on_where_requests_are_parsed():
    defaults = env_defaults()
    for name in REQUEST_PARSING:
        assert defaults.get(name) == "0", (
            f"serve.sh ENV_DEFAULTS sets {name}={defaults.get(name)!r}"
        )


def test_there_are_deployments_to_check():
    assert DEPLOYMENTS


@pytest.mark.parametrize("deployment", DEPLOYMENTS, ids=os.path.basename)
def test_no_deployment_turns_the_gc_off_where_requests_are_parsed(deployment):
    with open(deployment) as handle:
        config = yaml.safe_load(handle) or {}
    offending = [
        f"{where}.{name}={env[name]!r}"
        for where, env in env_blocks(config)
        for name in REQUEST_PARSING
        if name in env and str(env[name]).strip().lower() in ("1", "true", "yes", "on")
    ]
    assert not offending, f"{os.path.basename(deployment)} disables the cyclic GC: {offending}"
