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
"""What a run records about the code and chat template it serves.

`commit=` alone said nothing about the uncommitted edits production has run, so
each run and attempt also records whether the tree was dirty, a digest of the
tracked changes, a digest over every tensorrt_llm/**/*.py actually on disk, and
the chat template's digest. These drive `serve.sh provenance`, the same function
the launcher calls, against a scratch repository and checkpoint directory.
"""

import hashlib
import json
import os
import shutil
import subprocess

import pytest

SERVE_SH = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "serve.sh")
EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="needs git")


def git(repo, *args):
    # Pinned so a developer's global config (signing, hooks) cannot fail the commit.
    return subprocess.run(
        [
            "git",
            "-C",
            repo,
            "-c",
            "user.name=t",
            "-c",
            "user.email=t@t",
            "-c",
            "commit.gpgsign=false",
        ]
        + list(args),
        check=True,
        capture_output=True,
    ).stdout


def write(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as handle:
        handle.write(data)


def provenance(repo, model):
    """Run the collector; it must exit 0 whatever it finds."""
    done = subprocess.run(
        ["bash", SERVE_SH, "provenance", str(repo), str(model)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert done.returncode == 0, done.stdout + done.stderr
    values = {}
    for line in done.stdout.splitlines():
        key, sep, value = line.partition("=")
        assert sep, "not a key=value line: %r" % line
        values[key] = value
    return values


def manifest_digest(repo):
    """The digest computed independently, as `sha256sum` would list the files."""
    paths = []
    for dirpath, _, filenames in os.walk(os.path.join(repo, "tensorrt_llm")):
        paths += [os.path.join(dirpath, name) for name in filenames if name.endswith(".py")]
    lines = b""
    for path in sorted(os.path.relpath(path, repo) for path in paths):
        with open(os.path.join(repo, path), "rb") as handle:
            digest = hashlib.sha256(handle.read()).hexdigest()
        lines += ("%s  %s\n" % (digest, path)).encode()
    return hashlib.sha256(lines).hexdigest(), len(paths)


@pytest.fixture
def repo(tmp_path):
    root = str(tmp_path / "repo")
    write(os.path.join(root, "tensorrt_llm", "__init__.py"), b"VERSION = 1\n")
    write(os.path.join(root, "tensorrt_llm", "serve", "server.py"), b"def serve():\n    pass\n")
    write(os.path.join(root, "tensorrt_llm", "libs", "notes.txt"), b"not python\n")
    write(os.path.join(root, "README.md"), b"readme\n")
    git(root, "init", "-q")
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", "initial")
    return root


@pytest.fixture
def model(tmp_path):
    root = str(tmp_path / "model")
    os.makedirs(root)
    return root


def test_a_clean_tree_records_its_head_and_digests(repo, model):
    template = b"{% for message in messages %}{{ message.content }}{% endfor %}"
    write(os.path.join(model, "chat_template.jinja"), template)
    write(os.path.join(model, "tokenizer_config.json"), json.dumps({"chat_template": "x"}).encode())

    got = provenance(repo, model)

    assert got["commit"] == git(repo, "rev-parse", "HEAD").decode().strip()
    assert got["dirty"] == "no"
    assert got["dirty_count"] == "0"
    assert got["diff_sha256"] == EMPTY_SHA256
    assert (got["py_manifest_sha256"], int(got["py_files"])) == manifest_digest(repo)
    assert int(got["py_files"]) == 2
    # The jinja file wins over tokenizer_config.json, as it does when loading.
    assert got["chat_template"] == "chat_template.jinja"
    assert got["chat_template_sha256"] == hashlib.sha256(template).hexdigest()


def test_uncommitted_edits_are_visible(repo, model):
    clean = provenance(repo, model)
    write(os.path.join(repo, "tensorrt_llm", "serve", "server.py"), b"def serve():\n    return 1\n")
    write(os.path.join(repo, "tensorrt_llm", "serve", "new_module.py"), b"X = 2\n")

    got = provenance(repo, model)

    assert got["commit"] == clean["commit"], "the commit alone cannot tell these runs apart"
    assert got["dirty"] == "yes"
    assert got["dirty_count"] == "2"
    tracked = git(repo, "diff", "--binary", "--no-color", "--no-ext-diff", "HEAD")
    assert got["diff_sha256"] == hashlib.sha256(tracked).hexdigest() != EMPTY_SHA256
    # The untracked module is in no diff, but it is imported all the same.
    assert (got["py_manifest_sha256"], int(got["py_files"])) == manifest_digest(repo)
    assert got["py_manifest_sha256"] != clean["py_manifest_sha256"]


def test_manifest_digest_is_reproducible_with_coreutils(repo, model):
    """The documented one-liner gives the same digest, so anyone can check a run."""
    if not (shutil.which("sha256sum") and shutil.which("xargs")):
        pytest.skip("needs coreutils")
    by_hand = subprocess.run(
        "find tensorrt_llm -name '*.py' -type f -print0 | LC_ALL=C sort -z "
        "| xargs -0 sha256sum | sha256sum",
        shell=True,
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.split()[0]

    assert provenance(repo, model)["py_manifest_sha256"] == by_hand


def test_tokenizer_config_template_is_used_without_a_jinja_file(repo, model):
    template = "{{ bos_token }}{% for m in messages %}{{ m.content }}{% endfor %}"
    write(
        os.path.join(model, "tokenizer_config.json"),
        json.dumps({"chat_template": template, "model_max_length": 8}).encode(),
    )

    got = provenance(repo, model)

    assert got["chat_template"] == "tokenizer_config.json:chat_template"
    assert got["chat_template_sha256"] == hashlib.sha256(template.encode()).hexdigest()


def test_a_checkpoint_without_a_template_says_none(repo, model):
    write(os.path.join(model, "tokenizer_config.json"), b"{}")

    got = provenance(repo, model)

    assert got["chat_template"] == "none"
    assert got["chat_template_sha256"] == "none"


def test_what_cannot_be_read_is_unknown_and_never_fatal(tmp_path):
    not_a_repo = tmp_path / "plain"
    not_a_repo.mkdir()
    broken_model = tmp_path / "model"
    broken_model.mkdir()
    write(str(broken_model / "tokenizer_config.json"), b"{not json")

    got = provenance(not_a_repo, broken_model)

    for key in ("origin", "commit", "dirty", "dirty_count", "diff_sha256"):
        assert got[key] == "unknown", (key, got)
    assert got["py_manifest_sha256"] == "unknown"
    assert got["chat_template"] == "unknown"
    assert got["chat_template_sha256"] == "unknown"

    assert provenance(tmp_path / "missing", tmp_path / "missing")["commit"] == "unknown"
