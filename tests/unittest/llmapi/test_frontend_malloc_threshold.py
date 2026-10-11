# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
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
"""A disaggregated frontend keeps request-sized blocks out of glibc's heap.

glibc raises its mmap threshold to the largest block freed, so a frontend that
parses multi-megabyte request bodies ends up carving them from a heap that never
shrinks back, and its RSS climbs with free memory. The frontend processes pin
the threshold instead (``_bound_malloc_mmap_threshold``).
"""

import ctypes
import logging
import subprocess
import sys
import textwrap
from types import SimpleNamespace
from unittest import mock

import pytest

from tensorrt_llm.commands import serve

pytestmark = pytest.mark.cpu_only


def _fake_libc(result=1):
    mallopt = mock.Mock(return_value=result)
    return SimpleNamespace(mallopt=mallopt), mallopt


class TestBoundMallocMmapThreshold:
    def test_pins_glibcs_starting_threshold_by_default(self, monkeypatch):
        monkeypatch.delenv("TRTLLM_FRONTEND_MALLOC_MMAP_THRESHOLD", raising=False)
        libc, mallopt = _fake_libc()
        monkeypatch.setattr(serve.ctypes, "CDLL", lambda name: libc)
        serve._bound_malloc_mmap_threshold()
        mallopt.assert_called_once_with(-3, 131072)

    def test_the_environment_sets_the_threshold(self, monkeypatch):
        monkeypatch.setenv("TRTLLM_FRONTEND_MALLOC_MMAP_THRESHOLD", "262144")
        libc, mallopt = _fake_libc()
        monkeypatch.setattr(serve.ctypes, "CDLL", lambda name: libc)
        serve._bound_malloc_mmap_threshold()
        mallopt.assert_called_once_with(-3, 262144)

    @pytest.mark.parametrize("value", ["0", "-1", "lots", str(64 * 1024 * 1024)])
    def test_zero_bad_or_oversized_values_leave_glibc_alone(self, monkeypatch, value):
        monkeypatch.setenv("TRTLLM_FRONTEND_MALLOC_MMAP_THRESHOLD", value)
        libc, mallopt = _fake_libc()
        monkeypatch.setattr(serve.ctypes, "CDLL", lambda name: libc)
        serve._bound_malloc_mmap_threshold()
        mallopt.assert_not_called()

    def test_a_libc_without_mallopt_is_not_an_error(self, monkeypatch):
        monkeypatch.delenv("TRTLLM_FRONTEND_MALLOC_MMAP_THRESHOLD", raising=False)

        def no_glibc(name):
            raise OSError("libc.so.6: cannot open shared object file")

        monkeypatch.setattr(serve.ctypes, "CDLL", no_glibc)
        serve._bound_malloc_mmap_threshold()  # must not raise

    def test_a_refused_mallopt_only_warns(self, monkeypatch):
        monkeypatch.delenv("TRTLLM_FRONTEND_MALLOC_MMAP_THRESHOLD", raising=False)
        libc, mallopt = _fake_libc(result=0)
        monkeypatch.setattr(serve.ctypes, "CDLL", lambda name: libc)
        serve._bound_malloc_mmap_threshold()
        mallopt.assert_called_once()

    def test_every_fleet_worker_pins_it_for_itself(self, monkeypatch):
        """The setting does not survive exec, so each worker process applies it."""
        calls = []
        monkeypatch.setattr(serve, "_bound_malloc_mmap_threshold", lambda: calls.append(1))
        monkeypatch.setattr(serve.gc, "disable", lambda: None)
        monkeypatch.delenv("WEB_CONCURRENCY", raising=False)
        monkeypatch.delenv(serve.DisaggWorkerEnvs.TLLM_DISAGG_LOG_LEVEL, raising=False)
        # The worker setup re-formats the TRT-LLM log handlers for its own
        # process; put them back for the rest of this one.
        handlers = logging.getLogger("TRT-LLM").handlers
        formatters = [handler.formatter for handler in handlers]
        try:
            serve._init_fleet_worker_process()
        finally:
            for handler, formatter in zip(handlers, formatters):
                handler.setFormatter(formatter)
        assert calls == [1]


# The glibc behaviour the fix rests on, in a fresh process each: once a 4 MiB
# mmap'ed block is freed, glibc's dynamic threshold moves above 1 MiB, so the
# next 1 MiB block comes from the heap; with the threshold pinned it is mmap'ed
# and handed back to the kernel when freed.
_GLIBC_PROBE = textwrap.dedent(
    """
    import ctypes, sys

    class Mallinfo2(ctypes.Structure):
        _fields_ = [(n, ctypes.c_size_t) for n in (
            "arena", "ordblks", "smblks", "hblks", "hblkhd", "usmblks",
            "fsmblks", "uordblks", "fordblks", "keepcost")]

    libc = ctypes.CDLL("libc.so.6")
    libc.malloc.restype = ctypes.c_void_p
    libc.malloc.argtypes = [ctypes.c_size_t]
    libc.free.argtypes = [ctypes.c_void_p]
    libc.mallinfo2.restype = Mallinfo2
    if sys.argv[1] == "pinned":
        libc.mallopt.argtypes = [ctypes.c_int, ctypes.c_int]
        assert libc.mallopt(-3, 131072) == 1
    libc.free(libc.malloc(4 << 20))
    before = libc.mallinfo2()
    block = libc.malloc(1 << 20)
    after = libc.mallinfo2()
    mmapped = after.hblkhd - before.hblkhd >= 1 << 20
    print("mmap" if mmapped else "heap")
    libc.free(block)
    """
)


def _glibc_with_mallinfo2():
    try:
        return hasattr(ctypes.CDLL("libc.so.6"), "mallinfo2")
    except OSError:
        return False


@pytest.mark.skipif(not _glibc_with_mallinfo2(), reason="needs glibc >= 2.33")
@pytest.mark.parametrize("mode, expected", [("dynamic", "heap"), ("pinned", "mmap")])
def test_a_pinned_threshold_keeps_a_large_block_out_of_the_heap(mode, expected):
    result = subprocess.run(
        [sys.executable, "-c", _GLIBC_PROBE, mode],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == expected
