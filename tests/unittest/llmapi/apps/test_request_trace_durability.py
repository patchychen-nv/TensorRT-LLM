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
"""The request trace writer accounts for every record it is given.

A frozen copy of the trace is only as good as the claim that it holds what was
served. The writer therefore classifies each submitted record exactly once --
persisted (in its shard, newline-terminated, flushed and fsync()ed), dropped by
reason, unknown (bytes may have landed but the write or fsync holding them
failed), or still pending -- and publishes that accounting atomically beside the
shards, with each shard's durable byte range. These drive the real writer
through the failures that break that claim: a write that stops mid-record in the
middle of a UTF-8 character, a full disk, a failed fsync, a shard open that
fails with EEXIST, a stalled filesystem with a saturated queue, a writer that
dies and a successor that appends to the same file, and a sidecar write that
fails. In every case the books must
balance, no record is counted as persisted on bytes that were not fsync()ed,
and a fragment never corrupts the record written after it.
"""

import asyncio
import errno
import importlib.util
import json
import os
import threading
from pathlib import Path

import pytest

from tensorrt_llm.serve import request_trace
from tensorrt_llm.serve.request_trace import (
    _WRITER_QUEUE_SIZE,
    WRITER_SIDECAR_DIR,
    WRITER_SIDECAR_SCHEMA,
    RequestTraceWriter,
    verify_writer_shards,
)

pytestmark = pytest.mark.cpu_only

BUCKET = "2026-01-01T00"
SHARD = f"{BUCKET}/requests-w.jsonl"


def record(index, text="ok"):
    return {"session": "s", "index": index, "text": text}


def submit(writer, count, start=0, text="ok"):
    for index in range(start, start + count):
        writer._submit(BUCKET, "requests", record(index, text))


def sidecar(directory, writer):
    path = Path(directory) / WRITER_SIDECAR_DIR / f"{writer.generation}.json"
    return json.loads(path.read_text())


def balanced(document):
    """The books balance, and the shards' watermarks agree with the counts."""
    counts = document["counts"]
    accounted = (
        counts["persisted"]
        + sum(counts["dropped"].values())
        + counts["unknown"]
        + counts["pending"]
    )
    in_shards = sum(shard["persisted_records"] for shard in document["shards"])
    return (
        counts["submitted"] == accounted
        and counts["pending"] >= 0
        and in_shards == counts["persisted"]
    )


async def until(predicate, timeout=5.0):
    deadline = asyncio.get_running_loop().time() + timeout
    while not predicate():
        assert asyncio.get_running_loop().time() < deadline, "condition never became true"
        await asyncio.sleep(0.01)


async def started(directory):
    """A running writer whose shard names do not depend on the pid."""
    writer = RequestTraceWriter(str(directory), writer_suffix="-w")
    await writer.start()
    return writer


class _CutWrite:
    """A shard file whose first payload write lands only a prefix, then fails."""

    def __init__(self, handle, cut, error):
        self._handle = handle
        self._cut = cut
        self._error = error

    def write(self, data):
        if self._cut is not None and len(data) > 1:
            cut, self._cut = self._cut, None
            self._handle.write(data[:cut])
            self._handle.flush()
            raise self._error
        return self._handle.write(data)

    def __getattr__(self, name):
        return getattr(self._handle, name)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return self._handle.__exit__(*exc)


def cut_next_shard_write(monkeypatch, cut, error):
    """Make the next payload append to a shard land ``cut`` bytes, then raise."""
    real_open = Path.open
    armed = {"on": True}

    def opener(self, mode="r", *args, **kwargs):
        handle = real_open(self, mode, *args, **kwargs)
        if armed["on"] and mode == "ab" and self.suffix == ".jsonl":
            armed["on"] = False
            return _CutWrite(handle, cut, error)
        return handle

    monkeypatch.setattr(Path, "open", opener)


def fail_next_shard_opens(monkeypatch, count):
    """Make the next ``count`` opens of a shard for appending fail with EEXIST."""
    real_open = Path.open
    left = {"opens": count}

    def opener(self, mode="r", *args, **kwargs):
        if left["opens"] and mode == "ab" and self.suffix == ".jsonl":
            left["opens"] -= 1
            raise FileExistsError(errno.EEXIST, "File exists", str(self))
        return real_open(self, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", opener)
    return left


class TestBooksBalance:
    @pytest.mark.asyncio
    async def test_normal_appends_are_persisted_and_reconcile(self, tmp_path):
        writer = await started(tmp_path)
        submit(writer, 50)
        await writer.close()

        document = sidecar(tmp_path, writer)
        assert document["schema"] == WRITER_SIDECAR_SCHEMA
        assert document["closed"] is True
        assert document["counts"] == {
            "submitted": 50,
            "persisted": 50,
            "dropped": {"queue_full": 0, "unserializable": 0, "write_error": 0, "shutdown": 0},
            "unknown": 0,
            "pending": 0,
        }
        (shard,) = document["shards"]
        assert shard["file"] == SHARD
        assert shard["start_offset"] == 0
        assert shard["persisted_end"] == (tmp_path / SHARD).stat().st_size
        (check,) = verify_writer_shards(str(tmp_path), document)
        assert (check["records"], check["bad_lines"]) == (50, 0)

    @pytest.mark.asyncio
    async def test_a_record_counts_only_after_its_fsync(self, tmp_path, monkeypatch):
        writer = await started(tmp_path)
        """Persisted means fsync()ed: every persisted append was followed by one."""
        synced = []
        real_fsync = os.fsync

        def counting_fsync(fd):
            real_fsync(fd)
            synced.append(fd)

        monkeypatch.setattr(request_trace.os, "fsync", counting_fsync)
        submit(writer, 3)
        await until(lambda: writer.persisted == 3)
        assert synced, "records were counted as persisted without an fsync"

    @pytest.mark.asyncio
    async def test_sanitized_is_an_attribute_not_a_disposition(self, tmp_path):
        writer = await started(tmp_path)
        submit(writer, 1, text="cut \ud800 here")
        submit(writer, 1, start=1)
        await writer.close()

        document = sidecar(tmp_path, writer)
        assert document["attributes"] == {"sanitized": 1}
        assert document["counts"]["persisted"] == 2
        assert balanced(document)


class TestFailedWrites:
    @pytest.mark.asyncio
    async def test_a_write_cut_inside_a_utf8_character_is_unknown_not_persisted(
        self, tmp_path, monkeypatch
    ):
        writer = await started(tmp_path)
        """Bytes that landed without an fsync are never claimed, and never poison."""
        first = json.dumps(record(0, "你好"), ensure_ascii=False, separators=(",", ":")) + "\n"
        second = json.dumps(record(1, "你好"), ensure_ascii=False, separators=(",", ":")) + "\n"
        # Inside the first byte of the second record's three-byte character.
        cut = len(first.encode()) + second.encode().index("你".encode()) + 1
        cut_next_shard_write(monkeypatch, cut, OSError(errno.ENOSPC, "No space left on device"))

        submit(writer, 4, text="你好")
        await until(lambda: writer._write_error_count == 1)
        assert writer.persisted == 0
        # Record 0 landed whole and record 1 in part: neither is claimed.
        # Records 2 and 3 never reached the file.
        assert writer.unknown == 2
        assert writer.dropped["write_error"] == 2

        # The disk frees up: the next append must land whole and parse.
        submit(writer, 1, start=4, text="after")
        await writer.close()

        document = sidecar(tmp_path, writer)
        assert balanced(document)
        assert document["counts"]["persisted"] == 1
        assert document["counts"]["unknown"] == 2
        (shard,) = document["shards"]
        assert shard["fragments_isolated"] == 1
        assert shard["unknown_extents"] == [{"start": 0, "landed": cut, "records": 2}]
        (check,) = verify_writer_shards(str(tmp_path), document)
        assert (check["records"], check["bad_lines"], check["unknown_lines"]) == (1, 0, 2)

        lines = (tmp_path / SHARD).read_bytes().split(b"\n")
        # The fragment is one bad line; the record after it parses untouched.
        assert json.loads(lines[-2]) == record(4, "after")
        with pytest.raises(ValueError):
            json.loads(lines[-3])

    @pytest.mark.asyncio
    async def test_a_full_disk_drops_the_group_and_a_later_append_lands(
        self, tmp_path, monkeypatch
    ):
        writer = await started(tmp_path)
        cut_next_shard_write(monkeypatch, 0, OSError(errno.EDQUOT, "Disk quota exceeded"))
        submit(writer, 3)
        await until(lambda: writer._write_error_count == 1)
        assert writer.dropped["write_error"] == 3
        assert writer.unknown == 0

        submit(writer, 2, start=3)
        await writer.close()

        document = sidecar(tmp_path, writer)
        assert balanced(document)
        assert document["counts"]["persisted"] == 2
        assert "Disk quota exceeded" in document["errors"]["last_error"]
        (check,) = verify_writer_shards(str(tmp_path), document)
        assert (check["records"], check["bad_lines"]) == (2, 0)

    @pytest.mark.asyncio
    async def test_an_open_failing_with_eexist_is_retried_and_its_records_persist(
        self, tmp_path, monkeypatch
    ):
        """Appending opens without O_EXCL, so EEXIST is spurious; nothing was written yet."""
        monkeypatch.setattr(request_trace, "_SHARD_OPEN_RETRY_DELAYS", (0, 0, 0), raising=False)
        writer = await started(tmp_path)
        left = fail_next_shard_opens(monkeypatch, 2)
        submit(writer, 3)
        await writer.close()

        assert left["opens"] == 0
        document = sidecar(tmp_path, writer)
        assert balanced(document)
        assert document["counts"]["persisted"] == 3
        assert document["counts"]["dropped"]["write_error"] == 0
        assert document["errors"]["write_errors"] == 0
        (check,) = verify_writer_shards(str(tmp_path), document)
        assert (check["records"], check["bad_lines"]) == (3, 0)

    @pytest.mark.asyncio
    async def test_an_open_that_keeps_failing_with_eexist_drops_the_group(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(request_trace, "_SHARD_OPEN_RETRY_DELAYS", (0, 0, 0), raising=False)
        writer = await started(tmp_path)
        # The first open and all three retries.
        fail_next_shard_opens(monkeypatch, 4)
        submit(writer, 3)
        await until(lambda: writer._write_error_count == 1)
        assert writer.dropped["write_error"] == 3
        assert writer.unknown == 0

        submit(writer, 2, start=3)
        await writer.close()
        document = sidecar(tmp_path, writer)
        assert balanced(document)
        assert document["counts"]["persisted"] == 2
        assert "File exists" in document["errors"]["last_error"]
        (check,) = verify_writer_shards(str(tmp_path), document)
        assert (check["records"], check["bad_lines"]) == (2, 0)

    @pytest.mark.asyncio
    async def test_a_failed_fsync_leaves_its_records_unknown(self, tmp_path, monkeypatch):
        writer = await started(tmp_path)
        """The bytes are in the page cache, perhaps on disk: not durable, so not persisted."""
        real_fsync = os.fsync
        failures = iter([OSError(errno.EIO, "Input/output error")])

        def flaky_fsync(fd):
            if os.readlink(f"/proc/self/fd/{fd}").endswith(".jsonl"):
                error = next(failures, None)
                if error is not None:
                    raise error
            real_fsync(fd)

        monkeypatch.setattr(request_trace.os, "fsync", flaky_fsync)
        submit(writer, 3)
        await until(lambda: writer._write_error_count == 1)
        assert (writer.persisted, writer.unknown) == (0, 3)

        submit(writer, 1, start=3)
        await writer.close()
        document = sidecar(tmp_path, writer)
        assert balanced(document)
        assert (document["counts"]["persisted"], document["counts"]["unknown"]) == (1, 3)
        (check,) = verify_writer_shards(str(tmp_path), document)
        assert (check["records"], check["bad_lines"], check["unknown_lines"]) == (1, 0, 3)


class TestStallsAndShutdown:
    @staticmethod
    def stall_shard_fsync(monkeypatch):
        """Block every shard fsync until released; the sidecar's fsync still runs."""
        release = threading.Event()
        entered = threading.Event()
        real_fsync = os.fsync

        def stalled_fsync(fd):
            if os.readlink(f"/proc/self/fd/{fd}").endswith(".jsonl"):
                entered.set()
                release.wait(timeout=30)
            real_fsync(fd)

        monkeypatch.setattr(request_trace.os, "fsync", stalled_fsync)
        return entered, release

    @pytest.mark.asyncio
    async def test_a_stalled_filesystem_saturates_the_queue_and_still_balances(
        self, tmp_path, monkeypatch
    ):
        writer = await started(tmp_path)
        entered, release = self.stall_shard_fsync(monkeypatch)
        submit(writer, 1)
        await until(entered.is_set)
        # One record is in flight, stuck in fsync; fill the queue past its bound.
        submit(writer, _WRITER_QUEUE_SIZE + 7, start=1)

        # The cutoff: nothing beyond the in-flight batch has been classified,
        # and the overflow is dropped with its reason.
        books = writer.accounting()
        assert books["counts"]["dropped"]["queue_full"] == 7
        assert books["counts"]["persisted"] == 0
        assert books["counts"]["pending"] == 1 + _WRITER_QUEUE_SIZE
        assert balanced(books)

        release.set()
        await writer.close()
        document = sidecar(tmp_path, writer)
        assert balanced(document)
        assert document["counts"]["persisted"] == 1 + _WRITER_QUEUE_SIZE
        assert document["counts"]["pending"] == 0
        (check,) = verify_writer_shards(str(tmp_path), document)
        assert (check["records"], check["bad_lines"]) == (1 + _WRITER_QUEUE_SIZE, 0)

    @pytest.mark.asyncio
    async def test_a_shutdown_during_a_stall_drops_the_queue_and_claims_nothing_in_flight(
        self, tmp_path, monkeypatch
    ):
        writer = await started(tmp_path)
        monkeypatch.setattr(request_trace, "_WRITER_SHUTDOWN_TIMEOUT_SECONDS", 0.2)
        entered, release = self.stall_shard_fsync(monkeypatch)
        submit(writer, 1)
        await until(entered.is_set)
        submit(writer, 5, start=1)
        try:
            await writer.close()
            document = sidecar(tmp_path, writer)
            assert document["closed"] is True
            assert balanced(document)
            # The batch stuck in fsync may yet land, so it is not claimed; what
            # was still queued is dropped as never written.
            assert document["counts"]["unknown"] == 1
            assert document["counts"]["dropped"]["shutdown"] == 5
            assert document["counts"]["persisted"] == 0
        finally:
            release.set()
        # The abandoned write may still land once released. Its records stay
        # unknown, and no watermark moves for them: only the event loop books
        # an outcome, and that drain is gone.
        await asyncio.sleep(0.2)
        assert balanced(writer.accounting(closed=True))
        assert all(s["persisted_records"] == 0 for s in writer.accounting()["shards"])


class TestRestart:
    @pytest.mark.asyncio
    async def test_a_dead_writer_and_its_successor_stay_separable(self, tmp_path):
        first = RequestTraceWriter(str(tmp_path), writer_suffix="-w")
        await first.start()
        submit(first, 10)
        await until(lambda: first.persisted == 10)
        await first._publish(force=True)
        # SIGKILL: no close, no final sidecar, and an append cut mid-record.
        first._task.cancel()
        await asyncio.gather(first._task, return_exceptions=True)
        first._task = None
        with (tmp_path / SHARD).open("ab") as shard:
            shard.write(b'{"session":"s","index":10,"te')
        size_at_death = (tmp_path / SHARD).stat().st_size

        second = RequestTraceWriter(str(tmp_path), writer_suffix="-w")
        await second.start()
        assert second.generation != first.generation
        submit(second, 5, start=100)
        await second.close()

        dead, alive = sidecar(tmp_path, first), sidecar(tmp_path, second)
        assert dead["closed"] is False and alive["closed"] is True
        assert balanced(dead) and balanced(alive)
        (dead_check,) = verify_writer_shards(str(tmp_path), dead)
        (alive_check,) = verify_writer_shards(str(tmp_path), alive)
        assert (dead_check["records"], dead_check["bad_lines"]) == (10, 0)
        assert (alive_check["records"], alive_check["bad_lines"]) == (5, 0)
        assert alive_check["start_offset"] == size_at_death
        assert alive_check["fragments_isolated"] == 1
        # The successor's first record starts on a line of its own.
        lines = (tmp_path / SHARD).read_bytes().split(b"\n")
        assert json.loads(lines[11]) == record(100)


class TestSidecarPublication:
    @pytest.mark.asyncio
    async def test_a_failed_publication_leaves_the_previous_sidecar_whole(
        self, tmp_path, monkeypatch
    ):
        writer = await started(tmp_path)
        submit(writer, 2)
        await until(lambda: writer.persisted == 2)
        await writer._publish(force=True)
        before = sidecar(tmp_path, writer)

        real_replace = os.replace

        def failing_replace(src, dst):
            raise OSError(errno.EIO, "Input/output error")

        monkeypatch.setattr(request_trace.os, "replace", failing_replace)
        submit(writer, 1, start=2)
        await until(lambda: writer.persisted == 3)
        await writer._publish(force=True)
        assert sidecar(tmp_path, writer) == before  # still the last whole document
        assert writer._sidecar_errors == 1

        monkeypatch.setattr(request_trace.os, "replace", real_replace)
        await writer.close()
        after = sidecar(tmp_path, writer)
        assert after["errors"]["sidecar_errors"] == 1
        assert after["counts"]["persisted"] == 3
        assert balanced(after)


def _load_reconcile_tool():
    """The freeze-time reconciler, which must run without TensorRT-LLM installed."""
    root = Path(__file__).resolve().parents[4]
    path = (
        root
        / "examples"
        / "serve"
        / "large_scale_serving"
        / "analysis"
        / "reconcile_trace_writers.py"
    )
    spec = importlib.util.spec_from_file_location("reconcile_trace_writers", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestFreezeReconciliation:
    @pytest.mark.asyncio
    async def test_the_freeze_tool_agrees_with_the_writer_and_infers_nothing(
        self, tmp_path, monkeypatch
    ):
        # A predecessor that published, then persisted two more records it
        # never published, and died.
        first = await started(tmp_path)
        submit(first, 5)
        await until(lambda: first.persisted == 5)
        await first._publish(force=True)
        submit(first, 2, start=5)
        await until(lambda: first.persisted == 7)
        first._task.cancel()
        await asyncio.gather(first._task, return_exceptions=True)
        first._task = None

        # A successor whose first append is cut inside a character, then recovers.
        second = await started(tmp_path)
        text = json.dumps(record(100, "你好"), ensure_ascii=False, separators=(",", ":"))
        cut_next_shard_write(
            monkeypatch,
            len(text.encode()) + 3,
            OSError(errno.ENOSPC, "No space left on device"),
        )
        submit(second, 3, start=100, text="你好")
        await until(lambda: second._write_error_count == 1)
        submit(second, 2, start=200)
        await second.close()

        tool = _load_reconcile_tool()
        report = tool.reconcile(str(tmp_path))
        assert report["all_books_balance"] and report["all_shards_verify"]
        for generation in report["generations"]:
            document = json.loads((tmp_path / generation["sidecar"]).read_text())
            assert generation["shards"] == verify_writer_shards(str(tmp_path), document)
        # The predecessor's two unpublished records are on disk but in no
        # sidecar: reported as unaccounted, not credited to anyone.
        assert report["unaccounted"] == {
            SHARD: {
                "size": (tmp_path / SHARD).stat().st_size,
                "unclaimed_bytes": report["unaccounted"][SHARD]["unclaimed_bytes"],
                "whole_lines": 2,
                "cut_lines": 0,
            }
        }


class TestIdlePublication:
    @pytest.mark.asyncio
    async def test_a_quiet_writer_catches_its_sidecar_up(self, tmp_path, monkeypatch):
        """The last burst before a lull is published without waiting for more traffic."""
        monkeypatch.setattr(request_trace, "_WRITER_SIDECAR_INTERVAL_SECONDS", 1.0)
        writer = await started(tmp_path)
        # Within the interval of the start-up publication, so the batch itself
        # is not allowed to publish.
        submit(writer, 3)
        await until(lambda: writer.persisted == 3)
        assert sidecar(tmp_path, writer)["counts"]["persisted"] == 0
        await until(lambda: sidecar(tmp_path, writer)["counts"]["persisted"] == 3, timeout=5.0)
        document = sidecar(tmp_path, writer)
        assert document["closed"] is False
        assert balanced(document)
        (check,) = verify_writer_shards(str(tmp_path), document)
        assert (check["records"], check["bad_lines"]) == (3, 0)
        await writer.close()

    @pytest.mark.asyncio
    async def test_a_failed_publication_is_retried_while_the_writer_is_quiet(
        self, tmp_path, monkeypatch
    ):
        """A sidecar that did not land is retried once the filesystem recovers.

        The publication of a final burst fails. After that nothing is submitted
        and the writer is not closed, yet the sidecar must still catch up with
        the books: only a publication that landed may mark them published.
        """
        monkeypatch.setattr(request_trace, "_WRITER_SIDECAR_INTERVAL_SECONDS", 1.0)
        writer = await started(tmp_path)
        real_replace = os.replace
        refused = []

        def failing_replace(src, dst):
            refused.append(dst)
            raise OSError(errno.EIO, "Input/output error")

        monkeypatch.setattr(request_trace.os, "replace", failing_replace)
        submit(writer, 3)
        await until(lambda: writer.persisted == 3)
        # The idle drain publishes the burst, and the filesystem refuses it.
        await until(lambda: refused, timeout=5.0)
        assert sidecar(tmp_path, writer)["counts"]["persisted"] == 0
        monkeypatch.setattr(request_trace.os, "replace", real_replace)

        await until(lambda: sidecar(tmp_path, writer)["counts"]["persisted"] == 3, timeout=5.0)
        document = sidecar(tmp_path, writer)
        assert document["closed"] is False
        assert document["errors"]["sidecar_errors"] == len(refused)
        assert balanced(document)
        await writer.close()
