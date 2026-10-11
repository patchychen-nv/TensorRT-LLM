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
"""Dump what the model actually produced, before any endpoint formatter sees it.

Enabled by pointing ``TRTLLM_RAW_OUTPUT_DIR`` at a directory and passing
``--post_processor_hook tensorrt_llm.executor.raw_output_hook.RawOutputDump``;
with either half missing the whole thing is inert. One JSONL file per UTC hour::

    $TRTLLM_RAW_OUTPUT_DIR/
      2026-09-18T07/
        raw-<pid>.jsonl    one line per (request, output), when it finishes

What this records exists nowhere else. ``request_trace.py`` keeps the wire
payload, which is what the reasoning and tool parsers made of the text; when a
parser drops a tool call or raises "Incomplete DSML invoke", the text it was
looking at is already gone. This runs at the detok chokepoint -- after
``text``/``text_diff`` are populated and before any per-endpoint formatter reads
them -- so the two together show the transformation from both ends.

Reading it beside a trace
-------------------------
The join is the engine request id. It reaches the client as the response id
(``postprocess_handlers.py`` mints ``chatcmpl-<id>`` / ``cmpl-<id>``), so it is
inside the body ``request_trace`` records, and it is ``chunk.request_id`` here.

That holds through disaggregation, which is the case that needs it: the parsers
run only in the generation worker (``--tool_parser`` is generation-only) while
the request trace deliberately skips internal orchestrator hops, so the
generation worker writes no trace line at all and has no ``trace_id`` to stamp.
What makes the join survive anyway is that the proxy is a byte relay -- it
forwards the worker's frames without reassembling them -- so the id the client
sees, and the proxy records, is the one the generation worker minted.

Why the frames are kept apart
-----------------------------
A streaming record holds the deltas as a list rather than one joined string.
Incremental parsers are the thing being debugged and they fail *on the seams*:
a tool-call marker split across two chunks, a reasoning tag closed in the chunk
after it opened. Joining the deltas before writing destroys exactly the evidence
that distinguishes "the model never emitted it" from "the parser lost it at a
chunk boundary". ``text`` is stored too, so a reader wanting the whole output
does not have to join them.
"""

import json
import logging
import os
import threading
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from tensorrt_llm.executor.postprocessor_hook import (
    PostProcessorHookChunk,
    PostProcessorHookVerdict,
    emit,
)

# Stdlib logging, like ``postprocessor_hook`` itself and unlike the rest of the
# package: this module is imported inside each post-processing worker process,
# and that module keeps its dependency surface to the standard library for
# exactly that reason. Reaching for ``tensorrt_llm.logger`` here would drag the
# package import into a process that is meant to need only the hook.
logger = logging.getLogger(__name__)

RAW_OUTPUT_DIR_ENV = "TRTLLM_RAW_OUTPUT_DIR"

_HOUR_BUCKET_LEN = 13


def raw_output_dir_from_env() -> Optional[str]:
    """Read the enabling variable.

    A function rather than a module constant so the value is picked up when the
    hook is constructed. The hook is built once per owner -- the ``LLM`` for the
    in-proxy detok path, and separately inside each post-processing worker
    process -- and those are not all built at import time.
    """
    value = os.environ.get(RAW_OUTPUT_DIR_ENV)
    return value or None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class _Pending:
    """The deltas seen so far for one (request, output), and nothing else.

    A plain object rather than a dataclass because one of these exists per
    in-flight sequence and it is only ever touched by the two methods below.
    """

    __slots__ = ("frames", "started_at", "streaming", "disagg_request_id")

    def __init__(self, streaming: bool, disagg_request_id: Optional[int] = None):
        self.frames: List[str] = []
        self.started_at = _utc_now()
        self.streaming = streaming
        self.disagg_request_id = disagg_request_id


class RawOutputDump:
    """Record each output's detokenized text; never change it.

    Constructed with no arguments by ``load_post_processor_hook``, one instance
    per owner process, and called once per output per chunk.

    It is an observer, so every verdict is ``emit(chunk.text_diff)``: the applier
    rebuilds ``output.text`` as ``already-emitted prefix + verdict.text``, which
    for the unmodified diff is the text that was already there. Nothing
    downstream can tell this hook ran.

    **It never raises.** The hook seam fails closed on purpose -- a hook that
    vets output must not have its failures serve un-vetted text -- but this one
    vets nothing, and a full disk is not a reason to fail requests. Every error
    is swallowed and reported once.
    """

    def __init__(self) -> None:
        self._dir = raw_output_dir_from_env()
        self._suffix = f"-{os.getpid()}"
        # The in-proxy detok path runs under the event loop while the
        # post-processing workers each run their own; either way one instance
        # can be reached from more than one thread, and a torn line costs the
        # record rather than corrupting a neighbouring one only because a lock
        # is cheaper than reasoning about that.
        self._lock = threading.Lock()
        self._pending: Dict[Tuple[int, int], _Pending] = {}
        self._known_dirs: set = set()
        self._dropped = 0
        if self._dir:
            logger.info("Recording raw model output to %s", self._dir)

    # -- hook ----------------------------------------------------------------

    def __call__(self, chunk: PostProcessorHookChunk) -> PostProcessorHookVerdict:
        verdict = emit(chunk.text_diff)
        if not self._dir:
            return verdict
        try:
            self._observe(chunk)
        except Exception as error:  # noqa: BLE001 - a debug dump must not fail a request
            self._note_drop(error)
        return verdict

    def _observe(self, chunk: PostProcessorHookChunk) -> None:
        key = (chunk.request_id, chunk.output_index)
        with self._lock:
            pending = self._pending.get(key)
            if pending is None:
                pending = _Pending(
                    streaming=chunk.streaming, disagg_request_id=chunk.disagg_request_id
                )
                self._pending[key] = pending
            # Empty diffs are dropped rather than stored. A chunk that added no
            # text still arrives -- the final call after the last token is the
            # common one -- and keeping them would put blanks between the frames
            # that matter, which is noise in exactly the seam being read.
            if chunk.text_diff:
                pending.frames.append(chunk.text_diff)
            if not chunk.is_final:
                return
            # is_final is request-level, so for n>1 it fires once for every
            # output at the same moment. Each is popped by its own key. A
            # request that fails instead -- an executor error or a
            # postprocessing exception -- gets no final chunk; on_request_end
            # releases it. One that no path reports at all would stay until the
            # process ends, so every failure path must call it.
            self._pending.pop(key, None)
        self._write(chunk, pending)

    def on_request_end(self, request_id: int, reason: str) -> None:
        """Release a request that ended without a final chunk, recording what it had.

        Called through ``end_post_processor_hook`` when a request fails before
        its final chunk. Each of its outputs that produced text is written as a
        record with ``partial: true`` and ``end_reason``: a generation cut off
        by an error is evidence too, and is told apart from a finished one.
        """
        try:
            with self._lock:
                keys = [key for key in self._pending if key[0] == request_id]
                ended = [(key, self._pending.pop(key)) for key in keys]
            for (_, output_index), pending in ended:
                if pending.frames:
                    self._write_partial(request_id, output_index, pending, reason)
        except Exception as error:  # noqa: BLE001 - a debug dump must not fail a request
            self._note_drop(error)

    # -- writing -------------------------------------------------------------

    def _write(self, chunk: PostProcessorHookChunk, pending: _Pending) -> None:
        finished_at = _utc_now()
        record: Dict[str, Any] = {
            "event": "raw_output",
            # Two ids because one of them is not enough, which only showed up
            # under real traffic:
            #
            # `request_id` is this engine's own counter. On the aggregated
            # server it reaches the client inside the response id
            # ("chatcmpl-<id>") and joins a trace on that. On a disaggregated
            # one it does not leave the worker at all -- it counts 10, 11, 12
            # while the proxy knows the request as 10349323008212992 -- so a
            # join on it silently matches nothing.
            #
            # `disagg_request_id` is what the orchestrator gave the request, and
            # is on the proxy's trace line as a field. It is None when serving
            # is aggregated. Prefer it and fall back to `request_id`.
            #
            # The response id does not cover both either: the Responses API,
            # which is the protocol the agents actually speak, mints a fresh
            # `resp_<uuid>` unrelated to any engine id.
            "request_id": str(chunk.request_id),
            "disagg_request_id": (
                None if chunk.disagg_request_id is None else str(chunk.disagg_request_id)
            ),
            "output_index": chunk.output_index,
            "finished_at": finished_at,
            "started_at": pending.started_at,
            "streaming": pending.streaming,
            "aborted": chunk.aborted,
            # Post-detok, pre-formatter. `text` is what the model produced in
            # full; `frames` is how it arrived.
            "text": chunk.text,
            "frames": pending.frames,
        }
        self._append(finished_at[:_HOUR_BUCKET_LEN], record)

    def _write_partial(
        self, request_id: int, output_index: int, pending: _Pending, reason: str
    ) -> None:
        finished_at = _utc_now()
        record: Dict[str, Any] = {
            "event": "raw_output",
            "request_id": str(request_id),
            "disagg_request_id": (
                None if pending.disagg_request_id is None else str(pending.disagg_request_id)
            ),
            "output_index": output_index,
            "finished_at": finished_at,
            "started_at": pending.started_at,
            "streaming": pending.streaming,
            "aborted": True,
            "partial": True,
            "end_reason": reason,
            # A streaming output's frames are its deltas; a non-streaming one
            # sees its whole text in every chunk, so the last frame is it.
            "text": "".join(pending.frames) if pending.streaming else pending.frames[-1],
            "frames": pending.frames,
        }
        self._append(finished_at[:_HOUR_BUCKET_LEN], record)

    def _append(self, bucket: str, record: Dict[str, Any]) -> None:
        try:
            line = json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
        except (TypeError, ValueError) as error:
            self._note_drop(error)
            return
        directory = os.path.join(self._dir, bucket)
        try:
            if bucket not in self._known_dirs:
                os.makedirs(directory, exist_ok=True)
                self._known_dirs.add(bucket)
            path = os.path.join(directory, f"raw{self._suffix}.jsonl")
            # One open per record. The write happens once per finished sequence
            # rather than once per token, so this is off the hot path, and an
            # always-open handle would have to be closed from somewhere -- the
            # hook has no shutdown call.
            with open(path, "a", encoding="utf-8") as output:
                output.write(line)
        except OSError as error:
            self._note_drop(error)

    def _note_drop(self, error: BaseException) -> None:
        self._dropped += 1
        if self._dropped == 1 or self._dropped % 1000 == 0:
            logger.warning("Dropped %d raw output record(s): %s", self._dropped, error)
