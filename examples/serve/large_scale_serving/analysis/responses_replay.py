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
"""Offline replay of the Responses streaming path over recorded generations.

Why this exists
---------------
`_should_send_done_events` re-parses the whole accumulated `output.text` with the
*non-streaming* parsers on every chunk. The non-streaming tool regex needs a
closing tag, so while one call is closed and a second is still open, the open
one stays in `normal_text` and `tool_calls` is non-empty at the same time --
both halves of `if full_text and tool_calls:` are true and the open call's raw
markup is emitted as the final assistant text. See
`SPEC_responses_streaming_fix.md`.

This replays real recorded frames through the shipped code and checks the
invariant the fix is meant to establish:

    what an `output_text.done` event reports must be exactly what was already
    streamed as `output_text.delta` events for that item.

WHAT IS REAL AND WHAT IS NOT
----------------------------
This harness does **not** model the done-decision. It loads and executes the
real functions from the checked-in source tree:

  REAL, executed as shipped (loaded from the worktree's own .py files):
    * tensorrt_llm/serve/responses_utils.py
        - `_generate_streaming_event`   (the whole per-chunk event pipeline)
        - `_should_send_done_events`    (the defect under test)
        - `_close_open_item`, `_apply_tool_parser`, `_apply_reasoning_parser`
        - `ResponsesStreamingEventsHelper` / `ResponsesStreamingStateTracker`
        - `_get_chat_completion_function_tools`, `_tool_call_output_item`
    * tensorrt_llm/serve/openai_protocol.py   (ResponsesRequest, tool params)
    * tensorrt_llm/serve/tool_parser/**       (Glm47ToolParser and friends)
    * tensorrt_llm/llmapi/reasoning_parser.py (GlmReasoningParser)
    * tensorrt_llm/serve/web_search.py, responses_web_search.py
  The emitted events are the real `openai.types.responses` pydantic models.

  MODELLED / SUBSTITUTED (stated plainly so results are not over-read):
    1. Every other `tensorrt_llm.*` module is replaced by an auto-stub, because
       this worktree has no compiled C++ bindings and `import tensorrt_llm`
       fails at `tensorrt_llm.bindings`. The stubbed modules
       (`tensorrt_llm._utils`, `.executor`, `.llmapi` top level, `.llmapi.llm`,
       `.llmapi.tokenizer`, `.inputs.*`, `.sampling_params`,
       `.serve.harmony_adapter`, ...) supply only names that are imported for
       type annotations or for code paths this harness never enters.
       `verify_provenance()` asserts at run time that every module listed under
       REAL above really did come from a .py file in the tree.
    2. The driver. Production calls `_generate_streaming_event` from
       `ResponsesStreamingProcessor.process_single_output`, which also wraps
       each event in SSE framing and counts sequence numbers. This harness calls
       `_generate_streaming_event` directly with the same arguments that method
       passes (`output`, `request`, `finished_generation=res._done`, the helper,
       the parser ids and the two parser dicts). SSE framing is not exercised.
    3. `output` is a 3-field stand-in (`index`, `text`, `text_diff`) -- the only
       attributes `_generate_streaming_event` and `_should_send_done_events`
       read off it. This mirrors the checked-in unit test
       `tests/unittest/llmapi/apps/test_responses_streaming_tool_calls.py`.
    4. The tool *definitions*. The raw records carry the model's output, not the
       request that produced it, so the `tools` list is reconstructed from the
       call names observed in the text, each with a permissive object schema.
       Tool schemas feed `get_argument_type`, which only affects how argument
       values are typed into the call's JSON -- it does not affect `normal_text`
       and therefore does not affect the leak this measures.
    5. Reasoning/tool parser ids are supplied on the command line (defaults
       `glm47`/`glm47`, which is what the recorded fleet ran). Production reads
       them from server config.

Replay modes
------------
`stream`   one pass per record, exactly as production drives it:
           `finished_generation=True` on the final frame only. Prefix k means
           "everything emitted through frame k". This is the faithful replay.
`truncate` n independent replays per record; replay k feeds frames 0..k with
           `finished_generation=True` at k, i.e. "the stream ended here"
           (spec edge case 6: max_tokens / abort mid-call).

Usage
-----
    PY=/code/tensorrt_llm/.venv-3.12/bin/python3
    $PY examples/serve/large_scale_serving/analysis/responses_replay.py \
        --frames-json examples/serve/large_scale_serving/analysis/data/responses_replay_frames.json \
        --records examples/serve/large_scale_serving/analysis/data/responses_replay_multicall.jsonl \
        --out-json /tmp/responses_replay_BEFORE.json

Exit code is 1 when any invariant violation is found, so the same command is a
pass/fail gate before and after the fix. `--compare <old.json>` prints a
BEFORE/AFTER delta.
"""

from __future__ import annotations

import argparse
import difflib
import gzip
import hashlib
import importlib.abc
import importlib.util
import json
import logging
import os
import re
import sys
import types
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

# ---------------------------------------------------------------------------
# Standalone loader
# ---------------------------------------------------------------------------

# Modules loaded from their real source file. Everything else under
# `tensorrt_llm.` is stubbed.
_REAL_MODULES = {
    "tensorrt_llm.llmapi.reasoning_parser": "tensorrt_llm/llmapi/reasoning_parser.py",
    "tensorrt_llm.serve.openai_protocol": "tensorrt_llm/serve/openai_protocol.py",
    "tensorrt_llm.serve.web_search": "tensorrt_llm/serve/web_search.py",
    "tensorrt_llm.serve.responses_web_search": "tensorrt_llm/serve/responses_web_search.py",
    "tensorrt_llm.serve.responses_utils": "tensorrt_llm/serve/responses_utils.py",
}
# Whole packages loaded from real source, submodules included.
_REAL_PACKAGES = {
    "tensorrt_llm.serve.tool_parser": "tensorrt_llm/serve/tool_parser",
}


class _StubMeta(type):
    """Metaclass for auto-created stand-in types.

    Two behaviours the stubs need. Unknown attributes resolve to further
    stand-ins, so `SomeStubEnum.VALUE` used as a pydantic field default does not
    raise. And `__get_pydantic_core_schema__` returns `any_schema`, so a real
    pydantic model (`openai_protocol`'s, which this harness wants to execute for
    real) can still be built when one of its field annotations resolved to a
    stand-in.
    """

    def __getattr__(cls, name):
        if name.startswith("__"):
            raise AttributeError(name)
        value = _StubMeta(name, (object,), {})
        setattr(cls, name, value)
        return value

    def __get_pydantic_core_schema__(cls, source, handler):
        from pydantic_core import core_schema

        return core_schema.any_schema()

    def __iter__(cls):
        return iter(())


class _StubModule(types.ModuleType):
    """A `tensorrt_llm.*` module that hands back a stand-in for any name."""

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        value = _StubMeta(name, (object,), {"__module__": self.__name__})
        setattr(self, name, value)
        return value


class _TensorrtLlmFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Serve `tensorrt_llm.*` from source for the allowlist, stubs otherwise."""

    def __init__(self, root: Path, overrides: Optional[dict[str, str]] = None):
        self.root = Path(root)
        self.overrides = dict(overrides or {})

    def find_spec(self, fullname, path=None, target=None):
        if fullname != "tensorrt_llm" and not fullname.startswith("tensorrt_llm."):
            return None
        if fullname in self.overrides:
            return importlib.util.spec_from_file_location(fullname, self.overrides[fullname])
        for pkg, rel in _REAL_PACKAGES.items():
            if fullname == pkg or fullname.startswith(pkg + "."):
                suffix = fullname[len(pkg) :].lstrip(".")
                base = self.root / rel
                if suffix:
                    base = base.joinpath(*suffix.split("."))
                if base.is_dir():
                    return importlib.util.spec_from_file_location(
                        fullname, base / "__init__.py", submodule_search_locations=[str(base)]
                    )
                return importlib.util.spec_from_file_location(fullname, str(base) + ".py")
        if fullname in _REAL_MODULES:
            return importlib.util.spec_from_file_location(
                fullname, self.root / _REAL_MODULES[fullname]
            )
        return importlib.util.spec_from_loader(fullname, self, is_package=True)

    def create_module(self, spec):
        module = _StubModule(spec.name)
        module.__path__ = []
        return module

    def exec_module(self, module):
        # `tensorrt_llm.logger` is not in the allowlist (the real one drags in
        # the bindings) but the parsers log through it, and a stub would make a
        # parser exception invisible. Give it a real logger.
        if module.__name__ == "tensorrt_llm.logger":
            module.logger = logging.getLogger("trtllm.standalone")
        elif module.__name__ == "tensorrt_llm":
            module.logger = logging.getLogger("trtllm.standalone")


@dataclass
class Provenance:
    """Where each loaded module actually came from, and its content hash."""

    root: str
    real: dict[str, str]
    real_sha256: dict[str, str]
    stubbed: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "root": self.root,
            "real": self.real,
            "real_sha256": self.real_sha256,
            "stubbed": sorted(self.stubbed),
        }


def load_responses_utils(
    root: str | os.PathLike, responses_utils_path: Optional[str] = None
) -> tuple[Any, Provenance]:
    """Import the real `responses_utils` with the rest of TRT-LLM stubbed.

    `responses_utils_path` pins that one module to a specific file instead of
    the one in `root`. That is what makes a BEFORE baseline reproducible while
    the tree is being edited: point it at a snapshot of the pre-fix source
    (`git show <rev>:tensorrt_llm/serve/responses_utils.py`) and the same
    command with the flag dropped measures the fixed tree.

    Returns the module and a `Provenance` record. Raises if any module that is
    supposed to be real was not in fact loaded from a file in `root`.
    """
    root = Path(root).resolve()
    overrides = {}
    if responses_utils_path:
        overrides["tensorrt_llm.serve.responses_utils"] = str(Path(responses_utils_path).resolve())
    for name in list(sys.modules):
        if name == "tensorrt_llm" or name.startswith("tensorrt_llm."):
            del sys.modules[name]
    sys.meta_path.insert(0, _TensorrtLlmFinder(root, overrides))

    import tensorrt_llm.serve.responses_utils as responses_utils  # noqa: E402

    real: dict[str, str] = {}
    sha: dict[str, str] = {}
    stubbed: list[str] = []
    for name, module in sorted(sys.modules.items()):
        if not (name == "tensorrt_llm" or name.startswith("tensorrt_llm.")):
            continue
        origin = getattr(module, "__file__", None)
        if origin:
            real[name] = os.path.relpath(origin, root)
            sha[name] = hashlib.sha256(Path(origin).read_bytes()).hexdigest()
        else:
            stubbed.append(name)

    provenance = Provenance(root=str(root), real=real, real_sha256=sha, stubbed=stubbed)
    verify_provenance(provenance)
    return responses_utils, provenance


#: Modules whose real source must be executing for the results to mean anything.
REQUIRED_REAL = (
    "tensorrt_llm.serve.responses_utils",
    "tensorrt_llm.serve.openai_protocol",
    "tensorrt_llm.llmapi.reasoning_parser",
    "tensorrt_llm.serve.tool_parser.glm47_parser",
    "tensorrt_llm.serve.tool_parser.tool_parser_factory",
)


def verify_provenance(provenance: Provenance) -> None:
    """Fail loudly rather than silently report on a stub.

    A stub answers every attribute, so a mis-wired loader would produce a clean
    run with no events and no violations -- which reads exactly like a passing
    AFTER report.
    """
    missing = [m for m in REQUIRED_REAL if m not in provenance.real]
    if missing:
        raise RuntimeError(
            "these modules had to be loaded from real source but were stubbed: "
            + ", ".join(missing)
        )


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------

_TOOL_CALL_NAME = re.compile(r"<tool_call>([^<\n]*)")
MARKUP_TOKENS = (
    "<tool_call>",
    "</tool_call>",
    "<arg_key>",
    "</arg_key>",
    "<arg_value>",
    "</arg_value>",
)


@dataclass
class Record:
    record_id: str
    frames: list[str]
    text: str
    source: str = ""

    @property
    def n_tool_calls(self) -> int:
        return self.text.count("<tool_call>")

    def tool_names(self) -> list[str]:
        seen: list[str] = []
        for name in _TOOL_CALL_NAME.findall(self.text):
            name = name.strip()
            if name and name not in seen:
                seen.append(name)
        return seen


def load_records(paths: Iterable[str], min_calls: int = 0) -> list[Record]:
    """Read records from .json (single, `{frames, text}`) or .jsonl[.gz] files."""
    records: list[Record] = []
    for raw_path in paths:
        path = Path(raw_path)
        if path.suffix == ".json":
            blob = json.loads(path.read_text())
            records.append(
                Record(
                    record_id=blob.get("request_id") or path.stem,
                    frames=list(blob["frames"]),
                    text=blob["text"],
                    source=str(path),
                )
            )
            continue
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt") as handle:
            for lineno, line in enumerate(handle, 1):
                line = line.strip()
                if not line:
                    continue
                blob = json.loads(line)
                frames = blob.get("frames")
                if not frames:
                    continue
                records.append(
                    Record(
                        record_id=str(blob.get("request_id") or f"{path.name}:{lineno}"),
                        frames=list(frames),
                        text=blob.get("text") or "".join(frames),
                        source=f"{blob.get('_src', path.name)}:{blob.get('_line', lineno)}",
                    )
                )
    if min_calls:
        records = [r for r in records if r.n_tool_calls >= min_calls]
    return _uniquify_ids(records)


def _uniquify_ids(records: list[Record]) -> list[Record]:
    """Make `record_id` unique across the loaded set.

    `request_id` is the engine's own per-process counter, so the 42 raw dumps
    reuse the same small integers for completely different responses. Keying
    anything on it silently merges those responses: the counts come out low and
    a clean record can mask a leaking one under the same id.
    """
    taken: set[str] = set()
    for record in records:
        base = record.record_id
        candidate = base
        suffix = 1
        while candidate in taken:
            suffix += 1
            candidate = f"{base}#{suffix}"
        record.record_id = candidate
        taken.add(candidate)
    return records


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------


class _Output:
    """The three attributes the streaming path reads off `RequestOutput`."""

    def __init__(self, index: int = 0):
        self.index = index
        self.text = ""
        self.text_diff = ""


def _reconstructed_tool_names(record: Record, tool_parser_id: Optional[str]) -> list[str]:
    """Names to declare as tools, taken from the record's own markup.

    Asks the real parser what names it finds rather than re-deriving them here,
    so the declared set matches what `resolve_tool_name` will look up and the
    replay is not full of spurious "undefined function" warnings caused by the
    harness's own regex disagreeing with the parser's.
    """
    if tool_parser_id:
        try:
            factory = sys.modules["tensorrt_llm.serve.tool_parser.tool_parser_factory"]
            parser = factory.ToolParserFactory.create_tool_parser(tool_parser_id)
            # This pass declares no tools, so every name it finds is reported
            # as an undefined function. That warning is about the probe, not
            # about the replay, and printing it would misdescribe a run whose
            # real tool list is the one being built from these very names.
            logger = logging.getLogger("trtllm.standalone")
            previous = logger.level
            logger.setLevel(logging.ERROR)
            try:
                names = [
                    call.name
                    for call in parser.detect_and_parse(record.text, []).calls
                    if call.name
                ]
            finally:
                logger.setLevel(previous)
            if names:
                return list(dict.fromkeys(names))
        except Exception:
            pass
    return record.tool_names()


def _build_request(responses_utils: Any, record: Record, tool_parser_id: Optional[str]) -> Any:
    """A request carrying tool definitions reconstructed from the record.

    The raw dumps hold the model's output, not the request. The names are read
    back out of the emitted markup and given a permissive object schema; see the
    module docstring, item 4, for why this cannot move the leak measurement.
    """
    protocol = sys.modules["tensorrt_llm.serve.openai_protocol"]
    from openai.types.responses.tool import FunctionTool

    tools = [
        FunctionTool(
            type="function",
            name=name,
            description=None,
            parameters={
                "type": "object",
                "properties": {},
            },
            strict=False,
        )
        for name in _reconstructed_tool_names(record, tool_parser_id)
    ]
    try:
        return protocol.ResponsesRequest(input="", tools=tools)
    except Exception:  # pragma: no cover - only if the model gains required fields
        from types import SimpleNamespace

        return SimpleNamespace(tools=tools, chat_template_kwargs=None, reasoning=None)


@dataclass
class EventRow:
    """One emitted event, flattened to what the invariant needs."""

    prefix: int
    type: str
    item_id: str = ""
    output_index: int = -1
    content_index: int = -1
    payload: str = ""
    tool_name: Optional[str] = None
    tool_arguments: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        out = {
            "prefix": self.prefix,
            "type": self.type,
            "item_id": self.item_id,
            "output_index": self.output_index,
            "content_index": self.content_index,
        }
        if self.payload:
            out["payload"] = self.payload
        if self.tool_name is not None:
            out["tool_name"] = self.tool_name
            out["tool_arguments"] = self.tool_arguments
        return out


def _flatten(prefix: int, event: Any) -> Optional[EventRow]:
    etype = getattr(event, "type", "")
    if etype in ("response.output_text.delta", "response.reasoning_text.delta"):
        return EventRow(
            prefix=prefix,
            type=etype,
            item_id=getattr(event, "item_id", ""),
            output_index=getattr(event, "output_index", -1),
            content_index=getattr(event, "content_index", -1),
            payload=event.delta,
        )
    if etype in ("response.output_text.done", "response.reasoning_text.done"):
        return EventRow(
            prefix=prefix,
            type=etype,
            item_id=getattr(event, "item_id", ""),
            output_index=getattr(event, "output_index", -1),
            content_index=getattr(event, "content_index", -1),
            payload=event.text,
        )
    if etype == "response.output_item.done":
        item = getattr(event, "item", None)
        item_type = getattr(item, "type", None)
        if item_type in ("function_call", "custom_tool_call"):
            return EventRow(
                prefix=prefix,
                type="tool_call",
                item_id=getattr(item, "id", ""),
                output_index=getattr(event, "output_index", -1),
                tool_name=getattr(item, "name", None),
                tool_arguments=getattr(item, "arguments", None) or getattr(item, "input", None),
            )
    return None


@dataclass
class Replay:
    """Everything one replay of one record emitted."""

    record_id: str
    mode: str
    source: str
    n_frames: int
    n_tool_calls_in_text: int
    rows: list[EventRow] = field(default_factory=list)
    error: Optional[str] = None

    def text_deltas(self) -> list[EventRow]:
        return [r for r in self.rows if r.type == "response.output_text.delta"]

    def text_dones(self) -> list[EventRow]:
        return [r for r in self.rows if r.type == "response.output_text.done"]

    def tool_calls(self) -> list[EventRow]:
        return [r for r in self.rows if r.type == "tool_call"]


def replay_stream(
    responses_utils: Any,
    record: Record,
    *,
    reasoning_parser_id: Optional[str] = "glm47",
    tool_parser_id: Optional[str] = "glm47",
    upto: Optional[int] = None,
    finish_at_end: bool = True,
    mode: str = "stream",
) -> Replay:
    """Feed frames 0..upto through the real `_generate_streaming_event`.

    `upto=None` means the whole record. `finish_at_end` controls whether the
    last frame fed carries `finished_generation=True`.
    """
    last = len(record.frames) - 1 if upto is None else upto
    replay = Replay(
        record_id=record.record_id,
        mode=mode,
        source=record.source,
        n_frames=last + 1,
        n_tool_calls_in_text=record.n_tool_calls,
    )

    helper = responses_utils.ResponsesStreamingEventsHelper()
    reasoning_parser_dict: dict[int, Any] = {}
    tool_parser_dict: dict[int, Any] = {}
    request = _build_request(responses_utils, record, tool_parser_id)
    output = _Output(index=0)
    accumulated = ""

    for idx in range(last + 1):
        frame = record.frames[idx]
        accumulated += frame
        output.text = accumulated
        output.text_diff = frame
        finished = finish_at_end and idx == last
        try:
            events = list(
                responses_utils._generate_streaming_event(
                    output=output,
                    request=request,
                    finished_generation=finished,
                    streaming_events_helper=helper,
                    reasoning_parser_id=reasoning_parser_id,
                    tool_parser_id=tool_parser_id,
                    reasoning_parser_dict=reasoning_parser_dict,
                    tool_parser_dict=tool_parser_dict,
                )
            )
        except Exception as exc:  # the replay must survive to report the rest
            replay.error = f"frame {idx}: {type(exc).__name__}: {exc}"
            break
        for event in events:
            row = _flatten(idx, event)
            if row is not None:
                replay.rows.append(row)
    return replay


def replay_truncations(responses_utils: Any, record: Record, **kwargs: Any) -> list[Replay]:
    """One independent replay per prefix, each ending as if the stream stopped."""
    return [
        replay_stream(
            responses_utils, record, upto=k, finish_at_end=True, mode="truncate", **kwargs
        )
        for k in range(len(record.frames))
    ]


# ---------------------------------------------------------------------------
# The invariant
# ---------------------------------------------------------------------------


@dataclass
class Violation:
    """A `done` payload that is not the sum of the deltas that preceded it."""

    record_id: str
    mode: str
    prefix: int
    item_id: str
    kind: str  # "text" or "reasoning"
    streamed: str
    done: str
    extra_chars: int
    missing_chars: int
    markup_chars: int
    has_tool_markup: bool

    def diff(self, width: int = 3) -> str:
        return "\n".join(
            difflib.unified_diff(
                self.streamed.splitlines(keepends=True),
                self.done.splitlines(keepends=True),
                fromfile="concat(output_text.delta)",
                tofile="output_text.done",
                n=width,
                lineterm="",
            )
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "mode": self.mode,
            "prefix": self.prefix,
            "item_id": self.item_id,
            "kind": self.kind,
            "streamed": self.streamed,
            "done": self.done,
            "extra_chars": self.extra_chars,
            "missing_chars": self.missing_chars,
            "markup_chars": self.markup_chars,
            "has_tool_markup": self.has_tool_markup,
        }


def _diff_segments(streamed: str, done: str) -> tuple[list[str], int]:
    """(segments of `done` that were never streamed, chars streamed but dropped).

    Character-level opcodes rather than a whole-payload scan. A done payload can
    legitimately *contain* a markup token -- the model sometimes emits a stray
    `</tool_call>` that the parser passes through as ordinary text, and the
    deltas carry it too. Measuring markup over the whole payload scores that as
    a 2448-character leak when the actual difference is 12 characters. Only the
    inserted segments are the leak.
    """
    inserted: list[str] = []
    missing = 0
    matcher = difflib.SequenceMatcher(a=streamed, b=done, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag in ("insert", "replace"):
            inserted.append(done[j1:j2])
        if tag in ("delete", "replace"):
            missing += i2 - i1
    return inserted, missing


def _markup_chars(inserted: list[str]) -> int:
    """Chars of the leak that belong to a segment carrying tool-call markup."""
    return sum(
        len(segment) for segment in inserted if any(token in segment for token in MARKUP_TOKENS)
    )


def check_invariant(replay: Replay) -> list[Violation]:
    """concat(deltas for an item) == that item's done payload, for every item.

    Walked in emission order with one accumulator per open item, so a response
    with several message items (reasoning, then text, then text after a call) is
    checked item by item rather than in aggregate.
    """
    violations: list[Violation] = []
    pending: dict[tuple[str, str], str] = {}
    for row in replay.rows:
        if row.type.endswith(".delta"):
            kind = "reasoning" if "reasoning" in row.type else "text"
            pending[(kind, row.item_id)] = pending.get((kind, row.item_id), "") + row.payload
        elif row.type.endswith(".done"):
            kind = "reasoning" if "reasoning" in row.type else "text"
            streamed = pending.pop((kind, row.item_id), "")
            if streamed == row.payload:
                continue
            inserted, missing = _diff_segments(streamed, row.payload)
            markup = _markup_chars(inserted)
            violations.append(
                Violation(
                    record_id=replay.record_id,
                    mode=replay.mode,
                    prefix=row.prefix,
                    item_id=row.item_id,
                    kind=kind,
                    streamed=streamed,
                    done=row.payload,
                    extra_chars=sum(len(s) for s in inserted),
                    missing_chars=missing,
                    markup_chars=markup,
                    has_tool_markup=markup > 0,
                )
            )
    return violations


# ---------------------------------------------------------------------------
# Spec assertions for the 8-frame reference record
# ---------------------------------------------------------------------------


def reference_checks(replay: Replay) -> dict[str, Any]:
    """The two concrete claims in the spec's Verification section, point 1."""
    dones = replay.text_dones()
    calls = replay.tool_calls()
    leaking = [d for d in dones if any(t in d.payload for t in MARKUP_TOKENS)]
    final_text = dones[-1].payload if dones else ""
    return {
        "n_text_done_events": len(dones),
        "n_done_payloads_with_tool_markup": len(leaking),
        "leaking_prefixes": [d.prefix for d in leaking],
        "leaked_payload_lengths": [len(d.payload) for d in leaking],
        "n_tool_calls_surfaced": len(calls),
        "tool_call_names": [c.tool_name for c in calls],
        "final_text_done_len": len(final_text),
        "final_text_done": final_text,
        "no_markup_in_any_done": not leaking,
    }


# ---------------------------------------------------------------------------
# Runner + report
# ---------------------------------------------------------------------------


@dataclass
class RunResult:
    replays: list[Replay]
    violations: list[Violation]
    reference: Optional[dict[str, Any]] = None
    truncate_skipped: list[str] = field(default_factory=list)
    calls_by_record: dict[str, int] = field(default_factory=dict)
    calls_recovered: dict[str, int] = field(default_factory=dict)


def run(
    responses_utils: Any,
    records: list[Record],
    *,
    modes: tuple[str, ...],
    reasoning_parser_id: Optional[str],
    tool_parser_id: Optional[str],
    truncate_max_frames: int,
    reference_id: Optional[str] = None,
    progress: bool = False,
) -> RunResult:
    result = RunResult(replays=[], violations=[])
    kwargs = dict(reasoning_parser_id=reasoning_parser_id, tool_parser_id=tool_parser_id)

    for n, record in enumerate(records, 1):
        if progress and n % 250 == 0:
            print(f"  ... {n}/{len(records)} records", file=sys.stderr)
        result.calls_by_record[record.record_id] = record.n_tool_calls
        if "stream" in modes:
            replay = replay_stream(responses_utils, record, **kwargs)
            result.replays.append(replay)
            result.violations.extend(check_invariant(replay))
            result.calls_recovered[record.record_id] = len(replay.tool_calls())
            if reference_id and record.record_id == reference_id:
                result.reference = reference_checks(replay)
        if "truncate" in modes:
            if len(record.frames) > truncate_max_frames:
                result.truncate_skipped.append(record.record_id)
            else:
                for replay in replay_truncations(responses_utils, record, **kwargs):
                    result.replays.append(replay)
                    result.violations.extend(check_invariant(replay))
    return result


def summarise(result: RunResult) -> dict[str, Any]:
    by_mode: dict[str, dict[str, Any]] = {}
    for mode in sorted({r.mode for r in result.replays}):
        replays = [r for r in result.replays if r.mode == mode]
        mode_violations = [v for v in result.violations if v.mode == mode]
        bad_ids = {v.record_id for v in mode_violations}
        record_ids = {r.record_id for r in replays}
        by_mode[mode] = {
            "records": len(record_ids),
            "replays": len(replays),
            # Anti-vacuity: a run that emitted nothing satisfies the invariant
            # trivially. These have to stay non-zero for a clean report to be
            # worth anything.
            "output_text_delta_events": sum(len(r.text_deltas()) for r in replays),
            "output_text_done_events": sum(len(r.text_dones()) for r in replays),
            "tool_call_items": sum(len(r.tool_calls()) for r in replays),
            "records_violating": len(bad_ids),
            "violations": len(mode_violations),
            "violations_with_tool_markup": sum(1 for v in mode_violations if v.has_tool_markup),
            "records_with_tool_markup": len(
                {v.record_id for v in mode_violations if v.has_tool_markup}
            ),
            "total_extra_chars": sum(v.extra_chars for v in mode_violations),
            "total_markup_chars": sum(v.markup_chars for v in mode_violations),
            "max_markup_chars": max((v.markup_chars for v in mode_violations), default=0),
        }

    # Per (record, mode): the modes answer different questions and a combined
    # "first violating prefix" would report the truncate answer for a stream
    # row, which is the number a reader would quote.
    per_record: list[dict[str, Any]] = []
    keys = sorted({(v.record_id, v.mode) for v in result.violations}, key=lambda k: (k[1], k[0]))
    for record_id, mode in keys:
        record_violations = [
            v for v in result.violations if v.record_id == record_id and v.mode == mode
        ]
        per_record.append(
            {
                "record_id": record_id,
                "mode": mode,
                "n_tool_calls_in_text": result.calls_by_record.get(record_id),
                "first_violating_prefix": min(v.prefix for v in record_violations),
                "violating_prefixes": sorted({v.prefix for v in record_violations}),
                "n_violations": len(record_violations),
                "max_markup_chars": max(v.markup_chars for v in record_violations),
                "total_markup_chars": sum(v.markup_chars for v in record_violations),
                "max_extra_chars": max(v.extra_chars for v in record_violations),
                "has_tool_markup": any(v.has_tool_markup for v in record_violations),
            }
        )

    errored = [
        {"record_id": r.record_id, "mode": r.mode, "error": r.error}
        for r in result.replays
        if r.error
    ]

    # Rate by number of calls in the response - the shape the spec measured on
    # the fleet (1 call -> 0.1%, 2 -> 41.7%, 3 -> 100%).
    stream_bad = {v.record_id for v in result.violations if v.mode == "stream"}
    by_calls: dict[str, dict[str, Any]] = {}
    for record_id, n_calls in result.calls_by_record.items():
        bucket = by_calls.setdefault(
            str(n_calls),
            {
                "records": 0,
                "violating": 0,
                "calls_recovered": 0,
                "calls_expected": 0,
            },
        )
        bucket["records"] += 1
        bucket["violating"] += int(record_id in stream_bad)
        bucket["calls_expected"] += n_calls
        bucket["calls_recovered"] += result.calls_recovered.get(record_id, 0)
    for bucket in by_calls.values():
        bucket["violation_rate_pct"] = (
            round(100.0 * bucket["violating"] / bucket["records"], 1) if bucket["records"] else 0.0
        )

    # Spec verification point 3: "every record's tool calls must still be
    # recovered". `<tool_call>` markers in the raw text are the expectation;
    # a record whose markup is malformed legitimately recovers a different
    # number, so this is a list to eyeball, not an assertion.
    call_mismatch = [
        {
            "record_id": record_id,
            "markers_in_text": n_calls,
            "calls_surfaced": result.calls_recovered.get(record_id, 0),
        }
        for record_id, n_calls in sorted(result.calls_by_record.items())
        if record_id in result.calls_recovered and result.calls_recovered[record_id] != n_calls
    ]

    return {
        "by_mode": by_mode,
        "by_call_count": dict(sorted(by_calls.items(), key=lambda kv: int(kv[0]))),
        "records_violating": per_record,
        "errors": errored,
        "truncate_skipped_records": result.truncate_skipped,
        "call_recovery_mismatch": call_mismatch,
        # Kept per record so BEFORE/AFTER can be diffed: a fix that satisfies
        # the invariant by dropping tool calls would otherwise look perfect.
        "calls_recovered": dict(sorted(result.calls_recovered.items())),
        "calls_in_text": dict(sorted(result.calls_by_record.items())),
        "total_violations": len(result.violations),
        "reference": result.reference,
    }


def print_report(summary: dict[str, Any], provenance: Provenance, label: str) -> None:
    line = "=" * 78
    print(line)
    print(f"Responses streaming replay - {label}")
    print(line)
    target = "tensorrt_llm.serve.responses_utils"
    print(f"responses_utils.py sha256 : {provenance.real_sha256.get(target, '?')}")
    print(f"loaded real modules       : {len(provenance.real)}")
    print(f"stubbed modules           : {len(provenance.stubbed)}")
    print()

    for mode, stats in summary["by_mode"].items():
        print(f"--- mode: {mode} ---")
        print(f"  records replayed            : {stats['records']}")
        print(f"  replays run                 : {stats['replays']}")
        print(f"  output_text.delta events    : {stats['output_text_delta_events']}")
        print(f"  output_text.done events     : {stats['output_text_done_events']}")
        print(f"  tool call items surfaced    : {stats['tool_call_items']}")
        print(f"  records violating invariant : {stats['records_violating']}")
        print(f"  done events violating       : {stats['violations']}")
        print(
            f"  ... of which leak markup    : "
            f"{stats['violations_with_tool_markup']} "
            f"({stats['records_with_tool_markup']} records)"
        )
        print(f"  markup chars leaked (total) : {stats['total_markup_chars']}")
        print(f"  markup chars leaked (max)   : {stats['max_markup_chars']}")
        print(f"  unstreamed chars in done    : {stats['total_extra_chars']}")
        print()

    if summary.get("by_call_count"):
        print("--- violation rate by tool calls in the response (mode: stream) ---")
        print(
            f"  {'calls':>5} {'records':>8} {'violating':>10} {'rate':>7}  {'calls recovered':>16}"
        )
        for calls, stats in summary["by_call_count"].items():
            print(
                f"  {calls:>5} {stats['records']:>8} "
                f"{stats['violating']:>10} "
                f"{stats['violation_rate_pct']:>6}% "
                f"{stats['calls_recovered']:>9}/"
                f"{stats['calls_expected']}"
            )
        print()

    if summary["records_violating"]:
        print("--- per-record violations ---")
        header = (
            f"{'record_id':<22} {'mode':<9} {'calls':>5} {'first_k':>7} "
            f"{'n':>4} {'markup':>7} {'extra':>7}  prefixes"
        )
        print(header)
        print("-" * len(header))
        for entry in summary["records_violating"]:
            print(
                f"{entry['record_id']:<22} "
                f"{entry['mode']:<9} "
                f"{str(entry['n_tool_calls_in_text']):>5} "
                f"{entry['first_violating_prefix']:>7} "
                f"{entry['n_violations']:>4} "
                f"{entry['max_markup_chars']:>7} "
                f"{entry['max_extra_chars']:>7}  "
                f"{entry['violating_prefixes']}"
            )
        print()

    if summary.get("truncate_skipped_records"):
        print(
            f"--- {len(summary['truncate_skipped_records'])} records skipped "
            f"in truncate mode (--truncate-max-frames) ---"
        )
        print(f"  {summary['truncate_skipped_records']}")
        print()

    mismatch = summary.get("call_recovery_mismatch") or []
    if mismatch:
        print(
            f"--- {len(mismatch)} records where the calls surfaced differ "
            f"from the <tool_call> markers in the raw text ---"
        )
        for entry in mismatch[:20]:
            print(
                f"  {entry['record_id']:<22} markers="
                f"{entry['markers_in_text']} surfaced="
                f"{entry['calls_surfaced']}"
            )
        if len(mismatch) > 20:
            print(f"  ... {len(mismatch) - 20} more")
        print()

    if summary["errors"]:
        print("--- replay errors ---")
        for entry in summary["errors"][:20]:
            print(f"  {entry['record_id']} [{entry['mode']}]: {entry['error']}")
        if len(summary["errors"]) > 20:
            print(f"  ... {len(summary['errors']) - 20} more")
        print()

    reference = summary.get("reference")
    if reference:
        print("--- reference record (spec Verification, point 1) ---")
        print(f"  output_text.done events                 : {reference['n_text_done_events']}")
        print(
            f"  done payloads containing tool markup    : "
            f"{reference['n_done_payloads_with_tool_markup']} "
            f"at prefixes {reference['leaking_prefixes']} "
            f"lengths {reference['leaked_payload_lengths']}"
        )
        print(
            f"  tool calls surfaced                     : "
            f"{reference['n_tool_calls_surfaced']} "
            f"{reference['tool_call_names']}"
        )
        print(f"  final output_text.done length           : {reference['final_text_done_len']}")
        print(
            f"  spec: no markup in any done payload     : "
            f"{'PASS' if reference['no_markup_in_any_done'] else 'FAIL'}"
        )
        print(
            f"  spec: final text is 116 chars           : "
            f"{'PASS' if reference['final_text_done_len'] == 116 else 'FAIL'}"
        )
        print(
            f"  spec: both calls recovered              : "
            f"{'PASS' if reference['n_tool_calls_surfaced'] == 2 else 'FAIL'}"
        )
        print()

    # An errored replay emits nothing, and nothing trivially satisfies the
    # invariant. Reporting that as a pass is the one way this harness could
    # lie about a fix, so errors outrank the violation count.
    if summary["errors"]:
        verdict = f"INCONCLUSIVE - {len(summary['errors'])} replays raised before finishing"
    elif summary["total_violations"]:
        verdict = f"INVARIANT VIOLATED - {summary['total_violations']} violating done events"
    else:
        verdict = "INVARIANT HOLDS"
    print(f"VERDICT: {verdict}")
    print(line)


def print_violation_details(violations: list[Violation], limit: int) -> None:
    if not violations:
        return
    print()
    print("=" * 78)
    print(f"Violation detail (first {min(limit, len(violations))} of {len(violations)})")
    print("=" * 78)
    for violation in violations[:limit]:
        print(
            f"\nrecord {violation.record_id} mode={violation.mode} "
            f"prefix={violation.prefix} item={violation.item_id} "
            f"kind={violation.kind}"
        )
        print(f"  streamed via deltas : {len(violation.streamed)} chars")
        print(
            f"  reported by done    : {len(violation.done)} chars "
            f"(+{violation.extra_chars} never streamed, "
            f"-{violation.missing_chars} dropped, "
            f"{violation.markup_chars} chars of markup)"
        )
        print("  --- diff ---")
        for diff_line in violation.diff().splitlines():
            print(f"  {diff_line}")


def compare(previous: dict[str, Any], current: dict[str, Any]) -> None:
    print()
    print("=" * 78)
    print("BEFORE -> AFTER")
    print("=" * 78)
    for mode in sorted(set(previous["by_mode"]) | set(current["by_mode"])):
        before = previous["by_mode"].get(mode, {})
        after = current["by_mode"].get(mode, {})
        print(f"--- mode: {mode} ---")
        for key in (
            "records_violating",
            "violations",
            "violations_with_tool_markup",
            "total_markup_chars",
            "total_extra_chars",
            "output_text_delta_events",
            "output_text_done_events",
            "tool_call_items",
        ):
            print(f"  {key:<30} {before.get(key, '-'):>10} -> {after.get(key, '-'):>10}")
        print()

    # The one way a "fixed" run can be worse: fewer calls reach the client.
    # Satisfying the invariant by emitting less is not the fix being asked for.
    old_calls = previous.get("calls_recovered") or {}
    new_calls = current.get("calls_recovered") or {}
    regressed = sorted(
        (record_id, old_calls[record_id], new_calls[record_id])
        for record_id in set(old_calls) & set(new_calls)
        if new_calls[record_id] < old_calls[record_id]
    )
    gained = sum(
        1
        for record_id in set(old_calls) & set(new_calls)
        if new_calls[record_id] > old_calls[record_id]
    )
    print("--- tool call recovery ---")
    print(f"  records comparable            : {len(set(old_calls) & set(new_calls))}")
    print(f"  records surfacing FEWER calls : {len(regressed)}")
    print(f"  records surfacing more calls  : {gained}")
    for record_id, before_n, after_n in regressed[:25]:
        print(f"    {record_id:<22} {before_n} -> {after_n}")
    if len(regressed) > 25:
        print(f"    ... {len(regressed) - 25} more")
    print()


DEFAULT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_DATA = Path(__file__).resolve().parent / "data"


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--root", default=str(DEFAULT_ROOT), help="TensorRT-LLM checkout whose source is replayed"
    )
    parser.add_argument(
        "--responses-utils",
        default=None,
        help="pin tensorrt_llm/serve/responses_utils.py to "
        "this file instead of the one under --root; use a "
        "`git show <rev>:...` snapshot for a reproducible "
        "BEFORE baseline",
    )
    parser.add_argument(
        "--frames-json",
        default=str(DEFAULT_DATA / "responses_replay_frames.json"),
        help="the reference record ({frames, text}); pass '' to skip",
    )
    parser.add_argument(
        "--records", action="append", default=None, help=".jsonl[.gz] of raw records; repeatable"
    )
    parser.add_argument(
        "--min-calls",
        type=int,
        default=0,
        help="only replay records with >= this many <tool_call> markers",
    )
    parser.add_argument("--mode", choices=("stream", "truncate", "both"), default="both")
    parser.add_argument(
        "--truncate-max-frames",
        type=int,
        default=64,
        help="skip truncate mode for records longer than this (it is O(n^2) in frames)",
    )
    parser.add_argument("--reasoning-parser", default="glm47")
    parser.add_argument("--tool-parser", default="glm47")
    parser.add_argument("--label", default="BEFORE")
    parser.add_argument("--out-json", default=None)
    parser.add_argument("--compare", default=None, help="a previous --out-json to diff against")
    parser.add_argument(
        "--detail", type=int, default=3, help="how many violations to print in full"
    )
    parser.add_argument("--progress", action="store_true")
    parser.add_argument(
        "--parser-log-level",
        default="ERROR",
        help="level for the parsers' own logger; raise to "
        "WARNING to see 'undefined function' notices, which "
        "are mostly artefacts of the reconstructed tool list",
    )
    args = parser.parse_args(argv)

    # `Glm4ToolParser` argument coercion runs `ast.literal_eval` over
    # model-generated argument values (glm4_parser.py:79), so any value holding
    # a backslash escape Python does not know - `\|` in a shell command, say -
    # raises a SyntaxWarning from `<unknown>:1`. That is shipped behaviour
    # rather than anything this replay did, and at corpus scale it buries the
    # report.
    warnings.filterwarnings("ignore", category=SyntaxWarning)
    logging.basicConfig(stream=sys.stderr)
    logging.getLogger("trtllm.standalone").setLevel(
        getattr(logging, args.parser_log_level.upper(), logging.ERROR)
    )

    responses_utils, provenance = load_responses_utils(args.root, args.responses_utils)

    records: list[Record] = []
    reference_id: Optional[str] = None
    if args.frames_json:
        reference = load_records([args.frames_json])
        if reference:
            reference_id = reference[0].record_id
            records.extend(reference)
    paths = [p for p in (args.records or []) if p]
    if paths:
        existing = {r.text for r in records}
        for record in load_records(paths, min_calls=args.min_calls):
            if record.text not in existing:
                records.append(record)
    # The two loads uniquify independently; the combined list must be unique
    # too, or the reference record merges with a same-numbered corpus record.
    records = _uniquify_ids(records)

    if not records:
        parser.error("no records to replay")

    modes = ("stream", "truncate") if args.mode == "both" else (args.mode,)
    result = run(
        responses_utils,
        records,
        modes=modes,
        reasoning_parser_id=args.reasoning_parser or None,
        tool_parser_id=args.tool_parser or None,
        truncate_max_frames=args.truncate_max_frames,
        reference_id=reference_id,
        progress=args.progress,
    )
    summary = summarise(result)
    print_report(summary, provenance, args.label)
    print_violation_details(result.violations, args.detail)

    if args.out_json:
        blob = {
            "label": args.label,
            "provenance": provenance.to_dict(),
            "argv": sys.argv[1:],
            "parsers": {
                "reasoning": args.reasoning_parser,
                "tool": args.tool_parser,
            },
            "summary": summary,
            "violations": [v.to_dict() for v in result.violations],
        }
        Path(args.out_json).write_text(json.dumps(blob, indent=2, ensure_ascii=False))
        print(f"\nwrote {args.out_json}")

    if args.compare:
        previous = json.loads(Path(args.compare).read_text())
        compare(previous["summary"], summary)

    return 1 if (summary["total_violations"] or summary["errors"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
