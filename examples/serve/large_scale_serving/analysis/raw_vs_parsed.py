# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""What the model emitted, beside what the client received. Joined on the engine request id.

Sources inside one attempt directory:
  raw_output/<UTC hour>/raw-<pid>.jsonl        detokenized text, before any parser (RawOutputDump)
  request_trace/<UTC hour>/responses-<pid>.jsonl   the SSE stream the client got, after the parsers

Joining them takes two keys, because neither covers both deployments.

`disagg_request_id` is the one that matters here. The proxy stamps it on its trace line as a field,
and the orchestrator puts the same value in the request it sends the generation worker, so it is
shared across the two processes by construction. It is null when serving is aggregated.

`request_id` is the engine's own counter, and is the fallback for the aggregated case, where it
reaches the client inside the response id and so appears in the trace as the `<id>` in
`"id": "chatcmpl-<id>"`.

Using only the response id looked right and is wrong twice over, which real traffic showed and a
smoke test did not. On a disaggregated deployment the generation worker's counter never leaves it --
it reads 10, 11, 12 while the proxy knows the same requests as 10349323008212992 -- so the two sets
do not intersect at all. And the Responses API, which is what the agents actually speak, mints a
fresh `resp_<uuid>` per response that is derived from no engine id whatsoever. A chat-completions
request joins on the response id; nothing else does.

The two directories are not siblings by accident. `request_trace` is a symlink out to
`trace.request_root` (the traces are consumed by another team); `raw_output` is a real directory
under the attempt, because it is short-lived debugging output. Both are found from the attempt root
either way.

What to look at
---------------
`--only-diff` is the interesting mode: it keeps requests where the text the parsers emitted is not
the text the model produced. Some of that difference is the parsers doing their job -- a reasoning
parser moves text into `reasoning_content`, which this reassembles, so those come back equal. What
survives is text that went in and did not come out anywhere: a dropped tool call, a swallowed tag, a
truncated block.

`--frames` prints the per-chunk deltas. Incremental parsers fail on the seams -- a tool-call marker
split across two chunks reads as one the model never emitted -- and the seam is only visible here;
`text` has already joined them.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

# "id":"chatcmpl-288230376151711746" -> the engine request id. Both prefixes appear: chat mints
# `chatcmpl-`, /v1/completions mints `cmpl-`. Anchored on the quotes so a prefix appearing in
# generated text cannot be mistaken for the response's own id.
RESPONSE_ID = re.compile(r'"id"\s*:\s*"(?:chatcmpl|cmpl)-(\d+)"')

# Text the client was actually shown, in the order the parsers emitted it. `content` and
# `reasoning_content` are both included: a reasoning parser moves text from one to the other, and
# counting only `content` would report every reasoning model as losing its entire preamble.
DELTA_TEXT = re.compile(r'"(?:content|reasoning_content|text)"\s*:\s*"((?:[^"\\]|\\.)*)"')


def find_attempt(start: Path) -> Optional[Path]:
    """The attempt directory at or above `start`, recognised by holding raw_output."""
    for candidate in [start] + list(start.parents):
        if (candidate / "raw_output").is_dir():
            return candidate
    return None


def read_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    """Lines that parse. A torn last line is normal on a file still being appended to."""
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except ValueError:
                continue


def join_key(record: Dict[str, Any]) -> str:
    """The id a raw record shares with whatever else wrote about the request.

    `disagg_request_id` first: on a disaggregated deployment it is the only one
    the proxy also knows. `request_id` otherwise, which is what an aggregated
    server puts in the response id.
    """
    for field in ("disagg_request_id", "request_id"):
        value = record.get(field)
        if value not in (None, ""):
            return str(value)
    return ""


def load_raw(attempt: Path) -> Dict[str, Dict[str, Any]]:
    """Join key -> raw record. Later records win, which matters only for a reused id."""
    raw: Dict[str, Dict[str, Any]] = {}
    for path in sorted((attempt / "raw_output").glob("*/raw-*.jsonl")):
        for record in read_jsonl(path):
            key = join_key(record)
            if key:
                raw[key] = record
    return raw


def parsed_text(body: Any) -> str:
    r"""Reassemble what the client was shown from a trace response body.

    Streaming bodies are SSE text and non-streaming ones are a JSON object; both are searched with
    the same regex rather than parsed, because the goal is "every piece of text that reached the
    client" and the two schemas nest it differently. Escapes are decoded through json so a `\\n` in
    the stream compares equal to a newline in the raw record.
    """
    if not isinstance(body, str):
        body = json.dumps(body, ensure_ascii=False)
    out: List[str] = []
    for chunk in DELTA_TEXT.findall(body):
        try:
            out.append(json.loads('"%s"' % chunk))
        except ValueError:
            out.append(chunk)
    return "".join(out)


def load_traces(attempt: Path) -> Dict[str, Dict[str, Any]]:
    """Join key -> {trace_id, session, status, parsed}.

    A trace line is filed under every id it offers: its own `disagg_request_id`
    field, and any engine id embedded in a response id in the body. One of the
    two matches the raw side depending on the deployment, and filing both means
    the reader does not have to know which.
    """
    traces: Dict[str, Dict[str, Any]] = {}
    trace_dir = attempt / "request_trace"
    if not trace_dir.is_dir():
        return traces
    for path in sorted(trace_dir.glob("*/responses-*.jsonl")):
        for record in read_jsonl(path):
            body = (record.get("response") or {}).get("body")
            haystack = body if isinstance(body, str) else json.dumps(body, ensure_ascii=False)
            keys = set(RESPONSE_ID.findall(haystack or ""))
            for field in ("disagg_request_id", "client_id"):
                value = record.get(field)
                if value not in (None, ""):
                    keys.add(str(value))
            if not keys:
                continue
            entry = {
                "trace_id": record.get("trace_id", ""),
                "session": record.get("session", ""),
                "status": record.get("status", ""),
                "parsed": parsed_text(body),
            }
            for key in keys:
                traces[key] = entry
    return traces


def rows(attempt: Path) -> List[Dict[str, Any]]:
    raw, traces = load_raw(attempt), load_traces(attempt)
    out = []
    for key, record in sorted(raw.items()):
        trace = traces.get(key)
        model_text = record.get("text") or ""
        client_text = trace["parsed"] if trace else None
        out.append(
            {
                "request_id": record.get("request_id"),
                "disagg_request_id": record.get("disagg_request_id"),
                "key": key,
                "output_index": record.get("output_index"),
                "streaming": record.get("streaming"),
                "aborted": record.get("aborted"),
                "frames": record.get("frames") or [],
                "raw": model_text,
                "parsed": client_text,
                "trace_id": trace["trace_id"] if trace else "",
                "session": trace["session"] if trace else "",
                "status": trace["status"] if trace else "",
                # None means the request was never matched to a trace line at all, which is a
                # different thing from matching one whose text differs; kept apart so a missing
                # trace cannot be read as a parser dropping everything.
                "lost": None if client_text is None else len(model_text) - len(client_text),
            }
        )
    return out


def brief(text: str, limit: int) -> str:
    text = text.replace("\n", "\\n")
    return text if len(text) <= limit else text[: limit - 1] + "…"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "attempt",
        nargs="?",
        default=".",
        help="attempt directory, or anything inside one (default: cwd)",
    )
    parser.add_argument(
        "--only-diff",
        action="store_true",
        help="only requests where the client's text differs from the model's",
    )
    parser.add_argument(
        "--frames",
        action="store_true",
        help="print the per-chunk deltas, where incremental parsers break",
    )
    parser.add_argument("--request", default="", help="one request id, printed in full")
    parser.add_argument("--width", type=int, default=100, help="truncation width (0 = no limit)")
    parser.add_argument("--json", action="store_true", help="emit rows as JSONL instead of a table")
    args = parser.parse_args()

    start = Path(args.attempt).resolve()
    attempt = find_attempt(start)
    if attempt is None:
        print(
            "no raw_output/ at or above %s -- is this an attempt directory, and was the "
            "deployment run with server.raw_output: true?" % start,
            file=sys.stderr,
        )
        return 2

    table = rows(attempt)
    if args.request:
        table = [
            r
            for r in table
            if args.request in (r["key"], str(r["request_id"]), str(r["disagg_request_id"]))
        ]
        args.frames = True
        args.width = 0
    if args.only_diff:
        table = [r for r in table if r["parsed"] is not None and r["raw"] != r["parsed"]]

    if args.json:
        for row in table:
            print(json.dumps(row, ensure_ascii=False))
        return 0

    if not table:
        print("no rows (attempt: %s)" % attempt)
        return 0

    width = args.width or 10**9
    matched = sum(1 for r in table if r["parsed"] is not None)
    print("attempt: %s" % attempt)
    print("%d raw record(s), %d joined to a trace response\n" % (len(table), matched))
    for row in table:
        head = "request %s" % row["key"]
        if row["output_index"]:
            head += " [output %s]" % row["output_index"]
        if row["trace_id"]:
            head += "  trace=%s session=%s" % (row["trace_id"], row["session"])
        if row["aborted"]:
            head += "  ABORTED"
        print(head)
        print("  model :  %s" % brief(row["raw"], width))
        if row["parsed"] is None:
            print("  client:  (no trace response found for this id)")
        else:
            print("  client:  %s" % brief(row["parsed"], width))
            if row["lost"]:
                verdict = "%+d chars" % -row["lost"]
                print(
                    "  delta :  %s%s"
                    % (verdict, "  <-- text the client never saw" if row["lost"] > 0 else "")
                )
        if args.frames:
            print("  frames:  %d" % len(row["frames"]))
            for i, frame in enumerate(row["frames"]):
                print("    [%02d] %s" % (i, brief(frame, width)))
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
