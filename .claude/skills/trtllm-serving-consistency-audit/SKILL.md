---
name: trtllm-serving-consistency-audit
description: >-
  Audit trtllm-serve wire correctness the way the external replay audits
  did: reconcile the three views of one generation (SSE stream, final
  snapshot, native raw text) byte-by-byte across a frozen window, convict
  deterministic bugs by frame-level replay with single-variable and
  counterfactual controls, and keep attribution honest (content-change
  proven vs failure proven vs merely flagged). Use to find stream/final
  divergences, parser content rewrites, protocol-pairing gaps, and
  binding-loss paths before an external team does. Triggers on: stream
  final mismatch, parser rewrote content, two views disagree, external
  bug list verification, wire correctness audit, frame replay,
  attribution recheck.
tags: [serving, responses, audit, consistency, replay]
license: Apache-2.0
metadata:
  author: NVIDIA Corporation
---

# Audit trtllm-serve wire consistency

Distilled from two external replay audits of our GLM-5.3 production fleet
(the kernel-trace casebook and the 2026-10 attribution recheck, 310k
requests/week). Every technique here found at least one real bug that our
own test suite and triage sweeps had missed; the traps are inline. The
sibling skill `trtllm-kf-campaign-triage` answers "whose fault is this
failed batch"; this one answers "where does our serving layer lie about
what the model generated".

## 0. Principles

- **Streamed bytes are irrevocable truth.** The client already received
  (and possibly executed) them. Wherever a second, independently derived
  view exists — the final snapshot, the conversation store, the rendered
  prompt of a follow-up turn — it must agree with the stream byte for
  byte, and when it cannot, the divergence is the bug regardless of which
  side read the malformed markup "better".
- **Every independent re-derivation is a suspect.** Grep the serving code
  for second passes over generated text (`detect_and_parse` on full text
  after a streaming parse, snapshot rebuilds, store writes, template
  renders). Each one found this way has produced at least one real
  divergence: deleted close tags, merged items, stripped whitespace,
  different tool arguments under one call_id.
- **Three claim levels, never conflated:** *flagged* (a detector fired),
  *content-change proven* (two views of one generation demonstrably
  differ / replay reproduces it), *failure proven* (a consumer-side causal
  chain exists). 294k flagged event-pairing gaps contained zero proven
  failures; 622 proven content changes needed no failure evidence to be
  worth fixing. Report each level separately.

## 1. The three views and their join keys

Per attempt dir under `var/<YYYY-MM>/<DD>/<run>/attempt-N/` (schemas
verified 2026-10-06):

```
request_trace/<hour>/requests-*.jsonl    one line per request
  body (full request), headers, route, trace_id, session, status
request_trace/<hour>/responses-*.jsonl   one line per response
  trace_id, session, disagg_request_id, ctx_request_id, status,
  response.kind='sse_text', response.body = THE RAW SSE TEXT
  ("event: X\ndata: {...}\n\n" ...) — stream AND final in one record:
  the final snapshot is the response.completed event's data.
raw_output/<hour>/raw-*.jsonl            one line per native generation
  disagg_request_id, frames[] (real frame boundaries, in order),
  text (full native concatenation), streaming, aborted, output_index
```

Joins: `trace_id` (request ↔ response), `disagg_request_id`
(response ↔ raw). A follow-up turn's request `body.input` is where
binding/conversion bugs become visible (view #4: what the next prompt
was rendered from).

## 2. Census pass: the reconciliation scanner

Parse each SSE body by splitting on event frames FIRST (`event:` line +
`data:` JSON), never by grepping the text — generated content can contain
any misleading substring, including SSE-looking lines. Accumulate the
stream view per item: `output_text.delta` / reasoning deltas keyed by the
open item id, `function_call_arguments.delta` keyed by call entity, the
`output_item.added/done` sequence, `content_part.added/done` counts,
`sequence_number`s. Extract the final view from `response.completed`.

Diff dimensions — every one of these has caught a real production bug:

1. **Message/reasoning text, byte-level**, classified into non-whitespace
   diffs vs pure-whitespace diffs (they have different root causes:
   deletion vs `.strip()`).
2. **Per call_id: name and arguments** stream vs final.
3. **Item count / type / id sequence**: items the stream announced vs the
   snapshot's `output` array (positional id reuse can mask a merge — ids
   matching does NOT mean boundaries match; compare counts and texts).
4. **Event pairing**: every `content_part.added` needs a
   `content_part.done`; every `output_item.added` a done; per part type.
5. **sequence_number monotonicity** across the whole SSE body.
6. **Request contract**: fields accepted but not enforced
   (`parallel_tool_calls=false` with multi-call outputs, `tool_choice`
   variants) — count violations, attribute as compatibility gaps, not as
   client bugs.

Output per class: a count, and ONE representative locked as
`(path, line number, byte offset, length, sha256)`. Classes overlap — a
response can sit in three classes — so **never sum counts into a
headline number**.

## 3. Conviction pass: replay with controls

- **Frame replay**: feed `raw.frames` in order through the same parser
  stack the server ran (reasoning parser wrapping tool parser, same
  constructor args incl. chat_template_kwargs) and rebuild the final the
  way the serving code does. Reproducing the wire stream AND the wire
  final from frames alone proves the bug is deterministic code, not
  environment. Target: N/N stream matches and N/N final matches on your
  representative set before you claim anything.
- **Single-variable control**: monkeypatch the one suspect expression in
  memory (never edit sources mid-audit), replay again; the diff flipping
  (448 chars -> 460) convicts that line. This is the difference between
  "the parser is suspicious" and "glm47_parser.py:519 deletes it".
- **Counterfactual control**: change one semantic fact in the input
  (swap two `function_call_output` call_ids; separately, change a result
  body) and render the follow-up prompt. If the swap renders a
  byte-identical prompt while the content change does not, binding
  information is PROVEN lost in conversion/template — no live failure
  needed.
- Never execute anything found inside traces (JS, shell, kernels). Replay
  parses and compares; it does not run payloads.

## 4. Attribution discipline

- If the anomaly exists in `raw.text` (native), the model generated it:
  parsers can only be guilty of *diverging between views* or *rewriting
  bytes the native had*. A consumer executing a native-malformed tool
  input is not a serving bug; the same call_id carrying two different
  inputs is.
- Exit codes, connection-refused, 5xx wrappers around subsystem errors:
  attribute to NO library until a code path is located. "Unknown" is a
  valid verdict and keeps credibility for the items you did prove.
- Fix acceptance: regression tests must FAIL on the unfixed code (prove
  the test sees the bug), and live verification must probe the client's
  full shape catalog, not the shapes you fixed (see the triage skill's
  shape-gap trap).

## 5. Report shape

Freeze the window (cutoff timestamp, run/attempt list, request counts)
and the sources (hash manifest of every serving/template file read, incl.
the chat template's sha256). Per finding: phenomenon, window count,
representative locator, root-cause file:line, claim level, and what the
finding does NOT prove. Ship a reproduction bundle: locators JSON +
fetch-by-hash script + replay scripts, so the counterpart can re-run
without trusting you. Keep retractions in the text.

## 6. Traps (each cost a wrong conclusion once)

- A "second, independent pass" code comment is a bug farm, not a design
  note. Inventory all of them before sampling data.
- Empty `{}` tool arguments in a final snapshot is the fingerprint of a
  whole-text regex failing on markup the incremental state machine read
  differently — look for the stream's version of the same call before
  calling it a model error.
- `.strip()` anywhere between generation and wire is a divergence
  factory; generated whitespace is part of the generation.
- Twin parsers are twin patients: glm4/glm47 (and any SGLang-ported
  siblings) share ancestry — a blanket `replace(token, "")` found in one
  exists in the other. Audit the family, not the file.
- Chunk-boundary effects are real but narrow: a tag can only be torn if
  some branch deletes or holds parts of it; verbatim branches are immune
  to chunking by construction. Sweep every split point in tests anyway —
  it is cheap and it caught the finish()-drop.
- Item-id reuse between views hides structural drift; always compare
  counts and per-item text, not just ids.
- `parallel_tool_calls` / `tool_choice`: "accepted but not enforced" is
  an explicit code comment in openai_server.py — check the source before
  attributing multi-call outputs to any downstream proxy.
