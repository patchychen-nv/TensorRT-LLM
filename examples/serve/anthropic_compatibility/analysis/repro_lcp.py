#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Offline reproduction of Anthropic prompt-LCP drops from captured request bodies.

Replays captured `/v1/messages` bodies through the same server-side path that
feeds ``AnthropicPromptLcpTracker`` -- ``convert_anthropic_request`` then the
chat-template render then tokenization -- and compares the reproduced LCP against
the value the live server recorded in the audit log.

CPU only: no GPU and no model weights are needed, but the TensorRT-LLM runtime
deps (torch, transformers) and the served model's tokenizer must be importable,
so run this inside the serving container.

    python3 repro_lcp.py <run_dir> --model-dir /path/to/served/model

Control gate: turns whose recorded LCP is healthy must reproduce exactly. If they
do not, the harness does not faithfully mirror the server and its verdict on the
dropped turns is meaningless -- the script says so and stops.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Loading captured requests
# ---------------------------------------------------------------------------
def load_timeline(run_dir: Path) -> list[dict[str, Any]]:
    csv_path = run_dir / "analysis" / "timeline.csv"
    if not csv_path.exists():
        sys.exit(f"error: {csv_path} not found -- run analyze_audit.py first")
    with csv_path.open() as f:
        return list(csv.DictReader(f))


def load_body(run_dir: Path, capture_file: str) -> dict[str, Any]:
    path = run_dir / "anthropic_message_capture" / capture_file
    with gzip.open(path) as f:
        return json.load(f)["body"]


def pick_session(rows: list[dict[str, Any]], want: str | None) -> str:
    """Choose the session to replay: the requested one, else the one with the
    most LCP drops (the richest repro target)."""
    by_session: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_session[r["session_id"]].append(r)
    if want:
        matches = [s for s in by_session if s.startswith(want)]
        if not matches:
            sys.exit(f"error: no session starting with {want!r}")
        if len(matches) > 1:
            sys.exit(f"error: {want!r} is ambiguous: {matches}")
        return matches[0]

    def drops(session_rows):
        n = 0
        for r in session_rows:
            opp = r.get("current_reuse_opportunity_ratio") or ""
            if opp and float(opp) < 0.8 and int(r["session_turn_index"]) > 2:
                n += 1
        return n

    return max(by_session, key=lambda s: drops(by_session[s]))


# ---------------------------------------------------------------------------
# Mirror of the server-side render path
# ---------------------------------------------------------------------------
class Renderer:
    """body -> convert_anthropic_request -> chat template -> token ids.

    Mirrors OpenAIServer._prepare_chat_prompt (openai_server.py ~1505-1551).
    """

    def __init__(self, model_dir: str, tokenizer_dir: str | None = None,
                 custom_tokenizer: str | None = None):
        from transformers import AutoConfig

        from tensorrt_llm.inputs.utils import apply_chat_template
        from tensorrt_llm.tokenizer.tokenizer import (
            load_custom_tokenizer,
            tokenizer_factory,
        )
        from tensorrt_llm.serve.chat_utils import (
            parse_chat_messages_coroutines,
            resolve_top_level_model_type,
        )
        from tensorrt_llm.serve.anthropic_adapter import convert_anthropic_request

        self._apply_chat_template = apply_chat_template
        self._parse_messages = parse_chat_messages_coroutines
        self._resolve_model_type = resolve_top_level_model_type
        self._convert = convert_anthropic_request

        # The served config sets `custom_tokenizer: deepseek_v4`, and
        # apply_chat_template branches on the tokenizer *class* -- a plain
        # TransformersTokenizer takes the HF-template path instead of DeepSeek's
        # own, which would silently render a different prompt. Mirror the
        # server's construction exactly (py_executor_creator.py ~386).
        source = tokenizer_dir or model_dir
        if custom_tokenizer:
            self.tokenizer = load_custom_tokenizer(
                custom_tokenizer, source, trust_remote_code=True
            )
        else:
            self.tokenizer = tokenizer_factory(source)
        self.model_config = AutoConfig.from_pretrained(
            model_dir, trust_remote_code=True
        )
        self.model_type = self._resolve_model_type(self.model_config)

    def to_chat_request(self, body: dict[str, Any]):
        """Stage 1 -- the Anthropic -> OpenAI conversion (suspect A)."""
        from tensorrt_llm.serve.anthropic_protocol import AnthropicMessagesRequest

        return self._convert(AnthropicMessagesRequest(**body))

    def to_prompt(self, chat_request) -> str:
        """Stage 2 -- the chat-template render (suspect B)."""
        conversation, _mm_coros, mm_counts = self._parse_messages(
            chat_request.messages, self.model_config, None
        )
        tool_dicts = (
            None if chat_request.tools is None
            else [t.model_dump() for t in chat_request.tools]
        )
        return self._apply_chat_template(
            model_type=self.model_type,
            tokenizer=self.tokenizer,
            processor=None,
            conversation=conversation,
            add_generation_prompt=chat_request.add_generation_prompt,
            mm_placeholder_counts=mm_counts,
            tools=tool_dicts,
            documents=chat_request.documents,
            chat_template=chat_request.chat_template,
            chat_template_kwargs=chat_request.chat_template_kwargs or {},
        )

    def to_token_ids(self, prompt: str) -> list[int]:
        """Stage 3 -- tokenization."""
        return self.tokenizer.encode(prompt)

    def render(self, body: dict[str, Any]) -> tuple[Any, str, list[int]]:
        chat_request = self.to_chat_request(body)
        prompt = self.to_prompt(chat_request)
        return chat_request, prompt, self.to_token_ids(prompt)


# ---------------------------------------------------------------------------
# Three-level bisect
# ---------------------------------------------------------------------------
def first_diff(a, b) -> int | None:
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return None if len(a) == len(b) else n


def bisect_pair(renderer: Renderer, prev_body, cur_body, recorded_lcp) -> None:
    """Locate the layer that first introduces divergence.

    The first layer where the two requests differ is where the bug lives.
    """
    print("\n" + "=" * 74)
    print("THREE-LEVEL BISECT")
    print("=" * 74)

    prev_req, prev_prompt, prev_ids = renderer.render(prev_body)
    cur_req, cur_prompt, cur_ids = renderer.render(cur_body)

    # Level 1 -- conversion output
    prev_json = prev_req.model_dump_json()
    cur_json = cur_req.model_dump_json()
    d1 = first_diff(prev_json, cur_json)
    print(f"\n[L1] convert_anthropic_request output")
    print(f"     lengths {len(prev_json):,} vs {len(cur_json):,} chars")
    print(f"     first divergence at char {d1:,}" if d1 is not None
          else "     identical")
    if d1 is not None:
        lo = max(0, d1 - 90)
        print(f"     prev: …{prev_json[lo:d1 + 110]!r}")
        print(f"     cur : …{cur_json[lo:d1 + 110]!r}")

    # Level 2 -- rendered prompt string
    d2 = first_diff(prev_prompt, cur_prompt)
    print(f"\n[L2] rendered prompt string")
    print(f"     lengths {len(prev_prompt):,} vs {len(cur_prompt):,} chars")
    print(f"     first divergence at char {d2:,}" if d2 is not None
          else "     identical")
    if d2 is not None:
        lo = max(0, d2 - 90)
        print(f"     prev: …{prev_prompt[lo:d2 + 110]!r}")
        print(f"     cur : …{cur_prompt[lo:d2 + 110]!r}")

    # Level 3 -- token ids (this is what the tracker actually compares)
    d3 = first_diff(prev_ids, cur_ids)
    print(f"\n[L3] token ids  <-- what AnthropicPromptLcpTracker compares")
    print(f"     lengths {len(prev_ids):,} vs {len(cur_ids):,} tokens")
    print(f"     LCP = {d3:,}" if d3 is not None else "     identical")
    print(f"     server recorded LCP = {recorded_lcp:,}")
    if d3 is not None:
        match = "MATCH" if abs(d3 - recorded_lcp) <= 2 else "MISMATCH"
        print(f"     reproduced vs recorded: {match}")
        lo = max(0, d3 - 40)
        print(f"     prev decoded: …{renderer.tokenizer.decode(prev_ids[lo:d3 + 60])!r}")
        print(f"     cur  decoded: …{renderer.tokenizer.decode(cur_ids[lo:d3 + 60])!r}")

    print("\n" + "-" * 74)
    verdict = ("convert_anthropic_request" if d1 is not None else
               "chat-template render" if d2 is not None else
               "tokenizer" if d3 is not None else
               "nothing -- prompts are identical end to end")
    print(f"VERDICT: first divergence is introduced by: {verdict}")
    if d1 is None and d2 is None and d3 is None:
        print("  The two prompts are byte-identical, so an LCP of "
              f"{recorded_lcp:,} cannot come from the request pair alone.")
        print("  -> the drop is runtime state, not the conversion. Try --permute.")


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------
def replay(run_dir: Path, session: str, renderer: Renderer,
           limit: int, order: list[int] | None = None) -> tuple[int, int, list]:
    from tensorrt_llm.serve.anthropic_adapter import AnthropicPromptLcpTracker

    rows = [r for r in load_timeline(run_dir) if r["session_id"] == session]
    rows.sort(key=lambda r: r["started_at"])
    rows = rows[:limit]
    if order is not None:
        rows = [rows[i] for i in order]

    tracker = AnthropicPromptLcpTracker()
    ok = bad = 0
    results = []
    print(f"{'g':>5} {'turn':>5} {'recorded':>10} {'reproduced':>11} {'':>4}")
    for r in rows:
        body = load_body(run_dir, r["message_capture_file"])
        _req, _prompt, ids = renderer.render(body)
        metrics = tracker.observe(session, ids)
        repro = metrics["prompt_lcp_tokens"]
        rec = r["prompt_lcp_tokens"]
        rec_i = int(rec) if rec not in ("", "None", None) else None
        if rec_i is None or repro is None:
            flag = "-"
        elif abs(repro - rec_i) <= 2:
            flag = "ok"
            ok += 1
        else:
            flag = "FAIL"
            bad += 1
        print(f"{r['global_request_index']:>5} {r['session_turn_index']:>5} "
              f"{rec_i if rec_i is not None else '-':>10} "
              f"{repro if repro is not None else '-':>11} {flag:>4}")
        results.append((r, rec_i, repro))
    return ok, bad, results


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir", type=Path)
    p.add_argument("--model-dir", required=True,
                   help="Served model directory (for AutoConfig + tokenizer)")
    p.add_argument("--tokenizer-dir", default=None,
                   help="Tokenizer directory if different from --model-dir")
    p.add_argument("--custom-tokenizer", default=None,
                   help="Custom tokenizer alias from the served agg_config.yaml "
                        "(this run used 'deepseek_v4'); omit only if the config "
                        "has no custom_tokenizer key")
    p.add_argument("--session", default=None,
                   help="Session id prefix (default: the one with most drops)")
    p.add_argument("--limit", type=int, default=30,
                   help="Replay only the first N turns (default 30)")
    p.add_argument("--permute", type=int, default=0, metavar="N",
                   help="If the drop does not reproduce, try N random "
                        "orderings to test the async-observation hypothesis")
    args = p.parse_args()

    rows = load_timeline(args.run_dir)
    session = pick_session(rows, args.session)
    print(f"run     : {args.run_dir}")
    print(f"session : {session}")

    renderer = Renderer(args.model_dir, args.tokenizer_dir,
                        args.custom_tokenizer)
    print(f"model   : {renderer.model_type}")
    print(f"tokenizer: {type(renderer.tokenizer).__name__}\n")

    ok, bad, results = replay(args.run_dir, session, renderer, args.limit)

    print("\n" + "=" * 74)
    print(f"CONTROL GATE: {ok} turns reproduced exactly, {bad} mismatched")
    if bad:
        print("FAILED -- the harness does not mirror the server faithfully.")
        print("Its verdict on the dropped turns would be meaningless. Fix the")
        print("render path (tokenizer, model config, chat template) first.")
        sys.exit(2)
    print("PASSED -- reproduced every recorded LCP, harness is faithful.")

    # First reproduced drop -> bisect it
    drop = None
    for i, (r, rec, repro) in enumerate(results):
        if i and rec and repro and repro < 0.8 * (results[i - 1][2] or 1):
            drop = i
            break
    if drop is None:
        print("\nNo LCP drop in the replayed window; raise --limit.")
        return

    r_prev, _, _ = results[drop - 1]
    r_cur, rec_cur, _ = results[drop]
    print(f"\nReproduced drop at g={r_cur['global_request_index']} "
          f"(turn {r_cur['session_turn_index']}): LCP {rec_cur:,}")
    bisect_pair(renderer,
                load_body(args.run_dir, r_prev["message_capture_file"]),
                load_body(args.run_dir, r_cur["message_capture_file"]),
                rec_cur)


if __name__ == "__main__":
    main()
