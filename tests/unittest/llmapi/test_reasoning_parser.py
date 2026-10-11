# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import json
import os

import pytest

from tensorrt_llm.llmapi.reasoning_parser import (
    MODEL_TYPE_TO_REASONING_PARSER, GlmReasoningParser,
    NemotronV3ReasoningParser, ReasoningParserFactory, ReasoningParserResult,
    resolve_auto_reasoning_parser)

pytestmark = pytest.mark.cpu_only

R1_START, R1_END = "<think>", "</think>"


@pytest.mark.parametrize(("text", "content", "reasoning_context"), [
    ("a b", "", "a b"),
    (f"{R1_END} a b", " a b", ""),
    (f"a {R1_END} b", " b", "a "),
    (f"a b {R1_END}", "", "a b "),
    (f"{R1_START} a {R1_END} b", " b", f"{R1_START} a "),
])
def test_deepseek_r1_reasoning_parser(text: str, content: str,
                                      reasoning_context: str):
    reasoning_parser = ReasoningParserFactory.create_reasoning_parser(
        "deepseek-r1")
    result = reasoning_parser.parse(text)
    assert result.content == content
    assert result.reasoning_content == reasoning_context


@pytest.mark.parametrize(("delta_texts", "content", "reasoning_context"), [
    (["a", "b"], ["", ""], ["a", "b"]),
    ([R1_END, "a", "b"], ["", "a", "b"], ["", "", ""]),
    (["a", R1_END, "b"], ["", "", "b"], ["a", "", ""]),
    (["a", "b", R1_END], ["", "", ""], ["a", "b", ""]),
    (["a", f"l{R1_END}", "b"], ["", "", "b"], ["a", "l", ""]),
    (["a", f"l{R1_END}r", "b"], ["", "r", "b"], ["a", "l", ""]),
    (["a", f"{R1_END}r", "b"], ["", "r", "b"], ["a", "", ""]),
])
def test_deepseek_r1_reasoning_parser_stream(delta_texts: list, content: list,
                                             reasoning_context: list):
    reasoning_parser = ReasoningParserFactory.create_reasoning_parser(
        "deepseek-r1")
    for i, delta_text in enumerate(delta_texts):
        result = reasoning_parser.parse_delta(delta_text)
        assert result.content == content[i]
        assert result.reasoning_content == reasoning_context[i]


@pytest.mark.parametrize(
    ("parser_key", "text"),
    [
        # `finish()` flushes as reasoning: the stream starts inside the
        # reasoning block, so the withheld `<` is reasoning output.
        ("deepseek-r1", "a <"),
        # `finish()` flushes as content: no reasoning block was entered.
        ("qwen3", "a<"),
        # `finish()` discards: the buffer holds exactly a delimiter. Passes on
        # `main` too - it guards against a fix that leaks the tag instead.
        ("deepseek-r1", f"a{R1_END}"),
    ])
def test_deepseek_r1_reasoning_parser_stream_matches_non_stream(
        parser_key: str, text: str) -> None:
    """Streaming char-by-char then finishing must match a non-streaming parse.

    One `(parser_key, text)` pair per branch of `finish()`. This is the
    contract the missing flush violated, so it subsumes example-based tests
    of the individual branches.
    """
    expected = ReasoningParserFactory.create_reasoning_parser(parser_key).parse(
        text)
    reasoning_parser = ReasoningParserFactory.create_reasoning_parser(
        parser_key)
    content, reasoning_context = "", ""
    for char in text:
        result = reasoning_parser.parse_delta(char)
        content += result.content
        reasoning_context += result.reasoning_content
    result = reasoning_parser.finish()
    content += result.content
    reasoning_context += result.reasoning_content
    assert content == expected.content
    assert reasoning_context == expected.reasoning_content


def test_deepseek_r1_reasoning_parser_finish_flushes_partial_tag() -> None:
    """A partial tag arriving alongside text must still be flushed.

    Such a delta fills `_buffer` through the `rfind` branch of `parse_delta`,
    which one-character-at-a-time streaming never reaches - and it is the
    shape a real stream delivers.
    """
    reasoning_parser = ReasoningParserFactory.create_reasoning_parser(
        "deepseek-r1")
    assert reasoning_parser.parse_delta("a </thin").reasoning_content == "a "
    assert reasoning_parser.finish().reasoning_content == "</thin"


@pytest.mark.parametrize(("chat_template_kwargs", "flushed"), [
    ({
        "thinking": True
    }, "<"),
    ({
        "thinking": False
    }, ""),
])
def test_deepseek_v4_reasoning_parser_finish_delegates(
        chat_template_kwargs: dict[str, bool], flushed: str) -> None:
    """`finish()` must reach whichever parser the thinking flag selected.

    `DeepSeekV4ReasoningParser` delegates to two different targets:
    `DeepSeekR1Parser`, which now flushes, and `IdentityReasoningParser`,
    which withholds nothing to flush.
    """
    reasoning_parser = ReasoningParserFactory.create_reasoning_parser(
        "deepseek_v4", chat_template_kwargs)
    reasoning_parser.parse_delta("a <")
    result = reasoning_parser.finish()
    assert result.reasoning_content == flushed
    assert result.content == ""


@pytest.mark.parametrize("chat_template_kwargs", [{
    "thinking": True
}, {
    "enable_thinking": True
}])
def test_deepseek_v4_reasoning_parser_extracts_when_thinking(
        chat_template_kwargs: dict):
    reasoning_parser = ReasoningParserFactory.create_reasoning_parser(
        "deepseek_v4", chat_template_kwargs)

    result = reasoning_parser.parse(f"hidden{R1_END}visible")

    assert result.content == "visible"
    assert result.reasoning_content == "hidden"


def test_deepseek_v4_reasoning_parser_streams_when_thinking():
    reasoning_parser = ReasoningParserFactory.create_reasoning_parser(
        "deepseek_v4", {"enable_thinking": True})

    deltas = ["hid", f"den{R1_END}visible", " tail"]
    results = [reasoning_parser.parse_delta(delta) for delta in deltas]

    assert [result.content for result in results] == ["", "visible", " tail"]
    assert [result.reasoning_content
            for result in results] == ["hid", "den", ""]


TOOL_START = "<|tool_calls_section_begin|>"


@pytest.mark.parametrize(
    ("text", "content", "reasoning_context"),
    [
        # Standard <think>...</think> patterns.
        ("a<think>b</think>c", "c", "b"),
        ("<think>a</think>b", "b", "a"),
        ("<think>a", "", "a"),
        ("a", "a", ""),
        ("<think>", "", ""),
        # Interleaved thinking: tool call section implicitly ends reasoning.
        (f"<think>reasoning{TOOL_START}tool_call_data",
         f"{TOOL_START}tool_call_data", "reasoning"),
        # </think> before tool call section: standard end wins.
        (f"<think>reasoning</think>text{TOOL_START}tool_call_data",
         f"text{TOOL_START}tool_call_data", "reasoning"),
        # No <think> tag at all – just content.
        (f"content{TOOL_START}tool_call_data",
         f"content{TOOL_START}tool_call_data", ""),
    ])
def test_kimi_k2_reasoning_parser(text: str, content: str,
                                  reasoning_context: str):
    """Test kimi_k2 reasoning parser (non-streaming).

    Kimi-K2-Thinking generates <think>...</think> tags and may implicitly
    end reasoning via <|tool_calls_section_begin|>.
    """
    reasoning_parser = ReasoningParserFactory.create_reasoning_parser("kimi_k2")
    result = reasoning_parser.parse(text)
    assert result.content == content
    assert result.reasoning_content == reasoning_context


@pytest.mark.parametrize(
    ("delta_texts", "content", "reasoning_context"),
    [
        # Standard streaming cases (same as qwen3).
        (["<think>a", "l</think>r", "b"], ["", "r", "b"], ["a", "l", ""]),
        (["<th", "ink>a</think>b"], ["", "b"], ["", "a"]),
        (["<think>a</th", "ink>b"], ["", "b"], ["a", ""]),
        (["<think>", "a</think>b"], ["", "b"], ["", "a"]),
        (["<think>a</think>", "b"], ["", "b"], ["a", ""]),
        # Interleaved thinking: tool call section implicitly ends reasoning.
        # When the tool token arrives as a full token, the parser buffers it
        # (prefix check) and emits it combined with the next delta.
        (
            ["<think>", "reasoning", TOOL_START, "tool_data"],
            ["", "", "", TOOL_START + "tool_data"],
            ["", "reasoning", "", ""],
        ),
        # Tool section arrives combined with preceding reasoning text.
        (
            ["<think>", "reasoning" + TOOL_START + "tool_data"],
            ["", TOOL_START + "tool_data"],
            ["", "reasoning"],
        ),
        # Partial start-tag at end of delta should be buffered (not leaked).
        (
            ["content<th", "ink>reason</think>after"],
            ["content", "after"],
            ["", "reason"],
        ),
        # Partial tool section tag at end of content after </think>.
        (
            ["<think>reason</think>content<|tool", "_calls_section_begin|>td"],
            ["content", TOOL_START + "td"],
            ["reason", ""],
        ),
    ])
def test_kimi_k2_reasoning_parser_stream(delta_texts: list, content: list,
                                         reasoning_context: list):
    """Test kimi_k2 reasoning parser streaming."""
    reasoning_parser = ReasoningParserFactory.create_reasoning_parser("kimi_k2")
    for i, delta_text in enumerate(delta_texts):
        result = reasoning_parser.parse_delta(delta_text)
        assert result.content == content[i], \
            f"Step {i}: delta={delta_text!r}, expected content={content[i]!r}, got {result.content!r}"
        assert result.reasoning_content == reasoning_context[i], \
            f"Step {i}: delta={delta_text!r}, expected reasoning={reasoning_context[i]!r}, got {result.reasoning_content!r}"


@pytest.mark.parametrize(
    ("parser_name", "delta_texts", "content", "reasoning_context"),
    [
        # Interleaved thinking with reasoning_at_start=True (deepseek-r1, minimax_m2):
        # Model output starts directly with reasoning (no <think> tag).
        # Simulates: reason1</think>text1<think>reason2</think>text2
        (
            "minimax_m2",
            ["reason1", R1_END, "text1", R1_START, "reason2", R1_END, "text2"],
            ["", "", "text1", "", "", "", "text2"],
            ["reason1", "", "", "", "reason2", "", ""],
        ),
        (
            "deepseek-r1",
            ["reason1", R1_END, "text1", R1_START, "reason2", R1_END, "text2"],
            ["", "", "text1", "", "", "", "text2"],
            ["reason1", "", "", "", "reason2", "", ""],
        ),
        # Interleaved thinking with reasoning_at_start=False (qwen3, kimi_k2):
        # Model output contains <think>...</think> tags.
        # Simulates: <think>reason1</think>content1<think>reason2</think>content2
        (
            "qwen3",
            [
                R1_START, "reason1", R1_END, "content1", R1_START, "reason2",
                R1_END, "content2"
            ],
            ["", "", "", "content1", "", "", "", "content2"],
            ["", "reason1", "", "", "", "reason2", "", ""],
        ),
        (
            "kimi_k2",
            [
                R1_START, "reason1", R1_END, "content1", R1_START, "reason2",
                R1_END, "content2"
            ],
            ["", "", "", "content1", "", "", "", "content2"],
            ["", "reason1", "", "", "", "reason2", "", ""],
        ),
        # Kimi-K2 interleaved thinking: reasoning interrupted by tool calls.
        # Simulates: <think>reasoning<|tool_calls_section_begin|>tool_data
        # Note: when TOOL_START arrives as a full token, the parser buffers it
        # (prefix check) and emits it combined with the next delta.
        (
            "kimi_k2",
            [R1_START, "reasoning", TOOL_START, "tool_data"],
            ["", "", "", TOOL_START + "tool_data"],
            ["", "reasoning", "", ""],
        ),
    ],
)
def test_interleaved_thinking_stream(parser_name: str, delta_texts: list,
                                     content: list, reasoning_context: list):
    """Test that streaming parsers correctly handle interleaved thinking.

    Interleaved thinking allows models to reason between tool calls,
    producing multiple <think>...</think> blocks within a single generation.
    The streaming parser must correctly transition between reasoning and
    content modes across multiple think blocks.

    For kimi_k2, reasoning may also be implicitly ended by the tool call
    section token <|tool_calls_section_begin|> without an explicit </think>.
    """
    reasoning_parser = ReasoningParserFactory.create_reasoning_parser(
        parser_name)
    for i, delta_text in enumerate(delta_texts):
        result = reasoning_parser.parse_delta(delta_text)
        assert result.content == content[i], \
            f"Step {i}: delta={delta_text!r}, expected content={content[i]!r}, got {result.content!r}"
        assert result.reasoning_content == reasoning_context[i], \
            f"Step {i}: delta={delta_text!r}, expected reasoning={reasoning_context[i]!r}, got {result.reasoning_content!r}"


@pytest.mark.parametrize(("text", "content", "reasoning_context"), [
    ("a<think>b</think>c", "c", "b"),
    ("<think>a</think>b", "b", "a"),
    ("<think>a", "", "a"),
    ("a", "a", ""),
    ("<think>", "", ""),
])
def test_qwen3_reasoning_parser(text: str, content: str,
                                reasoning_context: str):
    reasoning_parser = ReasoningParserFactory.create_reasoning_parser("qwen3")
    result = reasoning_parser.parse(text)
    assert result.content == content
    assert result.reasoning_content == reasoning_context


@pytest.mark.parametrize(("delta_texts", "content", "reasoning_context"), [
    (["<think>a", "l</think>r", "b"], ["", "r", "b"], ["a", "l", ""]),
    (["<th", "ink>a</think>b"], ["", "b"], ["", "a"]),
    (["<think>a</th", "ink>b"], ["", "b"], ["a", ""]),
    (["<think>", "a</think>b"], ["", "b"], ["", "a"]),
    (["<think>a</think>", "b"], ["", "b"], ["a", ""]),
    (["<think>a</th", "ank></th", "ink>b"], ["", "", "b"
                                             ], ["a", "</thank>", ""]),
])
def test_qwen3_reasoning_parser_stream(delta_texts: list, content: list,
                                       reasoning_context: list):
    reasoning_parser = ReasoningParserFactory.create_reasoning_parser("qwen3")
    for i, delta_text in enumerate(delta_texts):
        result = reasoning_parser.parse_delta(delta_text)
        assert result.content == content[i]
        assert result.reasoning_content == reasoning_context[i]


@pytest.mark.parametrize(("text", "content", "reasoning_context"), [
    (f"hidden{R1_END}visible", "visible", "hidden"),
    (f"{R1_END}visible", "visible", ""),
    (R1_END, "", ""),
    ("unterminated", "", "unterminated"),
])
def test_poolside_v1_reasoning_parser_when_thinking(
        text: str, content: str, reasoning_context: str) -> None:
    reasoning_parser = ReasoningParserFactory.create_reasoning_parser(
        "poolside_v1", {"enable_thinking": True})
    result = reasoning_parser.parse(text)
    assert result.content == content
    assert result.reasoning_content == reasoning_context


@pytest.mark.parametrize("chat_template_kwargs", [
    {
        "enable_thinking": False
    },
    {
        "thinking": False
    },
])
def test_poolside_v1_reasoning_parser_when_not_thinking(
        chat_template_kwargs: dict[str, bool]) -> None:
    """Output is all visible content when thinking is off.

    The template emits `</think>` into the prompt in that mode, so the model
    output carries no markers at all.
    """
    reasoning_parser = ReasoningParserFactory.create_reasoning_parser(
        "poolside_v1", chat_template_kwargs)
    result = reasoning_parser.parse("visible")
    assert result.content == "visible"
    assert result.reasoning_content == ""


@pytest.mark.parametrize(("delta_texts", "content", "reasoning_context"), [
    (["a", f"l{R1_END}r", "b"], ["", "r", "b"], ["a", "l", ""]),
    (["a</th", "ink>b"], ["", "b"], ["a", ""]),
    (["", f"a{R1_END}b"], ["", "b"], ["", "a"]),
    ([f"a{R1_END}", "b"], ["", "b"], ["a", ""]),
    (["a</th", "ank></th", "ink>b"], ["", "", "b"], ["a", "</thank>", ""]),
])
def test_poolside_v1_reasoning_parser_stream(
        delta_texts: list[str], content: list[str],
        reasoning_context: list[str]) -> None:
    reasoning_parser = ReasoningParserFactory.create_reasoning_parser(
        "poolside_v1", {"enable_thinking": True})
    for i, delta_text in enumerate(delta_texts):
        result = reasoning_parser.parse_delta(delta_text)
        assert result.content == content[i]
        assert result.reasoning_content == reasoning_context[i]


def test_laguna_alias_of_poolside_v1() -> None:
    """`laguna` is kept as an alias so existing deployments keep working."""
    laguna = ReasoningParserFactory.create_reasoning_parser("laguna")
    poolside = ReasoningParserFactory.create_reasoning_parser("poolside_v1")
    assert type(laguna) is type(poolside)
    assert MODEL_TYPE_TO_REASONING_PARSER["laguna"] == "poolside_v1"


@pytest.mark.parametrize("parser", ["poolside_v1", "laguna"])
@pytest.mark.parametrize(("tail", "expected"), [(R1_START, True),
                                                (R1_END, False)])
def test_alias_resolves_identically(parser: str, tail: str,
                                    expected: bool) -> None:
    assert ReasoningParserFactory.resolve_prefilled_thinking(
        parser, f"<assistant>{tail}") is expected


@pytest.mark.parametrize(("prompt", "expected"), [
    (f"<user>hi</user>\n<assistant>{R1_START}", True),
    (f"<user>hi</user>\n<assistant>{R1_END}", False),
    (f"<user>hi</user>\n<assistant>\n{R1_END}", False),
    (f"<user>hi</user>\n<assistant>{R1_START}\n", True),
    ("<user>hi</user>\n<assistant>", None),
    (f"a prompt quoting {R1_END} mid-turn\n<assistant>", None),
])
def test_resolve_prefilled_thinking(prompt: str, expected: bool | None) -> None:
    assert ReasoningParserFactory.resolve_prefilled_thinking(
        "poolside_v1", prompt) is expected


def test_resolve_prefilled_thinking_unknown_parser() -> None:
    assert ReasoningParserFactory.resolve_prefilled_thinking(
        "not-a-parser", R1_START) is None


@pytest.mark.parametrize(("tail", "expected"), [(R1_START, True),
                                                (R1_END, False)])
def test_resolve_prefilled_thinking_opted_in(tail: str, expected: bool) -> None:
    assert ReasoningParserFactory.resolve_prefilled_thinking(
        "poolside_v1", f"<assistant>{tail}") is expected


@pytest.mark.parametrize("parser", ["poolside_v1", "laguna"])
@pytest.mark.parametrize(("text", "content", "reasoning_context"), [
    ("a<think>b</think>c", "c", "b"),
    ("<think>a</think>b", "b", "a"),
    ("<think>a", "", "a"),
    ("a", "a", ""),
])
def test_unresolved_mode_keeps_previous_behaviour(
        parser: str, text: str, content: str, reasoning_context: str) -> None:
    """With no mode supplied, split on a `<think>` the model emitted itself.

    Nothing resolves the mode for the offline LLM API, the disaggregated
    generation server, or `add_generation_prompt=false`. Falling back to
    `IdentityReasoningParser` there would drop the split these models still
    need, since they emit the tags in multi-turn and tool-calling flows.
    """
    reasoning_parser = ReasoningParserFactory.create_reasoning_parser(parser)

    result = reasoning_parser.parse(text)

    assert result.content == content
    assert result.reasoning_content == reasoning_context


@pytest.mark.parametrize("kwargs", [{
    "enable_thinking": False
}, {
    "thinking": False
}])
def test_explicit_no_thinking_still_uses_identity(
        kwargs: dict[str, bool]) -> None:
    """An explicit off is a resolved mode, so the template closed reasoning."""
    reasoning_parser = ReasoningParserFactory.create_reasoning_parser(
        "poolside_v1", kwargs)

    result = reasoning_parser.parse("a<think>b</think>c")

    assert result.content == "a<think>b</think>c"
    assert result.reasoning_content == ""


def test_clearing_only_enable_thinking_leaves_reasoning_on() -> None:
    """Why the server writes both keys: the parser ORs them.

    If it cleared only `enable_thinking`, a `thinking=True` the caller sent
    would still force reasoning mode and the answer would land in the wrong
    field. This pins the OR so that shortcut cannot be taken silently.
    """
    reasoning_parser = ReasoningParserFactory.create_reasoning_parser(
        "poolside_v1", {
            "thinking": True,
            "enable_thinking": False
        })

    result = reasoning_parser.parse(f"hidden{R1_END}visible")

    assert result.reasoning_content == "hidden"
    assert result.content == "visible"


@pytest.mark.parametrize("sent", [{
    "thinking": True
}, {
    "enable_thinking": True
}, {
    "thinking": True,
    "enable_thinking": True
}])
def test_resolved_mode_overrides_whatever_the_caller_sent(
        sent: dict[str, bool]) -> None:
    """Both keys written, as the server does, so the resolved mode wins."""
    resolved = {**sent, "thinking": False, "enable_thinking": False}
    reasoning_parser = ReasoningParserFactory.create_reasoning_parser(
        "poolside_v1", resolved)

    result = reasoning_parser.parse("visible")

    assert result.content == "visible"
    assert result.reasoning_content == ""


@pytest.mark.parametrize("parser", [
    "deepseek-r1", "deepseek_v4", "qwen3", "qwen3_5", "minimax_m2",
    "minimax_m3", "nemotron-v3", "nano-v3", "gemma4", "kimi_k2", "kimi_k25"
])
def test_resolve_prefilled_thinking_requires_opt_in(parser: str) -> None:
    """Parsers that have not opted in must never be resolved from the prompt.

    `deepseek_v4` shares the base class and `nemotron-v3` / `nano-v3` also read
    `enable_thinking`, so without the flag they would silently pick up a mode
    the server inferred.
    """
    # Otherwise a typo or a dropped registration passes vacuously, since an
    # unknown name also resolves to None.
    assert parser in ReasoningParserFactory.keys()
    for tail in (R1_START, R1_END, ""):
        assert ReasoningParserFactory.resolve_prefilled_thinking(
            parser, f"<assistant>{tail}") is None


@pytest.mark.parametrize(
    ("prompt_tail", "model_output", "content", "reasoning_content"), [
        (R1_START, f"hidden{R1_END}visible", "visible", "hidden"),
        (R1_END, "visible", "visible", ""),
    ])
def test_poolside_v1_mode_resolved_from_prompt(prompt_tail: str,
                                               model_output: str, content: str,
                                               reasoning_content: str) -> None:
    """Mirror the server path: resolve from the prompt, then parse.

    A request that sends no chat template kwargs must still land in the mode
    the template actually rendered.
    """
    prompt = f"<user>hi</user>\n<assistant>{prompt_tail}"
    thinking = ReasoningParserFactory.resolve_prefilled_thinking(
        "poolside_v1", prompt)
    reasoning_parser = ReasoningParserFactory.create_reasoning_parser(
        "poolside_v1", {"enable_thinking": thinking})
    result = reasoning_parser.parse(model_output)
    assert result.content == content
    assert result.reasoning_content == reasoning_content


TOOL_CALL = "<tool_call>"
TOOL_CALL_END = "</tool_call>"


@pytest.mark.parametrize(
    ("text", "content", "reasoning_context", "chat_template_kwargs"),
    [
        ("a b", "", "a b", None),
        (f"{R1_END} a b", " a b", "", None),
        (f"a {R1_END} b", " b", "a ", None),
        (f"a b {R1_END}", "", "a b ", None),
        (f"{R1_START} a {R1_END} b", " b", f"{R1_START} a ", None),
        # All without reasoning_context.
        ("a b", "a b", "", {
            "enable_thinking": False
        }),
        (f"{R1_END} a b", f"{R1_END} a b", "", {
            "enable_thinking": False
        }),
        (f"a {R1_END} b", f"a {R1_END} b", "", {
            "enable_thinking": False
        }),
        (f"a b {R1_END}", f"a b {R1_END}", "", {
            "enable_thinking": False
        }),
        # force_nonempty_content swaps reasoning into content when content is
        # empty (reasoning_at_start stays True, so parsing is unchanged).
        ("a b", "a b", "", {
            "force_nonempty_content": True
        }),
        (f"a {R1_END} b", " b", "a ", {
            "force_nonempty_content": True
        }),
        (f"a b {R1_END}", "a b ", "", {
            "force_nonempty_content": True
        }),
        # NVBug 6060281: whitespace-only content after </redacted_thinking> must
        # still trigger the reasoning-to-content swap when force_nonempty_content.
        (f"a {R1_END}\n", "a ", "", {
            "force_nonempty_content": True
        }),
        (f"a {R1_END} \t ", "a ", "", {
            "force_nonempty_content": True
        }),
        # NVBug 6082303: <tool_call> as implicit end-of-reasoning.
        # No </think> before <tool_call> — tool call must appear in content.
        (f"I need weather{TOOL_CALL}get_weather{TOOL_CALL_END}",
         f"{TOOL_CALL}get_weather{TOOL_CALL_END}", "I need weather", None),
        # </think> before <tool_call> — standard end wins.
        (f"reasoning{R1_END}{TOOL_CALL}get_weather{TOOL_CALL_END}",
         f"{TOOL_CALL}get_weather{TOOL_CALL_END}", "reasoning", None),
        # <tool_call> with enable_thinking=False — all goes to content.
        (f"content{TOOL_CALL}get_weather{TOOL_CALL_END}",
         f"content{TOOL_CALL}get_weather{TOOL_CALL_END}", "", {
             "enable_thinking": False
         }),
        # <tool_call> with force_nonempty_content — content is non-empty, no swap.
        (f"reasoning{TOOL_CALL}get_weather{TOOL_CALL_END}",
         f"{TOOL_CALL}get_weather{TOOL_CALL_END}", "reasoning", {
             "force_nonempty_content": True
         }),
        # <think> then <tool_call> without </think> with enable_thinking=False:
        # reasoning_at_start is False so parse() looks for <think> tag,
        # strips it, then tool_call acts as implicit end-of-reasoning.
        (f"{R1_START}reasoning{TOOL_CALL}get_weather{TOOL_CALL_END}",
         f"{TOOL_CALL}get_weather{TOOL_CALL_END}", "reasoning", {
             "enable_thinking": False
         }),
    ])
def test_nano_v3_reasoning_parser(text: str, content: str,
                                  reasoning_context: str,
                                  chat_template_kwargs: dict):
    reasoning_parser = ReasoningParserFactory.create_reasoning_parser(
        "nano-v3", chat_template_kwargs)
    result = reasoning_parser.parse(text)
    print(f"text: {text}, result: {result}")
    assert result.content == content
    assert result.reasoning_content == reasoning_context


@pytest.mark.parametrize(
    ("delta_texts", "content", "reasoning_context", "chat_template_kwargs"),
    [
        (["a", "b"], ["", ""], ["a", "b"], None),
        ([R1_END, "a", "b"], ["", "a", "b"], ["", "", ""], None),
        (["a", R1_END, "b"], ["", "", "b"], ["a", "", ""], None),
        (["a", "b", R1_END], ["", "", ""], ["a", "b", ""], None),
        (["a", f"l{R1_END}", "b"], ["", "", "b"], ["a", "l", ""], None),
        (["a", f"l{R1_END}r", "b"], ["", "r", "b"], ["a", "l", ""], None),
        (["a", f"{R1_END}r", "b"], ["", "r", "b"], ["a", "", ""], None),
        # All without reasoning_context.
        (["a", "b"], ["a", "b"], ["", ""], {
            "enable_thinking": False
        }),
        ([R1_END, "a", "b"], ["", f"{R1_END}a", "b"], ["", "", ""], {
            "enable_thinking": False
        }),
        (["a", R1_END, "b"], ["a", "", f"{R1_END}b"], ["", "", ""], {
            "enable_thinking": False
        }),
        (["a", "b", R1_END], ["a", "b", ""], ["", "", ""], {
            "enable_thinking": False
        }),
        (["a", f"l{R1_END}", "b"], ["a", f"l{R1_END}", "b"], ["", "", ""], {
            "enable_thinking": False
        }),
        (["a", f"l{R1_END}r", "b"], ["a", f"l{R1_END}r", "b"], ["", "", ""], {
            "enable_thinking": False
        }),
        (["a", f"{R1_END}r", "b"], ["a", f"{R1_END}r", "b"], ["", "", ""], {
            "enable_thinking": False
        }),
        # NVBug 6082303: <tool_call> as implicit end-of-reasoning in streaming.
        # Single-token <tool_call> after reasoning deltas.
        (
            ["I need ", "weather", TOOL_CALL, "get_weather"],
            ["", "", f"{TOOL_CALL}", "get_weather"],
            ["I need ", "weather", "", ""],
            None,
        ),
        # <tool_call> combined with preceding reasoning text in one delta.
        (
            ["reasoning" + TOOL_CALL + "tool_data"],
            [TOOL_CALL + "tool_data"],
            ["reasoning"],
            None,
        ),
        # </think> before <tool_call> — standard handling.
        (
            ["reasoning", R1_END, TOOL_CALL + "data"],
            ["", "", TOOL_CALL + "data"],
            ["reasoning", "", ""],
            None,
        ),
        # Token-level chunks mimicking the bug scenario from NVBug 6082303.
        (
            [
                "I", " need to check", " the weather in", " New York City",
                TOOL_CALL, "<function=get_weather>", "<parameter=location>NYC",
                "</parameter>", "</function>", TOOL_CALL_END
            ],
            [
                "", "", "", "", TOOL_CALL, "<function=get_weather>",
                "<parameter=location>NYC", "</parameter>", "</function>",
                TOOL_CALL_END
            ],
            [
                "I", " need to check", " the weather in", " New York City", "",
                "", "", "", "", ""
            ],
            None,
        ),
        # Parent buffers trailing "<" (prefix of "</think>") from a text
        # token, then the full <tool_call> special token arrives atomically.
        (
            ["reasoning<", "<tool_call>data"],
            ["", "<tool_call>data"],
            ["reasoning", "<"],
            None,
        ),
        # force_nonempty_content + streaming tool call: tool_call ends
        # reasoning, accumulated reasoning is discarded (not swapped into
        # content), and content carries the tool markup.
        (
            ["I need ", "weather", TOOL_CALL, "get_weather"],
            ["", "", TOOL_CALL, "get_weather"],
            ["I need ", "weather", "", ""],
            {
                "force_nonempty_content": True
            },
        ),
        # enable_thinking=False + streaming tool call: everything is content
        # (no buffering since <tool_call> is not a prefix of <think>).
        (
            ["content", TOOL_CALL, "data"],
            ["content", TOOL_CALL, "data"],
            ["", "", ""],
            {
                "enable_thinking": False
            },
        ),
    ])
def test_nano_v3_reasoning_parser_stream(delta_texts: list, content: list,
                                         reasoning_context: list,
                                         chat_template_kwargs: dict):
    reasoning_parser = ReasoningParserFactory.create_reasoning_parser(
        "nano-v3", chat_template_kwargs)
    for i, delta_text in enumerate(delta_texts):
        result = reasoning_parser.parse_delta(delta_text)
        print(f"delta_text: {delta_text}, result: {result}")
        assert result.content == content[i]
        assert result.reasoning_content == reasoning_context[i]


@pytest.mark.parametrize(
    ("delta_texts", "finish_content", "finish_reasoning",
     "chat_template_kwargs"),
    [
        (["a", "b"], "", "", None),
        ([R1_END, "a", "b"], "", "", None),
        (["a", R1_END, "b"], "", "", None),
        (["a", "b"], "", "", {
            "enable_thinking": False
        }),
        ([f"{R1_START}a", "b"], "", "", {
            "enable_thinking": False
        }),
        (["a", "b"], "", "", {
            "force_nonempty_content": False
        }),
        (["a", "b"], "ab", "", {
            "force_nonempty_content": True
        }),
        ([R1_END, "a", "b"], "", "", {
            "force_nonempty_content": True
        }),
        # NVBug 6082303: <tool_call> ends reasoning,
        # finish should return empty (tag acted as
        # implicit closing).
        (["reasoning", "<tool_call>data"], "", "", None),
        (["reasoning", "<tool_call>data"], "", "", {
            "force_nonempty_content": True
        }),
    ])
def test_nano_v3_reasoning_parser_finish(delta_texts: list, finish_content: str,
                                         finish_reasoning: str,
                                         chat_template_kwargs: dict):
    reasoning_parser = ReasoningParserFactory.create_reasoning_parser(
        "nano-v3", chat_template_kwargs)
    for delta_text in delta_texts:
        reasoning_parser.parse_delta(delta_text)
    result = reasoning_parser.finish()
    assert result.content == finish_content
    assert result.reasoning_content == finish_reasoning


# ---------------------------------------------------------------------------
# Auto-detection tests for resolve_auto_reasoning_parser
# ---------------------------------------------------------------------------


def _write_config(model_dir: str, model_type: str):
    """Write a minimal config.json with the given model_type."""
    with open(os.path.join(model_dir, "config.json"), "w") as f:
        json.dump({"model_type": model_type}, f)


def _write_tokenizer_config(model_dir: str, chat_template: str):
    """Write a minimal tokenizer_config.json with the given chat_template."""
    with open(os.path.join(model_dir, "tokenizer_config.json"), "w") as f:
        json.dump({"chat_template": chat_template}, f)


# Hybrid Qwen3: chat template contains "enable_thinking" → use "qwen3" parser
_HYBRID_TEMPLATE = (
    "{%- if enable_thinking is not defined %}{% set enable_thinking = true %}"
    "{% endif %}{%- if add_generation_prompt %}{%- if enable_thinking %}"
    "{{- '<|im_start|>assistant\\n<think>\\n' }}{%- else %}"
    "{{- '<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n' }}"
    "{%- endif %}{%- endif %}")

# Forced-thinking Qwen3: no "enable_thinking" but has "<think>" → "deepseek-r1"
_FORCED_THINKING_TEMPLATE = ("{%- if add_generation_prompt %}"
                             "{{- '<|im_start|>assistant\\n<think>\\n' }}"
                             "{%- endif %}")

# Forced-non-thinking Qwen3: no "enable_thinking" and no "<think>" → None
_FORCED_NON_THINKING_TEMPLATE = ("{%- if add_generation_prompt %}"
                                 "{{- '<|im_start|>assistant\\n' }}"
                                 "{%- endif %}")


def test_auto_detect_qwen3_hybrid(tmp_path):
    """Hybrid Qwen3 model with enable_thinking toggle → 'qwen3' parser."""
    model_dir = str(tmp_path / "Qwen3-235B-A22B")
    os.makedirs(model_dir)
    _write_config(model_dir, "qwen3_moe")
    _write_tokenizer_config(model_dir, _HYBRID_TEMPLATE)

    result = resolve_auto_reasoning_parser(model_dir)
    assert result == "qwen3"


def test_auto_detect_qwen3_forced_thinking(tmp_path):
    """Forced-thinking Qwen3 model → 'deepseek-r1' parser."""
    model_dir = str(tmp_path / "Qwen3-235B-A22B-Thinking-2507")
    os.makedirs(model_dir)
    _write_config(model_dir, "qwen3_moe")
    _write_tokenizer_config(model_dir, _FORCED_THINKING_TEMPLATE)

    result = resolve_auto_reasoning_parser(model_dir)
    assert result == "deepseek-r1"


def test_auto_detect_qwen3_forced_non_thinking(tmp_path):
    """Forced-non-thinking Qwen3 model → None (no parser needed)."""
    model_dir = str(tmp_path / "Qwen3-235B-A22B-Instruct-2507")
    os.makedirs(model_dir)
    _write_config(model_dir, "qwen3_moe")
    _write_tokenizer_config(model_dir, _FORCED_NON_THINKING_TEMPLATE)

    result = resolve_auto_reasoning_parser(model_dir)
    assert result is None


def test_auto_detect_qwen3_no_tokenizer_config(tmp_path):
    """Qwen3 model without tokenizer_config.json → falls back to 'qwen3'."""
    model_dir = str(tmp_path / "Qwen3-SomeModel")
    os.makedirs(model_dir)
    _write_config(model_dir, "qwen3")

    result = resolve_auto_reasoning_parser(model_dir)
    assert result == "qwen3"


def test_auto_detect_deepseek_r1(tmp_path):
    """DeepSeek R1 model → 'deepseek-r1' parser."""
    model_dir = str(tmp_path / "DeepSeek-R1")
    os.makedirs(model_dir)
    _write_config(model_dir, "deepseek_v3")

    result = resolve_auto_reasoning_parser(model_dir)
    assert result == "deepseek-r1"


def test_auto_detect_deepseek_non_r1(tmp_path):
    """DeepSeek non-R1 model → None."""
    model_dir = str(tmp_path / "DeepSeek-V3")
    os.makedirs(model_dir)
    _write_config(model_dir, "deepseek_v3")

    result = resolve_auto_reasoning_parser(model_dir)
    assert result is None


def test_auto_detect_unknown_model(tmp_path):
    """Unknown model type → None."""
    model_dir = str(tmp_path / "SomeUnknownModel")
    os.makedirs(model_dir)
    _write_config(model_dir, "unknown_type")

    result = resolve_auto_reasoning_parser(model_dir)
    assert result is None


def test_auto_detect_no_config(tmp_path):
    """No config.json → None."""
    model_dir = str(tmp_path / "EmptyDir")
    os.makedirs(model_dir)

    result = resolve_auto_reasoning_parser(model_dir)
    assert result is None


def test_auto_detect_gemma4(tmp_path):
    """Gemma 4 model → 'gemma4' parser."""
    model_dir = str(tmp_path / "gemma-4-26B-A4B-it")
    os.makedirs(model_dir)
    _write_config(model_dir, "gemma4")

    result = resolve_auto_reasoning_parser(model_dir)
    assert result == "gemma4"


def test_auto_detect_laguna(tmp_path):
    """Laguna model → 'poolside_v1' parser."""
    model_dir = str(tmp_path / "Laguna")
    os.makedirs(model_dir)
    _write_config(model_dir, "laguna")

    result = resolve_auto_reasoning_parser(model_dir)
    assert result == "poolside_v1"


@pytest.mark.parametrize("model_type", ["nemotron_h", "nemotron_h_puzzle"])
def test_auto_detect_nemotron_h(tmp_path, model_type):
    """Nemotron-H models → 'nemotron-v3' parser (preferred over 'nano-v3')."""
    model_dir = str(tmp_path / model_type)
    os.makedirs(model_dir)
    _write_config(model_dir, model_type)

    result = resolve_auto_reasoning_parser(model_dir)
    assert result == "nemotron-v3"


def test_nemotron_v3_alias_same_parser():
    """'nemotron-v3' and the legacy 'nano-v3' resolve to the same parser."""
    nemotron = ReasoningParserFactory.create_reasoning_parser("nemotron-v3")
    nano = ReasoningParserFactory.create_reasoning_parser("nano-v3")
    assert isinstance(nemotron, NemotronV3ReasoningParser)
    assert isinstance(nano, NemotronV3ReasoningParser)


# ---------------------------------------------------------------------------
# Gemma 4 reasoning parser tests
# ---------------------------------------------------------------------------

G4_OPEN, G4_CLOSE = "<|channel>", "<channel|>"


@pytest.mark.parametrize(
    ("text", "content", "reasoning_content"),
    [
        # No reasoning block: everything is content.
        ("hello world", "hello world", ""),
        # Channel block wrapping reasoning, followed by content.
        (f"{G4_OPEN}thought\nreasoning{G4_CLOSE}answer", "answer",
         "thought\nreasoning"),
        # Content before and after the reasoning block.
        (f"pre{G4_OPEN}r{G4_CLOSE}post", "prepost", "r"),
        # Unterminated channel: remainder treated as reasoning.
        (f"{G4_OPEN}abc", "", "abc"),
        # Multiple interleaved channel blocks.
        (f"a{G4_OPEN}r1{G4_CLOSE}b{G4_OPEN}r2{G4_CLOSE}c", "abc", "r1r2"),
        # Empty reasoning block (e.g. prefilled when enable_thinking=False).
        (f"{G4_OPEN}thought\n{G4_CLOSE}answer", "answer", "thought\n"),
    ],
)
def test_gemma4_reasoning_parser(text: str, content: str,
                                 reasoning_content: str):
    parser = ReasoningParserFactory.create_reasoning_parser("gemma4")
    result = parser.parse(text)
    assert result.content == content
    assert result.reasoning_content == reasoning_content


@pytest.mark.parametrize(
    ("delta_texts", "content", "reasoning_content"),
    [
        # No reasoning: plain content streams through.
        (["a", "b"], ["a", "b"], ["", ""]),
        # Open and close in a single delta.
        ([f"{G4_OPEN}r{G4_CLOSE}c"], ["c"], ["r"]),
        # Delimiters split across deltas.
        ([G4_OPEN, "r", G4_CLOSE, "c"], ["", "", "", "c"], ["", "r", "", ""]),
        # Partial open tag held back until complete.
        (["pre<|cha", "nnel>r<chan", "nel|>post"], ["pre", "", "post"
                                                    ], ["", "r", ""]),
        # Two reasoning blocks interleaved with content.
        (
            [f"{G4_OPEN}r1{G4_CLOSE}c1", f"{G4_OPEN}r2{G4_CLOSE}c2"],
            ["c1", "c2"],
            ["r1", "r2"],
        ),
        # Partial close tag at end of delta buffered.
        ([f"{G4_OPEN}reason<chan", "nel|>tail"], ["", "tail"], ["reason", ""]),
    ],
)
def test_gemma4_reasoning_parser_stream(delta_texts: list, content: list,
                                        reasoning_content: list):
    parser = ReasoningParserFactory.create_reasoning_parser("gemma4")
    for i, delta in enumerate(delta_texts):
        result = parser.parse_delta(delta)
        assert result.content == content[i], (
            f"Step {i}: delta={delta!r} expected content={content[i]!r} "
            f"got {result.content!r}")
        assert result.reasoning_content == reasoning_content[i], (
            f"Step {i}: delta={delta!r} expected reasoning="
            f"{reasoning_content[i]!r} got {result.reasoning_content!r}")


def test_gemma4_reasoning_parser_finish_flushes_buffer():
    """finish() should flush any buffered trailing text."""
    parser = ReasoningParserFactory.create_reasoning_parser("gemma4")
    # Send a partial open tag; parser holds it back.
    parser.parse_delta("some text<|cha")
    # Stream ended mid-tag: the held-back suffix flushes as content.
    result = parser.finish()
    assert result.content == "<|cha"
    assert result.reasoning_content == ""


def test_gemma4_reasoning_parser_finish_unterminated_reasoning():
    """Verify finish() flushes a held-back partial close tag as reasoning.

    When the stream ends mid-channel with a buffered partial close tag, the
    remainder should surface as reasoning content.
    """
    parser = ReasoningParserFactory.create_reasoning_parser("gemma4")
    # Enter reasoning; stream ends with a partial close tag (held back).
    stream = parser.parse_delta(f"{G4_OPEN}reasoning_start<chan")
    assert stream.reasoning_content == "reasoning_start"
    assert stream.content == ""
    # finish() should release the buffered "<chan" as reasoning since we are
    # still inside the channel block.
    result = parser.finish()
    assert result.content == ""
    assert result.reasoning_content == "<chan"


# ---------------------------------------------------------------------------
# Kimi K3 reasoning parser tests
#
# Fixture strings are transcribed from the checkpoint's `encoding_k3.py`
# rendering (the authoritative XTML chat template): tags render as
# `<|open|>tag key="value"<|sep|>` / `<|close|>tag<|sep|>` with no
# whitespace between segments, and the generation prompt ends inside
# `<|open|>think<|sep|>` (or `<|open|>response<|sep|>` when thinking=False),
# so completions start mid-channel and end with
# `<|close|>message<|sep|><|end_of_msg|>`.
# ---------------------------------------------------------------------------

K3_OPEN, K3_CLOSE, K3_SEP, K3_EOM = ("<|open|>", "<|close|>", "<|sep|>",
                                     "<|end_of_msg|>")

# One get_weather call, exactly as encoding_k3._render_assistant_segments
# renders it (attributes space-prefixed, index 1-based, string args raw).
K3_TOOLS_SECTION = (f'{K3_OPEN}tools{K3_SEP}'
                    f'{K3_OPEN}call tool="get_weather" index="1"{K3_SEP}'
                    f'{K3_OPEN}argument key="location" type="string"{K3_SEP}'
                    f'NYC'
                    f'{K3_CLOSE}argument{K3_SEP}'
                    f'{K3_CLOSE}call{K3_SEP}'
                    f'{K3_CLOSE}tools{K3_SEP}')

K3_MSG_END = f"{K3_CLOSE}message{K3_SEP}{K3_EOM}"


def _k3_completion(reasoning: str,
                   content: str,
                   tools_section: str = "",
                   terminated: bool = True) -> str:
    """A thinking-mode completion (prompt already opened the think channel)."""
    text = (f"{reasoning}{K3_CLOSE}think{K3_SEP}"
            f"{K3_OPEN}response{K3_SEP}{content}")
    if terminated:
        text += f"{K3_CLOSE}response{K3_SEP}{tools_section}{K3_MSG_END}"
    return text


@pytest.mark.parametrize(
    ("text", "content", "reasoning_content"),
    [
        # Fully terminated think + response message.
        (_k3_completion("step by step", "The answer is 4."), "The answer is 4.",
         "step by step"),
        # Tool-calling message: the tools section passes through into content
        # verbatim so the kimi_k3 tool parser can consume it downstream.
        (_k3_completion("pick a tool", "Checking.", K3_TOOLS_SECTION),
         "Checking." + K3_TOOLS_SECTION, "pick a tool"),
        # Length-capped mid-think: everything is reasoning.
        ("unterminated reasoning", "", "unterminated reasoning"),
        # Length-capped mid-response: finish() flushes the response tail.
        (_k3_completion("r", "partial resp",
                        terminated=False), "partial resp", "r"),
        # Empty think body.
        (_k3_completion("", "only content"), "only content", ""),
    ])
def test_kimi_k3_reasoning_parser(text: str, content: str,
                                  reasoning_content: str):
    parser = ReasoningParserFactory.create_reasoning_parser("kimi_k3")
    result = parser.parse(text)
    assert result.content == content
    assert result.reasoning_content == reasoning_content


@pytest.mark.parametrize(
    ("text", "content", "reasoning_content"),
    [
        # thinking=False: prompt opened the response channel directly.
        (f"plain answer{K3_CLOSE}response{K3_SEP}{K3_MSG_END}", "plain answer",
         ""),
        (f"answer{K3_CLOSE}response{K3_SEP}{K3_TOOLS_SECTION}{K3_MSG_END}",
         "answer" + K3_TOOLS_SECTION, ""),
    ])
def test_kimi_k3_reasoning_parser_non_thinking(text: str, content: str,
                                               reasoning_content: str):
    parser = ReasoningParserFactory.create_reasoning_parser(
        "kimi_k3", {"thinking": False})
    result = parser.parse(text)
    assert result.content == content
    assert result.reasoning_content == reasoning_content


@pytest.mark.parametrize(
    ("delta_texts", "content", "reasoning_content"),
    [
        # Plain reasoning streams straight through.
        (["a", "b"], ["", ""], ["a", "b"]),
        # Channel switch split across deltas mid-marker.
        (
            [
                "rea",
                f"son{K3_CLOSE}thi",  # codespell:ignore thi
                f"nk{K3_SEP}{K3_OPEN}response{K3_SEP}c",
                "d"
            ],
            ["", "", "c", "d"],
            ["rea", "son", "", ""]),
        # A partial marker at the end of a delta is held back, then released
        # as reasoning once it turns out not to be a marker.
        (["a<|clo", "x"], ["", ""], ["a", "<|clox"]),
        # Structural close/open pair arriving as one delta.
        ([f"r{K3_CLOSE}think{K3_SEP}{K3_OPEN}response{K3_SEP}c"], ["c"], ["r"]),
        # tools_pass: `<|close|>tools<|sep|>` has an internal `<`, so the
        # suffix hold must consider mid-marker splits like `...<|close|>to`.
        ([
            f"r{K3_CLOSE}think{K3_SEP}{K3_OPEN}response{K3_SEP}",
            f"{K3_OPEN}tools{K3_SEP}CALL{K3_CLOSE}to",
            f"ols{K3_SEP}{K3_MSG_END}",
        ], [
            "",
            f"{K3_OPEN}tools{K3_SEP}CALL",
            f"{K3_CLOSE}tools{K3_SEP}",
        ], ["r", "", ""]),
        # Message terminator split across deltas produces no output.
        ([
            f"r{K3_CLOSE}think{K3_SEP}{K3_OPEN}response{K3_SEP}c",
            "<|close|>mes", f"sage{K3_SEP}{K3_EOM}"
        ], ["c", "", ""], ["r", "", ""]),
    ])
def test_kimi_k3_reasoning_parser_stream(delta_texts: list, content: list,
                                         reasoning_content: list):
    parser = ReasoningParserFactory.create_reasoning_parser("kimi_k3")
    for i, delta_text in enumerate(delta_texts):
        result = parser.parse_delta(delta_text)
        assert result.content == content[i], \
            f"Step {i}: delta={delta_text!r}, expected content={content[i]!r}, got {result.content!r}"
        assert result.reasoning_content == reasoning_content[i], \
            f"Step {i}: delta={delta_text!r}, expected reasoning={reasoning_content[i]!r}, got {result.reasoning_content!r}"


@pytest.mark.parametrize("chunk_size", [1, 2, 3, 7])
@pytest.mark.parametrize("thinking", [True, False])
def test_kimi_k3_reasoning_parser_stream_matches_parse(chunk_size: int,
                                                       thinking: bool):
    """Streaming in arbitrary chunkings must reproduce the one-shot parse.

    This sweeps every marker-split position, which is the riskiest logic in
    the parser (`_partial_suffix_len` suffix holds).
    """
    if thinking:
        text = _k3_completion("Let me think.", "Answer: 4.", K3_TOOLS_SECTION)
    else:
        text = (f"Answer: 4.{K3_CLOSE}response{K3_SEP}{K3_TOOLS_SECTION}"
                f"{K3_MSG_END}")
    kwargs = None if thinking else {"thinking": False}

    oneshot = ReasoningParserFactory.create_reasoning_parser("kimi_k3",
                                                             kwargs).parse(text)

    streamer = ReasoningParserFactory.create_reasoning_parser("kimi_k3", kwargs)
    content, reasoning = [], []
    for start in range(0, len(text), chunk_size):
        result = streamer.parse_delta(text[start:start + chunk_size])
        content.append(result.content)
        reasoning.append(result.reasoning_content)
    tail = streamer.finish()
    content.append(tail.content)
    reasoning.append(tail.reasoning_content)

    assert "".join(content) == oneshot.content
    assert "".join(reasoning) == oneshot.reasoning_content


def test_kimi_k3_needs_raw_special_tokens():
    """The K3 delimiters are special tokens.

    The serving layer keys off this flag to disable skip_special_tokens for
    the request.
    """
    assert ReasoningParserFactory.needs_raw_special_tokens("kimi_k3") is True
    assert ReasoningParserFactory.needs_raw_special_tokens("KIMI_K3") is True
    assert ReasoningParserFactory.needs_raw_special_tokens(
        "deepseek-r1") is False
    assert ReasoningParserFactory.needs_raw_special_tokens(
        "no_such_parser") is False


def test_auto_detect_kimi_k3(tmp_path):
    """Kimi K3 model → 'kimi_k3' parser."""
    model_dir = str(tmp_path / "Kimi-K3")
    os.makedirs(model_dir)
    _write_config(model_dir, "kimi_k3")

    result = resolve_auto_reasoning_parser(model_dir)
    assert result == "kimi_k3"


class TestGlmReasoningParser:
    """GLM closes its reasoning block more than once per turn.

    Raw output from an agent round, verbatim:

        ...recovered.</think>Let me retry exec after waiting.</think><tool_call>

    DeepSeek-R1's rule -- everything after the first close is content --
    publishes the later tags as visible text. vLLM's GLM parser consumes a
    close seen outside a reasoning block instead; this parser does the same,
    under its own key so no other model's output is rewritten.
    """

    CAMPAIGN_OUTPUT = ("...environment recovered.</think>"
                       "Let me retry exec after waiting.</think>"
                       "<tool_call>collaboration.followup_task")

    @staticmethod
    def _parser():
        return ReasoningParserFactory.create_reasoning_parser("glm")

    @staticmethod
    def _stream(parser, text, chunk):
        content, reasoning = [], []
        for i in range(0, len(text), chunk):
            result = parser.parse_delta(text[i:i + chunk])
            if result.content:
                content.append(result.content)
            if result.reasoning_content:
                reasoning.append(result.reasoning_content)
        result = parser.finish()
        if result.content:
            content.append(result.content)
        if result.reasoning_content:
            reasoning.append(result.reasoning_content)
        return "".join(content), "".join(reasoning)

    def test_a_second_closing_tag_is_not_visible_text(self):
        result = self._parser().parse(self.CAMPAIGN_OUTPUT)

        assert result.reasoning_content == "...environment recovered."
        assert "</think>" not in result.content
        assert result.content == ("Let me retry exec after waiting."
                                  "<tool_call>collaboration.followup_task")

    @pytest.mark.parametrize("chunk", [1, 3, 8, 1000])
    def test_a_closing_tag_split_across_deltas_is_not_leaked(self, chunk):
        """The tag rarely arrives whole.

        Stripping only complete matches left `</t`, `hin`, `k>` going out one
        piece at a time; chunk sizes 3 and 8 both hit that, so the partial
        fragment has to be withheld rather than emitted.
        """
        content, reasoning = self._stream(self._parser(), self.CAMPAIGN_OUTPUT,
                                          chunk)

        assert "</think>" not in content
        assert reasoning == "...environment recovered."

    @pytest.mark.parametrize("chunk", [1, 3, 8, 1000])
    def test_streaming_agrees_with_parsing_the_whole_text(self, chunk):
        whole = self._parser().parse(self.CAMPAIGN_OUTPUT)
        content, reasoning = self._stream(self._parser(), self.CAMPAIGN_OUTPUT,
                                          chunk)

        assert content == whole.content
        assert reasoning == whole.reasoning_content

    def test_an_ordinary_single_block_is_split_as_before(self):
        """The common shape must be untouched by the extra handling."""
        result = self._parser().parse("Let me think.</think>The answer is 42.")

        assert result.reasoning_content == "Let me think."
        assert result.content == "The answer is 42."

    def test_deepseek_r1_still_passes_the_stray_tag_through(self):
        """The shared parser keeps its behaviour.

        A model that never emits a stray tag should not have its output
        rewritten on the chance that it might, which is the whole reason GLM
        got its own key.
        """
        result = ReasoningParserFactory.create_reasoning_parser(
            "deepseek-r1").parse(self.CAMPAIGN_OUTPUT)

        assert "</think>" in result.content


# The recorded failure, reduced to the shape the parser can react to.
#
# Live on 2026-09-18, instance 477479,
# `disagg_request_id=10897237976417924`: 26,055 characters delivered in 169
# frames, zero `<think>` and zero `</think>` anywhere, and one well-formed
# tool call at offset 25,392 -- arriving whole, inside frame 165. The prose
# between the markers cannot change any decision the parser makes, so only
# the load-bearing structure is kept: the 169 frame lengths, the offset and
# text of every `<` in the generation, and the call itself.
# `test_the_fixture_matches_the_recorded_generation` asserts each of those
# properties back, so a fixture that drifts from the capture fails loudly
# rather than passing quietly.
_RECORDED_LENGTH = 26055
_RECORDED_TOOL_CALL_OFFSET = 25392
_RECORDED_FRAME_LENS = (1, 153, 116, 185, 182, 172, 154, 134, 141, 149, 117,
                        160, 170, 263, 130, 126, 123, 123, 144, 170, 158, 114,
                        116, 141, 139, 153, 138, 119, 123, 188, 112, 134, 131,
                        132, 83, 119, 134, 171, 169, 160, 129, 167, 141, 163,
                        160, 129, 161, 174, 151, 190, 155, 104, 195, 159, 119,
                        106, 188, 181, 143, 158, 123, 175, 413, 179, 129, 127,
                        138, 121, 148, 165, 131, 153, 163, 174, 192, 147, 137,
                        143, 131, 139, 158, 120, 166, 183, 160, 157, 179, 175,
                        164, 137, 139, 130, 173, 140, 140, 164, 141, 156, 225,
                        121, 145, 173, 164, 156, 127, 150, 121, 125, 176, 158,
                        155, 166, 194, 177, 171, 141, 169, 148, 191, 124, 150,
                        153, 133, 167, 182, 146, 159, 159, 125, 146, 135, 156,
                        150, 161, 157, 154, 151, 153, 160, 137, 148, 135, 134,
                        123, 142, 195, 167, 171, 151, 156, 167, 191, 166, 194,
                        161, 142, 166, 158, 151, 118, 163, 200, 163, 186, 219,
                        224, 180, 150, 193)
# Every `<` the generation contains before the tool call. They are what the
# partial-marker holdbacks have to survive, so they keep their real offsets.
_RECORDED_ANGLE_BRACKETS = (
    (1843, "<float>()"),
    (1871, "<float>()"),
    (1914, "<float>()"),
    (1958, "<float>()"),
    (9407, "< total)"),
    (21388, "< 0.01 ULP"),
)
_RECORDED_TOOL_CALL = (
    "<tool_call>collaboration.send_message<arg_key>message</arg_key>"
    "<arg_value>"
    "Key finding: The Nemotron-3 chain is extremely sensitive to "
    "accumulation-order differences between cuBLAS gemm (beta=0, no bias) "
    "+ manual kernel-add-bias, versus PyTorch's `at::linear`/addmm (which "
    "fuses bias into the cuBLAS reduction). Difference is only ~1-2 ULP per "
    "step, but amplifies by ~sqrt(K) per GEMM through the 13-step chain, so "
    "the final error reaches 4.3e13 (rel 0.48). Fix: use cuBLASLt with "
    "CUBLASLT_EPILOGUE_BIAS to match `at::linear` numerically and remove "
    "bias-add from the element-wise kernels."
    "</arg_value><arg_key>target</arg_key><arg_value>root</arg_value>"
    "</tool_call>")


def _recorded_generation() -> tuple[str, list[str]]:
    """Rebuild the captured generation and its 169 frames."""
    filler = ("The maxdiff grows through the chain, so the bias-add kernels "
              "come out slightly off from at::linear. ")
    chars = list(
        (filler * (_RECORDED_LENGTH // len(filler) + 1))[:_RECORDED_LENGTH])
    for offset, snippet in _RECORDED_ANGLE_BRACKETS:
        chars[offset:offset + len(snippet)] = snippet
    chars[_RECORDED_TOOL_CALL_OFFSET:] = _RECORDED_TOOL_CALL
    text = "".join(chars)
    frames, at = [], 0
    for size in _RECORDED_FRAME_LENS:
        frames.append(text[at:at + size])
        at += size
    return text, frames


class TestGlmToolCallEndsReasoning:
    """A tool call opened without `</think>` must not stay in reasoning.

    GLM's chat template prefills `<think>`, so the parser starts inside the
    block. When the model reaches a tool call without ever closing it, the
    whole call -- markup included -- was classified as `reasoning_content`;
    the tool parser only ever sees the content half, so the call reached the
    client as reasoning text and never as a call.

    `TestGlmReasoningParser` above owns the opposite shape (GLM closing the
    block twice) and its tests double as the regression guard for it: this
    change must leave those outputs byte for byte identical.
    """

    TOOL_CALL = "<tool_call>"
    # Short enough to re-chunk exhaustively, and the same shape as the
    # recorded call: reasoning, then the marker, then `<arg_key>` markup
    # that a naive "split on the first `<`" would also have to survive.
    SPLIT_CASE = ("Planning the next step."
                  "<tool_call>collab.send<arg_key>m</arg_key></tool_call>")
    # The (content, reasoning) split `SPLIT_CASE` has to produce, no matter
    # how it is chunked. The sweeps below compare against this rather than
    # against a whole-marker run of the same parser: a parser that leaks the
    # call is self-consistent under every chunking, so agreement alone would
    # pass on the unfixed code.
    SPLIT_CASE_RESULT = (
        "<tool_call>collab.send<arg_key>m</arg_key></tool_call>",
        "Planning the next step.")

    @staticmethod
    def _parser(**kwargs):
        if kwargs:
            return GlmReasoningParser(**kwargs)
        return ReasoningParserFactory.create_reasoning_parser("glm")

    @classmethod
    def _stream(cls, deltas, **kwargs) -> tuple[str, str]:
        """Feed `deltas` to a fresh parser; return (content, reasoning)."""
        parser = cls._parser(**kwargs)
        content, reasoning = [], []
        for delta in deltas:
            result = parser.parse_delta(delta)
            content.append(result.content)
            reasoning.append(result.reasoning_content)
        result = parser.finish()
        content.append(result.content)
        reasoning.append(result.reasoning_content)
        return "".join(content), "".join(reasoning)

    @staticmethod
    def _chunks(text: str, size: int) -> list[str]:
        return [text[i:i + size] for i in range(0, len(text), size)]

    # -- case 1 / case 9: the live defect, streaming and not ---------------

    def test_a_tool_call_without_a_closing_tag_reaches_content(self):
        result = self._parser().parse(self.SPLIT_CASE)

        assert result.reasoning_content == "Planning the next step."
        assert result.content == (
            "<tool_call>collab.send<arg_key>m</arg_key></tool_call>")

    def test_the_marker_arriving_whole_reaches_content(self):
        assert self._stream([self.SPLIT_CASE]) == self.SPLIT_CASE_RESULT

    # -- the recorded 169-frame generation ---------------------------------

    def test_the_fixture_matches_the_recorded_generation(self):
        """Guards the reduction, not the parser.

        The fixture is the capture with its prose replaced; if any property
        the parser can observe drifts, the replay below stops testing what
        was recorded.
        """
        text, frames = _recorded_generation()

        assert len(text) == _RECORDED_LENGTH
        assert len(frames) == 169 and "".join(frames) == text
        assert "<think>" not in text and "</think>" not in text
        assert text.count(self.TOOL_CALL) == 1
        assert text.index(self.TOOL_CALL) == _RECORDED_TOOL_CALL_OFFSET
        # The call arrives whole, inside one frame -- which is why the
        # capture alone never exercised the split-marker path.
        assert sum(self.TOOL_CALL in frame for frame in frames) == 1
        assert frames[165].count(self.TOOL_CALL) == 1

    def test_the_recorded_generation_puts_the_call_in_content(self):
        text, frames = _recorded_generation()

        content, reasoning = self._stream(frames)

        assert self.TOOL_CALL not in reasoning
        assert content.startswith(_RECORDED_TOOL_CALL)
        assert reasoning == text[:_RECORDED_TOOL_CALL_OFFSET]
        assert reasoning + content == text

    def test_the_recorded_generation_streams_as_it_parses(self):
        text, frames = _recorded_generation()
        whole = self._parser().parse(text)

        assert whole.reasoning_content == text[:_RECORDED_TOOL_CALL_OFFSET]
        assert self._stream(frames) == (whole.content, whole.reasoning_content)

    @pytest.mark.parametrize("size", [1, 7, 12, 25, 40])
    def test_the_recorded_generation_survives_re_chunking(self, size):
        """The recorded framing alone never splits the marker.

        It arrives whole inside frame 165, so the frames replay only the
        first half of the defect. Re-chunked, sizes 1, 7, 25 and 40 all land
        a boundary inside the marker; 12 does not, and is here so the sweep
        covers both.
        """
        text, _ = _recorded_generation()

        content, reasoning = self._stream(self._chunks(text, size))

        assert reasoning == text[:_RECORDED_TOOL_CALL_OFFSET]
        assert content == text[_RECORDED_TOOL_CALL_OFFSET:]

    # -- case 2 / case 3: the marker split across deltas --------------------

    @pytest.mark.parametrize("size", range(1, len(SPLIT_CASE) + 2))
    def test_every_uniform_chunk_size_agrees_with_the_whole_marker(self, size):
        """Re-chunking must not change the split.

        One character per delta is included: the marker is then spread over
        eleven deltas and is complete in none of them.
        """
        assert self._stream(self._chunks(self.SPLIT_CASE,
                                         size)) == self.SPLIT_CASE_RESULT

    def test_every_two_way_split_point_agrees_with_the_whole_marker(self):
        """A sweep, not a case: any boundary inside the marker is the bug.

        Splitting on complete matches alone survives 69 of these 78
        boundaries -- and one of the nine it fails is saved by accident,
        because a delta ending in a bare `<` is already withheld as a
        possible `</think>`. A single hand-picked split proves nothing.
        """
        text = self.SPLIT_CASE
        end = len(text) + 1
        got = {self._stream([text[:i], text[i:]]) for i in range(end)}

        assert got == {self.SPLIT_CASE_RESULT}

    def test_every_three_way_split_point_agrees_with_the_whole_marker(self):
        """Case 3: the marker spread over three deltas, at every offset."""
        text = self.SPLIT_CASE
        end = len(text) + 1
        got = {
            self._stream([text[:i], text[i:j], text[j:]])
            for i in range(end)
            for j in range(i, end)
        }

        assert got == {self.SPLIT_CASE_RESULT}

    # -- the invariant ------------------------------------------------------

    CONSERVATION_CASES = (
        SPLIT_CASE,
        # No markup at all.
        "just some ordinary reasoning",
        # The guard fires with markup on both sides of it.
        "plan<tool_call>f</think>after",
        # A closing tag first, so the guard must stay out of the way.
        "plan</think>visible<tool_call>f",
        # GLM's repeated close, then a call.
        "a</think>b</think><tool_call>f",
        # A `<` that is a prefix of both markers, then neither.
        "a<b<tool_call>c",
        # A `</think>` fragment that never completes, then the marker.
        "x</t<tool_call>y",
        # Text that only looks like the start of the marker.
        "Planning <tools are useful, but",
        # Truncated mid-marker, and mid-closing-tag.
        "trailing<tool_c",
        "trailing</thi",  # codespell:ignore thi
        # Degenerate: nothing but the marker, and nothing but a `<`.
        "<tool_call>",
        "<",
    )

    @pytest.mark.parametrize("text", CONSERVATION_CASES)
    @pytest.mark.parametrize("size", [1, 2, 3, 5, 7, 8, 11, 1000])
    def test_nothing_is_dropped_or_duplicated(self, text, size):
        """Reasoning plus content reconstructs the input.

        `</think>` is the one exception, and by design: the base consumes the
        first one as the block's delimiter, and a later one is dropped only
        where it is a delimiter too -- the close a tool call preempted, or a
        redundant close standing directly before `<tool_call>` -- which
        covers every complete `</think>` these cases contain. Everything
        else -- including fragments that never grew into a marker -- has to
        come out somewhere, exactly once. (A quoted `</think>` in plain
        visible text is preserved; `TestGlmReasoningReEntry` owns that
        shape.)
        """
        expected = text.replace("</think>", "")
        content, reasoning = self._stream(self._chunks(text, size))

        assert self.TOOL_CALL not in reasoning
        assert reasoning + content == expected

    @pytest.mark.parametrize("text", CONSERVATION_CASES)
    def test_nothing_is_dropped_or_duplicated_without_streaming(self, text):
        expected = text.replace("</think>", "")
        result = self._parser().parse(text)

        assert self.TOOL_CALL not in result.reasoning_content
        assert result.reasoning_content + result.content == expected

    # -- the remaining edge cases -------------------------------------------

    def test_a_closing_tag_in_the_same_delta_still_ends_the_block(self):
        """Case 5: `</think>` first, so the block ends there.

        Splitting at the marker instead would leave the closing tag stranded
        at the end of `reasoning_content`.
        """
        text = "plan</think>say<tool_call>f"
        expected = ("say<tool_call>f", "plan")

        assert self._parser().parse(text).reasoning_content == "plan"
        assert self._stream([text]) == expected
        assert self._stream(self._chunks(text, 1)) == expected

    def test_text_that_only_looks_like_the_marker_is_released(self):
        """Case 6: nothing may be withheld forever."""
        got = self._stream(["Planning <tool", "s are useful"])

        assert got == ("", "Planning <tools are useful")

    def test_a_stream_ending_mid_marker_flushes_the_fragment(self):
        """Case 7: an incomplete tag is ordinary text, not markup.

        It stays on the reasoning side because that is the block it was
        withheld in -- the rule `finish()` already applies to a partial
        `</think>`.
        """
        parser = self._parser()

        assert parser.parse_delta("done<tool_c").reasoning_content == "done"
        assert parser.finish().reasoning_content == "<tool_c"

    @pytest.mark.parametrize("size", [1, 3, 1000])
    def test_the_guard_is_silent_when_reasoning_is_disabled(self, size):
        """Case 10: with no block open there is nothing to end."""
        text = "hello<tool_call>f"
        off = self._parser(reasoning_at_start=False)

        assert off.parse(text) == ReasoningParserResult(content=text)
        assert self._stream(self._chunks(text, size),
                            reasoning_at_start=False) == (text, "")

    def test_nemotron_is_left_alone(self):
        """The same hole exists there, and is a separate change.

        Asserted so that closing it is a deliberate edit to this test rather
        than a silent side effect of touching the shared helpers.
        """
        parser = ReasoningParserFactory.create_reasoning_parser("nemotron-v3")
        content, reasoning = "", ""
        for delta in ["Planning<tool_", "call>exec"]:
            result = parser.parse_delta(delta)
            content += result.content
            reasoning += result.reasoning_content

        assert content == "" and self.TOOL_CALL in reasoning


# The recorded re-entry failure, abridged: the model closed `</think>`,
# wrote visible text, then emitted `<think>` again mid-message, and the
# generation ends on a dangling `</`. Frame boundaries are the ones
# delivered; the elisions cut prose the parser cannot react to.
_REENTRY_FRAMES = (
    "Still",
    " maybe. Use 30 more.</think>## NCU run still in queue - polling once "
    "more, then reporting palette results and the algorithmic findings "
    "while we wait. Profiling takes long on jade\n\n```\n\n"
    "{Comment, palette, print}\nGive. Engineers. Kind",
    ". Color. One. Nice. Tune. Not.<think>It appears the trailing "
    "commentary text became garbled - possibly due to token corruption. "
    "Let me just say I'm waiting on the NCU run",
    ", one GPU request at a time.\n\nCan I proceed. Yes - poll NCU once "
    "more.\n</",
)
_REENTRY_TEXT = "".join(_REENTRY_FRAMES)


class TestGlmReasoningReEntry:
    """The model closes the block, writes text, then opens `<think>` again.

    Measured on the recorded frames above: the streamed view dropped the 30
    visible characters standing before the re-entry marker (the base class
    discards text ahead of a `<think>` it hunts down) and classified the
    second block as reasoning, while the whole-text parse kept both as
    content - stream and final described different documents, with
    characters existing in neither.

    The rule, on both paths: the first close wins. Once `</think>` has
    closed the block, a later `<think>` does not re-enter it - it is
    visible text, preserved verbatim, marker included - which is what the
    whole-text parse (one partition on the first close) always did. And a
    later `</think>` is dropped only in the delimiter shapes
    `TestGlmReasoningParser` pins; quoted in plain visible text it is
    preserved, where it used to be stripped out of tool output the model
    was echoing.
    """

    @staticmethod
    def _parser():
        return ReasoningParserFactory.create_reasoning_parser("glm")

    @classmethod
    def _stream(cls, deltas) -> tuple[str, str]:
        parser = cls._parser()
        content, reasoning = [], []
        for delta in deltas:
            result = parser.parse_delta(delta)
            content.append(result.content)
            reasoning.append(result.reasoning_content)
        result = parser.finish()
        content.append(result.content)
        reasoning.append(result.reasoning_content)
        return "".join(content), "".join(reasoning)

    @staticmethod
    def _chunks(text: str, size: int) -> list[str]:
        return [text[i:i + size] for i in range(0, len(text), size)]

    def test_the_whole_parse_reads_everything_after_the_close_as_content(self):
        result = self._parser().parse(_REENTRY_TEXT)

        assert result.reasoning_content == "Still maybe. Use 30 more."
        assert result.content == _REENTRY_TEXT.split("</think>", 1)[1]
        # Zero characters lost: the input is the split plus its delimiter.
        assert (result.reasoning_content + "</think>" +
                result.content == _REENTRY_TEXT)

    def test_the_recorded_frames_stream_as_they_parse(self):
        whole = self._parser().parse(_REENTRY_TEXT)

        content, reasoning = self._stream(_REENTRY_FRAMES)

        assert (content, reasoning) == (whole.content, whole.reasoning_content)

    def test_the_dropped_thirty_characters_are_streamed(self):
        """The measured loss, asserted back directly."""
        content, _ = self._stream(_REENTRY_FRAMES)

        assert ". Color. One. Nice. Tune. Not." in content
        assert "<think>It appears" in content
        assert content.endswith("\n</")

    @pytest.mark.parametrize("size", [1, 3, 7, 8, 11, 40, 1000])
    def test_every_uniform_chunk_size_agrees_with_the_whole_parse(self, size):
        whole = self._parser().parse(_REENTRY_TEXT)

        got = self._stream(self._chunks(_REENTRY_TEXT, size))

        assert got == (whole.content, whole.reasoning_content)

    def test_every_two_way_split_point_agrees_with_the_whole_parse(self):
        whole = self._parser().parse(_REENTRY_TEXT)
        expected = (whole.content, whole.reasoning_content)
        text = _REENTRY_TEXT

        got = {self._stream([text[:i], text[i:]]) for i in range(len(text) + 1)}

        assert got == {expected}

    @pytest.mark.parametrize("size", [1, 3, 8, 1000])
    def test_a_later_think_is_visible_text_not_a_re_entry(self, size):
        text = "reasoned</think>answer<think>afterthought"

        whole = self._parser().parse(text)
        streamed = self._stream(self._chunks(text, size))

        assert whole.content == "answer<think>afterthought"
        assert whole.reasoning_content == "reasoned"
        assert streamed == (whole.content, whole.reasoning_content)

    @pytest.mark.parametrize("size", [1, 3, 8, 1000])
    def test_a_quoted_close_in_visible_text_is_preserved(self, size):
        """`_without_stray_end` must not strip quoted markers.

        The other half of the fix: marker-like text inside content it merely
        passes through - tool output quoting the tag, source code printing
        it - stays. Only the delimiter shapes (`</think><tool_call>`, and
        the close a tool call preempted) are dropped.
        """
        text = ("plan</think>The tool printed a literal </think> marker "
                "inside its output text.")

        whole = self._parser().parse(text)
        streamed = self._stream(self._chunks(text, size))

        assert whole.content == ("The tool printed a literal </think> "
                                 "marker inside its output text.")
        assert whole.reasoning_content == "plan"
        assert streamed == (whole.content, whole.reasoning_content)

    @pytest.mark.parametrize("size", [1, 3, 8, 1000])
    def test_a_re_entered_block_may_still_carry_a_tool_call(self, size):
        """A call in the visible text keeps working after a re-entry."""
        text = ("think one</think>visible<think>more thoughts"
                "<tool_call>exec</tool_call>tail")

        whole = self._parser().parse(text)
        streamed = self._stream(self._chunks(text, size))

        assert whole.content == ("visible<think>more thoughts"
                                 "<tool_call>exec</tool_call>tail")
        assert streamed == (whole.content, whole.reasoning_content)

    @pytest.mark.parametrize("size", [1, 3, 8, 1000])
    def test_a_close_ending_the_stream_is_preserved(self, size):
        """A trailing close marker is text.

        At end of stream nothing follows the marker, so it cannot be the
        delimiter shape; both paths keep it as text.
        """
        text = "r</think>text</think>"

        whole = self._parser().parse(text)
        streamed = self._stream(self._chunks(text, size))

        assert whole.content == "text</think>"
        assert streamed == (whole.content, whole.reasoning_content)

    @pytest.mark.parametrize("size", [1, 3, 8, 1000])
    def test_a_think_marker_ending_the_stream_stays_visible(self, size):
        text = "r</think>c<think>"

        whole = self._parser().parse(text)
        streamed = self._stream(self._chunks(text, size))

        assert whole.content == "c<think>"
        assert streamed == (whole.content, whole.reasoning_content)


class TestGlmDoubleCloseResidue:
    """The model closes its block twice in a row before plain text.

    Recorded 2026-09-30 (one frame, verbatim shape): `...Let's
    do.</think></think>I'm going to try the CUDA WMMA path...`. The
    partition consumes the first close; the second stood at the very head
    of the content and, after the quoting-preservation narrowing, nothing
    dropped it any more - 32-35% of assistant turns shipped a literal
    `</think>` prefix to the client. A close before the first content
    character cannot be a quote (a quote needs text to carry it), so every
    consecutive leading close is delimiter residue, dropped on both views;
    a close after real content keeps the narrowed quoting rule.
    """

    TEXT = ("Search cutlass for wmma. Let's do."
            "</think></think>I'm going to try the CUDA WMMA path next.")

    @staticmethod
    def _parser():
        return ReasoningParserFactory.create_reasoning_parser("glm")

    @classmethod
    def _stream(cls, deltas) -> tuple[str, str]:
        parser = cls._parser()
        content, reasoning = [], []
        for delta in deltas:
            result = parser.parse_delta(delta)
            content.append(result.content)
            reasoning.append(result.reasoning_content)
        result = parser.finish()
        content.append(result.content)
        reasoning.append(result.reasoning_content)
        return "".join(content), "".join(reasoning)

    @staticmethod
    def _chunks(text: str, size: int) -> list[str]:
        return [text[i:i + size] for i in range(0, len(text), size)]

    def test_the_whole_parse_drops_the_leading_residue(self):
        result = self._parser().parse(self.TEXT)

        assert result.reasoning_content == ("Search cutlass for wmma. "
                                            "Let's do.")
        assert result.content == "I'm going to try the CUDA WMMA path next."

    @pytest.mark.parametrize("size", [1, 2, 3, 5, 7, 11, 16, 40, 200])
    def test_the_stream_agrees_on_every_chunking(self, size):
        whole = self._parser().parse(self.TEXT)

        content, reasoning = self._stream(self._chunks(self.TEXT, size))

        assert (content, reasoning) == (whole.content, whole.reasoning_content)

    def test_a_triple_close_is_still_residue(self):
        text = "plan</think></think></think>done"
        whole = self._parser().parse(text)
        assert whole.content == "done"

        content, _ = self._stream(self._chunks(text, 3))
        assert content == "done"

    def test_a_double_close_at_end_of_stream_leaves_no_content(self):
        text = "plan</think></think>"
        whole = self._parser().parse(text)
        assert whole.content == ""

        content, _ = self._stream(self._chunks(text, 4))
        assert content == ""

    def test_a_quoted_close_after_real_content_is_still_preserved(self):
        text = "plan</think>The tool printed </think> in its output."
        whole = self._parser().parse(text)
        assert whole.content == "The tool printed </think> in its output."

        content, _ = self._stream(self._chunks(text, 6))
        assert content == whole.content
