# Adapted from https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/function_call/glm4_moe_detector.py
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
import ast
import json
import math
import re
from collections import deque
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from tensorrt_llm.logger import logger
from tensorrt_llm.serve.openai_protocol import ChatCompletionToolsParam as Tool
from tensorrt_llm.serve.tool_parser.base_tool_parser import BaseToolParser
from tensorrt_llm.serve.tool_parser.core_types import (
    StreamingParseResult,
    ToolCallItem,
    _GetInfoFunc,
)

from .utils import infer_type_from_json_schema


class StreamState(str, Enum):
    """State machine states for XML to JSON streaming conversion."""

    INIT = "INIT"
    BETWEEN = "BETWEEN"
    IN_KEY = "IN_KEY"
    WAITING_VALUE = "WAITING_VALUE"
    IN_VALUE = "IN_VALUE"
    # A `</arg_value>` has been seen but not yet believed: it only closes the
    # value if the markup continues structurally. See classify_pending_close.
    PENDING_CLOSE = "PENDING_CLOSE"


# The two tokens that may legitimately follow a value's closing tag. Anything
# else after `</arg_value>` means the model was quoting the tag inside the
# value itself, which source-code payloads do.
_ARG_KEY_OPEN = "<arg_key>"
_TOOL_CALL_CLOSE = "</tool_call>"
_TOOL_CALL_OPEN = "<tool_call>"


def _strip_argument_separators(text: str) -> str:
    r"""Drop the filler GLM puts between tags: whitespace or a literal ``\n``.

    The same tolerance ``func_arg_regex`` spells as ``(?:\\n|\s)*``; the two
    must agree or the streaming path would read a boundary differently from
    the whole-text parse.
    """
    while True:
        stripped = text.lstrip()
        if stripped.startswith("\\n"):
            text = stripped[2:]
            continue
        return stripped


def classify_pending_close(buffer: str) -> str:
    r"""What the text after a ``</arg_value>`` says about that tag.

    The markup has no escaping, so a value that *contains* the closing tag -
    a shell or JS payload quoting it, say - is indistinguishable from the tag
    itself at the moment it appears. Cutting the value at the first match is
    what used to truncate such payloads. The tag is therefore only honored
    when what follows it is consistent with the enclosing structure: the next
    ``<arg_key>`` or the end of the call. Returns one of

    * ``"key"``     - separators then a complete ``<arg_key>``: a real close.
    * ``"pending"`` - the buffer could still become a real close: separators,
      a prefix of ``<arg_key>`` or of ``</tool_call>``, or a lone ``\\`` that
      may yet become a literal ``\\n`` separator.
    * ``"content"`` - anything else: the tag was part of the value.

    ``</tool_call>`` never reaches this buffer whole - the surrounding
    regexes cut the argument text at it - only as a partial tail, which stays
    pending until the caller finalizes the call and flushes.

    The residual ambiguity is honest: a value that itself contains a
    well-formed ``</arg_value><arg_key>`` (or ``</arg_value></tool_call>``)
    sequence reads as structure and truncates there. Nothing in the grammar
    can distinguish it; both parse paths read it the same way, which is the
    most that can be promised.
    """
    rest = _strip_argument_separators(buffer)
    if rest == _ARG_KEY_OPEN:
        return "key"
    if rest == "\\" or _ARG_KEY_OPEN.startswith(rest) or _TOOL_CALL_CLOSE.startswith(rest):
        return "pending"
    return "content"


def split_dead_close_buffer(buffer: str) -> Tuple[str, str]:
    r"""Split a tag buffer that just stopped being a prefix of ``</arg_value>``.

    Returns ``(released, kept)``: `released` is settled value content, `kept`
    is where the marker match restarts. The IN_VALUE buffer only grows while
    it still prefixes the closing tag, so it arrives here as a dead prefix
    plus the character that killed it - and since ``<`` appears nowhere in
    the tag past position 0, the only place a fresh match can begin is a
    trailing ``<``. Releasing the whole buffer instead, as this used to,
    swallowed that ``<`` as value content when it was really the tag's
    opener: a value ending in ``<`` (C++ template syntax, in a recorded grep
    pattern) put ``<<`` in the buffer, both characters went out as value,
    and the real ``</arg_value>`` behind them could never match from its
    first byte again - the close went unseen and the withheld value reached
    the client as ``"pattern": }``. The invariant this restores: no byte is
    ever dropped or double-counted - each is either value content or part of
    an exactly-matched marker.
    """
    restart = buffer.find("<", 1)
    if restart == -1:
        return buffer, ""
    return buffer[:restart], buffer[restart:]


def split_name_region(region: str) -> Tuple[str, str]:
    r"""Split what a name regex captured into ``(name, misplaced markup)``.

    A tool name never contains ``<``, so the name proper stops at the first
    one; everything from that ``<`` on is markup the model misplaced, for
    `classify_name_region` to judge. The glm47 regexes encode the same split
    structurally (``([^<]*)`` followed by a junk group); glm4's name group is
    delimited by the newline that ends the name line instead, so the split
    happens here.
    """
    name, sep, junk = region.partition("<")
    return name, sep + junk


def classify_name_region(name: str, junk: str, tool_indices: Dict[str, int]) -> Tuple[str, Any]:
    r"""What the text between ``<tool_call>`` and the arguments says the call is.

    `name` is the region up to its first ``<`` and `junk` everything from that
    ``<`` on. With the old ``(.*?)`` name groups the junk was swallowed INTO
    the name and delivered verbatim: two recorded GLM-5.3 calls reached their
    client named ``exec<tool_call>exec`` (the model restarted its call midway,
    doubling the opener) and ``exec<arg_value>input</arg_key>...`` (tags
    written out of order), and the client, seeing an unknown tool, lost the
    turn. Returns one of

    * ``("clean", name)`` - no junk. The well-formed path, unchanged.
    * ``("restart", offset)`` - `junk` holds a fresh ``<tool_call>`` (the last
      one, at `offset` within `junk`): the model abandoned the call it had
      opened and started over. Everything before the fresh opener is dead
      markup - its own name region is already junk-terminated, so no later
      text can make it parse - and the call that follows is the one the model
      meant. Callers release the dead prefix as ordinary text and parse on
      from the fresh opener.
    * ``("repaired", resolved)`` - the resolver strips the junk back to a
      declared tool (``apply_patch</arg_value>`` -> ``apply_patch``): the
      established recovery for an unbalanced tag fused onto a real name,
      preserved exactly because the arguments that follow are intact.
    * ``("malformed", None)`` - junk that maps onto nothing declared. The call
      must not be delivered: a name carrying markup is what the client just
      rejects wholesale, and inventing a call under a guessed name is worse.
      Callers release the call's entire text as ordinary output - the same
      stance `_flush_tool_parser` takes for unterminated markup, because
      silently losing model output is worse than showing a call that never
      parsed.

    Streaming callers may act on ``"restart"`` the moment it appears - a
    complete ``<tool_call>`` inside the junk is fixed text and final - but
    must gate the other verdicts on the structural token that seals the
    region (``<arg_key>`` or ``</tool_call>``): a half-arrived ``</arg_val``
    classifies as malformed while its completion repairs.
    """
    if not junk:
        return "clean", name
    restart = junk.rfind(_TOOL_CALL_OPEN)
    if restart != -1:
        return "restart", restart
    resolved = BaseToolParser.resolve_tool_name((name + junk).strip(), tool_indices)
    if resolved is not None:
        return "repaired", resolved
    return "malformed", None


def get_argument_type(func_name: str, arg_key: str, defined_tools: List[Tool]) -> Optional[str]:
    """Get the expected type of a function argument from tool definitions."""
    name2tool = {tool.function.name: tool for tool in defined_tools if tool.function.name}
    # Argument extraction sees the name as the model wrote it; parse_base_json
    # only repairs it afterwards. A bare spelling of a qualified declaration
    # (`exec` for `functions.exec` - the majority spelling on measured GLM-5
    # fleets) used to miss the schema here, so every argument of the call fell
    # through to the no-schema path however precisely its type was declared.
    # Resolve with the same rule the delivery path uses.
    resolved = BaseToolParser.resolve_tool_name(
        func_name, {name: i for i, name in enumerate(name2tool)}
    )
    if resolved is None or resolved not in name2tool:
        return None
    tool = name2tool[resolved]
    properties = (tool.function.parameters or {}).get("properties", {})
    if not isinstance(properties, dict):
        properties = {}
    if arg_key not in properties:
        return None
    return infer_type_from_json_schema(properties[arg_key])


def _convert_to_number(value: str) -> Any:
    """Convert string to appropriate number type (int or float)."""
    try:
        if "." in value or "e" in value.lower():
            return float(value)
        else:
            return int(value)
    except (ValueError, AttributeError):
        return value


def _is_json_finite(value: Any) -> bool:
    """Whether ``json.dumps`` would spell every number in `value` as JSON.

    ``json.loads`` and ``float`` overflow ``1e309`` to ``inf`` (and accept
    ``NaN``) without an exception, and ``json.dumps`` then emits the literal
    ``Infinity`` / ``NaN`` -- tokens the JSON grammar does not have, so the
    delivered arguments stop parsing downstream. Checked recursively: a
    non-finite float nested in an object-typed value poisons the whole
    serialization the same way.
    """
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(_is_json_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_is_json_finite(item) for item in value)
    return True


def parse_arguments(json_value: str, arg_type: Optional[str] = None) -> Tuple[Any, bool]:
    """Parse argument value with multiple fallback strategies.

    Only called for arguments whose declared schema type licenses a
    conversion (number, integer, boolean, object, array); string-typed and
    schema-less arguments never come here - they are passed through as the
    raw text between the markers. See ``_parse_argument_pairs``.

    A parse that yields a non-finite float is treated as no parse at all:
    the model wrote a decimal string, and if it cannot be represented
    faithfully as a JSON number the original text is passed through as a
    string rather than serialized to the non-JSON ``Infinity``. The last
    strategy below delivers exactly that (the raw text, JSON-quoted at the
    dumps site), so the guarded strategies simply fall through to it.

    Returns:
        Tuple of (parsed_value, is_valid_json)
    """
    numeric = ("number", "integer")
    try:
        parsed_value = json.loads(json_value)
        if arg_type in numeric and isinstance(parsed_value, str):
            parsed_value = _convert_to_number(parsed_value)
        if _is_json_finite(parsed_value):
            return parsed_value, True
    except (json.JSONDecodeError, ValueError):
        pass

    try:
        wrapped = json.loads('{"tmp": "' + json_value + '"}')
        parsed_value = json.loads(wrapped["tmp"])
        if arg_type in numeric and isinstance(parsed_value, str):
            parsed_value = _convert_to_number(parsed_value)
        if _is_json_finite(parsed_value):
            return parsed_value, True
    except (json.JSONDecodeError, ValueError, KeyError):
        pass

    try:
        parsed_value = ast.literal_eval(json_value)
        if _is_json_finite(parsed_value):
            return parsed_value, True
    except (ValueError, SyntaxError):
        pass

    try:
        quoted_value = json.dumps(str(json_value))
        return json.loads(quoted_value), True
    except (json.JSONDecodeError, ValueError):
        return json_value, False


class Glm4ToolParser(BaseToolParser):
    r"""Tool parser for GLM-4.5 and GLM-4.6 models.

    Assumes function call format (with actual newlines):
        <tool_call>get_weather
        <arg_key>city</arg_key>
        <arg_value>北京</arg_value>
        <arg_key>date</arg_key>
        <arg_value>2024-06-27</arg_value>
        </tool_call>

    Or with literal \n characters (escaped as \\n in the output):
        <tool_call>get_weather\n<arg_key>city</arg_key>\n<arg_value>北京</arg_value>\n</tool_call>

    Uses a streaming state machine to convert XML to JSON incrementally.
    """

    markup_tokens = (
        "<tool_call>",
        "</tool_call>",
        "<arg_key>",
        "</arg_key>",
        "<arg_value>",
        "</arg_value>",
    )

    def __init__(self):
        super().__init__()
        self.bot_token = "<tool_call>"  # nosec B105
        self.eot_token = "</tool_call>"  # nosec B105
        self.func_call_regex = r"<tool_call>.*?</tool_call>"
        self.func_detail_regex = re.compile(
            r"<tool_call>(.*?)(?:\\n|\n)(.*)</tool_call>", re.DOTALL
        )
        # The value ends at a `</arg_value>` that the *structure* confirms:
        # one followed (over separators) by the next `<arg_key>`, the end of
        # the call, or the end of the argument text. Stopping at the first
        # `</arg_value>` substring truncated any value that quotes the tag -
        # code payloads do - at that quote. The lookahead makes the lazy
        # match skip embedded tags; a value containing a full well-formed
        # `</arg_value><arg_key>` sequence remains ambiguous and reads as
        # structure (see classify_pending_close, which mirrors this rule for
        # the streaming path).
        self.func_arg_regex = re.compile(
            r"<arg_key>(.*?)</arg_key>(?:\\n|\s)*<arg_value>(.*?)</arg_value>"
            r"(?=(?:\\n|\s)*(?:<arg_key>|</tool_call>|$))",
            re.DOTALL,
        )
        self._last_arguments = ""
        self.current_tool_id = -1
        self.current_tool_name_sent = False
        self._streamed_raw_length = 0
        self._reset_streaming_state()

    def _reset_streaming_state(self) -> None:
        """Reset the streaming state machine for a new tool call."""
        self._stream_state = StreamState.INIT
        self._current_key = ""
        self._current_value = ""
        self._xml_tag_buffer = ""
        self._is_first_param = True
        self._value_started = False
        self._cached_value_type: Optional[str] = None
        # Key -> raw value text of every pair already put on the stream for
        # this call, so a repeated key can be recognized before any of its
        # JSON is emitted. See the duplicate handling in `_close_current_value`.
        self._streamed_pairs: Dict[str, str] = {}
        self._suppressing_duplicate = False

    @property
    def single_call_stop(self) -> Optional[str]:
        # Every call is its own <tool_call>...</tool_call> block with nothing
        # around it, so the closing tag completes exactly one call.
        return self.eot_token

    def has_tool_call(self, text: str) -> bool:
        """Check if the text contains a GLM-4 format tool call."""
        return self.bot_token in text

    def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult:
        """One-time parsing: Detects and parses tool calls in the provided text."""
        idx = text.find(self.bot_token)
        normal_text = text[:idx].strip() if idx != -1 else text
        if self.bot_token not in text:
            return StreamingParseResult(normal_text=normal_text, calls=[])
        match_result_list = re.findall(self.func_call_regex, text, re.DOTALL)
        calls = []
        try:
            for match_result in match_result_list:
                segment_calls, released = self._parse_call_segment(match_result, tools)
                calls.extend(segment_calls)
                # A segment (or a dead prefix of one) that the name rule
                # rejected is still model output: it joins the visible text
                # rather than vanishing - the same disposition
                # _flush_tool_parser takes for unterminated markup.
                normal_text += released
            return StreamingParseResult(normal_text=normal_text, calls=calls)
        except Exception as e:
            logger.error(f"Error in detect_and_parse: {e}")
            return StreamingParseResult(normal_text=text)

    def _parse_call_segment(
        self, segment: str, tools: List[Tool]
    ) -> Tuple[List[ToolCallItem], str]:
        """One sliced ``<tool_call>...</tool_call>`` -> (calls, text to release).

        The name region decides the segment's fate (see classify_name_region):
        a clean or resolver-repairable name delivers exactly as before; a
        restarted call re-anchors on the fresh opener with the abandoned
        prefix released as text; unrepairable junk releases the whole segment
        as text, because a name carrying markup must never reach a client -
        two recorded GLM-5.3 turns were lost to exactly that.
        """
        released_parts: List[str] = []
        while True:
            func_detail = self.func_detail_regex.search(segment)
            if func_detail is None:
                # No name line the format can read (glm4 requires a newline
                # after the name). This used to `continue`, silently deleting
                # the whole segment from the response; unreadable markup is
                # still model output and lands in the visible text instead.
                released_parts.append(segment)
                return [], "".join(released_parts)
            region = func_detail.group(1) if func_detail.group(1) else ""
            name_part, junk = split_name_region(region)
            verdict, recovered = classify_name_region(
                name_part.strip(), junk, self._get_tool_indices(tools)
            )
            if verdict != "restart":
                break
            # The model abandoned the call it had opened and started over.
            # The markup before the fresh opener can never parse - its name
            # region is already junk-terminated - so it is released as text
            # and parsing re-anchors on the call the model finished. One
            # pass suffices: the re-anchored region holds no further opener
            # (rfind took the last one).
            cut = func_detail.start(1) + len(name_part) + recovered
            released_parts.append(segment[:cut])
            segment = segment[cut:]

        if verdict == "malformed":
            logger.warning(
                f"Tool call name region carries markup that maps onto no declared tool "
                f"(name {name_part.strip()!r}, {len(junk)} junk chars); releasing the "
                f"{len(segment)}-char call as message text instead of a corrupted name"
            )
            released_parts.append(segment)
            return [], "".join(released_parts)

        # "clean" or "repaired": today's delivery, byte for byte. On the
        # repaired path the raw region goes through so parse_base_json
        # performs - and logs - the same name recovery it always has.
        func_name = region
        func_args = func_detail.group(2) if func_detail.group(2) else ""
        pairs = self.func_arg_regex.findall(func_args)
        arguments = self._parse_argument_pairs(pairs, func_name, tools)
        segment_calls = self.parse_base_json({"name": func_name, "parameters": arguments}, tools)
        return segment_calls, "".join(released_parts)

    def _get_value_type(self, func_name: str, key: str, tools: List[Tool]) -> str:
        """Get parameter type from tool definition, defaulting to string.

        Only a declared schema licenses a conversion. This used to fall back
        to sniffing the value's content - dead code in practice, since it ran
        at `<arg_value>` when the value was still empty, but had it ever run
        it would have promoted digit-leading text to a number the schema
        never asked for. With no schema the raw text is the only faithful
        answer, and it is what `_parse_argument_pairs` delivers on the
        whole-text path; the streaming path must agree.
        """
        return get_argument_type(func_name, key, tools) or "string"

    def _format_value_complete(self, value: str, value_type: str) -> str:
        """Format complete value based on type."""
        if value_type == "string":
            return json.dumps(value, ensure_ascii=False)
        # Mirror the whole-text path (`_parse_argument_pairs`): parse under
        # the declared type, fall back to the raw text as a JSON string.
        # str()-ing whatever came back, or emitting the raw text bare, put
        # unquoted non-JSON on the stream whenever the value did not parse -
        # and json.dumps of an overflowed float would put `Infinity` there,
        # which parse_arguments now refuses (see _is_json_finite).
        parsed_value, is_good_json = parse_arguments(value.strip(), value_type)
        if not is_good_json:
            logger.warning(f"Failed to parse '{value}' as {value_type}, treating as string")
            parsed_value = value.strip()
        return json.dumps(parsed_value, ensure_ascii=False)

    def _append_value_content(self, content: str) -> str:
        """Record `content` as value text, streaming it when that is sound.

        String values stream as JSON string content (opening quote on first
        use, characters escaped): any text is valid inside a JSON string, so
        nothing sent can turn out wrong. Non-string values are withheld until
        the close is confirmed - only the complete text says whether it
        parses under the declared type (a finite number, a well-formed
        object) or falls back to a quoted string, and bytes already sent
        cannot be recalled. `1e309` streamed raw would read as a number while
        the whole-text parse delivers the string "1e309"; the two views must
        not disagree. A pair whose key duplicates one already streamed emits
        nothing at all (see `_close_current_value`).
        """
        self._current_value += content
        if self._suppressing_duplicate:
            return ""
        value_type = self._cached_value_type or "string"
        if value_type != "string":
            return ""
        fragment = ""
        if not self._value_started:
            fragment += '"'
            self._value_started = True
        fragment += json.dumps(content, ensure_ascii=False)[1:-1]
        return fragment

    def _close_current_value(self) -> str:
        """The JSON that finishes the value whose close was just confirmed.

        A duplicate of a key already on the stream emits nothing: the first
        occurrence's bytes are already with the client and cannot be
        un-emitted, so the occurrence that reached the wire first wins - on
        this path and, identically, in `_parse_argument_pairs` - and the
        arguments JSON carries the key exactly once. An identical repeat
        collapses silently; a conflicting one is logged with the discarded
        value's length.
        """
        value_type = self._cached_value_type or "string"
        if self._suppressing_duplicate:
            first_value = self._streamed_pairs[self._current_key]
            discarded = self._current_value.strip()
            if discarded != first_value:
                logger.debug(
                    f"Duplicate tool argument key {self._current_key!r}: keeping the "
                    f"value already streamed, discarding a conflicting later value "
                    f"of {len(discarded)} chars"
                )
            fragment = ""
        else:
            if self._value_started:
                fragment = '"' if value_type == "string" else ""
            else:
                fragment = self._format_value_complete(self._current_value, value_type)
            self._streamed_pairs[self._current_key] = self._current_value.strip()
        self._stream_state = StreamState.BETWEEN
        self._current_value = ""
        self._value_started = False
        self._cached_value_type = None
        self._suppressing_duplicate = False
        self._xml_tag_buffer = ""
        return fragment

    def _process_xml_to_json_streaming(
        self, raw_increment: str, func_name: str, tools: List[Tool]
    ) -> str:
        """Convert XML increment to JSON streaming output using state machine.

        Processes XML fragments character by character and converts them
        to JSON format incrementally, maintaining state across calls.
        """
        json_output = ""

        # A deque rather than a plain loop so PENDING_CLOSE can push back the
        # text it looked ahead at when a `</arg_value>` turns out to be value
        # content; those characters then re-run through IN_VALUE like any
        # others. Replayed text can never hold a whole `</arg_value>`: the
        # lookahead breaks off at the first character that cannot extend
        # `<arg_key>` or `</tool_call>`, and `</arg_value>` diverges from both
        # by its third character. So nothing replays twice and the loop ends.
        pending_chars = deque(raw_increment)
        while pending_chars:
            char = pending_chars.popleft()
            self._xml_tag_buffer += char

            if self._stream_state in [StreamState.INIT, StreamState.BETWEEN]:
                if self._xml_tag_buffer.endswith("<arg_key>"):
                    self._stream_state = StreamState.IN_KEY
                    self._current_key = ""
                    self._xml_tag_buffer = ""

            elif self._stream_state == StreamState.IN_KEY:
                if self._xml_tag_buffer.endswith("</arg_key>"):
                    self._current_key = self._xml_tag_buffer[:-10].strip()
                    self._xml_tag_buffer = ""
                    self._stream_state = StreamState.WAITING_VALUE
                    # The separator waits for the key (rather than going out
                    # at `<arg_key>`) so a duplicate can be suppressed whole:
                    # a `, ` already emitted for a pair that then emits
                    # nothing else would corrupt the stream.
                    if self._current_key in self._streamed_pairs:
                        # Repeated key. Its first occurrence is already on
                        # the wire; this pair is swallowed and resolved at
                        # `_close_current_value`.
                        self._suppressing_duplicate = True
                    else:
                        json_output += "{" if self._is_first_param else ", "
                        self._is_first_param = False
                        json_output += json.dumps(self._current_key, ensure_ascii=False) + ": "

            elif self._stream_state == StreamState.WAITING_VALUE:
                if self._xml_tag_buffer.endswith("<arg_value>"):
                    self._stream_state = StreamState.IN_VALUE
                    self._current_value = ""
                    self._xml_tag_buffer = ""
                    self._value_started = False
                    self._cached_value_type = self._get_value_type(
                        func_name, self._current_key, tools
                    )

            elif self._stream_state == StreamState.IN_VALUE:
                if self._xml_tag_buffer.endswith("</arg_value>"):
                    # The tag alone does not end the value: a payload may be
                    # quoting it. Withhold judgement (and the closing quote)
                    # until the following text confirms it as structure -
                    # the mirror of func_arg_regex's lookahead.
                    self._current_value += self._xml_tag_buffer[:-12]
                    self._xml_tag_buffer = ""
                    self._stream_state = StreamState.PENDING_CLOSE
                else:
                    closing_tag = "</arg_value>"
                    is_potential_closing = len(self._xml_tag_buffer) <= len(
                        closing_tag
                    ) and closing_tag.startswith(self._xml_tag_buffer)

                    if not is_potential_closing:
                        # A dead buffer may still end where the real tag
                        # begins - a value's trailing `<` against the true
                        # `</arg_value>` - so only the bytes that cannot
                        # start the marker are released as value; see
                        # split_dead_close_buffer for the full account.
                        released, self._xml_tag_buffer = split_dead_close_buffer(
                            self._xml_tag_buffer
                        )
                        json_output += self._append_value_content(released)

            elif self._stream_state == StreamState.PENDING_CLOSE:
                verdict = classify_pending_close(self._xml_tag_buffer)
                if verdict == "key":
                    json_output += self._close_current_value()
                    # The lookahead consumed the whole `<arg_key>`, so take
                    # the BETWEEN -> IN_KEY transition here as well. The
                    # separator follows at `</arg_key>`, once the key can be
                    # checked against `_streamed_pairs`.
                    self._stream_state = StreamState.IN_KEY
                    self._current_key = ""
                    self._xml_tag_buffer = ""
                elif verdict == "content":
                    # The tag was value text after all: re-emit it as content
                    # and replay the looked-ahead characters through IN_VALUE.
                    json_output += self._append_value_content("</arg_value>")
                    replay = self._xml_tag_buffer
                    self._xml_tag_buffer = ""
                    self._stream_state = StreamState.IN_VALUE
                    pending_chars.extendleft(reversed(replay))
                # else: still pending - keep accumulating lookahead.

        return json_output

    def parse_streaming_increment(self, new_text: str, tools: List[Tool]) -> StreamingParseResult:
        """Streaming incremental parsing for GLM-4 format.

        Uses a state machine to convert XML to JSON incrementally for
        true character-by-character streaming.
        """
        self._buffer += new_text
        current_text = self._buffer

        has_tool_call = self.bot_token in current_text

        if not has_tool_call:
            is_potential_start = any(
                self.bot_token.startswith(current_text[-i:])
                for i in range(1, min(len(current_text), len(self.bot_token)) + 1)
            )

            if not is_potential_start:
                # A `</tool_call>` with no opener anywhere is prose, and goes
                # out byte-identical: detect_and_parse keeps it, and the two
                # views of one generation must agree. Stripping the eot token
                # here (as the ported code did) silently deleted it from the
                # stream while the final snapshot kept it - the same defect
                # recorded against the GLM-4.7 parser's twin branch. No close
                # tag the parser owns can reach this branch: a parsed call's
                # `</tool_call>` is consumed when finalization (or the
                # malformed-call release) re-anchors the buffer past it. Nor
                # can chunking tear the tag into a half-deleted state: its
                # only shared prefix with the bot token is `<`, which the
                # potential-start hold above already covers, and any longer
                # fragment is released verbatim here and completed verbatim
                # by the next increment.
                self._buffer = ""
                return StreamingParseResult(normal_text=current_text)
            else:
                return StreamingParseResult(normal_text="", calls=[])

        if not hasattr(self, "_tool_indices"):
            self._tool_indices = self._get_tool_indices(tools)

        calls: list[ToolCallItem] = []
        try:
            partial_match = re.search(
                pattern=r"<tool_call>(.*?)(?:\\n|\n)(.*?)(</tool_call>|$)",
                string=current_text,
                flags=re.DOTALL,
            )
            if partial_match:
                func_name_raw = partial_match.group(1)
                func_args_raw = partial_match.group(2)
                is_tool_end = partial_match.group(3)

                if func_name_raw is None or not func_name_raw.strip():
                    return StreamingParseResult(normal_text="", calls=[])

                func_name = func_name_raw.strip()
                func_args_raw = func_args_raw.strip() if func_args_raw else ""

                # A tool name never contains `<`. The name line is complete
                # by construction here (the regex requires its newline), so
                # the region's verdict is final; see classify_name_region.
                name_part, junk = split_name_region(func_name_raw)
                verdict, recovered = classify_name_region(
                    name_part.strip(), junk, self._tool_indices
                )
                if verdict == "restart":
                    # The model restarted its call: release the abandoned
                    # markup as visible text, re-anchor the buffer on the
                    # fresh opener, and parse on from it within this same
                    # increment - the same split detect_and_parse makes.
                    # Re-entering matters because this parser announces a
                    # name and only streams its arguments on the *next*
                    # increment: a restart that swallowed the current one
                    # would leave a whole call pending behind a single
                    # end-of-stream drain pass that expects progress. The
                    # re-entry is bounded: every pass removes at least the
                    # abandoned opener from the buffer.
                    cut = partial_match.start(1) + len(name_part) + recovered
                    abandoned = current_text[:cut]
                    self._buffer = current_text[cut:]
                    self._streamed_raw_length = 0
                    self._reset_streaming_state()
                    logger.debug(
                        f"Model restarted a tool call mid-name; releasing "
                        f"{len(abandoned)} chars of abandoned markup as message text"
                    )
                    reparsed = self.parse_streaming_increment("", tools)
                    return StreamingParseResult(
                        normal_text=abandoned + reparsed.normal_text,
                        calls=reparsed.calls,
                    )
                if verdict == "malformed":
                    if is_tool_end == self.eot_token:
                        # The call is complete and unreadable. Release its
                        # whole text - matching what detect_and_parse does
                        # with the same bytes - rather than delivering a
                        # name with markup fused into it.
                        segment_end = partial_match.end(3)
                        segment = current_text[:segment_end]
                        self._buffer = current_text[segment_end:]
                        self._streamed_raw_length = 0
                        self._reset_streaming_state()
                        logger.warning(
                            f"Tool call name region carries markup that maps onto no "
                            f"declared tool (name {name_part.strip()!r}, {len(junk)} junk "
                            f"chars); releasing the {len(segment)}-char call as message "
                            f"text instead of a corrupted name"
                        )
                        return StreamingParseResult(normal_text=segment, calls=[])
                    # Withhold until the close arrives: only then is the
                    # segment's extent known, and the whole of it goes out
                    # as text in one piece.
                    return StreamingParseResult(normal_text="", calls=[])
                # "clean" keeps today's raw name on the wire; "repaired"
                # sends the declared tool the resolver recovered, which is
                # what parse_base_json delivers for the same bytes whole.
                wire_name = recovered if verdict == "repaired" else func_name

                if self.current_tool_id == -1:
                    self.current_tool_id = 0
                    self.prev_tool_call_arr = []
                    self.streamed_args_for_tool = [""]
                    self._streamed_raw_length = 0
                    self.current_tool_name_sent = False
                    self._reset_streaming_state()

                while len(self.prev_tool_call_arr) <= self.current_tool_id:
                    self.prev_tool_call_arr.append({})
                while len(self.streamed_args_for_tool) <= self.current_tool_id:
                    self.streamed_args_for_tool.append("")

                if not self.current_tool_name_sent:
                    calls.append(
                        ToolCallItem(
                            tool_index=self.current_tool_id,
                            name=wire_name,
                            parameters="",
                        )
                    )
                    self.current_tool_name_sent = True
                    self._streamed_raw_length = 0
                    self._reset_streaming_state()
                    self.prev_tool_call_arr[self.current_tool_id] = {
                        "name": wire_name,
                        "arguments": {},
                    }
                else:
                    current_raw_length = len(func_args_raw)

                    if current_raw_length > self._streamed_raw_length:
                        raw_increment = func_args_raw[self._streamed_raw_length :]

                        json_increment = self._process_xml_to_json_streaming(
                            raw_increment, func_name, tools
                        )

                        self._streamed_raw_length = current_raw_length

                        if json_increment:
                            calls.append(
                                ToolCallItem(
                                    tool_index=self.current_tool_id,
                                    name=None,
                                    parameters=json_increment,
                                )
                            )
                            self._last_arguments += json_increment
                            self.streamed_args_for_tool[self.current_tool_id] += json_increment

                    if is_tool_end == self.eot_token:
                        if self._stream_state == StreamState.PENDING_CLOSE:
                            # End-of-call is the structural confirmation a
                            # pending `</arg_value>` was waiting for. The
                            # lookahead buffer holds only separators or the
                            # head of `</tool_call>` itself, never value text.
                            flushed = self._close_current_value()
                            if flushed:
                                calls.append(
                                    ToolCallItem(
                                        tool_index=self.current_tool_id,
                                        name=None,
                                        parameters=flushed,
                                    )
                                )
                                self._last_arguments += flushed
                                self.streamed_args_for_tool[self.current_tool_id] += flushed
                        if self._is_first_param:
                            empty_object = "{}"
                            calls.append(
                                ToolCallItem(
                                    tool_index=self.current_tool_id,
                                    name=None,
                                    parameters=empty_object,
                                )
                            )
                            self._last_arguments += empty_object
                        else:
                            # `_is_first_param` is the only thing that says
                            # whether an opening `{` was ever emitted, so it
                            # is the only thing that can say how to close.
                            # Sniffing the streamed text for a trailing `}`
                            # instead - as this did - mistakes an argument
                            # whose value is an object for the object being
                            # closed already, and swallows the brace the call
                            # itself needs: `{"opts": {"a": 1}` reached the
                            # client one `}` short. The GLM-4.7 parser fixed
                            # the same sniff in `_finalize_tool_call`; the
                            # close-confirmation delivery of object values
                            # makes this parser hit it on every object-final
                            # call.
                            closing_brace = "}"
                            calls.append(
                                ToolCallItem(
                                    tool_index=self.current_tool_id,
                                    name=None,
                                    parameters=closing_brace,
                                )
                            )
                            self._last_arguments += closing_brace
                            self.streamed_args_for_tool[self.current_tool_id] += closing_brace

                        try:
                            pairs = self.func_arg_regex.findall(func_args_raw)
                            if pairs:
                                arguments = self._parse_argument_pairs(pairs, func_name, tools)
                                self.prev_tool_call_arr[self.current_tool_id]["arguments"] = (
                                    arguments
                                )
                        except Exception as e:
                            logger.debug(f"Failed to parse arguments: {e}")

                        self._buffer = current_text[partial_match.end(3) :]

                        result = StreamingParseResult(normal_text="", calls=calls)
                        self.current_tool_id += 1
                        self._last_arguments = ""
                        self.current_tool_name_sent = False
                        self._streamed_raw_length = 0
                        self._reset_streaming_state()
                        return result

            return StreamingParseResult(normal_text="", calls=calls)

        except Exception as e:
            logger.error(f"Error in parse_streaming_increment: {e}")
            return StreamingParseResult(normal_text=current_text)

    def finish(self, tools: List[Tool]) -> StreamingParseResult:
        """Release text held back only as a potential ``<tool_call>`` start.

        The no-opener branch of ``parse_streaming_increment`` withholds a
        buffer whose tail could still grow into the bot token; when the
        stream ends instead, that text is prose and is released verbatim -
        withheld bytes may be delayed, never dropped. The chat completions
        path relies on this: it calls only ``finish`` at end of stream, with
        no external buffer drain. A buffer that holds actual tool-call markup
        is left in place - the stream was cut off inside a call, and that
        disposition stays with the serving layer (``_flush_tool_parser``
        releases it with a warning and the unfinished call's index; releasing
        it here as well would deliver the markup twice on that path).
        """
        held = self._buffer
        if not held or self.bot_token in held:
            return StreamingParseResult()
        self._buffer = ""
        return StreamingParseResult(normal_text=held)

    def _parse_argument_pairs(
        self, pairs: List[Tuple[str, str]], func_name: str, tools: List[Tool]
    ) -> Dict[str, Any]:
        """Parse argument key-value pairs, typed by the declared schema.

        Only a declared non-string type licenses a conversion. A string
        parameter - and any parameter no schema describes, such as those of a
        tool the request never declared - is delivered as the text between
        the markers, verbatim. Guessing a type from the value's shape is what
        corrupted live traffic: a `cell_id` the schema calls a string reached
        the client as the integer 99797 and failed its type check, and a
        freeform code payload of `true` was parsed to Python True and
        str()-round-tripped into "True", which the client then executed.
        json.dumps at delivery re-quotes and re-escapes the raw text, so
        passthrough here is still valid JSON there.

        A repeated key keeps its first occurrence, identically to the
        streaming path: there the first occurrence's bytes are already with
        the client when the repeat is recognized, so first-wins is the only
        policy under which the streamed JSON can carry the key once *and*
        agree with this parse. An identical repeat collapses silently; a
        conflicting one is logged with the discarded value's length.
        """
        arguments = {}
        seen_raw: Dict[str, str] = {}
        for arg_key, arg_value in pairs:
            arg_key = arg_key.strip()
            arg_value = arg_value.strip()
            if arg_key in seen_raw:
                if arg_value != seen_raw[arg_key]:
                    logger.debug(
                        f"Duplicate tool argument key {arg_key!r}: keeping the first "
                        f"occurrence, discarding a conflicting later value of "
                        f"{len(arg_value)} chars"
                    )
                continue
            seen_raw[arg_key] = arg_value
            arg_type = get_argument_type(func_name, arg_key, tools)
            if arg_type is None or arg_type == "string":
                arguments[arg_key] = arg_value
            else:
                parsed_value, is_good_json = parse_arguments(arg_value, arg_type)
                arguments[arg_key] = parsed_value if is_good_json else arg_value

        return arguments

    def supports_structural_tag(self) -> bool:
        return False

    def structure_info(self) -> _GetInfoFunc:
        raise NotImplementedError()
