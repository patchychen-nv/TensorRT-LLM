# Adapted from https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/function_call/glm47_moe_detector.py
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
import re
from collections import deque
from typing import Dict, List, Optional

from tensorrt_llm.logger import logger
from tensorrt_llm.serve.openai_protocol import ChatCompletionToolsParam as Tool
from tensorrt_llm.serve.tool_parser.base_tool_parser import BaseToolParser
from tensorrt_llm.serve.tool_parser.core_types import (
    StreamingParseResult,
    ToolCallItem,
    _GetInfoFunc,
)
from tensorrt_llm.serve.tool_parser.glm4_parser import (
    StreamState,
    classify_name_region,
    classify_pending_close,
    get_argument_type,
    parse_arguments,
    split_dead_close_buffer,
)


class Glm47ToolParser(BaseToolParser):
    r"""Tool parser for GLM-4.7 and GLM-5 models.

    GLM-4.7 uses a slightly different tool call format compared to GLM-4.5:
      - The function name may appear on the same line as ``<tool_call>`` without
        a newline separator before the first ``<arg_key>``.
      - Tool calls may have zero arguments
        (e.g. ``<tool_call>func</tool_call>``).

    Example format::

        <tool_call>get_weather<arg_key>city</arg_key><arg_value>Beijing</arg_value>
        <arg_key>date</arg_key><arg_value>2024-06-27</arg_value></tool_call>

    Or zero-argument::

        <tool_call>get_time</tool_call>
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
        self.func_call_regex = re.compile(r"<tool_call>.*?</tool_call>", re.DOTALL)
        # The name group is `([^<]*)`: a tool name never contains `<`. With
        # `(.*?)` instead, the group's only terminators were `<arg_key>` and
        # `</tool_call>`, so any malformed tag sequence after the name was
        # swallowed INTO it and delivered to the client verbatim - recorded
        # GLM-5.3 turns arrived named `exec<tool_call>exec` (a doubled opener)
        # and `exec<arg_value>input</arg_key>...` (tags out of order). Group 2
        # now catches whatever sits between the name and the first structural
        # token, and classify_name_region decides the call's fate from it:
        # empty is the well-formed path, a fresh `<tool_call>` is a restart,
        # strippable markup is repaired onto the declared tool, anything else
        # marks the call malformed and releases its text.
        self.func_detail_regex = re.compile(
            r"<tool_call>([^<]*)((?:(?!<arg_key>|</tool_call>).)*)(<arg_key>.*?)?</tool_call>",
            re.DOTALL,
        )
        # The value ends at a `</arg_value>` the structure confirms: one
        # followed (over separators) by the next `<arg_key>`, the end of the
        # call, or the end of the argument text. Stopping at the first
        # `</arg_value>` substring truncated any value quoting the tag - code
        # payloads do - at that quote. A value containing a full well-formed
        # `</arg_value><arg_key>` sequence remains genuinely ambiguous and
        # reads as structure; classify_pending_close mirrors this rule for
        # the streaming path.
        self.func_arg_regex = re.compile(
            r"<arg_key>(.*?)</arg_key>(?:\\n|\s)*<arg_value>(.*?)</arg_value>"
            r"(?=(?:\\n|\s)*(?:<arg_key>|</tool_call>|$))",
            re.DOTALL,
        )
        # Mirror of func_detail_regex for the streaming path: group 1 is the
        # name (never holding `<`), group 2 the misplaced markup between it
        # and the arguments. Group 2 may still be growing while neither
        # group 3 nor group 4 has matched, which is why streaming acts on its
        # verdict only once a structural token seals it (restarts excepted;
        # see classify_name_region).
        self._partial_stream_regex = re.compile(
            r"<tool_call>([^<]*)((?:(?!<arg_key>|</tool_call>).)*)"
            r"(?:(<arg_key.*?))?(?:(</tool_call>)|$)",
            re.DOTALL,
        )
        self._last_arguments = ""
        self.current_tool_id = -1
        self.current_tool_name_sent = False
        self._streamed_raw_length = 0
        # Tool indices of calls the stream announced and then abandoned: the
        # model opened a fresh `<tool_call>` inside their arguments without
        # closing them. Their markup is released as text; the serving layer
        # must not deliver them (see `_restart_in_arguments`).
        self.abandoned_tool_indices: set[int] = set()
        self._reset_streaming_state()

    def _reset_streaming_state(self) -> None:
        self._stream_state = StreamState.INIT
        self._current_key = ""
        self._current_value = ""
        self._xml_tag_buffer = ""
        self._is_first_param = True
        # Key -> raw value text of every pair already put on the stream for
        # this call, so a repeated key can be recognized before any of its
        # JSON is emitted. See the duplicate handling in `_commit_pending_value`.
        self._streamed_pairs: Dict[str, str] = {}
        self._suppressing_duplicate = False

    @property
    def single_call_stop(self) -> Optional[str]:
        # Every call is its own <tool_call>...</tool_call> block with nothing
        # around it, so the closing tag completes exactly one call.
        return self.eot_token

    def has_tool_call(self, text: str) -> bool:
        return self.bot_token in text

    def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult:
        if self.bot_token not in text:
            return StreamingParseResult(normal_text=text, calls=[])

        normal_text_parts = []
        last_end = 0
        calls = []
        try:
            for match in self.func_call_regex.finditer(text):
                if match.start() > last_end:
                    normal_text_parts.append(text[last_end : match.start()])
                last_end = match.end()
                segment_calls, released = self._parse_call_segment(match.group(0), tools)
                calls.extend(segment_calls)
                # A segment (or a dead prefix of one) that the name rule
                # rejected is still model output: it joins the visible text
                # in document order rather than vanishing - the same
                # disposition _flush_tool_parser takes for unterminated
                # markup.
                if released:
                    normal_text_parts.append(released)

            if last_end < len(text):
                normal_text_parts.append(text[last_end:])

            normal_text = "".join(normal_text_parts).strip()
            return StreamingParseResult(normal_text=normal_text, calls=calls)
        except Exception as e:
            logger.error(f"Error in detect_and_parse: {e}")
            return StreamingParseResult(normal_text=text)

    def _parse_call_segment(
        self, segment: str, tools: List[Tool]
    ) -> tuple[list[ToolCallItem], str]:
        """One sliced ``<tool_call>...</tool_call>`` -> (calls, text to release).

        The name region decides the segment's fate (see classify_name_region):
        a clean or resolver-repairable name delivers exactly as before; a
        restarted call re-anchors on the fresh opener with the abandoned
        prefix released as text; unrepairable junk releases the whole segment
        as text, because a name carrying markup must never reach a client -
        two recorded GLM-5.3 turns were lost to exactly that. A fresh opener
        inside the arguments is a restart too (see `_restart_in_arguments`),
        whatever the abandoned call's name region said.
        """
        released_parts: list[str] = []
        while True:
            func_detail = self.func_detail_regex.search(segment)
            if func_detail is None:
                # Structurally unreachable - the segment is bracketed by the
                # call tokens and every group between them can be empty - but
                # a dropped byte is never acceptable, so an unreadable
                # segment would be released whole.
                released_parts.append(segment)
                return [], "".join(released_parts)
            func_name = func_detail.group(1).strip() if func_detail.group(1) else ""
            junk = func_detail.group(2) or ""
            verdict, recovered = classify_name_region(
                func_name, junk, self._get_tool_indices(tools)
            )
            if verdict == "restart":
                # The model abandoned the call it had opened and started over.
                # The markup before the fresh opener can never parse - its name
                # region is already junk-terminated - so it is released as text
                # and parsing re-anchors on the call the model finished. The
                # re-anchored region holds no further opener in its name
                # (rfind took the last one).
                cut = func_detail.start(2) + recovered
            else:
                restart_at = self._restart_in_arguments(func_detail.group(3))
                if restart_at is None:
                    break
                cut = func_detail.start(3) + restart_at
                logger.warning(
                    f"Model opened a new tool call inside the arguments of {func_name!r} "
                    f"without closing it; releasing the {cut}-char abandoned call as "
                    f"message text"
                )
            released_parts.append(segment[:cut])
            segment = segment[cut:]

        if verdict == "malformed":
            logger.warning(
                f"Tool call name region carries markup that maps onto no declared tool "
                f"(name {func_name!r}, {len(junk)} junk chars); releasing the "
                f"{len(segment)}-char call as message text instead of a corrupted name"
            )
            released_parts.append(segment)
            return [], "".join(released_parts)

        # "clean" or "repaired": today's delivery, byte for byte. On the
        # repaired path the raw region (name + junk) goes through so
        # parse_base_json performs - and logs - the same name recovery it
        # always has, and the schema lookup resolves it the same way.
        if junk:
            func_name = (func_name + junk).strip()
        arguments = {}
        func_args = func_detail.group(3)
        if func_args:
            pairs = self.func_arg_regex.findall(func_args)
            arguments = self._parse_argument_pairs(pairs, func_name, tools)
        segment_calls = self.parse_base_json({"name": func_name, "parameters": arguments}, tools)
        return segment_calls, "".join(released_parts)

    def _restart_in_arguments(self, func_args: Optional[str]) -> Optional[int]:
        """Offset of a fresh ``<tool_call>`` inside an open call's arguments.

        The model opened a new call before closing the one it was in. Read as
        argument text, the fresh call's whole markup became part of the open
        call's value, so the call the model actually finished arrived nested
        inside one it had abandoned - recorded GLM-5.3 turns delivered
        ``exec`` calls whose ``input`` ended ``...)localObject<tool_call>exec
        <arg_key>input</arg_key><arg_value>...``, and the client ran that.
        Like a restart in the name region, the abandoned call is released as
        text and parsing re-anchors on the fresh opener, whatever the
        abandoned call's name region said: a malformed one is released either
        way, and only the re-anchoring keeps the call that follows it.

        The first opener counts, because that is where the stream sees the
        open call end; a later one restarts the re-anchored call in turn. A
        value that merely quotes complete call markup was never readable
        here: the segment already ended at the quoted ``</tool_call>``.
        """
        if not func_args:
            return None
        restart_at = func_args.find(self.bot_token)
        return None if restart_at == -1 else restart_at

    def _encode_finished_value(
        self, key: str, value: str, func_name: str, tools: List[Tool]
    ) -> str:
        """The JSON text for one argument whose value has fully arrived.

        Emission has to wait for a *confirmed* ``</arg_value>`` - one the
        following text backs up as structure (see classify_pending_close). A
        value that is still arriving is not the value: bytes already sent
        cannot be recalled, and a ``</arg_value>`` the payload was merely
        quoting would otherwise cut the value short right there.

        Coercion is delegated to ``_parse_argument_pairs``, the same routine
        ``detect_and_parse`` uses, so a streamed argument cannot disagree with
        the same argument parsed whole - which is the only definition of correct
        here that does not require inventing a second rule. What that routine
        does with the value - schema-typed parse or verbatim passthrough - is
        documented there.
        """
        arguments = self._parse_argument_pairs([(key, value)], func_name, tools)
        return json.dumps(next(iter(arguments.values())), ensure_ascii=False)

    def _commit_pending_value(self, func_name: str, tools: List[Tool]) -> str:
        """Encode the value whose pending ``</arg_value>`` was just confirmed.

        A duplicate of a key already on the stream encodes to nothing: the
        first occurrence's bytes are already with the client and cannot be
        un-emitted, so the occurrence that reached the wire first wins - on
        this path and, identically, in `_parse_argument_pairs` - and the
        arguments JSON carries the key exactly once. An identical repeat
        collapses silently; a conflicting one is logged with the discarded
        value's length.
        """
        if self._suppressing_duplicate:
            first_value = self._streamed_pairs[self._current_key]
            if self._current_value != first_value:
                logger.debug(
                    f"Duplicate tool argument key {self._current_key!r}: keeping the "
                    f"value already streamed, discarding a conflicting later value "
                    f"of {len(self._current_value)} chars"
                )
            fragment = ""
        else:
            self._streamed_pairs[self._current_key] = self._current_value
            fragment = self._encode_finished_value(
                self._current_key, self._current_value, func_name, tools
            )
        self._suppressing_duplicate = False
        self._current_value = ""
        self._xml_tag_buffer = ""
        self._stream_state = StreamState.BETWEEN
        return fragment

    def _process_xml_to_json_streaming(
        self, raw_increment: str, func_name: str, tools: List[Tool]
    ) -> str:
        """Convert XML increment to JSON streaming output using state machine."""
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
                    if self._current_key in self._streamed_pairs:
                        # Repeated key. Its first occurrence is already on
                        # the wire; this pair is swallowed whole and resolved
                        # at `_commit_pending_value`.
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

            elif self._stream_state == StreamState.IN_VALUE:
                if self._xml_tag_buffer.endswith("</arg_value>"):
                    # Whatever the tag buffer was still holding back is value,
                    # minus the closing tag itself. The tag alone does not end
                    # the value, though: a payload may be quoting it. Withhold
                    # the value until the following text confirms the close as
                    # structure - the mirror of func_arg_regex's lookahead.
                    self._current_value += self._xml_tag_buffer[:-12]
                    self._xml_tag_buffer = ""
                    self._stream_state = StreamState.PENDING_CLOSE
                else:
                    # `</arg_value>` can straddle deltas, so text that is still
                    # a prefix of it stays in the tag buffer until the next
                    # character settles which it is. Only text that can no
                    # longer grow into the closing tag joins the value, which
                    # keeps the buffer bounded by the tag's length and lets a
                    # value that merely looks like the tag through unharmed.
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
                        self._current_value += released

            elif self._stream_state == StreamState.PENDING_CLOSE:
                verdict = classify_pending_close(self._xml_tag_buffer)
                if verdict == "key":
                    json_output += self._commit_pending_value(func_name, tools)
                    # The lookahead consumed the whole `<arg_key>`, so take
                    # the BETWEEN -> IN_KEY transition here as well.
                    self._stream_state = StreamState.IN_KEY
                    self._current_key = ""
                    self._xml_tag_buffer = ""
                elif verdict == "content":
                    # The tag was value text after all: keep it as content
                    # and replay the looked-ahead characters through IN_VALUE.
                    self._current_value += "</arg_value>"
                    replay = self._xml_tag_buffer
                    self._xml_tag_buffer = ""
                    self._stream_state = StreamState.IN_VALUE
                    pending_chars.extendleft(reversed(replay))
                # else: still pending - keep accumulating lookahead.

        return json_output

    def _extract_match_groups(self, match: re.Match) -> tuple[str, str, str, str]:
        func_name = match.group(1).strip()
        junk = match.group(2) or ""
        func_args_raw = match.group(3).strip() if match.group(3) else ""
        is_tool_end = match.group(4) or ""
        return func_name, junk, func_args_raw, is_tool_end

    def _send_tool_name_if_needed(
        self, func_name: str, has_arg_key: bool, is_tool_end: str
    ) -> Optional[ToolCallItem]:
        if self.current_tool_name_sent:
            return None

        is_func_name_complete = has_arg_key or is_tool_end == self.eot_token
        if not is_func_name_complete:
            return None

        if not func_name:
            logger.warning("Empty function name detected, skipping tool call")
            return None

        # The name arrives markup-free: classify_name_region already settled
        # restarts, repairs and malformed regions before this point. What is
        # left to resolve is the qualifier/bare-spelling mismatch (`exec` for
        # a declared `functions.exec`, and the reverse), matching the repair
        # parse_base_json applies on the non-streaming path. A name that maps
        # onto nothing declared is still forwarded unchanged - a fallback
        # that is only safe because the name can no longer carry `<`.
        func_name = (
            self.resolve_tool_name(func_name, getattr(self, "_tool_indices", {})) or func_name
        )

        self.current_tool_name_sent = True
        self._streamed_raw_length = 0
        self._reset_streaming_state()

        self.prev_tool_call_arr[self.current_tool_id] = {
            "name": func_name,
            "arguments": {},
        }

        return ToolCallItem(
            tool_index=self.current_tool_id,
            name=func_name,
            parameters="",
        )

    def _process_arguments_streaming(
        self, func_name: str, func_args_raw: str, tools: List[Tool]
    ) -> Optional[ToolCallItem]:
        current_raw_length = len(func_args_raw)

        if current_raw_length <= self._streamed_raw_length:
            return None

        raw_increment = func_args_raw[self._streamed_raw_length :]

        json_increment = self._process_xml_to_json_streaming(raw_increment, func_name, tools)

        self._streamed_raw_length = current_raw_length

        if not json_increment:
            return None

        self._last_arguments += json_increment
        self.streamed_args_for_tool[self.current_tool_id] += json_increment

        return ToolCallItem(
            tool_index=self.current_tool_id,
            name=None,
            parameters=json_increment,
        )

    def _finalize_tool_call(
        self,
        func_name: str,
        func_args_raw: str,
        tools: List[Tool],
        match_end_pos: int,
        current_text: str,
    ) -> List[ToolCallItem]:
        calls = []
        if self._stream_state == StreamState.PENDING_CLOSE:
            # End-of-call is the structural confirmation the pending
            # `</arg_value>` was waiting for. The lookahead buffer holds only
            # separators or the head of `</tool_call>` itself, never value
            # text, so it is dropped with the commit.
            flushed = self._commit_pending_value(func_name, tools)
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
        # `_is_first_param` is the only thing that says whether an opening `{`
        # was ever emitted, so it is the only thing that can say how to close.
        # Sniffing the last fragment for a trailing `}` instead - as this did -
        # mistakes an argument whose value is an object for the object being
        # closed already, and swallows the brace the call itself needs:
        # `{"opts": {"a": 1}` reaches the client one `}` short.
        closing = "{}" if self._is_first_param else "}"
        calls.append(
            ToolCallItem(
                tool_index=self.current_tool_id,
                name=None,
                parameters=closing,
            )
        )
        self._last_arguments += closing
        self.streamed_args_for_tool[self.current_tool_id] += closing

        if func_args_raw:
            try:
                pairs = self.func_arg_regex.findall(func_args_raw)
                if pairs:
                    arguments = self._parse_argument_pairs(pairs, func_name, tools)
                    self.prev_tool_call_arr[self.current_tool_id]["arguments"] = arguments
            except Exception as e:
                logger.debug(f"Failed to parse arguments: {e}")

        self._buffer = current_text[match_end_pos:]

        self.current_tool_id += 1
        self._last_arguments = ""
        self.current_tool_name_sent = False
        self._streamed_raw_length = 0
        self._reset_streaming_state()

        return calls

    def parse_streaming_increment(self, new_text: str, tools: List[Tool]) -> StreamingParseResult:
        """Streaming incremental parsing for GLM-4.7 format.

        Uses a state machine to convert XML to JSON incrementally for
        true character-by-character streaming.
        """
        self._buffer += new_text
        current_text = self._buffer

        bot_idx = current_text.find(self.bot_token)

        if bot_idx == -1:
            tail_len = min(len(current_text), len(self.bot_token) - 1)
            is_potential_start = tail_len > 0 and any(
                self.bot_token.startswith(current_text[-i:]) for i in range(1, tail_len + 1)
            )

            if not is_potential_start:
                # A `</tool_call>` with no opener anywhere is prose, and goes
                # out byte-identical: detect_and_parse keeps it, and the two
                # views of one generation must agree. Stripping the eot token
                # here (as the ported code did) silently deleted it from the
                # stream - 612 recorded responses in one production week lost
                # exactly that tag from ordinary text while their final
                # snapshots kept it. No close tag the parser owns can reach
                # this branch: a parsed call's `</tool_call>` is consumed when
                # `_finalize_tool_call` (or the malformed-call release)
                # re-anchors the buffer past it. Nor can chunking tear the
                # tag into a half-deleted state: its only shared prefix with
                # the bot token is `<`, which the potential-start hold above
                # already covers, and any longer fragment is released verbatim
                # here and completed verbatim by the next increment.
                self._buffer = ""
                return StreamingParseResult(normal_text=current_text)
            return StreamingParseResult(normal_text="", calls=[])

        normal_text = ""
        if bot_idx > 0:
            normal_text = current_text[:bot_idx]
            current_text = current_text[bot_idx:]
            self._buffer = current_text

        if not hasattr(self, "_tool_indices"):
            self._tool_indices = self._get_tool_indices(tools)

        calls: list[ToolCallItem] = []
        try:
            partial_match = self._partial_stream_regex.search(current_text)

            if not partial_match:
                return StreamingParseResult(normal_text=normal_text, calls=[])

            func_name, junk, func_args_raw, is_tool_end = self._extract_match_groups(partial_match)

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

            has_arg_key = partial_match.group(3) is not None

            verdict, recovered = classify_name_region(func_name, junk, self._tool_indices)
            if verdict == "restart":
                # The model restarted its call. A complete `<tool_call>`
                # inside the junk region is fixed text, so this is final the
                # moment it appears - and it always appears before the name
                # could be sent, because the name gate needs a structural
                # token that would arrive later in the stream. Release the
                # abandoned markup as visible text and re-anchor the buffer
                # on the fresh opener - the same split detect_and_parse makes.
                cut = partial_match.start(2) + recovered
                abandoned = current_text[:cut]
                self._buffer = current_text[cut:]
                self._streamed_raw_length = 0
                self._reset_streaming_state()
                logger.debug(
                    f"Model restarted a tool call mid-name; releasing "
                    f"{len(abandoned)} chars of abandoned markup as message text"
                )
                return StreamingParseResult(normal_text=normal_text + abandoned, calls=calls)
            if junk and not (has_arg_key or is_tool_end == self.eot_token):
                # The junk region may still be growing (`</arg_val` reads as
                # malformed while its completion repairs), so every verdict
                # but a restart stays provisional until a structural token
                # seals the region. Hold, exactly as an incomplete name held
                # before.
                return StreamingParseResult(normal_text=normal_text, calls=calls)
            restart_at = self._restart_in_arguments(partial_match.group(3))
            if restart_at is not None:
                # A fresh `<tool_call>` inside the open call's arguments is
                # fixed text and final the moment it appears, as in the name
                # region, and the region before it is sealed by the
                # `<arg_key>` that started the arguments. The open call is
                # abandoned: its markup is released as visible text - the
                # same split detect_and_parse makes - and, if its name was
                # already announced, its index is recorded so the serving
                # layer does not deliver it. Its fragments are a JSON prefix
                # with no closing brace, and bytes already streamed cannot be
                # recalled; dropping the entity is the only consistent option.
                cut = partial_match.start(3) + restart_at
                abandoned = current_text[:cut]
                if self.current_tool_name_sent:
                    self.abandoned_tool_indices.add(self.current_tool_id)
                    self.current_tool_id += 1
                    self._last_arguments = ""
                    self.current_tool_name_sent = False
                self._buffer = current_text[cut:]
                self._streamed_raw_length = 0
                self._reset_streaming_state()
                logger.warning(
                    f"Model opened a new tool call inside the arguments of {func_name!r} "
                    f"without closing it; releasing the {len(abandoned)}-char abandoned "
                    f"call as message text"
                )
                return StreamingParseResult(normal_text=normal_text + abandoned, calls=calls)
            if verdict == "malformed":
                if is_tool_end == self.eot_token:
                    # The call is complete and unreadable. Release its whole
                    # text - matching what detect_and_parse does with the
                    # same bytes - rather than delivering a name with markup
                    # fused into it.
                    segment_end = partial_match.end()
                    segment = current_text[:segment_end]
                    self._buffer = current_text[segment_end:]
                    self._streamed_raw_length = 0
                    self._reset_streaming_state()
                    logger.warning(
                        f"Tool call name region carries markup that maps onto no "
                        f"declared tool (name {func_name!r}, {len(junk)} junk chars); "
                        f"releasing the {len(segment)}-char call as message text "
                        f"instead of a corrupted name"
                    )
                    return StreamingParseResult(normal_text=normal_text + segment, calls=calls)
                # Sealed by `<arg_key>` but the call has not closed: only the
                # close - or a fresh opener, handled above - tells the
                # segment's extent, so the release waits and nothing of the
                # call is streamed meanwhile.
                return StreamingParseResult(normal_text=normal_text, calls=calls)

            # "clean" sends the name through today's resolve-or-forward;
            # "repaired" sends the declared tool the resolver recovered
            # (resolve is idempotent on it). The argument machinery keeps
            # receiving the raw region so the schema lookup resolves the
            # same spelling the whole-text path sees.
            wire_name = recovered if verdict == "repaired" else func_name
            raw_region = (func_name + junk).strip() if junk else func_name

            tool_name_item = self._send_tool_name_if_needed(wire_name, has_arg_key, is_tool_end)
            if tool_name_item:
                calls.append(tool_name_item)

            if self.current_tool_name_sent:
                arg_item = self._process_arguments_streaming(raw_region, func_args_raw, tools)
                if arg_item:
                    calls.append(arg_item)

                if is_tool_end == self.eot_token:
                    finalize_calls = self._finalize_tool_call(
                        raw_region,
                        func_args_raw,
                        tools,
                        partial_match.end(),
                        current_text,
                    )
                    calls.extend(finalize_calls)
                    return StreamingParseResult(normal_text=normal_text, calls=calls)

        except Exception as e:
            logger.error(f"Error in parse_streaming_increment: {e}")
            return StreamingParseResult(normal_text=current_text)

        return StreamingParseResult(normal_text=normal_text, calls=calls)

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

    def _parse_argument_pairs(self, pairs, func_name: str, tools: List[Tool]) -> dict:
        """Parse argument key-value pairs, typed by the declared schema.

        Only a declared non-string type licenses a conversion. A string
        parameter - and any parameter no schema describes, such as those of a
        tool the request never declared - is delivered as the text between
        the markers, verbatim: no json.loads, no literal_eval, no str() of a
        parsed object. Guessing a type from the value's shape is what
        corrupted live traffic: a `cell_id` the schema calls a string reached
        the client as the integer 99797 and failed its type check, and a
        freeform code payload of `true` was parsed to Python True and
        str()-round-tripped into "True", which the client then executed as
        code. json.dumps at delivery re-quotes and re-escapes the raw text,
        so passthrough here is still valid JSON there.

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
