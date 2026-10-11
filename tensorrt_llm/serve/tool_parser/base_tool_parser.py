# Adapted from https://github.com/sgl-project/sglang/blob/083629c23564e1a64deaa052f1df5c5d914358d8/python/sglang/srt/function_call/base_format_detector.py
import json
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

from partial_json_parser.core.exceptions import MalformedJSON
from partial_json_parser.core.options import Allow

from tensorrt_llm.logger import logger

from ..openai_protocol import ChatCompletionToolsParam as Tool
from .core_types import StreamingParseResult, ToolCallItem, _GetInfoFunc
from .utils import find_common_prefix, is_complete_json, partial_json_loads

# A tool name, optionally qualified by the groups it was declared under:
# `apply_patch`, `functions.apply_patch`, `functions.collaboration.send_message`.
_QUALIFIED_NAME = re.compile(
    r"^[A-Za-z_][A-Za-z0-9_-]*(?:\.[A-Za-z_][A-Za-z0-9_-]*)*$")

# An opening or closing tag of the XML-ish tool-call markup, e.g. `<tool_call>`
# or `</arg_value>`, which models sometimes emit unbalanced into a tool name.
_MARKUP_TAG = re.compile(r"</?[A-Za-z_][A-Za-z0-9_-]*>")


class BaseToolParser(ABC):
    """Base class providing two sets of interfaces: one-time and streaming incremental."""

    needs_raw_special_tokens: bool = False
    # Parsers whose forced/named ``tool_choice`` output still carries the
    # model's native tool-call markup (rather than grammar-constrained bare
    # JSON arguments) set this to True so the serving layer runs extraction
    # on the named-choice path instead of passing raw text through.
    extracts_forced_tool_calls: bool = False

    def __init__(self):
        # Streaming state management
        # Buffer for accumulating incomplete patterns that arrive across multiple streaming chunks
        self._buffer = ""
        # Stores complete tool call info (name and arguments) for each tool being parsed.
        # Used by serving layer for completion handling when streaming ends.
        # Format: [{"name": str, "arguments": dict}, ...]
        self.prev_tool_call_arr: List[Dict] = []
        # Index of currently streaming tool call. Starts at -1 (no active tool),
        # increments as each tool completes. Tracks which tool's arguments are streaming.
        self.current_tool_id: int = -1
        # Flag for whether current tool's name has been sent to client.
        # Tool names sent first with empty parameters, then arguments stream incrementally.
        self.current_tool_name_sent: bool = False
        # Tracks raw JSON string content streamed to client for each tool's arguments.
        # Critical for serving layer to calculate remaining content when streaming ends.
        # Each index corresponds to a tool_id. Example: ['{"location": "San Francisco"', '{"temp": 72']
        self.streamed_args_for_tool: List[str] = []

        # Token configuration (override in subclasses)
        self.bot_token = ""  # nosec B105
        self.eot_token = ""  # nosec B105
        self.tool_call_separator = ", "

    def _get_tool_indices(self, tools: List[Tool]) -> Dict[str, int]:
        """
        Get a mapping of tool names to their indices in the tools list.

        This utility method creates a dictionary mapping function names to their
        indices in the tools list, which is commonly needed for tool validation
        and ToolCallItem creation.

        Args:
            tools: List of available tools

        Returns:
            Dictionary mapping tool names to their indices
        """
        return {
            tool.function.name: i
            for i, tool in enumerate(tools) if tool.function.name
        }

    @staticmethod
    def resolve_tool_name(name: Optional[str],
                          tool_indices: Dict[str, int]) -> Optional[str]:
        """Best-effort mapping of an emitted name onto a declared tool.

        Returns None when nothing declared matches. That is not a verdict on
        what the caller should do with the call -- each parser keeps its own
        policy for unmatched names -- only a statement that no recovery was
        possible.

        Three shapes are recovered, all observed in production traces:

        * A group qualifier: models prompted with nested tool groups emit
          ``functions.exec_command`` for a tool declared as ``exec_command``.
        * The same mismatch the other way round, which is the common one for
          a Codex-style client: the tools arrive inside a ``namespace`` and are
          declared ``functions.exec``, while the model writes ``exec``.
          Measured over one GLM-5.3 fleet, 3102 of 3730 calls used the bare
          spelling. Only an unambiguous tail is accepted.
        * Stray markup fused onto the name: ``apply_patch</arg_value>`` and
          ``<tool_call>exec_command`` both come from the model emitting an
          unbalanced tag, which shifts where the name regex starts or stops.

        All leave the arguments intact, so recovering the name turns a call
        the client would have rejected back into the one the model meant.

        What is deliberately *not* recovered is a name that only *contains* a
        declared tool somewhere inside prose. Taking the text after the last
        ``<tool_call>`` would map "ops_check - re-querying for current status,
        since my last report..." onto ``exec_command``, fabricating a call the
        model never made. Requiring the whole remainder to be a bare dotted
        identifier is what keeps narration out.
        """
        if not name:
            return None
        if name in tool_indices:
            return name

        # Declared tools indexed by their bare tail, so that a model writing
        # `exec` against a declared `functions.exec` can be matched. Only an
        # unambiguous tail qualifies: two namespaces may declare the same tool
        # name, and mis-routing a call is worse than not recovering it.
        # responses_utils._tool_resolution applies the same rule one layer up.
        by_tail: Dict[str, List[str]] = {}
        for declared in tool_indices:
            by_tail.setdefault(declared.rsplit(".", 1)[-1], []).append(declared)

        # The raw name first, then the same name with balanced markup removed.
        # Anything still holding a space, quote or bracket after that is prose
        # or source code, not a mangled identifier, and _QUALIFIED_NAME drops it.
        for candidate in (name, _MARKUP_TAG.sub("", name).strip()):
            if not _QUALIFIED_NAME.match(candidate):
                continue
            # The whole candidate before its tail: stripping the markup off
            # `functions.exec </arg_value>` leaves a name that is declared
            # verbatim, and testing only the tail used to miss it.
            if candidate in tool_indices:
                return candidate
            tail = candidate.rsplit(".", 1)[-1]
            if tail in tool_indices:
                return tail
            qualified = by_tail.get(tail)
            if qualified is not None and len(qualified) == 1:
                return qualified[0]
        return None

    def parse_base_json(self, action: Any,
                        tools: List[Tool]) -> List[ToolCallItem]:
        tool_indices = self._get_tool_indices(tools)
        if not isinstance(action, list):
            action = [action]

        results = []
        for act in action:
            name = act.get("name")
            if name:
                if name not in tool_indices:
                    # Recovery only rewrites the name when it maps onto a tool
                    # the caller declared. An unmatched name is still forwarded
                    # with tool_index=-1, which is this class's long-standing
                    # contract: the caller decides whether to reject it.
                    recovered = self.resolve_tool_name(name, tool_indices)
                    if recovered is not None:
                        logger.warning(
                            f"Recovered malformed tool name {name!r} as "
                            f"{recovered!r}")
                        name = recovered
                    else:
                        logger.warning(
                            f"Model attempted to call undefined function: {name}"
                        )
                results.append(
                    ToolCallItem(
                        tool_index=
                        -1,  # Caller should update this based on the actual tools array called
                        name=name,
                        parameters=json.dumps(
                            act.get("parameters") or act.get("arguments") or {},
                            ensure_ascii=False,
                        ),
                    ))

        return results

    @abstractmethod
    def detect_and_parse(self, text: str,
                         tools: List[Tool]) -> StreamingParseResult:
        """
        Parses the text in one go. Returns success=True if the format matches, otherwise False.
        Note that leftover_text here represents "content that this parser will not consume further".
        """
        action = json.loads(text)
        return StreamingParseResult(calls=self.parse_base_json(action, tools))

    def _ends_with_partial_token(self, buffer: str, bot_token: str) -> int:
        """
        Check if buffer ends with a partial bot_token.
        Return the length of the partial bot_token.

        For some format, the bot_token is not a token in model's vocabulary, such as
        `[TOOL_CALLS] [` in Mistral.
        """
        for i in range(1, min(len(buffer) + 1, len(bot_token))):
            if bot_token.startswith(buffer[-i:]):
                return i
        return 0

    def finish(self, tools: List[Tool]) -> StreamingParseResult:
        """Finalize a stream that ended before the format's closing markers.

        Called once by the serving layer when generation finishes, after the
        last ``parse_streaming_increment`` call, so parsers that buffer whole
        sections can emit whatever the truncated stream still holds. The
        default is a no-op to preserve the existing behavior of parsers that
        manage ``self._buffer`` incrementally.
        """
        return StreamingParseResult()

    def parse_streaming_increment(self, new_text: str,
                                  tools: List[Tool]) -> StreamingParseResult:
        """
        Streaming incremental parsing with tool validation.

        This base implementation works best with formats where:
        1. bot_token is followed immediately by JSON (e.g., bot_token + JSON_array)
        2. JSON can be parsed incrementally using partial_json_loads
        3. Multiple tool calls are separated by "; " or ", "

        Examples of incompatible formats (need custom implementation, may reuse some logic from this class):
        - Each tool call is wrapped in a separate block: See Qwen25Detector
        - Multiple separate blocks: [TOOL_CALLS] [...] \n [TOOL_CALLS] [...]
        - Tool call is Pythonic style

        For incompatible formats, detectors should override this method with custom logic.
        """
        # Append new text to buffer
        self._buffer += new_text
        current_text = self._buffer

        # The current_text has tool_call if it is the start of a new tool call sequence
        # or it is the start of a new tool call after a tool call separator, when there is a previous tool call
        if not (self.has_tool_call(current_text) or
                (self.current_tool_id > 0
                 and current_text.startswith(self.tool_call_separator))):
            # Only clear buffer if we're sure no tool call is starting
            if not self._ends_with_partial_token(self._buffer, self.bot_token):
                normal_text = self._buffer
                self._buffer = ""
                if self.eot_token in normal_text:
                    normal_text = normal_text.replace(self.eot_token, "")
                return StreamingParseResult(normal_text=normal_text)
            else:
                # Might be partial bot_token, keep buffering
                return StreamingParseResult()

        # Build tool indices if not already built
        if not hasattr(self, "_tool_indices"):
            self._tool_indices = self._get_tool_indices(tools)

        flags = Allow.ALL if self.current_tool_name_sent else Allow.ALL & ~Allow.STR

        try:
            try:
                tool_call_pos = current_text.find(self.bot_token)
                if tool_call_pos != -1:
                    start_idx = tool_call_pos + len(self.bot_token)
                elif self.current_tool_id > 0 and current_text.startswith(
                        self.tool_call_separator):
                    start_idx = len(self.tool_call_separator)
                else:
                    start_idx = 0

                if start_idx >= len(current_text):
                    return StreamingParseResult()

                (obj, end_idx) = partial_json_loads(current_text[start_idx:],
                                                    flags)

                is_current_complete = is_complete_json(
                    current_text[start_idx:start_idx + end_idx])

                # Handle parameters/arguments consistency
                # NOTE: we assume here that the obj is always partial of a single tool call
                if "parameters" in obj:
                    assert ("arguments" not in obj
                            ), "model generated both parameters and arguments"
                    obj["arguments"] = obj["parameters"]

                current_tool_call = obj

            except MalformedJSON:
                return StreamingParseResult()

            if not current_tool_call:
                return StreamingParseResult()

            # Case 1: Handle tool name streaming
            # This happens when we encounter a tool but haven't sent its name yet
            if not self.current_tool_name_sent:
                function_name = current_tool_call.get("name")

                if function_name:
                    # If this is a new tool (current_tool_id was -1), initialize it
                    if self.current_tool_id == -1:
                        self.current_tool_id = 0
                        self.streamed_args_for_tool.append("")
                    # If this is a subsequent tool, ensure streamed_args_for_tool is large enough
                    elif self.current_tool_id >= len(
                            self.streamed_args_for_tool):
                        while len(self.streamed_args_for_tool
                                  ) <= self.current_tool_id:
                            self.streamed_args_for_tool.append("")

                    # Send the tool name with empty parameters
                    res = StreamingParseResult(calls=[
                        ToolCallItem(
                            tool_index=self.current_tool_id,
                            name=function_name,
                            parameters="",
                        )
                    ], )
                    self.current_tool_name_sent = True
                else:
                    res = StreamingParseResult()

            # Case 2: Handle streaming arguments
            # This happens when we've already sent the tool name and now need to stream arguments incrementally
            else:
                cur_arguments = current_tool_call.get("arguments")
                res = StreamingParseResult()

                # A finished call may carry no "arguments" key at all, or an
                # explicit null. Both mean "no arguments" and are normalized to
                # {} so the call still completes, matching what parse_base_json
                # returns on the non-streaming path. While the JSON is still
                # partial a missing key only means "not streamed yet", so it is
                # left alone and the elif prev_arguments branch keeps handling it.
                if is_current_complete and cur_arguments is None:
                    cur_arguments = {}

                # An empty argument object is falsy but still has to be streamed and
                # completed, otherwise a zero-argument tool call never finishes and its
                # text stays in the buffer forever.
                if cur_arguments is not None:
                    # Calculate how much of the arguments we've already streamed
                    sent = len(
                        self.streamed_args_for_tool[self.current_tool_id])
                    cur_args_json = json.dumps(cur_arguments)
                    prev_arguments = None
                    if self.current_tool_id < len(self.prev_tool_call_arr):
                        prev_arguments = self.prev_tool_call_arr[
                            self.current_tool_id].get("arguments")

                    argument_diff = None

                    # If the current tool's JSON is complete, send all remaining arguments
                    if is_current_complete:
                        argument_diff = cur_args_json[sent:]
                        completing_tool_id = (
                            self.current_tool_id
                        )  # Save the ID of the tool that's completing

                        # Only remove the processed portion, keep unprocessed content
                        self._buffer = current_text[start_idx + end_idx:]

                        if self.current_tool_id < len(self.prev_tool_call_arr):
                            self.prev_tool_call_arr[
                                self.current_tool_id].clear()
                        self.current_tool_name_sent = False
                        self.streamed_args_for_tool[self.current_tool_id] = ""
                        self.current_tool_id += 1

                    # If the tool is still being parsed, send incremental changes
                    elif prev_arguments:
                        prev_args_json = json.dumps(prev_arguments)
                        if cur_args_json != prev_args_json:
                            prefix = find_common_prefix(prev_args_json,
                                                        cur_args_json)
                            argument_diff = prefix[sent:]

                    # Send the argument diff if there's something new
                    if argument_diff is not None:
                        # Use the correct tool_index: completing_tool_id for completed tools, current_tool_id for ongoing
                        tool_index_to_use = (completing_tool_id
                                             if is_current_complete else
                                             self.current_tool_id)
                        res = StreamingParseResult(calls=[
                            ToolCallItem(
                                tool_index=tool_index_to_use,
                                parameters=argument_diff,
                            )
                        ], )
                        if not is_current_complete:
                            self.streamed_args_for_tool[
                                self.current_tool_id] += argument_diff

            # Update prev_tool_call_arr with current state
            if self.current_tool_id >= 0:
                # Ensure prev_tool_call_arr is large enough
                while len(self.prev_tool_call_arr) <= self.current_tool_id:
                    self.prev_tool_call_arr.append({})
                self.prev_tool_call_arr[
                    self.current_tool_id] = current_tool_call

            return res

        except Exception as e:
            logger.error(f"Error in parse_streaming_increment: {e}")
            return StreamingParseResult()

    @abstractmethod
    def has_tool_call(self, text: str) -> bool:
        """
        Check if the given text contains function call markers specific to this format.
        """
        raise NotImplementedError()

    # The markup this format writes around and inside its calls. A delivered
    # call whose arguments still contain any of it was not cleanly separated
    # from the format; the request trace flags such calls.
    markup_tokens: Tuple[str, ...] = ()

    @property
    def single_call_stop(self) -> Optional[str]:
        """Text that completes exactly one tool call, if generation may end there.

        Serving honours ``parallel_tool_calls=false`` by stopping generation
        on this text and keeping it in the output, so a response carries at
        most one call and its output stays an exact prefix of what the model
        generated. None when the format has no such text, or when stopping on
        it would leave an enclosing block unclosed; such formats keep
        emitting every call the model writes.
        """
        return None

    def supports_structural_tag(self) -> bool:
        """Return True if this detector supports structural tag format."""
        return True

    def build_strict_structural_tag_format(
            self, tools: List[Tool]) -> Optional[Dict[str, Any]]:
        """Build a complete structural-tag format for strict-tool decoding.

        Override on parsers whose wire format cannot be expressed through
        the `structure_info` begin/end/trigger triples (e.g. kimi_k3's
        XTML call tags). Returns the xgrammar structural-tag format dict,
        or None to fall back to the `structure_info` path.
        """
        return None

    @abstractmethod
    def structure_info(self) -> _GetInfoFunc:
        """
        Return a function that creates StructureInfo for constrained generation.

        The returned function takes a tool name and returns a StructureInfo object
        containing the begin/end patterns and trigger tokens needed for constrained
        generation of function calls in this format.

        Returns:
            A function that takes a tool name (str) and returns StructureInfo
        """
        raise NotImplementedError()
