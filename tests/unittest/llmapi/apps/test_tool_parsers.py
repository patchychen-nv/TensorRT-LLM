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

import abc
import itertools
import json
import re
from typing import NamedTuple
from unittest.mock import Mock

import pytest

from tensorrt_llm.sampling_params import SamplingParams
from tensorrt_llm.serve import responses_utils
from tensorrt_llm.serve.openai_protocol import (ChatCompletionToolsParam,
                                                FunctionDefinition)
from tensorrt_llm.serve.postprocess_handlers import forced_tool_arguments_end
from tensorrt_llm.serve.responses_utils import (_abandoned_tool_indices,
                                                _accumulate_tool_call_fragments,
                                                _assembled_tool_calls,
                                                _flush_tool_parser)
from tensorrt_llm.serve.tool_parser.base_tool_parser import BaseToolParser
from tensorrt_llm.serve.tool_parser.core_types import (StreamingParseResult,
                                                       StructureInfo,
                                                       ToolCallItem)
from tensorrt_llm.serve.tool_parser.deepseekv3_parser import DeepSeekV3Parser
from tensorrt_llm.serve.tool_parser.deepseekv4_parser import DeepSeekV4Parser
from tensorrt_llm.serve.tool_parser.deepseekv31_parser import DeepSeekV31Parser
from tensorrt_llm.serve.tool_parser.deepseekv32_parser import DeepSeekV32Parser
from tensorrt_llm.serve.tool_parser.gemma4_parser import Gemma4ToolParser
from tensorrt_llm.serve.tool_parser.glm4_parser import Glm4ToolParser
from tensorrt_llm.serve.tool_parser.glm47_parser import Glm47ToolParser
from tensorrt_llm.serve.tool_parser.kimi_k2_tool_parser import KimiK2ToolParser
from tensorrt_llm.serve.tool_parser.kimi_k3_tool_parser import KimiK3ToolParser
from tensorrt_llm.serve.tool_parser.minimax_m2_parser import MiniMaxM2ToolParser
from tensorrt_llm.serve.tool_parser.poolside_v1_parser import \
    PoolsideV1ToolParser
from tensorrt_llm.serve.tool_parser.qwen3_coder_parser import \
    Qwen3CoderToolParser
from tensorrt_llm.serve.tool_parser.qwen3_tool_parser import Qwen3ToolParser
from tensorrt_llm.tokenizer.deepseek_v32.encoding import encode_messages

from tensorrt_llm.serve.tool_parser.gemma4_parser import (  # isort: skip
    BOT_TOKEN, CALL_PREFIX, EOT_TOKEN, STRING_DELIM, _extract_tool_calls,
    _find_matching_brace, _parse_gemma4_args, _parse_gemma4_array,
    _parse_gemma4_value,
)

pytestmark = pytest.mark.cpu_only


# Test fixtures for common tools
@pytest.fixture
def sample_tools():
    """Sample tools for testing."""
    return [
        ChatCompletionToolsParam(
            type="function",
            function=FunctionDefinition(
                name="get_weather",
                description="Get the current weather",
                parameters={
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state",
                        },
                        "unit": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                        },
                    },
                    "required": ["location"],
                },
            ),
        ),
        ChatCompletionToolsParam(
            type="function",
            function=FunctionDefinition(
                name="search_web",
                description="Search the web",
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query",
                        }
                    },
                    "required": ["query"],
                },
            ),
        ),
    ]


# Concrete implementation of BaseToolParser for testing
class ConcreteToolParser(BaseToolParser):
    """Concrete implementation of BaseToolParser for testing abstract methods."""

    def __init__(self):
        super().__init__()
        self.bot_token = "[TOOL_CALLS] "
        self.eot_token = "[/TOOL_CALLS]"

    def has_tool_call(self, text: str) -> bool:
        return self.bot_token in text

    def detect_and_parse(self, text: str, tools):
        # Placeholder to avoid NotImplementedError
        pass

    def structure_info(self):
        return lambda name: StructureInfo(
            begin=f'[TOOL_CALLS] {{"name":"{name}", "arguments":',
            end="}[/TOOL_CALLS]",
            trigger="[TOOL_CALLS]")


# ============================================================================
# BaseToolParser Tests
# ============================================================================


class TestBaseToolParser:
    """Test suite for BaseToolParser class."""

    def test_initialization(self):
        """Test that BaseToolParser initializes correctly."""
        parser = ConcreteToolParser()
        assert parser._buffer == ""
        assert parser.prev_tool_call_arr == []
        assert parser.current_tool_id == -1
        assert parser.current_tool_name_sent is False
        assert parser.streamed_args_for_tool == []

    def test_get_tool_indices(self, sample_tools):
        """Test _get_tool_indices correctly maps tool names to indices."""
        parser = ConcreteToolParser()
        indices = parser._get_tool_indices(sample_tools)

        assert len(indices) == len(sample_tools)
        assert indices["get_weather"] == 0
        assert indices["search_web"] == 1

    def test_get_tool_indices_empty(self):
        """Test _get_tool_indices with empty tools list."""
        parser = ConcreteToolParser()
        indices = parser._get_tool_indices([])
        assert indices == {}

    def test_parse_base_json_single_tool(self, sample_tools):
        """Test parse_base_json with a single tool call."""
        parser = ConcreteToolParser()
        action = {
            "name": "get_weather",
            "parameters": {
                "location": "San Francisco"
            }
        }

        results = parser.parse_base_json(action, sample_tools)

        assert len(results) == 1
        assert results[0].name == "get_weather"
        assert json.loads(results[0].parameters) == {
            "location": "San Francisco"
        }

    def test_parse_base_json_with_arguments_key(self, sample_tools):
        """Test parse_base_json handles 'arguments' key instead of 'parameters'."""
        parser = ConcreteToolParser()
        action = {"name": "search_web", "arguments": {"query": "TensorRT"}}

        results = parser.parse_base_json(action, sample_tools)

        assert len(results) == 1
        assert results[0].name == "search_web"
        assert json.loads(results[0].parameters) == {"query": "TensorRT"}

    def test_parse_base_json_multiple_tools(self, sample_tools):
        """Test parse_base_json with multiple tool calls."""
        parser = ConcreteToolParser()
        actions = [{
            "name": "get_weather",
            "parameters": {
                "location": "Boston"
            }
        }, {
            "name": "search_web",
            "arguments": {
                "query": "Python"
            }
        }]

        results = parser.parse_base_json(actions, sample_tools)

        assert len(results) == 2
        assert results[0].name == "get_weather"
        assert results[1].name == "search_web"

    def test_parse_base_json_undefined_function(self, sample_tools):
        """Test parse_base_json handles undefined function names gracefully."""
        parser = ConcreteToolParser()
        action = {"name": "undefined_function", "parameters": {}}

        results = parser.parse_base_json(action, sample_tools)

        # Should return the tool call with tool_index=-1 and log warning.
        assert len(results) == 1
        assert results[0].name == "undefined_function"
        assert results[0].tool_index == -1
        assert json.loads(results[0].parameters) == {}

    def test_parse_base_json_missing_parameters(self, sample_tools):
        """Test parse_base_json handles missing parameters."""
        parser = ConcreteToolParser()
        action = {"name": "get_weather"}

        results = parser.parse_base_json(action, sample_tools)

        assert len(results) == 1
        assert json.loads(results[0].parameters) == {}

    def test_parse_base_json_null_arguments(self, sample_tools):
        """Test parse_base_json handles an explicit null arguments value."""
        parser = ConcreteToolParser()
        action = {"name": "get_weather", "arguments": None}

        results = parser.parse_base_json(action, sample_tools)

        assert len(results) == 1
        assert json.loads(results[0].parameters) == {}

    def test_ends_with_partial_token(self):
        """Test _ends_with_partial_token detection."""
        parser = ConcreteToolParser()

        # Partial token at end (bot_token starts with the suffix)
        assert parser._ends_with_partial_token("Some text [TOOL",
                                               "[TOOL_CALLS] ") == 5
        assert parser._ends_with_partial_token("Some text [",
                                               "[TOOL_CALLS] ") == 1
        assert parser._ends_with_partial_token("Some text [TOOL_CALLS",
                                               "[TOOL_CALLS] ") == 11

        # No partial token
        assert parser._ends_with_partial_token("Some text",
                                               "[TOOL_CALLS] ") == 0
        assert parser._ends_with_partial_token("Some text [XYZ",
                                               "[TOOL_CALLS] ") == 0

        # Complete token at end (entire buffer is bot_token prefix but not complete match)
        # When buffer equals bot_token, it returns 0 because it's not a partial anymore
        assert parser._ends_with_partial_token("text [TOOL_CALLS] ",
                                               "[TOOL_CALLS] ") == 0

    def test_parse_streaming_increment_no_tool_call(self, sample_tools):
        """Test streaming parser returns normal text when no tool call present."""
        parser = ConcreteToolParser()

        result = parser.parse_streaming_increment("Hello, world!", sample_tools)

        assert result.normal_text == "Hello, world!"
        assert len(result.calls) == 0

    def test_parse_streaming_increment_partial_bot_token(self, sample_tools):
        """Test streaming parser buffers partial bot token."""
        parser = ConcreteToolParser()

        # Send partial bot token
        result = parser.parse_streaming_increment("[TOOL", sample_tools)

        # Should buffer and return nothing
        assert result.normal_text == ""
        assert len(result.calls) == 0
        assert parser._buffer == "[TOOL"

    def test_parse_streaming_increment_tool_name(self, sample_tools):
        """Test streaming parser handles tool name streaming."""
        parser = ConcreteToolParser()

        # Send bot token with partial JSON containing name
        result = parser.parse_streaming_increment(
            '[TOOL_CALLS] {"name":"get_weather"', sample_tools)

        # Should send tool name with empty parameters
        assert len(result.calls) == 1
        assert result.calls[0].name == "get_weather"
        assert result.calls[0].parameters == ""
        assert result.calls[0].tool_index == 0
        assert parser.current_tool_name_sent is True

    def test_parse_streaming_increment_tool_arguments(self, sample_tools):
        """Test streaming parser handles incremental argument streaming."""
        parser = ConcreteToolParser()

        # First send tool name
        result1 = parser.parse_streaming_increment(
            '[TOOL_CALLS] {"name":"get_weather"', sample_tools)
        # Should send tool name
        assert len(result1.calls) == 1
        assert result1.calls[0].name == "get_weather"

        # Then send complete arguments (parser needs complete JSON to parse incrementally)
        result2 = parser.parse_streaming_increment(
            ',"arguments":{"location":"San Francisco"}}', sample_tools)

        # Should stream arguments or complete the tool call
        # The base implementation uses partial JSON parsing, so it may return results
        assert result2 is not None  # Just verify it doesn't crash

    def test_parse_streaming_increment_complete_tool(self, sample_tools):
        """Test streaming parser handles complete tool call."""
        parser = ConcreteToolParser()

        # Send complete tool call in one chunk
        result = parser.parse_streaming_increment(
            '[TOOL_CALLS] {"name":"get_weather","arguments":{"location":"Boston"}}',
            sample_tools)

        # Should have sent tool name (first call)
        assert len(result.calls) == 1
        assert result.calls[0].name == "get_weather"

    def test_parse_streaming_increment_invalid_tool_name(self, sample_tools):
        """Test streaming parser handles invalid tool name."""
        parser = ConcreteToolParser()

        # Send invalid tool name - parser streams it through.
        result = parser.parse_streaming_increment(
            '[TOOL_CALLS] {"name":"invalid_tool"', sample_tools)

        # Should still return the tool call.
        assert len(result.calls) == 1
        assert result.calls[0].name == "invalid_tool"
        assert result.calls[0].tool_index == 0

    def test_supports_structural_tag(self):
        """Test supports_structural_tag returns True."""
        parser = ConcreteToolParser()
        assert parser.supports_structural_tag() is True

    def test_structure_info(self):
        """Test structure_info returns proper function."""
        parser = ConcreteToolParser()
        func = parser.structure_info()

        info = func("test_function")
        assert isinstance(info, StructureInfo)
        assert "test_function" in info.begin
        assert info.trigger == "[TOOL_CALLS]"

    def test_finish_default_noop(self, sample_tools):
        """The default finalization hook emits nothing and keeps state.

        Existing parsers manage ``self._buffer`` incrementally; the
        end-of-stream hook must not change their behavior.
        """
        parser = ConcreteToolParser()
        parser._buffer = '[TOOL_CALLS] {"name":"get_weather"'

        result = parser.finish(sample_tools)

        assert result.normal_text == ""
        assert result.calls == []
        assert parser._buffer == '[TOOL_CALLS] {"name":"get_weather"'

    def test_extracts_forced_tool_calls_default_false(self):
        """Forced tool_choice extraction is opt-in per parser."""
        assert ConcreteToolParser.extracts_forced_tool_calls is False


# ============================================================================
# Qwen3ToolParser Tests
# ============================================================================


class ToolParserTestCases(NamedTuple):
    has_tool_call_true: str
    detect_and_parse_single_tool: tuple[
        # Input text.
        str,
        # Expected `normal_text`.
        str,
        # Expected `name`.
        str,
        # Expected `parameters`.
        dict,
    ]
    detect_and_parse_multiple_tools: tuple[
        # Input text.
        str,
        # Expected names.
        tuple[str],
    ]
    detect_and_parse_malformed_tool: str
    detect_and_parse_with_parameters_key: tuple[
        # Input text.
        str,
        # Expected `name`.
        str,
        # Expected `parameters`.
        dict,
    ]
    parse_streaming_increment_partial_bot_token: str
    undefined_tool: str


class BaseToolParserTestClass:
    """Base class from which tests for actual implementations can be extended.

    NOTE: the name deliberately ends with `Class` so that `pytest` does not pick it up for execution
    automatically.
    """

    @abc.abstractmethod
    def make_parser(self):
        ...

    @property
    def make_tool_parser_test_cases(self) -> ToolParserTestCases:
        ...

    @pytest.fixture
    def parser(self):
        return self.make_parser()

    @pytest.fixture(scope="class")
    def tool_parser_test_cases(self) -> ToolParserTestCases:
        return self.make_tool_parser_test_cases()

    def test_has_tool_call_false(self, parser):
        """Test has_tool_call returns False when no tool call present."""
        text = "Just some regular text without tool calls"

        assert parser.has_tool_call(text) is False

    def test_has_tool_call_true(self, parser, tool_parser_test_cases):
        """Test has_tool_call returns True when tool call is present."""
        text = tool_parser_test_cases.has_tool_call_true

        assert parser.has_tool_call(text) is True

    def test_detect_and_parse_no_tool_call(self, sample_tools, parser):
        """Test detect_and_parse with text containing no tool calls."""
        text = "This is just a regular response."

        result = parser.detect_and_parse(text, sample_tools)

        assert result.normal_text == text
        assert len(result.calls) == 0

    def test_detect_and_parse_single_tool(self, sample_tools, parser,
                                          tool_parser_test_cases):
        """Test detect_and_parse with a single tool call."""
        text, normal_text, name, parameters = tool_parser_test_cases.detect_and_parse_single_tool

        result = parser.detect_and_parse(text, sample_tools)

        assert result.normal_text == normal_text
        assert len(result.calls) == 1
        assert result.calls[0].name == name
        assert json.loads(result.calls[0].parameters) == parameters

    def test_detect_and_parse_multiple_tools(self, sample_tools, parser,
                                             tool_parser_test_cases):
        """Test detect_and_parse with multiple tool calls."""
        text, call_names = tool_parser_test_cases.detect_and_parse_multiple_tools

        result = parser.detect_and_parse(text, sample_tools)

        assert tuple(call.name for call in result.calls) == call_names

    def test_detect_and_parse_malformed_tool(self, sample_tools, parser,
                                             tool_parser_test_cases):
        """Test detect_and_parse handles malformed tool call output from the model gracefully."""
        text = tool_parser_test_cases.detect_and_parse_malformed_tool

        result = parser.detect_and_parse(text, sample_tools)

        assert len(result.calls) == 0

    def test_detect_and_parse_with_parameters_key(self, sample_tools, parser,
                                                  tool_parser_test_cases):
        """Test detect_and_parse handles 'parameters' key."""
        text, name, parameters = tool_parser_test_cases.detect_and_parse_with_parameters_key

        result = parser.detect_and_parse(text, sample_tools)

        assert len(result.calls) == 1
        assert result.calls[0].name == name
        assert json.loads(result.calls[0].parameters) == parameters

    def test_parse_streaming_increment_normal_text(self, sample_tools, parser):
        """Test streaming parser handles normal text without tool calls."""
        text = "Hello, how can I help?"

        result = parser.parse_streaming_increment(text, sample_tools)

        assert result.normal_text == text
        assert len(result.calls) == 0

    def test_parse_streaming_increment_partial_bot_token(
            self, sample_tools, parser, tool_parser_test_cases):
        """Test streaming parser buffers partial bot token."""
        text = tool_parser_test_cases.parse_streaming_increment_partial_bot_token

        result = parser.parse_streaming_increment(text, sample_tools)

        assert result.normal_text == ""
        assert len(result.calls) == 0

    def test_undefined_tool(self, sample_tools, parser, tool_parser_test_cases):
        """Test the parser handles undefined tool gracefully."""
        text = tool_parser_test_cases.undefined_tool

        result = parser.detect_and_parse(text, sample_tools)

        # Should return the tool call with tool_index=-1.
        assert len(result.calls) == 1
        assert result.calls[0].tool_index == -1


class TestKimiK2ToolParser(BaseToolParserTestClass):
    """Test suite for KimiK2ToolParser class."""

    def make_parser(self):
        return KimiK2ToolParser()

    def make_tool_parser_test_cases(self):
        return ToolParserTestCases(
            has_tool_call_true=
            'Some text <|tool_calls_section_begin|><|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>{"location": "NYC"}<|tool_call_end|><|tool_calls_section_end|>',
            detect_and_parse_single_tool=(
                # Input text.
                ('Normal text'
                 '<|tool_calls_section_begin|><|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>{"location": "NYC"}<|tool_call_end|><|tool_calls_section_end|>'
                 ),
                # Expected `normal_text`.
                "Normal text",
                # Expected `name`.
                "get_weather",
                # Expected `parameters`.
                {
                    "location": "NYC"
                },
            ),
            detect_and_parse_multiple_tools=(
                # Input text.
                ('<|tool_calls_section_begin|><|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>{"location":"LA"}<|tool_call_end|>\n'
                 '<|tool_call_begin|>functions.search_web:0<|tool_call_argument_begin|>{"query":"AI"}<|tool_call_end|><|tool_calls_section_end|>'
                 ),
                # Expected names.
                ("get_weather", "search_web"),
            ),
            detect_and_parse_malformed_tool=
            ('<|tool_calls_section_begin|><|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>MALFORMED<|tool_call_end|><|tool_calls_section_end|>'
             ),
            detect_and_parse_with_parameters_key=(
                # Input text.
                ('<|tool_calls_section_begin|><|tool_call_begin|>functions.search_web:0<|tool_call_argument_begin|>{"query":"test"}<|tool_call_end|><|tool_calls_section_end|>'
                 ),
                # Expected `name`.
                "search_web",
                # Expected `parameters`.
                {
                    "query": "test"
                },
            ),
            parse_streaming_increment_partial_bot_token=
            "<|tool_calls_section_begin|><|tool_call_be",
            undefined_tool=
            '<|tool_calls_section_begin|><|tool_call_begin|>functions.undefined_func:0<|tool_call_argument_begin|>{"arg":"any value"}<|tool_call_end|><|tool_calls_section_end|>',
        )

    def test_initialization(self, parser):
        """Test that Qwen3ToolParser initializes correctly."""
        assert parser.bot_token == "<|tool_calls_section_begin|>"
        assert parser.eot_token == "<|tool_calls_section_end|>"

    def test_parse_streaming_increment_complete_tool_call(
            self, sample_tools, parser):
        """Test streaming parser with complete tool call in chunks."""

        # Send bot token
        parser.parse_streaming_increment("<|tool_calls_section_begin|>",
                                         sample_tools)

        # Send partial tool call with name
        result = parser.parse_streaming_increment(
            '<|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>{',
            sample_tools)

        # Should send tool name
        assert len(result.calls) == 1
        assert result.calls[0].name == "get_weather"
        assert result.calls[0].parameters == ""

        # Send arguments
        result = parser.parse_streaming_increment(
            '"location":"SF"}<|tool_call_end|>', sample_tools)

        # Should stream arguments
        assert len(result.calls) == 1
        assert json.loads(result.calls[0].parameters) == {"location": "SF"}

    def test_parse_streaming_increment_multiple_tools_streaming(
            self, sample_tools, parser):
        """Test streaming parser handles multiple tool calls."""

        # First tool
        parser.parse_streaming_increment('<|tool_calls_section_begin|>',
                                         sample_tools)
        parser.parse_streaming_increment(
            '<|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>{"location":"NYC"}<|tool_call_end|>',
            sample_tools)

        # Second tool
        parser.parse_streaming_increment(
            '<|tool_call_begin|>functions.search_web:0<|tool_call_argument_begin|>{"arg": "any value"}<|tool_call_end|>',
            sample_tools)

        result = parser.parse_streaming_increment('<|tool_calls_section_end|>',
                                                  sample_tools)
        # Should have started second tool
        assert result.calls[0].name == "search_web"
        assert result.calls[0].parameters == ""
        assert result.calls[0].tool_index == 1

    def test_structure_info_function(self):
        """Test structure_info returns correct lambda function."""
        parser = KimiK2ToolParser()
        func = parser.structure_info()

        info = func("test_function")

        assert isinstance(info, StructureInfo)
        assert info.begin == '<|tool_calls_section_begin|><|tool_call_begin|>functions.test_function:0<|tool_call_argument_begin|>'
        assert info.end == "<|tool_call_end|><|tool_calls_section_end|>"
        assert info.trigger == "<|tool_calls_section_begin|>"

    def test_structure_info_different_names(self):
        """Test structure_info works with different function names."""
        parser = KimiK2ToolParser()
        func = parser.structure_info()

        info1 = func("get_weather")
        info2 = func("search_web")

        assert "get_weather" in info1.begin
        assert "search_web" in info2.begin
        assert info1.end == info2.end == "<|tool_call_end|><|tool_calls_section_end|>"

    def test_undefined_tool(self, sample_tools, parser, tool_parser_test_cases):
        """KimiK2 has custom detect_and_parse that filters undefined tools."""
        text = tool_parser_test_cases.undefined_tool

        result = parser.detect_and_parse(text, sample_tools)

        assert len(result.calls) == 0

    def test_kimi_k2_format_compliance(self, sample_tools, parser):
        """Test that KimiK2ToolParser follows the documented format structure."""

        # Test the exact format from the docstring
        text = '<|tool_calls_section_begin|><|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>{"location":"Tokyo"}<|tool_call_end|><|tool_calls_section_end|>'

        result = parser.detect_and_parse(text, sample_tools)

        assert len(result.calls) == 1
        assert result.calls[0].name == "get_weather"
        assert json.loads(result.calls[0].parameters) == {"location": "Tokyo"}


class TestQwen3ToolParser(BaseToolParserTestClass):
    """Test suite for Qwen3ToolParser class."""

    def make_parser(self):
        return Qwen3ToolParser()

    def make_tool_parser_test_cases(self):
        return ToolParserTestCases(
            has_tool_call_true=
            'Some text <tool_call>\n{"name":"get_weather"}\n</tool_call>',
            detect_and_parse_single_tool=(
                # Input text.
                ('Normal text\n'
                 '<tool_call>\n'
                 '{"name":"get_weather","arguments":{"location":"NYC"}}\n'
                 '</tool_call>'),
                # Expected `normal_text`.
                "Normal text",
                # Expected `name`.
                "get_weather",
                # Expected `parameters`.
                {
                    "location": "NYC"
                },
            ),
            detect_and_parse_multiple_tools=(
                # Input text.
                ('<tool_call>\n{"name":"get_weather","arguments":{"location":"LA"}}\n</tool_call>\n'
                 '<tool_call>\n{"name":"search_web","arguments":{"query":"AI"}}\n</tool_call>'
                 ),
                # Expected names.
                ("get_weather", "search_web"),
            ),
            detect_and_parse_malformed_tool=
            ('<tool_call>\n{"name":"get_weather","arguments":MALFORMED}\n</tool_call>'
             ),
            detect_and_parse_with_parameters_key=(
                # Input text.
                ('<tool_call>\n{"name":"search_web","parameters":{"query":"test"}}\n</tool_call>'
                 ),
                # Expected `name`.
                "search_web",
                # Expected `parameters`.
                {
                    "query": "test"
                },
            ),
            parse_streaming_increment_partial_bot_token="<tool",
            undefined_tool=
            '<tool_call>\n{"name":"undefined_func","arguments":{}}\n</tool_call>',
        )

    def test_initialization(self, parser):
        """Test that Qwen3ToolParser initializes correctly."""
        assert parser.bot_token == "<tool_call>\n"
        assert parser.eot_token == "\n</tool_call>"
        assert parser.tool_call_separator == "\n"
        assert parser._normal_text_buffer == ""

    # NOTE: this is not put in the base class. Even though it could be made generic, the added logic
    # to do so loses the clarity of this more direct approach.
    def test_parse_streaming_increment_complete_tool_call(
            self, sample_tools, parser):
        """Test streaming parser with complete tool call in chunks."""

        # Send bot token
        parser.parse_streaming_increment("<tool_call>\n", sample_tools)

        # Send partial JSON with name
        result = parser.parse_streaming_increment('{"name":"get_weather"',
                                                  sample_tools)

        # Should send tool name
        assert len(result.calls) == 1
        assert result.calls[0].name == "get_weather"
        assert result.calls[0].parameters == ""

        # Send arguments
        result = parser.parse_streaming_increment(
            ',"arguments":{"location":"SF"}}\n</tool_call>', sample_tools)

        # Should stream arguments
        assert len(result.calls) == 1
        assert json.loads(result.calls[0].parameters) == {"location": "SF"}

    def test_parse_streaming_increment_end_token_handling(
            self, sample_tools, parser):
        """Test streaming parser handles end token correctly."""

        # Send complete tool call
        parser.parse_streaming_increment(
            '<tool_call>\n{"name":"get_weather","arguments":{"location":"NYC"}}\n</tool_call>',
            sample_tools)

        # The end token should be removed from normal text
        # Check buffer state
        assert parser._normal_text_buffer == ""

    def test_parse_streaming_increment_multiple_tools_streaming(
            self, sample_tools, parser):
        """Test streaming parser handles multiple tool calls."""

        # First tool
        parser.parse_streaming_increment('<tool_call>\n', sample_tools)
        parser.parse_streaming_increment(
            '{"name":"get_weather","arguments":{"location":"NYC"}}\n</tool_call>\n',
            sample_tools)

        # Second tool
        parser.parse_streaming_increment('<tool_call>\n', sample_tools)
        result = parser.parse_streaming_increment('{"name":"search_web"',
                                                  sample_tools)

        # Should have started second tool
        assert result.calls[0].name == "search_web"
        assert result.calls[0].parameters == ""
        assert result.calls[0].tool_index == 1

    def test_structure_info_function(self):
        """Test structure_info returns correct lambda function."""
        parser = Qwen3ToolParser()
        func = parser.structure_info()

        info = func("test_function")

        assert isinstance(info, StructureInfo)
        assert info.begin == '<tool_call>\n{"name":"test_function", "arguments":'
        assert info.end == "}\n</tool_call>"
        assert info.trigger == "<tool_call>"

    def test_structure_info_different_names(self):
        """Test structure_info works with different function names."""
        parser = Qwen3ToolParser()
        func = parser.structure_info()

        info1 = func("get_weather")
        info2 = func("search_web")

        assert "get_weather" in info1.begin
        assert "search_web" in info2.begin
        assert info1.end == info2.end == "}\n</tool_call>"

    def test_qwen3_format_compliance(self, sample_tools, parser):
        """Test that Qwen3ToolParser follows the documented format structure."""

        # Test the exact format from the docstring
        text = '<tool_call>\n{"name":"get_weather", "arguments":{"location":"Tokyo"}}\n</tool_call>'

        result = parser.detect_and_parse(text, sample_tools)

        assert len(result.calls) == 1
        assert result.calls[0].name == "get_weather"
        assert json.loads(result.calls[0].parameters) == {"location": "Tokyo"}

    # ------------------------------------------------------------------
    # NVBug 6240584: bare-JSON fallback in detect_and_parse
    #
    # Some Qwen3 chat templates (notably Qwen3.6 FP8 with
    # `--reasoning_parser qwen3_5 --tool_parser qwen3`) emit tool calls
    # as bare JSON, without a `<tool_call>...</tool_call>` wrapper, once
    # the reasoning parser strips the `</think>` block. The parser must
    # recover those before dropping the text into `normal_text`.
    # ------------------------------------------------------------------

    def test_detect_and_parse_bare_json_dict(self, sample_tools, parser):
        """Bare JSON dict without <tool_call> wrapper is parsed as a tool call."""
        text = '{"name":"get_weather","arguments":{"location":"Paris"}}'

        result = parser.detect_and_parse(text, sample_tools)

        assert result.normal_text == ""
        assert len(result.calls) == 1
        assert result.calls[0].name == "get_weather"
        assert json.loads(result.calls[0].parameters) == {"location": "Paris"}

    def test_detect_and_parse_bare_json_list(self, sample_tools, parser):
        """Bare JSON list of tool calls without wrapper is parsed."""
        text = '[{"name":"get_weather","arguments":{"location":"Paris"}}]'

        result = parser.detect_and_parse(text, sample_tools)

        assert result.normal_text == ""
        assert len(result.calls) == 1
        assert result.calls[0].name == "get_weather"
        assert json.loads(result.calls[0].parameters) == {"location": "Paris"}

    def test_detect_and_parse_bare_json_parameters_key(self, sample_tools,
                                                       parser):
        """Bare JSON with `parameters` (instead of `arguments`) is still parsed."""
        text = '{"name":"get_weather","parameters":{"location":"Paris"}}'

        result = parser.detect_and_parse(text, sample_tools)

        assert len(result.calls) == 1
        assert result.calls[0].name == "get_weather"
        assert json.loads(result.calls[0].parameters) == {"location": "Paris"}

    def test_detect_and_parse_non_json_text_falls_through(
            self, sample_tools, parser):
        """Plain non-JSON text passes through as normal_text with no calls."""
        text = "Hello world"

        result = parser.detect_and_parse(text, sample_tools)

        assert result.normal_text == "Hello world"
        assert result.calls == []

    def test_detect_and_parse_bare_json_scalar_falls_through(
            self, sample_tools, parser):
        """A JSON scalar (e.g. `"42"`) must fall through cleanly, not crash.

        This exercises the explicit `isinstance(parsed, (dict, list))` guard —
        `parse_base_json` would raise `AttributeError` on a bare int, so the
        guard prevents relying on exception catching for scalar JSON.
        """
        text = "42"

        result = parser.detect_and_parse(text, sample_tools)

        assert result.calls == []
        # No crash is the important part.

    def test_detect_and_parse_malformed_bare_json_falls_through(
            self, sample_tools, parser):
        """Malformed JSON without <tool_call> wrapper falls through cleanly."""
        text = '{"name": "get_weather", "arguments": MALFORMED}'

        result = parser.detect_and_parse(text, sample_tools)

        assert result.calls == []
        assert result.normal_text == text

    def test_detect_and_parse_bare_json_with_trailing_content(
            self, sample_tools, parser):
        """Bare JSON followed by trailing non-whitespace text is still parsed.

        NVBug 6240584 review follow-up: `json.loads(text.strip())` raises
        `json.JSONDecodeError: Extra data` on `'{...} trailing text'`, which
        used to drop the valid tool call into `normal_text`. The parser now
        uses `raw_decode` to consume only the leading JSON value and must
        recover the tool call regardless of what follows.
        """
        text = ('{"name":"get_weather","arguments":{"city":"Paris"}}\n'
                'Extra text after the tool call.')

        result = parser.detect_and_parse(text, sample_tools)

        assert len(result.calls) == 1
        assert result.calls[0].name == "get_weather"
        assert json.loads(result.calls[0].parameters) == {"city": "Paris"}

    # ------------------------------------------------------------------
    # NVBug 6240584: bare-JSON fallback in parse_streaming_increment
    #
    # The streaming path must also recover bare-JSON tool calls when the
    # `<tool_call>` wrapper never appears. Without this, streaming clients
    # receive the JSON as `delta.content` with `finish_reason="stop"`.
    # ------------------------------------------------------------------

    def test_streaming_bare_json_one_chunk(self, sample_tools, parser):
        """A complete bare-JSON tool call arriving in a single chunk emits calls."""
        result = parser.parse_streaming_increment(
            '{"name":"get_weather","arguments":{"city":"Paris"}}', sample_tools)

        names = [c.name for c in result.calls if c.name]
        assert "get_weather" in names
        params = "".join(c.parameters for c in result.calls if c.parameters)
        assert "Paris" in params
        assert result.normal_text == ""

    def test_streaming_bare_json_split_across_chunks(self, sample_tools,
                                                     parser):
        """Bare-JSON tool call split across multiple chunks parses on completion."""
        r1 = parser.parse_streaming_increment('{"name":"get_', sample_tools)
        r2 = parser.parse_streaming_increment('weather","arguments":',
                                              sample_tools)
        r3 = parser.parse_streaming_increment('{"city":"Paris"}}', sample_tools)

        all_calls = list(r1.calls) + list(r2.calls) + list(r3.calls)
        names = [c.name for c in all_calls if c.name]
        assert "get_weather" in names
        params = "".join(c.parameters for c in all_calls if c.parameters)
        assert "Paris" in params

    def test_streaming_bare_json_does_not_leak_content(self, sample_tools,
                                                       parser):
        """After a bare-JSON tool call is emitted, trailing text is not leaked.

        This must be the case even for subsequent empty/whitespace chunks:
        leaking any normal_text would flip `finish_reason` back to `stop`.
        """
        r1 = parser.parse_streaming_increment(
            '{"name":"get_weather","arguments":{"city":"Paris"}}', sample_tools)
        # Any subsequent chunks must not emit normal_text either.
        r2 = parser.parse_streaming_increment("", sample_tools)

        assert r1.normal_text == ""
        assert r2.normal_text == ""

    def test_streaming_non_json_text_flushed_as_normal(self, sample_tools,
                                                       parser):
        """Non-JSON text without a wrapper is flushed to normal_text."""
        result = parser.parse_streaming_increment("Hello world", sample_tools)

        assert result.normal_text == "Hello world"
        assert result.calls == []

    def test_streaming_bare_json_with_trailing_content(self, sample_tools,
                                                       parser):
        """Bare-JSON tool call plus trailing text: emit calls, don't buffer.

        NVBug 6240584 review follow-up: previously the streaming path called
        `json.loads(stripped)`, which fails with `Extra data` when the
        buffered content is `'{...} trailing text'`. The parser would then
        keep buffering forever and never emit the tool call. With
        `raw_decode`, the tool call must be emitted at the JSON boundary
        and the trailing text must be dropped (bare-JSON mode already
        suppresses subsequent chunks).
        """
        result = parser.parse_streaming_increment(
            '{"name":"get_weather","arguments":{"city":"Paris"}}\n'
            'Extra text after the tool call.', sample_tools)

        names = [c.name for c in result.calls if c.name]
        assert "get_weather" in names
        params = "".join(c.parameters for c in result.calls if c.parameters)
        assert "Paris" in params
        # Trailing text must NOT be surfaced as normal_text — it would flip
        # finish_reason back to "stop".
        assert result.normal_text == ""

    def test_streaming_bare_json_trailing_content_split_chunk(
            self, sample_tools, parser):
        """Same as above but the trailing text arrives in a later chunk.

        This exercises the state machine: chunk 1 completes the JSON (parser
        must emit calls now, not wait for more input), chunk 2 arrives after
        the parser is already in `_STREAM_MODE_BARE_JSON` and must be
        suppressed.
        """
        r1 = parser.parse_streaming_increment(
            '{"name":"get_weather","arguments":{"city":"Paris"}}', sample_tools)
        r2 = parser.parse_streaming_increment('\nExtra text.', sample_tools)

        names = [c.name for c in r1.calls if c.name]
        assert "get_weather" in names
        assert r1.normal_text == ""
        # Trailing chunk is fully suppressed.
        assert r2.calls == []
        assert r2.normal_text == ""

    def test_streaming_wrapped_form_unregressed(self, sample_tools, parser):
        """The pre-existing wrapped-form streaming path continues to work."""
        # Send bot token.
        parser.parse_streaming_increment("<tool_call>\n", sample_tools)

        # Partial JSON with name -> emits name with empty params.
        r_name = parser.parse_streaming_increment('{"name":"get_weather"',
                                                  sample_tools)
        assert len(r_name.calls) == 1
        assert r_name.calls[0].name == "get_weather"
        assert r_name.calls[0].parameters == ""

        # Complete the JSON and the wrapper.
        r_args = parser.parse_streaming_increment(
            ',"arguments":{"location":"SF"}}\n</tool_call>', sample_tools)
        assert len(r_args.calls) == 1
        assert json.loads(r_args.calls[0].parameters) == {"location": "SF"}

    @pytest.mark.parametrize(
        "arguments_chunk",
        [', "arguments": {}}', "}", ', "arguments": null}'],
        ids=["empty_object", "key_absent", "explicit_null"],
    )
    def test_streaming_zero_arg_tool(self, parser, arguments_chunk):
        """Test streaming a zero-argument tool call.

        A model can express "no arguments" as an empty object, by omitting the
        key, or as an explicit null. All three have to complete the call and
        stream "{}".
        """
        tools = [
            ChatCompletionToolsParam(
                type="function",
                function=FunctionDefinition(
                    name="get_time",
                    description="Get current time",
                    parameters={
                        "type": "object",
                        "properties": {},
                    },
                ),
            )
        ]
        chunks = [
            "<tool_call>\n",
            '{"name": "get_time"',
            arguments_chunk,
            "\n</tool_call>",
        ]

        results = [
            parser.parse_streaming_increment(chunk, tools) for chunk in chunks
        ]

        names = [c.name for r in results for c in r.calls if c.name]
        assert "get_time" in names

        # A zero-argument call still has to stream its arguments, otherwise the
        # client is left with arguments="", which is not valid JSON.
        params = "".join(c.parameters for r in results for c in r.calls)
        assert params == "{}", f"Expected '{{}}', got {params!r}"

        # The one-shot path resolves all three shapes to the same "{}".
        oneshot = parser.detect_and_parse("".join(chunks), tools)
        assert oneshot.calls[0].parameters == "{}"

        # The completed call must also be consumed from the buffer, otherwise the
        # parser stays in the tool-call branch and swallows the rest of the output.
        assert "<tool_call>" not in parser._buffer


class TestQwen3CoderToolParser(BaseToolParserTestClass):
    """Test suite for Qwen3CoderToolParser class."""

    def make_parser(self):
        return Qwen3CoderToolParser()

    def make_tool_parser_test_cases(self):
        return ToolParserTestCases(
            has_tool_call_true=("Some text <tool_call>\n"
                                "<function=get_weather>\n"
                                "<parameter=location>NYC</parameter>\n"
                                "</function>\n"
                                "</tool_call>"),
            detect_and_parse_single_tool=(
                # Input text.
                ("Normal text\n"
                 "<tool_call>\n"
                 "<function=get_weather>\n"
                 "<parameter=location>NYC</parameter>\n"
                 "</function>\n"
                 "</tool_call>"),
                # Expected `normal_text`.
                "Normal text\n",
                # Expected `name`.
                "get_weather",
                # Expected `parameters`.
                {
                    "location": "NYC"
                },
            ),
            detect_and_parse_multiple_tools=(
                # Input text.
                ("<tool_call>\n"
                 "<function=get_weather>\n"
                 "<parameter=location>LA</parameter>\n"
                 "</function>\n"
                 "</tool_call>\n"
                 "<tool_call>\n"
                 "<function=search_web>\n"
                 "<parameter=query>AI</parameter>\n"
                 "</function>\n"
                 "</tool_call>"),
                # Expected names.
                ("get_weather", "search_web"),
            ),
            detect_and_parse_malformed_tool=(
                # Typo.
                # NOTE: the regexes + logic in `Qwen3CoderToolParser` seems deliberately forgiving.
                # For example, forgetting the closing `</function>` is fine, as is the closing
                # `</parameter>`. However, the values returned in the function call information
                # might be dubious as a result.
                "<too_call>\n"
                "<function=get_weather>\n"
                "<parameter=location>San Francisco, CA</parameter>\n"
                "</function>\n"
                "</tool_call>"),
            detect_and_parse_with_parameters_key=(
                # Input text (Qwen3Coder uses "parameter", not "parameters").
                ("<tool_call>\n"
                 "<function=search_web>\n"
                 "<parameter=query>test</parameter>\n"
                 "</function>\n"
                 "</tool_call>"),
                # Expected `name`.
                "search_web",
                # Expected `parameters`.
                {
                    "query": "test"
                },
            ),
            parse_streaming_increment_partial_bot_token="<tool_call>",
            undefined_tool=("<tool_call>\n"
                            "<function=undefined_func>\n"
                            "<parameter=arg>value</parameter>\n"
                            "</function>\n"
                            "</tool_call>"),
        )

    def test_parse_streaming_increment_complete_tool_call(
            self, sample_tools, parser):
        """Test streaming parser with complete tool call in chunks."""

        # Send tool call start token
        result = parser.parse_streaming_increment("<tool_call>\n", sample_tools)
        assert len(result.calls) == 0

        # Send function declaration
        result = parser.parse_streaming_increment("<function=get_weather>\n",
                                                  sample_tools)

        # Should send tool name with empty parameters
        assert len(result.calls) == 1
        assert result.calls[0].name == "get_weather"
        assert result.calls[0].parameters == ""

        # Send parameter block
        result = parser.parse_streaming_increment(
            '<parameter=location>SF</parameter>\n</function>\n</tool_call>',
            sample_tools)

        # Should stream parameters
        assert len(result.calls) >= 1
        # Check that parameters were sent (could be in multiple chunks)
        all_params = "".join(call.parameters for call in result.calls
                             if call.parameters)
        assert "location" in all_params
        assert "SF" in all_params

    def test_parse_streaming_increment_end_token_handling(
            self, sample_tools, parser):
        """Test streaming parser handles end token correctly."""

        # Send complete tool call
        parser.parse_streaming_increment(
            "<tool_call>\n"
            "<function=get_weather>\n"
            "<parameter=location>NYC</parameter>\n"
            "</function>\n"
            "</tool_call>", sample_tools)

        # Check buffer state - should be cleared after complete tool call
        assert parser._buf == ""
        assert parser._in_tool_call is False

    def test_parse_streaming_increment_multiple_tools_streaming(
            self, sample_tools, parser):
        """Test streaming parser handles multiple tool calls."""

        # First tool.
        parser.parse_streaming_increment("<tool_call>\n", sample_tools)
        parser.parse_streaming_increment("<function=get_weather>\n",
                                         sample_tools)
        parser.parse_streaming_increment(
            "<parameter=location>NYC</parameter>\n</function>\n</tool_call>\n",
            sample_tools)

        # Second tool.
        parser.parse_streaming_increment("<tool_call>\n", sample_tools)
        result = parser.parse_streaming_increment("<function=search_web>\n",
                                                  sample_tools)

        # Should have started second tool.
        assert result.calls[0].name == "search_web"
        assert result.calls[0].parameters == ""
        assert result.calls[0].tool_index == 1

    def test_parse_streaming_increment_multiple_parameters(
            self, sample_tools, parser):
        """Test parser handles multiple parameters in a single function call."""

        tool_def = ChatCompletionToolsParam(
            type="function",
            function=FunctionDefinition(
                name="multi_param_func",
                description="Function with multiple parameters",
                parameters={
                    "type":
                    "object",
                    "properties": {
                        "param1": {
                            "type": "string"
                        },
                        "param2": {
                            "type": "float"
                        },
                        "param3": {
                            "type": "integer"
                        },
                        "param4": {
                            "type": "boolean"
                        },
                        "param5": {
                            "type": "object"
                        },
                        "param6": {
                            "type": "array"
                        },
                        "param7": {
                            "type": "null"
                        },
                        "param8": {
                            "type": "other_type"
                        }
                    },
                    "required": [
                        "param1", "param2", "param3", "param4", "param5",
                        "param6", "param7", "param8"
                    ]
                }))

        text = ("<tool_call>\n"
                "<function=multi_param_func>\n"
                "<parameter=param1>42</parameter>\n"
                "<parameter=param2>41.9</parameter>\n"
                "<parameter=param3>42</parameter>\n"
                "<parameter=param4>true</parameter>\n"
                "<parameter=param5>{\"key\": \"value\"}</parameter>\n"
                "<parameter=param6>[1, 2, 3]</parameter>\n"
                "<parameter=param7>null</parameter>\n"
                "<parameter=param8>{'arg1': 3, 'arg2': [1, 2]}</parameter>\n"
                "</function>\n"
                "</tool_call>")

        result = parser.detect_and_parse(text, [tool_def])

        assert len(result.calls) == 1
        assert result.calls[0].name == "multi_param_func"
        assert json.loads(result.calls[0].parameters) == {
            "param1": "42",
            "param2": 41.9,
            "param3": 42,
            "param4": True,
            "param5": {
                "key": "value"
            },
            "param6": [1, 2, 3],
            "param7": None,
            "param8": {
                "arg1": 3,
                "arg2": [1, 2]
            }
        }

    def test_parse_anyof_parameter_type_conversion(self, parser):
        """Test that parameters using anyOf schemas are correctly type-converted."""
        tool_def = ChatCompletionToolsParam(
            type="function",
            function=FunctionDefinition(
                name="create_record",
                description="Create a record with various optional fields",
                parameters={
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string"
                        },
                        "count": {
                            "anyOf": [{
                                "type": "integer"
                            }, {
                                "type": "null"
                            }],
                        },
                        "score": {
                            "anyOf": [{
                                "type": "number"
                            }, {
                                "type": "null"
                            }],
                        },
                        "active": {
                            "anyOf": [{
                                "type": "boolean"
                            }, {
                                "type": "null"
                            }],
                        },
                        "metadata": {
                            "anyOf": [{
                                "type": "object"
                            }, {
                                "type": "null"
                            }],
                        },
                        "tags": {
                            "anyOf": [{
                                "type": "array"
                            }, {
                                "type": "null"
                            }],
                        },
                        "label": {
                            "anyOf": [{
                                "type": "string"
                            }, {
                                "type": "null"
                            }],
                        },
                    },
                    "required": ["name"],
                },
            ),
        )

        text = ("<tool_call>\n"
                "<function=create_record>\n"
                "<parameter=name>test</parameter>\n"
                "<parameter=count>42</parameter>\n"
                "<parameter=score>3.14</parameter>\n"
                "<parameter=active>true</parameter>\n"
                '<parameter=metadata>{"key": "value"}</parameter>\n'
                "<parameter=tags>[1, 2, 3]</parameter>\n"
                "<parameter=label>hello</parameter>\n"
                "</function>\n"
                "</tool_call>")

        result = parser.detect_and_parse(text, [tool_def])

        assert len(result.calls) == 1
        params = json.loads(result.calls[0].parameters)
        assert params["name"] == "test"
        assert params["count"] == 42
        assert isinstance(params["count"], int)
        assert params["score"] == 3.14
        assert isinstance(params["score"], float)
        assert params["active"] is True
        assert params["metadata"] == {"key": "value"}
        assert params["tags"] == [1, 2, 3]
        assert params["label"] == "hello"

    def test_parse_anyof_null_value(self, parser):
        """Test that null values are handled correctly for anyOf parameters."""
        tool_def = ChatCompletionToolsParam(
            type="function",
            function=FunctionDefinition(
                name="set_value",
                description="Set a value",
                parameters={
                    "type": "object",
                    "properties": {
                        "value": {
                            "anyOf": [{
                                "type": "integer"
                            }, {
                                "type": "null"
                            }],
                        },
                    },
                },
            ),
        )

        text = ("<tool_call>\n"
                "<function=set_value>\n"
                "<parameter=value>null</parameter>\n"
                "</function>\n"
                "</tool_call>")

        result = parser.detect_and_parse(text, [tool_def])

        assert len(result.calls) == 1
        params = json.loads(result.calls[0].parameters)
        assert params["value"] is None

    def test_qwen3_coder_format_compliance(
        self,
        parser,
    ):
        """Test that Qwen3CoderToolParser follows the documented format structure."""

        # Test the exact format from the docstring
        text = ("<tool_call>\n"
                "<function=execute_bash>\n"
                "<parameter=command>\n"
                "pwd && ls\n"
                "</parameter>\n"
                "</function>\n"
                "</tool_call>")

        tool_def = ChatCompletionToolsParam(
            type="function",
            function=FunctionDefinition(
                name="execute_bash",
                description="Execute a bash command.",
                parameters={
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "The command to execute.",
                        },
                        "unit": {
                            "type": "string",
                        }
                    },
                    "required": ["command"],
                },
            ))

        result = parser.detect_and_parse(text, [tool_def])

        assert len(result.calls) == 1
        assert result.calls[0].name == "execute_bash"
        assert json.loads(result.calls[0].parameters) == {
            "command": "pwd && ls"
        }


# ============================================================================
# DeepSeek Parser Tests
# ============================================================================


class TestDeepSeekV3Parser(BaseToolParserTestClass):
    """Test suite for DeepSeekV3Parser class."""

    def make_parser(self):
        return DeepSeekV3Parser()

    def make_tool_parser_test_cases(self):
        calls_begin = "<｜tool▁calls▁begin｜>"
        calls_end = "<｜tool▁calls▁end｜>"
        call_begin = "<｜tool▁call▁begin｜>"
        call_end = "<｜tool▁call▁end｜>"
        sep = "<｜tool▁sep｜>"

        single_text = (
            f"Lead {calls_begin}{call_begin}function{sep}get_weather\n```json\n"
            f"{json.dumps({'location': 'Tokyo'})}\n```{call_end}{calls_end}")
        single_expected_normal = "Lead"  # the text is stripped
        single_expected_name = "get_weather"
        single_expected_params = {"location": "Tokyo"}

        # Provide one tool to satisfy type hints (tuple[str, tuple[str]])
        multiple_text = (
            f"{calls_begin}{call_begin}function{sep}get_weather\n```json\n"
            f"{json.dumps({'location': 'Paris'})}\n```{call_end}"
            f"{calls_begin}{call_begin}function{sep}search_web\n```json\n"
            f"{json.dumps({'query': 'AI'})}\n```{call_end}{calls_end}")
        multiple_names = ("get_weather", "search_web")

        malformed_text = (
            f"{calls_begin}{call_begin}function{sep}get_weather\n```json\n"
            "{'location': 'Paris'}\n```"
            f"{call_end}{calls_end}")

        with_parameters_key_text = (
            f"{calls_begin}{call_begin}function{sep}search_web\n```json\n"
            f"{json.dumps({'parameters': {'query': 'TensorRT'}})}\n```{call_end}{calls_end}"
        )
        with_parameters_key_name = "search_web"
        with_parameters_key_params = {"parameters": {"query": "TensorRT"}}

        partial_bot_token = "<｜tool▁cal"

        undefined_tool_text = (
            f"{calls_begin}{call_begin}function{sep}unknown\n```json\n"
            f"{json.dumps({'x': 1})}\n```{call_end}{calls_end}")

        return ToolParserTestCases(
            has_tool_call_true=f"Hello {calls_begin}",
            detect_and_parse_single_tool=(
                single_text,
                single_expected_normal,
                single_expected_name,
                single_expected_params,
            ),
            detect_and_parse_multiple_tools=(multiple_text, multiple_names),
            detect_and_parse_malformed_tool=malformed_text,
            detect_and_parse_with_parameters_key=(
                with_parameters_key_text,
                with_parameters_key_name,
                with_parameters_key_params,
            ),
            parse_streaming_increment_partial_bot_token=partial_bot_token,
            undefined_tool=undefined_tool_text,
        )


class TestDeepSeekV31Parser(BaseToolParserTestClass):
    """Test suite for DeepSeekV31Parser class."""

    def make_parser(self):
        return DeepSeekV31Parser()

    def make_tool_parser_test_cases(self):
        calls_begin = "<｜tool▁calls▁begin｜>"
        calls_end = "<｜tool▁calls▁end｜>"
        call_begin = "<｜tool▁call▁begin｜>"
        call_end = "<｜tool▁call▁end｜>"
        sep = "<｜tool▁sep｜>"

        single_text = (
            f"Intro {calls_begin}{call_begin}get_weather{sep}"
            f"{json.dumps({'location': 'Tokyo'})}{call_end}{calls_end}")
        single_expected_normal = "Intro"  # the text is stripped
        single_expected_name = "get_weather"
        single_expected_params = {"location": "Tokyo"}

        multiple_text = (f"{calls_begin}{call_begin}get_weather{sep}"
                         f"{json.dumps({'location': 'Paris'})}{call_end}"
                         f"{calls_begin}{call_begin}search_web{sep}"
                         f"{json.dumps({'query': 'AI'})}{call_end}{calls_end}")
        multiple_names = ("get_weather", "search_web")

        malformed_text = (
            f"{calls_begin}{call_begin}get_weather{sep}{{'location':'Paris'}}"
            f"{call_end}{calls_end}")

        with_parameters_key_text = (
            f"{calls_begin}{call_begin}search_web{sep}"
            f"{json.dumps({'parameters': {'query': 'TensorRT'}})}{call_end}{calls_end}"
        )
        with_parameters_key_name = "search_web"
        with_parameters_key_params = {"parameters": {"query": "TensorRT"}}

        partial_bot_token = "<｜tool▁cal"

        undefined_tool_text = (
            f"{calls_begin}{call_begin}unknown{sep}{json.dumps({'x': 1})}{call_end}{calls_end}"
        )

        return ToolParserTestCases(
            has_tool_call_true=f"Hi {calls_begin}",
            detect_and_parse_single_tool=(
                single_text,
                single_expected_normal,
                single_expected_name,
                single_expected_params,
            ),
            detect_and_parse_multiple_tools=(multiple_text, multiple_names),
            detect_and_parse_malformed_tool=malformed_text,
            detect_and_parse_with_parameters_key=(
                with_parameters_key_text,
                with_parameters_key_name,
                with_parameters_key_params,
            ),
            parse_streaming_increment_partial_bot_token=partial_bot_token,
            undefined_tool=undefined_tool_text,
        )


# ============================================================================
# DeepSeekV32Parser Tests
# ============================================================================


class TestDeepSeekV32Parser(BaseToolParserTestClass):
    """Test suite for DeepSeekV32Parser class."""

    def make_parser(self):
        return DeepSeekV32Parser()

    def make_tool_parser_test_cases(self):
        return ToolParserTestCases(
            has_tool_call_true=
            'Some text <｜DSML｜function_calls> <｜DSML｜invoke name="get_weather"> <｜DSML｜parameter name="location" string="true">NYC</｜DSML｜parameter> </｜DSML｜invoke> </｜DSML｜function_calls>',
            detect_and_parse_single_tool=(
                # Input text.
                ('Normal text'
                 '<｜DSML｜function_calls> <｜DSML｜invoke name="get_weather"> <｜DSML｜parameter name="location" string="true">NYC</｜DSML｜parameter> </｜DSML｜invoke> </｜DSML｜function_calls>'
                 ),
                # Expected `normal_text`.
                "Normal text",
                # Expected `name`.
                "get_weather",
                # Expected `parameters`.
                {
                    "location": "NYC"
                },
            ),
            detect_and_parse_multiple_tools=(
                # Input text.
                ('<｜DSML｜function_calls> <｜DSML｜invoke name="get_weather"> <｜DSML｜parameter name="location" string="true">NYC</｜DSML｜parameter> </｜DSML｜invoke> <｜DSML｜invoke name="search_web"> { "query": "AI" } </｜DSML｜invoke> </｜DSML｜function_calls>'
                 ),
                # Expected names.
                ("get_weather", "search_web"),
            ),
            detect_and_parse_malformed_tool=(
                # Format error: using "|" instead of "｜"
                '<|DSML|function_calls> <|DSML|invoke name="get_weather"> <|DSML|parameter name="location" string="true">NYC</|DSML|parameter> </|DSML|invoke> </|DSML|function_calls>'
            ),
            detect_and_parse_with_parameters_key=(
                # Input text.
                ('<｜DSML｜function_calls> <｜DSML｜invoke name="search_web"> { "query": "test" } </｜DSML｜invoke> </｜DSML｜function_calls>'
                 ),
                # Expected `name`.
                "search_web",
                # Expected `parameters`.
                {
                    "query": "test"
                },
            ),
            parse_streaming_increment_partial_bot_token="<｜DSML｜function_calls>",
            undefined_tool=
            ('<｜DSML｜function_calls> <｜DSML｜invoke name="undefined_func"> <｜DSML｜parameter name="arg" string="true">value</｜DSML｜parameter> </｜DSML｜invoke> </｜DSML｜function_calls>'
             ),
        )

    def test_initialization(self, parser):
        """Test that DeepSeekV32Parser initializes correctly."""
        assert parser.bot_token == "<｜DSML｜function_calls>"
        assert parser.eot_token == "</｜DSML｜function_calls>"
        assert parser.invoke_end_token == "</｜DSML｜invoke>"
        assert parser._last_arguments == ""

    def test_parse_streaming_increment_complete_tool_call(
            self, sample_tools, parser):
        """Test streaming parser with complete tool call in chunks."""
        parser.parse_streaming_increment('<｜DSML｜function_calls> ',
                                         sample_tools)
        result = parser.parse_streaming_increment(
            '<｜DSML｜invoke name="get_weather"> ', sample_tools)
        assert len(result.calls) == 0

        parser.parse_streaming_increment(
            '<｜DSML｜parameter name="location" string="true">NYC</｜DSML｜parameter> ',
            sample_tools)
        result = parser.parse_streaming_increment(
            '</｜DSML｜invoke> </｜DSML｜function_calls>', sample_tools)

        assert result.calls[0].name == "get_weather"
        assert json.loads(result.calls[1].parameters) == {"location": "NYC"}

    def test_parse_streaming_increment_multiple_tools_streaming(
            self, sample_tools, parser):
        """Test streaming parser handles end token correctly."""
        # First tool
        parser.parse_streaming_increment("<｜DSML｜function_calls> ",
                                         sample_tools)
        parser.parse_streaming_increment("<｜DSML｜invoke name=\"get_weather\"> ",
                                         sample_tools)
        parser.parse_streaming_increment(
            "<｜DSML｜parameter name=\"location\" string=\"true\">NYC</｜DSML｜parameter> ",
            sample_tools)
        parser.parse_streaming_increment("</｜DSML｜invoke> ", sample_tools)

        # Second tool
        parser.parse_streaming_increment("<｜DSML｜invoke name=\"search_web\"> ",
                                         sample_tools)
        parser.parse_streaming_increment(
            "<｜DSML｜parameter name=\"query\" string=\"true\">AI</｜DSML｜parameter> ",
            sample_tools)
        result = parser.parse_streaming_increment(
            "</｜DSML｜invoke> <｜DSML｜function_calls>", sample_tools)

        assert result.calls[0].name == "search_web"
        assert json.loads(result.calls[1].parameters) == {"query": "AI"}
        assert result.calls[1].tool_index == 1

    def test_structure_info_function(self):
        """Test that DeepSeekV32Parser structure_info returns correct lambda function."""
        parser = DeepSeekV32Parser()
        func = parser.structure_info()

        info = func("get_weather")

        assert info.begin == "<｜DSML｜invoke name=\"get_weather\">"
        assert info.end == "</｜DSML｜invoke>"
        assert info.trigger == "<｜DSML｜invoke name=\"get_weather\">"

    def test_structure_info_different_names(self):
        """Test that DeepSeekV32Parser structure_info returns correct lambda function."""
        parser = DeepSeekV32Parser()
        func = parser.structure_info()

        info1 = func("get_weather")
        info2 = func("search_web")

        assert "get_weather" in info1.begin
        assert "search_web" in info2.begin
        assert info1.end == info2.end == "</｜DSML｜invoke>"

    def test_deepseek_v32_format_compliance(self, sample_tools, parser):
        """Test that DeepSeekV32Parser follows the documented format structure."""

        # Test the exact format from the docstring
        text = "<｜DSML｜function_calls> <｜DSML｜invoke name=\"get_weather\"> <｜DSML｜parameter name=\"location\" string=\"true\">NYC</｜DSML｜parameter> </｜DSML｜invoke> </｜DSML｜function_calls>"
        result = parser.detect_and_parse(text, sample_tools)

        assert len(result.calls) == 1
        assert result.calls[0].name == "get_weather"
        assert json.loads(result.calls[0].parameters) == {"location": "NYC"}

    def test_encode_messages_multi_turn_with_tool_calls(self):
        """NVBug 5937478: encode_messages must handle dict-typed tool_call arguments.

        chat_utils deserializes arguments to dict; encode_arguments_to_dsml
        must not call json.loads() on it again.
        """
        messages = [
            {
                "role": "user",
                "content": "list files"
            },
            {
                "role":
                "assistant",
                "content":
                None,
                "tool_calls": [{
                    "id": "c1",
                    "type": "function",
                    "function": {
                        "name": "bash",
                        "arguments": {
                            "command": "ls"
                        },  # dict, not str
                    },
                }],
            },
            {
                "role": "tool",
                "content": "a.py b.py",
                "tool_call_id": "c1"
            },
            {
                "role": "user",
                "content": "open a.py"
            },
        ]
        result = encode_messages(messages, thinking_mode="chat")
        assert '<｜DSML｜invoke name="bash">' in result
        assert 'name="command"' in result
        assert ">ls<" in result


# ============================================================================
# DeepSeekV4Parser Tests
# ============================================================================


class TestDeepSeekV4Parser(BaseToolParserTestClass):
    """Test suite for DeepSeekV4Parser class."""

    def make_parser(self):
        return DeepSeekV4Parser()

    def make_tool_parser_test_cases(self):
        return ToolParserTestCases(
            has_tool_call_true=
            ('Some text <｜DSML｜tool_calls> <｜DSML｜invoke name="get_weather"> '
             '<｜DSML｜parameter name="location" string="true">NYC</｜DSML｜parameter> '
             "</｜DSML｜invoke> </｜DSML｜tool_calls>"),
            detect_and_parse_single_tool=(
                ('Normal text<｜DSML｜tool_calls> <｜DSML｜invoke name="get_weather"> '
                 '<｜DSML｜parameter name="location" string="true">NYC</｜DSML｜parameter> '
                 "</｜DSML｜invoke> </｜DSML｜tool_calls>"),
                "Normal text",
                "get_weather",
                {
                    "location": "NYC"
                },
            ),
            detect_and_parse_multiple_tools=(
                ('<｜DSML｜tool_calls> <｜DSML｜invoke name="get_weather"> '
                 '<｜DSML｜parameter name="location" string="true">NYC</｜DSML｜parameter> '
                 '</｜DSML｜invoke> <｜DSML｜invoke name="search_web"> '
                 '{ "query": "AI" } </｜DSML｜invoke> </｜DSML｜tool_calls>'),
                ("get_weather", "search_web"),
            ),
            detect_and_parse_malformed_tool=
            ('<|DSML|tool_calls> <|DSML|invoke name="get_weather"> '
             '<|DSML|parameter name="location" string="true">NYC</|DSML|parameter> '
             "</|DSML|invoke> </|DSML|tool_calls>"),
            detect_and_parse_with_parameters_key=(
                ('<｜DSML｜tool_calls> <｜DSML｜invoke name="search_web"> '
                 '{ "query": "test" } </｜DSML｜invoke> </｜DSML｜tool_calls>'),
                "search_web",
                {
                    "query": "test"
                },
            ),
            parse_streaming_increment_partial_bot_token="<｜DSML｜tool",
            undefined_tool=
            ('<｜DSML｜tool_calls> <｜DSML｜invoke name="undefined_func"> '
             '<｜DSML｜parameter name="arg" string="true">value</｜DSML｜parameter> '
             "</｜DSML｜invoke> </｜DSML｜tool_calls>"),
        )


# ============================================================================
# DeepSeek streaming text preservation
# ============================================================================


@pytest.mark.parametrize(
    "parser_cls",
    [DeepSeekV3Parser, DeepSeekV31Parser, DeepSeekV32Parser, DeepSeekV4Parser])
@pytest.mark.parametrize(
    "deltas",
    [
        # A delta that is itself a prefix of a tool-call start token.
        ["Use ", "<", "div> for a block element."],
        # A delta that ends on such a prefix after other text.
        ["The condition is a <", " b, so it holds."],
        # Text that starts like a start token and then diverges from it, for
        # both of the tokens the V3.2 and V4 parsers look for.
        ["Use <｜DSML｜function", "ality and <｜DSML｜invoke", "ality"],
    ],
)
def test_deepseek_streaming_preserves_withheld_text(
        sample_tools: list[ChatCompletionToolsParam],
        parser_cls: type[BaseToolParser], deltas: list[str]) -> None:
    """Withholding a delta may delay text but must never drop it."""
    parser = parser_cls()

    streamed = "".join(
        parser.parse_streaming_increment(delta, sample_tools).normal_text
        for delta in deltas)

    expected = "".join(deltas)
    assert streamed == expected, f"Expected {expected!r}, got {streamed!r}"
    assert parser_cls().detect_and_parse(expected,
                                         sample_tools).normal_text == expected


@pytest.mark.parametrize(
    "parser_cls, tool_call_text",
    [
        (
            DeepSeekV3Parser,
            ("<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>"
             'get_weather\n```json\n{"location": "Tokyo"}\n```'
             "<｜tool▁call▁end｜><｜tool▁calls▁end｜>"),
        ),
        (
            DeepSeekV31Parser,
            ("<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>get_weather<｜tool▁sep｜>"
             '{"location": "Tokyo"}<｜tool▁call▁end｜><｜tool▁calls▁end｜>'),
        ),
        (
            DeepSeekV32Parser,
            ('<｜DSML｜function_calls><｜DSML｜invoke name="get_weather">'
             '<｜DSML｜parameter name="location" string="true">Tokyo'
             "</｜DSML｜parameter></｜DSML｜invoke></｜DSML｜function_calls>"),
        ),
        (
            DeepSeekV4Parser,
            ('<｜DSML｜tool_calls><｜DSML｜invoke name="get_weather">'
             '<｜DSML｜parameter name="location" string="true">Tokyo'
             "</｜DSML｜parameter></｜DSML｜invoke></｜DSML｜tool_calls>"),
        ),
    ],
)
def test_deepseek_streaming_emits_text_before_tool_call(
        sample_tools: list[ChatCompletionToolsParam],
        parser_cls: type[BaseToolParser], tool_call_text: str) -> None:
    """Text that precedes a tool call in the same delta is content."""
    text = "Normal text" + tool_call_text

    result = parser_cls().parse_streaming_increment(text, sample_tools)

    assert result.normal_text == "Normal text"
    assert result.normal_text == parser_cls().detect_and_parse(
        text, sample_tools).normal_text
    assert "get_weather" in [call.name for call in result.calls if call.name]


@pytest.mark.parametrize(
    "parser_cls, tool_call_text",
    [
        (
            DeepSeekV3Parser,
            ("<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>"
             'get_weather\n```json\n{"location": "Tokyo"}\n```'
             "<｜tool▁call▁end｜><｜tool▁calls▁end｜>"),
        ),
        (
            DeepSeekV31Parser,
            ("<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>get_weather<｜tool▁sep｜>"
             '{"location": "Tokyo"}<｜tool▁call▁end｜><｜tool▁calls▁end｜>'),
        ),
        (
            DeepSeekV32Parser,
            ('<｜DSML｜function_calls><｜DSML｜invoke name="get_weather">'
             '{"location": "Tokyo"}</｜DSML｜invoke></｜DSML｜function_calls>'),
        ),
        (
            DeepSeekV4Parser,
            ('<｜DSML｜tool_calls><｜DSML｜invoke name="get_weather">'
             '{"location": "Tokyo"}</｜DSML｜invoke></｜DSML｜tool_calls>'),
        ),
    ],
)
def test_deepseek_streaming_prefix_is_delta_independent(
        sample_tools: list[ChatCompletionToolsParam],
        parser_cls: type[BaseToolParser], tool_call_text: str) -> None:
    """The prefix is streamed verbatim however the deltas are cut."""
    prefix = "  Normal text  "
    text = prefix + tool_call_text
    splits = [
        [text],
        [prefix, tool_call_text],
        [text[:8], text[8:]],
    ]

    for deltas in splits:
        parser = parser_cls()
        results = [
            parser.parse_streaming_increment(delta, sample_tools)
            for delta in deltas
        ]
        streamed = "".join(result.normal_text for result in results)
        names = [
            call.name for result in results for call in result.calls
            if call.name
        ]

        assert streamed == prefix, f"{deltas!r} streamed {streamed!r}"
        assert names == ["get_weather"], f"{deltas!r} called {names!r}"


# ============================================================================
# Glm4ToolParser Tests
# ============================================================================


class TestGlm4ToolParser(BaseToolParserTestClass):
    """Test suite for Glm4ToolParser class."""

    def make_parser(self):
        return Glm4ToolParser()

    def make_tool_parser_test_cases(self):
        single_text = ("Normal text"
                       "<tool_call>get_weather\n"
                       "<arg_key>location</arg_key>\n"
                       "<arg_value>NYC</arg_value>\n"
                       "</tool_call>")
        single_expected_normal = "Normal text"
        single_expected_name = "get_weather"
        single_expected_params = {"location": "NYC"}

        multiple_text = ("<tool_call>get_weather\n"
                         "<arg_key>location</arg_key>\n"
                         "<arg_value>LA</arg_value>\n"
                         "</tool_call>"
                         "<tool_call>search_web\n"
                         "<arg_key>query</arg_key>\n"
                         "<arg_value>AI</arg_value>\n"
                         "</tool_call>")
        multiple_names = ("get_weather", "search_web")

        malformed_text = ("<tool_call>get_weather"
                          "MALFORMED_NO_NEWLINE</tool_call>")

        with_parameters_text = ("<tool_call>search_web\n"
                                "<arg_key>query</arg_key>\n"
                                "<arg_value>test</arg_value>\n"
                                "</tool_call>")
        with_parameters_name = "search_web"
        with_parameters_params = {"query": "test"}

        partial_bot_token = "<tool_cal"

        undefined_tool_text = ("<tool_call>undefined_func\n"
                               "<arg_key>arg</arg_key>\n"
                               "<arg_value>value</arg_value>\n"
                               "</tool_call>")

        return ToolParserTestCases(
            has_tool_call_true=
            "Some text <tool_call>get_weather\n<arg_key>location</arg_key>\n<arg_value>NYC</arg_value>\n</tool_call>",
            detect_and_parse_single_tool=(
                single_text,
                single_expected_normal,
                single_expected_name,
                single_expected_params,
            ),
            detect_and_parse_multiple_tools=(multiple_text, multiple_names),
            detect_and_parse_malformed_tool=malformed_text,
            detect_and_parse_with_parameters_key=(
                with_parameters_text,
                with_parameters_name,
                with_parameters_params,
            ),
            parse_streaming_increment_partial_bot_token=partial_bot_token,
            undefined_tool=undefined_tool_text,
        )

    def test_initialization(self, parser):
        """Test that Glm4ToolParser initializes correctly."""
        assert parser.bot_token == "<tool_call>"
        assert parser.eot_token == "</tool_call>"

    def test_parse_streaming_increment_complete_tool_call(
            self, sample_tools, parser):
        """Test streaming parser with complete tool call in chunks."""

        # Send bot token with function name
        result = parser.parse_streaming_increment("<tool_call>get_weather\n",
                                                  sample_tools)

        # Should send tool name
        assert len(result.calls) == 1
        assert result.calls[0].name == "get_weather"
        assert result.calls[0].parameters == ""

        # Send arguments
        result = parser.parse_streaming_increment(
            "<arg_key>location</arg_key>\n"
            "<arg_value>SF</arg_value>\n"
            "</tool_call>", sample_tools)

        # Should stream arguments and complete the tool call
        all_params = "".join(call.parameters for call in result.calls
                             if call.parameters)
        assert "location" in all_params
        assert "SF" in all_params

    def test_parse_streaming_increment_multiple_tools_streaming(
            self, sample_tools, parser):
        """Test streaming parser handles multiple tool calls."""

        # First tool
        parser.parse_streaming_increment("<tool_call>get_weather\n",
                                         sample_tools)
        parser.parse_streaming_increment(
            "<arg_key>location</arg_key>\n"
            "<arg_value>NYC</arg_value>\n"
            "</tool_call>", sample_tools)

        # Second tool
        result = parser.parse_streaming_increment("<tool_call>search_web\n",
                                                  sample_tools)

        # Should have started second tool
        assert len(result.calls) == 1
        assert result.calls[0].name == "search_web"
        assert result.calls[0].parameters == ""
        assert result.calls[0].tool_index == 1

    def test_parse_streaming_multiple_params(self, sample_tools, parser):
        """Test streaming parser handles multiple parameters."""

        # Send function name
        parser.parse_streaming_increment("<tool_call>get_weather\n",
                                         sample_tools)

        # Send first parameter
        result1 = parser.parse_streaming_increment(
            "<arg_key>location</arg_key>\n"
            "<arg_value>NYC</arg_value>\n", sample_tools)

        params1 = "".join(call.parameters for call in result1.calls
                          if call.parameters)
        assert "location" in params1

        # Send second parameter and close
        result2 = parser.parse_streaming_increment(
            "<arg_key>unit</arg_key>\n"
            "<arg_value>celsius</arg_value>\n"
            "</tool_call>", sample_tools)

        params2 = "".join(call.parameters for call in result2.calls
                          if call.parameters)
        assert "unit" in params2

    def test_detect_and_parse_multiple_params(self, sample_tools):
        """Test one-shot parsing with multiple parameters."""
        parser = Glm4ToolParser()
        text = ("<tool_call>get_weather\n"
                "<arg_key>location</arg_key>\n"
                "<arg_value>Tokyo</arg_value>\n"
                "<arg_key>unit</arg_key>\n"
                "<arg_value>celsius</arg_value>\n"
                "</tool_call>")

        result = parser.detect_and_parse(text, sample_tools)

        assert len(result.calls) == 1
        assert result.calls[0].name == "get_weather"
        params = json.loads(result.calls[0].parameters)
        assert params == {"location": "Tokyo", "unit": "celsius"}

    def test_detect_and_parse_with_number_type(self):
        """Test parsing with number type coercion."""
        parser = Glm4ToolParser()
        tools = [
            ChatCompletionToolsParam(
                type="function",
                function=FunctionDefinition(
                    name="set_temperature",
                    description="Set temperature",
                    parameters={
                        "type": "object",
                        "properties": {
                            "value": {
                                "type": "number",
                            },
                            "label": {
                                "type": "string",
                            },
                        },
                        "required": ["value"],
                    },
                ),
            )
        ]

        text = ("<tool_call>set_temperature\n"
                "<arg_key>value</arg_key>\n"
                "<arg_value>72.5</arg_value>\n"
                "<arg_key>label</arg_key>\n"
                "<arg_value>room temp</arg_value>\n"
                "</tool_call>")

        result = parser.detect_and_parse(text, tools)

        assert len(result.calls) == 1
        params = json.loads(result.calls[0].parameters)
        assert params["value"] == 72.5
        assert params["label"] == "room temp"

    def test_glm4_format_compliance(self, sample_tools, parser):
        """Test that Glm4ToolParser follows the documented format structure."""

        text = ("<tool_call>get_weather\n"
                "<arg_key>location</arg_key>\n"
                "<arg_value>Tokyo</arg_value>\n"
                "</tool_call>")

        result = parser.detect_and_parse(text, sample_tools)

        assert len(result.calls) == 1
        assert result.calls[0].name == "get_weather"
        assert json.loads(result.calls[0].parameters) == {"location": "Tokyo"}

    def test_streaming_no_args(self, sample_tools, parser):
        """Test streaming a tool call with no arguments."""

        # First increment sends the tool name
        result1 = parser.parse_streaming_increment("<tool_call>get_weather\n",
                                                   sample_tools)
        names = [c.name for c in result1.calls if c.name]
        assert "get_weather" in names

        # Second increment closes the tool call with empty args
        result2 = parser.parse_streaming_increment("</tool_call>", sample_tools)
        params = "".join(c.parameters for c in result2.calls)
        assert "{}" in params

    def test_supports_structural_tag(self, parser):
        """Test that supports_structural_tag returns False."""
        assert parser.supports_structural_tag() is False

    # The GLM-4.5/4.6 parser shares the typing and delimiter machinery with
    # GLM-4.7 (see TestGlm47ArgumentCorruptions for the recorded traffic that
    # shaped it), so the same guarantees are pinned here in this parser's
    # newline-separated format.

    @staticmethod
    def _glm4_call(name, *pairs):
        body = "".join(
            f"<arg_key>{key}</arg_key>\n<arg_value>{value}</arg_value>\n"
            for key, value in pairs)
        return f"<tool_call>{name}\n{body}</tool_call>"

    @staticmethod
    def _streamed_params(chunks, tools):
        """Assemble the streamed argument string into a dict.

        A trailing empty increment stands in for the next delta, since this
        parser sends the name and only touches the arguments on a later
        increment.
        """
        parser = Glm4ToolParser()
        params = ""
        for chunk in list(chunks) + [""]:
            result = parser.parse_streaming_increment(chunk, tools)
            params += "".join(c.parameters for c in result.calls
                              if c.parameters)
        return json.loads(params)

    def test_string_schema_keeps_numeric_text_verbatim(self):
        """Corruption 1 in GLM-4.5 clothing: `99797` under a string schema."""
        tools = [
            ChatCompletionToolsParam(
                type="function",
                function=FunctionDefinition(
                    name="notebook_edit",
                    parameters={
                        "type": "object",
                        "properties": {
                            "cell_id": {
                                "type": "string"
                            }
                        },
                    },
                ),
            )
        ]
        text = self._glm4_call("notebook_edit", ("cell_id", "99797"))

        whole = Glm4ToolParser().detect_and_parse(text, tools)
        assert json.loads(whole.calls[0].parameters) == {"cell_id": "99797"}

        streamed = self._streamed_params([text], tools)
        assert streamed == {"cell_id": "99797"}

    def test_freeform_true_is_not_python_true(self):
        """Corruption 2: `true` must not str()-round-trip into "True"."""
        tools = [
            ChatCompletionToolsParam(
                type="function",
                function=FunctionDefinition(
                    name="apply_patch",
                    parameters={
                        "type": "object",
                        "properties": {
                            "input": {
                                "type": "string"
                            }
                        },
                    },
                ),
            )
        ]
        text = self._glm4_call("apply_patch", ("input", "true"))

        whole = Glm4ToolParser().detect_and_parse(text, tools)
        assert json.loads(whole.calls[0].parameters) == {"input": "true"}

        streamed = self._streamed_params([text], tools)
        assert streamed == {"input": "true"}

    def test_undeclared_arguments_pass_through_raw(self):
        """No schema, no conversion - here too."""
        tools = [
            ChatCompletionToolsParam(
                type="function",
                function=FunctionDefinition(name="exec",
                                            parameters={
                                                "type": "object",
                                                "properties": {},
                                            }),
            )
        ]
        text = self._glm4_call("exec_command", ("max_output_tokens", "8000"))

        whole = Glm4ToolParser().detect_and_parse(text, tools)
        assert json.loads(whole.calls[0].parameters) == {
            "max_output_tokens": "8000"
        }

        streamed = self._streamed_params([text], tools)
        assert streamed == {"max_output_tokens": "8000"}

    def test_value_containing_the_closing_tag_survives(self):
        """Corruption 3: the quoted tag stays in the value on both paths."""
        payload = 'let s = "</arg_value>"; run(s)'
        tools = [
            ChatCompletionToolsParam(
                type="function",
                function=FunctionDefinition(
                    name="run_js",
                    parameters={
                        "type": "object",
                        "properties": {
                            "code": {
                                "type": "string"
                            },
                            "timeout": {
                                "type": "integer"
                            },
                        },
                    },
                ),
            )
        ]
        text = self._glm4_call("run_js", ("code", payload), ("timeout", "30"))

        whole = Glm4ToolParser().detect_and_parse(text, tools)
        assert json.loads(whole.calls[0].parameters) == {
            "code": payload,
            "timeout": 30,
        }

        streamed = self._streamed_params([text], tools)
        assert streamed == {"code": payload, "timeout": 30}

        # And however the deltas fall, one character at a time included.
        assert self._streamed_params(list(text), tools) == {
            "code": payload,
            "timeout": 30,
        }

    @pytest.mark.parametrize("value", [
        "struct MMA_Traits<SM100_MMA_F16BF16_(TS|SS)<",
        "<",
        "nearly </arg_valu",
    ])
    def test_a_value_ending_in_a_prefix_of_the_close_marker_survives(
            self, value):
        """Corruption 4 in GLM-4.5 clothing: the traced grep pattern.

        A value's trailing `<` was held as a possible start of
        `</arg_value>`; when the real tag arrived, the dead buffer was
        released whole and the tag's opening `<` went out as value content,
        so the close was never recognized - the streamed string never
        terminated and the tag text leaked into it. The release now stops
        before a trailing `<` and re-anchors the match there (see
        split_dead_close_buffer). Swept over every split point, with the
        close confirmed by `<arg_key>` in one arrangement and by
        `</tool_call>` in the other.
        """
        tools = [
            ChatCompletionToolsParam(
                type="function",
                function=FunctionDefinition(
                    name="run_js",
                    parameters={
                        "type": "object",
                        "properties": {
                            "code": {
                                "type": "string"
                            },
                            "timeout": {
                                "type": "integer"
                            },
                        },
                    },
                ),
            )
        ]
        arrangements = (
            (("code", value), ("timeout", "30")),
            (("timeout", "30"), ("code", value)),
        )
        for pairs in arrangements:
            text = self._glm4_call("run_js", *pairs)
            expected = {"code": value, "timeout": 30}

            whole = Glm4ToolParser().detect_and_parse(text, tools)
            assert json.loads(whole.calls[0].parameters) == expected

            assert self._streamed_params([text], tools) == expected
            assert self._streamed_params(list(text), tools) == expected
            for i in range(1, len(text)):
                chunks = [text[:i], text[i:]]
                assert self._streamed_params(chunks, tools) == expected, (
                    f"streaming disagreed with the whole parse when split "
                    f"at {i}")

    @staticmethod
    def _streamed_text(chunks, tools):
        """The joined argument text itself, before any json.loads.

        The duplicate-key tests must look at the text: json.loads collapses
        repeated keys (last wins), so a stream still carrying the key twice
        would pass a value-level comparison while failing every typed client.
        """
        parser = Glm4ToolParser()
        params = ""
        for chunk in list(chunks) + [""]:
            result = parser.parse_streaming_increment(chunk, tools)
            params += "".join(c.parameters for c in result.calls
                              if c.parameters)
        return params

    _DUP_TOOLS = [
        ChatCompletionToolsParam(
            type="function",
            function=FunctionDefinition(
                name="notebook_edit",
                parameters={
                    "type": "object",
                    "properties": {
                        "cell_id": {
                            "type": "string"
                        },
                        "count": {
                            "type": "integer"
                        },
                    },
                },
            ),
        )
    ]

    @pytest.mark.parametrize("chunker", [
        lambda text: [text],
        list,
        lambda text: [text[i:i + 7] for i in range(0, len(text), 7)],
    ])
    def test_a_conflicting_duplicate_key_keeps_the_first_occurrence(
            self, chunker):
        """One production call wrote `cell_id` twice with different values.

        First-wins, identically on both paths: on the stream the first
        occurrence's bytes are already with the client when the repeat is
        recognized, so keeping them is the only policy under which the
        streamed JSON carries the key exactly once *and* equals the
        whole-text parse. See `_parse_argument_pairs`.
        """
        text = self._glm4_call("notebook_edit", ("cell_id", "nonexistent"),
                               ("cell_id", "nonexistent-cell"))

        whole = Glm4ToolParser().detect_and_parse(text, self._DUP_TOOLS)
        assert json.loads(whole.calls[0].parameters) == {
            "cell_id": "nonexistent"
        }

        streamed_text = self._streamed_text(chunker(text), self._DUP_TOOLS)
        assert streamed_text == whole.calls[0].parameters
        assert streamed_text.count('"cell_id"') == 1

    @pytest.mark.parametrize("chunker", [lambda text: [text], list])
    def test_an_identical_duplicate_key_collapses_to_one(self, chunker):
        text = self._glm4_call("notebook_edit", ("cell_id", "same"),
                               ("cell_id", "same"))

        whole = Glm4ToolParser().detect_and_parse(text, self._DUP_TOOLS)
        assert json.loads(whole.calls[0].parameters) == {"cell_id": "same"}

        streamed_text = self._streamed_text(chunker(text), self._DUP_TOOLS)
        assert streamed_text == whole.calls[0].parameters
        assert streamed_text.count('"cell_id"') == 1

    @pytest.mark.parametrize("chunker", [lambda text: [text], list])
    def test_a_duplicate_between_other_arguments_drops_cleanly(self, chunker):
        """The swallowed pair must not leave a stray `, ` on the stream."""
        text = self._glm4_call("notebook_edit", ("count", "1"),
                               ("cell_id", "a"), ("count", "2"))

        whole = Glm4ToolParser().detect_and_parse(text, self._DUP_TOOLS)
        assert json.loads(whole.calls[0].parameters) == {
            "count": 1,
            "cell_id": "a",
        }

        streamed_text = self._streamed_text(chunker(text), self._DUP_TOOLS)
        assert streamed_text == whole.calls[0].parameters

    @pytest.mark.parametrize("value", ["1e309", "-1e309"])
    @pytest.mark.parametrize("chunker", [lambda text: [text], list])
    def test_an_overflowing_number_is_delivered_as_the_text_the_model_wrote(
            self, value, chunker):
        """`1e309` under a number schema must not become `Infinity`.

        json.loads overflows it to float('inf') and json.dumps then emits
        the literal `Infinity` - a token the JSON grammar does not have. If
        the decimal string cannot be represented faithfully as a JSON
        number, the original text is passed through as a string instead.
        """
        text = self._glm4_call("notebook_edit", ("count", value))

        whole = Glm4ToolParser().detect_and_parse(text, self._DUP_TOOLS)
        assert json.loads(whole.calls[0].parameters) == {"count": value}

        streamed_text = self._streamed_text(chunker(text), self._DUP_TOOLS)
        assert streamed_text == whole.calls[0].parameters
        assert "Infinity" not in streamed_text

    @pytest.mark.parametrize("chunker", [lambda text: [text], list])
    def test_the_largest_finite_double_is_still_a_number(self, chunker):
        text = self._glm4_call("notebook_edit", ("count", "1e308"))

        whole = Glm4ToolParser().detect_and_parse(text, self._DUP_TOOLS)
        assert json.loads(whole.calls[0].parameters) == {"count": 1e308}

        streamed = self._streamed_params(chunker(text), self._DUP_TOOLS)
        assert streamed == {"count": 1e308}

    @pytest.mark.parametrize("chunker", [lambda text: [text], list])
    def test_an_object_final_argument_still_closes_the_call(self, chunker):
        """The closing `}` comes from `_is_first_param`, not a text sniff.

        Sniffing the streamed text for a trailing `}` mistook an argument
        whose value is an object for the call being closed already, and
        `{"opts": {"a": 1}` reached the client one `}` short - the same
        defect the GLM-4.7 parser's `_finalize_tool_call` fixed.
        """
        tools = [
            ChatCompletionToolsParam(
                type="function",
                function=FunctionDefinition(
                    name="configure",
                    parameters={
                        "type": "object",
                        "properties": {
                            "opts": {
                                "type": "object"
                            }
                        },
                    },
                ),
            )
        ]
        text = self._glm4_call("configure", ("opts", '{"a": 1}'))

        whole = Glm4ToolParser().detect_and_parse(text, tools)
        assert json.loads(whole.calls[0].parameters) == {"opts": {"a": 1}}

        streamed = self._streamed_params(chunker(text), tools)
        assert streamed == {"opts": {"a": 1}}


# ============================================================================
# Glm47ToolParser Tests
# ============================================================================


class TestGlm47ToolParser(BaseToolParserTestClass):
    """Test suite for Glm47ToolParser class (GLM-4.7/GLM-5 format)."""

    def make_parser(self):
        return Glm47ToolParser()

    def make_tool_parser_test_cases(self):
        # GLM-4.7 format: no newline required between func name and args
        single_text = ("Normal text"
                       "<tool_call>get_weather"
                       "<arg_key>location</arg_key>"
                       "<arg_value>NYC</arg_value>"
                       "</tool_call>")
        single_expected_normal = "Normal text"
        single_expected_name = "get_weather"
        single_expected_params = {"location": "NYC"}

        multiple_text = ("<tool_call>get_weather"
                         "<arg_key>location</arg_key>"
                         "<arg_value>LA</arg_value>"
                         "</tool_call>"
                         "<tool_call>search_web"
                         "<arg_key>query</arg_key>"
                         "<arg_value>AI</arg_value>"
                         "</tool_call>")
        multiple_names = ("get_weather", "search_web")

        # Malformed: no arg_key/arg_value and no closing pattern
        malformed_text = "<tool_call>MALFORMED_NO_ARGS"

        with_parameters_text = ("<tool_call>search_web"
                                "<arg_key>query</arg_key>"
                                "<arg_value>test</arg_value>"
                                "</tool_call>")
        with_parameters_name = "search_web"
        with_parameters_params = {"query": "test"}

        partial_bot_token = "<tool_cal"

        undefined_tool_text = ("<tool_call>undefined_func"
                               "<arg_key>arg</arg_key>"
                               "<arg_value>value</arg_value>"
                               "</tool_call>")

        return ToolParserTestCases(
            has_tool_call_true=
            "Some text <tool_call>get_weather<arg_key>location</arg_key><arg_value>NYC</arg_value></tool_call>",
            detect_and_parse_single_tool=(
                single_text,
                single_expected_normal,
                single_expected_name,
                single_expected_params,
            ),
            detect_and_parse_multiple_tools=(multiple_text, multiple_names),
            detect_and_parse_malformed_tool=malformed_text,
            detect_and_parse_with_parameters_key=(
                with_parameters_text,
                with_parameters_name,
                with_parameters_params,
            ),
            parse_streaming_increment_partial_bot_token=partial_bot_token,
            undefined_tool=undefined_tool_text,
        )

    def test_initialization(self, parser):
        """Test that Glm47ToolParser initializes correctly."""
        assert parser.bot_token == "<tool_call>"
        assert parser.eot_token == "</tool_call>"

    def test_zero_arg_tool_call(self, parser):
        """Test parsing a zero-argument tool call (GLM-4.7 feature)."""
        tools = [
            ChatCompletionToolsParam(
                type="function",
                function=FunctionDefinition(
                    name="get_time",
                    description="Get current time",
                    parameters={
                        "type": "object",
                        "properties": {},
                    },
                ),
            )
        ]
        text = "<tool_call>get_time</tool_call>"

        result = parser.detect_and_parse(text, tools)

        assert len(result.calls) == 1
        assert result.calls[0].name == "get_time"
        assert json.loads(result.calls[0].parameters) == {}

    def test_no_newline_format(self, sample_tools, parser):
        """Test parsing tool call without newline between name and args."""
        text = ("<tool_call>get_weather"
                "<arg_key>location</arg_key>"
                "<arg_value>Tokyo</arg_value>"
                "</tool_call>")

        result = parser.detect_and_parse(text, sample_tools)

        assert len(result.calls) == 1
        assert result.calls[0].name == "get_weather"
        assert json.loads(result.calls[0].parameters) == {"location": "Tokyo"}

    def test_newline_format_also_works(self, sample_tools, parser):
        """Test that GLM-4.5 newline format also works with GLM-4.7 parser."""
        text = ("<tool_call>get_weather\n"
                "<arg_key>location</arg_key>\n"
                "<arg_value>Tokyo</arg_value>\n"
                "</tool_call>")

        result = parser.detect_and_parse(text, sample_tools)

        assert len(result.calls) == 1
        assert result.calls[0].name == "get_weather"
        assert json.loads(result.calls[0].parameters) == {"location": "Tokyo"}

    def test_parse_streaming_increment_complete_tool_call(
            self, sample_tools, parser):
        """Test streaming parser with complete tool call in chunks."""
        # Send bot token with function name and first arg_key
        result = parser.parse_streaming_increment(
            "<tool_call>get_weather<arg_key>", sample_tools)

        # Should send tool name (has_arg_key is True)
        assert len(result.calls) == 1
        assert result.calls[0].name == "get_weather"
        assert result.calls[0].parameters == ""

        # Send arguments
        result = parser.parse_streaming_increment(
            "location</arg_key>"
            "<arg_value>SF</arg_value>"
            "</tool_call>", sample_tools)

        # Should stream arguments and complete the tool call
        all_params = "".join(call.parameters for call in result.calls
                             if call.parameters)
        assert "location" in all_params
        assert "SF" in all_params

    def test_parse_streaming_increment_multiple_tools_streaming(
            self, sample_tools, parser):
        """Test streaming parser handles multiple tool calls."""
        # First tool
        parser.parse_streaming_increment(
            "<tool_call>get_weather<arg_key>location</arg_key>"
            "<arg_value>NYC</arg_value></tool_call>", sample_tools)

        # Second tool
        result = parser.parse_streaming_increment(
            "<tool_call>search_web<arg_key>", sample_tools)

        # Should have started second tool
        assert len(result.calls) == 1
        assert result.calls[0].name == "search_web"
        assert result.calls[0].parameters == ""
        assert result.calls[0].tool_index == 1

    def test_streaming_zero_arg_tool(self, parser):
        """Test streaming a zero-argument tool call."""
        tools = [
            ChatCompletionToolsParam(
                type="function",
                function=FunctionDefinition(
                    name="get_time",
                    description="Get current time",
                    parameters={
                        "type": "object",
                        "properties": {},
                    },
                ),
            )
        ]

        # Send the complete zero-arg tool call
        result = parser.parse_streaming_increment(
            "<tool_call>get_time</tool_call>", tools)

        names = [c.name for c in result.calls if c.name]
        assert "get_time" in names

        # Should have sent empty object for no-arg function
        params = "".join(c.parameters for c in result.calls)
        assert "{}" in params

    def test_detect_and_parse_multiple_params(self, sample_tools):
        """Test one-shot parsing with multiple parameters."""
        parser = Glm47ToolParser()
        text = ("<tool_call>get_weather"
                "<arg_key>location</arg_key>"
                "<arg_value>Tokyo</arg_value>"
                "<arg_key>unit</arg_key>"
                "<arg_value>celsius</arg_value>"
                "</tool_call>")

        result = parser.detect_and_parse(text, sample_tools)

        assert len(result.calls) == 1
        assert result.calls[0].name == "get_weather"
        params = json.loads(result.calls[0].parameters)
        assert params == {"location": "Tokyo", "unit": "celsius"}

    def test_detect_and_parse_with_number_type(self):
        """Test parsing with number type coercion."""
        parser = Glm47ToolParser()
        tools = [
            ChatCompletionToolsParam(
                type="function",
                function=FunctionDefinition(
                    name="set_temperature",
                    description="Set temperature",
                    parameters={
                        "type": "object",
                        "properties": {
                            "value": {
                                "type": "number",
                            },
                            "label": {
                                "type": "string",
                            },
                        },
                        "required": ["value"],
                    },
                ),
            )
        ]

        text = ("<tool_call>set_temperature"
                "<arg_key>value</arg_key>"
                "<arg_value>72.5</arg_value>"
                "<arg_key>label</arg_key>"
                "<arg_value>room temp</arg_value>"
                "</tool_call>")

        result = parser.detect_and_parse(text, tools)

        assert len(result.calls) == 1
        params = json.loads(result.calls[0].parameters)
        assert params["value"] == 72.5
        assert params["label"] == "room temp"

    def test_supports_structural_tag(self, parser):
        """Test that supports_structural_tag returns False."""
        assert parser.supports_structural_tag() is False

    def test_normal_text_before_tool_call(self, sample_tools, parser):
        """Test that text before tool call is returned as normal_text."""
        text = ("Here is the weather info "
                "<tool_call>get_weather"
                "<arg_key>location</arg_key>"
                "<arg_value>NYC</arg_value>"
                "</tool_call>")

        result = parser.detect_and_parse(text, sample_tools)

        assert "Here is the weather info" in result.normal_text
        assert len(result.calls) == 1

    def test_text_between_tool_calls(self, sample_tools, parser):
        """Test that text between tool calls is preserved as normal_text."""
        text = ("<tool_call>get_weather"
                "<arg_key>location</arg_key>"
                "<arg_value>NYC</arg_value>"
                "</tool_call>"
                " some text between "
                "<tool_call>search_web"
                "<arg_key>query</arg_key>"
                "<arg_value>AI</arg_value>"
                "</tool_call>")

        result = parser.detect_and_parse(text, sample_tools)

        assert "some text between" in result.normal_text
        assert len(result.calls) == 2


# ============================================================================
# Glm47ToolParser — scenarios ported from sglang's reference test suite
# (sgl-project/sglang :: test/registered/unit/function_call/test_glm47_moe_detector.py).
# These are the edge cases most likely to surface in real GLM-5 MTP output.
# ============================================================================


class TestGlm47ToolParserSglangSuite:
    """Port of sglang's Glm47MoeDetector tests against our Glm47ToolParser."""

    @staticmethod
    def _tool(name, properties, required=None):
        params = {"type": "object", "properties": properties}
        if required is not None:
            params["required"] = required
        return ChatCompletionToolsParam(
            type="function",
            function=FunctionDefinition(name=name, parameters=params),
        )

    @staticmethod
    def _stream(parser, chunks, tools):
        calls = []
        normal = ""
        for chunk in chunks:
            result = parser.parse_streaming_increment(chunk, tools)
            calls.extend(result.calls)
            normal += result.normal_text or ""
        return calls, normal

    def test_mtp_func_and_string_split(self):
        """MTP splits the function name and string values mid-word."""
        tools = [
            self._tool(
                "create_task",
                {
                    "title": {
                        "type": "string"
                    },
                    "location": {
                        "type": "string"
                    },
                },
            )
        ]
        chunks = [
            "I'll create a task.",
            "<tool_call>create_ta",
            "sk<arg_key>title</arg_key><arg_value>Go to Bei",
            "jing</arg_value>",
            "<arg_key>location</arg_key><arg_value>San Fran",
            "cisco</arg_value></tool_call>",
        ]
        calls, normal = self._stream(Glm47ToolParser(), chunks, tools)

        assert "I'll create a task." in normal
        names = [c.name for c in calls if c.name]
        assert names == ["create_task"]
        params = json.loads("".join(c.parameters for c in calls
                                    if c.parameters))
        assert params == {"title": "Go to Beijing", "location": "San Francisco"}

    def test_mtp_noarg_and_multiple_calls(self):
        """No-arg call followed by regular call: state must reset cleanly."""
        tools = [
            self._tool("list_files", {}),
            self._tool("get_weather", {"city": {
                "type": "string"
            }}),
        ]
        chunks = [
            "<tool_call>list_files</tool_call>",
            "<tool_call>get_weather<arg_key>city</arg_key>"
            "<arg_value>Beijing</arg_value></tool_call>",
        ]
        calls, _ = self._stream(Glm47ToolParser(), chunks, tools)

        names = [c.name for c in calls if c.name]
        assert names == ["list_files", "get_weather"]

        empty_calls = [c for c in calls if c.parameters == "{}"]
        assert len(empty_calls) <= 1, \
            "No-arg function should emit at most one '{}'"

        weather_params = "".join(c.parameters for c in calls
                                 if c.parameters and c.tool_index == 1)
        assert json.loads(weather_params) == {"city": "Beijing"}

    def test_mtp_number_and_complex_json(self):
        """Numbers preserved as numbers; JSON array reassembled across splits."""
        tools = [
            self._tool(
                "create_todos",
                {
                    "priority": {
                        "type": "number"
                    },
                    "count": {
                        "type": "integer"
                    },
                    "items": {
                        "type": "array"
                    },
                },
            )
        ]
        chunks = [
            "<tool_call>create_todos",
            "<arg_key>priority</arg_key><arg_value>5.5</arg_value>",
            "<arg_key>count</arg_key><arg_value>10</arg_value>",
            '<arg_key>items</arg_key><arg_value>[{"description',
            '": "Test',
            'Todo 1"}, {"description": "TestTodo 2"}]</arg_value></tool_call>',
        ]
        calls, _ = self._stream(Glm47ToolParser(), chunks, tools)

        names = [c.name for c in calls if c.name]
        assert names == ["create_todos"]

        params = json.loads("".join(c.parameters for c in calls
                                    if c.parameters))
        assert params["priority"] == 5.5
        assert isinstance(params["priority"], (int, float))
        assert params["count"] == 10
        assert isinstance(params["count"], int)
        assert isinstance(params["items"], list)
        assert len(params["items"]) == 2
        assert params["items"][0]["description"] == "TestTodo 1"
        assert params["items"][1]["description"] == "TestTodo 2"

    def test_array_argument_with_escaped_json(self):
        r"""Arrays with escaped quotes, Windows paths, literal \n."""
        tools = [self._tool("todo_write", {"todos": {"type": "array"}})]
        parser = Glm47ToolParser()

        text = ('<tool_call>todo_write<arg_key>todos</arg_key><arg_value>'
                '[{"id": "1", "task": "Check file at C:\\\\Users\\\\test.txt", '
                '"status": "pending"}]'
                '</arg_value></tool_call>')
        result = parser.detect_and_parse(text, tools)
        params = json.loads(result.calls[0].parameters)
        assert params["todos"][0]["task"] == r"Check file at C:\Users\test.txt"

        parser = Glm47ToolParser()
        text = ('<tool_call>todo_write<arg_key>todos</arg_key><arg_value>'
                '[{"id": "1", "task": "Print \\\\n to see newline",'
                '"status": "pending"}]'
                '</arg_value></tool_call>')
        result = parser.detect_and_parse(text, tools)
        params = json.loads(result.calls[0].parameters)
        assert params["todos"][0]["task"] == r"Print \n to see newline"

    def test_boundary_param_value_extreme_split(self):
        """Worst-case: one character per chunk."""
        tools = [self._tool("search", {"query": {"type": "string"}})]
        chunks = [
            "<tool_call>search<arg_key>query</arg_key><arg_value>N",
            "e",
            "w ",
            "Y",
            "o",
            "rk</arg_value></tool_call>",
        ]
        calls, _ = self._stream(Glm47ToolParser(), chunks, tools)
        params = json.loads("".join(c.parameters for c in calls
                                    if c.parameters))
        assert params == {"query": "New York"}

    def test_boundary_empty_param_value(self):
        """Empty string values are preserved."""
        tools = [
            self._tool(
                "create_note",
                {
                    "title": {
                        "type": "string"
                    },
                    "content": {
                        "type": "string"
                    },
                },
            )
        ]
        text = ("<tool_call>create_note"
                "<arg_key>title</arg_key><arg_value>Test</arg_value>"
                "<arg_key>content</arg_key><arg_value></arg_value>"
                "</tool_call>")
        result = Glm47ToolParser().detect_and_parse(text, tools)
        params = json.loads(result.calls[0].parameters)
        assert params == {"title": "Test", "content": ""}

    def test_boundary_json_empty_structures(self):
        """Empty {} and [] as argument values shouldn't collide with no-arg '{}'."""
        tools = [
            self._tool(
                "create_structure",
                {
                    "empty_obj": {
                        "type": "object"
                    },
                    "empty_arr": {
                        "type": "array"
                    },
                },
            )
        ]
        text = ("<tool_call>create_structure"
                "<arg_key>empty_obj</arg_key><arg_value>{}</arg_value>"
                "<arg_key>empty_arr</arg_key><arg_value>[]</arg_value>"
                "</tool_call>")
        result = Glm47ToolParser().detect_and_parse(text, tools)
        params = json.loads(result.calls[0].parameters)
        assert params == {"empty_obj": {}, "empty_arr": []}

    def test_boundary_number_edge_values(self):
        """Zero, negative, scientific notation preserved as numbers."""
        tools = [
            self._tool(
                "calculate",
                {
                    "zero": {
                        "type": "number"
                    },
                    "negative": {
                        "type": "number"
                    },
                    "large": {
                        "type": "number"
                    },
                },
            )
        ]
        text = ("<tool_call>calculate"
                "<arg_key>zero</arg_key><arg_value>0</arg_value>"
                "<arg_key>negative</arg_key><arg_value>-42.5</arg_value>"
                "<arg_key>large</arg_key><arg_value>1e10</arg_value>"
                "</tool_call>")
        result = Glm47ToolParser().detect_and_parse(text, tools)
        params = json.loads(result.calls[0].parameters)
        assert params["zero"] == 0
        assert params["negative"] == -42.5
        assert params["large"] == 1e10

    def test_boundary_type_string_with_numeric_content(self):
        """Schema says string -> numeric-looking content stays string."""
        tools = [
            self._tool(
                "store_data",
                {
                    "id": {
                        "type": "string"
                    },
                    "code": {
                        "type": "string"
                    },
                },
            )
        ]
        text = ("<tool_call>store_data"
                "<arg_key>id</arg_key><arg_value>12345</arg_value>"
                "<arg_key>code</arg_key><arg_value>67.89</arg_value>"
                "</tool_call>")
        result = Glm47ToolParser().detect_and_parse(text, tools)
        params = json.loads(result.calls[0].parameters)
        assert isinstance(params["id"], str) and params["id"] == "12345"
        assert isinstance(params["code"], str) and params["code"] == "67.89"

    def test_error_undefined_tool(self, sample_tools):
        """Undefined tool names parse cleanly with tool_index=-1.

        TRT-LLM base behavior: warn + emit, unlike sglang which drops them.
        """
        text = ("<tool_call>nonexistent_function"
                "<arg_key>param</arg_key><arg_value>value</arg_value>"
                "</tool_call>")
        result = Glm47ToolParser().detect_and_parse(text, sample_tools)
        assert len(result.calls) == 1
        assert result.calls[0].name == "nonexistent_function"
        assert result.calls[0].tool_index == -1
        assert json.loads(result.calls[0].parameters) == {"param": "value"}

    def test_error_incomplete_buffer_at_end(self, sample_tools):
        """Stream ends mid-parse: no exception, returns a valid result."""
        parser = Glm47ToolParser()
        result = parser.parse_streaming_increment(
            "<tool_call>get_weather<arg_key>location</arg_key>"
            "<arg_value>Beijing", sample_tools)
        assert isinstance(result, StreamingParseResult)


class TestGlm47ToolParserFactory:
    """Test that GLM-4.7 parser is registered in the factory."""

    def test_glm47_registered(self):
        """Test that glm47 parser is registered in factory."""
        from tensorrt_llm.serve.tool_parser.tool_parser_factory import \
            ToolParserFactory
        assert "glm47" in ToolParserFactory.parsers

    def test_create_glm47_parser(self):
        """Test creating glm47 parser via factory."""
        from tensorrt_llm.serve.tool_parser.tool_parser_factory import \
            ToolParserFactory
        parser = ToolParserFactory.create_tool_parser("glm47")
        assert isinstance(parser, Glm47ToolParser)


# ============================================================================
# PoolsideV1ToolParser Tests
# ============================================================================


class TestPoolsideV1ToolParser(BaseToolParserTestClass):
    """Test suite for Poolside Laguna v1 tool calls."""

    def make_parser(self):
        return PoolsideV1ToolParser()

    def make_tool_parser_test_cases(self):
        return ToolParserTestCases(
            has_tool_call_true=("Some text <tool_call>get_weather\n"
                                "<arg_key>location</arg_key>\n"
                                "<arg_value>NYC</arg_value>\n"
                                "</tool_call>"),
            detect_and_parse_single_tool=(
                ("Normal text\n"
                 "<tool_call>get_weather\n"
                 "<arg_key>location</arg_key>\n"
                 "<arg_value>NYC</arg_value>\n"
                 "</tool_call>"),
                "Normal text",
                "get_weather",
                {
                    "location": "NYC"
                },
            ),
            detect_and_parse_multiple_tools=(
                ("<tool_call>get_weather\n"
                 "<arg_key>location</arg_key>\n"
                 "<arg_value>LA</arg_value>\n"
                 "</tool_call>\n"
                 "<tool_call>search_web\n"
                 "<arg_key>query</arg_key>\n"
                 "<arg_value>AI</arg_value>\n"
                 "</tool_call>"),
                ("get_weather", "search_web"),
            ),
            detect_and_parse_malformed_tool=("<tool_call>get_weather\n"
                                             "<arg_key>location</arg_key>\n"
                                             "<arg_value>NYC</arg_value>"),
            detect_and_parse_with_parameters_key=(
                ("<tool_call>search_web\n"
                 "<arg_key>query</arg_key>\n"
                 "<arg_value>test</arg_value>\n"
                 "</tool_call>"),
                "search_web",
                {
                    "query": "test"
                },
            ),
            parse_streaming_increment_partial_bot_token="<tool",
            undefined_tool=("<tool_call>undefined_func\n"
                            "<arg_key>arg</arg_key>\n"
                            "<arg_value>value</arg_value>\n"
                            "</tool_call>"),
        )

    def test_initialization(self, parser):
        assert parser.bot_token == "<tool_call>"
        assert parser.eot_token == "</tool_call>"
        assert parser.supports_structural_tag() is False

    def test_no_newline_format(self, sample_tools, parser):
        text = ("<tool_call>get_weather"
                "<arg_key>location</arg_key>"
                "<arg_value>Tokyo</arg_value>"
                "</tool_call>")

        result = parser.detect_and_parse(text, sample_tools)

        assert len(result.calls) == 1
        assert result.calls[0].name == "get_weather"
        assert json.loads(result.calls[0].parameters) == {"location": "Tokyo"}

    def test_zero_arg_tool_call(self, parser):
        tools = [
            ChatCompletionToolsParam(
                type="function",
                function=FunctionDefinition(
                    name="get_time",
                    description="Get current time",
                    parameters={
                        "type": "object",
                        "properties": {},
                    },
                ),
            )
        ]
        text = "<tool_call>get_time</tool_call>"

        result = parser.detect_and_parse(text, tools)

        assert len(result.calls) == 1
        assert result.calls[0].name == "get_time"
        assert json.loads(result.calls[0].parameters) == {}

    def test_detect_and_parse_preserves_suffix(self, sample_tools, parser):
        text = ("prefix "
                "<tool_call>get_weather\n"
                "<arg_key>location</arg_key>\n"
                "<arg_value>NYC</arg_value>\n"
                "</tool_call>"
                " suffix")

        result = parser.detect_and_parse(text, sample_tools)

        assert "prefix" in result.normal_text
        assert "suffix" in result.normal_text
        assert len(result.calls) == 1

    def test_detect_and_parse_allows_end_tag_text_in_string_arg(
            self, sample_tools, parser):
        text = ("<tool_call>search_web\n"
                "<arg_key>query</arg_key>\n"
                "<arg_value>literal </tool_call> marker</arg_value>\n"
                "</tool_call>")

        result = parser.detect_and_parse(text, sample_tools)

        assert result.normal_text == ""
        assert len(result.calls) == 1
        assert result.calls[0].name == "search_web"
        assert json.loads(result.calls[0].parameters) == {
            "query": "literal </tool_call> marker"
        }

    def test_schema_aware_argument_coercion(self, parser):
        tools = [
            ChatCompletionToolsParam(
                type="function",
                function=FunctionDefinition(
                    name="set_values",
                    description="Set typed values",
                    parameters={
                        "type": "object",
                        "properties": {
                            "string_value": {
                                "type": "string"
                            },
                            "integer_value": {
                                "type": "integer"
                            },
                            "number_value": {
                                "type": "number"
                            },
                            "boolean_value": {
                                "type": "boolean"
                            },
                            "array_value": {
                                "type": "array"
                            },
                            "object_value": {
                                "type": "object"
                            },
                        },
                    },
                ),
            )
        ]
        text = ("<tool_call>set_values\n"
                "<arg_key>string_value</arg_key>\n"
                "<arg_value>true</arg_value>\n"
                "<arg_key>integer_value</arg_key>\n"
                "<arg_value>42</arg_value>\n"
                "<arg_key>number_value</arg_key>\n"
                "<arg_value>3.5</arg_value>\n"
                "<arg_key>boolean_value</arg_key>\n"
                "<arg_value>true</arg_value>\n"
                "<arg_key>array_value</arg_key>\n"
                "<arg_value>[1, 2]</arg_value>\n"
                "<arg_key>object_value</arg_key>\n"
                '<arg_value>{"k": 1}</arg_value>\n'
                "</tool_call>")

        result = parser.detect_and_parse(text, tools)
        params = json.loads(result.calls[0].parameters)

        assert params == {
            "string_value": "true",
            "integer_value": 42,
            "number_value": 3.5,
            "boolean_value": True,
            "array_value": [1, 2],
            "object_value": {
                "k": 1
            },
        }

    def test_parse_streaming_increment_split_tags(self, sample_tools, parser):
        result = parser.parse_streaming_increment("<tool", sample_tools)
        assert result.normal_text == ""
        assert len(result.calls) == 0

        result = parser.parse_streaming_increment("_call>get_weather\n<arg",
                                                  sample_tools)
        assert result.normal_text == ""
        assert len(result.calls) == 0

        result = parser.parse_streaming_increment(
            "_key>location</arg_key>"
            "<arg_value>SF</arg_value>"
            "</tool_call>", sample_tools)
        assert len(result.calls) == 2
        assert result.calls[0].tool_index == 0
        assert result.calls[0].name == "get_weather"
        assert result.calls[0].parameters == ""
        assert result.calls[1].tool_index == 0
        assert json.loads(result.calls[1].parameters) == {"location": "SF"}

    def test_parse_streaming_increment_buffers_truncated_tool_call(
            self, sample_tools, parser):
        result = parser.parse_streaming_increment(
            "<tool_call>get_weather\n"
            "<arg_key>location</arg_key>\n"
            "<arg_value>SF",
            sample_tools,
        )

        assert result.normal_text == ""
        assert len(result.calls) == 0

    def test_parse_streaming_increment_allows_end_tag_text_in_string_arg(
            self, sample_tools, parser):
        result = parser.parse_streaming_increment(
            "<tool_call>search_web\n"
            "<arg_key>query</arg_key>\n"
            "<arg_value>literal </tool_call> marker</arg_value>\n"
            "</tool_call>",
            sample_tools,
        )

        assert len(result.calls) == 2
        assert result.calls[0].tool_index == 0
        assert result.calls[0].name == "search_web"
        assert result.calls[0].parameters == ""
        assert result.calls[1].tool_index == 0
        assert json.loads(result.calls[1].parameters) == {
            "query": "literal </tool_call> marker"
        }

    def test_parse_streaming_increment_multiple_tools(self, sample_tools,
                                                      parser):
        result = parser.parse_streaming_increment(
            "<tool_call>get_weather\n"
            "<arg_key>location</arg_key>\n"
            "<arg_value>NYC</arg_value>\n"
            "</tool_call>"
            "<tool_call>search_web\n"
            "<arg_key>query</arg_key>\n"
            "<arg_value>AI</arg_value>\n"
            "</tool_call>",
            sample_tools,
        )

        assert [call.name for call in result.calls if call.name] == [
            "get_weather",
            "search_web",
        ]
        assert [call.tool_index for call in result.calls if call.name] == [0, 1]
        params = [
            json.loads(call.parameters) for call in result.calls
            if call.parameters
        ]
        assert params == [{"location": "NYC"}, {"query": "AI"}]


class TestPoolsideV1ToolParserFactory:
    """Test that Poolside v1 parser is registered in the factory."""

    def test_poolside_v1_registered(self):
        from tensorrt_llm.serve.tool_parser.tool_parser_factory import \
            ToolParserFactory
        assert "poolside_v1" in ToolParserFactory.parsers

    def test_create_poolside_v1_parser(self):
        from tensorrt_llm.serve.tool_parser.tool_parser_factory import \
            ToolParserFactory
        parser = ToolParserFactory.create_tool_parser("poolside_v1")
        assert isinstance(parser, PoolsideV1ToolParser)

    def test_laguna_model_type_mapping(self):
        from tensorrt_llm.serve.tool_parser.tool_parser_factory import \
            MODEL_TYPE_TO_TOOL_PARSER
        assert MODEL_TYPE_TO_TOOL_PARSER["laguna"] == "poolside_v1"

    def test_auto_detect_laguna(self, tmp_path):
        from tensorrt_llm.serve.tool_parser.tool_parser_factory import \
            resolve_auto_tool_parser
        model_dir = tmp_path / "Laguna"
        model_dir.mkdir()
        (model_dir / "config.json").write_text(
            json.dumps({"model_type": "laguna"}))

        assert resolve_auto_tool_parser(str(model_dir)) == "poolside_v1"


# ============================================================================
# Integration Tests
# ============================================================================


class TestToolParserIntegration:
    """Integration tests for tool parsers."""

    def test_qwen3_5_reasoning_plus_qwen3_tool_parser_bare_json_pipeline(
            self, sample_tools):
        r"""NVBug 6240584: end-to-end reasoning + tool parser pipeline.

        Reproduces the exact scenario from the bug report: the Qwen3.6 FP8
        chat template pre-injects `<think>\n` into the assistant prompt
        prefix, so the model output starts *inside* the reasoning block
        with no opening `<think>` tag. Content up to `</think>` is the
        reasoning, and what follows is a bare JSON tool call (no
        `<tool_call>` wrapper). The `qwen3_5` reasoning parser (registered
        with `reasoning_at_start=True`) strips the thinking block, then
        the `qwen3` tool parser must recover the tool call so
        `args.has_tool_call[0]` is True and downstream logic sets
        `finish_reason="tool_calls"` (see chat_response_post_processor).
        """
        from tensorrt_llm.serve.openai_protocol import ChatCompletionRequest
        from tensorrt_llm.serve.postprocess_handlers import (
            ChatPostprocArgs, apply_reasoning_parser, apply_tool_parser)

        # Bug-report input: reasoning content (no leading `<think>` — the
        # chat template already injected it into the prompt prefix) followed
        # by a bare JSON tool call.
        text = ('Reasoning here.</think>\n'
                '{"name":"get_weather","arguments":{"city":"Paris"}}')

        # Build a minimal request so we can construct ChatPostprocArgs.
        req = ChatCompletionRequest(
            model="Qwen/Qwen3.6-27B-FP8",
            messages=[{
                "role": "user",
                "content": "What is the weather in Paris?"
            }],
            tools=sample_tools,
        )
        args = ChatPostprocArgs.from_request(req)
        args.reasoning_parser = "qwen3_5"
        args.tool_parser = "qwen3"

        # Non-streaming path.
        content, reasoning_content = apply_reasoning_parser(args,
                                                            output_index=0,
                                                            text=text,
                                                            streaming=False)
        assert reasoning_content == "Reasoning here."
        # The reasoning parser strips `<think>...</think>` — the remaining
        # content is the bare JSON, possibly with a leading newline.
        assert '"name":"get_weather"' in content

        normal_text, calls = apply_tool_parser(args,
                                               output_index=0,
                                               text=content,
                                               streaming=False)

        assert len(calls) == 1
        assert calls[0].name == "get_weather"
        assert json.loads(calls[0].parameters) == {"city": "Paris"}
        # Downstream (chat_response_post_processor) checks this flag to flip
        # finish_reason from "stop" to "tool_calls".
        assert args.has_tool_call.get(0) is True
        # And no bare JSON leaks into the visible content.
        assert normal_text == ""

    def test_qwen3_5_reasoning_plus_qwen3_tool_parser_bare_json_streaming(
            self, sample_tools):
        r"""NVBug 6240584: streaming variant of reasoning+tool parser pipeline.

        The bug most commonly reproduces on streamed chat completions —
        the model emits tokens one at a time and the OpenAI server relies
        on the tool parser to flip `finish_reason` to `tool_calls` before
        the stream ends. Feed the same reasoning + bare-JSON payload
        through `apply_reasoning_parser` / `apply_tool_parser` with
        `streaming=True` in small chunks and assert:
          - `args.has_tool_call[0]` is True at end-of-stream,
          - the accumulated tool-call name/arguments are correct,
          - the bare JSON never leaks into visible content.
        """
        from tensorrt_llm.serve.openai_protocol import ChatCompletionRequest
        from tensorrt_llm.serve.postprocess_handlers import (
            ChatPostprocArgs, apply_reasoning_parser, apply_tool_parser)

        text = ('Reasoning here.</think>\n'
                '{"name":"get_weather","arguments":{"city":"Paris"}}')

        # Chunk the input to force the streaming state machines to buffer
        # across boundaries. The split intentionally lands inside both the
        # `</think>` tag and the JSON payload.
        chunks = [
            'Reasoning ',
            'here.</thi',
            'nk>\n{"name":"get_',
            'weather","arguments":',
            '{"city":"Pa',
            'ris"}}',
        ]

        req = ChatCompletionRequest(
            model="Qwen/Qwen3.6-27B-FP8",
            messages=[{
                "role": "user",
                "content": "What is the weather in Paris?"
            }],
            tools=sample_tools,
        )
        args = ChatPostprocArgs.from_request(req)
        args.reasoning_parser = "qwen3_5"
        args.tool_parser = "qwen3"

        accumulated_content = ""
        accumulated_normal_text = ""
        collected_calls = []

        for chunk in chunks:
            content, _reasoning = apply_reasoning_parser(args,
                                                         output_index=0,
                                                         text=chunk,
                                                         streaming=True)
            accumulated_content += content
            if not content:
                continue
            normal_text, calls = apply_tool_parser(args,
                                                   output_index=0,
                                                   text=content,
                                                   streaming=True)
            if normal_text:
                accumulated_normal_text += normal_text
            collected_calls.extend(calls)

        # The reasoning parser must have stripped everything through the
        # `</think>` tag; the bare JSON survives into content.
        assert '"name":"get_weather"' in accumulated_content

        # The tool parser must have flipped `has_tool_call` before the
        # stream ended — this is the exact condition
        # `chat_response_post_processor` uses to set
        # `finish_reason="tool_calls"`.
        assert args.has_tool_call.get(0) is True

        # We must have received the tool name and its arguments (potentially
        # across multiple streaming increments).
        names = [c.name for c in collected_calls if c.name]
        assert names == ["get_weather"]
        params = "".join(c.parameters for c in collected_calls if c.parameters)
        assert json.loads(params) == {"city": "Paris"}

        # And no visible content is leaked from the bare-JSON payload.
        assert accumulated_normal_text == ""

    def test_end_to_end_single_tool(self, sample_tools):
        """Test end-to-end parsing of a single tool call."""
        parser = Qwen3ToolParser()

        # Simulate streaming
        chunks = [
            "<tool_call>\n", '{"name":"get', '_weather"', ',"arguments":',
            '{"location"', ':"Paris"}}\n', '</tool_call>'
        ]

        results = []
        for chunk in chunks:
            result = parser.parse_streaming_increment(chunk, sample_tools)
            if result.calls or result.normal_text:
                results.append(result)

        # Should have received tool name and arguments
        assert any(r.calls for r in results)

    def test_mixed_content_and_tool_calls(self, sample_tools):
        """Test parsing text that mixes normal content with tool calls."""
        parser = Qwen3ToolParser()

        text = (
            'I will check the weather for you.\n'
            '<tool_call>\n{"name":"get_weather","arguments":{"location":"London"}}\n</tool_call>\n'
            'Let me search that for you.')

        result = parser.detect_and_parse(text, sample_tools)

        assert "I will check the weather for you." in result.normal_text
        assert len(result.calls) == 1
        assert result.calls[0].name == "get_weather"

    def test_parser_state_reset(self, sample_tools):
        """Test that parser state can be used for multiple requests."""
        parser = Qwen3ToolParser()

        # First request
        result1 = parser.detect_and_parse(
            '<tool_call>\n{"name":"get_weather","arguments":{"location":"NYC"}}\n</tool_call>',
            sample_tools)

        # Reset internal state for new request
        parser2 = Qwen3ToolParser()

        # Second request
        result2 = parser2.detect_and_parse(
            '<tool_call>\n{"name":"search_web","arguments":{"query":"test"}}\n</tool_call>',
            sample_tools)

        assert result1.calls[0].name == "get_weather"
        assert result2.calls[0].name == "search_web"


# ============================================================================
# MiniMaxM2ToolParser Tests
# ============================================================================


class TestMiniMaxM2ToolParser:
    """Test suite for MiniMaxM2ToolParser class."""

    @pytest.fixture
    def parser(self):
        return MiniMaxM2ToolParser()

    def test_initialization(self, parser):
        """Test that MiniMaxM2ToolParser initializes correctly."""
        assert parser.bot_token == "<minimax:tool_call>"
        assert parser.eot_token == "</minimax:tool_call>"

    def test_has_tool_call_true(self, parser):
        """Test has_tool_call returns True when tool call present."""
        text = '<minimax:tool_call><invoke name="get_weather"><parameter name="location">NYC</parameter></invoke></minimax:tool_call>'
        assert parser.has_tool_call(text) is True

    def test_has_tool_call_false(self, parser):
        """Test has_tool_call returns False when no tool call present."""
        text = "Just some regular text without tool calls"
        assert parser.has_tool_call(text) is False

    def test_detect_and_parse_single_tool(self, sample_tools, parser):
        """Test detect_and_parse with a single tool call."""
        text = ('I will check the weather.'
                '<minimax:tool_call>'
                '<invoke name="get_weather">'
                '<parameter name="location">NYC</parameter>'
                '</invoke>'
                '</minimax:tool_call>')

        result = parser.detect_and_parse(text, sample_tools)

        assert "I will check the weather." in result.normal_text
        assert len(result.calls) == 1
        assert result.calls[0].name == "get_weather"
        params = json.loads(result.calls[0].parameters)
        assert params == {"location": "NYC"}

    def test_detect_and_parse_multiple_tools(self, sample_tools, parser):
        """Test detect_and_parse with multiple tool calls (parallel)."""
        text = ('<minimax:tool_call>'
                '<invoke name="get_weather">'
                '<parameter name="location">NYC</parameter>'
                '</invoke>'
                '<invoke name="search_web">'
                '<parameter name="query">AI news</parameter>'
                '</invoke>'
                '</minimax:tool_call>')

        result = parser.detect_and_parse(text, sample_tools)

        assert len(result.calls) == 2
        assert result.calls[0].name == "get_weather"
        assert result.calls[1].name == "search_web"
        assert json.loads(result.calls[0].parameters) == {"location": "NYC"}
        assert json.loads(result.calls[1].parameters) == {"query": "AI news"}

    def test_detect_and_parse_no_tool_call(self, sample_tools, parser):
        """Test detect_and_parse with text containing no tool calls."""
        text = "This is just a regular response."

        result = parser.detect_and_parse(text, sample_tools)

        assert result.normal_text == text
        assert len(result.calls) == 0

    def test_detect_and_parse_with_thinking(self, sample_tools, parser):
        """Test detect_and_parse with interleaved thinking content before tool call."""
        text = ('Let me think about this...\n'
                '<minimax:tool_call>'
                '<invoke name="search_web">'
                '<parameter name="query">weather forecast</parameter>'
                '</invoke>'
                '</minimax:tool_call>')

        result = parser.detect_and_parse(text, sample_tools)

        assert "Let me think about this..." in result.normal_text
        assert len(result.calls) == 1
        assert result.calls[0].name == "search_web"

    def test_detect_and_parse_multiple_params(self, sample_tools, parser):
        """Test parsing with multiple parameters."""
        text = ('<minimax:tool_call>'
                '<invoke name="get_weather">'
                '<parameter name="location">Tokyo</parameter>'
                '<parameter name="unit">celsius</parameter>'
                '</invoke>'
                '</minimax:tool_call>')

        result = parser.detect_and_parse(text, sample_tools)

        assert len(result.calls) == 1
        params = json.loads(result.calls[0].parameters)
        assert params == {"location": "Tokyo", "unit": "celsius"}

    def test_detect_and_parse_no_params(self):
        """Test parsing tool call with no parameters."""
        parser = MiniMaxM2ToolParser()
        tools = [
            ChatCompletionToolsParam(
                type="function",
                function=FunctionDefinition(
                    name="get_time",
                    description="Get current time",
                    parameters={
                        "type": "object",
                        "properties": {},
                    },
                ),
            )
        ]
        text = ('<minimax:tool_call>'
                '<invoke name="get_time">'
                '</invoke>'
                '</minimax:tool_call>')

        result = parser.detect_and_parse(text, tools)

        assert len(result.calls) == 1
        assert result.calls[0].name == "get_time"
        assert json.loads(result.calls[0].parameters) == {}

    def test_parse_streaming_increment_normal_text(self, sample_tools, parser):
        """Test streaming parser handles normal text without tool calls."""
        text = "Hello, how can I help?"

        result = parser.parse_streaming_increment(text, sample_tools)

        assert result.normal_text == text
        assert len(result.calls) == 0

    def test_parse_streaming_increment_partial_bot_token(
            self, sample_tools, parser):
        """Test streaming parser buffers partial bot token."""
        result = parser.parse_streaming_increment("<minimax:tool_cal",
                                                  sample_tools)

        assert result.normal_text == ""
        assert len(result.calls) == 0

    def test_parse_streaming_increment_complete_tool(self, sample_tools):
        """Test streaming parser with complete tool call."""
        parser = MiniMaxM2ToolParser()

        # Send the complete tool call
        result = parser.parse_streaming_increment(
            '<minimax:tool_call>'
            '<invoke name="get_weather">'
            '<parameter name="location">NYC</parameter>'
            '</invoke>'
            '</minimax:tool_call>', sample_tools)

        # Should have parsed the tool call
        names = [c.name for c in result.calls if c.name]
        assert "get_weather" in names

    def test_supports_structural_tag(self, parser):
        """Test that supports_structural_tag returns False."""
        assert parser.supports_structural_tag() is False

    def test_parse_param_value_string_not_coerced(self):
        """Test that string-typed params are not coerced by json.loads."""
        from tensorrt_llm.serve.tool_parser.minimax_m2_parser import \
            _parse_param_value

        # Values that json.loads would coerce if not short-circuited.
        assert _parse_param_value("42", "string") == "42"
        assert _parse_param_value("true", "string") == "true"
        assert _parse_param_value("null", "string") == "null"
        assert _parse_param_value('{"k": 1}', "string") == '{"k": 1}'
        assert _parse_param_value("[1,2]", "string") == "[1,2]"
        # Non-string types should still be parsed.
        assert _parse_param_value("42", "integer") == 42
        assert _parse_param_value("3.14", "number") == 3.14
        assert _parse_param_value("true", "boolean") is True
        assert _parse_param_value('{"k": 1}', "object") == {"k": 1}

    def test_detect_and_parse_preserves_suffix(self, sample_tools):
        """Test that text after </minimax:tool_call> is preserved."""
        parser = MiniMaxM2ToolParser()
        text = ('prefix text'
                '<minimax:tool_call>'
                '<invoke name="get_weather">'
                '<parameter name="location">NYC</parameter>'
                '</invoke>'
                '</minimax:tool_call>'
                ' suffix text')

        result = parser.detect_and_parse(text, sample_tools)

        assert "prefix text" in result.normal_text
        assert "suffix text" in result.normal_text
        assert len(result.calls) == 1
        assert result.calls[0].name == "get_weather"

    def test_streaming_preserves_prefix_in_same_chunk(self, sample_tools):
        """Test streaming returns prefix when it arrives with the tool token."""
        parser = MiniMaxM2ToolParser()
        result = parser.parse_streaming_increment(
            'Hello! <minimax:tool_call>'
            '<invoke name="get_weather">'
            '<parameter name="location">NYC</parameter>'
            '</invoke>'
            '</minimax:tool_call>', sample_tools)

        assert "Hello!" in result.normal_text
        names = [c.name for c in result.calls if c.name]
        assert "get_weather" in names


# ============================================================================
# Reasoning Parser Tests for Interleaved Thinking
# ============================================================================


class TestInterleavedThinkingReasoningParsers:
    """Test reasoning parsers used for interleaved thinking models."""

    def test_minimax_m2_reasoning_parser_registered(self):
        """Test that minimax_m2 reasoning parser is registered."""
        from tensorrt_llm.llmapi.reasoning_parser import ReasoningParserFactory
        assert "minimax_m2" in ReasoningParserFactory.keys()

    def test_minimax_m2_append_think_reasoning_parser_registered(self):
        """Test that minimax_m2_append_think reasoning parser is registered."""
        from tensorrt_llm.llmapi.reasoning_parser import ReasoningParserFactory
        assert "minimax_m2_append_think" in ReasoningParserFactory.keys()

    def test_minimax_m2_parser_handles_think_tags(self):
        """Test MiniMax-M2 reasoning parser correctly extracts thinking content."""
        from tensorrt_llm.llmapi.reasoning_parser import ReasoningParserFactory
        parser = ReasoningParserFactory.create_reasoning_parser("minimax_m2")

        text = "Let me reason about this...</think>Here is the answer."
        result = parser.parse(text)

        assert result.reasoning_content == "Let me reason about this..."
        assert result.content == "Here is the answer."

    def test_interleaved_thinking_minimax_with_tool_call(self):
        """Test MiniMax-M2 reasoning + tool parsing pipeline."""
        from tensorrt_llm.llmapi.reasoning_parser import ReasoningParserFactory

        parser = ReasoningParserFactory.create_reasoning_parser(
            "minimax_m2_append_think")
        text = ("Let me search for the latest news.</think>"
                '<minimax:tool_call>'
                '<invoke name="search_web">'
                '<parameter name="query">latest AI news</parameter>'
                '</invoke>'
                '</minimax:tool_call>')

        result = parser.parse(text)

        assert result.reasoning_content == "Let me search for the latest news."
        assert "<minimax:tool_call>" in result.content

        # Tool parser processes the content
        tool_parser = MiniMaxM2ToolParser()
        tool_result = tool_parser.detect_and_parse(result.content, [
            ChatCompletionToolsParam(
                type="function",
                function=FunctionDefinition(
                    name="search_web",
                    description="Search",
                    parameters={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string"
                            },
                        },
                    },
                ),
            )
        ])
        assert len(tool_result.calls) == 1
        assert tool_result.calls[0].name == "search_web"


# ============================================================================
# Gemma4 Tool Parser Tests
# ============================================================================

# Gemma4 helpers


def _g4_tc(func_name: str, args_str: str) -> str:
    """Build a Gemma4 tool call string."""
    return (f'{BOT_TOKEN}{CALL_PREFIX}{func_name}'
            f'{{{args_str}}}{EOT_TOKEN}')


def _g4_s(val: str) -> str:
    """Wrap a string value in Gemma4 string delimiters."""
    return f'{STRING_DELIM}{val}{STRING_DELIM}'


class TestGemma4ParsingHelpers:
    """Tests for low-level Gemma4 format parsing functions."""

    @pytest.mark.parametrize(
        "text,start,expected",
        [
            ("{hello}", 0, 6),
            ("{a:{b:1}}", 0, 8),
            # Brace inside a string-delim'd value is ignored.
            ('{key:' + STRING_DELIM + 'v{al}' + STRING_DELIM + '}', 0,
             len('{key:' + STRING_DELIM + 'v{al}' + STRING_DELIM + '}') - 1),
            # No matching closer => -1.
            ("{incomplete", 0, -1),
        ],
        ids=["simple", "nested", "string_delim", "unmatched"],
    )
    def test_find_matching_brace(self, text, start, expected):
        assert _find_matching_brace(text, start) == expected

    @pytest.mark.parametrize(
        "raw,expected",
        [
            (STRING_DELIM + "hello" + STRING_DELIM, "hello"),
            ("42", 42),
            ("3.14", 3.14),
            ("true", True),
            ("false", False),
            ("null", None),
        ],
        ids=["string", "int", "float", "bool_true", "bool_false", "null"],
    )
    def test_parse_value(self, raw, expected):
        assert _parse_gemma4_value(raw) == expected

    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("[]", []),
            (f'[{_g4_s("a")},{_g4_s("b")}]', ["a", "b"]),
            (f'[{_g4_s("hello")},42,true]', ["hello", 42, True]),
            (
                f'[{{name:{_g4_s("Alice")}}},{{name:{_g4_s("Bob")}}}]',
                [{
                    "name": "Alice"
                }, {
                    "name": "Bob"
                }],
            ),
        ],
        ids=["empty", "strings", "mixed_types", "nested_objects"],
    )
    def test_parse_array(self, raw, expected):
        assert _parse_gemma4_array(raw) == expected

    @pytest.mark.parametrize(
        "raw,expected",
        [
            (f'location:{_g4_s("Tokyo")}', {
                "location": "Tokyo"
            }),
            (
                f'location:{_g4_s("Tokyo")},unit:{_g4_s("celsius")}',
                {
                    "location": "Tokyo",
                    "unit": "celsius"
                },
            ),
            (
                f'name:{_g4_s("test")},count:42,active:true',
                {
                    "name": "test",
                    "count": 42,
                    "active": True
                },
            ),
            (
                f'loc:{{city:{_g4_s("Tokyo")},country:{_g4_s("Japan")}}}',
                {
                    "loc": {
                        "city": "Tokyo",
                        "country": "Japan"
                    }
                },
            ),
            (f'tags:[{_g4_s("a")},{_g4_s("b")}]', {
                "tags": ["a", "b"]
            }),
            ("", {}),
            # Strings carrying : or { must not be parsed as separators / braces.
            (f'url:{_g4_s("http://example.com:8080")}', {
                "url": "http://example.com:8080"
            }),
            ('tpl:' + _g4_s('Hello {name}'), {
                "tpl": "Hello {name}"
            }),
        ],
        ids=[
            "single_string",
            "multiple_values",
            "mixed_types",
            "nested_object",
            "with_array",
            "empty",
            "string_with_colon",
            "string_with_braces",
        ],
    )
    def test_parse_args(self, raw, expected):
        assert _parse_gemma4_args(raw) == expected

    def test_extract_tool_calls_single(self):
        text = _g4_tc("get_weather", f'location:{_g4_s("Tokyo")}')
        calls = _extract_tool_calls(text)
        assert len(calls) == 1
        assert calls[0][0] == "get_weather"

    def test_extract_tool_calls_multiple(self):
        text = (_g4_tc("get_weather", f'location:{_g4_s("Tokyo")}') +
                _g4_tc("search_web", f'query:{_g4_s("AI")}'))
        calls = _extract_tool_calls(text)
        assert len(calls) == 2
        assert calls[0][0] == "get_weather"
        assert calls[1][0] == "search_web"

    def test_extract_tool_calls_none(self):
        assert _extract_tool_calls("regular text") == []

    def test_extract_tool_calls_incomplete(self):
        # Missing EOT_TOKEN — must yield no calls.
        text = (f'{BOT_TOKEN}{CALL_PREFIX}'
                f'func{{arg:{_g4_s("val")}}}')
        assert _extract_tool_calls(text) == []


class TestGemma4ToolParser(BaseToolParserTestClass):
    """Test suite for Gemma4ToolParser class."""

    def make_parser(self):
        return Gemma4ToolParser()

    def make_tool_parser_test_cases(self):
        single_text = _g4_tc("get_weather", f'location:{_g4_s("NYC")}')
        single_expected_normal = ""
        single_expected_name = "get_weather"
        single_expected_params = {"location": "NYC"}

        multiple_text = (_g4_tc("get_weather", f'location:{_g4_s("LA")}') +
                         _g4_tc("search_web", f'query:{_g4_s("AI")}'))
        multiple_names = ("get_weather", "search_web")

        # Malformed: missing call: prefix, so no function name is found
        malformed_text = (f'{BOT_TOKEN}'
                          f'MALFORMED_NO_CALL_PREFIX{EOT_TOKEN}')

        with_parameters_text = _g4_tc("search_web", f'query:{_g4_s("test")}')
        with_parameters_name = "search_web"
        with_parameters_params = {"query": "test"}

        partial_bot_token = "<|tool"

        undefined_tool_text = _g4_tc("undefined_func", f'arg:{_g4_s("val")}')

        return ToolParserTestCases(
            has_tool_call_true=(
                f'Text {BOT_TOKEN}{CALL_PREFIX}'
                f'get_weather{{loc:{_g4_s("NYC")}}}{EOT_TOKEN}'),
            detect_and_parse_single_tool=(
                single_text,
                single_expected_normal,
                single_expected_name,
                single_expected_params,
            ),
            detect_and_parse_multiple_tools=(multiple_text, multiple_names),
            detect_and_parse_malformed_tool=malformed_text,
            detect_and_parse_with_parameters_key=(
                with_parameters_text,
                with_parameters_name,
                with_parameters_params,
            ),
            parse_streaming_increment_partial_bot_token=partial_bot_token,
            undefined_tool=undefined_tool_text,
        )

    def test_initialization(self, parser):
        assert parser.bot_token == BOT_TOKEN
        assert parser.eot_token == EOT_TOKEN
        assert parser.needs_raw_special_tokens is True

    def test_detect_and_parse_with_text_before(self, sample_tools, parser):
        text = ("Let me check. " +
                _g4_tc("get_weather", f'location:{_g4_s("NYC")}'))
        result = parser.detect_and_parse(text, sample_tools)
        assert result.normal_text == "Let me check."
        assert len(result.calls) == 1

    def test_detect_and_parse_multiple_params(self, sample_tools, parser):
        text = _g4_tc("get_weather",
                      f'location:{_g4_s("NYC")},unit:{_g4_s("celsius")}')
        result = parser.detect_and_parse(text, sample_tools)
        assert len(result.calls) == 1
        params = json.loads(result.calls[0].parameters)
        assert params == {"location": "NYC", "unit": "celsius"}

    def test_detect_and_parse_nested_object(self, sample_tools, parser):
        tools = [
            ChatCompletionToolsParam(
                type="function",
                function=FunctionDefinition(
                    name="create_event",
                    description="Create event",
                    parameters={
                        "type": "object",
                        "properties": {
                            "data": {
                                "type": "object"
                            }
                        },
                    },
                ),
            ),
        ]
        text = _g4_tc("create_event",
                      f'data:{{city:{_g4_s("Tokyo")},pop:1400}}')
        result = parser.detect_and_parse(text, tools)
        params = json.loads(result.calls[0].parameters)
        assert params == {"data": {"city": "Tokyo", "pop": 1400}}

    def test_detect_and_parse_array_param(self, sample_tools, parser):
        tools = [
            ChatCompletionToolsParam(
                type="function",
                function=FunctionDefinition(
                    name="add_tags",
                    description="Add tags",
                    parameters={
                        "type": "object",
                        "properties": {
                            "tags": {
                                "type": "array",
                                "items": {
                                    "type": "string"
                                },
                            },
                        },
                    },
                ),
            ),
        ]
        text = _g4_tc("add_tags", f'tags:[{_g4_s("py")},{_g4_s("ai")}]')
        result = parser.detect_and_parse(text, tools)
        params = json.loads(result.calls[0].parameters)
        assert params == {"tags": ["py", "ai"]}

    def test_detect_and_parse_empty_args(self, parser):
        tools = [
            ChatCompletionToolsParam(
                type="function",
                function=FunctionDefinition(
                    name="get_time",
                    description="Get time",
                    parameters={},
                ),
            ),
        ]
        text = _g4_tc("get_time", "")
        result = parser.detect_and_parse(text, tools)
        assert len(result.calls) == 1
        assert json.loads(result.calls[0].parameters) == {}

    def test_structure_info(self, parser):
        info_fn = parser.structure_info()
        info = info_fn("get_weather")
        assert info.begin == f'{BOT_TOKEN}{CALL_PREFIX}get_weather{{'
        assert info.end == f'}}{EOT_TOKEN}'
        assert info.trigger == BOT_TOKEN

    def test_parse_streaming_increment_complete_tool_call(
            self, sample_tools, parser):
        """Complete tool call in a single streaming chunk."""
        text = _g4_tc("get_weather", f'location:{_g4_s("Tokyo")}')
        result = parser.parse_streaming_increment(text, sample_tools)
        assert len(result.calls) >= 1
        assert result.calls[0].name == "get_weather"

    def test_parse_streaming_increment_multi_chunk(self, sample_tools, parser):
        """Tool call split across two chunks."""
        chunk1 = (f'{BOT_TOKEN}{CALL_PREFIX}'
                  f'get_weather{{location:')
        result1 = parser.parse_streaming_increment(chunk1, sample_tools)
        assert len(result1.calls) == 1
        assert result1.calls[0].name == "get_weather"

        chunk2 = f'{_g4_s("Tokyo")}}}{EOT_TOKEN}'
        result2 = parser.parse_streaming_increment(chunk2, sample_tools)
        assert len(result2.calls) >= 1

    def test_parse_streaming_increment_multiple_tools(self, sample_tools,
                                                      parser):
        """Multiple tool calls in streaming mode."""
        tc1 = _g4_tc("get_weather", f'location:{_g4_s("Tokyo")}')
        result1 = parser.parse_streaming_increment(tc1, sample_tools)
        assert len(result1.calls) >= 1

        tc2 = _g4_tc("search_web", f'query:{_g4_s("weather")}')
        result2 = parser.parse_streaming_increment(tc2, sample_tools)
        assert len(result2.calls) >= 1

    def test_parse_streaming_increment_text_then_tool(self, sample_tools,
                                                      parser):
        """Normal text followed by tool call."""
        result1 = parser.parse_streaming_increment("Checking...", sample_tools)
        assert result1.normal_text == "Checking..."

        tc = _g4_tc("get_weather", f'location:{_g4_s("NYC")}')
        result2 = parser.parse_streaming_increment(tc, sample_tools)
        assert len(result2.calls) >= 1

    def test_real_world_weather_query(self, sample_tools, parser):
        """Simulate real model output: text + tool call."""
        text = ("I'll check. " +
                _g4_tc("get_weather", (f'location:{_g4_s("San Francisco, CA")},'
                                       f'unit:{_g4_s("fahrenheit")}')))
        result = parser.detect_and_parse(text, sample_tools)
        assert result.normal_text == "I'll check."
        params = json.loads(result.calls[0].parameters)
        assert params == {
            "location": "San Francisco, CA",
            "unit": "fahrenheit",
        }

    def test_factory_creates_gemma4_parser(self):
        from tensorrt_llm.serve.tool_parser.tool_parser_factory import \
            ToolParserFactory
        p = ToolParserFactory.create_tool_parser("gemma4")
        assert isinstance(p, Gemma4ToolParser)

    def test_model_type_mapping(self):
        from tensorrt_llm.serve.tool_parser.tool_parser_factory import \
            MODEL_TYPE_TO_TOOL_PARSER
        assert MODEL_TYPE_TO_TOOL_PARSER.get("gemma4") == "gemma4"
        assert MODEL_TYPE_TO_TOOL_PARSER.get("gemma4_text") == "gemma4"


# ============================================================================
# Tool Parser Factory Tests
# ============================================================================


class TestToolParserFactory:
    """Test that the tool parser factory registers all expected parsers."""

    def test_minimax_m2_registered(self):
        """Test that minimax_m2 parser is registered in factory."""
        from tensorrt_llm.serve.tool_parser.tool_parser_factory import \
            ToolParserFactory
        assert "minimax_m2" in ToolParserFactory.parsers

    def test_create_minimax_m2_parser(self):
        """Test creating minimax_m2 parser via factory."""
        from tensorrt_llm.serve.tool_parser.tool_parser_factory import \
            ToolParserFactory
        parser = ToolParserFactory.create_tool_parser("minimax_m2")
        assert isinstance(parser, MiniMaxM2ToolParser)


# ============================================================================
# FunctionDefinition strict field and ChatCompletionRequest store field Tests
# ============================================================================


class TestFunctionDefinitionStrictField:
    """Test that FunctionDefinition accepts the strict field (TRTLLM-11616)."""

    def test_strict_true_accepted(self):
        """FunctionDefinition should accept strict=True without validation error."""
        func_def = FunctionDefinition(
            name="get_weather",
            description="Get weather",
            parameters={
                "type": "object",
                "properties": {}
            },
            strict=True,
        )
        assert func_def.strict is True

    def test_strict_false_accepted(self):
        """FunctionDefinition should accept strict=False without validation error."""
        func_def = FunctionDefinition(
            name="get_weather",
            description="Get weather",
            parameters={
                "type": "object",
                "properties": {}
            },
            strict=False,
        )
        assert func_def.strict is False

    def test_strict_none_by_default(self):
        """FunctionDefinition should default strict to None."""
        func_def = FunctionDefinition(
            name="get_weather",
            description="Get weather",
        )
        assert func_def.strict is None

    def test_tool_param_with_strict(self):
        """ChatCompletionToolsParam should accept function with strict field."""
        tool = ChatCompletionToolsParam(
            type="function",
            function=FunctionDefinition(
                name="search_web",
                description="Search",
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string"
                        }
                    },
                },
                strict=True,
            ),
        )
        assert tool.function.strict is True
        assert tool.function.name == "search_web"


# ============================================================================
# Strict tool structural tag constraint building Tests
# ============================================================================


class TestBuildToolStrictGuidedDecoding:
    """Test _build_tool_strict_guided_decoding_params from openai_server."""

    def test_no_strict_tools_returns_none(self):
        """Should return None when no tool has strict=True."""
        from tensorrt_llm.serve.openai_server import \
            _build_tool_strict_guided_decoding_params

        tools = [
            ChatCompletionToolsParam(
                type="function",
                function=FunctionDefinition(
                    name="get_weather",
                    parameters={
                        "type": "object",
                        "properties": {}
                    },
                ),
            ),
        ]
        result = _build_tool_strict_guided_decoding_params(tools, "qwen3")
        assert result is None

    def test_strict_false_returns_none(self):
        """Should return None when all tools have strict=False."""
        from tensorrt_llm.serve.openai_server import \
            _build_tool_strict_guided_decoding_params

        tools = [
            ChatCompletionToolsParam(
                type="function",
                function=FunctionDefinition(
                    name="get_weather",
                    parameters={
                        "type": "object",
                        "properties": {}
                    },
                    strict=False,
                ),
            ),
        ]
        result = _build_tool_strict_guided_decoding_params(tools, "qwen3")
        assert result is None

    def test_no_tools_returns_none(self):
        """Should return None when tools list is empty or None."""
        from tensorrt_llm.serve.openai_server import \
            _build_tool_strict_guided_decoding_params

        assert _build_tool_strict_guided_decoding_params(None, "qwen3") is None
        assert _build_tool_strict_guided_decoding_params([], "qwen3") is None

    def test_no_parser_returns_none(self):
        """Should return None when no tool parser is provided."""
        from tensorrt_llm.serve.openai_server import \
            _build_tool_strict_guided_decoding_params

        tools = [
            ChatCompletionToolsParam(
                type="function",
                function=FunctionDefinition(
                    name="get_weather",
                    parameters={
                        "type": "object",
                        "properties": {}
                    },
                    strict=True,
                ),
            ),
        ]
        assert _build_tool_strict_guided_decoding_params(tools, None) is None
        assert _build_tool_strict_guided_decoding_params(tools, "") is None

    def test_strict_tool_with_qwen3_parser(self):
        """Should build GuidedDecodingParams with structural_tag for Qwen3."""
        from tensorrt_llm.serve.openai_server import \
            _build_tool_strict_guided_decoding_params

        tools = [
            ChatCompletionToolsParam(
                type="function",
                function=FunctionDefinition(
                    name="get_weather",
                    parameters={
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string"
                            },
                        },
                        "required": ["location"],
                    },
                    strict=True,
                ),
            ),
        ]
        result = _build_tool_strict_guided_decoding_params(tools, "qwen3")
        assert result is not None
        assert result.structural_tag is not None

        stag = json.loads(result.structural_tag)
        assert stag["type"] == "structural_tag"
        fmt = stag["format"]
        assert fmt["type"] == "triggered_tags"
        assert "<tool_call>" in fmt["triggers"]
        assert len(fmt["tags"]) == 1

        tag = fmt["tags"][0]
        assert "get_weather" in tag["begin"]
        assert tag["content"]["type"] == "json_schema"
        assert tag["content"]["json_schema"]["properties"]["location"][
            "type"] == "string"

    def test_mixed_strict_and_non_strict(self):
        """Should constrain strict tools and allow any text for non-strict."""
        from tensorrt_llm.serve.openai_server import \
            _build_tool_strict_guided_decoding_params

        tools = [
            ChatCompletionToolsParam(
                type="function",
                function=FunctionDefinition(
                    name="get_weather",
                    parameters={
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string"
                            }
                        },
                    },
                    strict=True,
                ),
            ),
            ChatCompletionToolsParam(
                type="function",
                function=FunctionDefinition(
                    name="search_web",
                    parameters={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string"
                            }
                        },
                    },
                    strict=False,
                ),
            ),
        ]
        result = _build_tool_strict_guided_decoding_params(tools, "qwen3")
        assert result is not None

        stag = json.loads(result.structural_tag)
        fmt = stag["format"]
        assert len(fmt["tags"]) == 2

        # First tag (strict) should have json_schema content
        assert fmt["tags"][0]["content"]["type"] == "json_schema"
        # Second tag (non-strict) should have any_text content
        assert fmt["tags"][1]["content"]["type"] == "any_text"

    def test_unsupported_parser_returns_none(self):
        """Should return None for parsers that don't support structural tags."""
        from tensorrt_llm.serve.openai_server import \
            _build_tool_strict_guided_decoding_params

        tools = [
            ChatCompletionToolsParam(
                type="function",
                function=FunctionDefinition(
                    name="get_weather",
                    parameters={
                        "type": "object",
                        "properties": {}
                    },
                    strict=True,
                ),
            ),
        ]
        # glm4 does not support structural tags
        result = _build_tool_strict_guided_decoding_params(tools, "glm4")
        assert result is None

    def test_strict_tool_with_deepseek_parser(self):
        """Should build GuidedDecodingParams with structural_tag for DeepSeek."""
        from tensorrt_llm.serve.openai_server import \
            _build_tool_strict_guided_decoding_params

        tools = [
            ChatCompletionToolsParam(
                type="function",
                function=FunctionDefinition(
                    name="calculate",
                    parameters={
                        "type": "object",
                        "properties": {
                            "expr": {
                                "type": "string"
                            }
                        },
                    },
                    strict=True,
                ),
            ),
        ]
        result = _build_tool_strict_guided_decoding_params(tools, "deepseek_v3")
        assert result is not None
        assert result.structural_tag is not None

        stag = json.loads(result.structural_tag)
        fmt = stag["format"]
        assert fmt["type"] == "triggered_tags"
        assert len(fmt["tags"]) == 1
        assert "calculate" in fmt["tags"][0]["begin"]


# ============================================================================
# Named tool_choice (forced function call) Tests — TRTLLM-12758
# ============================================================================


def _make_tools(*specs):
    """Helper: build a list of ChatCompletionToolsParam from (name, params) specs."""
    return [
        ChatCompletionToolsParam(
            type="function",
            function=FunctionDefinition(name=name, parameters=params),
        ) for name, params in specs
    ]


_SCHEMA_LOCATION = {
    "type": "object",
    "properties": {
        "location": {
            "type": "string"
        },
    },
    "required": ["location"],
}

_SCHEMA_QUERY = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string"
        },
    },
    "required": ["query"],
}


class TestBuildForcedToolCallDecoding:
    """Test ``_build_forced_tool_call_decoding`` from openai_server.

    Covers OpenAI-spec ``tool_choice = {"type": "function",
    "function": {"name": "X"}}`` for non-harmony tool parsers (TRTLLM-12758).
    The helper returns ``(begin_prefix, GuidedDecodingParams)``: the caller
    prefix-injects ``begin_prefix`` into the rendered chat prompt and applies
    the guided-decoding params to the request, so the model is forced to
    start generation inside the tool call and the resulting arguments are
    JSON-schema valid.
    """

    @pytest.mark.parametrize(
        "parser_name",
        ["qwen3", "deepseek_v3", "kimi_k2", "gemma4"],
    )
    def test_forced_name_returns_prefix_and_json_schema(self, parser_name):
        """Forced-name path returns a parser-specific prefix and JSON schema.

        Across parser families, ``begin_prefix`` must contain the forced
        function name and the guided-decoding params must constrain the args
        to the function's ``parameters`` JSON Schema.
        """
        from tensorrt_llm.sampling_params import GuidedDecodingParams
        from tensorrt_llm.serve.openai_server import \
            _build_forced_tool_call_decoding
        from tensorrt_llm.serve.tool_parser.tool_parser_factory import \
            ToolParserFactory

        tools = _make_tools(("get_weather", _SCHEMA_LOCATION),
                            ("search_web", _SCHEMA_QUERY))
        begin_prefix, guided = _build_forced_tool_call_decoding(
            tools, parser_name, "get_weather")

        # ``begin_prefix`` is exactly the parser's tool-call begin string.
        parser = ToolParserFactory.parsers[parser_name.lower()]()
        expected_begin = parser.structure_info()("get_weather").begin
        assert begin_prefix == expected_begin
        assert "get_weather" in begin_prefix

        assert isinstance(guided, GuidedDecodingParams)
        assert guided.json == _SCHEMA_LOCATION
        assert guided.json_object is False
        assert guided.structural_tag is None

    def test_forced_name_no_parameters_uses_json_object(self):
        """Forced-name path falls back to ``json_object`` when no schema.

        When the forced function has no ``parameters``, the helper falls back
        to ``json_object=True`` so the synthesized ``arguments`` field is still
        well-formed JSON (typically ``{}``).
        """
        from tensorrt_llm.sampling_params import GuidedDecodingParams
        from tensorrt_llm.serve.openai_server import \
            _build_forced_tool_call_decoding

        tools = [
            ChatCompletionToolsParam(
                type="function",
                function=FunctionDefinition(name="ping"),
            ),
            ChatCompletionToolsParam(
                type="function",
                function=FunctionDefinition(name="other",
                                            parameters=_SCHEMA_QUERY),
            ),
        ]
        begin_prefix, guided = _build_forced_tool_call_decoding(
            tools, "qwen3", "ping")
        assert "ping" in begin_prefix
        assert isinstance(guided, GuidedDecodingParams)
        assert guided.json is None
        assert guided.json_object is True

    def test_forced_name_ignores_strict_flag(self):
        """The forced-call path engages regardless of any ``strict=True``."""
        from tensorrt_llm.serve.openai_server import (
            _build_forced_tool_call_decoding,
            _build_tool_strict_guided_decoding_params)

        tools = _make_tools(("get_weather", _SCHEMA_LOCATION))
        # Strict-tools path is unchanged: no strict → None.
        assert _build_tool_strict_guided_decoding_params(tools, "qwen3") is None
        # Forced-call path: always constrains.
        begin_prefix, guided = _build_forced_tool_call_decoding(
            tools, "qwen3", "get_weather")
        assert begin_prefix
        assert guided is not None

    def test_forced_name_missing_raises_value_error(self):
        """Forcing a function that is not in tools must raise ValueError."""
        from tensorrt_llm.serve.openai_server import \
            _build_forced_tool_call_decoding

        tools = _make_tools(("get_weather", _SCHEMA_LOCATION))
        with pytest.raises(ValueError) as exc:
            _build_forced_tool_call_decoding(tools, "qwen3", "missing_fn")
        msg = str(exc.value)
        assert "missing_fn" in msg
        assert "get_weather" in msg  # available functions reported

    def test_forced_name_no_tools_raises_value_error(self):
        """Forcing a function without any tools provided is a 4xx-class error."""
        from tensorrt_llm.serve.openai_server import \
            _build_forced_tool_call_decoding

        with pytest.raises(ValueError):
            _build_forced_tool_call_decoding([], "qwen3", "get_weather")
        with pytest.raises(ValueError):
            _build_forced_tool_call_decoding(None, "qwen3", "get_weather")

    def test_forced_name_no_parser_raises_value_error(self):
        """Forcing a function on a server without a tool_parser is a 4xx."""
        from tensorrt_llm.serve.openai_server import \
            _build_forced_tool_call_decoding

        tools = _make_tools(("get_weather", _SCHEMA_LOCATION))
        with pytest.raises(ValueError):
            _build_forced_tool_call_decoding(tools, None, "get_weather")
        with pytest.raises(ValueError):
            _build_forced_tool_call_decoding(tools, "", "get_weather")

    def test_forced_name_unsupported_parser_raises_value_error(self):
        """Parsers that do not support structural tags cannot honor forced names."""
        from tensorrt_llm.serve.openai_server import \
            _build_forced_tool_call_decoding

        tools = _make_tools(("get_weather", _SCHEMA_LOCATION))
        # glm4, glm47, qwen3_coder, minimax_m2 do not support structural tags.
        for parser_name in ("glm4", "glm47", "qwen3_coder", "minimax_m2"):
            with pytest.raises(ValueError) as exc:
                _build_forced_tool_call_decoding(tools, parser_name,
                                                 "get_weather")
            assert "structural" in str(exc.value).lower()

    def test_forced_name_unknown_parser_raises_value_error(self):
        """An unregistered parser name should also raise."""
        from tensorrt_llm.serve.openai_server import \
            _build_forced_tool_call_decoding

        tools = _make_tools(("get_weather", _SCHEMA_LOCATION))
        with pytest.raises(ValueError):
            _build_forced_tool_call_decoding(tools, "no_such_parser",
                                             "get_weather")

    def test_strict_path_unchanged_by_default(self):
        """Strict-tools path stays untouched without ``strict=True``.

        Regression guard: ``_build_tool_strict_guided_decoding_params`` must
        still return ``None`` when no tool has ``strict=True``, independent of
        the new forced-call helper.
        """
        from tensorrt_llm.serve.openai_server import \
            _build_tool_strict_guided_decoding_params

        tools = _make_tools(("get_weather", _SCHEMA_LOCATION))
        assert _build_tool_strict_guided_decoding_params(tools, "qwen3") is None


class TestForcedToolCallStreamingFinishReason:
    """A forced call that streams nothing must not claim ``tool_calls``.

    The streaming forced-call branch sets ``has_tool_call`` so the final chunk
    reports ``finish_reason="tool_calls"``. It used to do so on every
    iteration, including ones with empty ``delta_text``. When generation ends
    before any non-empty text -- ``max_completion_tokens`` reached before the
    first detokenized chunk, or an abort -- no ``DeltaToolCall`` is ever
    emitted, so the client saw a ``tool_calls`` finish reason with no tool
    call attached.
    """

    @staticmethod
    def _args(**overrides):
        from tensorrt_llm.serve.openai_protocol import (
            ChatCompletionNamedFunction, ChatCompletionNamedToolChoiceParam)
        from tensorrt_llm.serve.postprocess_handlers import ChatPostprocArgs

        args = ChatPostprocArgs(role="assistant", model="test-model")
        args.tool_parser = "qwen3"
        # The forced call is derived from the request, not from a dedicated
        # field: both ``tool_choice`` and ``tools`` must be set or
        # ``_forced_tool_choice`` / ``_forced_choice_uses_tool_parser`` will
        # not see a forced call and these assertions become vacuous.
        args.tool_choice = ChatCompletionNamedToolChoiceParam(
            function=ChatCompletionNamedFunction(name="get_weather"))
        args.tools = [
            ChatCompletionToolsParam(type="function",
                                     function=FunctionDefinition(
                                         name="get_weather",
                                         parameters=_SCHEMA_LOCATION))
        ]
        args.num_prompt_tokens = 3
        for key, value in overrides.items():
            setattr(args, key, value)
        return args

    @staticmethod
    def _rsp(text, finish_reason):
        output = Mock()
        output.index = 0
        output.text_diff = text
        output.text = text
        output.token_ids_diff = [1] if text else []
        output.token_ids = [1] if text else []
        output.finish_reason = finish_reason
        output.stop_reason = None
        output.logprobs_diff = None
        output.disaggregated_params = None

        rsp = Mock()
        rsp.outputs = [output]
        rsp._done = finish_reason is not None
        rsp.cached_tokens = 0
        rsp.prompt_token_ids = [1, 2, 3]
        # Read via getattr() and fed straight into a pydantic model, so it must
        # be a real value: a bare Mock fails float validation.
        rsp.avg_decoded_tokens_per_iter = None
        return rsp

    def _finish_reasons(self, chunks):
        reasons = []
        for chunk in chunks:
            for line in chunk.splitlines():
                if not line.startswith("data: ") or line.endswith("[DONE]"):
                    continue
                payload = json.loads(line[len("data: "):].strip())
                for choice in payload.get("choices", []):
                    if choice.get("finish_reason"):
                        reasons.append(choice["finish_reason"])
        return reasons

    def test_forced_call_with_no_text_does_not_report_tool_calls(self):
        """Generation ending before any text must not claim a tool call."""
        from tensorrt_llm.serve.postprocess_handlers import \
            chat_stream_post_processor

        args = self._args()
        # A single terminal iteration that never produced detokenized text.
        chunks = chat_stream_post_processor(self._rsp("", "length"), args)

        assert not args.has_tool_call.get(0, False), (
            "has_tool_call was set without any DeltaToolCall being emitted")
        reasons = self._finish_reasons(chunks)
        assert "tool_calls" not in reasons, (
            f"reported a tool-call finish reason with no tool call: {reasons}")
        assert reasons == ["length"]

    def test_forced_call_with_text_still_reports_tool_calls(self):
        """Baseline: once a delta is emitted the finish reason still flips."""
        from tensorrt_llm.serve.postprocess_handlers import \
            chat_stream_post_processor

        args = self._args()
        chat_stream_post_processor(self._rsp('{"city":', None), args)
        assert args.forced_tool_name_sent.get(0, False)
        assert args.has_tool_call.get(0, False)

        args.first_iteration = False
        chunks = chat_stream_post_processor(self._rsp(' "Rome"}', "stop"), args)
        assert self._finish_reasons(chunks) == ["tool_calls"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


class TestKimiK3ToolParser(BaseToolParserTestClass):
    """Test suite for KimiK3ToolParser (XTML tool-call format).

    Fixture strings follow the checkpoint's `encoding_k3.py` rendering:
    `<|open|>tag key="value"<|sep|>` / `<|close|>tag<|sep|>`, attributes
    space-prefixed and `&`/`"`-escaped, call indices 1-based, string
    argument bodies raw and non-string bodies JSON.
    """

    BOT = "<|open|>tools<|sep|>"
    EOT = "<|close|>tools<|sep|>"

    @staticmethod
    def _call(name: str, index: int, body: str) -> str:
        return (f'<|open|>call tool="{name}" index="{index}"<|sep|>'
                f'{body}<|close|>call<|sep|>')

    @staticmethod
    def _argument(key: str, type_: str, value: str) -> str:
        return (f'<|open|>argument key="{key}" type="{type_}"<|sep|>'
                f'{value}<|close|>argument<|sep|>')

    def _section(self, *calls: str) -> str:
        return self.BOT + "".join(calls) + self.EOT

    def make_parser(self):
        return KimiK3ToolParser()

    def make_tool_parser_test_cases(self):
        single_call = self._section(
            self._call("get_weather", 1,
                       self._argument("location", "string", "NYC")))
        return ToolParserTestCases(
            has_tool_call_true="Some text " + single_call,
            detect_and_parse_single_tool=(
                "Normal text" + single_call,
                "Normal text",
                "get_weather",
                {
                    "location": "NYC"
                },
            ),
            detect_and_parse_multiple_tools=(
                self._section(
                    self._call("get_weather", 1,
                               self._argument("location", "string", "LA")),
                    self._call("search_web", 2,
                               self._argument("query", "string", "AI")),
                ),
                ("get_weather", "search_web"),
            ),
            # A call without the mandatory tool="..." attribute is skipped.
            detect_and_parse_malformed_tool=self._section(
                '<|open|>call index="1"<|sep|>'
                '<|open|>argument key="location" type="string"<|sep|>NYC'
                '<|close|>argument<|sep|><|close|>call<|sep|>'),
            # K3 has no JSON "parameters" key wrapper; the closest analogue
            # is the raw-JSON call body variant.
            detect_and_parse_with_parameters_key=(
                self._section(
                    self._call(
                        "search_web", 1,
                        '<|open|>json type="object"<|sep|>{"query": "test"}'
                        '<|close|>json<|sep|>')),
                "search_web",
                {
                    "query": "test"
                },
            ),
            parse_streaming_increment_partial_bot_token="<|open|>too",
            undefined_tool=self._section(
                self._call("undefined_func", 1,
                           self._argument("arg", "string", "any value"))),
        )

    def test_initialization(self, parser):
        assert parser.bot_token == self.BOT
        assert parser.eot_token == self.EOT

    def test_undefined_tool(self, sample_tools, parser, tool_parser_test_cases):
        """Keep undefined-tool calls at their positional index.

        K3 warns about undefined tools rather than remapping ``tool_index`` to
        ``-1``.
        """
        text = tool_parser_test_cases.undefined_tool

        result = parser.detect_and_parse(text, sample_tools)

        assert len(result.calls) == 1
        assert result.calls[0].name == "undefined_func"
        assert result.calls[0].tool_index == 0

    def test_supports_structural_tag(self):
        """Reject JSON-schema structural tagging for XTML bodies.

        XTML bodies already use tag-structured text, so JSON-schema
        structural-tag constrained decoding does not apply.
        """
        parser = KimiK3ToolParser()
        assert parser.supports_structural_tag() is False
        with pytest.raises(NotImplementedError):
            parser.structure_info()

    def test_argument_type_coercion(self, sample_tools, parser):
        """Non-string argument bodies are JSON; string bodies stay raw."""
        text = self._section(
            self._call(
                "get_weather", 1,
                self._argument("location", "string", '"quoted" & raw') +
                self._argument("count", "number", "3") +
                self._argument("celsius", "boolean", "true") +
                self._argument("extra", "null", "null") +
                self._argument("nested", "object", '{"a": [1, 2]}') +
                self._argument("tags", "array", '["x", "y"]')))

        result = parser.detect_and_parse(text, sample_tools)

        assert len(result.calls) == 1
        assert json.loads(result.calls[0].parameters) == {
            "location": '"quoted" & raw',
            "count": 3,
            "celsius": True,
            "extra": None,
            "nested": {
                "a": [1, 2]
            },
            "tags": ["x", "y"],
        }

    def test_argument_invalid_json_falls_back_to_raw(self, sample_tools,
                                                     parser):
        """Keep a non-string argument's invalid JSON body as raw text.

        Invalid JSON should fall back to its original text instead of raising.
        """
        text = self._section(
            self._call("get_weather", 1,
                       self._argument("count", "number", "not-a-number")))

        result = parser.detect_and_parse(text, sample_tools)

        assert json.loads(result.calls[0].parameters) == {
            "count": "not-a-number"
        }

    def test_attribute_unescaping(self, sample_tools, parser):
        """Unescape encoded XTML attribute values.

        K3 attributes arrive escaped by ``encoding_k3._escape_attr_value``.
        """
        text = self._section(
            self._call(
                "get_weather", 1,
                self._argument("say &quot;hi&quot; &amp; bye", "string", "v")))

        result = parser.detect_and_parse(text, sample_tools)

        assert json.loads(result.calls[0].parameters) == {'say "hi" & bye': "v"}

    def test_empty_arguments(self, sample_tools, parser):
        """A call with no argument tags yields an empty JSON object."""
        text = self._section(self._call("search_web", 1, ""))

        result = parser.detect_and_parse(text, sample_tools)

        assert len(result.calls) == 1
        assert result.calls[0].parameters == "{}"

    def test_trailing_structural_markup_stripped(self, sample_tools, parser):
        """Strip trailing XTML terminators from standalone normal text.

        This covers parsing without a tools section or reasoning parser.
        """
        text = "The answer is 4.<|close|>message<|sep|><|end_of_msg|>"

        result = parser.detect_and_parse(text, sample_tools)

        assert result.normal_text == "The answer is 4."
        assert len(result.calls) == 0

    def test_streaming_buffers_section_until_complete(self, sample_tools,
                                                      parser):
        """Buffer an incomplete tools section while streaming response text.

        Response text is emitted immediately, but calls wait for the closing
        ``<|close|>tools<|sep|>`` marker.
        """
        result = parser.parse_streaming_increment("Checking. ", sample_tools)
        assert result.normal_text == "Checking. "
        assert result.calls == []

        # Section opener + call header: everything buffered.
        result = parser.parse_streaming_increment(
            self.BOT + '<|open|>call tool="get_weather" index="1"<|sep|>',
            sample_tools)
        assert result.normal_text == ""
        assert result.calls == []

        # Arguments still buffered.
        result = parser.parse_streaming_increment(
            self._argument("location", "string", "NYC"), sample_tools)
        assert result.normal_text == ""
        assert result.calls == []

        # Section close: the complete call is emitted.
        result = parser.parse_streaming_increment(
            "<|close|>call<|sep|>" + self.EOT, sample_tools)
        assert result.normal_text == ""
        assert len(result.calls) == 1
        assert result.calls[0].name == "get_weather"
        assert json.loads(result.calls[0].parameters) == {"location": "NYC"}

    def test_composes_with_kimi_k3_reasoning_parser(self, sample_tools, parser):
        """Parse tools passed through the Kimi-K3 reasoning parser.

        The reasoning parser preserves the tools section verbatim for this
        parser.
        """
        from tensorrt_llm.llmapi.reasoning_parser import ReasoningParserFactory

        completion = (
            "Need the weather.<|close|>think<|sep|>"
            "<|open|>response<|sep|>Checking."
            "<|close|>response<|sep|>" + self._section(
                self._call("get_weather", 1,
                           self._argument("location", "string", "NYC"))) +
            "<|close|>message<|sep|><|end_of_msg|>")

        reasoning = ReasoningParserFactory.create_reasoning_parser("kimi_k3")
        stage1 = reasoning.parse(completion)
        assert stage1.reasoning_content == "Need the weather."

        stage2 = parser.detect_and_parse(stage1.content, sample_tools)
        assert stage2.normal_text == "Checking."
        assert len(stage2.calls) == 1
        assert stage2.calls[0].name == "get_weather"
        assert json.loads(stage2.calls[0].parameters) == {"location": "NYC"}

    def test_extracts_forced_tool_calls(self):
        """K3 opts in to serve-level extraction on forced tool_choice.

        XTML has no structural-tag grammar, so a forced call still arrives
        as preamble + markup and must not be passed through raw.
        """
        assert KimiK3ToolParser.extracts_forced_tool_calls is True

    def test_streaming_early_end_flush(self, sample_tools, parser):
        """Emit a buffered complete call when the stream ends before EOT.

        Without the finalization hook the buffered section was silently
        dropped.
        """
        result = parser.parse_streaming_increment(
            "Sure. " + self.BOT + self._call(
                "get_weather", 1, self._argument("location", "string", "NYC")),
            sample_tools)
        assert result.normal_text == "Sure. "
        assert result.calls == []

        result = parser.finish(sample_tools)

        assert len(result.calls) == 1
        assert result.calls[0].name == "get_weather"
        assert json.loads(result.calls[0].parameters) == {"location": "NYC"}
        assert parser._buffer == ""

    def test_streaming_early_end_flush_salvages_complete_calls(
            self, sample_tools, parser):
        """A stream truncated mid-call still emits the calls that completed."""
        parser.parse_streaming_increment(
            self.BOT + self._call("get_weather", 1,
                                  self._argument("location", "string", "LA")) +
            '<|open|>call tool="search_web" index="2"<|sep|>'
            '<|open|>argument key="query" type="str', sample_tools)

        result = parser.finish(sample_tools)

        assert [call.name for call in result.calls] == ["get_weather"]
        assert json.loads(result.calls[0].parameters) == {"location": "LA"}

    def test_finish_after_complete_section_is_empty(self, sample_tools, parser):
        """A cleanly closed section leaves nothing for the flush to emit."""
        result = parser.parse_streaming_increment(
            self._section(
                self._call("get_weather", 1,
                           self._argument("location", "string", "NYC"))),
            sample_tools)
        assert len(result.calls) == 1

        result = parser.finish(sample_tools)

        assert result.normal_text == ""
        assert result.calls == []

    @pytest.mark.parametrize("residue", [
        "<|close|>message<|sep|>", "<|end_of_msg|>",
        "<|close|>message<|sep|><|end_of_msg|>"
    ])
    def test_streaming_post_section_residue_never_leaks(self, sample_tools,
                                                        parser, residue):
        """Structural framing after the section is stripped, not streamed.

        ``parse_streaming_increment`` leaves post-section residue in the
        buffer; if it arrives in a later increment the no-``bot_token`` path
        must not emit it as content. Streaming must match non-streaming, which
        strips the same residue via ``_trailing_structural``. Regression:
        without the ``_section_done`` guard the residue leaked as ``content``.
        """
        preamble = "Sure. "
        section = self._section(
            self._call("get_weather", 1,
                       self._argument("location", "string", "NYC")))
        # Section completes in the first increment; residue trickles in one
        # character at a time in later increments (worst case for partial
        # structural tokens).
        chunks = [preamble + section] + list(residue)

        normal_text = ""
        calls = []
        for chunk in chunks:
            result = parser.parse_streaming_increment(chunk, sample_tools)
            normal_text += result.normal_text
            calls.extend(result.calls)
        result = parser.finish(sample_tools)
        normal_text += result.normal_text
        calls.extend(result.calls)

        assert normal_text == preamble
        assert [call.name for call in calls] == ["get_weather"]
        # Streaming agrees with the non-streaming path on the same full text.
        non_streaming = self.make_parser().detect_and_parse(
            preamble + section + residue, sample_tools)
        assert non_streaming.normal_text == preamble

    def test_finish_flushes_held_partial_bot_token_as_text(
            self, sample_tools, parser):
        """A held-back bot_token prefix is plain text once the stream ends."""
        result = parser.parse_streaming_increment("A <|open|>too", sample_tools)
        assert result.normal_text == "A "

        result = parser.finish(sample_tools)

        assert result.normal_text == "<|open|>too"
        assert result.calls == []

    def test_streaming_multiple_calls_split_across_chunks(
            self, sample_tools, parser):
        """Both calls of a two-call section arrive once EOT lands."""
        first = self._call("get_weather", 1,
                           self._argument("location", "string", "LA"))
        second = self._call("search_web", 2,
                            self._argument("query", "string", "AI"))
        section = self.BOT + first + second + self.EOT
        # Split inside the second call header and inside the EOT token.
        chunks = [
            section[:len(self.BOT) + len(first) + 14],
            section[len(self.BOT) + len(first) + 14:-7],
            section[-7:],
        ]

        collected = []
        for chunk in chunks:
            collected.extend(
                parser.parse_streaming_increment(chunk, sample_tools).calls)

        assert [call.name
                for call in collected] == ["get_weather", "search_web"]
        assert json.loads(collected[0].parameters) == {"location": "LA"}
        assert json.loads(collected[1].parameters) == {"query": "AI"}

    def test_literal_lt_in_attribute_value(self, sample_tools, parser):
        """Attribute values may contain a literal ``<``.

        The K3 encoder escapes only ``&`` and ``"``, so ``<`` reaches the
        parser raw; the old header pattern (``[^<]*?``) silently dropped the
        whole call.
        """
        text = self._section(
            self._call("a<b", 1, self._argument("expr", "string", "x")) +
            self._call("get_weather", 2, self._argument("k<ey", "string", "v")))

        result = parser.detect_and_parse(text, sample_tools)

        assert [call.name for call in result.calls] == ["a<b", "get_weather"]
        assert json.loads(result.calls[1].parameters) == {"k<ey": "v"}


class TestForcedToolChoicePostprocessing:
    """Serve-level tests for named/forced ``tool_choice`` and stream flush.

    Drives the real ``chat_response_post_processor`` /
    ``chat_stream_post_processor`` with fake generation results, since the
    named-choice extraction and the end-of-stream flush live in the serving
    layer, above the parsers the rest of this file tests.
    """

    BOT = TestKimiK3ToolParser.BOT
    EOT = TestKimiK3ToolParser.EOT

    WEATHER_CALL_SECTION = (
        BOT + '<|open|>call tool="get_weather" index="1"<|sep|>'
        '<|open|>argument key="location" type="string"<|sep|>NYC'
        '<|close|>argument<|sep|>'
        '<|close|>call<|sep|>' + EOT)

    @staticmethod
    def _make_args(sample_tools,
                   tool_parser=None,
                   forced_tool_name=None,
                   stream=False):
        from tensorrt_llm.serve.openai_protocol import ChatCompletionRequest
        from tensorrt_llm.serve.postprocess_handlers import ChatPostprocArgs

        request_kwargs = dict(
            model="test-model",
            messages=[{
                "role": "user",
                "content": "What is the weather in NYC?"
            }],
            tools=sample_tools,
            stream=stream,
        )
        if forced_tool_name is not None:
            request_kwargs["tool_choice"] = {
                "type": "function",
                "function": {
                    "name": forced_tool_name
                },
            }
        req = ChatCompletionRequest(**request_kwargs)
        args = ChatPostprocArgs.from_request(req)
        args.tool_parser = tool_parser
        args.num_prompt_tokens = 5
        return args

    @staticmethod
    def _fake_response(text, finish_reason="stop"):
        from types import SimpleNamespace
        output = SimpleNamespace(index=0,
                                 text=text,
                                 token_ids=[1, 2, 3],
                                 finish_reason=finish_reason,
                                 stop_reason=None,
                                 disaggregated_params=None)
        return SimpleNamespace(outputs=[output], cached_tokens=0)

    @staticmethod
    def _stream_deltas(args, chunks, finish_reason="stop"):
        """Feed text chunks through the streaming postprocessor.

        Returns the decoded SSE payloads (one dict per emitted chunk).
        """
        from types import SimpleNamespace

        from tensorrt_llm.serve.postprocess_handlers import \
            chat_stream_post_processor

        payloads = []
        for chunk_index, chunk in enumerate(chunks):
            last = chunk_index == len(chunks) - 1
            output = SimpleNamespace(
                index=0,
                text_diff=chunk,
                token_ids_diff=[chunk_index],
                logprobs_diff=[],
                finish_reason=finish_reason if last else None,
                stop_reason=None,
                length=chunk_index + 1,
                disaggregated_params=None,
            )
            rsp = SimpleNamespace(outputs=[output],
                                  cached_tokens=0,
                                  id="test-request",
                                  _done=last)
            for line in chat_stream_post_processor(rsp, args):
                payloads.append(json.loads(line[len("data: "):]))
        return payloads

    def test_forced_choice_k3_extracts_arguments(self, sample_tools):
        """The headline TRTLLM-15176 bug.

        A forced tool_choice must return the extracted JSON, not the raw
        preamble + XTML markup.
        """
        from tensorrt_llm.serve.postprocess_handlers import \
            chat_response_post_processor

        args = self._make_args(sample_tools,
                               tool_parser="kimi_k3",
                               forced_tool_name="get_weather")
        rsp = self._fake_response("Let me check. " + self.WEATHER_CALL_SECTION)

        choice = chat_response_post_processor(rsp, args).choices[0]

        assert choice.finish_reason == "tool_calls"
        assert len(choice.message.tool_calls) == 1
        function = choice.message.tool_calls[0].function
        assert function.name == "get_weather"
        assert json.loads(function.arguments) == {"location": "NYC"}
        assert choice.message.content == "Let me check. "

    def test_forced_choice_k3_name_mismatch_uses_request_name(
            self, sample_tools):
        """The caller chose the tool; a disagreeing model only earns a log."""
        from tensorrt_llm.serve.postprocess_handlers import \
            chat_response_post_processor

        args = self._make_args(sample_tools,
                               tool_parser="kimi_k3",
                               forced_tool_name="search_web")
        rsp = self._fake_response(self.WEATHER_CALL_SECTION)

        choice = chat_response_post_processor(rsp, args).choices[0]

        function = choice.message.tool_calls[0].function
        assert function.name == "search_web"
        assert json.loads(function.arguments) == {"location": "NYC"}

    def test_forced_choice_k3_no_markup_returns_content(self, sample_tools):
        """Return content when a K3 forced call produced no markup.

        Nothing constrains K3 forced calls; if the model emits no markup,
        return the text as content rather than garbage arguments.
        """
        from tensorrt_llm.serve.postprocess_handlers import \
            chat_response_post_processor

        args = self._make_args(sample_tools,
                               tool_parser="kimi_k3",
                               forced_tool_name="get_weather")
        rsp = self._fake_response("I cannot help with that.")

        choice = chat_response_post_processor(rsp, args).choices[0]

        assert choice.finish_reason == "stop"
        assert not choice.message.tool_calls
        assert choice.message.content == "I cannot help with that."

    def test_forced_choice_raw_passthrough_keeps_text_as_arguments(
            self, sample_tools):
        """Ungated parsers report the constrained arguments value.

        Parsers without ``extracts_forced_tool_calls`` rely on JSON-schema
        guided decoding, so the generated text is the arguments value and is
        reported as-is, with finish_reason="tool_calls".
        """
        from tensorrt_llm.serve.postprocess_handlers import \
            chat_response_post_processor

        args = self._make_args(sample_tools,
                               tool_parser="qwen3",
                               forced_tool_name="get_weather")
        rsp = self._fake_response('{"location": "NYC"}')

        choice = chat_response_post_processor(rsp, args).choices[0]

        assert choice.finish_reason == "tool_calls"
        function = choice.message.tool_calls[0].function
        assert function.name == "get_weather"
        assert function.arguments == '{"location": "NYC"}'

    def test_forced_choice_truncates_overrun_arguments(self, sample_tools):
        """A model that overruns the arguments must not leak the tail.

        Generation starts inside the tool call, so an unconstrained model
        closes the enclosing object, emits the parser's end tag and keeps
        talking. Only the arguments value may be reported, otherwise the
        caller cannot json.loads() it.
        """
        from tensorrt_llm.serve.postprocess_handlers import \
            chat_response_post_processor

        args = self._make_args(sample_tools,
                               tool_parser="qwen3",
                               forced_tool_name="get_weather")
        rsp = self._fake_response(
            ' {"location":"NYC"}}\n</tool_call>\n\nOkay, the user just said '
            '"Just say hi."\n</think>\n\nHello! How can I assist you today?')

        choice = chat_response_post_processor(rsp, args).choices[0]

        function = choice.message.tool_calls[0].function
        assert function.name == "get_weather"
        assert json.loads(function.arguments) == {"location": "NYC"}
        assert choice.finish_reason == "tool_calls"

    def test_forced_choice_incomplete_arguments_returns_content(
            self, sample_tools):
        """A truncated arguments value must not be reported as a tool call.

        Guided decoding constrains the shape but not the length: hitting the
        token budget leaves a partial JSON value. Reporting it as `arguments`
        would hand the caller something json.loads() rejects, and claiming
        finish_reason="tool_calls" would assert a call that never completed.
        """
        from tensorrt_llm.serve.postprocess_handlers import \
            chat_response_post_processor

        args = self._make_args(sample_tools,
                               tool_parser="qwen3",
                               forced_tool_name="get_weather")
        rsp = self._fake_response('{"location": "San Fra')

        choice = chat_response_post_processor(rsp, args).choices[0]

        assert not choice.message.tool_calls
        assert choice.message.content == '{"location": "San Fra'
        assert choice.finish_reason != "tool_calls"

    def test_forced_choice_empty_generation_returns_no_tool_call(
            self, sample_tools):
        """An empty generation must not become a tool call with empty args."""
        from tensorrt_llm.serve.postprocess_handlers import \
            chat_response_post_processor

        args = self._make_args(sample_tools,
                               tool_parser="qwen3",
                               forced_tool_name="get_weather")
        rsp = self._fake_response("")

        choice = chat_response_post_processor(rsp, args).choices[0]

        assert not choice.message.tool_calls
        assert choice.finish_reason != "tool_calls"

    def test_forced_choice_streaming_stops_at_end_of_arguments(
            self, sample_tools):
        """The stream stops emitting once the arguments value completes."""
        args = self._make_args(sample_tools,
                               tool_parser="qwen3",
                               forced_tool_name="get_weather",
                               stream=True)
        payloads = self._stream_deltas(
            args,
            ['{"location"', ':"NYC"}', '}\n</tool_call>\n', 'Hello there!'])

        deltas = [p["choices"][0]["delta"] for p in payloads if p["choices"]]
        tool_deltas = [d for d in deltas if d.get("tool_calls")]
        arguments = "".join(
            d["tool_calls"][0]["function"].get("arguments") or ""
            for d in tool_deltas)
        assert json.loads(arguments) == {"location": "NYC"}
        # Name and id are sent exactly once, on the opening delta.
        named = [
            d for d in tool_deltas if d["tool_calls"][0]["function"].get("name")
        ]
        assert len(named) == 1
        assert named[0]["tool_calls"][0]["function"]["name"] == "get_weather"
        assert len([d for d in tool_deltas
                    if d["tool_calls"][0].get("id")]) == 1
        # A forced call produces no assistant text. Content here means the raw
        # generation -- including the overrun tail -- leaked into the message.
        assert not any(d.get("content") for d in deltas), (
            "forced call leaked content: "
            f"{[d.get('content') for d in deltas if d.get('content')]}")

    def test_forced_choice_k3_streaming_extracts(self, sample_tools):
        """Extract the forced call from a streamed K3 response.

        The preamble streams as content deltas and the extracted call as a
        tool_calls delta with the forced name.
        """
        args = self._make_args(sample_tools,
                               tool_parser="kimi_k3",
                               forced_tool_name="get_weather",
                               stream=True)
        section = self.WEATHER_CALL_SECTION
        payloads = self._stream_deltas(
            args, ["Let me check. ", section[:40], section[40:]])

        deltas = [p["choices"][0]["delta"] for p in payloads if p["choices"]]
        content = "".join(d.get("content") or "" for d in deltas)
        assert content == "Let me check. "
        tool_deltas = [d for d in deltas if d.get("tool_calls")]
        assert len(tool_deltas) == 1
        function = tool_deltas[0]["tool_calls"][0]["function"]
        assert function["name"] == "get_weather"
        assert json.loads(function["arguments"]) == {"location": "NYC"}
        finish_reasons = [
            p["choices"][0].get("finish_reason") for p in payloads
            if p["choices"]
        ]
        assert finish_reasons[-1] == "tool_calls"

    def test_streaming_early_end_flushes_buffered_call(self, sample_tools):
        """Flush the buffered call when an auto-choice stream ends early.

        A stream that ends before EOT still emits the buffered call via the
        finalization hook instead of dropping it.
        """
        args = self._make_args(sample_tools, tool_parser="kimi_k3", stream=True)
        truncated = self.WEATHER_CALL_SECTION[:-len(self.EOT)]
        payloads = self._stream_deltas(
            args, ["Sure. ", truncated[:30], truncated[30:]])

        deltas = [p["choices"][0]["delta"] for p in payloads if p["choices"]]
        tool_deltas = [d for d in deltas if d.get("tool_calls")]
        assert len(tool_deltas) == 1
        function = tool_deltas[0]["tool_calls"][0]["function"]
        assert function["name"] == "get_weather"
        assert json.loads(function["arguments"]) == {"location": "NYC"}
        finish_reasons = [
            p["choices"][0].get("finish_reason") for p in payloads
            if p["choices"]
        ]
        assert finish_reasons[-1] == "tool_calls"

    def test_forced_choice_k3_streaming_no_markup_is_content(
            self, sample_tools):
        """Streaming honest fallback: no markup means content, not a call."""
        args = self._make_args(sample_tools,
                               tool_parser="kimi_k3",
                               forced_tool_name="get_weather",
                               stream=True)
        payloads = self._stream_deltas(args, ["I cannot ", "help with that."])

        deltas = [p["choices"][0]["delta"] for p in payloads if p["choices"]]
        assert not any(d.get("tool_calls") for d in deltas)
        content = "".join(d.get("content") or "" for d in deltas)
        assert content == "I cannot help with that."
        finish_reasons = [
            p["choices"][0].get("finish_reason") for p in payloads
            if p["choices"]
        ]
        assert finish_reasons[-1] == "stop"


class TestConfigureParserSpecialTokenDecoding:
    """Test parser-specific detokenization settings in the OpenAI server."""

    @staticmethod
    def _configure(reasoning_parser_name: str | None = None,
                   tool_parser_name: str | None = None,
                   has_tools: bool = False) -> SamplingParams:
        from tensorrt_llm.serve.openai_server import \
            _configure_parser_special_token_decoding

        sampling_params = SamplingParams()
        _configure_parser_special_token_decoding(
            sampling_params,
            reasoning_parser_name=reasoning_parser_name,
            tool_parser_name=tool_parser_name,
            has_tools=has_tools)
        return sampling_params

    def test_kimi_k3_reasoning_parser_preserves_compact_xtml(self) -> None:
        sampling_params = self._configure(reasoning_parser_name="kimi_k3")

        assert sampling_params.skip_special_tokens is False
        assert sampling_params.spaces_between_special_tokens is False

    def test_kimi_k3_tool_parser_preserves_compact_xtml(self) -> None:
        sampling_params = self._configure(tool_parser_name="KIMI_K3",
                                          has_tools=True)

        assert sampling_params.skip_special_tokens is False
        assert sampling_params.spaces_between_special_tokens is False

    def test_tool_parser_does_not_apply_without_tools(self) -> None:
        sampling_params = self._configure(tool_parser_name="kimi_k3")

        assert sampling_params.skip_special_tokens is True
        assert sampling_params.spaces_between_special_tokens is True

    def test_other_raw_token_parser_keeps_spacing_contract(self) -> None:
        sampling_params = self._configure(tool_parser_name="deepseek_v32",
                                          has_tools=True)

        assert sampling_params.skip_special_tokens is False
        assert sampling_params.spaces_between_special_tokens is True


# ============================================================================
# Forced tool call: argument truncation — TRTLLM-12758
# ============================================================================


class TestParserExtractsForcedToolCalls:
    """Routing between the two forced-call strategies.

    Parsers that extract forced calls from their own markup must NOT be sent
    down the grammar path: none of them supports structural tags, so the
    server would reject every forced request against them and the extraction
    path in the post-processor would be unreachable.
    """

    def test_kimi_k3_is_routed_to_extraction(self) -> None:
        from tensorrt_llm.serve.openai_server import \
            _parser_extracts_forced_tool_calls

        assert _parser_extracts_forced_tool_calls("kimi_k3") is True
        # Guard the premise: K3 has no structural-tag support, so without the
        # routing above it would be rejected outright.
        from tensorrt_llm.serve.tool_parser.tool_parser_factory import \
            ToolParserFactory
        assert ToolParserFactory.parsers["kimi_k3"]().supports_structural_tag(
        ) is False

    def test_json_parsers_use_the_grammar_path(self) -> None:
        from tensorrt_llm.serve.openai_server import \
            _parser_extracts_forced_tool_calls

        assert _parser_extracts_forced_tool_calls("qwen3") is False
        assert _parser_extracts_forced_tool_calls(None) is False
        assert _parser_extracts_forced_tool_calls("not_a_parser") is False


class TestForcedToolArgumentsEnd:
    """Test ``forced_tool_arguments_end`` from postprocess_handlers.

    The forced-call path prefix-injects ``{"name": X, "arguments":`` into the
    prompt, so generation begins inside a tool-call object. Whatever the model
    produces after the arguments value -- the enclosing ``}``, the parser's end
    tag, reasoning, ordinary prose -- must not be reported as ``arguments``.
    """

    def test_returns_none_for_incomplete_json(self) -> None:
        assert forced_tool_arguments_end('{"location": "San Fra') is None

    def test_returns_none_for_empty_or_blank(self) -> None:
        assert forced_tool_arguments_end("") is None
        assert forced_tool_arguments_end("   ") is None

    def test_exact_json_consumes_everything(self) -> None:
        text = '{"location": "Paris"}'
        assert forced_tool_arguments_end(text) == len(text)

    def test_skips_leading_whitespace(self) -> None:
        text = '  {"location": "Paris"}'
        assert forced_tool_arguments_end(text) == len(text)

    def test_truncates_model_overrun(self) -> None:
        """Regression for the observed CI failure.

        The model closed the enclosing tool-call object, emitted the end tag,
        then carried on with reasoning and a chat reply. Only the leading
        arguments object may survive.
        """
        overrun = (' {"location":"Hello", "unit":"fahrenheit"}}\n</tool_call>\n'
                   '\nOkay, the user just said "Just say hi."\n</think>\n\n'
                   'Hello! How can I assist you today?')
        end = forced_tool_arguments_end(overrun)
        assert end is not None
        arguments = overrun[:end]
        # Must be parseable on its own -- this is exactly what the failing
        # test did with ``json.loads(forced_call.function.arguments)``.
        assert json.loads(arguments) == {
            "location": "Hello",
            "unit": "fahrenheit",
        }

    def test_streaming_incremental_cutoff(self) -> None:
        """Feeding the overrun in chunks yields the same arguments and stops."""
        chunks = [
            ' {"location":', '"Hello", "unit"', ':"fahrenheit"}', '}\n</tool_',
            'call>\n\nHello!'
        ]
        buffered, sent_len, done, streamed = "", 0, False, ""
        for chunk in chunks:
            if done:
                continue
            buffered += chunk
            end = forced_tool_arguments_end(buffered)
            limit = len(buffered) if end is None else end
            if end is not None:
                done = True
            streamed += buffered[sent_len:limit]
            sent_len = max(sent_len, limit)
        assert done, "stream never detected the end of the arguments"
        assert json.loads(streamed) == {
            "location": "Hello",
            "unit": "fahrenheit",
        }


# ============================================================================
# Glm47ToolParser: repairing tool names the model mangled
# ============================================================================
#
# Names drawn from production traces of GLM-5.2 driving a coding agent. The
# arguments in these calls parsed cleanly; only the name was damaged, so a
# repaired name turns a call the client would have rejected back into the one
# the model meant.


def _glm_tools(*names):
    """Build the tools list a GLM request would carry."""
    return [
        ChatCompletionToolsParam(type="function",
                                 function=FunctionDefinition(
                                     name=n,
                                     description=f"{n} description",
                                     parameters={
                                         "type": "object",
                                         "properties": {}
                                     })) for n in names
    ]


AGENT_TOOLS = _glm_tools("exec_command", "apply_patch", "update_plan",
                         "spawn_agent", "followup_task", "exec")

# (emitted name, declared tool it means). The group qualifier is what a model
# prompted with nested tool groups emits for a tool declared unqualified.
QUALIFIED_NAMES = [
    ("functions.exec_command", "exec_command"),
    ("functions.apply_patch", "apply_patch"),
    ("functions.update_plan", "update_plan"),
    ("collaboration.spawn_agent", "spawn_agent"),
    ("functions.collaboration.followup_task", "followup_task"),
]

# An unbalanced tag shifts where the name regex starts or stops, fusing markup
# onto an otherwise exact name.
MARKUP_FUSED_NAMES = [
    ("apply_patch</arg_value>", "apply_patch"),
    ("exec</arg_value>", "exec"),
    ("<tool_call>exec_command", "exec_command"),
    ("coll</arg_value><tool_call>collaboration.spawn_agent", "spawn_agent"),
]

# Narration and source code that happen to *contain* a declared tool name.
# Repairing these would fabricate a call the model never made, so each must
# survive as-is rather than collapsing onto the tool whose name it mentions.
PROSE_NAMES = [
    "ops_check - re-querying for current status. Since my last report",
    "exec_command depended on the tools. Let me use the correct approach",
    "exec_sudo_command onCompleteCommand=\"export SOLSWARM_SKILLS_DIR=/x\"",
    "collaboration.immediately_agent(\"target\" => \"/root/kernel_builder\")",
    "exec_command(cmd=\"cat /workspace/sol/kernel.cu\"",
]


def _one_call(name, arg_key="path", arg_value="/workspace"):
    return (f"<tool_call>{name}"
            f"<arg_key>{arg_key}</arg_key>"
            f"<arg_value>{arg_value}</arg_value>"
            f"</tool_call>")


class TestGlm47MangledToolNames:
    """Glm47ToolParser repairs mangled names without inventing calls."""

    @pytest.mark.parametrize("emitted,declared", QUALIFIED_NAMES)
    def test_group_qualifier_is_stripped(self, emitted, declared):
        result = Glm47ToolParser().detect_and_parse(_one_call(emitted),
                                                    AGENT_TOOLS)

        assert len(result.calls) == 1
        assert result.calls[0].name == declared

    @pytest.mark.parametrize("emitted,declared", MARKUP_FUSED_NAMES)
    def test_fused_markup_is_stripped(self, emitted, declared):
        result = Glm47ToolParser().detect_and_parse(_one_call(emitted),
                                                    AGENT_TOOLS)

        assert len(result.calls) == 1
        assert result.calls[0].name == declared

    def test_repair_preserves_the_arguments(self):
        """The arguments are why repairing beats dropping."""
        text = _one_call("functions.exec_command", "cmd", "ls -la /workspace")

        result = Glm47ToolParser().detect_and_parse(text, AGENT_TOOLS)

        assert len(result.calls) == 1
        assert result.calls[0].name == "exec_command"
        assert json.loads(result.calls[0].parameters) == {
            "cmd": "ls -la /workspace"
        }

    @pytest.mark.parametrize("prose", PROSE_NAMES)
    def test_prose_is_never_repaired_into_a_call(self, prose):
        """Narration mentioning a tool must not become a call to it."""
        result = Glm47ToolParser().detect_and_parse(_one_call(prose),
                                                    AGENT_TOOLS)

        declared = {t.function.name for t in AGENT_TOOLS}
        for call in result.calls:
            assert call.name not in declared, (
                f"prose was repaired into a call to {call.name!r}")

    def test_unrepairable_name_is_still_forwarded(self):
        """TRT-LLM forwards unmatched names with tool_index=-1; unchanged."""
        result = Glm47ToolParser().detect_and_parse(_one_call("no_such_tool"),
                                                    AGENT_TOOLS)

        assert len(result.calls) == 1
        assert result.calls[0].name == "no_such_tool"
        assert result.calls[0].tool_index == -1

    def test_declared_name_is_untouched(self):
        result = Glm47ToolParser().detect_and_parse(_one_call("exec_command"),
                                                    AGENT_TOOLS)

        assert len(result.calls) == 1
        assert result.calls[0].name == "exec_command"

    def test_partial_suffix_match_is_not_a_repair(self):
        """`my_exec_command` ends with a declared name but is not one."""
        result = Glm47ToolParser().detect_and_parse(
            _one_call("my_exec_command"), AGENT_TOOLS)

        assert len(result.calls) == 1
        assert result.calls[0].name == "my_exec_command"

    @pytest.mark.parametrize("emitted,declared",
                             QUALIFIED_NAMES + MARKUP_FUSED_NAMES)
    def test_streaming_repairs_the_same_way(self, emitted, declared):
        """A streamed response must not name the tool differently."""
        parser = Glm47ToolParser()
        names = []
        for chunk in _one_call(emitted):
            result = parser.parse_streaming_increment(chunk, AGENT_TOOLS)
            names += [c.name for c in result.calls if c.name]

        assert names == [declared]

    @pytest.mark.parametrize("prose", PROSE_NAMES)
    def test_streaming_never_repairs_prose(self, prose):
        parser = Glm47ToolParser()
        names = []
        for chunk in _one_call(prose):
            result = parser.parse_streaming_increment(chunk, AGENT_TOOLS)
            names += [c.name for c in result.calls if c.name]

        declared = {t.function.name for t in AGENT_TOOLS}
        assert not (set(names) & declared)


# ============================================================================
# Glm47ToolParser: a streamed argument is typed like the same argument parsed
# whole
# ============================================================================
#
# An argument's type comes from the declared schema or from nowhere. The
# schema lookup resolves the emitted name the way delivery does (`exec` finds
# `functions.exec`), so a bare spelling no longer hides the declaration; a
# parameter the schema calls a string - and any parameter no schema describes -
# is delivered as the raw text between the markers, verbatim.
#
# Both halves were learned from corrupted traffic. Typing off the value's
# shape turned a `cell_id` the schema calls a string into the integer 99797
# (rejected by the client's type check) and str()-round-tripped a freeform
# payload of `true` into Python's "True" (executed by the client, to a
# ReferenceError). Typing off the schema under the *unresolved* name missed
# the declaration entirely, so `max_output_tokens` went out as the string
# "8000" and the tool rejected `invalid type: string "8000", expected usize`.
# Guessing served one of those failures at the expense of the others; the
# schema serves them all, and where there is no schema nothing licenses a
# conversion.
#
# The property below is not "produces a number". It is equality with
# `detect_and_parse`: for every text and every chunking, the arguments a
# streaming client assembles must equal the arguments the same text yields when
# parsed whole. That is the only definition of correct that does not require
# inventing a second typing rule for the streaming path to follow, and it is
# what these tests assert even where the answer looks odd - the JSON-ish and
# boolean cases below are raw text on purpose, not accidents.
#
# Assertions are on the assembled call. `_process_xml_to_json_streaming` emits
# a fragment stream - `{`, `"key": `, the value, `, ` - that is only JSON once
# concatenated, and concatenating it is `responses_utils`' job, so these tests
# join the fragments with the same functions the serving layer uses. Asserting
# on one delta would measure a piece rather than what a client receives.


def _glm47_tool(name, properties=None):
    """One declared tool; `properties` is its JSON-schema property map."""
    return ChatCompletionToolsParam(type="function",
                                    function=FunctionDefinition(
                                        name=name,
                                        parameters={
                                            "type": "object",
                                            "properties": properties or {},
                                        }))


def _glm47_call(name, *pairs):
    """GLM-4.7 markup for one call with the given (key, value) arguments."""
    body = "".join(f"<arg_key>{key}</arg_key><arg_value>{value}</arg_value>"
                   for key, value in pairs)
    return f"<tool_call>{name}{body}</tool_call>"


# The shape live traffic has: the model calls a tool the request never
# declared, so `get_argument_type` finds nothing and the type can only come
# from the value. This is where the defect lived.
NO_SCHEMA_TOOLS = [
    _glm47_tool("exec"),
    _glm47_tool("wait"),
    _glm47_tool("request_user_input"),
]


def _streamed_calls(chunks, tools, parser_factory=Glm47ToolParser):
    """The tool calls a client assembles from `chunks`.

    Mirrors `_generate_streaming_event`: fragments accumulate per call, the
    parser is drained once the stream ends because it reports at most one
    finished call per increment, and a call the stream was cut off inside -
    or saw abandoned - is dropped.
    """
    parser = parser_factory()
    fragments = {}
    for chunk in chunks:
        result = parser.parse_streaming_increment(chunk, tools)
        _accumulate_tool_call_fragments(fragments, result.calls)
    _, flushed, unfinished = _flush_tool_parser(tools=tools,
                                                output_index=0,
                                                tool_parser_dict={0: parser})
    _accumulate_tool_call_fragments(fragments, flushed)
    return _assembled_tool_calls(fragments, unfinished,
                                 _abandoned_tool_indices(parser))


def _streamed_arguments(chunks, tools):
    """The arguments a client can run, which is the joined fragments parsed."""
    return [
        json.loads(call.parameters) for call in _streamed_calls(chunks, tools)
    ]


def _whole_arguments(text, tools):
    """What `detect_and_parse` concludes - the answer streaming must match."""
    result = Glm47ToolParser().detect_and_parse(text, tools)
    return [json.loads(call.parameters) for call in result.calls]


def _chunkings(text):
    """Every cut of `text` this suite checks, worst case first.

    One character per delta is not hypothetical: it is what the parser sees
    when the model emits a value token by token. The single split points cover
    every boundary that can land inside a tag or inside a value, which is the
    sweep the closing-tag handling has to survive.
    """
    yield "arriving whole", [text]
    yield "one character per delta", list(text)
    for size in (2, 3, 5, 8, 13, 40):
        yield f"{size} characters per delta", [
            text[i:i + size] for i in range(0, len(text), size)
        ]
    for i in range(1, len(text)):
        yield f"split at {i}", [text[:i], text[i:]]


def _multi_cut_chunkings(text, focus="</tool_call>", max_cuts=3):
    """Chunkings that tear `focus` into several pieces at once.

    `_chunkings` cuts a text at one point at a time, but a tag can just as
    well arrive in three or more pieces. A text no longer than `focus` is
    swept exhaustively - all 2048 chunkings of a bare `</tool_call>`; a
    longer one gets every combination of up to `max_cuts` cuts among the
    boundaries in and around each occurrence of `focus`.
    """
    if len(text) <= len(focus):
        positions = range(1, len(text))
        counts = range(len(text))
    else:
        starts = [m.start() for m in re.finditer(re.escape(focus), text)]
        positions = sorted({
            position
            for start in starts
            for position in range(max(1, start),
                                  min(len(text), start + len(focus) + 1))
        })
        counts = range(1, max_cuts + 1)
    for count in counts:
        for cuts in itertools.combinations(positions, count):
            bounds = (0, *cuts, len(text))
            yield f"cut at {cuts}", [
                text[a:b] for a, b in zip(bounds, bounds[1:])
            ]


def _assert_chunking_never_matters(text, tools):
    """The property, swept over every chunking. Returns the agreed arguments."""
    expected = _whole_arguments(text, tools)
    for label, chunks in _chunkings(text):
        assert _streamed_arguments(chunks, tools) == expected, (
            f"streaming disagreed with detect_and_parse when {label}")
    return expected


class TestGlm47StreamedArgumentTypes:
    """A streamed argument must equal the same argument parsed whole."""

    def test_numeric_argument_without_schema_stays_the_text_the_model_wrote(
            self):
        """No schema, no conversion.

        This exact shape once justified guessing: a tool takes
        `max_output_tokens` as a usize, the request never declared it, and the
        string "8000" was rejected. But the guess that repaired it corrupted
        declared-string parameters (`cell_id` "99797" arrived as an integer)
        and freeform payloads (`true` arrived as Python's "True") - see
        TestGlm47ArgumentCorruptions. The usize case is served by the schema
        instead: on the fleets that recorded it the tools *are* declared, only
        qualified, and the lookup now resolves the bare name (test below).
        For a tool that truly was never declared, nothing can validate a type,
        and the raw text is the only answer that cannot be wrong about it.
        """
        text = _glm47_call("exec_command", ("cmd", "ls -la /workspace"),
                           ("max_output_tokens", "8000"),
                           ("yield_time_ms", "10000"))

        arguments = _assert_chunking_never_matters(text, NO_SCHEMA_TOOLS)

        assert arguments == [{
            "cmd": "ls -la /workspace",
            "max_output_tokens": "8000",
            "yield_time_ms": "10000",
        }]
        assert isinstance(arguments[0]["max_output_tokens"], str)

    def test_a_bare_name_still_finds_the_qualified_schema(self):
        """The declared type survives the model's bare spelling.

        3102 of 3730 calls on the measured fleet wrote `exec` for a declared
        `functions.exec`. The schema lookup resolves the name the same way
        delivery does, so `max_output_tokens` is a number *because the schema
        says so*, not because it looks like one.
        """
        tools = [
            _glm47_tool(
                "functions.exec_command", {
                    "cmd": {
                        "type": "string"
                    },
                    "max_output_tokens": {
                        "type": "integer"
                    },
                })
        ]
        text = _glm47_call("exec_command", ("cmd", "ls"),
                           ("max_output_tokens", "8000"))

        arguments = _assert_chunking_never_matters(text, tools)

        assert arguments == [{"cmd": "ls", "max_output_tokens": 8000}]
        assert isinstance(arguments[0]["max_output_tokens"], int)

    def test_numeric_argument_with_a_numeric_schema_is_a_number(self):
        """Already worked through the schema path; must keep working."""
        tools = [
            _glm47_tool("resize", {
                "width": {
                    "type": "integer"
                },
                "ratio": {
                    "type": "number"
                },
            })
        ]
        text = _glm47_call("resize", ("width", "1920"), ("ratio", "1.5"))

        arguments = _assert_chunking_never_matters(text, tools)

        assert arguments == [{"width": 1920, "ratio": 1.5}]

    def test_string_schema_beats_a_numeric_looking_value(self):
        """A declared type wins over appearance, as the whole path already did."""
        tools = [_glm47_tool("store", {"id": {"type": "string"}})]
        text = _glm47_call("store", ("id", "8000"))

        arguments = _assert_chunking_never_matters(text, tools)

        assert arguments == [{"id": "8000"}]

    def test_ordinary_string_value_is_unchanged(self):
        tools = [_glm47_tool("exec_command", {"cmd": {"type": "string"}})]
        command = "grep -rn 'needle' /workspace --include=*.py"
        text = _glm47_call("exec_command", ("cmd", command))

        arguments = _assert_chunking_never_matters(text, tools)

        assert arguments == [{"cmd": command}]

    # JSON-looking text without a schema is still text. Parsing it used to be
    # the guess that corrupted string parameters; a caller that declared the
    # parameter an object gets an object (see the typed test above), and a
    # caller that declared nothing gets exactly what the model wrote. These
    # are pinned so a change to either path is visible, but what the streaming
    # path owes is the equality `_assert_chunking_never_matters` checks, not
    # these literals.
    @pytest.mark.parametrize("value", [
        '{"a": 1}',
        '{"a": {"b": [1, 2]}}',
        "[1, 2, 3]",
        '[{"description": "x"}]',
        "{}",
        "[]",
    ])
    def test_json_value_without_schema_stays_text(self, value):
        text = _glm47_call("exec_command", ("payload", value))

        arguments = _assert_chunking_never_matters(text, NO_SCHEMA_TOOLS)

        assert arguments == [{"payload": value}]

    @pytest.mark.parametrize("value", [
        "true",
        "false",
        "null",
    ])
    def test_boolean_and_null_without_schema_stay_text(self, value):
        text = _glm47_call("exec_command", ("flag", value))

        arguments = _assert_chunking_never_matters(text, NO_SCHEMA_TOOLS)

        assert arguments == [{"flag": value}]

    def test_empty_value_matches_the_whole_parse(self):
        """`<arg_value></arg_value>` - measured, not invented: empty string."""
        text = _glm47_call("exec_command", ("cmd", "ls"), ("stdin", ""))

        arguments = _assert_chunking_never_matters(text, NO_SCHEMA_TOOLS)

        assert arguments == [{"cmd": "ls", "stdin": ""}]

    def test_every_chunk_boundary_gives_what_arrived_whole(self):
        """The sweep: re-cut the same text everywhere, including per character.

        Typing at `</arg_value>` is only sound if no boundary can change the
        answer, so this compares every chunking against the value arriving in
        one piece rather than against a hand-written expectation.
        """
        tools = [
            _glm47_tool("exec_command", {
                "cmd": {
                    "type": "string"
                },
                "opts": {
                    "type": "object"
                },
            })
        ]
        text = _glm47_call("exec_command", ("cmd", "python3 -c 'print(1 + 1)'"),
                           ("opts", '{"timeout": 30, "shell": true}'),
                           ("max_output_tokens", "8000"), ("stdin", ""))

        whole = _streamed_arguments([text], tools)
        assert whole == [{
            "cmd": "python3 -c 'print(1 + 1)'",
            "opts": {
                "timeout": 30,
                "shell": True
            },
            # Not declared by the tool, so no conversion is licensed.
            "max_output_tokens": "8000",
            "stdin": "",
        }]

        for label, chunks in _chunkings(text):
            assert _streamed_arguments(chunks, tools) == whole, (
                f"re-chunking changed the arguments when {label}")

    # A value may contain something that reads like the closing tag. The tag
    # buffer holds back anything that could still grow into `</arg_value>` and
    # releases it once it cannot, which is what lets these through - and a
    # value containing the *complete* tag survives too, because the close is
    # only honored when the following text confirms it as structure; see
    # TestGlm47ArgumentCorruptions. A value whose *last* characters are a bare
    # prefix of the tag used to defeat the streaming path all the same: the
    # dead buffer was released whole, taking the real tag's opening `<` with
    # it, so the close was never seen and the value never delivered. The
    # release now stops before a trailing `<` and the match re-anchors there
    # (see split_dead_close_buffer, and corruption 4 in
    # TestGlm47ArgumentCorruptions for the trace), so the trailing-prefix
    # shapes below hold on every split too.
    @pytest.mark.parametrize("value", [
        "</arg_valueX>",
        "a</arg_valuex b",
        "<arg_value>",
        "<arg_key>not a key</arg_key>",
        "<b>bold</b>",
        "prints </arg_value then more text",
        "1 < 2 && 3 > 2",
        "ends with a template open<",
        "closes with </",
        "almost the tag </arg_valu",
        "almost the whole tag </arg_value",
    ])
    def test_a_value_that_looks_like_markup_survives(self, value):
        tools = [_glm47_tool("echo", {"text": {"type": "string"}})]
        text = _glm47_call("echo", ("text", value))

        arguments = _assert_chunking_never_matters(text, tools)

        assert arguments == [{"text": value}]

    @pytest.mark.parametrize("value", [
        "template<",
        "path</",
        "nearly </arg_valu",
    ])
    def test_a_trailing_marker_prefix_survives_before_another_argument(
            self, value):
        """The released `<` joins the value before the close is confirmed.

        With another argument following, the pending `</arg_value>` is
        confirmed by `<arg_key>` rather than by the call's end, so the
        trailing prefix has to be back in the value by the time
        PENDING_CLOSE commits it - releasing it any later would deliver the
        next pair fused onto this value.
        """
        tools = [
            _glm47_tool("echo", {
                "text": {
                    "type": "string"
                },
                "more": {
                    "type": "string"
                },
            })
        ]
        text = _glm47_call("echo", ("text", value), ("more", "x"))

        arguments = _assert_chunking_never_matters(text, tools)

        assert arguments == [{"text": value, "more": "x"}]

    def test_mixed_types_in_one_call_assemble_into_valid_json(self):
        """Each argument typed on its own, and the object still closes.

        `opts` is the case that matters: a value which is itself an object ends
        with `}`, which the closing logic used to read as the argument object
        having already been closed - delivering `{"opts": {...}` one brace
        short of parseable.
        """
        tools = [
            _glm47_tool(
                "configure", {
                    "name": {
                        "type": "string"
                    },
                    "retries": {
                        "type": "integer"
                    },
                    "opts": {
                        "type": "object"
                    },
                })
        ]
        text = _glm47_call("configure", ("name", "worker-1"), ("retries", "3"),
                           ("opts", '{"verbose": true}'), ("budget", "8000"),
                           ("note", ""))

        arguments = _assert_chunking_never_matters(text, tools)

        assert arguments == [{
            "name": "worker-1",
            "retries": 3,
            "opts": {
                "verbose": True
            },
            # `budget` is not in the schema: raw text, no guessing.
            "budget": "8000",
            "note": "",
        }]

    def test_an_argument_whose_value_is_an_object_still_closes_the_call(self):
        """The same brace bug with nothing else in the call to mask it."""
        tools = [_glm47_tool("apply", {"patch": {"type": "object"}})]
        text = _glm47_call("apply", ("patch", '{"path": "a.py"}'))

        streamed = _streamed_calls([text], tools)

        assert len(streamed) == 1
        assert streamed[0].parameters.endswith("}}")
        assert json.loads(streamed[0].parameters) == {"patch": {"path": "a.py"}}

    def test_stream_ending_inside_a_value_reports_no_call(self):
        """Unchanged: an unfinished call is dropped, and both paths agree."""
        chunks = [
            "<tool_call>exec_command<arg_key>cmd</arg_key>",
            "<arg_value>ls -la /works",
        ]

        assert _streamed_calls(chunks, NO_SCHEMA_TOOLS) == []
        assert _whole_arguments("".join(chunks), NO_SCHEMA_TOOLS) == []

    def test_several_calls_in_one_response_are_each_typed_alone(self):
        tools = [_glm47_tool("exec_command"), _glm47_tool("wait")]
        text = ("Running the check." +
                _glm47_call("exec_command", ("cmd", "pytest -q"),
                            ("max_output_tokens", "8000")) + " then" +
                _glm47_call("wait", ("seconds", "2.5"), ("reason", "startup")))

        arguments = _assert_chunking_never_matters(text, tools)

        # Both tools are declared without properties, so every argument is
        # schema-less and arrives as the text the model wrote.
        assert arguments == [
            {
                "cmd": "pytest -q",
                "max_output_tokens": "8000"
            },
            {
                "seconds": "2.5",
                "reason": "startup"
            },
        ]

    def test_zero_argument_call_still_sends_an_empty_object(self):
        tools = [_glm47_tool("get_time")]
        text = "<tool_call>get_time</tool_call>"

        arguments = _assert_chunking_never_matters(text, tools)

        assert arguments == [{}]


class TestGlm47ArgumentCorruptions:
    """The four argument corruptions recorded on GLM-5.3 fleets, as traced.

    Each test carries the shape of the traffic that failed. All are swept
    over every chunking, so the streaming and whole-text paths cannot drift
    apart on exactly the inputs that burned.
    """

    def test_string_schema_keeps_a_numeric_value_as_the_string_it_is(self):
        """Corruption 1: schema says string, model writes `99797`.

        The parser delivered the integer 99797 and the client's type
        validation rejected the call. The declared type is the contract;
        the value's shape is not.
        """
        tools = [_glm47_tool("notebook_edit", {"cell_id": {"type": "string"}})]
        text = _glm47_call("notebook_edit", ("cell_id", "99797"))

        arguments = _assert_chunking_never_matters(text, tools)

        assert arguments == [{"cell_id": "99797"}]
        assert isinstance(arguments[0]["cell_id"], str)

    def test_custom_tool_freeform_input_is_verbatim(self):
        """Corruption 2: a freeform payload of `true` arrived as "True".

        A custom (freeform-text) tool is described to the model as a single
        string parameter - the same shape `responses_utils.custom_parameters`
        builds - and its payload is forwarded verbatim. It used to be parsed
        as a JSON literal into Python True and str()-round-tripped: the
        client executed the payload and got `ReferenceError: True is not
        defined`.
        """
        input_arg = responses_utils.CUSTOM_TOOL_INPUT_ARG
        tools = [_glm47_tool("apply_patch", {
            input_arg: {
                "type": "string"
            },
        })]
        text = _glm47_call("apply_patch", (input_arg, "true"))

        arguments = _assert_chunking_never_matters(text, tools)

        assert arguments == [{input_arg: "true"}]
        assert arguments[0][input_arg] != "True"

    @pytest.mark.parametrize("value", ["true", "1e3", "null", "0099797"])
    def test_no_str_round_trip_for_any_literal_looking_string(self, value):
        """The same mechanism for every value str() would rewrite.

        json.loads("1e3") is 1000.0 and str() of it is "1000.0"; str(None)
        is "None". Verbatim passthrough makes the whole class impossible.
        """
        tools = [_glm47_tool("echo", {"text": {"type": "string"}})]
        text = _glm47_call("echo", ("text", value))

        arguments = _assert_chunking_never_matters(text, tools)

        assert arguments == [{"text": value}]

    def test_integer_schema_still_parses_the_number(self):
        """The declared type keeps licensing the conversion it names."""
        tools = [_glm47_tool("resize", {"width": {"type": "integer"}})]
        text = _glm47_call("resize", ("width", "42"))

        arguments = _assert_chunking_never_matters(text, tools)

        assert arguments == [{"width": 42}]
        assert isinstance(arguments[0]["width"], int)

    def test_boolean_schema_serializes_as_json_true(self):
        """A schema-typed boolean reaches the wire as `true`, never `True`."""
        tools = [_glm47_tool("toggle", {"on": {"type": "boolean"}})]
        text = _glm47_call("toggle", ("on", "true"))

        arguments = _assert_chunking_never_matters(text, tools)
        assert arguments == [{"on": True}]

        delivered = Glm47ToolParser().detect_and_parse(
            text, tools).calls[0].parameters
        assert '"on": true' in delivered
        assert "True" not in delivered

    def test_unknown_tool_arguments_pass_through_raw(self):
        """No schema anywhere: raw text. Guessing types is the bug."""
        tools = [_glm47_tool("declared_tool")]
        text = _glm47_call("invented_tool", ("flag", "true"), ("n", "8000"))

        arguments = _assert_chunking_never_matters(text, tools)

        assert arguments == [{"flag": "true", "n": "8000"}]

    # ----- corruption 3: values containing the markup delimiters -----

    def test_a_value_containing_the_closing_tag_survives(self):
        """Corruption 3: a code string containing `</arg_value>` mid-text.

        The value used to be cut at the first `</arg_value>` substring. The
        close now only counts when the following text is consistent with the
        enclosing structure - the next `<arg_key>` or the end of the call -
        so the quoted tag stays inside the value on both parse paths.
        """
        payload = 'let s = "</arg_value>"; console.log(s)'
        tools = [
            _glm47_tool("run_js", {
                "code": {
                    "type": "string"
                },
                "timeout": {
                    "type": "integer"
                },
            })
        ]
        text = _glm47_call("run_js", ("code", payload), ("timeout", "30"))

        arguments = _assert_chunking_never_matters(text, tools)

        assert arguments == [{"code": payload, "timeout": 30}]

    @pytest.mark.parametrize("payload", [
        "a</arg_value>b",
        "ends with the tag</arg_value>",
        "</arg_value> starts with it",
        "two</arg_value>of</arg_value>them",
        "tag then newline</arg_value>\nmore code",
    ])
    def test_embedded_closing_tags_in_the_last_argument(self, payload):
        """The last argument's value runs to the close the call's end confirms."""
        tools = [_glm47_tool("echo", {"text": {"type": "string"}})]
        text = _glm47_call("echo", ("text", payload))

        arguments = _assert_chunking_never_matters(text, tools)

        assert arguments == [{"text": payload}]

    def test_object_value_quoting_the_tag_still_parses_as_an_object(self):
        """The confirmation rule is type-agnostic, so typed values gain it too."""
        tools = [_glm47_tool("configure", {"opts": {"type": "object"}})]
        inner = '{"code": "x</arg_value>y", "n": 1}'
        text = _glm47_call("configure", ("opts", inner))

        arguments = _assert_chunking_never_matters(text, tools)

        assert arguments == [{"opts": {"code": "x</arg_value>y", "n": 1}}]

    def test_a_well_formed_structural_sequence_inside_a_value_is_ambiguous(
            self):
        """The honest limit, pinned so it reads as a decision.

        The markup has no escaping. A value containing a full
        `</arg_value><arg_key>` sequence is byte-identical to the value
        ending and the next argument beginning, so it reads as structure and
        the value truncates there. Both paths agree, which is the most the
        grammar allows; the common case - the tag quoted mid-text, not
        followed by more well-formed markup - is the one that survives.
        """
        tools = [_glm47_tool("echo", {"text": {"type": "string"}})]
        text = _glm47_call("echo", ("text", "x</arg_value><arg_key>y"))

        arguments = _assert_chunking_never_matters(text, tools)

        assert arguments == [{"text": "x"}]

    # ----- corruption 4: a value ending where the close marker begins -----

    # The audited call, verbatim: a grep over CUTLASS headers whose pattern
    # is C++ template syntax ending in `<`. The streaming path held that
    # trailing `<` as a possible start of `</arg_value>`; when the real tag
    # arrived, the dead buffer - now `<<` - was released whole, taking the
    # tag's opening `<` with it as value content. The close could never
    # match from its first byte again, so it went unseen, the withheld value
    # was never committed, and the client received `"pattern": }` - the
    # value gone, the JSON invalid. The whole-text parse always recovered
    # it. The release now stops before a trailing `<` and re-anchors the
    # match there (see split_dead_close_buffer).
    _A04_CALL = ("<tool_call>grep"
                 "<arg_key>-n</arg_key><arg_value>true</arg_value>"
                 "<arg_key>output_mode</arg_key>"
                 "<arg_value>content</arg_value>"
                 "<arg_key>path</arg_key>"
                 "<arg_value>/context/cutlass/include/cute/atom/"
                 "mma_traits_sm100.hpp</arg_value>"
                 "<arg_key>pattern</arg_key>"
                 "<arg_value>struct MMA_Traits<SM100_MMA_F16BF16_(TS|SS)<"
                 "</arg_value></tool_call>")

    _A04_TOOLS = [
        _glm47_tool(
            "grep", {
                "-n": {
                    "type": "boolean"
                },
                "output_mode": {
                    "type": "string"
                },
                "path": {
                    "type": "string"
                },
                "pattern": {
                    "type": "string"
                },
            })
    ]

    def test_a_value_ending_in_a_prefix_of_the_close_marker_is_delivered(self):
        """Corruption 4: the traced call, byte-equal on every cut.

        The pattern must arrive exactly as the model wrote it - the earlier
        template `<` and the trailing one included - and the assembled
        arguments must be valid JSON.
        """
        agreed = _assert_streamed_text_matches_whole(self._A04_CALL,
                                                     self._A04_TOOLS)

        assert json.loads(agreed) == {
            "-n": True,
            "output_mode": "content",
            "path": "/context/cutlass/include/cute/atom/mma_traits_sm100.hpp",
            "pattern": "struct MMA_Traits<SM100_MMA_F16BF16_(TS|SS)<",
        }


def _assert_streamed_text_matches_whole(text, tools):
    """The streamed argument *text* equals `detect_and_parse`'s, every cut.

    Stronger than `_assert_chunking_never_matters`, which compares parsed
    values: json.loads collapses a repeated key (last wins), so a stream
    still carrying the key twice would pass the value comparison while
    failing every typed client. Returns the agreed text, one call assumed.
    """
    whole = Glm47ToolParser().detect_and_parse(text, tools)
    assert len(whole.calls) == 1
    expected = whole.calls[0].parameters
    for label, chunks in _chunkings(text):
        calls = _streamed_calls(chunks, tools)
        assert len(calls) == 1, f"call count changed when {label}"
        assert calls[0].parameters == expected, (
            f"streamed text diverged from detect_and_parse when {label}")
    return expected


class TestGlm47DuplicateArgumentKeys:
    """One production call wrote the same `<arg_key>` twice.

    `cell_id` arrived as "nonexistent" and again as "nonexistent-cell". The
    streamed arguments carried both key instances - typed clients reject
    duplicate keys outright - while the whole-text parse silently kept the
    last: two views of the same bytes, disagreeing.

    The policy, identical on both paths: the first occurrence wins. On the
    stream the first occurrence's bytes are already with the client when the
    repeat's key closes, and bytes cannot be un-emitted; last-wins would
    therefore have to either emit the key twice (the recorded failure) or
    withhold every argument until the call ends (no streaming at all).
    First-wins is the one policy under which the streamed JSON carries the
    key exactly once *and* equals the whole-text parse. An identical repeat
    collapses silently; a conflicting one is logged with the discarded
    value's length (`_parse_argument_pairs`, `_commit_pending_value`).
    """

    TOOLS = [
        _glm47_tool("notebook_edit", {
            "cell_id": {
                "type": "string"
            },
            "count": {
                "type": "integer"
            },
        })
    ]

    def test_a_conflicting_duplicate_keeps_the_first_occurrence(self):
        text = _glm47_call("notebook_edit", ("cell_id", "nonexistent"),
                           ("cell_id", "nonexistent-cell"))

        delivered = _assert_streamed_text_matches_whole(text, self.TOOLS)

        assert json.loads(delivered) == {"cell_id": "nonexistent"}
        assert delivered.count('"cell_id"') == 1

    def test_an_identical_duplicate_collapses_to_one(self):
        text = _glm47_call("notebook_edit", ("cell_id", "same"),
                           ("cell_id", "same"))

        delivered = _assert_streamed_text_matches_whole(text, self.TOOLS)

        assert json.loads(delivered) == {"cell_id": "same"}
        assert delivered.count('"cell_id"') == 1

    def test_a_duplicate_between_other_arguments_drops_cleanly(self):
        """The swallowed pair leaves no stray separator on the stream."""
        text = _glm47_call("notebook_edit", ("count", "1"), ("cell_id", "a"),
                           ("count", "2"), ("cell_id", "b"))

        delivered = _assert_streamed_text_matches_whole(text, self.TOOLS)

        assert json.loads(delivered) == {"count": 1, "cell_id": "a"}

    def test_three_occurrences_still_deliver_one_key(self):
        text = _glm47_call("notebook_edit", ("cell_id", "v1"),
                           ("cell_id", "v2"), ("cell_id", "v3"))

        delivered = _assert_streamed_text_matches_whole(text, self.TOOLS)

        assert json.loads(delivered) == {"cell_id": "v1"}
        assert delivered.count('"cell_id"') == 1

    def test_a_duplicate_value_quoting_the_close_tag_is_still_swallowed(self):
        """Suppression survives the PENDING_CLOSE replay machinery."""
        text = _glm47_call("notebook_edit", ("cell_id", "first"),
                           ("cell_id", "x</arg_value>y"))

        delivered = _assert_streamed_text_matches_whole(text, self.TOOLS)

        assert json.loads(delivered) == {"cell_id": "first"}

    def test_no_schema_duplicates_behave_the_same(self):
        """Live traffic's common case: the tool was never declared."""
        text = _glm47_call("invented_tool", ("cmd", "ls"), ("cmd", "rm -rf /"))

        delivered = _assert_streamed_text_matches_whole(
            text, [_glm47_tool("declared_tool")])

        assert json.loads(delivered) == {"cmd": "ls"}


class TestGlm47NonFiniteNumbers:
    """`1e309` under a number schema must not reach the wire as `Infinity`.

    json.loads (and float()) overflow the literal to float('inf') without an
    exception, and json.dumps - allow_nan defaults to True - then spells it
    `Infinity`: a token the JSON grammar does not have, so the delivered
    arguments stop parsing downstream. The model wrote a decimal string; if
    it cannot be represented faithfully as a JSON number, the original text
    is passed through as a string instead (`_is_json_finite`,
    `parse_arguments`), the same no-guessing rule the raw passthrough
    already follows.
    """

    TOOLS = [
        _glm47_tool("calc", {
            "x": {
                "type": "number"
            },
            "n": {
                "type": "integer"
            },
        })
    ]

    @staticmethod
    def _strictly(delivered: str):
        """json.loads with the non-JSON number literals rejected.

        Python's default loads *accepts* Infinity/NaN, so a plain loads
        cannot see this defect - which is how it shipped.
        """

        def _reject(token):
            raise AssertionError(f"non-JSON literal {token!r} reached the wire")

        return json.loads(delivered, parse_constant=_reject)

    @pytest.mark.parametrize("value", ["1e309", "-1e309"])
    def test_an_overflowing_number_is_the_text_the_model_wrote(self, value):
        text = _glm47_call("calc", ("x", value))

        delivered = _assert_streamed_text_matches_whole(text, self.TOOLS)

        assert self._strictly(delivered) == {"x": value}

    def test_the_largest_finite_double_is_still_a_number(self):
        text = _glm47_call("calc", ("x", "1e308"))

        delivered = _assert_streamed_text_matches_whole(text, self.TOOLS)

        parsed = self._strictly(delivered)
        assert parsed == {"x": 1e308}
        assert isinstance(parsed["x"], float)

    def test_an_integer_schema_hits_the_same_guard(self):
        text = _glm47_call("calc", ("n", "1e309"))

        delivered = _assert_streamed_text_matches_whole(text, self.TOOLS)

        assert self._strictly(delivered) == {"n": "1e309"}

    @pytest.mark.parametrize("value", ["NaN", "Infinity", "-Infinity"])
    def test_the_literal_spellings_stay_text_too(self, value):
        """json.loads accepts these directly; the guard still refuses them."""
        text = _glm47_call("calc", ("x", value))

        delivered = _assert_streamed_text_matches_whole(text, self.TOOLS)

        assert self._strictly(delivered) == {"x": value}

    def test_a_non_finite_nested_in_an_object_value_falls_back_to_text(self):
        """The check is recursive: one nested inf poisons the whole dumps."""
        tools = [_glm47_tool("configure", {"opts": {"type": "object"}})]
        text = _glm47_call("configure", ("opts", '{"a": 1e309}'))

        delivered = _assert_streamed_text_matches_whole(text, tools)

        assert self._strictly(delivered) == {"opts": '{"a": 1e309}'}

    def test_a_finite_object_still_parses_as_an_object(self):
        tools = [_glm47_tool("configure", {"opts": {"type": "object"}})]
        text = _glm47_call("configure", ("opts", '{"a": 1e308}'))

        delivered = _assert_streamed_text_matches_whole(text, tools)

        assert self._strictly(delivered) == {"opts": {"a": 1e308}}


class TestGlm47StreamedArgumentDelivery:
    """The fragment stream the join depends on, and the cost of the fix."""

    @staticmethod
    def _argument_fragments(text, tools):
        """Every non-empty `parameters` delta, in order, one character in."""
        parser = Glm47ToolParser()
        return [
            call.parameters for chunk in text
            for call in parser.parse_streaming_increment(chunk, tools).calls
            if call.parameters
        ]

    def test_arguments_still_arrive_as_fragments_that_join_into_json(self):
        """Unchanged contract: the deltas are pieces, the join is the JSON."""
        tools = [_glm47_tool("exec_command", {"cmd": {"type": "string"}})]
        text = _glm47_call("exec_command", ("cmd", "ls"),
                           ("max_output_tokens", "8000"))

        fragments = self._argument_fragments(text, tools)

        assert len(fragments) > 1, "arguments were not delivered in pieces"
        assert fragments[0].startswith("{")
        assert json.loads("".join(fragments)) == {
            "cmd": "ls",
            "max_output_tokens": "8000",
        }

    def test_a_value_is_delivered_once_complete_rather_than_as_it_arrives(self):
        """The accepted cost, pinned so it reads as a decision not an accident.

        A `</arg_value>` only counts once the text after it confirms it as
        structure - a payload may be quoting the tag - so the value is held
        until that confirmation and then sent in one piece. A long `cmd`
        therefore no longer trickles out character by character. Nothing a
        client sees arrives later for it: whole calls were already only
        assembled when generation finished.
        """
        tools = [_glm47_tool("exec_command", {"cmd": {"type": "string"}})]
        text = _glm47_call("exec_command", ("cmd", "abcdefghij"))

        fragments = self._argument_fragments(text, tools)

        assert fragments == ['{"cmd": ', '"abcdefghij"', "}"]


# Values chosen to reach every branch of `parse_arguments` (via the typed
# schemas) and of the raw-passthrough rule (via string / no schema). The
# final group *ends* in a bare prefix of `</arg_value>` - the shape that
# used to defeat the streaming path by releasing the dead tag buffer whole
# and consuming the real close's opening `<` with it, so the streamed value
# vanished while the whole parse recovered it (corruption 4 in
# TestGlm47ArgumentCorruptions; fixed by split_dead_close_buffer).
_PROPERTY_VALUES = [
    "ls -la /workspace",
    "8000",
    "-42",
    "72.5",
    "1e10",
    "0",
    "007",
    '{"a": 1}',
    "[1, 2]",
    "true",
    "null",
    "",
    " ",
    '"quoted"',
    "8000 tokens",
    "1.2.3",
    "北京",
    "line one\nline two",
    "C:\\Users\\test.txt",
    'say "hi"',
    "</arg_valueX>",
    "a</arg_value>b",
    "<b>bold</b>",
    # Overflow json.loads to inf; every declared type must agree on the
    # raw-text fallback rather than serialize `Infinity`.
    "1e309",
    "-1e309",
    "1e308",
    # Trailing bare prefixes of the close marker (corruption 4), plus `<`
    # runs that must stream through without stalling in the tag buffer.
    "<",
    "a<",
    "a</",
    "a</arg_valu",
    "a</arg_value",
    "<<<",
    "a<<b<<<c<",
    "struct MMA_Traits<SM100_MMA_F16BF16_(TS|SS)<",
    # A quoted close *and* a trailing prefix: the PENDING_CLOSE lookahead
    # rules the tag content, replays it, and the replayed `<` must re-anchor
    # in the value buffer rather than be released as content.
    "a</arg_value><",
]

# Every declared type a JSON schema can give the argument, plus no schema at
# all - the case live traffic actually hits.
_PROPERTY_SCHEMAS = [
    None, "string", "number", "integer", "object", "array", "boolean"
]


@pytest.mark.parametrize("schema_type", _PROPERTY_SCHEMAS)
@pytest.mark.parametrize("value", _PROPERTY_VALUES)
def test_glm47_streamed_argument_equals_the_whole_parse(value, schema_type):
    """The property itself, over a corpus of values and every declared type.

    No expectation is written down: whatever `detect_and_parse` makes of the
    value is what streaming has to make of it too, however the text is cut up.
    """
    properties = {} if schema_type is None else {"v": {"type": schema_type}}
    tools = [_glm47_tool("f", properties)]
    text = _glm47_call("f", ("v", value))

    _assert_chunking_never_matters(text, tools)


# ============================================================================
# A call whose arguments are not valid JSON is never reported
# ============================================================================
#
# GLM-4.7 sometimes opens `<arg_value>`, writes the value, and then ends the
# block with `</think></tool_call>` without ever closing the value. Both
# `<tool_call>` tags are present and balanced, so nothing about the *markup*
# says anything is wrong: the call finalises as usual and its fragments
# assemble to `{"cmd": }`. A client stored that, replayed it in the next
# request's history, and was answered `tool_calls[0].function.arguments must
# be valid JSON`; the agent then retried to its backoff limit and the run was
# lost. 2 of 13,014 delivered calls - rare, and a whole run each time.
#
# So the rule keys on the result rather than on the markup: a call whose
# assembled arguments do not parse is dropped, and a warning names the tool.
# Repairing was rejected. Closing the quote and the brace would invent an
# argument the model never wrote, and a truncated shell command would then run
# with nothing to say that the rest of it was lost.
#
# The check lives in `_assembled_tool_calls` because that is the only place
# that both knows the whole and can still decline to report it: the parser
# streams `{`, `"cmd": ` and the value in separate increments, each already
# handed to the client's accumulator, and only learns at `</tool_call>` that
# they do not add up. These tests therefore drive the same join the serving
# layer performs rather than asserting on one delta.

# The recorded failure, `disagg_request_id=470103303783089`, with the command
# trimmed to two lines. What matters is the shape: `<arg_value>` opens, never
# closes, and `</think></tool_call>` ends the block.
_UNCLOSED_ARG_VALUE = ("<tool_call>exec_command"
                       "<arg_key>cmd</arg_key>"
                       "<arg_value>"
                       'kill -0 695 2>&1 || echo "seg1: 695 done"\n'
                       "wc -c /workspace/submit_output.log"
                       "</think></tool_call>")

# The prose the same response emitted just before that call.
_PROSE_BEFORE_THE_CALL = (
    "Let me try polling the submit output through a different method:")


def _streamed_text_and_calls(chunks, tools, parser_factory=Glm47ToolParser):
    """`_streamed_calls`, keeping the assistant text the parser released too.

    Needed to show that dropping a call leaves the message around it alone.
    """
    parser = parser_factory()
    fragments, released = {}, []
    for chunk in chunks:
        result = parser.parse_streaming_increment(chunk, tools)
        released.append(result.normal_text)
        _accumulate_tool_call_fragments(fragments, result.calls)
    flushed_text, flushed, unfinished = _flush_tool_parser(
        tools=tools, output_index=0, tool_parser_dict={0: parser})
    released.append(flushed_text)
    _accumulate_tool_call_fragments(fragments, flushed)
    return "".join(released), _assembled_tool_calls(
        fragments, unfinished, _abandoned_tool_indices(parser))


@pytest.fixture
def responses_warnings(monkeypatch):
    """Every warning `responses_utils` logs while the test runs.

    TRT-LLM logs through its own logger object rather than a stdlib one, so
    `caplog` never sees these; the module-level name is swapped for a recorder
    instead.
    """
    recorded = []

    class _Recorder:

        def warning(self, message):
            recorded.append(str(message))

        def __getattr__(self, name):
            return lambda *args, **kwargs: None

    monkeypatch.setattr(responses_utils, "logger", _Recorder())
    return recorded


class TestGlm47MalformedCallIsNotReported:
    """The delivery decision, one test per case the rule has to get right."""

    def test_the_recorded_failure_reports_no_call(self, responses_warnings):
        """Case 1: the live defect, as it was recorded."""
        assert _streamed_calls([_UNCLOSED_ARG_VALUE], NO_SCHEMA_TOOLS) == []

        assert len(responses_warnings) == 1
        assert "exec_command" in responses_warnings[0]
        assert "valid JSON" in responses_warnings[0]

    def test_what_was_dropped_was_the_unusable_call(self):
        """The drop is doing work, rather than the parser having gone quiet.

        Without this the test above passes just as well if the parser stops
        reporting the call for some unrelated reason, which would hide a much
        larger regression. So assert on the fragments themselves: they arrive,
        they name the tool, and their concatenation is the `{"cmd": }` that
        the client rejected.
        """
        parser = Glm47ToolParser()
        fragments = {}
        result = parser.parse_streaming_increment(_UNCLOSED_ARG_VALUE,
                                                  NO_SCHEMA_TOOLS)
        _accumulate_tool_call_fragments(fragments, result.calls)

        assert fragments[0]["name"] == "exec_command"
        assert fragments[0]["parameters"] == '{"cmd": }'
        with pytest.raises(json.JSONDecodeError):
            json.loads(fragments[0]["parameters"])

    def test_no_chunking_makes_the_malformed_call_reportable(self):
        """The verdict cannot depend on where the deltas happen to fall."""
        for label, chunks in _chunkings(_UNCLOSED_ARG_VALUE):
            assert _streamed_calls(chunks, NO_SCHEMA_TOOLS) == [], (
                f"the malformed call was reported when {label}")

    def test_a_well_formed_call_is_untouched(self):
        """Case 2: 99.98% of traffic. A regression here outweighs the bug."""
        text = _glm47_call("exec_command", ("cmd", "ls -la /workspace"),
                           ("max_output_tokens", "8000"))

        arguments = _assert_chunking_never_matters(text, NO_SCHEMA_TOOLS)

        assert arguments == [{
            "cmd": "ls -la /workspace",
            "max_output_tokens": "8000",
        }]

    def test_only_the_malformed_call_of_three_is_dropped(self):
        """Case 3: the good calls of a response are not collateral damage."""
        tools = [_glm47_tool("exec_command"), _glm47_tool("wait")]
        text = (_glm47_call("exec_command", ("cmd", "pytest -q")) +
                _UNCLOSED_ARG_VALUE + _glm47_call("wait", ("seconds", "2")))

        for label, chunks in _chunkings(text):
            calls = _streamed_calls(chunks, tools)
            assert [(c.name, json.loads(c.parameters)) for c in calls] == [
                ("exec_command", {
                    "cmd": "pytest -q"
                }),
                ("wait", {
                    "seconds": "2"
                }),
            ], f"the surviving calls changed when {label}"

    def test_the_prose_around_a_dropped_call_is_unaffected(self):
        """Case 4: the only call is dropped; the message still arrives."""
        text = _PROSE_BEFORE_THE_CALL + _UNCLOSED_ARG_VALUE

        released, calls = _streamed_text_and_calls([text], NO_SCHEMA_TOOLS)

        assert calls == []
        assert released == _PROSE_BEFORE_THE_CALL

    def test_zero_argument_call_is_still_reported(self):
        """Case 5: `{}` is valid JSON, so there is nothing to drop."""
        tools = [_glm47_tool("get_time")]

        calls = _streamed_calls(["<tool_call>get_time</tool_call>"], tools)

        assert [(c.name, c.parameters) for c in calls] == [("get_time", "{}")]

    @pytest.mark.parametrize("value", [
        'echo "}" >> a.json',
        '{"k": "v"}',
        'printf \'{"a": [1, 2]}\' | jq .',
        'sed -i \'s/"x"/"y"/\' f.json && echo "}"',
    ])
    def test_a_value_full_of_braces_and_quotes_is_still_reported(self, value):
        """Case 6: "contains a brace" is not "malformed"."""
        tools = [_glm47_tool("echo", {"text": {"type": "string"}})]
        text = _glm47_call("echo", ("text", value))

        arguments = _assert_chunking_never_matters(text, tools)

        assert arguments == [{"text": value}]

    def test_stream_cut_off_mid_call_is_dropped_once(self, responses_warnings):
        """Case 7: the older rule still owns this one, and says so once.

        `_flush_tool_parser` already drops a call the stream ended inside and
        warns about it. Letting the validity check warn again would report one
        lost call as two.
        """
        chunks = [
            "<tool_call>exec_command<arg_key>cmd</arg_key>",
            "<arg_value>ls -la /works",
        ]

        assert _streamed_calls(chunks, NO_SCHEMA_TOOLS) == []

        assert len(responses_warnings) == 1
        assert "Stream ended inside a tool call" in responses_warnings[0]

    def test_the_non_streaming_path_is_unchanged(self):
        """Case 8, measured rather than assumed - and it is not "no call".

        `func_detail_regex` matches the block whole and `func_arg_regex` then
        finds no `<arg_key>k</arg_key><arg_value>v</arg_value>` pair in it, so
        `detect_and_parse` reports the call with no arguments at all. `{}` is
        valid JSON, so the rule above does not fire and this path keeps the
        behaviour it has always had.

        The two paths therefore still disagree about this text: streaming now
        reports nothing, whole-parsing reports a call with empty arguments.
        Closing that gap means deciding what a call with markup it could not
        read should be, which is a different question from the one this change
        answers - the client can run `{}`, it could not run `{"cmd": }`.
        """
        result = Glm47ToolParser().detect_and_parse(_UNCLOSED_ARG_VALUE,
                                                    NO_SCHEMA_TOOLS)

        assert [(c.name, c.parameters)
                for c in result.calls] == [("exec_command", "{}")]

    def test_a_call_whose_name_never_arrived_is_still_dropped(self):
        """Case 9: unchanged, and not double-counted as a JSON failure."""
        fragments = {}
        _accumulate_tool_call_fragments(
            fragments,
            [ToolCallItem(tool_index=0, name=None, parameters='{"a": 1}')])

        assert _assembled_tool_calls(fragments) == []


# ============================================================================
# A tool name never contains '<'
# ============================================================================
#
# Two calls recorded on a GLM-5.3 fleet (~2 of 1042) reached their client
# with markup fused into the function name, and the client, treating the
# whole blob as an unknown tool, lost the turn:
#
#   exec<tool_call>exec               - the model restarted its call midway,
#                                       doubling the opener;
#   exec<arg_value>input</arg_key>... - tags written out of order, with
#                                       <arg_value> where <arg_key> belongs.
#
# The name regexes' `(.*?)` group terminated only at `<arg_key>` or
# `</tool_call>`, so the junk was swallowed INTO the name. The name group
# now stops at the first `<` and classify_name_region decides the call's
# fate from what follows: a doubled opener re-anchors on the fresh inner
# call, with the abandoned prefix released as message text; junk the
# resolver can strip back to a declared tool is repaired exactly as before
# (TestGlm47MangledToolNames pins that path); anything else marks the call
# malformed, and its whole text is released as the assistant's message -
# the text-over-silence stance `_flush_tool_parser` pins for unterminated
# markup - identically on the streaming and whole-text paths.

# Production shape 1, verbatim: the doubled opener. Byte accounting:
# `<tool_call>exec` (the abandoned restart) becomes message text; the rest
# is the delivered call, name `exec`, arguments {"k": "v"}.
_DOUBLED_OPENER = ("<tool_call>exec<tool_call>exec"
                   "<arg_key>k</arg_key><arg_value>v</arg_value></tool_call>")
_DOUBLED_OPENER_PREFIX = "<tool_call>exec"

# Production shape 2, with the recorded payload trimmed the way
# _UNCLOSED_ARG_VALUE trims its command: `<arg_value>` written where
# `<arg_key>` belongs. No call can be read out of it; every byte becomes
# message text.
_OUT_OF_ORDER_TAGS = ("<tool_call>exec<arg_value>input</arg_key>"
                      '<arg_value>const script = require("fs")</arg_value>'
                      "</tool_call>")


class TestGlm47ToolNameNeverContainsMarkup:
    """The two recorded name corruptions, pinned byte for byte."""

    def test_doubled_opener_recovers_the_restarted_call(self):
        """Whole-text: the inner, well-formed call is the one delivered."""
        result = Glm47ToolParser().detect_and_parse(_DOUBLED_OPENER,
                                                    NO_SCHEMA_TOOLS)

        assert [(c.name, json.loads(c.parameters))
                for c in result.calls] == [("exec", {
                    "k": "v"
                })]
        assert result.normal_text == _DOUBLED_OPENER_PREFIX

    def test_doubled_opener_streams_identically_on_every_cut(self):
        for label, chunks in _chunkings(_DOUBLED_OPENER):
            released, calls = _streamed_text_and_calls(chunks, NO_SCHEMA_TOOLS)
            assert [(c.name, json.loads(c.parameters))
                    for c in calls] == [("exec", {
                        "k": "v"
                    })], f"the recovered call changed when {label}"
            assert released == _DOUBLED_OPENER_PREFIX, (
                f"the abandoned prefix was not released as text when {label}")

    def test_out_of_order_tags_deliver_no_call_and_keep_the_text(self):
        result = Glm47ToolParser().detect_and_parse(_OUT_OF_ORDER_TAGS,
                                                    NO_SCHEMA_TOOLS)

        assert result.calls == []
        assert result.normal_text == _OUT_OF_ORDER_TAGS

    def test_out_of_order_tags_stream_identically_on_every_cut(self):
        for label, chunks in _chunkings(_OUT_OF_ORDER_TAGS):
            released, calls = _streamed_text_and_calls(chunks, NO_SCHEMA_TOOLS)
            assert calls == [], f"a call was invented when {label}"
            assert released == _OUT_OF_ORDER_TAGS, (
                f"characters were lost when {label}")

    @pytest.mark.parametrize("text", [_DOUBLED_OPENER, _OUT_OF_ORDER_TAGS])
    def test_no_delivered_name_contains_markup(self, text):
        """The invariant itself, swept over every cut of both shapes."""
        for label, chunks in _chunkings(text):
            for call in _streamed_calls(chunks, NO_SCHEMA_TOOLS):
                assert "<" not in call.name, (
                    f"name {call.name!r} carries markup when {label}")

    def test_surrounding_calls_survive_the_malformed_one(self):
        """The junk blob costs its own call and nothing else, on both views."""
        tools = [_glm47_tool("exec"), _glm47_tool("wait")]
        text = (_glm47_call("exec", ("cmd", "pytest -q")) + _OUT_OF_ORDER_TAGS +
                _glm47_call("wait", ("seconds", "2")))
        expected = [
            ("exec", {
                "cmd": "pytest -q"
            }),
            ("wait", {
                "seconds": "2"
            }),
        ]

        whole = Glm47ToolParser().detect_and_parse(text, tools)
        assert [(c.name, json.loads(c.parameters))
                for c in whole.calls] == expected
        assert whole.normal_text == _OUT_OF_ORDER_TAGS

        for label, chunks in _chunkings(text):
            released, calls = _streamed_text_and_calls(chunks, tools)
            assert [(c.name, json.loads(c.parameters)) for c in calls
                    ] == expected, (f"the surviving calls changed when {label}")
            assert released == _OUT_OF_ORDER_TAGS, (
                f"the malformed blob's bytes moved when {label}")

    def test_stream_cut_inside_the_malformed_blob_releases_it_at_flush(
            self, responses_warnings):
        """Truncated junk follows the older unterminated-markup rule."""
        cut = _OUT_OF_ORDER_TAGS[:-9]  # ends inside `</tool_call>`

        released, calls = _streamed_text_and_calls([cut], NO_SCHEMA_TOOLS)

        assert calls == []
        assert released == cut
        assert len(responses_warnings) == 1
        assert "Stream ended inside a tool call" in responses_warnings[0]


# ============================================================================
# A fresh `<tool_call>` inside an unclosed call's arguments
# ============================================================================
#
# GLM-5.3 also restarts a call after its arguments have begun: it opens a new
# `<tool_call>` before closing the one it is in, so the fresh opener lands in
# the open call's argument region instead of its name region. Read as argument
# text, the call the model went on to finish arrived nested inside the value of
# the call it had abandoned, and the client ran the abandoned call with that
# markup as its input - recorded `exec` inputs ended
# `...)localObject<tool_call>exec<arg_key>input</arg_key><arg_value>...`. A
# fresh opener there is now a restart exactly like one in the name region: the
# abandoned call's markup becomes message text and the call that follows is
# the one delivered, identically on the streaming and whole-text paths.

# Production shape, with the recorded payload trimmed. Byte accounting: the
# abandoned call (everything before the second opener) becomes message text;
# the finished call is delivered as `exec` with {"input": "ls -la"}.
_ABANDONED_CALL = ('<tool_call>exec<arg_key>input</arg_key>'
                   '<arg_value>notify("x")</arg_value>)localObject')
_FINISHED_CALL = _glm47_call("exec", ("input", "ls -la"))
_NESTED_CALL = _ABANDONED_CALL + _FINISHED_CALL
_FINISHED_ONLY = [("exec", {"input": "ls -la"})]


def _named_arguments(calls):
    return [(call.name, json.loads(call.parameters)) for call in calls]


class TestGlm47FreshCallInsideArguments:
    """An opener inside an open call's arguments abandons that call."""

    def test_whole_text_delivers_the_call_the_model_finished(self):
        result = Glm47ToolParser().detect_and_parse(_NESTED_CALL,
                                                    NO_SCHEMA_TOOLS)

        assert _named_arguments(result.calls) == _FINISHED_ONLY
        assert result.normal_text == _ABANDONED_CALL

    def test_every_cut_streams_the_same_split(self):
        for label, chunks in _chunkings(_NESTED_CALL):
            released, calls = _streamed_text_and_calls(chunks, NO_SCHEMA_TOOLS)
            assert _named_arguments(calls) == _FINISHED_ONLY, (
                f"the delivered call changed when {label}")
            assert released == _ABANDONED_CALL, (
                f"the abandoned call was not released as text when {label}")

    def test_no_delivered_argument_carries_call_markup(self):
        """The defect itself, swept over every cut: nothing nested survives."""
        for label, chunks in _chunkings(_NESTED_CALL):
            for call in _streamed_calls(chunks, NO_SCHEMA_TOOLS):
                assert "<tool_call>" not in call.parameters, (
                    f"arguments {call.parameters!r} nest a call when {label}")

    def test_an_announced_call_is_abandoned_without_a_second_warning(
            self, responses_warnings):
        """Token by token the abandoned call's name is already on the stream.

        Its index is recorded as abandoned and the finished call takes the
        next one. Its fragments are a JSON prefix that assembly would
        otherwise drop as a malformed call - with a warning that reads as a
        second loss, and a dropped-call slot the whole-text rebuild has no
        call for.
        """
        parser = Glm47ToolParser()
        fragments, released = {}, []
        for chunk in re.findall(r"<[^>]*>|[^<]+", _NESTED_CALL):
            result = parser.parse_streaming_increment(chunk, NO_SCHEMA_TOOLS)
            released.append(result.normal_text)
            _accumulate_tool_call_fragments(fragments, result.calls)

        assert parser.abandoned_tool_indices == {0}
        assert fragments[0]["name"] == "exec"
        assert "".join(released) == _ABANDONED_CALL

        calls = _assembled_tool_calls(fragments, None,
                                      _abandoned_tool_indices(parser))
        assert [(call.tool_index, call.name, json.loads(call.parameters))
                for call in calls] == [(1, "exec", {
                    "input": "ls -la"
                })]
        assert responses_warnings == []

    def test_surrounding_calls_and_prose_keep_their_places(self):
        tools = [_glm47_tool("exec"), _glm47_tool("wait")]
        before = _glm47_call("wait", ("seconds", "2"))
        text = "Checking. " + before + _NESTED_CALL + " Done."
        expected = [("wait", {"seconds": "2"})] + _FINISHED_ONLY

        whole = Glm47ToolParser().detect_and_parse(text, tools)
        assert _named_arguments(whole.calls) == expected
        assert whole.normal_text == "Checking. " + _ABANDONED_CALL + " Done."

        for label, chunks in _chunkings(text):
            released, calls = _streamed_text_and_calls(chunks, tools)
            assert _named_arguments(calls) == expected, (
                f"the delivered calls changed when {label}")
            assert released == "Checking. " + _ABANDONED_CALL + " Done.", (
                f"the text moved when {label}")

    def test_an_abandoned_malformed_call_still_yields_the_finished_one(self):
        """A malformed name region no longer takes the next call with it.

        Released whole, the segment - which runs to the finished call's
        `</tool_call>` - swallowed the call the model went on to make.
        """
        abandoned = ("<tool_call>exec<arg_value>input</arg_key>"
                     "<arg_key>input</arg_key><arg_value>x")
        text = abandoned + _FINISHED_CALL

        whole = Glm47ToolParser().detect_and_parse(text, NO_SCHEMA_TOOLS)
        assert _named_arguments(whole.calls) == _FINISHED_ONLY
        assert whole.normal_text == abandoned

        for label, chunks in _chunkings(text):
            released, calls = _streamed_text_and_calls(chunks, NO_SCHEMA_TOOLS)
            assert _named_arguments(calls) == _FINISHED_ONLY, (
                f"the finished call was lost when {label}")
            assert released == abandoned, f"the text moved when {label}"

    def test_a_stream_cut_inside_the_fresh_call_keeps_every_byte(self):
        """Neither call finished: no call on either view, all text kept."""
        text = _NESTED_CALL[:-len("ls -la</arg_value></tool_call>")]

        whole = Glm47ToolParser().detect_and_parse(text, NO_SCHEMA_TOOLS)
        assert whole.calls == []
        assert whole.normal_text == text

        for label, chunks in _chunkings(text):
            released, calls = _streamed_text_and_calls(chunks, NO_SCHEMA_TOOLS)
            assert calls == [], f"a call was invented when {label}"
            assert released == text, f"characters were lost when {label}"


# glm4 puts the name on its own line, so the same corruption needs the junk
# on that line; the two production shapes in glm4 markup.
_G4_DOUBLED_OPENER = ("<tool_call>exec<tool_call>exec\n"
                      "<arg_key>k</arg_key>\n"
                      "<arg_value>v</arg_value>\n</tool_call>")
_G4_OUT_OF_ORDER = ("<tool_call>exec<arg_value>input</arg_key>\n"
                    "<arg_key>k</arg_key>\n"
                    "<arg_value>v</arg_value>\n</tool_call>")


class TestGlm4ToolNameNeverContainsMarkup:
    """The same rule on the glm4 parser, whose name group shared the shape."""

    def test_doubled_opener_recovers_the_restarted_call(self):
        result = Glm4ToolParser().detect_and_parse(_G4_DOUBLED_OPENER,
                                                   NO_SCHEMA_TOOLS)

        assert [(c.name, json.loads(c.parameters))
                for c in result.calls] == [("exec", {
                    "k": "v"
                })]
        assert result.normal_text == _DOUBLED_OPENER_PREFIX

    def test_out_of_order_tags_deliver_no_call_and_keep_the_text(self):
        result = Glm4ToolParser().detect_and_parse(_G4_OUT_OF_ORDER,
                                                   NO_SCHEMA_TOOLS)

        assert result.calls == []
        assert result.normal_text == _G4_OUT_OF_ORDER

    def test_doubled_opener_streams_identically_on_every_cut(self):
        for label, chunks in _chunkings(_G4_DOUBLED_OPENER):
            released, calls = _streamed_text_and_calls(
                chunks, NO_SCHEMA_TOOLS, parser_factory=Glm4ToolParser)
            assert [(c.name, json.loads(c.parameters))
                    for c in calls] == [("exec", {
                        "k": "v"
                    })], f"the recovered call changed when {label}"
            assert released == _DOUBLED_OPENER_PREFIX, (
                f"the abandoned prefix was not released as text when {label}")

    def test_out_of_order_tags_stream_identically_on_every_cut(self):
        for label, chunks in _chunkings(_G4_OUT_OF_ORDER):
            released, calls = _streamed_text_and_calls(
                chunks, NO_SCHEMA_TOOLS, parser_factory=Glm4ToolParser)
            assert calls == [], f"a call was invented when {label}"
            assert released == _G4_OUT_OF_ORDER, (
                f"characters were lost when {label}")

    def test_a_slice_the_format_cannot_read_is_kept_as_text(self):
        """A slice the format cannot read lands in the text, not nowhere.

        glm4 requires a newline after the name; a sliced call without one
        used to vanish from the response entirely (`func_detail is None` ->
        `continue`). Unreadable markup is still model output.
        """
        text = ("<tool_call>exec<arg_key>k</arg_key>"
                "<arg_value>v</arg_value></tool_call>")

        result = Glm4ToolParser().detect_and_parse(text, NO_SCHEMA_TOOLS)

        assert result.calls == []
        assert result.normal_text == text


# ============================================================================
# GLM streaming prose preservation: `</tool_call>` in ordinary text
# ============================================================================
#
# A `</tool_call>` the model writes in plain prose - no `<tool_call>` opener
# anywhere - is model output, and `detect_and_parse` (which rebuilds the
# final view of a generation) keeps it byte for byte. The streaming
# no-opener branch used to strip it: 612 recorded responses in one
# production week streamed without the tag their final snapshot carried
# (one locked example: 448 streamed chars vs 460 final, differing by
# exactly one `</tool_call>`). Nothing the parser owns can reach that
# branch - a parsed call's close is consumed when finalization re-anchors
# the buffer past it - so the strip only ever deleted prose.

_GLM_PARSERS = [Glm4ToolParser, Glm47ToolParser]

# The same call in each parser's native markup, for the trailing-prose case.
_GLM_CALL_TEXTS = {
    Glm4ToolParser: ("<tool_call>get_weather\n"
                     "<arg_key>location</arg_key>\n"
                     "<arg_value>NYC</arg_value>\n"
                     "</tool_call>"),
    Glm47ToolParser: ("<tool_call>get_weather"
                      "<arg_key>location</arg_key>"
                      "<arg_value>NYC</arg_value>"
                      "</tool_call>"),
}


@pytest.mark.parametrize("parser_cls", _GLM_PARSERS)
class TestGlmStreamingProsePreservation:
    """The streamed text and the final snapshot are one document."""

    PROSE = "To end a call the model writes </tool_call> and then stops."

    def test_prose_close_tag_survives_every_chunking(self, parser_cls,
                                                     sample_tools):
        """Plain text with a lone `</tool_call>` streams byte-identical.

        Swept over every chunking, including every single split point - so
        the tag arrives torn at each of its internal boundaries, among them
        the split right after its leading `<`, the one byte the branch
        legitimately holds back as a potential `<tool_call>` start.
        """
        text = self.PROSE
        whole = parser_cls().detect_and_parse(text, sample_tools)
        assert whole.calls == []
        assert whole.normal_text == text

        for label, chunks in _chunkings(text):
            streamed, calls = _streamed_text_and_calls(
                chunks, sample_tools, parser_factory=parser_cls)
            assert calls == [], f"a call was invented when {label}"
            assert streamed == text, f"the stream lost bytes when {label}"

    def test_trailing_prose_after_a_call_keeps_its_close_tag(
            self, parser_cls, sample_tools):
        """A real call, then prose quoting `</tool_call>`: both delivered.

        The dangling close tag after a completed call is exactly what the
        stripped branch was presumably defending against; the whole-text
        parse keeps it as visible text, so the stream must too.
        """
        trailing = " A lone </tool_call> in prose must survive the call."
        text = _GLM_CALL_TEXTS[parser_cls] + trailing

        for label, chunks in _chunkings(text):
            streamed, calls = _streamed_text_and_calls(
                chunks, sample_tools, parser_factory=parser_cls)
            assert [c.name for c in calls
                    ] == ["get_weather"], (f"the call was lost when {label}")
            assert json.loads(calls[0].parameters) == {"location": "NYC"}
            assert streamed == trailing, (
                f"the trailing prose was corrupted when {label}")

    def test_partial_close_tail_is_released_not_held_forever(
            self, parser_cls, sample_tools):
        """A stream ending in `</tool_` still delivers every byte.

        `</tool_` shares no prefix with the bot token beyond `<` itself, so
        the branch releases it the moment it arrives rather than holding it
        for a completion that never comes; the end-of-stream flush then has
        nothing left to add.
        """
        chunks = ["The answer is 42. ", "See </tool_"]

        streamed, calls = _streamed_text_and_calls(chunks,
                                                   sample_tools,
                                                   parser_factory=parser_cls)

        assert calls == []
        assert streamed == "The answer is 42. See </tool_"

    def test_potential_start_tail_flushes_verbatim_at_end_of_stream(
            self, parser_cls, sample_tools):
        """`finish` releases text held back as a possible `<tool_call>`.

        Pinned against `finish` itself because the chat completions path
        calls only `parse_streaming_increment` + `finish` at end of stream
        (`apply_tool_parser`), with no external buffer drain: text withheld
        by the no-opener branch may be delayed, never dropped.
        """
        parser = parser_cls()
        held = "Compare a <tool"

        streamed = parser.parse_streaming_increment(held,
                                                    sample_tools).normal_text
        assert streamed == ""  # withheld: could still become the bot token

        flushed = parser.finish(sample_tools)
        assert flushed.normal_text == held
        assert flushed.calls == []
        assert parser._buffer == ""

    def test_finish_leaves_markup_buffers_to_the_serving_layer(
            self, parser_cls, sample_tools):
        """A stream cut off inside a call is not `finish`'s to release.

        `_flush_tool_parser` owns that disposition (release with a warning
        and the unfinished call's index); `finish` releasing it as well
        would deliver the markup twice on the responses path.
        """
        parser = parser_cls()
        parser.parse_streaming_increment("<tool_call>get_w", sample_tools)

        flushed = parser.finish(sample_tools)

        assert flushed.normal_text == ""
        assert flushed.calls == []
        assert parser._buffer == "<tool_call>get_w"

    @pytest.mark.parametrize("text", [
        "</tool_call>",
        "</tool_call> opens the reply.",
        "The reply ends on </tool_call>",
        "</tool_call></tool_call>",
        PROSE,
    ])
    def test_a_close_tag_survives_being_torn_into_several_pieces(
            self, parser_cls, sample_tools, text):
        """The KF witness shape - a message that is exactly the tag - and kin.

        The witness streamed no message item at all for a final output of
        exactly `</tool_call>`. With the tag at an edge of the message there
        is no prose around it to carry it, and a single split point does not
        show what three pieces of one tag do, so every multi-cut chunking is
        swept too: exhaustively for the bare tag.
        """
        whole = parser_cls().detect_and_parse(text, sample_tools)
        assert whole.calls == []
        assert whole.normal_text == text

        for label, chunks in itertools.chain(_chunkings(text),
                                             _multi_cut_chunkings(text)):
            streamed, calls = _streamed_text_and_calls(
                chunks, sample_tools, parser_factory=parser_cls)
            assert calls == [], f"a call was invented when {label}"
            assert streamed == text, f"the stream lost bytes when {label}"

    def test_a_close_tag_right_after_a_call_is_text(self, parser_cls,
                                                    sample_tools):
        """The tag lands exactly where the parser re-anchors past a call.

        Finalization leaves the buffer starting at the byte after the call's
        own `</tool_call>`, so a second one there is the first thing the
        no-opener branch sees - torn into pieces, on every cut around it.
        """
        text = _GLM_CALL_TEXTS[parser_cls] + "</tool_call>"

        for label, chunks in itertools.chain(_chunkings(text),
                                             _multi_cut_chunkings(text)):
            streamed, calls = _streamed_text_and_calls(
                chunks, sample_tools, parser_factory=parser_cls)
            assert [c.name for c in calls
                    ] == ["get_weather"], f"the call was lost when {label}"
            assert streamed == "</tool_call>", (
                f"the trailing tag was lost when {label}")


def test_glm47_streamed_prose_close_tag_matches_the_whole_parse(sample_tools):
    """The two views of the locked production example's shape agree.

    detect_and_parse strips the normal text around calls, so the comparison
    is modulo that documented strip; the `</tool_call>` itself must appear
    in both. (glm4's detect_and_parse drops text after the last call - a
    separate, pre-existing asymmetry - so this check is glm47's.)
    """
    trailing = " A lone </tool_call> in prose must survive the call."
    text = _GLM_CALL_TEXTS[Glm47ToolParser] + trailing

    whole = Glm47ToolParser().detect_and_parse(text, sample_tools)
    streamed, calls = _streamed_text_and_calls([text], sample_tools)

    assert whole.normal_text == trailing.strip()
    assert streamed.strip() == whole.normal_text
    assert [c.name for c in calls] == [c.name for c in whole.calls]


class TestAssembledToolCallArgumentsMustParse:
    """The rule at the level it is written, independent of any one parser."""

    @staticmethod
    def _fragments(*parameters):
        return {
            index: {
                "name": f"tool_{index}",
                "parameters": value
            }
            for index, value in enumerate(parameters)
        }

    @pytest.mark.parametrize("parameters,shape", [
        ('{"cmd": }', "a key with no value - the recorded failure"),
        ('{"cmd": "kill -0 695', "a string the closing quote never reached"),
        ('{"cmd": "ls"', "the object never closed"),
        ('{"a": 1,}', "a trailing comma"),
        ("{", "nothing but the opening brace"),
        ("", "a name announced and nothing after it"),
    ])
    def test_arguments_that_do_not_parse_are_dropped(self, parameters, shape,
                                                     responses_warnings):
        assert _assembled_tool_calls(self._fragments(parameters)) == [], shape
        assert len(responses_warnings) == 1
        assert "tool_0" in responses_warnings[0]

    @pytest.mark.parametrize("parameters", [
        "{}",
        '{"a": 1}',
        '{"a": {"b": [1, 2]}}',
        '{"text": "}"}',
        '{"text": "a \\" b"}',
        '{"text": "\\u00e9"}',
    ])
    def test_arguments_that_parse_are_kept(self, parameters,
                                           responses_warnings):
        assert [
            c.parameters
            for c in _assembled_tool_calls(self._fragments(parameters))
        ] == [parameters]
        assert responses_warnings == []

    def test_a_bad_call_does_not_take_the_good_ones_with_it(self):
        fragments = self._fragments('{"a": 1}', '{"b": ', '{"c": 3}')

        assert [c.tool_index
                for c in _assembled_tool_calls(fragments)] == [0, 2]

    def test_the_unfinished_call_is_not_reported_as_a_json_failure(
            self, responses_warnings):
        """Both rules match it; only the one that owns it may speak."""
        fragments = self._fragments('{"a": 1}', '{"b": ')

        assembled = _assembled_tool_calls(fragments, unfinished_tool_index=1)

        assert [c.tool_index for c in assembled] == [0]
        assert responses_warnings == []


# ---------------------------------------------------------------------------
# BaseToolParser.resolve_tool_name
#
# Recovery of a mangled tool name. Measured on a GLM-5.3 fleet serving a
# Codex client: of 99 calls the client answered `unsupported call`, the
# resolver recovered 0 -- it could not match anything at all on that workload,
# because the tools arrive inside a `namespace` and are therefore declared
# qualified (`functions.exec`) while the model writes them bare (`exec`).
# ---------------------------------------------------------------------------

_NAMESPACED = {
    "functions.exec": 0,
    "functions.wait": 1,
    "collaboration.spawn_agent": 2,
    "collaboration.list_agents": 3,
}


def _resolve(name, indices=None):
    from tensorrt_llm.serve.tool_parser.base_tool_parser import BaseToolParser

    return BaseToolParser.resolve_tool_name(
        name, _NAMESPACED if indices is None else indices)


def test_a_bare_name_maps_onto_the_declared_qualified_tool():
    """The direction this workload needs, and the one that was missing.

    3102 of 3730 calls on the measured fleet used the bare spelling.
    """
    assert _resolve("exec") == "functions.exec"
    assert _resolve("spawn_agent") == "collaboration.spawn_agent"


def test_markup_fused_onto_a_bare_name_is_stripped_then_mapped():
    """Both defects at once: strip the stray tag, then bare -> qualified."""
    assert _resolve("<tool_call>exec") == "functions.exec"
    assert _resolve("exec</arg_value>") == "functions.exec"


def test_markup_fused_onto_a_qualified_name_matches_it_verbatim():
    """A stripped candidate is matched whole, not only by its tail.

    Regression: testing only the tail meant a declared qualified name that
    the strip left intact still failed to match.
    """
    assert _resolve("functions.exec </arg_value>") == "functions.exec"


def test_a_qualifier_is_still_dropped_when_the_bare_tool_is_declared():
    """The original direction must keep working."""
    assert _resolve("functions.exec_command",
                    {"exec_command": 0}) == "exec_command"


def test_an_ambiguous_bare_name_is_not_guessed():
    """An ambiguous bare name is left alone rather than guessed.

    Two namespaces offering the same tool cannot be told apart from the bare
    name, and mis-routing a call is worse than not recovering it.
    """
    ambiguous = {"a.status": 0, "b.status": 1}
    assert _resolve("status", ambiguous) is None


def test_an_exact_declared_name_wins_over_a_tail_match():
    both = {"exec": 0, "functions.exec": 1}
    assert _resolve("exec", both) == "exec"


def test_prose_containing_a_tool_name_is_not_turned_into_a_call():
    """Prose is never repaired into a call.

    This is the guard the original implementation was built around: a name
    that merely *contains* a declared tool must not fabricate a call the
    model never made.
    """
    assert _resolve("collab? no. Use exec.<tool_call>exec") is None
    assert _resolve("ops_check - re-querying for current status, since my "
                    "last report") is None
    assert _resolve(
        "exec surg? No, actually use proper functions.exec tool") is None
    assert _resolve("exec_command_placeholder</arg_value>") is None


def test_unknown_and_empty_names_resolve_to_nothing():
    assert _resolve("totally_undeclared") is None
    assert _resolve("") is None
    assert _resolve(None) is None
