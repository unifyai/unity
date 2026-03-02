"""
tests/async_tool_loop/test_utils.py
=========================================

Unit tests for utility functions in unity.common._async_tool.
"""

import asyncio
import json
import logging
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel, Field

from unity.common._async_tool.utils import (
    format_json_for_log,
    format_llm_response_for_log,
    get_handle_paused_state,
    maybe_await,
    try_parse_json,
)
from unity.common._async_tool.formatting import serialize_tool_content
from unity.logger import _MillisFormatter
from unity.syntax_highlight import highlight_code_blocks

# =============================================================================
# get_handle_paused_state tests
# =============================================================================


class TestGetHandlePausedState:
    """Tests for get_handle_paused_state helper function."""

    def test_returns_true_when_paused(self):
        """Returns True when handle's _pause_event is cleared (paused)."""
        mock_handle = MagicMock()
        mock_handle._pause_event = asyncio.Event()
        mock_handle._pause_event.clear()  # Paused state

        result = get_handle_paused_state(mock_handle)
        assert result is True

    def test_returns_false_when_running(self):
        """Returns False when handle's _pause_event is set (running)."""
        mock_handle = MagicMock()
        mock_handle._pause_event = asyncio.Event()
        mock_handle._pause_event.set()  # Running state

        result = get_handle_paused_state(mock_handle)
        assert result is False

    def test_returns_none_when_no_pause_event(self):
        """Returns None when handle has no _pause_event attribute."""
        mock_handle = MagicMock(spec=[])  # No attributes

        result = get_handle_paused_state(mock_handle)
        assert result is None

    def test_returns_none_when_pause_event_is_none(self):
        """Returns None when handle._pause_event is None."""
        mock_handle = MagicMock(spec=["_pause_event"])
        mock_handle._pause_event = None

        result = get_handle_paused_state(mock_handle)
        assert result is None

    def test_returns_none_when_pause_event_has_no_is_set(self):
        """Returns None when _pause_event doesn't have is_set method."""
        mock_handle = MagicMock(spec=["_pause_event"])
        mock_handle._pause_event = "not an event"

        result = get_handle_paused_state(mock_handle)
        assert result is None

    def test_handles_exception_gracefully(self):
        """Returns None when accessing _pause_event raises an exception."""
        mock_handle = MagicMock()
        type(mock_handle)._pause_event = property(
            lambda self: (_ for _ in ()).throw(RuntimeError("test error")),
        )

        result = get_handle_paused_state(mock_handle)
        assert result is None

    def test_with_mocked_is_set(self):
        """Works with mocked is_set return values."""
        # Test with is_set returning True (running)
        mock_handle = MagicMock()
        mock_handle._pause_event = MagicMock()
        mock_handle._pause_event.is_set.return_value = True

        result = get_handle_paused_state(mock_handle)
        assert result is False  # Running (not paused)

        # Test with is_set returning False (paused)
        mock_handle._pause_event.is_set.return_value = False

        result = get_handle_paused_state(mock_handle)
        assert result is True  # Paused

    def test_with_none_handle(self):
        """Returns None when handle is None."""
        result = get_handle_paused_state(None)
        assert result is None

    def test_with_pause_event_proxy(self):
        """Works with proxy objects that expose is_set()."""

        class _PauseStateProxy:
            """Minimal proxy exposing is_set()."""

            def __init__(self, paused: bool):
                self._paused = paused

            def is_set(self) -> bool:
                return not self._paused  # Event set = running

        # Test paused state
        mock_handle = MagicMock(spec=["_pause_event"])
        mock_handle._pause_event = _PauseStateProxy(paused=True)
        assert get_handle_paused_state(mock_handle) is True

        # Test running state
        mock_handle._pause_event = _PauseStateProxy(paused=False)
        assert get_handle_paused_state(mock_handle) is False


# =============================================================================
# maybe_await tests
# =============================================================================


class TestMaybeAwait:
    """Tests for maybe_await helper function."""

    @pytest.mark.asyncio
    async def test_returns_value_for_non_awaitable(self):
        """Returns value directly if not awaitable."""
        result = await maybe_await(42)
        assert result == 42

    @pytest.mark.asyncio
    async def test_returns_value_for_string(self):
        """Returns string directly."""
        result = await maybe_await("hello")
        assert result == "hello"

    @pytest.mark.asyncio
    async def test_awaits_coroutine(self):
        """Awaits and returns result for coroutine."""

        async def async_func():
            return "async result"

        result = await maybe_await(async_func())
        assert result == "async result"

    @pytest.mark.asyncio
    async def test_awaits_future(self):
        """Awaits and returns result for Future."""
        future = asyncio.Future()
        future.set_result("future result")

        result = await maybe_await(future)
        assert result == "future result"


# =============================================================================
# try_parse_json tests
# =============================================================================


class TestTryParseJson:
    """Tests for try_parse_json helper function."""

    def test_parses_valid_json_string(self):
        """Parses valid JSON string."""
        result = try_parse_json('{"key": "value"}')
        assert result == {"key": "value"}

    def test_parses_json_array(self):
        """Parses JSON array."""
        result = try_parse_json("[1, 2, 3]")
        assert result == [1, 2, 3]

    def test_returns_non_string_unchanged(self):
        """Returns non-string values unchanged."""
        dict_val = {"key": "value"}
        result = try_parse_json(dict_val)
        assert result is dict_val

    def test_returns_invalid_json_unchanged(self):
        """Returns invalid JSON string unchanged."""
        invalid = "not valid json"
        result = try_parse_json(invalid)
        assert result == invalid

    def test_returns_int_unchanged(self):
        """Returns int unchanged."""
        result = try_parse_json(42)
        assert result == 42

    def test_returns_none_unchanged(self):
        """Returns None unchanged."""
        result = try_parse_json(None)
        assert result is None


# =============================================================================
# format_json_for_log / format_llm_response_for_log tests
# =============================================================================


class TestFormatJsonForLog:
    """Tests for format_json_for_log — newline expansion in terminal logs."""

    def test_expands_newlines_in_string_values(self):
        """Escaped newlines in string values become real newlines."""
        body = {"content": "Line one\nLine two\nLine three"}
        result = format_json_for_log(body)
        lines = result.split("\n")

        content_line = next(l for l in lines if "Line one" in l)
        idx = lines.index(content_line)
        assert "Line two" in lines[idx + 1]
        assert "Line three" in lines[idx + 2]

    def test_preserves_json_structure(self):
        """Keys, brackets, and non-newline content are unchanged."""
        body = {"key": "value", "nested": {"inner": "data"}}
        result = format_json_for_log(body)
        assert '"key"' in result
        assert '"nested"' in result
        assert '"inner"' in result
        assert '"data"' in result

    def test_markdown_content_renders_naturally(self):
        """Markdown with headers and bullets renders across lines."""
        body = {
            "content": "### Title\n- Item one\n- Item two\n\nParagraph.",
            "role": "assistant",
        }
        result = format_json_for_log(body)
        lines = result.split("\n")

        title_line = next(l for l in lines if "### Title" in l)
        idx = lines.index(title_line)
        assert "- Item one" in lines[idx + 1]
        assert "- Item two" in lines[idx + 2]


class TestFormatLlmResponseForLog:
    """Tests for format_llm_response_for_log — bespoke execute_code rendering."""

    def test_execute_code_gets_markdown_fences(self):
        """execute_code tool calls have markdown fenced code blocks."""
        msg = {
            "content": "Running the code.",
            "role": "assistant",
            "tool_calls": [
                {
                    "function": {
                        "name": "execute_code",
                        "arguments": json.dumps(
                            {
                                "thought": "Render the PDF",
                                "language": "python",
                                "code": "\nimport os\nprint(os.getcwd())\n",
                            },
                        ),
                    },
                },
            ],
        }
        result = format_llm_response_for_log(msg)

        assert "```python" in result
        lines = result.split("\n")
        open_idx = next(i for i, l in enumerate(lines) if "```python" in l)
        close_indices = [
            i
            for i, l in enumerate(lines)
            if i > open_idx and l.strip() == '```"' or l.strip() == "```"
        ]
        assert len(close_indices) >= 1
        code_lines = lines[open_idx + 1 : close_indices[0]]
        code_text = "\n".join(l.strip() for l in code_lines)
        assert "import os" in code_text
        assert "print(os.getcwd())" in code_text

    def test_non_execute_code_tool_calls_unaffected(self):
        """Tool calls other than execute_code don't get fenced delimiters."""
        msg = {
            "content": "",
            "role": "assistant",
            "tool_calls": [
                {
                    "function": {
                        "name": "some_other_tool",
                        "arguments": json.dumps({"query": "hello\nworld"}),
                    },
                },
            ],
        }
        result = format_llm_response_for_log(msg)

        assert "```" not in result
        lines = result.split("\n")
        hello_line = next(l for l in lines if "hello" in l)
        idx = lines.index(hello_line)
        assert "world" in lines[idx + 1]

    def test_does_not_mutate_original_message(self):
        """The original message dict is not modified."""
        original_args = json.dumps({"code": "x = 1\ny = 2\n", "language": "python"})
        msg = {
            "role": "assistant",
            "tool_calls": [
                {"function": {"name": "execute_code", "arguments": original_args}},
            ],
        }
        format_llm_response_for_log(msg)
        assert msg["tool_calls"][0]["function"]["arguments"] == original_args


# =============================================================================
# highlight_code_blocks / _MillisFormatter TTY-aware highlighting
# =============================================================================


class TestHighlightCodeBlocks:
    """Tests for Pygments-based syntax highlighting of markdown-fenced code blocks."""

    def test_highlights_python_code_block(self):
        """Code between ```lang and ``` receives ANSI escape codes."""
        text = "some prefix\n" "```python\n" "x = 42\n" "```\n" "some suffix"
        result = highlight_code_blocks(text)
        assert "\033[" in result
        assert "42" in result
        assert "some prefix" in result
        assert "some suffix" in result

    def test_preserves_text_outside_delimiters(self):
        """Text outside code blocks is unchanged."""
        text = "before\n" "```python\n" "pass\n" "```\n" "after"
        result = highlight_code_blocks(text)
        assert result.startswith("before\n")
        assert result.endswith("after")

    def test_no_delimiters_returns_unchanged(self):
        """Text without code block delimiters passes through unchanged."""
        text = "just a regular log message with no code"
        assert highlight_code_blocks(text) == text

    def test_unknown_language_falls_back_to_plain(self):
        """Unrecognised language leaves the code block unchanged."""
        text = "```nonexistent_lang_xyz\n" "some code\n" "```"
        result = highlight_code_blocks(text)
        assert "some code" in result

    def test_multiple_code_blocks(self):
        """Multiple code blocks in the same message are each highlighted."""
        text = (
            "```python\n"
            "x = 1\n"
            "```\n"
            "middle text\n"
            "```python\n"
            "y = 2\n"
            "```"
        )
        result = highlight_code_blocks(text)
        assert "middle text" in result
        ansi_count = result.count("\033[")
        assert ansi_count > 2

    def test_with_indentation(self):
        """Indented blocks (from JSON expansion) are highlighted."""
        text = (
            "                ```python\n"
            "                img = render(page=0)\n"
            "                ```"
        )
        result = highlight_code_blocks(text)
        assert "\033[" in result
        assert "render" in result

    def test_closing_not_confused_with_opening(self):
        """Closing ``` is not confused with an opening ```lang."""
        text = "```python\n" "x = 1\n" "```\n" "```python\n" "y = 2\n" "```"
        result = highlight_code_blocks(text)
        ansi_count = result.count("\033[")
        assert ansi_count > 2


class TestMillisFormatterTtyHighlighting:
    """Tests for _MillisFormatter conditional TTY highlighting."""

    def _make_record(self, msg: str) -> logging.LogRecord:
        return logging.LogRecord(
            name="unity",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg=msg,
            args=(),
            exc_info=None,
        )

    def test_tty_formatter_highlights_code_blocks(self):
        """When stream.isatty() is True, code blocks get ANSI highlighting."""

        class FakeTTY:
            def isatty(self):
                return True

        fmt = _MillisFormatter(stream=FakeTTY())
        msg = (
            "🤖 [CodeActActor.act(abcd)] {\n"
            "    ```python\n"
            "    x = 42\n"
            "    ```\n"
            "}"
        )
        result = fmt.format(self._make_record(msg))
        assert "\033[" in result
        assert "42" in result

    def test_non_tty_formatter_does_not_highlight(self):
        """When stream.isatty() is False, code blocks are plain text."""

        class FakeFile:
            def isatty(self):
                return False

        fmt = _MillisFormatter(stream=FakeFile())
        msg = (
            "🤖 [CodeActActor.act(abcd)] {\n"
            "    ```python\n"
            "    x = 42\n"
            "    ```\n"
            "}"
        )
        result = fmt.format(self._make_record(msg))
        assert "\033[" not in result
        assert "x = 42" in result

    def test_no_stream_defaults_to_no_highlighting(self):
        """When no stream is provided, highlighting is off."""
        fmt = _MillisFormatter()
        msg = "```python\n" "x = 42\n" "```"
        result = fmt.format(self._make_record(msg))
        assert "\033[" not in result


# =============================================================================
# serialize_tool_content – Pydantic model payloads
# =============================================================================


class _SampleModel(BaseModel):
    name: str = Field(..., description="A name.")
    value: int = Field(..., description="A number.")


class TestSerializeToolContentPydanticModel:
    """serialize_tool_content should serialize Pydantic models as clean JSON,
    not fall through to str() which produces Python repr."""

    def test_pydantic_model_serialized_as_json(self):
        """A BaseModel payload should produce valid JSON, not a Python repr."""
        model = _SampleModel(name="alice", value=42)

        result = serialize_tool_content(
            tool_name="my_tool",
            payload=model,
            is_final=True,
        )

        assert isinstance(result, str), f"Expected str, got {type(result).__name__}"
        parsed = json.loads(result)
        assert parsed["name"] == "alice"
        assert parsed["value"] == 42
