"""
tests/async_tool_loop/test_utils.py
=========================================

Unit tests for utility functions in unity.common._async_tool.
"""

import asyncio
import json
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel, Field

from unity.common._async_tool.utils import (
    get_handle_paused_state,
    maybe_await,
    try_parse_json,
)
from unity.common._async_tool.formatting import serialize_tool_content

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
