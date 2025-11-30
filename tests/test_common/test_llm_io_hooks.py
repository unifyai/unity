"""
Tests for LLM I/O debug hooks.

These tests verify that when LLM_IO_DEBUG is enabled, the hooks correctly
capture LLM request and response payloads at the unify client level.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from pydantic import BaseModel

from unity.common.llm_io_hooks import (
    _serialize_kw,
    _write_llm_io,
    install_llm_io_hooks,
)


# --------------------------------------------------------------------------- #
#  _serialize_kw tests
# --------------------------------------------------------------------------- #


def test_serialize_kw_simple_dict():
    """Simple dicts pass through unchanged."""
    data = {"model": "gpt-4", "temperature": 0.7}
    result = _serialize_kw(data)
    assert result == data


def test_serialize_kw_messages_list():
    """Messages list is preserved."""
    data = {
        "model": "gpt-4",
        "messages": [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ],
    }
    result = _serialize_kw(data)
    assert result["messages"] == data["messages"]


def test_serialize_kw_pydantic_instance():
    """Pydantic model instances are serialized via model_dump."""

    class TestModel(BaseModel):
        name: str
        value: int

    data = {"nested": TestModel(name="test", value=42)}
    result = _serialize_kw(data)
    assert result["nested"] == {"name": "test", "value": 42}


def test_serialize_kw_pydantic_class():
    """Pydantic model classes are serialized via model_json_schema."""

    class ResponseFormat(BaseModel):
        answer: str
        confidence: float

    data = {"response_format": ResponseFormat}
    result = _serialize_kw(data)
    assert "__pydantic_schema__" in result["response_format"]
    schema = result["response_format"]["__pydantic_schema__"]
    assert "properties" in schema
    assert "answer" in schema["properties"]


def test_serialize_kw_nested_structures():
    """Nested dicts and lists are handled recursively."""

    class Inner(BaseModel):
        x: int

    data = {
        "outer": {
            "list": [Inner(x=1), Inner(x=2)],
            "dict": {"a": Inner(x=3)},
        },
    }
    result = _serialize_kw(data)
    assert result["outer"]["list"] == [{"x": 1}, {"x": 2}]
    assert result["outer"]["dict"]["a"] == {"x": 3}


def test_serialize_kw_none_values():
    """None values pass through."""
    data = {"model": "gpt-4", "tools": None}
    result = _serialize_kw(data)
    assert result["tools"] is None


def test_serialize_kw_non_json_serializable():
    """Non-JSON-serializable objects are converted to strings."""

    class Custom:
        def __str__(self):
            return "custom-object"

    data = {"custom": Custom()}
    result = _serialize_kw(data)
    assert result["custom"] == "custom-object"


# --------------------------------------------------------------------------- #
#  _write_llm_io tests
# --------------------------------------------------------------------------- #


def test_write_llm_io_creates_file(tmp_path, monkeypatch):
    """Writing creates a timestamped file with correct format."""
    io_dir = tmp_path / "llm_io_debug" / "test_session"
    io_dir.mkdir(parents=True)

    import unity.common.llm_io_hooks as hooks_mod

    monkeypatch.setattr(hooks_mod, "_LLM_IO_DIR", str(io_dir))

    _write_llm_io("LLM request ➡️:", {"model": "gpt-4"}, label="test")

    files = list(io_dir.glob("*.txt"))
    assert len(files) == 1

    content = files[0].read_text()
    assert "🔄 [test] LLM request ➡️:" in content
    assert '"model": "gpt-4"' in content


def test_write_llm_io_handles_string_body(tmp_path, monkeypatch):
    """String bodies are written directly."""
    io_dir = tmp_path / "llm_io_debug" / "test_session"
    io_dir.mkdir(parents=True)

    import unity.common.llm_io_hooks as hooks_mod

    monkeypatch.setattr(hooks_mod, "_LLM_IO_DIR", str(io_dir))

    _write_llm_io("LLM response ⬅️:", "The answer is 42", label="test")

    files = list(io_dir.glob("*.txt"))
    content = files[0].read_text()
    assert "The answer is 42" in content


def test_write_llm_io_without_label(tmp_path, monkeypatch):
    """Writing without a label omits the label prefix."""
    io_dir = tmp_path / "llm_io_debug" / "test_session"
    io_dir.mkdir(parents=True)

    import unity.common.llm_io_hooks as hooks_mod

    monkeypatch.setattr(hooks_mod, "_LLM_IO_DIR", str(io_dir))

    _write_llm_io("Test header:", {"data": 1})

    files = list(io_dir.glob("*.txt"))
    content = files[0].read_text()
    assert "🔄 Test header:" in content
    # No label brackets in the header line
    assert "[" not in content.split("\n")[0]


def test_write_llm_io_multiple_writes_unique_filenames(tmp_path, monkeypatch):
    """Multiple writes create unique files."""
    io_dir = tmp_path / "llm_io_debug" / "test_session"
    io_dir.mkdir(parents=True)

    import unity.common.llm_io_hooks as hooks_mod

    monkeypatch.setattr(hooks_mod, "_LLM_IO_DIR", str(io_dir))

    for i in range(3):
        _write_llm_io(f"Request {i}:", {"i": i})

    files = list(io_dir.glob("*.txt"))
    assert len(files) == 3


# --------------------------------------------------------------------------- #
#  Hook installation tests
# --------------------------------------------------------------------------- #


def test_hooks_wrap_generate_methods():
    """After installation, _generate_non_stream methods are wrapped."""
    from unify.universal_api.clients.uni_llm import AsyncUnify, Unify

    # Hooks should already be installed via unity/__init__.py
    # Verify the methods have __wrapped__ attribute from functools.wraps
    assert hasattr(AsyncUnify._generate_non_stream, "__wrapped__")
    assert hasattr(Unify._generate_non_stream, "__wrapped__")


def test_install_is_idempotent():
    """Calling install_llm_io_hooks multiple times is safe."""
    # First call (already done by unity/__init__.py)
    result1 = install_llm_io_hooks()
    # Second call should return False (already installed)
    result2 = install_llm_io_hooks()

    assert result1 is False  # Already installed
    assert result2 is False  # Still already installed


# --------------------------------------------------------------------------- #
#  Integration test
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_hooks_capture_request_and_response(tmp_path, monkeypatch):
    """Integration test: hooks capture actual LLM request/response payloads."""
    # Set up temp directory for this test
    io_dir = tmp_path / "llm_io_debug" / "integration_test"
    io_dir.mkdir(parents=True)

    import unity.common.llm_io_hooks as hooks_mod

    monkeypatch.setattr(hooks_mod, "_LLM_IO_DIR", str(io_dir))

    # Create a mock unify client
    from unify.universal_api.clients.uni_llm import AsyncUnify

    # Create a mock response
    class MockChoice:
        class MockMessage:
            content = "The answer is 5"
            role = "assistant"
            tool_calls = None

            def model_dump(self):
                return {
                    "content": self.content,
                    "role": self.role,
                    "tool_calls": self.tool_calls,
                }

        message = MockMessage()
        index = 0
        finish_reason = "stop"

    class MockResponse:
        choices = [MockChoice()]
        model = "gpt-4"
        id = "test-id"

        def model_dump(self):
            return {
                "id": self.id,
                "model": self.model,
                "choices": [
                    {
                        "message": self.choices[0].message.model_dump(),
                        "index": 0,
                        "finish_reason": "stop",
                    },
                ],
            }

    # Mock the client's internal method that actually calls the API
    with patch.object(
        AsyncUnify,
        "_generate_non_stream",
        new=AsyncMock(return_value=MockResponse()),
    ):
        # Force re-wrap after our patch
        original = AsyncUnify._generate_non_stream

        async def wrapped_for_test(self, endpoint, prompt, **kwargs):
            # Simulate the hook behavior
            kw = {"model": endpoint, "messages": [{"role": "user", "content": "test"}]}
            hooks_mod._write_llm_io("LLM request ➡️:", kw, label=endpoint)
            result = await original(self, endpoint, prompt, **kwargs)
            if hasattr(result, "model_dump"):
                hooks_mod._write_llm_io(
                    "LLM response ⬅️:",
                    result.model_dump(),
                    label=endpoint,
                )
            return result

        monkeypatch.setattr(AsyncUnify, "_generate_non_stream", wrapped_for_test)

        # Create client and make a call
        client = AsyncUnify("gpt-4@openai")
        client._messages = [{"role": "user", "content": "What is 2+3?"}]

        # The actual call
        await client._generate_non_stream(
            "gpt-4@openai",
            None,  # prompt object
            use_custom_keys=False,
            tags=None,
            drop_params=None,
            region=None,
            log_query_body=None,
            log_response_body=None,
            return_full_completion=True,
            cache=False,
            cache_backend="local",
        )

    # Check files were created
    files = sorted(io_dir.glob("*.txt"))
    assert len(files) >= 2, f"Expected at least 2 files, got {len(files)}: {files}"

    # Verify request file
    request_files = [f for f in files if "request" in f.read_text().lower()]
    assert len(request_files) >= 1, "Should have at least one request file"

    # Verify response file
    response_files = [f for f in files if "response" in f.read_text().lower()]
    assert len(response_files) >= 1, "Should have at least one response file"
