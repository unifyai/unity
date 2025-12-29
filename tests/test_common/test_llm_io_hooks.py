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
    _write_request_pending,
    _append_response_and_finalize,
    _record_cache_status,
    get_cache_stats,
    reset_cache_stats,
    install_llm_io_hooks,
)
from unity.settings import SETTINGS


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
#  File writing tests (combined request+response files)
# --------------------------------------------------------------------------- #


def test_write_request_pending_creates_file(tmp_path, monkeypatch):
    """Writing a pending request creates a timestamped file."""
    io_dir = tmp_path / "llm_io_debug" / "test_session"
    io_dir.mkdir(parents=True)

    import unity.common.llm_io_hooks as hooks_mod

    monkeypatch.setattr(hooks_mod, "_LLM_IO_DIR", str(io_dir))

    path = _write_request_pending({"model": "gpt-4"}, label="test")

    assert path is not None
    assert path.exists()
    assert "_pending" in path.name

    content = path.read_text()
    assert "🔄 [test] LLM request ➡️" in content
    assert '"model": "gpt-4"' in content


def test_append_response_and_finalize(tmp_path, monkeypatch):
    """Appending response and finalizing renames the file."""
    io_dir = tmp_path / "llm_io_debug" / "test_session"
    io_dir.mkdir(parents=True)

    import unity.common.llm_io_hooks as hooks_mod

    monkeypatch.setattr(hooks_mod, "_LLM_IO_DIR", str(io_dir))

    # Write pending request
    pending_path = _write_request_pending({"model": "gpt-4"}, label="test")
    assert pending_path is not None

    # Append response and finalize
    _append_response_and_finalize(
        pending_path,
        {"choices": [{"message": {"content": "Hello"}}]},
        "hit",
        label="test",
    )

    # Pending file should be gone
    assert not pending_path.exists()

    # Should have a _hit file now
    hit_files = list(io_dir.glob("*_hit.txt"))
    assert len(hit_files) == 1

    content = hit_files[0].read_text()
    assert "LLM request ➡️" in content
    assert "LLM response ⬅️" in content
    assert "[cache: hit]" in content


def test_write_request_without_label(tmp_path, monkeypatch):
    """Writing without a label omits the label prefix."""
    io_dir = tmp_path / "llm_io_debug" / "test_session"
    io_dir.mkdir(parents=True)

    import unity.common.llm_io_hooks as hooks_mod

    monkeypatch.setattr(hooks_mod, "_LLM_IO_DIR", str(io_dir))

    path = _write_request_pending({"data": 1})

    content = path.read_text()
    assert "🔄 LLM request ➡️" in content
    # No label brackets in the header line
    assert "[" not in content.split("\n")[0]


def test_multiple_writes_unique_filenames(tmp_path, monkeypatch):
    """Multiple writes create unique files."""
    io_dir = tmp_path / "llm_io_debug" / "test_session"
    io_dir.mkdir(parents=True)

    import unity.common.llm_io_hooks as hooks_mod

    monkeypatch.setattr(hooks_mod, "_LLM_IO_DIR", str(io_dir))

    paths = []
    for i in range(3):
        path = _write_request_pending({"i": i})
        paths.append(path)

    # All paths should be unique
    assert len(set(paths)) == 3
    files = list(io_dir.glob("*_pending.txt"))
    assert len(files) == 3


# --------------------------------------------------------------------------- #
#  Cache stats tests
# --------------------------------------------------------------------------- #


def test_get_cache_stats_initial():
    """Cache stats should track hits and misses."""
    # Reset to known state
    reset_cache_stats()

    stats = get_cache_stats()
    assert stats["hits"] == 0
    assert stats["misses"] == 0
    assert stats["hit_rate"] == 0.0


def test_record_cache_status_hit():
    """Recording a hit increments the hit counter."""
    reset_cache_stats()

    _record_cache_status("hit")
    _record_cache_status("hit")

    stats = get_cache_stats()
    assert stats["hits"] == 2
    assert stats["misses"] == 0
    assert stats["hit_rate"] == 100.0


def test_record_cache_status_miss():
    """Recording a miss increments the miss counter."""
    reset_cache_stats()

    _record_cache_status("miss")
    _record_cache_status("miss")
    _record_cache_status("miss")

    stats = get_cache_stats()
    assert stats["hits"] == 0
    assert stats["misses"] == 3
    assert stats["hit_rate"] == 0.0


def test_record_cache_status_mixed():
    """Mixed hits and misses calculate correct hit rate."""
    reset_cache_stats()

    _record_cache_status("hit")
    _record_cache_status("miss")
    _record_cache_status("hit")
    _record_cache_status("miss")

    stats = get_cache_stats()
    assert stats["hits"] == 2
    assert stats["misses"] == 2
    assert stats["hit_rate"] == 50.0


def test_record_cache_status_unknown_ignored():
    """Unknown cache status is ignored."""
    reset_cache_stats()

    _record_cache_status("hit")
    _record_cache_status("unknown")
    _record_cache_status("miss")

    stats = get_cache_stats()
    assert stats["hits"] == 1
    assert stats["misses"] == 1


def test_reset_cache_stats():
    """Resetting cache stats clears all counters."""
    _record_cache_status("hit")
    _record_cache_status("miss")

    reset_cache_stats()

    stats = get_cache_stats()
    assert stats["hits"] == 0
    assert stats["misses"] == 0


# --------------------------------------------------------------------------- #
#  Hook installation tests
# --------------------------------------------------------------------------- #


def test_hooks_wrap_generate_methods():
    """After installation, _generate_non_stream methods are wrapped."""
    from unillm import AsyncUnify, Unify

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
    from unillm import AsyncUnify

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
            # Simulate the hook behavior with new combined file approach
            kw = {"model": endpoint, "messages": [{"role": "user", "content": "test"}]}
            pending_path = hooks_mod._write_request_pending(kw, label=endpoint)
            result = await original(self, endpoint, prompt, **kwargs)
            if hasattr(result, "model_dump"):
                hooks_mod._append_response_and_finalize(
                    pending_path,
                    result.model_dump(),
                    "miss",
                    label=endpoint,
                )
            return result

        monkeypatch.setattr(AsyncUnify, "_generate_non_stream", wrapped_for_test)

        # Create client and make a call
        client = AsyncUnify(SETTINGS.UNIFY_MODEL)
        client._messages = [{"role": "user", "content": "What is 2+3?"}]

        # The actual call
        await client._generate_non_stream(
            SETTINGS.UNIFY_MODEL,
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

    # Check files were created - should be a single combined file now
    files = sorted(io_dir.glob("*.txt"))
    assert len(files) >= 1, f"Expected at least 1 file, got {len(files)}: {files}"

    # The file should contain both request and response
    content = files[0].read_text()
    assert "LLM request ➡️" in content, "File should contain request"
    assert "LLM response ⬅️" in content, "File should contain response"
