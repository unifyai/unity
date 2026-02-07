"""
tests/async_tool_loop/test_composite_handle_return.py
=====================================================

Tests for composite tool returns: a tool returns both intermediate data
*and* a SteerableToolHandle nested inside a dict / list / tuple.

The loop extracts the handle, replaces it with a sentinel in the
intermediate data, presents the data to the LLM as in-flight progress,
and continues steering the handle as usual.
"""

import asyncio

import pytest

from unity.common.async_tool_loop import (
    SteerableToolHandle,
    start_async_tool_loop,
)
from unity.common._async_tool.tools_data import (
    _extract_nested_handle,
    _HANDLE_SENTINEL,
)
from unity.common.tool_spec import ToolSpec
from unity.common.llm_client import new_llm_client
from tests.async_helpers import (
    _wait_for_tool_request,
    _wait_for_tool_result,
    _wait_for_assistant_call_prefix,
    real_tool_messages,
)

# ─────────────────────────────────────────────────────────────────────────────
#  Unit tests for _extract_nested_handle
# ─────────────────────────────────────────────────────────────────────────────


class _MinimalHandle(SteerableToolHandle):
    """Lightweight concrete handle for unit tests (no event loop needed)."""

    def __init__(self):
        pass

    async def ask(self, question, **kw):
        return ""

    async def interject(self, message, **kw):
        pass

    def stop(self, reason=None, **kw):
        pass

    def pause(self, **kw):
        pass

    def resume(self, **kw):
        pass

    def done(self):
        return True

    async def result(self):
        return "done"

    async def next_clarification(self):
        return {}

    async def next_notification(self):
        return {}

    async def answer_clarification(self, cid, ans):
        pass


class TestExtractNestedHandle:
    """Pure-function unit tests – no LLM calls needed."""

    def _make_handle(self):
        return _MinimalHandle()

    def test_no_handle_returns_none(self):
        obj = {"a": 1, "b": [2, 3]}
        handle, cleaned = _extract_nested_handle(obj)
        assert handle is None
        assert cleaned is obj

    def test_direct_handle_returned(self):
        h = self._make_handle()
        handle, cleaned = _extract_nested_handle(h)
        assert handle is h
        assert cleaned == _HANDLE_SENTINEL

    def test_handle_in_dict(self):
        h = self._make_handle()
        obj = {"data": 42, "handle": h}
        handle, cleaned = _extract_nested_handle(obj)
        assert handle is h
        assert cleaned == {"data": 42, "handle": _HANDLE_SENTINEL}

    def test_handle_in_list(self):
        h = self._make_handle()
        obj = ["some_data", h]
        handle, cleaned = _extract_nested_handle(obj)
        assert handle is h
        assert cleaned == ["some_data", _HANDLE_SENTINEL]

    def test_handle_in_tuple(self):
        h = self._make_handle()
        obj = (False, h)
        handle, cleaned = _extract_nested_handle(obj)
        assert handle is h
        assert cleaned == (False, _HANDLE_SENTINEL)

    def test_handle_deeply_nested(self):
        h = self._make_handle()
        obj = {"result": {"nested": [{"deep": h}]}}
        handle, cleaned = _extract_nested_handle(obj)
        assert handle is h
        assert cleaned == {"result": {"nested": [{"deep": _HANDLE_SENTINEL}]}}

    def test_multiple_handles_raises(self):
        h1 = self._make_handle()
        h2 = self._make_handle()
        obj = {"a": h1, "b": h2}
        with pytest.raises(ValueError, match="2 SteerableToolHandles"):
            _extract_nested_handle(obj)

    def test_plain_values_untouched(self):
        obj = {"x": 1, "y": "hello", "z": [True, None, 3.14]}
        handle, cleaned = _extract_nested_handle(obj)
        assert handle is None
        assert cleaned is obj


# ─────────────────────────────────────────────────────────────────────────────
#  Integration test: composite dict return with handle
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_composite_dict_return_with_handle(llm_config):
    """Tool returns ``{"found": True, "handle": <handle>}``.

    The outer loop should:
    1. Extract the handle and adopt it for steering.
    2. Present the intermediate data (with sentinel) as progress to the LLM.
    3. Expose dynamic steering helpers (stop_*, etc.).
    4. Eventually complete with the handle's final result.
    """

    inner_gate = asyncio.Event()

    class SimpleHandle(SteerableToolHandle):
        """Minimal handle backed by an asyncio gate."""

        def __init__(self):
            self._done = asyncio.Event()
            self._gate = inner_gate
            self._stopped = False

        async def ask(self, question, **kw):
            return "progress: waiting for gate"

        async def interject(self, message, **kw):
            pass

        def stop(self, reason=None, **kw):
            self._stopped = True
            self._done.set()

        def pause(self, **kw):
            pass

        def resume(self, **kw):
            pass

        def done(self):
            return self._done.is_set()

        async def result(self):
            await self._gate.wait()
            self._done.set()
            return "inner-complete"

        async def next_clarification(self):
            return {}

        async def next_notification(self):
            return {}

        async def answer_clarification(self, cid, ans):
            pass

    async def composite_tool() -> dict:
        """Return intermediate data alongside a steerable handle."""
        handle = SimpleHandle()
        return {"found": True, "count": 7, "handle": handle}

    composite_tool.__name__ = "composite_tool"
    composite_tool.__qualname__ = "composite_tool"

    client = new_llm_client(**llm_config)
    client.set_system_message(
        "You are running inside an automated test. Perform the steps exactly:\n"
        "1️⃣  Call `composite_tool` with no arguments.\n"
        "2️⃣  Observe the intermediate result. The tool is still running.\n"
        "3️⃣  Continue waiting for it to complete.\n"
        "4️⃣  Once it finishes, respond with exactly 'all done'.",
    )

    outer_handle = start_async_tool_loop(
        client=client,
        message="start",
        tools={"composite_tool": ToolSpec(fn=composite_tool, max_total_calls=1)},
        max_steps=20,
        timeout=240,
    )

    # Wait for the tool to be called and the placeholder to be inserted
    await _wait_for_tool_request(client, "composite_tool")
    await _wait_for_tool_result(client, tool_name="composite_tool", min_results=1)

    # Verify the intermediate content is visible in the placeholder
    tool_msgs = real_tool_messages(client.messages, tool_name="composite_tool")
    assert tool_msgs, "Expected a tool message placeholder for composite_tool"
    placeholder_content = tool_msgs[0].get("content", "")
    # The intermediate data should contain "found" and the sentinel
    assert (
        "found" in placeholder_content
    ), f"Intermediate data should contain 'found'; got: {placeholder_content}"
    assert (
        "_placeholder" in placeholder_content
    ), f"Intermediate data should be formatted as progress; got: {placeholder_content}"

    # Release the inner gate so the handle completes
    inner_gate.set()

    final = await outer_handle.result()
    assert final is not None, "Loop should complete with a response"

    # After completion, the tool message should contain the handle's final result
    tool_msgs_final = real_tool_messages(client.messages, tool_name="composite_tool")
    assert tool_msgs_final, "Expected final tool message for composite_tool"
    # The final content replaces the progress placeholder
    final_content = tool_msgs_final[0].get("content", "")
    assert (
        "inner-complete" in final_content
    ), f"Final tool result should contain the handle's result; got: {final_content}"


# ─────────────────────────────────────────────────────────────────────────────
#  Integration test: composite return with steering (stop)
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_composite_return_stop_steering(llm_config):
    """Tool returns composite data + handle. The outer loop should expose
    ``stop_*`` helpers that propagate to the inner handle.
    """

    stop_called = {"count": 0}

    class StoppableHandle(SteerableToolHandle):
        def __init__(self):
            self._done = asyncio.Event()

        async def ask(self, question, **kw):
            return "ask response"

        async def interject(self, message, **kw):
            pass

        def stop(self, reason=None, **kw):
            stop_called["count"] += 1
            self._done.set()

        def pause(self, **kw):
            pass

        def resume(self, **kw):
            pass

        def done(self):
            return self._done.is_set()

        async def result(self):
            await self._done.wait()
            return "stopped-result"

        async def next_clarification(self):
            return {}

        async def next_notification(self):
            return {}

        async def answer_clarification(self, cid, ans):
            pass

    async def lookup_tool() -> dict:
        """Return partial lookup results and a handle for the ongoing operation."""
        return {"partial_results": ["item_a", "item_b"], "handle": StoppableHandle()}

    lookup_tool.__name__ = "lookup_tool"
    lookup_tool.__qualname__ = "lookup_tool"

    client = new_llm_client(**llm_config)
    client.set_system_message(
        "You are running inside an automated test.\n"
        "1️⃣  Call `lookup_tool` with no arguments.\n"
        "2️⃣  If the user says 'stop', immediately call the helper whose name "
        "starts with `stop_` (e.g. `stop_lookup_tool_<id>`) exactly once.\n"
        "3️⃣  Once stopped, reply with exactly 'stopped'.",
    )

    outer_handle = start_async_tool_loop(
        client=client,
        message="start",
        tools={"lookup_tool": ToolSpec(fn=lookup_tool, max_total_calls=1)},
        max_steps=20,
        timeout=240,
    )

    await _wait_for_tool_request(client, "lookup_tool")
    await _wait_for_tool_result(client, tool_name="lookup_tool", min_results=1)

    # Interject to trigger the stop helper
    await outer_handle.interject("stop")
    await _wait_for_assistant_call_prefix(client, "stop_")

    final = await outer_handle.result()
    assert final is not None, "Loop should complete"
    assert stop_called["count"] >= 1, "Inner handle stop() should be invoked"


# ─────────────────────────────────────────────────────────────────────────────
#  Integration test: composite return with nested async tool loop
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_composite_return_with_nested_loop(llm_config):
    """Tool returns a dict with metadata + an AsyncToolLoopHandle from
    ``start_async_tool_loop``. The outer loop should present the metadata
    as progress and steer the inner loop to completion.
    """

    async def inner_echo() -> str:
        """Simple inner tool."""
        return "echo-reply"

    inner_echo.__name__ = "inner_echo"
    inner_echo.__qualname__ = "inner_echo"

    async def composite_with_loop() -> dict:
        """Launch a nested loop and return it alongside metadata."""
        inner_client = new_llm_client(**llm_config)
        inner_client.set_system_message(
            "Call `inner_echo` then reply with exactly the result.",
        )
        handle = start_async_tool_loop(
            client=inner_client,
            message="go",
            tools={"inner_echo": inner_echo},
            max_steps=5,
            timeout=60,
        )
        return {"status": "launched", "task_id": "abc-123", "handle": handle}

    composite_with_loop.__name__ = "composite_with_loop"
    composite_with_loop.__qualname__ = "composite_with_loop"

    client = new_llm_client(**llm_config)
    client.set_system_message(
        "You are running inside an automated test.\n"
        "1️⃣  Call `composite_with_loop` with no arguments.\n"
        "2️⃣  Observe the intermediate progress (it should show status and task_id).\n"
        "3️⃣  Wait for it to complete.\n"
        "4️⃣  Reply with exactly 'finished'.",
    )

    outer_handle = start_async_tool_loop(
        client=client,
        message="start",
        tools={
            "composite_with_loop": ToolSpec(
                fn=composite_with_loop,
                max_total_calls=1,
            ),
        },
        max_steps=20,
        timeout=240,
    )

    final = await outer_handle.result()
    assert final is not None, "Outer loop should complete"

    # The intermediate content should have been visible as progress
    tool_msgs = [
        m
        for m in client.messages
        if m.get("role") == "tool" and m.get("name") == "composite_with_loop"
    ]
    assert tool_msgs, "Expected tool messages for composite_with_loop"


# ─────────────────────────────────────────────────────────────────────────────
#  Integration test: multiple handles in return raises
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_multiple_handles_in_return_raises(llm_config):
    """Returning more than one handle in a composite structure should raise."""

    class DummyHandle(SteerableToolHandle):
        def __init__(self):
            self._done = asyncio.Event()

        async def ask(self, question, **kw):
            return ""

        async def interject(self, message, **kw):
            pass

        def stop(self, reason=None, **kw):
            self._done.set()

        def pause(self, **kw):
            pass

        def resume(self, **kw):
            pass

        def done(self):
            return self._done.is_set()

        async def result(self):
            await self._done.wait()
            return "done"

        async def next_clarification(self):
            return {}

        async def next_notification(self):
            return {}

        async def answer_clarification(self, cid, ans):
            pass

    async def bad_tool() -> dict:
        """Returns two handles — should trigger an error."""
        return {"h1": DummyHandle(), "h2": DummyHandle()}

    bad_tool.__name__ = "bad_tool"
    bad_tool.__qualname__ = "bad_tool"

    client = new_llm_client(**llm_config)
    client.set_system_message(
        "Call `bad_tool` with no arguments. Reply with 'done' when finished.",
    )

    outer_handle = start_async_tool_loop(
        client=client,
        message="start",
        tools={"bad_tool": bad_tool},
        max_steps=10,
        timeout=60,
    )

    # The loop should surface the ValueError as a tool error (not crash the loop).
    # The LLM will see the traceback and should still produce a final reply.
    final = await outer_handle.result()
    assert final is not None, "Loop should still complete despite the tool error"

    # Verify the error was surfaced in a tool message
    tool_msgs = [
        m
        for m in client.messages
        if m.get("role") == "tool" and m.get("name") == "bad_tool"
    ]
    assert tool_msgs, "Expected a tool message for bad_tool"
    error_content = tool_msgs[0].get("content", "")
    assert (
        "SteerableToolHandle" in error_content or "ValueError" in error_content
    ), f"Error should mention multiple handles; got: {error_content}"


# ─────────────────────────────────────────────────────────────────────────────
#  Integration test: tuple return with handle
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_composite_tuple_return(llm_config):
    """Tool returns ``(metadata_dict, handle)`` as a tuple."""

    gate = asyncio.Event()

    class TupleHandle(SteerableToolHandle):
        def __init__(self):
            self._done = asyncio.Event()

        async def ask(self, question, **kw):
            return ""

        async def interject(self, message, **kw):
            pass

        def stop(self, reason=None, **kw):
            self._done.set()

        def pause(self, **kw):
            pass

        def resume(self, **kw):
            pass

        def done(self):
            return self._done.is_set()

        async def result(self):
            await gate.wait()
            self._done.set()
            return "tuple-handle-done"

        async def next_clarification(self):
            return {}

        async def next_notification(self):
            return {}

        async def answer_clarification(self, cid, ans):
            pass

    async def tuple_tool() -> tuple:
        """Return a tuple of (metadata, handle)."""
        return ({"ready": True, "item_count": 3}, TupleHandle())

    tuple_tool.__name__ = "tuple_tool"
    tuple_tool.__qualname__ = "tuple_tool"

    client = new_llm_client(**llm_config)
    client.set_system_message(
        "You are running inside an automated test.\n"
        "1️⃣  Call `tuple_tool` with no arguments.\n"
        "2️⃣  Wait for it to complete.\n"
        "3️⃣  Reply with exactly 'done'.",
    )

    outer_handle = start_async_tool_loop(
        client=client,
        message="start",
        tools={"tuple_tool": ToolSpec(fn=tuple_tool, max_total_calls=1)},
        max_steps=20,
        timeout=240,
    )

    await _wait_for_tool_request(client, "tuple_tool")
    await _wait_for_tool_result(client, tool_name="tuple_tool", min_results=1)

    # Verify intermediate content shows the tuple data with sentinel
    tool_msgs = real_tool_messages(client.messages, tool_name="tuple_tool")
    assert tool_msgs, "Expected a tool message for tuple_tool"
    ph_content = tool_msgs[0].get("content", "")
    assert (
        "ready" in ph_content
    ), f"Intermediate data should contain 'ready'; got: {ph_content}"

    # Release gate
    gate.set()

    final = await outer_handle.result()
    assert final is not None, "Loop should complete"

    # Final tool result should contain the handle's result
    tool_msgs_final = real_tool_messages(client.messages, tool_name="tuple_tool")
    final_content = tool_msgs_final[0].get("content", "")
    assert (
        "tuple-handle-done" in final_content
    ), f"Final result should contain handle result; got: {final_content}"
