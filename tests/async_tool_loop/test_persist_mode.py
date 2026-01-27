"""
Tests for the `persist=True` mode in the async tool loop.

When `persist=True`, the loop does not terminate when the LLM produces content
without tool calls. Instead, it blocks waiting for the next interjection. This
enables a single persistent loop that can process multiple events over time.
"""

from __future__ import annotations

import asyncio

import pytest
from tests.helpers import _handle_project
from unity.common.llm_client import new_llm_client
from tests.async_helpers import (
    _wait_for_condition,
    _wait_for_tool_request,
)

from unity.common.async_tool_loop import start_async_tool_loop


# --------------------------------------------------------------------------- #
#  TOOL IMPLEMENTATIONS                                                       #
# --------------------------------------------------------------------------- #
async def echo(text: str) -> str:
    """Echo back the input text."""
    await asyncio.sleep(0.05)
    return f"echoed: {text}"


def add(x: int, y: int) -> int:
    """Add two numbers."""
    return x + y


# --------------------------------------------------------------------------- #
#  BASIC PERSIST MODE                                                         #
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
@_handle_project
async def test_persist_mode_waits_for_interjection(model):
    """
    In persist mode, after the LLM produces content without tool calls,
    the loop blocks waiting for an interjection instead of returning.
    """
    client = new_llm_client(model=model)

    handle = start_async_tool_loop(
        client,
        message="Say hello and nothing else.",
        tools={},
        persist=True,
        timeout=60,
    )

    # Wait for the LLM to produce initial content
    async def _has_assistant_response() -> bool:
        return any(
            m.get("role") == "assistant" and m.get("content")
            for m in (client.messages or [])
        )

    await _wait_for_condition(_has_assistant_response, poll=0.05, timeout=30.0)

    # The loop should NOT be done - it should be waiting for an interjection
    await asyncio.sleep(0.2)  # Small delay to ensure loop has reached wait state
    assert not handle.done(), "Persist loop should not terminate after first response"

    # Now interject to give it something to process
    await handle.interject("Now say goodbye.")

    # Wait for a second assistant response
    async def _has_two_responses() -> bool:
        assistant_msgs = [
            m
            for m in (client.messages or [])
            if m.get("role") == "assistant" and m.get("content")
        ]
        return len(assistant_msgs) >= 2

    await _wait_for_condition(_has_two_responses, poll=0.05, timeout=30.0)

    # Still should not be done - waiting for more
    await asyncio.sleep(0.2)
    assert not handle.done(), "Persist loop should continue waiting after interjection"

    # Stop the loop explicitly
    handle.stop()
    result = await handle.result()
    assert result == "processed stopped early, no result"


@pytest.mark.asyncio
@_handle_project
async def test_persist_mode_processes_multiple_interjections(model):
    """
    The persist loop should process multiple interjections sequentially.
    """
    client = new_llm_client(model=model)

    handle = start_async_tool_loop(
        client,
        message="Reply with 'ready' and wait for my instructions.",
        tools={"add": add},
        persist=True,
        timeout=60,
    )

    # Wait for initial response
    async def _has_response() -> bool:
        return any(
            m.get("role") == "assistant" and m.get("content")
            for m in (client.messages or [])
        )

    await _wait_for_condition(_has_response, poll=0.05, timeout=30.0)

    # First interjection - ask it to add
    await handle.interject(
        "Use the add tool to compute 2 + 3, then tell me the result.",
    )

    # Wait for the tool to be called and response generated
    await _wait_for_tool_request(client, "add")

    async def _has_tool_result() -> bool:
        return any(m.get("role") == "tool" for m in (client.messages or []))

    await _wait_for_condition(_has_tool_result, poll=0.05, timeout=30.0)

    # Wait for assistant to process tool result
    async def _has_result_response() -> bool:
        msgs = client.messages or []
        tool_idx = next(
            (i for i, m in enumerate(msgs) if m.get("role") == "tool"),
            -1,
        )
        if tool_idx < 0:
            return False
        return any(
            m.get("role") == "assistant" and m.get("content")
            for m in msgs[tool_idx + 1 :]
        )

    await _wait_for_condition(_has_result_response, poll=0.05, timeout=30.0)

    # Verify the result is in the response
    assistant_msgs = [
        m.get("content", "")
        for m in (client.messages or [])
        if m.get("role") == "assistant" and m.get("content")
    ]
    assert any("5" in msg for msg in assistant_msgs), "Should have computed 2+3=5"

    # Second interjection
    await handle.interject("Now add 10 and 20.")

    # Wait for second tool call
    async def _has_second_tool_call() -> bool:
        tool_calls = [
            m
            for m in (client.messages or [])
            if m.get("role") == "tool" and m.get("name") == "add"
        ]
        return len(tool_calls) >= 2

    await _wait_for_condition(_has_second_tool_call, poll=0.05, timeout=30.0)

    # Stop and verify
    handle.stop()
    await handle.result()

    # Verify both additions were performed
    tool_msgs = [m for m in (client.messages or []) if m.get("role") == "tool"]
    assert len(tool_msgs) >= 2, "Should have made at least 2 tool calls"


@pytest.mark.asyncio
@_handle_project
async def test_persist_mode_terminates_on_stop(model):
    """
    A persist loop should terminate gracefully when handle.stop() is called.
    """
    client = new_llm_client(model=model)

    handle = start_async_tool_loop(
        client,
        message="Say 'waiting' and wait.",
        tools={},
        persist=True,
        timeout=60,
    )

    # Wait for initial response
    async def _has_response() -> bool:
        return any(m.get("role") == "assistant" for m in (client.messages or []))

    await _wait_for_condition(_has_response, poll=0.05, timeout=30.0)

    # Stop the loop
    handle.stop()

    # Should terminate with the standard stop message
    result = await handle.result()
    assert result == "processed stopped early, no result"
    assert handle.done()


# --------------------------------------------------------------------------- #
#  NON-PERSIST MODE REGRESSION TEST                                           #
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
@_handle_project
async def test_non_persist_mode_terminates_normally(model):
    """
    Without persist=True, the loop should terminate as normal when the LLM
    produces content without tool calls (regression test).
    """
    client = new_llm_client(model=model)

    handle = start_async_tool_loop(
        client,
        message="Say hello and nothing else.",
        tools={},
        persist=False,  # Explicitly false (the default)
        timeout=60,
    )

    # The loop should terminate with the assistant's response
    result = await handle.result()

    assert result.strip(), "Should have a non-empty response"
    assert handle.done(), "Loop should be done"

    # Verify only one assistant message (no waiting for interjections)
    assistant_msgs = [
        m for m in (client.messages or []) if m.get("role") == "assistant"
    ]
    assert len(assistant_msgs) == 1, "Should have exactly one assistant response"


@pytest.mark.asyncio
@_handle_project
async def test_persist_mode_with_tool_calls(model):
    """
    Persist mode should handle tool calls normally and only wait after
    the LLM produces content without tool calls.
    """
    client = new_llm_client(model=model)

    handle = start_async_tool_loop(
        client,
        message="Use the add tool to compute 1 + 1, then tell me the result.",
        tools={"add": add},
        persist=True,
        timeout=60,
    )

    # Wait for the tool to be called
    await _wait_for_tool_request(client, "add")

    # Wait for the assistant to respond with the result
    async def _has_result_response() -> bool:
        msgs = client.messages or []
        tool_idx = next(
            (i for i, m in enumerate(msgs) if m.get("role") == "tool"),
            -1,
        )
        if tool_idx < 0:
            return False
        return any(
            m.get("role") == "assistant" and m.get("content")
            for m in msgs[tool_idx + 1 :]
        )

    await _wait_for_condition(_has_result_response, poll=0.05, timeout=30.0)

    # Verify the result contains "2"
    assistant_msgs = [
        m.get("content", "")
        for m in (client.messages or [])
        if m.get("role") == "assistant" and m.get("content")
    ]
    assert any("2" in msg for msg in assistant_msgs), "Should have computed 1+1=2"

    # Loop should still be alive, waiting for next interjection
    await asyncio.sleep(0.2)
    assert not handle.done(), "Persist loop should wait after tool call completes"

    # Clean up
    handle.stop()
    await handle.result()
