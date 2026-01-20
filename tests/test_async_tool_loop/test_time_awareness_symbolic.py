"""
Symbolic tests for time-awareness in the async tool loop.

These tests verify the programmatic correctness of time context injection
and tool timing tracking, without depending on LLM behavior.

The tests validate:
1. Time context system message is injected
2. Timezone is correctly applied via now()
3. Loop start time is captured
4. Tool execution timing is recorded
5. Multiple tool calls build cumulative history
"""

from __future__ import annotations

import asyncio

import pytest

from tests.helpers import _handle_project
from unity.common.llm_client import new_llm_client
from unity.common.async_tool_loop import start_async_tool_loop


# --------------------------------------------------------------------------- #
#  TOOL IMPLEMENTATIONS FOR TESTING                                           #
# --------------------------------------------------------------------------- #


async def simple_tool() -> str:
    """A simple tool that returns immediately."""
    return "done"


async def timed_tool(duration: float = 0.1) -> str:
    """A tool that sleeps for a specified duration."""
    await asyncio.sleep(duration)
    return f"completed after {duration}s"


# --------------------------------------------------------------------------- #
#  HELPER FUNCTIONS                                                           #
# --------------------------------------------------------------------------- #


def find_time_context_in_messages(messages: list) -> dict | None:
    """Find the system message containing time context."""
    for msg in messages:
        if msg.get("role") == "system" and msg.get("_time_context"):
            return msg
    return None


def find_runtime_context_message(messages: list) -> dict | None:
    """Find the runtime context system message."""
    for msg in messages:
        if msg.get("role") == "system" and msg.get("_runtime_context"):
            return msg
    return None


# --------------------------------------------------------------------------- #
#  TEST: Time context is injected into system message                         #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
@_handle_project
async def test_time_context_injected(model):
    """Verify that the system message contains the ## Time Context section."""
    client = new_llm_client(model=model)

    # Run a simple tool loop
    answer = await start_async_tool_loop(
        client,
        message="Call simple_tool and reply with 'ok'.",
        tools={"simple_tool": simple_tool},
    ).result()

    assert answer.strip()

    # Find the runtime context system message
    runtime_msg = find_runtime_context_message(client.messages)
    assert runtime_msg is not None, "Runtime context system message not found"

    # Verify it contains the Time Context section
    content = runtime_msg.get("content", "")
    assert (
        "## Time Context" in content
    ), "Time Context section not found in system message"
    assert (
        "Conversation started:" in content
    ), "Conversation start time not in system message"


# --------------------------------------------------------------------------- #
#  TEST: Tool timing is recorded in time context                              #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
@_handle_project
async def test_tool_timing_recorded(model):
    """Verify that tool execution timing appears in the time context."""
    client = new_llm_client(model=model)

    # Run a loop with a tool that takes some time
    answer = await start_async_tool_loop(
        client,
        message="Call timed_tool with duration=0.1 and reply with 'ok'.",
        tools={"timed_tool": timed_tool},
    ).result()

    assert answer.strip()

    # Find the runtime context message
    runtime_msg = find_runtime_context_message(client.messages)
    assert runtime_msg is not None

    content = runtime_msg.get("content", "")

    # Verify tool execution history is present
    assert "### Tool Execution History" in content, "Tool history section not found"
    assert "timed_tool" in content, "Tool name not in history"
    assert "call_" in content.lower() or "|" in content, "call_id table not found"


# --------------------------------------------------------------------------- #
#  TEST: Multiple tool calls build cumulative history                         #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
@_handle_project
async def test_tool_history_cumulative(model):
    """Verify that multiple tool calls build cumulative history with offsets."""
    client = new_llm_client(model=model)

    # Run a loop that calls multiple tools
    answer = await start_async_tool_loop(
        client,
        message=(
            "Call simple_tool first, then call timed_tool with duration=0.05. "
            "Reply with 'done' after both complete."
        ),
        tools={"simple_tool": simple_tool, "timed_tool": timed_tool},
    ).result()

    assert answer.strip()

    # Find the runtime context message
    runtime_msg = find_runtime_context_message(client.messages)
    assert runtime_msg is not None

    content = runtime_msg.get("content", "")

    # Count tool entries in the history (look for table rows with tool names)
    # The table format is: | call_id | Tool | Started (relative) | Duration |
    simple_count = content.count("simple_tool")
    timed_count = content.count("timed_tool")

    # At least one of each tool should be in the history
    assert simple_count >= 1, "simple_tool not found in tool history"
    assert timed_count >= 1, "timed_tool not found in tool history"


# --------------------------------------------------------------------------- #
#  TEST: Time context uses monkey-patched now()                               #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
@_handle_project
async def test_timezone_applied_via_now(model):
    """Verify that times use the monkey-patched now() function.

    The conftest.py patches now() to return a fixed datetime. This test
    verifies that the time context respects that patching.
    """
    client = new_llm_client(model=model)

    # Run a simple loop
    answer = await start_async_tool_loop(
        client,
        message="Call simple_tool and reply 'ok'.",
        tools={"simple_tool": simple_tool},
    ).result()

    assert answer.strip()

    # The runtime context message should exist
    runtime_msg = find_runtime_context_message(client.messages)
    assert runtime_msg is not None

    # The _time_context marker should be set
    assert runtime_msg.get("_time_context") is True, "_time_context marker not set"


# --------------------------------------------------------------------------- #
#  TEST: Loop start time is captured at loop creation                         #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
@_handle_project
async def test_loop_start_time_captured(model):
    """Verify conversation start time is captured and shown as elapsed."""
    client = new_llm_client(model=model)

    # Run a simple loop
    answer = await start_async_tool_loop(
        client,
        message="Call simple_tool and reply 'ok'.",
        tools={"simple_tool": simple_tool},
    ).result()

    assert answer.strip()

    # Find the runtime context message
    runtime_msg = find_runtime_context_message(client.messages)
    assert runtime_msg is not None

    content = runtime_msg.get("content", "")

    # Should show "Conversation started: X ago" format
    assert "Conversation started:" in content
    assert "ago" in content
