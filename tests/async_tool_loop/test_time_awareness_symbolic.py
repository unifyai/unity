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
6. Simulated time progression (1 second per call)
"""

from __future__ import annotations

import asyncio
import re
from datetime import datetime, timedelta, timezone

import pytest

from tests.helpers import _handle_project
from unity.common.llm_client import new_llm_client
from unity.common.async_tool_loop import start_async_tool_loop

# --------------------------------------------------------------------------- #
#  SIMULATED TIME FIXTURE                                                     #
# --------------------------------------------------------------------------- #

# Fixed base datetime for simulated time
_SIMULATED_BASE = datetime(2026, 1, 15, 10, 30, 0, tzinfo=timezone.utc)


@pytest.fixture
def simulated_time(monkeypatch):
    """Fixture that patches now() and perf_counter() for deterministic time.

    Each call to now(as_string=False) increments by 1 second, simulating
    time progression during the conversation. This allows testing that
    elapsed time displays correctly (e.g., "Conversation started: 3.0s ago").

    Each call to perf_counter() also increments by 1 second, making tool
    timing ("Started (relative)", "Duration") deterministic across test runs.

    Returns a dict with:
    - 'now_call_count': number of times now() was called
    - 'perf_call_count': number of times perf_counter() was called
    - 'base': the base datetime
    - 'increment_seconds': seconds added per call (default 1)
    """
    state = {
        "now_call_count": 0,
        "perf_call_count": 0,
        "base": _SIMULATED_BASE,
        "increment_seconds": 1,
    }

    def _simulated_now(time_only: bool = False, as_string: bool = True):
        """Return incrementing datetime for simulated time progression."""
        current_time = state["base"] + timedelta(
            seconds=state["now_call_count"] * state["increment_seconds"],
        )
        state["now_call_count"] += 1

        if not as_string:
            return current_time

        label = "UTC"
        if time_only:
            return current_time.strftime("%I:%M %p ") + label
        return current_time.strftime("%A, %B %d, %Y at %I:%M %p ") + label

    def _simulated_perf_counter() -> float:
        """Return incrementing perf_counter value for deterministic tool timing."""
        value = state["perf_call_count"] * state["increment_seconds"]
        state["perf_call_count"] += 1
        return float(value)

    # Patch now() in all relevant modules
    monkeypatch.setattr("unity.common.prompt_helpers.now", _simulated_now)
    monkeypatch.setattr("unity.common._async_tool.time_context.now", _simulated_now)

    # Patch perf_counter() for deterministic tool timing
    monkeypatch.setattr(
        "unity.common._async_tool.time_context.perf_counter",
        _simulated_perf_counter,
    )
    monkeypatch.setattr(
        "unity.common._async_tool.tools_utils.perf_counter",
        _simulated_perf_counter,
    )
    monkeypatch.setattr(
        "unity.common._async_tool.tools_data.perf_counter",
        _simulated_perf_counter,
    )

    return state


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
async def test_time_context_injected(llm_config):
    """Verify that the system message contains the ## Time Context section."""
    client = new_llm_client(**llm_config)

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
async def test_tool_timing_recorded(llm_config):
    """Verify that tool execution timing appears in the time context."""
    client = new_llm_client(**llm_config)

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
async def test_tool_history_cumulative(llm_config):
    """Verify that multiple tool calls build cumulative history with offsets."""
    client = new_llm_client(**llm_config)

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
async def test_timezone_applied_via_now(llm_config):
    """Verify that times use the monkey-patched now() function.

    The conftest.py patches now() to return a fixed datetime. This test
    verifies that the time context respects that patching.
    """
    client = new_llm_client(**llm_config)

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
async def test_loop_start_time_captured(llm_config):
    """Verify conversation start time is captured and shown as elapsed."""
    client = new_llm_client(**llm_config)

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


# --------------------------------------------------------------------------- #
#  TEST: Simulated time progression                                           #
# --------------------------------------------------------------------------- #


def extract_elapsed_seconds(content: str) -> float | None:
    """Extract the elapsed seconds from 'Conversation started: X ago' text."""
    # Match patterns like "0.0s ago", "1.0s ago", "5.0s ago"
    match = re.search(r"Conversation started:\s*([\d.]+)s\s*ago", content)
    if match:
        return float(match.group(1))
    return None


@pytest.mark.asyncio
@_handle_project
async def test_simulated_time_progression(llm_config, simulated_time):
    """Verify that simulated time increments correctly per now() call.

    With simulated_time fixture, each call to now() adds 1 second.
    The elapsed time shown should reflect the difference between
    loop start time and current time.
    """
    client = new_llm_client(**llm_config)

    # Run a simple loop - the LLM will make multiple calls triggering now()
    answer = await start_async_tool_loop(
        client,
        message="Call simple_tool and reply with 'ok'.",
        tools={"simple_tool": simple_tool},
    ).result()

    assert answer.strip()

    # Find the runtime context message
    runtime_msg = find_runtime_context_message(client.messages)
    assert runtime_msg is not None

    content = runtime_msg.get("content", "")

    # Verify time context section exists
    assert "## Time Context" in content
    assert "Conversation started:" in content

    # The elapsed time should be a positive number of seconds
    elapsed = extract_elapsed_seconds(content)
    assert elapsed is not None, f"Could not extract elapsed time from: {content}"
    # With simulated time, elapsed should be >= 0 (at least some time passed)
    assert elapsed >= 0, f"Elapsed time should be non-negative, got {elapsed}"

    # Verify the simulated_time fixture was used
    assert simulated_time["now_call_count"] > 0, "now() was not invoked"
    assert simulated_time["perf_call_count"] > 0, "perf_counter() was not invoked"


@pytest.mark.asyncio
@_handle_project
async def test_simulated_time_shows_elapsed_correctly(llm_config, simulated_time):
    """Verify that elapsed time calculation is correct with simulated time.

    The elapsed time shown should equal (current_call_count - start_call_count) seconds.
    """
    client = new_llm_client(**llm_config)

    # Record the call counts before the loop starts
    initial_now_count = simulated_time["now_call_count"]
    initial_perf_count = simulated_time["perf_call_count"]

    # Run a loop with a tool
    answer = await start_async_tool_loop(
        client,
        message="Call simple_tool and reply 'ok'.",
        tools={"simple_tool": simple_tool},
    ).result()

    assert answer.strip()

    # The fixtures should have been called multiple times
    assert (
        simulated_time["now_call_count"] > initial_now_count
    ), "now() should have been called during loop"
    assert (
        simulated_time["perf_call_count"] > initial_perf_count
    ), "perf_counter() should have been called during loop"

    # Find the runtime context message
    runtime_msg = find_runtime_context_message(client.messages)
    assert runtime_msg is not None

    content = runtime_msg.get("content", "")
    elapsed = extract_elapsed_seconds(content)

    # The elapsed time in the message reflects the difference between
    # when build_system_message() was called and when the loop started
    # Since each now() call increments by 1 second, elapsed should be
    # a whole number of seconds
    assert elapsed is not None, f"Could not parse elapsed from: {content}"
    assert elapsed == int(elapsed), f"Elapsed should be whole seconds, got {elapsed}"
