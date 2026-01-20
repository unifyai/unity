"""
Eval tests for time-awareness in the async tool loop.

These tests verify that the LLM can reason about time based on the
injected time context. They test real LLM behavior with cached responses.

The tests validate:
1. LLM can report the current time
2. LLM knows when the conversation started
3. LLM can report how long a tool took to execute
4. LLM can identify and re-call the faster of two tools
"""

from __future__ import annotations

import asyncio

import pytest

from tests.helpers import _handle_project
from unity.common.llm_client import new_llm_client
from unity.common.async_tool_loop import start_async_tool_loop


# Module-level marker: all tests in this file are eval tests
pytestmark = pytest.mark.eval


# --------------------------------------------------------------------------- #
#  TOOL IMPLEMENTATIONS (Neutral Names - No Speed Hints)                      #
# --------------------------------------------------------------------------- #


async def tool_alpha() -> str:
    """Performs operation alpha."""
    await asyncio.sleep(0.5)  # Slow
    return "alpha_complete"


async def tool_beta() -> str:
    """Performs operation beta."""
    await asyncio.sleep(0.05)  # Fast
    return "beta_complete"


async def timed_operation(duration: float = 0.2) -> str:
    """Performs a timed operation."""
    await asyncio.sleep(duration)
    return f"operation completed after {duration} seconds"


async def get_data() -> str:
    """Retrieves some data."""
    await asyncio.sleep(0.1)
    return "data retrieved successfully"


# --------------------------------------------------------------------------- #
#  HELPER: Count tool calls by name                                           #
# --------------------------------------------------------------------------- #


def count_tool_calls(messages: list, tool_name: str) -> int:
    """Count how many times a tool was called in the message history."""
    count = 0
    for msg in messages:
        if msg.get("role") == "assistant":
            for tc in msg.get("tool_calls") or []:
                fn = tc.get("function", {})
                if fn.get("name") == tool_name:
                    count += 1
    return count


# --------------------------------------------------------------------------- #
#  TEST: LLM is aware of current time                                         #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
@_handle_project
async def test_current_time_awareness(model):
    """Verify the LLM can answer questions about the current time.

    The current time is injected via the time context in the system message.
    The LLM should be able to reference this in its response.
    """
    client = new_llm_client(model=model)

    answer = await start_async_tool_loop(
        client,
        message=(
            "Look at the Time Context section in the system message. "
            "Based on that context, what time is it approximately? "
            "Just give me a brief answer with the time."
        ),
        tools={},  # No tools needed
    ).result()

    # The answer should contain some time-related content
    answer_lower = answer.lower()
    # Check for time indicators (hour, AM/PM, UTC, time references, etc.)
    time_indicators = [
        "am",
        "pm",
        "utc",
        ":",
        "o'clock",
        "noon",
        "midnight",
        "january",  # From the fixed test datetime
        "2026",  # From the fixed test datetime
        "time",
        "context",
        "second",  # e.g., "0 seconds after conversation started"
        "now",  # e.g., "right now"
        "ago",  # e.g., "started X ago"
        "started",
    ]
    has_time = any(ind in answer_lower for ind in time_indicators)
    assert has_time, f"Expected time reference in response, got: {answer}"


# --------------------------------------------------------------------------- #
#  TEST: LLM knows when conversation started                                  #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
@_handle_project
async def test_conversation_start_awareness(model):
    """Verify the LLM knows when the conversation started."""
    client = new_llm_client(model=model)

    answer = await start_async_tool_loop(
        client,
        message=(
            "How long ago did this conversation start? "
            "Just give me a brief answer about the elapsed time."
        ),
        tools={},  # No tools needed
    ).result()

    # The answer should acknowledge the conversation just started
    # or mention some elapsed time
    answer_lower = answer.lower()
    time_indicators = [
        "second",
        "moment",
        "just",
        "ago",
        "started",
        "recently",
        "beginning",
        "0",
    ]
    has_time_ref = any(ind in answer_lower for ind in time_indicators)
    assert has_time_ref, f"Expected time reference in response, got: {answer}"


# --------------------------------------------------------------------------- #
#  TEST: LLM can report tool execution duration                               #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
@_handle_project
async def test_tool_duration_awareness(model):
    """Verify the LLM can report how long a tool took to execute."""
    client = new_llm_client(model=model)

    answer = await start_async_tool_loop(
        client,
        message=(
            "First, call get_data to retrieve some data. "
            "Then tell me how long that tool call took to execute. "
            "Focus on the execution duration in your answer."
        ),
        tools={"get_data": get_data},
    ).result()

    # Verify get_data was called
    call_count = count_tool_calls(client.messages, "get_data")
    assert call_count >= 1, "get_data should have been called"

    # The answer should mention duration/time
    answer_lower = answer.lower()
    duration_indicators = [
        "second",
        "0.",
        "took",
        "duration",
        "completed",
        "ms",
        "millisecond",
    ]
    has_duration = any(ind in answer_lower for ind in duration_indicators)
    assert has_duration, f"Expected duration reference in response, got: {answer}"


# --------------------------------------------------------------------------- #
#  TEST: LLM can identify and re-call the faster tool                         #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
@_handle_project
async def test_faster_tool_identification(model):
    """Verify the LLM can identify which tool was faster and re-call it.

    Uses neutral tool names (tool_alpha, tool_beta) so the LLM must
    infer speed from the execution timing in the Time Context.
    """
    client = new_llm_client(model=model)

    answer = await start_async_tool_loop(
        client,
        message=(
            "Call both tool_alpha and tool_beta (in parallel if you can). "
            "After they complete, look at the execution times in the Time Context "
            "to determine which tool was faster. Then call that faster tool again. "
            "Finally, tell me which tool was faster and why."
        ),
        tools={"tool_alpha": tool_alpha, "tool_beta": tool_beta},
    ).result()

    # Count tool calls
    alpha_count = count_tool_calls(client.messages, "tool_alpha")
    beta_count = count_tool_calls(client.messages, "tool_beta")

    # Both should be called at least once initially
    assert alpha_count >= 1, "tool_alpha should have been called at least once"
    assert beta_count >= 1, "tool_beta should have been called at least once"

    # tool_beta is faster (0.05s vs 0.5s), so it should be called twice
    # (once initially, once as the re-call)
    assert beta_count >= 2, (
        f"tool_beta should have been called twice (it's faster), "
        f"but was called {beta_count} times"
    )

    # The answer should identify beta as faster
    answer_lower = answer.lower()
    assert (
        "beta" in answer_lower
    ), f"Expected 'beta' to be identified as faster: {answer}"


# --------------------------------------------------------------------------- #
#  TEST: LLM can reason about relative tool timing                            #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
@_handle_project
async def test_relative_timing_comparison(model):
    """Verify the LLM can compare execution times of different tools."""
    client = new_llm_client(model=model)

    answer = await start_async_tool_loop(
        client,
        message=(
            "Call tool_alpha first, then call tool_beta. "
            "After both complete, compare their execution times. "
            "Which one took longer and by approximately how much?"
        ),
        tools={"tool_alpha": tool_alpha, "tool_beta": tool_beta},
    ).result()

    # Both tools should be called
    alpha_count = count_tool_calls(client.messages, "tool_alpha")
    beta_count = count_tool_calls(client.messages, "tool_beta")
    assert alpha_count >= 1, "tool_alpha should have been called"
    assert beta_count >= 1, "tool_beta should have been called"

    # The answer should mention alpha took longer (0.5s vs 0.05s)
    answer_lower = answer.lower()
    assert "alpha" in answer_lower, "Expected alpha to be mentioned in comparison"

    # Should mention time comparison
    comparison_indicators = [
        "longer",
        "faster",
        "slower",
        "more",
        "less",
        "second",
        "time",
    ]
    has_comparison = any(ind in answer_lower for ind in comparison_indicators)
    assert has_comparison, f"Expected time comparison in response, got: {answer}"
