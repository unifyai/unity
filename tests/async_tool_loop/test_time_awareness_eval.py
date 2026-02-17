"""
Eval tests for time-awareness in the async tool loop.

These tests verify that the LLM can reason about time based on the
injected time context. They use structured JSON response schemas
(Pydantic models via ``response_format``) so assertions are on concrete
field values rather than keyword scanning.

The ``now()`` helper is monkey-patched globally by conftest.py to return
a fixed datetime (2025-06-13 12:00:00 UTC). This means:
- The system prompt footer always reads "… June 13, 2025 at 12:00 PM UTC".
- ``elapsed_since_start()`` always returns 0 (same fixed datetime).
- Tool durations use real ``time.perf_counter`` (not patched here).

The tests validate:
1. LLM can report the current time
2. LLM knows when the conversation started
3. LLM can report how long a tool took to execute
4. LLM can identify and re-call the faster of two tools
5. LLM can compare relative tool timing
"""

from __future__ import annotations

import asyncio

import pytest
from pydantic import BaseModel, Field

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
#  TEST: LLM knows when conversation started                                  #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
@_handle_project
async def test_conversation_start_awareness(llm_config):
    """Verify the LLM reports conversation elapsed time from Time Context.

    Because ``now()`` is monkey-patched to always return the same fixed
    datetime, ``elapsed_since_start()`` is always 0.0 s.
    """

    class ElapsedTimeResponse(BaseModel):
        elapsed_seconds: float = Field(
            ...,
            description="Number of seconds elapsed since the conversation started.",
        )

    client = new_llm_client(**llm_config)

    answer = await start_async_tool_loop(
        client,
        message="How long ago did this conversation start, in seconds?",
        tools={},
        response_format=ElapsedTimeResponse,
    ).result()

    assert isinstance(answer, ElapsedTimeResponse)
    # With a fixed now(), elapsed is always 0.
    assert (
        answer.elapsed_seconds <= 1.0
    ), f"Expected ~0 s elapsed (now() is fixed), got {answer.elapsed_seconds}"


# --------------------------------------------------------------------------- #
#  TEST: LLM can report tool execution duration                               #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
@_handle_project
async def test_tool_duration_awareness(llm_config):
    """Verify the LLM can report a tool's execution duration from the
    Tool Execution History table in the Time Context."""

    class ToolDurationResponse(BaseModel):
        tool_name: str = Field(
            ...,
            description="The name of the tool that was called.",
        )
        duration_seconds: float = Field(
            ...,
            description="How long the tool took to execute in seconds.",
        )

    client = new_llm_client(**llm_config)

    answer = await start_async_tool_loop(
        client,
        message=(
            "Call get_data to retrieve some data, then tell me "
            "how long that call took to execute."
        ),
        tools={"get_data": get_data},
        response_format=ToolDurationResponse,
    ).result()

    # Verify get_data was actually called
    call_count = count_tool_calls(client.messages, "get_data")
    assert call_count >= 1, "get_data should have been called"

    assert isinstance(answer, ToolDurationResponse)
    assert answer.tool_name == "get_data"
    # The tool sleeps for 0.1s; allow a generous range for scheduling overhead.
    assert (
        0.05 <= answer.duration_seconds <= 1.0
    ), f"Expected duration ~0.1s, got {answer.duration_seconds}"


# --------------------------------------------------------------------------- #
#  TEST: LLM can identify and re-call the faster tool                         #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
@_handle_project
async def test_faster_tool_identification(llm_config):
    """Verify the LLM can identify which tool was faster and re-call it.

    Uses neutral tool names (tool_alpha, tool_beta) so the LLM must
    infer speed from the execution timing in the Time Context.
    tool_alpha sleeps 0.5 s; tool_beta sleeps 0.05 s.
    """

    class FasterToolResponse(BaseModel):
        faster_tool: str = Field(
            ...,
            description="The name of the tool with the shorter execution time.",
        )
        faster_duration_seconds: float = Field(
            ...,
            description="Execution duration of the faster tool in seconds.",
        )
        slower_tool: str = Field(
            ...,
            description="The name of the tool with the longer execution time.",
        )
        slower_duration_seconds: float = Field(
            ...,
            description="Execution duration of the slower tool in seconds.",
        )

    client = new_llm_client(**llm_config)

    answer = await start_async_tool_loop(
        client,
        message=(
            "Call both tool_alpha and tool_beta (in parallel if you can). "
            "After they complete, determine which tool was faster "
            "and call that faster tool again. "
            "Finally, report which tool was faster and which was slower, "
            "along with their durations."
        ),
        tools={"tool_alpha": tool_alpha, "tool_beta": tool_beta},
        response_format=FasterToolResponse,
    ).result()

    # Count tool calls
    alpha_count = count_tool_calls(client.messages, "tool_alpha")
    beta_count = count_tool_calls(client.messages, "tool_beta")

    # Both should be called at least once initially
    assert alpha_count >= 1, "tool_alpha should have been called at least once"
    assert beta_count >= 1, "tool_beta should have been called at least once"

    # tool_beta is faster (0.05s vs 0.5s), so it should be called twice
    assert beta_count >= 2, (
        f"tool_beta should have been called twice (it's faster), "
        f"but was called {beta_count} times"
    )

    assert isinstance(answer, FasterToolResponse)
    assert answer.faster_tool == "tool_beta"
    assert answer.slower_tool == "tool_alpha"
    assert answer.faster_duration_seconds < answer.slower_duration_seconds
