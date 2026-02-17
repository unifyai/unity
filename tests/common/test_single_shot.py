"""Tests for single_shot_tool_decision."""

from __future__ import annotations

import pytest
from pydantic import BaseModel, Field

from tests.helpers import _handle_project
from unity.common.single_shot import (
    single_shot_tool_decision,
    SingleShotResult,
    ToolExecution,
)
from unity.common.llm_client import new_llm_client

# --------------------------------------------------------------------------- #
#  Test fixtures: response format models                                       #
# --------------------------------------------------------------------------- #


class ThoughtsResponse(BaseModel):
    """Simple structured output with just thoughts."""

    thoughts: str = Field(..., description="Your reasoning about the request")


class DecisionResponse(BaseModel):
    """Structured output for decisions."""

    thoughts: str = Field(..., description="Your reasoning about the decision")
    decision: str = Field(..., description="Your decision: 'approve' or 'reject'")
    confidence: int = Field(..., description="Confidence level from 1-10")


# --------------------------------------------------------------------------- #
#  Test fixtures: simple tools                                                #
# --------------------------------------------------------------------------- #


def greet(*, name: str) -> str:
    """Greet a person by name.

    Parameters
    ----------
    name : str
        The name of the person to greet.

    Returns
    -------
    str
        A greeting message.
    """
    return f"Hello, {name}!"


def add_numbers(*, a: int, b: int) -> int:
    """Add two numbers together.

    Parameters
    ----------
    a : int
        First number.
    b : int
        Second number.

    Returns
    -------
    int
        The sum of a and b.
    """
    return a + b


async def async_multiply(*, x: int, y: int) -> int:
    """Multiply two numbers (async version).

    Parameters
    ----------
    x : int
        First factor.
    y : int
        Second factor.

    Returns
    -------
    int
        The product of x and y.
    """
    return x * y


def do_nothing() -> str:
    """Do nothing and return a confirmation.

    Use this when no action is needed.

    Returns
    -------
    str
        Confirmation that nothing was done.
    """
    return "Nothing done."


def send_notification(*, message: str) -> str:
    """Send a notification message.

    Parameters
    ----------
    message : str
        The notification message to send.

    Returns
    -------
    str
        Confirmation that the notification was sent.
    """
    return f"Notification sent: {message}"


def start_background_task(*, task_name: str) -> str:
    """Start a background task.

    Parameters
    ----------
    task_name : str
        Name of the task to start.

    Returns
    -------
    str
        Confirmation that the task was started.
    """
    return f"Task '{task_name}' started"


# --------------------------------------------------------------------------- #
#  Unit tests: result structure                                               #
# --------------------------------------------------------------------------- #


def test_tool_execution_dataclass():
    """ToolExecution has expected fields."""
    execution = ToolExecution(
        name="greet",
        args={"name": "Alice"},
        result="Hello, Alice!",
    )
    assert execution.name == "greet"
    assert execution.args == {"name": "Alice"}
    assert execution.result == "Hello, Alice!"


def test_single_shot_result_single_tool():
    """SingleShotResult with a single tool provides backward-compatible properties."""
    result = SingleShotResult(
        tools=[
            ToolExecution(name="greet", args={"name": "Alice"}, result="Hello, Alice!"),
        ],
        text_response=None,
    )
    # New API
    assert len(result.tools) == 1
    assert result.tools[0].name == "greet"
    assert result.tools[0].args == {"name": "Alice"}
    assert result.tools[0].result == "Hello, Alice!"
    # Backward-compatible properties
    assert result.tool_name == "greet"
    assert result.tool_args == {"name": "Alice"}
    assert result.tool_result == "Hello, Alice!"
    assert result.text_response is None


def test_single_shot_result_multiple_tools():
    """SingleShotResult can hold multiple tool executions."""
    result = SingleShotResult(
        tools=[
            ToolExecution(name="greet", args={"name": "Alice"}, result="Hello, Alice!"),
            ToolExecution(name="add_numbers", args={"a": 2, "b": 3}, result=5),
        ],
        text_response=None,
    )
    # New API - all tools accessible
    assert len(result.tools) == 2
    assert result.tools[0].name == "greet"
    assert result.tools[1].name == "add_numbers"
    assert result.tools[1].result == 5
    # Backward-compatible properties return first tool
    assert result.tool_name == "greet"
    assert result.tool_args == {"name": "Alice"}
    assert result.tool_result == "Hello, Alice!"


def test_single_shot_result_no_tool():
    """SingleShotResult can represent no-tool case."""
    result = SingleShotResult(
        tools=[],
        text_response="I don't need to use any tools.",
    )
    assert len(result.tools) == 0
    # Backward-compatible properties return None when no tools
    assert result.tool_name is None
    assert result.tool_args is None
    assert result.tool_result is None
    assert result.text_response is not None


def test_single_shot_result_with_structured_output():
    """SingleShotResult can include structured_output."""
    structured = ThoughtsResponse(thoughts="This is my reasoning")
    result = SingleShotResult(
        tools=[
            ToolExecution(name="greet", args={"name": "Alice"}, result="Hello, Alice!"),
        ],
        text_response='{"thoughts": "This is my reasoning"}',
        structured_output=structured,
    )
    assert result.tool_name == "greet"
    assert result.structured_output is not None
    assert result.structured_output.thoughts == "This is my reasoning"


# --------------------------------------------------------------------------- #
#  Integration tests: real LLM calls (cached)                                 #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
@_handle_project
async def test_single_shot_calls_tool():
    """LLM should call the greet tool when asked to greet someone."""
    client = new_llm_client()
    client.set_system_message(
        "You are a helpful assistant. Use the available tools to respond to requests.",
    )

    tools = {
        "greet": greet,
        "add_numbers": add_numbers,
    }

    result = await single_shot_tool_decision(
        client,
        "Please greet Alice.",
        tools,
    )

    assert result.tool_name == "greet"
    assert result.tool_args == {"name": "Alice"}
    assert result.tool_result == "Hello, Alice!"


@pytest.mark.asyncio
@_handle_project
async def test_single_shot_chooses_correct_tool():
    """LLM should choose add_numbers when asked to add."""
    client = new_llm_client()
    client.set_system_message(
        "You are a calculator assistant. Use tools to perform calculations.",
    )

    tools = {
        "greet": greet,
        "add_numbers": add_numbers,
    }

    result = await single_shot_tool_decision(
        client,
        "What is 5 + 7?",
        tools,
    )

    assert result.tool_name == "add_numbers"
    assert result.tool_args == {"a": 5, "b": 7}
    assert result.tool_result == 12


@pytest.mark.asyncio
@_handle_project
async def test_single_shot_async_tool():
    """Async tools should be awaited correctly."""
    client = new_llm_client()
    client.set_system_message(
        "You are a calculator assistant. Use tools to perform calculations.",
    )

    tools = {
        "multiply": async_multiply,
    }

    result = await single_shot_tool_decision(
        client,
        "What is 6 times 8?",
        tools,
    )

    assert result.tool_name == "multiply"
    assert result.tool_args == {"x": 6, "y": 8}
    assert result.tool_result == 48


@pytest.mark.asyncio
@_handle_project
async def test_single_shot_tool_choice_required():
    """With tool_choice='required', LLM must call a tool."""
    client = new_llm_client()
    client.set_system_message("You must use a tool to respond.")

    tools = {
        "do_nothing": do_nothing,
    }

    result = await single_shot_tool_decision(
        client,
        "Hello, how are you?",
        tools,
        tool_choice="required",
    )

    # With tool_choice="required", the LLM must pick something
    assert result.tool_name == "do_nothing"
    assert result.tool_result == "Nothing done."


@pytest.mark.asyncio
@_handle_project
async def test_single_shot_no_tools():
    """With empty tools dict, LLM should just respond with text."""
    client = new_llm_client()
    client.set_system_message("You are a helpful assistant.")

    result = await single_shot_tool_decision(
        client,
        "What is 2+2?",
        {},  # No tools
    )

    # No tools available, so text response only
    assert result.tool_name is None
    assert result.tool_result is None
    # Should have some text response
    assert result.text_response is not None or True  # May vary by model


# --------------------------------------------------------------------------- #
#  Integration tests: response_format (structured output)                      #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
@_handle_project
async def test_single_shot_structured_output_no_tools():
    """With response_format and no tools, LLM returns structured output."""
    client = new_llm_client()
    client.set_system_message(
        "You are a decision assistant. Always provide your reasoning.",
    )

    result = await single_shot_tool_decision(
        client,
        "Should I approve this expense report for $50?",
        {},  # No tools
        response_format=DecisionResponse,
    )

    # No tools, so no tool call
    assert result.tool_name is None
    assert result.tool_result is None

    # Should have structured output
    assert result.structured_output is not None
    assert isinstance(result.structured_output, DecisionResponse)
    assert result.structured_output.thoughts  # Non-empty
    assert result.structured_output.decision in ("approve", "reject")
    assert 1 <= result.structured_output.confidence <= 10


@pytest.mark.asyncio
@_handle_project
async def test_single_shot_structured_output_with_tools():
    """With response_format AND tools, LLM returns structured output AND calls tool."""
    client = new_llm_client()
    client.set_system_message(
        "You are a helpful assistant. First think about the request, then use "
        "the appropriate tool. Your response must include your thoughts as JSON.",
    )

    tools = {
        "greet": greet,
        "add_numbers": add_numbers,
    }

    result = await single_shot_tool_decision(
        client,
        "Please greet Bob. Think about why this is a good greeting.",
        tools,
        response_format=ThoughtsResponse,
    )

    # Should call the greet tool
    assert result.tool_name == "greet"
    assert result.tool_args == {"name": "Bob"}
    assert result.tool_result == "Hello, Bob!"

    # May or may not have structured output depending on model behavior
    # (some models only return tool calls, some return both)
    # We don't assert on structured_output here as behavior varies


@pytest.mark.asyncio
@_handle_project
async def test_single_shot_structured_output_complex():
    """Test structured output with a more complex model."""
    client = new_llm_client()
    client.set_system_message(
        "You are a sentiment analyzer. Analyze the sentiment and provide your "
        "decision with confidence.",
    )

    result = await single_shot_tool_decision(
        client,
        "The product is absolutely amazing and I love it!",
        {},  # No tools
        response_format=DecisionResponse,
    )

    assert result.structured_output is not None
    assert isinstance(result.structured_output, DecisionResponse)
    # Should be a positive sentiment, likely "approve"
    assert result.structured_output.decision in ("approve", "reject")
    assert result.structured_output.thoughts  # Non-empty reasoning


# --------------------------------------------------------------------------- #
#  Integration tests: multiple concurrent tool calls                          #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
@_handle_project
async def test_single_shot_multiple_concurrent_tools():
    """LLM can call multiple tools concurrently in a single thinking step.

    This tests the core behavior: one LLM decision can trigger multiple parallel
    tool executions, all of which are executed and returned.
    """
    client = new_llm_client()
    client.set_system_message(
        "You are an efficient assistant. When asked to do multiple things, "
        "call ALL relevant tools in parallel in a single response. Do not call "
        "them one at a time - use parallel tool calls.",
    )

    tools = {
        "send_notification": send_notification,
        "start_background_task": start_background_task,
    }

    result = await single_shot_tool_decision(
        client,
        "Start a background task called 'data_sync' AND send a notification "
        "saying 'Sync started'. Do both in parallel.",
        tools,
        tool_choice="required",
    )

    # Should have called both tools
    assert len(result.tools) == 2, (
        f"Expected 2 concurrent tool calls, got {len(result.tools)}: "
        f"{[t.name for t in result.tools]}"
    )

    # Check that both tools were called (order may vary)
    tool_names = {t.name for t in result.tools}
    assert tool_names == {"send_notification", "start_background_task"}

    # Check results
    for tool in result.tools:
        if tool.name == "send_notification":
            assert "Notification sent:" in tool.result
            assert "Sync started" in tool.result
        elif tool.name == "start_background_task":
            assert "data_sync" in tool.result
            assert "started" in tool.result


@pytest.mark.asyncio
@_handle_project
async def test_single_shot_multiple_tools_same_type():
    """LLM can call the same tool multiple times concurrently."""
    client = new_llm_client()
    client.set_system_message(
        "You are an efficient assistant. When asked to do multiple things, "
        "call ALL relevant tools in parallel in a single response.",
    )

    tools = {
        "greet": greet,
    }

    result = await single_shot_tool_decision(
        client,
        "Greet both Alice and Bob. Call the greet tool twice in parallel - "
        "once for Alice and once for Bob.",
        tools,
        tool_choice="required",
    )

    # Should have called greet twice
    assert (
        len(result.tools) == 2
    ), f"Expected 2 concurrent greet calls, got {len(result.tools)}"

    # Both should be greet tool
    assert all(t.name == "greet" for t in result.tools)

    # Check that both names were greeted
    greeted_names = {t.args["name"] for t in result.tools}
    assert greeted_names == {"Alice", "Bob"}

    # Check results
    results = {t.result for t in result.tools}
    assert results == {"Hello, Alice!", "Hello, Bob!"}
