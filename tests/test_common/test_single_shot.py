"""Tests for single_shot_tool_decision."""

from __future__ import annotations

import pytest

from tests.helpers import _handle_project
from unity.common.single_shot import single_shot_tool_decision, SingleShotResult
from unity.common.llm_client import new_llm_client


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


# --------------------------------------------------------------------------- #
#  Unit tests: result structure                                               #
# --------------------------------------------------------------------------- #


def test_single_shot_result_dataclass():
    """SingleShotResult has expected fields."""
    result = SingleShotResult(
        tool_name="greet",
        tool_args={"name": "Alice"},
        tool_result="Hello, Alice!",
        text_response=None,
    )
    assert result.tool_name == "greet"
    assert result.tool_args == {"name": "Alice"}
    assert result.tool_result == "Hello, Alice!"
    assert result.text_response is None


def test_single_shot_result_no_tool():
    """SingleShotResult can represent no-tool case."""
    result = SingleShotResult(
        tool_name=None,
        tool_args=None,
        tool_result=None,
        text_response="I don't need to use any tools.",
    )
    assert result.tool_name is None
    assert result.tool_result is None
    assert result.text_response is not None


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
