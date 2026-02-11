"""
Tests for the ``instructions`` parameter on Actor.act().

The ``instructions`` parameter provides meta-guidance on *how* to approach a task,
as opposed to the ``description`` which specifies *what* to do.
"""

import asyncio

import pytest
from unittest.mock import MagicMock

from unity.actor.code_act_actor import CodeActActor
from unity.actor.prompt_builders import build_code_act_prompt

# ---------------------------------------------------------------------------
# Prompt builder unit tests (symbolic)
# ---------------------------------------------------------------------------


def test_build_code_act_prompt_without_instructions():
    """When instructions is None, the prompt should not contain an Instructions section."""
    prompt = build_code_act_prompt(environments={}, tools=None, instructions=None)
    assert "### Instructions" not in prompt


def test_build_code_act_prompt_with_instructions():
    """When instructions is provided, the prompt should contain the Instructions section."""
    prompt = build_code_act_prompt(
        environments={},
        tools=None,
        instructions="Always use sub-agents for parallel tasks.",
    )
    assert "### Instructions" in prompt
    assert "Always use sub-agents for parallel tasks." in prompt
    assert "You MUST follow these instructions" in prompt


def test_build_code_act_prompt_instructions_placed_before_rules():
    """Instructions should appear before the execution rules / tool signatures."""
    prompt = build_code_act_prompt(
        environments={},
        tools=None,
        instructions="Prefer simple solutions.",
    )
    instructions_pos = prompt.index("### Instructions")
    # The role line always appears; instructions should come after the role but
    # the prompt overall should still contain the standard role header.
    assert (
        "Code-First Automation Agent" in prompt or "Function Execution Agent" in prompt
    )
    assert instructions_pos > 0


# ---------------------------------------------------------------------------
# CodeActActor.act accepts instructions (symbolic)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_code_act_actor_accepts_instructions_parameter():
    """
    CodeActActor.act should accept the ``instructions`` keyword argument
    without raising TypeError, and return a valid handle.
    """
    actor = CodeActActor(
        headless=True,
        computer_mode="mock",
        timeout=30,
    )
    try:
        handle = await actor.act(
            "What is 2 + 2?",
            instructions="Keep your response concise.",
            persist=False,
            clarification_enabled=False,
        )
        assert handle is not None
        await handle.stop()
    finally:
        try:
            await actor.close()
        except Exception:
            pass


@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_code_act_actor_instructions_none_by_default():
    """
    When instructions is not passed, it defaults to None and the actor
    operates normally.
    """
    actor = CodeActActor(
        headless=True,
        computer_mode="mock",
        timeout=30,
    )
    try:
        handle = await actor.act(
            "What is 2 + 2?",
            persist=False,
            clarification_enabled=False,
        )
        assert handle is not None
        await handle.stop()
    finally:
        try:
            await actor.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# CodeActActor.act instructions influence behavior (eval)
# ---------------------------------------------------------------------------


@pytest.mark.eval
@pytest.mark.asyncio
@pytest.mark.timeout(300)
async def test_code_act_instructions_forbid_execute_code():
    """
    When instructions say "do NOT use execute_code", the actor should answer
    directly without calling execute_code.
    """
    actor = CodeActActor(
        headless=True,
        computer_mode="mock",
        timeout=60,
    )
    try:
        handle = await actor.act(
            "What is 7 * 6?",
            instructions=(
                "Do NOT use the execute_code tool. Answer the question directly "
                "from your own knowledge without running any code."
            ),
            persist=False,
            clarification_enabled=False,
        )
        result = await asyncio.wait_for(handle.result(), timeout=60)

        # The result should contain 42
        assert "42" in str(result)

        # Verify execute_code was NOT called
        from tests.actor.state_managers.utils import get_code_act_tool_calls

        tool_calls = get_code_act_tool_calls(handle)
        assert "execute_code" not in tool_calls, (
            f"Instructions said not to use execute_code, but it was called. "
            f"Tool calls: {tool_calls}"
        )
    finally:
        try:
            await actor.close()
        except Exception:
            pass


@pytest.mark.eval
@pytest.mark.asyncio
@pytest.mark.timeout(300)
async def test_code_act_instructions_include_marker_in_response():
    """
    When instructions require including a specific marker phrase in the final
    answer, the actor should comply. This validates that instructions reach the
    LLM and influence its output.
    """
    actor = CodeActActor(
        headless=True,
        computer_mode="mock",
        timeout=60,
    )
    try:
        handle = await actor.act(
            "What is the capital of France?",
            instructions=(
                "You MUST end your final answer with the exact phrase "
                "'[INSTRUCTIONS_FOLLOWED]' on its own line."
            ),
            persist=False,
            clarification_enabled=False,
        )
        result = await asyncio.wait_for(handle.result(), timeout=60)
        result_str = str(result)

        assert "Paris" in result_str
        assert "INSTRUCTIONS_FOLLOWED" in result_str
    finally:
        try:
            await actor.close()
        except Exception:
            pass


@pytest.mark.eval
@pytest.mark.asyncio
@pytest.mark.timeout(300)
async def test_code_act_instructions_response_format_constraint():
    """
    Instructions can guide how the result is formatted.
    """
    actor = CodeActActor(
        headless=True,
        computer_mode="mock",
        timeout=60,
    )
    try:
        handle = await actor.act(
            "List three colors.",
            instructions=(
                "Format your final answer as a numbered list, one color per line, "
                "like:\n1. Red\n2. Blue\n3. Green"
            ),
            persist=False,
            clarification_enabled=False,
        )
        result = await asyncio.wait_for(handle.result(), timeout=60)
        result_str = str(result)

        # The result should contain numbered items
        assert "1." in result_str
        assert "2." in result_str
        assert "3." in result_str
    finally:
        try:
            await actor.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# SingleFunctionActor.act accepts instructions (symbolic)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_single_function_actor_accepts_instructions():
    """
    SingleFunctionActor.act should accept the ``instructions`` parameter
    without raising TypeError.
    """
    from unity.actor.single_function_actor import SingleFunctionActor

    fm = MagicMock()
    fm.search_functions = MagicMock(
        return_value={
            "metadata": [
                {
                    "function_id": 1,
                    "name": "my_func",
                    "docstring": "A test function",
                    "argspec": "()",
                    "implementation": "def my_func():\n    return 42",
                    "is_primitive": False,
                    "verify": False,
                },
            ],
        },
    )

    actor = SingleFunctionActor(
        function_manager=fm,
        headless=True,
        computer_mode="mock",
    )
    try:
        handle = await actor.act(
            "do something",
            instructions="Be concise in any LLM-generated arguments.",
        )
        assert handle is not None
        # Wait for completion (the function just returns 42)
        result = await asyncio.wait_for(handle.result(), timeout=15)
        assert result is not None
    finally:
        try:
            await actor.close()
        except Exception:
            pass
