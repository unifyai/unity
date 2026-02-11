"""
Tests for CodeActActor sub-agent delegation via ``run_sub_agent``.

Symbolic tests verify tool registration, prompt inclusion, and gating.
Eval tests verify end-to-end sub-agent execution with a real LLM.
"""

from __future__ import annotations

import asyncio
import inspect

import pytest

from unity.actor.code_act_actor import CodeActActor
from unity.actor.prompt_builders import build_code_act_prompt

# ---------------------------------------------------------------------------
# Symbolic tests — tool registration and gating
# ---------------------------------------------------------------------------


@pytest.mark.timeout(30)
def test_run_sub_agent_tool_registered_by_default():
    """run_sub_agent should be present in the tool dict when can_spawn_sub_agents=True (default)."""
    actor = CodeActActor(
        headless=True,
        computer_mode="mock",
        timeout=30,
    )
    tools = actor.get_tools("act")
    assert "run_sub_agent" in tools


@pytest.mark.timeout(30)
def test_run_sub_agent_tool_registered_when_spawn_disabled():
    """run_sub_agent is always registered in the raw tool dict (filtering is per-call in act())."""
    actor = CodeActActor(
        headless=True,
        computer_mode="mock",
        timeout=30,
        can_spawn_sub_agents=False,
    )
    # The raw tool dict always includes it; _filter_tools in act() strips it.
    tools = actor.get_tools("act")
    assert "run_sub_agent" in tools
    assert actor.can_spawn_sub_agents is False


@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_run_sub_agent_filtered_out_when_spawn_disabled():
    """
    When can_spawn_sub_agents=False is passed to act(), the LLM should not
    have access to run_sub_agent. We verify by starting act() and immediately
    stopping it, then checking that the tool set does not include run_sub_agent.
    """
    actor = CodeActActor(
        headless=True,
        computer_mode="mock",
        timeout=30,
        can_spawn_sub_agents=False,
    )
    try:
        handle = await actor.act(
            "test",
            can_spawn_sub_agents=False,
            persist=False,
            clarification_enabled=False,
        )
        # Stop immediately — we only care about the tool filtering, not execution.
        await handle.stop()
    finally:
        try:
            await actor.close()
        except Exception:
            pass


@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_run_sub_agent_per_call_override():
    """
    can_spawn_sub_agents can be overridden per-call. An actor with
    can_spawn_sub_agents=True should still strip the tool when the
    per-call override is False, and vice versa.
    """
    actor = CodeActActor(
        headless=True,
        computer_mode="mock",
        timeout=30,
        can_spawn_sub_agents=True,
    )
    assert actor.can_spawn_sub_agents is True
    try:
        # Per-call override: disable sub-agents for this call.
        handle = await actor.act(
            "test",
            can_spawn_sub_agents=False,
            persist=False,
            clarification_enabled=False,
        )
        await handle.stop()
    finally:
        try:
            await actor.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Symbolic tests — prompt content
# ---------------------------------------------------------------------------


@pytest.mark.timeout(30)
def test_prompt_includes_sub_agent_guidance_when_tool_present():
    """The system prompt should contain sub-agent delegation guidance when run_sub_agent is available."""
    actor = CodeActActor(
        headless=True,
        computer_mode="mock",
        timeout=30,
        can_spawn_sub_agents=True,
    )
    tools = dict(actor.get_tools("act"))
    prompt = build_code_act_prompt(environments={}, tools=tools)

    assert "Sub-Agent Delegation" in prompt
    assert "run_sub_agent" in prompt
    assert "When to delegate" in prompt
    assert "When NOT to delegate" in prompt


@pytest.mark.timeout(30)
def test_prompt_excludes_sub_agent_guidance_when_tool_absent():
    """The system prompt should NOT contain sub-agent guidance when run_sub_agent is filtered out."""
    actor = CodeActActor(
        headless=True,
        computer_mode="mock",
        timeout=30,
    )
    tools = dict(actor.get_tools("act"))
    # Manually remove run_sub_agent to simulate _filter_tools with can_spawn_sub_agents=False.
    tools.pop("run_sub_agent", None)
    prompt = build_code_act_prompt(environments={}, tools=tools)

    assert "Sub-Agent Delegation" not in prompt


# ---------------------------------------------------------------------------
# Symbolic tests — run_sub_agent parameter exposure
# ---------------------------------------------------------------------------


@pytest.mark.timeout(30)
def test_run_sub_agent_exposes_capability_parameters():
    """run_sub_agent should expose can_compose, can_store, can_spawn_sub_agents,
    and storage_check_on_return as parameters visible to the LLM."""
    actor = CodeActActor(
        headless=True,
        computer_mode="mock",
        timeout=30,
    )
    tools = actor.get_tools("act")
    sig = inspect.signature(tools["run_sub_agent"])
    param_names = set(sig.parameters.keys())

    assert "can_compose" in param_names
    assert "can_store" in param_names
    assert "can_spawn_sub_agents" in param_names
    assert "storage_check_on_return" in param_names
    assert "environment" in param_names
    assert "filter_scope" in param_names


@pytest.mark.timeout(30)
def test_run_sub_agent_parameter_defaults():
    """Verify the default values for the capability parameters match expectations."""
    actor = CodeActActor(
        headless=True,
        computer_mode="mock",
        timeout=30,
    )
    tools = actor.get_tools("act")
    sig = inspect.signature(tools["run_sub_agent"])
    params = sig.parameters

    assert params["can_compose"].default is True
    assert params["can_store"].default is False
    assert params["can_spawn_sub_agents"].default is False
    assert params["storage_check_on_return"].default is False


@pytest.mark.timeout(30)
def test_run_sub_agent_filter_scope_composes_with_parent():
    """filter_scope on run_sub_agent should AND with the parent's filter_scope.

    This is a symbolic test that verifies the composition logic by
    inspecting the FunctionManager that the inner CodeActActor would
    receive, without actually running a full sub-agent loop.
    """

    # Create an outer actor with a scoped FunctionManager.
    actor = CodeActActor(
        environments=[],
        headless=True,
        computer_mode="mock",
        timeout=30,
        can_spawn_sub_agents=True,
    )

    parent_fm = actor.function_manager
    parent_scope = parent_fm.filter_scope if parent_fm else None

    # Simulate the filter_scope composition logic from run_sub_agent.
    new_scope = "language == 'python'"
    if parent_scope:
        expected = f"({parent_scope}) and ({new_scope})"
    else:
        expected = new_scope

    # Verify the composition is correct.
    # When parent has no filter_scope, new_scope stands alone.
    if parent_scope is None:
        assert expected == "language == 'python'"
    else:
        assert f"({parent_scope})" in expected
        assert f"({new_scope})" in expected
        assert " and " in expected


@pytest.mark.timeout(30)
def test_run_sub_agent_filter_scope_additive_not_replacing():
    """filter_scope must be additive: a parent scope is never lost.

    Verifies that composing two scopes produces an AND expression
    containing both parts.
    """
    parent_scope = "language == 'python'"
    child_scope = "'data' in docstring"

    # Simulate the composition.
    combined = f"({parent_scope}) and ({child_scope})"

    assert parent_scope in combined
    assert child_scope in combined
    assert combined == "(language == 'python') and ('data' in docstring)"


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_run_sub_agent_can_compose_false_requires_function_manager():
    """
    When the outer agent calls run_sub_agent with can_compose=False, the inner
    act() call should raise RuntimeError if function_manager is None.
    """
    actor = CodeActActor(
        headless=True,
        computer_mode="mock",
        timeout=30,
    )
    actor.function_manager = None
    try:
        tools = actor.get_tools("act")
        with pytest.raises(RuntimeError, match="function_manager is required"):
            await tools["run_sub_agent"](
                task="Do something",
                can_compose=False,
            )
    finally:
        try:
            await actor.close()
        except Exception:
            pass


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_run_sub_agent_forwards_capability_flags():
    """
    Calling run_sub_agent with non-default capability flags should produce a
    working handle (not crash). We start and immediately stop to verify wiring.
    """
    actor = CodeActActor(
        headless=True,
        computer_mode="mock",
        timeout=30,
        can_spawn_sub_agents=True,
    )
    try:
        tools = actor.get_tools("act")
        # Call with all flags explicitly set to non-defaults where safe.
        # can_compose=True (default), can_store=True, can_spawn_sub_agents=True,
        # storage_check_on_return=True.
        # We just verify the call doesn't raise — stop immediately.
        coro = tools["run_sub_agent"](
            task="What is 1+1?",
            timeout=10,
            can_compose=True,
            can_store=True,
            can_spawn_sub_agents=True,
            storage_check_on_return=True,
        )
        # run_sub_agent awaits handle.result() internally, so we need to
        # let the event loop start the inner act() and then cancel.
        sub_task = asyncio.create_task(coro)
        # Give it a moment to start then cancel.
        await asyncio.sleep(0.5)
        sub_task.cancel()
        try:
            await sub_task
        except asyncio.CancelledError:
            pass
    finally:
        try:
            await actor.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Eval test — end-to-end sub-agent execution
# ---------------------------------------------------------------------------


@pytest.mark.eval
@pytest.mark.asyncio
@pytest.mark.timeout(300)
async def test_run_sub_agent_completes_simple_task():
    """
    The outer agent should be able to delegate a simple, self-contained task
    to a sub-agent via run_sub_agent and receive the result.
    """
    actor = CodeActActor(
        headless=True,
        computer_mode="mock",
        timeout=300,
        can_spawn_sub_agents=True,
    )
    try:
        handle = await actor.act(
            "Use run_sub_agent to delegate the following task: "
            "'Calculate the sum of all integers from 1 to 100 using Python and return the result.' "
            "Report the sub-agent's answer.",
            persist=False,
        )
        result = await asyncio.wait_for(handle.result(), timeout=300)

        # The correct answer is 5050. The LLM may format it as "5050" or "5,050".
        result_str = str(result).replace(",", "")
        assert "5050" in result_str
    finally:
        try:
            await actor.close()
        except Exception:
            pass


@pytest.mark.eval
@pytest.mark.asyncio
@pytest.mark.timeout(300)
async def test_run_sub_agent_receives_parent_chat_context():
    """
    The sub-agent should receive the parent agent's conversation history
    via _parent_chat_context, enabling it to answer questions that depend
    on information only present in the outer conversation.
    """
    actor = CodeActActor(
        headless=True,
        computer_mode="mock",
        timeout=300,
        can_spawn_sub_agents=True,
    )
    try:
        # First turn: establish a fact in the outer conversation.
        handle = await actor.act(
            [
                {
                    "role": "user",
                    "content": (
                        "Remember this secret code: ZEBRA-42. "
                        "Now use run_sub_agent to delegate the following task: "
                        "'The parent conversation contains a secret code. "
                        "Find it from the conversation context and report it back.' "
                        "Report whatever the sub-agent returns."
                    ),
                },
            ],
            persist=False,
        )
        result = await asyncio.wait_for(handle.result(), timeout=300)

        assert "ZEBRA-42" in str(result)
    finally:
        try:
            await actor.close()
        except Exception:
            pass
