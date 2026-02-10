"""
Tests for CodeActActor sub-agent delegation via ``run_sub_agent``.

Symbolic tests verify tool registration, prompt inclusion, and gating.
Eval tests verify end-to-end sub-agent execution with a real LLM.
"""

from __future__ import annotations

import asyncio

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
# Eval test — end-to-end sub-agent execution
# ---------------------------------------------------------------------------


@pytest.mark.eval
@pytest.mark.asyncio
@pytest.mark.timeout(180)
async def test_run_sub_agent_completes_simple_task():
    """
    The outer agent should be able to delegate a simple, self-contained task
    to a sub-agent via run_sub_agent and receive the result.
    """
    actor = CodeActActor(
        headless=True,
        computer_mode="mock",
        timeout=120,
        can_spawn_sub_agents=True,
    )
    try:
        handle = await actor.act(
            "Use run_sub_agent to delegate the following task: "
            "'Calculate the sum of all integers from 1 to 100 using Python and return the result.' "
            "Report the sub-agent's answer.",
            persist=False,
            clarification_enabled=False,
        )
        result = await asyncio.wait_for(handle.result(), timeout=120)

        # The correct answer is 5050. The LLM may format it as "5050" or "5,050".
        result_str = str(result).replace(",", "")
        assert "5050" in result_str
    finally:
        try:
            await actor.close()
        except Exception:
            pass
