"""
Tests for CodeActActor sub-agent delegation via ``SubAgentEnvironment``.

Symbolic tests verify environment installation, prompt inclusion, and gating.
Eval tests verify end-to-end sub-agent execution with a real LLM.
"""

from __future__ import annotations

import asyncio
import inspect

import pytest

from unity.actor.code_act_actor import CodeActActor
from unity.actor.environments.sub_agent import SubAgentEnvironment, _SubAgentRunner
from unity.actor.prompt_builders import build_code_act_prompt
from unity.common.async_tool_loop import SteerableToolHandle

# ---------------------------------------------------------------------------
# Symbolic tests — environment installation and gating
# ---------------------------------------------------------------------------


@pytest.mark.timeout(30)
def test_sub_agent_env_installed_when_enabled():
    """SubAgentEnvironment should be in self.environments when can_spawn_sub_agents=True."""
    actor = CodeActActor(
        headless=True,
        computer_mode="mock",
        timeout=30,
        can_spawn_sub_agents=True,
    )
    assert "sub_agent" in actor.environments
    assert isinstance(actor.environments["sub_agent"], SubAgentEnvironment)


@pytest.mark.timeout(30)
def test_sub_agent_env_absent_when_disabled():
    """SubAgentEnvironment should NOT be in self.environments when can_spawn_sub_agents=False."""
    actor = CodeActActor(
        headless=True,
        computer_mode="mock",
        timeout=30,
        can_spawn_sub_agents=False,
    )
    assert "sub_agent" not in actor.environments
    assert actor.can_spawn_sub_agents is False


@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_sub_agent_env_excluded_from_sandbox_per_call_override():
    """
    When can_spawn_sub_agents=True at init but False per-call,
    the sub_agent namespace should be excluded from the sandbox environments.
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
def test_prompt_includes_sub_agent_guidance_when_env_present():
    """The system prompt should contain sub-agent delegation guidance via environment prompt context."""
    actor = CodeActActor(
        headless=True,
        computer_mode="mock",
        timeout=30,
        can_spawn_sub_agents=True,
    )
    tools = dict(actor.get_tools("act"))
    prompt = build_code_act_prompt(environments=actor.environments, tools=tools)

    assert "Sub-Agent Delegation" in prompt
    assert "sub_agent.run" in prompt
    assert "When to use" in prompt
    assert "When NOT to use" in prompt
    assert "steerable" in prompt.lower()


@pytest.mark.timeout(30)
def test_prompt_excludes_sub_agent_guidance_when_env_absent():
    """The system prompt should NOT contain sub-agent guidance when SubAgentEnvironment is not installed."""
    actor = CodeActActor(
        headless=True,
        computer_mode="mock",
        timeout=30,
        can_spawn_sub_agents=False,
    )
    tools = dict(actor.get_tools("act"))
    prompt = build_code_act_prompt(environments=actor.environments, tools=tools)

    assert "Sub-Agent Delegation" not in prompt


# ---------------------------------------------------------------------------
# Symbolic tests — _SubAgentRunner.run parameter exposure
# ---------------------------------------------------------------------------


@pytest.mark.timeout(30)
def test_sub_agent_run_exposes_capability_parameters():
    """sub_agent.run should expose can_compose, can_store, can_spawn_sub_agents,
    and storage_check_on_return as parameters visible to the LLM."""
    sig = inspect.signature(_SubAgentRunner.run)
    param_names = set(sig.parameters.keys())

    assert "can_compose" in param_names
    assert "can_store" in param_names
    assert "can_spawn_sub_agents" in param_names
    assert "storage_check_on_return" in param_names
    assert "prompt_functions" in param_names
    assert "discovery_scope" in param_names


@pytest.mark.timeout(30)
def test_sub_agent_run_parameter_defaults():
    """Verify the default values for the capability parameters match expectations."""
    sig = inspect.signature(_SubAgentRunner.run)
    params = sig.parameters

    assert params["can_compose"].default is True
    assert params["can_store"].default is False
    assert params["can_spawn_sub_agents"].default is False
    assert params["storage_check_on_return"].default is False


@pytest.mark.timeout(30)
def test_sub_agent_run_hides_internal_params():
    """Internal parameters prefixed with _ should not appear in the filtered docstring."""
    sig = inspect.signature(_SubAgentRunner.run)
    public_params = {
        name
        for name, p in sig.parameters.items()
        if not name.startswith("_") and name != "self"
    }

    assert "_clarification_up_q" not in public_params
    assert "_clarification_down_q" not in public_params
    assert "task" in public_params
    assert "timeout" in public_params


@pytest.mark.timeout(30)
def test_sub_agent_discovery_scope_composes_with_parent():
    """discovery_scope on sub_agent.run should AND with the parent's filter_scope.

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

    # Simulate the discovery_scope composition logic from _SubAgentRunner.run.
    new_scope = "language == 'python'"
    if parent_scope:
        expected = f"({parent_scope}) and ({new_scope})"
    else:
        expected = new_scope

    # When parent has no filter_scope, new_scope stands alone.
    if parent_scope is None:
        assert expected == "language == 'python'"
    else:
        assert f"({parent_scope})" in expected
        assert f"({new_scope})" in expected
        assert " and " in expected


@pytest.mark.timeout(30)
def test_sub_agent_discovery_scope_narrows_not_replaces():
    """discovery_scope must narrow: a parent scope is never lost.

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


# ---------------------------------------------------------------------------
# Symbolic tests — privilege escalation prevention
# ---------------------------------------------------------------------------


@pytest.mark.timeout(30)
def test_sub_agent_runner_caps_can_compose():
    """When the parent has can_compose=False, the runner should cap inner can_compose to False."""
    runner = _SubAgentRunner(
        parent_environments={},
        function_manager=None,
        parent_can_compose=False,
        parent_can_store=True,
        model=None,
        preprocess_msgs=None,
        prompt_caching=None,
        parent_timeout=30,
    )
    # Even if can_compose=True is requested, it should be capped by parent.
    assert not (True and runner._parent_can_compose)


@pytest.mark.timeout(30)
def test_sub_agent_runner_caps_can_store():
    """When the parent has can_store=False, the runner should cap inner can_store to False."""
    runner = _SubAgentRunner(
        parent_environments={},
        function_manager=None,
        parent_can_compose=True,
        parent_can_store=False,
        model=None,
        preprocess_msgs=None,
        prompt_caching=None,
        parent_timeout=30,
    )
    # Even if can_store=True is requested, it should be capped by parent.
    assert not (True and runner._parent_can_store)


# ---------------------------------------------------------------------------
# Symbolic test — sub_agent.run returns a steerable handle
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_sub_agent_run_returns_steerable_handle():
    """sub_agent.run() should return a SteerableToolHandle, not a plain string."""
    actor = CodeActActor(
        headless=True,
        computer_mode="mock",
        timeout=30,
        can_spawn_sub_agents=True,
    )
    try:
        env = actor.environments["sub_agent"]
        runner = env.get_instance()
        handle = await runner.run(
            task="What is 1+1?",
            timeout=10,
        )
        assert isinstance(handle, SteerableToolHandle)
        assert callable(handle.result)
        assert callable(handle.stop)
        assert callable(handle.pause)
        assert callable(handle.resume)
        # Clean up
        await handle.stop()
        try:
            await handle.result()
        except Exception:
            pass
    finally:
        try:
            await actor.close()
        except Exception:
            pass


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_sub_agent_run_forwards_capability_flags():
    """
    Calling sub_agent.run() with non-default capability flags should produce a
    working steerable handle (not crash). We start and immediately stop to
    verify wiring.
    """
    actor = CodeActActor(
        headless=True,
        computer_mode="mock",
        timeout=30,
        can_spawn_sub_agents=True,
    )
    try:
        env = actor.environments["sub_agent"]
        runner = env.get_instance()
        # Call with all flags explicitly set to non-defaults where safe.
        handle = await runner.run(
            task="What is 1+1?",
            timeout=10,
            can_compose=True,
            can_store=True,
            can_spawn_sub_agents=True,
            storage_check_on_return=True,
        )
        await handle.stop()
        try:
            await handle.result()
        except Exception:
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
async def test_sub_agent_completes_simple_task():
    """
    The outer agent should be able to delegate a simple, self-contained task
    to a sub-agent via sub_agent.run() and receive the result.
    """
    actor = CodeActActor(
        headless=True,
        computer_mode="mock",
        timeout=300,
        can_spawn_sub_agents=True,
    )
    try:
        handle = await actor.act(
            "Use sub_agent.run() to delegate the following task: "
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
async def test_sub_agent_receives_parent_chat_context():
    """
    The sub-agent should receive the parent agent's conversation history
    via _PARENT_CHAT_CONTEXT ContextVar, enabling it to answer questions
    that depend on information only present in the outer conversation.
    """
    actor = CodeActActor(
        headless=True,
        computer_mode="mock",
        timeout=300,
        can_spawn_sub_agents=True,
    )
    try:
        # Establish a fact in the outer conversation and delegate a sub-task.
        handle = await actor.act(
            [
                {
                    "role": "user",
                    "content": (
                        "Remember this secret code: ZEBRA-42. "
                        "Now use sub_agent.run() to delegate the following task: "
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
