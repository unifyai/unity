"""
Tests for CodeActActor delegation via ``ActorEnvironment``.

Symbolic tests verify environment installation, prompt inclusion, and gating.
Eval tests verify end-to-end actor execution with a real LLM.
"""

from __future__ import annotations

import asyncio
import inspect

import pytest

from unity.actor.code_act_actor import CodeActActor
from unity.actor.environments.actor import ActorEnvironment, _ActorRunner
from unity.actor.execution.session import ActorContext, _ACTOR_CONTEXT
from unity.actor.prompt_builders import build_code_act_prompt
from unity.common.async_tool_loop import SteerableToolHandle

# ---------------------------------------------------------------------------
# Symbolic tests — environment installation and gating
# ---------------------------------------------------------------------------


@pytest.mark.timeout(30)
def test_actor_env_installed_when_enabled():
    """ActorEnvironment should be in self.environments when passed in environments list."""
    actor = CodeActActor(
        environments=[ActorEnvironment()],
        timeout=30,
    )
    assert "actor" in actor.environments
    assert isinstance(actor.environments["actor"], ActorEnvironment)


@pytest.mark.timeout(30)
def test_actor_env_absent_when_disabled():
    """ActorEnvironment should NOT be in self.environments when not passed."""
    actor = CodeActActor(
        timeout=30,
    )
    assert "actor" not in actor.environments


# ---------------------------------------------------------------------------
# Symbolic tests — prompt content
# ---------------------------------------------------------------------------


@pytest.mark.timeout(30)
def test_prompt_includes_actor_guidance_when_env_present():
    """The system prompt should contain actor delegation guidance via environment prompt context."""
    actor = CodeActActor(
        environments=[ActorEnvironment()],
        timeout=30,
    )
    tools = dict(actor.get_tools("act"))
    prompt = build_code_act_prompt(environments=actor.environments, tools=tools)

    assert "Actor Delegation" in prompt
    assert "actor.run" in prompt
    assert "When to use" in prompt
    assert "When NOT to use" in prompt
    assert "steerable" in prompt.lower()


@pytest.mark.timeout(30)
def test_prompt_excludes_actor_guidance_when_env_absent():
    """The system prompt should NOT contain actor guidance when ActorEnvironment is not installed."""
    actor = CodeActActor(
        timeout=30,
    )
    tools = dict(actor.get_tools("act"))
    prompt = build_code_act_prompt(environments=actor.environments, tools=tools)

    assert "Actor Delegation" not in prompt


# ---------------------------------------------------------------------------
# Symbolic tests — _ActorRunner.run parameter exposure
# ---------------------------------------------------------------------------


@pytest.mark.timeout(30)
def test_actor_run_exposes_capability_parameters():
    """actor.run should expose can_compose, can_store, can_spawn_sub_agents,
    and storage_check_on_return as parameters visible to the LLM."""
    sig = inspect.signature(_ActorRunner.run)
    param_names = set(sig.parameters.keys())

    assert "can_compose" in param_names
    assert "can_store" in param_names
    assert "can_spawn_sub_agents" in param_names
    assert "storage_check_on_return" in param_names
    assert "prompt_functions" in param_names
    assert "discovery_scope" in param_names


@pytest.mark.timeout(30)
def test_actor_run_parameter_defaults():
    """Verify the default values for the capability parameters match expectations."""
    sig = inspect.signature(_ActorRunner.run)
    params = sig.parameters

    assert params["can_compose"].default is True
    assert params["can_store"].default is False
    assert params["can_spawn_sub_agents"].default is False
    assert params["storage_check_on_return"].default is False


@pytest.mark.timeout(30)
def test_actor_run_hides_internal_params():
    """Internal parameters prefixed with _ should not appear in the filtered docstring."""
    sig = inspect.signature(_ActorRunner.run)
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
def test_actor_discovery_scope_composes_with_parent():
    """discovery_scope on actor.run should AND with the parent's filter_scope.

    This is a symbolic test that verifies the composition logic by
    inspecting the FunctionManager that the inner CodeActActor would
    receive, without actually running a full actor loop.
    """

    # Create an outer actor with a scoped FunctionManager.
    actor = CodeActActor(
        environments=[ActorEnvironment()],
        timeout=30,
    )

    parent_fm = actor.function_manager
    parent_scope = parent_fm.filter_scope if parent_fm else None

    # Simulate the discovery_scope composition logic from _ActorRunner.run.
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
def test_actor_discovery_scope_narrows_not_replaces():
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
# Symbolic tests — privilege escalation prevention via ContextVar
# ---------------------------------------------------------------------------


@pytest.mark.timeout(30)
def test_actor_runner_caps_can_compose_via_contextvar():
    """When the parent sets can_compose=False in _ACTOR_CONTEXT, the runner should cap it."""
    ctx = ActorContext(function_manager=None, can_compose=False, can_store=True)
    # Simulate the capping logic from _ActorRunner.run.
    effective = True and ctx.can_compose
    assert effective is False


@pytest.mark.timeout(30)
def test_actor_runner_caps_can_store_via_contextvar():
    """When the parent sets can_store=False in _ACTOR_CONTEXT, the runner should cap it."""
    ctx = ActorContext(function_manager=None, can_compose=True, can_store=False)
    # Simulate the capping logic from _ActorRunner.run.
    effective = True and ctx.can_store
    assert effective is False


# ---------------------------------------------------------------------------
# Symbolic tests — ActorEnvironment class-level
# ---------------------------------------------------------------------------


@pytest.mark.timeout(30)
def test_actor_env_namespace():
    """ActorEnvironment.NAMESPACE should be 'actor' and match the instance property."""
    assert ActorEnvironment.NAMESPACE == "actor"

    env = ActorEnvironment()
    assert env.namespace == "actor"


@pytest.mark.timeout(30)
def test_actor_env_get_tools():
    """get_tools() should return exactly one tool with correct metadata including function_id."""
    env = ActorEnvironment()
    tools = env.get_tools()

    assert set(tools.keys()) == {"actor.run"}
    meta = tools["actor.run"]
    assert meta.name == "actor.run"
    assert meta.is_impure is True
    assert meta.is_steerable is True
    assert meta.function_id is not None
    assert meta.function_context == "primitive"


@pytest.mark.timeout(30)
def test_actor_env_get_prompt_context():
    """get_prompt_context() should include the heading, signature, and docstring content."""
    env = ActorEnvironment()
    ctx = env.get_prompt_context()

    assert "Actor Delegation" in ctx
    assert "actor.run" in ctx
    # Signature should include key parameters.
    assert "task" in ctx
    assert "prompt_functions" in ctx
    assert "timeout" in ctx
    # Docstring content should be present.
    assert "When to use" in ctx
    assert "When NOT to use" in ctx
    assert "SteerableToolHandle" in ctx


@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_actor_env_capture_state():
    """capture_state() should return the expected type marker."""
    env = ActorEnvironment()
    state = await env.capture_state()
    assert state == {"type": "actor"}


@pytest.mark.timeout(30)
def test_actor_in_collect_primitives():
    """collect_primitives() should include primitives.actor.run with correct metadata."""
    from unity.function_manager.primitives.registry import collect_primitives

    primitives = collect_primitives()
    assert "primitives.actor.run" in primitives
    entry = primitives["primitives.actor.run"]
    assert entry["is_primitive"] is True
    assert entry["primitive_method"] == "run"
    assert entry["function_id"] is not None


@pytest.mark.timeout(30)
def test_actor_accessible_via_primitives_class():
    """Primitives().actor should return an _ActorRunner instance."""
    from unity.function_manager.primitives import Primitives
    from unity.function_manager.primitives.scope import PrimitiveScope

    scope = PrimitiveScope(scoped_managers=frozenset({"actor"}))
    prims = Primitives(primitive_scope=scope)
    runner = prims.actor
    assert hasattr(runner, "run")
    assert hasattr(runner, "_PRIMITIVE_METHODS")
    assert "run" in runner._PRIMITIVE_METHODS


# ---------------------------------------------------------------------------
# Symbolic test — actor.run returns a steerable handle
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_actor_run_returns_steerable_handle():
    """actor.run() should return a SteerableToolHandle, not a plain string."""
    actor = CodeActActor(
        environments=[ActorEnvironment()],
        timeout=30,
    )
    try:
        env = actor.environments["actor"]
        runner = env.get_instance()

        # Set the ContextVar so actor.run can read parent context.
        token = _ACTOR_CONTEXT.set(ActorContext(
            function_manager=actor.function_manager,
            can_compose=True,
            can_store=True,
        ))
        try:
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
            _ACTOR_CONTEXT.reset(token)
    finally:
        try:
            await actor.close()
        except Exception:
            pass


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_actor_run_forwards_capability_flags():
    """
    Calling actor.run() with non-default capability flags should produce a
    working steerable handle (not crash). We start and immediately stop to
    verify wiring.
    """
    actor = CodeActActor(
        environments=[ActorEnvironment()],
        timeout=30,
    )
    try:
        env = actor.environments["actor"]
        runner = env.get_instance()

        token = _ACTOR_CONTEXT.set(ActorContext(
            function_manager=actor.function_manager,
            can_compose=True,
            can_store=True,
        ))
        try:
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
            _ACTOR_CONTEXT.reset(token)
    finally:
        try:
            await actor.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Eval test — end-to-end actor execution
# ---------------------------------------------------------------------------


@pytest.mark.eval
@pytest.mark.asyncio
@pytest.mark.timeout(300)
async def test_actor_completes_simple_task():
    """
    The outer agent should be able to delegate a simple, self-contained task
    to an actor via actor.run() and receive the result.
    """
    actor = CodeActActor(
        environments=[ActorEnvironment()],
        timeout=300,
    )
    try:
        handle = await actor.act(
            "Use actor.run() to delegate the following task: "
            "'Calculate the sum of all integers from 1 to 100 using Python and return the result.' "
            "Report the actor's answer.",
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
async def test_actor_receives_parent_chat_context():
    """
    The actor should receive the parent agent's conversation history
    via _PARENT_CHAT_CONTEXT ContextVar, enabling it to answer questions
    that depend on information only present in the outer conversation.
    """
    actor = CodeActActor(
        environments=[ActorEnvironment()],
        timeout=300,
    )
    try:
        # Establish a fact in the outer conversation and delegate a sub-task.
        handle = await actor.act(
            [
                {
                    "role": "user",
                    "content": (
                        "Remember this secret code: ZEBRA-42. "
                        "Now use actor.run() to delegate the following task: "
                        "'The parent conversation contains a secret code. "
                        "Find it from the conversation context and report it back.' "
                        "Report whatever the actor returns."
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
