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
    assert "actor.act" in prompt
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
# Symbolic tests — _ActorRunner.act parameter exposure
# ---------------------------------------------------------------------------


@pytest.mark.timeout(30)
def test_actor_act_exposes_capability_parameters():
    """actor.act should expose can_compose, can_store, can_spawn_sub_agents,
    and guidelines as parameters visible to the LLM."""
    sig = inspect.signature(_ActorRunner.act)
    param_names = set(sig.parameters.keys())

    assert "can_compose" in param_names
    assert "can_store" in param_names
    assert "can_spawn_sub_agents" in param_names
    assert "prompt_functions" in param_names
    assert "discovery_scope" in param_names
    assert "guidelines" in param_names


@pytest.mark.timeout(30)
def test_actor_act_parameter_defaults():
    """Verify the default values for the capability parameters match expectations."""
    sig = inspect.signature(_ActorRunner.act)
    params = sig.parameters

    assert params["can_compose"].default is True
    assert params["can_store"].default is False
    assert params["can_spawn_sub_agents"].default is False
    assert params["guidelines"].default is None


@pytest.mark.timeout(30)
def test_actor_act_hides_internal_params():
    """Internal parameters prefixed with _ should not appear in the filtered docstring."""
    sig = inspect.signature(_ActorRunner.act)
    public_params = {
        name
        for name, p in sig.parameters.items()
        if not name.startswith("_") and name != "self"
    }

    assert "_clarification_up_q" not in public_params
    assert "_clarification_down_q" not in public_params
    assert "request" in public_params
    assert "timeout" in public_params


@pytest.mark.timeout(30)
def test_actor_discovery_scope_used_directly():
    """discovery_scope on actor.act is used directly as the FM's filter_scope.

    Since _ActorRunner.act() constructs a fresh FunctionManager (no parent
    inheritance), the discovery_scope becomes the filter_scope verbatim.
    """
    from unity.actor.environments.actor import _build_scoped_fm

    fm = _build_scoped_fm("language == 'python'")
    assert fm.filter_scope == "language == 'python'"

    fm_none = _build_scoped_fm(None)
    assert fm_none.filter_scope is None


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

    assert set(tools.keys()) == {"actor.act"}
    meta = tools["actor.act"]
    assert meta.name == "actor.act"
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
    assert "actor.act" in ctx
    # Signature should include key parameters.
    assert "request" in ctx
    assert "prompt_functions" in ctx
    assert "timeout" in ctx
    assert "guidelines" in ctx
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
    """collect_primitives() should include primitives.actor.act with correct metadata."""
    from unity.function_manager.primitives.registry import collect_primitives

    primitives = collect_primitives()
    assert "primitives.actor.act" in primitives
    entry = primitives["primitives.actor.act"]
    assert entry["is_primitive"] is True
    assert entry["primitive_method"] == "act"
    assert entry["function_id"] is not None


@pytest.mark.timeout(30)
def test_actor_accessible_via_primitives_class():
    """Primitives().actor should return an _ActorRunner instance."""
    from unity.function_manager.primitives import Primitives
    from unity.function_manager.primitives.scope import PrimitiveScope

    scope = PrimitiveScope(scoped_managers=frozenset({"actor"}))
    prims = Primitives(primitive_scope=scope)
    runner = prims.actor
    assert hasattr(runner, "act")
    assert hasattr(runner, "_PRIMITIVE_METHODS")
    assert "act" in runner._PRIMITIVE_METHODS


# ---------------------------------------------------------------------------
# Symbolic test — actor.act returns a steerable handle
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_actor_act_returns_steerable_handle():
    """actor.act() should return a SteerableToolHandle, not a plain string."""
    runner = _ActorRunner()
    handle = await runner.act(
        request="What is 1+1?",
        timeout=10,
    )
    try:
        assert isinstance(handle, SteerableToolHandle)
        assert callable(handle.result)
        assert callable(handle.stop)
        assert callable(handle.pause)
        assert callable(handle.resume)
    finally:
        await handle.stop()
        try:
            await handle.result()
        except Exception:
            pass


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_actor_act_forwards_capability_flags():
    """
    Calling actor.act() with non-default capability flags should produce a
    working steerable handle (not crash). We start and immediately stop to
    verify wiring.
    """
    runner = _ActorRunner()
    handle = await runner.act(
        request="What is 1+1?",
        timeout=10,
        can_compose=True,
        can_store=True,
        can_spawn_sub_agents=True,
    )
    await handle.stop()
    try:
        await handle.result()
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
    to an actor via actor.act() and receive the result.
    """
    actor = CodeActActor(
        environments=[ActorEnvironment()],
        timeout=300,
    )
    try:
        handle = await actor.act(
            "Use actor.act() to delegate the following request: "
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
