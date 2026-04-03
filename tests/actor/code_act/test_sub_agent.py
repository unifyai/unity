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
    assert "primitives" in actor.environments
    env = actor.environments["primitives"]
    tools = env.get_tools()
    assert "primitives.actor.act" in tools


@pytest.mark.timeout(30)
def test_actor_env_absent_when_disabled():
    """ActorEnvironment should NOT be in self.environments when not passed."""
    actor = CodeActActor(
        timeout=30,
    )
    primitives_env = actor.environments.get("primitives")
    if primitives_env is not None:
        tools = primitives_env.get_tools()
        assert not any(k.startswith("primitives.actor.") for k in tools)
    else:
        pass  # No primitives environment at all — actor tools are absent.


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
    assert "primitives.actor.act" in prompt
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
    """ActorEnvironment.NAMESPACE should be 'primitives' and match the instance property."""
    assert ActorEnvironment.NAMESPACE == "primitives"

    env = ActorEnvironment()
    assert env.namespace == "primitives"


@pytest.mark.timeout(30)
def test_actor_env_get_tools():
    """get_tools() should return exactly one tool with correct metadata including function_id."""
    env = ActorEnvironment()
    tools = env.get_tools()

    assert set(tools.keys()) == {"primitives.actor.act"}
    meta = tools["primitives.actor.act"]
    assert meta.name == "primitives.actor.act"
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
    assert "primitives.actor.act" in ctx
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
@pytest.mark.llm_call
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
@pytest.mark.llm_call
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
# Symbolic tests — construct_sandbox_root("primitives") round-trip
# ---------------------------------------------------------------------------


@pytest.mark.timeout(30)
def test_construct_sandbox_root_primitives_returns_primitives_with_actor():
    """construct_sandbox_root("primitives") should return a Primitives instance
    whose .actor attribute is an _ActorRunner.

    This is the factory used by _inject_dependencies to satisfy the
    "primitives.actor.act" dependency at runtime when a stored function is
    executed outside of a live CodeActActor sandbox.
    """
    from unity.function_manager.primitives.registry import construct_sandbox_root
    from unity.function_manager.primitives.runtime import Primitives

    root = construct_sandbox_root("primitives")
    assert root is not None
    assert isinstance(root, Primitives)
    assert hasattr(root.actor, "act")
    assert callable(root.actor.act)


@pytest.mark.timeout(30)
def test_construct_sandbox_root_primitives_is_stateless():
    """Each call to construct_sandbox_root("primitives") returns an independent instance.

    Statelessness is load-bearing: stored compositional functions that call
    primitives.actor.act(...) receive a freshly constructed Primitives via
    _inject_dependencies, with no shared state between invocations.
    """
    from unity.function_manager.primitives.registry import construct_sandbox_root
    from unity.function_manager.primitives.runtime import Primitives

    root_a = construct_sandbox_root("primitives")
    root_b = construct_sandbox_root("primitives")
    assert root_a is not root_b
    assert isinstance(root_a, Primitives)
    assert isinstance(root_b, Primitives)


@pytest.mark.timeout(30)
def test_construct_sandbox_root_primitives_has_act_as_primitive_method():
    """The _ActorRunner exposed via construct_sandbox_root("primitives").actor
    should have 'act' in _PRIMITIVE_METHODS so the registry can discover it."""
    from unity.function_manager.primitives.registry import construct_sandbox_root

    root = construct_sandbox_root("primitives")
    runner = root.actor
    assert hasattr(runner, "_PRIMITIVE_METHODS")
    assert "act" in runner._PRIMITIVE_METHODS


@pytest.mark.timeout(30)
def test_construct_sandbox_root_actor_returns_none():
    """construct_sandbox_root("actor") is no longer a valid root and returns None."""
    from unity.function_manager.primitives.registry import construct_sandbox_root

    assert construct_sandbox_root("actor") is None


@pytest.mark.timeout(30)
def test_construct_sandbox_root_unknown_returns_none():
    """construct_sandbox_root with an unknown root name returns None."""
    from unity.function_manager.primitives.registry import construct_sandbox_root

    assert construct_sandbox_root("nonexistent_root") is None


# ---------------------------------------------------------------------------
# Eval test — end-to-end actor execution
# ---------------------------------------------------------------------------


@pytest.mark.eval
@pytest.mark.asyncio
@pytest.mark.llm_call
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
            "Use primitives.actor.act() to delegate the following request: "
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


# ---------------------------------------------------------------------------
# Eval test — stored function with primitives.actor.act dependency re-execution
# ---------------------------------------------------------------------------


@pytest.mark.eval
@pytest.mark.asyncio
@pytest.mark.llm_call
@pytest.mark.timeout(300)
async def test_stored_actor_function_reexecutes_successfully():
    """A stored compositional function that calls primitives.actor.act() works when re-executed.

    This is the end-to-end test for the "store then re-execute" pipeline:

    1. Programmatically store a function that calls primitives.actor.act(...)
    2. Retrieve it and prepare it via _inject_callables_for_functions
       (which calls _inject_dependencies -> construct_sandbox_root("primitives"))
    3. Call the resulting callable with a trivially simple request
    4. Verify the inner actor (freshly constructed _ActorRunner) runs an
       LLM loop to completion and returns a meaningful result

    This exercises the chain:
    stored function -> _inject_dependencies -> construct_sandbox_root("primitives")
    -> Primitives() -> .actor -> _ActorRunner.act() -> inner CodeActActor -> result
    """
    from tests.helpers import _handle_project
    from unity.function_manager.function_manager import FunctionManager
    from unity.function_manager.execution_env import create_base_globals

    @_handle_project
    async def _inner():
        fm = FunctionManager(include_primitives=False)

        source = (
            "async def quick_compute(request: str):\n"
            '    """Delegate a computation to a sub-agent."""\n'
            "    handle = await primitives.actor.act(\n"
            "        request=request,\n"
            "        timeout=60,\n"
            "    )\n"
            "    return await handle.result()\n"
        )

        result = fm.add_functions(implementations=source)
        assert result == {"quick_compute": "added"}

        func_data = fm._get_function_data_by_name(name="quick_compute")
        assert func_data is not None
        assert "primitives.actor.act" in func_data.get("depends_on", [])

        namespace = create_base_globals()
        callables = fm._inject_callables_for_functions(
            [func_data],
            namespace=namespace,
        )

        assert len(callables) == 1
        assert "primitives" in namespace
        assert hasattr(namespace["primitives"], "actor")
        assert isinstance(namespace["primitives"].actor, _ActorRunner)
        assert "quick_compute" in namespace

        fn = namespace["quick_compute"]
        result = await asyncio.wait_for(
            fn("What is 2 + 2? Reply with just the number."),
            timeout=120,
        )
        result_str = str(result)
        assert (
            "4" in result_str
        ), f"Expected '4' in inner actor result, got: {result_str}"

    await _inner()
