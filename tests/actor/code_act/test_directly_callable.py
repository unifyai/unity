"""
Tests for the sub-agent environment feature.

Covers:
1. matches_segment — dotted-path segment matching
2. resolve_directly_callable — pattern expansion + error handling
3. StateManagerEnvironment per-method filtering (allowed_methods)
4. _build_sub_agent_environments — environment construction from patterns
"""

import pytest

from unity.actor.environments.base import (
    matches_segment,
    resolve_directly_callable,
)
from unity.actor.environments.state_managers import StateManagerEnvironment
from unity.function_manager.primitives import Primitives, PrimitiveScope
from unity.function_manager.primitives.registry import get_registry

# ────────────────────────────────────────────────────────────────────────────
# 1. matches_segment
# ────────────────────────────────────────────────────────────────────────────


def test_matches_segment_exact():
    """Exact match returns True."""
    assert matches_segment("primitives.contacts.ask", "primitives.contacts.ask")


def test_matches_segment_ancestor_one_level():
    """One-level ancestor matches."""
    assert matches_segment("primitives.contacts", "primitives.contacts.ask")


def test_matches_segment_ancestor_two_levels():
    """Two-level ancestor matches."""
    assert matches_segment("primitives", "primitives.contacts.ask")


def test_matches_segment_partial_segment_no_match():
    """Partial segment (not on a dot boundary) does NOT match."""
    assert not matches_segment("primitives.con", "primitives.contacts.ask")


def test_matches_segment_different_root():
    """Different root does not match."""
    assert not matches_segment("functions", "primitives.contacts.ask")


def test_matches_segment_longer_pattern():
    """Pattern longer than name does not match."""
    assert not matches_segment(
        "primitives.contacts.ask.extra",
        "primitives.contacts.ask",
    )


def test_matches_segment_single_segment():
    """Single-segment names work for both pattern and name."""
    assert matches_segment("alpha", "alpha")
    assert not matches_segment("alpha", "beta")


def test_matches_segment_namespace_matches_children():
    """Namespace-only pattern matches all children."""
    assert matches_segment("functions", "functions.alpha")
    assert matches_segment("functions", "functions.beta")
    assert matches_segment("my_service", "my_service.do_something")


# ────────────────────────────────────────────────────────────────────────────
# 2. resolve_directly_callable
# ────────────────────────────────────────────────────────────────────────────


_ALL_TOOLS = {
    "primitives.contacts.ask",
    "primitives.contacts.update",
    "primitives.tasks.ask",
    "primitives.tasks.update",
    "primitives.tasks.execute",
    "functions.alpha",
    "functions.beta",
    "my_service.do_something",
    "my_service.other_method",
}


def test_resolve_exact_match():
    """Exact name resolves to itself."""
    result = resolve_directly_callable(["functions.alpha"], _ALL_TOOLS)
    assert result == {"functions.alpha"}


def test_resolve_namespace_expands():
    """Namespace pattern expands to all children."""
    result = resolve_directly_callable(["primitives.contacts"], _ALL_TOOLS)
    assert result == {"primitives.contacts.ask", "primitives.contacts.update"}


def test_resolve_top_level_namespace():
    """Top-level namespace expands to everything under it."""
    result = resolve_directly_callable(["primitives"], _ALL_TOOLS)
    assert result == {
        "primitives.contacts.ask",
        "primitives.contacts.update",
        "primitives.tasks.ask",
        "primitives.tasks.update",
        "primitives.tasks.execute",
    }


def test_resolve_multiple_patterns():
    """Multiple patterns are unioned."""
    result = resolve_directly_callable(
        ["primitives.contacts.ask", "functions.alpha"],
        _ALL_TOOLS,
    )
    assert result == {"primitives.contacts.ask", "functions.alpha"}


def test_resolve_mixed_granularity():
    """Mix of exact and namespace patterns works."""
    result = resolve_directly_callable(
        ["primitives.contacts", "functions.alpha"],
        _ALL_TOOLS,
    )
    assert result == {
        "primitives.contacts.ask",
        "primitives.contacts.update",
        "functions.alpha",
    }


def test_resolve_unknown_pattern_raises():
    """Pattern matching zero tools raises ValueError."""
    with pytest.raises(ValueError, match="did not match"):
        resolve_directly_callable(["nonexistent"], _ALL_TOOLS)


def test_resolve_partial_segment_raises():
    """Partial segment pattern raises (not a dotted ancestor)."""
    with pytest.raises(ValueError, match="did not match"):
        resolve_directly_callable(["primitives.con"], _ALL_TOOLS)


def test_resolve_custom_env_namespace():
    """Custom environment namespace resolves to its children."""
    result = resolve_directly_callable(["my_service"], _ALL_TOOLS)
    assert result == {"my_service.do_something", "my_service.other_method"}


# ────────────────────────────────────────────────────────────────────────────
# 3. StateManagerEnvironment per-method filtering
# ────────────────────────────────────────────────────────────────────────────


def test_sme_allowed_methods_filters_get_tools():
    """get_tools() only returns methods in the allowed set."""
    env = StateManagerEnvironment(
        allowed_methods={"primitives.contacts.ask"},
    )
    tools = env.get_tools()

    assert "primitives.contacts.ask" in tools
    assert "primitives.contacts.update" not in tools
    # No other manager methods should appear
    for name in tools:
        assert name == "primitives.contacts.ask", f"Unexpected tool: {name}"


def test_sme_allowed_methods_multiple():
    """Multiple allowed methods from different managers."""
    env = StateManagerEnvironment(
        allowed_methods={
            "primitives.contacts.ask",
            "primitives.tasks.update",
        },
    )
    tools = env.get_tools()

    assert "primitives.contacts.ask" in tools
    assert "primitives.tasks.update" in tools
    assert len(tools) == 2


def test_sme_allowed_methods_preserves_function_id():
    """Filtered tools still have correct function_id and function_context."""
    registry = get_registry()
    env = StateManagerEnvironment(
        allowed_methods={"primitives.contacts.ask"},
    )
    tools = env.get_tools()
    meta = tools["primitives.contacts.ask"]

    assert meta.function_id == registry.get_function_id("contacts", "ask")
    assert meta.function_context == "primitive"


def test_sme_allowed_methods_none_returns_all():
    """allowed_methods=None (default) returns all methods."""
    env_all = StateManagerEnvironment()
    env_filtered = StateManagerEnvironment(
        allowed_methods=None,
    )

    assert env_all.get_tools().keys() == env_filtered.get_tools().keys()


def test_sme_allowed_methods_filters_prompt_context():
    """get_prompt_context() only documents allowed methods."""
    env = StateManagerEnvironment(
        allowed_methods={
            "primitives.contacts.ask",
        },
    )
    context = env.get_prompt_context()

    # contacts.ask should be documented
    assert ".ask" in context
    # contacts.update should NOT be documented (not in allowed set)
    assert ".update" not in context or "primitives.contacts.update" not in context


def test_sme_allowed_methods_prompt_includes_manager_header():
    """Filtered prompt context still includes the manager header."""
    env = StateManagerEnvironment(
        allowed_methods={"primitives.contacts.ask"},
    )
    context = env.get_prompt_context()

    assert "primitives.contacts" in context


def test_sme_allowed_methods_scoped_primitives():
    """allowed_methods works with a scoped Primitives instance."""
    scope = PrimitiveScope(scoped_managers=frozenset({"contacts", "tasks"}))
    prims = Primitives(primitive_scope=scope)
    env = StateManagerEnvironment(
        prims,
        allowed_methods={"primitives.contacts.ask", "primitives.tasks.execute"},
    )
    tools = env.get_tools()

    assert "primitives.contacts.ask" in tools
    assert "primitives.tasks.execute" in tools
    assert len(tools) == 2


# ────────────────────────────────────────────────────────────────────────────
# 4. _build_sub_agent_environments
# ────────────────────────────────────────────────────────────────────────────


def test_build_envs_primitives_only():
    """Primitive patterns produce a scoped StateManagerEnvironment."""
    from unity.actor.code_act_actor import _build_sub_agent_environments

    parent_env = StateManagerEnvironment()
    envs = _build_sub_agent_environments(
        environment=["primitives.contacts.ask"],
        parent_environments={"primitives": parent_env},
        function_manager=None,
    )

    # Should produce exactly one environment (StateManagerEnvironment)
    assert len(envs) == 1
    env = envs[0]
    assert isinstance(env, StateManagerEnvironment)
    tools = env.get_tools()
    assert "primitives.contacts.ask" in tools
    # Only the requested method
    assert all(
        name == "primitives.contacts.ask" for name in tools
    ), f"Unexpected tools: {list(tools.keys())}"


def test_build_envs_local_custom_env():
    """Local custom environment functions are passed through."""
    from unity.actor.code_act_actor import _build_sub_agent_environments
    from unity.actor.environments import create_env

    class MyService:
        async def do_something(self):
            """Does something."""

        async def other_method(self):
            """Other."""

    parent_env = create_env("my_service", MyService())
    envs = _build_sub_agent_environments(
        environment=["my_service.do_something"],
        parent_environments={"my_service": parent_env},
        function_manager=None,
    )

    # Should pass through the parent env
    assert len(envs) == 1
    assert envs[0] is parent_env


def test_build_envs_mixed_sources():
    """Mixed primitive + local patterns produce multiple environments."""
    from unity.actor.code_act_actor import _build_sub_agent_environments
    from unity.actor.environments import create_env

    class MyService:
        async def run(self):
            """Run."""

    sm_env = StateManagerEnvironment()
    svc_env = create_env("my_service", MyService())

    envs = _build_sub_agent_environments(
        environment=["primitives.contacts.ask", "my_service.run"],
        parent_environments={"primitives": sm_env, "my_service": svc_env},
        function_manager=None,
    )

    # Should produce two environments
    assert len(envs) == 2
    types = {type(e).__name__ for e in envs}
    assert "StateManagerEnvironment" in types


def test_build_envs_unknown_pattern_raises():
    """Unknown pattern raises ValueError."""
    from unity.actor.code_act_actor import _build_sub_agent_environments

    with pytest.raises(ValueError, match="did not match"):
        _build_sub_agent_environments(
            environment=["nonexistent"],
            parent_environments={},
            function_manager=None,
        )


def test_build_envs_empty_environment():
    """Empty environment produces no environments (but doesn't raise)."""
    from unity.actor.code_act_actor import _build_sub_agent_environments

    envs = _build_sub_agent_environments(
        environment=[],
        parent_environments={},
        function_manager=None,
    )
    assert envs == []


def test_build_envs_namespace_expansion():
    """Namespace pattern expands and creates correctly scoped environment."""
    from unity.actor.code_act_actor import _build_sub_agent_environments

    parent_env = StateManagerEnvironment()
    envs = _build_sub_agent_environments(
        environment=["primitives.contacts"],
        parent_environments={"primitives": parent_env},
        function_manager=None,
    )

    assert len(envs) == 1
    tools = envs[0].get_tools()
    # Should include all contacts methods
    assert any("contacts.ask" in n for n in tools)
    assert any("contacts.update" in n for n in tools)
    # Should NOT include other managers
    assert not any("tasks" in n for n in tools)
