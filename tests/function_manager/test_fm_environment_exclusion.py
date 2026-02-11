"""Tests for FunctionManager environment exclusion (function_id masking).

Verifies:
1. ToolMetadata supports function_id + function_context
2. ToolSurfaceRegistry.get_function_id() matches collect_primitives() IDs
3. StateManagerEnvironment populates function_id/function_context for each primitive
4. FunctionManager exclusion filter generation (context-aware: primitive vs compositional)
"""

import pytest
from types import SimpleNamespace

from unity.actor.environments.base import ToolMetadata
from unity.actor.environments.state_managers import StateManagerEnvironment
from unity.function_manager.function_manager import FunctionManager
from unity.function_manager.primitives import Primitives
from unity.function_manager.primitives.scope import PrimitiveScope
from unity.function_manager.primitives.registry import get_registry, _get_stable_id

# ────────────────────────────────────────────────────────────────────────────
# ToolMetadata function_id + function_context fields
# ────────────────────────────────────────────────────────────────────────────


def test_tool_metadata_function_id_defaults_to_none():
    """function_id defaults to None when not specified."""
    meta = ToolMetadata(name="foo", is_impure=True)
    assert meta.function_id is None
    assert meta.function_context is None


def test_tool_metadata_function_id_with_context():
    """function_id and function_context can be set together."""
    meta = ToolMetadata(
        name="foo",
        is_impure=True,
        function_id=42,
        function_context="primitive",
    )
    assert meta.function_id == 42
    assert meta.function_context == "primitive"

    meta2 = ToolMetadata(
        name="bar",
        is_impure=False,
        function_id=7,
        function_context="compositional",
    )
    assert meta2.function_id == 7
    assert meta2.function_context == "compositional"


# ────────────────────────────────────────────────────────────────────────────
# Registry get_function_id
# ────────────────────────────────────────────────────────────────────────────


def test_registry_get_function_id_contacts_ask():
    """get_function_id matches the ID from _get_stable_id for contacts.ask."""
    registry = get_registry()
    fid = registry.get_function_id("contacts", "ask")
    expected = _get_stable_id("ContactManager", "ask")
    assert fid == expected


def test_registry_get_function_id_tasks_execute():
    """get_function_id matches the ID from _get_stable_id for tasks.execute."""
    registry = get_registry()
    fid = registry.get_function_id("tasks", "execute")
    expected = _get_stable_id("TaskScheduler", "execute")
    assert fid == expected


def test_registry_get_function_id_invalid_alias():
    """get_function_id raises ValueError for unknown alias."""
    registry = get_registry()
    with pytest.raises(ValueError, match="Unknown manager alias"):
        registry.get_function_id("nonexistent_manager", "ask")


def test_registry_get_function_id_matches_collect_primitives():
    """get_function_id produces the same IDs as collect_primitives."""
    registry = get_registry()
    scope = PrimitiveScope.all_managers()
    collected = registry.collect_primitives(scope)

    for name, row in collected.items():
        parts = name.split(".")
        assert len(parts) == 3 and parts[0] == "primitives"
        alias, method = parts[1], parts[2]
        fid = registry.get_function_id(alias, method)
        assert fid == row["function_id"], (
            f"get_function_id({alias!r}, {method!r}) = {fid} "
            f"but collect_primitives has {row['function_id']} for {name}"
        )


# ────────────────────────────────────────────────────────────────────────────
# StateManagerEnvironment function_id + function_context population
# ────────────────────────────────────────────────────────────────────────────


def test_state_manager_env_get_tools_has_function_ids():
    """Every tool from StateManagerEnvironment has function_id and function_context."""
    env = StateManagerEnvironment()
    tools = env.get_tools()
    assert len(tools) > 0, "Expected at least some tools"
    for fq_name, meta in tools.items():
        assert meta.function_id is not None, f"Tool {fq_name} should have a function_id"
        assert isinstance(meta.function_id, int)
        assert (
            meta.function_context == "primitive"
        ), f"Tool {fq_name} should have function_context='primitive'"


def test_state_manager_env_function_ids_match_registry():
    """function_ids from get_tools() match registry.get_function_id()."""
    registry = get_registry()
    env = StateManagerEnvironment()
    tools = env.get_tools()

    for fq_name, meta in tools.items():
        parts = fq_name.split(".")
        assert len(parts) == 3 and parts[0] == "primitives"
        alias, method = parts[1], parts[2]
        expected_id = registry.get_function_id(alias, method)
        assert (
            meta.function_id == expected_id
        ), f"Tool {fq_name}: function_id={meta.function_id}, expected {expected_id}"


def test_state_manager_env_scoped_has_function_ids():
    """Scoped StateManagerEnvironment still populates function_ids."""
    scope = PrimitiveScope(scoped_managers=frozenset({"contacts"}))
    primitives = Primitives(primitive_scope=scope)
    env = StateManagerEnvironment(primitives)
    tools = env.get_tools()

    assert len(tools) > 0
    for fq_name, meta in tools.items():
        assert fq_name.startswith("primitives.contacts.")
        assert meta.function_id is not None
        assert meta.function_context == "primitive"


def test_state_manager_env_function_ids_are_unique():
    """All function_ids from get_tools() are unique."""
    env = StateManagerEnvironment()
    tools = env.get_tools()
    ids = [meta.function_id for meta in tools.values()]
    assert len(ids) == len(set(ids)), "function_ids should be unique"


# ────────────────────────────────────────────────────────────────────────────
# FunctionManager context-aware exclusion filters
# ────────────────────────────────────────────────────────────────────────────


def _make_fm_stub(
    *,
    filter_scope=None,
    exclude_primitive_ids=None,
    exclude_compositional_ids=None,
    primitive_scope=None,
):
    """Create a minimal stub that has the real FM methods bound."""
    registry = get_registry()
    ns = SimpleNamespace(
        _filter_scope=filter_scope,
        _exclude_primitive_ids=(
            frozenset(exclude_primitive_ids) if exclude_primitive_ids else None
        ),
        _exclude_compositional_ids=(
            frozenset(exclude_compositional_ids) if exclude_compositional_ids else None
        ),
        _primitive_scope=primitive_scope,
        _registry=registry,
        _build_id_exclusion=FunctionManager._build_id_exclusion,
    )
    ns._scoped_filter = lambda cf: FunctionManager._scoped_filter(ns, cf)
    ns._scoped_primitive_filter = lambda: FunctionManager._scoped_primitive_filter(ns)
    return ns


# ── _build_id_exclusion (static helper) ──────────────────────────────────


def test_build_id_exclusion_none_when_empty():
    """_build_id_exclusion returns None for empty/None sets."""
    assert FunctionManager._build_id_exclusion(None) is None
    assert FunctionManager._build_id_exclusion(frozenset()) is None


def test_build_id_exclusion_single_id():
    """_build_id_exclusion builds correct expression for a single ID."""
    result = FunctionManager._build_id_exclusion(frozenset({42}))
    assert result == "function_id != 42"


def test_build_id_exclusion_multiple_ids_sorted():
    """_build_id_exclusion builds sorted expression for multiple IDs."""
    result = FunctionManager._build_id_exclusion(frozenset({30, 10, 20}))
    assert result == "function_id != 10 and function_id != 20 and function_id != 30"


# ── _scoped_filter (compositional context) ───────────────────────────────


def test_scoped_filter_includes_compositional_exclusion():
    """_scoped_filter applies compositional exclusions to compositional queries."""
    fm = _make_fm_stub(
        filter_scope="language == 'python'",
        exclude_compositional_ids={42},
    )
    result = fm._scoped_filter("name == 'foo'")
    assert "name == 'foo'" in result
    assert "language == 'python'" in result
    assert "function_id != 42" in result


def test_scoped_filter_ignores_primitive_exclusion():
    """_scoped_filter must NOT apply primitive exclusions.

    Primitive function_ids live in a different DB context and could collide
    with compositional auto-incremented IDs.
    """
    fm = _make_fm_stub(
        filter_scope="language == 'python'",
        exclude_primitive_ids={42},
    )
    result = fm._scoped_filter("name == 'foo'")
    assert "function_id" not in result


def test_scoped_filter_filter_scope_only():
    """_scoped_filter works with only filter_scope (no exclusions)."""
    fm = _make_fm_stub(filter_scope="language == 'python'")
    result = fm._scoped_filter(None)
    assert result == "language == 'python'"


def test_scoped_filter_compositional_exclusion_only():
    """_scoped_filter applies compositional exclusion even without filter_scope."""
    fm = _make_fm_stub(exclude_compositional_ids={99})
    result = fm._scoped_filter(None)
    assert result == "function_id != 99"


def test_scoped_filter_all_none_returns_none():
    """_scoped_filter returns None when everything is empty."""
    fm = _make_fm_stub()
    result = fm._scoped_filter(None)
    assert result is None


# ── _scoped_primitive_filter (primitive context) ─────────────────────────


def test_scoped_primitive_filter_with_exclusion():
    """_scoped_primitive_filter combines primitive_row_filter with primitive exclusion."""
    registry = get_registry()
    scope = PrimitiveScope.single("contacts")
    fm = _make_fm_stub(exclude_primitive_ids={42}, primitive_scope=scope)
    result = fm._scoped_primitive_filter()
    base_filter = registry.primitive_row_filter(scope)
    assert base_filter in result
    assert "function_id != 42" in result
    assert " and " in result


def test_scoped_primitive_filter_ignores_compositional_exclusion():
    """_scoped_primitive_filter must NOT apply compositional exclusions."""
    registry = get_registry()
    scope = PrimitiveScope.single("contacts")
    fm = _make_fm_stub(exclude_compositional_ids={42}, primitive_scope=scope)
    result = fm._scoped_primitive_filter()
    expected = registry.primitive_row_filter(scope)
    assert result == expected
    assert "function_id != 42" not in result


def test_scoped_primitive_filter_no_exclusion():
    """_scoped_primitive_filter returns base filter when no exclusions set."""
    registry = get_registry()
    scope = PrimitiveScope.single("contacts")
    fm = _make_fm_stub(primitive_scope=scope)
    result = fm._scoped_primitive_filter()
    expected = registry.primitive_row_filter(scope)
    assert result == expected
