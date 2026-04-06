"""
Tests for Actor discovering and using DashboardManager primitives.

These tests verify that:
1. DashboardManager is properly registered in Primitives
2. Actor can access DashboardManager methods via primitives.dashboards
3. Sync methods are wrapped to be async (without modifying the singleton)
4. Registry metadata is correct (domain, routing guidance, examples)
"""

from __future__ import annotations

import asyncio

from unity.function_manager.primitives import (
    Primitives,
    _AsyncPrimitiveWrapper,
    _create_async_wrapper,
)

# ────────────────────────────────────────────────────────────────────────────
# Discovery Tests
# ────────────────────────────────────────────────────────────────────────────


def test_dashboard_manager_in_primitives():
    """DashboardManager should be accessible via primitives.dashboards."""
    primitives = Primitives()
    dm = primitives.dashboards
    assert dm is not None
    assert hasattr(dm, "create_tile")
    assert hasattr(dm, "create_dashboard")
    assert hasattr(dm, "get_tile")
    assert hasattr(dm, "get_dashboard")


def test_dashboard_manager_has_expected_tile_methods():
    """DashboardManager should have all expected tile CRUD methods."""
    primitives = Primitives()
    dm = primitives.dashboards

    assert callable(getattr(dm, "create_tile", None))
    assert callable(getattr(dm, "get_tile", None))
    assert callable(getattr(dm, "update_tile", None))
    assert callable(getattr(dm, "delete_tile", None))
    assert callable(getattr(dm, "list_tiles", None))


def test_dashboard_manager_has_expected_dashboard_methods():
    """DashboardManager should have all expected dashboard CRUD methods."""
    primitives = Primitives()
    dm = primitives.dashboards

    assert callable(getattr(dm, "create_dashboard", None))
    assert callable(getattr(dm, "get_dashboard", None))
    assert callable(getattr(dm, "update_dashboard", None))
    assert callable(getattr(dm, "delete_dashboard", None))
    assert callable(getattr(dm, "list_dashboards", None))


def test_dashboard_manager_metadata_registered():
    """DashboardManager metadata should be in ToolSurfaceRegistry."""
    from unity.function_manager.primitives import get_registry

    registry = get_registry()
    spec = registry.get_manager_spec("dashboards")
    assert spec is not None
    assert spec.domain == "Visualizations & Dashboards"
    methods = registry.primitive_methods(manager_alias="dashboards")
    assert "create_tile" in methods
    assert "create_dashboard" in methods
    assert "get_tile" in methods
    assert "list_tiles" in methods


def test_dashboard_manager_in_primitive_registry():
    """DashboardManager should be registered in ToolSurfaceRegistry."""
    from unity.function_manager.primitives import get_registry

    registry = get_registry()
    spec = registry.get_manager_spec("dashboards")
    assert spec is not None
    assert (
        spec.primitive_class_path
        == "unity.dashboard_manager.dashboard_manager.DashboardManager"
    )


def test_dashboard_manager_alias_to_getter():
    """DashboardManager should have entry in _ALIAS_TO_GETTER."""
    from unity.function_manager.primitives.runtime import _ALIAS_TO_GETTER

    assert "dashboards" in _ALIAS_TO_GETTER
    assert _ALIAS_TO_GETTER["dashboards"] == "get_dashboard_manager"


def test_dashboard_manager_in_sync_managers():
    """DashboardManager should be listed in _SYNC_MANAGERS for async wrapping."""
    from unity.function_manager.primitives.runtime import _SYNC_MANAGERS

    assert "dashboards" in _SYNC_MANAGERS


# ────────────────────────────────────────────────────────────────────────────
# Async Patching Tests
# ────────────────────────────────────────────────────────────────────────────


def test_dashboard_manager_sync_methods_patched_to_async():
    """DashboardManager sync methods should be patched to be async via primitives."""
    primitives = Primitives()
    dm = primitives.dashboards

    assert asyncio.iscoroutinefunction(
        dm.create_tile,
    ), "create_tile should be async after patching"
    assert asyncio.iscoroutinefunction(
        dm.create_dashboard,
    ), "create_dashboard should be async after patching"
    assert asyncio.iscoroutinefunction(
        dm.get_tile,
    ), "get_tile should be async after patching"
    assert asyncio.iscoroutinefunction(
        dm.list_tiles,
    ), "list_tiles should be async after patching"
    assert asyncio.iscoroutinefunction(
        dm.delete_tile,
    ), "delete_tile should be async after patching"


def test_async_patching_does_not_modify_original():
    """Creating a wrapper should not modify the original manager singleton."""
    from unity.manager_registry import ManagerRegistry

    dm = ManagerRegistry.get_dashboard_manager()

    wrapper = _create_async_wrapper(dm, "dashboards")
    assert asyncio.iscoroutinefunction(wrapper.create_tile)
    assert not asyncio.iscoroutinefunction(
        dm.create_tile,
    ), "Original sync method unchanged"


def test_primitives_dashboards_returns_async_wrapper():
    """primitives.dashboards should return an async wrapper."""
    primitives = Primitives()
    dm_wrapper = primitives.dashboards

    assert isinstance(dm_wrapper, _AsyncPrimitiveWrapper), "Should return wrapper"
    assert asyncio.iscoroutinefunction(dm_wrapper.create_tile)

    from unity.manager_registry import ManagerRegistry

    dm_original = ManagerRegistry.get_dashboard_manager()
    assert not asyncio.iscoroutinefunction(
        dm_original.create_tile,
    ), "Original unchanged"


# ────────────────────────────────────────────────────────────────────────────
# Routing Guidance & Examples
# ────────────────────────────────────────────────────────────────────────────


def test_dashboard_not_in_data_routing_guidance():
    """Dashboards and data are orthogonal -- no routing guidance between them."""
    from unity.function_manager.primitives import get_registry
    from unity.function_manager.primitives.scope import PrimitiveScope

    registry = get_registry()
    scope = PrimitiveScope(scoped_managers=frozenset({"dashboards", "data"}))
    context = registry.prompt_context(scope)
    assert "primitives.dashboards.*` vs `primitives.data.*" not in context


def test_dashboard_example_generators_registered():
    """Dashboard example generators should be registered in _EXAMPLE_GENERATORS."""
    from unity.function_manager.primitives.registry import _EXAMPLE_GENERATORS

    assert "dashboards" in _EXAMPLE_GENERATORS
    generators = _EXAMPLE_GENERATORS["dashboards"]
    assert "get_primitives_dashboards_baked_in_example" in generators
    assert "get_primitives_dashboards_live_data_example" in generators
    assert "get_primitives_dashboards_composition_example" in generators
