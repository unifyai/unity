"""Tests for DashboardManager registration in ManagerRegistry and primitives."""

from unity.dashboard_manager.base import BaseDashboardManager
from unity.function_manager.primitives.registry import (
    _EXAMPLE_GENERATORS,
    _MANAGER_SPECS,
    get_registry,
)
from unity.function_manager.primitives.scope import (
    VALID_MANAGER_ALIASES,
    PrimitiveScope,
)


class TestManagerRegistration:
    def test_dashboard_manager_registered(self):
        from unity.manager_registry import ManagerRegistry

        dm = ManagerRegistry.get_dashboard_manager()
        assert isinstance(dm, BaseDashboardManager)

    def test_simulated_dashboard_manager(self):
        from unity.dashboard_manager.simulated import SimulatedDashboardManager

        dm = SimulatedDashboardManager()
        assert isinstance(dm, BaseDashboardManager)
        assert isinstance(dm, SimulatedDashboardManager)


class TestPrimitivesRegistry:
    def test_dashboards_in_manager_specs(self):
        aliases = {s.manager_alias for s in _MANAGER_SPECS}
        assert "dashboards" in aliases

    def test_dashboards_in_valid_manager_aliases(self):
        assert "dashboards" in VALID_MANAGER_ALIASES

    def test_dashboards_in_example_generators(self):
        assert "dashboards" in _EXAMPLE_GENERATORS

    def test_dashboard_example_generator_names(self):
        names = _EXAMPLE_GENERATORS["dashboards"]
        assert "get_primitives_dashboards_baked_in_example" in names
        assert "get_primitives_dashboards_live_data_example" in names
        assert "get_primitives_dashboards_composition_example" in names

    def test_dashboards_in_alias_to_getter(self):
        from unity.function_manager.primitives.runtime import _ALIAS_TO_GETTER

        assert "dashboards" in _ALIAS_TO_GETTER

    def test_dashboards_in_sync_managers(self):
        from unity.function_manager.primitives.runtime import _SYNC_MANAGERS

        assert "dashboards" in _SYNC_MANAGERS


class TestPrimitivesDiscovery:
    """Test that DashboardManager primitives are correctly discovered."""

    def test_dashboard_methods_discovered(self):
        registry = get_registry()
        methods = registry.primitive_methods(manager_alias="dashboards")
        expected = {
            "create_dashboard",
            "create_tile",
            "delete_dashboard",
            "delete_tile",
            "get_dashboard",
            "get_tile",
            "list_dashboards",
            "list_tiles",
            "update_dashboard",
            "update_tile",
        }
        assert set(methods) == expected
        assert "set_tile_data_scope" not in methods

    def test_dashboard_manager_spec_metadata(self):
        spec = get_registry().get_manager_spec("dashboards")
        assert spec is not None
        assert spec.domain == "Visualizations & Dashboards"
        assert "DashboardManager" in spec.primitive_class_path

    def test_dashboard_primitives_collected(self):
        scope = PrimitiveScope.single("dashboards")
        primitives = get_registry().collect_primitives(scope)
        assert len(primitives) > 0
        assert "primitives.dashboards.create_tile" in primitives
        assert "primitives.dashboards.create_dashboard" in primitives

    def test_dashboard_methods_have_docstrings(self):
        registry = get_registry()
        methods = registry.primitive_methods(manager_alias="dashboards")
        scope = PrimitiveScope.single("dashboards")
        primitives = registry.collect_primitives(scope)
        for method in methods:
            key = f"primitives.dashboards.{method}"
            assert key in primitives, f"Missing primitive: {key}"
            doc = primitives[key]["docstring"]
            assert doc, f"Empty docstring for {key}"
            assert len(doc) > 100, f"Docstring for {key} too short ({len(doc)} chars)"

    def test_dashboard_write_docstrings_explain_destination_and_data_scope(self):
        scope = PrimitiveScope.single("dashboards")
        primitives = get_registry().collect_primitives(scope)

        write_methods = (
            "create_tile",
            "update_tile",
            "delete_tile",
            "create_dashboard",
            "update_dashboard",
            "delete_dashboard",
        )
        for method in write_methods:
            doc = primitives[f"primitives.dashboards.{method}"]["docstring"]
            assert "destination" in doc
            assert "Accessible shared spaces" in doc

        for method in ("create_tile", "update_tile"):
            doc = primitives[f"primitives.dashboards.{method}"]["docstring"]
            assert "data_scope" in doc
            assert "dashboard" in doc
            assert "space:<id>" in doc

    def test_dashboard_prompt_context_generated(self):
        scope = PrimitiveScope.single("dashboards")
        context = get_registry().prompt_context(scope)
        assert "primitives.dashboards" in context
        assert "create_tile" in context
        assert "create_dashboard" in context

    def test_dashboard_prompt_examples_generated(self):
        scope = PrimitiveScope.single("dashboards")
        examples = get_registry().prompt_examples(scope)
        assert "primitives.dashboards.create_tile" in examples
        assert "primitives.dashboards.create_dashboard" in examples
        assert "on_data" in examples

    def test_no_dashboard_vs_data_routing_guidance(self):
        """Dashboards and data are orthogonal -- no routing guidance needed."""
        scope = PrimitiveScope(
            scoped_managers=frozenset({"dashboards", "data"}),
        )
        context = get_registry().prompt_context(scope)
        assert "primitives.dashboards.*` vs `primitives.data.*" not in context


class TestPromptExampleFunctions:
    """Test that prompt example functions execute without errors."""

    def test_baked_in_example_returns_string(self):
        from unity.actor.prompt_examples import (
            get_primitives_dashboards_baked_in_example,
        )

        result = get_primitives_dashboards_baked_in_example()
        assert isinstance(result, str)
        assert "create_tile" in result
        assert "include_plotlyjs" in result

    def test_live_data_example_returns_string(self):
        from unity.actor.prompt_examples import (
            get_primitives_dashboards_live_data_example,
        )

        result = get_primitives_dashboards_live_data_example()
        assert isinstance(result, str)
        assert "on_data" in result
        assert "data_bindings" in result
        assert "data.sales" in result

    def test_live_data_example_includes_query_params(self):
        from unity.actor.prompt_examples import (
            get_primitives_dashboards_live_data_example,
        )

        result = get_primitives_dashboards_live_data_example()
        assert 'columns=["month", "revenue"]' in result
        assert 'order_by="month"' in result

    def test_live_data_example_no_unifydata_calls(self):
        from unity.actor.prompt_examples import (
            get_primitives_dashboards_live_data_example,
        )

        result = get_primitives_dashboards_live_data_example()
        assert "UnifyData.filter(" not in result
        assert "UnifyData.reduce(" not in result

    def test_rich_live_data_example_returns_string(self):
        from unity.actor.prompt_examples import (
            get_primitives_dashboards_rich_live_data_example,
        )

        result = get_primitives_dashboards_rich_live_data_example()
        assert isinstance(result, str)
        assert "on_data" in result
        assert "JoinBinding" in result
        assert "JoinReduceBinding" in result
        assert "data.orders" in result
        assert "data.by_category" in result

    def test_rich_live_data_example_no_unifydata_calls(self):
        from unity.actor.prompt_examples import (
            get_primitives_dashboards_rich_live_data_example,
        )

        result = get_primitives_dashboards_rich_live_data_example()
        assert "UnifyData.join(" not in result
        assert "UnifyData.joinReduce(" not in result

    def test_composition_example_returns_string(self):
        from unity.actor.prompt_examples import (
            get_primitives_dashboards_composition_example,
        )

        result = get_primitives_dashboards_composition_example()
        assert isinstance(result, str)
        assert "create_dashboard" in result
        assert "TilePosition" in result
