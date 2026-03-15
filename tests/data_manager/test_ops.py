"""
Tests for DataManager ops builder helpers.

Covers the config-dict builder functions in
``data_manager.ops.plot_ops`` and ``data_manager.ops.table_view_ops``.
These builders translate Pydantic config models into the raw dicts
expected by the Unify Plot / Table-View API.
"""

from __future__ import annotations

from unity.data_manager.ops.plot_ops import (
    PlotConfig,
    _build_plot_config_dict,
    _build_project_config_dict,
)
from unity.data_manager.ops.table_view_ops import (
    _build_table_config_dict,
    _build_project_config_dict as _build_tv_project_config_dict,
)
from unity.data_manager.types.table_view import TableViewConfig
from unity.session_details import UNASSIGNED_USER_CONTEXT, UNASSIGNED_ASSISTANT_CONTEXT

_CTX = f"{UNASSIGNED_USER_CONTEXT}/{UNASSIGNED_ASSISTANT_CONTEXT}/Files/Local/test"


# =============================================================================
# plot_ops: _build_plot_config_dict
# =============================================================================


class TestBuildPlotConfigDict:

    def test_minimal_omits_optional_keys(self):
        """Only required fields appear; optional keys like y_axis are absent."""
        d = _build_plot_config_dict(PlotConfig(plot_type="bar", x_axis="Category"))
        assert d["type"] == "bar"
        assert d["x_axis"] == "Category"
        for key in (
            "y_axis",
            "group_by",
            "metric",
            "scale_x",
            "scale_y",
            "bin_count",
            "show_regression",
        ):
            assert key not in d

    def test_full_config_propagates_all_fields(self):
        d = _build_plot_config_dict(
            PlotConfig(
                plot_type="scatter",
                x_axis="X",
                y_axis="Y",
                group_by="Group",
                metric="mean",
                scale_x="log",
                scale_y="linear",
                bin_count=20,
                show_regression=True,
            ),
        )
        assert d == {
            "type": "scatter",
            "x_axis": "X",
            "y_axis": "Y",
            "group_by": "Group",
            "metric": "mean",
            "scale_x": "log",
            "scale_y": "linear",
            "bin_count": 20,
            "show_regression": True,
        }


# =============================================================================
# plot_ops: _build_project_config_dict
# =============================================================================


class TestBuildProjectConfigDict:

    def test_defaults(self):
        """Minimal call produces randomize=False and omits optional keys."""
        d = _build_project_config_dict(project_name="P", context=_CTX)
        assert d["project_name"] == "P"
        assert d["context"] == _CTX
        assert d["randomize"] is False
        for key in ("filter_expr", "exclude_fields", "group_by"):
            assert key not in d

    def test_all_optional_fields(self):
        d = _build_project_config_dict(
            project_name="P",
            context=_CTX,
            filter_expr="status == 'active'",
            randomize=True,
            exclude_fields=["password"],
            group_by="region",
        )
        assert d["filter_expr"] == "status == 'active'"
        assert d["randomize"] is True
        assert d["exclude_fields"] == ["password"]
        assert d["group_by"] == "region"


# =============================================================================
# table_view_ops: _build_table_config_dict
# =============================================================================


class TestBuildTableConfigDict:

    def test_empty_config_yields_empty_dict(self):
        assert _build_table_config_dict(TableViewConfig()) == {}

    def test_columns_nested_structure(self):
        """visible, hidden, and order are grouped under a 'columns' key."""
        d = _build_table_config_dict(
            TableViewConfig(
                columns_visible=["name", "email"],
                columns_hidden=["ssn"],
                columns_order=["email", "name"],
            ),
        )
        assert d["columns"]["visible"] == ["name", "email"]
        assert d["columns"]["hidden"] == ["ssn"]
        assert d["columns"]["order"] == ["email", "name"]

    def test_full_config(self):
        d = _build_table_config_dict(
            TableViewConfig(
                columns_visible=["a", "b"],
                columns_order=["b", "a"],
                row_limit=100,
                sort_by="a",
                sort_order="asc",
            ),
        )
        assert d["columns"]["visible"] == ["a", "b"]
        assert d["columns"]["order"] == ["b", "a"]
        assert d["row_limit"] == 100
        assert d["sort_by"] == "a"
        assert d["sort_order"] == "asc"


# =============================================================================
# table_view_ops: _build_project_config_dict
# =============================================================================


class TestBuildTVProjectConfigDict:

    def test_defaults(self):
        d = _build_tv_project_config_dict(project_name="P", context="Data/sales")
        assert d["project_name"] == "P"
        assert d["context"] == "Data/sales"
        assert "filter_expr" not in d

    def test_with_filter(self):
        d = _build_tv_project_config_dict(
            project_name="P",
            context="Data/sales",
            filter_expr="status == 'active'",
        )
        assert d["filter_expr"] == "status == 'active'"
