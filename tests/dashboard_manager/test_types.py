"""Tests for DashboardManager Pydantic types."""

import json

import pytest
from pydantic import TypeAdapter

from unity.dashboard_manager.types.tile import (
    DataBinding,
    FilterBinding,
    JoinBinding,
    JoinReduceBinding,
    ReduceBinding,
    TileRecord,
    TileRecordRow,
    TileResult,
)
from unity.dashboard_manager.types.dashboard import (
    DashboardRecordRow,
    DashboardResult,
    TilePosition,
)


class TestTileTypes:
    def test_tile_record_row_minimal(self):
        row = TileRecordRow(
            token="abc123abc123",
            title="Test Tile",
            html_content="<h1>Hello</h1>",
        )
        assert row.token == "abc123abc123"
        assert row.has_data_bindings is False
        assert row.data_binding_contexts is None

    def test_tile_record_with_bindings(self):
        row = TileRecordRow(
            token="abc123abc123",
            title="Live Tile",
            html_content="<div></div>",
            has_data_bindings=True,
            data_binding_contexts="Data/Sales,Data/Orders",
        )
        assert row.has_data_bindings is True
        assert "Data/Sales" in row.data_binding_contexts

    def test_tile_record_includes_id(self):
        record = TileRecord(
            tile_id=42,
            token="abc123abc123",
            title="Test",
            html_content="<p>content</p>",
        )
        assert record.tile_id == 42

    def test_tile_result_succeeded(self):
        result = TileResult(
            url="https://example.com/tile/view/abc123",
            token="abc123",
            title="Test",
        )
        assert result.succeeded is True

    def test_tile_result_failed(self):
        result = TileResult(error="Something went wrong")
        assert result.succeeded is False


class TestFilterBinding:
    def test_minimal(self):
        b = FilterBinding(context="Data/X")
        assert b.operation == "filter"
        assert b.context == "Data/X"
        assert b.alias is None
        assert b.filter is None
        assert b.columns is None
        assert b.exclude_columns is None
        assert b.order_by is None
        assert b.descending is False
        assert b.limit is None
        assert b.offset is None
        assert b.group_by is None

    def test_with_query_params(self):
        b = FilterBinding(
            context="Data/Sales/Monthly",
            alias="sales",
            filter="year == 2025",
            columns=["month", "revenue"],
            exclude_columns=["internal_id"],
            order_by="month",
            descending=True,
            limit=500,
            offset=10,
            group_by=["region"],
        )
        assert b.filter == "year == 2025"
        assert b.columns == ["month", "revenue"]
        assert b.limit == 500
        assert b.group_by == ["region"]

    def test_default_operation_field(self):
        b = FilterBinding(context="Data/X")
        assert b.operation == "filter"


class TestReduceBinding:
    def test_minimal(self):
        b = ReduceBinding(context="Data/Sales", metric="count", columns="id")
        assert b.operation == "reduce"
        assert b.metric == "count"
        assert b.columns == "id"

    def test_with_group_by(self):
        b = ReduceBinding(
            context="Data/Sales",
            metric="sum",
            columns=["revenue", "cost"],
            filter="year == 2025",
            group_by=["region"],
            result_where="revenue > 1000",
        )
        assert b.group_by == ["region"]
        assert b.result_where == "revenue > 1000"


class TestJoinBinding:
    def test_minimal(self):
        b = JoinBinding(
            tables=["Data/Orders", "Data/Customers"],
            join_expr="Data/Orders.cust_id == Data/Customers.id",
            select={"Data/Orders.amount": "amount", "Data/Customers.name": "name"},
        )
        assert b.operation == "join"
        assert len(b.tables) == 2
        assert b.mode == "inner"
        assert b.result_limit == 100
        assert b.result_offset == 0

    def test_full_params(self):
        b = JoinBinding(
            tables=["Data/A", "Data/B"],
            join_expr="Data/A.id == Data/B.fk",
            select={"Data/A.val": "val"},
            mode="left",
            left_where="active == True",
            right_where="created > '2025-01-01'",
            result_where="val > 10",
            result_limit=50,
            result_offset=25,
        )
        assert b.mode == "left"
        assert b.left_where == "active == True"
        assert b.result_limit == 50


class TestJoinReduceBinding:
    def test_minimal(self):
        b = JoinReduceBinding(
            tables=["Data/Orders", "Data/Products"],
            join_expr="Data/Orders.pid == Data/Products.id",
            select={"Data/Orders.amount": "amount", "Data/Products.cat": "cat"},
            metric="sum",
            columns="amount",
        )
        assert b.operation == "join_reduce"
        assert b.metric == "sum"

    def test_with_group_by(self):
        b = JoinReduceBinding(
            tables=["Data/A", "Data/B"],
            join_expr="Data/A.id == Data/B.fk",
            select={"Data/A.val": "val", "Data/B.cat": "cat"},
            metric="avg",
            columns=["val"],
            group_by=["cat"],
            result_where="val > 5",
        )
        assert b.group_by == ["cat"]


class TestDataBindingDiscriminator:
    """Verify the discriminated union dispatches correctly from dicts."""

    adapter = TypeAdapter(DataBinding)

    def test_filter_from_dict(self):
        b = self.adapter.validate_python(
            {"operation": "filter", "context": "Data/X", "columns": ["a"]},
        )
        assert isinstance(b, FilterBinding)

    def test_filter_default_operation(self):
        b = self.adapter.validate_python(
            {"operation": "filter", "context": "Data/X"},
        )
        assert isinstance(b, FilterBinding)

    def test_reduce_from_dict(self):
        b = self.adapter.validate_python(
            {
                "operation": "reduce",
                "context": "Data/X",
                "metric": "count",
                "columns": "id",
            },
        )
        assert isinstance(b, ReduceBinding)

    def test_join_from_dict(self):
        b = self.adapter.validate_python(
            {
                "operation": "join",
                "tables": ["Data/A", "Data/B"],
                "join_expr": "Data/A.id == Data/B.fk",
                "select": {"Data/A.x": "x"},
            },
        )
        assert isinstance(b, JoinBinding)

    def test_join_reduce_from_dict(self):
        b = self.adapter.validate_python(
            {
                "operation": "join_reduce",
                "tables": ["Data/A", "Data/B"],
                "join_expr": "Data/A.id == Data/B.fk",
                "select": {"Data/A.x": "x"},
                "metric": "sum",
                "columns": "x",
            },
        )
        assert isinstance(b, JoinReduceBinding)

    def test_missing_operation_raises(self):
        with pytest.raises(Exception):
            self.adapter.validate_python({"context": "Data/X"})


class TestDashboardTypes:
    def test_tile_position_defaults(self):
        pos = TilePosition(tile_token="tok123")
        assert pos.x == 0
        assert pos.y == 0
        assert pos.w == 6
        assert pos.h == 4

    def test_tile_position_custom(self):
        pos = TilePosition(tile_token="tok123", x=4, y=2, w=8, h=6)
        assert pos.x == 4
        assert pos.w == 8

    def test_dashboard_record_row(self):
        layout = json.dumps([{"tile_token": "t1", "x": 0, "y": 0, "w": 6, "h": 4}])
        row = DashboardRecordRow(
            token="dash123dash1",
            title="My Dashboard",
            layout=layout,
            tile_count=1,
        )
        assert row.tile_count == 1

    def test_dashboard_result_succeeded(self):
        result = DashboardResult(
            url="https://example.com/dashboard/view/abc123",
            token="abc123",
            title="Dashboard",
            tiles=[TilePosition(tile_token="t1")],
        )
        assert result.succeeded is True

    def test_dashboard_result_failed(self):
        result = DashboardResult(error="Failed")
        assert result.succeeded is False
