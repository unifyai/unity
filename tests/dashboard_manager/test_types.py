"""Tests for DashboardManager Pydantic types."""

import json

from unity.dashboard_manager.types.tile import (
    DataBinding,
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

    def test_data_binding(self):
        binding = DataBinding(context="Data/Sales/Monthly", alias="sales")
        assert binding.context == "Data/Sales/Monthly"
        assert binding.alias == "sales"

    def test_data_binding_with_query_params(self):
        binding = DataBinding(
            context="Data/Sales/Monthly",
            alias="sales",
            filter="year == 2025",
            columns=["month", "revenue"],
            exclude_columns=["internal_id"],
            order_by="month",
            descending=True,
        )
        assert binding.filter == "year == 2025"
        assert binding.columns == ["month", "revenue"]
        assert binding.exclude_columns == ["internal_id"]
        assert binding.order_by == "month"
        assert binding.descending is True

    def test_data_binding_minimal_still_works(self):
        binding = DataBinding(context="Data/X")
        assert binding.context == "Data/X"
        assert binding.alias is None
        assert binding.filter is None
        assert binding.columns is None
        assert binding.exclude_columns is None
        assert binding.order_by is None
        assert binding.descending is False

    def test_data_binding_defaults(self):
        binding = DataBinding(context="Data/Y", columns=["a"])
        assert binding.filter is None
        assert binding.exclude_columns is None
        assert binding.order_by is None
        assert binding.descending is False
        assert binding.columns == ["a"]


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
