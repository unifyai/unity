"""Tests for SimulatedDashboardManager -- full CRUD coverage."""

import json

from unity.dashboard_manager.types.tile import (
    FilterBinding,
    JoinBinding,
    JoinReduceBinding,
    ReduceBinding,
)
from unity.dashboard_manager.types.dashboard import TilePosition


class TestSimulatedTileCRUD:
    def test_create_tile(self, simulated_dm):
        result = simulated_dm.create_tile("<h1>Test</h1>", title="Test Tile")
        assert result.succeeded
        assert result.token is not None
        assert "tile/view/" in result.url

    def test_create_tile_with_filter_binding(self, simulated_dm):
        result = simulated_dm.create_tile(
            "<div></div>",
            title="Live Tile",
            data_bindings=[FilterBinding(context="Data/Sales")],
        )
        assert result.succeeded
        tile = simulated_dm.get_tile(result.token)
        assert tile.has_data_bindings is True
        assert "Data/Sales" in tile.data_binding_contexts

    def test_create_tile_with_enriched_filter_bindings(self, simulated_dm):
        result = simulated_dm.create_tile(
            "<div>chart</div>",
            title="Enriched Tile",
            data_bindings=[
                FilterBinding(
                    context="Data/Sales/Monthly",
                    columns=["month", "revenue"],
                    order_by="month",
                    filter="year == 2025",
                ),
                FilterBinding(
                    context="Data/Products",
                    alias="products",
                    exclude_columns=["internal_id"],
                    descending=True,
                ),
            ],
        )
        assert result.succeeded
        tile = simulated_dm.get_tile(result.token)
        assert tile.has_data_bindings is True
        assert "Data/Sales/Monthly" in tile.data_binding_contexts
        assert "Data/Products" in tile.data_binding_contexts

    def test_create_tile_with_reduce_binding(self, simulated_dm):
        result = simulated_dm.create_tile(
            "<div>kpi</div>",
            title="KPI Tile",
            data_bindings=[
                ReduceBinding(
                    context="Data/Sales",
                    metric="sum",
                    columns="revenue",
                ),
            ],
        )
        assert result.succeeded
        tile = simulated_dm.get_tile(result.token)
        assert tile.has_data_bindings is True
        assert "Data/Sales" in tile.data_binding_contexts

    def test_create_tile_with_join_binding(self, simulated_dm):
        result = simulated_dm.create_tile(
            "<div>joined</div>",
            title="Join Tile",
            data_bindings=[
                JoinBinding(
                    tables=["Data/Orders", "Data/Customers"],
                    join_expr="Data/Orders.cust_id == Data/Customers.id",
                    select={
                        "Data/Orders.amount": "amount",
                        "Data/Customers.name": "name",
                    },
                ),
            ],
        )
        assert result.succeeded
        tile = simulated_dm.get_tile(result.token)
        assert tile.has_data_bindings is True
        assert "Data/Orders" in tile.data_binding_contexts
        assert "Data/Customers" in tile.data_binding_contexts

    def test_create_tile_with_join_reduce_binding(self, simulated_dm):
        result = simulated_dm.create_tile(
            "<div>aggregated</div>",
            title="Join-Reduce Tile",
            data_bindings=[
                JoinReduceBinding(
                    tables=["Data/Orders", "Data/Products"],
                    join_expr="Data/Orders.pid == Data/Products.id",
                    select={"Data/Orders.amount": "amount", "Data/Products.cat": "cat"},
                    metric="sum",
                    columns="amount",
                    group_by=["cat"],
                ),
            ],
        )
        assert result.succeeded
        tile = simulated_dm.get_tile(result.token)
        assert tile.has_data_bindings is True
        assert "Data/Orders" in tile.data_binding_contexts
        assert "Data/Products" in tile.data_binding_contexts

    def test_create_tile_with_mixed_bindings(self, simulated_dm):
        result = simulated_dm.create_tile(
            "<div>mixed</div>",
            title="Mixed Bindings Tile",
            data_bindings=[
                FilterBinding(context="Data/Sales", columns=["month"]),
                ReduceBinding(context="Data/Sales", metric="count", columns="id"),
                JoinBinding(
                    tables=["Data/A", "Data/B"],
                    join_expr="Data/A.id == Data/B.fk",
                    select={"Data/A.x": "x"},
                ),
            ],
        )
        assert result.succeeded
        tile = simulated_dm.get_tile(result.token)
        assert tile.has_data_bindings is True
        ctxs = tile.data_binding_contexts
        assert "Data/Sales" in ctxs
        assert "Data/A" in ctxs
        assert "Data/B" in ctxs

    def test_get_tile(self, simulated_dm):
        result = simulated_dm.create_tile("<p>Hello</p>", title="Get Test")
        tile = simulated_dm.get_tile(result.token)
        assert tile is not None
        assert tile.html_content == "<p>Hello</p>"
        assert tile.title == "Get Test"

    def test_get_tile_not_found(self, simulated_dm):
        assert simulated_dm.get_tile("nonexistent") is None

    def test_update_tile(self, simulated_dm):
        result = simulated_dm.create_tile("<h1>V1</h1>", title="Original")
        updated = simulated_dm.update_tile(
            result.token,
            html="<h1>V2</h1>",
            title="Updated",
        )
        assert updated.succeeded
        assert updated.title == "Updated"
        tile = simulated_dm.get_tile(result.token)
        assert tile.html_content == "<h1>V2</h1>"

    def test_update_tile_not_found(self, simulated_dm):
        result = simulated_dm.update_tile("nonexistent", title="Nope")
        assert not result.succeeded

    def test_delete_tile(self, simulated_dm):
        result = simulated_dm.create_tile("<p>Delete me</p>", title="Doomed")
        assert simulated_dm.delete_tile(result.token) is True
        assert simulated_dm.get_tile(result.token) is None

    def test_delete_tile_not_found(self, simulated_dm):
        assert simulated_dm.delete_tile("nonexistent") is False

    def test_list_tiles(self, simulated_dm):
        simulated_dm.create_tile("<p>1</p>", title="Tile 1")
        simulated_dm.create_tile("<p>2</p>", title="Tile 2")
        tiles = simulated_dm.list_tiles()
        assert len(tiles) == 2
        assert all(t.html_content == "" for t in tiles)

    def test_list_tiles_limit(self, simulated_dm):
        for i in range(5):
            simulated_dm.create_tile(f"<p>{i}</p>", title=f"Tile {i}")
        tiles = simulated_dm.list_tiles(limit=3)
        assert len(tiles) == 3


class TestSimulatedTileOnData:
    def test_create_tile_with_on_data(self, simulated_dm):
        result = simulated_dm.create_tile(
            "<div id='tbl'>Loading...</div>",
            title="On-Data Tile",
            data_bindings=[
                FilterBinding(context="Data/Sales", alias="sales"),
            ],
            on_data="document.getElementById('tbl').textContent = data.sales.length;",
        )
        assert result.succeeded
        tile = simulated_dm.get_tile(result.token)
        assert tile.on_data_script is not None
        assert "data.sales" in tile.on_data_script
        assert tile.data_bindings_json is not None
        parsed = json.loads(tile.data_bindings_json)
        assert len(parsed) == 1
        assert parsed[0]["alias"] == "sales"

    def test_create_tile_with_on_data_auto_alias(self, simulated_dm):
        result = simulated_dm.create_tile(
            "<div></div>",
            title="Auto-Alias",
            data_bindings=[FilterBinding(context="Data/Sales/Monthly")],
            on_data="console.log(data.monthly);",
        )
        assert result.succeeded
        tile = simulated_dm.get_tile(result.token)
        parsed = json.loads(tile.data_bindings_json)
        assert parsed[0]["alias"] == "monthly"

    def test_create_tile_with_mixed_bindings_on_data(self, simulated_dm):
        result = simulated_dm.create_tile(
            "<div></div>",
            title="Mixed",
            data_bindings=[
                FilterBinding(context="Data/Sales", alias="sales"),
                ReduceBinding(
                    context="Data/Sales",
                    alias="total",
                    metric="sum",
                    columns="revenue",
                ),
            ],
            on_data="console.log(data.sales, data.total);",
        )
        assert result.succeeded
        tile = simulated_dm.get_tile(result.token)
        parsed = json.loads(tile.data_bindings_json)
        assert len(parsed) == 2
        aliases = {b["alias"] for b in parsed}
        assert aliases == {"sales", "total"}

    def test_create_tile_without_on_data_no_bindings_json(self, simulated_dm):
        result = simulated_dm.create_tile(
            "<div></div>",
            title="Baked",
        )
        assert result.succeeded
        tile = simulated_dm.get_tile(result.token)
        assert tile.on_data_script is None
        assert tile.data_bindings_json is None

    def test_create_tile_bindings_without_on_data_still_works(self, simulated_dm):
        result = simulated_dm.create_tile(
            "<div></div>",
            title="Legacy Live",
            data_bindings=[FilterBinding(context="Data/X")],
        )
        assert result.succeeded
        tile = simulated_dm.get_tile(result.token)
        assert tile.has_data_bindings is True
        assert tile.on_data_script is None
        assert tile.data_bindings_json is not None

    def test_update_tile_with_on_data(self, simulated_dm):
        result = simulated_dm.create_tile(
            "<div></div>",
            title="Update Test",
            data_bindings=[FilterBinding(context="Data/X", alias="x")],
            on_data="console.log(data.x);",
        )
        assert result.succeeded

        updated = simulated_dm.update_tile(
            result.token,
            on_data="document.body.textContent = JSON.stringify(data.x);",
        )
        assert updated.succeeded
        tile = simulated_dm.get_tile(result.token)
        assert "JSON.stringify" in tile.on_data_script

    def test_update_tile_with_new_bindings(self, simulated_dm):
        result = simulated_dm.create_tile(
            "<div></div>",
            title="Binding Update",
            data_bindings=[FilterBinding(context="Data/A", alias="a")],
            on_data="console.log(data.a);",
        )
        assert result.succeeded

        updated = simulated_dm.update_tile(
            result.token,
            data_bindings=[
                FilterBinding(context="Data/B", alias="b"),
                FilterBinding(context="Data/C", alias="c"),
            ],
            on_data="console.log(data.b, data.c);",
        )
        assert updated.succeeded
        tile = simulated_dm.get_tile(result.token)
        parsed = json.loads(tile.data_bindings_json)
        assert len(parsed) == 2
        assert tile.data_binding_contexts == "Data/B,Data/C"

    def test_update_tile_clear_on_data(self, simulated_dm):
        result = simulated_dm.create_tile(
            "<div></div>",
            title="Clear Test",
            data_bindings=[FilterBinding(context="Data/X", alias="x")],
            on_data="console.log(data.x);",
        )
        assert result.succeeded

        updated = simulated_dm.update_tile(result.token, on_data="")
        assert updated.succeeded
        tile = simulated_dm.get_tile(result.token)
        assert tile.on_data_script is None


class TestSimulatedDashboardCRUD:
    def test_create_dashboard(self, simulated_dm):
        t = simulated_dm.create_tile("<p>T</p>", title="T")
        result = simulated_dm.create_dashboard(
            "Test Dashboard",
            tiles=[TilePosition(tile_token=t.token, x=0, y=0, w=12, h=4)],
        )
        assert result.succeeded
        assert len(result.tiles) == 1

    def test_create_empty_dashboard(self, simulated_dm):
        result = simulated_dm.create_dashboard("Empty")
        assert result.succeeded
        assert result.tiles == []

    def test_get_dashboard(self, simulated_dm):
        result = simulated_dm.create_dashboard("Get Test")
        dash = simulated_dm.get_dashboard(result.token)
        assert dash is not None
        assert dash.title == "Get Test"

    def test_get_dashboard_not_found(self, simulated_dm):
        assert simulated_dm.get_dashboard("nonexistent") is None

    def test_update_dashboard(self, simulated_dm):
        t = simulated_dm.create_tile("<p>T</p>", title="T")
        result = simulated_dm.create_dashboard("Original")
        updated = simulated_dm.update_dashboard(
            result.token,
            title="Updated",
            tiles=[TilePosition(tile_token=t.token)],
        )
        assert updated.succeeded
        assert updated.title == "Updated"
        assert len(updated.tiles) == 1

    def test_update_dashboard_not_found(self, simulated_dm):
        result = simulated_dm.update_dashboard("nonexistent", title="Nope")
        assert not result.succeeded

    def test_delete_dashboard(self, simulated_dm):
        result = simulated_dm.create_dashboard("Doomed")
        assert simulated_dm.delete_dashboard(result.token) is True
        assert simulated_dm.get_dashboard(result.token) is None

    def test_delete_dashboard_not_found(self, simulated_dm):
        assert simulated_dm.delete_dashboard("nonexistent") is False

    def test_list_dashboards(self, simulated_dm):
        simulated_dm.create_dashboard("D1")
        simulated_dm.create_dashboard("D2")
        dashboards = simulated_dm.list_dashboards()
        assert len(dashboards) == 2


class TestSimulatedClear:
    def test_clear_resets_everything(self, simulated_dm):
        simulated_dm.create_tile("<p>T</p>", title="T")
        simulated_dm.create_dashboard("D")
        simulated_dm.clear()
        assert simulated_dm.list_tiles() == []
        assert simulated_dm.list_dashboards() == []


class TestSimulatedDocstringInheritance:
    def test_create_tile_has_docstring(self, simulated_dm):
        assert "HTML" in simulated_dm.create_tile.__doc__

    def test_create_dashboard_has_docstring(self, simulated_dm):
        assert "dashboard" in simulated_dm.create_dashboard.__doc__.lower()
