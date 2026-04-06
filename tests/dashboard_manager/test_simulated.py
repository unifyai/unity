"""Tests for SimulatedDashboardManager -- full CRUD coverage."""

from unity.dashboard_manager.types.tile import DataBinding
from unity.dashboard_manager.types.dashboard import TilePosition


class TestSimulatedTileCRUD:
    def test_create_tile(self, simulated_dm):
        result = simulated_dm.create_tile("<h1>Test</h1>", title="Test Tile")
        assert result.succeeded
        assert result.token is not None
        assert "tile/view/" in result.url

    def test_create_tile_with_bindings(self, simulated_dm):
        result = simulated_dm.create_tile(
            "<div></div>",
            title="Live Tile",
            data_bindings=[DataBinding(context="Data/Sales")],
        )
        assert result.succeeded
        tile = simulated_dm.get_tile(result.token)
        assert tile.has_data_bindings is True
        assert "Data/Sales" in tile.data_binding_contexts

    def test_create_tile_with_enriched_bindings(self, simulated_dm):
        result = simulated_dm.create_tile(
            "<div>chart</div>",
            title="Enriched Tile",
            data_bindings=[
                DataBinding(
                    context="Data/Sales/Monthly",
                    columns=["month", "revenue"],
                    order_by="month",
                    filter="year == 2025",
                ),
                DataBinding(
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
