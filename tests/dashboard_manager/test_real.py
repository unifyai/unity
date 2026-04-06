"""
Integration tests for the real DashboardManager against the Unify backend.

These tests exercise the full DashboardManager public API (tile CRUD, dashboard
CRUD, list/filter operations) using the real backend.

The DashboardManager stores tile HTML and dashboard layouts in Unify contexts
(via DataManager) and registers tokens with Orchestra. Token registration may
fail silently if Orchestra lacks the dashboard endpoint, but the core CRUD
operations still complete because the data layer is independent of token
registration.

Each test gets a fresh, isolated Unify context that is cleaned up after the test.
"""

from __future__ import annotations

from unity.dashboard_manager.dashboard_manager import DashboardManager
from unity.dashboard_manager.types.dashboard import TilePosition
from unity.dashboard_manager.types.tile import DataBinding
from unity.manager_registry import ManagerRegistry
from tests.helpers import _handle_project

# ────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────


def _fresh_dm() -> DashboardManager:
    """Create a fresh DashboardManager instance (clears registry singleton)."""
    ManagerRegistry.clear()
    return DashboardManager()


# ────────────────────────────────────────────────────────────────────────────
# Tile CRUD
# ────────────────────────────────────────────────────────────────────────────


@_handle_project
def test_create_tile_basic():
    """create_tile should store HTML in Unify and return a TileResult."""
    dm = _fresh_dm()

    result = dm.create_tile(
        "<h1>Hello World</h1>",
        title="Test Tile",
        description="A simple test tile",
    )

    assert result.succeeded, f"create_tile failed: {result.error}"
    assert result.token
    assert len(result.token) == 12
    assert result.title == "Test Tile"
    assert "/tile/view/" in result.url


@_handle_project
def test_create_tile_with_data_bindings():
    """create_tile should accept data_bindings for live-data tiles."""
    dm = _fresh_dm()

    result = dm.create_tile(
        "<div id='chart'>Loading...</div>",
        title="Live Chart",
        data_bindings=[
            DataBinding(context="Data/monthly_stats"),
            DataBinding(context="Data/revenue", alias="rev"),
        ],
    )

    assert result.succeeded, f"create_tile failed: {result.error}"
    assert result.token


@_handle_project
def test_create_tile_rejects_invalid_binding_context():
    """create_tile should fail with clear error when binding context is invalid."""
    dm = _fresh_dm()

    result = dm.create_tile(
        "<div>chart</div>",
        title="Bad Context Tile",
        data_bindings=[
            DataBinding(
                context="Data/NonexistentContext/DoesNotExist",
                columns=["foo"],
            ),
        ],
    )

    assert not result.succeeded
    assert result.error is not None
    assert len(result.error) > 0


@_handle_project
def test_get_tile():
    """get_tile should return full TileRecord including html_content."""
    dm = _fresh_dm()

    created = dm.create_tile(
        "<p>Retrievable content</p>",
        title="Get Test",
        description="Testing get_tile",
    )
    assert created.succeeded

    tile = dm.get_tile(created.token)
    assert tile is not None
    assert tile.token == created.token
    assert tile.title == "Get Test"
    assert tile.html_content == "<p>Retrievable content</p>"
    assert tile.description == "Testing get_tile"


@_handle_project
def test_get_tile_not_found():
    """get_tile should return None for a nonexistent token."""
    dm = _fresh_dm()
    tile = dm.get_tile("nonexistent_")
    assert tile is None


@_handle_project
def test_update_tile():
    """update_tile should modify tile fields."""
    dm = _fresh_dm()

    created = dm.create_tile(
        "<h1>Original</h1>",
        title="Before Update",
    )
    assert created.succeeded

    updated = dm.update_tile(
        created.token,
        html="<h1>Updated</h1>",
        title="After Update",
    )
    assert updated.succeeded

    tile = dm.get_tile(created.token)
    assert tile is not None
    assert tile.html_content == "<h1>Updated</h1>"
    assert tile.title == "After Update"


@_handle_project
def test_delete_tile():
    """delete_tile should remove the tile from the context."""
    dm = _fresh_dm()

    created = dm.create_tile("<p>Delete me</p>", title="To Delete")
    assert created.succeeded

    deleted = dm.delete_tile(created.token)
    assert deleted is True

    tile = dm.get_tile(created.token)
    assert tile is None


@_handle_project
def test_delete_tile_not_found():
    """delete_tile should return False for a nonexistent token."""
    dm = _fresh_dm()
    deleted = dm.delete_tile("nonexistent_")
    assert deleted is False


@_handle_project
def test_list_tiles():
    """list_tiles should return tiles without html_content."""
    dm = _fresh_dm()

    dm.create_tile("<h1>Tile A</h1>", title="Tile A")
    dm.create_tile("<h1>Tile B</h1>", title="Tile B")

    tiles = dm.list_tiles()
    assert len(tiles) >= 2

    titles = {t.title for t in tiles}
    assert "Tile A" in titles
    assert "Tile B" in titles


@_handle_project
def test_list_tiles_with_limit():
    """list_tiles should respect the limit parameter."""
    dm = _fresh_dm()

    for i in range(5):
        dm.create_tile(f"<p>Item {i}</p>", title=f"Tile {i}")

    tiles = dm.list_tiles(limit=3)
    assert len(tiles) <= 3


# ────────────────────────────────────────────────────────────────────────────
# Dashboard CRUD
# ────────────────────────────────────────────────────────────────────────────


@_handle_project
def test_create_dashboard_basic():
    """create_dashboard should store layout in Unify and return a DashboardResult."""
    dm = _fresh_dm()

    t1 = dm.create_tile("<h1>KPI</h1>", title="KPI Card")
    t2 = dm.create_tile("<div>Chart</div>", title="Chart")
    assert t1.succeeded and t2.succeeded

    result = dm.create_dashboard(
        "Test Dashboard",
        description="Integration test dashboard",
        tiles=[
            TilePosition(tile_token=t1.token, x=0, y=0, w=6, h=4),
            TilePosition(tile_token=t2.token, x=6, y=0, w=6, h=4),
        ],
    )

    assert result.succeeded, f"create_dashboard failed: {result.error}"
    assert result.token
    assert len(result.token) == 12
    assert result.title == "Test Dashboard"
    assert "/dashboard/view/" in result.url
    assert len(result.tiles) == 2


@_handle_project
def test_create_dashboard_no_tiles():
    """create_dashboard with no tiles should succeed (empty layout)."""
    dm = _fresh_dm()

    result = dm.create_dashboard("Empty Dashboard")
    assert result.succeeded
    assert result.tiles == []


@_handle_project
def test_get_dashboard():
    """get_dashboard should return full DashboardResult with tile positions."""
    dm = _fresh_dm()

    tile = dm.create_tile("<h1>Solo</h1>", title="Solo Tile")
    assert tile.succeeded

    created = dm.create_dashboard(
        "Get Test Dashboard",
        tiles=[TilePosition(tile_token=tile.token, x=0, y=0, w=12, h=6)],
    )
    assert created.succeeded

    dashboard = dm.get_dashboard(created.token)
    assert dashboard is not None
    assert dashboard.token == created.token
    assert dashboard.title == "Get Test Dashboard"
    assert len(dashboard.tiles) == 1
    assert dashboard.tiles[0].tile_token == tile.token
    assert dashboard.tiles[0].w == 12


@_handle_project
def test_get_dashboard_not_found():
    """get_dashboard should return None for a nonexistent token."""
    dm = _fresh_dm()
    dashboard = dm.get_dashboard("nonexistent_")
    assert dashboard is None


@_handle_project
def test_update_dashboard():
    """update_dashboard should modify dashboard fields and layout."""
    dm = _fresh_dm()

    t1 = dm.create_tile("<p>A</p>", title="A")
    t2 = dm.create_tile("<p>B</p>", title="B")
    assert t1.succeeded and t2.succeeded

    created = dm.create_dashboard(
        "Before",
        tiles=[TilePosition(tile_token=t1.token, x=0, y=0, w=12, h=6)],
    )
    assert created.succeeded

    updated = dm.update_dashboard(
        created.token,
        title="After",
        tiles=[
            TilePosition(tile_token=t1.token, x=0, y=0, w=6, h=4),
            TilePosition(tile_token=t2.token, x=6, y=0, w=6, h=4),
        ],
    )
    assert updated.succeeded
    assert len(updated.tiles) == 2

    dashboard = dm.get_dashboard(created.token)
    assert dashboard.title == "After"
    assert len(dashboard.tiles) == 2


@_handle_project
def test_delete_dashboard():
    """delete_dashboard should remove the dashboard from the context."""
    dm = _fresh_dm()

    created = dm.create_dashboard("To Delete")
    assert created.succeeded

    deleted = dm.delete_dashboard(created.token)
    assert deleted is True

    dashboard = dm.get_dashboard(created.token)
    assert dashboard is None


@_handle_project
def test_delete_dashboard_not_found():
    """delete_dashboard should return False for a nonexistent token."""
    dm = _fresh_dm()
    deleted = dm.delete_dashboard("nonexistent_")
    assert deleted is False


@_handle_project
def test_list_dashboards():
    """list_dashboards should return stored dashboards."""
    dm = _fresh_dm()

    dm.create_dashboard("Dashboard Alpha")
    dm.create_dashboard("Dashboard Beta")

    dashboards = dm.list_dashboards()
    assert len(dashboards) >= 2

    titles = {d.title for d in dashboards}
    assert "Dashboard Alpha" in titles
    assert "Dashboard Beta" in titles


@_handle_project
def test_list_dashboards_with_limit():
    """list_dashboards should respect the limit parameter."""
    dm = _fresh_dm()

    for i in range(5):
        dm.create_dashboard(f"Dashboard {i}")

    dashboards = dm.list_dashboards(limit=3)
    assert len(dashboards) <= 3


# ────────────────────────────────────────────────────────────────────────────
# Full lifecycle
# ────────────────────────────────────────────────────────────────────────────


@_handle_project
def test_full_tile_lifecycle():
    """Complete tile lifecycle: create -> get -> update -> list -> delete."""
    dm = _fresh_dm()

    created = dm.create_tile(
        "<h1>Lifecycle Test</h1>",
        title="Lifecycle Tile",
        description="Full lifecycle",
    )
    assert created.succeeded

    fetched = dm.get_tile(created.token)
    assert fetched is not None
    assert fetched.html_content == "<h1>Lifecycle Test</h1>"

    dm.update_tile(created.token, title="Updated Lifecycle Tile")
    fetched2 = dm.get_tile(created.token)
    assert fetched2.title == "Updated Lifecycle Tile"

    tiles = dm.list_tiles()
    assert any(t.token == created.token for t in tiles)

    dm.delete_tile(created.token)
    assert dm.get_tile(created.token) is None


@_handle_project
def test_full_dashboard_lifecycle():
    """Complete dashboard lifecycle: create tiles -> compose -> update -> delete."""
    dm = _fresh_dm()

    t1 = dm.create_tile("<h1>KPI</h1>", title="KPI")
    t2 = dm.create_tile("<canvas>Chart</canvas>", title="Chart")
    assert t1.succeeded and t2.succeeded

    dashboard = dm.create_dashboard(
        "Lifecycle Dashboard",
        tiles=[
            TilePosition(tile_token=t1.token, x=0, y=0, w=6, h=4),
            TilePosition(tile_token=t2.token, x=6, y=0, w=6, h=4),
        ],
    )
    assert dashboard.succeeded

    fetched = dm.get_dashboard(dashboard.token)
    assert fetched is not None
    assert len(fetched.tiles) == 2

    dm.update_dashboard(dashboard.token, title="Updated Dashboard")
    fetched2 = dm.get_dashboard(dashboard.token)
    assert fetched2.title == "Updated Dashboard"

    dm.delete_dashboard(dashboard.token)
    assert dm.get_dashboard(dashboard.token) is None
