"""Destination routing coverage for DashboardManager shared-space rows."""

from __future__ import annotations

from unity.common.context_registry import ContextRegistry
from unity.session_details import SESSION_DETAILS
from tests.dashboard_manager.helpers import (
    active_read_root,
    context_titles,
    fresh_dashboard_manager,
)
from tests.helpers import _handle_project


@_handle_project
def test_tile_writes_route_to_destination_and_reads_merge_roots(
    dashboard_manager_spaces,
):
    """Tile rows write to one root while reads merge accessible roots."""
    first_space, _ = dashboard_manager_spaces
    manager = fresh_dashboard_manager()
    personal_root = active_read_root()

    personal_tile = manager.create_tile("<h1>Personal</h1>", title="Personal Tile")
    shared_tile = manager.create_tile(
        "<h1>Patch</h1>",
        title="Patch Tile",
        destination=f"space:{first_space}",
    )
    delete_candidate = manager.create_tile(
        "<h1>Delete</h1>",
        title="Patch Tile To Delete",
        destination=f"space:{first_space}",
    )

    assert personal_tile.succeeded, personal_tile.error
    assert shared_tile.succeeded, shared_tile.error
    assert delete_candidate.succeeded, delete_candidate.error

    assert "Personal Tile" in context_titles(f"{personal_root}/Dashboards/Tiles")
    assert "Patch Tile" in context_titles(f"Spaces/{first_space}/Dashboards/Tiles")

    assert manager.get_tile(shared_tile.token).title == "Patch Tile"
    tile_titles = {tile.title for tile in manager.list_tiles()}
    assert tile_titles >= {"Personal Tile", "Patch Tile"}
    assert not manager.update_tile(shared_tile.token, title="Wrong Root").succeeded
    assert manager.update_tile(
        shared_tile.token,
        title="Patch Tile Updated",
        destination=f"space:{first_space}",
    ).succeeded
    assert manager.get_tile(shared_tile.token).title == "Patch Tile Updated"
    assert manager.delete_tile(shared_tile.token) is False
    assert manager.delete_tile(
        delete_candidate.token,
        destination=f"space:{first_space}",
    )
    assert manager.get_tile(delete_candidate.token) is None

    SESSION_DETAILS.space_ids = []
    ContextRegistry.clear()

    assert manager.get_tile(shared_tile.token) is None
    visible_tile_titles = {tile.title for tile in manager.list_tiles()}
    assert visible_tile_titles >= {"Personal Tile"}
    assert "Patch Tile Updated" not in visible_tile_titles


@_handle_project
def test_dashboard_layout_writes_route_to_destination_and_reads_merge_roots(
    dashboard_manager_spaces,
):
    """Dashboard layout rows write to one root while reads merge accessible roots."""
    _, second_space = dashboard_manager_spaces
    manager = fresh_dashboard_manager()
    personal_root = active_read_root()

    personal_dashboard = manager.create_dashboard("Personal Dashboard")
    shared_dashboard = manager.create_dashboard(
        "Patch Dashboard",
        destination=f"space:{second_space}",
    )
    delete_candidate = manager.create_dashboard(
        "Patch Dashboard To Delete",
        destination=f"space:{second_space}",
    )

    assert personal_dashboard.succeeded, personal_dashboard.error
    assert shared_dashboard.succeeded, shared_dashboard.error
    assert delete_candidate.succeeded, delete_candidate.error

    assert "Personal Dashboard" in context_titles(
        f"{personal_root}/Dashboards/Layouts",
    )
    assert "Patch Dashboard" in context_titles(
        f"Spaces/{second_space}/Dashboards/Layouts",
    )

    assert manager.get_dashboard(shared_dashboard.token).title == "Patch Dashboard"
    dashboard_titles = {dashboard.title for dashboard in manager.list_dashboards()}
    assert dashboard_titles >= {
        "Personal Dashboard",
        "Patch Dashboard",
    }
    assert not manager.update_dashboard(
        shared_dashboard.token,
        title="Wrong Root",
    ).succeeded
    assert manager.update_dashboard(
        shared_dashboard.token,
        title="Patch Dashboard Updated",
        destination=f"space:{second_space}",
    ).succeeded
    assert (
        manager.get_dashboard(shared_dashboard.token).title == "Patch Dashboard Updated"
    )
    assert manager.delete_dashboard(shared_dashboard.token) is False
    assert manager.delete_dashboard(
        delete_candidate.token,
        destination=f"space:{second_space}",
    )
    assert manager.get_dashboard(delete_candidate.token) is None

    SESSION_DETAILS.space_ids = []
    ContextRegistry.clear()

    assert manager.get_dashboard(shared_dashboard.token) is None
    visible_dashboard_titles = {
        dashboard.title for dashboard in manager.list_dashboards()
    }
    assert visible_dashboard_titles >= {"Personal Dashboard"}
    assert "Patch Dashboard Updated" not in visible_dashboard_titles


def test_simulated_tile_destinations_match_real_visibility(simulated_dm):
    """Simulated tile rows honor destination writes and Read-A visibility."""
    original_space_ids = list(SESSION_DETAILS.space_ids)
    SESSION_DETAILS.space_ids = [7]
    try:
        personal = simulated_dm.create_tile("<p>Personal</p>", title="Personal")
        shared = simulated_dm.create_tile(
            "<p>Shared</p>",
            title="Shared",
            destination="space:7",
        )

        assert personal.succeeded
        assert shared.succeeded
        assert {tile.title for tile in simulated_dm.list_tiles()} == {
            "Personal",
            "Shared",
        }
        assert not simulated_dm.update_tile(shared.token, title="Wrong").succeeded
        assert simulated_dm.update_tile(
            shared.token,
            title="Right",
            destination="space:7",
        ).succeeded

        SESSION_DETAILS.space_ids = []
        assert simulated_dm.get_tile(shared.token) is None
        assert {tile.title for tile in simulated_dm.list_tiles()} == {"Personal"}
    finally:
        SESSION_DETAILS.space_ids = original_space_ids


def test_simulated_dashboard_destinations_match_real_visibility(simulated_dm):
    """Simulated dashboard rows honor destination writes and Read-A visibility."""
    original_space_ids = list(SESSION_DETAILS.space_ids)
    SESSION_DETAILS.space_ids = [7]
    try:
        personal = simulated_dm.create_dashboard("Personal Dashboard")
        shared = simulated_dm.create_dashboard(
            "Shared Dashboard",
            destination="space:7",
        )

        assert personal.succeeded
        assert shared.succeeded
        assert {dashboard.title for dashboard in simulated_dm.list_dashboards()} == {
            "Personal Dashboard",
            "Shared Dashboard",
        }
        assert not simulated_dm.update_dashboard(
            shared.token,
            title="Wrong",
        ).succeeded
        assert simulated_dm.update_dashboard(
            shared.token,
            title="Right",
            destination="space:7",
        ).succeeded

        SESSION_DETAILS.space_ids = []
        assert simulated_dm.get_dashboard(shared.token) is None
        assert {dashboard.title for dashboard in simulated_dm.list_dashboards()} == {
            "Personal Dashboard",
        }
    finally:
        SESSION_DETAILS.space_ids = original_space_ids
