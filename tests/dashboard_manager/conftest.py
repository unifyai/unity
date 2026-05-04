"""Fixtures for DashboardManager tests."""

import time

import pytest

from unity.common.context_registry import ContextRegistry
from unity.dashboard_manager.simulated import SimulatedDashboardManager
from unity.session_details import SESSION_DETAILS


@pytest.fixture
def simulated_dm():
    """Fresh SimulatedDashboardManager instance."""
    return SimulatedDashboardManager()


@pytest.fixture
def seeded_dm(simulated_dm):
    """SimulatedDashboardManager pre-seeded with sample tiles and a dashboard."""
    t1 = simulated_dm.create_tile(
        "<h1>KPI Card</h1>",
        title="KPI Overview",
        description="Key performance indicators",
    )
    t2 = simulated_dm.create_tile(
        "<div id='chart'>chart placeholder</div>",
        title="Revenue Chart",
    )

    from unity.dashboard_manager.types.dashboard import TilePosition

    simulated_dm.create_dashboard(
        "Test Dashboard",
        tiles=[
            TilePosition(tile_token=t1.token, x=0, y=0, w=6, h=4),
            TilePosition(tile_token=t2.token, x=6, y=0, w=6, h=4),
        ],
    )
    return simulated_dm


@pytest.fixture
def dashboard_manager_spaces():
    """Assign unique shared-space memberships for dashboard routing tests."""
    base_id = time.time_ns() % 1_000_000_000
    space_ids = (base_id, base_id + 1)
    original_space_ids = list(SESSION_DETAILS.space_ids)
    SESSION_DETAILS.space_ids = list(space_ids)
    ContextRegistry.clear()
    try:
        yield space_ids
    finally:
        SESSION_DETAILS.space_ids = original_space_ids
        ContextRegistry.clear()
