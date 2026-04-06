"""Fixtures for DashboardManager tests."""

import pytest

from unity.dashboard_manager.simulated import SimulatedDashboardManager


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
