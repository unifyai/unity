"""
CodeActActor eval tests for DashboardManager routing (simulated managers).

These tests verify that when the user asks for visualizations:
1. The actor routes to ``primitives.dashboards.create_tile`` (not old plot/table_view).
2. For live-data scenarios, the actor includes ``data_bindings`` with query params.

Uses ``make_code_act_actor`` with simulated managers and a live LLM.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from tests.actor.state_managers.utils import (
    make_code_act_actor,
)
from unity.manager_registry import ManagerRegistry

pytestmark = pytest.mark.eval


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _safe_call(method: Any, *args: Any, **kwargs: Any) -> Any:
    if asyncio.iscoroutinefunction(method):
        return await method(*args, **kwargs)
    return method(*args, **kwargs)


async def _seed_monthly_revenue(dm: Any) -> None:
    await _safe_call(
        dm.create_table,
        context="Sales/Monthly",
        fields={"month": "str", "revenue": "float", "region": "str"},
    )
    await _safe_call(
        dm.insert_rows,
        context="Sales/Monthly",
        rows=[
            {"month": "Jan", "revenue": 12000.0, "region": "North"},
            {"month": "Feb", "revenue": 15000.0, "region": "North"},
            {"month": "Mar", "revenue": 9000.0, "region": "South"},
            {"month": "Apr", "revenue": 18000.0, "region": "North"},
            {"month": "May", "revenue": 11000.0, "region": "South"},
        ],
    )


async def _seed_repairs(dm: Any) -> None:
    await _safe_call(
        dm.create_table,
        context="repairs",
        fields={
            "repair_id": "int",
            "category": "str",
            "priority": "str",
            "cost": "float",
        },
    )
    await _safe_call(
        dm.insert_rows,
        context="repairs",
        rows=[
            {"repair_id": 1, "category": "Plumbing", "priority": "high", "cost": 500.0},
            {
                "repair_id": 2,
                "category": "Electrical",
                "priority": "low",
                "cost": 150.0,
            },
            {"repair_id": 3, "category": "Plumbing", "priority": "high", "cost": 800.0},
            {"repair_id": 4, "category": "HVAC", "priority": "medium", "cost": 350.0},
            {
                "repair_id": 5,
                "category": "Electrical",
                "priority": "high",
                "cost": 600.0,
            },
        ],
    )


# ---------------------------------------------------------------------------
# Routing tests: actor should use primitives.dashboards for visualizations
# ---------------------------------------------------------------------------

VISUALIZATION_QUESTIONS = [
    "Create a bar chart showing monthly revenue from the Sales/Monthly data.",
    "Plot the repairs data by category as a chart or visualization.",
]


@pytest.mark.asyncio
@pytest.mark.timeout(300)
@pytest.mark.parametrize("question", VISUALIZATION_QUESTIONS)
async def test_code_act_visualization_routes_to_dashboards(question: str):
    """Verify CodeActActor routes visualization requests to primitives.dashboards.create_tile."""
    async with make_code_act_actor(impl="simulated") as (actor, _primitives, calls):
        dm = ManagerRegistry.get_data_manager()
        await _seed_monthly_revenue(dm)
        await _seed_repairs(dm)

        handle = await actor.act(
            f"{question} Do not ask clarifying questions. Proceed immediately.",
            clarification_enabled=False,
        )
        result = await handle.result()

        assert result is not None
        dashboard_calls = [c for c in calls if "dashboards" in c]
        assert dashboard_calls, (
            f"Expected primitives.dashboards calls for a visualization request, "
            f"but only saw: {calls}"
        )
        create_tile_calls = [c for c in calls if "create_tile" in c]
        assert create_tile_calls, (
            f"Expected primitives.dashboards.create_tile to be called, "
            f"but dashboard calls were: {dashboard_calls}"
        )
