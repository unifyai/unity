"""
CodeActActor tests for DataManager operations (simulated managers).

Data operations (filter, reduce, join) may require multi-step composition,
so ``execute_code`` is acceptable here. Both tools are exposed; the primary
assertion is correct routing to data primitives.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from tests.actor.state_managers.utils import make_code_act_actor
from unity.manager_registry import ManagerRegistry

pytestmark = [pytest.mark.eval, pytest.mark.llm_call]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _safe_call(method: Any, *args: Any, **kwargs: Any) -> Any:
    """Call a method that may be sync or async (due to in-place patching)."""
    if asyncio.iscoroutinefunction(method):
        return await method(*args, **kwargs)
    return method(*args, **kwargs)


async def _seed_repairs(dm: Any) -> None:
    """Seed a repairs table with sample data."""
    await _safe_call(
        dm.create_table,
        context="repairs",
        fields={
            "repair_id": "int",
            "description": "str",
            "priority": "str",
            "cost": "float",
        },
    )
    await _safe_call(
        dm.insert_rows,
        context="repairs",
        rows=[
            {
                "repair_id": 1,
                "description": "Engine overhaul",
                "priority": "high",
                "cost": 5000.0,
            },
            {
                "repair_id": 2,
                "description": "Oil change",
                "priority": "low",
                "cost": 150.0,
            },
            {
                "repair_id": 3,
                "description": "Brake replacement",
                "priority": "high",
                "cost": 2500.0,
            },
            {
                "repair_id": 4,
                "description": "Tire rotation",
                "priority": "medium",
                "cost": 200.0,
            },
            {
                "repair_id": 5,
                "description": "Transmission repair",
                "priority": "high",
                "cost": 8000.0,
            },
        ],
    )


async def _seed_monthly_stats(dm: Any) -> None:
    """Seed a monthly_stats table with sample data."""
    await _safe_call(
        dm.create_table,
        context="monthly_stats",
        fields={"region": "str", "month": "str", "amount": "int"},
    )
    await _safe_call(
        dm.insert_rows,
        context="monthly_stats",
        rows=[
            {"region": "North", "month": "Jan", "amount": 1200},
            {"region": "South", "month": "Jan", "amount": 800},
            {"region": "North", "month": "Feb", "amount": 1500},
            {"region": "South", "month": "Feb", "amount": 600},
            {"region": "North", "month": "Mar", "amount": 900},
        ],
    )


async def _seed_arrears(dm: Any) -> None:
    """Seed an arrears table with sample data."""
    await _safe_call(
        dm.create_table,
        context="arrears",
        fields={
            "account_id": "str",
            "account_name": "str",
            "overdue_amount": "float",
        },
    )
    await _safe_call(
        dm.insert_rows,
        context="arrears",
        rows=[
            {
                "account_id": "ACC-001",
                "account_name": "Acme Corp",
                "overdue_amount": 1250.0,
            },
            {
                "account_id": "ACC-002",
                "account_name": "Beta Inc",
                "overdue_amount": 320.0,
            },
            {
                "account_id": "ACC-003",
                "account_name": "Gamma LLC",
                "overdue_amount": 875.0,
            },
            {
                "account_id": "ACC-004",
                "account_name": "Delta Co",
                "overdue_amount": 150.0,
            },
        ],
    )


async def _seed_payments(dm: Any) -> None:
    """Seed a payments table with sample data."""
    await _safe_call(
        dm.create_table,
        context="payments",
        fields={
            "account_id": "str",
            "payment_amount": "float",
            "status": "str",
        },
    )
    await _safe_call(
        dm.insert_rows,
        context="payments",
        rows=[
            {"account_id": "ACC-001", "payment_amount": 500.0, "status": "partial"},
            {"account_id": "ACC-002", "payment_amount": 320.0, "status": "paid"},
            {"account_id": "ACC-003", "payment_amount": 200.0, "status": "partial"},
        ],
    )


async def _seed_tenants(dm: Any) -> None:
    """Seed a tenants table with sample data."""
    await _safe_call(
        dm.create_table,
        context="tenants",
        fields={"tenant_id": "int", "tenant_name": "str", "unit": "str"},
    )
    await _safe_call(
        dm.insert_rows,
        context="tenants",
        rows=[
            {"tenant_id": 1, "tenant_name": "Alice Smith", "unit": "A1"},
            {"tenant_id": 2, "tenant_name": "Bob Jones", "unit": "B3"},
            {"tenant_id": 3, "tenant_name": "Carol Lee", "unit": "C2"},
        ],
    )


async def _seed_repairs_with_tenant_id(dm: Any) -> None:
    """Seed a repairs table that references tenants via tenant_id."""
    await _safe_call(
        dm.create_table,
        context="repairs",
        fields={
            "repair_id": "int",
            "tenant_id": "int",
            "description": "str",
            "cost": "float",
        },
    )
    await _safe_call(
        dm.insert_rows,
        context="repairs",
        rows=[
            {
                "repair_id": 1,
                "tenant_id": 1,
                "description": "Leaky faucet",
                "cost": 150.0,
            },
            {
                "repair_id": 2,
                "tenant_id": 2,
                "description": "Broken window",
                "cost": 300.0,
            },
            {"repair_id": 3, "tenant_id": 1, "description": "AC repair", "cost": 500.0},
        ],
    )


# ---------------------------------------------------------------------------
# Seeding map: question → seeder functions
# ---------------------------------------------------------------------------

_FILTER_SEEDERS = {
    "Filter the repairs table to show only high priority items.": [_seed_repairs],
    "Get all rows from the Data/Pipeline/monthly_stats context where amount > 1000.": [
        _seed_monthly_stats,
    ],
    "Query the arrears dataset and filter for overdue amounts greater than 500.": [
        _seed_arrears,
    ],
}

_REDUCE_SEEDERS = {
    "Calculate the total sum of the 'amount' column in the repairs dataset.": [
        _seed_repairs,
    ],
    "What is the average repair cost in the Data/Repairs/2024 context?": [
        _seed_repairs,
    ],
    "Count how many rows are in the monthly_stats table, grouped by region.": [
        _seed_monthly_stats,
    ],
}

_JOIN_SEEDERS = {
    "Join the repairs table with the tenants table on tenant_id.": [
        _seed_repairs_with_tenant_id,
        _seed_tenants,
    ],
    "Combine the arrears and payments data to find outstanding balances.": [
        _seed_arrears,
        _seed_payments,
    ],
}


DATA_FILTER_QUESTIONS: list[str] = list(_FILTER_SEEDERS.keys())
DATA_REDUCE_QUESTIONS: list[str] = list(_REDUCE_SEEDERS.keys())
DATA_JOIN_QUESTIONS: list[str] = list(_JOIN_SEEDERS.keys())


@pytest.mark.asyncio
@pytest.mark.timeout(240)
@pytest.mark.parametrize("question", DATA_FILTER_QUESTIONS)
async def test_code_act_filter_questions_use_data_primitives(
    question: str,
):
    """Verify CodeActActor calls primitives.data.* for filter queries (both tools exposed)."""
    async with make_code_act_actor(impl="simulated") as (actor, _primitives, calls):
        dm = ManagerRegistry.get_data_manager()
        for seeder in _FILTER_SEEDERS[question]:
            await seeder(dm)

        handle = await actor.act(
            f"{question} Do not ask clarifying questions. Proceed with the best interpretation.",
            clarification_enabled=False,
        )
        result = await handle.result()

        assert result is not None
        assert calls, "Expected at least one state manager call."
        data_calls = [c for c in calls if "data" in c]
        assert data_calls, f"Expected data primitive calls, saw: {calls}"


@pytest.mark.asyncio
@pytest.mark.timeout(240)
@pytest.mark.parametrize("question", DATA_REDUCE_QUESTIONS)
async def test_code_act_reduce_questions_use_data_primitives(
    question: str,
):
    """Verify CodeActActor calls primitives.data.* for aggregation queries (both tools exposed)."""
    async with make_code_act_actor(impl="simulated") as (actor, _primitives, calls):
        dm = ManagerRegistry.get_data_manager()
        for seeder in _REDUCE_SEEDERS[question]:
            await seeder(dm)

        handle = await actor.act(
            f"{question} Do not ask clarifying questions. Proceed with the best interpretation.",
            clarification_enabled=False,
        )
        result = await handle.result()

        assert result is not None
        assert calls, "Expected at least one state manager call."
        data_calls = [c for c in calls if "data" in c]
        assert data_calls, f"Expected data primitive calls, saw: {calls}"


@pytest.mark.asyncio
@pytest.mark.timeout(240)
@pytest.mark.parametrize("question", DATA_JOIN_QUESTIONS)
async def test_code_act_join_questions_use_data_primitives(
    question: str,
):
    """Verify CodeActActor calls primitives.data.* for join queries (both tools exposed)."""
    async with make_code_act_actor(impl="simulated") as (actor, _primitives, calls):
        dm = ManagerRegistry.get_data_manager()
        for seeder in _JOIN_SEEDERS[question]:
            await seeder(dm)

        handle = await actor.act(
            f"{question} Do not ask clarifying questions. Proceed with the best interpretation.",
            clarification_enabled=False,
        )
        result = await handle.result()

        assert result is not None
        assert calls, "Expected at least one state manager call."
        data_calls = [c for c in calls if "data" in c]
        assert data_calls, f"Expected data primitive calls, saw: {calls}"
