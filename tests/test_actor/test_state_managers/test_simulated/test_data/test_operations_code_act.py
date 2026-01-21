"""
CodeActActor tests for DataManager operations (simulated managers).

Mirrors `test_operations.py` but validates CodeActActor produces Python that calls
`primitives.data.*` (on-the-fly; no FunctionManager).

Pattern: On-the-fly planning (Actor generates plans dynamically)
"""

from __future__ import annotations

import pytest

from tests.test_actor.test_state_managers.utils import make_code_act_actor

pytestmark = pytest.mark.eval


DATA_FILTER_QUESTIONS: list[str] = [
    "Filter the repairs table to show only high priority items.",
    "Get all rows from the Data/Pipeline/monthly_stats context where amount > 1000.",
    "Query the arrears dataset and filter for overdue amounts greater than 500.",
]

DATA_REDUCE_QUESTIONS: list[str] = [
    "Calculate the total sum of the 'amount' column in the repairs dataset.",
    "What is the average repair cost in the Data/Repairs/2024 context?",
    "Count how many rows are in the monthly_stats table, grouped by region.",
]

DATA_JOIN_QUESTIONS: list[str] = [
    "Join the repairs table with the tenants table on tenant_id.",
    "Combine the arrears and payments data to find outstanding balances.",
]


@pytest.mark.asyncio
@pytest.mark.timeout(240)
@pytest.mark.parametrize("question", DATA_FILTER_QUESTIONS)
async def test_code_act_filter_questions_use_data_primitives(
    question: str,
):
    """Verify CodeActActor produces Python calling primitives.data.* for filter queries."""
    async with make_code_act_actor(impl="simulated") as (actor, _primitives, calls):
        handle = await actor.act(
            f"{question} Do not ask clarifying questions. Proceed with the best interpretation.",
            clarification_enabled=False,
        )
        result = await handle.result()

        # Verify result is non-empty
        assert isinstance(result, str) and result.strip()

        # Routing: must hit data primitives for data queries
        assert calls, "Expected at least one state manager call."
        data_calls = [c for c in calls if "data" in c]
        assert data_calls, f"Expected data primitive calls, saw: {calls}"


@pytest.mark.asyncio
@pytest.mark.timeout(240)
@pytest.mark.parametrize("question", DATA_REDUCE_QUESTIONS)
async def test_code_act_reduce_questions_use_data_primitives(
    question: str,
):
    """Verify CodeActActor produces Python calling primitives.data.* for aggregation queries."""
    async with make_code_act_actor(impl="simulated") as (actor, _primitives, calls):
        handle = await actor.act(
            f"{question} Do not ask clarifying questions. Proceed with the best interpretation.",
            clarification_enabled=False,
        )
        result = await handle.result()

        # Verify result is non-empty
        assert isinstance(result, str) and result.strip()

        # Routing: must hit data primitives for aggregation
        assert calls, "Expected at least one state manager call."
        data_calls = [c for c in calls if "data" in c]
        assert data_calls, f"Expected data primitive calls, saw: {calls}"


@pytest.mark.asyncio
@pytest.mark.timeout(240)
@pytest.mark.parametrize("question", DATA_JOIN_QUESTIONS)
async def test_code_act_join_questions_use_data_primitives(
    question: str,
):
    """Verify CodeActActor produces Python calling primitives.data.* for join queries."""
    async with make_code_act_actor(impl="simulated") as (actor, _primitives, calls):
        handle = await actor.act(
            f"{question} Do not ask clarifying questions. Proceed with the best interpretation.",
            clarification_enabled=False,
        )
        result = await handle.result()

        # Verify result is non-empty
        assert isinstance(result, str) and result.strip()

        # Routing: must hit data primitives for join
        assert calls, "Expected at least one state manager call."
        data_calls = [c for c in calls if "data" in c]
        assert data_calls, f"Expected data primitive calls, saw: {calls}"
