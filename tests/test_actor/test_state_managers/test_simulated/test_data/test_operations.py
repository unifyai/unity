"""
Actor tests for DataManager operations.

Tests that HierarchicalActor correctly generates plans calling `primitives.data.*`
for data query, join, and aggregation operations.

Pattern: On-the-fly planning (Actor generates plans dynamically)
"""

from __future__ import annotations

import pytest

from tests.test_actor.test_state_managers.utils import (
    get_state_manager_tools,
    make_hierarchical_actor,
)

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
async def test_filter_questions_use_data_primitives(
    question: str,
    mock_verification,
):
    """Verify Actor generates plans calling primitives.data.filter for filter queries."""
    async with make_hierarchical_actor(impl="simulated") as actor:

        handle = await actor.act(
            f"{question} Do not ask clarifying questions. Generate the full plan.",
            persist=False,
        )
        result = await handle.result()

        # Verify result is non-empty
        assert isinstance(result, str) and result.strip()

        # Verify plan was generated
        assert handle.plan_source_code
        assert "async def" in handle.plan_source_code
        assert "primitives." in handle.plan_source_code

        # Verify data primitives were called
        state_manager_tools = get_state_manager_tools(handle)
        assert state_manager_tools, "Expected at least one state manager tool call"

        # Should call data.filter or data.search
        data_tools = [t for t in state_manager_tools if "data" in t]
        assert data_tools, f"Expected data primitive calls, saw: {state_manager_tools}"


@pytest.mark.asyncio
@pytest.mark.timeout(240)
@pytest.mark.parametrize("question", DATA_REDUCE_QUESTIONS)
async def test_reduce_questions_use_data_primitives(
    question: str,
    mock_verification,
):
    """Verify Actor generates plans calling primitives.data.reduce for aggregation queries."""
    async with make_hierarchical_actor(impl="simulated") as actor:

        handle = await actor.act(
            f"{question} Do not ask clarifying questions. Generate the full plan.",
            persist=False,
        )
        result = await handle.result()

        # Verify result is non-empty
        assert isinstance(result, str) and result.strip()

        # Verify plan was generated
        assert handle.plan_source_code
        assert "async def" in handle.plan_source_code

        # Verify data primitives were called
        state_manager_tools = get_state_manager_tools(handle)
        data_tools = [t for t in state_manager_tools if "data" in t]
        assert data_tools, f"Expected data primitive calls, saw: {state_manager_tools}"


@pytest.mark.asyncio
@pytest.mark.timeout(240)
@pytest.mark.parametrize("question", DATA_JOIN_QUESTIONS)
async def test_join_questions_use_data_primitives(
    question: str,
    mock_verification,
):
    """Verify Actor generates plans calling primitives.data.*_join for join queries."""
    async with make_hierarchical_actor(impl="simulated") as actor:

        handle = await actor.act(
            f"{question} Do not ask clarifying questions. Generate the full plan.",
            persist=False,
        )
        result = await handle.result()

        # Verify result is non-empty
        assert isinstance(result, str) and result.strip()

        # Verify plan was generated
        assert handle.plan_source_code
        assert "async def" in handle.plan_source_code

        # Verify data primitives were called
        state_manager_tools = get_state_manager_tools(handle)
        data_tools = [t for t in state_manager_tools if "data" in t]
        assert data_tools, f"Expected data primitive calls, saw: {state_manager_tools}"
