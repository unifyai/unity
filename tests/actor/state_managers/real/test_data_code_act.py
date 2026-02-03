"""Real DataManager CodeAct tests for Actor.

Tests that CodeActActor correctly generates Python code that
composes DataManager primitives for complex data workflows.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from tests.helpers import _handle_project
from tests.actor.state_managers.utils import (
    make_code_act_actor,
)
from unity.manager_registry import ManagerRegistry


async def _safe_call(method, *args, **kwargs) -> Any:
    """
    Call a method that may be sync or async (due to in-place patching).

    After primitives.data is accessed, the DataManager singleton gets patched
    to have async methods. This helper handles both cases.
    """
    if asyncio.iscoroutinefunction(method):
        return await method(*args, **kwargs)
    else:
        return method(*args, **kwargs)


@pytest.mark.asyncio
@pytest.mark.timeout(300)
@pytest.mark.eval
@_handle_project
async def test_code_act_composes_data_operations():
    """Test that CodeActActor generates code composing multiple DataManager operations."""
    async with make_code_act_actor(impl="real") as (actor, _primitives, calls):

        # Access real DataManager and create test tables
        dm = ManagerRegistry.get_data_manager()

        # Create sales table
        sales_context = "Data/Test/code_act_sales"
        await _safe_call(
            dm.create_table,
            context=sales_context,
            fields={"region": "str", "amount": "int"},
        )
        await _safe_call(
            dm.insert_rows,
            context=sales_context,
            rows=[
                {"region": "North", "amount": 1000},
                {"region": "South", "amount": 2000},
                {"region": "North", "amount": 1500},
                {"region": "South", "amount": 500},
            ],
        )

        try:
            # Call actor with complex query requiring composition
            handle = await actor.act(
                f"Using the data in {sales_context}, calculate the total amount per region "
                "and tell me which region has the higher total.",
                clarification_enabled=False,
            )

            # Wait for result
            result = await handle.result()

            # Assert result mentions regions and comparison
            result_lower = result.lower()
            assert "north" in result_lower or "south" in result_lower

            # Assert data primitives were called
            data_calls = [c for c in calls if "data" in c]
            assert data_calls, f"Expected data primitive calls, saw: {calls}"

        finally:
            # Cleanup - handles both sync and async cases
            await _safe_call(dm.delete_table, context=sales_context, dangerous_ok=True)


@pytest.mark.asyncio
@pytest.mark.timeout(300)
@pytest.mark.eval
@_handle_project
async def test_code_act_data_pipeline():
    """Test that CodeActActor generates code for a data transformation pipeline."""
    async with make_code_act_actor(impl="real") as (actor, _primitives, calls):

        # Access real DataManager and create source table
        dm = ManagerRegistry.get_data_manager()

        source_context = "Data/Test/code_act_source"
        await _safe_call(
            dm.create_table,
            context=source_context,
            fields={"name": "str", "score": "int", "category": "str"},
        )
        await _safe_call(
            dm.insert_rows,
            context=source_context,
            rows=[
                {"name": "Alice", "score": 85, "category": "A"},
                {"name": "Bob", "score": 72, "category": "B"},
                {"name": "Charlie", "score": 91, "category": "A"},
                {"name": "Diana", "score": 68, "category": "B"},
            ],
        )

        try:
            # Call actor with pipeline query
            handle = await actor.act(
                f"From the data in {source_context}, filter to only category 'A' entries, "
                "then calculate the average score of those entries.",
                clarification_enabled=False,
            )

            # Wait for result
            result = await handle.result()

            # Assert result mentions the average (Alice: 85, Charlie: 91 -> avg = 88)
            assert "88" in result or "average" in result.lower()

            # Assert data primitives were called
            data_calls = [c for c in calls if "data" in c]
            assert data_calls, f"Expected data primitive calls, saw: {calls}"

        finally:
            # Cleanup - handles both sync and async cases
            await _safe_call(dm.delete_table, context=source_context, dangerous_ok=True)


@pytest.mark.asyncio
@pytest.mark.timeout(300)
@pytest.mark.eval
@_handle_project
async def test_code_act_filter_calls_data_manager():
    """Test that CodeActActor calls DataManager.filter for data filtering queries."""
    async with make_code_act_actor(impl="real") as (actor, _primitives, calls):

        # Access real DataManager and create a test table with data
        dm = ManagerRegistry.get_data_manager()

        # Create test context and seed data
        test_context = "Data/Test/code_act_filter_test"
        await _safe_call(
            dm.create_table,
            context=test_context,
            fields={"name": "str", "amount": "int", "region": "str"},
        )
        await _safe_call(
            dm.insert_rows,
            context=test_context,
            rows=[
                {"name": "Item A", "amount": 100, "region": "North"},
                {"name": "Item B", "amount": 500, "region": "South"},
                {"name": "Item C", "amount": 200, "region": "North"},
            ],
        )

        try:
            # Call actor with data query
            handle = await actor.act(
                f"Filter the data in {test_context} to show only items where amount > 150.",
                clarification_enabled=False,
            )

            # Wait for result
            result = await handle.result()

            # Assert result mentions the filtered data
            assert "Item B" in result or "500" in result or "Item C" in result

            # Assert data primitives were called
            data_calls = [c for c in calls if "data" in c]
            assert data_calls, f"Expected data primitive calls, saw: {calls}"

        finally:
            # Cleanup - handles both sync and async cases
            await _safe_call(dm.delete_table, context=test_context, dangerous_ok=True)


@pytest.mark.asyncio
@pytest.mark.timeout(300)
@pytest.mark.eval
@_handle_project
async def test_code_act_reduce_calls_data_manager():
    """Test that CodeActActor calls DataManager.reduce for aggregation queries."""
    async with make_code_act_actor(impl="real") as (actor, _primitives, calls):

        # Access real DataManager
        dm = ManagerRegistry.get_data_manager()

        # Create test context and seed data
        test_context = "Data/Test/code_act_reduce_test"
        await _safe_call(
            dm.create_table,
            context=test_context,
            fields={"product": "str", "revenue": "int"},
        )
        await _safe_call(
            dm.insert_rows,
            context=test_context,
            rows=[
                {"product": "Widget", "revenue": 1000},
                {"product": "Gadget", "revenue": 2000},
                {"product": "Doohickey", "revenue": 1500},
            ],
        )

        try:
            # Call actor with aggregation query
            handle = await actor.act(
                f"What is the total sum of the revenue column in {test_context}?",
                clarification_enabled=False,
            )

            # Wait for result
            result = await handle.result()

            # Assert result contains the sum (4500)
            assert "4500" in result or "4,500" in result

            # Assert data primitives were called
            data_calls = [c for c in calls if "data" in c]
            assert data_calls, f"Expected data primitive calls, saw: {calls}"

        finally:
            # Cleanup - handles both sync and async cases
            await _safe_call(dm.delete_table, context=test_context, dangerous_ok=True)
