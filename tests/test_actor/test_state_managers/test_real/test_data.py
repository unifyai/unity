"""Real DataManager tests for Actor.

Tests that Actor correctly calls real DataManager methods and verifies
actual data operations against the Unify backend.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from tests.helpers import _handle_project
from tests.test_actor.test_state_managers.utils import (
    get_state_manager_tools,
    make_actor,
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
async def test_filter_calls_data_manager(mock_verification):
    """Test that Actor calls DataManager.filter for data filtering queries."""
    async with make_actor(impl="real") as actor:

        # Access real DataManager and create a test table with data
        dm = ManagerRegistry.get_data_manager()

        # Create test context and seed data
        test_context = "Data/Test/actor_filter_test"
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
                persist=False,
            )

            # Wait for result
            result = await handle.result()

            # Assert result mentions the filtered data
            assert "Item B" in result or "500" in result or "Item C" in result

            # Assert data primitives were called
            state_manager_tools = get_state_manager_tools(handle)
            data_tools = [t for t in state_manager_tools if "data" in t]
            assert (
                data_tools
            ), f"Expected data primitive calls, saw: {state_manager_tools}"

        finally:
            # Cleanup - handles both sync and async cases
            await _safe_call(dm.delete_table, context=test_context, dangerous_ok=True)


@pytest.mark.asyncio
@pytest.mark.timeout(300)
@pytest.mark.eval
@_handle_project
async def test_reduce_calls_data_manager(mock_verification):
    """Test that Actor calls DataManager.reduce for aggregation queries."""
    async with make_actor(impl="real") as actor:

        # Access real DataManager
        dm = ManagerRegistry.get_data_manager()

        # Create test context and seed data
        test_context = "Data/Test/actor_reduce_test"
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
                persist=False,
            )

            # Wait for result
            result = await handle.result()

            # Assert result contains the sum (4500)
            assert "4500" in result or "4,500" in result

            # Assert data primitives were called
            state_manager_tools = get_state_manager_tools(handle)
            data_tools = [t for t in state_manager_tools if "data" in t]
            assert (
                data_tools
            ), f"Expected data primitive calls, saw: {state_manager_tools}"

        finally:
            # Cleanup - handles both sync and async cases
            await _safe_call(dm.delete_table, context=test_context, dangerous_ok=True)


@pytest.mark.asyncio
@pytest.mark.timeout(300)
@pytest.mark.eval
@_handle_project
async def test_describe_table_calls_data_manager(mock_verification):
    """Test that Actor calls DataManager.describe_table for schema queries."""
    async with make_actor(impl="real") as actor:

        # Access real DataManager
        dm = ManagerRegistry.get_data_manager()

        # Create test context
        test_context = "Data/Test/actor_describe_test"
        await _safe_call(
            dm.create_table,
            context=test_context,
            fields={"id": "int", "name": "str", "score": "float"},
        )

        try:
            # Call actor with schema query
            handle = await actor.act(
                f"What columns are in the table at {test_context}? List them.",
                persist=False,
            )

            # Wait for result
            result = await handle.result()

            # Assert result mentions the columns
            result_lower = result.lower()
            assert (
                "id" in result_lower
                or "name" in result_lower
                or "score" in result_lower
            )

            # Assert data primitives were called
            state_manager_tools = get_state_manager_tools(handle)
            data_tools = [t for t in state_manager_tools if "data" in t]
            assert (
                data_tools
            ), f"Expected data primitive calls, saw: {state_manager_tools}"

        finally:
            # Cleanup - handles both sync and async cases
            await _safe_call(dm.delete_table, context=test_context, dangerous_ok=True)
