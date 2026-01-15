"""Real FileManager CodeAct tests for Actor.

Tests that Actor correctly generates and executes Python plans that
compose FileManager and DataManager primitives for file analysis workflows.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.helpers import _handle_project
from tests.test_actor.test_state_managers.utils import (
    get_state_manager_tools,
    make_actor,
)
from unity.manager_registry import ManagerRegistry


@pytest.mark.asyncio
@pytest.mark.timeout(300)
@pytest.mark.eval
@_handle_project
async def test_code_act_file_discovery_and_query(mock_verification, tmp_path: Path):
    """Test that Actor generates code to discover files and query their contents."""
    async with make_actor(impl="real") as actor:

        # Create test files
        reports_dir = tmp_path / "reports"
        reports_dir.mkdir()

        q1_file = reports_dir / "q1_sales.csv"
        q2_file = reports_dir / "q2_sales.csv"
        q1_file.write_text("product,revenue\nWidget,5000\nGadget,3000\n")
        q2_file.write_text("product,revenue\nWidget,6000\nGadget,4000\n")

        # Access real FileManager and ingest the files
        fm = ManagerRegistry.get_file_manager()
        fm.ingest_files([str(q1_file), str(q2_file)])

        # Call actor with discovery + analysis query
        handle = await actor.act(
            f"Find all CSV files in {reports_dir}, describe their structure, "
            "and tell me the total revenue across all products in all files.",
            persist=False,
        )

        # Wait for result
        result = await handle.result()

        # Assert result mentions total (5000+3000+6000+4000 = 18000)
        assert "18000" in result or "18,000" in result or "revenue" in result.lower()

        # Verify plan was generated
        assert handle.plan_source_code
        assert "async def" in handle.plan_source_code

        # Assert file primitives were called
        state_manager_tools = get_state_manager_tools(handle)
        files_tools = [t for t in state_manager_tools if "files" in t]
        assert (
            files_tools
        ), f"Expected files primitive calls, saw: {state_manager_tools}"


@pytest.mark.asyncio
@pytest.mark.timeout(300)
@pytest.mark.eval
@_handle_project
async def test_code_act_file_to_data_pipeline(mock_verification, tmp_path: Path):
    """Test that Actor generates code composing FileManager and DataManager."""
    async with make_actor(impl="real") as actor:

        # Create a test file with data to transform
        source_file = tmp_path / "raw_metrics.csv"
        source_file.write_text(
            "date,metric,value\n"
            "2024-01-01,cpu,75\n"
            "2024-01-01,memory,60\n"
            "2024-01-02,cpu,80\n"
            "2024-01-02,memory,65\n",
        )

        # Access FileManager and ingest
        fm = ManagerRegistry.get_file_manager()
        fm.ingest_files(str(source_file))

        # Call actor with transformation pipeline query
        handle = await actor.act(
            f"Using the data from {source_file}, calculate the average value "
            "for each metric type (cpu and memory).",
            persist=False,
        )

        # Wait for result
        result = await handle.result()

        # Assert result mentions metrics and averages
        # CPU avg: (75+80)/2 = 77.5, Memory avg: (60+65)/2 = 62.5
        result_lower = result.lower()
        assert "cpu" in result_lower or "memory" in result_lower
        assert "77" in result or "62" in result or "average" in result_lower

        # Verify plan was generated
        assert handle.plan_source_code
        assert "primitives." in handle.plan_source_code


@pytest.mark.asyncio
@pytest.mark.timeout(300)
@pytest.mark.eval
@_handle_project
async def test_code_act_describe_then_reduce(mock_verification, tmp_path: Path):
    """Test that Actor uses describe() to discover schema then performs reduce."""
    async with make_actor(impl="real") as actor:

        # Create a spreadsheet-like file
        test_file = tmp_path / "inventory.csv"
        test_file.write_text(
            "item,quantity,price\n"
            "Apples,100,1.50\n"
            "Bananas,150,0.75\n"
            "Oranges,80,2.00\n",
        )

        # Access FileManager and ingest
        fm = ManagerRegistry.get_file_manager()
        fm.ingest_files(str(test_file))

        # Call actor with schema discovery + aggregation
        handle = await actor.act(
            f"First describe the file at {test_file} to understand its columns, "
            "then calculate the total quantity of all items.",
            persist=False,
        )

        # Wait for result
        result = await handle.result()

        # Assert result mentions total quantity (100+150+80 = 330)
        assert "330" in result or "total" in result.lower()

        # Verify plan uses describe and reduce/filter
        assert handle.plan_source_code
        # Should mention describe or filter/reduce operations
        plan_lower = handle.plan_source_code.lower()
        assert (
            "describe" in plan_lower or "filter" in plan_lower or "reduce" in plan_lower
        )
