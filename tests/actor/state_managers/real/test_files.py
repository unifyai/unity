"""Real FileManager tests for Actor.

Tests that Actor correctly calls real FileManager methods and verifies
actual file operations against the Unify backend.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.helpers import _handle_project
from tests.actor.state_managers.utils import (
    get_state_manager_tools,
    make_hierarchical_actor,
)
from unity.manager_registry import ManagerRegistry


@pytest.mark.asyncio
@pytest.mark.timeout(300)
@pytest.mark.eval
@_handle_project
async def test_describe_calls_file_manager(mock_verification, tmp_path: Path):
    """Test that Actor calls FileManager.describe for file discovery."""
    async with make_hierarchical_actor(impl="real") as actor:

        # Create a test file
        test_file = tmp_path / "actor_test_report.csv"
        test_file.write_text("name,value\nAlice,100\nBob,200\nCharlie,300\n")

        # Access real FileManager and ingest the file
        fm = ManagerRegistry.get_file_manager()
        fm.ingest_files(str(test_file))

        # Call actor with file description query
        handle = await actor.act(
            f"Describe the storage layout of the file at {test_file}. What contexts and tables does it have?",
            persist=False,
        )

        # Wait for result
        result = await handle.result()

        # Assert result mentions file storage concepts (handle str, dict, or BaseModel)
        result_lower = str(result).lower()
        assert any(
            term in result_lower
            for term in ["context", "table", "column", "storage", "csv"]
        )

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
async def test_filter_files_calls_file_manager(mock_verification, tmp_path: Path):
    """Test that Actor calls FileManager.filter_files for file queries."""
    async with make_hierarchical_actor(impl="real") as actor:

        # Create test files
        test_file1 = tmp_path / "sales_q1.csv"
        test_file2 = tmp_path / "sales_q2.csv"
        test_file1.write_text("quarter,revenue\nQ1,1000\n")
        test_file2.write_text("quarter,revenue\nQ2,1500\n")

        # Access real FileManager and ingest the files
        fm = ManagerRegistry.get_file_manager()
        fm.ingest_files([str(test_file1), str(test_file2)])

        # Call actor with file list query
        handle = await actor.act(
            f"What files have been ingested from the {tmp_path} directory?",
            persist=False,
        )

        # Wait for result
        result = await handle.result()

        # Assert result mentions the files (handle str, dict, or BaseModel)
        result_lower = str(result).lower()
        assert "sales" in result_lower or "csv" in result_lower or "q1" in result_lower

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
async def test_ask_about_file_calls_file_manager(mock_verification, tmp_path: Path):
    """Test that Actor calls FileManager.ask for file content queries."""
    async with make_hierarchical_actor(impl="real") as actor:

        # Create a test file with content
        test_file = tmp_path / "company_report.txt"
        test_file.write_text(
            "Annual Report 2024\n\n"
            "Revenue: $5.2 million\n"
            "Employees: 150\n"
            "Headquarters: San Francisco\n",
        )

        # Access real FileManager and ingest the file
        fm = ManagerRegistry.get_file_manager()
        fm.ingest_files(str(test_file))

        # Call actor with content query
        handle = await actor.act(
            f"What is the revenue mentioned in the file at {test_file}?",
            persist=False,
        )

        # Wait for result
        result = await handle.result()

        # Assert result mentions the revenue (handle str, dict, or BaseModel)
        result_str = str(result)
        assert "5.2" in result_str or "million" in result_str.lower()

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
async def test_file_manager_delegates_to_data_manager(
    mock_verification,
    tmp_path: Path,
):
    """Test that FileManager correctly delegates data operations to DataManager."""
    async with make_hierarchical_actor(impl="real") as actor:

        # Create a CSV with numeric data for aggregation
        test_file = tmp_path / "metrics.csv"
        test_file.write_text("metric,value\nCPU,75\nMemory,60\nDisk,45\n")

        # Access real FileManager and ingest the file
        fm = ManagerRegistry.get_file_manager()
        fm.ingest_files(str(test_file))

        # Get the storage map to find the table context
        storage = fm.describe(file_path=str(test_file))

        # Call actor with aggregation query on file data
        handle = await actor.act(
            f"What is the average of the 'value' column in the file at {test_file}?",
            persist=False,
        )

        # Wait for result
        result = await handle.result()

        # Assert result mentions the average (60) (handle str, dict, or BaseModel)
        result_str = str(result)
        assert "60" in result_str or "average" in result_str.lower()

        # Assert either files or data primitives were called (delegation may happen internally)
        state_manager_tools = get_state_manager_tools(handle)
        relevant_tools = [t for t in state_manager_tools if "files" in t or "data" in t]
        assert (
            relevant_tools
        ), f"Expected files/data primitive calls, saw: {state_manager_tools}"
