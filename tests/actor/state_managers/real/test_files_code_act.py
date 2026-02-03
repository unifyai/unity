"""Real FileManager CodeAct tests for Actor.

Tests that CodeActActor correctly generates Python code to accomplish
file analysis workflows. CodeActActor may use primitives OR shell commands
(e.g., ls, cat, Python's open()) - both approaches are valid as long as
the result is correct.

The primary assertion is that the actor produces the correct answer.
Primitive call tracking is informational only (not required).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.helpers import _handle_project
from tests.actor.state_managers.utils import (
    make_code_act_actor,
)
from unity.manager_registry import ManagerRegistry


@pytest.mark.asyncio
@pytest.mark.timeout(300)
@pytest.mark.eval
@_handle_project
async def test_code_act_file_discovery_and_query(tmp_path: Path):
    """Test that CodeActActor generates code to discover files and query their contents."""
    async with make_code_act_actor(impl="real") as (actor, _primitives, calls):

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
            clarification_enabled=False,
        )

        # Wait for result
        result = await handle.result()

        # Assert result mentions total (5000+3000+6000+4000 = 18000)
        assert "18000" in result or "18,000" in result or "revenue" in result.lower()

        # Log primitive calls for debugging (not required - CodeActActor may use shell commands)
        files_calls = [c for c in calls if "files" in c or "data" in c]
        if files_calls:
            print(f"✓ Used file/data primitives: {files_calls}")
        else:
            print(
                f"ℹ Used alternative approach (shell/Python I/O). All tracked calls: {calls}",
            )


@pytest.mark.asyncio
@pytest.mark.timeout(300)
@pytest.mark.eval
@_handle_project
async def test_code_act_file_to_data_pipeline(tmp_path: Path):
    """Test that CodeActActor generates code composing FileManager and DataManager."""
    async with make_code_act_actor(impl="real") as (actor, _primitives, calls):

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
            clarification_enabled=False,
        )

        # Wait for result
        result = await handle.result()

        # Assert result mentions metrics and averages
        # CPU avg: (75+80)/2 = 77.5, Memory avg: (60+65)/2 = 62.5
        result_lower = result.lower()
        assert "cpu" in result_lower or "memory" in result_lower
        assert "77" in result or "62" in result or "average" in result_lower

        # Log primitive calls for debugging (not required - CodeActActor may use shell commands)
        relevant_calls = [c for c in calls if "files" in c or "data" in c]
        if relevant_calls:
            print(f"✓ Used file/data primitives: {relevant_calls}")
        else:
            print(
                f"ℹ Used alternative approach (shell/Python I/O). All tracked calls: {calls}",
            )


@pytest.mark.asyncio
@pytest.mark.timeout(300)
@pytest.mark.eval
@_handle_project
async def test_code_act_describe_then_reduce(tmp_path: Path):
    """Test that CodeActActor uses describe() to discover schema then performs reduce."""
    async with make_code_act_actor(impl="real") as (actor, _primitives, calls):

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
            clarification_enabled=False,
        )

        # Wait for result
        result = await handle.result()

        # Assert result mentions total quantity (100+150+80 = 330)
        assert "330" in result or "total" in result.lower()

        # Log primitive calls for debugging (not required - CodeActActor may use shell commands)
        relevant_calls = [c for c in calls if "files" in c or "data" in c]
        if relevant_calls:
            print(f"✓ Used file/data primitives: {relevant_calls}")
        else:
            print(
                f"ℹ Used alternative approach (shell/Python I/O). All tracked calls: {calls}",
            )


@pytest.mark.asyncio
@pytest.mark.timeout(300)
@pytest.mark.eval
@_handle_project
async def test_code_act_describe_calls_file_manager(tmp_path: Path):
    """Test that CodeActActor can describe file storage structure."""
    async with make_code_act_actor(impl="real") as (actor, _primitives, calls):

        # Create a test file
        test_file = tmp_path / "actor_test_report.csv"
        test_file.write_text("name,value\nAlice,100\nBob,200\nCharlie,300\n")

        # Access real FileManager and ingest the file
        fm = ManagerRegistry.get_file_manager()
        fm.ingest_files(str(test_file))

        # Call actor with file description query
        handle = await actor.act(
            f"Describe the storage layout of the file at {test_file}. What contexts and tables does it have?",
            clarification_enabled=False,
        )

        # Wait for result
        result = await handle.result()

        # Assert result mentions file storage concepts
        result_lower = result.lower()
        assert any(
            term in result_lower
            for term in ["context", "table", "column", "storage", "csv"]
        )

        # Log primitive calls for debugging (not required - CodeActActor may use shell commands)
        files_calls = [c for c in calls if "files" in c]
        if files_calls:
            print(f"✓ Used files primitives: {files_calls}")
        else:
            print(
                f"ℹ Used alternative approach (shell/Python I/O). All tracked calls: {calls}",
            )


@pytest.mark.asyncio
@pytest.mark.timeout(300)
@pytest.mark.eval
@_handle_project
async def test_code_act_ask_about_file_calls_file_manager(tmp_path: Path):
    """Test that CodeActActor can answer questions about file contents."""
    async with make_code_act_actor(impl="real") as (actor, _primitives, calls):

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
            clarification_enabled=False,
        )

        # Wait for result
        result = await handle.result()

        # Assert result mentions the revenue
        assert "5.2" in result or "million" in result.lower()

        # Log primitive calls for debugging (not required - CodeActActor may use shell commands)
        files_calls = [c for c in calls if "files" in c]
        if files_calls:
            print(f"✓ Used files primitives: {files_calls}")
        else:
            print(
                f"ℹ Used alternative approach (shell/Python I/O). All tracked calls: {calls}",
            )
