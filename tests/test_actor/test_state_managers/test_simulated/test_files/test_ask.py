"""
Actor tests for FileManager.ask operations.

Tests that HierarchicalActor correctly generates plans calling `primitives.files.*`
for file-related queries and operations.

Pattern: On-the-fly planning (Actor generates plans dynamically)
"""

from __future__ import annotations

import pytest

from tests.test_actor.test_state_managers.utils import (
    get_state_manager_tools,
    make_hierarchical_actor,
)

pytestmark = pytest.mark.eval


FILE_QUESTIONS: list[str] = [
    "What files are in the /reports directory?",
    "Search for files containing 'quarterly revenue' in the documents folder.",
    "Describe the storage layout of the file at /data/Q4_2024.csv.",
    "What columns are in the spreadsheet at /reports/monthly.xlsx?",
]

FILE_ASK_QUESTIONS: list[str] = [
    "What is the main topic of the document at /reports/summary.pdf?",
    "Summarize the contents of the file at /docs/meeting_notes.docx.",
    "What data is in the CSV file at /data/sales.csv?",
]


@pytest.mark.asyncio
@pytest.mark.timeout(240)
@pytest.mark.parametrize("question", FILE_QUESTIONS)
async def test_file_questions_use_files_primitives(
    question: str,
    mock_verification,
):
    """Verify Actor generates plans calling primitives.files.* for file queries."""
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

        # Verify file primitives were called
        state_manager_tools = get_state_manager_tools(handle)
        assert state_manager_tools, "Expected at least one state manager tool call"

        # Should call files.* primitives
        files_tools = [t for t in state_manager_tools if "files" in t]
        assert (
            files_tools
        ), f"Expected files primitive calls, saw: {state_manager_tools}"


@pytest.mark.asyncio
@pytest.mark.timeout(240)
@pytest.mark.parametrize("question", FILE_ASK_QUESTIONS)
async def test_file_ask_questions_use_ask_about_file(
    question: str,
    mock_verification,
):
    """Verify Actor generates plans calling primitives.files.ask for file content queries."""
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

        # Verify file primitives were called
        state_manager_tools = get_state_manager_tools(handle)
        files_tools = [t for t in state_manager_tools if "files" in t]
        assert (
            files_tools
        ), f"Expected files primitive calls, saw: {state_manager_tools}"
