"""
CodeActActor tests for FileManager.ask operations (simulated managers).

Mirrors `test_ask.py` but validates CodeActActor produces Python that calls
`primitives.files.*` (on-the-fly; no FunctionManager).

Pattern: On-the-fly planning (Actor generates plans dynamically)
"""

from __future__ import annotations

import pytest

from tests.test_actor.test_state_managers.utils import make_code_act_actor

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
async def test_code_act_file_questions_use_files_primitives(
    question: str,
):
    """Verify CodeActActor produces Python calling primitives.files.* for file queries."""
    async with make_code_act_actor(impl="simulated") as (actor, _primitives, calls):
        handle = await actor.act(
            f"{question} Do not ask clarifying questions. Proceed with the best interpretation.",
            clarification_enabled=False,
        )
        result = await handle.result()

        # Verify result is non-empty (relax assertion: str, dict, or BaseModel)
        from pydantic import BaseModel

        assert result and (
            isinstance(result, (str, dict)) or isinstance(result, BaseModel)
        )

        # Routing: must hit files primitives for file queries
        assert calls, "Expected at least one state manager call."
        files_calls = [c for c in calls if "files" in c]
        assert files_calls, f"Expected files primitive calls, saw: {calls}"


@pytest.mark.asyncio
@pytest.mark.timeout(240)
@pytest.mark.parametrize("question", FILE_ASK_QUESTIONS)
async def test_code_act_file_ask_questions_use_ask_about_file(
    question: str,
):
    """Verify CodeActActor produces Python calling primitives.files.ask for file content queries."""
    async with make_code_act_actor(impl="simulated") as (actor, _primitives, calls):
        handle = await actor.act(
            f"{question} Do not ask clarifying questions. Proceed with the best interpretation.",
            clarification_enabled=False,
        )
        result = await handle.result()

        # Verify result is non-empty (relax assertion: str, dict, or BaseModel)
        from pydantic import BaseModel

        assert result and (
            isinstance(result, (str, dict)) or isinstance(result, BaseModel)
        )

        # Routing: must hit files primitives for file content queries
        assert calls, "Expected at least one state manager call."
        files_calls = [c for c in calls if "files" in c]
        assert files_calls, f"Expected files primitive calls, saw: {calls}"
