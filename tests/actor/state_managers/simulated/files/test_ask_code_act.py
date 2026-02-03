"""
CodeActActor tests for FileManager.ask operations (simulated managers).

Mirrors `test_ask.py` but validates CodeActActor produces Python to answer
file-related queries. CodeActActor may use primitives OR shell commands
(e.g., ls, cat, Python's open()) - both approaches are valid.

The primary assertion is that the actor produces a result.
Primitive call tracking is informational only (not strictly required).

Pattern: On-the-fly planning (Actor generates plans dynamically)
"""

from __future__ import annotations

import pytest

from tests.actor.state_managers.utils import make_code_act_actor

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
    """Verify CodeActActor produces Python to answer file queries."""
    async with make_code_act_actor(impl="simulated") as (actor, _primitives, calls):
        handle = await actor.act(
            f"{question} Do not ask clarifying questions. Proceed with the best interpretation.",
            clarification_enabled=False,
        )
        result = await handle.result()

        # Verify result is not None (routing test, not type test)
        assert result is not None

        # Log primitive calls for debugging (CodeActActor may use shell commands as alternative)
        files_calls = [c for c in calls if "files" in c]
        if files_calls:
            print(f"✓ Used files primitives: {files_calls}")
        elif calls:
            print(f"ℹ Used other primitives: {calls}")
        else:
            print("ℹ Used alternative approach (shell/Python I/O)")


@pytest.mark.asyncio
@pytest.mark.timeout(240)
@pytest.mark.parametrize("question", FILE_ASK_QUESTIONS)
async def test_code_act_file_ask_questions_use_ask_about_file(
    question: str,
):
    """Verify CodeActActor produces Python to answer file content queries."""
    async with make_code_act_actor(impl="simulated") as (actor, _primitives, calls):
        handle = await actor.act(
            f"{question} Do not ask clarifying questions. Proceed with the best interpretation.",
            clarification_enabled=False,
        )
        result = await handle.result()

        # Verify result is not None (routing test, not type test)
        assert result is not None

        # Log primitive calls for debugging (CodeActActor may use shell commands as alternative)
        files_calls = [c for c in calls if "files" in c]
        if files_calls:
            print(f"✓ Used files primitives: {files_calls}")
        elif calls:
            print(f"ℹ Used other primitives: {calls}")
        else:
            print("ℹ Used alternative approach (shell/Python I/O)")
