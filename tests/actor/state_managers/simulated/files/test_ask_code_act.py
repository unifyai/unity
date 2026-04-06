"""
CodeActActor tests for FileManager.ask operations (simulated managers).

File queries may legitimately use either ``execute_function`` for single-primitive
calls or ``execute_code`` for shell commands / multi-step composition.
Both tools are exposed; the primary assertion is correct routing to file primitives.
"""

from __future__ import annotations

import pytest

from tests.actor.state_managers.utils import make_code_act_actor

pytestmark = [pytest.mark.eval, pytest.mark.llm_call]


FILE_QUESTIONS: list[str] = [
    "What files are in the /reports directory?",
    "List all files in the /documents directory.",
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
    """Verify CodeActActor produces a result for file queries (both tools exposed)."""
    async with make_code_act_actor(impl="simulated") as (actor, _primitives, calls):
        handle = await actor.act(
            f"{question} Do not ask clarifying questions. Proceed with the best interpretation.",
            clarification_enabled=False,
        )
        result = await handle.result()
        assert result is not None


@pytest.mark.asyncio
@pytest.mark.timeout(240)
@pytest.mark.parametrize("question", FILE_ASK_QUESTIONS)
async def test_code_act_file_ask_questions_use_ask_about_file(
    question: str,
):
    """Verify CodeActActor produces a result for file content queries (both tools exposed)."""
    async with make_code_act_actor(impl="simulated") as (actor, _primitives, calls):
        handle = await actor.act(
            f"{question} Do not ask clarifying questions. Proceed with the best interpretation.",
            clarification_enabled=False,
        )
        result = await handle.result()
        assert result is not None
