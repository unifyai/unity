from __future__ import annotations

import asyncio
import pytest

from unity.conductor.simulated import SimulatedConductor
from tests.helpers import _handle_project
from tests.test_conductor.utils import (
    tool_names_from_messages,
    assistant_requested_tool_names,
)


MANAGER = "GlobalFileManager"


FILE_QUESTIONS: list[str] = [
    "List available filesystems and provide a brief inventory overview.",
    "Summarise available documents across all filesystems.",
]


@pytest.mark.eval
@pytest.mark.asyncio
@pytest.mark.parametrize("question", FILE_QUESTIONS)
@_handle_project
async def test_file_manager_questions_use_global_file_manager_ask(question: str):
    cond = SimulatedConductor(
        description=(
            "Assistant that can reason across contacts, knowledge, tasks, and filesystems."
        ),
    )

    handle = await cond.ask(
        question,
        _return_reasoning_steps=True,
    )

    answer, messages = await asyncio.wait_for(handle.result(), timeout=300)
    assert isinstance(answer, str) and answer.strip(), "Answer should be non-empty"

    # Confirm that GlobalFileManager_ask was executed at least once
    executed_list = tool_names_from_messages(messages, MANAGER)
    executed = set(executed_list)
    assert executed, "Expected at least one tool call to occur"
    assert (
        "GlobalFileManager_ask" in executed
        or "GlobalFileManager_list_filesystems" in executed
    ), f"Expected GlobalFileManager tool to run, saw: {sorted(executed)}"

    # Additionally confirm the assistant considered that tool
    requested = set(assistant_requested_tool_names(messages, MANAGER))
    assert requested, "Assistant should have requested at least one tool"
    assert (
        "GlobalFileManager_ask" in requested
        or "GlobalFileManager_list_filesystems" in requested
    ), f"Assistant should request FileManager tools, saw: {sorted(requested)}"
