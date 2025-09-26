from __future__ import annotations

import asyncio

import pytest

from unity.conductor.simulated import SimulatedConductor
from tests.helpers import _handle_project
from tests.test_conductor.utils import (
    tool_names_from_messages,
    assistant_requested_tool_names,
)


MANAGER = "SimulatedTaskScheduler"


TASK_QUESTIONS: list[str] = [
    "Which tasks are due today?",
    "List all high-priority tasks.",
    "What tasks are scheduled for tomorrow?",
    "Show tasks assigned to Alice that are still open.",
]


@pytest.mark.eval
@pytest.mark.asyncio
@pytest.mark.parametrize("question", TASK_QUESTIONS)
@_handle_project
async def test_task_questions_use_only_task_scheduler_tool(question: str):
    cond = SimulatedConductor(
        description=(
            "Assistant focused on tasks; contacts, transcripts, and knowledge exist but are not needed for these queries."
        ),
    )

    handle = await cond.ask(
        question,
        _return_reasoning_steps=True,
    )

    answer, messages = await asyncio.wait_for(handle.result(), timeout=300)
    assert isinstance(answer, str) and answer.strip(), "Answer should be non-empty"

    # The only executed tool must be SimulatedTaskScheduler.ask and it should run exactly once
    executed_list = tool_names_from_messages(messages, MANAGER)
    executed = set(executed_list)
    assert executed, "Expected at least one tool call to occur"
    assert executed == {
        "SimulatedTaskScheduler_ask",
    }, f"Only SimulatedTaskScheduler_ask should run, saw: {sorted(executed)}"
    assert (
        executed_list.count("SimulatedTaskScheduler_ask") == 1
    ), f"Expected exactly one SimulatedTaskScheduler_ask call, saw order: {executed_list}"

    # Additionally confirm that any assistant tool selection(s) referenced only that tool
    requested = set(assistant_requested_tool_names(messages, MANAGER))
    assert requested, "Assistant should have requested at least one tool"
    assert requested <= {
        "SimulatedTaskScheduler_ask",
    }, f"Assistant should request only SimulatedTaskScheduler_ask, saw: {sorted(requested)}"
