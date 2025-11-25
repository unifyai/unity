from __future__ import annotations

import asyncio

import pytest

from unity.conductor.simulated import SimulatedConductor
from tests.helpers import _handle_project
from tests.test_conductor.utils import (
    tool_names_from_messages,
    assistant_requested_tool_names,
)


MANAGER = "TaskScheduler"


# Each query intentionally contains an unrelated read (ask) and write (update)
COMBINED_REQUESTS: list[str] = [
    (
        "Which tasks are due tomorrow? Also create a new task: Call Alice about the Q3 budget tomorrow at 09:00."
    ),
    (
        "List all high-priority tasks. Also update the priority of 'Draft Budget FY26' to high."
    ),
    (
        "What tasks are assigned to Bob Johnson? Also delete the task named 'Old Onboarding Checklist'."
    ),
    (
        "Summarise tasks scheduled for next week. Also set 'Prepare slides for kickoff' to start today at 10:00."
    ),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("request_text", COMBINED_REQUESTS)
@_handle_project
async def test_combined_task_queries_call_both_ask_and_update_once_each(
    request_text: str,
):
    cond = SimulatedConductor(
        description=(
            "Assistant that can both read from and update the task list; combined queries include a separate question and update."
        ),
    )

    handle = await cond.request(
        request_text,
        _return_reasoning_steps=True,
    )

    answer, messages = await asyncio.wait_for(handle.result(), timeout=300)
    assert isinstance(answer, str) and answer.strip(), "Answer should be non-empty"

    executed_list = tool_names_from_messages(messages, MANAGER)
    executed = set(executed_list)
    assert executed, "Expected at least one tool call to occur"

    # Must include both at least once (dynamic continue tools permitted)
    assert (
        executed_list.count("TaskScheduler_ask") >= 1
    ), f"Expected at least one TaskScheduler_ask call, saw order: {executed_list}"
    assert (
        executed_list.count("TaskScheduler_update") >= 1
    ), f"Expected at least one TaskScheduler_update call, saw order: {executed_list}"

    # Allow at most one ContactManager_ask in addition to TaskScheduler ask/update
    contact_ask_count = executed_list.count("ContactManager_ask")
    assert (
        contact_ask_count <= 1
    ), f"At most one ContactManager_ask call allowed, saw {contact_ask_count} in order: {executed_list}"

    # Only these should have executed (TaskScheduler ask/update and optional single ContactManager_ask)
    allowed_tools = {"TaskScheduler_ask", "TaskScheduler_update", "ContactManager_ask"}
    assert {
        "TaskScheduler_ask",
        "TaskScheduler_update",
    }.issubset(
        executed,
    ), f"Both ask and update must be executed, saw: {sorted(executed)}"
    assert (
        executed <= allowed_tools
    ), f"Unexpected tools executed: {sorted(executed - allowed_tools)}"

    # Assistant tool requests should reference only ask/update (dynamic continues normalised)
    requested = set(assistant_requested_tool_names(messages, MANAGER))
    assert requested, "Assistant should have requested at least one tool"
    # Allow at most one ContactManager_ask request as well
    requested_list = assistant_requested_tool_names(messages, MANAGER)
    requested_contact_ask_count = requested_list.count("ContactManager_ask")
    assert (
        requested_contact_ask_count <= 1
    ), f"At most one ContactManager_ask request allowed, saw {requested_contact_ask_count} in order: {requested_list}"
    assert requested <= {
        "TaskScheduler_ask",
        "TaskScheduler_update",
        "ContactManager_ask",
    }, (
        f"Assistant should only request TaskScheduler ask/update and optional ContactManager_ask, "
        f"saw: {sorted(requested)}"
    )
