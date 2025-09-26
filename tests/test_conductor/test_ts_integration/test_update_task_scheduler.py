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


UPDATE_QUERIES: list[str] = [
    "Create a new task: Call Alice about the Q3 budget tomorrow at 09:00.",
    "Update the priority of 'Draft Budget FY26' to high.",
    "Delete the task named 'Old Onboarding Checklist'.",
    "Create a task to email Contoso about invoices and set it due next Friday.",
]


@pytest.mark.asyncio
@pytest.mark.parametrize("request_text", UPDATE_QUERIES)
@_handle_project
async def test_update_only_queries_call_only_update_and_not_ask_first(
    request_text: str,
):
    cond = SimulatedConductor(
        description=(
            "Assistant maintaining a task list; update requests should directly write without preliminary reads."
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

    # Must only be SimulatedTaskScheduler_update (dynamic continue permitted)
    assert executed <= {
        "SimulatedTaskScheduler_update",
    }, f"Only SimulatedTaskScheduler_update should run, saw: {sorted(executed)}"
    assert (
        executed_list[0] == "SimulatedTaskScheduler_update"
    ), f"The first call must be SimulatedTaskScheduler_update, saw order: {executed_list}"
    assert (
        executed_list.count("SimulatedTaskScheduler_update") >= 1
    ), f"Expected at least one SimulatedTaskScheduler_update call, saw order: {executed_list}"

    # Additionally ensure no ask() calls were requested by the assistant
    requested = set(assistant_requested_tool_names(messages, MANAGER))
    assert requested, "Assistant should have requested at least one tool"
    assert (
        "SimulatedTaskScheduler_ask" not in requested
    ), f"Assistant must not request SimulatedTaskScheduler_ask, saw: {sorted(requested)}"
