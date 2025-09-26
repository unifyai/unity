from __future__ import annotations

import asyncio

import pytest

from unity.conductor.simulated import SimulatedConductor
from tests.helpers import _handle_project
from tests.test_conductor.utils import (
    tool_names_from_messages,
    assistant_requested_tool_names,
)


MANAGER = "SimulatedKnowledgeManager"


UPDATE_QUERIES: list[str] = [
    "Store: Office hours are 9–5 PT.",
    "Add that Tesla's battery warranty is eight years.",
    "Create a knowledge entry: Our refund window is 30 days for unopened items.",
    "Update the onboarding policy to require security training in week one.",
]


@pytest.mark.asyncio
@pytest.mark.parametrize("request_text", UPDATE_QUERIES)
@_handle_project
async def test_update_only_queries_call_only_update_and_not_ask_first(
    request_text: str,
):
    cond = SimulatedConductor(
        description=(
            "Assistant maintaining a structured knowledge-base; update requests should directly write without preliminary reads."
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

    # Must only be SimulatedKnowledgeManager_update (dynamic continue permitted)
    assert executed <= {
        "SimulatedKnowledgeManager_update",
    }, f"Only SimulatedKnowledgeManager_update should run, saw: {sorted(executed)}"
    assert (
        executed_list[0] == "SimulatedKnowledgeManager_update"
    ), f"The first call must be SimulatedKnowledgeManager_update, saw order: {executed_list}"
    assert (
        executed_list.count("SimulatedKnowledgeManager_update") >= 1
    ), f"Expected at least one SimulatedKnowledgeManager_update call, saw order: {executed_list}"

    # Additionally ensure no ask() calls were requested by the assistant
    requested = set(assistant_requested_tool_names(messages, MANAGER))
    assert requested, "Assistant should have requested at least one tool"
    assert (
        "SimulatedKnowledgeManager_ask" not in requested
    ), f"Assistant must not request SimulatedKnowledgeManager_ask, saw: {sorted(requested)}"
