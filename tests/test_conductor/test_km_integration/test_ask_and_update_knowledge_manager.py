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


# Each query intentionally contains an unrelated read (ask) and write (update)
COMBINED_QUERIES: list[str] = [
    (
        "What are our office hours? Also store that the support inbox is monitored 7 days a week."
    ),
    (
        "Summarise the onboarding policy for engineers. Also add that a security review occurs in week two."
    ),
    (
        "Tell me the refund policy for ACME by date; also record that exchanges are allowed within 45 days."
    ),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("combined_text", COMBINED_QUERIES)
@_handle_project
async def test_combined_queries_call_both_ask_and_update_once_each(combined_text: str):
    cond = SimulatedConductor(
        description=(
            "Assistant that can both read from and update the knowledge-base; combined queries include a separate question and update."
        ),
    )

    handle = await cond.request(
        combined_text,
        _return_reasoning_steps=True,
    )

    answer, messages = await asyncio.wait_for(handle.result(), timeout=300)
    assert isinstance(answer, str) and answer.strip(), "Answer should be non-empty"

    executed_list = tool_names_from_messages(messages, MANAGER)
    executed = set(executed_list)
    assert executed, "Expected at least one tool call to occur"

    # Must include both at least once (dynamic continue tools permitted)
    assert (
        executed_list.count("SimulatedKnowledgeManager_ask") >= 1
    ), f"Expected at least one SimulatedKnowledgeManager_ask call, saw order: {executed_list}"
    assert (
        executed_list.count("SimulatedKnowledgeManager_update") >= 1
    ), f"Expected at least one SimulatedKnowledgeManager_update call, saw order: {executed_list}"

    # Order is not strictly enforced, but both must appear
    assert {
        "SimulatedKnowledgeManager_ask",
        "SimulatedKnowledgeManager_update",
    }.issubset(
        executed,
    ), f"Both ask and update must be executed, saw: {sorted(executed)}"

    # Assistant tool requests should reference only ask/update (dynamic continues normalised)
    requested = set(assistant_requested_tool_names(messages, MANAGER))
    assert requested, "Assistant should have requested at least one tool"
    assert requested <= {
        "SimulatedKnowledgeManager_ask",
        "SimulatedKnowledgeManager_update",
    }, f"Assistant should only request ask/update for KnowledgeManager, saw: {sorted(requested)}"
