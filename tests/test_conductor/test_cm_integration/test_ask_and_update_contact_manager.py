from __future__ import annotations

import asyncio

import pytest

from unity.conductor.simulated import SimulatedConductor
from tests.helpers import _handle_project
from tests.test_conductor.utils import (
    tool_names_from_messages,
    assistant_requested_tool_names,
)


MANAGER = "SimulatedContactManager"


COMBINED_REQUESTS: list[str] = [
    (
        "Ask for the current phone number for Bob Johnson using the contacts manager. "
        "Also update his phone number to 555-222-3333. "
        "Use only ContactManager.ask and ContactManager.update; do not use any other tools."
    ),
    (
        "Ask for the total number of contacts currently stored using the contacts manager. "
        "Also set Jane Doe's description to 'Preferred contact is email'. "
        "Use only ContactManager.ask and ContactManager.update; do not use any other tools."
    ),
    (
        "Answer this question using the contacts manager – what is Alice Smith's current email? "
        "Also update Alice Smith's WhatsApp number to +1-555-101-2020. "
        "Use only ContactManager.ask and ContactManager.update; no other tools."
    ),
]


@pytest.mark.eval
@pytest.mark.asyncio
@pytest.mark.parametrize("request_text", COMBINED_REQUESTS)
@_handle_project
async def test_contact_combined_ask_and_update_only_expected_tools(request_text: str):
    cond = SimulatedConductor(
        description=(
            "Operations assistant managing contacts; tasks, transcripts, and knowledge are irrelevant here."
        ),
    )

    handle = await cond.request(
        request_text,
        _return_reasoning_steps=True,
    )

    answer, messages = await asyncio.wait_for(handle.result(), timeout=300)
    assert isinstance(answer, str) and answer.strip(), "Answer should be non-empty"

    executed_list = tool_names_from_messages(messages, MANAGER)
    requested_list = assistant_requested_tool_names(messages, MANAGER)
    executed = set(executed_list)
    requested = set(requested_list)

    # Must include both ask and update in the executed tools
    assert executed, "Expected at least some tool activity"
    assert {
        "SimulatedContactManager_ask",
        "SimulatedContactManager_update",
    }.issubset(
        executed,
    ), f"Expected both SimulatedContactManager_ask and SimulatedContactManager_update to run; saw: {sorted(executed)}"

    # Strictly no other tools should have been executed
    assert executed <= {
        "SimulatedContactManager_ask",
        "SimulatedContactManager_update",
    }, f"Unexpected tools executed: {sorted(executed - {'SimulatedContactManager_ask', 'SimulatedContactManager_update'})}"

    # Assistant's requested tools should also be only these two (order/duplicates allowed)
    assert requested, "Assistant should have requested tools"
    assert requested <= {
        "SimulatedContactManager_ask",
        "SimulatedContactManager_update",
    }, f"Unexpected tool requests: {sorted(requested - {'SimulatedContactManager_ask', 'SimulatedContactManager_update'})}"

    # Tighter checks: at least one ask and at least one update call were executed
    assert (
        executed_list.count("SimulatedContactManager_ask") >= 1
    ), "Expected at least one ask execution"
    assert (
        executed_list.count("SimulatedContactManager_update") >= 1
    ), "Expected at least one update execution"

    # Order is not enforced; tools may be called in any sequence or in parallel.
