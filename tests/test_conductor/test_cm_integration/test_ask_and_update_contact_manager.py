from __future__ import annotations

import asyncio

import pytest

from unity.conductor.simulated import SimulatedConductor
from tests.helpers import _handle_project


def _tool_names_from_messages(msgs: list[dict]) -> list[str]:
    names: list[str] = []
    for m in msgs:
        if m.get("role") == "tool":
            name = m.get("name") or ""
            if name and not str(name).startswith("check_status_"):
                names.append(str(name))
    return names


def _assistant_requested_tool_names(msgs: list[dict]) -> list[str]:
    names: list[str] = []
    for m in msgs:
        if m.get("role") == "assistant" and m.get("tool_calls"):
            for tc in m.get("tool_calls") or []:
                fn = (tc or {}).get("function", {}) or {}
                name = fn.get("name") or ""
                if name and not str(name).startswith("check_status_"):
                    names.append(str(name))
    return names


COMBINED_REQUESTS: list[str] = [
    (
        "First, look up the current phone number for Bob Johnson using the contacts manager, "
        "then update his phone number to 555-222-3333. Use only the ContactManager.ask and ContactManager.update tools; "
        "do not use any other tools."
    ),
    (
        "If there exists a contact with email jane.d@example.com (check using the contacts manager), "
        "update their description to 'Preferred contact is email'; otherwise do nothing and say 'no change'. "
        "Use only ContactManager.ask and ContactManager.update; do not use any other tools."
    ),
    (
        "Answer this question using the contacts manager – what is Alice Smith's current email? "
        "Then immediately update Alice Smith's WhatsApp number to +1-555-101-2020. "
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

    executed_list = _tool_names_from_messages(messages)
    requested_list = _assistant_requested_tool_names(messages)
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

    # Order assertion: the first ask should occur before the first update
    first_ask_idx = executed_list.index("SimulatedContactManager_ask")
    first_update_idx = executed_list.index("SimulatedContactManager_update")
    assert (
        first_ask_idx < first_update_idx
    ), f"Expected ask before update. Order was: {executed_list}"

    # If both tools were explicitly requested by the assistant, assert the first
    # request order mirrors ask before update (best-effort; skip if one missing)
    if (
        "SimulatedContactManager_ask" in requested_list
        and "SimulatedContactManager_update" in requested_list
    ):
        r_ask_idx = requested_list.index("SimulatedContactManager_ask")
        r_upd_idx = requested_list.index("SimulatedContactManager_update")
        assert (
            r_ask_idx < r_upd_idx
        ), f"Expected tool selection to request ask before update. Order was: {requested_list}"
