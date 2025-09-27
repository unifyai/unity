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


CONTACT_UPDATE_REQUESTS: list[str] = [
    "Create a new contact: Jane Doe, email jane.d@example.com, phone 15551234567.",
    "Update Bob Johnson's phone number to 555-222-1111.",
    "Delete the contact with email diana@themyscira.com.",
    (
        "Merge duplicate contacts named Alice into a single entry, preferring the one "
        "with email alice.smith@example.com."
    ),
]


@pytest.mark.eval
@pytest.mark.asyncio
@pytest.mark.parametrize("request_text", CONTACT_UPDATE_REQUESTS)
@_handle_project
async def test_contact_updates_call_contact_manager_update(request_text: str):
    cond = SimulatedConductor(
        description=(
            "Operations assistant managing contacts, tasks, transcripts, and knowledge."
        ),
    )

    handle = await cond.request(
        request_text,
        _return_reasoning_steps=True,
    )

    answer, messages = await asyncio.wait_for(handle.result(), timeout=300)
    assert isinstance(answer, str) and answer.strip(), "Answer should be non-empty"

    # Verify that only contact ask/update tools executed, and update ran at least once.
    executed_list = _tool_names_from_messages(messages)
    executed = set(executed_list)
    assert executed, "Expected at least one tool call to occur"
    assert (
        "ContactManager_update" in executed
    ), f"Expected SimulatedContactManager_update to run, saw: {sorted(executed)}"
    assert executed <= {
        "ContactManager_ask",
        "ContactManager_update",
    }, f"Unexpected tools executed: {sorted(executed - {'ContactManager_ask', 'ContactManager_update'})}"

    # It's fine if the assistant also requested SimulatedContactManager.ask; no assertion needed.
    _ = _assistant_requested_tool_names(messages)
