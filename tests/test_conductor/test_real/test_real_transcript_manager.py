from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import pytest

from unity.conductor.simulated import SimulatedConductor
from unity.transcript_manager.transcript_manager import TranscriptManager
from unity.contact_manager.contact_manager import ContactManager
from unity.contact_manager.types.contact import Contact

from tests.helpers import _handle_project
from tests.test_conductor.utils import (
    tool_names_from_messages,
    assistant_requested_tool_names,
)


# ---------------------------------------------------------------------------
#  Test-local fixture – ensure deterministic assistant/user sync
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _stub_assistant_and_user_sync(monkeypatch):
    """Prevent network access during manager initialisation.

    Mirrors the approach used in ContactManager real-integration tests so creating
    real manager instances inside Conductor does not attempt external calls.
    """

    import unity

    # Clear any previously cached assistant record
    monkeypatch.setattr(unity, "ASSISTANT", None, raising=False)
    # Ensure discovery returns no real assistants
    monkeypatch.setattr(unity, "_list_all_assistants", lambda: [], raising=False)

    # Stub user info fetch to a local, deterministic value for ContactManager
    monkeypatch.setattr(
        ContactManager,
        "_fetch_user_info",
        lambda self: {
            "first_name": "John",
            "last_name": "Doe",
            "email": "john.doe@example.com",
        },
        raising=False,
    )


# ---------------------------------------------------------------------------
#  Real Conductor → TranscriptManager.ask
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@_handle_project
async def test_real_conductor_transcript_ask_calls_transcript_manager():
    # Seed a message via the real TranscriptManager (with a real ContactManager)
    cm = ContactManager()
    tm = TranscriptManager(contact_manager=cm)

    # Create two contacts (will be created on-the-fly during log_messages)
    alice = Contact(
        first_name="Alice",
        surname="Smith",
        email_address="alice.smith@example.com",
    )
    bob = Contact(
        first_name="Bob",
        surname="Jones",
        email_address="bob.jones@example.com",
    )

    # Log a message that mentions the budget
    _ = tm.log_messages(
        {
            "medium": "email",
            "sender_id": alice,
            "receiver_ids": [bob],
            "timestamp": datetime.now(timezone.utc),
            "content": "Subject: Q3 Budget\nBody: Final numbers are ready for review.",
        },
        synchronous=True,
    )

    # Wire a SimulatedConductor to the real TranscriptManager + ContactManager instances
    cond = SimulatedConductor(contact_manager=cm, transcript_manager=tm)

    handle = await cond.ask(
        "Show the most recent message that mentions the budget.",
        _return_reasoning_steps=True,
    )
    answer, messages = await asyncio.wait_for(handle.result(), timeout=300)

    # Basic content check
    assert isinstance(answer, str) and answer.strip(), "Answer should be non-empty"

    # Ensure TranscriptManager.ask was invoked (and no other manager tools ran)
    executed_list = tool_names_from_messages(messages, "TranscriptManager")
    requested_list = assistant_requested_tool_names(messages, "TranscriptManager")
    assert executed_list, "Expected at least one tool call"
    assert set(executed_list) == {
        "TranscriptManager_ask",
    }, f"Only TranscriptManager_ask should run; saw: {sorted(set(executed_list))}"
    assert (
        executed_list.count("TranscriptManager_ask") == 1
    ), f"Expected exactly one TranscriptManager_ask call, saw order: {executed_list}"
    assert set(requested_list) <= {
        "TranscriptManager_ask",
    }, f"Assistant should request only TranscriptManager_ask, saw: {sorted(set(requested_list))}"

    # Global exclusivity: verify no other manager tools ran
    all_tool_names = [
        str(m.get("name"))
        for m in messages
        if m.get("role") == "tool"
        and not str(m.get("name") or "").startswith("check_status_")
    ]
    assert all_tool_names, "Expected at least one tool call overall"
    assert all(
        n.startswith("TranscriptManager_ask")
        or n.startswith("continue_TranscriptManager_ask")
        for n in all_tool_names
    ), f"Unexpected tools executed: {sorted(set(all_tool_names))}"
    assert (
        len(all_tool_names) == 1
    ), f"Only one tool call expected; saw: {all_tool_names}"
