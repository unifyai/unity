from __future__ import annotations

import asyncio
import pytest

from unity.conductor.simulated import SimulatedConductor
from unity.contact_manager.contact_manager import ContactManager

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
    """Prevent network access during ContactManager initialisation.

    Mirrors the approach used in contact-manager tests so creating a real
    ContactManager inside Conductor does not attempt external calls.
    """

    import unity

    # Clear any previously cached assistant record
    monkeypatch.setattr(unity, "ASSISTANT", None, raising=False)
    # Ensure discovery returns no real assistants
    monkeypatch.setattr(unity, "_list_all_assistants", lambda: [], raising=False)

    # Stub user info fetch to a local, deterministic value
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
#  Real Conductor → ContactManager.ask
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@_handle_project
async def test_real_conductor_contact_ask_calls_contact_manager():
    # Seed a contact via the real ContactManager
    cm = ContactManager()
    cm._create_contact(
        first_name="Eve",
        surname="Adams",
        email_address="eve.adams@example.com",
    )

    # SimulatedConductor wired to the real ContactManager instance
    cond = SimulatedConductor(contact_manager=cm)

    handle = await cond.ask(
        "What is Eve Adams' email address?",
        _return_reasoning_steps=True,
    )
    answer, messages = await asyncio.wait_for(handle.result(), timeout=300)

    # Basic content check
    assert isinstance(answer, str) and "eve.adams@example.com" in answer.lower()

    # Ensure ContactManager.ask was invoked (and nothing else from any manager)
    executed_list = tool_names_from_messages(messages, "ContactManager")
    requested_list = assistant_requested_tool_names(messages, "ContactManager")
    assert executed_list, "Expected at least one tool call"
    assert set(executed_list) == {
        "ContactManager_ask",
    }, f"Only ContactManager_ask should run; saw: {sorted(set(executed_list))}"
    assert (
        executed_list.count("ContactManager_ask") == 1
    ), f"Expected exactly one ContactManager_ask call, saw order: {executed_list}"
    assert set(requested_list) <= {
        "ContactManager_ask",
    }, f"Assistant should request only ContactManager_ask, saw: {sorted(set(requested_list))}"

    # Global exclusivity: verify no other manager tools ran
    all_tool_names = [
        str(m.get("name"))
        for m in messages
        if m.get("role") == "tool"
        and not str(m.get("name") or "").startswith("check_status_")
    ]
    assert all_tool_names, "Expected at least one tool call overall"
    assert all(
        n.startswith("ContactManager_ask")
        or n.startswith("continue_ContactManager_ask")
        for n in all_tool_names
    ), f"Unexpected tools executed: {sorted(set(all_tool_names))}"
    assert (
        len(all_tool_names) == 1
    ), f"Only one tool call expected; saw: {all_tool_names}"


# ---------------------------------------------------------------------------
#  Real Conductor → ContactManager.update
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@_handle_project
async def test_real_conductor_contact_update_calls_contact_manager():
    # Seed a contact we will modify
    cm = ContactManager()
    cm._create_contact(
        first_name="Bob",
        surname="Test",
        email_address="bob.update@example.com",
    )

    cond = SimulatedConductor(contact_manager=cm)

    request_text = "Update the contact with email bob.update@example.com: set the phone to 555-777-8888."
    handle = await cond.request(request_text, _return_reasoning_steps=True)
    answer, messages = await asyncio.wait_for(handle.result(), timeout=300)

    assert isinstance(answer, str) and answer.strip(), "Answer should be non-empty"

    # Ensure only ContactManager.update was invoked (not ask) and nothing else
    executed_list = tool_names_from_messages(messages, "ContactManager")
    requested_list = assistant_requested_tool_names(messages, "ContactManager")
    assert executed_list, "Expected at least one tool call"
    assert (
        executed_list[0] == "ContactManager_update"
    ), f"First call must be ContactManager_update, saw order: {executed_list}"
    assert set(executed_list) <= {
        "ContactManager_update",
    }, f"Only ContactManager_update should run, saw: {sorted(set(executed_list))}"
    assert "ContactManager_ask" not in set(
        executed_list,
    ), f"ContactManager_ask must not run, saw: {sorted(set(executed_list))}"
    assert set(requested_list) <= {
        "ContactManager_update",
    }, f"Assistant should request only ContactManager_update, saw: {sorted(set(requested_list))}"

    # Verify the mutation took effect
    rows = cm._filter_contacts(filter="email_address == 'bob.update@example.com'")
    assert rows and rows[0].phone_number == "5557778888"

    # Global exclusivity: verify no other manager tools ran
    all_tool_names = [
        str(m.get("name"))
        for m in messages
        if m.get("role") == "tool"
        and not str(m.get("name") or "").startswith("check_status_")
    ]
    assert all_tool_names, "Expected at least one tool call overall"
    assert all(
        n.startswith("ContactManager_update")
        or n.startswith("continue_ContactManager_update")
        for n in all_tool_names
    ), f"Unexpected tools executed: {sorted(set(all_tool_names))}"
    assert (
        len(all_tool_names) == 1
    ), f"Only one tool call expected; saw: {all_tool_names}"
