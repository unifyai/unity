import pytest

from unity.contact_manager.contact_manager import ContactManager
from tests.helpers import _handle_project


@pytest.mark.unit
@_handle_project
def test_dummy_assistant_created(monkeypatch):
    """When the account has no assistants, a dummy assistant with ID 0 should be created."""

    # Force _fetch_assistant_info to return an empty list
    monkeypatch.setattr(ContactManager, "_fetch_assistant_info", lambda self: [])

    cm = ContactManager()

    assistants = cm._search_contacts(filter="contact_id == 0")
    assert len(assistants) == 1, "Exactly one assistant contact (ID 0) should exist"

    a = assistants[0]
    assert a.first_name == "Unify"
    assert a.surname == "Assistant"
    assert a.email_address == "unify.assistant@unify.ai"
    assert a.phone_number == "+10000000000"


@pytest.mark.unit
@_handle_project
def test_real_assistant_synced(monkeypatch):
    """If exactly one assistant is returned by the API, its details should populate contact ID 0."""

    sample_info = [
        {
            "agent_id": "123",
            "first_name": "Alice",
            "surname": "Smith",
            "phone": "+15551234567",
            "email": "alice.smith@example.com",
            "about": "Helpful assistant",
            "region": "North America",
        },
    ]

    monkeypatch.setattr(
        ContactManager,
        "_fetch_assistant_info",
        lambda self: sample_info,
    )

    cm = ContactManager()

    assistants = cm._search_contacts(filter="contact_id == 0")
    assert len(assistants) == 1

    a = assistants[0]
    # Core fields mapped directly
    assert a.first_name == "Alice"
    assert a.surname == "Smith"
    assert a.email_address == "alice.smith@example.com"
    assert a.phone_number == "+15551234567"
