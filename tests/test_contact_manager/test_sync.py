import pytest
import unity

from unity.contact_manager.contact_manager import ContactManager
from unity.session_details import SESSION_DETAILS
from tests.helpers import _handle_project


# ---------------------------------------------------------------------------
#  Test-local fixture – ensure deterministic assistant state
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clear_cached_assistant(monkeypatch):
    """Force *unity* to behave as if no real assistant were configured.

    We clear SESSION_DETAILS.assistant_record so that every time a
    ``ContactManager`` instance synchronises the assistant (id 0) it sees
    *None* and therefore falls back to the dummy placeholder record.

    The fixture is *autouse* and therefore applies to every test in this
    module without having to be listed explicitly.
    """

    # 1. Clear any previously cached assistant record (from earlier tests)
    SESSION_DETAILS.assistant_record = None

    # 2. Ensure future `unity.init()` calls cannot discover a real assistant
    #    by monkey-patching the internal helper it relies on.
    monkeypatch.setattr(
        unity,
        "_list_all_assistants",
        lambda: [],
        raising=False,
    )

    # Note: With SESSION_DETAILS.is_initialized=False (the default in tests),
    # _resolve_user_details automatically returns defaults without API calls.
    # No additional patching needed.


@_handle_project
def test_dummy_assistant(monkeypatch):
    """When the account has no assistants, a default assistant with ID 0 should be created."""
    from unity.session_details import (
        DEFAULT_ASSISTANT_EMAIL,
        DEFAULT_ASSISTANT_FIRST_NAME,
        DEFAULT_ASSISTANT_PHONE,
        DEFAULT_ASSISTANT_SURNAME,
    )

    # Force assistant discovery helper to return an empty list
    monkeypatch.setattr(
        "unity.contact_manager.system_contacts._list_assistants",
        lambda self: [],
        raising=True,
    )

    cm = ContactManager()

    assistants = cm.filter_contacts(filter="contact_id == 0")["contacts"]
    assert len(assistants) == 1, "Exactly one assistant contact (ID 0) should exist"

    a = assistants[0]
    assert a.first_name == DEFAULT_ASSISTANT_FIRST_NAME
    assert a.surname == DEFAULT_ASSISTANT_SURNAME
    assert a.email_address == DEFAULT_ASSISTANT_EMAIL
    assert a.phone_number == DEFAULT_ASSISTANT_PHONE
    # System contact timezone should be hard-coded to UTC for now
    assert a.timezone == "UTC"

    # Default user (id 1) should also have UTC for now
    users = cm.filter_contacts(filter="contact_id == 1")["contacts"]
    assert users, "Default user should exist"
    assert users[0].timezone == "UTC"


@_handle_project
def test_real_assistant(monkeypatch):
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
        "unity.contact_manager.system_contacts._list_assistants",
        lambda self: sample_info,
        raising=True,
    )

    cm = ContactManager()

    assistants = cm.filter_contacts(filter="contact_id == 0")["contacts"]
    assert len(assistants) == 1

    a = assistants[0]
    # Core fields mapped directly
    assert a.first_name == "Alice"
    assert a.surname == "Smith"
    assert a.email_address == "alice.smith@example.com"
    assert a.phone_number == "+15551234567"
    # System contact timezone should be hard-coded to UTC for now
    assert a.timezone == "UTC"

    users = cm.filter_contacts(filter="contact_id == 1")["contacts"]
    assert users, "Default user should exist"
    assert users[0].timezone == "UTC"
