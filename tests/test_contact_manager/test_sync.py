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
    """If a real assistant is configured, its details should populate contact ID 0."""
    from unity.session_details import SESSION_DETAILS

    sample_record = {
        "agent_id": "123",
        "first_name": "Alice",
        "surname": "Smith",
        "phone": "+15551234567",
        "email": "alice.smith@example.com",
        "about": "Helpful assistant",
        "region": "North America",
        "timezone": "America/New_York",
    }

    # Simulate a real session with an assistant record.
    # This is how production works: unity.init() sets assistant_record,
    # which _resolve_assistant_details() then uses to populate the contact.
    monkeypatch.setattr(SESSION_DETAILS, "_initialized", True)
    monkeypatch.setattr(SESSION_DETAILS, "assistant_record", sample_record)

    cm = ContactManager()

    assistants = cm.filter_contacts(filter="contact_id == 0")["contacts"]
    assert len(assistants) == 1

    a = assistants[0]
    # Core fields mapped directly
    assert a.first_name == "Alice"
    assert a.surname == "Smith"
    assert a.email_address == "alice.smith@example.com"
    assert a.phone_number == "+15551234567"
    # Timezone should be synced from the assistant record
    assert a.timezone == "America/New_York"

    users = cm.filter_contacts(filter="contact_id == 1")["contacts"]
    assert users, "Default user should exist"


@_handle_project
def test_system_contacts_have_is_system_flag(monkeypatch):
    """Assistant and user contacts should have is_system=True."""
    # Force assistant discovery helper to return an empty list
    monkeypatch.setattr(
        "unity.contact_manager.system_contacts._list_assistants",
        lambda self: [],
        raising=True,
    )

    cm = ContactManager()

    # Assistant (id=0) should have is_system=True
    assistants = cm.filter_contacts(filter="contact_id == 0")["contacts"]
    assert len(assistants) == 1, "Exactly one assistant contact (ID 0) should exist"
    assert (
        assistants[0].is_system is True
    ), "Assistant contact should have is_system=True"

    # User (id=1) should have is_system=True
    users = cm.filter_contacts(filter="contact_id == 1")["contacts"]
    assert len(users) == 1, "Exactly one user contact (ID 1) should exist"
    assert users[0].is_system is True, "User contact should have is_system=True"


# ---------------------------------------------------------------------------
#  Org member provisioning tests
# ---------------------------------------------------------------------------


@_handle_project
def test_org_members_provisioned_as_system_contacts(monkeypatch):
    """Org members should be created as system contacts with correct fields."""
    # Mock org members API to return test data
    fake_org_members = [
        {"email": "alice@org.com", "name": "Alice Johnson"},
        {"email": "bob@org.com", "name": "Bob"},  # Single name (no surname)
    ]
    monkeypatch.setattr(
        "unity.contact_manager.system_contacts._fetch_org_members",
        lambda: fake_org_members,
    )

    cm = ContactManager()

    # Find Alice
    alice_contacts = cm.filter_contacts(filter="email_address == 'alice@org.com'")[
        "contacts"
    ]
    assert len(alice_contacts) == 1, "Alice should be created"
    alice = alice_contacts[0]
    assert alice.first_name == "Alice"
    assert alice.surname == "Johnson"
    assert alice.is_system is True, "Org member should have is_system=True"
    assert alice.should_respond is True, "Org member should have should_respond=True"
    assert alice.response_policy == "", "Org member should have blank response_policy"

    # Find Bob (single name case)
    bob_contacts = cm.filter_contacts(filter="email_address == 'bob@org.com'")[
        "contacts"
    ]
    assert len(bob_contacts) == 1, "Bob should be created"
    bob = bob_contacts[0]
    assert bob.first_name == "Bob"
    assert bob.surname is None, "Single name should have no surname"
    assert bob.is_system is True


@_handle_project
def test_org_member_skips_primary_user_email(monkeypatch):
    """Org member with same email as primary user (id=1) should be skipped."""
    from unity.session_details import DEFAULT_USER_EMAIL

    # Include the primary user's email in org members list
    fake_org_members = [
        {"email": DEFAULT_USER_EMAIL, "name": "Primary User Duplicate"},
        {"email": "other@org.com", "name": "Other Member"},
    ]
    monkeypatch.setattr(
        "unity.contact_manager.system_contacts._fetch_org_members",
        lambda: fake_org_members,
    )

    cm = ContactManager()

    # Primary user should still be id=1, not duplicated
    primary_users = cm.filter_contacts(filter="contact_id == 1")["contacts"]
    assert len(primary_users) == 1
    # The name should NOT be overwritten to "Primary User Duplicate"
    assert primary_users[0].first_name != "Primary"

    # Other member should be created
    other_contacts = cm.filter_contacts(filter="email_address == 'other@org.com'")[
        "contacts"
    ]
    assert len(other_contacts) == 1, "Other org member should be created"


@_handle_project
def test_existing_contact_updated_to_system_for_org_member(monkeypatch):
    """If contact with org member email exists, it should be marked is_system=True."""
    # First create ContactManager with no org members to create a regular contact
    monkeypatch.setattr(
        "unity.contact_manager.system_contacts._fetch_org_members",
        lambda: [],
    )
    cm = ContactManager()

    # Create a regular contact
    result = cm._create_contact(
        first_name="PreExisting",
        email_address="preexisting@org.com",
    )
    cid = result["details"]["contact_id"]

    # Verify it's not a system contact
    contacts = cm.filter_contacts(filter=f"contact_id == {cid}")["contacts"]
    assert contacts[0].is_system is False

    # Now simulate org members API returning this email
    monkeypatch.setattr(
        "unity.contact_manager.system_contacts._fetch_org_members",
        lambda: [{"email": "preexisting@org.com", "name": "Pre Existing"}],
    )

    # Re-sync by calling provision directly
    from unity.contact_manager.system_contacts import provision_org_member_contacts

    provision_org_member_contacts(cm)

    # Contact should now be marked as system
    contacts = cm.filter_contacts(filter=f"contact_id == {cid}")["contacts"]
    assert contacts[0].is_system is True, "Existing contact should be updated to system"


@_handle_project
def test_no_org_members_when_personal_api_key(monkeypatch):
    """When _fetch_org_members returns empty (personal key), no extra contacts created."""
    monkeypatch.setattr(
        "unity.contact_manager.system_contacts._fetch_org_members",
        lambda: [],
    )

    cm = ContactManager()

    # Should only have assistant (0) and user (1)
    all_contacts = cm.filter_contacts()["contacts"]
    contact_ids = {c.contact_id for c in all_contacts}

    # Only system contacts should exist
    assert 0 in contact_ids, "Assistant should exist"
    assert 1 in contact_ids, "User should exist"
    # No other contacts created from org members
    assert len(contact_ids) == 2, "Only assistant and user should exist"
