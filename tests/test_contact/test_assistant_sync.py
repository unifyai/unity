import pytest
import unity  # Added to patch global assistant

from unity.contact_manager.contact_manager import ContactManager
from tests.helpers import _handle_project


# ---------------------------------------------------------------------------
#  Test-local fixture – ensure deterministic assistant state
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clear_cached_assistant(monkeypatch):
    """Force *unity* to behave as if no real assistant were configured.

    We patch the global ``unity.ASSISTANT`` that `unity.init()` caches so that
    every time a ``ContactManager`` instance synchronises the assistant (id 0)
    it sees *None* and therefore falls back to the dummy placeholder record.

    The fixture is *autouse* and therefore applies to every test in this
    module without having to be listed explicitly.
    """

    # 1. Clear any previously cached assistant record (from earlier tests)
    monkeypatch.setattr(unity, "ASSISTANT", None, raising=False)

    # 2. Ensure future `unity.init()` calls cannot discover a real assistant
    #    by monkey-patching the internal helper it relies on.
    monkeypatch.setattr(
        unity,
        "_list_all_assistants",
        lambda: [],
        raising=False,
    )

    # 3. Prevent ContactManager from touching the network when synchronising
    #    the default *user* contact (id == 1). We replace the helper with a
    #    stub that returns an *empty* dict so no metadata is available but –
    #    crucially – the call succeeds without needing a real backend and
    #    without relying on ``unity.ASSISTANT`` being a mapping.
    from unity.contact_manager.contact_manager import ContactManager

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
