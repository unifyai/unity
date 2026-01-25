import pytest
import re

from tests.helpers import _handle_project, get_or_create_contact
from tests.test_conversation_manager.conftest import BOSS
from tests.test_conversation_manager.test_actions.integration.helpers import (
    assert_no_errors,
    get_actor_started_event,
    wait_for_actor_completion,
    verify_contact_in_db,
)
from unity.conversation_manager.events import SMSReceived

pytestmark = [pytest.mark.integration, pytest.mark.eval]


@pytest.mark.asyncio
@pytest.mark.timeout(120)
@_handle_project
async def test_contact_query_smoke(initialized_cm_codeact):
    """Ask for a contact detail via CM→Actor and get back the correct phone number."""
    cm = initialized_cm_codeact

    result = await cm.step_until_wait(
        # Use an exact identifier (email) to avoid semantic search (which can require
        # embedding/derived-log infrastructure and be flaky in local test backends).
        SMSReceived(
            contact=BOSS,
            content="What's the phone number for the contact with email alice@example.com?",
        ),
    )

    actor_event = get_actor_started_event(result)
    handle_id = actor_event.handle_id

    final = await wait_for_actor_completion(cm, handle_id, timeout=90)

    assert "alice" in final.lower()
    digits = re.sub(r"\D", "", final)
    assert "15555552222" in digits, f"Expected Alice phone in result, got: {final}"
    assert_no_errors(result)


@pytest.mark.asyncio
@pytest.mark.timeout(120)
@_handle_project
async def test_contact_create_and_verify_db_smoke(initialized_cm_codeact):
    """Create a new contact via CM→Actor and verify it was written to ContactManager storage."""
    cm = initialized_cm_codeact

    result = await cm.step_until_wait(
        SMSReceived(
            contact=BOSS,
            content="Please save a new contact: Jane Doe, email jane@example.com",
        ),
    )

    actor_event = get_actor_started_event(result)
    handle_id = actor_event.handle_id

    _final = await wait_for_actor_completion(cm, handle_id, timeout=90)

    # Find the created contact by filtering ContactManager context (deterministic).
    payload = cm.cm.contact_manager.filter_contacts(
        filter="email_address == 'jane@example.com'",
        limit=5,
    )
    contacts = payload.get("contacts") or []
    assert (
        contacts
    ), f"Expected to find created contact for jane@example.com, got: {payload}"
    c0 = contacts[0]
    if isinstance(c0, dict):
        contact_id = int(c0.get("contact_id"))
    else:
        contact_id = int(getattr(c0, "contact_id"))

    verify_contact_in_db(
        cm,
        contact_id,
        expected_fields={
            "first_name": "Jane",
            "surname": "Doe",
            "email_address": "jane@example.com",
        },
    )
    assert_no_errors(result)


@pytest.mark.asyncio
@pytest.mark.timeout(120)
@_handle_project
async def test_contact_update_and_verify_db_smoke(initialized_cm_codeact):
    """Update an existing contact via CM→Actor and verify the change persisted."""
    cm = initialized_cm_codeact

    # Create a baseline contact deterministically.
    created = get_or_create_contact(
        cm.cm.contact_manager,
        first_name="Eve",
        surname="Adams",
        email_address="eve@example.com",
        phone_number="+15555550900",
    )
    assert created is not None

    result = await cm.step_until_wait(
        SMSReceived(
            contact=BOSS,
            content="Update the contact with email eve@example.com to have phone number +15555550901.",
        ),
    )

    actor_event = get_actor_started_event(result)
    handle_id = actor_event.handle_id
    _final = await wait_for_actor_completion(cm, handle_id, timeout=90)

    payload = cm.cm.contact_manager.filter_contacts(
        filter="email_address == 'eve@example.com'",
        limit=5,
    )
    contacts = payload.get("contacts") or []
    assert contacts
    c0 = contacts[0]
    contact_id = (
        int(c0.get("contact_id")) if isinstance(c0, dict) else int(c0.contact_id)
    )

    verify_contact_in_db(
        cm,
        contact_id,
        expected_fields={
            "email_address": "eve@example.com",
            "phone_number": "+15555550901",
        },
    )
    assert_no_errors(result)
