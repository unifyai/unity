import pytest
import re

from tests.helpers import _handle_project
from tests.test_conversation_manager.conftest import BOSS
from tests.test_conversation_manager.test_actions.integration.helpers import (
    assert_no_errors,
    get_actor_started_event,
    wait_for_actor_completion,
    verify_contact_in_db,
)
from unity.conversation_manager.events import SMSReceived

pytestmark = [pytest.mark.integration, pytest.mark.codeact, pytest.mark.eval]


@pytest.mark.asyncio
@pytest.mark.timeout(120)
@_handle_project
async def test_contact_query_smoke(initialized_cm_codeact):
    """Smoke: CM → CodeActActor queries ContactManager and returns correct contact info."""
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
    """Smoke: CM → CodeActActor creates a contact and we verify DB state."""
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
