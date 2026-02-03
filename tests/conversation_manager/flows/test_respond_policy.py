"""
tests/conversation_manager/flows/test_respond_policy.py
=================================================================

Tests for the `should_respond` boolean flag and `response_policy` text field
on contacts. These tests verify:

1. The `should_respond` flag is presented to the brain in contact_index
2. Outbound communication is blocked when `should_respond=False`
3. The `response_policy` text influences assistant behavior

Uses the CMStepDriver "step_until_wait" API pattern from test_comms.py.
"""

import pytest

from tests.helpers import _handle_project, get_or_create_contact
from tests.conversation_manager.cm_helpers import (
    filter_events_by_type,
    assert_has_one,
)
from unity.conversation_manager.events import (
    SMSReceived,
    SMSSent,
    EmailReceived,
    EmailSent,
    UnifyMessageReceived,
    UnifyMessageSent,
)

pytestmark = pytest.mark.eval


# ---------------------------------------------------------------------------
#  Test contact definitions (contact_id populated dynamically)
# ---------------------------------------------------------------------------

# Contact that CAN be responded to (should_respond=True)
CONTACT_RESPOND_YES_DEF = {
    "first_name": "Alice",
    "surname": "Responder",
    "email_address": "alice@respond.com",
    "phone_number": "+15555550010",
    "should_respond": True,
    "response_policy": "Respond promptly and professionally to all inquiries.",
}

# Contact that CANNOT be responded to (should_respond=False)
CONTACT_RESPOND_NO_DEF = {
    "first_name": "Bob",
    "surname": "NoContact",
    "email_address": "bob@nocontact.com",
    "phone_number": "+15555550011",
    "should_respond": False,
    "response_policy": "Do not respond - this contact has opted out of communications.",
}

# Contact with should_respond=False but no response_policy
CONTACT_RESPOND_NO_POLICY_DEF = {
    "first_name": "Carol",
    "surname": "Silent",
    "email_address": "carol@silent.com",
    "phone_number": "+15555550012",
    "should_respond": False,
    "response_policy": None,
}

# Contact with should_respond=True and specific response_policy
CONTACT_RESPOND_WITH_POLICY_DEF = {
    "first_name": "David",
    "surname": "VIP",
    "email_address": "david@vip.com",
    "phone_number": "+15555550013",
    "should_respond": True,
    "response_policy": "This is a VIP client. Always be extra polite and thorough.",
}


# ---------------------------------------------------------------------------
#  Helper to set up test contacts on the conversation manager
# ---------------------------------------------------------------------------


def setup_test_contacts(cm) -> dict:
    """
    Add test contacts to the conversation manager via ContactManager.

    Returns a dict mapping definition names to fully populated contact dicts
    (with actual contact_ids from the database).
    """
    contacts = {}

    if cm.contact_manager is None:
        return contacts

    contact_defs = [
        ("respond_yes", CONTACT_RESPOND_YES_DEF),
        ("respond_no", CONTACT_RESPOND_NO_DEF),
        ("respond_no_policy", CONTACT_RESPOND_NO_POLICY_DEF),
        ("respond_with_policy", CONTACT_RESPOND_WITH_POLICY_DEF),
    ]

    for name, contact_def in contact_defs:
        # Create contact and get the actual contact_id from the database
        contact_id = get_or_create_contact(
            cm.contact_manager,
            first_name=contact_def["first_name"],
            surname=contact_def.get("surname"),
            email_address=contact_def.get("email_address"),
            phone_number=contact_def.get("phone_number"),
        )

        # Update should_respond and response_policy using the actual contact_id
        cm.contact_manager.update_contact(
            contact_id=contact_id,
            should_respond=contact_def.get("should_respond", True),
            response_policy=contact_def.get("response_policy") or "",
        )

        # Build the full contact dict with actual contact_id
        contacts[name] = {
            "contact_id": contact_id,
            **contact_def,
        }

    return contacts


# ---------------------------------------------------------------------------
#  Tests: should_respond=True allows outbound communication
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@_handle_project
async def test_sms_allowed_when_should_respond_true(initialized_cm):
    """SMS response is allowed when contact has should_respond=True."""
    cm = initialized_cm
    contacts = setup_test_contacts(cm)
    contact = contacts["respond_yes"]

    result = await cm.step_until_wait(
        SMSReceived(
            contact=contact,
            content="Tell me a joke",
        ),
    )

    # Should have SMS sent (response allowed)
    assert_has_one(result.output_events, SMSSent)
    sms = filter_events_by_type(result.output_events, SMSSent)[0]
    assert sms.contact["phone_number"] == contact["phone_number"]
    assert sms.content


@pytest.mark.asyncio
@_handle_project
async def test_email_allowed_when_should_respond_true(initialized_cm):
    """Email response is allowed when contact has should_respond=True."""
    cm = initialized_cm
    contacts = setup_test_contacts(cm)
    contact = contacts["respond_yes"]

    result = await cm.step_until_wait(
        EmailReceived(
            contact=contact,
            subject="Test Subject",
            body="Tell me a joke",
            email_id="test_email_id",
        ),
    )

    # Should have email sent (response allowed)
    assert_has_one(result.output_events, EmailSent)
    email = filter_events_by_type(result.output_events, EmailSent)[0]
    assert email.contact["email_address"] == contact["email_address"]
    assert email.body


@pytest.mark.asyncio
@_handle_project
async def test_unify_message_allowed_when_should_respond_true(initialized_cm):
    """Unify message response is allowed when contact has should_respond=True."""
    cm = initialized_cm
    contacts = setup_test_contacts(cm)
    contact = contacts["respond_yes"]

    result = await cm.step_until_wait(
        UnifyMessageReceived(
            contact=contact,
            content="Tell me a joke",
        ),
    )

    # Should have unify message sent (response allowed)
    assert_has_one(result.output_events, UnifyMessageSent)
    msg = filter_events_by_type(result.output_events, UnifyMessageSent)[0]
    assert msg.contact["contact_id"] == contact["contact_id"]
    assert msg.content


# ---------------------------------------------------------------------------
#  Tests: should_respond=False blocks outbound communication
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@_handle_project
async def test_sms_blocked_when_should_respond_false(initialized_cm):
    """SMS response is blocked when contact has should_respond=False."""
    cm = initialized_cm
    contacts = setup_test_contacts(cm)
    contact = contacts["respond_no"]

    result = await cm.step_until_wait(
        SMSReceived(
            contact=contact,
            content="Tell me a joke",
        ),
    )

    # Should NOT have any SMSSent events TO THE BLOCKED CONTACT.
    # (The LLM may inform the boss that it can't respond, which is allowed.)
    sms_events = filter_events_by_type(result.output_events, SMSSent)
    sms_to_blocked = [
        e for e in sms_events if e.contact.get("contact_id") == contact["contact_id"]
    ]
    assert len(sms_to_blocked) == 0, (
        f"Expected 0 SMSSent events to should_respond=False contact, "
        f"got {len(sms_to_blocked)}"
    )


@pytest.mark.asyncio
@_handle_project
async def test_email_blocked_when_should_respond_false(initialized_cm):
    """Email response is blocked when contact has should_respond=False."""
    cm = initialized_cm
    contacts = setup_test_contacts(cm)
    contact = contacts["respond_no"]

    result = await cm.step_until_wait(
        EmailReceived(
            contact=contact,
            subject="Test Subject",
            body="Tell me a joke",
            email_id="test_email_id",
        ),
    )

    # Should NOT have any EmailSent events TO THE BLOCKED CONTACT.
    # (The LLM may inform the boss that it can't respond, which is allowed.)
    email_events = filter_events_by_type(result.output_events, EmailSent)
    email_to_blocked = [
        e for e in email_events if e.contact.get("contact_id") == contact["contact_id"]
    ]
    assert len(email_to_blocked) == 0, (
        f"Expected 0 EmailSent events to should_respond=False contact, "
        f"got {len(email_to_blocked)}"
    )


@pytest.mark.asyncio
@_handle_project
async def test_unify_message_blocked_when_should_respond_false(initialized_cm):
    """Unify message response is blocked when contact has should_respond=False."""
    cm = initialized_cm
    contacts = setup_test_contacts(cm)
    contact = contacts["respond_no"]

    result = await cm.step_until_wait(
        UnifyMessageReceived(
            contact=contact,
            content="Tell me a joke",
        ),
    )

    # Should NOT have any UnifyMessageSent events TO THE BLOCKED CONTACT.
    # (The LLM may inform the boss that it can't respond, which is allowed.)
    msg_events = filter_events_by_type(result.output_events, UnifyMessageSent)
    msg_to_blocked = [
        e for e in msg_events if e.contact.get("contact_id") == contact["contact_id"]
    ]
    assert len(msg_to_blocked) == 0, (
        f"Expected 0 UnifyMessageSent events to should_respond=False contact, "
        f"got {len(msg_to_blocked)}"
    )


@pytest.mark.asyncio
@_handle_project
async def test_sms_blocked_no_policy_when_should_respond_false(initialized_cm):
    """SMS is blocked when should_respond=False even without response_policy."""
    cm = initialized_cm
    contacts = setup_test_contacts(cm)
    contact = contacts["respond_no_policy"]

    result = await cm.step_until_wait(
        SMSReceived(
            contact=contact,
            content="Tell me a joke",
        ),
    )

    # Should NOT have any SMSSent events TO THE BLOCKED CONTACT.
    # (The LLM may inform the boss that it can't respond, which is allowed.)
    sms_events = filter_events_by_type(result.output_events, SMSSent)
    sms_to_blocked = [
        e for e in sms_events if e.contact.get("contact_id") == contact["contact_id"]
    ]
    assert len(sms_to_blocked) == 0, (
        f"Expected 0 SMSSent events to should_respond=False contact, "
        f"got {len(sms_to_blocked)}"
    )


# ---------------------------------------------------------------------------
#  Tests: response_policy text is respected
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@_handle_project
async def test_vip_policy_allows_response(initialized_cm):
    """Contact with VIP response_policy and should_respond=True gets a response."""
    cm = initialized_cm
    contacts = setup_test_contacts(cm)
    contact = contacts["respond_with_policy"]

    result = await cm.step_until_wait(
        SMSReceived(
            contact=contact,
            content="Hello, I need help with my account.",
        ),
    )

    # Should have SMS sent (response allowed for VIP)
    assert_has_one(result.output_events, SMSSent)
    sms = filter_events_by_type(result.output_events, SMSSent)[0]
    assert sms.contact["phone_number"] == contact["phone_number"]
    assert sms.content


# ---------------------------------------------------------------------------
#  Tests: Cross-channel with policy combinations
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@_handle_project
async def test_cross_channel_blocked_when_should_respond_false(initialized_cm):
    """Cross-channel communication (SMS asking for email) is blocked when should_respond=False."""
    cm = initialized_cm
    contacts = setup_test_contacts(cm)
    contact = contacts["respond_no"]

    result = await cm.step_until_wait(
        SMSReceived(
            contact=contact,
            content="Send me a joke via email",
        ),
    )

    # Should NOT have any EmailSent events TO THE BLOCKED CONTACT.
    # (The LLM may inform the boss that it can't respond, which is allowed.)
    email_events = filter_events_by_type(result.output_events, EmailSent)
    email_to_blocked = [
        e for e in email_events if e.contact.get("contact_id") == contact["contact_id"]
    ]
    assert len(email_to_blocked) == 0, (
        f"Expected 0 EmailSent events to should_respond=False contact, "
        f"got {len(email_to_blocked)}"
    )

    # Should also NOT have any SMSSent events TO THE BLOCKED CONTACT.
    sms_events = filter_events_by_type(result.output_events, SMSSent)
    sms_to_blocked = [
        e for e in sms_events if e.contact.get("contact_id") == contact["contact_id"]
    ]
    assert len(sms_to_blocked) == 0, (
        f"Expected 0 SMSSent events to should_respond=False contact, "
        f"got {len(sms_to_blocked)}"
    )


@pytest.mark.asyncio
@_handle_project
async def test_cross_channel_allowed_when_should_respond_true(initialized_cm):
    """Cross-channel communication (SMS asking for email) is allowed when should_respond=True."""
    cm = initialized_cm
    contacts = setup_test_contacts(cm)
    contact = contacts["respond_yes"]

    result = await cm.step_until_wait(
        SMSReceived(
            contact=contact,
            content="Send me a joke via email",
        ),
    )

    # Should have email sent (cross-channel allowed)
    assert_has_one(result.output_events, EmailSent)
    email = filter_events_by_type(result.output_events, EmailSent)[0]
    assert email.contact["email_address"] == contact["email_address"]
    assert email.body
