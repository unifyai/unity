"""
tests/test_conversation_manager/test_flows/test_respond_policy.py
=================================================================

Tests for the `should_respond` boolean flag and `response_policy` text field
on contacts. These tests verify:

1. The `should_respond` flag is presented to the brain in contact_index
2. Outbound communication is blocked when `should_respond=False`
3. The `response_policy` text influences assistant behavior

Uses the CMStepDriver "step_until_wait" API pattern from test_comms.py.
"""

import pytest

from tests.helpers import _handle_project
from tests.test_conversation_manager.cm_helpers import (
    filter_events_by_type,
    assert_has_one,
)
from unity.conversation_manager.domains.contact_index import Contact
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
#  Test contacts with different should_respond / response_policy settings
# ---------------------------------------------------------------------------

# Contact that CAN be responded to (should_respond=True)
CONTACT_RESPOND_YES = {
    "contact_id": 10,
    "first_name": "Alice",
    "surname": "Responder",
    "email_address": "alice@respond.com",
    "phone_number": "+15555550010",
    "should_respond": True,
    "response_policy": "Respond promptly and professionally to all inquiries.",
}

# Contact that CANNOT be responded to (should_respond=False)
CONTACT_RESPOND_NO = {
    "contact_id": 11,
    "first_name": "Bob",
    "surname": "NoContact",
    "email_address": "bob@nocontact.com",
    "phone_number": "+15555550011",
    "should_respond": False,
    "response_policy": "Do not respond - this contact has opted out of communications.",
}

# Contact with should_respond=False but no response_policy
CONTACT_RESPOND_NO_POLICY = {
    "contact_id": 12,
    "first_name": "Carol",
    "surname": "Silent",
    "email_address": "carol@silent.com",
    "phone_number": "+15555550012",
    "should_respond": False,
    "response_policy": None,
}

# Contact with should_respond=True and specific response_policy
CONTACT_RESPOND_WITH_POLICY = {
    "contact_id": 13,
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


def setup_test_contacts(cm):
    """Add test contacts to the conversation manager's contact_index."""
    for contact_dict in [
        CONTACT_RESPOND_YES,
        CONTACT_RESPOND_NO,
        CONTACT_RESPOND_NO_POLICY,
        CONTACT_RESPOND_WITH_POLICY,
    ]:
        cm.contact_index.contacts[contact_dict["contact_id"]] = Contact(**contact_dict)


# ---------------------------------------------------------------------------
#  Tests: should_respond=True allows outbound communication
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@_handle_project
async def test_sms_allowed_when_should_respond_true(initialized_cm):
    """SMS response is allowed when contact has should_respond=True."""
    cm = initialized_cm
    setup_test_contacts(cm)
    contact = CONTACT_RESPOND_YES

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
    setup_test_contacts(cm)
    contact = CONTACT_RESPOND_YES

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
    setup_test_contacts(cm)
    contact = CONTACT_RESPOND_YES

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
    setup_test_contacts(cm)
    contact = CONTACT_RESPOND_NO

    result = await cm.step_until_wait(
        SMSReceived(
            contact=contact,
            content="Tell me a joke",
        ),
    )

    # Should NOT have any SMSSent events (response blocked)
    sms_events = filter_events_by_type(result.output_events, SMSSent)
    assert len(sms_events) == 0, (
        f"Expected 0 SMSSent events for should_respond=False contact, "
        f"got {len(sms_events)}"
    )


@pytest.mark.asyncio
@_handle_project
async def test_email_blocked_when_should_respond_false(initialized_cm):
    """Email response is blocked when contact has should_respond=False."""
    cm = initialized_cm
    setup_test_contacts(cm)
    contact = CONTACT_RESPOND_NO

    result = await cm.step_until_wait(
        EmailReceived(
            contact=contact,
            subject="Test Subject",
            body="Tell me a joke",
            email_id="test_email_id",
        ),
    )

    # Should NOT have any EmailSent events (response blocked)
    email_events = filter_events_by_type(result.output_events, EmailSent)
    assert len(email_events) == 0, (
        f"Expected 0 EmailSent events for should_respond=False contact, "
        f"got {len(email_events)}"
    )


@pytest.mark.asyncio
@_handle_project
async def test_unify_message_blocked_when_should_respond_false(initialized_cm):
    """Unify message response is blocked when contact has should_respond=False."""
    cm = initialized_cm
    setup_test_contacts(cm)
    contact = CONTACT_RESPOND_NO

    result = await cm.step_until_wait(
        UnifyMessageReceived(
            contact=contact,
            content="Tell me a joke",
        ),
    )

    # Should NOT have any UnifyMessageSent events (response blocked)
    msg_events = filter_events_by_type(result.output_events, UnifyMessageSent)
    assert len(msg_events) == 0, (
        f"Expected 0 UnifyMessageSent events for should_respond=False contact, "
        f"got {len(msg_events)}"
    )


@pytest.mark.asyncio
@_handle_project
async def test_sms_blocked_no_policy_when_should_respond_false(initialized_cm):
    """SMS is blocked when should_respond=False even without response_policy."""
    cm = initialized_cm
    setup_test_contacts(cm)
    contact = CONTACT_RESPOND_NO_POLICY

    result = await cm.step_until_wait(
        SMSReceived(
            contact=contact,
            content="Tell me a joke",
        ),
    )

    # Should NOT have any SMSSent events (response blocked)
    sms_events = filter_events_by_type(result.output_events, SMSSent)
    assert len(sms_events) == 0, (
        f"Expected 0 SMSSent events for should_respond=False contact, "
        f"got {len(sms_events)}"
    )


# ---------------------------------------------------------------------------
#  Tests: response_policy text is respected
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@_handle_project
async def test_vip_policy_allows_response(initialized_cm):
    """Contact with VIP response_policy and should_respond=True gets a response."""
    cm = initialized_cm
    setup_test_contacts(cm)
    contact = CONTACT_RESPOND_WITH_POLICY

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
    setup_test_contacts(cm)
    contact = CONTACT_RESPOND_NO

    result = await cm.step_until_wait(
        SMSReceived(
            contact=contact,
            content="Send me a joke via email",
        ),
    )

    # Should NOT have any EmailSent events (response blocked)
    email_events = filter_events_by_type(result.output_events, EmailSent)
    assert len(email_events) == 0, (
        f"Expected 0 EmailSent events for should_respond=False contact, "
        f"got {len(email_events)}"
    )

    # Should also NOT have any SMSSent events
    sms_events = filter_events_by_type(result.output_events, SMSSent)
    assert len(sms_events) == 0, (
        f"Expected 0 SMSSent events for should_respond=False contact, "
        f"got {len(sms_events)}"
    )


@pytest.mark.asyncio
@_handle_project
async def test_cross_channel_allowed_when_should_respond_true(initialized_cm):
    """Cross-channel communication (SMS asking for email) is allowed when should_respond=True."""
    cm = initialized_cm
    setup_test_contacts(cm)
    contact = CONTACT_RESPOND_YES

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
