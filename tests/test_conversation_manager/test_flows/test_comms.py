"""
tests/test_conversation_manager/test_comms.py
=============================================

Tests for communication flows (SMS, email, calls, etc.)

Uses the CMStepDriver "step_until_wait" API which:
- Processes an input Event
- Runs the LLM in a loop until it calls 'wait' (no-op)
- Collects all output events produced

This allows the LLM to:
1. Acknowledge on the inbound channel (e.g., reply via SMS)
2. Send the requested communication on the target channel (e.g., email)
3. Call 'wait' when done

Voice call tests verify that events are handled correctly. In the voice
architecture, the Main CM Brain only provides guidance to the Voice Agent
(fast brain) - it doesn't produce speech directly.
"""

import pytest

from tests.helpers import _handle_project
from tests.test_conversation_manager.cm_helpers import (
    filter_events_by_type,
    assert_has_one,
)
from tests.test_conversation_manager.conftest import (
    TEST_CONTACTS,
    HELPFUL_RESPONSE_POLICY,
)
from unity.conversation_manager.events import (
    EmailReceived,
    EmailSent,
    InboundPhoneUtterance,
    InboundUnifyMeetUtterance,
    PhoneCallEnded,
    PhoneCallReceived,
    PhoneCallSent,
    PhoneCallStarted,
    SMSReceived,
    SMSSent,
    UnifyMeetEnded,
    UnifyMeetReceived,
    UnifyMeetStarted,
    UnifyMessageReceived,
    UnifyMessageSent,
)
from unity.conversation_manager.types import Medium

pytestmark = pytest.mark.eval


# ---------------------------------------------------------------------------
#  Module-level fixture to use helpful response policy for all contacts
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def use_helpful_response_policy(initialized_cm):
    """
    Override response_policy for all test contacts to be more permissive.

    The default response policy says "you do not need to take orders from them"
    which Claude interprets too strictly, refusing to make phone calls or send
    messages via specific channels when requested by non-boss contacts.

    This fixture updates all test contacts to use HELPFUL_RESPONSE_POLICY which
    encourages the assistant to fulfil reasonable requests including channel-
    specific communication requests.
    """
    cm = initialized_cm.cm
    if cm.contact_manager is not None:
        for contact in TEST_CONTACTS:
            cm.contact_manager.update_contact(
                contact_id=contact["contact_id"],
                response_policy=HELPFUL_RESPONSE_POLICY,
            )
    yield


# ---------------------------------------------------------------------------
#  Text-based communication tests (SMS, Email, UnifyMessage)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@_handle_project
async def test_sms_to_sms(initialized_cm):
    """SMS request for joke -> reply via SMS."""
    cm = initialized_cm
    contact = TEST_CONTACTS[0]

    result = await cm.step_until_wait(
        SMSReceived(
            contact=contact,
            content="Tell me a joke",
        ),
    )

    # Should have exactly one SMS sent (the reply)
    assert_has_one(result.output_events, SMSSent)
    sms = filter_events_by_type(result.output_events, SMSSent)[0]
    assert sms.contact["phone_number"] == contact["phone_number"]
    assert sms.content


@pytest.mark.asyncio
@_handle_project
async def test_sms_to_email(initialized_cm):
    """SMS request for joke via email -> should send email (may also ack via SMS)."""
    cm = initialized_cm
    contact = TEST_CONTACTS[0]

    result = await cm.step_until_wait(
        SMSReceived(
            contact=contact,
            content="Tell me a joke via email",
        ),
    )

    # Must have exactly one email sent (the target)
    assert_has_one(result.output_events, EmailSent)
    email = filter_events_by_type(result.output_events, EmailSent)[0]
    assert email.contact["email_address"] == contact["email_address"]
    assert email.body


@pytest.mark.asyncio
@_handle_project
async def test_sms_to_unify_message(initialized_cm):
    """SMS request for joke via unify message -> should send unify message."""
    cm = initialized_cm
    contact = TEST_CONTACTS[0]

    result = await cm.step_until_wait(
        SMSReceived(
            contact=contact,
            content="Tell me a joke via unify message",
        ),
    )

    # Must have exactly one unify message sent (the target)
    assert_has_one(result.output_events, UnifyMessageSent)
    msg = filter_events_by_type(result.output_events, UnifyMessageSent)[0]
    assert msg.contact["contact_id"] == contact["contact_id"]
    assert msg.content


@pytest.mark.asyncio
@_handle_project
async def test_sms_to_phone_call(initialized_cm):
    """SMS request for joke via phone call -> should initiate call."""
    cm = initialized_cm
    contact = TEST_CONTACTS[0]

    result = await cm.step_until_wait(
        SMSReceived(
            contact=contact,
            content="Tell me a joke via phone call",
        ),
    )

    # Must have exactly one phone call initiated (the target)
    assert_has_one(result.output_events, PhoneCallSent)
    call = filter_events_by_type(result.output_events, PhoneCallSent)[0]
    assert call.contact["phone_number"] == contact["phone_number"]


@pytest.mark.asyncio
@_handle_project
async def test_email_to_email(initialized_cm):
    """Email request for joke -> reply via email."""
    cm = initialized_cm
    contact = TEST_CONTACTS[0]

    result = await cm.step_until_wait(
        EmailReceived(
            contact=contact,
            subject="Test Subject",
            body="Tell me a joke",
            email_id="test_email_id",
        ),
    )

    # Should have exactly one email sent (the reply)
    assert_has_one(result.output_events, EmailSent)
    email = filter_events_by_type(result.output_events, EmailSent)[0]
    assert email.contact["email_address"] == contact["email_address"]
    assert email.subject
    assert email.body


@pytest.mark.asyncio
@_handle_project
async def test_email_with_attachment_visible(initialized_cm):
    """Email with attachment -> assistant confirms receipt and can share download path."""
    cm = initialized_cm
    contact = TEST_CONTACTS[0]

    # Step 1: Send email with attachment
    result = await cm.step_until_wait(
        EmailReceived(
            contact=contact,
            subject="Document for review",
            body="I've attached the quarterly report. Can you confirm you received it?",
            email_id="test_email_with_attachment",
            attachments=["quarterly_report.pdf"],
        ),
    )

    # Should have exactly one email sent (the reply)
    assert_has_one(result.output_events, EmailSent)
    email = filter_events_by_type(result.output_events, EmailSent)[0]
    # The reply should confirm receipt of the attachment
    body_lower = email.body.lower()
    assert any(
        term in body_lower
        for term in [
            "received",
            "see",
            "got",
            "attachment",
            "quarterly",
            "report",
            "pdf",
        ]
    ), f"Expected reply to confirm attachment receipt, got: {email.body}"

    # Step 2: Ask about the download path
    result2 = await cm.step_until_wait(
        EmailReceived(
            contact=contact,
            subject="Re: Document for review",
            body="Great! Did you download it? What's the file path?",
            email_id="test_email_followup",
            attachments=[],
        ),
    )

    # Should reply with the file path
    assert_has_one(result2.output_events, EmailSent)
    email2 = filter_events_by_type(result2.output_events, EmailSent)[0]
    body2_lower = email2.body.lower()
    # The reply should mention the Downloads path
    assert "downloads" in body2_lower and "quarterly_report.pdf" in body2_lower, (
        f"Expected reply to include download path (Downloads/quarterly_report.pdf), "
        f"got: {email2.body}"
    )


@pytest.mark.asyncio
@_handle_project
async def test_email_missing_attachment_detected(initialized_cm):
    """Email asks about attachment but none attached -> assistant notes missing attachment."""
    cm = initialized_cm
    contact = TEST_CONTACTS[0]

    result = await cm.step_until_wait(
        EmailReceived(
            contact=contact,
            subject="Document for review",
            body="I've attached the quarterly report. Can you confirm you received it?",
            email_id="test_email_missing_attachment",
            attachments=[],  # No attachments despite the body mentioning one
        ),
    )

    # Should have exactly one email sent (the reply)
    assert_has_one(result.output_events, EmailSent)
    email = filter_events_by_type(result.output_events, EmailSent)[0]
    # The reply should mention that the attachment is missing
    # Normalize curly apostrophes to straight apostrophes for matching
    body_lower = email.body.lower().replace("'", "'").replace("'", "'")
    assert any(
        term in body_lower
        for term in [
            "missing",
            "forgot",
            "don't see",
            "didn't see",
            "not seeing",
            "no attachment",
            "didn't",
            "not attached",
            "can't find",
            "unable",
            "couldn't",
            "resend",
            "re-send",
            "check",
        ]
    ), f"Expected reply to note missing attachment, got: {email.body}"


@pytest.mark.asyncio
@_handle_project
async def test_unify_message_with_attachment_visible(initialized_cm):
    """Unify message with attachment -> assistant confirms receipt and can share download path."""
    cm = initialized_cm
    contact = TEST_CONTACTS[0]

    # Step 1: Send unify message with attachment
    result = await cm.step_until_wait(
        UnifyMessageReceived(
            contact=contact,
            content="I've attached the quarterly report. Can you confirm you received it?",
            attachments=["quarterly_report.pdf"],
        ),
    )

    # Should have exactly one unify message sent (the reply)
    assert_has_one(result.output_events, UnifyMessageSent)
    msg = filter_events_by_type(result.output_events, UnifyMessageSent)[0]
    # The reply should confirm receipt of the attachment
    content_lower = msg.content.lower()
    assert any(
        term in content_lower
        for term in [
            "received",
            "see",
            "got",
            "attachment",
            "quarterly",
            "report",
            "pdf",
        ]
    ), f"Expected reply to confirm attachment receipt, got: {msg.content}"

    # Step 2: Ask about the download path
    result2 = await cm.step_until_wait(
        UnifyMessageReceived(
            contact=contact,
            content="Great! Did you download it? What's the file path?",
            attachments=[],
        ),
    )

    # Should reply with the file path
    assert_has_one(result2.output_events, UnifyMessageSent)
    msg2 = filter_events_by_type(result2.output_events, UnifyMessageSent)[0]
    content2_lower = msg2.content.lower()
    # The reply should mention the Downloads path
    assert "downloads" in content2_lower and "quarterly_report.pdf" in content2_lower, (
        f"Expected reply to include download path (Downloads/quarterly_report.pdf), "
        f"got: {msg2.content}"
    )


@pytest.mark.asyncio
@_handle_project
async def test_unify_message_missing_attachment_detected(initialized_cm):
    """Unify message asks about attachment but none attached -> assistant notes missing attachment."""
    cm = initialized_cm
    contact = TEST_CONTACTS[0]

    result = await cm.step_until_wait(
        UnifyMessageReceived(
            contact=contact,
            content="I've attached the quarterly report. Can you confirm you received it?",
            attachments=[],  # No attachments despite the content mentioning one
        ),
    )

    # Should have exactly one unify message sent (the reply)
    assert_has_one(result.output_events, UnifyMessageSent)
    msg = filter_events_by_type(result.output_events, UnifyMessageSent)[0]
    # The reply should mention that the attachment is missing
    # Normalize curly apostrophes to straight apostrophes for matching
    content_lower = msg.content.lower().replace("'", "'").replace("'", "'")
    assert any(
        term in content_lower
        for term in [
            "missing",
            "forgot",
            "don't see",
            "didn't see",
            "not seeing",
            "no attachment",
            "didn't",
            "not attached",
            "can't find",
            "unable",
            "couldn't",
            "resend",
            "re-send",
            "check",
        ]
    ), f"Expected reply to note missing attachment, got: {msg.content}"


@pytest.mark.asyncio
@_handle_project
async def test_email_to_sms(initialized_cm):
    """Email request for joke via SMS -> should send SMS."""
    cm = initialized_cm
    contact = TEST_CONTACTS[0]

    result = await cm.step_until_wait(
        EmailReceived(
            contact=contact,
            subject="Test Subject",
            body="Tell me a joke via SMS",
            email_id="test_email_id",
        ),
    )

    # Must have SMS sent (the target)
    assert_has_one(result.output_events, SMSSent)
    sms = filter_events_by_type(result.output_events, SMSSent)[0]
    assert sms.contact["phone_number"] == contact["phone_number"]
    assert sms.content


@pytest.mark.asyncio
@_handle_project
async def test_email_to_unify_message(initialized_cm):
    """Email request for joke via unify message -> should send unify message."""
    cm = initialized_cm
    contact = TEST_CONTACTS[0]

    result = await cm.step_until_wait(
        EmailReceived(
            contact=contact,
            subject="Test Subject",
            body="Tell me a joke via unify message",
            email_id="test_email_id",
        ),
    )

    # Must have exactly one unify message sent (the target)
    assert_has_one(result.output_events, UnifyMessageSent)
    msg = filter_events_by_type(result.output_events, UnifyMessageSent)[0]
    assert msg.contact["contact_id"] == contact["contact_id"]
    assert msg.content


@pytest.mark.asyncio
@_handle_project
async def test_email_to_phone_call(initialized_cm):
    """Email request for joke via phone call -> should initiate call."""
    cm = initialized_cm
    contact = TEST_CONTACTS[0]

    result = await cm.step_until_wait(
        EmailReceived(
            contact=contact,
            subject="Test Subject",
            body="Tell me a joke via phone call",
            email_id="test_email_id",
        ),
    )

    # Must have exactly one phone call initiated (the target)
    assert_has_one(result.output_events, PhoneCallSent)
    call = filter_events_by_type(result.output_events, PhoneCallSent)[0]
    assert call.contact["phone_number"] == contact["phone_number"]


@pytest.mark.asyncio
@_handle_project
async def test_unify_message_to_unify_message(initialized_cm):
    """Unify message request for joke -> reply via unify message."""
    cm = initialized_cm
    contact = TEST_CONTACTS[0]

    result = await cm.step_until_wait(
        UnifyMessageReceived(
            contact=contact,
            content="Tell me a joke",
        ),
    )

    # Should have exactly one unify message sent (the reply)
    assert_has_one(result.output_events, UnifyMessageSent)
    msg = filter_events_by_type(result.output_events, UnifyMessageSent)[0]
    assert msg.contact["contact_id"] == contact["contact_id"]
    assert msg.content


@pytest.mark.asyncio
@_handle_project
async def test_unify_message_to_sms(initialized_cm):
    """Unify message request for joke via SMS -> should send SMS."""
    cm = initialized_cm
    contact = TEST_CONTACTS[0]

    result = await cm.step_until_wait(
        UnifyMessageReceived(
            contact=contact,
            content="Tell me a joke via SMS",
        ),
    )

    # Must have SMS sent (the target)
    assert_has_one(result.output_events, SMSSent)
    sms = filter_events_by_type(result.output_events, SMSSent)[0]
    assert sms.contact["phone_number"] == contact["phone_number"]
    assert sms.content


@pytest.mark.asyncio
@_handle_project
async def test_unify_message_to_email(initialized_cm):
    """Unify message request for joke via email -> should send email."""
    cm = initialized_cm
    contact = TEST_CONTACTS[0]

    result = await cm.step_until_wait(
        UnifyMessageReceived(
            contact=contact,
            content="Tell me a joke via email",
        ),
    )

    # Must have exactly one email sent (the target)
    assert_has_one(result.output_events, EmailSent)
    email = filter_events_by_type(result.output_events, EmailSent)[0]
    assert email.contact["email_address"] == contact["email_address"]
    assert email.body


@pytest.mark.asyncio
@_handle_project
async def test_unify_message_to_phone_call(initialized_cm):
    """Unify message request for joke via phone call -> should initiate call."""
    cm = initialized_cm
    contact = TEST_CONTACTS[0]

    result = await cm.step_until_wait(
        UnifyMessageReceived(
            contact=contact,
            content="Tell me a joke via phone call",
        ),
    )

    # Must have exactly one phone call initiated (the target)
    assert_has_one(result.output_events, PhoneCallSent)
    call = filter_events_by_type(result.output_events, PhoneCallSent)[0]
    assert call.contact["phone_number"] == contact["phone_number"]


# ---------------------------------------------------------------------------
#  Voice call tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@_handle_project
async def test_phone_call(initialized_cm):
    """Basic phone call flow - just verify utterance is recorded."""
    cm = initialized_cm
    contact = TEST_CONTACTS[0]

    await cm.step(
        PhoneCallReceived(
            contact=contact,
            conference_name="test_conference",
        ),
    )
    await cm.step(PhoneCallStarted(contact=contact))
    await cm.step(InboundPhoneUtterance(contact=contact, content="Tell me a joke"))

    phone_thread = list(
        cm.contact_index.active_conversations[contact["contact_id"]].threads[
            Medium.PHONE_CALL
        ],
    )
    assert any(getattr(m, "content", None) == "Tell me a joke" for m in phone_thread)

    await cm.step(PhoneCallEnded(contact=contact))


@pytest.mark.asyncio
@_handle_project
async def test_phone_call_to_sms(initialized_cm):
    """During phone call, request joke via SMS -> should send SMS."""
    cm = initialized_cm
    contact = TEST_CONTACTS[0]

    await cm.step(
        PhoneCallReceived(
            contact=contact,
            conference_name="test_conference",
        ),
    )
    await cm.step(PhoneCallStarted(contact=contact))

    result = await cm.step_until_wait(
        InboundPhoneUtterance(
            contact=contact,
            content="Send me a joke via SMS right now",
        ),
    )

    # Must have SMS sent
    assert_has_one(result.output_events, SMSSent)

    await cm.step(PhoneCallEnded(contact=contact))


@pytest.mark.asyncio
@_handle_project
async def test_phone_call_to_email(initialized_cm):
    """During phone call, request joke via email -> should send email."""
    cm = initialized_cm
    contact = TEST_CONTACTS[0]

    await cm.step(
        PhoneCallReceived(
            contact=contact,
            conference_name="test_conference",
        ),
    )
    await cm.step(PhoneCallStarted(contact=contact))

    result = await cm.step_until_wait(
        InboundPhoneUtterance(
            contact=contact,
            content="Send me a joke via email right now",
        ),
    )

    # Must have exactly one email sent
    assert_has_one(result.output_events, EmailSent)

    await cm.step(PhoneCallEnded(contact=contact))


@pytest.mark.asyncio
@_handle_project
async def test_phone_call_to_unify_message(initialized_cm):
    """During phone call, request joke via unify message -> should send unify message."""
    cm = initialized_cm
    contact = TEST_CONTACTS[0]

    await cm.step(
        PhoneCallReceived(
            contact=contact,
            conference_name="test_conference",
        ),
    )
    await cm.step(PhoneCallStarted(contact=contact))

    result = await cm.step_until_wait(
        InboundPhoneUtterance(
            contact=contact,
            content="Send me a joke via unify message right now",
        ),
    )

    # Must have exactly one unify message sent
    assert_has_one(result.output_events, UnifyMessageSent)

    await cm.step(PhoneCallEnded(contact=contact))


# ---------------------------------------------------------------------------
#  Unify Meet tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@_handle_project
async def test_unify_meet(initialized_cm):
    """Basic unify meet flow - just verify utterance is recorded."""
    cm = initialized_cm
    contact = TEST_CONTACTS[0]

    await cm.step(UnifyMeetReceived(contact=contact))
    await cm.step(UnifyMeetStarted(contact=contact))
    await cm.step(InboundUnifyMeetUtterance(contact=contact, content="Tell me a joke"))

    meet_thread = list(
        cm.contact_index.active_conversations[contact["contact_id"]].threads[
            Medium.UNIFY_MEET
        ],
    )
    assert any(getattr(m, "content", None) == "Tell me a joke" for m in meet_thread)

    await cm.step(UnifyMeetEnded(contact=contact))


# Note: There is no test_unify_meet_to_phone_call test because the system does not
# support maintaining multiple simultaneous voice-based conversations. While on a
# unify meet, the assistant cannot initiate an outbound phone call.


@pytest.mark.asyncio
@_handle_project
async def test_unify_meet_to_sms(initialized_cm):
    """During unify meet, request joke via SMS -> should send SMS."""
    cm = initialized_cm
    contact = TEST_CONTACTS[0]

    await cm.step(UnifyMeetReceived(contact=contact))
    await cm.step(UnifyMeetStarted(contact=contact))

    result = await cm.step_until_wait(
        InboundUnifyMeetUtterance(
            contact=contact,
            content="Send me a joke via sms right now",
        ),
    )

    # Must have SMS sent
    assert_has_one(result.output_events, SMSSent)

    await cm.step(UnifyMeetEnded(contact=contact))


@pytest.mark.asyncio
@_handle_project
async def test_unify_meet_to_email(initialized_cm):
    """During unify meet, request joke via email -> should send email."""
    cm = initialized_cm
    contact = TEST_CONTACTS[0]

    await cm.step(UnifyMeetReceived(contact=contact))
    await cm.step(UnifyMeetStarted(contact=contact))

    result = await cm.step_until_wait(
        InboundUnifyMeetUtterance(
            contact=contact,
            content="Send me a joke via email right now",
        ),
    )

    # Must have exactly one email sent
    assert_has_one(result.output_events, EmailSent)

    await cm.step(UnifyMeetEnded(contact=contact))


@pytest.mark.asyncio
@_handle_project
async def test_unify_meet_to_unify_message(initialized_cm):
    """During unify meet, request joke via unify message -> should send unify message."""
    cm = initialized_cm
    contact = TEST_CONTACTS[0]

    await cm.step(UnifyMeetReceived(contact=contact))
    await cm.step(UnifyMeetStarted(contact=contact))

    result = await cm.step_until_wait(
        InboundUnifyMeetUtterance(
            contact=contact,
            content="Send me a joke via unify message right now",
        ),
    )

    # Must have exactly one unify message sent
    assert_has_one(result.output_events, UnifyMessageSent)

    await cm.step(UnifyMeetEnded(contact=contact))
