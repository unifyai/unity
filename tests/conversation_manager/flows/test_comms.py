"""
tests/conversation_manager/test_comms.py
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

from unittest.mock import AsyncMock, patch

import pytest

from tests.helpers import _handle_project
from tests.conversation_manager.cm_helpers import (
    filter_events_by_type,
    assert_has_one,
    make_contacts_visible,
)
from tests.conversation_manager.conftest import (
    TEST_CONTACTS,
    BOSS,
    HELPFUL_RESPONSE_POLICY,
)
from unity.conversation_manager.events import (
    ApiMessageReceived,
    ApiMessageSent,
    DiscordMessageReceived,
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
from unity.conversation_manager.domains import comms_utils
from unity.conversation_manager.cm_types import Medium

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
            attachments=[{"id": "att-email-1", "filename": "quarterly_report.pdf"}],
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
    # The reply should mention the Attachments path
    assert "attachments" in body2_lower and "quarterly_report.pdf" in body2_lower, (
        f"Expected reply to include attachment path (Attachments/att-email-1_quarterly_report.pdf), "
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
    body_lower = email.body.lower().replace("\u2018", "'").replace("\u2019", "'")
    assert any(
        term in body_lower
        for term in [
            "missing",
            "forgot",
            "don't see",
            "doesn't look",
            "doesn't appear",
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
            attachments=[{"id": "att-1", "filename": "quarterly_report.pdf"}],
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
    # The reply should mention the Attachments path
    assert (
        "attachments" in content2_lower and "quarterly_report.pdf" in content2_lower
    ), (
        f"Expected reply to include attachment path (Attachments/att-1_quarterly_report.pdf), "
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
    # Normalize curly apostrophes (U+2018, U+2019) to straight apostrophes for matching
    content_lower = msg.content.lower().replace("\u2018", "'").replace("\u2019", "'")
    assert any(
        term in content_lower
        for term in [
            "missing",
            "forgot",
            "don't see",
            "doesn't look",
            "doesn't appear",
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


@pytest.mark.asyncio
@_handle_project
async def test_discord_message_to_discord_message(initialized_cm):
    """Discord DM request for joke -> reply via Discord DM through brain action tools."""
    cm = initialized_cm
    contact = TEST_CONTACTS[0]
    discord_id = "discord-user-123"

    cm.cm.assistant_discord_bot_id = "discord-bot-123"
    assert cm.cm.contact_manager is not None
    cm.cm.contact_manager.update_contact(
        contact_id=contact["contact_id"],
        discord_id=discord_id,
    )

    with patch(
        "unity.comms.primitives.comms_utils.send_discord_message",
        new=AsyncMock(return_value={"success": True}),
    ):
        result = await cm.step_until_wait(
            DiscordMessageReceived(
                contact={**contact, "discord_id": discord_id},
                content="Tell me a joke",
                channel_id="dm-channel-1",
                bot_id="discord-bot-123",
                message_id="discord-message-1",
            ),
        )

    assert "send_discord_message" in cm.all_tool_calls
    discord_thread = cm.contact_index.get_messages_for_contact(
        contact["contact_id"],
        Medium.DISCORD_MESSAGE,
    )
    assistant_messages = [
        message
        for message in discord_thread
        if getattr(message, "role", None) == "assistant"
    ]
    assert assistant_messages
    assert assistant_messages[-1].content


# ---------------------------------------------------------------------------
#  API message tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@_handle_project
async def test_api_message_to_api_response(initialized_cm):
    """API message request for joke -> reply via send_api_response."""
    cm = initialized_cm
    contact = TEST_CONTACTS[0]

    cm.cm._pending_api_message_id = "test-api-msg-001"

    result = await cm.step_until_wait(
        ApiMessageReceived(
            contact=contact,
            content="Tell me a joke",
            api_message_id="test-api-msg-001",
        ),
    )

    assert_has_one(result.output_events, ApiMessageSent)
    api_resp = filter_events_by_type(result.output_events, ApiMessageSent)[0]
    assert api_resp.contact["contact_id"] == contact["contact_id"]
    assert api_resp.content
    assert api_resp.api_message_id == "test-api-msg-001"


@pytest.mark.asyncio
@_handle_project
async def test_api_message_to_sms(initialized_cm):
    """API message requesting joke via SMS -> should send SMS."""
    cm = initialized_cm
    contact = TEST_CONTACTS[0]

    result = await cm.step_until_wait(
        ApiMessageReceived(
            contact=contact,
            content="Tell me a joke via SMS",
            api_message_id="test-api-msg-002",
        ),
    )

    assert_has_one(result.output_events, SMSSent)
    sms = filter_events_by_type(result.output_events, SMSSent)[0]
    assert sms.contact["phone_number"] == contact["phone_number"]
    assert sms.content


@pytest.mark.asyncio
@_handle_project
async def test_api_message_tags_echoed_in_response(initialized_cm):
    """API message with tags -> send_api_response should echo tags back."""
    cm = initialized_cm
    contact = TEST_CONTACTS[0]

    cm.cm._pending_api_message_id = "test-api-msg-tags"

    result = await cm.step_until_wait(
        ApiMessageReceived(
            contact=contact,
            content="Tell me a joke",
            api_message_id="test-api-msg-tags",
            tags=["source:slack", "channel:#general"],
        ),
    )

    assert_has_one(result.output_events, ApiMessageSent)
    api_resp = filter_events_by_type(result.output_events, ApiMessageSent)[0]
    assert api_resp.content
    assert api_resp.tags == [
        "source:slack",
        "channel:#general",
    ], f"Expected tags to be echoed, got {api_resp.tags}"


@pytest.mark.asyncio
@_handle_project
async def test_api_message_with_attachment_visible(initialized_cm):
    """API message with attachment -> assistant confirms receipt."""
    cm = initialized_cm
    contact = TEST_CONTACTS[0]

    cm.cm._pending_api_message_id = "test-api-msg-att"

    result = await cm.step_until_wait(
        ApiMessageReceived(
            contact=contact,
            content="I've attached the quarterly report. Can you confirm you received it?",
            api_message_id="test-api-msg-att",
            attachments=[{"id": "att-1", "filename": "quarterly_report.pdf"}],
        ),
    )

    assert_has_one(result.output_events, ApiMessageSent)
    msg = filter_events_by_type(result.output_events, ApiMessageSent)[0]
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

    phone_thread = cm.contact_index.get_messages_for_contact(
        contact["contact_id"],
        Medium.PHONE_CALL,
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

    meet_thread = cm.contact_index.get_messages_for_contact(
        contact["contact_id"],
        Medium.UNIFY_MEET,
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


# ---------------------------------------------------------------------------
#  Email recipient tests (to, cc, bcc)
#
#  These tests verify the new to/cc/bcc functionality using inline email
#  addresses in the boss's request, avoiding contact lookup delegation.
#
#  Note: The basic single-recipient email test is already covered by
#  test_email_with_inline_email_address in test_multi_contact_outbound.py.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@_handle_project
async def test_email_with_cc(initialized_cm):
    """Boss sends email with CC recipient."""
    cm = initialized_cm

    make_contacts_visible(cm, 2, 3)  # Alice, Bob

    result = await cm.step_until_wait(
        SMSReceived(
            contact=BOSS,
            content=(
                "Could you email Alice and tell her the meeting "
                "is confirmed for 3pm tomorrow. CC Bob."
            ),
        ),
    )

    # Should have exactly one email sent
    assert_has_one(result.output_events, EmailSent)
    email = filter_events_by_type(result.output_events, EmailSent)[0]

    # Verify recipients
    assert (
        "alice@example.com" in email.to
    ), f"Expected alice@example.com in 'to', got to={email.to}"
    assert (
        "bob@example.com" in email.cc
    ), f"Expected bob@example.com in 'cc', got cc={email.cc}"


@pytest.mark.asyncio
@_handle_project
async def test_email_with_bcc(initialized_cm):
    """Boss sends email with BCC recipient."""
    cm = initialized_cm
    make_contacts_visible(cm, 2, 4)  # Alice, Charlie

    result = await cm.step_until_wait(
        SMSReceived(
            contact=BOSS,
            content=(
                "Could you email Alice and tell her the budget "
                "has been approved. BCC Charlie."
            ),
        ),
    )

    # Should have exactly one email sent
    assert_has_one(result.output_events, EmailSent)
    email = filter_events_by_type(result.output_events, EmailSent)[0]

    # Verify recipients
    assert (
        "alice@example.com" in email.to
    ), f"Expected alice@example.com in 'to', got to={email.to}"
    assert (
        "charlie@example.com" in email.bcc
    ), f"Expected charlie@example.com in 'bcc', got bcc={email.bcc}"


@pytest.mark.asyncio
@_handle_project
async def test_email_with_multiple_cc(initialized_cm):
    """Boss sends email with multiple CC recipients."""
    cm = initialized_cm
    make_contacts_visible(cm, 2, 3, 4)  # Alice, Bob, Charlie

    result = await cm.step_until_wait(
        SMSReceived(
            contact=BOSS,
            content=(
                "Could you email Alice and tell her the quarterly "
                "results look great. CC Bob and Charlie."
            ),
        ),
    )

    # Should have exactly one email sent
    assert_has_one(result.output_events, EmailSent)
    email = filter_events_by_type(result.output_events, EmailSent)[0]

    # Verify recipients
    assert (
        "alice@example.com" in email.to
    ), f"Expected alice@example.com in 'to', got to={email.to}"
    assert (
        "bob@example.com" in email.cc
    ), f"Expected bob@example.com in 'cc', got cc={email.cc}"
    assert (
        "charlie@example.com" in email.cc
    ), f"Expected charlie@example.com in 'cc', got cc={email.cc}"


@pytest.mark.asyncio
@_handle_project
async def test_email_with_cc_and_bcc(initialized_cm):
    """Boss sends email with TO, CC, and BCC recipients."""
    cm = initialized_cm
    make_contacts_visible(cm, 2, 3, 5)  # Alice, Bob, Diana

    result = await cm.step_until_wait(
        SMSReceived(
            contact=BOSS,
            content=(
                "Could you email Alice and tell her the project "
                "deadline is Friday. CC Bob and BCC Diana."
            ),
        ),
    )

    # Should have exactly one email sent
    assert_has_one(result.output_events, EmailSent)
    email = filter_events_by_type(result.output_events, EmailSent)[0]

    # Verify recipients in correct fields
    assert (
        "alice@example.com" in email.to
    ), f"Expected alice@example.com in 'to', got to={email.to}"
    assert (
        "bob@example.com" in email.cc
    ), f"Expected bob@example.com in 'cc', got cc={email.cc}"
    assert (
        "diana@example.com" in email.bcc
    ), f"Expected diana@example.com in 'bcc', got bcc={email.bcc}"


@pytest.mark.asyncio
@_handle_project
async def test_email_to_multiple_recipients(initialized_cm):
    """Boss sends email to multiple TO recipients."""
    cm = initialized_cm
    make_contacts_visible(cm, 2, 3)  # Alice, Bob

    result = await cm.step_until_wait(
        SMSReceived(
            contact=BOSS,
            content=(
                "Could you email both Alice and Bob "
                "and tell them we're excited to move forward with the partnership."
            ),
        ),
    )

    # Should have exactly one email sent
    assert_has_one(result.output_events, EmailSent)
    email = filter_events_by_type(result.output_events, EmailSent)[0]

    # Verify both addresses are in recipients
    all_recipients = email.to + email.cc
    assert (
        "alice@example.com" in all_recipients
    ), f"Expected alice@example.com in recipients, got to={email.to}, cc={email.cc}"
    assert (
        "bob@example.com" in all_recipients
    ), f"Expected bob@example.com in recipients, got to={email.to}, cc={email.cc}"


@pytest.mark.asyncio
@_handle_project
async def test_email_with_multiple_bcc(initialized_cm):
    """Boss sends email with multiple BCC recipients."""
    cm = initialized_cm
    make_contacts_visible(cm, 2, 3, 4)  # Alice, Bob, Charlie

    result = await cm.step_until_wait(
        SMSReceived(
            contact=BOSS,
            content=(
                "Could you email Alice and tell her the contract "
                "is ready for signature. BCC Bob and Charlie "
                "for their records."
            ),
        ),
    )

    # Should have exactly one email sent
    assert_has_one(result.output_events, EmailSent)
    email = filter_events_by_type(result.output_events, EmailSent)[0]

    # Verify TO recipient
    assert (
        "alice@example.com" in email.to
    ), f"Expected alice@example.com in 'to', got to={email.to}"

    # Verify both BCC recipients
    assert (
        "bob@example.com" in email.bcc
    ), f"Expected bob@example.com in 'bcc', got bcc={email.bcc}"
    assert (
        "charlie@example.com" in email.bcc
    ), f"Expected charlie@example.com in 'bcc', got bcc={email.bcc}"


@pytest.mark.asyncio
@_handle_project
async def test_email_with_all_plural_recipients(initialized_cm):
    """Boss sends email with TO, CC, and BCC all populated."""
    cm = initialized_cm
    make_contacts_visible(cm, 2, 3, 4, 5)  # Alice, Bob, Charlie, Diana

    result = await cm.step_until_wait(
        SMSReceived(
            contact=BOSS,
            content=(
                "Send an email to Alice and Bob "
                "telling them the board meeting is scheduled for Monday at 10am. "
                "CC Charlie for visibility, "
                "and BCC Diana for records."
            ),
        ),
    )

    # Should have exactly one email sent
    assert_has_one(result.output_events, EmailSent)
    email = filter_events_by_type(result.output_events, EmailSent)[0]

    # Verify TO recipients
    all_to_cc = email.to + email.cc
    assert (
        "alice@example.com" in all_to_cc
    ), f"Expected alice@example.com in to/cc, got to={email.to}, cc={email.cc}"
    assert (
        "bob@example.com" in all_to_cc
    ), f"Expected bob@example.com in to/cc, got to={email.to}, cc={email.cc}"

    # Verify CC recipient
    assert (
        "charlie@example.com" in all_to_cc
    ), f"Expected charlie@example.com in to/cc, got to={email.to}, cc={email.cc}"

    # Verify BCC recipient
    assert (
        "diana@example.com" in email.bcc
    ), f"Expected diana@example.com in 'bcc', got bcc={email.bcc}"


# ---------------------------------------------------------------------------
#  Outbound comms failure recovery
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@_handle_project
async def test_failed_sms_surfaces_error_in_conversation_thread(initialized_cm):
    """When send_sms fails at the comms layer, the error should be pushed into
    the conversation thread so the brain can see what went wrong on the next turn.

    Today, failed outbound comms publish an Error event that has no registered
    handler. The error is silently dropped: nothing appears in the conversation
    thread, no notification is pushed, and no follow-up brain turn is triggered
    in production. The brain is completely blind to the failure.

    This test verifies that after a comms-layer SMS failure, the conversation
    thread contains evidence of the error — enabling the brain to reason about
    the failure and attempt recovery on subsequent turns.
    """
    cm = initialized_cm
    contact = TEST_CONTACTS[0]  # Alice (has phone number)

    # Patch comms to always fail so we can inspect the thread after the attempt.
    original_send_sms = comms_utils.send_sms_message_via_number

    async def always_fail(*args, **kwargs):
        return {"success": False}

    comms_utils.send_sms_message_via_number = always_fail
    try:
        result = await cm.step_until_wait(
            SMSReceived(
                contact=contact,
                content="Tell me a joke",
            ),
            max_steps=3,
        )
    finally:
        comms_utils.send_sms_message_via_number = original_send_sms

    # The brain should have attempted send_sms at least once.
    assert (
        "send_sms" in cm.all_tool_calls
    ), f"Expected brain to attempt send_sms, but tool calls were: {cm.all_tool_calls}"

    # The failure should be visible in the conversation thread for this contact.
    # After the fix, the comms error will be pushed as a system message so the
    # brain can see it on the next turn and decide how to recover.
    contact_messages = cm.contact_index.get_messages_for_contact(
        contact["contact_id"],
        medium=Medium.SMS_MESSAGE,
    )
    thread_text = " ".join(getattr(m, "content", "") for m in contact_messages).lower()

    assert "fail" in thread_text or "error" in thread_text, (
        "Failed outbound SMS should produce a visible error in the conversation "
        "thread so the brain can reason about the failure on subsequent turns. "
        "Currently the Error event is silently dropped (no registered handler). "
        f"SMS thread messages: {[getattr(m, 'content', repr(m)) for m in contact_messages]}"
    )
