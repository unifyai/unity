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
from tests.test_conversation_manager.conftest import TEST_CONTACTS
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

pytestmark = pytest.mark.eval


def _only(events, typ):
    return [e for e in events if isinstance(e, typ)]


def _has_one(events, typ):
    """Assert exactly one event of the given type exists."""
    matches = _only(events, typ)
    count = len(matches)
    assert count == 1, f"Expected exactly 1 {typ.__name__}, got {count}"
    return True


# ---------------------------------------------------------------------------
#  Text-based communication tests (SMS, Email, UnifyMessage)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@_handle_project
async def test_sms_to_sms(initialized_cm):
    """SMS request for joke -> reply via SMS."""
    cm = initialized_cm
    contact = TEST_CONTACTS[1]

    result = await cm.step_until_wait(
        SMSReceived(
            contact=contact,
            content="Tell me a joke",
        ),
    )

    # Should have exactly one SMS sent (the reply)
    _has_one(result.output_events, SMSSent)
    sms = _only(result.output_events, SMSSent)[0]
    assert sms.contact["phone_number"] == contact["phone_number"]
    assert sms.content


@pytest.mark.asyncio
@_handle_project
async def test_sms_to_email(initialized_cm):
    """SMS request for joke via email -> should send email (may also ack via SMS)."""
    cm = initialized_cm
    contact = TEST_CONTACTS[1]

    result = await cm.step_until_wait(
        SMSReceived(
            contact=contact,
            content="Tell me a joke via email",
        ),
    )

    # Must have exactly one email sent (the target)
    _has_one(result.output_events, EmailSent)
    email = _only(result.output_events, EmailSent)[0]
    assert email.contact["email_address"] == contact["email_address"]
    assert email.body


@pytest.mark.asyncio
@_handle_project
async def test_sms_to_unify_message(initialized_cm):
    """SMS request for joke via unify message -> should send unify message."""
    cm = initialized_cm
    contact = TEST_CONTACTS[1]

    result = await cm.step_until_wait(
        SMSReceived(
            contact=contact,
            content="Tell me a joke via unify message",
        ),
    )

    # Must have exactly one unify message sent (the target)
    _has_one(result.output_events, UnifyMessageSent)
    msg = _only(result.output_events, UnifyMessageSent)[0]
    assert msg.contact["contact_id"] == contact["contact_id"]
    assert msg.content


@pytest.mark.asyncio
@_handle_project
async def test_sms_to_phone_call(initialized_cm):
    """SMS request for joke via phone call -> should initiate call."""
    cm = initialized_cm
    contact = TEST_CONTACTS[1]

    result = await cm.step_until_wait(
        SMSReceived(
            contact=contact,
            content="Tell me a joke via phone call",
        ),
    )

    # Must have exactly one phone call initiated (the target)
    _has_one(result.output_events, PhoneCallSent)
    call = _only(result.output_events, PhoneCallSent)[0]
    assert call.contact["phone_number"] == contact["phone_number"]


@pytest.mark.asyncio
@_handle_project
async def test_email_to_email(initialized_cm):
    """Email request for joke -> reply via email."""
    cm = initialized_cm
    contact = TEST_CONTACTS[1]

    result = await cm.step_until_wait(
        EmailReceived(
            contact=contact,
            subject="Test Subject",
            body="Tell me a joke",
            email_id="test_email_id",
        ),
    )

    # Should have exactly one email sent (the reply)
    _has_one(result.output_events, EmailSent)
    email = _only(result.output_events, EmailSent)[0]
    assert email.contact["email_address"] == contact["email_address"]
    assert email.subject
    assert email.body


@pytest.mark.asyncio
@_handle_project
async def test_email_to_sms(initialized_cm):
    """Email request for joke via SMS -> should send SMS."""
    cm = initialized_cm
    contact = TEST_CONTACTS[1]

    result = await cm.step_until_wait(
        EmailReceived(
            contact=contact,
            subject="Test Subject",
            body="Tell me a joke via SMS",
            email_id="test_email_id",
        ),
    )

    # Must have SMS sent (the target)
    _has_one(result.output_events, SMSSent)
    sms = _only(result.output_events, SMSSent)[0]
    assert sms.contact["phone_number"] == contact["phone_number"]
    assert sms.content


@pytest.mark.asyncio
@_handle_project
async def test_email_to_unify_message(initialized_cm):
    """Email request for joke via unify message -> should send unify message."""
    cm = initialized_cm
    contact = TEST_CONTACTS[1]

    result = await cm.step_until_wait(
        EmailReceived(
            contact=contact,
            subject="Test Subject",
            body="Tell me a joke via unify message",
            email_id="test_email_id",
        ),
    )

    # Must have exactly one unify message sent (the target)
    _has_one(result.output_events, UnifyMessageSent)
    msg = _only(result.output_events, UnifyMessageSent)[0]
    assert msg.contact["contact_id"] == contact["contact_id"]
    assert msg.content


@pytest.mark.asyncio
@_handle_project
async def test_email_to_phone_call(initialized_cm):
    """Email request for joke via phone call -> should initiate call."""
    cm = initialized_cm
    contact = TEST_CONTACTS[1]

    result = await cm.step_until_wait(
        EmailReceived(
            contact=contact,
            subject="Test Subject",
            body="Tell me a joke via phone call",
            email_id="test_email_id",
        ),
    )

    # Must have exactly one phone call initiated (the target)
    _has_one(result.output_events, PhoneCallSent)
    call = _only(result.output_events, PhoneCallSent)[0]
    assert call.contact["phone_number"] == contact["phone_number"]


@pytest.mark.asyncio
@_handle_project
async def test_unify_message_to_unify_message(initialized_cm):
    """Unify message request for joke -> reply via unify message."""
    cm = initialized_cm
    contact = TEST_CONTACTS[1]

    result = await cm.step_until_wait(
        UnifyMessageReceived(
            contact=contact,
            content="Tell me a joke",
        ),
    )

    # Should have exactly one unify message sent (the reply)
    _has_one(result.output_events, UnifyMessageSent)
    msg = _only(result.output_events, UnifyMessageSent)[0]
    assert msg.contact["contact_id"] == contact["contact_id"]
    assert msg.content


@pytest.mark.asyncio
@_handle_project
async def test_unify_message_to_sms(initialized_cm):
    """Unify message request for joke via SMS -> should send SMS."""
    cm = initialized_cm
    contact = TEST_CONTACTS[1]

    result = await cm.step_until_wait(
        UnifyMessageReceived(
            contact=contact,
            content="Tell me a joke via SMS",
        ),
    )

    # Must have SMS sent (the target)
    _has_one(result.output_events, SMSSent)
    sms = _only(result.output_events, SMSSent)[0]
    assert sms.contact["phone_number"] == contact["phone_number"]
    assert sms.content


@pytest.mark.asyncio
@_handle_project
async def test_unify_message_to_email(initialized_cm):
    """Unify message request for joke via email -> should send email."""
    cm = initialized_cm
    contact = TEST_CONTACTS[1]

    result = await cm.step_until_wait(
        UnifyMessageReceived(
            contact=contact,
            content="Tell me a joke via email",
        ),
    )

    # Must have exactly one email sent (the target)
    _has_one(result.output_events, EmailSent)
    email = _only(result.output_events, EmailSent)[0]
    assert email.contact["email_address"] == contact["email_address"]
    assert email.body


@pytest.mark.asyncio
@_handle_project
async def test_unify_message_to_phone_call(initialized_cm):
    """Unify message request for joke via phone call -> should initiate call."""
    cm = initialized_cm
    contact = TEST_CONTACTS[1]

    result = await cm.step_until_wait(
        UnifyMessageReceived(
            contact=contact,
            content="Tell me a joke via phone call",
        ),
    )

    # Must have exactly one phone call initiated (the target)
    _has_one(result.output_events, PhoneCallSent)
    call = _only(result.output_events, PhoneCallSent)[0]
    assert call.contact["phone_number"] == contact["phone_number"]


# ---------------------------------------------------------------------------
#  Voice call tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@_handle_project
async def test_phone_call(initialized_cm):
    """Basic phone call flow - just verify utterance is recorded."""
    cm = initialized_cm
    contact = TEST_CONTACTS[1]

    await cm.step(
        PhoneCallReceived(
            contact=contact,
            conference_name="test_conference",
        ),
    )
    await cm.step(PhoneCallStarted(contact=contact))
    await cm.step(InboundPhoneUtterance(contact=contact, content="Tell me a joke"))

    voice_thread = list(
        cm.contact_index.active_conversations[contact["contact_id"]].threads["voice"],
    )
    assert any(getattr(m, "content", None) == "Tell me a joke" for m in voice_thread)

    await cm.step(PhoneCallEnded(contact=contact))


@pytest.mark.asyncio
@_handle_project
async def test_phone_call_to_sms(initialized_cm):
    """During phone call, request joke via SMS -> should send SMS."""
    cm = initialized_cm
    contact = TEST_CONTACTS[1]

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
    _has_one(result.output_events, SMSSent)

    await cm.step(PhoneCallEnded(contact=contact))


@pytest.mark.asyncio
@_handle_project
async def test_phone_call_to_email(initialized_cm):
    """During phone call, request joke via email -> should send email."""
    cm = initialized_cm
    contact = TEST_CONTACTS[1]

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
    _has_one(result.output_events, EmailSent)

    await cm.step(PhoneCallEnded(contact=contact))


@pytest.mark.asyncio
@_handle_project
async def test_phone_call_to_unify_message(initialized_cm):
    """During phone call, request joke via unify message -> should send unify message."""
    cm = initialized_cm
    contact = TEST_CONTACTS[1]

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
    _has_one(result.output_events, UnifyMessageSent)

    await cm.step(PhoneCallEnded(contact=contact))


# ---------------------------------------------------------------------------
#  Unify Meet tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@_handle_project
async def test_unify_meet(initialized_cm):
    """Basic unify meet flow - just verify utterance is recorded."""
    cm = initialized_cm
    contact = TEST_CONTACTS[1]

    await cm.step(UnifyMeetReceived(contact=contact))
    await cm.step(UnifyMeetStarted(contact=contact))
    await cm.step(InboundUnifyMeetUtterance(contact=contact, content="Tell me a joke"))

    voice_thread = list(
        cm.contact_index.active_conversations[contact["contact_id"]].threads["voice"],
    )
    assert any(getattr(m, "content", None) == "Tell me a joke" for m in voice_thread)

    await cm.step(UnifyMeetEnded(contact=contact))


# Note: There is no test_unify_meet_to_phone_call test because the system does not
# support maintaining multiple simultaneous voice-based conversations. While on a
# unify meet, the assistant cannot initiate an outbound phone call.


@pytest.mark.asyncio
@_handle_project
async def test_unify_meet_to_sms(initialized_cm):
    """During unify meet, request joke via SMS -> should send SMS."""
    cm = initialized_cm
    contact = TEST_CONTACTS[1]

    await cm.step(UnifyMeetReceived(contact=contact))
    await cm.step(UnifyMeetStarted(contact=contact))

    result = await cm.step_until_wait(
        InboundUnifyMeetUtterance(
            contact=contact,
            content="Send me a joke via sms right now",
        ),
    )

    # Must have SMS sent
    _has_one(result.output_events, SMSSent)

    await cm.step(UnifyMeetEnded(contact=contact))


@pytest.mark.asyncio
@_handle_project
async def test_unify_meet_to_email(initialized_cm):
    """During unify meet, request joke via email -> should send email."""
    cm = initialized_cm
    contact = TEST_CONTACTS[1]

    await cm.step(UnifyMeetReceived(contact=contact))
    await cm.step(UnifyMeetStarted(contact=contact))

    result = await cm.step_until_wait(
        InboundUnifyMeetUtterance(
            contact=contact,
            content="Send me a joke via email right now",
        ),
    )

    # Must have exactly one email sent
    _has_one(result.output_events, EmailSent)

    await cm.step(UnifyMeetEnded(contact=contact))


@pytest.mark.asyncio
@_handle_project
async def test_unify_meet_to_unify_message(initialized_cm):
    """During unify meet, request joke via unify message -> should send unify message."""
    cm = initialized_cm
    contact = TEST_CONTACTS[1]

    await cm.step(UnifyMeetReceived(contact=contact))
    await cm.step(UnifyMeetStarted(contact=contact))

    result = await cm.step_until_wait(
        InboundUnifyMeetUtterance(
            contact=contact,
            content="Send me a joke via unify message right now",
        ),
    )

    # Must have exactly one unify message sent
    _has_one(result.output_events, UnifyMessageSent)

    await cm.step(UnifyMeetEnded(contact=contact))
