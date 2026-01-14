"""
tests/test_conversation_manager/test_comms.py
=============================================

Tests for communication flows (SMS, email, calls, etc.)

Uses the ConversationManager "single-step" API:
- Construct an input Event
- Call `await cm._step(event)`
- Assert on the returned output events and/or resulting state

Voice call tests verify that events are handled correctly. In the voice
architecture, the Main CM Brain only provides guidance to the Voice Agent
(fast brain) - it doesn't produce speech directly. The Voice Agent handles
all conversational responses. These tests verify event flow, not speech output.
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


@pytest.mark.asyncio
@_handle_project
async def test_sms_to_sms(initialized_cm):
    cm = initialized_cm
    contact = TEST_CONTACTS[1]

    result = await cm._step(
        SMSReceived(
            contact=contact,
            content="Tell me a joke",
        ),
    )

    out = _only(result.output_events, SMSSent)
    assert len(out) >= 1
    sms = out[0]
    assert sms.contact["phone_number"] == contact["phone_number"]
    assert sms.content


@pytest.mark.asyncio
@_handle_project
async def test_sms_to_email(initialized_cm):
    cm = initialized_cm
    contact = TEST_CONTACTS[1]

    result = await cm._step(
        SMSReceived(
            contact=contact,
            content="Tell me a joke via email",
        ),
    )

    out = _only(result.output_events, EmailSent)
    assert len(out) >= 1
    email = out[0]
    assert email.contact["email_address"] == contact["email_address"]
    assert email.body


@pytest.mark.asyncio
@_handle_project
async def test_sms_to_unify_message(initialized_cm):
    cm = initialized_cm
    contact = TEST_CONTACTS[1]

    result = await cm._step(
        SMSReceived(
            contact=contact,
            content="Tell me a joke via unify message",
        ),
    )

    out = _only(result.output_events, UnifyMessageSent)
    assert len(out) >= 1
    msg = out[0]
    assert msg.contact["contact_id"] == contact["contact_id"]
    assert msg.content


@pytest.mark.asyncio
@_handle_project
async def test_sms_to_phone_call(initialized_cm):
    cm = initialized_cm
    contact = TEST_CONTACTS[1]

    result = await cm._step(
        SMSReceived(
            contact=contact,
            content="Tell me a joke via phone call",
        ),
    )

    out = _only(result.output_events, PhoneCallSent)
    assert len(out) >= 1
    call = out[0]
    assert call.contact["phone_number"] == contact["phone_number"]


@pytest.mark.asyncio
@_handle_project
async def test_email_to_email(initialized_cm):
    cm = initialized_cm
    contact = TEST_CONTACTS[1]

    result = await cm._step(
        EmailReceived(
            contact=contact,
            subject="Test Subject",
            body="Tell me a joke",
            email_id="test_email_id",
        ),
    )

    out = _only(result.output_events, EmailSent)
    assert len(out) >= 1
    email = out[0]
    assert email.contact["email_address"] == contact["email_address"]
    assert email.subject
    assert email.body


@pytest.mark.asyncio
@_handle_project
async def test_email_to_sms(initialized_cm):
    cm = initialized_cm
    contact = TEST_CONTACTS[1]

    result = await cm._step(
        EmailReceived(
            contact=contact,
            subject="Test Subject",
            body="Tell me a joke via SMS",
            email_id="test_email_id",
        ),
    )

    out = _only(result.output_events, SMSSent)
    assert len(out) >= 1
    sms = out[0]
    assert sms.contact["phone_number"] == contact["phone_number"]
    assert sms.content


@pytest.mark.asyncio
@_handle_project
async def test_email_to_unify_message(initialized_cm):
    cm = initialized_cm
    contact = TEST_CONTACTS[1]

    result = await cm._step(
        EmailReceived(
            contact=contact,
            subject="Test Subject",
            body="Tell me a joke via unify message",
            email_id="test_email_id",
        ),
    )

    out = _only(result.output_events, UnifyMessageSent)
    assert len(out) >= 1
    msg = out[0]
    assert msg.contact["contact_id"] == contact["contact_id"]
    assert msg.content


@pytest.mark.asyncio
@_handle_project
async def test_email_to_phone_call(initialized_cm):
    cm = initialized_cm
    contact = TEST_CONTACTS[1]

    result = await cm._step(
        EmailReceived(
            contact=contact,
            subject="Test Subject",
            body="Tell me a joke via phone call",
            email_id="test_email_id",
        ),
    )

    out = _only(result.output_events, PhoneCallSent)
    assert len(out) >= 1
    call = out[0]
    assert call.contact["phone_number"] == contact["phone_number"]


@pytest.mark.asyncio
@_handle_project
async def test_unify_message_to_unify_message(initialized_cm):
    cm = initialized_cm
    contact = TEST_CONTACTS[1]

    result = await cm._step(
        UnifyMessageReceived(
            contact=contact,
            content="Tell me a joke",
        ),
    )

    out = _only(result.output_events, UnifyMessageSent)
    assert len(out) >= 1
    msg = out[0]
    assert msg.contact["contact_id"] == contact["contact_id"]
    assert msg.content


@pytest.mark.asyncio
@_handle_project
async def test_unify_message_to_sms(initialized_cm):
    cm = initialized_cm
    contact = TEST_CONTACTS[1]

    result = await cm._step(
        UnifyMessageReceived(
            contact=contact,
            content="Tell me a joke via SMS",
        ),
    )

    out = _only(result.output_events, SMSSent)
    assert len(out) >= 1
    sms = out[0]
    assert sms.contact["phone_number"] == contact["phone_number"]
    assert sms.content


@pytest.mark.asyncio
@_handle_project
async def test_unify_message_to_email(initialized_cm):
    cm = initialized_cm
    contact = TEST_CONTACTS[1]

    result = await cm._step(
        UnifyMessageReceived(
            contact=contact,
            content="Tell me a joke via email",
        ),
    )

    out = _only(result.output_events, EmailSent)
    assert len(out) >= 1
    email = out[0]
    assert email.contact["email_address"] == contact["email_address"]
    assert email.body


@pytest.mark.asyncio
@_handle_project
async def test_unify_message_to_phone_call(initialized_cm):
    cm = initialized_cm
    contact = TEST_CONTACTS[1]

    result = await cm._step(
        UnifyMessageReceived(
            contact=contact,
            content="Tell me a joke via phone call",
        ),
    )

    out = _only(result.output_events, PhoneCallSent)
    assert len(out) >= 1
    call = out[0]
    assert call.contact["phone_number"] == contact["phone_number"]


@pytest.mark.asyncio
@_handle_project
async def test_phone_call(initialized_cm):
    cm = initialized_cm
    contact = TEST_CONTACTS[1]

    await cm._step(
        PhoneCallReceived(
            contact=contact,
            conference_name="test_conference",
        ),
    )
    await cm._step(PhoneCallStarted(contact=contact))
    await cm._step(InboundPhoneUtterance(contact=contact, content="Tell me a joke"))

    voice_thread = list(
        cm.contact_index.active_conversations[contact["contact_id"]].threads["voice"],
    )
    assert any(getattr(m, "content", None) == "Tell me a joke" for m in voice_thread)

    await cm._step(PhoneCallEnded(contact=contact))


@pytest.mark.asyncio
@_handle_project
async def test_phone_call_to_sms(initialized_cm):
    cm = initialized_cm
    contact = TEST_CONTACTS[1]

    await cm._step(
        PhoneCallReceived(
            contact=contact,
            conference_name="test_conference",
        ),
    )
    await cm._step(PhoneCallStarted(contact=contact))

    result = await cm._step(
        InboundPhoneUtterance(
            contact=contact,
            content="Tell me a joke via SMS right now",
        ),
    )

    out = _only(result.output_events, SMSSent)
    assert len(out) >= 1

    await cm._step(PhoneCallEnded(contact=contact))


@pytest.mark.asyncio
@_handle_project
async def test_phone_call_to_email(initialized_cm):
    cm = initialized_cm
    contact = TEST_CONTACTS[1]

    await cm._step(
        PhoneCallReceived(
            contact=contact,
            conference_name="test_conference",
        ),
    )
    await cm._step(PhoneCallStarted(contact=contact))

    result = await cm._step(
        InboundPhoneUtterance(
            contact=contact,
            content="Tell me a joke via email right now",
        ),
    )

    out = _only(result.output_events, EmailSent)
    assert len(out) >= 1

    await cm._step(PhoneCallEnded(contact=contact))


@pytest.mark.asyncio
@_handle_project
async def test_phone_call_to_unify_message(initialized_cm):
    cm = initialized_cm
    contact = TEST_CONTACTS[1]

    await cm._step(
        PhoneCallReceived(
            contact=contact,
            conference_name="test_conference",
        ),
    )
    await cm._step(PhoneCallStarted(contact=contact))

    result = await cm._step(
        InboundPhoneUtterance(
            contact=contact,
            content="Tell me a joke via unify message right now",
        ),
    )

    out = _only(result.output_events, UnifyMessageSent)
    assert len(out) >= 1

    await cm._step(PhoneCallEnded(contact=contact))


@pytest.mark.asyncio
@_handle_project
async def test_unify_meet(initialized_cm):
    cm = initialized_cm
    contact = TEST_CONTACTS[1]

    await cm._step(UnifyMeetReceived(contact=contact))
    await cm._step(UnifyMeetStarted(contact=contact))
    await cm._step(InboundUnifyMeetUtterance(contact=contact, content="Tell me a joke"))

    voice_thread = list(
        cm.contact_index.active_conversations[contact["contact_id"]].threads["voice"],
    )
    assert any(getattr(m, "content", None) == "Tell me a joke" for m in voice_thread)

    await cm._step(UnifyMeetEnded(contact=contact))


# Note: There is no test_unify_meet_to_phone_call test because the system does not
# support maintaining multiple simultaneous voice-based conversations. While on a
# unify meet, the assistant cannot initiate an outbound phone call.


@pytest.mark.asyncio
@_handle_project
async def test_unify_meet_to_sms(initialized_cm):
    cm = initialized_cm
    contact = TEST_CONTACTS[1]

    await cm._step(UnifyMeetReceived(contact=contact))
    await cm._step(UnifyMeetStarted(contact=contact))

    result = await cm._step(
        InboundUnifyMeetUtterance(
            contact=contact,
            content="Tell me a joke via sms right now",
        ),
    )

    out = _only(result.output_events, SMSSent)
    assert len(out) >= 1

    await cm._step(UnifyMeetEnded(contact=contact))


@pytest.mark.asyncio
@_handle_project
async def test_unify_meet_to_email(initialized_cm):
    cm = initialized_cm
    contact = TEST_CONTACTS[1]

    await cm._step(UnifyMeetReceived(contact=contact))
    await cm._step(UnifyMeetStarted(contact=contact))

    result = await cm._step(
        InboundUnifyMeetUtterance(
            contact=contact,
            content="Tell me a joke via email right now",
        ),
    )

    out = _only(result.output_events, EmailSent)
    assert len(out) >= 1

    await cm._step(UnifyMeetEnded(contact=contact))


@pytest.mark.asyncio
@_handle_project
async def test_unify_meet_to_unify_message(initialized_cm):
    cm = initialized_cm
    contact = TEST_CONTACTS[1]

    await cm._step(UnifyMeetReceived(contact=contact))
    await cm._step(UnifyMeetStarted(contact=contact))

    result = await cm._step(
        InboundUnifyMeetUtterance(
            contact=contact,
            content="Tell me a joke via unify message right now",
        ),
    )

    out = _only(result.output_events, UnifyMessageSent)
    assert len(out) >= 1

    await cm._step(UnifyMeetEnded(contact=contact))
