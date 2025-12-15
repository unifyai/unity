"""
tests/test_conversation_manager/test_events.py
==============================================

Tests for conversation manager event publishing across all communication mediums.

Verifies that the correct inbound and outbound events are published for:
- Browser voice (UNIFY_MEET): InboundUnifyMeetUtterance
- Phone calls (PHONE_CALL): InboundPhoneUtterance
- Text chat (UNIFY_MESSAGE): UnifyMessageReceived / UnifyMessageSent
- SMS (SMS_MESSAGE): SMSReceived / SMSSent
- Email (EMAIL): EmailReceived / EmailSent

Note: Voice modes (UNIFY_MEET, PHONE_CALL) only verify inbound events since
the Main CM Brain provides guidance to the Voice Agent (fast brain) which
handles all speech output. The Voice Agent isn't running in these tests.
"""

import asyncio

import pytest

from tests.helpers import _handle_project
from tests.test_conversation_manager.helpers import (
    contacts,
    send_incoming_call,
    send_incoming_sms,
    send_incoming_email,
    send_incoming_unify_message,
)
from unity.conversation_manager.events import (
    EmailReceived,
    EmailSent,
    InboundPhoneUtterance,
    InboundUnifyMeetUtterance,
    PhoneCallEnded,
    SMSReceived,
    SMSSent,
    UnifyMeetEnded,
    UnifyMessageReceived,
    UnifyMessageSent,
)


# =============================================================================
# Voice Medium Tests
# =============================================================================


@pytest.mark.asyncio
@_handle_project
async def test_unify_meet_publishes_utterance_events(
    test_redis_client,
    event_capture,
):
    """
    Verify that browser voice calls publish inbound utterance events.

    In the voice mode architecture, the Main CM Brain only provides guidance
    to the Voice Agent (fast brain) - it doesn't produce speech directly.
    The Voice Agent (not running in these tests) handles all conversational
    responses.
    """
    event_capture.clear()

    contact = contacts[1]

    pubsub = await send_incoming_call(
        test_redis_client,
        contact,
        "test_conference",
        "Tell me a joke",
        mode="unify_meet",
    )

    # Give Main CM Brain time to process (it may or may not produce guidance)
    await asyncio.sleep(3.0)

    # Verify inbound event was recorded
    inbound_events = event_capture.get_events(InboundUnifyMeetUtterance)
    assert len(inbound_events) >= 1
    inbound = inbound_events[0]
    assert inbound.content == "Tell me a joke"

    await pubsub.unsubscribe("app:call:call_guidance")
    await pubsub.aclose()

    await test_redis_client.publish(
        "app:comms:unify_meet_ended",
        UnifyMeetEnded(contact=contact).to_json(),
    )


@pytest.mark.asyncio
@_handle_project
async def test_phone_call_publishes_utterance_events(
    test_redis_client,
    event_capture,
):
    """
    Verify that phone calls publish inbound utterance events.

    In the voice mode architecture, the Main CM Brain only provides guidance
    to the Voice Agent (fast brain) - it doesn't produce speech directly.
    The Voice Agent (not running in these tests) handles all conversational
    responses. The Main CM Brain may or may not send guidance depending on
    whether it has data to exchange with the Voice Agent.
    """
    event_capture.clear()

    contact = contacts[1]

    pubsub = await send_incoming_call(
        test_redis_client,
        contact,
        "test_conference",
        "Tell me a joke",
        mode="call",
    )

    # Give Main CM Brain time to process (it may or may not produce guidance)
    await asyncio.sleep(3.0)

    # Verify inbound event was recorded
    inbound_events = event_capture.get_events(InboundPhoneUtterance)
    assert len(inbound_events) >= 1
    inbound = inbound_events[0]
    assert inbound.content == "Tell me a joke"

    await pubsub.unsubscribe("app:call:call_guidance")
    await pubsub.aclose()

    await test_redis_client.publish(
        "app:comms:phone_call_ended",
        PhoneCallEnded(contact=contact).to_json(),
    )


# =============================================================================
# Text Medium Tests
# =============================================================================


@pytest.mark.asyncio
@_handle_project
async def test_unify_message_publishes_events(
    test_redis_client,
    event_capture,
):
    """
    Verify that text chat messages publish inbound and outbound events.
    """
    event_capture.clear()

    contact = contacts[1]

    await send_incoming_unify_message(test_redis_client, contact, "Tell me a joke")

    # Wait for outbound event
    outbound = await event_capture.wait_for_event(UnifyMessageSent, timeout=60.0)

    # Verify inbound event
    inbound_events = event_capture.get_events(UnifyMessageReceived)
    assert len(inbound_events) >= 1
    inbound = inbound_events[0]
    assert inbound.content == "Tell me a joke"

    # Verify outbound event
    assert hasattr(outbound, "contact")
    assert hasattr(outbound, "content")
    assert len(outbound.content) > 0


@pytest.mark.asyncio
@_handle_project
async def test_sms_publishes_events(
    test_redis_client,
    event_capture,
):
    """
    Verify that SMS messages publish inbound and outbound events.
    """
    event_capture.clear()

    contact = contacts[1]

    await send_incoming_sms(test_redis_client, contact, "Tell me a joke")

    # Wait for outbound event
    outbound = await event_capture.wait_for_event(SMSSent, timeout=60.0)

    # Verify inbound event
    inbound_events = event_capture.get_events(SMSReceived)
    assert len(inbound_events) >= 1
    inbound = inbound_events[0]
    assert inbound.content == "Tell me a joke"

    # Verify outbound event
    assert hasattr(outbound, "contact")
    assert hasattr(outbound, "content")
    assert len(outbound.content) > 0


@pytest.mark.asyncio
@_handle_project
async def test_email_publishes_events(
    test_redis_client,
    event_capture,
):
    """
    Verify that email messages publish inbound and outbound events.
    """
    event_capture.clear()

    contact = contacts[1]

    await send_incoming_email(
        test_redis_client,
        contact,
        subject="Test Subject",
        body="Tell me a joke",
        email_id="test_email_123",
    )

    # Wait for outbound event
    outbound = await event_capture.wait_for_event(EmailSent, timeout=60.0)

    # Verify inbound event
    inbound_events = event_capture.get_events(EmailReceived)
    assert len(inbound_events) >= 1
    inbound = inbound_events[0]
    assert inbound.subject == "Test Subject"
    assert inbound.body == "Tell me a joke"

    # Verify outbound event
    assert hasattr(outbound, "contact")
    assert hasattr(outbound, "subject")
    assert hasattr(outbound, "body")
    assert len(outbound.body) > 0
