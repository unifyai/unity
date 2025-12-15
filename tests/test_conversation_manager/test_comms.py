"""
tests/test_conversation_manager/test_comms.py
=============================================

Tests for communication flows (SMS, email, calls, etc.)

Voice call tests verify that events are handled correctly. In the voice
architecture, the Main CM Brain only provides guidance to the Voice Agent
(fast brain) - it doesn't produce speech directly. The Voice Agent handles
all conversational responses. These tests verify event flow, not speech output.
"""

import asyncio

import pytest

pytestmark = pytest.mark.eval

from tests.helpers import _handle_project
from tests.test_conversation_manager.helpers import (
    contacts,
    capture_outgoing_email,
    capture_outgoing_phone_call,
    capture_outgoing_sms,
    capture_outgoing_unify_message,
    send_incoming_email,
    send_incoming_call,
    send_incoming_sms,
    send_incoming_unify_message,
)
from unity.conversation_manager.events import PhoneCallEnded, UnifyMeetEnded


@pytest.mark.asyncio
@_handle_project
async def test_sms_to_sms(test_redis_client, event_capture):
    """
    Test basic SMS flow: send an incoming SMS and receive a response.

    Flow:
    1. Send SMSRecieved event with a question
    2. CM processes it with LLM
    3. CM publishes SMSSent event with response
    4. We capture and verify the response
    """
    # Clear any events from initialization
    event_capture.clear()

    # Send incoming SMS
    contact = contacts[1]
    await send_incoming_sms(test_redis_client, contact, "Tell me a joke")

    # Capture outgoing SMS and verify response
    await capture_outgoing_sms(event_capture, contact)


@pytest.mark.asyncio
@_handle_project
async def test_sms_to_email(test_redis_client, event_capture):
    """
    Test SMS to email flow: send an incoming SMS and receive a response.
    """
    # Clear any events from initialization
    event_capture.clear()

    # Send incoming SMS
    contact = contacts[1]
    await send_incoming_sms(test_redis_client, contact, "Tell me a joke via email")

    # Capture outgoing email and verify response
    await capture_outgoing_email(event_capture, contact)


@pytest.mark.asyncio
@_handle_project
async def test_sms_to_unify_message(test_redis_client, event_capture):
    """
    Test SMS to unify message flow: send an incoming SMS and receive a response.
    """
    # Clear any events from initialization
    event_capture.clear()

    # Send incoming SMS
    contact = contacts[1]
    await send_incoming_sms(
        test_redis_client,
        contact,
        "Tell me a joke via unify message",
    )

    # Capture outgoing unify message and verify response
    await capture_outgoing_unify_message(event_capture, contact)


@pytest.mark.asyncio
@_handle_project
async def test_sms_to_phone_call(test_redis_client, event_capture):
    """
    Test SMS to phone call flow: send an incoming SMS and receive a response.
    """
    # Clear any events from initialization
    event_capture.clear()

    # Send incoming SMS
    contact = contacts[1]
    await send_incoming_sms(test_redis_client, contact, "Tell me a joke via phone call")

    # Capture outgoing phone call and verify response
    await capture_outgoing_phone_call(event_capture, contact)

    # End the phone call
    await test_redis_client.publish(
        "app:comms:phone_call_ended",
        PhoneCallEnded(contact=contact).to_json(),
    )


@pytest.mark.asyncio
@_handle_project
async def test_email_to_email(test_redis_client, event_capture):
    """
    Test basic email flow: send an incoming email and receive a response.

    Flow:
    1. Send EmailRecieved event with a question
    2. CM processes it with LLM
    3. CM publishes EmailSent event with response
    4. We capture and verify the response
    """
    # Clear any events from initialization
    event_capture.clear()

    # Send incoming email
    contact = contacts[1]
    email_id = "test_email_id"
    await send_incoming_email(
        test_redis_client,
        contact,
        "Test Subject",
        "Tell me a joke",
        email_id,
    )

    # Capture outgoing email and verify response
    await capture_outgoing_email(event_capture, contact, email_id)


@pytest.mark.asyncio
@_handle_project
async def test_email_to_sms(test_redis_client, event_capture):
    """
    Test email to SMS flow: send an incoming email and receive a response.
    """
    # Clear any events from initialization
    event_capture.clear()

    # Send incoming email
    contact = contacts[1]
    email_id = "test_email_id"
    await send_incoming_email(
        test_redis_client,
        contact,
        "Test Subject",
        "Tell me a joke via SMS",
        email_id,
    )

    # Capture outgoing SMS and verify response
    await capture_outgoing_sms(event_capture, contact)


@pytest.mark.asyncio
@_handle_project
async def test_email_to_unify_message(test_redis_client, event_capture):
    """
    Test email to unify message flow: send an incoming email and receive a response.
    """
    # Clear any events from initialization
    event_capture.clear()

    # Send incoming email
    contact = contacts[1]
    email_id = "test_email_id"
    await send_incoming_email(
        test_redis_client,
        contact,
        "Test Subject",
        "Tell me a joke via unify message",
        email_id,
    )

    # Capture outgoing unify message and verify response
    await capture_outgoing_unify_message(event_capture, contact)


@pytest.mark.asyncio
@_handle_project
async def test_email_to_phone_call(test_redis_client, event_capture):
    """
    Test email to phone call flow: send an incoming email and receive a response.
    """
    # Clear any events from initialization
    event_capture.clear()

    # Send incoming email
    contact = contacts[1]
    email_id = "test_email_id"
    await send_incoming_email(
        test_redis_client,
        contact,
        "Test Subject",
        "Tell me a joke via phone call",
        email_id,
    )

    # Capture outgoing phone call and verify response
    await capture_outgoing_phone_call(event_capture, contact)

    # End the phone call
    await test_redis_client.publish(
        "app:comms:phone_call_ended",
        PhoneCallEnded(contact=contact).to_json(),
    )


@pytest.mark.asyncio
@_handle_project
async def test_unify_message_to_unify_message(test_redis_client, event_capture):
    """
    Test unify message to unify message flow: send an incoming unify message and
    receive a response.
    """
    # Clear any events from initialization
    event_capture.clear()

    # Send incoming unify message
    contact = contacts[1]
    await send_incoming_unify_message(test_redis_client, contact, "Tell me a joke")

    # Capture outgoing unify message and verify response
    await capture_outgoing_unify_message(event_capture, contact)


@pytest.mark.asyncio
@_handle_project
async def test_unify_message_to_sms(test_redis_client, event_capture):
    """
    Test unify message to SMS flow: send an incoming unify message and receive
    a response via SMS.
    """
    # Clear any events from initialization
    event_capture.clear()

    # Send incoming unify message
    contact = contacts[1]
    await send_incoming_unify_message(
        test_redis_client,
        contact,
        "Tell me a joke via SMS",
    )

    # Capture outgoing SMS and verify response
    await capture_outgoing_sms(event_capture, contact)


@pytest.mark.asyncio
@_handle_project
async def test_unify_message_to_email(test_redis_client, event_capture):
    """
    Test unify message to email flow: send an incoming unify message and receive a
    response.
    """
    # Clear any events from initialization
    event_capture.clear()

    # Send incoming unify message
    contact = contacts[1]
    await send_incoming_unify_message(
        test_redis_client,
        contact,
        "Tell me a joke via email",
    )

    # Capture outgoing email and verify response
    await capture_outgoing_email(event_capture, contact)


@pytest.mark.asyncio
@_handle_project
async def test_unify_message_to_phone_call(test_redis_client, event_capture):
    """
    Test unify message to phone call flow: send an incoming unify message and receive a response.
    """
    # Clear any events from initialization
    event_capture.clear()

    # Send incoming unify message
    contact = contacts[1]
    await send_incoming_unify_message(
        test_redis_client,
        contact,
        "Tell me a joke via phone call",
    )

    # Capture outgoing phone call and verify response
    await capture_outgoing_phone_call(event_capture, contact)

    # End the phone call
    await test_redis_client.publish(
        "app:comms:phone_call_ended",
        PhoneCallEnded(contact=contact).to_json(),
    )


@pytest.mark.asyncio
@_handle_project
async def test_phone_call(test_redis_client, event_capture):
    """
    Test phone call flow.

    In the voice architecture, the Main CM Brain only provides guidance to the
    Voice Agent (fast brain) - it doesn't produce speech directly. The Voice
    Agent handles all conversational responses. We verify the call events are
    processed correctly.
    """
    from unity.conversation_manager.events import InboundPhoneUtterance

    # Clear any events from initialization
    event_capture.clear()

    # Send incoming phone call
    contact = contacts[1]
    pubsub = await send_incoming_call(
        test_redis_client,
        contact,
        "test_conference",
        "Tell me a joke",
    )

    # Give Main CM Brain time to process
    await asyncio.sleep(3.0)

    # Verify inbound utterance event was recorded
    inbound_events = event_capture.get_events(InboundPhoneUtterance)
    assert len(inbound_events) >= 1, "Should record inbound phone utterance"
    assert inbound_events[0].content == "Tell me a joke"

    # Cleanup subscription
    await pubsub.unsubscribe("app:call:call_guidance")
    await pubsub.aclose()

    print("\n✅ Phone call test complete!")

    # End the phone call
    await test_redis_client.publish(
        "app:comms:phone_call_ended",
        PhoneCallEnded(contact=contact).to_json(),
    )


@pytest.mark.asyncio
@_handle_project
async def test_phone_call_to_sms(test_redis_client, event_capture):
    """
    Test phone call to SMS flow: user on a call requests SMS, verify SMS is sent.

    The Main CM Brain provides guidance to the Voice Agent but doesn't produce
    speech directly. The key assertion is that the SMS action is executed.
    """
    # Clear any events from initialization
    event_capture.clear()

    # Send incoming phone call
    contact = contacts[1]
    pubsub = await send_incoming_call(
        test_redis_client,
        contact,
        "test_conference",
        "Tell me a joke via SMS right now",
    )

    # Cleanup subscription (we don't expect voice streaming)
    await pubsub.unsubscribe("app:call:call_guidance")
    await pubsub.aclose()

    # Capture outgoing SMS and verify response
    await capture_outgoing_sms(event_capture, contact)

    # End the phone call
    await test_redis_client.publish(
        "app:comms:phone_call_ended",
        PhoneCallEnded(contact=contact).to_json(),
    )


@pytest.mark.asyncio
@_handle_project
async def test_phone_call_to_email(test_redis_client, event_capture):
    """
    Test phone call to email flow: user on a call requests email, verify email is sent.

    The Main CM Brain provides guidance to the Voice Agent but doesn't produce
    speech directly. The key assertion is that the email action is executed.
    """
    # Clear any events from initialization
    event_capture.clear()

    # Send incoming phone call
    contact = contacts[1]
    pubsub = await send_incoming_call(
        test_redis_client,
        contact,
        "test_conference",
        "Tell me a joke via email right now",
    )

    # Cleanup subscription (we don't expect voice streaming)
    await pubsub.unsubscribe("app:call:call_guidance")
    await pubsub.aclose()

    # Capture outgoing email and verify response
    await capture_outgoing_email(event_capture, contact)

    # End the phone call
    await test_redis_client.publish(
        "app:comms:phone_call_ended",
        PhoneCallEnded(contact=contact).to_json(),
    )


@pytest.mark.asyncio
@_handle_project
async def test_phone_call_to_unify_message(test_redis_client, event_capture):
    """
    Test phone call to unify message flow: user on a call requests a message.

    The Main CM Brain provides guidance to the Voice Agent but doesn't produce
    speech directly. The key assertion is that the unify message action is executed.
    """
    # Clear any events from initialization
    event_capture.clear()

    # Send incoming phone call
    contact = contacts[1]
    pubsub = await send_incoming_call(
        test_redis_client,
        contact,
        "test_conference",
        "Tell me a joke via unify message right now",
    )

    # Cleanup subscription (we don't expect voice streaming)
    await pubsub.unsubscribe("app:call:call_guidance")
    await pubsub.aclose()

    # Capture outgoing unify message and verify response
    await capture_outgoing_unify_message(event_capture, contact)

    # End the phone call
    await test_redis_client.publish(
        "app:comms:phone_call_ended",
        PhoneCallEnded(contact=contact).to_json(),
    )


@pytest.mark.asyncio
@_handle_project
async def test_unify_meet(test_redis_client, event_capture):
    """
    Test unify meet flow.

    In the voice architecture, the Main CM Brain only provides guidance to the
    Voice Agent (fast brain) - it doesn't produce speech directly. The Voice
    Agent handles all conversational responses. We verify the call events are
    processed correctly.
    """
    from unity.conversation_manager.events import InboundUnifyMeetUtterance

    # Clear any events from initialization
    event_capture.clear()

    # Send incoming unify meet
    contact = contacts[1]
    pubsub = await send_incoming_call(
        test_redis_client,
        contact,
        "test_conference",
        "Tell me a joke",
        mode="unify_meet",
    )

    # Give Main CM Brain time to process
    await asyncio.sleep(3.0)

    # Verify inbound utterance event was recorded
    inbound_events = event_capture.get_events(InboundUnifyMeetUtterance)
    assert len(inbound_events) >= 1, "Should record inbound unify meet utterance"
    assert inbound_events[0].content == "Tell me a joke"

    # Cleanup subscription
    await pubsub.unsubscribe("app:call:call_guidance")
    await pubsub.aclose()

    print("\n✅ Unify meet test complete!")

    # End the unify meet
    await test_redis_client.publish(
        "app:comms:unify_meet_ended",
        UnifyMeetEnded(contact=contact).to_json(),
    )


# Note: There is no test_unify_meet_to_phone_call test because the system does not
# support maintaining multiple simultaneous voice-based conversations. While on a
# unify meet, the assistant cannot initiate an outbound phone call.


@pytest.mark.asyncio
@_handle_project
async def test_unify_meet_to_sms(test_redis_client, event_capture):
    """
    Test unify meet to SMS flow: user on a call requests SMS, verify SMS is sent.

    The Main CM Brain provides guidance to the Voice Agent but doesn't produce
    speech directly. The key assertion is that the SMS action is executed.
    """
    # Clear any events from initialization
    event_capture.clear()

    # Send incoming unify meet
    contact = contacts[1]
    pubsub = await send_incoming_call(
        test_redis_client,
        contact,
        "test_conference",
        "Tell me a joke via sms right now",
        mode="unify_meet",
    )

    # Cleanup subscription (we don't expect voice streaming)
    await pubsub.unsubscribe("app:call:call_guidance")
    await pubsub.aclose()

    # Capture outgoing SMS and verify response
    await capture_outgoing_sms(event_capture, contact)

    # End the unify meet
    await test_redis_client.publish(
        "app:comms:unify_meet_ended",
        UnifyMeetEnded(contact=contact).to_json(),
    )


@pytest.mark.asyncio
@_handle_project
async def test_unify_meet_to_email(test_redis_client, event_capture):
    """
    Test unify meet to email flow: user on a call requests email, verify email is sent.

    The Main CM Brain provides guidance to the Voice Agent but doesn't produce
    speech directly. The key assertion is that the email action is executed.
    """
    # Clear any events from initialization
    event_capture.clear()

    # Send incoming unify meet
    contact = contacts[1]
    pubsub = await send_incoming_call(
        test_redis_client,
        contact,
        "test_conference",
        "Tell me a joke via email right now",
        mode="unify_meet",
    )

    # Cleanup subscription (we don't expect voice streaming)
    await pubsub.unsubscribe("app:call:call_guidance")
    await pubsub.aclose()

    # Capture outgoing email and verify response
    await capture_outgoing_email(event_capture, contact)

    # End the unify meet
    await test_redis_client.publish(
        "app:comms:unify_meet_ended",
        UnifyMeetEnded(contact=contact).to_json(),
    )


@pytest.mark.asyncio
@_handle_project
async def test_unify_meet_to_unify_message(test_redis_client, event_capture):
    """
    Test unify meet to unify message flow: user on a call requests a message.

    The Main CM Brain provides guidance to the Voice Agent but doesn't produce
    speech directly. The key assertion is that the unify message action is executed.
    """
    # Clear any events from initialization
    event_capture.clear()

    # Send incoming unify meet
    contact = contacts[1]
    pubsub = await send_incoming_call(
        test_redis_client,
        contact,
        "test_conference",
        "Tell me a joke via unify message right now",
        mode="unify_meet",
    )

    # Cleanup subscription (we don't expect voice streaming)
    await pubsub.unsubscribe("app:call:call_guidance")
    await pubsub.aclose()

    # Capture outgoing unify message and verify response
    await capture_outgoing_unify_message(event_capture, contact)

    # End the unify meet
    await test_redis_client.publish(
        "app:comms:unify_meet_ended",
        UnifyMeetEnded(contact=contact).to_json(),
    )
