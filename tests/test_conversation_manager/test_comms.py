"""
tests/test_conversation_manager/test_comms.py
=============================================

Tests for communication flows (SMS, email, calls, etc.)
"""

import pytest
from tests.helpers import _handle_project
from tests.test_conversation_manager.helpers import (
    capture_outgoing_email,
    capture_outgoing_phone_call,
    capture_outgoing_sms,
    capture_outgoing_unify_message,
    capture_stream_response,
    send_incoming_email,
    send_incoming_call,
    send_incoming_sms,
    send_incoming_unify_message,
)
from unity.conversation_manager_2.new_events import PhoneCallEnded, UnifyCallEnded


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
    contact_number = "+15555551111"
    await send_incoming_sms(test_redis_client, contact_number, "Tell me a joke")

    # Capture outgoing SMS and verify response
    await capture_outgoing_sms(event_capture, contact_number)


@pytest.mark.asyncio
@_handle_project
async def test_sms_to_email(test_redis_client, event_capture):
    """
    Test SMS to email flow: send an incoming SMS and receive a response.
    """
    # Clear any events from initialization
    event_capture.clear()

    # Send incoming SMS
    contact_number = "+15555551111"
    email_address = "test@contact.com"
    await send_incoming_sms(
        test_redis_client,
        contact_number,
        "Tell me a joke via email",
    )

    # Capture outgoing email and verify response
    await capture_outgoing_email(event_capture, email_address)


@pytest.mark.asyncio
@_handle_project
async def test_sms_to_unify_message(test_redis_client, event_capture):
    """
    Test SMS to unify message flow: send an incoming SMS and receive a response.
    """
    # Clear any events from initialization
    event_capture.clear()

    # Send incoming SMS
    contact_number = "+15555551111"
    contact_id = 1
    await send_incoming_sms(
        test_redis_client,
        contact_number,
        "Tell me a joke via unify message",
    )

    # Capture outgoing unify message and verify response
    await capture_outgoing_unify_message(event_capture, contact_id)


@pytest.mark.asyncio
@_handle_project
async def test_sms_to_phone_call(test_redis_client, event_capture):
    """
    Test SMS to phone call flow: send an incoming SMS and receive a response.
    """
    # Clear any events from initialization
    event_capture.clear()

    # Send incoming SMS
    contact_number = "+15555551111"
    await send_incoming_sms(
        test_redis_client,
        contact_number,
        "Tell me a joke via phone call",
    )

    # Capture outgoing phone call and verify response
    await capture_outgoing_phone_call(event_capture, contact_number)

    # End the phone call
    await test_redis_client.publish(
        "app:comms:phone_call_ended",
        PhoneCallEnded(contact=contact_number).to_json(),
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
    contact_email = "test@contact.com"
    message_id = "test_message_id"
    await send_incoming_email(
        test_redis_client,
        contact_email,
        "Test Subject",
        "Tell me a joke",
        message_id,
    )

    # Capture outgoing email and verify response
    await capture_outgoing_email(event_capture, contact_email, message_id)


@pytest.mark.asyncio
@_handle_project
async def test_email_to_sms(test_redis_client, event_capture):
    """
    Test email to SMS flow: send an incoming email and receive a response.
    """
    # Clear any events from initialization
    event_capture.clear()

    # Send incoming email
    contact_number = "+15555551111"
    email_address = "test@contact.com"
    await send_incoming_email(
        test_redis_client,
        email_address,
        "Test Subject",
        "Tell me a joke via SMS",
        "test_message_id",
    )

    # Capture outgoing SMS and verify response
    await capture_outgoing_sms(event_capture, contact_number)


@pytest.mark.asyncio
@_handle_project
async def test_email_to_unify_message(test_redis_client, event_capture):
    """
    Test email to unify message flow: send an incoming email and receive a response.
    """
    # Clear any events from initialization
    event_capture.clear()

    # Send incoming email
    email_address = "test@contact.com"
    await send_incoming_email(
        test_redis_client,
        email_address,
        "Test Subject",
        "Tell me a joke via unify message",
        "test_message_id",
    )

    # Capture outgoing unify message and verify response
    await capture_outgoing_unify_message(event_capture, 1)


@pytest.mark.asyncio
@_handle_project
async def test_email_to_phone_call(test_redis_client, event_capture):
    """
    Test email to phone call flow: send an incoming email and receive a response.
    """
    # Clear any events from initialization
    event_capture.clear()

    # Send incoming email
    contact_number = "+15555551111"
    email_address = "test@contact.com"
    await send_incoming_email(
        test_redis_client,
        email_address,
        "Test Subject",
        "Tell me a joke via phone call",
        "test_message_id",
    )

    # Capture outgoing phone call and verify response
    await capture_outgoing_phone_call(event_capture, contact_number)

    # End the phone call
    await test_redis_client.publish(
        "app:comms:phone_call_ended",
        PhoneCallEnded(contact=contact_number).to_json(),
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
    contact_id = 1
    await send_incoming_unify_message(test_redis_client, contact_id, "Tell me a joke")

    # Capture outgoing unify message and verify response
    await capture_outgoing_unify_message(event_capture, contact_id)


@pytest.mark.asyncio
@_handle_project
async def test_unify_message_to_sms(test_redis_client, event_capture):
    """
    Test unify message to unify call flow: send an incoming unify message and receive
    a response.
    """
    # Clear any events from initialization
    event_capture.clear()

    # Send incoming unify message
    contact_id = 1
    contact_number = "+15555551111"
    await send_incoming_unify_message(
        test_redis_client,
        contact_id,
        "Tell me a joke via SMS",
    )

    # Capture outgoing SMS and verify response
    await capture_outgoing_sms(event_capture, contact_number)


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
    contact_id = 1
    email_address = "test@contact.com"
    await send_incoming_unify_message(
        test_redis_client,
        contact_id,
        "Tell me a joke via email",
    )

    # Capture outgoing email and verify response
    await capture_outgoing_email(event_capture, email_address)


@pytest.mark.asyncio
@_handle_project
async def test_unify_message_to_phone_call(test_redis_client, event_capture):
    """
    Test unify message to phone call flow: send an incoming unify message and receive a response.
    """
    # Clear any events from initialization
    event_capture.clear()

    # Send incoming unify message
    contact_id = 1
    contact_number = "+15555551111"
    await send_incoming_unify_message(
        test_redis_client,
        contact_id,
        "Tell me a joke via phone call",
    )

    # Capture outgoing phone call and verify response
    await capture_outgoing_phone_call(event_capture, contact_number)

    # End the phone call
    await test_redis_client.publish(
        "app:comms:phone_call_ended",
        PhoneCallEnded(contact=contact_number).to_json(),
    )


@pytest.mark.asyncio
@_handle_project
async def test_phone_call_flow(test_redis_client, event_capture):
    """Test phone call flow."""
    # Clear any events from initialization
    event_capture.clear()

    # Send incoming phone call
    contact_number = "+15555551111"
    pubsub = await send_incoming_call(
        test_redis_client,
        contact_number,
        "test_conference",
        "Tell me a joke",
    )

    # Capture the assistant's response to the user utterance
    print("📞 Waiting for assistant's response to user...")
    start2, chunks2, end2 = await capture_stream_response(pubsub, "Response to user")

    assert start2, "Should receive start_gen for response"
    assert len(chunks2) > 0, "Should receive chunks for response"
    assert end2, "Should receive end_gen for response"

    # Cleanup subscription
    await pubsub.unsubscribe("app:call:response_gen")
    await pubsub.aclose()

    # Verify exchange completed successfully
    print(f"\n✅ Phone call test complete!")
    print(f"   Exchange 2 (Response to user): {len(''.join(chunks2))} characters")

    # End the phone call
    await test_redis_client.publish(
        "app:comms:phone_call_ended",
        PhoneCallEnded(contact=contact_number).to_json(),
    )


@pytest.mark.asyncio
@_handle_project
async def test_phone_call_to_sms(test_redis_client, event_capture):
    """
    Test phone call to SMS flow: send an incoming phone call and receive a response.
    """
    # Clear any events from initialization
    event_capture.clear()

    # Send incoming phone call
    contact_number = "+15555551111"
    pubsub = await send_incoming_call(
        test_redis_client,
        contact_number,
        "test_conference",
        "Tell me a joke via SMS right now",
    )

    # Capture the assistant's response to the user utterance
    print("📞 Waiting for assistant's response to user...")
    start2, chunks2, end2 = await capture_stream_response(pubsub, "Response to user")

    assert start2, "Should receive start_gen for response"
    assert len(chunks2) > 0, "Should receive chunks for response"
    assert end2, "Should receive end_gen for response"

    # Cleanup subscription
    await pubsub.unsubscribe("app:call:response_gen")
    await pubsub.aclose()

    # Capture outgoing SMS and verify response
    await capture_outgoing_sms(event_capture, contact_number)

    # End the phone call
    await test_redis_client.publish(
        "app:comms:phone_call_ended",
        PhoneCallEnded(contact=contact_number).to_json(),
    )


@pytest.mark.asyncio
@_handle_project
async def test_phone_call_to_email(test_redis_client, event_capture):
    """
    Test phone call to email flow: send an incoming phone call and receive a response.
    """
    # Clear any events from initialization
    event_capture.clear()

    # Send incoming phone call
    contact_number = "+15555551111"
    email_address = "test@contact.com"
    pubsub = await send_incoming_call(
        test_redis_client,
        contact_number,
        "test_conference",
        "Tell me a joke via email right now",
    )

    # Capture the assistant's response to the user utterance
    print("📞 Waiting for assistant's response to user...")
    start2, chunks2, end2 = await capture_stream_response(pubsub, "Response to user")
    assert start2, "Should receive start_gen for response"
    assert len(chunks2) > 0, "Should receive chunks for response"
    assert end2, "Should receive end_gen for response"

    # Cleanup subscription
    await pubsub.unsubscribe("app:call:response_gen")
    await pubsub.aclose()

    # Capture outgoing email and verify response
    await capture_outgoing_email(event_capture, email_address)

    # End the phone call
    await test_redis_client.publish(
        "app:comms:phone_call_ended",
        PhoneCallEnded(contact=contact_number).to_json(),
    )


@pytest.mark.asyncio
@_handle_project
async def test_phone_call_to_unify_message(test_redis_client, event_capture):
    """
    Test phone call to unify message flow: send an incoming phone call and receive a response.
    """
    # Clear any events from initialization
    event_capture.clear()

    # Send incoming phone call
    contact_number = "+15555551111"
    contact_id = 1
    pubsub = await send_incoming_call(
        test_redis_client,
        contact_number,
        "test_conference",
        "Tell me a joke via unify message right now",
    )

    # Capture the assistant's response to the user utterance
    print("📞 Waiting for assistant's response to user...")
    start2, chunks2, end2 = await capture_stream_response(pubsub, "Response to user")
    assert start2, "Should receive start_gen for response"
    assert len(chunks2) > 0, "Should receive chunks for response"
    assert end2, "Should receive end_gen for response"

    # Cleanup subscription
    await pubsub.unsubscribe("app:call:response_gen")
    await pubsub.aclose()

    # Capture outgoing unify message and verify response
    await capture_outgoing_unify_message(event_capture, contact_id)

    # End the phone call
    await test_redis_client.publish(
        "app:comms:phone_call_ended",
        PhoneCallEnded(contact=contact_number).to_json(),
    )


@pytest.mark.asyncio
@_handle_project
async def test_unify_call_flow(test_redis_client, event_capture):
    """Test unify call flow."""
    # Clear any events from initialization
    event_capture.clear()

    # Send incoming unify call
    contact_id = 1
    pubsub = await send_incoming_call(
        test_redis_client,
        contact_id,
        "test_conference",
        "Tell me a joke",
        mode="unify_call",
    )

    # Capture the assistant's response to the user utterance
    print("📞 Waiting for assistant's response to user...")
    start2, chunks2, end2 = await capture_stream_response(pubsub, "Response to user")
    assert start2, "Should receive start_gen for response"
    assert len(chunks2) > 0, "Should receive chunks for response"
    assert end2, "Should receive end_gen for response"

    # Cleanup subscription
    await pubsub.unsubscribe("app:unify_call:response_gen")
    await pubsub.aclose()

    # Verify exchange completed successfully
    print(f"\n✅ Unify call test complete!")
    print(f"   Exchange 2 (Response to user): {len(''.join(chunks2))} characters")

    # End the unify call
    await test_redis_client.publish(
        "app:comms:unify_call_ended",
        UnifyCallEnded(contact=contact_id).to_json(),
    )


@pytest.mark.asyncio
@_handle_project
async def test_unify_call_to_sms(test_redis_client, event_capture):
    """
    Test unify call to SMS flow: send an incoming unify call and receive a response.
    """
    # Clear any events from initialization
    event_capture.clear()

    # Send incoming unify call
    contact_id = 1
    contact_number = "+15555551111"
    pubsub = await send_incoming_call(
        test_redis_client,
        contact_id,
        "test_conference",
        "Tell me a joke via sms right now",
        mode="unify_call",
    )

    # Capture the assistant's response to the user utterance
    print("📞 Waiting for assistant's response to user...")
    start2, chunks2, end2 = await capture_stream_response(pubsub, "Response to user")
    assert start2, "Should receive start_gen for response"
    assert len(chunks2) > 0, "Should receive chunks for response"
    assert end2, "Should receive end_gen for response"

    # Cleanup subscription
    await pubsub.unsubscribe("app:unify_call:response_gen")
    await pubsub.aclose()

    # Capture outgoing SMS and verify response
    await capture_outgoing_sms(event_capture, contact_number)

    # End the unify call
    await test_redis_client.publish(
        "app:comms:unify_call_ended",
        UnifyCallEnded(contact=contact_id).to_json(),
    )


@pytest.mark.asyncio
@_handle_project
async def test_unify_call_to_email(test_redis_client, event_capture):
    """
    Test unify call to email flow: send an incoming unify call and receive a response.
    """
    # Clear any events from initialization
    event_capture.clear()

    # Send incoming unify call
    contact_id = 1
    email_address = "test@contact.com"
    pubsub = await send_incoming_call(
        test_redis_client,
        contact_id,
        "test_conference",
        "Tell me a joke via email right now",
        mode="unify_call",
    )

    # Capture the assistant's response to the user utterance
    print("📞 Waiting for assistant's response to user...")
    start2, chunks2, end2 = await capture_stream_response(pubsub, "Response to user")
    assert start2, "Should receive start_gen for response"
    assert len(chunks2) > 0, "Should receive chunks for response"
    assert end2, "Should receive end_gen for response"

    # Cleanup subscription
    await pubsub.unsubscribe("app:unify_call:response_gen")
    await pubsub.aclose()

    # Capture outgoing email and verify response
    await capture_outgoing_email(event_capture, email_address)

    # End the unify call
    await test_redis_client.publish(
        "app:comms:unify_call_ended",
        UnifyCallEnded(contact=contact_id).to_json(),
    )


@pytest.mark.asyncio
@_handle_project
async def test_unify_call_to_unify_message(test_redis_client, event_capture):
    """
    Test unify call to unify message flow: send an incoming unify call and receive a response.
    """
    # Clear any events from initialization
    event_capture.clear()

    # Send incoming unify call
    contact_id = 1
    pubsub = await send_incoming_call(
        test_redis_client,
        contact_id,
        "test_conference",
        "Tell me a joke via unify message right now",
        mode="unify_call",
    )

    # Capture the assistant's response to the user utterance
    print("📞 Waiting for assistant's response to user...")
    start2, chunks2, end2 = await capture_stream_response(pubsub, "Response to user")
    assert start2, "Should receive start_gen for response"
    assert len(chunks2) > 0, "Should receive chunks for response"
    assert end2, "Should receive end_gen for response"

    # Cleanup subscription
    await pubsub.unsubscribe("app:unify_call:response_gen")
    await pubsub.aclose()

    # Capture outgoing unify message and verify response
    await capture_outgoing_unify_message(event_capture, contact_id)

    # End the unify call
    await test_redis_client.publish(
        "app:comms:unify_call_ended",
        UnifyCallEnded(contact=contact_id).to_json(),
    )
