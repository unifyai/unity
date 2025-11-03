"""
tests/test_conversation_manager/test_comms.py
=============================================

Tests for communication flows (SMS, email, calls, etc.)
"""

import asyncio
import pytest

from tests.test_conversation_manager.helpers import (
    capture_stream_response,
    send_incoming_email,
    send_incoming_phone_call,
    send_incoming_sms,
    send_incoming_unify_message,
)
from unity.conversation_manager_2.new_events import (
    EmailRecieved,
    EmailSent,
    PhoneCallEnded,
    PhoneCallRecieved,
    PhoneCallSent,
    PhoneCallStarted,
    PhoneUtterance,
    SMSSent,
    UnifyMessageRecieved,
    UnifyMessageSent,
)
from tests.helpers import _handle_project


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

    # Wait for the assistant's response
    print("⏳ Waiting for SMS response (timeout: 60s)...")
    response = await event_capture.wait_for_event(
        SMSSent,
        timeout=60.0,
        contact=contact_number,
    )

    # Verify response
    assert isinstance(response, SMSSent)
    assert response.contact == contact_number
    assert len(response.content) > 0

    print(f"✅ Got SMS response: {response.content[:100]}...")
    print(f"   Full response length: {len(response.content)} characters")


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
        test_redis_client, contact_number, "Tell me a joke via email"
    )

    # Wait for the assistant's response
    print("⏳ Waiting for email response (timeout: 60s)...")
    response = await event_capture.wait_for_event(
        EmailSent,
        timeout=60.0,
        contact=email_address,
    )

    # Verify response
    assert isinstance(response, EmailSent)
    assert response.contact == email_address
    assert len(response.body) > 0

    print(f"✅ Got email response: {response.body[:100]}...")
    print(f"   Full response length: {len(response.body)} characters")


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
        test_redis_client, contact_number, "Tell me a joke via unify message"
    )

    # Wait for the assistant's response
    print("⏳ Waiting for unify message response (timeout: 60s)...")
    response = await event_capture.wait_for_event(
        UnifyMessageSent,
        timeout=60.0,
        contact=contact_id,
    )

    # Verify response
    assert isinstance(response, UnifyMessageSent)
    assert response.contact == contact_id
    assert len(response.content) > 0

    print(f"✅ Got unify message response: {response.content[:100]}...")
    print(f"   Full response length: {len(response.content)} characters")


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
        test_redis_client, contact_number, "Tell me a joke via phone call"
    )

    # Wait for the assistant's response
    print("⏳ Waiting for phone call response (timeout: 60s)...")
    response = await event_capture.wait_for_event(
        PhoneCallSent,
        timeout=60.0,
        contact=contact_number,
    )

    # Verify response
    assert isinstance(response, PhoneCallSent)
    assert response.contact == contact_number

    # End the phone call
    end_phone_call = PhoneCallEnded(
        contact=contact_number,
    )
    await test_redis_client.publish(
        "app:comms:phone_call_ended", end_phone_call.to_json()
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

    # Wait for the assistant's response
    print("⏳ Waiting for email response (timeout: 60s)...")
    response = await event_capture.wait_for_event(
        EmailSent,
        timeout=60.0,
        contact=contact_email,
    )

    # Verify response
    assert isinstance(response, EmailSent)
    assert response.message_id == message_id
    assert response.contact == contact_email
    assert len(response.body) > 0

    print(f"✅ Got email response: {response.body[:100]}...")
    print(f"   Full response length: {len(response.body)} characters")


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

    # Wait for the assistant's response
    print("⏳ Waiting for SMS response (timeout: 60s)...")
    response = await event_capture.wait_for_event(
        SMSSent,
        timeout=60.0,
        contact=contact_number,
    )

    # Verify response
    assert isinstance(response, SMSSent)
    assert response.contact == contact_number
    assert len(response.content) > 0

    print(f"✅ Got SMS response: {response.content[:100]}...")
    print(f"   Full response length: {len(response.content)} characters")


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

    # Wait for the assistant's response
    print("⏳ Waiting for unify message response (timeout: 60s)...")
    response = await event_capture.wait_for_event(
        UnifyMessageSent,
        timeout=60.0,
        contact=1,
    )

    # Verify response
    assert isinstance(response, UnifyMessageSent)
    assert response.contact == 1
    assert len(response.content) > 0

    print(f"✅ Got unify message response: {response.content[:100]}...")
    print(f"   Full response length: {len(response.content)} characters")


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

    # Wait for the assistant's response
    print("⏳ Waiting for phone call response (timeout: 60s)...")
    response = await event_capture.wait_for_event(
        PhoneCallSent,
        timeout=60.0,
        contact=contact_number,
    )

    # Verify response
    assert isinstance(response, PhoneCallSent)
    assert response.contact == contact_number

    # End the phone call
    end_phone_call = PhoneCallEnded(
        contact=contact_number,
    )
    await test_redis_client.publish(
        "app:comms:phone_call_ended", end_phone_call.to_json()
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

    # Wait for the assistant's response
    print("⏳ Waiting for unify message response (timeout: 60s)...")
    response = await event_capture.wait_for_event(
        UnifyMessageSent,
        timeout=60.0,
        contact=contact_id,
    )

    # Verify response
    assert isinstance(response, UnifyMessageSent)
    assert response.contact == contact_id
    assert len(response.content) > 0

    print(f"✅ Got unify message response: {response.content[:100]}...")
    print(f"   Full response length: {len(response.content)} characters")


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
        test_redis_client, contact_id, "Tell me a joke via SMS"
    )

    # Wait for the assistant's response
    print("⏳ Waiting for SMS response (timeout: 60s)...")
    response = await event_capture.wait_for_event(
        SMSSent,
        timeout=60.0,
        contact=contact_number,
    )

    # Verify response
    assert isinstance(response, SMSSent)
    assert response.contact == contact_number
    assert len(response.content) > 0

    print(f"✅ Got unify call started response: {response.content[:100]}...")
    print(f"   Full response length: {len(response.content)} characters")


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
        test_redis_client, contact_id, "Tell me a joke via email"
    )

    # Wait for the assistant's response
    print("⏳ Waiting for email response (timeout: 60s)...")
    response = await event_capture.wait_for_event(
        EmailSent,
        timeout=60.0,
        contact=email_address,
    )

    # Verify response
    assert isinstance(response, EmailSent)
    assert response.contact == email_address
    assert len(response.body) > 0

    print(f"✅ Got email response: {response.body[:100]}...")
    print(f"   Full response length: {len(response.body)} characters")


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
        test_redis_client, contact_id, "Tell me a joke via phone call"
    )

    # Wait for the assistant's response
    print("⏳ Waiting for phone call response (timeout: 60s)...")
    response = await event_capture.wait_for_event(
        PhoneCallSent,
        timeout=60.0,
        contact=contact_number,
    )

    # Verify response
    assert isinstance(response, PhoneCallSent)
    assert response.contact == contact_number

    # End the phone call
    end_phone_call = PhoneCallEnded(
        contact=contact_number,
    )
    await test_redis_client.publish(
        "app:comms:phone_call_ended", end_phone_call.to_json()
    )


@pytest.mark.asyncio
@_handle_project
async def test_phone_call_flow(test_redis_client, event_capture):
    """Test phone call flow."""
    # Clear any events from initialization
    event_capture.clear()

    # Send incoming phone call
    contact_number = "+15555551111"
    pubsub = await send_incoming_phone_call(
        test_redis_client, contact_number, "test_conference", "Tell me a joke"
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
    end_phone_call = PhoneCallEnded(
        contact=contact_number,
    )
    await test_redis_client.publish(
        "app:comms:phone_call_ended", end_phone_call.to_json()
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
    pubsub = await send_incoming_phone_call(
        test_redis_client, contact_number, "test_conference", "Tell me a joke via SMS"
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

    # Wait for the assistant's response
    print("⏳ Waiting for SMS response (timeout: 60s)...")
    response = await event_capture.wait_for_event(
        SMSSent,
        timeout=60.0,
        contact=contact_number,
    )

    # Verify response
    assert isinstance(response, SMSSent)
    assert response.contact == contact_number
    assert len(response.content) > 0

    # Verify exchange completed successfully
    print(f"✅ Got SMS response: {response.content[:100]}...")
    print(f"   Full response length: {len(response.content)} characters")

    # End the phone call
    end_phone_call = PhoneCallEnded(
        contact=contact_number,
    )
    await test_redis_client.publish(
        "app:comms:phone_call_ended", end_phone_call.to_json()
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
    pubsub = await send_incoming_phone_call(
        test_redis_client, contact_number, "test_conference", "Tell me a joke via email"
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

    # Wait for the assistant's response
    print("⏳ Waiting for email response (timeout: 60s)...")
    response = await event_capture.wait_for_event(
        EmailSent,
        timeout=60.0,
        contact=email_address,
    )

    # Verify response
    assert isinstance(response, EmailSent)
    assert response.contact == email_address
    assert len(response.body) > 0

    # Verify exchange completed successfully
    print(f"✅ Got email response: {response.body[:100]}...")
    print(f"   Full response length: {len(response.body)} characters")

    # End the phone call
    end_phone_call = PhoneCallEnded(
        contact=contact_number,
    )
    await test_redis_client.publish(
        "app:comms:phone_call_ended", end_phone_call.to_json()
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
    pubsub = await send_incoming_phone_call(
        test_redis_client,
        contact_number,
        "test_conference",
        "Tell me a joke via unify message",
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

    # Wait for the assistant's response
    print("⏳ Waiting for unify message response (timeout: 60s)...")
    response = await event_capture.wait_for_event(
        UnifyMessageSent,
        timeout=60.0,
        contact=contact_id,
    )

    # Verify response
    assert isinstance(response, UnifyMessageSent)
    assert response.contact == contact_id
    assert len(response.content) > 0

    # Verify exchange completed successfully
    print(f"✅ Got unify message response: {response.content[:100]}...")
    print(f"   Full response length: {len(response.content)} characters")

    # End the phone call
    end_phone_call = PhoneCallEnded(
        contact=contact_number,
    )
    await test_redis_client.publish(
        "app:comms:phone_call_ended", end_phone_call.to_json()
    )
