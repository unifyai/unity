"""
tests/test_conversation_manager/test_comms.py
=============================================

Tests for communication flows (SMS, email, calls, etc.)
"""

import asyncio
import pytest

from tests.test_conversation_manager.helpers import capture_stream_response
from unity.conversation_manager_2.new_events import (
    EmailRecieved,
    EmailSent,
    PhoneCallEnded,
    PhoneCallRecieved,
    PhoneCallSent,
    PhoneCallStarted,
    PhoneUtterance,
    SMSRecieved,
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
    incoming_sms = SMSRecieved(
        contact=contact_number,
        content="Tell me a joke",
    )

    print(f"\n📱 Sending SMS from {contact_number}")
    await test_redis_client.publish("app:comms:sms_received", incoming_sms.to_json())

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

    contact_number = "+15555551111"
    email_address = "test@contact.com"

    # Send incoming SMS
    incoming_sms = SMSRecieved(
        contact=contact_number,
        content="Tell me a joke via email",
    )

    print(f"\n📱 Sending SMS from {contact_number}")
    await test_redis_client.publish("app:comms:sms_received", incoming_sms.to_json())

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
    incoming_sms = SMSRecieved(
        contact=contact_number,
        content="Tell me a joke via unify message",
    )

    print(f"\n📱 Sending SMS from {contact_number}")
    await test_redis_client.publish("app:comms:sms_received", incoming_sms.to_json())

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
async def test_sms_to_phone_call(test_redis_client, event_capture):
    """
    Test SMS to phone call flow: send an incoming SMS and receive a response.
    """
    # Clear any events from initialization
    event_capture.clear()

    contact_number = "+15555551111"
    incoming_sms = SMSRecieved(
        contact=contact_number,
        content="Tell me a joke via phone call",
    )

    print(f"\n📱 Sending SMS from {contact_number}")
    await test_redis_client.publish("app:comms:sms_received", incoming_sms.to_json())

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
    incoming_email = EmailRecieved(
        contact=contact_email,
        body="Tell me a joke",
        subject="Test Subject",
        message_id="test_message_id",
    )

    print(f"\n📧 Sending email from {contact_email}")
    await test_redis_client.publish(
        "app:comms:email_received", incoming_email.to_json()
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
    assert response.message_id == incoming_email.message_id
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

    contact_number = "+15555551111"
    email_address = "test@contact.com"

    # Send incoming email
    incoming_email = EmailRecieved(
        contact=email_address,
        body="Tell me a joke via SMS",
        subject="Test Subject",
        message_id="test_message_id",
    )

    print(f"\n📧 Sending email from {email_address}")
    await test_redis_client.publish(
        "app:comms:email_received", incoming_email.to_json()
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
    incoming_email = EmailRecieved(
        contact=email_address,
        body="Tell me a joke via unify message",
        subject="Test Subject",
        message_id="test_message_id",
    )

    print(f"\n📧 Sending email from {email_address}")
    await test_redis_client.publish(
        "app:comms:email_received", incoming_email.to_json()
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

    # Send incoming email
    incoming_email = EmailRecieved(
        contact=email_address,
        body="Tell me a joke via phone call",
        subject="Test Subject",
        message_id="test_message_id",
    )

    print(f"\n📧 Sending email from {email_address}")
    await test_redis_client.publish(
        "app:comms:email_received", incoming_email.to_json()
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
    incoming_unify_message = UnifyMessageRecieved(
        contact=1,
        content="Tell me a joke",
    )

    print(f"\n📧 Sending unify message from 1")
    await test_redis_client.publish(
        "app:comms:unify_message_received", incoming_unify_message.to_json()
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
async def test_unify_message_to_sms(test_redis_client, event_capture):
    """
    Test unify message to unify call flow: send an incoming unify message and receive
    a response.
    """
    # Clear any events from initialization
    event_capture.clear()

    # Send incoming unify message
    incoming_unify_message = UnifyMessageRecieved(
        contact=1,
        content="Tell me a joke via SMS",
    )

    print(f"\n📧 Sending unify message from 1")
    await test_redis_client.publish(
        "app:comms:unify_message_received", incoming_unify_message.to_json()
    )

    # Wait for the assistant's response
    contact_number = "+15555551111"
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
    incoming_unify_message = UnifyMessageRecieved(
        contact=1,
        content="Tell me a joke via email",
    )

    print(f"\n📧 Sending unify message from 1")
    await test_redis_client.publish(
        "app:comms:unify_message_received", incoming_unify_message.to_json()
    )

    # Wait for the assistant's response
    email_address = "test@contact.com"
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

    contact_number = "+15555551111"

    # Send incoming unify message
    incoming_unify_message = UnifyMessageRecieved(
        contact=1,
        content="Tell me a joke via phone call",
    )

    print(f"\n📧 Sending unify message from 1")
    await test_redis_client.publish(
        "app:comms:unify_message_received", incoming_unify_message.to_json()
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

    contact_number = "+15555551111"

    # Step 1: Send PhoneCallReceived
    # (CM will try to spawn call script, but we ignore that)
    incoming_call = PhoneCallRecieved(
        contact=contact_number,
        conference_name="test_conference",
    )

    print(f"\n📞 Step 1: Sending PhoneCallReceived from {contact_number}")
    await test_redis_client.publish("app:comms:call_recieved", incoming_call.to_json())

    # Give CM a moment to process and attempt to spawn script
    await asyncio.sleep(0.5)

    # Step 2: Subscribe to the response streaming channel FIRST
    # We must subscribe BEFORE PhoneCallStarted to catch the initial greeting
    print("📞 Step 2: Subscribing to app:call:response_gen channel")
    pubsub = test_redis_client.pubsub()
    await pubsub.subscribe("app:call:response_gen")

    # Wait a moment for subscription to be ready
    await asyncio.sleep(0.2)

    # Step 3: Act as the call script - publish PhoneCallStarted
    # This triggers the initial greeting to be streamed
    call_started = PhoneCallStarted(contact=contact_number)

    print("📞 Step 3: Acting as call script - sending PhoneCallStarted")
    await test_redis_client.publish(
        "app:comms:phone_call_started", call_started.to_json()
    )

    # Step 4: Capture the initial greeting (triggered by PhoneCallStarted above)
    print("📞 Step 4: Waiting for assistant's initial greeting...")
    start1, chunks1, end1 = await capture_stream_response(pubsub, "Initial greeting")

    assert start1, "Should receive start_gen for initial greeting"
    assert len(chunks1) > 0, "Should receive chunks for initial greeting"
    assert end1, "Should receive end_gen for initial greeting"

    # Step 5: Send a user utterance
    user_utterance = PhoneUtterance(contact=contact_number, content="Tell me a joke")

    print("\n📞 Step 5: Sending user utterance (PhoneUtterance)")
    await test_redis_client.publish(
        "app:comms:phone_utterance", user_utterance.to_json()
    )

    # Step 6: Capture the assistant's response to the user utterance
    print("📞 Step 6: Waiting for assistant's response to user...")
    start2, chunks2, end2 = await capture_stream_response(pubsub, "Response to user")

    assert start2, "Should receive start_gen for response"
    assert len(chunks2) > 0, "Should receive chunks for response"
    assert end2, "Should receive end_gen for response"

    # Cleanup subscription
    await pubsub.unsubscribe("app:call:response_gen")
    await pubsub.aclose()

    # Step 7: Verify both exchanges completed successfully
    print(f"\n✅ Phone call test complete!")
    print(f"   Exchange 1 (Initial greeting): {len(''.join(chunks1))} characters")
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

    contact_number = "+15555551111"

    # Send incoming phone call
    incoming_phone_call = PhoneCallRecieved(
        contact=contact_number,
        conference_name="test_conference",
    )

    print(f"\n📞 Sending phone call from {contact_number}")
    await test_redis_client.publish(
        "app:comms:call_recieved", incoming_phone_call.to_json()
    )

    # Give CM a moment to process and attempt to spawn script
    await asyncio.sleep(0.5)

    # Step 2: Subscribe to the response streaming channel FIRST
    # We must subscribe BEFORE PhoneCallStarted to catch the initial greeting
    print("📞 Step 2: Subscribing to app:call:response_gen channel")
    pubsub = test_redis_client.pubsub()
    await pubsub.subscribe("app:call:response_gen")

    # Wait a moment for subscription to be ready
    await asyncio.sleep(0.2)

    # Step 3: Act as the call script - publish PhoneCallStarted
    # This triggers the initial greeting to be streamed
    call_started = PhoneCallStarted(contact=contact_number)

    print("📞 Step 3: Acting as call script - sending PhoneCallStarted")
    await test_redis_client.publish(
        "app:comms:phone_call_started", call_started.to_json()
    )

    # Step 4: Capture the initial greeting (triggered by PhoneCallStarted above)
    print("📞 Step 4: Waiting for assistant's initial greeting...")
    start1, chunks1, end1 = await capture_stream_response(pubsub, "Initial greeting")

    assert start1, "Should receive start_gen for initial greeting"
    assert len(chunks1) > 0, "Should receive chunks for initial greeting"
    assert end1, "Should receive end_gen for initial greeting"

    # Step 5: Send a user utterance
    user_utterance = PhoneUtterance(
        contact=contact_number, content="Tell me a joke via SMS"
    )

    print("\n📞 Step 5: Sending user utterance (PhoneUtterance)")
    await test_redis_client.publish(
        "app:comms:phone_utterance", user_utterance.to_json()
    )

    # Step 6: Capture the assistant's response to the user utterance
    print("📞 Step 6: Waiting for assistant's response to user...")
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

    contact_number = "+15555551111"
    email_address = "test@contact.com"

    # Send incoming phone call
    incoming_phone_call = PhoneCallRecieved(
        contact=contact_number,
        conference_name="test_conference",
    )

    print(f"\n📞 Sending phone call from {contact_number}")
    await test_redis_client.publish(
        "app:comms:call_recieved", incoming_phone_call.to_json()
    )

    # Give CM a moment to process and attempt to spawn script
    await asyncio.sleep(0.5)

    # Step 2: Subscribe to the response streaming channel FIRST
    # We must subscribe BEFORE PhoneCallStarted to catch the initial greeting
    print("📞 Step 2: Subscribing to app:call:response_gen channel")
    pubsub = test_redis_client.pubsub()
    await pubsub.subscribe("app:call:response_gen")

    # Wait a moment for subscription to be ready
    await asyncio.sleep(0.2)

    # Step 3: Act as the call script - publish PhoneCallStarted
    # This triggers the initial greeting to be streamed
    call_started = PhoneCallStarted(contact=contact_number)

    print("📞 Step 3: Acting as call script - sending PhoneCallStarted")
    await test_redis_client.publish(
        "app:comms:phone_call_started", call_started.to_json()
    )

    # Step 4: Capture the initial greeting (triggered by PhoneCallStarted above)
    print("📞 Step 4: Waiting for assistant's initial greeting...")
    start1, chunks1, end1 = await capture_stream_response(pubsub, "Initial greeting")
    assert start1, "Should receive start_gen for initial greeting"
    assert len(chunks1) > 0, "Should receive chunks for initial greeting"
    assert end1, "Should receive end_gen for initial greeting"

    # Step 5: Send a user utterance
    user_utterance = PhoneUtterance(
        contact=contact_number, content="Tell me a joke via email"
    )

    print("\n📞 Step 5: Sending user utterance (PhoneUtterance)")
    await test_redis_client.publish(
        "app:comms:phone_utterance", user_utterance.to_json()
    )

    # Step 6: Capture the assistant's response to the user utterance
    print("📞 Step 6: Waiting for assistant's response to user...")
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

    contact_number = "+15555551111"
    unify_message_id = 1

    # Send incoming phone call
    incoming_phone_call = PhoneCallRecieved(
        contact=contact_number,
        conference_name="test_conference",
    )

    print(f"\n📞 Sending phone call from {contact_number}")
    await test_redis_client.publish(
        "app:comms:call_recieved", incoming_phone_call.to_json()
    )

    # Give CM a moment to process and attempt to spawn script
    await asyncio.sleep(0.5)

    # Step 2: Subscribe to the response streaming channel FIRST
    # We must subscribe BEFORE PhoneCallStarted to catch the initial greeting
    print("📞 Step 2: Subscribing to app:call:response_gen channel")
    pubsub = test_redis_client.pubsub()
    await pubsub.subscribe("app:call:response_gen")

    # Wait a moment for subscription to be ready
    await asyncio.sleep(0.2)

    # Step 3: Act as the call script - publish PhoneCallStarted
    # This triggers the initial greeting to be streamed
    call_started = PhoneCallStarted(contact=contact_number)

    print("📞 Step 3: Acting as call script - sending PhoneCallStarted")
    await test_redis_client.publish(
        "app:comms:phone_call_started", call_started.to_json()
    )

    # Step 4: Capture the initial greeting (triggered by PhoneCallStarted above)
    print("📞 Step 4: Waiting for assistant's initial greeting...")
    start1, chunks1, end1 = await capture_stream_response(pubsub, "Initial greeting")
    assert start1, "Should receive start_gen for initial greeting"
    assert len(chunks1) > 0, "Should receive chunks for initial greeting"
    assert end1, "Should receive end_gen for initial greeting"

    # Step 5: Send a user utterance
    user_utterance = PhoneUtterance(
        contact=contact_number, content="Tell me a joke via unify message"
    )

    print("\n📞 Step 5: Sending user utterance (PhoneUtterance)")
    await test_redis_client.publish(
        "app:comms:phone_utterance", user_utterance.to_json()
    )

    # Step 6: Capture the assistant's response to the user utterance
    print("📞 Step 6: Waiting for assistant's response to user...")
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
        contact=unify_message_id,
    )

    # Verify response
    assert isinstance(response, UnifyMessageSent)
    assert response.contact == unify_message_id
    assert len(response.content) > 0

    # End the phone call
    end_phone_call = PhoneCallEnded(
        contact=contact_number,
    )
    await test_redis_client.publish(
        "app:comms:phone_call_ended", end_phone_call.to_json()
    )
