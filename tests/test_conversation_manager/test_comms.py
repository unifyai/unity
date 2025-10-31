"""
tests/test_conversation_manager/test_comms.py
=============================================

Tests for communication flows (SMS, email, calls, etc.)
"""

import pytest

from unity.conversation_manager_2.new_events import (
    EmailRecieved,
    EmailSent,
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
    incoming_sms = SMSRecieved(
        contact=1,
        content="Tell me a joke via unify message",
    )

    print(f"\n📱 Sending SMS from 1")
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

    contact_number = "+15555551111"
    email_address = "test@contact.com"

    # Send incoming email
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
async def test_unify_message_to_sms(test_redis_client, event_capture):
    """
    Test unify message to SMS flow: send an incoming unify message and receive a response.
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
    print("⏳ Waiting for SMS response (timeout: 60s)...")
    response = await event_capture.wait_for_event(
        SMSSent,
        timeout=60.0,
        contact=1,
    )

    # Verify response
    assert isinstance(response, SMSSent)
    assert response.contact == 1
    assert len(response.content) > 0

    print(f"✅ Got SMS response: {response.content[:100]}...")
    print(f"   Full response length: {len(response.content)} characters")


@pytest.mark.asyncio
@_handle_project
async def test_unify_message_to_unify_message(test_redis_client, event_capture):
    """
    Test unify message to unify message flow: send an incoming unify message and receive a response.
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
    Test unify message to unify call flow: send an incoming unify message and receive a response.
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
    print("⏳ Waiting for SMS response (timeout: 60s)...")
    response = await event_capture.wait_for_event(
        SMSSent,
        timeout=60.0,
        contact=1,
    )

    # Verify response
    assert isinstance(response, SMSSent)
    assert response.contact == 1
    assert len(response.content) > 0

    print(f"✅ Got unify call started response: {response.content[:100]}...")
    print(f"   Full response length: {len(response.content)} characters")


@pytest.mark.asyncio
@_handle_project
async def test_unify_message_to_email(test_redis_client, event_capture):
    """
    Test unify message to email flow: send an incoming unify message and receive a response.
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
    print("⏳ Waiting for email response (timeout: 60s)...")
    response = await event_capture.wait_for_event(
        EmailSent,
        timeout=60.0,
        contact="test@contact.com",
    )

    # Verify response
    assert isinstance(response, EmailSent)
    assert response.contact == "test@contact.com"
    assert len(response.body) > 0

    print(f"✅ Got email response: {response.body[:100]}...")
    print(f"   Full response length: {len(response.body)} characters")
