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
)
from tests.helpers import _handle_project


@pytest.mark.asyncio
@_handle_project
async def test_sms_to_sms(initialized_system):
    """
    Test basic SMS flow: send an incoming SMS and receive a response.

    Flow:
    1. Send SMSRecieved event with a question
    2. CM processes it with LLM
    3. CM publishes SMSSent event with response
    4. We capture and verify the response
    """
    redis_client = initialized_system["redis_client"]
    event_capture = initialized_system["event_capture"]

    # Clear any events from initialization
    event_capture.clear()

    # Send incoming SMS
    contact_number = "+15555551111"
    incoming_sms = SMSRecieved(
        contact=contact_number,
        content="What is the capital of France?",
    )

    print(f"\n📱 Sending SMS from {contact_number}")
    await redis_client.publish("app:comms:sms_received", incoming_sms.to_json())

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
async def test_email_to_email(initialized_system):
    """
    Test basic email flow: send an incoming email and receive a response.

    Flow:
    1. Send EmailRecieved event with a question
    2. CM processes it with LLM
    3. CM publishes EmailSent event with response
    4. We capture and verify the response
    """
    redis_client = initialized_system["redis_client"]
    event_capture = initialized_system["event_capture"]

    # Clear any events from initialization
    event_capture.clear()

    # Send incoming email
    contact_email = "test@contact.com"
    incoming_email = EmailRecieved(
        contact=contact_email,
        body="What is the capital of France?",
        subject="Test Subject",
        message_id="test_message_id",
    )

    print(f"\n📧 Sending email from {contact_email}")
    await redis_client.publish("app:comms:email_received", incoming_email.to_json())

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
