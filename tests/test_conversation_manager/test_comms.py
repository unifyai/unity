"""
tests/test_conversation_manager/test_comms.py
=============================================

Tests for communication flows (SMS, email, calls, etc.)

Uses **direct handler testing** pattern:
- Call EventHandler.handle_event() directly instead of publishing events
- Check CM state directly instead of waiting for published events
- Same pattern as ContactManager tests

Voice call tests verify that events are handled correctly. In the voice
architecture, the Main CM Brain only provides guidance to the Voice Agent
(fast brain) - it doesn't produce speech directly.
"""

import pytest

from tests.helpers import _handle_project
from tests.test_conversation_manager.conftest import TEST_CONTACTS
from unity.conversation_manager.events import (
    SMSReceived,
    EmailReceived,
    UnifyMessageReceived,
)
from unity.conversation_manager.domains.event_handlers import EventHandler

pytestmark = pytest.mark.eval


def get_sms_thread(cm, contact_id: int):
    """Get SMS thread for a contact from CM state."""
    active = cm.contact_index.active_conversations.get(contact_id)
    if not active:
        return []
    return list(active.threads.get("sms", []))


def get_email_thread(cm, contact_id: int):
    """Get email thread for a contact from CM state."""
    active = cm.contact_index.active_conversations.get(contact_id)
    if not active:
        return []
    return list(active.threads.get("email", []))


def get_unify_message_thread(cm, contact_id: int):
    """Get unify_message thread for a contact from CM state."""
    active = cm.contact_index.active_conversations.get(contact_id)
    if not active:
        return []
    return list(active.threads.get("unify_message", []))


def find_assistant_message(thread):
    """Find the first assistant message (name='You') in a thread."""
    for msg in thread:
        if getattr(msg, "name", "") == "You":
            return msg
    return None


async def handle_message_and_respond(cm, event):
    """Handle an incoming message event and run LLM directly.

    This is the test-friendly pattern that avoids debouncer/background tasks:
    1. Call EventHandler.handle_event() - adds message to thread
    2. Call cm._run_llm() directly - runs LLM synchronously
       (with test_sync_actions=True, actions are awaited synchronously)
    3. Process response events directly (bypass event broker)
    """
    from unity.conversation_manager.events import SMSSent, EmailSent, UnifyMessageSent

    # Subscribe to capture events BEFORE running LLM
    captured_events = []

    # Store original publish method
    original_publish = cm.event_broker.publish

    async def capturing_publish(channel, message):
        # Capture events we care about
        if channel.startswith("app:comms:"):
            try:
                from unity.conversation_manager.events import Event

                evt = Event.from_json(message)
                if isinstance(evt, (SMSSent, EmailSent, UnifyMessageSent)):
                    captured_events.append(evt)
            except Exception:
                pass
        return await original_publish(channel, message)

    # Temporarily replace publish to capture events
    cm.event_broker.publish = capturing_publish

    try:
        await EventHandler.handle_event(event, cm)
        await cm._run_llm()
    finally:
        # Restore original publish
        cm.event_broker.publish = original_publish

    # Process captured response events to update thread state
    for evt in captured_events:
        await EventHandler.handle_event(evt, cm)


@pytest.mark.asyncio
@_handle_project
async def test_sms_to_sms(initialized_cm):
    """
    Test basic SMS flow: incoming SMS triggers LLM response via SMS.

    Uses direct handler call instead of event publishing to avoid
    background task / event loop issues with pytest-asyncio.
    """
    cm = initialized_cm
    contact = TEST_CONTACTS[1]

    event = SMSReceived(
        contact=contact,
        content="Tell me a joke",
    )

    print(f"\n📱 Processing SMS from {contact['phone_number']}")
    await handle_message_and_respond(cm, event)

    # Check response in CM state directly
    sms_thread = get_sms_thread(cm, contact["contact_id"])
    print(f"📱 SMS thread has {len(sms_thread)} messages")

    # Find assistant response
    assistant_msg = find_assistant_message(sms_thread)
    assert assistant_msg is not None, "Expected assistant SMS response"
    assert len(assistant_msg.content) > 0, "Expected non-empty response"

    print(f"✅ Got SMS response: {assistant_msg.content[:100]}...")


@pytest.mark.asyncio
@_handle_project
async def test_sms_to_email(initialized_cm):
    """Test SMS → LLM → email response flow."""
    cm = initialized_cm
    contact = TEST_CONTACTS[1]

    event = SMSReceived(
        contact=contact,
        content="Tell me a joke via email",
    )

    print(f"\n📱 Processing SMS from {contact['phone_number']}")
    await handle_message_and_respond(cm, event)

    # Check email thread for response
    email_thread = get_email_thread(cm, contact["contact_id"])
    assistant_msg = find_assistant_message(email_thread)
    assert assistant_msg is not None, "Expected assistant email response"

    print(f"✅ Got email response: {assistant_msg.body[:100]}...")


@pytest.mark.asyncio
@_handle_project
async def test_email_to_email(initialized_cm):
    """Test basic email flow: incoming email triggers LLM response via email."""
    cm = initialized_cm
    contact = TEST_CONTACTS[1]

    event = EmailReceived(
        contact=contact,
        subject="Test Subject",
        body="Tell me a joke",
        email_id="test_email_id",
    )

    print(f"\n📧 Processing email from {contact['email_address']}")
    await handle_message_and_respond(cm, event)

    # Check email thread for response
    email_thread = get_email_thread(cm, contact["contact_id"])
    assistant_msg = find_assistant_message(email_thread)
    assert assistant_msg is not None, "Expected assistant email response"

    print(f"✅ Got email response: {assistant_msg.body[:100]}...")


@pytest.mark.asyncio
@_handle_project
async def test_email_to_sms(initialized_cm):
    """Test email → LLM → SMS response flow."""
    cm = initialized_cm
    contact = TEST_CONTACTS[1]

    event = EmailReceived(
        contact=contact,
        subject="Test Subject",
        body="Tell me a joke via SMS",
        email_id="test_email_id",
    )

    print(f"\n📧 Processing email from {contact['email_address']}")
    await handle_message_and_respond(cm, event)

    # Check SMS thread for response
    sms_thread = get_sms_thread(cm, contact["contact_id"])
    assistant_msg = find_assistant_message(sms_thread)
    assert assistant_msg is not None, "Expected assistant SMS response"

    print(f"✅ Got SMS response: {assistant_msg.content[:100]}...")


@pytest.mark.asyncio
@_handle_project
async def test_unify_message_to_unify_message(initialized_cm):
    """Test unify message flow: incoming message triggers LLM response."""
    cm = initialized_cm
    contact = TEST_CONTACTS[1]

    event = UnifyMessageReceived(
        contact=contact,
        content="Tell me a joke",
    )

    print(f"\n💬 Processing unify message from contact {contact['contact_id']}")
    await handle_message_and_respond(cm, event)

    # Check unify_message thread for response
    thread = get_unify_message_thread(cm, contact["contact_id"])
    assistant_msg = find_assistant_message(thread)
    assert assistant_msg is not None, "Expected assistant unify message response"

    print(f"✅ Got unify message response: {assistant_msg.content[:100]}...")


@pytest.mark.asyncio
@_handle_project
async def test_unify_message_to_sms(initialized_cm):
    """Test unify message → LLM → SMS response flow."""
    cm = initialized_cm
    contact = TEST_CONTACTS[1]

    event = UnifyMessageReceived(
        contact=contact,
        content="Tell me a joke via SMS",
    )

    print(f"\n💬 Processing unify message from contact {contact['contact_id']}")
    await handle_message_and_respond(cm, event)

    # Check SMS thread for response
    sms_thread = get_sms_thread(cm, contact["contact_id"])
    assistant_msg = find_assistant_message(sms_thread)
    assert assistant_msg is not None, "Expected assistant SMS response"

    print(f"✅ Got SMS response: {assistant_msg.content[:100]}...")


@pytest.mark.asyncio
@_handle_project
async def test_unify_message_to_email(initialized_cm):
    """Test unify message → LLM → email response flow."""
    cm = initialized_cm
    contact = TEST_CONTACTS[1]

    event = UnifyMessageReceived(
        contact=contact,
        content="Tell me a joke via email",
    )

    print(f"\n💬 Processing unify message from contact {contact['contact_id']}")
    await handle_message_and_respond(cm, event)

    # Check email thread for response
    email_thread = get_email_thread(cm, contact["contact_id"])
    assistant_msg = find_assistant_message(email_thread)
    assert assistant_msg is not None, "Expected assistant email response"

    print(f"✅ Got email response: {assistant_msg.body[:100]}...")
