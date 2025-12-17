"""
tests/test_conversation_manager/test_comms.py
=============================================

Tests for communication flows (SMS, email, calls, etc.)

Uses **direct handler testing** pattern:
- Call EventHandler.handle_event() directly instead of publishing events
- Call cm._run_llm() directly instead of relying on debouncer
- Check CM state directly instead of waiting for published events
- Same pattern as ContactManager tests

Voice call tests verify that events are handled correctly. In the voice
architecture, the Main CM Brain only provides guidance to the Voice Agent
(fast brain) - it doesn't produce speech directly. The Voice Agent handles
all conversational responses. These tests verify event flow, not speech output.
"""

import pytest

from tests.helpers import _handle_project
from tests.test_conversation_manager.conftest import TEST_CONTACTS
from unity.conversation_manager.events import (
    SMSReceived,
    EmailReceived,
    UnifyMessageReceived,
    PhoneCallReceived,
    PhoneCallStarted,
    PhoneCallEnded,
    InboundPhoneUtterance,
    UnifyMeetReceived,
    UnifyMeetStarted,
    UnifyMeetEnded,
    InboundUnifyMeetUtterance,
    SMSSent,
    EmailSent,
    UnifyMessageSent,
    PhoneCallSent,
)
from unity.conversation_manager.domains.event_handlers import EventHandler

pytestmark = pytest.mark.eval


# =============================================================================
# Helper functions for direct state inspection
# =============================================================================


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


def get_voice_thread(cm, contact_id: int):
    """Get voice thread for a contact from CM state."""
    active = cm.contact_index.active_conversations.get(contact_id)
    if not active:
        return []
    return list(active.threads.get("voice", []))


def find_assistant_message(thread):
    """Find the first assistant message (name='You') in a thread."""
    for msg in thread:
        if getattr(msg, "name", "") == "You":
            return msg
    return None


# =============================================================================
# Core test helper for message handling
# =============================================================================


async def handle_message_and_respond(cm, event):
    """Handle an incoming message event and run LLM directly.

    This is the test-friendly pattern that avoids debouncer/background tasks:
    1. Call EventHandler.handle_event() - adds message to thread
    2. Call cm._run_llm() directly - runs LLM synchronously
    3. Capture and process response events to update thread state
    """
    # Capture events published by actions
    captured_events = []
    original_publish = cm.event_broker.publish

    async def capturing_publish(channel, message):
        if channel.startswith("app:comms:"):
            try:
                from unity.conversation_manager.events import Event

                evt = Event.from_json(message)
                if isinstance(
                    evt,
                    (SMSSent, EmailSent, UnifyMessageSent, PhoneCallSent),
                ):
                    captured_events.append(evt)
            except Exception:
                pass
        return await original_publish(channel, message)

    cm.event_broker.publish = capturing_publish

    try:
        await EventHandler.handle_event(event, cm)
        await cm._run_llm()
    finally:
        cm.event_broker.publish = original_publish

    # Process captured response events to update thread state
    for evt in captured_events:
        await EventHandler.handle_event(evt, cm)

    return captured_events


async def handle_voice_call_setup(cm, contact, mode="call"):
    """Set up a voice call by handling the received and started events."""
    if mode == "call":
        received_event = PhoneCallReceived(contact=contact, conference_name="test_conf")
        started_event = PhoneCallStarted(contact=contact)
    else:
        received_event = UnifyMeetReceived(contact=contact)
        started_event = UnifyMeetStarted(contact=contact)

    await EventHandler.handle_event(received_event, cm)
    await EventHandler.handle_event(started_event, cm)


async def handle_voice_utterance_and_respond(cm, contact, content, mode="call"):
    """Handle an incoming voice utterance and run LLM."""
    if mode == "call":
        utterance_event = InboundPhoneUtterance(contact=contact, content=content)
    else:
        utterance_event = InboundUnifyMeetUtterance(contact=contact, content=content)

    return await handle_message_and_respond(cm, utterance_event)


async def handle_voice_call_end(cm, contact, mode="call"):
    """End a voice call."""
    if mode == "call":
        end_event = PhoneCallEnded(contact=contact)
    else:
        end_event = UnifyMeetEnded(contact=contact)
    await EventHandler.handle_event(end_event, cm)


# =============================================================================
# Text-based message tests (SMS, Email, UnifyMessage)
# =============================================================================


@pytest.mark.asyncio
@_handle_project
async def test_sms_to_sms(initialized_cm):
    """Test basic SMS flow: incoming SMS triggers LLM response via SMS."""
    cm = initialized_cm
    contact = TEST_CONTACTS[1]

    event = SMSReceived(contact=contact, content="Tell me a joke")
    print(f"\n📱 Processing SMS from {contact['phone_number']}")

    await handle_message_and_respond(cm, event)

    sms_thread = get_sms_thread(cm, contact["contact_id"])
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

    event = SMSReceived(contact=contact, content="Tell me a joke via email")
    print(f"\n📱 Processing SMS from {contact['phone_number']}")

    await handle_message_and_respond(cm, event)

    email_thread = get_email_thread(cm, contact["contact_id"])
    assistant_msg = find_assistant_message(email_thread)
    assert assistant_msg is not None, "Expected assistant email response"
    print(f"✅ Got email response: {assistant_msg.body[:100]}...")


@pytest.mark.asyncio
@_handle_project
async def test_sms_to_unify_message(initialized_cm):
    """Test SMS → LLM → unify message response flow."""
    cm = initialized_cm
    contact = TEST_CONTACTS[1]

    event = SMSReceived(contact=contact, content="Tell me a joke via unify message")
    print(f"\n📱 Processing SMS from {contact['phone_number']}")

    await handle_message_and_respond(cm, event)

    thread = get_unify_message_thread(cm, contact["contact_id"])
    assistant_msg = find_assistant_message(thread)
    assert assistant_msg is not None, "Expected assistant unify message response"
    print(f"✅ Got unify message response: {assistant_msg.content[:100]}...")


@pytest.mark.asyncio
@_handle_project
async def test_sms_to_phone_call(initialized_cm):
    """Test SMS → LLM → phone call response flow."""
    cm = initialized_cm
    contact = TEST_CONTACTS[1]

    event = SMSReceived(contact=contact, content="Tell me a joke via phone call")
    print(f"\n📱 Processing SMS from {contact['phone_number']}")

    captured = await handle_message_and_respond(cm, event)

    # Verify phone call was initiated
    phone_calls = [e for e in captured if isinstance(e, PhoneCallSent)]
    assert len(phone_calls) >= 1, "Expected phone call to be initiated"
    print("✅ Phone call initiated")


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

    sms_thread = get_sms_thread(cm, contact["contact_id"])
    assistant_msg = find_assistant_message(sms_thread)
    assert assistant_msg is not None, "Expected assistant SMS response"
    print(f"✅ Got SMS response: {assistant_msg.content[:100]}...")


@pytest.mark.asyncio
@_handle_project
async def test_email_to_unify_message(initialized_cm):
    """Test email → LLM → unify message response flow."""
    cm = initialized_cm
    contact = TEST_CONTACTS[1]

    event = EmailReceived(
        contact=contact,
        subject="Test Subject",
        body="Tell me a joke via unify message",
        email_id="test_email_id",
    )
    print(f"\n📧 Processing email from {contact['email_address']}")

    await handle_message_and_respond(cm, event)

    thread = get_unify_message_thread(cm, contact["contact_id"])
    assistant_msg = find_assistant_message(thread)
    assert assistant_msg is not None, "Expected assistant unify message response"
    print(f"✅ Got unify message response: {assistant_msg.content[:100]}...")


@pytest.mark.asyncio
@_handle_project
async def test_email_to_phone_call(initialized_cm):
    """Test email → LLM → phone call response flow."""
    cm = initialized_cm
    contact = TEST_CONTACTS[1]

    event = EmailReceived(
        contact=contact,
        subject="Test Subject",
        body="Tell me a joke via phone call",
        email_id="test_email_id",
    )
    print(f"\n📧 Processing email from {contact['email_address']}")

    captured = await handle_message_and_respond(cm, event)

    phone_calls = [e for e in captured if isinstance(e, PhoneCallSent)]
    assert len(phone_calls) >= 1, "Expected phone call to be initiated"
    print("✅ Phone call initiated")


@pytest.mark.asyncio
@_handle_project
async def test_unify_message_to_unify_message(initialized_cm):
    """Test unify message flow: incoming message triggers LLM response."""
    cm = initialized_cm
    contact = TEST_CONTACTS[1]

    event = UnifyMessageReceived(contact=contact, content="Tell me a joke")
    print(f"\n💬 Processing unify message from contact {contact['contact_id']}")

    await handle_message_and_respond(cm, event)

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

    event = UnifyMessageReceived(contact=contact, content="Tell me a joke via SMS")
    print(f"\n💬 Processing unify message from contact {contact['contact_id']}")

    await handle_message_and_respond(cm, event)

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

    event = UnifyMessageReceived(contact=contact, content="Tell me a joke via email")
    print(f"\n💬 Processing unify message from contact {contact['contact_id']}")

    await handle_message_and_respond(cm, event)

    email_thread = get_email_thread(cm, contact["contact_id"])
    assistant_msg = find_assistant_message(email_thread)
    assert assistant_msg is not None, "Expected assistant email response"
    print(f"✅ Got email response: {assistant_msg.body[:100]}...")


@pytest.mark.asyncio
@_handle_project
async def test_unify_message_to_phone_call(initialized_cm):
    """Test unify message → LLM → phone call response flow."""
    cm = initialized_cm
    contact = TEST_CONTACTS[1]

    event = UnifyMessageReceived(
        contact=contact,
        content="Tell me a joke via phone call",
    )
    print(f"\n💬 Processing unify message from contact {contact['contact_id']}")

    captured = await handle_message_and_respond(cm, event)

    phone_calls = [e for e in captured if isinstance(e, PhoneCallSent)]
    assert len(phone_calls) >= 1, "Expected phone call to be initiated"
    print("✅ Phone call initiated")


# =============================================================================
# Phone call tests
# =============================================================================


@pytest.mark.asyncio
@_handle_project
async def test_phone_call(initialized_cm):
    """
    Test phone call flow.

    In the voice architecture, the Main CM Brain only provides guidance to the
    Voice Agent (fast brain) - it doesn't produce speech directly. The Voice
    Agent handles all conversational responses. We verify the call events are
    processed correctly.
    """
    cm = initialized_cm
    contact = TEST_CONTACTS[1]

    print(f"\n📞 Processing phone call from {contact['phone_number']}")

    # Set up the call
    await handle_voice_call_setup(cm, contact, mode="call")

    # Send utterance and get response
    await handle_voice_utterance_and_respond(cm, contact, "Tell me a joke", mode="call")

    # Verify voice thread has the utterance
    voice_thread = get_voice_thread(cm, contact["contact_id"])
    assert len(voice_thread) >= 1, "Should record inbound phone utterance"

    # End the call
    await handle_voice_call_end(cm, contact, mode="call")
    print("✅ Phone call test complete!")


@pytest.mark.asyncio
@_handle_project
async def test_phone_call_to_sms(initialized_cm):
    """
    Test phone call to SMS flow: user on a call requests SMS, verify SMS is sent.
    """
    cm = initialized_cm
    contact = TEST_CONTACTS[1]

    print(f"\n📞 Processing phone call from {contact['phone_number']}")

    await handle_voice_call_setup(cm, contact, mode="call")
    await handle_voice_utterance_and_respond(
        cm,
        contact,
        "Tell me a joke via SMS right now",
        mode="call",
    )

    sms_thread = get_sms_thread(cm, contact["contact_id"])
    assistant_msg = find_assistant_message(sms_thread)
    assert assistant_msg is not None, "Expected assistant SMS response"

    await handle_voice_call_end(cm, contact, mode="call")
    print(f"✅ Got SMS response: {assistant_msg.content[:100]}...")


@pytest.mark.asyncio
@_handle_project
async def test_phone_call_to_email(initialized_cm):
    """
    Test phone call to email flow: user on a call requests email, verify email is sent.
    """
    cm = initialized_cm
    contact = TEST_CONTACTS[1]

    print(f"\n📞 Processing phone call from {contact['phone_number']}")

    await handle_voice_call_setup(cm, contact, mode="call")
    await handle_voice_utterance_and_respond(
        cm,
        contact,
        "Tell me a joke via email right now",
        mode="call",
    )

    email_thread = get_email_thread(cm, contact["contact_id"])
    assistant_msg = find_assistant_message(email_thread)
    assert assistant_msg is not None, "Expected assistant email response"

    await handle_voice_call_end(cm, contact, mode="call")
    print(f"✅ Got email response: {assistant_msg.body[:100]}...")


@pytest.mark.asyncio
@_handle_project
async def test_phone_call_to_unify_message(initialized_cm):
    """
    Test phone call to unify message flow: user on a call requests a message.
    """
    cm = initialized_cm
    contact = TEST_CONTACTS[1]

    print(f"\n📞 Processing phone call from {contact['phone_number']}")

    await handle_voice_call_setup(cm, contact, mode="call")
    await handle_voice_utterance_and_respond(
        cm,
        contact,
        "Tell me a joke via unify message right now",
        mode="call",
    )

    thread = get_unify_message_thread(cm, contact["contact_id"])
    assistant_msg = find_assistant_message(thread)
    assert assistant_msg is not None, "Expected assistant unify message response"

    await handle_voice_call_end(cm, contact, mode="call")
    print(f"✅ Got unify message response: {assistant_msg.content[:100]}...")


# =============================================================================
# Unify meet tests
# =============================================================================


@pytest.mark.asyncio
@_handle_project
async def test_unify_meet(initialized_cm):
    """
    Test unify meet flow.

    In the voice architecture, the Main CM Brain only provides guidance to the
    Voice Agent (fast brain) - it doesn't produce speech directly. The Voice
    Agent handles all conversational responses. We verify the call events are
    processed correctly.
    """
    cm = initialized_cm
    contact = TEST_CONTACTS[1]

    print(f"\n🎥 Processing unify meet from contact {contact['contact_id']}")

    await handle_voice_call_setup(cm, contact, mode="unify_meet")
    await handle_voice_utterance_and_respond(
        cm,
        contact,
        "Tell me a joke",
        mode="unify_meet",
    )

    voice_thread = get_voice_thread(cm, contact["contact_id"])
    assert len(voice_thread) >= 1, "Should record inbound unify meet utterance"

    await handle_voice_call_end(cm, contact, mode="unify_meet")
    print("✅ Unify meet test complete!")


# Note: There is no test_unify_meet_to_phone_call test because the system does not
# support maintaining multiple simultaneous voice-based conversations. While on a
# unify meet, the assistant cannot initiate an outbound phone call.


@pytest.mark.asyncio
@_handle_project
async def test_unify_meet_to_sms(initialized_cm):
    """
    Test unify meet to SMS flow: user on a call requests SMS, verify SMS is sent.
    """
    cm = initialized_cm
    contact = TEST_CONTACTS[1]

    print(f"\n🎥 Processing unify meet from contact {contact['contact_id']}")

    await handle_voice_call_setup(cm, contact, mode="unify_meet")
    await handle_voice_utterance_and_respond(
        cm,
        contact,
        "Tell me a joke via sms right now",
        mode="unify_meet",
    )

    sms_thread = get_sms_thread(cm, contact["contact_id"])
    assistant_msg = find_assistant_message(sms_thread)
    assert assistant_msg is not None, "Expected assistant SMS response"

    await handle_voice_call_end(cm, contact, mode="unify_meet")
    print(f"✅ Got SMS response: {assistant_msg.content[:100]}...")


@pytest.mark.asyncio
@_handle_project
async def test_unify_meet_to_email(initialized_cm):
    """
    Test unify meet to email flow: user on a call requests email, verify email is sent.
    """
    cm = initialized_cm
    contact = TEST_CONTACTS[1]

    print(f"\n🎥 Processing unify meet from contact {contact['contact_id']}")

    await handle_voice_call_setup(cm, contact, mode="unify_meet")
    await handle_voice_utterance_and_respond(
        cm,
        contact,
        "Tell me a joke via email right now",
        mode="unify_meet",
    )

    email_thread = get_email_thread(cm, contact["contact_id"])
    assistant_msg = find_assistant_message(email_thread)
    assert assistant_msg is not None, "Expected assistant email response"

    await handle_voice_call_end(cm, contact, mode="unify_meet")
    print(f"✅ Got email response: {assistant_msg.body[:100]}...")


@pytest.mark.asyncio
@_handle_project
async def test_unify_meet_to_unify_message(initialized_cm):
    """
    Test unify meet to unify message flow: user on a call requests a message.
    """
    cm = initialized_cm
    contact = TEST_CONTACTS[1]

    print(f"\n🎥 Processing unify meet from contact {contact['contact_id']}")

    await handle_voice_call_setup(cm, contact, mode="unify_meet")
    await handle_voice_utterance_and_respond(
        cm,
        contact,
        "Tell me a joke via unify message right now",
        mode="unify_meet",
    )

    thread = get_unify_message_thread(cm, contact["contact_id"])
    assistant_msg = find_assistant_message(thread)
    assert assistant_msg is not None, "Expected assistant unify message response"

    await handle_voice_call_end(cm, contact, mode="unify_meet")
    print(f"✅ Got unify message response: {assistant_msg.content[:100]}...")
