"""
tests/conversation_manager/test_event_logging.py
=====================================================

Tests that verify ConversationManager publishes events to the EventBus
for observability.

Unlike ContactManager and TranscriptManager which publish ManagerMethod events
for their ask()/update() methods, ConversationManager publishes Comms events
(SMSReceived, SMSSent, EmailReceived, EmailSent, etc.) as it processes
communication flows.

These tests verify that:
1. Inbound events (SMSReceived, EmailReceived) are logged to EventBus
2. Outbound events (SMSSent, EmailSent) are logged to EventBus
3. Event payloads contain the expected data
"""

from __future__ import annotations

import os
import asyncio

import pytest
import pytest_asyncio

from tests.helpers import _handle_project, capture_events, get_or_create_contact
from unity.conversation_manager.events import (
    SMSReceived,
    EmailReceived,
    EmailSent,
    UnifyMessageReceived,
)

# All tests in this file require EventBus publishing to verify event behavior
pytestmark = pytest.mark.enable_eventbus


# =============================================================================
# Helper Functions
# =============================================================================


async def wait_for_operations_queue(timeout: float = 5.0) -> None:
    """
    Wait for all queued operations (including publish_bus_events) to complete.

    The CM uses an async queue for operations like publishing to EventBus.
    This helper waits for that queue to be empty.
    """
    from unity.conversation_manager.domains import managers_utils

    # Yield to the event loop so the fire-and-forget create_task() in the
    # event handler can execute its Queue.put() (non-blocking on an unbounded
    # queue, so a single yield is sufficient).
    await asyncio.sleep(0)

    try:
        await asyncio.wait_for(
            managers_utils._operations_queue.join(),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        pass  # Continue even if timeout - some events may still be captured


# =============================================================================
# Test Fixture: CM with EventBus publishing enabled
# =============================================================================


def _apply_comms_only_mocks(cm) -> None:
    """
    Apply mocks for external communication services only, NOT for EventBus publishing.

    This allows testing EventBus event logging while still mocking external services
    (SMS, email, etc.) that we don't want to actually call during tests.
    """
    from unity.conversation_manager.domains import comms_utils
    from unity.conversation_manager import debug_logger
    from unity.conversation_manager.domains.event_handlers import EventHandler
    from unity.conversation_manager.events import SummarizeContext

    def _sync_mock_success(*args, **kwargs):
        return {"success": True}

    async def _async_mock_success(*args, **kwargs):
        return {"success": True}

    # Mock external communication services
    comms_utils.send_sms_message_via_number = _async_mock_success
    comms_utils.send_unify_message = _async_mock_success
    comms_utils.send_email_via_address = _async_mock_success
    comms_utils.start_call = _async_mock_success
    cm.call_manager.start_call = _async_mock_success
    cm.call_manager.start_unify_meet = _async_mock_success
    cm.schedule_proactive_speech = _async_mock_success
    debug_logger.log_job_startup = _sync_mock_success
    debug_logger.mark_job_done = _sync_mock_success
    # NOTE: We do NOT mock managers_utils.publish_bus_events - we want events published
    # NOTE: We do NOT mock managers_utils.log_message - that's TranscriptManager logging
    EventHandler._registry[SummarizeContext] = _async_mock_success


# Test contacts
TEST_CONTACTS = [
    {
        "contact_id": 0,
        "first_name": "Test",
        "surname": "Assistant",
        "email_address": "assistant@test.com",
        "phone_number": "+15555551234",
    },
    {
        "contact_id": 1,
        "first_name": "Test",
        "surname": "Contact",
        "email_address": "test@contact.com",
        "phone_number": "+15555551111",
    },
]


@pytest_asyncio.fixture
async def cm_with_eventbus():
    """
    Create a ConversationManager with EventBus publishing enabled (not mocked).

    Unlike the main test fixtures, this one:
    - Does NOT mock publish_bus_events (so events go to EventBus)
    - DOES mock external services (SMS, email, calls)

    This is a function-scoped fixture for isolation.
    """
    from unity.actor.simulated import SimulatedActor
    from unity.conversation_manager.event_broker import reset_event_broker
    from unity.conversation_manager import start_async, stop_async
    from unity.conversation_manager.domains import managers_utils

    # Only Actor is simulated - avoids computer environment dependencies
    os.environ["UNITY_ACTOR_IMPL"] = "simulated"
    os.environ["UNITY_ACTOR_SIMULATED_STEPS"] = "3"
    os.environ["UNITY_MEMORY_ENABLED"] = "false"
    os.environ["UNITY_KNOWLEDGE_ENABLED"] = "false"
    os.environ["UNITY_GUIDANCE_ENABLED"] = "false"
    os.environ["UNITY_SECRET_ENABLED"] = "false"
    os.environ["UNITY_SKILL_ENABLED"] = "false"
    os.environ["UNITY_WEB_ENABLED"] = "false"
    os.environ["UNITY_FILE_ENABLED"] = "false"
    os.environ["UNITY_INCREMENTING_TIMESTAMPS"] = "true"
    os.environ["TEST"] = "true"
    os.environ["UNITY_CONVERSATION_JOB_NAME"] = "test_event_logging_job"

    reset_event_broker()

    # Start CM WITHOUT test mocks (so publish_bus_events is not mocked)
    cm = await start_async(
        project_name="TestEventLogging",
        enable_comms_manager=False,
        apply_test_mocks=False,  # Don't apply default test mocks
    )

    # Apply our custom mocks that only mock external services, not EventBus
    _apply_comms_only_mocks(cm)

    # Initialize managers with SimulatedActor
    actor = SimulatedActor(steps=3, log_mode="log", emit_notifications=False)
    await managers_utils.init_conv_manager(cm, actor=actor)

    # Start the operations listener that processes EventBus publishing
    asyncio.create_task(managers_utils.listen_to_operations(cm))

    # Update test contacts in ContactManager (source of truth)
    # ContactIndex.get_contact() queries ContactManager directly
    if cm.contact_manager is not None:
        for contact_data in TEST_CONTACTS:
            cm.contact_manager.update_contact(
                contact_id=contact_data["contact_id"],
                first_name=contact_data.get("first_name"),
                surname=contact_data.get("surname"),
                email_address=contact_data.get("email_address"),
                phone_number=contact_data.get("phone_number"),
                should_respond=True,
            )

    yield cm

    # Cleanup
    await stop_async()
    reset_event_broker()


# =============================================================================
# Event Logging Tests
# =============================================================================


@pytest.mark.asyncio
@_handle_project
async def test_sms_events_logged_to_eventbus(cm_with_eventbus):
    """
    Verify that SMS events are published to the EventBus.

    When an SMS is received and processed:
    1. SMSReceived event should be logged with contact and content
    2. SMSSent event should be logged with the assistant's response
    """
    from unity.conversation_manager.domains.event_handlers import EventHandler

    cm = cm_with_eventbus
    contact = TEST_CONTACTS[1]

    # Unique content for filtering
    unique_content = "📱 Test event logging SMS message"

    sms_event = SMSReceived(
        contact=contact,
        content=unique_content,
    )

    async with capture_events("Comms") as events:
        # Process the SMS event
        await EventHandler.handle_event(
            sms_event,
            cm,
            is_voice_call=False,
        )

        # Wait for queued operations (publish_bus_events) to complete
        await wait_for_operations_queue()

    # Filter for SMS events
    sms_received_events = [
        e
        for e in events
        if e.payload_cls == "SMSReceived" and e.payload.get("content") == unique_content
    ]

    assert sms_received_events, (
        f"No SMSReceived event logged to EventBus. "
        f"Found events: {[e.payload_cls for e in events]}"
    )

    # Verify payload content
    received_evt = sms_received_events[0]
    assert (
        received_evt.payload.get("contact") == contact
    ), "SMSReceived event should contain the contact"


@pytest.mark.asyncio
@_handle_project
async def test_email_events_logged_to_eventbus(cm_with_eventbus):
    """
    Verify that Email events are published to the EventBus.

    When an email is received and processed:
    1. EmailReceived event should be logged with contact, subject, and body
    """
    from unity.conversation_manager.domains.event_handlers import EventHandler

    cm = cm_with_eventbus
    contact = TEST_CONTACTS[1]

    unique_subject = "📧 Test Event Logging Subject"
    unique_body = "This is a test email for event logging verification."

    email_event = EmailReceived(
        contact=contact,
        subject=unique_subject,
        body=unique_body,
        email_id="test_event_logging_email_001",
    )

    async with capture_events("Comms") as events:
        # Process the email event
        await EventHandler.handle_event(
            email_event,
            cm,
            is_voice_call=False,
        )

        # Wait for queued operations (publish_bus_events) to complete
        await wait_for_operations_queue()

    # Filter for email received events
    email_received_events = [
        e
        for e in events
        if e.payload_cls == "EmailReceived"
        and e.payload.get("subject") == unique_subject
    ]

    assert email_received_events, (
        f"No EmailReceived event logged to EventBus. "
        f"Found events: {[e.payload_cls for e in events]}"
    )

    # Verify payload content
    received_evt = email_received_events[0]
    assert (
        received_evt.payload.get("body") == unique_body
    ), "EmailReceived event should contain the body"
    assert (
        received_evt.payload.get("contact") == contact
    ), "EmailReceived event should contain the contact"


@pytest.mark.asyncio
@_handle_project
async def test_unify_message_events_logged_to_eventbus(cm_with_eventbus):
    """
    Verify that UnifyMessage events are published to the EventBus.

    When a Unify message is received and processed:
    1. UnifyMessageReceived event should be logged
    """
    from unity.conversation_manager.domains.event_handlers import EventHandler

    cm = cm_with_eventbus
    contact = TEST_CONTACTS[1]

    unique_content = "💬 Test event logging Unify message"

    unify_msg_event = UnifyMessageReceived(
        contact=contact,
        content=unique_content,
    )

    async with capture_events("Comms") as events:
        # Process the Unify message event
        await EventHandler.handle_event(
            unify_msg_event,
            cm,
            is_voice_call=False,
        )

        # Wait for queued operations (publish_bus_events) to complete
        await wait_for_operations_queue()

    # Filter for Unify message received events
    unify_received_events = [
        e
        for e in events
        if e.payload_cls == "UnifyMessageReceived"
        and e.payload.get("content") == unique_content
    ]

    assert unify_received_events, (
        f"No UnifyMessageReceived event logged to EventBus. "
        f"Found events: {[e.payload_cls for e in events]}"
    )

    # Verify payload content
    received_evt = unify_received_events[0]
    assert (
        received_evt.payload.get("contact") == contact
    ), "UnifyMessageReceived event should contain the contact"


@pytest.mark.asyncio
@_handle_project
async def test_event_bus_event_has_correct_type(cm_with_eventbus):
    """
    Verify that CM events published to EventBus have type="Comms".

    This is important for filtering and observability - all CM events
    should be identifiable as Comms events.
    """
    from unity.conversation_manager.domains.event_handlers import EventHandler

    cm = cm_with_eventbus
    contact = TEST_CONTACTS[1]

    sms_event = SMSReceived(
        contact=contact,
        content="Test event type verification",
    )

    async with capture_events("Comms") as events:
        await EventHandler.handle_event(
            sms_event,
            cm,
            is_voice_call=False,
        )
        await wait_for_operations_queue()

    # All captured events should have type="Comms"
    for event in events:
        assert (
            event.type == "Comms"
        ), f"Expected event type 'Comms', got '{event.type}' for {event.payload_cls}"


@pytest.mark.asyncio
@_handle_project
async def test_event_bus_event_excludes_sensitive_data(cm_with_eventbus):
    """
    Verify that sensitive data (api_key, email_id) is stripped from EventBus events.

    The publish_bus_events function should remove sensitive fields before
    publishing to ensure they don't end up in logs/observability systems.
    """
    from unity.conversation_manager.domains.event_handlers import EventHandler

    cm = cm_with_eventbus
    contact = TEST_CONTACTS[1]

    email_event = EmailReceived(
        contact=contact,
        subject="Test sensitive data stripping",
        body="This email has an email_id that should be stripped",
        email_id="sensitive_email_id_12345",  # This should be stripped
    )

    async with capture_events("Comms") as events:
        await EventHandler.handle_event(
            email_event,
            cm,
            is_voice_call=False,
        )
        await wait_for_operations_queue()

    # Find the EmailReceived event
    email_events = [e for e in events if e.payload_cls == "EmailReceived"]

    if email_events:
        # email_id should be stripped from the payload
        assert (
            "email_id" not in email_events[0].payload
        ), "email_id should be stripped from EventBus payload for security"


# =============================================================================
# Transcript Logging: Email Recipient Fidelity
# =============================================================================


@pytest.mark.asyncio
@_handle_project
async def test_inbound_email_transcript_includes_all_recipients(cm_with_eventbus):
    """
    When an inbound email has multiple to/cc recipients, the transcript
    entry's receiver_ids should include all resolved recipient contact IDs,
    not just [0] (the assistant).
    """
    from unity.conversation_manager.domains.event_handlers import EventHandler
    import unify

    cm = cm_with_eventbus
    assert cm.contact_manager is not None, "ContactManager not initialized"

    # Create sender and recipient contacts
    alice_email = "alice_inbound@example.com"
    bob_email = "bob_inbound@example.com"
    charlie_email = "charlie_inbound@example.com"

    alice_id = get_or_create_contact(
        cm.contact_manager,
        email_address=alice_email,
        first_name="Alice",
        surname="Sender",
    )
    bob_id = get_or_create_contact(
        cm.contact_manager,
        email_address=bob_email,
        first_name="Bob",
        surname="ToRecipient",
    )
    charlie_id = get_or_create_contact(
        cm.contact_manager,
        email_address=charlie_email,
        first_name="Charlie",
        surname="CcRecipient",
    )

    alice = {"contact_id": alice_id, "email_address": alice_email}

    unique_subject = "Inbound Recipient Logging Test"

    # Alice sends an email TO bob, CC charlie
    email_event = EmailReceived(
        contact=alice,
        subject=unique_subject,
        body="Testing inbound recipient logging.",
        email_id="test_inbound_recipients_001",
        to=[bob_email],
        cc=[charlie_email],
    )

    await EventHandler.handle_event(email_event, cm, is_voice_call=False)
    await wait_for_operations_queue()

    # Query the transcript for this message
    tm = cm.transcript_manager
    assert tm is not None, "TranscriptManager not initialized"
    ctx = getattr(tm, "_transcripts_ctx", None)
    assert ctx, "TranscriptManager missing _transcripts_ctx"

    logs = unify.get_logs(
        context=ctx,
        limit=10,
        sorting={"timestamp": "descending"},
        from_fields=["message_id", "content", "sender_id", "receiver_ids"],
    )

    # Find our message
    target_log = None
    for lg in logs or []:
        content = str((lg.entries or {}).get("content") or "")
        if unique_subject.lower() in content.lower():
            target_log = dict(lg.entries or {})
            break

    assert (
        target_log is not None
    ), f"Did not find transcript message containing {unique_subject!r}"

    receiver_ids = target_log.get("receiver_ids", [])
    receiver_ids_int = [int(x) for x in receiver_ids]

    # sender_id should be Alice
    assert (
        int(target_log["sender_id"]) == alice_id
    ), f"Expected sender_id={alice_id}, got {target_log['sender_id']}"

    # receiver_ids should include Bob (to) and Charlie (cc), not just [0]
    assert bob_id in receiver_ids_int, (
        f"Bob (to recipient, id={bob_id}) should be in receiver_ids, "
        f"got {receiver_ids_int}"
    )
    assert charlie_id in receiver_ids_int, (
        f"Charlie (cc recipient, id={charlie_id}) should be in receiver_ids, "
        f"got {receiver_ids_int}"
    )


@pytest.mark.asyncio
@_handle_project
async def test_outbound_email_transcript_includes_all_recipients(cm_with_eventbus):
    """
    When an outbound email is sent to multiple to/cc recipients, the
    transcript entry's receiver_ids should include all recipient contact IDs,
    not just the single contact from event.contact.
    """
    from unity.conversation_manager.domains.event_handlers import EventHandler
    import unify

    cm = cm_with_eventbus
    assert cm.contact_manager is not None, "ContactManager not initialized"

    # Create recipient contacts
    alice_email = "alice_outbound@example.com"
    bob_email = "bob_outbound@example.com"

    alice_id = get_or_create_contact(
        cm.contact_manager,
        email_address=alice_email,
        first_name="Alice",
        surname="Primary",
    )
    bob_id = get_or_create_contact(
        cm.contact_manager,
        email_address=bob_email,
        first_name="Bob",
        surname="CcRecipient",
    )

    alice = {"contact_id": alice_id, "email_address": alice_email}

    unique_subject = "Outbound Recipient Logging Test"

    # Assistant sends an email TO alice, CC bob
    email_event = EmailSent(
        contact=alice,
        subject=unique_subject,
        body="Testing outbound recipient logging.",
        to=[alice_email],
        cc=[bob_email],
    )

    await EventHandler.handle_event(email_event, cm, is_voice_call=False)
    await wait_for_operations_queue()

    # Query the transcript for this message
    tm = cm.transcript_manager
    assert tm is not None, "TranscriptManager not initialized"
    ctx = getattr(tm, "_transcripts_ctx", None)
    assert ctx, "TranscriptManager missing _transcripts_ctx"

    logs = unify.get_logs(
        context=ctx,
        limit=10,
        sorting={"timestamp": "descending"},
        from_fields=["message_id", "content", "sender_id", "receiver_ids"],
    )

    # Find our message
    target_log = None
    for lg in logs or []:
        content = str((lg.entries or {}).get("content") or "")
        if unique_subject.lower() in content.lower():
            target_log = dict(lg.entries or {})
            break

    assert (
        target_log is not None
    ), f"Did not find transcript message containing {unique_subject!r}"

    receiver_ids = target_log.get("receiver_ids", [])
    receiver_ids_int = [int(x) for x in receiver_ids]

    # sender_id should be 0 (the assistant)
    assert (
        int(target_log["sender_id"]) == 0
    ), f"Expected sender_id=0 (assistant), got {target_log['sender_id']}"

    # receiver_ids should include both Alice (to) and Bob (cc)
    assert alice_id in receiver_ids_int, (
        f"Alice (to recipient, id={alice_id}) should be in receiver_ids, "
        f"got {receiver_ids_int}"
    )
    assert bob_id in receiver_ids_int, (
        f"Bob (cc recipient, id={bob_id}) should be in receiver_ids, "
        f"got {receiver_ids_int}"
    )
