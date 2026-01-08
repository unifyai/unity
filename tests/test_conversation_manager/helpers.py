import asyncio

from unity.conversation_manager.events import (
    ActorClarificationRequest,
    ActorHandleResponse,
    ActorHandleStarted,
    EmailReceived,
    EmailSent,
    PhoneCallReceived,
    PhoneCallSent,
    PhoneCallStarted,
    InboundPhoneUtterance,
    SMSReceived,
    SMSSent,
    UnifyMeetReceived,
    UnifyMeetStarted,
    InboundUnifyMeetUtterance,
    UnifyMessageReceived,
    UnifyMessageSent,
)


contacts = [
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


async def send_incoming_sms(event_broker, contact: dict, content: str):
    incoming_sms = SMSReceived(
        contact=contact,
        content=content,
    )
    print(f"\n📱 Sending SMS from {contact['phone_number']}")
    await event_broker.publish("app:comms:sms_received", incoming_sms.to_json())


async def send_incoming_email(
    event_broker,
    contact: dict,
    subject: str,
    body: str,
    email_id: str,
):
    incoming_email = EmailReceived(
        contact=contact,
        subject=subject,
        body=body,
        email_id=email_id,
    )
    print(f"\n📧 Sending email from {contact['email_address']}")
    await event_broker.publish(
        "app:comms:email_received",
        incoming_email.to_json(),
    )


async def send_incoming_unify_message(
    event_broker,
    contact: dict,
    content: str,
):
    incoming_unify_message = UnifyMessageReceived(
        contact=contact,
        content=content,
    )
    print(f"\n📧 Sending unify message from {contact['contact_id']}")
    await event_broker.publish(
        "app:comms:unify_message_received",
        incoming_unify_message.to_json(),
    )


async def send_incoming_call(
    event_broker,
    contact: dict,
    conference_name: str,
    user_utterance: str,
    mode: str = "call",
):
    """
    Simulate an incoming voice call and send a user utterance.

    In the new voice architecture, the Main CM Brain only provides guidance
    to the Voice Agent (fast brain) - it doesn't produce speech directly.
    The Voice Agent handles all conversational responses. In tests, the
    Voice Agent isn't running, so we don't expect streaming speech content.
    """
    # Send call received event
    if mode == "call":
        incoming_call = PhoneCallReceived(
            contact=contact,
            conference_name=conference_name,
        )
        contact_str = contact["phone_number"]
    else:
        incoming_call = UnifyMeetReceived(contact=contact)
        contact_str = contact["contact_id"]
    print(f"\n📞 Sending {incoming_call.to_dict()['event_name']} from {contact_str}")
    await event_broker.publish(
        f"app:comms:{mode}_received",
        incoming_call.to_json(),
    )
    await asyncio.sleep(0.5)

    # Subscribe to the call_guidance channel (for any guidance from Main CM Brain)
    print(f"📞 Subscribing to app:call:call_guidance channel")
    # InMemoryEventBroker.pubsub() is an async context manager, so we need to enter it
    pubsub = await event_broker.pubsub().__aenter__()
    await pubsub.subscribe("app:call:call_guidance")
    await asyncio.sleep(0.5)

    # Send call started event
    print(f"📞 Sending call started event from {contact_str}")
    if mode == "call":
        await event_broker.publish(
            "app:comms:phone_call_started",
            PhoneCallStarted(contact=contact).to_json(),
        )
    else:
        await event_broker.publish(
            "app:comms:unify_meet_started",
            UnifyMeetStarted(contact=contact).to_json(),
        )
    # Give Main CM Brain time to process and respond
    await asyncio.sleep(2.0)

    # Send a user utterance
    print(f"📞 Sending user utterance from {contact_str}")
    if mode == "call":
        await event_broker.publish(
            "app:comms:phone_utterance",
            InboundPhoneUtterance(contact=contact, content=user_utterance).to_json(),
        )
    else:
        await event_broker.publish(
            "app:comms:unify_meet_utterance",
            InboundUnifyMeetUtterance(
                contact=contact,
                content=user_utterance,
            ).to_json(),
        )
    print(f"📞 User utterance sent: {user_utterance}")
    return pubsub


async def capture_outgoing_sms(event_capture, contact: dict):
    # Wait for the assistant's response
    print("⏳ Waiting for SMS response (timeout: 300s)...")
    response = await event_capture.wait_for_event(SMSSent, timeout=300.0)

    # Verify response
    assert isinstance(response, SMSSent)
    assert response.contact["phone_number"] == contact["phone_number"]
    assert len(response.content) > 0

    print(f"✅ Got SMS response: {response.content[:100]}...")
    print(f"   Full response length: {len(response.content)} characters")


async def capture_outgoing_email(event_capture, contact: dict, email_id: str = None):
    # Wait for the assistant's response
    print("⏳ Waiting for email response (timeout: 60s)...")
    response = await event_capture.wait_for_event(
        EmailSent,
        timeout=300.0,
    )

    # Verify response
    assert isinstance(response, EmailSent)
    assert response.contact["email_address"] == contact["email_address"]
    if email_id:
        assert response.email_id_replied_to == email_id
    assert len(response.body) > 0

    print(f"✅ Got email response: {response.body[:100]}...")
    print(f"   Full response length: {len(response.body)} characters")


async def capture_outgoing_unify_message(event_capture, contact: int):
    # Wait for the assistant's response
    print("⏳ Waiting for unify message response (timeout: 60s)...")
    response = await event_capture.wait_for_event(
        UnifyMessageSent,
        timeout=300.0,
    )

    # Verify response
    assert isinstance(response, UnifyMessageSent)
    assert response.contact["contact_id"] == contact["contact_id"]
    assert len(response.content) > 0

    print(f"✅ Got unify message response: {response.content[:100]}...")
    print(f"   Full response length: {len(response.content)} characters")


async def capture_outgoing_phone_call(event_capture, contact: dict):
    # Wait for the assistant's response
    print("⏳ Waiting for phone call response (timeout: 60s)...")
    response = await event_capture.wait_for_event(
        PhoneCallSent,
        timeout=300.0,
    )

    # Verify response
    assert isinstance(response, PhoneCallSent)
    assert response.contact["phone_number"] == contact["phone_number"]


async def capture_task_started(
    event_capture,
    action_name: str,
    timeout: float = 300.0,
):
    """Wait for and capture an ActorHandleStarted event (task started)."""
    print(f"⏳ Waiting for task {action_name} request (timeout: 60s)...")
    handle_started = await event_capture.wait_for_event(
        ActorHandleStarted,
        timeout=timeout,
        action_name=action_name,
    )

    assert isinstance(handle_started, ActorHandleStarted)
    assert handle_started.action_name == action_name
    assert len(handle_started.query) > 0

    print(f"✅ Got task {action_name} request")
    print(f"   Query: {handle_started.query[:100]}...")
    return handle_started


async def capture_task_action_response(
    event_capture,
    handle_id: int,
    operation: str,
    call_id: str = "",
    timeout: float = 300.0,
):
    """Wait for and capture an ActorHandleResponse event for a task action.

    Args:
        event_capture: The event capture fixture
        handle_id: The task handle ID to match
        operation: The steering operation to match (e.g., "stop", "ask", "interject")
        call_id: Optional call_id for clarification responses
        timeout: Timeout in seconds
    """
    from unity.conversation_manager.task_actions import parse_action_name, OPERATION_MAP

    print(
        f"⏳ Waiting for task action '{operation}' on handle {handle_id} (timeout: {timeout}s)...",
    )

    def match_action(event):
        if not isinstance(event, ActorHandleResponse):
            return False
        if event.handle_id != handle_id:
            return False
        parsed = parse_action_name(event.action_name)
        return parsed.operation == operation and parsed.handle_id == handle_id

    response = await event_capture.wait_for_event_with_matcher(
        ActorHandleResponse,
        match_action,
        timeout=timeout,
    )

    assert isinstance(response, ActorHandleResponse)
    assert response.handle_id == handle_id

    op = OPERATION_MAP.get(operation)
    if op and op.param_name in ("query", "message", "answer"):
        assert len(response.query) > 0

    if call_id:
        assert response.call_id == call_id

    print(
        f"✅ Got task action response: handle_id={handle_id}, action={response.action_name}",
    )
    print(f"   Query: {response.query[:100] if response.query else '(none)'}...")
    return response


async def send_actor_clarification_request(
    event_broker,
    handle_id: int,
    query: str,
    call_id: str,
):
    """Manually send an ActorClarificationRequest event."""
    clarification_request = ActorClarificationRequest(
        handle_id=handle_id,
        query=query,
        call_id=call_id,
    )
    print(
        f"\n❓ Sending ActorClarificationRequest: handle_id={handle_id}, call_id={call_id}",
    )
    print(f"   Query: {query}")
    await event_broker.publish(
        "app:actor:clarification_request",
        clarification_request.to_json(),
    )
