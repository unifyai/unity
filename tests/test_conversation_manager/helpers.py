import asyncio
import json

from unity.conversation_manager.events import (
    ConductorClarificationRequest,
    ConductorHandleResponse,
    ConductorHandleStarted,
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


async def send_incoming_sms(test_redis_client, contact: dict, content: str):
    incoming_sms = SMSReceived(
        contact=contact,
        content=content,
    )
    print(f"\n📱 Sending SMS from {contact['phone_number']}")
    await test_redis_client.publish("app:comms:sms_received", incoming_sms.to_json())


async def send_incoming_email(
    test_redis_client,
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
    await test_redis_client.publish(
        "app:comms:email_received",
        incoming_email.to_json(),
    )


async def send_incoming_unify_message(
    test_redis_client,
    contact: dict,
    content: str,
):
    incoming_unify_message = UnifyMessageReceived(
        contact=contact,
        content=content,
    )
    print(f"\n📧 Sending unify message from {contact['contact_id']}")
    await test_redis_client.publish(
        "app:comms:unify_message_received",
        incoming_unify_message.to_json(),
    )


async def send_incoming_call(
    test_redis_client,
    contact: dict,
    conference_name: str,
    user_utterance: str,
    mode: str = "call",
):
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
    await test_redis_client.publish(
        f"app:comms:{mode}_received",
        incoming_call.to_json(),
    )
    await asyncio.sleep(0.5)

    # Subscribe to the response streaming channel
    print(f"📞 Subscribing to app:{mode}:response_gen channel")
    pubsub = test_redis_client.pubsub()
    await pubsub.subscribe(f"app:{mode}:response_gen")
    await asyncio.sleep(0.5)

    # Send call started event
    print(f"📞 Sending call started event from {contact_str}")
    if mode == "call":
        await test_redis_client.publish(
            "app:comms:phone_call_started",
            PhoneCallStarted(contact=contact).to_json(),
        )
    else:
        await test_redis_client.publish(
            "app:comms:unify_meet_started",
            UnifyMeetStarted(contact=contact).to_json(),
        )
    await asyncio.sleep(0.5)

    # Capture the initial greeting
    print("📞 Waiting for assistant's initial greeting...")
    start1, chunks1, end1 = await capture_stream_response(pubsub, "Initial greeting")
    assert start1, "Should receive start_gen for initial greeting"
    assert len(chunks1) > 0, "Should receive chunks for initial greeting"
    assert end1, "Should receive end_gen for initial greeting"

    # Send a user utterance
    print(f"📞 Sending user utterance from {contact_str}")
    if mode == "call":
        await test_redis_client.publish(
            "app:comms:phone_utterance",
            InboundPhoneUtterance(contact=contact, content=user_utterance).to_json(),
        )
    else:
        await test_redis_client.publish(
            "app:comms:unify_meet_utterance",
            InboundUnifyMeetUtterance(
                contact=contact,
                content=user_utterance,
            ).to_json(),
        )
    print(f"   Exchange 1 (Initial greeting): {len(''.join(chunks1))} characters")
    return pubsub


async def capture_outgoing_sms(event_capture, contact: dict):
    # Wait for the assistant's response
    print("⏳ Waiting for SMS response (timeout: 60s)...")
    response = await event_capture.wait_for_event(SMSSent, timeout=60.0)

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
        timeout=60.0,
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
        timeout=60.0,
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
        timeout=60.0,
    )

    # Verify response
    assert isinstance(response, PhoneCallSent)
    assert response.contact["phone_number"] == contact["phone_number"]


async def capture_stream_response(pubsub, label: str, timeout: float = 60.0):
    """Capture start_gen -> chunks -> end_gen"""
    chunks = []
    start_time = asyncio.get_event_loop().time()
    got_start = False
    got_end = False

    while (asyncio.get_event_loop().time() - start_time) < timeout:
        try:
            msg = await asyncio.wait_for(
                pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0),
                timeout=5.0,
            )

            if msg and msg["type"] == "message":
                data = json.loads(msg["data"])

                if data["type"] == "start_gen":
                    got_start = True
                    print(f"   ✓ {label}: Got start_gen")

                elif data["type"] == "gen_chunk":
                    chunk_content = data.get("chunk", "")
                    chunks.append(chunk_content)
                    print(f"   ✓ {label}: Got chunk: {chunk_content}")

                elif data["type"] == "end_gen":
                    got_end = True
                    full_response = "".join(chunks)
                    print(
                        f"   ✓ {label}: Got {len(chunks)} chunks, {len(full_response)} chars total",
                    )
                    print(f"   ✓ {label}: Preview: {full_response[:80]}...")
                    return got_start, chunks, got_end

        except asyncio.TimeoutError:
            continue

    return got_start, chunks, got_end


async def capture_task_started(
    event_capture,
    action_name: str,
    timeout: float = 60.0,
):
    """Wait for and capture a ConductorHandleStarted event (task started)"""
    print(f"⏳ Waiting for task {action_name} request (timeout: 60s)...")
    handle_started = await event_capture.wait_for_event(
        ConductorHandleStarted,
        timeout=timeout,
        action_name=action_name,
    )

    assert isinstance(handle_started, ConductorHandleStarted)
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
    timeout: float = 60.0,
):
    """Wait for and capture a ConductorHandleResponse event for a task action.

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
        if not isinstance(event, ConductorHandleResponse):
            return False
        if event.handle_id != handle_id:
            return False
        parsed = parse_action_name(event.action_name)
        return parsed.operation == operation and parsed.handle_id == handle_id

    response = await event_capture.wait_for_event_with_matcher(
        ConductorHandleResponse,
        match_action,
        timeout=timeout,
    )

    assert isinstance(response, ConductorHandleResponse)
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


async def send_conductor_clarification_request(
    test_redis_client,
    handle_id: int,
    query: str,
    call_id: str,
):
    """Manually send a ConductorClarificationRequest event"""
    clarification_request = ConductorClarificationRequest(
        handle_id=handle_id,
        query=query,
        call_id=call_id,
    )
    print(
        f"\n❓ Sending ConductorClarificationRequest: handle_id={handle_id}, call_id={call_id}",
    )
    print(f"   Query: {query}")
    await test_redis_client.publish(
        "app:conductor:output_events",
        clarification_request.to_json(),
    )
