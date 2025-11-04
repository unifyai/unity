import asyncio
import json

from unity.conversation_manager_2.new_events import (
    EmailRecieved,
    PhoneCallRecieved,
    PhoneCallStarted,
    PhoneUtterance,
    SMSRecieved,
    UnifyCallReceived,
    UnifyCallStarted,
    UnifyCallUtterance,
    UnifyMessageRecieved,
)


async def send_incoming_sms(test_redis_client, contact_number: str, content: str):
    incoming_sms = SMSRecieved(
        contact=contact_number,
        content=content,
    )
    print(f"\n📱 Sending SMS from {contact_number}")
    await test_redis_client.publish("app:comms:sms_received", incoming_sms.to_json())


async def send_incoming_email(
    test_redis_client,
    email_address: str,
    subject: str,
    body: str,
    message_id: str,
):
    incoming_email = EmailRecieved(
        contact=email_address,
        subject=subject,
        body=body,
        message_id=message_id,
    )
    print(f"\n📧 Sending email from {email_address}")
    await test_redis_client.publish(
        "app:comms:email_received",
        incoming_email.to_json(),
    )


async def send_incoming_unify_message(
    test_redis_client,
    contact: int,
    content: str,
):
    incoming_unify_message = UnifyMessageRecieved(
        contact=contact,
        content=content,
    )
    print(f"\n📧 Sending unify message from {contact}")
    await test_redis_client.publish(
        "app:comms:unify_message_received",
        incoming_unify_message.to_json(),
    )


async def send_incoming_call(
    test_redis_client,
    contact: str | int,
    conference_name: str,
    user_utterance: str,
    mode: str = "call",
):
    # Send call received event
    if mode == "call":
        incoming_call = PhoneCallRecieved(
            contact=contact,
            conference_name=conference_name,
        )
    else:
        incoming_call = UnifyCallReceived(contact=contact)
    print(f"\n📞 Sending {incoming_call.to_dict()['event_name']} from {contact}")
    await test_redis_client.publish(
        f"app:comms:{mode}_recieved",
        incoming_call.to_json(),
    )
    await asyncio.sleep(0.5)

    # Subscribe to the response streaming channel
    print(f"📞 Subscribing to app:{mode}:response_gen channel")
    pubsub = test_redis_client.pubsub()
    await pubsub.subscribe(f"app:{mode}:response_gen")
    await asyncio.sleep(0.5)

    # Send call started event
    print(f"📞 Sending call started event from {contact}")
    if mode == "call":
        await test_redis_client.publish(
            "app:comms:phone_call_started",
            PhoneCallStarted(contact=contact).to_json(),
        )
    else:
        await test_redis_client.publish(
            "app:comms:unify_call_started",
            UnifyCallStarted(contact=contact).to_json(),
        )
    await asyncio.sleep(0.5)

    # Capture the initial greeting
    print("📞 Waiting for assistant's initial greeting...")
    start1, chunks1, end1 = await capture_stream_response(pubsub, "Initial greeting")
    assert start1, "Should receive start_gen for initial greeting"
    assert len(chunks1) > 0, "Should receive chunks for initial greeting"
    assert end1, "Should receive end_gen for initial greeting"

    # Send a user utterance
    print(f"📞 Sending user utterance from {contact}")
    if mode == "call":
        await test_redis_client.publish(
            "app:comms:phone_utterance",
            PhoneUtterance(contact=contact, content=user_utterance).to_json(),
        )
    else:
        await test_redis_client.publish(
            "app:comms:unify_call_utterance",
            UnifyCallUtterance(contact=contact, content=user_utterance).to_json(),
        )
    print(f"   Exchange 1 (Initial greeting): {len(''.join(chunks1))} characters")
    return pubsub


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
