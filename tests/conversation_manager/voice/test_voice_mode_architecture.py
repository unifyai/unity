"""
tests/conversation_manager/test_voice_mode_architecture.py
================================================================

Integration tests for voice mode event flows and cross-thread delivery.

These tests validate:
1. Voice call events flow correctly through the event broker
2. Call guidance events are published and received correctly
3. Cross-thread event delivery works (voice agents run in background threads)

Note: Unit tests for schema structure, config values, and method existence have been
removed as they are implementation-locked and don't catch production bugs.
"""

import asyncio
import json

import pytest
import pytest_asyncio

# =============================================================================
# Local Fixtures
# =============================================================================


@pytest_asyncio.fixture
async def event_broker():
    """
    Local in-memory broker for this test module.

    This avoids starting a full ConversationManager instance (slow, requires env),
    while still exercising the same pub/sub API that voice mode uses.
    """
    from unity.conversation_manager.event_broker import create_event_broker

    broker = create_event_broker()
    yield broker
    await broker.aclose()


# =============================================================================
# Integration Tests: Voice Call Flow (In-Memory Broker)
# =============================================================================


@pytest.mark.asyncio
class TestVoiceCallFlowIntegration:
    """
    Integration tests for voice call flows.

    These tests validate the event flow using the in-memory event broker.
    """

    @pytest.fixture
    def boss_contact(self):
        return {
            "contact_id": 1,
            "first_name": "Test",
            "surname": "Boss",
            "phone_number": "+15555555678",
            "email_address": "boss@test.com",
        }

    async def test_phone_call_started_event_flow(
        self,
        event_broker,
        boss_contact,
    ):
        """
        Verify phone call start event is properly published and captured.
        """
        from unity.conversation_manager.events import Event, PhoneCallStarted

        async with event_broker.pubsub() as pubsub:
            await pubsub.psubscribe("app:comms:*")

            # Publish a call started event
            event = PhoneCallStarted(contact=boss_contact)
            await event_broker.publish(
                "app:comms:phone_call_started",
                event.to_json(),
            )

            msg = await pubsub.get_message(
                timeout=2.0,
                ignore_subscribe_messages=True,
            )
            assert msg is not None
            captured = Event.from_json(msg["data"])
            assert isinstance(captured, PhoneCallStarted)

    async def test_call_guidance_event_flow(
        self,
        event_broker,
        boss_contact,
    ):
        """
        Verify call guidance events flow through the system.
        """
        from unity.conversation_manager.events import FastBrainNotification, Event

        async with event_broker.pubsub() as pubsub:
            await pubsub.subscribe("app:call:notification")

            # Publish a guidance event
            event = FastBrainNotification(
                contact=boss_contact,
                content="Please ask about their schedule",
            )
            await event_broker.publish("app:call:notification", event.to_json())

            msg = await pubsub.get_message(
                timeout=2.0,
                ignore_subscribe_messages=True,
            )
            assert msg is not None
            captured = Event.from_json(msg["data"])
            assert isinstance(captured, FastBrainNotification)
            assert captured.content == "Please ask about their schedule"


# =============================================================================
# Integration Tests: Voice Guidance Channel
# =============================================================================


@pytest.mark.asyncio
class TestFastBrainNotificationChannel:
    """Tests for the call_guidance channel."""

    async def test_call_guidance_channel_format(self, event_broker):
        """Verify call_guidance channel message format."""
        async with event_broker.pubsub() as pubsub:
            await pubsub.subscribe("app:call:notification")

            # Consume the subscription confirmation message
            await pubsub.get_message(timeout=1.0)

            # Publish guidance (the format used by Main CM Brain)
            await event_broker.publish(
                "app:call:notification",
                json.dumps({"content": "Please ask about their schedule"}),
            )

            msg = await pubsub.get_message(
                ignore_subscribe_messages=True,
                timeout=2.0,
            )

            assert msg is not None, "Expected to receive published message"
            assert msg["type"] == "message"
            data = json.loads(msg["data"])
            assert "content" in data
            assert data["content"] == "Please ask about their schedule"


# =============================================================================
# Threading Tests: In-process Voice Agent Uses Shared Broker
# =============================================================================


@pytest.mark.asyncio
async def test_event_broker_delivers_across_threads(event_broker):
    """
    Voice agents run in a background thread but must share the same in-memory broker.

    This validates that a subscriber created on a different event loop/thread can
    still receive published messages.
    """
    import queue
    import threading

    ready = threading.Event()
    received: "queue.Queue[dict]" = queue.Queue()

    def _subscriber_thread() -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        async def _run() -> None:
            async with event_broker.pubsub() as pubsub:
                await pubsub.subscribe("app:call:status")
                ready.set()
                msg = await pubsub.get_message(
                    timeout=2.0,
                    ignore_subscribe_messages=True,
                )
                received.put(msg)

        loop.run_until_complete(_run())
        loop.close()

    t = threading.Thread(target=_subscriber_thread, daemon=True)
    t.start()

    # Wait until the subscriber is ready (no sleeps for event alignment).
    assert await asyncio.to_thread(ready.wait, 2.0)

    await event_broker.publish("app:call:status", json.dumps({"type": "stop"}))

    msg = await asyncio.to_thread(received.get, True, 2.0)
    assert msg is not None
    assert json.loads(msg["data"])["type"] == "stop"
    t.join(timeout=2.0)


# =============================================================================
# Integration Tests: UnifyLLM Adapter Behavior
# =============================================================================


@pytest.mark.asyncio
async def test_unify_llm_chat_returns_stream():
    """UnifyLLM.chat() returns a UnifyLLMStream instance."""
    from livekit.agents import llm
    from unity.conversation_manager.livekit_unify_adapter import (
        UnifyLLM,
        UnifyLLMStream,
    )
    from unity.settings import SETTINGS

    llm_instance = UnifyLLM(model=SETTINGS.conversation.FAST_BRAIN_MODEL)

    # Create a minimal chat context
    chat_ctx = llm.ChatContext()
    chat_ctx.add_message(role="user", content="Hello")

    stream = llm_instance.chat(chat_ctx=chat_ctx)
    assert isinstance(stream, UnifyLLMStream)

    # Clean up the stream to avoid task warnings
    await stream.aclose()
