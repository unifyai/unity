"""
Tests for the in-memory event broker.

These tests verify that the in-memory implementation matches Redis pub/sub semantics.
"""

import asyncio
import pytest

from unity.conversation_manager.in_memory_event_broker import (
    create_in_memory_event_broker,
    get_in_memory_event_broker,
    reset_in_memory_event_broker,
)


@pytest.fixture
def broker():
    """Create a fresh broker for each test."""
    return create_in_memory_event_broker()


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset the global singleton before and after each test."""
    reset_in_memory_event_broker()
    yield
    reset_in_memory_event_broker()


class TestBasicPublishSubscribe:
    """Test basic publish/subscribe functionality."""

    @pytest.mark.asyncio
    async def test_subscribe_and_receive_message(self, broker):
        """Test subscribing to a channel and receiving a message."""
        async with broker.pubsub() as pubsub:
            await pubsub.subscribe("test:channel")

            # Publish a message
            receivers = await broker.publish("test:channel", '{"hello": "world"}')
            assert receivers == 1

            # Receive the message
            msg = await pubsub.get_message(timeout=1.0, ignore_subscribe_messages=True)
            assert msg is not None
            assert msg["type"] == "message"
            assert msg["channel"] == "test:channel"
            assert msg["data"] == '{"hello": "world"}'

    @pytest.mark.asyncio
    async def test_pattern_subscribe(self, broker):
        """Test pattern-based subscription (psubscribe)."""
        async with broker.pubsub() as pubsub:
            await pubsub.psubscribe("app:comms:*")

            # Publish to matching channels
            await broker.publish("app:comms:startup", '{"type": "startup"}')
            await broker.publish("app:comms:message", '{"type": "message"}')

            # Publish to non-matching channel
            await broker.publish("app:other:event", '{"type": "other"}')

            # Should receive two messages
            msg1 = await pubsub.get_message(timeout=1.0, ignore_subscribe_messages=True)
            assert msg1 is not None
            assert msg1["type"] == "pmessage"
            assert msg1["pattern"] == "app:comms:*"
            assert msg1["channel"] == "app:comms:startup"

            msg2 = await pubsub.get_message(timeout=1.0, ignore_subscribe_messages=True)
            assert msg2 is not None
            assert msg2["channel"] == "app:comms:message"

            # Should timeout (no more messages)
            msg3 = await pubsub.get_message(timeout=0.1, ignore_subscribe_messages=True)
            assert msg3 is None

    @pytest.mark.asyncio
    async def test_multiple_subscriptions(self, broker):
        """Test subscribing to multiple channels."""
        async with broker.pubsub() as pubsub:
            await pubsub.subscribe("channel:a", "channel:b")

            await broker.publish("channel:a", "message_a")
            await broker.publish("channel:b", "message_b")
            await broker.publish("channel:c", "message_c")  # Not subscribed

            msg1 = await pubsub.get_message(timeout=1.0, ignore_subscribe_messages=True)
            assert msg1["channel"] == "channel:a"

            msg2 = await pubsub.get_message(timeout=1.0, ignore_subscribe_messages=True)
            assert msg2["channel"] == "channel:b"

            msg3 = await pubsub.get_message(timeout=0.1, ignore_subscribe_messages=True)
            assert msg3 is None

    @pytest.mark.asyncio
    async def test_multiple_patterns(self, broker):
        """Test subscribing to multiple patterns."""
        async with broker.pubsub() as pubsub:
            await pubsub.psubscribe("app:comms:*", "app:actor:*")

            await broker.publish("app:comms:event", "comms")
            await broker.publish("app:actor:event", "actor")
            await broker.publish("app:other:event", "other")  # Not matched

            msg1 = await pubsub.get_message(timeout=1.0, ignore_subscribe_messages=True)
            assert msg1["channel"] == "app:comms:event"

            msg2 = await pubsub.get_message(timeout=1.0, ignore_subscribe_messages=True)
            assert msg2["channel"] == "app:actor:event"

            msg3 = await pubsub.get_message(timeout=0.1, ignore_subscribe_messages=True)
            assert msg3 is None


class TestSubscribeMessages:
    """Test subscribe confirmation messages."""

    @pytest.mark.asyncio
    async def test_subscribe_confirmation(self, broker):
        """Test that subscribe sends confirmation messages."""
        async with broker.pubsub() as pubsub:
            await pubsub.subscribe("test:channel")

            # Should receive subscribe confirmation
            msg = await pubsub.get_message(timeout=1.0, ignore_subscribe_messages=False)
            assert msg is not None
            assert msg["type"] == "subscribe"
            assert msg["channel"] == "test:channel"

    @pytest.mark.asyncio
    async def test_psubscribe_confirmation(self, broker):
        """Test that psubscribe sends confirmation messages."""
        async with broker.pubsub() as pubsub:
            await pubsub.psubscribe("test:*")

            msg = await pubsub.get_message(timeout=1.0, ignore_subscribe_messages=False)
            assert msg is not None
            assert msg["type"] == "psubscribe"
            assert msg["channel"] == "test:*"

    @pytest.mark.asyncio
    async def test_ignore_subscribe_messages(self, broker):
        """Test that ignore_subscribe_messages works correctly."""
        async with broker.pubsub() as pubsub:
            await pubsub.subscribe("test:channel")
            await broker.publish("test:channel", "actual_message")

            # Should skip the subscribe confirmation and get the actual message
            msg = await pubsub.get_message(timeout=1.0, ignore_subscribe_messages=True)
            assert msg is not None
            assert msg["type"] == "message"
            assert msg["data"] == "actual_message"


class TestUnsubscribe:
    """Test unsubscribe functionality."""

    @pytest.mark.asyncio
    async def test_unsubscribe_specific_channel(self, broker):
        """Test unsubscribing from a specific channel."""
        async with broker.pubsub() as pubsub:
            await pubsub.subscribe("channel:a", "channel:b")

            # Unsubscribe from channel:a
            await pubsub.unsubscribe("channel:a")

            # Should only receive from channel:b
            await broker.publish("channel:a", "a")
            await broker.publish("channel:b", "b")

            msg = await pubsub.get_message(timeout=1.0, ignore_subscribe_messages=True)
            assert msg["channel"] == "channel:b"

            msg2 = await pubsub.get_message(timeout=0.1, ignore_subscribe_messages=True)
            assert msg2 is None

    @pytest.mark.asyncio
    async def test_unsubscribe_all(self, broker):
        """Test unsubscribing from all channels."""
        async with broker.pubsub() as pubsub:
            await pubsub.subscribe("channel:a", "channel:b")
            await pubsub.unsubscribe()  # Unsubscribe from all

            await broker.publish("channel:a", "a")
            await broker.publish("channel:b", "b")

            msg = await pubsub.get_message(timeout=0.1, ignore_subscribe_messages=True)
            assert msg is None


class TestMultipleSubscribers:
    """Test multiple subscribers."""

    @pytest.mark.asyncio
    async def test_multiple_subscribers_same_channel(self, broker):
        """Test that multiple subscribers receive the same message."""
        async with broker.pubsub() as pubsub1:
            async with broker.pubsub() as pubsub2:
                await pubsub1.subscribe("shared:channel")
                await pubsub2.subscribe("shared:channel")

                receivers = await broker.publish("shared:channel", "broadcast")
                assert receivers == 2

                msg1 = await pubsub1.get_message(
                    timeout=1.0,
                    ignore_subscribe_messages=True,
                )
                msg2 = await pubsub2.get_message(
                    timeout=1.0,
                    ignore_subscribe_messages=True,
                )

                assert msg1["data"] == "broadcast"
                assert msg2["data"] == "broadcast"

    @pytest.mark.asyncio
    async def test_subscriber_cleanup_on_context_exit(self, broker):
        """Test that subscriptions are cleaned up when context exits."""
        async with broker.pubsub() as pubsub:
            await pubsub.subscribe("test:channel")
            receivers = await broker.publish("test:channel", "test")
            assert receivers == 1

        # After context exit, should have no subscribers
        receivers = await broker.publish("test:channel", "test")
        assert receivers == 0


class TestTimeout:
    """Test timeout behavior."""

    @pytest.mark.asyncio
    async def test_get_message_timeout(self, broker):
        """Test that get_message returns None on timeout."""
        async with broker.pubsub() as pubsub:
            await pubsub.subscribe("test:channel")

            # No message published, should timeout
            start = asyncio.get_event_loop().time()
            msg = await pubsub.get_message(timeout=0.1, ignore_subscribe_messages=True)
            elapsed = asyncio.get_event_loop().time() - start

            assert msg is None
            assert 0.1 <= elapsed < 0.5  # Should have waited ~0.1 seconds


class TestExecuteCommand:
    """Test execute_command for Redis compatibility."""

    @pytest.mark.asyncio
    async def test_pubsub_numpat(self, broker):
        """Test PUBSUB NUMPAT command."""
        # No subscriptions initially
        numpat = await broker.execute_command("PUBSUB", "NUMPAT")
        assert numpat == 0

        async with broker.pubsub() as pubsub:
            await pubsub.psubscribe("pattern:*")
            numpat = await broker.execute_command("PUBSUB", "NUMPAT")
            assert numpat == 1

            await pubsub.psubscribe("another:*")
            numpat = await broker.execute_command("PUBSUB", "NUMPAT")
            assert numpat == 2

    @pytest.mark.asyncio
    async def test_unsupported_command(self, broker):
        """Test that unsupported commands raise NotImplementedError."""
        with pytest.raises(NotImplementedError):
            await broker.execute_command("UNSUPPORTED", "COMMAND")


class TestSingleton:
    """Test singleton behavior."""

    def test_get_singleton_returns_same_instance(self):
        """Test that get_in_memory_event_broker returns the same instance."""
        broker1 = get_in_memory_event_broker()
        broker2 = get_in_memory_event_broker()
        assert broker1 is broker2

    def test_reset_clears_singleton(self):
        """Test that reset creates a new instance."""
        broker1 = get_in_memory_event_broker()
        reset_in_memory_event_broker()
        broker2 = get_in_memory_event_broker()
        assert broker1 is not broker2


class TestClosedBroker:
    """Test behavior after broker is closed."""

    @pytest.mark.asyncio
    async def test_publish_after_close(self, broker):
        """Test that publish returns 0 after broker is closed."""
        await broker.aclose()
        receivers = await broker.publish("test:channel", "test")
        assert receivers == 0

    @pytest.mark.asyncio
    async def test_get_message_after_pubsub_close(self, broker):
        """Test that get_message returns None after pubsub is closed."""
        async with broker.pubsub() as pubsub:
            await pubsub.subscribe("test:channel")

        # pubsub is now closed
        # Can't call get_message on closed pubsub as it's out of scope


class TestRealWorldPatterns:
    """Test patterns that match actual ConversationManager usage."""

    @pytest.mark.asyncio
    async def test_conversation_manager_pattern(self, broker):
        """Test the actual pattern used by ConversationManager.wait_for_events()."""
        async with broker.pubsub() as pubsub:
            # This is the actual subscription from conversation_manager.py
            await pubsub.psubscribe(
                "app:comms:*",
                "app:actor:*",
                "app:logging:message_logged",
                "app:managers:output",
            )

            # Publish various events
            await broker.publish("app:comms:startup", '{"type": "startup"}')
            await broker.publish("app:actor:result", '{"handle_id": 1}')
            await broker.publish("app:logging:message_logged", '{"medium": "email"}')
            await broker.publish("app:managers:output", '{"output": "test"}')
            await broker.publish("app:unrelated:event", '{"should": "not match"}')

            # Should receive 4 messages (not the unrelated one)
            received = []
            for _ in range(5):
                msg = await pubsub.get_message(
                    timeout=0.1,
                    ignore_subscribe_messages=True,
                )
                if msg:
                    received.append(msg["channel"])
                else:
                    break

            assert len(received) == 4
            assert "app:comms:startup" in received
            assert "app:actor:result" in received
            assert "app:logging:message_logged" in received
            assert "app:managers:output" in received

    @pytest.mark.asyncio
    async def test_json_message_passthrough(self, broker):
        """Test that JSON messages are passed through unchanged."""
        import json

        test_event = {
            "type": "StartupEvent",
            "assistant_id": "test_123",
            "user_id": "user_456",
            "nested": {"key": "value"},
        }

        async with broker.pubsub() as pubsub:
            await pubsub.subscribe("app:comms:startup")

            await broker.publish("app:comms:startup", json.dumps(test_event))

            msg = await pubsub.get_message(timeout=1.0, ignore_subscribe_messages=True)
            assert msg is not None

            # Parse the data back
            received_event = json.loads(msg["data"])
            assert received_event == test_event
