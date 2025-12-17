"""
In-memory event broker for async pub/sub communication.

Provides:
- Async publish/subscribe API
- Pattern-based subscriptions (psubscribe with glob patterns)
- No serialization overhead (messages passed by reference)
- No external dependencies
"""

from __future__ import annotations

import asyncio
import fnmatch
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, AsyncIterator


@dataclass
class _Message:
    """Internal message structure for pub/sub."""

    type: str  # "message", "pmessage", "subscribe", "psubscribe"
    channel: str
    pattern: str | None  # Only set for pmessage
    data: str


@dataclass
class _Subscription:
    """Tracks a single subscription (channel or pattern)."""

    is_pattern: bool
    value: str  # channel name or pattern
    queue: asyncio.Queue[_Message]


class InMemoryPubSub:
    """
    In-memory pub/sub subscription context.

    Usage:
        async with broker.pubsub() as pubsub:
            await pubsub.psubscribe("app:comms:*")
            msg = await pubsub.get_message(timeout=1.0, ignore_subscribe_messages=True)
    """

    def __init__(self, broker: "InMemoryEventBroker"):
        self._broker = broker
        self._subscriptions: list[_Subscription] = []
        self._message_queue: asyncio.Queue[_Message] = asyncio.Queue()
        self._closed = False

    async def subscribe(self, *channels: str) -> None:
        """Subscribe to one or more channels."""
        for channel in channels:
            sub = _Subscription(
                is_pattern=False,
                value=channel,
                queue=self._message_queue,
            )
            self._subscriptions.append(sub)
            self._broker._add_subscription(sub)

            # Queue a subscribe confirmation message
            await self._message_queue.put(
                _Message(
                    type="subscribe",
                    channel=channel,
                    pattern=None,
                    data=str(len(self._subscriptions)),
                ),
            )

    async def psubscribe(self, *patterns: str) -> None:
        """Subscribe to one or more channel patterns (glob-style)."""
        for pattern in patterns:
            sub = _Subscription(
                is_pattern=True,
                value=pattern,
                queue=self._message_queue,
            )
            self._subscriptions.append(sub)
            self._broker._add_subscription(sub)

            # Queue a psubscribe confirmation message
            await self._message_queue.put(
                _Message(
                    type="psubscribe",
                    channel=pattern,
                    pattern=None,
                    data=str(len(self._subscriptions)),
                ),
            )

    async def unsubscribe(self, *channels: str) -> None:
        """Unsubscribe from channels. If no channels specified, unsubscribe from all."""
        if not channels:
            # Unsubscribe from all
            for sub in self._subscriptions:
                self._broker._remove_subscription(sub)
            self._subscriptions.clear()
        else:
            for channel in channels:
                for sub in self._subscriptions[:]:
                    if not sub.is_pattern and sub.value == channel:
                        self._broker._remove_subscription(sub)
                        self._subscriptions.remove(sub)

    async def punsubscribe(self, *patterns: str) -> None:
        """Unsubscribe from patterns. If no patterns specified, unsubscribe from all."""
        if not patterns:
            for sub in self._subscriptions[:]:
                if sub.is_pattern:
                    self._broker._remove_subscription(sub)
                    self._subscriptions.remove(sub)
        else:
            for pattern in patterns:
                for sub in self._subscriptions[:]:
                    if sub.is_pattern and sub.value == pattern:
                        self._broker._remove_subscription(sub)
                        self._subscriptions.remove(sub)

    async def get_message(
        self,
        *,
        timeout: float | None = None,
        ignore_subscribe_messages: bool = False,
    ) -> dict[str, Any] | None:
        """
        Get the next message from subscribed channels.

        Args:
            timeout: Maximum time to wait (seconds). None = no timeout.
            ignore_subscribe_messages: If True, skip subscribe/psubscribe confirmations.

        Returns:
            Message dict with keys: type, channel, pattern (if pmessage), data.
            Returns None on timeout.
        """
        if self._closed:
            return None

        while True:
            try:
                if timeout is not None:
                    msg = await asyncio.wait_for(
                        self._message_queue.get(),
                        timeout=timeout,
                    )
                else:
                    msg = await self._message_queue.get()

                # Skip subscription confirmations if requested
                if ignore_subscribe_messages and msg.type in (
                    "subscribe",
                    "psubscribe",
                    "unsubscribe",
                    "punsubscribe",
                ):
                    continue

                # Return message as dict
                result = {
                    "type": msg.type,
                    "channel": msg.channel,
                    "data": msg.data,
                }
                if msg.pattern is not None:
                    result["pattern"] = msg.pattern
                return result

            except asyncio.TimeoutError:
                return None
            except asyncio.CancelledError:
                raise

    async def aclose(self) -> None:
        """Close the pubsub connection and clean up subscriptions."""
        self._closed = True
        for sub in self._subscriptions:
            self._broker._remove_subscription(sub)
        self._subscriptions.clear()


class InMemoryEventBroker:
    """
    In-memory event broker for async publish/subscribe.

    Messages are passed by reference (no serialization) for efficiency.

    Usage:
        broker = InMemoryEventBroker()

        # Publishing
        await broker.publish("app:comms:startup", event.to_json())

        # Subscribing
        async with broker.pubsub() as pubsub:
            await pubsub.psubscribe("app:comms:*")
            while True:
                msg = await pubsub.get_message(timeout=1.0)
                if msg:
                    process(msg)
    """

    def __init__(self):
        self._subscriptions: list[_Subscription] = []
        self._lock = asyncio.Lock()
        self._closed = False

    def _add_subscription(self, sub: _Subscription) -> None:
        """Register a subscription (called by InMemoryPubSub)."""
        self._subscriptions.append(sub)

    def _remove_subscription(self, sub: _Subscription) -> None:
        """Unregister a subscription (called by InMemoryPubSub)."""
        if sub in self._subscriptions:
            self._subscriptions.remove(sub)

    async def publish(self, channel: str, message: str) -> int:
        """
        Publish a message to a channel.

        Args:
            channel: The channel name (e.g., "app:comms:startup")
            message: The message payload (typically JSON string)

        Returns:
            Number of subscribers that received the message.
        """
        if self._closed:
            return 0

        receivers = 0

        for sub in self._subscriptions[:]:  # Copy to avoid mutation during iteration
            matched = False
            msg_type = "message"
            pattern = None

            if sub.is_pattern:
                # Pattern matching (glob-style)
                if fnmatch.fnmatch(channel, sub.value):
                    matched = True
                    msg_type = "pmessage"
                    pattern = sub.value
            else:
                # Exact channel match
                if sub.value == channel:
                    matched = True

            if matched:
                try:
                    await sub.queue.put(
                        _Message(
                            type=msg_type,
                            channel=channel,
                            pattern=pattern,
                            data=message,
                        ),
                    )
                    receivers += 1
                except Exception:
                    # Queue might be closed/full - skip this subscriber
                    pass

        return receivers

    @asynccontextmanager
    async def pubsub(self) -> AsyncIterator[InMemoryPubSub]:
        """
        Create a pub/sub context for subscribing to channels.

        Usage:
            async with broker.pubsub() as pubsub:
                await pubsub.psubscribe("app:*")
                msg = await pubsub.get_message(timeout=1.0)
        """
        ps = InMemoryPubSub(self)
        try:
            yield ps
        finally:
            await ps.aclose()

    async def aclose(self) -> None:
        """Close the broker and all subscriptions."""
        self._closed = True
        self._subscriptions.clear()

    async def execute_command(self, *args) -> Any:
        """
        Execute a command (limited support for test compatibility).

        Currently supports:
            PUBSUB NUMPAT - returns number of pattern subscriptions
        """
        if len(args) >= 2 and args[0] == "PUBSUB" and args[1] == "NUMPAT":
            return sum(1 for sub in self._subscriptions if sub.is_pattern)
        raise NotImplementedError(f"Command not supported: {args}")


# Global singleton instance
_broker: InMemoryEventBroker | None = None


def get_in_memory_event_broker() -> InMemoryEventBroker:
    """Get or create the global in-memory event broker singleton."""
    global _broker
    if _broker is None:
        _broker = InMemoryEventBroker()
    return _broker


def create_in_memory_event_broker() -> InMemoryEventBroker:
    """Create a new in-memory event broker instance (for testing)."""
    return InMemoryEventBroker()


def reset_in_memory_event_broker() -> None:
    """Reset the global singleton (useful for testing)."""
    global _broker
    if _broker is not None:
        # Don't await aclose since this might be called from sync context
        _broker._closed = True
        _broker._subscriptions.clear()
    _broker = None
