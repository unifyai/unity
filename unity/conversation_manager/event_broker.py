"""
Event broker factory for ConversationManager.

Supports two backends:
- "redis": Redis pub/sub (default for production deployments)
- "in_memory": In-memory event broker (for testing and in-process operation)

The backend is selected via UNITY_EVENT_BROKER environment variable or setting.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Union

import redis.asyncio as redis

from unity.conversation_manager.in_memory_event_broker import (
    InMemoryEventBroker,
    get_in_memory_event_broker,
    create_in_memory_event_broker,
)

if TYPE_CHECKING:
    # Type alias for either broker type
    EventBroker = Union[redis.Redis, InMemoryEventBroker]


_broker: "EventBroker | None" = None


def _get_redis_port() -> int:
    """Get Redis port from environment or default to 6379."""
    return int(os.environ.get("REDIS_PORT", "6379"))


def _get_broker_type() -> str:
    """
    Get the event broker type from settings.

    Returns "redis" or "in_memory" based on UNITY_EVENT_BROKER setting.
    Defaults to "redis" for backward compatibility.
    """
    # Import here to avoid circular imports
    from unity.settings import SETTINGS

    return getattr(SETTINGS, "UNITY_EVENT_BROKER", "redis")


def _create_redis_broker() -> redis.Redis:
    """Create a Redis event broker."""
    return redis.Redis(
        host="localhost",
        port=_get_redis_port(),
        decode_responses=True,
    )


def get_event_broker() -> "EventBroker":
    """
    Get the global event broker singleton.

    Returns either a Redis client or InMemoryEventBroker based on
    the UNITY_EVENT_BROKER setting.
    """
    global _broker

    if _broker is None:
        broker_type = _get_broker_type()
        if broker_type == "in_memory":
            _broker = get_in_memory_event_broker()
        else:
            _broker = _create_redis_broker()

    return _broker


def create_event_broker() -> "EventBroker":
    """
    Create and return a new event broker instance.

    Use this when you need a separate client for a different asyncio
    event loop or thread. Returns the appropriate type based on
    UNITY_EVENT_BROKER setting.
    """
    broker_type = _get_broker_type()
    if broker_type == "in_memory":
        return create_in_memory_event_broker()
    else:
        return _create_redis_broker()


def reset_event_broker() -> None:
    """
    Reset the global event broker singleton.

    Useful for testing to ensure a fresh broker between tests.
    """
    global _broker

    if _broker is not None:
        # For in-memory broker, also reset its internal state
        if isinstance(_broker, InMemoryEventBroker):
            from unity.conversation_manager.in_memory_event_broker import (
                reset_in_memory_event_broker,
            )

            reset_in_memory_event_broker()

    _broker = None
