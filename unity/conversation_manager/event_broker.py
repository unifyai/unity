"""
Event broker for ConversationManager.

Provides in-memory pub/sub for event-driven communication between components.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from unity.conversation_manager.in_memory_event_broker import (
    InMemoryEventBroker,
    get_in_memory_event_broker,
    create_in_memory_event_broker,
    reset_in_memory_event_broker,
)

if TYPE_CHECKING:
    EventBroker = InMemoryEventBroker


_broker: InMemoryEventBroker | None = None


def get_event_broker() -> InMemoryEventBroker:
    """
    Get the global event broker singleton.

    Returns the shared InMemoryEventBroker instance.
    """
    global _broker

    if _broker is None:
        _broker = get_in_memory_event_broker()

    return _broker


def create_event_broker() -> InMemoryEventBroker:
    """
    Create and return a new event broker instance.

    Use this when you need a separate broker for testing or isolation.
    """
    return create_in_memory_event_broker()


def reset_event_broker() -> None:
    """
    Reset the global event broker singleton.

    Useful for testing to ensure a fresh broker between tests.
    """
    global _broker

    if _broker is not None:
        reset_in_memory_event_broker()

    _broker = None
