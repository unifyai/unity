"""
tests/test_conversation_manager/conftest.py
==============================================

Fixtures for conversation manager integration tests.

Uses **direct handler testing** pattern (same as ContactManager tests):
- No event-driven initialization (no background task dependencies)
- Direct calls to event handlers
- Direct state inspection
- Works reliably with pytest-asyncio

The tests use simulated implementations for all managers (ContactManager,
TranscriptManager, TaskScheduler, etc.) to avoid connecting to real backends.
"""

from __future__ import annotations

import asyncio
import os
import pytest
from datetime import datetime, timezone
from typing import TYPE_CHECKING, List, Type

import pytest_asyncio

from unity.conversation_manager.events import Event

if TYPE_CHECKING:
    from unity.conversation_manager.conversation_manager import ConversationManager
    from unity.conversation_manager.in_memory_event_broker import InMemoryEventBroker


# Fixed datetime for LLM cache consistency - must match tests/conftest.py
_FIXED_DATETIME = datetime(2025, 6, 13, 12, 0, 0, tzinfo=timezone.utc)

# Test contacts used across all tests
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


# =============================================================================
# Module-level setup: Configure environment for in-process mode
# =============================================================================


def pytest_configure(config):
    """Configure environment variables before any tests run."""
    # Use simulated implementations for all managers
    os.environ["UNITY_ACTOR_IMPL"] = "simulated"
    os.environ["UNITY_CONTACT_IMPL"] = "simulated"
    os.environ["UNITY_TRANSCRIPT_IMPL"] = "simulated"
    os.environ["UNITY_TASK_IMPL"] = "simulated"
    os.environ["UNITY_CONVERSATION_IMPL"] = "simulated"

    # Steps for SimulatedActor - 3 allows for pause+resume interactions
    os.environ["UNITY_ACTOR_SIMULATED_STEPS"] = "3"

    # Disable optional managers that might connect to real backends
    os.environ["UNITY_MEMORY_ENABLED"] = "false"
    os.environ["UNITY_KNOWLEDGE_ENABLED"] = "false"
    os.environ["UNITY_GUIDANCE_ENABLED"] = "false"
    os.environ["UNITY_SECRET_ENABLED"] = "false"
    os.environ["UNITY_SKILL_ENABLED"] = "false"
    os.environ["UNITY_WEB_ENABLED"] = "false"
    os.environ["UNITY_FILE_ENABLED"] = "false"

    # Fixed datetime for LLM cache consistency
    os.environ["UNITY_FIXED_DATETIME"] = _FIXED_DATETIME.isoformat()

    # Mark as test mode
    os.environ["TEST"] = "true"
    os.environ["UNITY_CONVERSATION_JOB_NAME"] = "test_job"


# =============================================================================
# ConversationManager Fixtures (Direct Handler Testing)
# =============================================================================


@pytest_asyncio.fixture(scope="module")
async def conversation_manager() -> "ConversationManager":
    """
    Start and initialize ConversationManager in-process for the test module.

    Uses DIRECT initialization (not event-driven) to avoid background task
    issues with pytest-asyncio. This follows the same pattern as ContactManager
    tests - direct method calls, not event publishing.
    """
    from unity.conversation_manager.event_broker import reset_event_broker
    from unity.conversation_manager import start_async, stop_async
    from unity.conversation_manager.domains import managers_utils

    # Reset any existing event broker state
    reset_event_broker()

    print("\n✓ Starting ConversationManager in-process...")
    cm = await start_async(
        project_name="TestProject",
        enable_comms_manager=False,  # Don't start CommsManager (requires GCP)
        apply_test_mocks=True,
    )
    print("✓ ConversationManager started (in-process mode)")
    print("  Using simulated implementations for all managers")

    # Initialize managers DIRECTLY (not via event handler)
    # This avoids the background task / event loop interleaving issues
    print("⏳ Initializing managers directly...")
    await managers_utils.init_conv_manager(cm)
    print("✅ Managers initialized")

    # Set test contacts directly on contact_index
    cm.contact_index.set_contacts(TEST_CONTACTS)
    print(f"✅ Test contacts set: {len(TEST_CONTACTS)}")

    yield cm

    # Cleanup
    print("\n✓ Stopping ConversationManager...")
    await stop_async()
    reset_event_broker()


@pytest.fixture
def initialized_cm(
    conversation_manager: "ConversationManager",
) -> "ConversationManager":
    """
    Per-test fixture that provides a clean ConversationManager.

    Clears conversation state between tests for isolation while reusing
    the expensive module-scoped CM instance.
    """
    # Clear any conversation state from previous tests
    conversation_manager.contact_index.clear_conversations()
    return conversation_manager


# =============================================================================
# Event Capture Helper (for tests that need event inspection)
# =============================================================================


class EventCapture:
    """
    Captures events from the in-memory event broker for test assertions.

    Used by tests that need to verify specific events were published.
    """

    def __init__(self, event_broker: "InMemoryEventBroker"):
        self._broker = event_broker
        self._captured_events: List[Event] = []
        self._pubsub = None
        self._capture_task = None
        self._running = False

    async def start_capturing(self, patterns: List[str]):
        """Start capturing events matching the given patterns."""
        self._pubsub = await self._broker.pubsub().__aenter__()
        await self._pubsub.psubscribe(*patterns)
        self._running = True
        self._capture_task = asyncio.create_task(self._capture_loop())

    async def _capture_loop(self):
        """Background task that captures all published events."""
        while self._running:
            try:
                msg = await self._pubsub.get_message(
                    timeout=0.1,
                    ignore_subscribe_messages=True,
                )
                if msg and msg["type"] == "pmessage":
                    try:
                        event = Event.from_json(msg["data"])
                        self._captured_events.append(event)
                    except Exception:
                        pass  # Skip unparseable events
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception:
                if self._running:
                    break

    def get_events(
        self,
        event_type: Type[Event] | None = None,
        **attributes,
    ) -> List[Event]:
        """Get all captured events, optionally filtered."""
        events = self._captured_events
        if event_type:
            events = [e for e in events if isinstance(e, event_type)]
        if attributes:
            events = [
                e
                for e in events
                if all(getattr(e, k, None) == v for k, v in attributes.items())
            ]
        return events

    def clear(self):
        """Clear all captured events."""
        self._captured_events.clear()

    async def stop(self):
        """Stop capturing events."""
        self._running = False
        if self._capture_task:
            self._capture_task.cancel()
            try:
                await self._capture_task
            except asyncio.CancelledError:
                pass
        if self._pubsub:
            await self._pubsub.aclose()


@pytest_asyncio.fixture
async def event_capture(initialized_cm: "ConversationManager") -> EventCapture:
    """
    EventCapture instance that listens to all conversation manager events.

    Uses the in-memory event broker directly - no Redis needed.
    """
    capture = EventCapture(initialized_cm.event_broker)
    await capture.start_capturing(["app:comms:*", "app:actor:*", "app:managers:*"])
    yield capture
    await capture.stop()


# =============================================================================
# Convenience Fixtures
# =============================================================================


@pytest.fixture
def event_broker(initialized_cm: "ConversationManager") -> "InMemoryEventBroker":
    """
    Direct access to the in-memory event broker.

    Useful for tests that need to publish/subscribe directly.
    """
    return initialized_cm.event_broker
