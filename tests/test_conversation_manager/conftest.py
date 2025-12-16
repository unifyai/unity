"""
tests/test_conversation_manager/conftest.py
==============================================

Fixtures for conversation manager integration tests.

Uses **in-process mode** for simple, fast testing:
- No Redis server required
- No subprocess spawning
- Direct access to ConversationManager instance
- Direct monkey-patching support

The tests use simulated implementations for all managers (ContactManager,
TranscriptManager, TaskScheduler, etc.) to avoid connecting to real backends.
"""

from __future__ import annotations

import asyncio
import os
import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING, List, Type

import pytest_asyncio

from unity.conversation_manager.events import (
    Event,
    GetContactsResponse,
    StartupEvent,
)

if TYPE_CHECKING:
    from unity.conversation_manager.conversation_manager import ConversationManager
    from unity.conversation_manager.in_memory_event_broker import InMemoryEventBroker


# Fixed datetime for LLM cache consistency - must match tests/conftest.py
_FIXED_DATETIME = datetime(2025, 6, 13, 12, 0, 0, tzinfo=timezone.utc)


# =============================================================================
# Module-level setup: Configure environment for in-process mode
# =============================================================================


def pytest_configure(config):
    """Configure environment variables before any tests run."""
    # Use simulated implementations for all managers
    os.environ["UNITY_ACTOR_IMPL"] = "simulated"
    os.environ["UNITY_CONTACTS_IMPL"] = "simulated"
    os.environ["UNITY_TRANSCRIPTS_IMPL"] = "simulated"
    os.environ["UNITY_TASKS_IMPL"] = "simulated"
    os.environ["UNITY_CONVERSATION_IMPL"] = "simulated"
    os.environ["UNITY_CONDUCTOR_IMPL"] = "simulated"

    # Steps for SimulatedActor - 3 allows for pause+resume interactions
    os.environ["UNITY_SIMULATED_ACTOR_STEPS"] = "3"

    # Disable optional managers that might connect to real backends
    os.environ["UNITY_KNOWLEDGE_ENABLED"] = "false"
    os.environ["UNITY_GUIDANCE_ENABLED"] = "false"
    os.environ["UNITY_SECRETS_ENABLED"] = "false"
    os.environ["UNITY_SKILLS_ENABLED"] = "false"
    os.environ["UNITY_WEB_SEARCH_ENABLED"] = "false"
    os.environ["UNITY_FILES_ENABLED"] = "false"

    # Fixed datetime for LLM cache consistency
    os.environ["UNITY_FIXED_DATETIME"] = _FIXED_DATETIME.isoformat()

    # Mark as test mode
    os.environ["TEST"] = "true"
    os.environ["JOB_NAME"] = "test_job"


# =============================================================================
# Event Capture Helper
# =============================================================================


class EventCapture:
    """
    Captures events from the in-memory event broker for test assertions.

    Much simpler than the Redis-based version since we're in the same process.
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

    async def wait_for_event(
        self,
        event_type: Type[Event],
        timeout: float = 30.0,
        **attributes,
    ) -> Event:
        """Wait for a specific event type with optional attribute matching."""
        start = time.perf_counter()
        while time.perf_counter() - start < timeout:
            for event in self._captured_events:
                if isinstance(event, event_type):
                    if all(getattr(event, k, None) == v for k, v in attributes.items()):
                        return event
            await asyncio.sleep(0.05)

        raise TimeoutError(
            f"Timeout waiting for {event_type.__name__} with {attributes}",
        )

    async def wait_for_event_with_matcher(
        self,
        event_type: Type[Event],
        matcher: callable,
        timeout: float = 30.0,
    ) -> Event:
        """Wait for a specific event type that matches a custom matcher function."""
        start = time.perf_counter()
        while time.perf_counter() - start < timeout:
            for event in self._captured_events:
                if isinstance(event, event_type) and matcher(event):
                    return event
            await asyncio.sleep(0.05)

        raise TimeoutError(
            f"Timeout waiting for {event_type.__name__} matching custom criteria",
        )

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


# =============================================================================
# ConversationManager Fixtures (In-Process Mode)
# =============================================================================


@pytest_asyncio.fixture(scope="module")
async def conversation_manager() -> "ConversationManager":
    """
    Start ConversationManager in-process for the test module.

    This is much simpler than the subprocess approach:
    - No Redis server needed
    - No subprocess spawning
    - Direct access to the CM instance
    - Direct monkey-patching support
    """
    # Reset any existing event broker state
    from unity.conversation_manager.event_broker import reset_event_broker

    reset_event_broker()

    # Import and start CM in-process
    from unity.conversation_manager import start_async, stop_async

    print("\n✓ Starting ConversationManager in-process...")
    cm = await start_async(
        project_name="TestProject",
        enable_comms_manager=False,  # Don't start CommsManager (requires GCP)
        apply_test_mocks=True,
    )
    print(f"✓ ConversationManager started (in-process mode)")
    print("  Using simulated implementations for all managers")

    yield cm

    # Cleanup
    print("\n✓ Stopping ConversationManager...")
    await stop_async()
    reset_event_broker()


@pytest_asyncio.fixture(scope="module")
async def initialized_conversation_manager(
    conversation_manager: "ConversationManager",
) -> "ConversationManager":
    """
    Initialize the ConversationManager with startup and contacts events.

    Publishes the startup event and contacts, then waits for initialization
    to complete. Much faster than subprocess mode since we're in-process.
    """
    cm = conversation_manager
    event_broker = cm.event_broker

    # Create startup event
    startup = StartupEvent(
        api_key=os.getenv("UNIFY_KEY", "test_key"),
        medium="test",
        assistant_id="test_assistant_1",
        user_id="test_user_1",
        assistant_name="Test Assistant",
        assistant_age="25",
        assistant_nationality="US",
        assistant_about="A helpful test assistant",
        assistant_number="+15555551234",
        assistant_email="assistant@test.com",
        user_name="Test User",
        user_number="+15555555678",
        user_email="user@test.com",
        voice_provider="cartesia",
        voice_id="test_voice",
    )

    # Subscribe to initialization_complete before sending startup
    async with event_broker.pubsub() as pubsub:
        await pubsub.subscribe("app:comms:initialization_complete")

        # Send startup event
        print("📤 Publishing startup event...")
        await event_broker.publish("app:comms:startup", startup.to_json())

        # Send contacts
        print("📤 Publishing contacts...")
        contacts_event = GetContactsResponse(
            contacts=[
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
            ],
        )
        await event_broker.publish("app:comms:contacts", contacts_event.to_json())

        # Wait for initialization (with timeout)
        print("⏳ Waiting for initialization...")
        init_timeout = 60  # Much shorter than subprocess mode
        start_time = time.perf_counter()

        while time.perf_counter() - start_time < init_timeout:
            msg = await pubsub.get_message(timeout=1.0, ignore_subscribe_messages=True)
            if msg and msg["type"] == "message":
                try:
                    from unity.conversation_manager.events import InitializationComplete

                    event = Event.from_json(msg["data"])
                    if isinstance(event, InitializationComplete):
                        elapsed = time.perf_counter() - start_time
                        print(f"✅ Initialization complete after {elapsed:.1f}s")
                        break
                except Exception:
                    pass
        else:
            raise RuntimeError(
                f"Timeout ({init_timeout}s) waiting for InitializationComplete",
            )

    print("✅ System initialized and ready")
    return cm


@pytest_asyncio.fixture
async def event_capture(
    initialized_conversation_manager: "ConversationManager",
) -> EventCapture:
    """
    EventCapture instance that listens to all conversation manager events.

    Uses the in-memory event broker directly - no Redis needed.
    """
    cm = initialized_conversation_manager
    capture = EventCapture(cm.event_broker)
    await capture.start_capturing(["app:comms:*", "app:conductor:*", "app:managers:*"])
    yield capture
    await capture.stop()


# =============================================================================
# Convenience Fixtures
# =============================================================================


@pytest_asyncio.fixture
async def event_broker(
    initialized_conversation_manager: "ConversationManager",
) -> "InMemoryEventBroker":
    """
    Direct access to the in-memory event broker.

    Useful for tests that need to publish/subscribe directly.
    """
    return initialized_conversation_manager.event_broker
