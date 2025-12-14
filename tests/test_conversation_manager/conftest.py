"""
tests/test_conversation_manager/conftest.py
==============================================

Fixtures for conversation manager integration tests.

Sets up:
- Redis server (dynamic port to support parallel test runs)
- Conversation manager process (using simulated implementations)
- Event capture utilities

IMPORTANT: These tests are compatible with `-t` per-test parallelism because
each test gets its own Redis server on a unique port.

The tests use simulated implementations for all managers (ContactManager,
TranscriptManager, TaskScheduler, etc.) to avoid connecting to real backends.
This is controlled via UNITY_*_IMPL environment variables.
"""

import asyncio
import os
import pytest
import pytest_asyncio
import socket
import subprocess
import time
from pathlib import Path
from typing import List, Type
import redis.asyncio as redis

from unity.conversation_manager.events import (
    Event,
    GetContactsResponse,
    StartupEvent,
)


def _find_free_port() -> int:
    """Find an available port by binding to port 0 and reading the assigned port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# ============================================================================
# Redis Server Management
# ============================================================================


@pytest.fixture(scope="module")
def redis_server():
    """
    Start a Redis server on a dynamically allocated port.

    This enables parallel test execution with `-t` flag since each test
    module gets its own Redis instance on a unique port.
    """
    redis_port = _find_free_port()
    redis_proc = subprocess.Popen(
        [
            "redis-server",
            "--port",
            str(redis_port),
            "--save",
            "",
            "--appendonly",
            "no",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # Wait for Redis to be ready
    for i in range(50):
        try:
            # Use sync Redis client for the health check
            import redis as redis_sync

            test_client = redis_sync.Redis(port=redis_port, decode_responses=False)
            test_client.ping()
            test_client.close()
            print(f"✓ Redis server started on port {redis_port}")
            break
        except (redis_sync.ConnectionError, redis_sync.ResponseError):
            if i == 49:
                redis_proc.kill()
                raise RuntimeError(f"Redis failed to start on port {redis_port}")
            time.sleep(0.1)

    yield redis_port

    # Cleanup: Stop Redis
    print(f"\n✓ Stopping Redis server (port {redis_port})")
    redis_proc.terminate()
    try:
        redis_proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        redis_proc.kill()
        redis_proc.wait()


# ============================================================================
# Test Redis Client
# ============================================================================


@pytest_asyncio.fixture
async def test_redis_client(redis_server):
    """Redis client for publishing events in tests."""
    client = redis.Redis(port=redis_server)
    yield client
    await client.aclose()


# ============================================================================
# Event Capture Helper
# ============================================================================


class EventCapture:
    """Captures events published to Redis for test assertions."""

    def __init__(self, redis_client: redis.Redis):
        self._client = redis_client
        self._captured_events: List[Event] = []
        self._pubsub = None
        self._capture_task = None
        self._running = False

    async def start_capturing(self, patterns: List[str]):
        """Start capturing events matching the given patterns."""
        self._pubsub = self._client.pubsub()
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

    def get_events(self, event_type: Type[Event] = None, **attributes) -> List[Event]:
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
            await self._pubsub.unsubscribe()
            await self._pubsub.aclose()


@pytest_asyncio.fixture
async def event_capture(redis_server):
    """
    EventCapture instance that listens to all conversation manager events.

    Creates its own Redis client to avoid event loop conflicts with
    module-scoped fixtures.
    """
    client = redis.Redis(port=redis_server)
    capture = EventCapture(client)
    await capture.start_capturing(["app:comms:*", "app:conductor:*", "app:managers:*"])
    yield capture
    await capture.stop()
    await client.aclose()


# ============================================================================
# Conversation Manager Process
# ============================================================================


@pytest_asyncio.fixture(scope="module")
async def conversation_manager_process(redis_server):
    """Start the conversation manager as a background process with simulated managers."""
    import sys

    test_env = os.environ.copy()
    test_env.update(
        {
            "JOB_NAME": "test_job",
            "UNIFY_CACHE": "true",
            "TEST": "true",
            "REDIS_PORT": str(redis_server),
            # Use simulated implementations for all managers
            "UNITY_ACTOR_IMPL": "simulated",
            "UNITY_CONTACTS_IMPL": "simulated",
            "UNITY_TRANSCRIPTS_IMPL": "simulated",
            "UNITY_TASKS_IMPL": "simulated",
            "UNITY_CONVERSATION_IMPL": "simulated",
            "UNITY_CONDUCTOR_IMPL": "simulated",
            # Disable optional managers that might connect to real backends
            "UNITY_KNOWLEDGE_ENABLED": "false",
            "UNITY_GUIDANCE_ENABLED": "false",
            "UNITY_SECRETS_ENABLED": "false",
            "UNITY_SKILLS_ENABLED": "false",
            "UNITY_WEB_SEARCH_ENABLED": "false",
            "UNITY_FILES_ENABLED": "false",
        },
    )

    # Start conversation manager (logs go to terminal)
    repo_root = Path(__file__).parent.parent.parent
    cm_proc = subprocess.Popen(
        [sys.executable, "start.py"],
        cwd=repo_root,
        env=test_env,
    )

    # Wait for it to start (simple wait)
    start_time = time.time()
    while time.time() - start_time < 10:
        if cm_proc.poll() is not None:
            raise RuntimeError("CM process died during startup")
        if time.time() - start_time > 3:
            break
        time.sleep(0.5)

    print(
        f"✓ Conversation manager started (PID: {cm_proc.pid}, Redis port: {redis_server})",
    )
    print("  Using simulated implementations for all managers")

    yield cm_proc

    # Cleanup
    print(f"\n✓ Stopping conversation manager (PID: {cm_proc.pid})")
    cm_proc.terminate()
    try:
        cm_proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        cm_proc.kill()
        cm_proc.wait()


# ============================================================================
# Initialized System (Convenience)
# ============================================================================


@pytest_asyncio.fixture(scope="module", autouse=True)
async def initialized_conversation_manager(conversation_manager_process, redis_server):
    """
    Initialize the conversation manager with startup and contacts events.

    This fixture is module-scoped and runs automatically (autouse=True) for all
    tests in this module, so tests don't need to explicitly request it.

    Waits for the CM to subscribe to Redis channels, then publishes startup
    and contacts events.

    Returns: cm_process
    """
    temp_client = redis.Redis(port=redis_server)

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

    # wait for the conversation manager to initialize
    print("Waiting for initialization to complete...")
    await asyncio.sleep(20)

    # Wait for CM to subscribe to channels by checking for active pattern subscriptions
    print("⏳ Waiting for conversation manager to subscribe to Redis channels...")
    max_wait = 30
    wait_interval = 0.5
    waited = 0
    num_patterns = 0
    while waited < max_wait:
        # Check if there are at least 1 active pattern subscription from CM's wait_for_events()
        num_patterns = await temp_client.execute_command("PUBSUB", "NUMPAT")
        if num_patterns >= 1:
            print(
                f"✅ Found {num_patterns} active pattern subscription(s) after {waited:.1f}s",
            )
            break
        await asyncio.sleep(wait_interval)
        waited += wait_interval
    else:
        print(
            f"⚠️  Expected pattern subscriptions after {max_wait}s, found {num_patterns}",
        )
        raise RuntimeError("Conversation manager did not subscribe to Redis channels")

    # Brief additional wait to ensure CM's get_message() loop is actively polling
    await asyncio.sleep(1)

    # Send startup event
    print("📤 Publishing startup event...")
    await temp_client.publish("app:comms:startup", startup.to_json())

    # Send contacts list
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
    await temp_client.publish("app:comms:contacts", contacts_event.to_json())

    print("✅ System initialized and ready")

    await temp_client.aclose()

    return conversation_manager_process
