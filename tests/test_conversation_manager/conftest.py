"""
tests/test_conversation_manager/conftest.py
==============================================

Fixtures for conversation manager integration tests.

Sets up:
- Redis server (default port 6379)
- Conversation manager process
- Event capture utilities
"""

import asyncio
import os
import pytest
import pytest_asyncio
import subprocess
import time
from pathlib import Path
from typing import List, Type
import redis.asyncio as redis

from unity.conversation_manager_2.new_events import Event, StartupEvent


# ============================================================================
# Redis Server Management
# ============================================================================


@pytest.fixture(scope="module")
def redis_server():
    """
    Start a Redis server on the default port (6379).

    Note: Make sure no other Redis instance is running on this port.
    """
    # Start redis-server (uses default port 6379)
    redis_proc = subprocess.Popen(
        [
            "redis-server",
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
            test_client = redis_sync.Redis(decode_responses=False)
            test_client.ping()
            test_client.close()
            print(f"✓ Redis server started")
            break
        except (redis_sync.ConnectionError, redis_sync.ResponseError):
            if i == 49:
                redis_proc.kill()
                raise RuntimeError("Redis failed to start")
            time.sleep(0.1)

    yield

    # Cleanup: Stop Redis
    print(f"\n✓ Stopping Redis server")
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
    client = redis.Redis()
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
async def event_capture(test_redis_client):
    """EventCapture instance that listens to all conversation manager events."""
    capture = EventCapture(test_redis_client)
    await capture.start_capturing(["app:comms:*", "app:conductor:*", "app:managers:*"])
    yield capture
    await capture.stop()


# ============================================================================
# Conversation Manager Process
# ============================================================================


@pytest_asyncio.fixture
async def conversation_manager_process(redis_server):
    """Start the conversation manager as a background process."""
    test_env = os.environ.copy()
    test_env.update(
        {
            "JOB_NAME": "test_job",
            "UNIFY_TRACED": "true",
            "UNIFY_CACHE": "true",
            "TEST": "true",
        },
    )

    # Start conversation manager (logs go to terminal)
    repo_root = Path(__file__).parent.parent.parent
    cm_proc = subprocess.Popen(
        ["python", "start.py"],
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

    print(f"✓ Conversation manager started (PID: {cm_proc.pid})")

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


@pytest_asyncio.fixture
async def initialized_system(
    conversation_manager_process,
    test_redis_client,
    event_capture,
):
    """
    Convenience fixture: fully initialized system with startup event published.

    Returns dict with: redis_client, event_capture, cm_process
    """

    startup = StartupEvent(
        api_key=os.getenv("UNIFY_KEY", "test_key"),
        medium="test",
        assistant_id="test_assistant_1",
        user_id="test_user_1",
        assistant_name="Test Assistant",
        assistant_age="25",
        assistant_region="US",
        assistant_about="A helpful test assistant",
        assistant_number="+15555551234",
        assistant_email="assistant@test.com",
        user_name="Test User",
        user_number="+15555555678",
        user_whatsapp_number="+15555555678",
        user_email="user@test.com",
        voice_provider="cartesia",
        voice_id="test_voice",
    )

    await test_redis_client.publish("app:comms:startup", startup.to_json())
    await asyncio.sleep(1)  # Let it initialize

    return {
        "redis_client": test_redis_client,
        "event_capture": event_capture,
        "cm_process": conversation_manager_process,
    }
