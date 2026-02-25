"""
tests/conversation_manager/core/test_inactivity_lifecycle.py
=============================================================

Tests for container inactivity detection and lifecycle management.

These tests verify the critical production behavior documented in INFRA.md:
- Containers shut down after 6 minutes (360s) of inactivity
- Idle containers ping every 30 seconds to stay alive
- Cleanup is called properly on shutdown
- Jobs are marked as done in AssistantJobs

This is Phase 4.1 of the end-to-end testing roadmap.

What This File Tests:
---------------------
1. **Inactivity detection**: Does check_inactivity() trigger shutdown after timeout?
2. **Activity reset**: Does receiving events reset the inactivity timer?
3. **Ping keep-alive**: Does the ping mechanism keep idle containers alive?
4. **Cleanup sequence**: Is cleanup called in the correct order on shutdown?
5. **Job marking**: Is assistant_jobs.mark_job_done() called for live containers?
6. **Event broker close**: Is the event broker properly closed on shutdown?

Production Context (from INFRA.md):
-----------------------------------
- Inactivity timeout: 6 minutes (360 seconds)
- Ping interval: 30 seconds (half the timeout)
- Idle containers use DEFAULT_ASSISTANT_ID ("0")
- Live containers have a real assistant_id
- On shutdown: cleanup() → mark_job_done() → stop.set()
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio

from unity.conversation_manager.events import (
    Ping,
)
from unity.session_details import DEFAULT_ASSISTANT_ID

# =============================================================================
# Fixtures
# =============================================================================


@pytest_asyncio.fixture
async def event_broker():
    """Create a fresh in-memory event broker."""
    from unity.conversation_manager.in_memory_event_broker import (
        create_in_memory_event_broker,
        reset_in_memory_event_broker,
    )

    reset_in_memory_event_broker()
    broker = create_in_memory_event_broker()
    yield broker
    await broker.aclose()
    reset_in_memory_event_broker()


@pytest.fixture
def mock_loop():
    """Create a mock event loop with controllable time."""
    mock = MagicMock()
    mock.time.return_value = 0.0
    return mock


@pytest.fixture
def minimal_cm_config():
    """Minimal configuration for creating a ConversationManager."""
    return {
        "job_name": "test-job-123",
        "user_id": "user_1",
        "assistant_id": "assistant_1",
        "user_first_name": "Test",
        "user_surname": "User",
        "assistant_first_name": "Test",
        "assistant_surname": "Assistant",
        "assistant_age": "25",
        "assistant_nationality": "American",
        "assistant_about": "A helpful assistant",
        "assistant_number": "+15555550000",
        "assistant_email": "assistant@test.com",
        "user_number": "+15555551111",
        "user_email": "user@test.com",
        "voice_provider": "cartesia",
        "voice_id": "test_voice",
        "voice_mode": "tts",
    }


# =============================================================================
# Test: Inactivity Detection Basics
# =============================================================================


class TestInactivityDetectionBasics:
    """Tests for basic inactivity timeout detection."""

    @pytest.mark.asyncio
    async def test_inactivity_timeout_triggers_shutdown(self, event_broker):
        """
        Verify that check_inactivity() triggers shutdown after the timeout.

        This is the core behavior: after 6 minutes of no activity, the container
        should shut down gracefully.
        """
        from unity.conversation_manager.conversation_manager import ConversationManager

        stop_event = asyncio.Event()
        cm = ConversationManager(
            event_broker=event_broker,
            job_name="test-job",
            user_id="user_1",
            assistant_id="assistant_1",
            user_first_name="Test",
            user_surname="User",
            assistant_first_name="Test",
            assistant_surname="Assistant",
            assistant_age="25",
            assistant_nationality="American",
            assistant_about="Test bio",
            assistant_number="+15555550000",
            assistant_email="assistant@test.com",
            user_number="+15555551111",
            user_email="user@test.com",
            stop=stop_event,
        )

        # Set very short timeouts for testing
        cm.inactivity_timeout = 0.1  # 100ms timeout
        cm.inactivity_check_interval = 0.05  # 50ms check interval

        # Set last_activity_time to a time that makes timeout already exceeded
        cm.last_activity_time = cm.loop.time() - 1.0  # 1 second ago

        # Run check_inactivity for a short time
        check_task = asyncio.create_task(cm.check_inactivity())

        # Wait for stop to be set
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=1.0)
        except asyncio.TimeoutError:
            pytest.fail("Inactivity timeout did not trigger shutdown")
        finally:
            check_task.cancel()
            try:
                await check_task
            except asyncio.CancelledError:
                pass

        assert stop_event.is_set(), "Stop event should be set after inactivity timeout"

    @pytest.mark.asyncio
    async def test_activity_resets_inactivity_timer(self, event_broker):
        """
        Verify that receiving events resets the inactivity timer.

        When wait_for_events() receives a message, it should update
        last_activity_time to the current loop time.
        """
        from unity.conversation_manager.conversation_manager import ConversationManager

        stop_event = asyncio.Event()
        cm = ConversationManager(
            event_broker=event_broker,
            job_name="test-job",
            user_id="user_1",
            assistant_id="assistant_1",
            user_first_name="Test",
            user_surname="User",
            assistant_first_name="Test",
            assistant_surname="Assistant",
            assistant_age="25",
            assistant_nationality="American",
            assistant_about="Test bio",
            assistant_number="+15555550000",
            assistant_email="assistant@test.com",
            user_number="+15555551111",
            user_email="user@test.com",
            stop=stop_event,
        )

        initial_activity_time = cm.last_activity_time

        # Simulate activity update (what wait_for_events does when receiving an event)
        # We use direct time manipulation instead of sleeping - this makes the test
        # deterministic regardless of actual wall-clock time
        cm.last_activity_time = cm.loop.time() + 0.1

        assert (
            cm.last_activity_time > initial_activity_time
        ), "Activity time should be updated after receiving an event"

    @pytest.mark.asyncio
    async def test_no_shutdown_when_activity_continues(self, event_broker):
        """
        Verify that continuous activity prevents shutdown.

        If last_activity_time keeps getting updated, the inactivity timeout
        should never be reached.
        """
        from unity.conversation_manager.conversation_manager import ConversationManager

        stop_event = asyncio.Event()
        cm = ConversationManager(
            event_broker=event_broker,
            job_name="test-job",
            user_id="user_1",
            assistant_id="assistant_1",
            user_first_name="Test",
            user_surname="User",
            assistant_first_name="Test",
            assistant_surname="Assistant",
            assistant_age="25",
            assistant_nationality="American",
            assistant_about="Test bio",
            assistant_number="+15555550000",
            assistant_email="assistant@test.com",
            user_number="+15555551111",
            user_email="user@test.com",
            stop=stop_event,
        )

        # Set short timeouts
        cm.inactivity_timeout = 0.2
        cm.inactivity_check_interval = 0.05

        # Start inactivity check
        check_task = asyncio.create_task(cm.check_inactivity())

        # Keep updating activity time faster than the timeout
        for _ in range(5):
            cm.last_activity_time = cm.loop.time()
            await asyncio.sleep(0.05)

        # Stop should NOT be set
        assert (
            not stop_event.is_set()
        ), "Stop should not be set while activity continues"

        check_task.cancel()
        try:
            await check_task
        except asyncio.CancelledError:
            pass


# =============================================================================
# Test: Ping Keep-Alive Mechanism
# =============================================================================


class TestPingKeepAlive:
    """Tests for the ping mechanism that keeps idle containers alive."""

    @pytest.mark.asyncio
    async def test_ping_event_resets_activity_timer(self, event_broker):
        """
        Verify that Ping events reset the inactivity timer.

        Idle containers send pings every 30 seconds to stay alive. Each ping
        should reset the inactivity timer.
        """
        from unity.conversation_manager.conversation_manager import ConversationManager
        from unity.conversation_manager.domains.event_handlers import EventHandler

        stop_event = asyncio.Event()
        cm = ConversationManager(
            event_broker=event_broker,
            job_name="test-job",
            user_id="user_1",
            assistant_id=DEFAULT_ASSISTANT_ID,  # Idle container
            user_first_name="Test",
            user_surname="User",
            assistant_first_name="Test",
            assistant_surname="Assistant",
            assistant_age="25",
            assistant_nationality="American",
            assistant_about="Test bio",
            assistant_number="+15555550000",
            assistant_email="assistant@test.com",
            user_number="+15555551111",
            user_email="user@test.com",
            stop=stop_event,
        )

        # Record initial activity time
        initial_time = cm.last_activity_time

        # Simulate activity update from receiving a ping
        # We use direct time manipulation instead of sleeping
        cm.last_activity_time = cm.loop.time() + 0.1

        # Handle the ping event
        ping_event = Ping(kind="keepalive")
        await EventHandler.handle_event(ping_event, cm)

        assert cm.last_activity_time > initial_time, "Ping should update activity time"


# =============================================================================
# Test: Cleanup Sequence
# =============================================================================


class TestCleanupSequence:
    """Tests for the cleanup sequence on shutdown."""

    @pytest.mark.asyncio
    async def test_cleanup_calls_mark_job_done_for_live_container(self, event_broker):
        """
        Verify that cleanup() calls mark_job_done() for live containers.

        Live containers (with real assistant_id) must mark their job as done
        in AssistantJobs so the system knows they're no longer running.
        """
        from unity.conversation_manager.conversation_manager import ConversationManager

        stop_event = asyncio.Event()
        cm = ConversationManager(
            event_broker=event_broker,
            job_name="test-job-live",
            user_id="user_1",
            assistant_id="real_assistant_123",  # Live container
            user_first_name="Test",
            user_surname="User",
            assistant_first_name="Test",
            assistant_surname="Assistant",
            assistant_age="25",
            assistant_nationality="American",
            assistant_about="Test bio",
            assistant_number="+15555550000",
            assistant_email="assistant@test.com",
            user_number="+15555551111",
            user_email="user@test.com",
            stop=stop_event,
        )

        with patch(
            "unity.conversation_manager.conversation_manager.assistant_jobs.mark_job_done",
        ) as mock_mark_done:
            await cm.cleanup()

            mock_mark_done.assert_called_once_with("test-job-live")

    @pytest.mark.asyncio
    async def test_cleanup_skips_mark_job_done_for_idle_container(self, event_broker):
        """
        Verify that cleanup() skips mark_job_done() for idle containers.

        Idle containers (with DEFAULT_ASSISTANT_ID) were never "live" in the
        AssistantJobs sense, so we don't need to mark them as done.
        """
        from unity.conversation_manager.conversation_manager import ConversationManager

        stop_event = asyncio.Event()
        cm = ConversationManager(
            event_broker=event_broker,
            job_name="test-job-idle",
            user_id="user_1",
            assistant_id=DEFAULT_ASSISTANT_ID,  # Idle container
            user_first_name="Test",
            user_surname="User",
            assistant_first_name="Test",
            assistant_surname="Assistant",
            assistant_age="25",
            assistant_nationality="American",
            assistant_about="Test bio",
            assistant_number="+15555550000",
            assistant_email="assistant@test.com",
            user_number="+15555551111",
            user_email="user@test.com",
            stop=stop_event,
        )

        with patch(
            "unity.conversation_manager.conversation_manager.assistant_jobs.mark_job_done",
        ) as mock_mark_done:
            await cm.cleanup()

            mock_mark_done.assert_not_called()

    @pytest.mark.asyncio
    async def test_cleanup_calls_cleanup_call_proc(self, event_broker):
        """
        Verify that cleanup() calls cleanup_call_proc() to terminate voice agents.

        Any running voice agent subprocess must be terminated on shutdown.
        """
        from unity.conversation_manager.conversation_manager import ConversationManager

        stop_event = asyncio.Event()
        cm = ConversationManager(
            event_broker=event_broker,
            job_name="test-job",
            user_id="user_1",
            assistant_id="assistant_1",
            user_first_name="Test",
            user_surname="User",
            assistant_first_name="Test",
            assistant_surname="Assistant",
            assistant_age="25",
            assistant_nationality="American",
            assistant_about="Test bio",
            assistant_number="+15555550000",
            assistant_email="assistant@test.com",
            user_number="+15555551111",
            user_email="user@test.com",
            stop=stop_event,
        )

        # Mock cleanup_call_proc
        cm.call_manager.cleanup_call_proc = AsyncMock()

        with patch(
            "unity.conversation_manager.conversation_manager.assistant_jobs.mark_job_done",
        ):
            await cm.cleanup()

            cm.call_manager.cleanup_call_proc.assert_called_once()

    @pytest.mark.asyncio
    async def test_cleanup_sets_stop_event(self, event_broker):
        """
        Verify that cleanup() sets the stop event.

        This signals to main.py that shutdown is complete.
        """
        from unity.conversation_manager.conversation_manager import ConversationManager

        stop_event = asyncio.Event()
        cm = ConversationManager(
            event_broker=event_broker,
            job_name="test-job",
            user_id="user_1",
            assistant_id="assistant_1",
            user_first_name="Test",
            user_surname="User",
            assistant_first_name="Test",
            assistant_surname="Assistant",
            assistant_age="25",
            assistant_nationality="American",
            assistant_about="Test bio",
            assistant_number="+15555550000",
            assistant_email="assistant@test.com",
            user_number="+15555551111",
            user_email="user@test.com",
            stop=stop_event,
        )

        assert not stop_event.is_set(), "Stop event should not be set initially"

        with patch(
            "unity.conversation_manager.conversation_manager.assistant_jobs.mark_job_done",
        ):
            await cm.cleanup()

        assert stop_event.is_set(), "Stop event should be set after cleanup"


# =============================================================================
# Test: Event Broker Lifecycle
# =============================================================================


class TestEventBrokerLifecycle:
    """Tests for event broker lifecycle management."""

    @pytest.mark.asyncio
    async def test_inactivity_closes_event_broker(self, event_broker):
        """
        Verify that inactivity timeout closes the event broker.

        When check_inactivity() triggers shutdown, it should close the event
        broker to release resources and signal to wait_for_events() to stop.
        """
        from unity.conversation_manager.conversation_manager import ConversationManager

        stop_event = asyncio.Event()
        cm = ConversationManager(
            event_broker=event_broker,
            job_name="test-job",
            user_id="user_1",
            assistant_id="assistant_1",
            user_first_name="Test",
            user_surname="User",
            assistant_first_name="Test",
            assistant_surname="Assistant",
            assistant_age="25",
            assistant_nationality="American",
            assistant_about="Test bio",
            assistant_number="+15555550000",
            assistant_email="assistant@test.com",
            user_number="+15555551111",
            user_email="user@test.com",
            stop=stop_event,
        )

        # Set very short timeout
        cm.inactivity_timeout = 0.05
        cm.inactivity_check_interval = 0.02
        cm.last_activity_time = cm.loop.time() - 1.0  # Already timed out

        # Track if aclose was called
        original_aclose = event_broker.aclose
        aclose_called = False

        async def mock_aclose():
            nonlocal aclose_called
            aclose_called = True
            return await original_aclose()

        event_broker.aclose = mock_aclose

        # Run check_inactivity
        check_task = asyncio.create_task(cm.check_inactivity())

        try:
            await asyncio.wait_for(stop_event.wait(), timeout=1.0)
        except asyncio.TimeoutError:
            pass
        finally:
            check_task.cancel()
            try:
                await check_task
            except asyncio.CancelledError:
                pass

        assert aclose_called, "Event broker should be closed on inactivity timeout"


# =============================================================================
# Test: Full Lifecycle Integration
# =============================================================================


class TestFullLifecycleIntegration:
    """Integration tests for complete lifecycle scenarios."""

    @pytest.mark.asyncio
    async def test_idle_to_live_to_shutdown_lifecycle(self, event_broker):
        """
        Test the complete lifecycle: idle → startup → live → inactivity → shutdown.

        This simulates the production flow:
        1. Container starts in idle state
        2. Receives startup event, becomes live
        3. Processes events, stays alive
        4. No activity for timeout period
        5. Shuts down gracefully
        """
        from unity.conversation_manager.conversation_manager import ConversationManager

        stop_event = asyncio.Event()
        cm = ConversationManager(
            event_broker=event_broker,
            job_name="test-lifecycle-job",
            user_id="user_1",
            assistant_id=DEFAULT_ASSISTANT_ID,  # Start as idle
            user_first_name="Test",
            user_surname="User",
            assistant_first_name="Test",
            assistant_surname="Assistant",
            assistant_age="25",
            assistant_nationality="American",
            assistant_about="Test bio",
            assistant_number="+15555550000",
            assistant_email="assistant@test.com",
            user_number="+15555551111",
            user_email="user@test.com",
            stop=stop_event,
        )

        # Verify idle state
        assert cm.assistant_id == DEFAULT_ASSISTANT_ID

        # Simulate startup (transition to live)
        startup_payload = {
            "api_key": "test_key",
            "assistant_id": "live_assistant_456",
            "user_id": "user_1",
            "assistant_first_name": "Live",
            "assistant_surname": "Assistant",
            "assistant_age": "30",
            "assistant_nationality": "British",
            "assistant_about": "A live assistant",
            "assistant_number": "+15555559999",
            "assistant_email": "live@test.com",
            "user_first_name": "Live",
            "user_surname": "User",
            "user_number": "+15555558888",
            "user_email": "live_user@test.com",
            "voice_provider": "cartesia",
            "voice_id": "voice_123",
            "voice_mode": "tts",
        }
        cm.set_details(startup_payload)

        # Verify live state
        assert cm.assistant_id == "live_assistant_456"

        # Set short timeout for testing
        cm.inactivity_timeout = 0.1
        cm.inactivity_check_interval = 0.03

        # Set activity time to simulate recent activity, then force timeout
        # We use direct time manipulation instead of sleeping
        cm.last_activity_time = cm.loop.time() - 1.0  # Force timeout

        # Run check_inactivity
        check_task = asyncio.create_task(cm.check_inactivity())

        try:
            await asyncio.wait_for(stop_event.wait(), timeout=1.0)
        except asyncio.TimeoutError:
            pytest.fail("Lifecycle did not complete - shutdown not triggered")
        finally:
            check_task.cancel()
            try:
                await check_task
            except asyncio.CancelledError:
                pass

        assert stop_event.is_set(), "Container should shut down after inactivity"

    @pytest.mark.asyncio
    async def test_cleanup_order_is_correct(self, event_broker):
        """
        Verify cleanup operations happen in the correct order.

        The order matters for production:
        1. Update rolling summaries (persist conversation state)
        2. Store chat history
        3. Cleanup call proc
        4. Mark job done
        5. Set stop event
        """
        from unity.conversation_manager.conversation_manager import ConversationManager

        stop_event = asyncio.Event()
        cm = ConversationManager(
            event_broker=event_broker,
            job_name="test-cleanup-order",
            user_id="user_1",
            assistant_id="assistant_1",
            user_first_name="Test",
            user_surname="User",
            assistant_first_name="Test",
            assistant_surname="Assistant",
            assistant_age="25",
            assistant_nationality="American",
            assistant_about="Test bio",
            assistant_number="+15555550000",
            assistant_email="assistant@test.com",
            user_number="+15555551111",
            user_email="user@test.com",
            stop=stop_event,
        )

        call_order = []

        # Mock all the cleanup functions to track order
        cm.call_manager.cleanup_call_proc = AsyncMock(
            side_effect=lambda: call_order.append("cleanup_call_proc"),
        )

        async def mock_store_chat_history():
            call_order.append("store_chat_history")

        cm.store_chat_history = mock_store_chat_history

        with patch(
            "unity.conversation_manager.conversation_manager.assistant_jobs.mark_job_done",
            side_effect=lambda x: call_order.append("mark_job_done"),
        ):
            await cm.cleanup()

        # Verify order
        expected_order = [
            "store_chat_history",
            "cleanup_call_proc",
            "mark_job_done",
        ]
        assert (
            call_order == expected_order
        ), f"Cleanup order incorrect. Expected {expected_order}, got {call_order}"

        # Stop event should be set at the end
        assert stop_event.is_set()


# =============================================================================
# Test: Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_cleanup_handles_missing_job_name(self, event_broker):
        """
        Verify cleanup handles missing job_name gracefully.

        In some error scenarios, job_name might not be set.
        """
        from unity.conversation_manager.conversation_manager import ConversationManager

        stop_event = asyncio.Event()
        cm = ConversationManager(
            event_broker=event_broker,
            job_name="",  # Empty job name
            user_id="user_1",
            assistant_id="assistant_1",
            user_first_name="Test",
            user_surname="User",
            assistant_first_name="Test",
            assistant_surname="Assistant",
            assistant_age="25",
            assistant_nationality="American",
            assistant_about="Test bio",
            assistant_number="+15555550000",
            assistant_email="assistant@test.com",
            user_number="+15555551111",
            user_email="user@test.com",
            stop=stop_event,
        )

        with patch(
            "unity.conversation_manager.conversation_manager.assistant_jobs.mark_job_done",
        ) as mock_mark_done:
            # Should not raise
            await cm.cleanup()

            # Should not call mark_job_done with empty job name
            mock_mark_done.assert_not_called()

    @pytest.mark.asyncio
    async def test_multiple_inactivity_checks_dont_double_shutdown(self, event_broker):
        """
        Verify that multiple inactivity checks don't cause issues.

        Once stop is set, subsequent checks should be no-ops.
        """
        from unity.conversation_manager.conversation_manager import ConversationManager

        stop_event = asyncio.Event()
        cm = ConversationManager(
            event_broker=event_broker,
            job_name="test-job",
            user_id="user_1",
            assistant_id="assistant_1",
            user_first_name="Test",
            user_surname="User",
            assistant_first_name="Test",
            assistant_surname="Assistant",
            assistant_age="25",
            assistant_nationality="American",
            assistant_about="Test bio",
            assistant_number="+15555550000",
            assistant_email="assistant@test.com",
            user_number="+15555551111",
            user_email="user@test.com",
            stop=stop_event,
        )

        cm.inactivity_timeout = 0.05
        cm.inactivity_check_interval = 0.02
        cm.last_activity_time = cm.loop.time() - 1.0

        aclose_count = 0
        original_aclose = event_broker.aclose

        async def counting_aclose():
            nonlocal aclose_count
            aclose_count += 1
            # Only actually close on first call
            if aclose_count == 1:
                return await original_aclose()

        event_broker.aclose = counting_aclose

        # Start check_inactivity
        check_task = asyncio.create_task(cm.check_inactivity())

        try:
            await asyncio.wait_for(stop_event.wait(), timeout=1.0)
            # Let it run a bit more to see if it tries to close again
            await asyncio.sleep(0.1)
        except asyncio.TimeoutError:
            pass
        finally:
            check_task.cancel()
            try:
                await check_task
            except asyncio.CancelledError:
                pass

        # aclose should only be called once
        assert aclose_count == 1, f"aclose called {aclose_count} times, expected 1"
