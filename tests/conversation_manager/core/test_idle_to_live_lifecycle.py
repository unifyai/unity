"""
tests/conversation_manager/core/test_idle_to_live_lifecycle.py
==============================================================

Integration tests for the idle → live container lifecycle.

These tests verify the critical invariant that the ConversationManager event
loop can process events at ALL times — including the idle state before any
assistant identity is known and before manager initialization.

WHY THESE TESTS MATTER:
-----------------------
In production, Unity containers start in an idle state: no assistant ID,
no API key, no user details. The container must stay alive by processing
Ping keepalives while waiting for a StartupEvent from the adapter. Manager
initialization must ONLY be triggered by StartupEvent (production) or
directly in main.py (local dev).

Commit b3814ca2 broke this by adding `await wait_for_initialization()` to
the event loop for all non-SessionConfig events. In the idle state,
`cm.initialized` is False and never becomes True (no StartupEvent has
arrived), so the event loop deadlocks. Even Ping keepalives couldn't get
through, causing the container to hit the inactivity timeout and die.

Commit 95c2243b similarly blocked inbound calls by adding per-handler
`wait_for_initialization()` calls.

These tests would have caught those bugs immediately:
- test_idle_container_processes_pings: Pings blocked → timeout → failure
- test_event_loop_does_not_block_sms_on_initialization: SMS blocked → failure
- test_startup_event_triggers_initialization: Init in wrong place → failure
- test_full_idle_to_live_flow: Full lifecycle broken → failure
- test_act_before_initialization_queued_not_blocking: act() blocks → failure
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio

from unity.conversation_manager.in_memory_event_broker import (
    create_in_memory_event_broker,
    reset_in_memory_event_broker,
)

# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_startup_event_kwargs() -> dict:
    """Minimal kwargs for constructing a StartupEvent."""
    return dict(
        api_key="test_api_key",
        medium="sms",
        assistant_id="test_assistant_42",
        user_id="test_user_123",
        assistant_first_name="Test",
        assistant_surname="Assistant",
        assistant_age="25",
        assistant_nationality="American",
        assistant_about="A test assistant for lifecycle tests",
        assistant_number="+15555550000",
        assistant_email="assistant@test.com",
        user_first_name="Boss",
        user_surname="User",
        user_number="+15555550001",
        user_email="boss@test.com",
        voice_provider="cartesia",
        voice_id="test_voice",
    )


def _make_idle_cm(event_broker):
    """Create a ConversationManager in idle state (uninitialized, no assistant)."""
    from unity.conversation_manager.conversation_manager import ConversationManager

    stop = asyncio.Event()
    cm = ConversationManager(
        event_broker=event_broker,
        job_name="test_idle_job",
        user_id="",
        assistant_id="0",
        user_first_name="",
        user_surname="",
        assistant_first_name="",
        assistant_surname="",
        assistant_age="",
        assistant_nationality="",
        assistant_about="",
        assistant_number="",
        assistant_email="",
        user_number="",
        user_email="",
        voice_provider="",
        voice_id="",
        project_name="TestProject",
        stop=stop,
    )
    return cm


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest_asyncio.fixture
async def event_broker():
    """Isolated in-memory event broker for each test."""
    reset_in_memory_event_broker()
    broker = create_in_memory_event_broker()
    yield broker
    await broker.aclose()
    reset_in_memory_event_broker()


# ── Tests ────────────────────────────────────────────────────────────────────


class TestIdleContainerEventProcessing:
    """
    Tests that verify the event loop works in the idle state.

    The idle state is the default state of any newly created container.
    No assistant identity is known, cm.initialized is False, and manager
    initialization has NOT been triggered. The event loop must still
    process events (particularly Ping keepalives) without blocking.
    """

    @pytest.mark.asyncio
    async def test_idle_container_processes_pings(self, event_broker):
        """
        Ping events must be processed by an uninitialized CM without blocking.

        If anyone adds wait_for_initialization() or any init gate to the
        event loop for non-config events, this test will deadlock/timeout.
        This directly reproduces the bug from commit b3814ca2.
        """
        from unity.conversation_manager.events import Ping

        cm = _make_idle_cm(event_broker)

        # Verify precondition: CM is NOT initialized
        assert not cm.initialized, "CM should start uninitialized"

        # Start the event loop in the background
        loop_task = asyncio.create_task(cm.wait_for_events())

        try:
            # Give the event loop a moment to subscribe
            await asyncio.sleep(0.1)

            # Publish a Ping event
            ping = Ping(kind="keepalive")
            await event_broker.publish("app:comms:ping", ping.to_json())

            # The event loop should process this immediately. We verify by
            # checking that last_activity_time was updated (Ping processing
            # updates it via the event loop's msg handler).
            initial_activity = cm.last_activity_time

            # Publish another ping after a brief delay
            await asyncio.sleep(0.15)
            await event_broker.publish(
                "app:comms:ping",
                Ping(kind="keepalive").to_json(),
            )

            # Wait briefly for the event to be processed
            await asyncio.sleep(0.15)

            # last_activity_time should have advanced
            assert cm.last_activity_time > initial_activity, (
                "Event loop did not process Ping event. "
                "The event loop is likely blocked on initialization. "
                "This is the exact bug from commit b3814ca2."
            )
        finally:
            loop_task.cancel()
            try:
                await loop_task
            except asyncio.CancelledError:
                pass

    @pytest.mark.asyncio
    async def test_event_loop_does_not_block_sms_on_initialization(self, event_broker):
        """
        SMS events must be processable by an uninitialized CM.

        The handler should not block waiting for initialization. Manager-
        dependent work (like logging to TranscriptManager) is deferred
        via queue_operation, but the event itself must flow through.
        """
        from unity.conversation_manager.events import SMSReceived

        cm = _make_idle_cm(event_broker)

        # Set up minimal mocks so the SMSReceived handler doesn't crash
        # on missing attributes that aren't relevant to this test.
        cm.contact_index.set_fallback_contacts(
            [
                {
                    "contact_id": 1,
                    "first_name": "Boss",
                    "surname": "User",
                    "phone_number": "+15555550001",
                    "email_address": "boss@test.com",
                },
            ],
        )
        cm.cancel_proactive_speech = AsyncMock()
        cm.request_llm_run = AsyncMock()

        assert not cm.initialized

        loop_task = asyncio.create_task(cm.wait_for_events())

        try:
            await asyncio.sleep(0.1)

            sms = SMSReceived(
                contact={
                    "contact_id": 1,
                    "first_name": "Boss",
                    "surname": "User",
                    "phone_number": "+15555550001",
                    "email_address": "boss@test.com",
                },
                content="Hello from idle state!",
            )
            await event_broker.publish("app:comms:msg_message", sms.to_json())

            # Wait for event processing (should be near-instant)
            await asyncio.sleep(0.3)

            # The SMS should have been added to the contact index
            from unity.conversation_manager.types import Medium

            msgs = cm.contact_index.get_messages_for_contact(1, Medium.SMS_MESSAGE)
            assert len(msgs) >= 1, (
                "SMSReceived was not processed by the event loop. "
                "The event loop is likely blocked on initialization."
            )
            assert msgs[0].content == "Hello from idle state!"
        finally:
            loop_task.cancel()
            try:
                await loop_task
            except asyncio.CancelledError:
                pass

    @pytest.mark.asyncio
    async def test_multiple_pings_processed_while_idle(self, event_broker):
        """
        Multiple pings should all be processed while the CM stays idle.

        This simulates the production scenario where an idle container
        sends keepalive pings every 30 seconds to avoid inactivity timeout.
        """
        from unity.conversation_manager.events import Ping

        cm = _make_idle_cm(event_broker)
        assert not cm.initialized

        loop_task = asyncio.create_task(cm.wait_for_events())

        try:
            await asyncio.sleep(0.1)

            # Send 5 pings in succession
            for i in range(5):
                await event_broker.publish(
                    "app:comms:ping",
                    Ping(kind="keepalive").to_json(),
                )
                await asyncio.sleep(0.05)

            # Wait for all to be processed
            await asyncio.sleep(0.3)

            # CM should still be uninitialized (pings don't trigger init)
            assert not cm.initialized, (
                "Pings should not trigger initialization. "
                "Initialization must only come from StartupEvent."
            )
        finally:
            loop_task.cancel()
            try:
                await loop_task
            except asyncio.CancelledError:
                pass


class TestStartupTriggersInitialization:
    """
    Tests that verify StartupEvent correctly triggers manager initialization.

    StartupEvent is the ONLY event that should trigger initialization in
    production (in local dev, main.py triggers it directly).
    """

    @pytest.mark.asyncio
    async def test_startup_event_triggers_initialization(self, event_broker):
        """
        StartupEvent should trigger manager initialization.

        We mock init_conv_manager to verify it's called and to flip
        cm.initialized without needing real Orchestra/Unify backends.
        """
        from unity.conversation_manager.events import StartupEvent
        from unity.conversation_manager.domains import managers_utils

        cm = _make_idle_cm(event_broker)
        assert not cm.initialized

        init_called = asyncio.Event()

        async def mock_init_conv_manager(cm_arg, **kwargs):
            """Mock that simulates successful initialization."""
            cm_arg.initialized = True
            init_called.set()

        async def mock_listen_to_operations(cm_arg):
            """Mock that does nothing (no real operations to process)."""

        async def mock_startup_sequence(cm_arg):
            """Mock that does nothing (no real job logging)."""

        loop_task = asyncio.create_task(cm.wait_for_events())

        try:
            await asyncio.sleep(0.1)

            with (
                patch.object(
                    managers_utils,
                    "init_conv_manager",
                    mock_init_conv_manager,
                ),
                patch.object(
                    managers_utils,
                    "listen_to_operations",
                    mock_listen_to_operations,
                ),
                patch(
                    "unity.conversation_manager.domains.event_handlers._startup_sequence",
                    mock_startup_sequence,
                ),
            ):
                startup = StartupEvent(**_make_startup_event_kwargs())
                await event_broker.publish("app:comms:startup", startup.to_json())

                # Wait for initialization to be triggered
                try:
                    await asyncio.wait_for(init_called.wait(), timeout=3.0)
                except asyncio.TimeoutError:
                    pytest.fail(
                        "init_conv_manager was not called after StartupEvent. "
                        "The StartupEvent handler may be broken.",
                    )

            assert cm.initialized, "CM should be initialized after StartupEvent."
        finally:
            loop_task.cancel()
            try:
                await loop_task
            except asyncio.CancelledError:
                pass


class TestFullIdleToLiveFlow:
    """
    End-to-end test of the idle → startup → live lifecycle.

    This simulates the production flow:
    1. Container starts idle (no assistant)
    2. Pings keep it alive
    3. StartupEvent arrives with assistant details
    4. Managers initialize
    5. Inbound SMS arrives and is handled
    """

    @pytest.mark.asyncio
    async def test_full_idle_to_live_flow(self, event_broker):
        """
        Full lifecycle: idle pings → startup → init → inbound SMS.

        This is the "does our system hang forever" guardrail test.
        """
        from unity.conversation_manager.events import (
            Ping,
            StartupEvent,
            SMSReceived,
            InitializationComplete,
        )
        from unity.conversation_manager.domains import managers_utils

        cm = _make_idle_cm(event_broker)
        assert not cm.initialized

        # We need to mock request_llm_run and cancel_proactive_speech since
        # they have side effects that aren't relevant to this test.
        cm.cancel_proactive_speech = AsyncMock()
        cm.request_llm_run = AsyncMock()

        init_complete = asyncio.Event()

        async def mock_init_conv_manager(cm_arg, **kwargs):
            """Simulates initialization: sets up fallback contacts and flips flag."""
            cm_arg.initialized = True
            # Publish InitializationComplete like real init does
            await event_broker.publish(
                "app:comms:initialization_complete",
                InitializationComplete().to_json(),
            )
            init_complete.set()

        async def mock_listen_to_operations(cm_arg):
            pass

        async def mock_startup_sequence(cm_arg):
            pass

        loop_task = asyncio.create_task(cm.wait_for_events())

        try:
            await asyncio.sleep(0.1)

            # ── Phase 1: Idle state — pings should work ──
            await event_broker.publish(
                "app:comms:ping",
                Ping(kind="keepalive").to_json(),
            )
            await asyncio.sleep(0.15)
            assert not cm.initialized, "Should still be idle after pings"

            # ── Phase 2: StartupEvent arrives ──
            with (
                patch.object(
                    managers_utils,
                    "init_conv_manager",
                    mock_init_conv_manager,
                ),
                patch.object(
                    managers_utils,
                    "listen_to_operations",
                    mock_listen_to_operations,
                ),
                patch(
                    "unity.conversation_manager.domains.event_handlers._startup_sequence",
                    mock_startup_sequence,
                ),
            ):
                startup = StartupEvent(**_make_startup_event_kwargs())
                await event_broker.publish("app:comms:startup", startup.to_json())

                try:
                    await asyncio.wait_for(init_complete.wait(), timeout=3.0)
                except asyncio.TimeoutError:
                    pytest.fail("Initialization did not complete after StartupEvent")

            assert cm.initialized, "CM should be initialized after StartupEvent"

            # ── Phase 3: Inbound SMS after initialization ──
            # Set up fallback contacts so SMS handler can resolve the sender
            cm.contact_index.set_fallback_contacts(
                [
                    {
                        "contact_id": 1,
                        "first_name": "Boss",
                        "surname": "User",
                        "phone_number": "+15555550001",
                        "email_address": "boss@test.com",
                    },
                ],
            )

            sms = SMSReceived(
                contact={
                    "contact_id": 1,
                    "first_name": "Boss",
                    "surname": "User",
                    "phone_number": "+15555550001",
                    "email_address": "boss@test.com",
                },
                content="Hey, I just hired you!",
            )
            await event_broker.publish("app:comms:msg_message", sms.to_json())

            # Wait for SMS to be processed
            await asyncio.sleep(0.3)

            from unity.conversation_manager.types import Medium

            msgs = cm.contact_index.get_messages_for_contact(1, Medium.SMS_MESSAGE)
            assert len(msgs) >= 1, (
                "Inbound SMS was not handled after initialization. "
                "The idle → live transition may be broken."
            )
            assert msgs[0].content == "Hey, I just hired you!"

        finally:
            loop_task.cancel()
            try:
                await loop_task
            except asyncio.CancelledError:
                pass


class TestActBeforeInitialization:
    """
    Tests that act() in brain_action_tools correctly queues work instead
    of blocking on initialization.

    Commit e7b882a6 added wait_for_initialization() in act(), which would
    block the slow brain's tool call. Ved's fix (1b367a5c) changed this to
    queue_operation() so act() returns immediately and the actual actor
    invocation runs after initialization completes.
    """

    @pytest.mark.asyncio
    async def test_act_before_initialization_queued_not_blocking(self, event_broker):
        """
        Calling act() on an uninitialized CM should return immediately
        by queueing the actor invocation, not blocking on initialization.

        If act() blocks on wait_for_initialization, this test will timeout.
        """
        from unity.conversation_manager.domains import managers_utils
        from unity.conversation_manager.domains.brain_action_tools import (
            ConversationManagerBrainActionTools,
        )

        # Drain any leftover operations from other tests
        while not managers_utils._operations_queue.empty():
            try:
                managers_utils._operations_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

        mock_cm = MagicMock()
        mock_cm.initialized = False
        mock_cm.in_flight_actions = {}
        mock_cm._current_state_snapshot = None
        mock_cm._current_snapshot_state = None

        with patch(
            "unity.conversation_manager.domains.brain_action_tools.get_event_broker",
        ) as mock_get_broker:
            mock_broker = MagicMock()
            mock_broker.publish = AsyncMock()
            mock_get_broker.return_value = mock_broker

            tools = ConversationManagerBrainActionTools(mock_cm)

            # act() should return within 2 seconds (not block on init)
            try:
                result = await asyncio.wait_for(
                    tools.act(query="look up the weather", requesting_contact_id=1),
                    timeout=2.0,
                )
            except asyncio.TimeoutError:
                pytest.fail(
                    "act() blocked for >2s, likely waiting on initialization. "
                    "act() should queue the work via queue_operation and "
                    "return immediately. This is the bug from commit e7b882a6.",
                )

        assert result["status"] == "acting"

        # The actor.act() should NOT have been called yet (queued for later)
        mock_cm.actor.act.assert_not_called()

        # The operation should be in the queue
        assert not managers_utils._operations_queue.empty(), (
            "The actor invocation should be queued via queue_operation, "
            "not executed synchronously before initialization."
        )

        # Clean up the queue
        while not managers_utils._operations_queue.empty():
            try:
                managers_utils._operations_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
