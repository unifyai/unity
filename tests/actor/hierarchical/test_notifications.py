"""
Tests for the notify() internal primitive in HierarchicalActor.

The notify() primitive allows generated plans to send progress notifications
to the supervising handle. It's an internal tool similar to request_clarification(),
defined via closure in _prepare_execution_environment().

These tests verify:
- notify() is properly injected into the execution namespace
- notify() pushes messages to the plan's _notification_q
- notify() logs messages to action_log
- Multiple notifications are queued in order
"""

import asyncio

import pytest

from unity.actor.hierarchical_actor import (
    HierarchicalActor,
    HierarchicalActorHandle,
)

from unittest.mock import AsyncMock


class SimpleMockVerificationClient:
    """Mock verification client that always returns success."""

    def __init__(self):
        # HierarchicalActorHandle expects `.generate` to be awaitable and return JSON.
        self.generate = AsyncMock(side_effect=self._side_effect)

    async def _side_effect(self, *args, **kwargs):
        _ = (args, kwargs)
        # Keep it minimal; HierarchicalActor parses this into a VerificationAssessment.
        return '{"status": "ok", "reason": "Mock verification success."}'


# ────────────────────────────────────────────────────────────────────────────
# Tests
# ────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_notify_is_injected_into_execution_namespace():
    """Verify that notify() is available in the plan's execution namespace."""
    # Create actor with computer_mode="mock" (uses MockComputerBackend automatically)
    actor = HierarchicalActor(
        headless=True,
        computer_mode="mock",
        connect_now=False,
    )

    active_task = None
    try:
        active_task = HierarchicalActorHandle(actor=actor, goal="Test notify injection")

        if active_task._execution_task:
            active_task._execution_task.cancel()
            try:
                await active_task._execution_task
            except asyncio.CancelledError:
                pass

        # Prepare the execution environment
        await actor._prepare_execution_environment(active_task)

        # Verify notify is in the namespace
        assert "notify" in active_task.execution_namespace
        assert callable(active_task.execution_namespace["notify"])

        print("✅ notify() is properly injected into execution namespace")

    finally:
        if active_task and not active_task.done():
            try:
                await active_task.stop()
            except Exception:
                pass
        if actor:
            try:
                await actor.close()
            except Exception:
                pass


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_notify_pushes_to_notification_queue():
    """Verify that notify() pushes messages to _notification_q."""
    # Create actor with computer_mode="mock" (uses MockComputerBackend automatically)
    actor = HierarchicalActor(
        headless=True,
        computer_mode="mock",
        connect_now=False,
    )

    active_task = None
    try:
        active_task = HierarchicalActorHandle(actor=actor, goal="Test notify queue")

        if active_task._execution_task:
            active_task._execution_task.cancel()
            try:
                await active_task._execution_task
            except asyncio.CancelledError:
                pass

        # Mock verification
        active_task.verification_client = SimpleMockVerificationClient()

        # Prepare and inject the plan
        await actor._prepare_execution_environment(active_task)

        # Get the notify function from the namespace
        notify_fn = active_task.execution_namespace["notify"]

        # Call notify directly
        await notify_fn("Test message 1")
        await notify_fn("Test message 2")

        # Verify messages are in the queue
        assert active_task._notification_q.qsize() == 2

        msg1 = active_task._notification_q.get_nowait()
        assert msg1["type"] == "notification"
        assert msg1["message"] == "Test message 1"

        msg2 = active_task._notification_q.get_nowait()
        assert msg2["type"] == "notification"
        assert msg2["message"] == "Test message 2"

        print("✅ notify() correctly pushes messages to _notification_q")

    finally:
        if active_task and not active_task.done():
            try:
                await active_task.stop()
            except Exception:
                pass
        if actor:
            try:
                await actor.close()
            except Exception:
                pass


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_notify_logs_to_action_log():
    """Verify that notify() logs messages to action_log."""
    # Create actor with computer_mode="mock" (uses MockComputerBackend automatically)
    actor = HierarchicalActor(
        headless=True,
        computer_mode="mock",
        connect_now=False,
    )

    active_task = None
    try:
        active_task = HierarchicalActorHandle(actor=actor, goal="Test notify logging")

        if active_task._execution_task:
            active_task._execution_task.cancel()
            try:
                await active_task._execution_task
            except asyncio.CancelledError:
                pass

        # Mock verification
        active_task.verification_client = SimpleMockVerificationClient()

        # Prepare and inject the plan
        await actor._prepare_execution_environment(active_task)

        # Get the notify function from the namespace
        notify_fn = active_task.execution_namespace["notify"]

        # Clear action log to isolate our test
        initial_log_len = len(active_task.action_log)

        # Call notify
        await notify_fn("Progress update: 50% complete")

        # Verify message is logged
        log_after = active_task.action_log[initial_log_len:]
        assert any("Progress update: 50% complete" in entry for entry in log_after)

        print("✅ notify() correctly logs messages to action_log")

    finally:
        if active_task and not active_task.done():
            try:
                await active_task.stop()
            except Exception:
                pass
        if actor:
            try:
                await actor.close()
            except Exception:
                pass


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_notify_multiple_messages_queued_in_order():
    """Verify that multiple notify() calls queue messages in the correct order."""
    # Create actor with computer_mode="mock" (uses MockComputerBackend automatically)
    actor = HierarchicalActor(
        headless=True,
        computer_mode="mock",
        connect_now=False,
    )

    active_task = None
    try:
        active_task = HierarchicalActorHandle(actor=actor, goal="Test notify ordering")

        if active_task._execution_task:
            active_task._execution_task.cancel()
            try:
                await active_task._execution_task
            except asyncio.CancelledError:
                pass

        # Mock verification
        active_task.verification_client = SimpleMockVerificationClient()

        # Prepare and inject the plan
        await actor._prepare_execution_environment(active_task)

        # Get the notify function from the namespace
        notify_fn = active_task.execution_namespace["notify"]

        # Send multiple notifications
        await notify_fn("Step 1: Starting...")
        await notify_fn("Step 2: Processing...")
        await notify_fn("Step 3: Complete!")

        # Verify all messages are in the queue in order
        assert active_task._notification_q.qsize() == 3

        msg1 = active_task._notification_q.get_nowait()
        assert msg1["message"] == "Step 1: Starting..."

        msg2 = active_task._notification_q.get_nowait()
        assert msg2["message"] == "Step 2: Processing..."

        msg3 = active_task._notification_q.get_nowait()
        assert msg3["message"] == "Step 3: Complete!"

        print("✅ notify() correctly queues multiple messages in order")

    finally:
        if active_task and not active_task.done():
            try:
                await active_task.stop()
            except Exception:
                pass
        if actor:
            try:
                await actor.close()
            except Exception:
                pass
