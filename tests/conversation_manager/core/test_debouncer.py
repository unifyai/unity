"""
Tests for the Debouncer utility.

The Debouncer is a core utility that manages async task execution with
debouncing semantics. It supports:
- Pending task replacement (debouncing)
- Optional running task cancellation (cancel_running parameter)
- Task queuing (at most 1 running + 1 pending)

These tests verify the Debouncer's behavior independent of the
ConversationManager integration.

NOTE: Debouncer tests are inherently timing-sensitive since the debouncer
manages task timing. Some fixed sleeps are intentional to test the debounce
behavior (e.g., simulating LLM thinking time). Where possible, we use
event-based synchronization for task coordination.
"""

import asyncio
import time as _time

import pytest


async def _wait_for_condition(
    predicate,
    *,
    timeout: float = 5.0,
    poll: float = 0.02,
) -> bool:
    """Poll predicate() until True or timeout. Returns whether condition was met."""
    start = _time.perf_counter()
    while _time.perf_counter() - start < timeout:
        if predicate():
            return True
        await asyncio.sleep(poll)
    return False


class TestDebouncerCancelRunning:
    """Tests for the cancel_running parameter behavior."""

    @pytest.mark.asyncio
    async def test_preserves_running_task_when_cancel_running_false(self):
        """
        Verify that with cancel_running=False, the running task completes.

        This is the core behavior needed for voice mode:
        - When cancel_running=False, the currently running task must complete
        - New submissions replace the pending task, not the running one
        """
        from unity.conversation_manager.domains.utils import Debouncer

        debouncer = Debouncer()

        execution_log = []
        task1_started = asyncio.Event()
        task1_can_complete = asyncio.Event()

        async def task1():
            """First task - signals when started, waits for permission to complete."""
            execution_log.append("task1:started")
            task1_started.set()
            try:
                await asyncio.wait_for(task1_can_complete.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                execution_log.append("task1:timeout")
                return
            except asyncio.CancelledError:
                execution_log.append("task1:cancelled")
                raise
            execution_log.append("task1:completed")

        async def task2():
            """Second task."""
            execution_log.append("task2:started")
            await asyncio.sleep(0.1)
            execution_log.append("task2:completed")

        # Submit first task
        await debouncer.submit(task1, cancel_running=False)

        # Wait for first task to start
        await asyncio.wait_for(task1_started.wait(), timeout=2.0)

        # Now submit second task WITH cancel_running=False
        # This should NOT cancel task1
        await debouncer.submit(task2, cancel_running=False)

        # Wait for task2 to be queued (pending)
        await _wait_for_condition(lambda: debouncer.pending_task is not None)

        # Now allow task1 to complete
        task1_can_complete.set()

        # Wait for both tasks to complete (poll instead of fixed sleep)
        await _wait_for_condition(lambda: "task1:completed" in execution_log)
        await _wait_for_condition(lambda: "task2:completed" in execution_log)

        # KEY ASSERTION: Task 1 should have COMPLETED, not been cancelled
        assert "task1:completed" in execution_log, (
            f"Task 1 should have completed when cancel_running=False!\n"
            f"  Execution log: {execution_log}\n"
            f"\n"
            f"With cancel_running=False, the running task should be allowed\n"
            f"to complete. Only the pending task should be replaced."
        )

        assert "task1:cancelled" not in execution_log, (
            f"Task 1 was cancelled even though cancel_running=False!\n"
            f"  Execution log: {execution_log}"
        )

    @pytest.mark.asyncio
    async def test_cancels_running_task_when_cancel_running_true(self):
        """
        Verify that with cancel_running=True, the running task IS cancelled.

        This is useful for text mode where we want to cancel stale work
        and start fresh with new context.
        """
        from unity.conversation_manager.domains.utils import Debouncer

        debouncer = Debouncer()

        execution_log = []
        task1_started = asyncio.Event()

        async def task1():
            """First task - will be cancelled."""
            execution_log.append("task1:started")
            task1_started.set()
            try:
                await asyncio.sleep(5.0)  # Long sleep - should be cancelled
                execution_log.append("task1:completed")
            except asyncio.CancelledError:
                execution_log.append("task1:cancelled")
                raise

        async def task2():
            """Second task."""
            execution_log.append("task2:started")
            await asyncio.sleep(0.1)
            execution_log.append("task2:completed")

        # Submit first task
        await debouncer.submit(task1, cancel_running=False)

        # Wait for first task to start
        await asyncio.wait_for(task1_started.wait(), timeout=2.0)

        # Submit second task WITH cancel_running=True
        # This SHOULD cancel task1
        await debouncer.submit(task2, cancel_running=True)

        # Wait for task2 to complete (poll instead of fixed sleep)
        await _wait_for_condition(lambda: "task2:completed" in execution_log)

        # Task 1 should have been CANCELLED (not completed)
        assert "task1:cancelled" in execution_log, (
            f"Task 1 should have been cancelled when cancel_running=True!\n"
            f"  Execution log: {execution_log}"
        )

        # Task 2 should have run
        assert "task2:completed" in execution_log, (
            f"Task 2 should have completed!\n" f"  Execution log: {execution_log}"
        )


class TestDebouncerCancellationPropagation:
    """
    Tests for cancellation propagation behavior.

    In Python 3.11+, cancelling a task that is awaiting another task will
    also cancel the inner task. The Debouncer must use asyncio.shield()
    to protect running tasks from this propagation when a pending task
    is cancelled (debounced).
    """

    @pytest.mark.asyncio
    async def test_pending_cancellation_does_not_cancel_running_task(self):
        """
        REGRESSION TEST: Cancelling a pending task must NOT cancel the running task.

        This tests the asyncio.shield() fix for Python 3.11+ behavior where
        cancelling a task that awaits another task propagates the cancellation.

        Scenario:
        1. Task A starts running
        2. Task B is submitted (pending, waiting for A)
        3. Task C is submitted (cancels pending B)
        4. Task A must continue and complete (not be cancelled by B's cancellation)

        Before fix: Task A would be cancelled when Task B was cancelled
        After fix: Task A completes normally, Task C runs after
        """
        from unity.conversation_manager.domains.utils import Debouncer

        debouncer = Debouncer()
        results = []

        async def slow_task(task_id):
            """A task that takes 0.3s to complete."""
            try:
                results.append(f"task{task_id}:started")
                await asyncio.sleep(0.3)
                results.append(f"task{task_id}:completed")
            except asyncio.CancelledError:
                results.append(f"task{task_id}:cancelled")
                raise

        # Submit task 0 - it starts running
        await debouncer.submit(slow_task, args=(0,), cancel_running=False)
        # Wait for task 0 to start
        await _wait_for_condition(lambda: "task0:started" in results)

        # Submit task 1 - becomes pending, waiting for task 0
        await debouncer.submit(slow_task, args=(1,), cancel_running=False)
        # Wait for task 1 to be queued as pending
        await _wait_for_condition(lambda: debouncer.pending_task is not None)

        # Submit task 2 - this cancels pending task 1 (debounce)
        # The bug: task 1's cancellation would propagate to task 0
        await debouncer.submit(slow_task, args=(2,), cancel_running=False)

        # Wait for task 0 and task 2 to complete (poll instead of fixed sleep)
        await _wait_for_condition(
            lambda: "task0:completed" in results and "task2:completed" in results,
        )

        # Task 0 MUST have completed (not cancelled)
        assert "task0:completed" in results, (
            f"Task 0 should have completed!\n"
            f"  Results: {results}\n"
            f"\n"
            f"This regression indicates the asyncio.shield() fix is missing.\n"
            f"In Python 3.11+, cancelling a pending task that awaits the running\n"
            f"task will propagate the cancellation unless shield() is used."
        )

        assert "task0:cancelled" not in results, (
            f"Task 0 was cancelled by pending task cancellation!\n"
            f"  Results: {results}\n"
            f"\n"
            f"The Debouncer must use asyncio.shield() to protect the running\n"
            f"task from cancellation propagation when pending tasks are cancelled."
        )

        # Task 2 should also complete (it was the final pending task)
        assert "task2:completed" in results, (
            f"Task 2 should have completed!\n" f"  Results: {results}"
        )

    @pytest.mark.asyncio
    async def test_rapid_submissions_with_cancel_running_false(self):
        """
        Test rapid submissions don't cause unexpected cancellations.

        Simulates the voice mode scenario where rapid user utterances
        trigger many submissions in quick succession.
        """
        from unity.conversation_manager.domains.utils import Debouncer

        debouncer = Debouncer()
        results = []

        async def task(task_id):
            try:
                results.append(f"task{task_id}:started")
                await asyncio.sleep(0.5)  # Simulates LLM thinking time
                results.append(f"task{task_id}:completed")
            except asyncio.CancelledError:
                results.append(f"task{task_id}:cancelled")
                raise

        # Rapid submissions (faster than task completion time)
        # NOTE: The 0.1s delay between submissions is intentional - it simulates
        # rapid user utterances in voice mode (faster than LLM thinking time)
        for i in range(5):
            await debouncer.submit(task, args=(i,), cancel_running=False)
            if i < 4:
                await asyncio.sleep(
                    0.1,
                )  # 0.1s between submissions (intentional timing)

        # Wait for final tasks to complete (poll instead of fixed 2s sleep)
        # With cancel_running=False: task0 runs to completion, then task4 runs
        await _wait_for_condition(
            lambda: sum(1 for r in results if "completed" in r) >= 2,
            timeout=5.0,
        )

        # Count results
        completed = sum(1 for r in results if "completed" in r)
        cancelled = sum(1 for r in results if "cancelled" in r)

        # With cancel_running=False and proper shielding:
        # - Task 0 completes (first running task)
        # - Task 4 completes (final pending task after task 0 finishes)
        # - No tasks should be cancelled
        assert completed >= 2, (
            f"Expected at least 2 completions (first + final task)\n"
            f"  Completed: {completed}, Cancelled: {cancelled}\n"
            f"  Results: {results}"
        )

        assert cancelled == 0, (
            f"No tasks should be cancelled with cancel_running=False!\n"
            f"  Completed: {completed}, Cancelled: {cancelled}\n"
            f"  Results: {results}"
        )


class TestDebouncerUserOriginProtection:
    """Tests for user-origin pending task protection.

    When a user utterance is pending in the debouncer, non-user events
    (actor lifecycle, system events) must not replace it. This prevents
    the scenario where a user speaks, their request is queued behind a
    running task, and then an actor event arrives and silently replaces
    the user's request.
    """

    @pytest.mark.asyncio
    async def test_non_user_cannot_replace_pending_user(self):
        """A non-user submission is skipped when a user utterance is pending."""
        from unity.conversation_manager.domains.utils import Debouncer

        debouncer = Debouncer(name="TestCM")

        results = []
        task0_started = asyncio.Event()

        async def slow_task(label):
            if label == "running":
                task0_started.set()
            try:
                results.append(f"{label}:started")
                await asyncio.sleep(0.5)
                results.append(f"{label}:completed")
            except asyncio.CancelledError:
                results.append(f"{label}:cancelled")
                raise

        # Task 0: running (non-user)
        await debouncer.submit(slow_task, args=("running",), cancel_running=False)
        await asyncio.wait_for(task0_started.wait(), timeout=2.0)

        # User utterance: pending behind running task
        await debouncer.submit(
            slow_task,
            args=("user",),
            cancel_running=False,
            is_user_origin=True,
        )
        await _wait_for_condition(lambda: debouncer.pending_task is not None)
        assert debouncer._pending_is_user_origin is True

        # Actor event: should be SKIPPED because user utterance is pending
        await debouncer.submit(
            slow_task,
            args=("actor",),
            cancel_running=False,
            is_user_origin=False,
        )

        # Wait for the user task to run (after running task completes)
        await _wait_for_condition(
            lambda: "user:completed" in results,
            timeout=5.0,
        )

        assert "user:completed" in results, (
            f"User task should have completed, not been replaced.\n"
            f"  Results: {results}"
        )
        assert "actor:started" not in results, (
            f"Actor task should have been skipped (user pending).\n"
            f"  Results: {results}"
        )

    @pytest.mark.asyncio
    async def test_user_can_replace_pending_non_user(self):
        """A user submission replaces a pending non-user submission normally."""
        from unity.conversation_manager.domains.utils import Debouncer

        debouncer = Debouncer()

        results = []
        task0_started = asyncio.Event()

        async def slow_task(label):
            if label == "running":
                task0_started.set()
            try:
                results.append(f"{label}:started")
                await asyncio.sleep(0.3)
                results.append(f"{label}:completed")
            except asyncio.CancelledError:
                results.append(f"{label}:cancelled")
                raise

        # Task 0: running
        await debouncer.submit(slow_task, args=("running",), cancel_running=False)
        await asyncio.wait_for(task0_started.wait(), timeout=2.0)

        # Actor event: pending
        await debouncer.submit(
            slow_task,
            args=("actor",),
            cancel_running=False,
            is_user_origin=False,
        )

        # User utterance: should REPLACE the actor pending
        await debouncer.submit(
            slow_task,
            args=("user",),
            cancel_running=False,
            is_user_origin=True,
        )

        await _wait_for_condition(
            lambda: "user:completed" in results,
            timeout=5.0,
        )

        assert "user:completed" in results
        assert "actor:started" not in results, (
            f"Actor task should have been replaced by user task.\n"
            f"  Results: {results}"
        )

    @pytest.mark.asyncio
    async def test_user_can_replace_pending_user(self):
        """A newer user submission replaces an older pending user submission."""
        from unity.conversation_manager.domains.utils import Debouncer

        debouncer = Debouncer()

        results = []
        task0_started = asyncio.Event()

        async def slow_task(label):
            if label == "running":
                task0_started.set()
            try:
                results.append(f"{label}:started")
                await asyncio.sleep(0.3)
                results.append(f"{label}:completed")
            except asyncio.CancelledError:
                results.append(f"{label}:cancelled")
                raise

        # Task 0: running
        await debouncer.submit(slow_task, args=("running",), cancel_running=False)
        await asyncio.wait_for(task0_started.wait(), timeout=2.0)

        # User utterance 1: pending
        await debouncer.submit(
            slow_task,
            args=("user1",),
            cancel_running=False,
            is_user_origin=True,
        )

        # User utterance 2: replaces user 1 (newer user beats older user)
        await debouncer.submit(
            slow_task,
            args=("user2",),
            cancel_running=False,
            is_user_origin=True,
        )

        await _wait_for_condition(
            lambda: "user2:completed" in results,
            timeout=5.0,
        )

        assert "user2:completed" in results
        assert "user1:started" not in results, (
            f"Older user task should have been replaced by newer user task.\n"
            f"  Results: {results}"
        )
