"""
Tests for the Debouncer class.

The Debouncer provides rate-limiting for async function calls with these behaviors:
1. Debouncing: Multiple rapid submits result in only the last one running
2. Sequential execution: Running tasks complete before the next starts (by default)
3. Cancel running: Optionally cancel currently running task for immediate new execution
4. Delay: Optionally wait before starting execution
"""

import asyncio
import pytest

from unity.conversation_manager.domains.utils import Debouncer


class TestBasicExecution:
    """Test basic submit and execution functionality."""

    @pytest.mark.asyncio
    async def test_submit_executes_async_function(self):
        """Test that a submitted function is actually executed."""
        debouncer = Debouncer()
        result = []

        async def task():
            result.append("executed")

        await debouncer.submit(task)
        # Wait for the pending task to schedule and run
        await asyncio.sleep(0.01)

        assert result == ["executed"]

    @pytest.mark.asyncio
    async def test_submit_with_args(self):
        """Test that args are passed correctly to the function."""
        debouncer = Debouncer()
        result = []

        async def task(a, b, c):
            result.append((a, b, c))

        await debouncer.submit(task, args=(1, 2, 3))
        await asyncio.sleep(0.01)

        assert result == [(1, 2, 3)]

    @pytest.mark.asyncio
    async def test_submit_with_kwargs(self):
        """Test that kwargs are passed correctly to the function."""
        debouncer = Debouncer()
        result = []

        async def task(x=None, y=None):
            result.append({"x": x, "y": y})

        await debouncer.submit(task, kwargs={"x": 10, "y": 20})
        await asyncio.sleep(0.01)

        assert result == [{"x": 10, "y": 20}]

    @pytest.mark.asyncio
    async def test_submit_with_args_and_kwargs(self):
        """Test that both args and kwargs are passed correctly."""
        debouncer = Debouncer()
        result = []

        async def task(a, b, x=None):
            result.append((a, b, x))

        await debouncer.submit(task, args=(1, 2), kwargs={"x": "hello"})
        await asyncio.sleep(0.01)

        assert result == [(1, 2, "hello")]


class TestDebouncingBehavior:
    """Test that multiple rapid submits result in only the last one running."""

    @pytest.mark.asyncio
    async def test_rapid_submits_only_last_runs(self):
        """Test that rapid submits debounce - only the last submitted task runs."""
        debouncer = Debouncer()
        result = []

        async def task(value):
            result.append(value)

        # Submit multiple tasks rapidly
        await debouncer.submit(task, args=("first",))
        await debouncer.submit(task, args=("second",))
        await debouncer.submit(task, args=("third",))

        # Wait for execution
        await asyncio.sleep(0.05)

        # Only the last one should have run
        assert result == ["third"]

    @pytest.mark.asyncio
    async def test_debounce_cancels_pending_not_running(self):
        """Test that debouncing cancels pending tasks but not running ones."""
        debouncer = Debouncer()
        execution_order = []
        started_event = asyncio.Event()

        async def slow_task(value):
            execution_order.append(f"start:{value}")
            started_event.set()
            await asyncio.sleep(0.1)
            execution_order.append(f"end:{value}")

        async def fast_task(value):
            execution_order.append(f"fast:{value}")

        # Start a slow task
        await debouncer.submit(slow_task, args=("slow",))

        # Wait for slow task to start running
        await started_event.wait()

        # Submit multiple fast tasks while slow is running
        # These become pending and get debounced
        await debouncer.submit(fast_task, args=("pending1",))
        await debouncer.submit(fast_task, args=("pending2",))

        # Wait for everything to complete
        await asyncio.sleep(0.2)

        # Slow task should complete, and only pending2 should run (pending1 debounced)
        assert "start:slow" in execution_order
        assert "end:slow" in execution_order
        assert "fast:pending1" not in execution_order
        assert "fast:pending2" in execution_order


class TestSequentialExecution:
    """Test that tasks execute sequentially by default."""

    @pytest.mark.asyncio
    async def test_tasks_execute_sequentially(self):
        """Test that a new task waits for the running task to complete."""
        debouncer = Debouncer()
        execution_order = []
        first_started = asyncio.Event()

        async def task(value, duration):
            execution_order.append(f"start:{value}")
            if value == "first":
                first_started.set()
            await asyncio.sleep(duration)
            execution_order.append(f"end:{value}")

        # Start first task
        await debouncer.submit(task, args=("first", 0.1))

        # Wait for first to start
        await first_started.wait()

        # Submit second task (should wait for first)
        await debouncer.submit(task, args=("second", 0.05))

        # Wait for both to complete
        await asyncio.sleep(0.3)

        # First should complete before second starts
        assert execution_order.index("end:first") < execution_order.index(
            "start:second",
        )

    @pytest.mark.asyncio
    async def test_running_task_not_cancelled_by_default(self):
        """Test that the running task is not cancelled when a new task is submitted."""
        debouncer = Debouncer()
        cancelled = []
        first_started = asyncio.Event()

        async def task(value):
            try:
                execution_order = []
                execution_order.append(f"start:{value}")
                if value == "first":
                    first_started.set()
                await asyncio.sleep(0.1)
                execution_order.append(f"end:{value}")
            except asyncio.CancelledError:
                cancelled.append(value)
                raise

        # Start first task
        await debouncer.submit(task, args=("first",))

        # Wait for it to start
        await first_started.wait()

        # Submit new task (should NOT cancel first)
        await debouncer.submit(task, args=("second",))

        # Wait for completion
        await asyncio.sleep(0.3)

        # First task should NOT have been cancelled
        assert "first" not in cancelled


class TestCancelRunning:
    """Test cancel_running=True behavior."""

    @pytest.mark.asyncio
    async def test_cancel_running_cancels_current_task(self):
        """Test that cancel_running=True cancels the currently running task."""
        debouncer = Debouncer()
        cancelled = []
        completed = []
        first_started = asyncio.Event()

        async def task(value):
            try:
                if value == "first":
                    first_started.set()
                await asyncio.sleep(0.5)
                completed.append(value)
            except asyncio.CancelledError:
                cancelled.append(value)
                raise

        # Start first task
        await debouncer.submit(task, args=("first",))

        # Wait for it to start
        await first_started.wait()

        # Submit with cancel_running=True
        await debouncer.submit(task, args=("second",), cancel_running=True)

        # Wait for completion
        await asyncio.sleep(0.6)

        # First should be cancelled, second should complete
        assert "first" in cancelled
        assert "second" in completed

    @pytest.mark.asyncio
    async def test_cancel_running_allows_immediate_new_task(self):
        """Test that cancel_running allows new task to start immediately."""
        debouncer = Debouncer()
        start_times = {}
        first_started = asyncio.Event()

        async def task(value):
            start_times[value] = asyncio.get_event_loop().time()
            if value == "first":
                first_started.set()
            await asyncio.sleep(0.5)

        start_time = asyncio.get_event_loop().time()

        # Start first task
        await debouncer.submit(task, args=("first",))

        # Wait for it to start
        await first_started.wait()

        # Submit with cancel_running=True
        await debouncer.submit(task, args=("second",), cancel_running=True)

        # Wait a bit for second to start
        await asyncio.sleep(0.05)

        # Second should have started quickly (not waiting 0.5s for first)
        assert "second" in start_times
        second_delay = start_times["second"] - start_time
        assert second_delay < 0.3  # Should start well before first would complete


class TestDelay:
    """Test delay parameter functionality."""

    @pytest.mark.asyncio
    async def test_delay_before_execution(self):
        """Test that task execution is delayed by the specified amount."""
        debouncer = Debouncer()
        start_time = asyncio.get_event_loop().time()
        execution_time = []

        async def task():
            execution_time.append(asyncio.get_event_loop().time())

        await debouncer.submit(task, delay=0.1)

        # Wait for execution
        await asyncio.sleep(0.2)

        assert len(execution_time) == 1
        elapsed = execution_time[0] - start_time
        assert 0.08 <= elapsed <= 0.2  # Should have waited ~0.1s

    @pytest.mark.asyncio
    async def test_delayed_task_debounced_before_delay(self):
        """Test that a delayed task can be debounced before its delay expires."""
        debouncer = Debouncer()
        result = []

        async def task(value):
            result.append(value)

        # Submit with delay
        await debouncer.submit(task, args=("first",), delay=0.1)

        # Submit again before delay expires (debounces first)
        await debouncer.submit(task, args=("second",), delay=0.0)

        # Wait for execution
        await asyncio.sleep(0.2)

        # Only second should run (first was debounced)
        assert result == ["second"]

    @pytest.mark.asyncio
    async def test_delay_zero_executes_immediately(self):
        """Test that delay=0 executes without waiting."""
        debouncer = Debouncer()
        start_time = asyncio.get_event_loop().time()
        execution_time = []

        async def task():
            execution_time.append(asyncio.get_event_loop().time())

        await debouncer.submit(task, delay=0)

        # Wait briefly for async execution
        await asyncio.sleep(0.02)

        assert len(execution_time) == 1
        elapsed = execution_time[0] - start_time
        assert elapsed < 0.05  # Should be nearly immediate


class TestDelayAndCancelRunning:
    """Test interaction between delay and cancel_running."""

    @pytest.mark.asyncio
    async def test_delay_with_cancel_running(self):
        """Test that delay works with cancel_running=True."""
        debouncer = Debouncer()
        cancelled = []
        start_times = {}
        first_started = asyncio.Event()

        async def task(value):
            try:
                start_times[value] = asyncio.get_event_loop().time()
                if value == "first":
                    first_started.set()
                await asyncio.sleep(0.5)
            except asyncio.CancelledError:
                cancelled.append(value)
                raise

        submit_time = asyncio.get_event_loop().time()

        # Start first task
        await debouncer.submit(task, args=("first",))

        # Wait for it to start
        await first_started.wait()

        # Submit with delay AND cancel_running
        await debouncer.submit(task, args=("second",), delay=0.1, cancel_running=True)

        # Wait for everything
        await asyncio.sleep(0.3)

        # First should be cancelled
        assert "first" in cancelled
        # Second should have started after ~0.1s delay
        assert "second" in start_times
        second_delay = start_times["second"] - submit_time
        assert 0.08 <= second_delay <= 0.2


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.mark.asyncio
    async def test_submit_when_nothing_running(self):
        """Test submit works correctly when no task is currently running."""
        debouncer = Debouncer()
        result = []

        async def task():
            result.append("ran")

        # Initial state - nothing running
        assert debouncer.running_task is None
        assert debouncer.pending_task is None

        await debouncer.submit(task)
        await asyncio.sleep(0.02)

        assert result == ["ran"]

    @pytest.mark.asyncio
    async def test_multiple_sequential_submits(self):
        """Test multiple submits that each complete before the next."""
        debouncer = Debouncer()
        result = []

        async def task(value):
            result.append(value)

        # Submit and wait for completion
        await debouncer.submit(task, args=("first",))
        await asyncio.sleep(0.02)

        await debouncer.submit(task, args=("second",))
        await asyncio.sleep(0.02)

        await debouncer.submit(task, args=("third",))
        await asyncio.sleep(0.02)

        # All should have run
        assert result == ["first", "second", "third"]

    @pytest.mark.asyncio
    async def test_cancel_running_when_nothing_running(self):
        """Test that cancel_running=True works when nothing is running."""
        debouncer = Debouncer()
        result = []

        async def task():
            result.append("ran")

        # cancel_running should be a no-op when nothing is running
        await debouncer.submit(task, cancel_running=True)
        await asyncio.sleep(0.02)

        assert result == ["ran"]

    @pytest.mark.asyncio
    async def test_task_exception_does_not_break_debouncer(self):
        """Test that an exception in a task doesn't break subsequent submits."""
        debouncer = Debouncer()
        result = []

        async def failing_task():
            raise ValueError("intentional error")

        async def working_task():
            result.append("worked")

        # Submit failing task
        await debouncer.submit(failing_task)
        await asyncio.sleep(0.02)

        # Subsequent submit should still work
        await debouncer.submit(working_task)
        await asyncio.sleep(0.02)

        assert result == ["worked"]

    @pytest.mark.asyncio
    async def test_none_args_kwargs_defaults(self):
        """Test that None args/kwargs are handled correctly."""
        debouncer = Debouncer()
        result = []

        async def task():
            result.append("no-args")

        # Explicitly pass None
        await debouncer.submit(task, args=None, kwargs=None)
        await asyncio.sleep(0.02)

        assert result == ["no-args"]


class TestTaskStateTracking:
    """Test internal state tracking of running and pending tasks."""

    @pytest.mark.asyncio
    async def test_running_task_set_during_execution(self):
        """Test that running_task is set while task is executing."""
        debouncer = Debouncer()
        running_task_during_execution = []
        task_started = asyncio.Event()

        async def task():
            task_started.set()
            running_task_during_execution.append(debouncer.running_task)
            await asyncio.sleep(0.05)

        await debouncer.submit(task)

        # Wait for task to start and capture state
        await task_started.wait()

        # running_task should be set
        assert len(running_task_during_execution) == 1
        assert running_task_during_execution[0] is not None
        assert isinstance(running_task_during_execution[0], asyncio.Task)

    @pytest.mark.asyncio
    async def test_pending_task_cleared_after_becomes_running(self):
        """Test that pending_task is cleared once the task starts running."""
        debouncer = Debouncer()
        task_started = asyncio.Event()

        async def task():
            task_started.set()
            await asyncio.sleep(0.05)

        await debouncer.submit(task)

        # Immediately after submit, pending_task should be set
        assert debouncer.pending_task is not None

        # Wait for task to start running
        await task_started.wait()
        await asyncio.sleep(0.01)  # Small delay for state update

        # pending_task should be cleared (it's now running_task)
        assert debouncer.pending_task is None
        assert debouncer.running_task is not None


class TestCancelTasks:
    """Test the _cancel_tasks internal method behavior."""

    @pytest.mark.asyncio
    async def test_cancel_tasks_cancels_pending(self):
        """Test that _cancel_tasks(pending=True) cancels pending task."""
        debouncer = Debouncer()
        result = []

        async def task():
            result.append("ran")

        # Submit with delay so it stays pending
        await debouncer.submit(task, delay=0.5)

        # Cancel pending
        await debouncer._cancel_tasks(pending=True, running=False)

        # Wait - task should not run
        await asyncio.sleep(0.6)

        assert result == []

    @pytest.mark.asyncio
    async def test_cancel_tasks_running_false_does_not_cancel_running(self):
        """Test that _cancel_tasks(running=False) does not cancel running task."""
        debouncer = Debouncer()
        completed = []
        started = asyncio.Event()

        async def task():
            started.set()
            await asyncio.sleep(0.1)
            completed.append("done")

        await debouncer.submit(task)
        await started.wait()

        # Cancel with running=False
        await debouncer._cancel_tasks(pending=True, running=False)

        # Wait for task to complete
        await asyncio.sleep(0.2)

        # Task should have completed (not cancelled)
        assert completed == ["done"]
