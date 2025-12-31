"""
Regression tests for concurrent task execution guard in delegate mode.

Verifies that the TaskScheduler correctly prevents nested task execution attempts
(e.g., when a task's execution tries to start another task) by raising RuntimeError,
and that such failures do not corrupt the Tasks table or leave orphaned 'active' rows.
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import Any, Optional

import pytest

from tests.helpers import _handle_project
from unity.actor.simulated import SimulatedActor
from unity.common.task_execution_context import current_task_execution_delegate
from unity.task_scheduler.task_scheduler import TaskScheduler
from unity.task_scheduler.types.status import Status


class _Delegate:
    """Test-only TaskExecutionDelegate that triggers a nested execute attempt."""

    def __init__(
        self,
        *,
        actor: SimulatedActor,
        scheduler: TaskScheduler,
        nested_task_id: int,
    ) -> None:
        self._actor = actor
        self._scheduler = scheduler
        self._nested_task_id = nested_task_id
        self._trigger_nested = asyncio.Event()
        self._nested_exc: "asyncio.Queue[BaseException]" = asyncio.Queue(maxsize=1)
        self._nested_task: Optional[asyncio.Task] = None

    def trigger_nested(self) -> None:
        self._trigger_nested.set()

    async def wait_for_nested_error(self, *, timeout: float = 5.0) -> BaseException:
        return await asyncio.wait_for(self._nested_exc.get(), timeout=timeout)

    async def cancel_background(self) -> None:
        """Cancel any pending background tasks started by this delegate."""
        if self._nested_task is None:
            return
        if self._nested_task.done():
            return
        self._nested_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._nested_task

    async def start_task_run(
        self,
        *,
        task_description: str,
        entrypoint: int | None,
        parent_chat_context: list[dict] | None,
        clarification_up_q: Optional[asyncio.Queue[str]],
        clarification_down_q: Optional[asyncio.Queue[str]],
        images: Any | None = None,
        **kwargs: Any,
    ):
        # Start the task on the underlying actor (SimulatedActor ignores extra kwargs).
        handle = await self._actor.act(
            task_description,
            _parent_chat_context=parent_chat_context,
            _clarification_up_q=clarification_up_q,
            _clarification_down_q=clarification_down_q,
            entrypoint=entrypoint,
            **kwargs,
        )

        async def _attempt_nested_execute() -> None:
            await self._trigger_nested.wait()
            try:
                await self._scheduler.execute(task_id=self._nested_task_id)
            except BaseException as e:  # noqa: BLE001 - test harness captures error type/message
                try:
                    self._nested_exc.put_nowait(e)
                except Exception:
                    pass

        # Defer the nested attempt until the test triggers it, ensuring the outer
        # scheduler has set `_active_task` (and the Tasks row is active).
        self._nested_task = asyncio.create_task(_attempt_nested_execute())

        # Optional: avoid unused param warnings while keeping the signature aligned.
        _ = images
        return handle


@pytest.mark.asyncio
@pytest.mark.timeout(120)
@_handle_project
async def test_nested_execute_raises_runtime_error_in_delegate_mode():
    actor = SimulatedActor(steps=None, duration=None)
    scheduler = TaskScheduler(actor=actor)

    task_a = scheduler._create_task(name="task_a", description="task_a")["details"][
        "task_id"
    ]
    task_b = scheduler._create_task(name="task_b", description="task_b")["details"][
        "task_id"
    ]

    delegate = _Delegate(actor=actor, scheduler=scheduler, nested_task_id=task_b)
    token = current_task_execution_delegate.set(delegate)
    handle = None
    try:
        handle = await scheduler.execute(task_id=task_a)

        assert scheduler._active_task is not None
        assert scheduler._active_task.task_id == task_a

        # Trigger a nested execute attempt while task_a is running.
        delegate.trigger_nested()
        err = await delegate.wait_for_nested_error(timeout=5.0)

        assert isinstance(err, RuntimeError)
        assert str(err) == "Another task is already running – stop it first."

        # Verify task_a is still active and task_b was not corrupted.
        row_a = scheduler._filter_tasks(filter=f"task_id == {task_a}", limit=1)[0]
        row_b = scheduler._filter_tasks(filter=f"task_id == {task_b}", limit=1)[0]
        assert row_a.status == Status.active
        assert row_b.status != Status.active
        assert scheduler._active_task is not None
        assert scheduler._active_task.task_id == task_a

    finally:
        try:
            await delegate.cancel_background()
        except Exception:
            pass
        try:
            current_task_execution_delegate.reset(token)
        except Exception:
            pass
        if handle is not None:
            handle.stop(cancel=True)
            await handle.result()
        assert scheduler._active_task is None


@pytest.mark.asyncio
@pytest.mark.timeout(120)
@_handle_project
async def test_failed_nested_execute_does_not_corrupt_tasks_table():
    actor = SimulatedActor(steps=None, duration=None)
    scheduler = TaskScheduler(actor=actor)

    task_a = scheduler._create_task(name="task_a", description="task_a")["details"][
        "task_id"
    ]
    task_b = scheduler._create_task(name="task_b", description="task_b")["details"][
        "task_id"
    ]

    delegate = _Delegate(actor=actor, scheduler=scheduler, nested_task_id=task_b)
    token = current_task_execution_delegate.set(delegate)
    handle = None
    try:
        handle = await scheduler.execute(task_id=task_a)

        # Attempt multiple nested executions; all should be rejected.
        for _ in range(2):
            try:
                await scheduler.execute(task_id=task_b)
                raise AssertionError("Expected RuntimeError for concurrent execute")
            except RuntimeError as e:
                assert str(e) == "Another task is already running – stop it first."

        active_rows = scheduler._filter_tasks(filter="status == 'active'")
        assert len(active_rows) == 1
        assert active_rows[0].task_id == task_a

    finally:
        try:
            await delegate.cancel_background()
        except Exception:
            pass
        try:
            current_task_execution_delegate.reset(token)
        except Exception:
            pass
        if handle is not None:
            handle.stop(cancel=True)
            await handle.result()
        # After cleanup, there should be no active rows and no active pointer.
        assert scheduler._active_task is None
        assert scheduler._filter_tasks(filter="status == 'active'") == []
