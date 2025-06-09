import pytest
import functools
import asyncio

from unity.task_manager.simulated import SimulatedTaskManager
from unity.task_scheduler.simulated import SimulatedTaskScheduler
from tests.helpers import _handle_project


@pytest.mark.asyncio
@_handle_project
async def test_request_calls_start_task(monkeypatch):
    """
    A 'start this task now' request should call TaskScheduler.start_task once
    (via the wrapped _start_task_call_ helper inside TaskManager.request).
    """
    calls = {"count": 0}
    original = SimulatedTaskScheduler.start_task

    @functools.wraps(original)
    def spy(self, task_id: int, **kwargs):
        calls["count"] += 1
        return original(self, task_id, **kwargs)

    monkeypatch.setattr(SimulatedTaskScheduler, "start_task", spy, raising=True)

    tm = SimulatedTaskManager("Demo – deployment pipeline.")
    handle = tm.request(
        "Please start task with 'task id == 17' right away – we need the build running.",
    )
    await asyncio.wait_for(handle.result(), timeout=60)

    assert (
        calls["count"] == 1
    ), "TaskScheduler.start_task should be invoked exactly once."
