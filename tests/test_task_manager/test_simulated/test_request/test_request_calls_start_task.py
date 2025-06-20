import pytest
import functools
import asyncio

from unity.conductor.simulated import SimulatedConductor
from unity.task_scheduler.simulated import SimulatedTaskScheduler
from tests.helpers import _handle_project


@pytest.mark.asyncio
@_handle_project
async def test_request_calls_execute_task(monkeypatch):
    """
    A 'start this task now' request should call TaskScheduler.execute_task once
    (via the wrapped _execute_task_call_ helper inside Conductor.request).
    """
    calls = {"count": 0}
    original = SimulatedTaskScheduler.execute_task

    @functools.wraps(original)
    async def spy(self, task_id: int, **kwargs):
        calls["count"] += 1
        return await original(self, task_id, **kwargs)

    monkeypatch.setattr(SimulatedTaskScheduler, "execute_task", spy, raising=True)

    tm = SimulatedConductor("Demo – deployment pipeline.")
    handle = await tm.request(
        "Please start task with 'task id == 17' right away – we need the build running.",
    )
    await asyncio.wait_for(handle.result(), timeout=6000)

    assert (
        calls["count"] == 1
    ), "TaskScheduler.execute_task should be invoked exactly once."
