import pytest
import functools
import asyncio

from unity.task_manager.simulated import SimulatedTaskManager
from unity.task_scheduler.simulated import SimulatedTaskScheduler
from tests.helpers import _handle_project


@pytest.mark.asyncio
@_handle_project
async def test_calls_task_scheduler_update(monkeypatch):
    """Creating or editing a task must invoke TaskScheduler.update exactly once."""
    calls = {"count": 0}
    original = SimulatedTaskScheduler.update

    @functools.wraps(original)
    def spy(self, text: str, **kwargs):
        calls["count"] += 1
        return original(self, text, **kwargs)

    monkeypatch.setattr(SimulatedTaskScheduler, "update", spy, raising=True)

    tm = SimulatedTaskManager("Demo – marketing backlog.")
    handle = tm.request("Add a task: 'Design new landing page', due next Tuesday.")
    await asyncio.wait_for(handle.result(), timeout=60)

    assert calls["count"] == 1, "TaskScheduler.update must be called once."
