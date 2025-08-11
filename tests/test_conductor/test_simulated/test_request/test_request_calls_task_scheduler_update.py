import pytest
import functools

from unity.conductor.simulated import SimulatedConductor
from unity.task_scheduler.simulated import SimulatedTaskScheduler
from tests.helpers import _handle_project


@pytest.mark.asyncio
@_handle_project
async def test_request_calls_task_scheduler_update(monkeypatch):
    """Creating or editing a task must invoke TaskScheduler.update exactly once."""
    calls = {"count": 0}
    original = SimulatedTaskScheduler.update

    @functools.wraps(original)
    async def spy(self, text: str, **kwargs):
        calls["count"] += 1
        return await original(self, text, **kwargs)

    monkeypatch.setattr(SimulatedTaskScheduler, "update", spy, raising=True)

    cond = SimulatedConductor("Demo – marketing backlog.")
    handle = await cond.request(
        "Add a task: 'Design new landing page', due next Tuesday.",
    )
    await handle.result()

    assert calls["count"] == 1, "TaskScheduler.update must be called once."
