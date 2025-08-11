import pytest
import functools

from unity.conductor.simulated import SimulatedConductor
from unity.task_scheduler.simulated import SimulatedTaskScheduler
from tests.helpers import _handle_project


@pytest.mark.asyncio
@_handle_project
async def test_request_calls_task_scheduler_ask(monkeypatch):
    """
    After mutating the task list the assistant should query it once, triggering TaskScheduler.ask.
    """
    calls = {"count": 0}
    original = SimulatedTaskScheduler.ask

    @functools.wraps(original)
    async def spy(self, text: str, **kwargs):
        calls["count"] += 1
        return await original(self, text, **kwargs)

    monkeypatch.setattr(SimulatedTaskScheduler, "ask", spy, raising=True)

    cond = SimulatedConductor("Sprint board demo.")
    handle = await cond.request(
        "List all of the tasks which are still due this week, and then update them all to be high priority",
    )
    await handle.result()

    assert calls["count"] == 1, "TaskScheduler.ask should be called exactly once."
