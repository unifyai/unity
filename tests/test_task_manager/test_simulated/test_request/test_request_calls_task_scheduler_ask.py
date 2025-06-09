import pytest
import functools
import asyncio

from unity.task_manager.simulated import SimulatedTaskManager
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
    def spy(self, text: str, **kwargs):
        calls["count"] += 1
        return original(self, text, **kwargs)

    monkeypatch.setattr(SimulatedTaskScheduler, "ask", spy, raising=True)

    tm = SimulatedTaskManager("Sprint board demo.")
    handle = tm.request(
        "List all of the tasks which are still due this week, and then update them all to be high priority",
    )
    await asyncio.wait_for(handle.result(), timeout=60)

    assert calls["count"] == 1, "TaskScheduler.ask should be called exactly once."
