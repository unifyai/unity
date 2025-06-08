import pytest
import functools
import asyncio

from unity.task_manager.simulated import SimulatedTaskManager
from unity.task_scheduler.simulated import SimulatedTaskScheduler
from tests.helpers import _handle_project


@pytest.mark.asyncio
@_handle_project
async def test_ask_calls_task_scheduler_ask(monkeypatch):
    """Questions about the backlog should consult TaskScheduler.ask once."""
    calls = {"count": 0}
    original = SimulatedTaskScheduler.ask

    @functools.wraps(original)
    def spy(self, text: str, **kwargs):
        calls["count"] += 1
        return original(self, text, **kwargs)

    monkeypatch.setattr(SimulatedTaskScheduler, "ask", spy, raising=True)

    tm = SimulatedTaskManager("Demo – engineering sprint board.")
    handle = tm.ask("Which tasks are due before Friday?")
    await asyncio.wait_for(handle.result(), timeout=6000)

    assert calls["count"] == 1, "TaskScheduler.ask should be triggered exactly once."
