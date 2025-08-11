import pytest
import functools

from unity.conductor.simulated import SimulatedConductor
from unity.task_scheduler.simulated import SimulatedTaskScheduler
from tests.helpers import _handle_project


@pytest.mark.asyncio
@_handle_project
async def test_ask_calls_task_scheduler_ask(monkeypatch):
    """Questions about the backlog should consult TaskScheduler.ask once."""
    calls = {"count": 0}
    original = SimulatedTaskScheduler.ask

    @functools.wraps(original)
    async def spy(self, text: str, **kwargs):
        calls["count"] += 1
        return await original(self, text, **kwargs)

    monkeypatch.setattr(SimulatedTaskScheduler, "ask", spy, raising=True)

    cond = SimulatedConductor("Demo – engineering sprint board.")
    handle = await cond.ask("Which tasks are due before Friday?")
    await handle.result()

    assert calls["count"] == 1, "TaskScheduler.ask should be triggered exactly once."
