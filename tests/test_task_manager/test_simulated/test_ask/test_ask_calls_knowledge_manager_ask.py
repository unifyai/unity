import pytest
import functools
import asyncio

from unity.task_manager.simulated import SimulatedTaskManager
from unity.knowledge_manager.simulated import SimulatedKnowledgeManager
from tests.helpers import _handle_project


@pytest.mark.asyncio
@_handle_project
async def test_ask_calls_knowledge_manager_ask(monkeypatch):
    """A knowledge lookup should route to KnowledgeManager.retrieve once."""
    calls = {"count": 0}
    original = SimulatedKnowledgeManager.ask

    @functools.wraps(original)
    async def spy(self, text: str, **kwargs):
        calls["count"] += 1
        return await original(self, text, **kwargs)

    monkeypatch.setattr(SimulatedKnowledgeManager, "retrieve", spy, raising=True)

    tm = SimulatedTaskManager("Demo – internal KB for product specs.")
    handle = await tm.ask("What warranty info do we store about the X200 battery pack?")
    await asyncio.wait_for(handle.result(), timeout=60)

    assert calls["count"] == 1, "KnowledgeManager.retrieve must be called exactly once."
