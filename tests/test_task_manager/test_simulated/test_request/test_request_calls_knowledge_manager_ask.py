import pytest
import functools
import asyncio

from unity.task_manager.simulated import SimulatedTaskManager
from unity.knowledge_manager.simulated import SimulatedKnowledgeManager
from tests.helpers import _handle_project


@pytest.mark.asyncio
@_handle_project
async def test_request_calls_knowledge_manager_ask(monkeypatch):
    """
    A write request that first checks existing KB facts should call KnowledgeManager.retrieve once.
    """
    calls = {"count": 0}
    original = SimulatedKnowledgeManager.retrieve

    @functools.wraps(original)
    async def spy(self, text: str, **kwargs):
        calls["count"] += 1
        return await original(self, text, **kwargs)

    monkeypatch.setattr(SimulatedKnowledgeManager, "retrieve", spy, raising=True)

    tm = SimulatedTaskManager("Ops run-book demo.")
    handle = await tm.request(
        "Update the knowledge-base to say the X200 battery warranty is now three years. "
        "First, check what warranty period we currently have recorded so we can note the change.",
    )
    await asyncio.wait_for(handle.result(), timeout=60)

    assert calls["count"] == 1, "KnowledgeManager.retrieve must be called exactly once."
