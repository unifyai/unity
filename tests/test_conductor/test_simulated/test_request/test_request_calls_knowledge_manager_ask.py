import pytest
import functools

from unity.conductor.simulated import SimulatedConductor
from unity.knowledge_manager.simulated import SimulatedKnowledgeManager
from tests.helpers import _handle_project


@pytest.mark.asyncio
@_handle_project
async def test_request_calls_knowledge_manager_ask(monkeypatch):
    """
    A write request that first checks existing KB facts should call KnowledgeManager.retrieve once.
    """
    calls = {"count": 0}
    original = SimulatedKnowledgeManager.ask

    @functools.wraps(original)
    async def spy(self, text: str, **kwargs):
        calls["count"] += 1
        return await original(self, text, **kwargs)

    monkeypatch.setattr(SimulatedKnowledgeManager, "ask", spy, raising=True)

    cond = SimulatedConductor("Ops run-book demo.")
    handle = await cond.request(
        "Update the knowledge-base to say the X200 battery warranty is now three years. "
        "First, check what warranty period we currently have recorded so we can note the change.",
    )
    await handle.result()

    assert calls["count"] == 1, "KnowledgeManager.ask must be called exactly once."
