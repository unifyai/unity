import pytest
import functools

from unity.conductor.simulated import SimulatedConductor
from unity.knowledge_manager.simulated import SimulatedKnowledgeManager
from tests.helpers import _handle_project


@pytest.mark.asyncio
@_handle_project
async def test_request_calls_knowledge_manager_store(monkeypatch):
    """A write-request to remember new facts should hit KnowledgeManager.store once."""
    calls = {"count": 0}
    original = SimulatedKnowledgeManager.update

    @functools.wraps(original)
    async def spy(self, text: str, **kwargs):
        calls["count"] += 1
        return await original(self, text, **kwargs)

    monkeypatch.setattr(SimulatedKnowledgeManager, "update", spy, raising=True)

    cond = SimulatedConductor("Demo – ops run-book KB.")
    handle = await cond.request(
        "Remember that the new Wi-Fi password is 'P@ssw0rd2025'.",
    )
    await handle.result()

    assert calls["count"] == 1, "KnowledgeManager.update should be called exactly once."
