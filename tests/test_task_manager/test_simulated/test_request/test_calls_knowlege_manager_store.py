# test_task_manager/test_simulated/test_request/test_calls_knowlege_manager_store.py
import pytest
import functools
import asyncio

from unity.task_manager.simulated import SimulatedTaskManager
from unity.knowledge_manager.simulated import SimulatedKnowledgeManager
from tests.helpers import _handle_project


@pytest.mark.asyncio
@_handle_project
async def test_calls_knowledge_manager_store(monkeypatch):
    """A write-request to remember new facts should hit KnowledgeManager.store once."""
    calls = {"count": 0}
    original = SimulatedKnowledgeManager.store

    @functools.wraps(original)
    def spy(self, text: str, **kwargs):
        calls["count"] += 1
        return original(self, text, **kwargs)

    monkeypatch.setattr(SimulatedKnowledgeManager, "store", spy, raising=True)

    tm = SimulatedTaskManager("Demo – ops run-book KB.")
    handle = tm.request("Remember that the new Wi-Fi password is 'P@ssw0rd2025'.")
    await asyncio.wait_for(handle.result(), timeout=60)

    assert calls["count"] == 1, "KnowledgeManager.store should be called exactly once."
