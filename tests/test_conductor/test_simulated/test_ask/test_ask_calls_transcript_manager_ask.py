import pytest
import functools

from unity.conductor.simulated import SimulatedConductor
from unity.transcript_manager.simulated import SimulatedTranscriptManager
from tests.helpers import _handle_project


@pytest.mark.asyncio
@_handle_project
async def test_ask_calls_transcript_manager_ask(monkeypatch):
    """Task-level question about recent chats should hit TranscriptManager.ask."""
    calls = {"count": 0}
    original = SimulatedTranscriptManager.ask

    @functools.wraps(original)
    async def spy(self, text: str, **kwargs):
        calls["count"] += 1
        return await original(self, text, **kwargs)

    monkeypatch.setattr(SimulatedTranscriptManager, "ask", spy, raising=True)

    cond = SimulatedConductor("Demo – support-team chat archive.")
    handle = await cond.ask(
        "Show me the last Slack message Frank sent about ticket #381.",
    )
    await handle.result()

    assert calls["count"] == 1, "TranscriptManager.ask should be invoked once."
