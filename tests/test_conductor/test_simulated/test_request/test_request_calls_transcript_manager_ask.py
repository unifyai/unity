import pytest
import functools

from unity.conductor.simulated import SimulatedConductor
from unity.transcript_manager.simulated import SimulatedTranscriptManager
from tests.helpers import _handle_project


@pytest.mark.asyncio
@_handle_project
async def test_request_calls_transcript_manager_ask(monkeypatch):
    """
    A mutation request that also needs a transcript lookup should hit TranscriptManager.ask once.
    """
    calls = {"count": 0}
    original = SimulatedTranscriptManager.ask

    @functools.wraps(original)
    async def spy(self, text: str, **kwargs):
        calls["count"] += 1
        return await original(self, text, **kwargs)

    monkeypatch.setattr(SimulatedTranscriptManager, "ask", spy, raising=True)

    cond = SimulatedConductor("Support chats demo.")
    handle = await cond.request(
        "Archive yesterday's Slack conversation about bug #4321. "
        "Before archiving, tell me the final message in that thread so I can paste it in the ticket.",
    )
    await handle.result()

    assert calls["count"] >= 1, "TranscriptManager.ask should be invoked at least once."
