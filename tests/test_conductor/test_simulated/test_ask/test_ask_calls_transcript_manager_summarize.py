import pytest
import functools

from unity.conductor.simulated import SimulatedConductor
from unity.transcript_manager.simulated import SimulatedTranscriptManager
from tests.helpers import _handle_project


@pytest.mark.asyncio
@_handle_project
async def test_ask_calls_transcript_manager_summarize(monkeypatch):
    """Ask request to summarise transcripts should call summarize once."""
    calls = {"count": 0}
    original = SimulatedTranscriptManager.summarize

    @functools.wraps(original)
    async def spy(self, **kwargs):
        calls["count"] += 1
        return await original(self, **kwargs)

    monkeypatch.setattr(SimulatedTranscriptManager, "summarize", spy, raising=True)

    cond = SimulatedConductor("Demo – sales-call recordings.")
    handle = await cond.request(
        "Can you please give me a summary of the recent exchange with id==123.",
    )
    await handle.result()

    assert calls["count"] == 1, "summarize must be invoked exactly once."
