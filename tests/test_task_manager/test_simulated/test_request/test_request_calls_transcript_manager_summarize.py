import pytest
import functools
import asyncio

from unity.task_manager.simulated import SimulatedTaskManager
from unity.transcript_manager.simulated import SimulatedTranscriptManager
from tests.helpers import _handle_project


@pytest.mark.asyncio
@_handle_project
async def test_request_calls_transcript_manager_summarize(monkeypatch):
    """Mutation request to summarise transcripts should call summarize once."""
    calls = {"count": 0}
    original = SimulatedTranscriptManager.summarize

    @functools.wraps(original)
    async def spy(self, **kwargs):
        calls["count"] += 1
        return await original(self, **kwargs)

    monkeypatch.setattr(SimulatedTranscriptManager, "summarize", spy, raising=True)

    tm = SimulatedTaskManager("Demo – sales-call recordings.")
    handle = await tm.request(
        "Summarise yesterday’s call with ACME Corp (with exchange id 123) and save it.",
    )
    await asyncio.wait_for(handle.result(), timeout=60)

    assert calls["count"] == 1, "summarize must be invoked exactly once."
