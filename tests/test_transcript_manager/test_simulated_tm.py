from __future__ import annotations

import asyncio
import pytest

from unity.transcript_manager.simulated import (
    SimulatedTranscriptManager,
    _SimulatedTranscriptHandle,
)

# Helper identical to the one used elsewhere in the test-suite
from tests.helpers import _handle_project


# ────────────────────────────────────────────────────────────────────────────
# 1.  Basic start-and-ask                                                    #
# ────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@_handle_project
async def test_start_and_ask_simulated_tm():
    tm = SimulatedTranscriptManager("Demo transcript DB.")
    handle = tm.ask("Show me my unread emails.")
    answer = await handle.result()
    assert isinstance(answer, str) and answer.strip(), "Answer should be non-empty"


# ────────────────────────────────────────────────────────────────────────────
# 2.  Interject                                                             #
# ────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@_handle_project
async def test_interject_simulated_tm(monkeypatch):
    counts = {"interject": 0}
    original_interject = _SimulatedTranscriptHandle.interject

    def wrapped(self, message: str) -> str:  # type: ignore[override]
        counts["interject"] += 1
        return original_interject(self, message)

    monkeypatch.setattr(
        _SimulatedTranscriptHandle,
        "interject",
        wrapped,
        raising=True,
    )

    tm = SimulatedTranscriptManager()
    handle = tm.ask("Summarise yesterday's Slack exchange with Bob.")
    # interject while running
    await asyncio.sleep(0.05)
    reply = handle.interject("Also include any emojis Bob used.")
    assert "ack" in reply.lower()

    await handle.result()
    assert counts["interject"] == 1, ".interject should be called exactly once"


# ────────────────────────────────────────────────────────────────────────────
# 3.  Stop                                                                  #
# ────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@_handle_project
async def test_stop_simulated_tm():
    tm = SimulatedTranscriptManager()
    handle = tm.ask("Produce a full export of all messages.")
    await asyncio.sleep(0.05)
    handle.stop()

    with pytest.raises(asyncio.CancelledError):
        await handle.result()

    assert handle.done(), "Handle should report done after stop()"


# ────────────────────────────────────────────────────────────────────────────
# 4.  Clarification handshake                                               #
# ────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@_handle_project
async def test_tm_requests_clarification():
    tm = SimulatedTranscriptManager()

    up_q: asyncio.Queue[str] = asyncio.Queue()
    down_q: asyncio.Queue[str] = asyncio.Queue()

    handle = tm.ask(
        "Find important messages.",
        clarification_up_q=up_q,
        clarification_down_q=down_q,
    )

    # Must ask for clarification first
    question = await asyncio.wait_for(up_q.get(), timeout=30)
    assert "clarify" in question.lower()

    # Provide clarification
    await down_q.put("Focus on project Alpha deadlines.")
    answer = await handle.result()

    assert isinstance(answer, str) and answer.strip(), "Answer should not be empty"
