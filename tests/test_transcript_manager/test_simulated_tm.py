from __future__ import annotations

import asyncio
import pytest
import functools

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
    handle = await tm.ask("Show me my unread emails.")
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

    @functools.wraps(original_interject)
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
    handle = await tm.ask("Summarise yesterday's Slack exchange with Bob.")
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
    handle = await tm.ask("Produce a full export of all messages.")
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

    handle = await tm.ask(
        "Find important messages.",
        clarification_up_q=up_q,
        clarification_down_q=down_q,
        _requests_clarification=True,
    )

    # Must ask for clarification first
    question = await asyncio.wait_for(up_q.get(), timeout=60)
    assert "clarify" in question.lower()

    # Provide clarification
    await down_q.put("Focus on project Alpha deadlines.")
    answer = await handle.result()

    assert isinstance(answer, str) and answer.strip(), "Answer should not be empty"


# ────────────────────────────────────────────────────────────────────────────
# 5.  Stateful memory across serial asks                                     #
# ────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@_handle_project
async def test_tm_stateful_memory():
    """
    Two consecutive .ask() calls should share the same conversation context
    because the manager's LLM is stateful.
    """
    tm = SimulatedTranscriptManager()

    # 1) Ask for a unique codename – expect a non-empty answer
    handle1 = await tm.ask(
        "Please invent a unique project codename for our upcoming initiative. "
        "Respond with *only* the codename.",
    )
    codename = (await handle1.result()).strip()
    assert codename, "Codename should not be empty"

    # 2) Ask the LLM to recall what it just said
    handle2 = await tm.ask("Great. What codename did you suggest earlier?")
    answer2 = (await handle2.result()).lower()

    # The second answer should mention the same codename exactly
    assert (
        codename.lower().split(" ")[-1] in answer2
    ), "LLM should recall the previous codename"


# ────────────────────────────────────────────────────────────────────────────
# 7.  Doc-string inheritance                                                 #
# ────────────────────────────────────────────────────────────────────────────
def test_simulated_tm_docstrings_match_base():
    """
    Public methods in SimulatedTranscriptManager should copy the real
    BaseTranscriptManager doc-strings one-for-one (via functools.wraps).
    """
    from unity.transcript_manager.base import BaseTranscriptManager
    from unity.transcript_manager.simulated import SimulatedTranscriptManager

    assert (
        BaseTranscriptManager.ask.__doc__.strip()
        in SimulatedTranscriptManager.ask.__doc__.strip()
    ), ".store doc-string was not copied correctly"


# ────────────────────────────────────────────────────────────────────────────
# 8.  Pause → Resume round-trip + valid_tools                                #
# ────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@_handle_project
async def test_pause_and_resume_simulated_tm(monkeypatch):
    """
    Ensure a `_SimulatedTranscriptHandle` can be paused and resumed and that
    `valid_tools` flips correctly between the two states.
    """
    counts = {"pause": 0, "resume": 0}

    # --- patch pause -------------------------------------------------------
    orig_pause = _SimulatedTranscriptHandle.pause

    @functools.wraps(orig_pause)
    def _patched_pause(self):  # type: ignore[override]
        counts["pause"] += 1
        return orig_pause(self)

    monkeypatch.setattr(
        _SimulatedTranscriptHandle,
        "pause",
        _patched_pause,
        raising=True,
    )

    # --- patch resume ------------------------------------------------------
    orig_resume = _SimulatedTranscriptHandle.resume

    @functools.wraps(orig_resume)
    def _patched_resume(self):  # type: ignore[override]
        counts["resume"] += 1
        return orig_resume(self)

    monkeypatch.setattr(
        _SimulatedTranscriptHandle,
        "resume",
        _patched_resume,
        raising=True,
    )

    tm = SimulatedTranscriptManager()
    handle = await tm.ask("List unread DMs.")

    # Initially, pause should be available and resume absent.
    tools_initial = handle.valid_tools
    assert "pause" in tools_initial and "resume" not in tools_initial

    # Pause the handle.
    pause_reply = handle.pause()
    assert "pause" in pause_reply.lower()

    tools_paused = handle.valid_tools
    assert "resume" in tools_paused and "pause" not in tools_paused

    # Start result() – it should block while paused.
    res_task = asyncio.create_task(handle.result())
    await asyncio.sleep(0.1)
    assert not res_task.done(), "result() must wait while paused"

    # Resume and ensure execution proceeds.
    resume_reply = handle.resume()
    assert "resume" in resume_reply.lower() or "running" in resume_reply.lower()

    tools_running = handle.valid_tools
    assert "pause" in tools_running and "resume" not in tools_running

    answer = await asyncio.wait_for(res_task, timeout=60)
    assert isinstance(answer, str) and answer.strip()

    # Each steering method should have been called exactly once.
    assert counts == {
        "pause": 1,
        "resume": 1,
    }, "pause/resume should each be called once"
