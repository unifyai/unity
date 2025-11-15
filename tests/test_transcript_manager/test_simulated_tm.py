from __future__ import annotations

import asyncio
import pytest
import functools

from unity.transcript_manager.simulated import (
    SimulatedTranscriptManager,
    _SimulatedTranscriptHandle,
)

# Helper identical to the one used elsewhere in the test-suite
from tests.helpers import (
    _handle_project,
    _ack_ok,
    _assert_blocks_while_paused,
    DEFAULT_TIMEOUT,
    _unique_token,
)


# ────────────────────────────────────────────────────────────────────────────
# 1.  Doc-string inheritance                                                 #
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
# 2.  Basic start-and-ask                                                    #
# ────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@_handle_project
async def test_start_and_ask_simulated_tm():
    tm = SimulatedTranscriptManager("Demo transcript DB.")
    handle = await tm.ask("Show me my unread emails.")
    answer = await handle.result()
    assert isinstance(answer, str) and answer.strip(), "Answer should be non-empty"


# ────────────────────────────────────────────────────────────────────────────
# 3.  Stateful memory – serial asks                                         #
# ────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@_handle_project
async def test_tm_stateful_memory_serial_asks():
    """
    Two consecutive .ask() calls should share the same conversation context
    because the manager's LLM is stateful.

    To reduce brittleness from formatting/phrasing, we seed the first turn with
    a unique token and then require that the second turn recalls that exact
    token somewhere in its response.
    """
    tm = SimulatedTranscriptManager()

    # 1) Seed a unique token inside a realistic transcript request
    token = _unique_token("TICKET")
    handle1 = await tm.ask(
        "Please produce exactly one realistic transcript message. "
        f"Ensure the message content includes this exact ticket number verbatim: {token}",
    )
    first_answer = (await handle1.result()).strip()
    assert first_answer, "First answer should not be empty"

    # 2) Ask the LLM to recall the previously mentioned token
    handle2 = await tm.ask(
        "What ticket number did you mention earlier? Quote it verbatim in your answer.",
    )
    answer2 = await handle2.result()
    assert (
        isinstance(answer2, str) and answer2.strip()
    ), "Second answer should be non-empty"

    # The second answer should mention the exact token (substring check is robust to formatting)
    assert token in answer2, "LLM should recall the previously mentioned token"


# ────────────────────────────────────────────────────────────────────────────
# Steerable handle tests                                                     #
# ────────────────────────────────────────────────────────────────────────────


# ────────────────────────────────────────────────────────────────────────────
# 4.  Interject                                                             #
# ────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@_handle_project
async def test_handle_interject(monkeypatch):
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
    assert _ack_ok(reply)

    await handle.result()
    assert counts["interject"] == 1, ".interject should be called exactly once"


# ────────────────────────────────────────────────────────────────────────────
# 5.  Stop                                                                  #
# ────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@_handle_project
async def test_handle_stop():
    tm = SimulatedTranscriptManager()
    handle = await tm.ask("Produce a full export of all messages.")
    await asyncio.sleep(0.05)
    handle.stop()
    await handle.result()
    assert handle.done(), "Handle should report done after stop()"


# ────────────────────────────────────────────────────────────────────────────
# 6.  Clarification handshake                                               #
# ────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@_handle_project
async def test_handle_requests_clarification():
    tm = SimulatedTranscriptManager()

    up_q: asyncio.Queue[str] = asyncio.Queue()
    down_q: asyncio.Queue[str] = asyncio.Queue()

    handle = await tm.ask(
        "Find important messages.",
        _clarification_up_q=up_q,
        _clarification_down_q=down_q,
        _requests_clarification=True,
    )

    # Must ask for clarification first
    question = await asyncio.wait_for(up_q.get(), timeout=DEFAULT_TIMEOUT)
    assert "clarify" in question.lower()

    # Provide clarification
    await down_q.put("Focus on project Alpha deadlines.")
    answer = await handle.result()

    assert isinstance(answer, str) and answer.strip(), "Answer should not be empty"


# ────────────────────────────────────────────────────────────────────────────
# 7.  Pause → Resume round-trip                                              #
# ────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@_handle_project
async def test_handle_pause_and_resume(monkeypatch):
    """
    Ensure a `_SimulatedTranscriptHandle` can be paused and resumed.
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

    # Pause the handle.
    pause_reply = handle.pause()
    assert "pause" in pause_reply.lower()

    # Start result() – it should block while paused.
    res_task = await _assert_blocks_while_paused(handle.result())

    # Resume and ensure execution proceeds.
    resume_reply = handle.resume()
    assert "resume" in resume_reply.lower() or "running" in resume_reply.lower()

    answer = await asyncio.wait_for(res_task, timeout=DEFAULT_TIMEOUT)
    assert isinstance(answer, str) and answer.strip()

    # Each steering method should have been called exactly once.
    assert counts == {
        "pause": 1,
        "resume": 1,
    }, "pause/resume should each be called once"


# ────────────────────────────────────────────────────────────────────────────
# 8.  Nested ask on handle                                                   #
# ────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@_handle_project
async def test_handle_ask():
    """
    The internal handle returned by SimulatedTranscriptManager.ask exposes a
    dynamic ask() method that should produce a nested handle whose result can
    be awaited independently of the parent.
    """
    tm = SimulatedTranscriptManager()

    # Start an initial ask to obtain the live handle
    handle = await tm.ask("Summarize all unread messages this week.")

    # Add extra context to ensure nested prompt includes it
    handle.interject("Focus on European enterprise accounts.")

    # Invoke the dynamic ask on the running handle
    nested = await handle.ask("What is the key point to emphasize?")

    nested_answer = await nested.result()
    assert isinstance(nested_answer, str) and nested_answer.strip(), (
        "Nested ask() should yield a non-empty string answer",
    )
    assert any(substr in nested_answer.lower() for substr in ("europe", "eu"))

    # The original handle should still be awaitable and produce an answer
    handle_answer = await handle.result()
    assert isinstance(handle_answer, str) and handle_answer.strip(), (
        "Handle should still yield a non-empty answer after nested ask",
    )
