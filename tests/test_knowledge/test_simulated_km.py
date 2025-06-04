from __future__ import annotations

import asyncio
import pytest
import functools

from unity.knowledge_manager.simulated import (
    SimulatedKnowledgeManager,
    _SimulatedKnowledgeHandle,
)

# helper that wraps each test in its own Unify project / trace context
from tests.helpers import _handle_project


# ────────────────────────────────────────────────────────────────────────────
# 1.  Basic start-and-retrieve                                               #
# ────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@_handle_project
async def test_start_and_retrieve_simulated_km():
    km = SimulatedKnowledgeManager("Demo KB for unit-tests.")
    h = km.retrieve("What do we already know about Mars?")
    ans = await h.result()
    assert isinstance(ans, str) and ans.strip(), "Answer should be non-empty"


# ────────────────────────────────────────────────────────────────────────────
# 2.  Interject                                                             #
# ────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@_handle_project
async def test_interject_simulated_km(monkeypatch):
    calls = {"interject": 0}
    orig = _SimulatedKnowledgeHandle.interject

    @functools.wraps(orig)
    def wrapped(self, msg: str) -> str:  # type: ignore[override]
        calls["interject"] += 1
        return orig(self, msg)

    monkeypatch.setattr(_SimulatedKnowledgeHandle, "interject", wrapped, raising=True)

    km = SimulatedKnowledgeManager()
    h = km.retrieve("Show me all facts about Paris.")
    await asyncio.sleep(0.05)
    reply = h.interject("Only include historical facts.")
    assert "ack" in reply.lower() or "noted" in reply.lower()
    await h.result()
    assert calls["interject"] == 1, ".interject should be called exactly once"


# ────────────────────────────────────────────────────────────────────────────
# 3.  Stop                                                                  #
# ────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@_handle_project
async def test_stop_simulated_km():
    km = SimulatedKnowledgeManager()
    h = km.retrieve("Generate a 100-page report of all knowledge.")
    await asyncio.sleep(0.05)
    h.stop()
    with pytest.raises(asyncio.CancelledError):
        await h.result()
    assert h.done(), "Handle should report done after stop()"


# ────────────────────────────────────────────────────────────────────────────
# 4.  Clarification handshake                                               #
# ────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@_handle_project
async def test_km_requests_clarification():
    km = SimulatedKnowledgeManager()

    up_q: asyncio.Queue[str] = asyncio.Queue()
    down_q: asyncio.Queue[str] = asyncio.Queue()

    h = km.retrieve(
        "Please summarise the knowledge base.",
        clarification_up_q=up_q,
        clarification_down_q=down_q,
    )

    question = await asyncio.wait_for(up_q.get(), timeout=30)
    assert "clarify" in question.lower()

    await down_q.put("Focus on scientific facts.")
    ans = await h.result()
    assert isinstance(ans, str) and ans.strip()
    assert "science" in ans.lower() or "scientific" in ans.lower()


# ────────────────────────────────────────────────────────────────────────────
# 5.  Stateful memory – store then retrieve                                  #
# ────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@_handle_project
async def test_km_stateful_store_then_retrieve():
    """
    A fact stored via .store() should be recalled by a later .retrieve().
    """
    km = SimulatedKnowledgeManager()
    fact = "The CEO of Acme Corp is Jane Doe."

    # store a new fact
    h_store = km.store(fact)
    await h_store.result()

    # retrieve it
    h_ret = km.retrieve("Who is the CEO of Acme Corp?")
    answer = (await h_ret.result()).lower()
    assert "jane" in answer and "doe" in answer


# ────────────────────────────────────────────────────────────────────────────
# 6.  Stateful memory – serial retrieves                                     #
# ────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@_handle_project
async def test_km_stateful_serial_retrieves():
    """
    Two consecutive .retrieve() calls should share context.
    """
    km = SimulatedKnowledgeManager()

    # first question – ask for a single‐word theme of the KB
    h1 = km.retrieve(
        "Using one word only, how would you describe the overall theme of our knowledge base?",
    )
    theme = (await h1.result()).strip()
    assert theme, "Theme word should not be empty"

    # follow-up question
    h2 = km.retrieve(
        "What single word did you just use to describe the knowledge base?",
    )
    ans2 = (await h2.result()).lower()
    assert theme.lower() in ans2, "LLM should recall the theme it produced earlier"


# ────────────────────────────────────────────────────────────────────────────
# 7.  Doc-string inheritance                                                 #
# ────────────────────────────────────────────────────────────────────────────
def test_simulated_km_docstrings_match_real():
    """
    Public methods in SimulatedKnowledgeManager should copy the real
    BaseKnowledgeManager doc-strings one-for-one (via functools.wraps).
    """
    from unity.knowledge_manager.base import BaseKnowledgeManager
    from unity.knowledge_manager.simulated import SimulatedKnowledgeManager

    assert (
        SimulatedKnowledgeManager.store.__doc__.strip()
        == BaseKnowledgeManager.store.__doc__.strip()
    ), ".store doc-string was not copied correctly"

    assert (
        SimulatedKnowledgeManager.retrieve.__doc__.strip()
        == BaseKnowledgeManager.retrieve.__doc__.strip()
    ), ".retrieve doc-string was not copied correctly"


# ────────────────────────────────────────────────────────────────────────────
# 8.  Pause → Resume round-trip + valid_tools                                #
# ────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@_handle_project
async def test_pause_and_resume_simulated_km(monkeypatch):
    """
    Ensure a `_SimulatedKnowledgeHandle` can be paused and resumed and that
    its `valid_tools` property flips appropriately.
    """
    counts = {"pause": 0, "resume": 0}

    # --- patch pause -------------------------------------------------------
    orig_pause = _SimulatedKnowledgeHandle.pause

    @functools.wraps(orig_pause)
    def _patched_pause(self):  # type: ignore[override]
        counts["pause"] += 1
        return orig_pause(self)

    monkeypatch.setattr(
        _SimulatedKnowledgeHandle,
        "pause",
        _patched_pause,
        raising=True,
    )

    # --- patch resume ------------------------------------------------------
    orig_resume = _SimulatedKnowledgeHandle.resume

    @functools.wraps(orig_resume)
    def _patched_resume(self):  # type: ignore[override]
        counts["resume"] += 1
        return orig_resume(self)

    monkeypatch.setattr(
        _SimulatedKnowledgeHandle,
        "resume",
        _patched_resume,
        raising=True,
    )

    km = SimulatedKnowledgeManager()
    handle = km.retrieve("Summarise everything we know about quantum gravity.")

    # Before pausing: pause should be available, resume not.
    tools_initial = handle.valid_tools
    assert "pause" in tools_initial and "resume" not in tools_initial

    # Pause the handle
    pause_msg = handle.pause()
    assert "paused" in pause_msg.lower()

    # After pausing: resume should be available, pause not.
    tools_paused = handle.valid_tools
    assert "resume" in tools_paused and "pause" not in tools_paused

    # Start result() while still paused – it should await
    res_task = asyncio.create_task(handle.result())
    await asyncio.sleep(0.1)
    assert not res_task.done(), "result() must block while paused"

    # Resume execution
    resume_msg = handle.resume()
    assert "resume" in resume_msg.lower() or "running" in resume_msg.lower()

    # After resuming: pause available again, resume gone.
    tools_running = handle.valid_tools
    assert "pause" in tools_running and "resume" not in tools_running

    # Now result() should finish
    answer = await asyncio.wait_for(res_task, timeout=30)
    assert isinstance(answer, str) and answer.strip()

    # Each steering method must have been invoked exactly once
    assert counts == {"pause": 1, "resume": 1}, "pause/resume must each be called once"
