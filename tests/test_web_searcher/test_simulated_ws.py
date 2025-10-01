from __future__ import annotations

import asyncio
import functools
import pytest

from unity.web_searcher.simulated import (
    SimulatedWebSearcher,
    _SimulatedWebSearcherHandle,
)

# keeps each test isolated in its own Unify project / trace context
from tests.helpers import _handle_project


# ────────────────────────────────────────────────────────────────────────────
# 1.  Basic start-and-ask                                                    #
# ────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@_handle_project
async def test_start_and_ask_simulated_ws():
    ws = SimulatedWebSearcher("Demo web-search for unit-tests.")
    h = await ws.ask("What happened in vector DBs in Q1 2025?")
    answer = await h.result()
    assert isinstance(answer, str) and answer.strip(), "Answer should be non-empty"


# ────────────────────────────────────────────────────────────────────────────
# 2.  Stateful memory – serial asks                                         #
# ────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@_handle_project
async def test_ws_stateful_memory_serial_asks():
    """
    Two consecutive .ask() calls share context because the manager keeps a
    stateful LLM.
    """
    ws = SimulatedWebSearcher()

    h1 = await ws.ask(
        "Please propose a short unique report code for my research, "
        "and reply with only that code.",
    )
    code = (await h1.result()).strip()
    assert code, "Code should not be empty"

    h2 = await ws.ask("Great. What code did you just propose?")
    answer2 = (await h2.result()).lower()
    assert code.lower() in answer2, "LLM should recall the code it generated"


# ────────────────────────────────────────────────────────────────────────────
# Steerable handle tests                                                     #
# ────────────────────────────────────────────────────────────────────────────


# ────────────────────────────────────────────────────────────────────────────
# 3.  Interject                                                             #
# ────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@_handle_project
async def test_handle_interject(monkeypatch):
    calls = {"interject": 0}
    orig = _SimulatedWebSearcherHandle.interject

    @functools.wraps(orig)
    def wrapped(self, msg: str) -> str:  # type: ignore[override]
        calls["interject"] += 1
        return orig(self, msg)

    monkeypatch.setattr(
        _SimulatedWebSearcherHandle,
        "interject",
        wrapped,
        raising=True,
    )

    ws = SimulatedWebSearcher()
    h = await ws.ask("Summarize latest announcements from major vendors.")
    await asyncio.sleep(0.05)
    reply = h.interject("Prefer primary sources and release notes.")
    assert "ack" in reply.lower() or "noted" in reply.lower()
    await h.result()
    assert calls["interject"] == 1, ".interject should be invoked exactly once"


# ────────────────────────────────────────────────────────────────────────────
# 4.  Stop                                                                  #
# ────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@_handle_project
async def test_handle_stop():
    ws = SimulatedWebSearcher()
    h = await ws.ask("Generate a long market analysis.")
    await asyncio.sleep(0.05)
    h.stop()
    await h.result()
    assert h.done(), "Handle should report done after stop()"


# ────────────────────────────────────────────────────────────────────────────
# 5.  Clarification handshake                                               #
# ────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@_handle_project
async def test_handle_requests_clarification():
    ws = SimulatedWebSearcher()

    up_q: asyncio.Queue[str] = asyncio.Queue()
    down_q: asyncio.Queue[str] = asyncio.Queue()

    h = await ws.ask(
        "Find the latest updates for my client's product,",
        clarification_up_q=up_q,
        clarification_down_q=down_q,
        _requests_clarification=True,
    )

    question = await asyncio.wait_for(up_q.get(), timeout=60)
    assert "clarify" in question.lower()
    await down_q.put("Focus on European updates only.")

    answer = await h.result()
    assert isinstance(answer, str) and answer.strip()
    assert "europe" in answer.lower()


# ────────────────────────────────────────────────────────────────────────────
# 6.  Pause → Resume round-trip                                             #
# ────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@_handle_project
async def test_handle_pause_and_resume(monkeypatch):
    call_counts = {"pause": 0, "resume": 0}

    original_pause = _SimulatedWebSearcherHandle.pause

    @functools.wraps(original_pause)
    def _patched_pause(self):  # type: ignore[override]
        call_counts["pause"] += 1
        return original_pause(self)

    monkeypatch.setattr(
        _SimulatedWebSearcherHandle,
        "pause",
        _patched_pause,
        raising=True,
    )

    original_resume = _SimulatedWebSearcherHandle.resume

    @functools.wraps(original_resume)
    def _patched_resume(self):  # type: ignore[override]
        call_counts["resume"] += 1
        return original_resume(self)

    monkeypatch.setattr(
        _SimulatedWebSearcherHandle,
        "resume",
        _patched_resume,
        raising=True,
    )

    ws = SimulatedWebSearcher()
    handle = await ws.ask("Give a concise overview of Q1 trends.")

    pause_msg = handle.pause()
    assert "pause" in pause_msg.lower()

    res_task = asyncio.create_task(handle.result())
    await asyncio.sleep(0.1)
    assert not res_task.done()

    resume_msg = handle.resume()
    assert "resume" in resume_msg.lower() or "running" in resume_msg.lower()

    answer = await asyncio.wait_for(res_task, timeout=60)
    assert isinstance(answer, str) and answer.strip()

    assert call_counts == {"pause": 1, "resume": 1}


# ────────────────────────────────────────────────────────────────────────────
# 7.  Nested ask on handle                                                   #
# ────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@_handle_project
async def test_handle_ask():
    ws = SimulatedWebSearcher()

    handle = await ws.ask("Summarize key research findings.")

    handle.interject("Focus on European enterprise accounts.")

    nested = await handle.ask("What is the key point to emphasize?")

    nested_answer = await nested.result()
    assert isinstance(nested_answer, str) and nested_answer.strip()
    assert "europe" in nested_answer.lower()

    handle_answer = await handle.result()
    assert isinstance(handle_answer, str) and handle_answer.strip()
