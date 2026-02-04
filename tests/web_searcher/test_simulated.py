from __future__ import annotations

from dotenv import load_dotenv

load_dotenv(override=True)

import asyncio
import functools
import pytest

from unity.web_searcher.simulated import (
    SimulatedWebSearcher,
    _SimulatedWebSearcherHandle,
)

# keeps each test isolated in its own Unify project / trace context
from tests.helpers import (
    _handle_project,
    _ack_ok,
    _assert_blocks_while_paused,
    DEFAULT_TIMEOUT,
)


# ────────────────────────────────────────────────────────────────────────────
# 0.  Doc-string inheritance                                                 #
# ────────────────────────────────────────────────────────────────────────────
def test_docstrings_match_base():
    """
    Public methods in SimulatedWebSearcher should copy the real
    BaseWebSearcher doc-strings one-for-one (via functools.wraps).
    """
    from unity.web_searcher.base import BaseWebSearcher
    from unity.web_searcher.simulated import SimulatedWebSearcher

    assert (
        BaseWebSearcher.ask.__doc__.strip() in SimulatedWebSearcher.ask.__doc__.strip()
    ), ".ask doc-string was not copied correctly"

    assert (
        BaseWebSearcher.update.__doc__.strip()
        in SimulatedWebSearcher.update.__doc__.strip()
    ), ".update doc-string was not copied correctly"


# ────────────────────────────────────────────────────────────────────────────
# 1.  Basic start-and-ask                                                    #
# ────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@_handle_project
async def test_start_and_ask():
    ws = SimulatedWebSearcher("Demo web-search for unit-tests.")
    h = await ws.ask("What happened in vector DBs in Q1 2025?")
    answer = await h.result()
    assert isinstance(answer, str) and answer.strip(), "Answer should be non-empty"


# ────────────────────────────────────────────────────────────────────────────
# 2.  Stateful memory – serial asks                                         #
# ────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@_handle_project
async def test_stateful_memory_serial_asks():
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
async def test_interject(monkeypatch):
    calls = {"interject": 0}
    orig = _SimulatedWebSearcherHandle.interject

    @functools.wraps(orig)
    async def wrapped(self, msg: str, **kwargs) -> str:  # type: ignore[override]
        calls["interject"] += 1
        return await orig(self, msg, **kwargs)

    monkeypatch.setattr(
        _SimulatedWebSearcherHandle,
        "interject",
        wrapped,
        raising=True,
    )

    ws = SimulatedWebSearcher()
    h = await ws.ask("Summarize latest announcements from major vendors.")
    await asyncio.sleep(0.05)
    reply = await h.interject("Prefer primary sources and release notes.")
    assert _ack_ok(reply)
    await h.result()
    assert calls["interject"] == 1, ".interject should be invoked exactly once"


# ────────────────────────────────────────────────────────────────────────────
# 4.  Stop                                                                  #
# ────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@_handle_project
async def test_stop():
    ws = SimulatedWebSearcher()
    h = await ws.ask("Generate a long market analysis.")
    await asyncio.sleep(0.05)
    await h.stop()
    await h.result()
    assert h.done(), "Handle should report done after stop()"


# ────────────────────────────────────────────────────────────────────────────
# 5.  Clarification handshake                                               #
# ────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@_handle_project
async def test_requests_clarification():
    ws = SimulatedWebSearcher()

    up_q: asyncio.Queue[str] = asyncio.Queue()
    down_q: asyncio.Queue[str] = asyncio.Queue()

    h = await ws.ask(
        "Find the latest updates on the Toys R US social media accounts,",
        _clarification_up_q=up_q,
        _clarification_down_q=down_q,
        _requests_clarification=True,
    )

    question = await asyncio.wait_for(up_q.get(), timeout=DEFAULT_TIMEOUT)
    assert "clarify" in question.lower()
    await down_q.put(
        "Let's focus on Twitter, and let me know if the most recent was a tweet or a retweet.",
    )

    answer = await h.result()
    assert isinstance(answer, str) and answer.strip()
    assert "tweet" in answer.lower()


# ────────────────────────────────────────────────────────────────────────────
# 6.  Pause → Resume round-trip                                             #
# ────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@_handle_project
async def test_pause_and_resume(monkeypatch):
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

    pause_msg = await handle.pause()
    assert "pause" in pause_msg.lower()

    res_task = asyncio.create_task(handle.result())
    await _assert_blocks_while_paused(res_task)

    resume_msg = await handle.resume()
    assert "resume" in resume_msg.lower() or "running" in resume_msg.lower()

    answer = await res_task
    assert isinstance(answer, str) and answer.strip()

    assert call_counts == {"pause": 1, "resume": 1}


# ────────────────────────────────────────────────────────────────────────────
# 7.  Nested ask on handle                                                   #
# ────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@_handle_project
async def test_nested_ask():
    ws = SimulatedWebSearcher()

    handle = await ws.ask("Summarize key research findings.")

    await handle.interject("Focus on European enterprise accounts.")

    nested = await handle.ask("What is the key point to emphasize?")

    nested_answer = await nested.result()
    assert isinstance(nested_answer, str) and nested_answer.strip()
    assert any(substr in nested_answer.lower() for substr in ("europe", "eu"))

    handle_answer = await handle.result()
    assert isinstance(handle_answer, str) and handle_answer.strip()


# ────────────────────────────────────────────────────────────────────────────
# 8.  Structured output with response_format                                  #
# ────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@_handle_project
async def test_ask_with_response_format():
    from pydantic import BaseModel, Field

    class SimpleSummary(BaseModel):
        summary: str = Field(..., description="One-sentence summary")

    ws = SimulatedWebSearcher()
    h = await ws.ask(
        "Provide a one-sentence summary of a recent technology news item; output JSON matching the provided schema.",
        response_format=SimpleSummary,
    )
    final = await h.result()

    parsed = SimpleSummary.model_validate_json(final)
    assert isinstance(parsed.summary, str) and parsed.summary.strip() != ""


@_handle_project
def test_clear_reinitialises():
    """
    Ensure SimulatedWebSearcher.clear re-runs the constructor (fresh stateful LLM
    and tools mapping stays provisioned).
    """
    from unity.web_searcher.simulated import SimulatedWebSearcher

    sim = SimulatedWebSearcher()
    old_llm = getattr(sim, "_llm", None)
    assert old_llm is not None
    assert isinstance(sim._ask_tools, dict) and sim._ask_tools

    sim.clear()

    # After clear, llm handle should be replaced and tools still present
    assert getattr(sim, "_llm", None) is not None and sim._llm is not old_llm
    assert isinstance(sim._ask_tools, dict) and sim._ask_tools


# ────────────────────────────────────────────────────────────────────────────
# 9.  Update – basic completion                                              #
# ────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@_handle_project
async def test_update_basic_completion():
    """
    SimulatedWebSearcher.update should return a live handle that completes.
    """
    ws = SimulatedWebSearcher()
    h = await ws.update(
        "Create a website entry for host=example.com with tags ['docs', 'security']",
    )
    answer = await asyncio.wait_for(h.result(), timeout=DEFAULT_TIMEOUT)
    assert isinstance(answer, str) and answer.strip()


# ────────────────────────────────────────────────────────────────────────────
# 10.  Stop while paused should finish immediately                           #
# ────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@_handle_project
async def test_stop_while_paused_finishes_immediately():
    ws = SimulatedWebSearcher()
    h = await ws.ask("Generate a very long market report.")
    await h.pause()
    res_task = asyncio.create_task(h.result())
    await asyncio.sleep(0.1)
    assert not res_task.done()
    await h.stop("cancelled")
    out = await asyncio.wait_for(res_task, timeout=DEFAULT_TIMEOUT)
    assert isinstance(out, str)
    assert h.done()


# ────────────────────────────────────────────────────────────────────────────
# 11.  Stop while waiting for clarification should finish immediately         #
# ────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@_handle_project
async def test_stop_while_waiting_for_clarification_finishes_immediately():
    ws = SimulatedWebSearcher()
    up_q: asyncio.Queue[str] = asyncio.Queue()
    down_q: asyncio.Queue[str] = asyncio.Queue()
    h = await ws.ask(
        "Find recent coverage.",
        _clarification_up_q=up_q,
        _clarification_down_q=down_q,
        _requests_clarification=True,
    )
    q = await asyncio.wait_for(up_q.get(), timeout=DEFAULT_TIMEOUT)
    assert "clarify" in q.lower()
    await h.stop("no longer needed")
    out = await asyncio.wait_for(h.result(), timeout=DEFAULT_TIMEOUT)
    assert isinstance(out, str)
    assert h.done()
