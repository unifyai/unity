from __future__ import annotations

import asyncio
import pytest
import functools

from unity.knowledge_manager.simulated import (
    SimulatedKnowledgeManager,
    _SimulatedKnowledgeHandle,
)

# helper that wraps each test in its own Unify project / trace context
from tests.helpers import (
    _handle_project,
    _ack_ok,
    _assert_blocks_while_paused,
    DEFAULT_TIMEOUT,
)


# ────────────────────────────────────────────────────────────────────────────
# 1.  Doc-string inheritance                                                 #
# ────────────────────────────────────────────────────────────────────────────
def test_simulated_km_docstrings_match_base():
    """
    Public methods in SimulatedKnowledgeManager should copy the real
    BaseKnowledgeManager doc-strings one-for-one (via functools.wraps).
    """
    from unity.knowledge_manager.base import BaseKnowledgeManager
    from unity.knowledge_manager.simulated import SimulatedKnowledgeManager

    assert (
        BaseKnowledgeManager.ask.__doc__.strip()
        in SimulatedKnowledgeManager.ask.__doc__.strip()
    ), ".retrieve doc-string was not copied correctly"

    assert (
        BaseKnowledgeManager.update.__doc__.strip()
        in SimulatedKnowledgeManager.update.__doc__.strip()
    ), ".store doc-string was not copied correctly"

    assert (
        BaseKnowledgeManager.refactor.__doc__.strip()
        in SimulatedKnowledgeManager.refactor.__doc__.strip()
    ), ".refactor doc-string was not copied correctly"


# ────────────────────────────────────────────────────────────────────────────
# 2.  Basic start-and-ask                                                    #
# ────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@_handle_project
async def test_start_and_ask_simulated_km():
    km = SimulatedKnowledgeManager("Demo KB for unit-tests.")
    handle = await km.ask("What do we already know about Zebulon?")
    answer = await handle.result()
    assert isinstance(answer, str) and answer.strip(), "Answer should be non-empty"


# ────────────────────────────────────────────────────────────────────────────
# 3.  Stateful memory – serial retrieves                                     #
# ────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@_handle_project
async def test_km_stateful_serial_retrieves():
    """
    Two consecutive .retrieve() calls should share context.
    """
    km = SimulatedKnowledgeManager()

    # first question – ask for a single‐word theme of the KB
    h1 = await km.ask(
        "Using one word only, how would you describe the overall theme of our knowledge base?",
    )
    theme = (await h1.result()).strip()
    assert theme, "Theme word should not be empty"

    # follow-up question
    h2 = await km.ask(
        "What single word did you just use to describe the knowledge base?",
    )
    ans2 = (await h2.result()).lower()
    assert theme.lower() in ans2, "LLM should recall the theme it produced earlier"


# ────────────────────────────────────────────────────────────────────────────
# 4.  Update then ask – state carries through                                #
# ────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@_handle_project
async def test_km_stateful_update_then_retrieve():
    """
    A fact stored via .store() should be recalled by a later .retrieve().
    """
    km = SimulatedKnowledgeManager()
    fact = "The flagship product of Acme Corp is the Quantum Widget."

    # store a new fact
    h_store = await km.update(fact)
    await h_store.result()

    # retrieve it
    h_ret = await km.ask("What is the flagship product of Acme Corp?")
    answer = (await h_ret.result()).lower()
    assert "quantum" in answer and "widget" in answer


# ────────────────────────────────────────────────────────────────────────────
# 5.  Basic refactor                                                         #
# ────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@_handle_project
async def test_refactor_simulated_km():
    """
    The simulated KM should return a non-empty migration plan in response to
    .refactor().  We do **not** verify the content in detail – only that the
    string is present and mentions something schema-related (e.g. "column").
    """
    km = SimulatedKnowledgeManager(
        "Tiny demo KB where Contacts duplicate company opening hours.",
    )

    handle = await km.refactor(
        "Please remove duplicated columns and introduce proper primary keys.",
    )
    migration_plan = await handle.result()

    assert (
        isinstance(migration_plan, str) and migration_plan.strip()
    ), "Migration plan should be non-empty"
    assert "column" in migration_plan.lower() or "table" in migration_plan.lower(), (
        "Plan should mention schema elements (columns/tables).",
        migration_plan,
    )


# ────────────────────────────────────────────────────────────────────────────
# Steerable handle tests                                                     #
# ────────────────────────────────────────────────────────────────────────────


# ────────────────────────────────────────────────────────────────────────────
# 6.  Interject                                                             #
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
    handle = await km.ask("Show me all facts about Zebulon.")
    await asyncio.sleep(0.05)
    reply = handle.interject("Only include historical facts.")
    assert _ack_ok(reply)
    await handle.result()
    assert calls["interject"] == 1, ".interject should be called exactly once"


# ────────────────────────────────────────────────────────────────────────────
# 7.  Stop                                                                  #
# ────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@_handle_project
async def test_stop_simulated_km():
    km = SimulatedKnowledgeManager()
    handle = await km.ask("Generate a 100-page report of all knowledge.")
    await asyncio.sleep(0.05)
    handle.stop()
    await handle.result()
    assert handle.done(), "Handle should report done after stop()"


# ────────────────────────────────────────────────────────────────────────────
# 8.  Clarification handshake                                               #
# ────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@_handle_project
async def test_km_requests_clarification():
    km = SimulatedKnowledgeManager()

    up_q: asyncio.Queue[str] = asyncio.Queue()
    down_q: asyncio.Queue[str] = asyncio.Queue()

    handle = await km.ask(
        "Please summarise the knowledge base.",
        _clarification_up_q=up_q,
        _clarification_down_q=down_q,
        _requests_clarification=True,
    )

    question = await asyncio.wait_for(up_q.get(), timeout=DEFAULT_TIMEOUT)
    assert "clarify" in question.lower()

    await down_q.put("Focus on scientific facts.")
    answer = await handle.result()
    assert isinstance(answer, str) and answer.strip()
    assert "science" in answer.lower() or "scientific" in answer.lower()


# ────────────────────────────────────────────────────────────────────────────
# 9.  Pause → Resume round-trip                                              #
# ────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@_handle_project
async def test_pause_and_resume_simulated_km(monkeypatch):
    """
    Ensure a `_SimulatedKnowledgeHandle` can be paused and resumed.
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
    handle = await km.ask("Summarise everything we know about quantum gravity.")

    # Pause the handle
    pause_msg = await handle.pause()
    assert "pause" in pause_msg.lower() or "paused" in pause_msg.lower()

    # Start result() while still paused – it should await
    res_task = await _assert_blocks_while_paused(handle.result())

    # Resume execution
    resume_msg = await handle.resume()
    assert "resume" in resume_msg.lower() or "running" in resume_msg.lower()

    # Now result() should finish
    answer = await asyncio.wait_for(res_task, timeout=DEFAULT_TIMEOUT)
    assert isinstance(answer, str) and answer.strip()

    # Each steering method must have been invoked exactly once
    assert counts == {"pause": 1, "resume": 1}, "pause/resume must each be called once"


# ────────────────────────────────────────────────────────────────────────────
# 10. Nested ask on handle                                                   #
# ────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@_handle_project
async def test_handle_ask():
    """
    The internal handle returned by SimulatedKnowledgeManager.ask exposes a
    dynamic ask() method that should produce a nested handle whose result can
    be awaited independently of the parent.
    """
    km = SimulatedKnowledgeManager()

    # Start an initial ask to obtain the live handle
    handle = await km.ask("Summarize all relevant knowledge this quarter.")

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


# ────────────────────────────────────────────────────────────────────────────
# 11. Clear – reset and remain usable                                         #
# ────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@_handle_project
async def test_simulated_clear():
    """
    SimulatedKnowledgeManager.clear should reset the manager (hard-coded completion)
    and remain usable afterwards.
    """
    km = SimulatedKnowledgeManager()
    # Seed some prior state via an update call
    h_store = await km.update("Add a temporary fact about Project Phoenix.")
    await asyncio.wait_for(h_store.result(), timeout=DEFAULT_TIMEOUT)

    # Clear should not raise and should be quick (no LLM roundtrip requirement)
    km.clear()

    # Post-clear, an ask should still work
    h_q = await km.ask("List any knowledge stored today.")
    answer = await asyncio.wait_for(h_q.result(), timeout=DEFAULT_TIMEOUT)
    assert (
        isinstance(answer, str) and answer.strip()
    ), "Answer should be non-empty after clear()"


@_handle_project
def test_simulated_knowledge_manager_reduce_shapes():
    km = SimulatedKnowledgeManager()

    scalar = km.reduce(table="Content", metric="sum", keys="row_id")
    assert isinstance(scalar, (int, float))

    multi = km.reduce(table="Content", metric="max", keys=["row_id"])
    assert isinstance(multi, dict)
    assert set(multi.keys()) == {"row_id"}

    grouped = km.reduce(
        table="Content",
        metric="sum",
        keys="row_id",
        group_by="row_id",
    )
    assert isinstance(grouped, dict)


# ────────────────────────────────────────────────────────────────────────────
# 12.  Stop while paused should finish immediately                           #
# ────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@_handle_project
async def test_stop_while_paused_finishes_immediately():
    km = SimulatedKnowledgeManager()
    h = await km.ask("Generate a long knowledge export.")
    await h.pause()
    res_task = asyncio.create_task(h.result())
    await asyncio.sleep(0.1)
    assert not res_task.done()
    h.stop("cancelled by user")
    out = await asyncio.wait_for(res_task, timeout=DEFAULT_TIMEOUT)
    assert isinstance(out, str)
    assert h.done()


# ────────────────────────────────────────────────────────────────────────────
# 13.  Stop while waiting for clarification should finish immediately         #
# ────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@_handle_project
async def test_stop_while_waiting_for_clarification_finishes_immediately():
    km = SimulatedKnowledgeManager()
    up_q: asyncio.Queue[str] = asyncio.Queue()
    down_q: asyncio.Queue[str] = asyncio.Queue()
    h = await km.ask(
        "Summarise the knowledge base.",
        _clarification_up_q=up_q,
        _clarification_down_q=down_q,
        _requests_clarification=True,
    )
    q = await asyncio.wait_for(up_q.get(), timeout=DEFAULT_TIMEOUT)
    assert "clarify" in q.lower()
    h.stop("no longer needed")
    out = await asyncio.wait_for(h.result(), timeout=DEFAULT_TIMEOUT)
    assert isinstance(out, str)
    assert h.done()
