from __future__ import annotations

import asyncio
import functools
import pytest

from unity.guidance_manager.simulated import (
    SimulatedGuidanceManager,
    _SimulatedGuidanceHandle,
)

# keeps each test isolated in its own Unify project / trace context
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
def test_docstrings_match_base():
    """
    Public methods in SimulatedGuidanceManager should copy the real
    BaseGuidanceManager doc-strings one-for-one (via functools.wraps).
    """
    from unity.guidance_manager.base import BaseGuidanceManager
    from unity.guidance_manager.simulated import SimulatedGuidanceManager

    assert (
        BaseGuidanceManager.ask.__doc__.strip()
        in SimulatedGuidanceManager.ask.__doc__.strip()
    ), ".ask doc-string was not copied correctly"

    assert (
        BaseGuidanceManager.update.__doc__.strip()
        in SimulatedGuidanceManager.update.__doc__.strip()
    ), ".update doc-string was not copied correctly"


# ────────────────────────────────────────────────────────────────────────────
# 2.  Basic start-and-ask                                                    #
# ────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@_handle_project
async def test_start_and_ask():
    gm = SimulatedGuidanceManager("Demo guidance for unit-tests.")
    h = await gm.ask("List the top 3 guidance items.")
    answer = await h.result()
    assert isinstance(answer, str) and answer.strip(), "Answer should be non-empty"


# ────────────────────────────────────────────────────────────────────────────
# 3.  Stateful memory – serial asks                                         #
# ────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@_handle_project
async def test_stateful_memory_serial_asks():
    """
    Two consecutive .ask() calls share context because the manager keeps a
    stateful LLM.
    """
    gm = SimulatedGuidanceManager()

    token = _unique_token("GUIDE")
    h1 = await gm.ask(
        "Please generate a short guidance snippet and include this exact token verbatim: "
        + token,
    )
    first = (await h1.result()).strip()
    assert first, "First answer should not be empty"

    h2 = await gm.ask("What unique token did you mention earlier? Quote it verbatim.")
    answer2 = await h2.result()
    assert isinstance(answer2, str) and answer2.strip()
    assert token in answer2, "LLM should recall the previously mentioned token"


# ────────────────────────────────────────────────────────────────────────────
# 4.  Update then ask – state carries through                                #
# ────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@_handle_project
async def test_stateful_update_then_ask():
    gm = SimulatedGuidanceManager()
    title = "Deployment Playbook"

    # create a fictitious guidance entry
    upd = await gm.update(
        f"Create a new guidance entry titled '{title}' that focuses on rollout steps and checklist.",
    )
    await upd.result()

    # ask about it
    hq = await gm.ask("Do we have guidance about deployment?")
    ans = (await hq.result()).lower()
    assert "deploy" in ans, "Answer should reference the guidance added via update"


# ────────────────────────────────────────────────────────────────────────────
# Steerable handle tests                                                     #
# ────────────────────────────────────────────────────────────────────────────


# ────────────────────────────────────────────────────────────────────────────
# 5.  Interject                                                             #
# ────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@_handle_project
async def test_handle_interject(monkeypatch):
    calls = {"interject": 0}
    orig = _SimulatedGuidanceHandle.interject

    @functools.wraps(orig)
    async def wrapped(self, msg: str, **kwargs) -> str:  # type: ignore[override]
        calls["interject"] += 1
        return await orig(self, msg, **kwargs)

    monkeypatch.setattr(_SimulatedGuidanceHandle, "interject", wrapped, raising=True)

    gm = SimulatedGuidanceManager()
    h = await gm.ask("Summarize our onboarding guidance.")
    await asyncio.sleep(0.05)
    reply = await h.interject("Focus on European enterprise scenarios.")
    assert _ack_ok(reply)
    await h.result()
    assert calls["interject"] == 1, ".interject should be invoked exactly once"


# ────────────────────────────────────────────────────────────────────────────
# 6.  Stop                                                                  #
# ────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@_handle_project
async def test_handle_stop():
    gm = SimulatedGuidanceManager()
    h = await gm.ask("Generate a full guidance export.")
    await asyncio.sleep(0.05)
    await h.stop()
    await h.result()
    assert h.done(), "Handle should report done after stop()"


# ────────────────────────────────────────────────────────────────────────────
# 7.  Clarification handshake                                               #
# ────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@_handle_project
async def test_handle_requests_clarification():
    gm = SimulatedGuidanceManager()

    up_q: asyncio.Queue[str] = asyncio.Queue()
    down_q: asyncio.Queue[str] = asyncio.Queue()

    h = await gm.ask(
        "What is the best practice for layout?",
        _clarification_up_q=up_q,
        _clarification_down_q=down_q,
        _requests_clarification=True,
    )

    question = await asyncio.wait_for(up_q.get(), timeout=DEFAULT_TIMEOUT)
    assert "clarify" in question.lower()
    await down_q.put("Focus on onboarding flows and mobile breakpoints.")

    answer = await h.result()
    assert isinstance(answer, str) and answer.strip()


# ────────────────────────────────────────────────────────────────────────────
# 8.  Pause → Resume round-trip                                             #
# ────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@_handle_project
async def test_handle_pause_and_resume(monkeypatch):
    """
    Verify that a `_SimulatedGuidanceHandle` can be paused and later resumed
    and that the *result()* coroutine blocks while the handle is paused.
    """
    call_counts = {"pause": 0, "resume": 0}

    # --- monkey-patch pause ------------------------------------------------
    original_pause = _SimulatedGuidanceHandle.pause

    @functools.wraps(original_pause)
    def _patched_pause(self):  # type: ignore[override]
        call_counts["pause"] += 1
        return original_pause(self)

    monkeypatch.setattr(
        _SimulatedGuidanceHandle,
        "pause",
        _patched_pause,
        raising=True,
    )

    # --- monkey-patch resume ----------------------------------------------
    original_resume = _SimulatedGuidanceHandle.resume

    @functools.wraps(original_resume)
    def _patched_resume(self):  # type: ignore[override]
        call_counts["resume"] += 1
        return original_resume(self)

    monkeypatch.setattr(
        _SimulatedGuidanceHandle,
        "resume",
        _patched_resume,
        raising=True,
    )

    gm = SimulatedGuidanceManager()
    handle = await gm.ask("Generate a short summary of all UI guidance.")

    # 1️⃣ Pause before awaiting the result
    pause_msg = await handle.pause()
    assert "pause" in pause_msg.lower()

    # 2️⃣ Kick off result() – it should block while paused
    res_task = await _assert_blocks_while_paused(handle.result())

    # 3️⃣ Resume and ensure the task now completes
    resume_msg = await handle.resume()
    assert "resume" in resume_msg.lower() or "running" in resume_msg.lower()

    answer = await asyncio.wait_for(res_task, timeout=DEFAULT_TIMEOUT)
    assert isinstance(answer, str) and answer.strip(), "Answer should be non-empty"

    # 4️⃣ Exactly one pause and one resume call must have been recorded
    assert call_counts == {
        "pause": 1,
        "resume": 1,
    }, "pause / resume should each be invoked exactly once"


# ────────────────────────────────────────────────────────────────────────────
# 9.  Nested ask on handle                                                   #
# ────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@_handle_project
async def test_handle_ask():
    """
    The internal handle returned by SimulatedGuidanceManager.ask exposes a
    dynamic ask() method that should produce a nested handle whose result can
    be awaited independently of the parent.
    """
    gm = SimulatedGuidanceManager()

    # Start an initial ask to obtain the live handle
    handle = await gm.ask("Summarize all onboarding guidance this quarter.")

    # Add extra context to ensure nested prompt includes it
    await handle.interject("Focus on European enterprise accounts.")

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
# 10.  Clear – reset and remain usable                                        #
# ────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@_handle_project
async def test_clear():
    """
    SimulatedGuidanceManager.clear should reset the manager and remain usable.
    """
    gm = SimulatedGuidanceManager()
    # Create some prior state in the stateful LLM
    upd = await gm.update("Create a temporary guidance entry about onboarding.")
    await asyncio.wait_for(upd.result(), timeout=DEFAULT_TIMEOUT)

    # Clear should not raise and should be quick
    gm.clear()

    # Post-clear, an ask should still work
    h = await gm.ask("List our guidance focus areas.")
    answer = await asyncio.wait_for(h.result(), timeout=DEFAULT_TIMEOUT)
    assert (
        isinstance(answer, str) and answer.strip()
    ), "Answer should be non-empty after clear()"


# ────────────────────────────────────────────────────────────────────────────
# 11.  Stop while paused should finish immediately                           #
# ────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@_handle_project
async def test_stop_while_paused_finishes_immediately():
    gm = SimulatedGuidanceManager()
    h = await gm.ask("Produce an exhaustive guidance export.")
    await h.pause()
    res_task = asyncio.create_task(h.result())
    await asyncio.sleep(0.1)
    assert not res_task.done()
    await h.stop("cancelled by user")
    out = await asyncio.wait_for(res_task, timeout=DEFAULT_TIMEOUT)
    assert isinstance(out, str)
    assert h.done()


# ────────────────────────────────────────────────────────────────────────────
# 12.  Stop while waiting for clarification should finish immediately         #
# ────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@_handle_project
async def test_stop_while_waiting_for_clarification_finishes_immediately():
    gm = SimulatedGuidanceManager()
    up_q: asyncio.Queue[str] = asyncio.Queue()
    down_q: asyncio.Queue[str] = asyncio.Queue()
    h = await gm.ask(
        "What is our policy here?",
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
