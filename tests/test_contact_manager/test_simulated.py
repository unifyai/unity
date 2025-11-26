from __future__ import annotations

import asyncio
import functools
import pytest

from unity.contact_manager.simulated import (
    SimulatedContactManager,
    _SimulatedContactHandle,
)

# keeps each test isolated in its own Unify project / trace context
from tests.helpers import (
    _handle_project,
    _ack_ok,
    _assert_blocks_while_paused,
    DEFAULT_TIMEOUT,
)


# ────────────────────────────────────────────────────────────────────────────
# 1.  Doc-string inheritance                                                 #
# ────────────────────────────────────────────────────────────────────────────
def test_docstrings_match_base():
    """
    Public methods in SimulatedContactManager should copy the real
    BaseContactManager doc-strings one-for-one (via functools.wraps).
    """
    from unity.contact_manager.base import BaseContactManager
    from unity.contact_manager.simulated import SimulatedContactManager

    assert (
        BaseContactManager.ask.__doc__.strip()
        in SimulatedContactManager.ask.__doc__.strip()
    ), ".store doc-string was not copied correctly"

    assert (
        BaseContactManager.update.__doc__.strip()
        in SimulatedContactManager.update.__doc__.strip()
    ), ".retrieve doc-string was not copied correctly"


# ────────────────────────────────────────────────────────────────────────────
# 2.  Basic start-and-ask                                                    #
# ────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@_handle_project
async def test_start_and_ask():
    cm = SimulatedContactManager("Demo CRM for unit-tests.")
    h = await cm.ask("List all my contacts.")
    answer = await h.result()
    assert isinstance(answer, str) and answer.strip(), "Answer should be non-empty"


# ────────────────────────────────────────────────────────────────────────────
# 3.  Stateful memory – serial asks                                         #
# ────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@_handle_project
async def test_stateful_serial_asks():
    """
    Two consecutive .ask() calls share context because the manager keeps a
    stateful LLM.
    """
    cm = SimulatedContactManager()

    h1 = await cm.ask(
        "Please suggest a unique reference code for a new prospect, "
        "and reply with *only* that code.",
    )
    ref_code = (await h1.result()).strip()
    assert ref_code, "Reference code should not be empty"

    h2 = await cm.ask("Great. What reference code did you just propose?")
    answer2 = (await h2.result()).lower()
    assert ref_code.lower() in answer2, "LLM should recall the code it generated"


# ────────────────────────────────────────────────────────────────────────────
# 4.  Update then ask – state carries through                                #
# ────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@_handle_project
async def test_stateful_update_then_ask():
    cm = SimulatedContactManager()
    full_name = "Johnathan Doe"
    email = "john.doe@example.com"

    # create a fictitious contact
    upd = await cm.update(
        f"Create a new contact: {full_name}, email {email}, mark as high priority.",
    )
    await upd.result()

    # ask about it
    hq = await cm.ask("Do we have Johnathan's contact details on file?")
    ans = (await hq.result()).lower()
    assert (
        "john" in ans and "email" in ans
    ), "Contact created via update should be recalled"


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
    orig = _SimulatedContactHandle.interject

    @functools.wraps(orig)
    def wrapped(self, msg: str) -> str:  # type: ignore[override]
        calls["interject"] += 1
        return orig(self, msg)

    monkeypatch.setattr(_SimulatedContactHandle, "interject", wrapped, raising=True)

    cm = SimulatedContactManager()
    h = await cm.ask("Show me all contacts created this quarter.")
    await asyncio.sleep(0.05)
    reply = h.interject("Filter only VIP customers.")
    assert _ack_ok(reply)
    await h.result()
    assert calls["interject"] == 1, ".interject should be invoked exactly once"


# ────────────────────────────────────────────────────────────────────────────
# 6.  Stop                                                                  #
# ────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@_handle_project
async def test_handle_stop():
    cm = SimulatedContactManager()
    h = await cm.ask("Generate a full CRM export.")
    await asyncio.sleep(0.05)
    h.stop()
    await h.result()
    assert h.done(), "Handle should report done after stop()"


# ────────────────────────────────────────────────────────────────────────────
# 7.  Clarification handshake                                               #
# ────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@_handle_project
async def test_handle_requests_clarification():
    cm = SimulatedContactManager()

    up_q: asyncio.Queue[str] = asyncio.Queue()
    down_q: asyncio.Queue[str] = asyncio.Queue()

    h = await cm.ask(
        "What is David's phone number?",
        _clarification_up_q=up_q,
        _clarification_down_q=down_q,
        _requests_clarification=True,
    )

    question = await asyncio.wait_for(up_q.get(), timeout=DEFAULT_TIMEOUT)
    assert "clarify" in question.lower()
    await down_q.put("It's the one ending in 123")

    answer = await h.result()
    assert isinstance(answer, str) and answer.strip()
    assert "123" in answer.lower()


# ────────────────────────────────────────────────────────────────────────────
# 8.  Pause → Resume round-trip                                             #
# ────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@_handle_project
async def test_handle_pause_and_resume(monkeypatch):
    """
    Verify that a `_SimulatedContactHandle` can be paused and later resumed
    and that the *result()* coroutine blocks while the handle is paused.
    """
    call_counts = {"pause": 0, "resume": 0}

    # --- monkey-patch pause ------------------------------------------------
    original_pause = _SimulatedContactHandle.pause

    @functools.wraps(original_pause)
    def _patched_pause(self):  # type: ignore[override]
        call_counts["pause"] += 1
        return original_pause(self)

    monkeypatch.setattr(
        _SimulatedContactHandle,
        "pause",
        _patched_pause,
        raising=True,
    )

    # --- monkey-patch resume ----------------------------------------------
    original_resume = _SimulatedContactHandle.resume

    @functools.wraps(original_resume)
    def _patched_resume(self):  # type: ignore[override]
        call_counts["resume"] += 1
        return original_resume(self)

    monkeypatch.setattr(
        _SimulatedContactHandle,
        "resume",
        _patched_resume,
        raising=True,
    )

    cm = SimulatedContactManager()
    handle = await cm.ask("Generate a short summary of all open opportunities.")

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
    The internal handle returned by SimulatedContactManager.ask exposes a
    dynamic ask() method that should produce a nested handle whose result can
    be awaited independently of the parent.
    """
    cm = SimulatedContactManager()

    # Start an initial ask to obtain the live handle
    handle = await cm.ask("Summarize all open opportunities this quarter.")

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
# 11.  Stop while paused should finish immediately                           #
# ────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@_handle_project
async def test_stop_while_paused():
    cm = SimulatedContactManager()
    h = await cm.ask("Produce a long contact export.")
    # Enter paused state
    await h.pause()
    # result() should be blocked while paused
    res_task = asyncio.create_task(h.result())
    await asyncio.sleep(0.1)
    assert not res_task.done()
    # Stop should unblock and complete promptly
    h.stop("cancelled by user")
    out = await asyncio.wait_for(res_task, timeout=DEFAULT_TIMEOUT)
    assert isinstance(out, str)
    assert h.done()


# ────────────────────────────────────────────────────────────────────────────
# 12.  Stop while waiting for clarification should finish immediately         #
# ────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@_handle_project
async def test_stop_while_waiting_clarification():
    cm = SimulatedContactManager()
    up_q: asyncio.Queue[str] = asyncio.Queue()
    down_q: asyncio.Queue[str] = asyncio.Queue()
    h = await cm.ask(
        "Find David's phone number.",
        _clarification_up_q=up_q,
        _clarification_down_q=down_q,
        _requests_clarification=True,
    )
    # Wait until a clarification is requested
    q = await asyncio.wait_for(up_q.get(), timeout=DEFAULT_TIMEOUT)
    assert "clarify" in q.lower()
    # Stop without answering; should return promptly
    h.stop("no longer needed")
    out = await asyncio.wait_for(h.result(), timeout=DEFAULT_TIMEOUT)
    assert isinstance(out, str)
    assert h.done()


# ────────────────────────────────────────────────────────────────────────────
# 10.  Simulated private helpers                                             #
# ────────────────────────────────────────────────────────────────────────────
@_handle_project
def test_filter_sync():
    """
    SimulatedContactManager.filter_contacts should produce a plausible list of
    contacts synchronously (cannot be called from an active event loop).
    """
    cm = SimulatedContactManager()
    # Use a permissive filter; just validate basic shape and limit behaviour
    results = cm.filter_contacts(filter="True", limit=3)
    assert isinstance(results, list), "Expected list of contacts"
    assert len(results) <= 3, "Limit should cap the number of returned contacts"
    if results:
        first = results[0]
        assert hasattr(first, "contact_id"), "Each contact should have contact_id"


@_handle_project
def test_update_sync():
    """
    SimulatedContactManager.update_contact should return a structured confirmation
    with 'outcome' and 'details.contact_id'.
    """
    cm = SimulatedContactManager()
    out = cm.update_contact(contact_id=123, first_name="Alice")
    assert isinstance(out, dict), "update_contact yields a dict-like outcome"
    assert "outcome" in out, "Outcome should include 'outcome' message"
    assert "details" in out and isinstance(out["details"], dict)
    assert isinstance(out["details"].get("contact_id"), int)


@_handle_project
def test_clear_sync():
    """
    SimulatedContactManager.clear should reset the manager (hard-coded completion)
    and remain usable afterwards.
    """
    cm = SimulatedContactManager()
    # Do a synchronous operation to create some prior state
    cm.update_contact(contact_id=1, surname="Smith")
    # Clear should not raise and should be quick (no LLM roundtrip)
    cm.clear()
    # Post-clear, synchronous helper still works
    post = cm.filter_contacts(limit=1)
    assert isinstance(post, list)


@_handle_project
def test_simulated_contact_manager_reduce_shapes():
    cm = SimulatedContactManager()

    scalar = cm.reduce(metric="sum", keys="contact_id")
    assert isinstance(scalar, (int, float))

    multi = cm.reduce(metric="max", keys=["contact_id"])
    assert isinstance(multi, dict)
    assert set(multi.keys()) == {"contact_id"}

    grouped = cm.reduce(metric="sum", keys="contact_id", group_by="segment")
    assert isinstance(grouped, dict)
