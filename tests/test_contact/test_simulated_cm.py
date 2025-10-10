from __future__ import annotations

import asyncio
import functools
import pytest

from unity.contact_manager.simulated import (
    SimulatedContactManager,
    _SimulatedContactHandle,
)
from unity.contact_manager.types.contact import Contact

# keeps each test isolated in its own Unify project / trace context
from tests.helpers import _handle_project


# ────────────────────────────────────────────────────────────────────────────
# 1.  Doc-string inheritance                                                 #
# ────────────────────────────────────────────────────────────────────────────
def test_simulated_cm_docstrings_match_base():
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
async def test_start_and_ask_simulated_cm():
    cm = SimulatedContactManager("Demo CRM for unit-tests.")
    h = await cm.ask("List all my contacts.")
    answer = await h.result()
    assert isinstance(answer, str) and answer.strip(), "Answer should be non-empty"


# ────────────────────────────────────────────────────────────────────────────
# 3.  Stateful memory – serial asks                                         #
# ────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@_handle_project
async def test_cm_stateful_memory_serial_asks():
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
async def test_cm_stateful_update_then_ask():
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
    assert "ack" in reply.lower() or "noted" in reply.lower()
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
        clarification_up_q=up_q,
        clarification_down_q=down_q,
        _requests_clarification=True,
    )

    question = await asyncio.wait_for(up_q.get(), timeout=60)
    assert "clarify" in question.lower()
    await down_q.put("Yes – focus on European clients.")

    answer = await h.result()
    assert isinstance(answer, str) and answer.strip()
    assert "europe" in answer.lower()


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
    pause_msg = handle.pause()
    assert "pause" in pause_msg.lower()

    # 2️⃣ Kick off result() – it should block while paused
    res_task = asyncio.create_task(handle.result())
    await asyncio.sleep(0.1)  # give the coroutine a moment to enter the wait-loop
    assert not res_task.done(), "result() should block while the handle is paused"

    # 3️⃣ Resume and ensure the task now completes
    resume_msg = handle.resume()
    assert "resume" in resume_msg.lower() or "running" in resume_msg.lower()

    answer = await asyncio.wait_for(res_task, timeout=60)
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
    assert "europe" in nested_answer.lower()

    # The original handle should still be awaitable and produce an answer
    handle_answer = await handle.result()
    assert isinstance(handle_answer, str) and handle_answer.strip(), (
        "Handle should still yield a non-empty answer after nested ask",
    )


# ────────────────────────────────────────────────────────────────────────────
# 10. Private: _filter_contacts                                              #
# ────────────────────────────────────────────────────────────────────────────
@_handle_project
def test_private_filter_contacts_basic():
    cm = SimulatedContactManager(
        "Simulated CRM for private method tests.",
    )

    # Use a permissive filter and small limit to keep runtime low
    results = cm._filter_contacts(
        filter="first_name is None or first_name is not None",
        offset=0,
        limit=3,
    )

    assert isinstance(results, list)
    assert len(results) <= 3
    # All returned items should be Contact models
    assert all(isinstance(c, Contact) for c in results)


# ────────────────────────────────────────────────────────────────────────────
# 11. Private: _update_contact                                               #
# ────────────────────────────────────────────────────────────────────────────
@_handle_project
def test_private_update_contact_returns_structured_outcome():
    cm = SimulatedContactManager()

    outcome = cm._update_contact(
        contact_id=42,
        first_name="Alice",
        surname="Example",
        response_policy="Share weekly updates",
        custom_fields={"priority": "high"},
    )

    assert isinstance(outcome, dict)
    assert "outcome" in outcome and isinstance(outcome["outcome"], str)
    assert "details" in outcome and isinstance(outcome["details"], dict)
    assert outcome["details"].get("contact_id") == 42


# ────────────────────────────────────────────────────────────────────────────
# 12. Private: _delete_contact                                               #
# ────────────────────────────────────────────────────────────────────────────
@_handle_project
def test_private_delete_contact_returns_structured_outcome():
    cm = SimulatedContactManager()

    outcome = cm._delete_contact(contact_id=77)

    assert isinstance(outcome, dict)
    assert outcome.get("details", {}).get("contact_id") == 77
    assert isinstance(outcome.get("outcome", ""), str)


# ────────────────────────────────────────────────────────────────────────────
# 13. Private: _merge_contacts                                               #
# ────────────────────────────────────────────────────────────────────────────
@_handle_project
def test_private_merge_contacts_returns_structured_outcome():
    cm = SimulatedContactManager()

    cid1, cid2 = 12, 34
    outcome = cm._merge_contacts(
        contact_id_1=cid1,
        contact_id_2=cid2,
        overrides={"email_address": 2},
    )

    assert isinstance(outcome, dict)
    details = outcome.get("details", {})
    assert isinstance(details, dict)

    kept = details.get("kept_contact_id")
    deleted = details.get("deleted_contact_id")
    assert kept is not None and deleted is not None and kept != deleted
    assert {kept, deleted} == {cid1, cid2}
    assert isinstance(details.get("overrides", {}), dict)


# ────────────────────────────────────────────────────────────────────────────
# 14. Private: _create_contact                                               #
# ────────────────────────────────────────────────────────────────────────────
@_handle_project
def test_private_create_contact_returns_structured_outcome():
    cm = SimulatedContactManager()

    outcome = cm._create_contact(
        first_name="Alice",
        surname="Example",
        email_address="alice@example.com",
        respond_to=True,
        custom_fields={"priority": "high"},
    )

    assert isinstance(outcome, dict)
    assert "outcome" in outcome and isinstance(outcome["outcome"], str)
    assert "details" in outcome and isinstance(outcome["details"], dict)
    cid = outcome["details"].get("contact_id")
    assert isinstance(cid, int)
