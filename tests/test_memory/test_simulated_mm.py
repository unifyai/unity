"""
Integration-style tests for *SimulatedMemoryManager*.

For every public method we:

• monkey-patch the **sub-manager** helpers that the MemoryManager
  delegates to (`SimulatedContactManager` / `SimulatedTranscriptManager`)
  so they increment counters;
• spin-up a fresh `SimulatedMemoryManager` (after patches so the spies
  are active);
• invoke the target method with a JSON-encoded 50-message transcript
  where the “useful” line is buried near the end; and
• assert the patched helpers were called as expected *and* that the
  method returns a non-empty string.

No steerability checks are needed – all MemoryManager methods return a
final string immediately.
"""

from __future__ import annotations

import json
import functools
import pytest

from unity.memory_manager.simulated import SimulatedMemoryManager
from unity.contact_manager.simulated import SimulatedContactManager
from unity.knowledge_manager.simulated import SimulatedKnowledgeManager
from unity.task_scheduler.simulated import SimulatedTaskScheduler

# shared helper used throughout the test-suite – isolates each test run
from tests.helpers import _handle_project


# --------------------------------------------------------------------------- #
# Utilities                                                                   #
# --------------------------------------------------------------------------- #
def _build_transcript(useful_line: str) -> str:
    """
    Make a 50-message transcript JSON blob (`indent=4`) with `useful_line`
    hidden at position 37.  The other 49 entries are innocuous.
    """
    msgs = [{"sender": "User", "content": f"Random chit-chat {i}."} for i in range(50)]
    msgs[37]["content"] = useful_line
    return json.dumps(msgs, indent=4)


# --------------------------------------------------------------------------- #
# 1. update_contacts – delegation map                                         #
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
@_handle_project
async def test_mm_update_contacts_invokes_expected_tools(monkeypatch):
    counts = {"cm_update": 0}

    # --- patch SimulatedContactManager.update ------------------------------
    orig_cm_upd = SimulatedContactManager.update

    @functools.wraps(orig_cm_upd)
    async def spy_cm_upd(self, text: str, **kw):
        counts["cm_update"] += 1
        return await orig_cm_upd(self, text, **kw)

    monkeypatch.setattr(SimulatedContactManager, "update", spy_cm_upd, raising=True)

    # --- run the method ----------------------------------------------------
    mm = SimulatedMemoryManager("CRM enrichment demo.")
    transcript = _build_transcript(
        "FYI: New contact – Dana Fox, dana.fox@example.com, phone +14155550123.",
    )
    answer = await mm.update_contacts(transcript)

    # --- expectations ------------------------------------------------------
    assert isinstance(answer, str) and answer.strip(), "Return should be non-empty"
    # At least one call to update contacts
    assert counts["cm_update"] >= 1


# --------------------------------------------------------------------------- #
# 2. update_contact_bio – restricted column write                             #
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
@_handle_project
async def test_mm_update_contact_bio_calls_inner_helpers(monkeypatch):
    counts = {"cm_update": 0}

    # --- patch SimulatedContactManager.update ------------------------------
    orig_cm_upd = SimulatedContactManager.update

    @functools.wraps(orig_cm_upd)
    async def spy_cm_upd(self, text: str, **kw):
        counts["cm_update"] += 1
        return await orig_cm_upd(self, text, **kw)

    monkeypatch.setattr(SimulatedContactManager, "update", spy_cm_upd, raising=True)

    # run --------------------------------------------------------------------
    mm = SimulatedMemoryManager("Bio refresh demo.")
    transcript = _build_transcript("BTW – Dana just moved to Berlin.")
    new_bio = await mm.update_contact_bio(
        transcript,
        contact_id=1,
    )

    # check ------------------------------------------------------------------
    assert isinstance(new_bio, str) and new_bio.strip()
    # At least one call to update contacts
    assert counts["cm_update"] >= 1


# --------------------------------------------------------------------------- #
# 3. update_contact_rolling_summary – restricted column write                 #
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
@_handle_project
async def test_mm_update_contact_rolling_summary_invocations(monkeypatch):
    counts = {"cm_update": 0}

    # --- patch SimulatedContactManager.update ------------------------------
    orig_cm_upd = SimulatedContactManager.update

    @functools.wraps(orig_cm_upd)
    async def spy_cm_upd(self, text: str, **kw):
        counts["cm_update"] += 1
        return await orig_cm_upd(self, text, **kw)

    monkeypatch.setattr(SimulatedContactManager, "update", spy_cm_upd, raising=True)

    # run --------------------------------------------------------------------
    mm = SimulatedMemoryManager("Rolling-summary refresh demo.")
    transcript = _build_transcript(
        "Action items: finalise KPI dashboard by Friday and schedule follow-up.",
    )

    new_summary = await mm.update_contact_rolling_summary(
        transcript,
        contact_id=1,
    )

    # check ------------------------------------------------------------------
    assert isinstance(new_summary, str) and new_summary.strip()
    # At least one call to update contacts
    assert counts["cm_update"] >= 1


# --------------------------------------------------------------------------- #
# 4. update_knowledge – should call KnowledgeManager.update at least once     #
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
@_handle_project
async def test_mm_update_knowledge_invokes_kb_update(monkeypatch):
    counts = {"kb_update": 0}

    orig_kb_update = SimulatedKnowledgeManager.update

    @functools.wraps(orig_kb_update)
    async def spy_kb_update(self, text: str, **kw):
        counts["kb_update"] += 1
        return await orig_kb_update(self, text, **kw)

    monkeypatch.setattr(
        SimulatedKnowledgeManager,
        "update",
        spy_kb_update,
        raising=True,
    )

    mm = SimulatedMemoryManager("Knowledge ingestion demo.")
    transcript = _build_transcript(
        "Fun fact: The company standardised on Kubernetes for all deployments back in 2021.",
    )

    result = await mm.update_knowledge(transcript)

    assert isinstance(result, str) and result.strip()
    assert counts["kb_update"] >= 1, "KnowledgeManager.update should be invoked"


# --------------------------------------------------------------------------- #
# 5. update_tasks – should call TaskScheduler.update at least once            #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
@_handle_project
async def test_mm_update_tasks_invokes_scheduler_update(monkeypatch):
    counts = {"ts_update": 0}

    orig_ts_update = SimulatedTaskScheduler.update

    @functools.wraps(orig_ts_update)
    async def spy_ts_update(self, text: str, **kw):
        counts["ts_update"] += 1
        return await orig_ts_update(self, text, **kw)

    monkeypatch.setattr(
        SimulatedTaskScheduler,
        "update",
        spy_ts_update,
        raising=True,
    )

    mm = SimulatedMemoryManager("Task list update demo.")
    transcript = _build_transcript(
        "Please create a task to organise the quarterly review meeting next Monday.",
    )

    result = await mm.update_tasks(transcript)

    assert isinstance(result, str) and result.strip()
    assert counts["ts_update"] >= 1, "TaskScheduler.update should be invoked"


# ---------------------------------------------------------------------------
# Response policy helper
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@_handle_project
async def test_mm_update_contact_response_policy_invocations(monkeypatch):
    orig_cm_upd = SimulatedContactManager.update

    calls = {"cm_update": 0}

    @functools.wraps(orig_cm_upd)
    async def spy_cm_upd(self, text: str, **kw):  # noqa: D401 – imperative helper
        calls["cm_update"] += 1
        return await orig_cm_upd(self, text, **kw)

    monkeypatch.setattr(SimulatedContactManager, "update", spy_cm_upd, raising=True)

    mm = SimulatedMemoryManager("response policy helper demo")
    transcript = _build_transcript("Please be more formal when replying to Jane.")

    await mm.update_contact_response_policy(transcript, contact_id=1)

    # One invocation of ContactManager.update expected
    assert (
        calls["cm_update"] >= 1
    ), "ContactManager.update should be called at least once for response policy"
