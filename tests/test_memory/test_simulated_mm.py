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
from unity.transcript_manager.simulated import SimulatedTranscriptManager

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
    counts = {"cm_ask": 0, "cm_update": 0, "tm_ask": 0}

    # --- patch SimulatedContactManager.ask ----------------------------------
    orig_cm_ask = SimulatedContactManager.ask

    @functools.wraps(orig_cm_ask)
    async def spy_cm_ask(self, text: str, **kw):
        counts["cm_ask"] += 1
        return await orig_cm_ask(self, text, **kw)

    monkeypatch.setattr(SimulatedContactManager, "ask", spy_cm_ask, raising=True)

    # --- patch SimulatedContactManager.update ------------------------------
    orig_cm_upd = SimulatedContactManager.update

    @functools.wraps(orig_cm_upd)
    async def spy_cm_upd(self, text: str, **kw):
        counts["cm_update"] += 1
        return await orig_cm_upd(self, text, **kw)

    monkeypatch.setattr(SimulatedContactManager, "update", spy_cm_upd, raising=True)

    # --- patch SimulatedTranscriptManager.ask ------------------------------
    orig_tm_ask = SimulatedTranscriptManager.ask

    @functools.wraps(orig_tm_ask)
    async def spy_tm_ask(self, text: str, **kw):
        counts["tm_ask"] += 1
        return await orig_tm_ask(self, text, **kw)

    monkeypatch.setattr(SimulatedTranscriptManager, "ask", spy_tm_ask, raising=True)

    # --- run the method ----------------------------------------------------
    mm = SimulatedMemoryManager("CRM enrichment demo.")
    transcript = _build_transcript(
        "FYI: New contact – Dana Fox, dana.fox@example.com, phone +14155550123.",
    )
    answer = await mm.update_contacts(transcript)

    # --- expectations ------------------------------------------------------
    assert isinstance(answer, str) and answer.strip(), "Return should be non-empty"
    # At least one ask to contacts & transcript, plus ≥1 update to contacts
    assert counts["cm_ask"] >= 1
    assert counts["tm_ask"] >= 1
    assert counts["cm_update"] >= 1


# --------------------------------------------------------------------------- #
# 2. update_contact_bio – restricted column write                             #
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
@_handle_project
async def test_mm_update_contact_bio_calls_inner_helpers(monkeypatch):
    counts = {"cm_ask": 0, "tm_ask": 0, "_upd": 0}

    # spies ------------------------------------------------------------------
    o_cm_ask = SimulatedContactManager.ask
    o_tm_ask = SimulatedTranscriptManager.ask
    o_cm_upd = SimulatedContactManager._update_contact

    @functools.wraps(o_cm_ask)
    async def spy_cm_ask(self, text: str, **kw):
        counts["cm_ask"] += 1
        return await o_cm_ask(self, text, **kw)

    @functools.wraps(o_tm_ask)
    async def spy_tm_ask(self, text: str, **kw):
        counts["tm_ask"] += 1
        return await o_tm_ask(self, text, **kw)

    @functools.wraps(o_cm_upd)
    async def spy_cm_upd(self, **kw):
        counts["_upd"] += 1
        return await o_cm_upd(self, **kw)

    monkeypatch.setattr(SimulatedContactManager, "ask", spy_cm_ask, raising=True)
    monkeypatch.setattr(SimulatedTranscriptManager, "ask", spy_tm_ask, raising=True)
    monkeypatch.setattr(
        SimulatedContactManager,
        "_update_contact",
        spy_cm_upd,
        raising=True,
    )

    # run --------------------------------------------------------------------
    mm = SimulatedMemoryManager("Bio refresh demo.")
    latest_bio = "Dana Fox – Marketing Lead, NYC."
    transcript = _build_transcript("BTW – Dana just moved to Berlin.")
    new_bio = await mm.update_contact_bio(transcript, latest_bio=latest_bio)

    # check ------------------------------------------------------------------
    assert isinstance(new_bio, str) and new_bio.strip()
    assert counts["_upd"] == 1, "_update_contact should be called exactly once"
    assert counts["cm_ask"] >= 1
    assert counts["tm_ask"] >= 1


# --------------------------------------------------------------------------- #
# 3. update_contact_rolling_summary – restricted column write                 #
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
@_handle_project
async def test_mm_update_contact_rolling_summary_invocations(monkeypatch):
    counts = {"cm_ask": 0, "tm_ask": 0, "_upd": 0}

    o_cm_ask = SimulatedContactManager.ask
    o_tm_ask = SimulatedTranscriptManager.ask
    o_cm_upd = SimulatedContactManager._update_contact

    @functools.wraps(o_cm_ask)
    async def spy_cm_ask(self, text: str, **kw):
        counts["cm_ask"] += 1
        return await o_cm_ask(self, text, **kw)

    @functools.wraps(o_tm_ask)
    async def spy_tm_ask(self, text: str, **kw):
        counts["tm_ask"] += 1
        return await o_tm_ask(self, text, **kw)

    @functools.wraps(o_cm_upd)
    async def spy_cm_upd(self, **kw):
        counts["_upd"] += 1
        return await o_cm_upd(self, **kw)

    monkeypatch.setattr(SimulatedContactManager, "ask", spy_cm_ask, raising=True)
    monkeypatch.setattr(SimulatedTranscriptManager, "ask", spy_tm_ask, raising=True)
    monkeypatch.setattr(
        SimulatedContactManager,
        "_update_contact",
        spy_cm_upd,
        raising=True,
    )

    mm = SimulatedMemoryManager("Rolling-summary refresh demo.")
    prev_summary = "Discussing Q3 marketing launch."
    transcript = _build_transcript(
        "Action items: finalise KPI dashboard by Friday and schedule follow-up.",
    )

    new_summary = await mm.update_contact_rolling_summary(
        transcript,
        latest_rolling_summary=prev_summary,
    )

    assert isinstance(new_summary, str) and new_summary.strip()
    assert counts["_upd"] == 1, "_update_contact should be invoked once"
    assert counts["cm_ask"] >= 1 and counts["tm_ask"] >= 1
