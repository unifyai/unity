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
    counts = {"_upd": 0}

    async def stub_update_contact(
        self,
        *,
        contact_id: int,
        custom_fields: dict,
        **kw,
    ):
        # Increment counter, pretend success
        counts["_upd"] += 1
        return {"outcome": "stub ok", "details": {"contact_id": contact_id}}

    monkeypatch.setattr(
        SimulatedContactManager,
        "_update_contact",
        stub_update_contact,
        raising=False,
    )

    # run --------------------------------------------------------------------
    mm = SimulatedMemoryManager("Bio refresh demo.")
    latest_bio = "Dana Fox – Marketing Lead, NYC."
    transcript = _build_transcript("BTW – Dana just moved to Berlin.")
    new_bio = await mm.update_contact_bio(transcript, latest_bio=latest_bio)

    # check ------------------------------------------------------------------
    assert isinstance(new_bio, str) and new_bio.strip()
    assert counts["_upd"] == 1, "_update_contact should be called exactly once"


# --------------------------------------------------------------------------- #
# 3. update_contact_rolling_summary – restricted column write                 #
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
@_handle_project
async def test_mm_update_contact_rolling_summary_invocations(monkeypatch):
    counts = {"_upd": 0}

    # ---- ensure _update_contact is available & spied ----------------------
    async def stub_update_contact(
        self,
        *,
        contact_id: int,
        custom_fields: dict,
        **kw,
    ):
        counts["_upd"] += 1
        return {"outcome": "stub ok", "details": {"contact_id": contact_id}}

    monkeypatch.setattr(
        SimulatedContactManager,
        "_update_contact",
        stub_update_contact,
        raising=False,
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
