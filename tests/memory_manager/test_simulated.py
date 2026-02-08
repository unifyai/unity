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
import asyncio

from unity.memory_manager.simulated import SimulatedMemoryManager
from unity.contact_manager.simulated import SimulatedContactManager
from unity.knowledge_manager.simulated import SimulatedKnowledgeManager
from unity.task_scheduler.simulated import SimulatedTaskScheduler
from unity.contact_manager.types.contact import Contact

# shared helper used throughout the test-suite – isolates each test run
from tests.helpers import _handle_project, DEFAULT_TIMEOUT


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
async def test_update_contacts_invokes_expected_tools(monkeypatch):
    counts = {"cm_create": 0}

    # --- patch SimulatedContactManager._create_contact ---------------------
    orig_cm_create = SimulatedContactManager._create_contact

    def spy_cm_create(self, **kw):  # synchronous private helper
        counts["cm_create"] += 1
        return orig_cm_create(self, **kw)

    monkeypatch.setattr(
        SimulatedContactManager,
        "_create_contact",
        spy_cm_create,
        raising=True,
    )

    # --- ensure the named person does NOT exist yet -----------------------
    # This gently guides the simulation toward a contact creation rather than
    # inventing an existing record and performing an update instead.
    def _fakefilter_contacts(self, *, filter=None, offset=0, limit=1):
        return []

    monkeypatch.setattr(
        SimulatedContactManager,
        "filter_contacts",
        _fakefilter_contacts,
        raising=True,
    )

    # --- run the method ----------------------------------------------------
    cm = SimulatedContactManager(
        description=(
            "TEST SCENARIO: Contacts update path. SimulatedContactManager MUST treat the address book as missing"
            " the named person; its ask helpers must NOT invent existing records. It should deterministically"
            " accept create/update calls and return concise confirmations. SimulatedTranscriptManager should"
            " return straightforward results. No external I/O."
        ),
    )
    mm = SimulatedMemoryManager(
        contact_manager=cm,
    )
    transcript = _build_transcript(
        "FYI: New contact – Dana Fox, dana.fox@example.com, phone +14155550123.",
    )
    answer = await mm.update_contacts(transcript)

    # --- expectations ------------------------------------------------------
    assert isinstance(answer, str) and answer.strip(), "Return should be non-empty"
    # At least one contact creation should be invoked
    assert counts["cm_create"] >= 1


# --------------------------------------------------------------------------- #
# 1b. update_contacts – nameless service contact keeps no name                #
# --------------------------------------------------------------------------- #
@pytest.mark.eval
@pytest.mark.asyncio
@_handle_project
async def test_update_contacts_preserves_nameless_service_contact(monkeypatch):
    """When a transcript mentions a named representative on a service/org
    contact (no first_name/surname, bio describes the entity), the
    MemoryManager must NOT write the representative's name into the
    contact's name fields.
    """
    captured_update_kwargs: list[dict] = []
    captured_create_kwargs: list[dict] = []

    # --- spy on update_contact to capture kwargs -------------------------
    orig_cm_update = SimulatedContactManager.update_contact

    def spy_cm_update(self, **kw):
        captured_update_kwargs.append(kw)
        return orig_cm_update(self, **kw)

    monkeypatch.setattr(
        SimulatedContactManager,
        "update_contact",
        spy_cm_update,
        raising=True,
    )

    # --- spy on _create_contact to capture kwargs ------------------------
    orig_cm_create = SimulatedContactManager._create_contact

    def spy_cm_create(self, **kw):
        captured_create_kwargs.append(kw)
        return orig_cm_create(self, **kw)

    monkeypatch.setattr(
        SimulatedContactManager,
        "_create_contact",
        spy_cm_create,
        raising=True,
    )

    # --- filter_contacts returns the nameless service contact -------------
    def _fake_filter_contacts(self, *, filter=None, offset=0, limit=1):
        return [
            Contact(
                contact_id=10,
                first_name=None,
                surname=None,
                phone_number="8005550199",
                bio="Acme Corp billing support line",
            ),
        ]

    monkeypatch.setattr(
        SimulatedContactManager,
        "filter_contacts",
        _fake_filter_contacts,
        raising=True,
    )

    # --- run the method --------------------------------------------------
    cm = SimulatedContactManager(
        description=(
            "TEST SCENARIO: Nameless service contact. The contacts list contains"
            " exactly one contact: contact_id=10, first_name=None, surname=None,"
            " phone_number='8005550199', bio='Acme Corp billing support line'."
            " This is an organisation contact, NOT a person. Any names mentioned"
            " in the transcript belong to transient representatives, not to this"
            " contact's identity. SimulatedContactManager should accept updates"
            " and return concise confirmations. No external I/O."
        ),
    )
    mm = SimulatedMemoryManager(
        contact_manager=cm,
    )
    transcript = _build_transcript(
        "I called the Acme Corp billing line (8005550199). Sarah from the "
        "billing department answered and confirmed our next invoice is due "
        "March 15. She said to email billing@acme.com for follow-ups."
    )
    answer = await mm.update_contacts(
        transcript,
        update_bios=False,
        update_rolling_summaries=False,
        update_response_policies=False,
    )

    assert isinstance(answer, str) and answer.strip(), "Return should be non-empty"

    # Verify: no update_contact call set first_name or surname
    for kw in captured_update_kwargs:
        assert kw.get("first_name") is None, (
            f"update_contact should not set first_name on a service contact, got: {kw}"
        )
        assert kw.get("surname") is None, (
            f"update_contact should not set surname on a service contact, got: {kw}"
        )

    # Verify: no create_contact call was made with a name resembling the rep
    for kw in captured_create_kwargs:
        fn = (kw.get("first_name") or "").lower()
        sn = (kw.get("surname") or "").lower()
        assert "sarah" not in fn and "sarah" not in sn, (
            f"Should not create a new contact for a transient representative, got: {kw}"
        )


# --------------------------------------------------------------------------- #
# 2. update_contact_bio – restricted column write                             #
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
@_handle_project
async def test_update_contact_bio_calls_inner_helpers(monkeypatch):
    counts = {"cm_update": 0}

    # --- patch SimulatedContactManager.update_contact ---------------------
    orig_cm_upd = SimulatedContactManager.update_contact

    @functools.wraps(orig_cm_upd)
    def spy_cm_upd(self, **kw):
        counts["cm_update"] += 1
        return orig_cm_upd(self, **kw)

    monkeypatch.setattr(
        SimulatedContactManager,
        "update_contact",
        spy_cm_upd,
        raising=True,
    )

    # --- align names: ensure contact_id==1 refers to Dana -----------------
    def _fakefilter_contacts(self, *, filter=None, offset=0, limit=1):
        # Deterministic single result matching the transcript's person
        return [
            Contact(
                contact_id=1,
                first_name="Dana",
                surname="Fox",
                bio="Lead Project Manager at Tech Solutions, specializing in software development.",
            ),
        ]

    monkeypatch.setattr(
        SimulatedContactManager,
        "filter_contacts",
        _fakefilter_contacts,
        raising=True,
    )

    # run --------------------------------------------------------------------
    cm = SimulatedContactManager(
        description=(
            "TEST SCENARIO: Bio refresh. SimulatedContactManager MUST accept a single deterministic update to the"
            " 'bio' column for the target contact Dana; it must NOT refuse and must NOT claim the bio is already"
            " correct. SimulatedTranscriptManager should provide straightforward results. No external I/O."
        ),
    )
    mm = SimulatedMemoryManager(
        contact_manager=cm,
    )
    transcript = _build_transcript(
        "BTW – Dana Fox was promoted to Senior Project Manager at Tech Solutions.",
    )
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
async def test_update_contact_rolling_summary_invocations(monkeypatch):
    counts = {"cm_update": 0}

    # --- patch SimulatedContactManager.update_contact ---------------------
    orig_cm_upd = SimulatedContactManager.update_contact

    @functools.wraps(orig_cm_upd)
    def spy_cm_upd(self, **kw):
        counts["cm_update"] += 1
        return orig_cm_upd(self, **kw)

    monkeypatch.setattr(
        SimulatedContactManager,
        "update_contact",
        spy_cm_upd,
        raising=True,
    )

    # --- align names: ensure contact_id==1 refers to the person in transcript
    def _fakefilter_contacts(self, *, filter=None, offset=0, limit=1):
        # Deterministic single result matching the transcript's person
        return [
            Contact(
                contact_id=1,
                first_name="Dana",
                surname="Fox",
                rolling_summary="Drives KPI dashboard work; coordinates follow-ups.",
            ),
        ]

    monkeypatch.setattr(
        SimulatedContactManager,
        "filter_contacts",
        _fakefilter_contacts,
        raising=True,
    )

    # run --------------------------------------------------------------------
    cm = SimulatedContactManager(
        description=(
            "TEST SCENARIO: Rolling summary refresh. SimulatedContactManager MUST accept a single deterministic"
            " update to 'rolling_summary' for the target contact; do NOT claim it's already up to date."
            " SimulatedTranscriptManager answers simply. No external I/O."
        ),
    )
    mm = SimulatedMemoryManager(
        contact_manager=cm,
    )
    transcript = _build_transcript(
        "Dana Fox – action items: finalise the KPI dashboard by Friday and schedule a follow-up.",
    )

    new_summary = await mm.update_contact_rolling_summary(
        transcript,
        contact_id=1,
    )

    # check ------------------------------------------------------------------
    assert isinstance(new_summary, str) and new_summary.strip()
    # At least one call to ContactManager.update_contact
    assert (
        counts["cm_update"] >= 1
    ), "ContactManager.update_contact should be called at least once for rolling summary"


# --------------------------------------------------------------------------- #
# 4. update_contact_response_policy – restricted column write                 #
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
@_handle_project
async def test_update_contact_response_policy_invocations(monkeypatch):
    orig_cm_upd = SimulatedContactManager.update_contact

    calls = {"cm_update": 0}

    @functools.wraps(orig_cm_upd)
    def spy_cm_upd(self, **kw):  # noqa: D401 – imperative helper
        calls["cm_update"] += 1
        return orig_cm_upd(self, **kw)

    monkeypatch.setattr(
        SimulatedContactManager,
        "update_contact",
        spy_cm_upd,
        raising=True,
    )

    # --- align names: ensure contact_id==1 refers to Jane -----------------
    def _fakefilter_contacts(self, *, filter=None, offset=0, limit=1):
        # Deterministic single result matching the transcript's person
        return [
            Contact(
                contact_id=1,
                first_name="Jane",
                surname="Doe",
                response_policy="Respond within 24 hours to all inquiries.",
            ),
        ]

    monkeypatch.setattr(
        SimulatedContactManager,
        "filter_contacts",
        _fakefilter_contacts,
        raising=True,
    )

    cm = SimulatedContactManager(
        description=(
            "TEST SCENARIO: Response policy update. SimulatedContactManager MUST accept a single deterministic"
            " update to 'response_policy' for the target contact; do NOT claim it is already set."
            " SimulatedTranscriptManager answers simply. No external I/O."
        ),
    )
    mm = SimulatedMemoryManager(
        contact_manager=cm,
    )
    transcript = _build_transcript("Please be more formal when replying to Jane.")

    await mm.update_contact_response_policy(transcript, contact_id=1)

    # One invocation of ContactManager.update_contact expected
    assert (
        calls["cm_update"] >= 1
    ), "ContactManager.update_contact should be called at least once for response policy"


# --------------------------------------------------------------------------- #
# 5. update_knowledge – should call KnowledgeManager.update at least once     #
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
@_handle_project
async def test_update_knowledge_invokes_kb_update(monkeypatch):
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

    km = SimulatedKnowledgeManager(
        description=(
            "TEST SCENARIO: Knowledge update. SimulatedKnowledgeManager MUST treat the knowledge base as EMPTY for"
            " this run. All KnowledgeManager.ask calls MUST respond 'not found/absent' for facts introduced in"
            " this transcript unless the transcript itself includes an explicit KnowledgeManager.update"
            " manager-method proving prior storage. Therefore the flow should proceed to call"
            " KnowledgeManager.update exactly once to store the new fact. No external I/O."
        ),
    )
    mm = SimulatedMemoryManager(
        knowledge_manager=km,
    )
    transcript = _build_transcript(
        "Fun fact: The company standardised on Kubernetes for all deployments back in 2021.",
    )

    result = await mm.update_knowledge(transcript)

    assert isinstance(result, str) and result.strip()
    assert counts["kb_update"] >= 1, "KnowledgeManager.update should be invoked"


# --------------------------------------------------------------------------- #
# 6. update_tasks – should call TaskScheduler.update at least once            #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
@_handle_project
async def test_update_tasks_invokes_scheduler_update(monkeypatch):
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

    ts = SimulatedTaskScheduler(
        description=(
            "TEST SCENARIO: Task creation/update. SimulatedTaskScheduler MUST accept a deterministic update"
            " creating the requested task; do NOT claim the task already exists. SimulatedTranscriptManager"
            " returns straightforward results. No external I/O."
        ),
    )
    mm = SimulatedMemoryManager(
        task_scheduler=ts,
    )
    transcript = _build_transcript(
        "Please create a task to organise the quarterly review meeting next Monday.",
    )

    result = await mm.update_tasks(transcript)

    assert isinstance(result, str) and result.strip()
    assert counts["ts_update"] >= 1, "TaskScheduler.update should be invoked"


# --------------------------------------------------------------------------- #
# 7. reset – should complete and manager remains usable                        #
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
@_handle_project
async def test_reset_remains_usable():
    mm = SimulatedMemoryManager()
    # Reset should not raise
    await asyncio.wait_for(mm.reset(), timeout=DEFAULT_TIMEOUT)
    # After reset, a public method should still be usable
    result = await asyncio.wait_for(
        mm.update_tasks("Please schedule a quick follow-up for tomorrow."),
        timeout=DEFAULT_TIMEOUT,
    )
    assert isinstance(result, str) and result.strip()


# --------------------------------------------------------------------------- #
# 8. build_plain_transcript – includes manager_method JSON lines               #
# --------------------------------------------------------------------------- #
@_handle_project
def test_build_plain_transcript_includes_manager_method_json():
    msgs = [
        {"sender": "Alice Example", "content": "Hi Bob"},
        {
            "kind": "manager_method",
            "data": {
                "source": "ConversationManager",
                "phase": "incoming",
                "method": "ContactManager.update",
                "args": {"contact_id": 1, "bio": "Updated."},
            },
        },
        {"sender": "Bob Example", "content": "Hi Alice"},
    ]
    out = SimulatedMemoryManager.build_plain_transcript(msgs)
    assert isinstance(out, str) and out.strip()
    # Plain chat lines should be rendered
    assert "Alice Example: Hi Bob" in out
    assert "Bob Example: Hi Alice" in out
    # Manager-method event should be appended as a JSON line
    assert '"kind": "manager_method"' in out
