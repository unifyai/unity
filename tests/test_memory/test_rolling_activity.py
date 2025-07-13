"""
Simple unit test for `MemoryManager.get_rolling_activity`.

Ensures that when **no** activity has been recorded yet, the helper
returns an *empty* string – callers can then safely omit the Historic
Activity block from prompts.
"""

from __future__ import annotations


import unify

from tests.helpers import _handle_project

import pytest
import random


# ---------------------------------------------------------------------------
#  Test – empty rolling activity                                             |
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@_handle_project
async def test_get_rolling_activity_empty(monkeypatch):
    """`get_rolling_activity` should return an empty string with no logs."""

    from unity.memory_manager.memory_manager import MemoryManager

    # 1.  Stub heavy helpers so instantiation is lightweight
    async def _noop(self, *_, **__):
        """Async no-op used to replace `_setup_rolling_callbacks`."""

    # Avoid costly callback registration & context/field creation
    monkeypatch.setattr(MemoryManager, "_setup_rolling_callbacks", _noop, raising=True)
    monkeypatch.setattr(
        MemoryManager,
        "_ensure_rolling_context",
        lambda self: "ctx",
        raising=True,
    )

    # Ensure *no* rows are returned so the method must fall back to "empty"
    monkeypatch.setattr(unify, "get_logs", lambda *a, **kw: [], raising=True)

    # 2.  Exercise & verify
    mm = MemoryManager()

    assert (
        mm.get_rolling_activity() == ""
    ), "Expected empty string when no activity logged"


# ---------------------------------------------------------------------------
#  Test – single manager call populates rolling activity                     |
# ---------------------------------------------------------------------------

# Manager test doubles
from unity.contact_manager.simulated import SimulatedContactManager
from unity.transcript_manager.simulated import SimulatedTranscriptManager
from unity.knowledge_manager.simulated import SimulatedKnowledgeManager
from unity.task_scheduler.simulated import SimulatedTaskScheduler

# MemoryManager (subject under test)
from unity.memory_manager.memory_manager import MemoryManager

# Handy type alias for the param table
from typing import Callable, Any, Tuple

_ManagerFactory = Callable[[], Any]

# ---------------------------------------------------------------------------
# Parameter table                                                           |
# ---------------------------------------------------------------------------

# Each tuple: (id, injector, factory, call-factory)
#   • id         – readable test id used by pytest
#   • injector   – which kwarg of MemoryManager should receive the manager
#                  ("contact" | "transcript" | "knowledge" | "none")
#   • factory    – zero-arg callable that returns a *fresh* manager instance
#   • call_fn    – lambda that, given the manager, triggers ONE public method
#                  and yields an awaitable SteerableToolHandle.

MANAGER_TEST_CASES: Tuple[
    Tuple[str, str, _ManagerFactory, Callable[[Any], Any]],
    ...,
] = (
    (
        "contact_ask",
        "contact",
        lambda: SimulatedContactManager(log_events=True),
        lambda m: m.ask("Hello from contact ask."),
    ),
    (
        "contact_update",
        "contact",
        lambda: SimulatedContactManager(log_events=True),
        lambda m: m.update("Please create a new imaginary contact."),
    ),
    (
        "transcript_ask",
        "transcript",
        lambda: SimulatedTranscriptManager(log_events=True),
        lambda m: m.ask("What's the latest message?"),
    ),
    (
        "transcript_summarize",
        "transcript",
        lambda: SimulatedTranscriptManager(log_events=True),
        lambda m: m.summarize(from_messages=[1]),
    ),
    (
        "knowledge_ask",
        "knowledge",
        lambda: SimulatedKnowledgeManager(log_events=True),
        lambda m: m.ask("Tell me what we know about batteries."),
    ),
    (
        "knowledge_update",
        "knowledge",
        lambda: SimulatedKnowledgeManager(log_events=True),
        lambda m: m.update("Store that Tesla batteries last 8 years."),
    ),
    (
        "knowledge_refactor",
        "knowledge",
        lambda: SimulatedKnowledgeManager(log_events=True),
        lambda m: m.refactor("Normalise manufacturer tables."),
    ),
    (
        "taskscheduler_ask",
        "none",
        lambda: SimulatedTaskScheduler(log_events=True),
        lambda m: m.ask("Which tasks are due tomorrow?"),
    ),
    (
        "taskscheduler_update",
        "none",
        lambda: SimulatedTaskScheduler(log_events=True),
        lambda m: m.update("Add a task to send summary email tomorrow."),
    ),
    (
        "taskscheduler_execute",
        "none",
        lambda: SimulatedTaskScheduler(log_events=True),
        lambda m: m.execute_task(1),
    ),
)

# ---------------------------------------------------------------------------
#  Manager-specific parameter subsets                                        |
# ---------------------------------------------------------------------------

CONTACT_TEST_CASES = [c for c in MANAGER_TEST_CASES if c[1] == "contact"]
TRANSCRIPT_TEST_CASES = [c for c in MANAGER_TEST_CASES if c[1] == "transcript"]
KNOWLEDGE_TEST_CASES = [c for c in MANAGER_TEST_CASES if c[1] == "knowledge"]
TASKSCHEDULER_TEST_CASES = [
    c for c in MANAGER_TEST_CASES if c[0].startswith("taskscheduler")
]

# ---------------------------------------------------------------------------
#  Test-specific: shrink window sizes & patch MemoryManager for fast roll-ups |
# ---------------------------------------------------------------------------


# Small deterministic windows so that a handful of manager calls exercise **all**
# hierarchy levels without long waits or hundreds of calls.  They keep the
# original naming order so the pretty-printing helper renders the same labels.

SMALL_COUNT_WINDOWS = {
    "past_interaction": 1,
    "past_10_interactions": 2,
    "past_40_interactions": 4,
    "past_120_interactions": 8,
    "past_520_interactions": 16,
}

SMALL_TIME_WINDOWS = {
    "past_day": 1,  # ‘1 day’ → keep wording ‘Past Day’
    "past_week": 2,  # 2 days
    "past_4_weeks": 4,  # 4 days
    "past_12_weeks": 8,  # 8 days
    "past_52_weeks": 16,  # 16 days
}

# Preserve the canonical window order from the production code
SMALL_COUNT_ORDER = [
    "past_interaction",
    "past_10_interactions",
    "past_40_interactions",
    "past_120_interactions",
    "past_520_interactions",
]

SMALL_TIME_ORDER = [
    "past_day",
    "past_week",
    "past_4_weeks",
    "past_12_weeks",
    "past_52_weeks",
]

# Build parent-mapping dicts exactly like the production class-level code.

SMALL_COUNT_PARENT: dict[str, tuple[str, int]] = {}
for i in range(1, len(SMALL_COUNT_ORDER)):
    child, parent = SMALL_COUNT_ORDER[i], SMALL_COUNT_ORDER[i - 1]
    SMALL_COUNT_PARENT[child] = (
        parent,
        SMALL_COUNT_WINDOWS[child] // SMALL_COUNT_WINDOWS[parent],
    )

SMALL_TIME_PARENT: dict[str, tuple[str, int]] = {}
for i in range(1, len(SMALL_TIME_ORDER)):
    child, parent = SMALL_TIME_ORDER[i], SMALL_TIME_ORDER[i - 1]
    SMALL_TIME_PARENT[child] = (
        parent,
        SMALL_TIME_WINDOWS[child] // SMALL_TIME_WINDOWS[parent],
    )


def _patch_memory_manager_windows(monkeypatch):
    """Apply window-size overrides to *MemoryManager* and patch Subscription.

    Must be called **before** instantiating a new MemoryManager so that the
    constructor registers callbacks with the shrunken thresholds.
    """

    from unity.memory_manager.memory_manager import MemoryManager
    from unity.events.event_bus import Subscription

    # ---- override window constants --------------------------------------
    monkeypatch.setattr(
        MemoryManager,
        "_COUNT_WINDOWS",
        SMALL_COUNT_WINDOWS,
        raising=True,
    )
    monkeypatch.setattr(
        MemoryManager,
        "_TIME_WINDOWS",
        SMALL_TIME_WINDOWS,
        raising=True,
    )

    monkeypatch.setattr(MemoryManager, "_COUNT_ORDER", SMALL_COUNT_ORDER, raising=True)
    monkeypatch.setattr(MemoryManager, "_TIME_ORDER", SMALL_TIME_ORDER, raising=True)

    monkeypatch.setattr(
        MemoryManager,
        "_COUNT_PARENT",
        SMALL_COUNT_PARENT,
        raising=True,
    )
    monkeypatch.setattr(MemoryManager, "_TIME_PARENT", SMALL_TIME_PARENT, raising=True)

    # ---- monkey-patch the Subscription trigger so time-based windows fire
    #      immediately (no real waiting needed).

    def _test_should_trigger(self, evt):  # noqa: D401 – imperative helper
        # Preserve original count-based logic
        if self.count_step is not None:
            self.local_count += 1
            if self.local_count >= self.count_step:
                return True

        # For time-based subscriptions we trigger **every** event so higher-level
        # roll-ups occur without real-time delays.
        if self.time_step is not None:
            return True

        return False

    monkeypatch.setattr(
        Subscription,
        "should_trigger",
        _test_should_trigger,
        raising=True,
    )

    # ---- patch MemoryManager._build_activity_summary so headings use the
    #      *actual* threshold values from SMALL_COUNT_WINDOWS rather than the
    #      hard-coded digits embedded in the window name.  This keeps the
    #      rendered markdown in sync with the shrunken test setup.

    orig_build = MemoryManager._build_activity_summary

    def _patched_build(self, entries: dict[str, str], mode: str = "time") -> str:  # type: ignore[override]
        # Delegate to the original implementation first
        text = orig_build(self, entries, mode)

        # Adjust interaction-based headings (as before) *and* time-based
        # headings so that they reflect the **actual test thresholds** from
        # `SMALL_TIME_WINDOWS` instead of the literal names (e.g. “Week”).
        for win, thresh in SMALL_COUNT_WINDOWS.items():
            if win == "past_interaction":
                # Already handled explicitly in the original pretty helper
                continue

            # Extract the *original* number from the window name – e.g.
            # "past_10_interactions" → "10".
            try:
                original_num = win.split("_")[1]
            except IndexError:
                continue  # malformed, skip

            orig_heading = f"Past {original_num} Interactions"
            new_heading = (
                "Past Interaction" if thresh == 1 else f"Past {thresh} Interactions"
            )
            # Replace both the level-2 heading line and any inline mentions
            text = text.replace(orig_heading, new_heading)

        # ---- patch time headings --------------------------------------
        for win, thresh in SMALL_TIME_WINDOWS.items():
            if win == "past_day":
                continue  # “Past Day” remains unchanged

            # Build the original heading (Past Week / Past 4 Weeks / …)
            parts = win.split("_")
            orig_heading = "Past " + " ".join(
                p.capitalize() if not p.isdigit() else p for p in parts[1:]
            )

            # New replacement heading reflecting the *day* threshold
            plural = "Days" if thresh != 1 else "Day"
            new_heading = f"Past {thresh} {plural}"

            text = text.replace(orig_heading, new_heading)

        return text

    monkeypatch.setattr(
        MemoryManager,
        "_build_activity_summary",
        _patched_build,
        raising=True,
    )


# ---------------------------------------------------------------------------
#  Shared helper to run a manager test case                                   |
# ---------------------------------------------------------------------------


async def _run_manager_case(
    manager: Any,
    mm: MemoryManager,
    call_factories: list[Callable[[Any], Any]],
    n_calls: int,
    case_id: str,
):
    """Execute *n_calls* randomly chosen method calls and assert log count."""

    await mm.reset()

    from unity.events.event_bus import EVENT_BUS

    # Record baseline number of rolling activity logs
    baseline_logs = len(
        unify.get_logs(
            context=mm._rolling_ctx,
        ),
    )

    rng = random.Random(42)  # deterministic

    # Perform the manager calls
    for _ in range(n_calls):
        factory = rng.choice(call_factories)
        handle = await factory(manager)
        await handle.result()

    # Flush events & *all* callbacks, including nested cascades triggered by
    # lower-level summaries, using the new `cascade=True` flag.
    EVENT_BUS.join_published()
    EVENT_BUS.join_callbacks(cascade=True)

    # Build summary (interaction mode) – should be non-empty
    summary = mm.get_rolling_activity(mode="interaction")
    assert (
        summary.strip()
    ), f"Expected non-empty summary for {case_id} after {n_calls} call(s)"

    # ------------------------------------------------------------------
    # 1.  Expected headings (dynamic)
    # ------------------------------------------------------------------

    _manager_titles = {
        "ContactManager": "# Contacts",
        "TranscriptManager": "# Transcripts",
        "KnowledgeManager": "# Knowledge",
        "TaskScheduler": "# Tasks",
    }
    expected_title = _manager_titles.get(case_id)
    if expected_title is None:
        raise AssertionError(f"No expected title mapping for case_id={case_id}")

    # Pretty helper (mirrors MemoryManager._build_activity_summary)
    def _pretty(w: str) -> str:
        if w == "all_time":
            return "All Time"
        # Special-case the single-interaction window so we keep the historic wording
        if w == "past_interaction":
            return "Past Interaction"

        # For interaction-based windows use the *configured* threshold from
        # SMALL_COUNT_WINDOWS so the heading reflects the **actual** value
        # (powers-of-two in the shrunk test setup) instead of the literal
        # number embedded in the name (e.g. "past_10_interactions").
        if w.endswith("_interactions") and w in SMALL_COUNT_WINDOWS:
            threshold = SMALL_COUNT_WINDOWS[w]
            plural = "Interactions" if threshold != 1 else "Interaction"
            return f"Past {threshold} {plural}"

        # For time-based windows use SMALL_TIME_WINDOWS thresholds so the
        # heading matches the patched _build_activity_summary output.
        if w in SMALL_TIME_WINDOWS:
            thresh_t = SMALL_TIME_WINDOWS[w]
            if w == "past_day":
                return "Past Day"
            plural = "Days" if thresh_t != 1 else "Day"
            return f"Past {thresh_t} {plural}"

        # Fallback – keep the original behaviour
        parts = w.split("_")
        return "Past " + " ".join(
            p.capitalize() if not p.isdigit() else p for p in parts[1:]
        )

    # --- interaction-mode windows triggered so far
    # Windows with a threshold **up to and including** the executed number of
    # calls should now be present in the summary because each window triggers
    # as soon as its `count_step` or `time_step` is reached.
    active_count: list[str] = []
    for w in SMALL_COUNT_ORDER:
        thresh = SMALL_COUNT_WINDOWS[w]
        if w == "past_interaction":
            if n_calls >= thresh:
                active_count.append(w)
        else:
            # With the small deterministic windows each higher-level summary
            # is emitted after exactly **`thresh`** ManagerMethod events – the
            # ratio between successive windows is 2×, but one summary of the
            # parent window is sufficient to trigger the child.  Therefore we
            # mark the window active as soon as `n_calls >= thresh`.
            if n_calls >= thresh:
                active_count.append(w)
    expected_interaction_headings = {expected_title} | {
        f"## {_pretty(w)}" for w in active_count
    }

    # --- time-mode windows triggered so far
    active_time: list[str] = []
    for w in SMALL_TIME_ORDER:
        thresh = SMALL_TIME_WINDOWS[w]
        if w == "past_day":
            # Base-level time window fires immediately after the first event.
            if n_calls >= 1:
                active_time.append(w)
        else:
            # Same reasoning as for interaction windows: one lower-level
            # summary per `thresh` events means the child window is triggered
            # after exactly `thresh` ManagerMethod events.
            if n_calls >= thresh:
                active_time.append(w)

    time_summary = mm.get_rolling_activity(mode="time")
    expected_time_headings = {expected_title} | {
        f"## {_pretty(w)}" for w in active_time
    }

    # --- Positive presence checks --------------------------------------
    for hdr in expected_interaction_headings:
        assert (
            hdr in summary
        ), f"Expected heading '{hdr}' in interaction summary for {case_id}.\nSummary:\n{summary}"

    for hdr in expected_time_headings:
        assert (
            hdr in time_summary
        ), f"Expected heading '{hdr}' in time summary for {case_id}.\nSummary:\n{time_summary}"

    # ------------------------------------------------------------------
    # 2.  Verify that at least the expected minimum number of snapshot rows
    #     were created.  Each call produces one *count* and one *time* snapshot.
    #     Additional rows appear when higher-level roll-ups are written.
    # ------------------------------------------------------------------

    total_logs = len(
        unify.get_logs(
            context=mm._rolling_ctx,
        ),
    )
    new_logs = total_logs - baseline_logs
    min_expected_rows = n_calls * 2  # 1 count + 1 time per call
    assert new_logs >= min_expected_rows, (
        f"Expected at least {min_expected_rows} rolling activity logs for {case_id} after {n_calls} call(s), "
        f"found {new_logs}."
    )

    # ---------------- Negative assertions – no unexpected headings ---------
    def _extract_headings(text: str) -> set[str]:
        """Return all lines that start with one or more '#' characters."""
        return {ln.strip() for ln in text.splitlines() if ln.lstrip().startswith("#")}

    # Interaction summary should contain *only* the dynamically expected headings.
    interaction_headings = _extract_headings(summary)

    unexpected_interaction = interaction_headings - expected_interaction_headings
    assert not unexpected_interaction, (
        "Found unexpected headings in interaction-based rolling activity summary "
        f"for {case_id}: {unexpected_interaction}. Full summary:\n{summary}"
    )

    # Time-based summary should contain *only* the dynamically expected headings.
    time_headings = _extract_headings(time_summary)
    unexpected_time = time_headings - expected_time_headings
    assert not unexpected_time, (
        "Found unexpected headings in time-based rolling activity summary "
        f"for {case_id}: {unexpected_time}. Full summary:\n{time_summary}"
    )


# ---------------------------------------------------------------------------
#  Build (n_calls) list – we only test 1 and 2 calls for now                 |
# ---------------------------------------------------------------------------

_N_CALLS_TO_TEST = [1, 2, 4, 8, 16]

# ---------------------------------------------------------------------------
#  Build lists of call_factories per manager ---------------------------------

CONTACT_CALL_FACTORIES = [c[3] for c in CONTACT_TEST_CASES]
CONTACT_MANAGER_FACTORY = CONTACT_TEST_CASES[0][2]

TRANSCRIPT_CALL_FACTORIES = [c[3] for c in TRANSCRIPT_TEST_CASES]
TRANSCRIPT_MANAGER_FACTORY = TRANSCRIPT_TEST_CASES[0][2]

KNOWLEDGE_CALL_FACTORIES = [c[3] for c in KNOWLEDGE_TEST_CASES]
KNOWLEDGE_MANAGER_FACTORY = KNOWLEDGE_TEST_CASES[0][2]

TASK_CALL_FACTORIES = [c[3] for c in TASKSCHEDULER_TEST_CASES]
TASK_MANAGER_FACTORY = TASKSCHEDULER_TEST_CASES[0][2]

# ---------------------------------------------------------------------------
#  ContactManager specific tests                                             |
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@_handle_project
@pytest.mark.parametrize("n_calls", _N_CALLS_TO_TEST)
async def test_contact_manager_methods_populate_interaction_rolling_activity(
    monkeypatch,
    n_calls,
):
    _patch_memory_manager_windows(monkeypatch)
    manager = CONTACT_MANAGER_FACTORY()
    mm = MemoryManager(contact_manager=manager)

    await _run_manager_case(
        manager,
        mm,
        CONTACT_CALL_FACTORIES,
        n_calls,
        "ContactManager",
    )


# ---------------------------------------------------------------------------
#  TranscriptManager specific tests                                          |
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@_handle_project
@pytest.mark.parametrize("n_calls", _N_CALLS_TO_TEST)
async def test_transcript_manager_methods_populate_interaction_rolling_activity(
    monkeypatch,
    n_calls,
):
    _patch_memory_manager_windows(monkeypatch)
    manager = TRANSCRIPT_MANAGER_FACTORY()
    mm = MemoryManager(transcript_manager=manager)

    await _run_manager_case(
        manager,
        mm,
        TRANSCRIPT_CALL_FACTORIES,
        n_calls,
        "TranscriptManager",
    )


# ---------------------------------------------------------------------------
#  KnowledgeManager specific tests                                           |
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@_handle_project
@pytest.mark.parametrize("n_calls", _N_CALLS_TO_TEST)
async def test_knowledge_manager_methods_populate_interaction_rolling_activity(
    monkeypatch,
    n_calls,
):
    _patch_memory_manager_windows(monkeypatch)
    manager = KNOWLEDGE_MANAGER_FACTORY()
    mm = MemoryManager(knowledge_manager=manager)

    await _run_manager_case(
        manager,
        mm,
        KNOWLEDGE_CALL_FACTORIES,
        n_calls,
        "KnowledgeManager",
    )


# ---------------------------------------------------------------------------
#  TaskScheduler specific tests                                              |
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@_handle_project
@pytest.mark.parametrize("n_calls", _N_CALLS_TO_TEST)
async def test_taskscheduler_methods_populate_interaction_rolling_activity(
    monkeypatch,
    n_calls,
):
    _patch_memory_manager_windows(monkeypatch)
    manager = TASK_MANAGER_FACTORY()
    mm = MemoryManager()

    await _run_manager_case(
        manager,
        mm,
        TASK_CALL_FACTORIES,
        n_calls,
        "TaskScheduler",
    )


# ---------------------------------------------------------------------------
#  New tests – verify time-based rolling activity summaries include **all**
#  expected sub-headings for every manager (no window-shrinking monkey-patch)
# ---------------------------------------------------------------------------

import datetime as dt

from unity.events.event_bus import EVENT_BUS, Event
from unity.memory_manager.memory_manager import MemoryManager


_TIME_BASED_HEADINGS = [
    "## Past Day",
    "## Past Week",
    "## Past 4 Weeks",
    "## Past 12 Weeks",
    "## Past 52 Weeks",
]

# Manager class-names registered in MemoryManager._MANAGERS
_TIME_SUMMARY_MANAGERS = [
    "ContactManager",
    "TranscriptManager",
    "KnowledgeManager",
    "TaskScheduler",
    "Conductor",
]

# Mapping of window → threshold in *days*
_TIME_WINDOW_THRESHOLDS = {
    "past_day": 1,
    "past_week": 2,
    "past_4_weeks": 4,
    "past_12_weeks": 8,
    "past_52_weeks": 16,
}


def _pretty_time_window(w: str) -> str:
    """Return heading reflecting SMALL_TIME_WINDOWS thresholds."""
    thresh = SMALL_TIME_WINDOWS.get(w)
    if thresh is None:
        # fallback to original naming
        parts = w.split("_")
        return "Past " + " ".join(
            p.capitalize() if not p.isdigit() else p for p in parts[1:]
        )
    if thresh == 1:
        return "Past Day"
    return f"Past {thresh} Days"


async def _assert_time_based_headings_for_manager(
    mgr_cls: str,
    total_days: int,
    monkeypatch,
):
    """Publish enough *ManagerMethod* events with simulated timestamps so that
    all time-based windows *up to* the size implied by *total_days* are generated,
    then verify the presence of the corresponding headings in the summary.
    """

    # Fresh MemoryManager instance (ensures callback subscriptions) ----------
    _patch_memory_manager_windows(monkeypatch)
    mm = MemoryManager()
    await mm.reset()

    # Publish one event per *simulated* day for the requested span ----------
    base_ts = dt.datetime(2025, 1, 1, tzinfo=dt.UTC)

    for day in range(total_days):
        ts = base_ts + dt.timedelta(days=day)
        await EVENT_BUS.publish(
            Event(
                type="ManagerMethod",
                timestamp=ts,
                payload={
                    "manager": mgr_cls,
                    "method": "unit_test",
                    "phase": "outgoing",
                },
            ),
        )

    # Flush logger & wait for all cascading callbacks -----------------------
    EVENT_BUS.join_published()
    EVENT_BUS.join_callbacks(cascade=True)

    # Retrieve the *time-based* rolling activity summary --------------------
    summary = mm.get_rolling_activity(mode="time")

    # Determine which windows should have triggered given total_days
    triggered_windows = [
        w for w, thresh in _TIME_WINDOW_THRESHOLDS.items() if total_days >= thresh
    ]

    expected_headings = {f"## {_pretty_time_window(w)}" for w in triggered_windows}

    missing = [hdr for hdr in expected_headings if hdr not in summary]
    assert (
        not missing
    ), f"Missing expected heading(s) for {mgr_cls} after {total_days} day(s): {missing}.\nSummary:\n{summary}"


# ---------------------------------------------------------------------------
#  Per-manager TIME-based rolling-activity tests                             |
# ---------------------------------------------------------------------------

_TIME_DAYS_TO_TEST = [1, 2, 4, 8, 16]


@pytest.mark.asyncio
@_handle_project
@pytest.mark.parametrize(
    "total_days",
    _TIME_DAYS_TO_TEST,
    ids=["1_day", "2_days", "4_days", "8_days", "16_days"],
)
async def test_contact_manager_methods_populate_time_rolling_activity(
    monkeypatch,
    total_days,
):
    await _assert_time_based_headings_for_manager(
        "ContactManager",
        total_days,
        monkeypatch,
    )


@pytest.mark.asyncio
@_handle_project
@pytest.mark.parametrize(
    "total_days",
    _TIME_DAYS_TO_TEST,
    ids=["1_day", "2_days", "4_days", "8_days", "16_days"],
)
async def test_transcript_manager_methods_populate_time_rolling_activity(
    monkeypatch,
    total_days,
):
    await _assert_time_based_headings_for_manager(
        "TranscriptManager",
        total_days,
        monkeypatch,
    )


@pytest.mark.asyncio
@_handle_project
@pytest.mark.parametrize(
    "total_days",
    _TIME_DAYS_TO_TEST,
    ids=["1_day", "2_days", "4_days", "8_days", "16_days"],
)
async def test_knowledge_manager_methods_populate_time_rolling_activity(
    monkeypatch,
    total_days,
):
    await _assert_time_based_headings_for_manager(
        "KnowledgeManager",
        total_days,
        monkeypatch,
    )


@pytest.mark.asyncio
@_handle_project
@pytest.mark.parametrize(
    "total_days",
    _TIME_DAYS_TO_TEST,
    ids=["1_day", "2_days", "4_days", "8_days", "16_days"],
)
async def test_taskscheduler_methods_populate_time_rolling_activity(
    monkeypatch,
    total_days,
):
    await _assert_time_based_headings_for_manager(
        "TaskScheduler",
        total_days,
        monkeypatch,
    )
