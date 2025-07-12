"""
Simple unit test for `MemoryManager.get_rolling_activity`.

Ensures that when **no** activity has been recorded yet, the helper
returns an *empty* string – callers can then safely omit the Historic
Activity block from prompts.
"""

from __future__ import annotations


import unify
import asyncio

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

from unity.contact_manager.simulated import SimulatedContactManager
from unity.transcript_manager.simulated import SimulatedTranscriptManager
from unity.knowledge_manager.simulated import SimulatedKnowledgeManager
from unity.task_scheduler.simulated import SimulatedTaskScheduler

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

# Shared helper to run a manager test case ----------------------------------


async def _run_manager_case(
    injector: str,
    manager_factory: _ManagerFactory,
    call_factories: list[Callable[[Any], Any]],
    n_calls: int,
    expected_summaries: int,
    case_id: str,
):
    """Execute *n_calls* randomly chosen method calls and assert summary headings."""

    from unity.memory_manager.memory_manager import MemoryManager
    from unity.events.event_bus import EVENT_BUS

    EVENT_BUS.reset()

    manager = manager_factory()

    # Wire the chosen manager into MemoryManager where possible
    if injector == "contact":
        mm = MemoryManager(contact_manager=manager)
    elif injector == "transcript":
        mm = MemoryManager(transcript_manager=manager)
    elif injector == "knowledge":
        mm = MemoryManager(knowledge_manager=manager)
    else:
        mm = MemoryManager()

    # Allow async callback setup to complete
    await asyncio.sleep(0.05)

    # Baseline – should be empty
    assert not unify.get_logs(
        context=mm._rolling_ctx,
        limit=1,
    ), "RollingActivity context not empty before test"

    rng = random.Random(42)  # deterministic

    # Perform the manager calls
    for _ in range(n_calls):
        factory = rng.choice(call_factories)
        handle = await factory(manager)
        await handle.result()

    # Flush events & callbacks
    EVENT_BUS.join_published()
    EVENT_BUS.join_callbacks()

    # Build summary (interaction mode)
    summary = mm.get_rolling_activity(mode="interaction")
    assert (
        summary.strip()
    ), f"Expected non-empty summary for {case_id} after {n_calls} call(s)"

    # Count headings for individual windows (lines starting with '## ')
    heading_lines = [ln for ln in summary.splitlines() if ln.startswith("## ")]
    assert len(heading_lines) == expected_summaries, (
        f"Expected {expected_summaries} summary headings for {case_id} after {n_calls} call(s), "
        f"found {len(heading_lines)}.\nSummary:\n{summary}"
    )


# ---------------------------------------------------------------------------
#  Build (n_calls, expected) pairs based on MemoryManager._COUNT_WINDOWS      |
# ---------------------------------------------------------------------------

from unity.memory_manager.memory_manager import MemoryManager as _MM

_COUNT_THRESHOLDS = sorted(set(_MM._COUNT_WINDOWS.values()))  # ascending
_CALLS_EXPECTED_PAIRS = [
    ((cnt + 1) // 2, idx + 1)  # half the events → method calls, round up
    for idx, cnt in enumerate(_COUNT_THRESHOLDS)
]

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
@pytest.mark.parametrize("n_calls, expected_summaries", _CALLS_EXPECTED_PAIRS)
async def test_contact_manager_methods_populate_rolling_activity(
    n_calls,
    expected_summaries,
):
    await _run_manager_case(
        "contact",
        CONTACT_MANAGER_FACTORY,
        CONTACT_CALL_FACTORIES,
        n_calls,
        expected_summaries,
        "ContactManager",
    )


# ---------------------------------------------------------------------------
#  TranscriptManager specific tests                                          |
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@_handle_project
@pytest.mark.parametrize("n_calls, expected_summaries", _CALLS_EXPECTED_PAIRS)
async def test_transcript_manager_methods_populate_rolling_activity(
    n_calls,
    expected_summaries,
):
    await _run_manager_case(
        "transcript",
        TRANSCRIPT_MANAGER_FACTORY,
        TRANSCRIPT_CALL_FACTORIES,
        n_calls,
        expected_summaries,
        "TranscriptManager",
    )


# ---------------------------------------------------------------------------
#  KnowledgeManager specific tests                                           |
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@_handle_project
@pytest.mark.parametrize("n_calls, expected_summaries", _CALLS_EXPECTED_PAIRS)
async def test_knowledge_manager_methods_populate_rolling_activity(
    n_calls,
    expected_summaries,
):
    await _run_manager_case(
        "knowledge",
        KNOWLEDGE_MANAGER_FACTORY,
        KNOWLEDGE_CALL_FACTORIES,
        n_calls,
        expected_summaries,
        "KnowledgeManager",
    )


# ---------------------------------------------------------------------------
#  TaskScheduler specific tests                                              |
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@_handle_project
@pytest.mark.parametrize("n_calls, expected_summaries", _CALLS_EXPECTED_PAIRS)
async def test_taskscheduler_methods_populate_rolling_activity(
    n_calls,
    expected_summaries,
):
    await _run_manager_case(
        "none",
        TASK_MANAGER_FACTORY,
        TASK_CALL_FACTORIES,
        n_calls,
        expected_summaries,
        "TaskScheduler",
    )
