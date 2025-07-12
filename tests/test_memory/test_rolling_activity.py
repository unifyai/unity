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

# Shared helper to run a manager test case ----------------------------------


async def _run_manager_case(
    manager: Any,
    mm: MemoryManager,
    call_factories: list[Callable[[Any], Any]],
    n_calls: int,
    case_id: str,
):
    """Execute *n_calls* randomly chosen method calls and assert log count."""

    from unity.events.event_bus import EVENT_BUS

    EVENT_BUS.reset()

    # Allow async callback setup to complete
    await asyncio.sleep(0.05)

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

    # Flush events & callbacks
    EVENT_BUS.join_published()
    EVENT_BUS.join_callbacks()

    # Build summary (interaction mode) – should be non-empty
    summary = mm.get_rolling_activity(mode="interaction")
    assert (
        summary.strip()
    ), f"Expected non-empty summary for {case_id} after {n_calls} call(s)"

    # Verify that exactly *n_calls* new rolling activity logs were created
    total_logs = len(
        unify.get_logs(
            context=mm._rolling_ctx,
        ),
    )
    new_logs = total_logs - baseline_logs
    assert new_logs == n_calls + 1, (
        f"Expected {n_calls} rolling activity logs for {case_id} after {n_calls} call(s), "
        f"found {new_logs}.\nSummary:\n{summary}"
    )


# ---------------------------------------------------------------------------
#  Build (n_calls) list – we only test 1 and 2 calls for now                 |
# ---------------------------------------------------------------------------

_N_CALLS_TO_TEST = [1, 2]

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
async def test_contact_manager_methods_populate_rolling_activity(n_calls):
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
async def test_transcript_manager_methods_populate_rolling_activity(n_calls):
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
async def test_knowledge_manager_methods_populate_rolling_activity(n_calls):
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
async def test_taskscheduler_methods_populate_rolling_activity(n_calls):
    manager = TASK_MANAGER_FACTORY()
    mm = MemoryManager()

    await _run_manager_case(
        manager,
        mm,
        TASK_CALL_FACTORIES,
        n_calls,
        "TaskScheduler",
    )
