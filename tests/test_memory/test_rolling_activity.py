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


@pytest.mark.asyncio
@_handle_project
@pytest.mark.parametrize(
    "case_id, injector, manager_factory, call_factory",
    MANAGER_TEST_CASES,
    ids=[c[0] for c in MANAGER_TEST_CASES],
)
async def test_manager_methods_populate_rolling_activity(
    case_id,
    injector,
    manager_factory,
    call_factory,
):
    """Ensure that *every* simulated manager method produces a Rolling-Activity snapshot."""

    from unity.memory_manager.memory_manager import MemoryManager
    from unity.events.event_bus import EVENT_BUS

    EVENT_BUS.reset()

    # Fresh manager instance (emits ManagerMethod events)
    manager = manager_factory()

    # Wire the chosen manager into MemoryManager where possible
    if injector == "contact":
        mm = MemoryManager(contact_manager=manager)
    elif injector == "transcript":
        mm = MemoryManager(transcript_manager=manager)
    elif injector == "knowledge":
        mm = MemoryManager(knowledge_manager=manager)
    else:
        # TaskScheduler or unsupported injector – MemoryManager still registers callbacks
        mm = MemoryManager()

    # Allow async callback setup to complete
    await asyncio.sleep(0.05)

    # Baseline – current number of RollingActivity rows
    initial_rows = len(unify.get_logs(context=mm._rolling_ctx, limit=100))
    assert initial_rows == 0

    # Trigger **one** outgoing manager call via the provided factory
    handle = await call_factory(manager)
    await handle.result()

    # Ensure events & callbacks are fully processed
    EVENT_BUS.join_published()
    EVENT_BUS.join_callbacks()

    updated_rows = len(unify.get_logs(context=mm._rolling_ctx, limit=100))

    assert updated_rows == 1

    # A non-empty interaction summary must now be available
    summary = mm.get_rolling_activity(mode="interaction")
    assert summary.strip(), f"Expected non-empty summary for {case_id}"
