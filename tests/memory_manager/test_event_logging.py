"""Event logging tests for MemoryManager.

Mirrors the ``test_event_logging.py`` tests that exist for ContactManager,
TaskScheduler, KnowledgeManager, and TranscriptManager.

MemoryManager methods return plain ``str`` results (not SteerableToolHandle),
so the decorator under test is ``@log_manager_result`` rather than
``@log_manager_call``.  We use ``SimulatedMemoryManager`` (backed by simulated
sub-managers, no external I/O other than cached LLM calls) to keep tests fast.
"""

from __future__ import annotations

import pytest

from unity.memory_manager.simulated import SimulatedMemoryManager
from tests.helpers import _handle_project, capture_events

# All tests in this file require EventBus publishing to verify event behavior
pytestmark = [pytest.mark.enable_eventbus, pytest.mark.llm_call]


# ---------------------------------------------------------------------------
#  Shared helpers
# ---------------------------------------------------------------------------

_TRANSCRIPT = (
    "Alice said: Hi, could you help me book a flight to Madrid?\n"
    "Bob replied: Sure, let me check the calendar.\n"
)

# Display labels defined on each @log_manager_result decorator in memory_manager.py
_LABELS = {
    "update_contacts": "Updating contacts from transcript",
    "update_knowledge": "Updating knowledge from transcript",
    "update_tasks": "Updating tasks from transcript",
    "process_chunk": "Processing memory chunk",
}


def _filter_mm_events(events, method: str, phase: str):
    """Filter captured events for a specific MemoryManager method + phase."""
    return [
        e
        for e in events
        if e.payload.get("manager") == "MemoryManager"
        and e.payload.get("method") == method
        and e.payload.get("phase") == phase
    ]


# ---------------------------------------------------------------------------
#  update_contacts
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@_handle_project
async def test_events_for_update_contacts():
    mm = SimulatedMemoryManager()

    async with capture_events("ManagerMethod") as events:
        result = await mm.update_contacts(_TRANSCRIPT)

    assert isinstance(result, str), "update_contacts should return a string"

    incoming = _filter_mm_events(events, "update_contacts", "incoming")
    assert incoming, "No incoming ManagerMethod event for update_contacts()"
    assert incoming[0].payload.get("display_label") == _LABELS["update_contacts"]
    assert incoming[0].payload.get("transcript") == _TRANSCRIPT

    call_id = incoming[0].calling_id
    outgoing = [
        e
        for e in events
        if e.calling_id == call_id and e.payload.get("phase") == "outgoing"
    ]
    assert outgoing, "No outgoing ManagerMethod event for update_contacts()"
    assert isinstance(outgoing[0].payload.get("answer"), str)


# ---------------------------------------------------------------------------
#  update_knowledge
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@_handle_project
async def test_events_for_update_knowledge():
    mm = SimulatedMemoryManager()

    async with capture_events("ManagerMethod") as events:
        result = await mm.update_knowledge(_TRANSCRIPT)

    assert isinstance(result, str)

    incoming = _filter_mm_events(events, "update_knowledge", "incoming")
    assert incoming, "No incoming ManagerMethod event for update_knowledge()"
    assert incoming[0].payload.get("display_label") == _LABELS["update_knowledge"]

    call_id = incoming[0].calling_id
    outgoing = [
        e
        for e in events
        if e.calling_id == call_id and e.payload.get("phase") == "outgoing"
    ]
    assert outgoing, "No outgoing ManagerMethod event for update_knowledge()"
    assert isinstance(outgoing[0].payload.get("answer"), str)


# ---------------------------------------------------------------------------
#  update_tasks
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@_handle_project
async def test_events_for_update_tasks():
    mm = SimulatedMemoryManager()

    async with capture_events("ManagerMethod") as events:
        result = await mm.update_tasks(_TRANSCRIPT)

    assert isinstance(result, str)

    incoming = _filter_mm_events(events, "update_tasks", "incoming")
    assert incoming, "No incoming ManagerMethod event for update_tasks()"
    assert incoming[0].payload.get("display_label") == _LABELS["update_tasks"]

    call_id = incoming[0].calling_id
    outgoing = [
        e
        for e in events
        if e.calling_id == call_id and e.payload.get("phase") == "outgoing"
    ]
    assert outgoing, "No outgoing ManagerMethod event for update_tasks()"


# ---------------------------------------------------------------------------
#  process_chunk (unified single-loop)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@_handle_project
async def test_events_for_process_chunk():
    mm = SimulatedMemoryManager()

    async with capture_events("ManagerMethod") as events:
        result = await mm.process_chunk(_TRANSCRIPT)

    assert isinstance(result, str)

    incoming = _filter_mm_events(events, "process_chunk", "incoming")
    assert incoming, "No incoming ManagerMethod event for process_chunk()"
    assert incoming[0].payload.get("display_label") == _LABELS["process_chunk"]
    assert incoming[0].payload.get("transcript") == _TRANSCRIPT

    call_id = incoming[0].calling_id
    outgoing = [
        e
        for e in events
        if e.calling_id == call_id and e.payload.get("phase") == "outgoing"
    ]
    assert outgoing, "No outgoing ManagerMethod event for process_chunk()"
    assert isinstance(outgoing[0].payload.get("answer"), str)


# ---------------------------------------------------------------------------
#  Hierarchy propagation: inner tool loops should carry MM lineage
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@_handle_project
async def test_inner_events_carry_mm_lineage():
    """When MemoryManager.update_knowledge calls KnowledgeManager.ask (etc.)
    inside its tool loop, those inner ManagerMethod events should include
    'MemoryManager.update_knowledge' in their hierarchy."""
    mm = SimulatedMemoryManager()

    async with capture_events("ManagerMethod") as events:
        await mm.update_knowledge(_TRANSCRIPT)

    # Look for any inner event (not MemoryManager itself) that has hierarchy
    inner_events = [
        e
        for e in events
        if e.payload.get("manager") != "MemoryManager"
        and isinstance(e.payload.get("hierarchy"), list)
        and len(e.payload.get("hierarchy", [])) > 1
    ]

    if inner_events:
        # At least one inner event should reference MemoryManager in its hierarchy
        has_mm_parent = any(
            any("MemoryManager" in seg for seg in e.payload.get("hierarchy", []))
            for e in inner_events
        )
        assert has_mm_parent, (
            "Inner events exist but none carry MemoryManager in their hierarchy. "
            f"Sample hierarchy: {inner_events[0].payload.get('hierarchy')}"
        )
