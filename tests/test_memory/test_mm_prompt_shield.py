import asyncio
import datetime as dt
from typing import Dict

import pytest

from tests.helpers import _handle_project

from unity.events.event_bus import EVENT_BUS, Event
from unity.events.manager_event_logging import publish_manager_method_event, new_call_id
from unity.memory_manager.memory_manager import MemoryManager

# Simulated manager test doubles (no external I/O)
from unity.contact_manager.simulated import SimulatedContactManager
from unity.transcript_manager.simulated import SimulatedTranscriptManager
from unity.knowledge_manager.simulated import SimulatedKnowledgeManager
from unity.task_scheduler.simulated import SimulatedTaskScheduler
from unity.transcript_manager.types.message import Message, Medium


# ---------------------------------------------------------------------------
#  Helper – factory that returns a **fresh** MemoryManager instance wired to
#           simulated sub-managers (so we can patch methods easily).
# ---------------------------------------------------------------------------


def _make_mm(monkeypatch, kb_counter: Dict[str, int]):
    """Return a new MemoryManager with patched KnowledgeManager.update."""

    async def _stub_kb_update(
        self,
        text: str,
        *_,
        **__,
    ):  # noqa: D401 – imperative helper
        kb_counter["calls"] += 1
        return "ok"

    # Patch before instantiation so the new instance picks up the stub.
    monkeypatch.setattr(
        SimulatedKnowledgeManager,
        "update",
        _stub_kb_update,
        raising=True,
    )

    # Keep *all other* heavy maintenance helpers lightweight
    async def _noop(self, *_, **__):  # noqa: D401 – imperative helper
        return "noop"

    monkeypatch.setattr(MemoryManager, "update_contacts", _noop, raising=True)
    monkeypatch.setattr(MemoryManager, "update_tasks", _noop, raising=True)
    monkeypatch.setattr(MemoryManager, "update_contact_bio", _noop, raising=True)
    monkeypatch.setattr(
        MemoryManager,
        "update_contact_rolling_summary",
        _noop,
        raising=True,
    )

    mm = MemoryManager(
        contact_manager=SimulatedContactManager(description="prompt-shield"),
        transcript_manager=SimulatedTranscriptManager(description="prompt-shield"),
        knowledge_manager=SimulatedKnowledgeManager(description="prompt-shield"),
        task_scheduler=SimulatedTaskScheduler(description="prompt-shield"),
    )

    # Shrink chunk size so tests run quickly
    mm._CHUNK_SIZE = 3  # type: ignore[attr-defined]

    return mm


# ---------------------------------------------------------------------------
#  1.  Shield *blocks* duplicate KnowledgeManager.update when explicit call
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@_handle_project
async def test_prompt_shield_blocks_duplicate_kb_update(monkeypatch):
    kb_counter: Dict[str, int] = {"calls": 0}
    mm = _make_mm(monkeypatch, kb_counter)

    # Allow async callback registration
    await asyncio.sleep(0.05)

    # Build events – one chat message + explicit KM.update (incoming/outgoing)
    ts_base = dt.datetime(2025, 1, 1, tzinfo=dt.UTC)
    msg = Message(
        medium=Medium.SMS_MESSAGE,
        sender_id=1,
        receiver_ids=[0],
        timestamp=ts_base,
        content="Remember the new SLA details.",
        exchange_id=1,
    )
    await EVENT_BUS.publish(Event(type="Message", payload=msg))

    call_id = new_call_id()
    await publish_manager_method_event(
        call_id,
        "KnowledgeManager",
        "update",
        phase="incoming",
        request="store_sla",
        source="ConversationManager",
    )
    await publish_manager_method_event(
        call_id,
        "KnowledgeManager",
        "update",
        phase="outgoing",
        result="stored",
        source="ConversationManager",
    )

    EVENT_BUS.join_published()

    # Wait for chunk processing
    await asyncio.sleep(0.2)

    # Passive update_knowledge should NOT invoke KnowledgeManager.update again
    assert (
        kb_counter["calls"] == 0
    ), "KnowledgeManager.update should NOT be called when explicit ConversationManager call exists"


# ---------------------------------------------------------------------------
#  2.  Shield does **not** block when explicit call targets a different manager
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@_handle_project
async def test_prompt_shield_allows_kb_update_when_irrelevant_explicit_call(
    monkeypatch,
):
    kb_counter: Dict[str, int] = {"calls": 0}
    mm = _make_mm(monkeypatch, kb_counter)

    await asyncio.sleep(0.05)

    # Publish one chat message + explicit *ContactManager.update* event
    ts_base = dt.datetime(2025, 1, 2, tzinfo=dt.UTC)
    msg = Message(
        medium=Medium.SMS_MESSAGE,
        sender_id=1,
        receiver_ids=[0],
        timestamp=ts_base,
        content="Please remember this knowledge fact.",
        exchange_id=1,
    )
    await EVENT_BUS.publish(Event(type="Message", payload=msg))

    call_id = new_call_id()
    await publish_manager_method_event(
        call_id,
        "ContactManager",
        "update",
        phase="incoming",
        request="update_contact",
        source="ConversationManager",
    )
    await publish_manager_method_event(
        call_id,
        "ContactManager",
        "update",
        phase="outgoing",
        result="stored",
        source="ConversationManager",
    )

    EVENT_BUS.join_published()

    # Wait for chunk processing
    await asyncio.sleep(0.2)

    # Passive update_knowledge SHOULD still invoke KnowledgeManager.update
    assert (
        kb_counter["calls"] >= 1
    ), "KnowledgeManager.update should fire when no explicit KM.update present in the chunk"
