import asyncio
import datetime as dt
import functools
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
        contact_manager=SimulatedContactManager(
            description="Assume the contact list starts out **totally empty**",
        ),
        transcript_manager=SimulatedTranscriptManager(
            description="Assume the transcripts start out **totally empty**",
        ),
        knowledge_manager=SimulatedKnowledgeManager(
            description="Assume the knowledge base starts out **totally empty**",
        ),
        task_scheduler=SimulatedTaskScheduler(
            description="Assume the task list starts out **totally empty**",
        ),
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
    # Track how many times the *passive* MemoryManager.update_knowledge helper fires
    mm_kb_counter: Dict[str, int] = {"calls": 0}

    # Preserve the original coroutine so we can delegate after incrementing our counter
    original_mm_update_knowledge = MemoryManager.update_knowledge

    @functools.wraps(original_mm_update_knowledge)
    async def _stub_mm_update_knowledge(
        self,
        *args,
        **kwargs,
    ):  # noqa: D401 – imperative helper
        mm_kb_counter["calls"] += 1
        # Delegate to the real implementation so KnowledgeManager.update is still invoked
        return await original_mm_update_knowledge(self, *args, **kwargs)

    monkeypatch.setattr(
        MemoryManager,
        "update_knowledge",
        _stub_mm_update_knowledge,
        raising=True,
    )

    _make_mm(monkeypatch, kb_counter)

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
    EVENT_BUS.join_callbacks()

    # Wait for chunk processing
    await asyncio.sleep(0.2)

    # The *passive* MemoryManager.update_knowledge helper itself MUST still run once
    assert (
        mm_kb_counter["calls"] >= 1
    ), "MemoryManager.update_knowledge should still execute for the chunk"

    # Passive update_knowledge should NOT invoke KnowledgeManager.update again
    assert (
        kb_counter["calls"] == 0
    ), "KnowledgeManager.update should NOT be called when explicit ConversationManager call exists"


# ---------------------------------------------------------------------------
#  2.  Shield does **not** block when explicit call targets a different manager
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@_handle_project
async def test_prompt_shield_allows_km_update_when_irrelevant_explicit_call(
    monkeypatch,
):
    kb_counter: Dict[str, int] = {"calls": 0}
    # Track how many times the *passive* MemoryManager.update_knowledge helper fires
    mm_kb_counter: Dict[str, int] = {"calls": 0}

    # Patch the coroutine to increment our counter while remaining lightweight
    # Preserve the original coroutine so we can delegate after incrementing our counter
    original_mm_update_knowledge = MemoryManager.update_knowledge

    @functools.wraps(original_mm_update_knowledge)
    async def _stub_mm_update_knowledge(self, *args, **kwargs):  # noqa: D401
        mm_kb_counter["calls"] += 1
        # Delegate to the real implementation so KnowledgeManager.update is still invoked
        return await original_mm_update_knowledge(self, *args, **kwargs)

    monkeypatch.setattr(
        MemoryManager,
        "update_knowledge",
        _stub_mm_update_knowledge,
        raising=True,
    )

    mm = _make_mm(monkeypatch, kb_counter)

    await asyncio.sleep(0.05)

    # Publish one chat message + explicit *ContactManager.update* event
    ts_base = dt.datetime(2025, 1, 2, tzinfo=dt.UTC)
    msg = Message(
        medium=Medium.SMS_MESSAGE,
        sender_id=1,
        receiver_ids=[0],
        timestamp=ts_base,
        content="Please remember this import fact: the office is *always* closed on a Friday",
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
    EVENT_BUS.join_callbacks()

    # Wait for chunk processing
    await asyncio.sleep(0.2)

    # We expect the MemoryManager.update_knowledge helper itself to have run exactly once
    assert (
        mm_kb_counter["calls"] >= 1
    ), "MemoryManager.update_knowledge should execute for the chunk"

    # Passive update_knowledge SHOULD still invoke KnowledgeManager.update
    assert (
        kb_counter["calls"] >= 1
    ), "KnowledgeManager.update should fire when no explicit KM.update present in the chunk"
