import asyncio
import datetime as dt
from typing import Dict

import pytest

from tests.helpers import _handle_project

from unity.events.event_bus import EVENT_BUS, Event
from unity.events.manager_event_logging import publish_manager_method_event, new_call_id
from unity.memory_manager.memory_manager import MemoryManager

# Simulated manager stand-ins (no external I/O)
from unity.contact_manager.simulated import SimulatedContactManager
from unity.transcript_manager.simulated import SimulatedTranscriptManager
from unity.knowledge_manager.simulated import SimulatedKnowledgeManager
from unity.task_scheduler.simulated import SimulatedTaskScheduler
from unity.transcript_manager.types.message import Message, Medium


@pytest.mark.asyncio
@_handle_project
async def test_explicit_conversation_manager_calls_are_visible_in_passive_chunk(
    monkeypatch,
):
    """A ConversationManager-originated explicit tool call should appear in the
    transcript chunk handed to *update_knowledge* so the model can reason that
    the knowledge was already stored explicitly.
    """

    # ---------------------------------------------------------------
    # 0.  Capture the transcript blob received by update_knowledge
    # ---------------------------------------------------------------
    captured: Dict[str, str] = {}

    async def _stub_update_knowledge(self, transcript: str, *_, **__):  # noqa: D401
        captured["transcript"] = transcript
        return "ok"

    # Keep the other heavy update helpers lightweight as well
    async def _noop(self, *_, **__):  # noqa: D401 – imperative helper
        return "noop"

    monkeypatch.setattr(
        MemoryManager,
        "update_knowledge",
        _stub_update_knowledge,
        raising=True,
    )
    monkeypatch.setattr(MemoryManager, "update_contacts", _noop, raising=True)
    monkeypatch.setattr(MemoryManager, "update_tasks", _noop, raising=True)
    monkeypatch.setattr(MemoryManager, "update_contact_bio", _noop, raising=True)
    monkeypatch.setattr(
        MemoryManager,
        "update_contact_rolling_summary",
        _noop,
        raising=True,
    )

    # ---------------------------------------------------------------
    # 1.  Instantiate MemoryManager with a tiny chunk size (3) so the
    #     test runs quickly.
    # ---------------------------------------------------------------
    mm = MemoryManager(
        contact_manager=SimulatedContactManager(description="shield-test"),
        transcript_manager=SimulatedTranscriptManager(description="shield-test"),
        knowledge_manager=SimulatedKnowledgeManager(description="shield-test"),
        task_scheduler=SimulatedTaskScheduler(description="shield-test"),
    )
    mm._CHUNK_SIZE = 3  # type: ignore[attr-defined]

    # Allow async callback registration
    await asyncio.sleep(0.05)

    # ---------------------------------------------------------------
    # 2.  Publish   ① one Message   +   ②/③ explicit ManagerMethod
    # ---------------------------------------------------------------
    base_ts = dt.datetime(2025, 1, 1, tzinfo=dt.UTC)

    # (①) plain chat message
    msg = Message(
        medium=Medium.SMS_MESSAGE,
        sender_id=1,
        receiver_ids=[0],
        timestamp=base_ts,
        content="Please remember this important fact.",
        exchange_id=1,
    )
    await EVENT_BUS.publish(Event(type="Message", payload=msg))

    # (②/③) explicit KnowledgeManager.update call triggered by ConversationManager
    call_id = new_call_id()
    await publish_manager_method_event(
        call_id,
        "KnowledgeManager",
        "update",
        phase="incoming",
        request="remember_fact",
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

    # Ensure events hit backend queues
    EVENT_BUS.join_published()

    # ---------------------------------------------------------------
    # 3.  Wait briefly for chunk processing
    # ---------------------------------------------------------------
    await asyncio.sleep(0.2)

    # ---------------------------------------------------------------
    # 4.  Assertions – update_knowledge was invoked once and its
    #     transcript blob contains the manager_method records.
    # ---------------------------------------------------------------
    blob = captured.get("transcript")
    assert blob is not None, "update_knowledge should have been called once"
    assert (
        '"kind": "manager_method"' in blob
    ), "ManagerMethod events must be included in the transcript chunk"
