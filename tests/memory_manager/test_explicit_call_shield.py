import asyncio
import datetime as dt
from typing import Dict

import pytest

from tests.helpers import _handle_project

from unity.events.event_bus import EVENT_BUS, Event

# This test publishes events to verify MemoryManager behavior
pytestmark = pytest.mark.enable_eventbus
from unity.events.manager_event_logging import publish_manager_method_event, new_call_id
from unity.memory_manager.memory_manager import MemoryManager

# Simulated manager stand-ins (no external I/O)
from unity.contact_manager.simulated import SimulatedContactManager
from unity.transcript_manager.simulated import SimulatedTranscriptManager
from unity.knowledge_manager.simulated import SimulatedKnowledgeManager
from unity.task_scheduler.simulated import SimulatedTaskScheduler
from unity.transcript_manager.types.message import Message
from unity.conversation_manager.types import Medium


@pytest.mark.asyncio
@_handle_project
async def test_explicit_calls_visible_in_passive_chunk(
    monkeypatch,
):
    """A ConversationManager-originated explicit tool call should appear in the
    transcript chunk handed to *process_chunk* so the model can reason that
    the knowledge was already stored explicitly.
    """

    # ---------------------------------------------------------------
    # 0.  Capture the transcript blob received by process_chunk
    # ---------------------------------------------------------------
    captured: Dict[str, str] = {}

    async def _stub_process_chunk(self, transcript: str, *_, **__):  # noqa: D401
        captured["transcript"] = transcript
        return "ok"

    monkeypatch.setattr(
        MemoryManager,
        "process_chunk",
        _stub_process_chunk,
        raising=True,
    )

    # ---------------------------------------------------------------
    # 1.  Instantiate MemoryManager with a tiny chunk size (3) so the
    #     test runs quickly.
    # ---------------------------------------------------------------
    mm = MemoryManager(
        contact_manager=SimulatedContactManager(
            description=(
                "TEST SCENARIO: Explicit ConversationManager calls are present. As a"
                " simulated ContactManager, act deterministically, accept updates, and"
                " avoid external I/O."
            ),
        ),
        transcript_manager=SimulatedTranscriptManager(
            description=(
                "TEST SCENARIO: Explicit ConversationManager calls are present. Return"
                " straightforward results for transcript queries."
            ),
        ),
        knowledge_manager=SimulatedKnowledgeManager(
            description=(
                "TEST SCENARIO: Explicit ConversationManager calls are present. If the"
                " transcript contains an explicit KnowledgeManager.update manager-method,"
                " treat the fact as already stored; otherwise behave as if absent."
            ),
        ),
        task_scheduler=SimulatedTaskScheduler(
            description=(
                "TEST SCENARIO: Explicit ConversationManager calls are present. Accept simple"
                " task updates deterministically and keep responses minimal."
            ),
        ),
    )
    mm._CHUNK_SIZE = 3  # type: ignore[attr-defined]

    # Allow async callback registration
    await asyncio.sleep(0.05)

    # ---------------------------------------------------------------
    # 2.  Publish one Message + explicit ManagerMethod events
    # ---------------------------------------------------------------
    base_ts = dt.datetime(2025, 1, 1, tzinfo=dt.UTC)

    msg = Message(
        medium=Medium.SMS_MESSAGE,
        sender_id=1,
        receiver_ids=[0],
        timestamp=base_ts,
        content="Please remember this important fact.",
        exchange_id=1,
    )
    await EVENT_BUS.publish(Event(type="Message", payload=msg))

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

    EVENT_BUS.join_published()

    # ---------------------------------------------------------------
    # 3.  Wait briefly for chunk processing
    # ---------------------------------------------------------------
    await asyncio.sleep(0.2)

    # ---------------------------------------------------------------
    # 4.  Assertions -- process_chunk was invoked once and its
    #     transcript blob contains the manager_method records.
    # ---------------------------------------------------------------
    blob = captured.get("transcript")
    assert blob is not None, "process_chunk should have been called once"
    assert (
        '"kind": "manager_method"' in blob
    ), "ManagerMethod events must be included in the transcript chunk"
