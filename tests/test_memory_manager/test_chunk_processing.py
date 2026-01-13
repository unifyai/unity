"""Unit test verifying MemoryManager automatic *chunk* processing (every N messages).

The production MemoryManager triggers its maintenance helpers once every
``_CHUNK_SIZE`` transcript messages.  To keep the test fast we monkey-patch the
instance attribute to **3** so that publishing three synthetic `message` events
is sufficient to fire one *chunk* update.

We patch the heavy `update_*` methods with lightweight stubs that merely record
how many times they were invoked – no LLM calls or DB writes.  Using the
*simulated* manager classes further reduces dependencies.
"""

from __future__ import annotations

import asyncio
import datetime as dt
from typing import Dict

import pytest

from tests.helpers import _handle_project

# Core classes under test ----------------------------------------------------
from unity.memory_manager.memory_manager import MemoryManager
from unity.events.event_bus import EVENT_BUS, Event

# This test publishes events to verify MemoryManager chunk processing
pytestmark = pytest.mark.enable_eventbus

# Simulated manager test-doubles (no external I/O) ---------------------------
from unity.contact_manager.simulated import SimulatedContactManager
from unity.transcript_manager.simulated import SimulatedTranscriptManager
from unity.knowledge_manager.simulated import SimulatedKnowledgeManager
from unity.task_scheduler.simulated import SimulatedTaskScheduler

from unity.transcript_manager.types.message import Message
from unity.transcript_manager.types.medium import Medium


# ---------------------------------------------------------------------------
#                             TEST CASE                                      #
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@_handle_project
async def test_parse_transcript_for_updates(monkeypatch):
    """Publishing *three* messages should trigger exactly **one** maintenance
    cycle comprising:

    • update_contacts           – once
    • update_knowledge          – once
    • update_tasks              – once
    • update_contact_bio        – once per unique contact (here: 2)
    • update_contact_rolling_summary – once per unique contact (here: 2)
    """

    # ------------------------------------------------------------------
    # 1.  Counters & lightweight stubs replacing the heavy update helpers
    # ------------------------------------------------------------------
    counts: Dict[str, int] = {
        "contacts": 0,
        "knowledge": 0,
        "tasks": 0,
        "bio": 0,
        "rolling": 0,
        "policy": 0,
    }

    async def _stub_factory(key: str):  # noqa: D401 – imperative helper
        async def _stub(self, *_, **__):  # noqa: D401 – imperative helper
            counts[key] += 1
            return key  # light placeholder return value

        return _stub

    # Patch the *class* methods before instantiation so the new instance
    # immediately uses the stubs.
    monkeypatch.setattr(
        MemoryManager,
        "update_contacts",
        await _stub_factory("contacts"),
        raising=True,
    )
    monkeypatch.setattr(
        MemoryManager,
        "update_knowledge",
        await _stub_factory("knowledge"),
        raising=True,
    )
    monkeypatch.setattr(
        MemoryManager,
        "update_tasks",
        await _stub_factory("tasks"),
        raising=True,
    )
    monkeypatch.setattr(
        MemoryManager,
        "update_contact_bio",
        await _stub_factory("bio"),
        raising=True,
    )
    monkeypatch.setattr(
        MemoryManager,
        "update_contact_rolling_summary",
        await _stub_factory("rolling"),
        raising=True,
    )
    monkeypatch.setattr(
        MemoryManager,
        "update_contact_response_policy",
        await _stub_factory("policy"),
        raising=True,
    )

    # ------------------------------------------------------------------
    # 2.  Instantiate MemoryManager wired to simulated sub-managers
    # ------------------------------------------------------------------
    mm = MemoryManager(
        contact_manager=SimulatedContactManager(
            description=(
                "TEST SCENARIO: Chunk processing. You are a simulated ContactManager used"
                " inside a real MemoryManager; behave deterministically and lightweight,"
                " avoiding any external I/O. Accept updates when asked without requiring"
                " clarification, and keep outputs concise so the MemoryManager can run a"
                " single maintenance cycle."
            ),
        ),
        transcript_manager=SimulatedTranscriptManager(
            description=(
                "TEST SCENARIO: Chunk processing. Simulated TranscriptManager should return"
                " straightforward results for simple ask operations and avoid side effects."
            ),
        ),
        knowledge_manager=SimulatedKnowledgeManager(
            description=(
                "TEST SCENARIO: Chunk processing. Simulated KnowledgeManager can assume an"
                " empty store and respond quickly; no refusers or clarifications—just"
                " lightweight acknowledgements."
            ),
        ),
        task_scheduler=SimulatedTaskScheduler(
            description=(
                "TEST SCENARIO: Chunk processing. Simulated TaskScheduler should accept"
                " simple updates without hesitation and keep responses minimal."
            ),
        ),
    )

    # Shrink the chunk size to just **3** messages so the test runs fast.
    mm._CHUNK_SIZE = 3  # type: ignore[attr-defined]

    # Allow asynchronous callback registration to complete.
    await asyncio.sleep(0.05)

    # ------------------------------------------------------------------
    # 3.  Publish three synthetic messages – two distinct contacts (id 1 & 2)
    # ------------------------------------------------------------------
    base_ts = dt.datetime(2025, 1, 1, tzinfo=dt.UTC)

    msgs = [
        Message(  # contact 1 → assistant (id 0)
            medium=Medium.SMS_MESSAGE,
            sender_id=1,
            receiver_ids=[0],
            timestamp=base_ts,
            content="Hi!",
            exchange_id=1,
        ),
        Message(  # assistant → contact 1
            medium=Medium.SMS_MESSAGE,
            sender_id=0,
            receiver_ids=[1],
            timestamp=base_ts + dt.timedelta(seconds=1),
            content="Hello!",
            exchange_id=1,
        ),
        Message(  # contact 2 → assistant (id 0)
            medium=Medium.SMS_MESSAGE,
            sender_id=2,
            receiver_ids=[0],
            timestamp=base_ts + dt.timedelta(seconds=2),
            content="Hey there!",
            exchange_id=2,
        ),
    ]

    for m in msgs:
        await EVENT_BUS.publish(Event(type="Message", payload=m))

    EVENT_BUS.join_published()

    # ------------------------------------------------------------------
    # 4.  Wait for the MemoryManager to finish the background chunk task
    # ------------------------------------------------------------------
    await asyncio.sleep(0.2)  # small grace-period; task runs concurrently

    # ------------------------------------------------------------------
    # 5.  Assertions – exactly one global update + 2 per-contact updates
    # ------------------------------------------------------------------
    assert counts["contacts"] == 1, "update_contacts should fire once per chunk"
    assert counts["knowledge"] == 1, "update_knowledge should fire once per chunk"
    assert counts["tasks"] == 1, "update_tasks should fire once per chunk"
    assert counts["bio"] == 2, "update_contact_bio should fire once per unique contact"
    assert (
        counts["rolling"] == 2
    ), "update_contact_rolling_summary should fire once per unique contact"
    assert (
        counts["policy"] == 2
    ), "update_contact_response_policy should fire once per unique contact"
