"""Unit test verifying MemoryManager automatic *chunk* processing (every N messages).

The production MemoryManager triggers its unified ``process_chunk`` method once
every ``_CHUNK_SIZE`` transcript messages.  To keep the test fast we
monkey-patch the instance attribute to **3** so that publishing three synthetic
``message`` events is sufficient to fire one chunk update.

We patch ``process_chunk`` with a lightweight stub that merely records how many
times it was invoked -- no LLM calls or DB writes.
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
from unity.conversation_manager.types import Medium

# ---------------------------------------------------------------------------
#                             TEST CASE                                      #
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@_handle_project
async def test_parse_transcript_for_updates(monkeypatch):
    """Publishing *three* messages should trigger exactly **one** call to
    ``process_chunk``.
    """

    # ------------------------------------------------------------------
    # 1.  Counter & lightweight stub replacing the heavy process_chunk
    # ------------------------------------------------------------------
    counts: Dict[str, int] = {"process_chunk": 0}

    async def _stub_process_chunk(self, *_, **__):  # noqa: D401
        counts["process_chunk"] += 1
        return "ok"

    monkeypatch.setattr(
        MemoryManager,
        "process_chunk",
        _stub_process_chunk,
        raising=True,
    )

    # ------------------------------------------------------------------
    # 2.  Instantiate MemoryManager wired to simulated sub-managers
    # ------------------------------------------------------------------
    mm = MemoryManager(
        contact_manager=SimulatedContactManager(
            description=(
                "TEST SCENARIO: Chunk processing. Simulated ContactManager used"
                " inside a real MemoryManager; behave deterministically and"
                " lightweight, avoiding any external I/O."
            ),
        ),
        transcript_manager=SimulatedTranscriptManager(
            description=(
                "TEST SCENARIO: Chunk processing. Simulated TranscriptManager"
                " should return straightforward results."
            ),
        ),
        knowledge_manager=SimulatedKnowledgeManager(
            description=(
                "TEST SCENARIO: Chunk processing. Simulated KnowledgeManager"
                " can assume an empty store and respond quickly."
            ),
        ),
        task_scheduler=SimulatedTaskScheduler(
            description=(
                "TEST SCENARIO: Chunk processing. Simulated TaskScheduler"
                " should accept simple updates without hesitation."
            ),
        ),
    )

    # Shrink the chunk size to just **3** messages so the test runs fast.
    mm._CHUNK_SIZE = 3  # type: ignore[attr-defined]

    # Allow asynchronous callback registration to complete.
    await asyncio.sleep(0.05)

    # ------------------------------------------------------------------
    # 3.  Publish three synthetic messages -- two distinct contacts (id 1 & 2)
    # ------------------------------------------------------------------
    base_ts = dt.datetime(2025, 1, 1, tzinfo=dt.UTC)

    msgs = [
        Message(  # contact 1 -> assistant (id 0)
            medium=Medium.SMS_MESSAGE,
            sender_id=1,
            receiver_ids=[0],
            timestamp=base_ts,
            content="Hi!",
            exchange_id=1,
        ),
        Message(  # assistant -> contact 1
            medium=Medium.SMS_MESSAGE,
            sender_id=0,
            receiver_ids=[1],
            timestamp=base_ts + dt.timedelta(seconds=1),
            content="Hello!",
            exchange_id=1,
        ),
        Message(  # contact 2 -> assistant (id 0)
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
    # 5.  Assertion -- exactly one call to process_chunk
    # ------------------------------------------------------------------
    assert (
        counts["process_chunk"] == 1
    ), "process_chunk should fire exactly once per chunk"
