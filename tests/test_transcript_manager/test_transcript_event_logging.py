from __future__ import annotations

import pytest

from unity.transcript_manager.transcript_manager import TranscriptManager
from unity.events.event_bus import EVENT_BUS
from tests.helpers import _handle_project


async def _gather_managermethod_events():
    """
    Convenience helper: fetch *all* ManagerMethod events currently in memory.
    """
    events = await EVENT_BUS.search(filter='type == "ManagerMethod"', limit=1000)
    return [e for e in events if e.type == "ManagerMethod"]


# ─────────────────────────  ask() logging  ──────────────────────────


@pytest.mark.unit
@pytest.mark.asyncio
@_handle_project
async def test_managermethod_events_for_ask():
    tm = TranscriptManager()

    user_q = "📝 What did Alice say to Bob yesterday?"  # unique text
    handle = await tm.ask(user_q)
    await handle.result()

    # ensure async logger has flushed
    EVENT_BUS.join_published()

    events = await _gather_managermethod_events()

    incoming = [
        e
        for e in events
        if e.payload.get("manager") == "TranscriptManager"
        and e.payload.get("method") == "ask"
        and e.payload.get("phase") == "incoming"
        and e.payload.get("question") == user_q
    ]
    assert incoming, "No incoming ManagerMethod event recorded for ask()"
    call_id = incoming[0].calling_id

    outgoing = [
        e
        for e in events
        if e.calling_id == call_id and e.payload.get("phase") == "outgoing"
    ]
    assert outgoing, "No outgoing ManagerMethod event recorded for ask()"
    assert (
        isinstance(outgoing[0].payload.get("answer"), str)
        and outgoing[0].payload["answer"].strip()
    ), "Outgoing ask event should carry the assistant answer"


# ───────────────────────  summarize() logging  ──────────────────────


@pytest.mark.unit
@pytest.mark.asyncio
@_handle_project
async def test_managermethod_events_for_summarize():
    tm = TranscriptManager()

    # Seed a minimal transcript: single exchange containing two messages.
    ex_id = 999
    tm.log_messages(
        {
            "medium": "email",
            "sender_id": 1,
            "receiver_id": 2,
            "timestamp": "2025-01-01T12:00:00Z",
            "content": "Hi Bob, can we meet tomorrow?",
            "exchange_id": ex_id,
            "message_id": 1,
        },
    )
    tm.log_messages(
        {
            "medium": "email",
            "sender_id": 2,
            "receiver_id": 1,
            "timestamp": "2025-01-01T12:05:00Z",
            "content": "Sure Alice, 10 am works.",
            "exchange_id": ex_id,
            "message_id": 2,
        },
    )
    # Make sure the messages hit the backend before we summarise.
    tm.join_published()

    handle = await tm.summarize(from_exchanges=ex_id)
    await handle.result()

    EVENT_BUS.join_published()

    events = await _gather_managermethod_events()

    incoming = [
        e
        for e in events
        if e.payload.get("manager") == "TranscriptManager"
        and e.payload.get("method") == "summarize"
        and e.payload.get("phase") == "incoming"
        and ex_id in (e.payload.get("from_exchanges") or [])
    ]
    assert incoming, "No incoming ManagerMethod event recorded for summarize()"
    call_id = incoming[0].calling_id

    outgoing = [
        e
        for e in events
        if e.calling_id == call_id and e.payload.get("phase") == "outgoing"
    ]
    assert outgoing, "No outgoing ManagerMethod event recorded for summarize()"
    assert "answer" in outgoing[0].payload, "Outgoing summarize event missing 'answer'"
