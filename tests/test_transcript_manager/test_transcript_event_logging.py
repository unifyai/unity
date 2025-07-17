from __future__ import annotations

import asyncio
from datetime import datetime

import pytest

from unity.events.event_bus import EVENT_BUS
from unity.transcript_manager.transcript_manager import TranscriptManager
from unity.transcript_manager.types.message import Message
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
    tm.log_message(
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
    tm.log_message(
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


# ───────────────────────  log_message() with audio logging  ──────────────────────


@pytest.mark.unit
@pytest.mark.asyncio
@_handle_project
async def test_log_message_with_audio():
    """
    Verify that `log_message` correctly persists the `audio` field with a GCS URI.
    """
    tm = TranscriptManager()

    test_audio_gcs_uri = "gs://my-test-bucket/audio/test.wav"
    test_content = f"Test message with audio @ {datetime.now()}"

    test_message = Message(
        medium="phone_call",
        sender_id=123,
        receiver_id=456,
        timestamp=datetime.now(),
        content=test_content,
        audio=test_audio_gcs_uri,
    )

    # Log the message and wait for it to be persisted
    tm.log_message(test_message)
    tm.join_published()
    await asyncio.sleep(2)  # Allow time for backend to index

    # Retrieve the message using the unique content as a filter
    results = tm._search_messages(filter=f"content == '{test_content}'", limit=1)

    # Assertions
    assert len(results) == 1, "Message was not found after logging"
    retrieved_message = results[0]
    assert retrieved_message.audio == test_audio_gcs_uri, "Audio GCS URI was not persisted correctly"