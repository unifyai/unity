from __future__ import annotations

import pytest

from unity.transcript_manager.transcript_manager import TranscriptManager
from tests.helpers import _handle_project, capture_events

# All tests in this file require EventBus publishing to verify event behavior
pytestmark = [pytest.mark.enable_eventbus, pytest.mark.llm_call]


# ─────────────────────────  ask() logging  ──────────────────────────


@pytest.mark.asyncio
@_handle_project
async def test_events_ask():
    tm = TranscriptManager()

    user_q = "📝 What did Alice say to Bob yesterday?"  # unique text

    async with capture_events("ManagerMethod") as events:
        handle = await tm.ask(user_q)
        await handle.result()

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
