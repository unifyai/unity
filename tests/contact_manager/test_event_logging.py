from __future__ import annotations

import pytest

from unity.contact_manager.contact_manager import ContactManager
from tests.helpers import _handle_project, capture_events

# All tests in this file require EventBus publishing to verify event behavior
pytestmark = pytest.mark.enable_eventbus


@pytest.mark.asyncio
@_handle_project
async def test_ask_events():
    cm = ContactManager()

    user_q = "📅 Echo back today's date, please."  # unique text → easy filtering

    async with capture_events("ManagerMethod") as events:
        handle = await cm.ask(user_q)
        await handle.result()

    incoming = [
        e
        for e in events
        if e.payload.get("method") == "ask"
        and e.payload.get("phase") == "incoming"
        and e.payload.get("question") == user_q
    ]
    print([e.payload for e in events if e.payload.get("method") == "ask"])
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
    ), "Outgoing event should carry the assistant answer"


@pytest.mark.slow
@pytest.mark.asyncio
@_handle_project
async def test_update_events():
    cm = ContactManager()

    nl_cmd = "Create a contact named Logan Paul, email logan@example.com."

    async with capture_events("ManagerMethod") as events:
        handle = await cm.update(nl_cmd)
        await handle.result()

    incoming = [
        e
        for e in events
        if e.payload.get("method") == "update"
        and e.payload.get("phase") == "incoming"
        and e.payload.get("request") == nl_cmd
    ]
    assert incoming, "No incoming ManagerMethod event recorded for update()"
    call_id = incoming[0].calling_id

    outgoing = [
        e
        for e in events
        if e.calling_id == call_id and e.payload.get("phase") == "outgoing"
    ]
    assert outgoing, "No outgoing ManagerMethod event recorded for update()"
    # for updates we don't mandate any specific answer text, just presence
    assert "answer" in outgoing[0].payload, "Outgoing update event missing 'answer'"
