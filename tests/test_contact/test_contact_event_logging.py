from __future__ import annotations

import pytest

from unity.contact_manager.contact_manager import ContactManager
from unity.events.event_bus import EVENT_BUS
from tests.helpers import _handle_project


async def _gather_managermethod_events():
    """
    Convenience helper: fetch *all* ManagerMethod events currently in memory.
    """
    events = await EVENT_BUS.search(filter='type == "ManagerMethod"', limit=500)
    return [e for e in events if e.type == "ManagerMethod"]


@pytest.mark.unit
@pytest.mark.asyncio
@_handle_project
async def test_managermethod_events_for_ask():
    cm = ContactManager()

    user_q = "📅 Echo back today's date, please."  # unique text → easy filtering
    handle = await cm.ask(user_q)
    await handle.result()

    # ensure async logger has flushed
    EVENT_BUS.join_published()

    events = await _gather_managermethod_events()

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
@pytest.mark.unit
@pytest.mark.asyncio
@_handle_project
async def test_managermethod_events_for_update():
    cm = ContactManager()

    nl_cmd = "Create a contact named Logan Paul, email logan@example.com."
    handle = await cm.update(nl_cmd)
    await handle.result()

    EVENT_BUS.join_published()

    events = await _gather_managermethod_events()

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
