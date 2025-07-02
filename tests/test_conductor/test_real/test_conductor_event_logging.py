from __future__ import annotations

import pytest

from unity.conductor.conductor import Conductor
from unity.events.event_bus import EVENT_BUS
from tests.helpers import _handle_project


async def _gather_events():
    e = await EVENT_BUS.search(filter='type == "ManagerMethod"', limit=1000)
    return [ev for ev in e if ev.type == "ManagerMethod"]


@pytest.mark.unit
@pytest.mark.asyncio
@_handle_project
async def test_managermethod_events_for_ask():
    c = Conductor()
    q = "You are part of a unit test. Do *not* call *any* tools, just reply 'hello'."
    h = await c.ask(q)
    await h.result()
    EVENT_BUS.join_published()
    ev = await _gather_events()
    incoming = [
        e
        for e in ev
        if e.payload.get("manager") == "Conductor"
        and e.payload.get("method") == "ask"
        and e.payload.get("phase") == "incoming"
    ]
    assert incoming
    call_id = incoming[0].calling_id
    outgoing = [
        e
        for e in ev
        if e.calling_id == call_id and e.payload.get("phase") == "outgoing"
    ]
    assert outgoing


@pytest.mark.unit
@pytest.mark.asyncio
@_handle_project
async def test_managermethod_events_for_request():
    c = Conductor()
    cmd = "You are part of a unit test. Do *not* call *any* tools, just reply 'hello'."
    h = await c.request(cmd)
    await h.result()
    EVENT_BUS.join_published()
    ev = await _gather_events()
    incoming = [
        e
        for e in ev
        if e.payload.get("manager") == "Conductor"
        and e.payload.get("method") == "request"
        and e.payload.get("phase") == "incoming"
    ]
    assert incoming
    call_id = incoming[0].calling_id
    outgoing = [
        e
        for e in ev
        if e.calling_id == call_id and e.payload.get("phase") == "outgoing"
    ]
    assert outgoing
