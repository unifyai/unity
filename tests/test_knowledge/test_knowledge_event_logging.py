from __future__ import annotations

import pytest

from unity.knowledge_manager.knowledge_manager import KnowledgeManager
from unity.events.event_bus import EVENT_BUS
from tests.helpers import _handle_project


async def _gather_managermethod_events():
    events = await EVENT_BUS.search(filter='type == "ManagerMethod"', limit=1000)
    return [e for e in events if e.type == "ManagerMethod"]


# ─────────────────────────  ask() logging  ──────────────────────────


@pytest.mark.unit
@pytest.mark.asyncio
@_handle_project
async def test_managermethod_events_for_ask():
    km = KnowledgeManager()

    user_q = "🔎 How many tables do I have?"
    handle = await km.ask(user_q)
    await handle.result()

    EVENT_BUS.join_published()

    events = await _gather_managermethod_events()

    incoming = [
        e
        for e in events
        if e.payload.get("manager") == "KnowledgeManager"
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
    )


# ─────────────────────────  update() logging  ───────────────────────


@pytest.mark.unit
@pytest.mark.asyncio
@_handle_project
async def test_managermethod_events_for_update():
    km = KnowledgeManager()

    nl_cmd = "Create a table Foo with a column bar:string."
    handle = await km.update(nl_cmd)
    await handle.result()

    EVENT_BUS.join_published()
    events = await _gather_managermethod_events()

    incoming = [
        e
        for e in events
        if e.payload.get("manager") == "KnowledgeManager"
        and e.payload.get("method") == "update"
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
    assert "answer" in outgoing[0].payload


# ─────────────────────────  refactor() logging  ─────────────────────


@pytest.mark.unit
@pytest.mark.asyncio
@_handle_project
async def test_managermethod_events_for_refactor():
    km = KnowledgeManager()

    cmd = "Refactor the Foo table to rename bar → baz."
    handle = await km.refactor(cmd)
    await handle.result()

    EVENT_BUS.join_published()
    events = await _gather_managermethod_events()

    incoming = [
        e
        for e in events
        if e.payload.get("manager") == "KnowledgeManager"
        and e.payload.get("method") == "refactor"
        and e.payload.get("phase") == "incoming"
        and e.payload.get("request") == cmd
    ]
    assert incoming, "No incoming ManagerMethod event for refactor()"
    call_id = incoming[0].calling_id

    outgoing = [
        e
        for e in events
        if e.calling_id == call_id and e.payload.get("phase") == "outgoing"
    ]
    assert outgoing, "No outgoing ManagerMethod event for refactor()"
    assert "answer" in outgoing[0].payload
