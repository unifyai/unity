from __future__ import annotations

import pytest

from unity.task_scheduler.task_scheduler import TaskScheduler
from unity.events.event_bus import EVENT_BUS
from tests.helpers import _handle_project


async def _gather_events():
    ev = await EVENT_BUS.search(filter='type == "ManagerMethod"', limit=1000)
    return [e for e in ev if e.type == "ManagerMethod"]


@pytest.mark.unit
@pytest.mark.asyncio
@_handle_project
async def test_managermethod_events_for_ask():
    ts = TaskScheduler()
    q = "📋 List all tasks."
    h = await ts.ask(q)
    await h.result()
    EVENT_BUS.join_published()
    ev = await _gather_events()
    incoming = [
        e
        for e in ev
        if e.payload.get("manager") == "TaskScheduler"
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
async def test_managermethod_events_for_update():
    ts = TaskScheduler()
    cmd = "Create a task 'Submit report' for tomorrow."
    h = await ts.update(cmd)
    await h.result()
    EVENT_BUS.join_published()
    ev = await _gather_events()
    incoming = [
        e
        for e in ev
        if e.payload.get("manager") == "TaskScheduler"
        and e.payload.get("method") == "update"
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
async def test_managermethod_events_for_execute():
    ts = TaskScheduler()

    # create a simple task first
    outcome = ts._create_task(name="Demo", description="Run a demo task")
    task_id = outcome["details"]["task_id"]

    h = await ts.execute(text=str(task_id))
    await h.result()
    EVENT_BUS.join_published()
    ev = await _gather_events()
    incoming = [
        e
        for e in ev
        if e.payload.get("manager") == "TaskScheduler"
        and e.payload.get("method") == "execute"
        and e.payload.get("request") == str(task_id)
    ]
    assert incoming
    call_id = incoming[0].calling_id
    outgoing = [
        e
        for e in ev
        if e.calling_id == call_id and e.payload.get("phase") == "outgoing"
    ]
    assert outgoing
