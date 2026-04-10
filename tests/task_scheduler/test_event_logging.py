from __future__ import annotations

import pytest

from unity.task_scheduler.task_scheduler import TaskScheduler
from tests.helpers import _handle_project, capture_events

# All tests in this file require EventBus publishing to verify event behavior
pytestmark = [pytest.mark.enable_eventbus, pytest.mark.llm_call]


@pytest.mark.asyncio
@_handle_project
async def test_managermethod_events_for_ask():
    ts = TaskScheduler()
    q = "📋 List all tasks."

    async with capture_events("ManagerMethod") as events:
        h = await ts.ask(q)
        await h.result()

    incoming = [
        e
        for e in events
        if e.payload.get("manager") == "TaskScheduler"
        and e.payload.get("method") == "ask"
        and e.payload.get("phase") == "incoming"
    ]
    assert incoming
    call_id = incoming[0].calling_id
    outgoing = [
        e
        for e in events
        if e.calling_id == call_id and e.payload.get("phase") == "outgoing"
    ]
    assert outgoing


@pytest.mark.asyncio
@_handle_project
async def test_managermethod_events_for_update():
    ts = TaskScheduler()
    cmd = "Create a task 'Submit report' for tomorrow."

    async with capture_events("ManagerMethod") as events:
        h = await ts.update(cmd)
        await h.result()

    incoming = [
        e
        for e in events
        if e.payload.get("manager") == "TaskScheduler"
        and e.payload.get("method") == "update"
        and e.payload.get("phase") == "incoming"
    ]
    assert incoming
    call_id = incoming[0].calling_id
    outgoing = [
        e
        for e in events
        if e.calling_id == call_id and e.payload.get("phase") == "outgoing"
    ]
    assert outgoing


@pytest.mark.asyncio
@_handle_project
async def test_managermethod_events_for_execute():
    ts = TaskScheduler()

    # create a simple task first
    outcome = ts._create_task(name="Demo", description="Run a demo task")
    task_id = outcome["details"]["task_id"]

    async with capture_events("ManagerMethod") as events:
        h = await ts.execute(task_id=task_id)
        await h.result()

    incoming = [
        e
        for e in events
        if e.payload.get("manager") == "TaskScheduler"
        and e.payload.get("method") == "execute"
        and e.payload.get("request") == task_id
    ]
    assert incoming
    call_id = incoming[0].calling_id
    outgoing = [
        e
        for e in events
        if e.calling_id == call_id and e.payload.get("phase") == "outgoing"
    ]
    assert outgoing
