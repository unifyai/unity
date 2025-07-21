import datetime as dt
import pytest

from unity.events.event_bus import EventBus, Event
from tests.helpers import _handle_project


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _mk_evt(call_id: str, seq: int, etype: str = "Msg") -> Event:
    return Event(
        type=etype,
        calling_id=call_id,
        timestamp=dt.datetime.now(dt.UTC),
        payload={"seq": seq},
    )


# ---------------------------------------------------------------------------
# 1. explicit pin / unpin API
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@_handle_project
async def test_pin_preserves_events() -> None:
    bus = EventBus()
    bus.set_window("Msg", 2)  # keep only 2 unpinned events

    # publish two events (seq 0,1) with distinct call-ids
    e0 = _mk_evt("c1", 0)
    e1 = _mk_evt("c2", 1)
    await bus.publish(e0)
    await bus.publish(e1)

    # Pin the first call-id before overflowing the window
    bus.pin_call_id("c1")

    # Publish a third event – would normally evict e0, but it's pinned
    e2 = _mk_evt("c3", 2)
    await bus.publish(e2)

    res = await bus.search(filter='type == "Msg"', limit=10)
    seqs = [ev.payload["seq"] for ev in res]
    # Expect the pinned e0 (seq 0) + the two newest (1,2)
    assert set(seqs) == {0, 1, 2} and len(seqs) == 3

    # Now unpin and publish a 4th event → oldest (seq 0) should fall off
    bus.unpin_call_id("c1")
    e3 = _mk_evt("c4", 3)
    await bus.publish(e3)

    # Deque should now contain only the two newest unpinned events (seq 3 & 2)
    dq_seqs = [ev.payload["seq"] for ev in bus._deques["Msg"]]
    assert dq_seqs == [2, 3] or dq_seqs == [3, 2]


# ---------------------------------------------------------------------------
# 2. auto-pin / auto-unpin rules
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@_handle_project
async def test_auto_pin_rule() -> None:
    bus = EventBus()
    bus.set_window("Task", 1)  # tiny window so eviction is obvious

    # auto-pin on "start", unpin on "end", keyed by task_id in payload
    bus.register_auto_pin(
        event_type="Task",
        open_predicate=lambda e: e.payload.get("stage") == "start",
        close_predicate=lambda e: e.payload.get("stage") == "end",
        key_fn=lambda e: e.payload["task_id"],
    )

    start_evt = Event(
        type="Task",
        payload={"task_id": "T1", "stage": "start"},
        calling_id="T1",
        timestamp=dt.datetime.now(dt.UTC),
    )
    await bus.publish(start_evt)

    # Publish another event that would overflow the window but uses same task_id
    mid_evt = Event(
        type="Task",
        payload={"task_id": "T1", "info": "progress"},
        calling_id="T1",
        timestamp=dt.datetime.now(dt.UTC),
    )
    await bus.publish(mid_evt)

    res = await bus.search(filter='type == "Task"', limit=10)
    # Both events must be present despite window=1
    assert len(res) == 2

    # Send end event – should unpin and allow eviction
    end_evt = Event(
        type="Task",
        payload={"task_id": "T1", "stage": "end"},
        calling_id="T1",
        timestamp=dt.datetime.now(dt.UTC),
    )
    await bus.publish(end_evt)

    # Another event to force trimming
    overflow_evt = Event(
        type="Task",
        payload={"task_id": "T2", "info": "other"},
        calling_id="T2",
        timestamp=dt.datetime.now(dt.UTC),
    )
    await bus.publish(overflow_evt)

    # The in-memory deque for the event-type should now contain only one
    # unpinned event (window size =1) and it must belong to the new task.
    assert len(bus._deques["Task"]) == 1
    newest = bus._deques["Task"][0]
    assert newest.payload.get("task_id") == "T2"
