import pytest
import asyncio
import datetime as dt
from collections import deque

from unity.events.event_bus import EventBus, Event
from unity.transcript_manager.types.message import Message
from tests.helpers import _handle_project


@pytest.mark.asyncio
@_handle_project
async def test_basic_publish():
    """Publishing a valid event should complete without exceptions
    and the event should be stored in the in-memory deque.
    """
    bus = EventBus()  # use defaults (50-event windows)

    # create a minimal Message payload; model_construct() skips field validation,
    # so it works even if Message has required fields we don’t care about here
    payload = Message.model_construct()

    event = Event(
        type="Message",
        timestamp=dt.datetime.now(dt.UTC).isoformat(),
        payload=payload,
    )

    # This should run cleanly …
    await bus.publish(event)

    # … and the event should now be in the per-type deque
    assert event in bus._deques["Message"]


@pytest.mark.asyncio
@_handle_project
async def test_concurrent_integrity():
    """
    Do a burst of concurrent publishes across two event types; all should succeed
    and be visible afterwards, demonstrating that the internal asyncio.Lock
    protects the critical section.
    """
    window = 200
    bus = EventBus()
    bus.set_default_window(200)

    # Clear any pre-existing state for determinism
    for typ in "Message":
        bus._deques.setdefault(typ, deque(maxlen=window)).clear()

    base_ts = dt.datetime.now(dt.UTC)
    n_events = 100
    events: list[Event] = []
    publish_tasks = []
    etype, payload_cls = "Message", Message

    for i in range(n_events):
        evt = Event(
            type=etype,
            timestamp=base_ts
            + dt.timedelta(microseconds=i),  # unique, strictly increasing
            payload=payload_cls.model_construct(),
        )
        events.append(evt)
        publish_tasks.append(asyncio.create_task(bus.publish(evt)))

    # Run all publishes concurrently; will raise if any individual publish fails
    await asyncio.gather(*publish_tasks)

    # Join published
    bus.join_published()

    # Fetch back everything; limit well above what we sent
    latest = await bus.search(limit=window, grouped_by_type=True)
    latest = latest["Message"]

    # Keep only the events we just published (ignore any older prefilled logs)
    our_ts = {e.timestamp for e in events}
    latest_ours = [e for e in latest if e.timestamp in our_ts]

    # Every event we published must be present
    assert len(latest_ours) == n_events
