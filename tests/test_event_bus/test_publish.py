import pytest
import asyncio
import datetime as dt
from collections import deque

from unity.events.event_bus import EventBus, Event
from unity.events.types.message import Message
from unity.events.types.message_exchange_summary import MessageExchangeSummary
from tests.helpers import _handle_project


@pytest.mark.asyncio
@_handle_project
async def test_publish():
    """Publishing a valid event should complete without exceptions
    and the event should be stored in the in-memory deque.
    """
    bus = EventBus()          # use defaults (50-event windows)

    # create a minimal Message payload; model_construct() skips field validation,
    # so it works even if Message has required fields we don’t care about here
    payload = Message.model_construct()

    event = Event(
        context="message",
        timestamp=dt.datetime.now(dt.UTC).isoformat(),
        payload=payload,
    )

    # This should run cleanly …
    await bus.publish(event)

    # … and the event should now be in the per-type deque
    assert event in bus._deques["message"]


@pytest.mark.asyncio
@_handle_project
async def test_concurrent_publishes_lock_integrity():
    """
    Do a burst of concurrent publishes across two event types; all should succeed
    and be visible afterwards, demonstrating that the internal asyncio.Lock
    protects the critical section.
    """
    window = 200
    bus = EventBus(
        windows_sizes={"message": window, "message_exchange_summary": window}
    )

    # Clear any pre-existing state for determinism
    for typ in ("message", "message_exchange_summary"):
        bus._deques.setdefault(typ, deque(maxlen=window)).clear()

    base_ts = dt.datetime.now(dt.UTC)
    n_events = 100
    events: list[Event] = []
    publish_tasks = []

    for i in range(n_events):
        etype, payload_cls = (
            ("message", Message) if i % 2 == 0 else ("message_exchange_summary", MessageExchangeSummary)
        )
        evt = Event(
            context=etype,
            timestamp=base_ts + dt.timedelta(microseconds=i),   # unique, strictly increasing
            payload=payload_cls.model_construct(),
        )
        events.append(evt)
        publish_tasks.append(asyncio.create_task(bus.publish(evt)))

    # Run all publishes concurrently; will raise if any individual publish fails
    await asyncio.gather(*publish_tasks)

    # Fetch back everything; limit well above what we sent
    latest = await bus.get_latest(limit=window)

    # Keep only the events we just published (ignore any older prefilled logs)
    our_ts = {e.timestamp for e in events}
    latest_ours = [e for e in latest if e.timestamp in our_ts]

    # Every event we published must be present
    assert len(latest_ours) == n_events
