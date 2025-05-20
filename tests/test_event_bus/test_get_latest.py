import pytest
import datetime as dt
from collections import deque

from unity.events.event_bus import EventBus, Event
from unity.events.types.message import Message
from unity.events.types.message_exchange_summary import MessageExchangeSummary
from tests.helpers import _handle_project


@pytest.mark.asyncio
@_handle_project
async def test_get_latest():
    """A single publish should be retrievable via get_latest()."""
    bus = EventBus()

    payload = Message.model_construct()
    event = Event(type="message", ts=dt.datetime.now(dt.UTC), payload=payload)

    await bus.publish(event)

    # Read back through the public API
    latest = await bus.get_latest(types=["message"], limit=1)

    # There should be at least one event, and it should be the one we just published
    assert latest and latest[0] == event


@pytest.mark.asyncio
@_handle_project
async def test_get_latest_mixed_types_ordering():
    """Interwoven publishing of two event types should come back newest-first."""
    window_sizes = {"message": 10, "message_exchange_summary": 10}
    bus = EventBus(windows_sizes=window_sizes)

    # Start from a clean slate for deterministic assertions
    for t in ("message", "message_exchange_summary"):
        bus._deques.setdefault(t, deque(maxlen=window_sizes[t])).clear()

    base_ts = dt.datetime.now(dt.UTC)
    events = []

    # Publish 6 events: message, summary, message, …
    for idx in range(6):
        etype, payload_cls = (
            ("message", Message) if idx % 2 == 0 else ("message_exchange_summary", MessageExchangeSummary)
        )

        evt = Event(
            type=etype,
            ts=base_ts + dt.timedelta(seconds=idx),   # strictly ascending timestamps
            payload=payload_cls.model_construct(),
        )
        events.append(evt)
        await bus.publish(evt)

    # Retrieve the newest 10 events (more than we published)
    latest = await bus.get_latest(limit=10)

    # Filter the list to only the events we just wrote
    latest_ours = [e for e in latest if e in events]

    # They should be returned newest-first, i.e. exactly reverse of the order written
    assert latest_ours == list(reversed(events))
