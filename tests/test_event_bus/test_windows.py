import datetime as dt
import pytest
from collections import deque

from unity.events.event_bus import EventBus, Event
from unity.events.types.message import Message
from tests.helpers import _handle_project


@pytest.mark.asyncio
@_handle_project
async def test_window_eviction_at_limit():
    """When more than *window* events are published, the oldest should fall off."""
    window = 3
    bus = EventBus(windows_sizes={"message": window})

    # Start from a known clean state for this type (harmless use of a private attr)
    bus._deques.setdefault("message", bus._deques.get("message", deque(maxlen=window))).clear()

    # Publish window + 1 events with ascending timestamps
    events = []
    base_ts = dt.datetime.now(dt.UTC)
    for i in range(window + 1):
        evt = Event(
            type="message",
            ts=base_ts + dt.timedelta(seconds=i),
            payload=Message.model_construct(),
        )
        events.append(evt)
        await bus.publish(evt)

    # Fetch everything currently buffered for "message"
    latest = await bus.get_latest(types=["message"], limit=10)

    # Filter to the events we just published (there may be pre-existing logs)
    latest_ours = [e for e in latest if e in events]

    # We expect only *window* of our events (the newest three) to remain
    assert len(latest_ours) == window
    assert events[0] not in latest_ours            # the earliest one was evicted
    assert latest_ours[0] == events[-1]            # newest appears first (newest-first order)
