import datetime as dt
import pytest
import random
from collections import deque

from unity.events.event_bus import EventBus, Event
from unity.transcript_manager.types.message import Message, Medium
from tests.helpers import _handle_project


@pytest.mark.asyncio
@_handle_project
async def test_window_cache_is_faster():
    """When more than *window* events are published, the oldest should fall off."""
    window = 3
    bus = EventBus()
    bus.set_window("message", window)

    # Start from a known clean state for this type (harmless use of a private attr)
    bus._deques.setdefault(
        "message",
        bus._deques.get("message", deque(maxlen=window)),
    ).clear()

    # Publish window + 1 events with ascending timestamps
    event_ids = []
    base_ts = dt.datetime.now(dt.UTC)
    for i in range(window + 1):
        evt = Event(
            type="message",
            timestamp=base_ts + dt.timedelta(seconds=i),
            payload=Message(
                medium=random.choice(list(Medium)),
                sender_id=random.randint(0, 10),
                receiver_id=random.randint(0, 10),
                timestamp=dt.datetime.now(dt.UTC).isoformat(),
                content=f"{i}",
                exchange_id=0,
            ),
        )
        event_ids.append(evt.event_id)
        await bus.publish(evt, sync=True)

    # Time fetching just from cache vs having to hit backend
    await bus.search(filter="type == 'message'", limit=3)  # warm up
    t0 = dt.datetime.now(dt.UTC)
    await bus.search(filter="type == 'message'", limit=3)
    t1 = dt.datetime.now(dt.UTC)
    await bus.search(filter="type == 'message'", limit=4)
    t2 = dt.datetime.now(dt.UTC)

    cache_time = (t1 - t0).total_seconds()
    backend_time = (t2 - t1).total_seconds()
    assert (
        cache_time * 2 < backend_time
    ), f"Cache ({cache_time:.3f}s) should be faster than backend ({backend_time:.3f}s)"
