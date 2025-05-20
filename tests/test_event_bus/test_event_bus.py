# tests/test_event_bus.py
import datetime as dt
import pytest

from unity.events.event_bus import EventBus, Event
from unity.events.types.message import Message


@pytest.mark.asyncio
async def test_publish():
    """Publishing a valid event should complete without exceptions
    and the event should be stored in the in-memory deque.
    """
    bus = EventBus()          # use defaults (50-event windows)

    # create a minimal Message payload; model_construct() skips field validation,
    # so it works even if Message has required fields we don’t care about here
    payload = Message.model_construct()

    event = Event(
        type="message",
        ts=dt.datetime.now(dt.UTC).isoformat(),
        payload=payload,
    )

    # This should run cleanly …
    await bus.publish(event)

    # … and the event should now be in the per-type deque
    assert event in bus._deques["message"]
