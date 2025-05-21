import pytest
import asyncio
import datetime as dt

from unity.events.event_bus import EventBus, Event
from unity.communication.types.message import Message
from tests.helpers import _handle_project


@pytest.mark.asyncio
@_handle_project
async def test_prefill_from_upstream_on_new_instance():
    """
    After some events are published with one EventBus, a brand-new EventBus
    should hydrate those same events from Unify logs into its in-memory window.
    """
    window = 10
    bus1 = EventBus()
    bus1.set_default_window(window)

    base_ts = dt.datetime.now(dt.UTC)
    published: list[Event] = []

    # Publish five message events with ascending timestamps
    for i in range(5):
        evt = Event(
            type="message",
            timestamp=base_ts + dt.timedelta(seconds=i),
            payload=Message.model_construct(),
        )
        published.append(evt)
        await bus1.publish(evt)

    # Give the async logger a brief moment (usually unnecessary, but harmless)
    await asyncio.sleep(0.05)

    # Create a *new* EventBus that should preload from persisted logs
    bus2 = EventBus()
    bus2.set_window("messages", window)

    latest = await bus2.get_latest(types=["message"], limit=window)

    # Each originally-sent event (identified by its ts & payload) must be present
    for sent in published:
        assert any(
            rec.timestamp == sent.timestamp and rec.payload == sent.payload
            for rec in latest
        ), f"Event with ts {sent.timestamp.isoformat()} not found in prefilled window"
