import pytest
import random
import asyncio
import datetime as dt

from unity.events.event_bus import EventBus, Event
from unity.transcript_manager.types.message import Message, Medium
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
            type="Message",
            timestamp=base_ts + dt.timedelta(seconds=i),
            payload=Message(
                medium=random.choice(list(Medium)),
                sender_id=random.randint(0, 10),
                receiver_ids=[random.randint(0, 10)],
                timestamp=dt.datetime.now(dt.UTC),
                content="hello",
                exchange_id=0,
            ),
        )
        published.append(evt)
        await bus1.publish(evt)

    # Give the async logger a brief moment (usually unnecessary, but harmless)
    await asyncio.sleep(0.05)

    # Create a *new* EventBus that should preload from persisted logs
    bus2 = EventBus()
    bus2.set_window("messages", window)

    latest = await bus2.search(filter="type == 'Message'", limit=window)

    # Each originally-sent event (identified by its ts & payload) must be present
    for sent in published:
        assert any(
            rec.timestamp == sent.timestamp for rec in latest
        ), f"Event with ts {sent.timestamp} not found in prefilled window"
