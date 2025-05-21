import pytest
import asyncio
import datetime as dt
from collections import deque

from unity.events.event_bus import EventBus, Event
from unity.communication.types.message import Message
from unity.communication.types.message_exchange_summary import MessageExchangeSummary
from tests.helpers import _handle_project


@pytest.mark.asyncio
@_handle_project
async def test_get_latest():
    """A single publish should be retrievable via get_latest()."""
    bus = EventBus()

    payload = Message.model_construct()
    event = Event(type="Messages", timestamp=dt.datetime.now(dt.UTC), payload=payload)

    await bus.publish(event)

    # Read back through the public API
    latest = await bus.get_latest(types=["Messages"], limit=1)

    # There should be at least one event, and it should be the one we just published
    assert latest and latest[0] == event


@pytest.mark.asyncio
@_handle_project
async def test_get_latest_mixed_types_ordering():
    """Interwoven publishing of two event types should come back newest-first."""
    window_sizes = {"Messages": 10, "MessageExchangeSummary": 10}
    bus = EventBus()
    [bus.set_window(k, v) for k, v in window_sizes.items()]

    # Start from a clean slate for deterministic assertions
    for t in ("Messages", "MessageExchangeSummary"):
        bus._deques.setdefault(t, deque(maxlen=window_sizes[t])).clear()

    base_ts = dt.datetime.now(dt.UTC)
    events = []

    # Publish 6 events: message, summary, message, …
    for idx in range(6):
        etype, payload_cls = (
            ("Messages", Message)
            if idx % 2 == 0
            else ("MessageExchangeSummary", MessageExchangeSummary)
        )

        evt = Event(
            type=etype,
            timestamp=base_ts
            + dt.timedelta(seconds=idx),  # strictly ascending timestamps
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


@pytest.mark.asyncio
@_handle_project
async def test_concurrent_get_latest_lock_integrity():
    """
    Fire several concurrent `get_latest` requests with different `types` filters
    and limits.  All should complete without dead-locking and return the
    correct (newest-first) slices, proving the read-side lock holds up.
    """
    windows = {"Messages": 100, "MessageExchangeSummary": 100}
    bus = EventBus()
    [bus.set_window(k, v) for k, v in windows.items()]

    # Ensure a clean slate so results are deterministic
    for t, w in windows.items():
        bus._deques.setdefault(t, deque(maxlen=w)).clear()

    # ── Publish 40 interleaved events ───────────────────────────────
    base_ts = dt.datetime.now(dt.UTC)
    events = []
    for i in range(40):
        etype, payload_cls = (
            ("Messages", Message)
            if i % 2 == 0
            else ("MessageExchangeSummary", MessageExchangeSummary)
        )
        evt = Event(
            type=etype,
            timestamp=base_ts + dt.timedelta(microseconds=i),
            payload=payload_cls.model_construct(),
        )
        events.append(evt)
        await bus.publish(evt)

    # Pre-compute expected slices (newest-first order)
    all_newest = list(reversed(events))
    messages_newest = [e for e in all_newest if e.type == "Messages"]
    summaries_newest = [e for e in all_newest if e.type == "MessageExchangeSummary"]

    expected_r1 = messages_newest[:5]
    expected_r2 = summaries_newest[:7]
    expected_r3 = []  # empty filter → empty result
    expected_r4 = all_newest[:15]

    # ── Concurrent read tasks ──────────────────────────────────────
    tasks = [
        asyncio.create_task(bus.get_latest(types=["Messages"], limit=5)),  # r1
        asyncio.create_task(
            bus.get_latest(types=["MessageExchangeSummary"], limit=7),
        ),  # r2
        asyncio.create_task(bus.get_latest(types=[], limit=10)),  # r3 (no types)
        asyncio.create_task(bus.get_latest(types=None, limit=15)),  # r4 (both types)
    ]

    r1, r2, r3, r4 = await asyncio.gather(*tasks)

    # ── Assertions ─────────────────────────────────────────────────
    assert r1 == expected_r1
    assert r2 == expected_r2
    assert r3 == expected_r3
    assert r4 == expected_r4
