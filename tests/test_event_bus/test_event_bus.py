"""Comprehensive unit‑tests for the in‑process Event, EventBus and Subscription
implementation shown in the guide.

Run with:
    pytest -q

These tests exercise:
* dataclass immutability and automatic timestamping (``Event``)
* singleton behaviour, append‑only log, trim policy, and publish fan‑out (``EventBus``)
* filtering, replay, dynamic filter updates, and async iteration (``Subscription``)
"""

import asyncio
import dataclasses
import datetime as dt
from typing import List

import pytest

# Adjust this import to where you placed the implementation.
# For instance, if the classes live in ``event_system.py`` in the same folder
# the following works:
from events.event_bus import Event, EventBus  # noqa: E402

# ─────────────────────────── Fixtures & helpers ────────────────────────────────

@pytest.fixture()
def bus() -> EventBus:
    """Return a *fresh* EventBus instance for each test.

    The EventBus is a singleton, so we clear the cached instance first to avoid
    cross‑test interference.
    """
    EventBus._instance = None  # type: ignore[attr-defined]
    return EventBus(maxlen=5)  # small maxlen lets us test trimming fast


async def _publish_many(bus: EventBus, events: List[Event]):
    for ev in events:
        await bus.publish(ev)


# ────────────────────────────   Tests   ───────────────────────────────────────


def test_event_is_immutable():
    """Event should be *frozen* — attempting to mutate raises FrozenInstanceError."""
    ev = Event(type="foo")
    with pytest.raises(dataclasses.FrozenInstanceError):
        ev.type = "bar"  # type: ignore[misc]


@pytest.mark.asyncio
async def test_publish_and_subscribe_basic(bus: EventBus):
    """A subscriber with an *event_types* filter only receives matching events."""
    sub = bus.subscribe(event_types={"foo"})
    e1 = Event("foo", payload={"x": 1})
    e2 = Event("bar")

    await _publish_many(bus, [e1, e2])

    received = await asyncio.wait_for(sub._queue.get(), 0.1)  # type: ignore[attr-defined]
    assert received == e1
    assert sub._queue.empty()  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_replay_returns_filtered_history(bus: EventBus):
    """Subscription.replay() returns *past* events that match the filter in order."""
    e1, e2, e3 = Event("a"), Event("b"), Event("a")
    await _publish_many(bus, [e1, e2, e3])

    sub = bus.subscribe(event_types={"a"})
    assert sub.replay() == [e1, e3]


@pytest.mark.asyncio
async def test_set_types_updates_filter(bus: EventBus):
    """After calling set_types, only the *new* set is considered for future events."""
    sub = bus.subscribe(event_types={"x"})
    ex, ey, ez = Event("x"), Event("y"), Event("y")

    await _publish_many(bus, [ex, ey])
    # Only *x* should have been delivered so far
    assert await sub._queue.get() == ex  # type: ignore[attr-defined]
    assert sub._queue.empty()  # type: ignore[attr-defined]

    # Narrow to just "y" and publish another y event
    sub.set_types("y")
    await bus.publish(ez)
    assert await sub._queue.get() == ez  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_multiple_subscribers_receive_only_their_events(bus: EventBus):
    sub_a = bus.subscribe(event_types={"a"})
    sub_b = bus.subscribe(event_types={"b"})
    ea, eb = Event("a"), Event("b")

    await _publish_many(bus, [ea, eb])

    assert await sub_a._queue.get() == ea  # type: ignore[attr-defined]
    assert await sub_b._queue.get() == eb  # type: ignore[attr-defined]
    assert sub_a._queue.empty() and sub_b._queue.empty()  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_maxlen_trim(bus: EventBus):
    """EventBus deque should trim the oldest items past *maxlen*."""
    events = [Event(f"e{i}") for i in range(6)]  # maxlen is 5
    await _publish_many(bus, events)

    assert len(bus._log) == 5  # type: ignore[attr-defined]
    assert list(bus._log)[0] == events[1]  # oldest (e0) has been trimmed


@pytest.mark.asyncio
async def test_async_iteration(bus: EventBus):
    """__aiter__ should yield events in real time."""
    sub = bus.subscribe(event_types={"ping"})
    ping = Event("ping")

    async def producer():
        await asyncio.sleep(0)  # let iterator start
        await bus.publish(ping)

    asyncio.create_task(producer())

    async for ev in sub:  # will break after first ping
        assert ev == ping
        break


@pytest.mark.asyncio
async def test_predicate_subscription(bus: EventBus):
    """A custom predicate works the same as event_types filtering."""
    pred = lambda e: e.type.startswith("sys.")
    sub = bus.subscribe(match=pred)

    ok, skip = Event("sys.start"), Event("user.login")
    await _publish_many(bus, [ok, skip])

    assert await sub._queue.get() == ok  # type: ignore[attr-defined]
    assert sub._queue.empty()  # type: ignore[attr-defined]
