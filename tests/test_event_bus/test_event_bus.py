"""Comprehensive unit-tests for the per‑type sliding‑window *EventBus*.

Run with:
    pytest -q

This suite covers:
* ``Event`` immutability and automatic timestamping
* Per‑type window behaviour (default 50, configurable at runtime)
* Publishing / subscribing, filtering, replay, dynamic filter updates
* Async iteration and coexistence of multiple subscribers

Implementation under test lives in ``event_system.py`` (same folder).
"""

from __future__ import annotations

import asyncio
import dataclasses
from typing import List

import pytest

# ─── Import system under test ────────────────────────────────────────────────
from events.event_bus import Event, EventBus  # noqa: E402

# ─── Fixtures & helpers ──────────────────────────────────────────────────────

@pytest.fixture()
def bus() -> EventBus:
    """Return a *fresh* EventBus instance for every test (reset singleton)."""
    EventBus._instance = None  # type: ignore[attr-defined]
    return EventBus()


async def _publish_many(bus: EventBus, events: List[Event]):
    for ev in events:
        await bus.publish(ev)


# ─── Tests ───────────────────────────────────────────────────────────────────


def test_event_is_immutable():
    ev = Event(type="foo")
    with pytest.raises(dataclasses.FrozenInstanceError):
        ev.type = "bar"  # type: ignore[misc]


@pytest.mark.asyncio
async def test_publish_and_subscribe_basic(bus: EventBus):
    sub = bus.subscribe(event_types={"foo"})
    e1 = Event("foo", payload={"x": 1})
    e2 = Event("bar")
    await _publish_many(bus, [e1, e2])

    received = await asyncio.wait_for(sub._queue.get(), 0.1)  # type: ignore[attr-defined]
    assert received == e1
    assert sub._queue.empty()  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_replay_returns_filtered_history(bus: EventBus):
    e1, e2, e3 = Event("a"), Event("b"), Event("a")
    await _publish_many(bus, [e1, e2, e3])

    sub = bus.subscribe(event_types={"a"})
    assert sub.replay() == [e1, e3]


@pytest.mark.asyncio
async def test_set_types_updates_filter(bus: EventBus):
    sub = bus.subscribe(event_types={"x"})
    ex, ey1, ey2 = Event("x"), Event("y"), Event("y")

    await _publish_many(bus, [ex, ey1])
    assert await sub._queue.get() == ex  # type: ignore[attr-defined]
    assert sub._queue.empty()  # type: ignore[attr-defined]

    sub.set_types("y")
    await bus.publish(ey2)
    assert await sub._queue.get() == ey2  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_multiple_subscribers_receive_only_their_events(bus: EventBus):
    sa = bus.subscribe(event_types={"a"})
    sb = bus.subscribe(event_types={"b"})
    ea, eb = Event("a"), Event("b")
    await _publish_many(bus, [ea, eb])

    assert await sa._queue.get() == ea  # type: ignore[attr-defined]
    assert await sb._queue.get() == eb  # type: ignore[attr-defined]
    assert sa._queue.empty() and sb._queue.empty()  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_window_trim_per_type(bus: EventBus):
    """Oldest events of a type are trimmed once its window is exceeded."""
    bus.set_window("login", 5)  # shrink for fast test
    events = [Event("login", payload={"n": i}) for i in range(6)]
    await _publish_many(bus, events)

    log = bus._log_by_type["login"]  # type: ignore[attr-defined]
    assert len(log) == 5
    # First (events[0]) was dropped, events[1] is now oldest
    assert log[0] == events[1]


@pytest.mark.asyncio
async def test_async_iteration(bus: EventBus):
    sub = bus.subscribe(event_types={"ping"})
    ping = Event("ping")

    async def producer():
        await asyncio.sleep(0)  # let iterator start
        await bus.publish(ping)

    asyncio.create_task(producer())

    async for ev in sub:
        assert ev == ping
        break


@pytest.mark.asyncio
async def test_predicate_subscription(bus: EventBus):
    pred = lambda e: e.type.startswith("sys.")
    sub = bus.subscribe(match=pred)

    ok, skip = Event("sys.start"), Event("user.login")
    await _publish_many(bus, [ok, skip])

    assert await sub._queue.get() == ok  # type: ignore[attr-defined]
    assert sub._queue.empty()  # type: ignore[attr-defined]
