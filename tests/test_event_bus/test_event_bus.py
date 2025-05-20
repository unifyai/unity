"""Unit tests for the *typed* EventBus that only allows models in *events/types/*.

Run with::

    pytest -q

This suite verifies:
* Validation – unknown types or wrong payload classes raise.
* Normal publish/subscribe cycle for the **Message** and **MessageExchangeSummary**
  event types.
* Per‑type sliding‑window trimming.

NOTE: Tests monkey‑patch *unify* so no real logging back‑end is required.
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any

import pytest


# ─── System under test ──────────────────────────────────────────────────────
from unity.events.event_bus import Event, EventBus  # noqa: E402
from unity.events.types.message import Message, Medium  # noqa: E402
from unity.events.types.message_exchange_summary import (  # noqa: E402
    MessageExchangeSummary,
)

# ─── Fixtures ───────────────────────────────────────────────────────────────


@pytest.fixture()
def bus() -> EventBus:
    EventBus._instance = None  # type: ignore[attr-defined]
    return EventBus()


async def _publish(bus: EventBus, evt: Event):
    await bus.publish(evt)


# ─── Tests ─────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_publish_message(bus: EventBus):
    sub = bus.subscribe(event_types={"message"})
    msg = Message(
        medium=Medium.EMAIL,
        sender_id=1,
        receiver_id=2,
        timestamp="2025-05-20T12:00:00Z",
        content="hi",
        exchange_id=42,
    )
    evt = Event(type="message", payload=msg)
    await _publish(bus, evt)

    got = await asyncio.wait_for(sub._queue.get(), 0.1)  # type: ignore[attr-defined]
    assert got.payload == msg


@pytest.mark.asyncio
async def test_publish_summary(bus: EventBus):
    sub = bus.subscribe(event_types={"message_exchange_summary"})
    summary = MessageExchangeSummary(exchange_ids=[1, 2], summary="All good")
    evt = Event(type="message_exchange_summary", payload=summary)
    await _publish(bus, evt)
    got = await sub._queue.get()  # type: ignore[attr-defined]
    assert got.payload == summary


@pytest.mark.asyncio
async def test_window_trim(bus: EventBus):
    bus.set_window("message", 3)
    for i in range(5):
        msg = Message(
            medium=Medium.SMS_MESSAGE,
            sender_id=i,
            receiver_id=i + 1,
            timestamp=datetime.utcnow().isoformat(),
            content=str(i),
            exchange_id=i,
        )
        await _publish(bus, Event(type="message", payload=msg))

    # Only the last 3 should remain
    history = bus.get_history(lambda e: e.type == "message")
    assert len(history) == 3
    assert history[0].payload.content == "2"


def test_unknown_type_validation(bus: EventBus):
    evt = Event(type="does_not_exist", payload=None)
    with pytest.raises(ValueError):
        asyncio.run(bus.publish(evt))


def test_wrong_payload_class(bus: EventBus):
    wrong = MessageExchangeSummary(exchange_ids=[1], summary="x")
    evt = Event(type="message", payload=wrong)  # mismatch on purpose
    with pytest.raises(TypeError):
        asyncio.run(bus.publish(evt))
