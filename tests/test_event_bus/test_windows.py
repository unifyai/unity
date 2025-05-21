import datetime as dt
import pytest
from collections import deque

from unity.events.event_bus import EventBus, Event
from unity.communication.types.message import Message
from unity.communication.types.message_exchange_summary import MessageExchangeSummary
from tests.helpers import _handle_project


@pytest.mark.asyncio
@_handle_project
async def test_window_eviction_at_limit():
    """When more than *window* events are published, the oldest should fall off."""
    window = 3
    bus = EventBus(windows_sizes={"message": window})

    # Start from a known clean state for this type (harmless use of a private attr)
    bus._deques.setdefault(
        "message",
        bus._deques.get("message", deque(maxlen=window)),
    ).clear()

    # Publish window + 1 events with ascending timestamps
    events = []
    base_ts = dt.datetime.now(dt.UTC)
    for i in range(window + 1):
        evt = Event(
            type="message",
            timestamp=base_ts + dt.timedelta(seconds=i),
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
    assert events[0] not in latest_ours  # the earliest one was evicted
    assert latest_ours[0] == events[-1]  # newest appears first (newest-first order)


@pytest.mark.asyncio
@_handle_project
async def test_window_eviction_mixed_sizes_and_ordering():
    """
    Verify that *different* per-type window sizes are respected simultaneously
    and that `get_latest()` returns the surviving events newest-first.
    """
    # Different windows: 2 for Message, 3 for MessageExchangeSummary
    windows = {"message": 2, "message_exchange_summary": 3}
    bus = EventBus(windows_sizes=windows)

    # Clear any prefilled data so we know exactly what's in memory
    for t, w in windows.items():
        bus._deques.setdefault(t, deque(maxlen=w)).clear()

    base_ts = dt.datetime.now(dt.UTC)

    # Publish seven events with strictly ascending timestamps
    # pattern: M, S, M, S, M, S, S
    publish_plan = [
        ("message", Message),
        ("message_exchange_summary", MessageExchangeSummary),
        ("message", Message),
        ("message_exchange_summary", MessageExchangeSummary),
        ("message", Message),
        ("message_exchange_summary", MessageExchangeSummary),
        ("message_exchange_summary", MessageExchangeSummary),
    ]
    events = []
    for idx, (etype, payload_cls) in enumerate(publish_plan):
        evt = Event(
            type=etype,
            timestamp=base_ts + dt.timedelta(seconds=idx),
            payload=payload_cls.model_construct(),
        )
        events.append(evt)
        await bus.publish(evt)

    # Retrieve more than enough to capture everything that survived
    latest = await bus.get_latest(limit=10)

    # Keep only the events we just published (ignore any earlier logs)
    ours = [e for e in latest if e in events]

    # Expected survivors by window:
    expected_messages = events[2:5:2]  # idx 2 and 4 → 2 newest messages
    expected_summaries = events[3:]  # idx 3,5,6 → 3 newest summaries
    expected_survivors = list(reversed(expected_summaries + expected_messages))
    # reversed() because get_latest() returns newest-first

    # 1️⃣ Correct counts per type
    assert sum(e.type == "message" for e in ours) == windows["message"]
    assert (
        sum(e.type == "message_exchange_summary" for e in ours)
        == windows["message_exchange_summary"]
    )

    # 2️⃣ Overall ordering newest-first
    assert ours == expected_survivors, "Events not in expected newest-first order"
