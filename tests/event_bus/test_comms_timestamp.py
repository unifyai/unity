"""Tests for Comms event timestamp handling.

Verifies that:
1. CommsPayload.timestamp is typed as datetime (not str)
2. The full roundtrip (Event -> BusEvent -> Event) preserves timestamp correctly
3. Publishing Comms events with timestamps works without type mismatch errors
"""

import pytest
import datetime as dt
from typing import get_type_hints, get_origin, get_args, Union

from tests.helpers import _handle_project
from unity.events.event_bus import EventBus, Event as BusEvent
from unity.events.types.comms import CommsPayload
from unity.conversation_manager.events import (
    SMSReceived,
    EmailReceived,
    UnifyMessageReceived,
    Event as CommsEvent,
)


# -------------------------------------------------------------------
#  CommsPayload timestamp type tests
# -------------------------------------------------------------------


def test_timestamp_field_is_datetime_type():
    """CommsPayload.timestamp should be Optional[datetime], not str."""
    hints = get_type_hints(CommsPayload)
    timestamp_type = hints.get("timestamp")

    # Should be Optional[datetime] which is Union[datetime, None]
    assert get_origin(timestamp_type) is Union
    args = get_args(timestamp_type)
    assert dt.datetime in args
    assert type(None) in args


def test_comms_payload_accepts_datetime():
    """CommsPayload should accept datetime objects for timestamp."""
    now = dt.datetime.now(dt.UTC)
    payload = CommsPayload(timestamp=now, content="test")
    assert payload.timestamp == now


def test_comms_payload_coerces_string_to_datetime():
    """CommsPayload should coerce ISO string to datetime."""
    iso_str = "2025-01-15T10:30:00+00:00"
    payload = CommsPayload(timestamp=iso_str, content="test")
    assert isinstance(payload.timestamp, dt.datetime)
    assert payload.timestamp.isoformat() == iso_str


# -------------------------------------------------------------------
#  Event roundtrip tests (Event -> BusEvent -> Event)
# -------------------------------------------------------------------


def test_sms_received_roundtrip():
    """SMSReceived event should preserve timestamp through bus conversion."""
    original_ts = dt.datetime(2025, 6, 15, 14, 30, 0, tzinfo=dt.UTC)
    original = SMSReceived(
        timestamp=original_ts,
        contact={"contact_id": 1, "name": "Test User"},
        content="Hello world",
    )

    # Convert to bus event
    bus_event = original.to_bus_event()

    # Verify bus event has correct type
    assert bus_event.type == "Comms"
    assert bus_event.payload_cls == "SMSReceived"

    # Convert back to comms event
    restored = CommsEvent.from_bus_event(bus_event)

    # Verify timestamp is preserved
    assert restored.timestamp == original_ts
    assert isinstance(restored.timestamp, dt.datetime)


def test_email_received_roundtrip():
    """EmailReceived event should preserve timestamp through bus conversion."""
    original_ts = dt.datetime(2025, 6, 15, 14, 30, 0, tzinfo=dt.UTC)
    original = EmailReceived(
        timestamp=original_ts,
        contact={"contact_id": 2, "email": "test@example.com"},
        subject="Test Subject",
        body="Test body content",
    )

    bus_event = original.to_bus_event()
    restored = CommsEvent.from_bus_event(bus_event)

    assert restored.timestamp == original_ts
    assert isinstance(restored.timestamp, dt.datetime)


def test_unify_message_roundtrip():
    """UnifyMessageReceived event should preserve timestamp through bus conversion."""
    original_ts = dt.datetime(2025, 6, 15, 14, 30, 0, tzinfo=dt.UTC)
    original = UnifyMessageReceived(
        timestamp=original_ts,
        contact={"contact_id": 3, "name": "Console User"},
        content="Message from console",
    )

    bus_event = original.to_bus_event()
    restored = CommsEvent.from_bus_event(bus_event)

    assert restored.timestamp == original_ts
    assert isinstance(restored.timestamp, dt.datetime)


# -------------------------------------------------------------------
#  EventBus publishing tests
# -------------------------------------------------------------------


@pytest.mark.asyncio
@_handle_project
async def test_publish_comms_with_timestamp():
    """Publishing Comms event with timestamp should not raise type errors."""
    bus = EventBus()

    now = dt.datetime.now(dt.UTC)
    event = BusEvent(
        type="Comms",
        timestamp=now,
        payload=CommsPayload(
            timestamp=now,
            contact={"contact_id": 1},
            content="Test message",
        ),
    )

    # This should not raise - previously failed with:
    # "Type mismatch for field 'timestamp': field has strict type 'str',
    #  but value has inferred type 'datetime'"
    await bus.publish(event)

    # Verify event is in the deque
    assert event in bus._deques["Comms"]


@pytest.mark.asyncio
@_handle_project
async def test_publish_and_search_comms_timestamp():
    """Published Comms event should be searchable with correct timestamp."""
    bus = EventBus()

    original_ts = dt.datetime(2025, 7, 20, 12, 0, 0, tzinfo=dt.UTC)
    event = BusEvent(
        type="Comms",
        timestamp=original_ts,
        payload=CommsPayload(
            timestamp=original_ts,
            contact={"contact_id": 1},
            content="Searchable message",
        ),
    )

    await bus.publish(event)

    # Search for the event
    results = await bus.search(
        filter='type == "Comms"',
        limit=10,
    )

    # Find our event (there may be others from previous tests)
    our_events = [
        e for e in results if e.payload.get("content") == "Searchable message"
    ]
    assert len(our_events) == 1

    # Verify timestamp is preserved as datetime
    found = our_events[0]
    assert found.timestamp == original_ts
