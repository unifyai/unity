"""Tests verifying the distinct schema strategies for global vs type-specific contexts.

Global context (`Events`): Stores payload as single `payload_json` column to avoid
cross-type schema conflicts when different event types have same field names with
different types.

Type-specific contexts (`Events/ToolLoop`, etc.): Spreads payload fields for
queryability within a homogeneous event type.
"""

import json
import pytest
import datetime as dt

import unify

from unity.events.event_bus import EventBus, Event
from tests.helpers import _handle_project


@pytest.mark.asyncio
@_handle_project
async def test_global_context_stores_payload_json():
    """Global context should store payload as a single payload_json column,
    not spread individual payload fields.
    """
    bus = EventBus()

    # Publish an event with a structured payload
    payload = {"field_a": "value_a", "field_b": 123, "nested": {"x": 1}}
    event = Event(
        type="TestGlobalSchema",
        timestamp=dt.datetime.now(dt.UTC),
        payload=payload,
    )
    await bus.publish(event)
    bus.join_published()

    # Query the global context directly to inspect raw schema
    global_ctx = bus._global_ctx
    logs = unify.get_logs(
        project=unify.active_project(),
        context=global_ctx,
        filter=f"event_id == '{event.event_id}'",
    )

    assert len(logs) == 1
    log_entry = logs[0].entries

    # Should have payload_json as a single column
    assert "payload_json" in log_entry
    parsed = json.loads(log_entry["payload_json"])
    assert parsed == payload

    # Should NOT have spread payload fields
    assert "field_a" not in log_entry
    assert "field_b" not in log_entry
    assert "nested" not in log_entry


@pytest.mark.asyncio
@_handle_project
async def test_specific_context_spreads_payload():
    """Type-specific context should spread payload fields for queryability."""
    bus = EventBus()

    # Publish an event with a structured payload
    payload = {"method": "test_method", "status": "success", "count": 42}
    event = Event(
        type="TestSpecificSchema",
        timestamp=dt.datetime.now(dt.UTC),
        payload=payload,
    )
    await bus.publish(event)
    bus.join_published()

    # Query the type-specific context directly
    specific_ctx = bus._specific_ctxs["TestSpecificSchema"]
    logs = unify.get_logs(
        project=unify.active_project(),
        context=specific_ctx,
        filter=f"event_id == '{event.event_id}'",
    )

    assert len(logs) == 1
    log_entry = logs[0].entries

    # Should have spread payload fields (for queryability)
    assert log_entry["method"] == "test_method"
    assert log_entry["status"] == "success"
    assert log_entry["count"] == 42

    # Should NOT have payload_json in specific context
    assert "payload_json" not in log_entry


@pytest.mark.asyncio
@_handle_project
async def test_complex_payloads_no_type_conflict():
    """Publishing events with conflicting field types should not cause errors
    because global context uses payload_json instead of spreading.
    """
    bus = EventBus()

    # Event 1: message field is Dict[str, str]
    event1 = Event(
        type="TestTypeConflictA",
        timestamp=dt.datetime.now(dt.UTC),
        payload={"message": {"role": "user", "content": "hello"}},
    )
    await bus.publish(event1)
    bus.join_published()

    # Event 2: message field is Dict[str, Any] with None and nested structures
    # (This would fail with spread fields due to type inference conflict)
    event2 = Event(
        type="TestTypeConflictB",
        timestamp=dt.datetime.now(dt.UTC),
        payload={
            "message": {
                "content": None,
                "role": "assistant",
                "tool_calls": [
                    {
                        "function": {"arguments": '{"k": 1}', "name": "search"},
                        "id": "call_123",
                        "type": "function",
                    },
                ],
                "function_call": None,
            },
        },
    )
    await bus.publish(event2)
    bus.join_published()  # Should not raise type mismatch error

    # Verify both events are retrievable from global context
    global_ctx = bus._global_ctx
    logs = unify.get_logs(
        project=unify.active_project(),
        context=global_ctx,
        filter=f"event_id == '{event1.event_id}' or event_id == '{event2.event_id}'",
    )

    assert len(logs) == 2


@pytest.mark.asyncio
@_handle_project
async def test_rehydration_from_payload_json():
    """Events retrieved via search() should correctly rehydrate payload from
    payload_json column in global context.
    """
    bus = EventBus()

    # Publish an event with a complex payload
    original_payload = {
        "method": "test_rehydration",
        "args": [1, 2, 3],
        "kwargs": {"key": "value"},
        "nested": {"deep": {"structure": True}},
    }
    event = Event(
        type="TestRehydration",
        timestamp=dt.datetime.now(dt.UTC),
        payload=original_payload,
    )
    await bus.publish(event)
    bus.join_published()

    # Search for the event (uses global context read path)
    results = await bus.search(
        filter=f"event_id == '{event.event_id}'",
        limit=1,
    )

    assert len(results) == 1
    retrieved = results[0]

    # Payload should be correctly rehydrated
    assert retrieved.payload == original_payload
    assert retrieved.event_id == event.event_id
