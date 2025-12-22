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
from unity.events.types.manager_method import ManagerMethodPayload
from unity.events.types.tool_loop import ToolLoopPayload
from tests.helpers import _handle_project


@pytest.mark.asyncio
@_handle_project
async def test_global_context_stores_payload_json():
    """Global context should store payload as a single payload_json column,
    not spread individual payload fields.
    """
    bus = EventBus()

    # Publish a ManagerMethod event
    payload = ManagerMethodPayload(
        manager="TestManager",
        method="test_method",
        phase="incoming",
        question="test question",
    )
    event = Event(
        type="ManagerMethod",
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
    assert parsed["manager"] == "TestManager"
    assert parsed["method"] == "test_method"

    # Should NOT have spread payload fields in global context
    assert "manager" not in log_entry
    assert "method" not in log_entry


@pytest.mark.asyncio
@_handle_project
async def test_specific_context_spreads_payload():
    """Type-specific context should spread payload fields for queryability."""
    bus = EventBus()

    # Publish a ManagerMethod event
    payload = ManagerMethodPayload(
        manager="QueryManager",
        method="ask",
        phase="incoming",
        question="what is the answer?",
    )
    event = Event(
        type="ManagerMethod",
        timestamp=dt.datetime.now(dt.UTC),
        payload=payload,
    )
    await bus.publish(event)
    bus.join_published()

    # Query the type-specific context directly
    specific_ctx = bus._specific_ctxs["ManagerMethod"]
    logs = unify.get_logs(
        project=unify.active_project(),
        context=specific_ctx,
        filter=f"event_id == '{event.event_id}'",
    )

    assert len(logs) == 1
    log_entry = logs[0].entries

    # Should have spread payload fields (for queryability)
    assert log_entry["manager"] == "QueryManager"
    assert log_entry["method"] == "ask"
    assert log_entry["phase"] == "incoming"

    # Should NOT have payload_json in specific context
    assert "payload_json" not in log_entry


@pytest.mark.asyncio
@_handle_project
async def test_toolloop_complex_message_no_type_conflict():
    """ToolLoop events with complex LLM messages should not cause type conflicts.

    The 'message' field is typed as Dict[str, Any] to handle varied LLM response
    shapes including None values, nested arrays, etc.
    """
    bus = EventBus()

    # Simple message
    simple_payload = ToolLoopPayload(
        message={"role": "user", "content": "hello"},
        method="test_method",
        hierarchy=["root"],
        hierarchy_label="Test",
    )
    event1 = Event(
        type="ToolLoop",
        timestamp=dt.datetime.now(dt.UTC),
        payload=simple_payload,
    )
    await bus.publish(event1)
    bus.join_published()

    # Complex message with None and nested structures
    complex_payload = ToolLoopPayload(
        message={
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
        method="test_method",
        hierarchy=["root"],
        hierarchy_label="Test",
    )
    event2 = Event(
        type="ToolLoop",
        timestamp=dt.datetime.now(dt.UTC),
        payload=complex_payload,
    )
    await bus.publish(event2)
    bus.join_published()  # Should not raise type mismatch error

    # Verify both events are retrievable
    results = await bus.search(
        filter=f"event_id == '{event1.event_id}' or event_id == '{event2.event_id}'",
        limit=10,
    )
    assert len(results) == 2


@pytest.mark.asyncio
@_handle_project
async def test_rehydration_from_payload_json():
    """Events retrieved via search() should correctly rehydrate payload from
    payload_json column in global context.
    """
    bus = EventBus()

    # Publish a ToolLoop event with a complex message
    original_payload = ToolLoopPayload(
        message={"role": "assistant", "content": "test content", "extra": [1, 2, 3]},
        method="test_rehydration",
        hierarchy=["level1", "level2"],
        hierarchy_label="Test > Nested",
        origin="test_origin",
    )
    event = Event(
        type="ToolLoop",
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

    # Payload is always a dict after validation
    assert retrieved.payload["method"] == "test_rehydration"
    assert retrieved.payload["hierarchy"] == ["level1", "level2"]

    assert retrieved.event_id == event.event_id
