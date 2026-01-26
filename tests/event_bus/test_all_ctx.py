"""Tests for aggregation context mirroring and private field injection in EventBus."""

from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest
import unify

from tests.helpers import _handle_project
from unity.common.log_utils import _derive_all_contexts
from unity.events.event_bus import EVENT_BUS, Event
from unity.events.types import ManagerMethodPayload


def _get_raw_log_by_event_id(ctx: str, event_id: str):
    """Get raw log entry including private fields."""
    logs = unify.get_logs(
        context=ctx,
        filter=f'event_id == "{event_id}"',
        limit=1,
    )
    return logs[0] if logs else None


@pytest.mark.asyncio
@_handle_project
async def test_event_creates_all_contexts_entries():
    """Publishing an event should mirror to both aggregation contexts."""
    # Create a unique event
    payload = ManagerMethodPayload(
        manager="TestManager",
        method="test_all_ctx",
        phase="complete",
        result="success",
    )
    event = Event(type="ManagerMethod", payload=payload)

    # Publish the event
    await EVENT_BUS.publish(event, blocking=True)

    # Wait for aggregation callbacks to complete
    EVENT_BUS.join_published()
    await asyncio.sleep(0.2)  # Allow callbacks to execute

    # Get the type-specific context
    specific_ctx = EVENT_BUS._specific_ctxs["ManagerMethod"]

    # Derive both aggregation contexts
    all_ctxs = _derive_all_contexts(specific_ctx)
    assert len(all_ctxs) == 2, "Should have user-level and global aggregation contexts"

    # Verify the event exists in the primary context
    primary_log = _get_raw_log_by_event_id(specific_ctx, event.event_id)
    assert primary_log is not None, "Event should exist in primary context"

    # Verify it was mirrored to both aggregation contexts
    for all_ctx in all_ctxs:
        agg_log = _get_raw_log_by_event_id(all_ctx, event.event_id)
        assert agg_log is not None, f"Event should be mirrored to {all_ctx}"


@pytest.mark.asyncio
@_handle_project
async def test_global_events_table_also_mirrored():
    """The global Events table should also be mirrored to aggregation contexts."""
    payload = ManagerMethodPayload(
        manager="TestManager",
        method="test_global_mirror",
        phase="complete",
        result="success",
    )
    event = Event(type="ManagerMethod", payload=payload)

    await EVENT_BUS.publish(event, blocking=True)
    EVENT_BUS.join_published()
    await asyncio.sleep(0.2)

    # Get the global Events context
    global_ctx = EVENT_BUS._global_ctx

    # Derive aggregation contexts for the global table
    all_ctxs = _derive_all_contexts(global_ctx)
    assert len(all_ctxs) == 2, "Should have user-level and global aggregation contexts"

    # Verify it exists in global context
    global_log = _get_raw_log_by_event_id(global_ctx, event.event_id)
    assert global_log is not None, "Event should exist in global Events context"

    # Verify it was mirrored to both aggregation contexts
    for all_ctx in all_ctxs:
        agg_log = _get_raw_log_by_event_id(all_ctx, event.event_id)
        assert agg_log is not None, f"Event should be mirrored to {all_ctx}"


@pytest.mark.asyncio
@_handle_project
async def test_user_field_injected():
    """Event logs should have _user field set to user name."""
    test_user_name = "TestUserName"

    with patch(
        "unity.common.log_utils._get_user_name",
        return_value=test_user_name,
    ):
        payload = ManagerMethodPayload(
            manager="TestManager",
            method="test_user_injection",
            phase="complete",
            result="success",
        )
        event = Event(type="ManagerMethod", payload=payload)

        await EVENT_BUS.publish(event, blocking=True)
        EVENT_BUS.join_published()

        specific_ctx = EVENT_BUS._specific_ctxs["ManagerMethod"]
        log = _get_raw_log_by_event_id(specific_ctx, event.event_id)
        assert log is not None, "Log should exist"

        entries = log.entries
        assert (
            entries.get("_user") == test_user_name
        ), f"_user should be '{test_user_name}', got {entries.get('_user')}"


@pytest.mark.asyncio
@_handle_project
async def test_assistant_field_injected():
    """Event logs should have _assistant field set to assistant name."""
    test_assistant_name = "TestAssistantName"

    with patch(
        "unity.common.log_utils._get_assistant_name",
        return_value=test_assistant_name,
    ):
        payload = ManagerMethodPayload(
            manager="TestManager",
            method="test_assistant_injection",
            phase="complete",
            result="success",
        )
        event = Event(type="ManagerMethod", payload=payload)

        await EVENT_BUS.publish(event, blocking=True)
        EVENT_BUS.join_published()

        specific_ctx = EVENT_BUS._specific_ctxs["ManagerMethod"]
        log = _get_raw_log_by_event_id(specific_ctx, event.event_id)
        assert log is not None, "Log should exist"

        entries = log.entries
        assert (
            entries.get("_assistant") == test_assistant_name
        ), f"_assistant should be '{test_assistant_name}', got {entries.get('_assistant')}"


@pytest.mark.asyncio
@_handle_project
async def test_assistant_id_field_injected():
    """Event logs should have _assistant_id field set to assistant's agent_id."""
    test_assistant_id = "test-agent-789"

    with patch(
        "unity.common.log_utils._get_assistant_id",
        return_value=test_assistant_id,
    ):
        payload = ManagerMethodPayload(
            manager="TestManager",
            method="test_assistant_id_injection",
            phase="complete",
            result="success",
        )
        event = Event(type="ManagerMethod", payload=payload)

        await EVENT_BUS.publish(event, blocking=True)
        EVENT_BUS.join_published()

        specific_ctx = EVENT_BUS._specific_ctxs["ManagerMethod"]
        log = _get_raw_log_by_event_id(specific_ctx, event.event_id)
        assert log is not None, "Log should exist"

        entries = log.entries
        assert (
            entries.get("_assistant_id") == test_assistant_id
        ), f"_assistant_id should be '{test_assistant_id}', got {entries.get('_assistant_id')}"


@pytest.mark.asyncio
@_handle_project
async def test_user_id_field_injected():
    """Event logs should have _user_id field set to user's ID."""
    test_user_id = "test-user-456"

    with patch(
        "unity.common.log_utils._get_user_id",
        return_value=test_user_id,
    ):
        payload = ManagerMethodPayload(
            manager="TestManager",
            method="test_user_id_injection",
            phase="complete",
            result="success",
        )
        event = Event(type="ManagerMethod", payload=payload)

        await EVENT_BUS.publish(event, blocking=True)
        EVENT_BUS.join_published()

        specific_ctx = EVENT_BUS._specific_ctxs["ManagerMethod"]
        log = _get_raw_log_by_event_id(specific_ctx, event.event_id)
        assert log is not None, "Log should exist"

        entries = log.entries
        assert (
            entries.get("_user_id") == test_user_id
        ), f"_user_id should be '{test_user_id}', got {entries.get('_user_id')}"


@pytest.mark.asyncio
@_handle_project
async def test_all_contexts_created_on_init():
    """Aggregation contexts should be created when EventBus initializes."""
    # EventBus creates contexts in __init__ via _ensure_known_contexts()
    # Verify aggregation contexts exist for the global Events context
    all_ctxs = _derive_all_contexts(EVENT_BUS._global_ctx)
    assert len(all_ctxs) == 2, "Should have user-level and global aggregation contexts"

    contexts = unify.get_contexts()
    for all_ctx in all_ctxs:
        assert all_ctx in contexts, f"{all_ctx} context should be created"


@pytest.mark.asyncio
@_handle_project
async def test_type_specific_aggregation_contexts_created():
    """Aggregation contexts should be created for each event type."""
    # Check a specific event type's aggregation contexts
    specific_ctx = EVENT_BUS._specific_ctxs.get("ManagerMethod")
    assert specific_ctx is not None, "ManagerMethod context should exist"

    all_ctxs = _derive_all_contexts(specific_ctx)
    assert len(all_ctxs) == 2, "Should have user-level and global aggregation contexts"

    contexts = unify.get_contexts()
    for all_ctx in all_ctxs:
        assert all_ctx in contexts, f"{all_ctx} context should be created"


@pytest.mark.asyncio
@_handle_project
async def test_private_fields_in_global_table():
    """Private fields should also be present in the global Events table."""
    test_user_name = "GlobalTableUser"
    test_assistant_name = "GlobalTableAssistant"

    with (
        patch(
            "unity.common.log_utils._get_user_name",
            return_value=test_user_name,
        ),
        patch(
            "unity.common.log_utils._get_assistant_name",
            return_value=test_assistant_name,
        ),
    ):
        payload = ManagerMethodPayload(
            manager="TestManager",
            method="test_global_private_fields",
            phase="complete",
            result="success",
        )
        event = Event(type="ManagerMethod", payload=payload)

        await EVENT_BUS.publish(event, blocking=True)
        EVENT_BUS.join_published()

        # Check global Events table
        global_ctx = EVENT_BUS._global_ctx
        log = _get_raw_log_by_event_id(global_ctx, event.event_id)
        assert log is not None, "Log should exist in global table"

        entries = log.entries
        assert (
            entries.get("_user") == test_user_name
        ), f"_user should be in global table, got {entries.get('_user')}"
        assert (
            entries.get("_assistant") == test_assistant_name
        ), f"_assistant should be in global table, got {entries.get('_assistant')}"


@pytest.mark.asyncio
@_handle_project
async def test_aggregation_is_by_reference():
    """Events in aggregation contexts should be references, not copies."""
    payload = ManagerMethodPayload(
        manager="TestManager",
        method="test_reference_check",
        phase="complete",
        result="success",
    )
    event = Event(type="ManagerMethod", payload=payload)

    await EVENT_BUS.publish(event, blocking=True)
    EVENT_BUS.join_published()
    await asyncio.sleep(0.2)

    specific_ctx = EVENT_BUS._specific_ctxs["ManagerMethod"]
    all_ctxs = _derive_all_contexts(specific_ctx)

    # Get the primary log
    primary_log = _get_raw_log_by_event_id(specific_ctx, event.event_id)
    assert primary_log is not None, "Primary log should exist"

    # Verify aggregation logs have the same ID (by reference, not copy)
    for all_ctx in all_ctxs:
        agg_log = _get_raw_log_by_event_id(all_ctx, event.event_id)
        assert agg_log is not None, f"Aggregation log should exist in {all_ctx}"
        assert agg_log.id == primary_log.id, (
            f"Aggregation log in {all_ctx} should reference same log ID. "
            f"Expected {primary_log.id}, got {agg_log.id}"
        )
