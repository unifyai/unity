"""Tests for aggregation context mirroring and private field injection."""

from __future__ import annotations

from datetime import datetime, UTC
from unittest.mock import patch

import unify
from tests.helpers import _handle_project
from unity.common.log_utils import _derive_all_contexts
from unity.transcript_manager.transcript_manager import TranscriptManager
from unity.transcript_manager.types.message import Message


@_handle_project
def test_async_log_messages_mirrors_to_all_contexts():
    """Async log_messages (synchronous=False) should also mirror to aggregation contexts."""
    tm = TranscriptManager()

    msg = Message(
        medium="email",
        sender_id=0,
        receiver_ids=[1],
        timestamp=datetime.now(UTC),
        content="Async mirror test message",
        exchange_id=0,
    )
    # Use async path (synchronous=False is default)
    tm.log_messages(msg, synchronous=False)

    # Wait for async logger to flush
    tm.join_published()

    # Verify message exists in primary context
    result = tm._filter_messages(filter="content == 'Async mirror test message'")
    messages = result["messages"]
    assert len(messages) >= 1, "Message should exist in manager's context"
    message_id = messages[0].message_id

    # Derive aggregation contexts
    all_ctxs = _derive_all_contexts(tm._transcripts_ctx)
    assert len(all_ctxs) == 2, "Should have user-level and global aggregation contexts"

    # Verify it was mirrored to both aggregation contexts
    for all_ctx in all_ctxs:
        all_logs = unify.get_logs(
            context=all_ctx,
            filter=f"message_id == {message_id}",
        )
        assert len(all_logs) >= 1, f"Async log should be mirrored to {all_ctx}"


def _get_raw_log_by_message_id(ctx: str, message_id: int):
    """Get raw log entry including private fields."""
    logs = unify.get_logs(
        context=ctx,
        filter=f"message_id == {message_id}",
        limit=1,
    )
    return logs[0] if logs else None


@_handle_project
def test_log_creates_all_transcripts_entries():
    """Creating a message should mirror to both aggregation contexts."""
    tm = TranscriptManager()

    # Create a message
    msg = Message(
        medium="email",
        sender_id=0,
        receiver_ids=[1],
        timestamp=datetime.now(UTC),
        content="Test message for All/Ctx",
        exchange_id=0,
    )
    tm.log_messages(msg)
    tm.join_published()

    # Get the message_id from the created message
    result = tm._filter_messages(filter="content == 'Test message for All/Ctx'")
    messages = result["messages"]
    assert len(messages) >= 1, "Message should exist in manager's context"
    message_id = messages[0].message_id

    # Derive both aggregation contexts from the manager's context
    all_ctxs = _derive_all_contexts(tm._transcripts_ctx)
    assert len(all_ctxs) == 2, "Should have user-level and global aggregation contexts"

    # Verify it was mirrored to both aggregation contexts
    for all_ctx in all_ctxs:
        all_logs = unify.get_logs(
            context=all_ctx,
            filter=f"message_id == {message_id}",
        )
        assert len(all_logs) >= 1, f"Message should be mirrored to {all_ctx}"


@_handle_project
def test_user_field_injected():
    """Logs should have _user field set to user name."""
    test_user_name = "TestUserName"

    with patch(
        "unity.common.log_utils._get_user_context",
        return_value=test_user_name,
    ):
        tm = TranscriptManager()
        msg = Message(
            medium="sms_message",
            sender_id=0,
            receiver_ids=[1],
            timestamp=datetime.now(UTC),
            content="User field test",
            exchange_id=0,
        )
        tm.log_messages(msg)
        tm.join_published()

        result = tm._filter_messages(filter="content == 'User field test'")
        messages = result["messages"]
        assert len(messages) >= 1
        message_id = messages[0].message_id

        log = _get_raw_log_by_message_id(tm._transcripts_ctx, message_id)
        assert log is not None, "Log should exist"

        entries = log.entries
        assert (
            entries.get("_user") == test_user_name
        ), f"_user should be '{test_user_name}', got {entries.get('_user')}"


@_handle_project
def test_assistant_field_injected():
    """Logs should have _assistant field set to assistant name."""
    test_assistant_name = "TestAssistantName"

    with patch(
        "unity.common.log_utils._get_assistant_context",
        return_value=test_assistant_name,
    ):
        tm = TranscriptManager()
        msg = Message(
            medium="sms_message",
            sender_id=0,
            receiver_ids=[1],
            timestamp=datetime.now(UTC),
            content="Assistant field test",
            exchange_id=0,
        )
        tm.log_messages(msg)
        tm.join_published()

        result = tm._filter_messages(filter="content == 'Assistant field test'")
        messages = result["messages"]
        assert len(messages) >= 1
        message_id = messages[0].message_id

        log = _get_raw_log_by_message_id(tm._transcripts_ctx, message_id)
        assert log is not None, "Log should exist"

        entries = log.entries
        assert (
            entries.get("_assistant") == test_assistant_name
        ), f"_assistant should be '{test_assistant_name}', got {entries.get('_assistant')}"


@_handle_project
def test_assistant_id_field_injected():
    """Logs should have _assistant_id field set to assistant's agent_id."""
    test_assistant_id = "test-agent-789"

    with patch(
        "unity.common.log_utils._get_assistant_id",
        return_value=test_assistant_id,
    ):
        tm = TranscriptManager()
        msg = Message(
            medium="sms_message",
            sender_id=0,
            receiver_ids=[1],
            timestamp=datetime.now(UTC),
            content="Assistant ID field test",
            exchange_id=0,
        )
        tm.log_messages(msg)
        tm.join_published()

        result = tm._filter_messages(filter="content == 'Assistant ID field test'")
        messages = result["messages"]
        assert len(messages) >= 1
        message_id = messages[0].message_id

        log = _get_raw_log_by_message_id(tm._transcripts_ctx, message_id)
        assert log is not None, "Log should exist"

        entries = log.entries
        assert (
            entries.get("_assistant_id") == test_assistant_id
        ), f"_assistant_id should be '{test_assistant_id}', got {entries.get('_assistant_id')}"


@_handle_project
def test_user_id_field_injected():
    """Logs should have _user_id field set to user's ID."""
    test_user_id = "test-user-456"

    with patch(
        "unity.common.log_utils._get_user_id",
        return_value=test_user_id,
    ):
        tm = TranscriptManager()
        msg = Message(
            medium="email",
            sender_id=0,
            receiver_ids=[1],
            timestamp=datetime.now(UTC),
            content="User ID field test",
            exchange_id=0,
        )
        tm.log_messages(msg)
        tm.join_published()

        result = tm._filter_messages(filter="content == 'User ID field test'")
        messages = result["messages"]
        assert len(messages) >= 1
        message_id = messages[0].message_id

        log = _get_raw_log_by_message_id(tm._transcripts_ctx, message_id)
        assert log is not None, "Log should exist"

        entries = log.entries
        assert (
            entries.get("_user_id") == test_user_id
        ), f"_user_id should be '{test_user_id}', got {entries.get('_user_id')}"


@_handle_project
def test_all_contexts_created_on_provision():
    """Aggregation contexts should be created when TranscriptManager provisions storage."""
    # TranscriptManager provisions storage via ContextRegistry.get_context() in __init__
    tm = TranscriptManager()

    # Derive the expected aggregation contexts for Transcripts
    all_ctxs = _derive_all_contexts(tm._transcripts_ctx)
    assert len(all_ctxs) == 2, "Should have user-level and global aggregation contexts"

    # Verify both aggregation contexts exist
    contexts = unify.get_contexts()
    for all_ctx in all_ctxs:
        assert all_ctx in contexts, f"{all_ctx} context should be created"


@_handle_project
def test_private_fields_excluded_from_filter_messages():
    """Private fields should be excluded when reading messages via public API."""
    tm = TranscriptManager()

    msg = Message(
        medium="email",
        sender_id=0,
        receiver_ids=[1],
        timestamp=datetime.now(UTC),
        content="Private field exclusion test",
        exchange_id=0,
    )
    tm.log_messages(msg)
    tm.join_published()

    # Get message via public filter_messages API
    result = tm._filter_messages(filter="content == 'Private field exclusion test'")
    messages = result["messages"]
    assert len(messages) >= 1

    message = messages[0]
    # Private fields should NOT be in the Message model (they're excluded on read)
    assert not hasattr(message, "_user"), "_user should not be exposed"
    assert not hasattr(message, "_user_id"), "_user_id should not be exposed"
    assert not hasattr(message, "_assistant"), "_assistant should not be exposed"
    assert not hasattr(message, "_assistant_id"), "_assistant_id should not be exposed"


@_handle_project
def test_deleting_message_removes_from_all_ctxs():
    """Deleting a message should also remove it from all aggregation contexts."""
    tm = TranscriptManager()

    # Create a message
    msg = Message(
        medium="email",
        sender_id=0,
        receiver_ids=[1],
        timestamp=datetime.now(UTC),
        content="Message to be deleted",
        exchange_id=0,
    )
    tm.log_messages(msg)
    tm.join_published()

    result = tm._filter_messages(filter="content == 'Message to be deleted'")
    messages = result["messages"]
    assert len(messages) >= 1
    message_id = messages[0].message_id

    # Derive the aggregation contexts
    all_ctxs = _derive_all_contexts(tm._transcripts_ctx)
    assert len(all_ctxs) == 2, "Should have user-level and global aggregation contexts"

    # Verify it exists in all aggregation contexts before deletion
    for all_ctx in all_ctxs:
        all_logs_before = unify.get_logs(
            context=all_ctx,
            filter=f"message_id == {message_id}",
        )
        assert (
            len(all_logs_before) >= 1
        ), f"Message should exist in {all_ctx} before deletion"

    # Delete the message using unify.delete_logs
    logs_to_delete = unify.get_logs(
        context=tm._transcripts_ctx,
        filter=f"message_id == {message_id}",
    )
    if logs_to_delete:
        log_ids = [lg.id for lg in logs_to_delete]
        unify.delete_logs(logs=log_ids, context=tm._transcripts_ctx)

    # Verify it's removed from all aggregation contexts after deletion
    for all_ctx in all_ctxs:
        all_logs_after = unify.get_logs(
            context=all_ctx,
            filter=f"message_id == {message_id}",
        )
        assert (
            len(all_logs_after) == 0
        ), f"Message should be removed from {all_ctx} after deletion"


@_handle_project
def test_update_syncs_to_all_aggregation_contexts():
    """Updating a message should be immediately visible in all aggregation contexts."""
    tm = TranscriptManager()

    # Create a message with a specific sender_id
    msg = Message(
        medium="email",
        sender_id=100,
        receiver_ids=[200],
        timestamp=datetime.now(UTC),
        content="Update sync test message",
        exchange_id=0,
    )
    tm.log_messages(msg)
    tm.join_published()

    result = tm._filter_messages(filter="content == 'Update sync test message'")
    messages = result["messages"]
    assert len(messages) >= 1
    message_id = messages[0].message_id

    # Derive aggregation contexts
    all_ctxs = _derive_all_contexts(tm._transcripts_ctx)
    assert len(all_ctxs) == 2, "Should have user-level and global aggregation contexts"

    # Verify initial sender_id in all contexts
    for ctx in [tm._transcripts_ctx, *all_ctxs]:
        log = _get_raw_log_by_message_id(ctx, message_id)
        assert log is not None, f"Log should exist in {ctx}"
        assert log.entries.get("sender_id") == 100, f"Initial sender_id in {ctx}"

    # Update the contact_id from 100 to 999
    tm.update_contact_id(original_contact_id=100, new_contact_id=999)

    # Verify the update is immediately visible in ALL contexts (primary + aggregations)
    for ctx in [tm._transcripts_ctx, *all_ctxs]:
        log = _get_raw_log_by_message_id(ctx, message_id)
        assert log is not None, f"Log should exist in {ctx} after update"
        assert log.entries.get("sender_id") == 999, (
            f"Updated sender_id should be visible in {ctx}. "
            f"Expected 999, got '{log.entries.get('sender_id')}'"
        )


@_handle_project
def test_log_id_unchanged_after_update():
    """Updates should modify the existing log entry, not create a new one."""
    tm = TranscriptManager()

    # Create a message
    msg = Message(
        medium="email",
        sender_id=101,
        receiver_ids=[201],
        timestamp=datetime.now(UTC),
        content="Log ID test message",
        exchange_id=0,
    )
    tm.log_messages(msg)
    tm.join_published()

    result = tm._filter_messages(filter="content == 'Log ID test message'")
    messages = result["messages"]
    assert len(messages) >= 1
    message_id = messages[0].message_id

    # Get the original log ID
    original_log = _get_raw_log_by_message_id(tm._transcripts_ctx, message_id)
    original_log_id = original_log.id

    # Update the contact_id
    tm.update_contact_id(original_contact_id=101, new_contact_id=888)

    # Verify the log ID is unchanged (in-place update, not delete+create)
    updated_log = _get_raw_log_by_message_id(tm._transcripts_ctx, message_id)
    assert updated_log.id == original_log_id, (
        f"Log ID should be unchanged after update. "
        f"Original: {original_log_id}, After update: {updated_log.id}"
    )

    # Verify all aggregation contexts still reference the same log ID
    all_ctxs = _derive_all_contexts(tm._transcripts_ctx)
    for all_ctx in all_ctxs:
        agg_log = _get_raw_log_by_message_id(all_ctx, message_id)
        assert agg_log.id == original_log_id, (
            f"Aggregation context {all_ctx} should still reference the same log. "
            f"Expected {original_log_id}, got {agg_log.id}"
        )
