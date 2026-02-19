"""Tests for aggregation context mirroring and private field injection."""

from __future__ import annotations

from unittest.mock import patch

import unify
from tests.helpers import _handle_project
from unity.common.log_utils import _derive_all_contexts
from unity.blacklist_manager.blacklist_manager import BlackListManager


def _get_raw_log_by_blacklist_id(ctx: str, blacklist_id: int):
    """Get raw log entry including private fields."""
    logs = unify.get_logs(
        context=ctx,
        filter=f"blacklist_id == {blacklist_id}",
        limit=1,
    )
    return logs[0] if logs else None


@_handle_project
def test_log_creates_all_blacklist_entries():
    """Creating a blacklist entry should mirror to both aggregation contexts."""
    bm = BlackListManager()

    # Create a blacklist entry
    result = bm.create_blacklist_entry(
        medium="email",
        contact_detail="spam@example.com",
        reason="Test blacklist entry for All/Ctx",
    )
    assert result["outcome"] == "blacklist entry created"
    blacklist_id = result["details"]["blacklist_id"]

    # Verify it exists in the manager's context
    entries = bm.filter_blacklist(filter=f"blacklist_id == {blacklist_id}")["entries"]
    assert len(entries) == 1, "Blacklist entry should exist in manager's context"

    # Derive both aggregation contexts from the manager's context
    all_ctxs = _derive_all_contexts(bm._ctx)
    assert len(all_ctxs) == 2, "Should have user-level and global aggregation contexts"

    # Verify it was mirrored to both aggregation contexts
    for all_ctx in all_ctxs:
        all_logs = unify.get_logs(
            context=all_ctx,
            filter=f"blacklist_id == {blacklist_id}",
        )
        assert len(all_logs) >= 1, f"Blacklist entry should be mirrored to {all_ctx}"


@_handle_project
def test_user_field_injected():
    """Logs should have _user field set to user name."""
    test_user_name = "TestUserName"

    with patch(
        "unity.common.log_utils._get_user_context",
        return_value=test_user_name,
    ):
        bm = BlackListManager()
        result = bm.create_blacklist_entry(
            medium="email",
            contact_detail="user-test@example.com",
            reason="Testing user field",
        )
        assert result["outcome"] == "blacklist entry created"
        blacklist_id = result["details"]["blacklist_id"]

        log = _get_raw_log_by_blacklist_id(bm._ctx, blacklist_id)
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
        bm = BlackListManager()
        result = bm.create_blacklist_entry(
            medium="sms_message",
            contact_detail="+1234567890",
            reason="Testing assistant field",
        )
        assert result["outcome"] == "blacklist entry created"
        blacklist_id = result["details"]["blacklist_id"]

        log = _get_raw_log_by_blacklist_id(bm._ctx, blacklist_id)
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
        bm = BlackListManager()
        result = bm.create_blacklist_entry(
            medium="sms_message",
            contact_detail="+9876543210",
            reason="Testing assistant ID field",
        )
        assert result["outcome"] == "blacklist entry created"
        blacklist_id = result["details"]["blacklist_id"]

        log = _get_raw_log_by_blacklist_id(bm._ctx, blacklist_id)
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
        bm = BlackListManager()
        result = bm.create_blacklist_entry(
            medium="email",
            contact_detail="userid-test@example.com",
            reason="Testing user ID field",
        )
        assert result["outcome"] == "blacklist entry created"
        blacklist_id = result["details"]["blacklist_id"]

        log = _get_raw_log_by_blacklist_id(bm._ctx, blacklist_id)
        assert log is not None, "Log should exist"

        entries = log.entries
        assert (
            entries.get("_user_id") == test_user_id
        ), f"_user_id should be '{test_user_id}', got {entries.get('_user_id')}"


@_handle_project
def test_all_contexts_created_on_provision():
    """Aggregation contexts should be created when BlackListManager provisions storage."""
    # BlackListManager provisions storage via ContextRegistry.get_context() in __init__
    bm = BlackListManager()

    # Derive the expected aggregation contexts
    all_ctxs = _derive_all_contexts(bm._ctx)
    assert len(all_ctxs) == 2, "Should have user-level and global aggregation contexts"

    # Verify both aggregation contexts exist
    contexts = unify.get_contexts()
    for all_ctx in all_ctxs:
        assert all_ctx in contexts, f"{all_ctx} context should be created"


@_handle_project
def test_private_fields_excluded_from_filter_blacklist():
    """Private fields should be excluded when reading blacklist via public API."""
    bm = BlackListManager()

    result = bm.create_blacklist_entry(
        medium="email",
        contact_detail="private-test@example.com",
        reason="Testing private field exclusion",
    )
    assert result["outcome"] == "blacklist entry created"
    blacklist_id = result["details"]["blacklist_id"]

    # Get blacklist entry via filter_blacklist API
    entries = bm.filter_blacklist(filter=f"blacklist_id == {blacklist_id}")["entries"]
    assert len(entries) == 1

    entry = entries[0]
    # Private fields should NOT be in the BlackList model (they're excluded on read)
    assert not hasattr(entry, "_user"), "_user should not be exposed"
    assert not hasattr(entry, "_user_id"), "_user_id should not be exposed"
    assert not hasattr(entry, "_assistant"), "_assistant should not be exposed"
    assert not hasattr(entry, "_assistant_id"), "_assistant_id should not be exposed"


@_handle_project
def test_deleting_blacklist_entry_removes_from_all_ctxs():
    """Deleting a blacklist entry should also remove it from all aggregation contexts."""
    bm = BlackListManager()

    # Create a blacklist entry
    result = bm.create_blacklist_entry(
        medium="email",
        contact_detail="delete-test@example.com",
        reason="Entry to be deleted",
    )
    assert result["outcome"] == "blacklist entry created"
    blacklist_id = result["details"]["blacklist_id"]

    # Derive the aggregation contexts
    all_ctxs = _derive_all_contexts(bm._ctx)
    assert len(all_ctxs) == 2, "Should have user-level and global aggregation contexts"

    # Verify it exists in all aggregation contexts before deletion
    for all_ctx in all_ctxs:
        all_logs_before = unify.get_logs(
            context=all_ctx,
            filter=f"blacklist_id == {blacklist_id}",
        )
        assert (
            len(all_logs_before) >= 1
        ), f"Blacklist entry should exist in {all_ctx} before deletion"

    # Delete the blacklist entry
    bm.delete_blacklist_entry(blacklist_id=blacklist_id)

    # Verify it's removed from all aggregation contexts after deletion
    for all_ctx in all_ctxs:
        all_logs_after = unify.get_logs(
            context=all_ctx,
            filter=f"blacklist_id == {blacklist_id}",
        )
        assert (
            len(all_logs_after) == 0
        ), f"Blacklist entry should be removed from {all_ctx} after deletion"


@_handle_project
def test_update_syncs_to_all_aggregation_contexts():
    """Updating a blacklist entry should be immediately visible in all aggregation contexts."""
    bm = BlackListManager()

    # Create a blacklist entry with initial values
    result = bm.create_blacklist_entry(
        medium="email",
        contact_detail="update-sync@example.com",
        reason="Original reason",
    )
    assert result["outcome"] == "blacklist entry created"
    blacklist_id = result["details"]["blacklist_id"]

    # Derive aggregation contexts
    all_ctxs = _derive_all_contexts(bm._ctx)
    assert len(all_ctxs) == 2, "Should have user-level and global aggregation contexts"

    # Verify initial reason in all contexts
    for ctx in [bm._ctx, *all_ctxs]:
        log = _get_raw_log_by_blacklist_id(ctx, blacklist_id)
        assert log is not None, f"Log should exist in {ctx}"
        assert (
            log.entries.get("reason") == "Original reason"
        ), f"Initial reason in {ctx}"

    # Update the blacklist entry's reason
    bm.update_blacklist_entry(blacklist_id=blacklist_id, reason="Updated reason")

    # Verify the update is immediately visible in ALL contexts (primary + aggregations)
    for ctx in [bm._ctx, *all_ctxs]:
        log = _get_raw_log_by_blacklist_id(ctx, blacklist_id)
        assert log is not None, f"Log should exist in {ctx} after update"
        assert log.entries.get("reason") == "Updated reason", (
            f"Updated reason should be visible in {ctx}. "
            f"Expected 'Updated reason', got '{log.entries.get('reason')}'"
        )


@_handle_project
def test_log_id_unchanged_after_update():
    """Updates should modify the existing log entry, not create a new one."""
    bm = BlackListManager()

    # Create a blacklist entry
    result = bm.create_blacklist_entry(
        medium="email",
        contact_detail="log-id-test@example.com",
        reason="Before update",
    )
    assert result["outcome"] == "blacklist entry created"
    blacklist_id = result["details"]["blacklist_id"]

    # Get the original log ID
    original_log = _get_raw_log_by_blacklist_id(bm._ctx, blacklist_id)
    original_log_id = original_log.id

    # Update the blacklist entry
    bm.update_blacklist_entry(blacklist_id=blacklist_id, reason="After update")

    # Verify the log ID is unchanged (in-place update, not delete+create)
    updated_log = _get_raw_log_by_blacklist_id(bm._ctx, blacklist_id)
    assert updated_log.id == original_log_id, (
        f"Log ID should be unchanged after update. "
        f"Original: {original_log_id}, After update: {updated_log.id}"
    )

    # Verify all aggregation contexts still reference the same log ID
    all_ctxs = _derive_all_contexts(bm._ctx)
    for all_ctx in all_ctxs:
        agg_log = _get_raw_log_by_blacklist_id(all_ctx, blacklist_id)
        assert agg_log.id == original_log_id, (
            f"Aggregation context {all_ctx} should still reference the same log. "
            f"Expected {original_log_id}, got {agg_log.id}"
        )
