"""Tests for aggregation context mirroring and private field injection."""

from __future__ import annotations

from unittest.mock import patch

import unify
from tests.helpers import _handle_project
from unity.common.log_utils import _derive_all_contexts
from unity.guidance_manager.guidance_manager import GuidanceManager


def _get_raw_log_by_guidance_id(ctx: str, guidance_id: int):
    """Get raw log entry including private fields."""
    logs = unify.get_logs(
        context=ctx,
        filter=f"guidance_id == {guidance_id}",
        limit=1,
    )
    return logs[0] if logs else None


@_handle_project
def test_log_creates_all_guidance_entries():
    """Creating a guidance entry should mirror to both aggregation contexts."""
    gm = GuidanceManager()

    # Create a guidance entry
    result = gm.add_guidance(
        title="Test Guidance",
        content="Test guidance content for All/Ctx",
    )
    guidance_id = result["details"]["guidance_id"]

    # Verify it exists in the manager's context
    guidance = gm.filter(filter=f"guidance_id == {guidance_id}")
    assert len(guidance) == 1, "Guidance should exist in manager's context"

    # Derive both aggregation contexts from the manager's context
    all_ctxs = _derive_all_contexts(gm._ctx)
    assert len(all_ctxs) == 2, "Should have user-level and global aggregation contexts"

    # Verify it was mirrored to both aggregation contexts
    for all_ctx in all_ctxs:
        all_logs = unify.get_logs(
            context=all_ctx,
            filter=f"guidance_id == {guidance_id}",
        )
        assert len(all_logs) >= 1, f"Guidance should be mirrored to {all_ctx}"


@_handle_project
def test_user_field_injected():
    """Logs should have _user field set to user name."""
    test_user_name = "TestUserName"

    with patch(
        "unity.common.log_utils._get_user_context",
        return_value=test_user_name,
    ):
        gm = GuidanceManager()
        result = gm.add_guidance(
            title="User Test Guidance",
            content="Testing user field injection",
        )
        guidance_id = result["details"]["guidance_id"]

        log = _get_raw_log_by_guidance_id(gm._ctx, guidance_id)
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
        gm = GuidanceManager()
        result = gm.add_guidance(
            title="Assistant Test Guidance",
            content="Testing assistant field injection",
        )
        guidance_id = result["details"]["guidance_id"]

        log = _get_raw_log_by_guidance_id(gm._ctx, guidance_id)
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
        gm = GuidanceManager()
        result = gm.add_guidance(
            title="Assistant ID Test",
            content="Testing assistant ID field injection",
        )
        guidance_id = result["details"]["guidance_id"]

        log = _get_raw_log_by_guidance_id(gm._ctx, guidance_id)
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
        gm = GuidanceManager()
        result = gm.add_guidance(
            title="User ID Test",
            content="Testing user ID field injection",
        )
        guidance_id = result["details"]["guidance_id"]

        log = _get_raw_log_by_guidance_id(gm._ctx, guidance_id)
        assert log is not None, "Log should exist"

        entries = log.entries
        assert (
            entries.get("_user_id") == test_user_id
        ), f"_user_id should be '{test_user_id}', got {entries.get('_user_id')}"


@_handle_project
def test_all_contexts_created_on_provision():
    """Aggregation contexts should be created when GuidanceManager provisions storage."""
    # GuidanceManager provisions storage via ContextRegistry.get_context() in __init__
    gm = GuidanceManager()

    # Derive the expected aggregation contexts
    all_ctxs = _derive_all_contexts(gm._ctx)
    assert len(all_ctxs) == 2, "Should have user-level and global aggregation contexts"

    # Verify both aggregation contexts exist
    contexts = unify.get_contexts()
    for all_ctx in all_ctxs:
        assert all_ctx in contexts, f"{all_ctx} context should be created"


@_handle_project
def test_private_fields_excluded_from_filter():
    """Private fields should be excluded when reading guidance via public API."""
    gm = GuidanceManager()

    result = gm.add_guidance(
        title="Private Field Test",
        content="Testing private field exclusion",
    )
    guidance_id = result["details"]["guidance_id"]

    # Get guidance via filter API
    guidance_list = gm.filter(filter=f"guidance_id == {guidance_id}")
    assert len(guidance_list) == 1

    guidance = guidance_list[0]
    # Private fields should NOT be in the Guidance model (they're excluded on read)
    assert not hasattr(guidance, "_user"), "_user should not be exposed"
    assert not hasattr(guidance, "_user_id"), "_user_id should not be exposed"
    assert not hasattr(guidance, "_assistant"), "_assistant should not be exposed"
    assert not hasattr(guidance, "_assistant_id"), "_assistant_id should not be exposed"


@_handle_project
def test_deleting_guidance_removes_from_all_ctxs():
    """Deleting a guidance entry should also remove it from all aggregation contexts."""
    gm = GuidanceManager()

    # Create a guidance entry
    result = gm.add_guidance(
        title="Delete Test Guidance",
        content="Guidance to be deleted",
    )
    guidance_id = result["details"]["guidance_id"]

    # Derive the aggregation contexts
    all_ctxs = _derive_all_contexts(gm._ctx)
    assert len(all_ctxs) == 2, "Should have user-level and global aggregation contexts"

    # Verify it exists in all aggregation contexts before deletion
    for all_ctx in all_ctxs:
        all_logs_before = unify.get_logs(
            context=all_ctx,
            filter=f"guidance_id == {guidance_id}",
        )
        assert (
            len(all_logs_before) >= 1
        ), f"Guidance should exist in {all_ctx} before deletion"

    # Delete the guidance
    gm.delete_guidance(guidance_id=guidance_id)

    # Verify it's removed from all aggregation contexts after deletion
    for all_ctx in all_ctxs:
        all_logs_after = unify.get_logs(
            context=all_ctx,
            filter=f"guidance_id == {guidance_id}",
        )
        assert (
            len(all_logs_after) == 0
        ), f"Guidance should be removed from {all_ctx} after deletion"


@_handle_project
def test_update_syncs_to_all_aggregation_contexts():
    """Updating a guidance entry should be immediately visible in all aggregation contexts."""
    gm = GuidanceManager()

    # Create a guidance entry with initial values
    result = gm.add_guidance(
        title="Update Sync Guidance",
        content="Original content",
    )
    guidance_id = result["details"]["guidance_id"]

    # Derive aggregation contexts
    all_ctxs = _derive_all_contexts(gm._ctx)
    assert len(all_ctxs) == 2, "Should have user-level and global aggregation contexts"

    # Verify initial content in all contexts
    for ctx in [gm._ctx, *all_ctxs]:
        log = _get_raw_log_by_guidance_id(ctx, guidance_id)
        assert log is not None, f"Log should exist in {ctx}"
        assert (
            log.entries.get("content") == "Original content"
        ), f"Initial content in {ctx}"

    # Update the guidance's content
    gm.update_guidance(guidance_id=guidance_id, content="Updated content")

    # Verify the update is immediately visible in ALL contexts (primary + aggregations)
    for ctx in [gm._ctx, *all_ctxs]:
        log = _get_raw_log_by_guidance_id(ctx, guidance_id)
        assert log is not None, f"Log should exist in {ctx} after update"
        assert log.entries.get("content") == "Updated content", (
            f"Updated content should be visible in {ctx}. "
            f"Expected 'Updated content', got '{log.entries.get('content')}'"
        )


@_handle_project
def test_log_id_unchanged_after_update():
    """Updates should modify the existing log entry, not create a new one."""
    gm = GuidanceManager()

    # Create a guidance entry
    result = gm.add_guidance(
        title="Log ID Test Guidance",
        content="Before update",
    )
    guidance_id = result["details"]["guidance_id"]

    # Get the original log ID
    original_log = _get_raw_log_by_guidance_id(gm._ctx, guidance_id)
    original_log_id = original_log.id

    # Update the guidance
    gm.update_guidance(guidance_id=guidance_id, content="After update")

    # Verify the log ID is unchanged (in-place update, not delete+create)
    updated_log = _get_raw_log_by_guidance_id(gm._ctx, guidance_id)
    assert updated_log.id == original_log_id, (
        f"Log ID should be unchanged after update. "
        f"Original: {original_log_id}, After update: {updated_log.id}"
    )

    # Verify all aggregation contexts still reference the same log ID
    all_ctxs = _derive_all_contexts(gm._ctx)
    for all_ctx in all_ctxs:
        agg_log = _get_raw_log_by_guidance_id(all_ctx, guidance_id)
        assert agg_log.id == original_log_id, (
            f"Aggregation context {all_ctx} should still reference the same log. "
            f"Expected {original_log_id}, got {agg_log.id}"
        )
