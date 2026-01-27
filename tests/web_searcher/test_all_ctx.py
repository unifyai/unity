"""Tests for aggregation context mirroring and private field injection."""

from __future__ import annotations

from unittest.mock import patch

import unify
from tests.helpers import _handle_project
from unity.common.log_utils import _derive_all_contexts
from unity.web_searcher.web_searcher import WebSearcher


def _get_raw_log_by_website_id(ctx: str, website_id: int):
    """Get raw log entry including private fields."""
    logs = unify.get_logs(
        context=ctx,
        filter=f"website_id == {website_id}",
        limit=1,
    )
    return logs[0] if logs else None


@_handle_project
def test_log_creates_all_websites_entries():
    """Creating a website should mirror to both aggregation contexts."""
    ws = WebSearcher()

    # Create a website
    result = ws._create_website(
        name="Test Website",
        host="test-all-ctx.example.com",
        gated=False,
        subscribed=False,
        notes="Test website for All/Ctx",
    )
    assert result["outcome"] == "website created"

    # Get the website_id
    websites = ws._filter_websites(filter="host == 'test-all-ctx.example.com'")
    assert len(websites) == 1, "Website should exist in manager's context"
    website_id = websites[0].website_id

    # Derive both aggregation contexts from the manager's context
    all_ctxs = _derive_all_contexts(ws._websites_ctx)
    assert len(all_ctxs) == 2, "Should have user-level and global aggregation contexts"

    # Verify it was mirrored to both aggregation contexts
    for all_ctx in all_ctxs:
        all_logs = unify.get_logs(
            context=all_ctx,
            filter=f"website_id == {website_id}",
        )
        assert len(all_logs) >= 1, f"Website should be mirrored to {all_ctx}"


@_handle_project
def test_user_field_injected():
    """Logs should have _user field set to user name."""
    test_user_name = "TestUserName"

    with patch(
        "unity.common.log_utils._get_user_name",
        return_value=test_user_name,
    ):
        ws = WebSearcher()
        result = ws._create_website(
            name="User Test Site",
            host="user-test.example.com",
            gated=False,
            subscribed=False,
            notes="Testing user field",
        )
        assert result["outcome"] == "website created"

        websites = ws._filter_websites(filter="host == 'user-test.example.com'")
        assert len(websites) >= 1
        website_id = websites[0].website_id

        log = _get_raw_log_by_website_id(ws._websites_ctx, website_id)
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
        "unity.common.log_utils._get_assistant_name",
        return_value=test_assistant_name,
    ):
        ws = WebSearcher()
        result = ws._create_website(
            name="Assistant Test Site",
            host="assistant-test.example.com",
            gated=False,
            subscribed=False,
            notes="Testing assistant field",
        )
        assert result["outcome"] == "website created"

        websites = ws._filter_websites(filter="host == 'assistant-test.example.com'")
        assert len(websites) >= 1
        website_id = websites[0].website_id

        log = _get_raw_log_by_website_id(ws._websites_ctx, website_id)
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
        ws = WebSearcher()
        result = ws._create_website(
            name="Assistant ID Test Site",
            host="assistant-id-test.example.com",
            gated=False,
            subscribed=False,
            notes="Testing assistant ID field",
        )
        assert result["outcome"] == "website created"

        websites = ws._filter_websites(filter="host == 'assistant-id-test.example.com'")
        assert len(websites) >= 1
        website_id = websites[0].website_id

        log = _get_raw_log_by_website_id(ws._websites_ctx, website_id)
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
        ws = WebSearcher()
        result = ws._create_website(
            name="User ID Test Site",
            host="user-id-test.example.com",
            gated=False,
            subscribed=False,
            notes="Testing user ID field",
        )
        assert result["outcome"] == "website created"

        websites = ws._filter_websites(filter="host == 'user-id-test.example.com'")
        assert len(websites) >= 1
        website_id = websites[0].website_id

        log = _get_raw_log_by_website_id(ws._websites_ctx, website_id)
        assert log is not None, "Log should exist"

        entries = log.entries
        assert (
            entries.get("_user_id") == test_user_id
        ), f"_user_id should be '{test_user_id}', got {entries.get('_user_id')}"


@_handle_project
def test_all_contexts_created_on_provision():
    """Aggregation contexts should be created when WebSearcher provisions storage."""
    # WebSearcher provisions storage via ContextRegistry.get_context() in __init__
    ws = WebSearcher()

    # Derive the expected aggregation contexts
    all_ctxs = _derive_all_contexts(ws._websites_ctx)
    assert len(all_ctxs) == 2, "Should have user-level and global aggregation contexts"

    # Verify both aggregation contexts exist
    contexts = unify.get_contexts()
    for all_ctx in all_ctxs:
        assert all_ctx in contexts, f"{all_ctx} context should be created"


@_handle_project
def test_private_fields_excluded_from_filter_websites():
    """Private fields should be excluded when reading websites via public API."""
    ws = WebSearcher()

    result = ws._create_website(
        name="Private Test Site",
        host="private-test.example.com",
        gated=False,
        subscribed=False,
        notes="Testing private field exclusion",
    )
    assert result["outcome"] == "website created"

    # Get website via _filter_websites API
    websites = ws._filter_websites(filter="host == 'private-test.example.com'")
    assert len(websites) == 1

    website = websites[0]
    # Private fields should NOT be in the Website model (they're excluded on read)
    assert not hasattr(website, "_user"), "_user should not be exposed"
    assert not hasattr(website, "_user_id"), "_user_id should not be exposed"
    assert not hasattr(website, "_assistant"), "_assistant should not be exposed"
    assert not hasattr(website, "_assistant_id"), "_assistant_id should not be exposed"


@_handle_project
def test_deleting_website_removes_from_all_ctxs():
    """Deleting a website should also remove it from all aggregation contexts."""
    ws = WebSearcher()

    # Create a website
    result = ws._create_website(
        name="Delete Test Site",
        host="delete-test.example.com",
        gated=False,
        subscribed=False,
        notes="Website to be deleted",
    )
    assert result["outcome"] == "website created"

    websites = ws._filter_websites(filter="host == 'delete-test.example.com'")
    assert len(websites) >= 1
    website_id = websites[0].website_id

    # Derive the aggregation contexts
    all_ctxs = _derive_all_contexts(ws._websites_ctx)
    assert len(all_ctxs) == 2, "Should have user-level and global aggregation contexts"

    # Verify it exists in all aggregation contexts before deletion
    for all_ctx in all_ctxs:
        all_logs_before = unify.get_logs(
            context=all_ctx,
            filter=f"website_id == {website_id}",
        )
        assert (
            len(all_logs_before) >= 1
        ), f"Website should exist in {all_ctx} before deletion"

    # Delete the website
    ws._delete_website(host="delete-test.example.com")

    # Verify it's removed from all aggregation contexts after deletion
    for all_ctx in all_ctxs:
        all_logs_after = unify.get_logs(
            context=all_ctx,
            filter=f"website_id == {website_id}",
        )
        assert (
            len(all_logs_after) == 0
        ), f"Website should be removed from {all_ctx} after deletion"


@_handle_project
def test_update_syncs_to_all_aggregation_contexts():
    """Updating a website should be immediately visible in all aggregation contexts."""
    ws = WebSearcher()

    # Create a website with initial values
    result = ws._create_website(
        name="Update Sync Site",
        host="update-sync.example.com",
        gated=False,
        subscribed=False,
        notes="Original notes",
    )
    assert result["outcome"] == "website created"

    websites = ws._filter_websites(filter="host == 'update-sync.example.com'")
    assert len(websites) >= 1
    website_id = websites[0].website_id

    # Derive aggregation contexts
    all_ctxs = _derive_all_contexts(ws._websites_ctx)
    assert len(all_ctxs) == 2, "Should have user-level and global aggregation contexts"

    # Verify initial notes in all contexts
    for ctx in [ws._websites_ctx, *all_ctxs]:
        log = _get_raw_log_by_website_id(ctx, website_id)
        assert log is not None, f"Log should exist in {ctx}"
        assert log.entries.get("notes") == "Original notes", f"Initial notes in {ctx}"

    # Update the website's notes
    ws._update_website(website_id=website_id, notes="Updated notes")

    # Verify the update is immediately visible in ALL contexts (primary + aggregations)
    for ctx in [ws._websites_ctx, *all_ctxs]:
        log = _get_raw_log_by_website_id(ctx, website_id)
        assert log is not None, f"Log should exist in {ctx} after update"
        assert log.entries.get("notes") == "Updated notes", (
            f"Updated notes should be visible in {ctx}. "
            f"Expected 'Updated notes', got '{log.entries.get('notes')}'"
        )


@_handle_project
def test_log_id_unchanged_after_update():
    """Updates should modify the existing log entry, not create a new one."""
    ws = WebSearcher()

    # Create a website
    result = ws._create_website(
        name="Log ID Test Site",
        host="log-id-test.example.com",
        gated=False,
        subscribed=False,
        notes="Before update",
    )
    assert result["outcome"] == "website created"

    websites = ws._filter_websites(filter="host == 'log-id-test.example.com'")
    assert len(websites) >= 1
    website_id = websites[0].website_id

    # Get the original log ID
    original_log = _get_raw_log_by_website_id(ws._websites_ctx, website_id)
    original_log_id = original_log.id

    # Update the website
    ws._update_website(website_id=website_id, notes="After update")

    # Verify the log ID is unchanged (in-place update, not delete+create)
    updated_log = _get_raw_log_by_website_id(ws._websites_ctx, website_id)
    assert updated_log.id == original_log_id, (
        f"Log ID should be unchanged after update. "
        f"Original: {original_log_id}, After update: {updated_log.id}"
    )

    # Verify all aggregation contexts still reference the same log ID
    all_ctxs = _derive_all_contexts(ws._websites_ctx)
    for all_ctx in all_ctxs:
        agg_log = _get_raw_log_by_website_id(all_ctx, website_id)
        assert agg_log.id == original_log_id, (
            f"Aggregation context {all_ctx} should still reference the same log. "
            f"Expected {original_log_id}, got {agg_log.id}"
        )
