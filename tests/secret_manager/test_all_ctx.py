"""Tests for aggregation context mirroring and private field injection."""

from __future__ import annotations

from unittest.mock import patch

import unify
from tests.helpers import _handle_project
from unity.common.log_utils import _derive_all_contexts
from unity.secret_manager.secret_manager import SecretManager


def _get_raw_log_by_secret_id(ctx: str, secret_id: int):
    """Get raw log entry including private fields."""
    logs = unify.get_logs(
        context=ctx,
        filter=f"secret_id == {secret_id}",
        limit=1,
    )
    return logs[0] if logs else None


@_handle_project
def test_log_creates_all_secrets_entries():
    """Creating a secret should mirror to both aggregation contexts."""
    sm = SecretManager()

    # Create a secret
    result = sm._create_secret(
        name="test_secret_all_ctx",
        value="test-value-123",
        description="Test secret for All/Ctx",
    )
    assert result["outcome"] == "secret created"

    # Get the secret_id
    secrets = sm._filter_secrets(filter="name == 'test_secret_all_ctx'")
    assert len(secrets) == 1, "Secret should exist in manager's context"
    secret_id = secrets[0].secret_id

    # Derive both aggregation contexts from the manager's context
    all_ctxs = _derive_all_contexts(sm._ctx)
    assert len(all_ctxs) == 2, "Should have user-level and global aggregation contexts"

    # Verify it was mirrored to both aggregation contexts
    for all_ctx in all_ctxs:
        all_logs = unify.get_logs(
            context=all_ctx,
            filter=f"secret_id == {secret_id}",
        )
        assert len(all_logs) >= 1, f"Secret should be mirrored to {all_ctx}"


@_handle_project
def test_user_field_injected():
    """Logs should have _user field set to user name."""
    test_user_name = "TestUserName"

    with patch(
        "unity.common.log_utils._get_user_context",
        return_value=test_user_name,
    ):
        sm = SecretManager()
        result = sm._create_secret(
            name="user_test_secret",
            value="test-value",
            description="Testing user field",
        )
        assert result["outcome"] == "secret created"

        secrets = sm._filter_secrets(filter="name == 'user_test_secret'")
        assert len(secrets) >= 1
        secret_id = secrets[0].secret_id

        log = _get_raw_log_by_secret_id(sm._ctx, secret_id)
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
        sm = SecretManager()
        result = sm._create_secret(
            name="assistant_test_secret",
            value="test-value",
            description="Testing assistant field",
        )
        assert result["outcome"] == "secret created"

        secrets = sm._filter_secrets(filter="name == 'assistant_test_secret'")
        assert len(secrets) >= 1
        secret_id = secrets[0].secret_id

        log = _get_raw_log_by_secret_id(sm._ctx, secret_id)
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
        sm = SecretManager()
        result = sm._create_secret(
            name="assistant_id_test_secret",
            value="test-value",
            description="Testing assistant ID field",
        )
        assert result["outcome"] == "secret created"

        secrets = sm._filter_secrets(filter="name == 'assistant_id_test_secret'")
        assert len(secrets) >= 1
        secret_id = secrets[0].secret_id

        log = _get_raw_log_by_secret_id(sm._ctx, secret_id)
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
        sm = SecretManager()
        result = sm._create_secret(
            name="user_id_test_secret",
            value="test-value",
            description="Testing user ID field",
        )
        assert result["outcome"] == "secret created"

        secrets = sm._filter_secrets(filter="name == 'user_id_test_secret'")
        assert len(secrets) >= 1
        secret_id = secrets[0].secret_id

        log = _get_raw_log_by_secret_id(sm._ctx, secret_id)
        assert log is not None, "Log should exist"

        entries = log.entries
        assert (
            entries.get("_user_id") == test_user_id
        ), f"_user_id should be '{test_user_id}', got {entries.get('_user_id')}"


@_handle_project
def test_all_contexts_created_on_provision():
    """Aggregation contexts should be created when SecretManager provisions storage."""
    # SecretManager provisions storage via ContextRegistry.get_context() in __init__
    sm = SecretManager()

    # Derive the expected aggregation contexts
    all_ctxs = _derive_all_contexts(sm._ctx)
    assert len(all_ctxs) == 2, "Should have user-level and global aggregation contexts"

    # Verify both aggregation contexts exist
    contexts = unify.get_contexts()
    for all_ctx in all_ctxs:
        assert all_ctx in contexts, f"{all_ctx} context should be created"


@_handle_project
def test_private_fields_excluded_from_filter_secrets():
    """Private fields should be excluded when reading secrets via public API."""
    sm = SecretManager()

    result = sm._create_secret(
        name="private_test_secret",
        value="test-value",
        description="Testing private field exclusion",
    )
    assert result["outcome"] == "secret created"

    # Get secret via _filter_secrets API
    secrets = sm._filter_secrets(filter="name == 'private_test_secret'")
    assert len(secrets) == 1

    secret = secrets[0]
    # Private fields should NOT be in the Secret model (they're excluded on read)
    assert not hasattr(secret, "_user"), "_user should not be exposed"
    assert not hasattr(secret, "_user_id"), "_user_id should not be exposed"
    assert not hasattr(secret, "_assistant"), "_assistant should not be exposed"
    assert not hasattr(secret, "_assistant_id"), "_assistant_id should not be exposed"


@_handle_project
def test_deleting_secret_removes_from_all_ctxs():
    """Deleting a secret should also remove it from all aggregation contexts."""
    sm = SecretManager()

    # Create a secret
    result = sm._create_secret(
        name="delete_test_secret",
        value="test-value",
        description="Secret to be deleted",
    )
    assert result["outcome"] == "secret created"

    secrets = sm._filter_secrets(filter="name == 'delete_test_secret'")
    assert len(secrets) >= 1
    secret_id = secrets[0].secret_id

    # Derive the aggregation contexts
    all_ctxs = _derive_all_contexts(sm._ctx)
    assert len(all_ctxs) == 2, "Should have user-level and global aggregation contexts"

    # Verify it exists in all aggregation contexts before deletion
    for all_ctx in all_ctxs:
        all_logs_before = unify.get_logs(
            context=all_ctx,
            filter=f"secret_id == {secret_id}",
        )
        assert (
            len(all_logs_before) >= 1
        ), f"Secret should exist in {all_ctx} before deletion"

    # Delete the secret
    sm._delete_secret(name="delete_test_secret")

    # Verify it's removed from all aggregation contexts after deletion
    for all_ctx in all_ctxs:
        all_logs_after = unify.get_logs(
            context=all_ctx,
            filter=f"secret_id == {secret_id}",
        )
        assert (
            len(all_logs_after) == 0
        ), f"Secret should be removed from {all_ctx} after deletion"


@_handle_project
def test_update_syncs_to_all_aggregation_contexts():
    """Updating a secret should be immediately visible in all aggregation contexts."""
    sm = SecretManager()

    # Create a secret with initial values
    result = sm._create_secret(
        name="update_sync_secret",
        value="test-value",
        description="Original description",
    )
    assert result["outcome"] == "secret created"

    secrets = sm._filter_secrets(filter="name == 'update_sync_secret'")
    assert len(secrets) >= 1
    secret_id = secrets[0].secret_id

    # Derive aggregation contexts
    all_ctxs = _derive_all_contexts(sm._ctx)
    assert len(all_ctxs) == 2, "Should have user-level and global aggregation contexts"

    # Verify initial description in all contexts
    for ctx in [sm._ctx, *all_ctxs]:
        log = _get_raw_log_by_secret_id(ctx, secret_id)
        assert log is not None, f"Log should exist in {ctx}"
        assert (
            log.entries.get("description") == "Original description"
        ), f"Initial description in {ctx}"

    # Update the secret's description
    sm._update_secret(name="update_sync_secret", description="Updated description")

    # Verify the update is immediately visible in ALL contexts (primary + aggregations)
    for ctx in [sm._ctx, *all_ctxs]:
        log = _get_raw_log_by_secret_id(ctx, secret_id)
        assert log is not None, f"Log should exist in {ctx} after update"
        assert log.entries.get("description") == "Updated description", (
            f"Updated description should be visible in {ctx}. "
            f"Expected 'Updated description', got '{log.entries.get('description')}'"
        )


@_handle_project
def test_log_id_unchanged_after_update():
    """Updates should modify the existing log entry, not create a new one."""
    sm = SecretManager()

    # Create a secret
    result = sm._create_secret(
        name="log_id_test_secret",
        value="test-value",
        description="Before update",
    )
    assert result["outcome"] == "secret created"

    secrets = sm._filter_secrets(filter="name == 'log_id_test_secret'")
    assert len(secrets) >= 1
    secret_id = secrets[0].secret_id

    # Get the original log ID
    original_log = _get_raw_log_by_secret_id(sm._ctx, secret_id)
    original_log_id = original_log.id

    # Update the secret
    sm._update_secret(name="log_id_test_secret", description="After update")

    # Verify the log ID is unchanged (in-place update, not delete+create)
    updated_log = _get_raw_log_by_secret_id(sm._ctx, secret_id)
    assert updated_log.id == original_log_id, (
        f"Log ID should be unchanged after update. "
        f"Original: {original_log_id}, After update: {updated_log.id}"
    )

    # Verify all aggregation contexts still reference the same log ID
    all_ctxs = _derive_all_contexts(sm._ctx)
    for all_ctx in all_ctxs:
        agg_log = _get_raw_log_by_secret_id(all_ctx, secret_id)
        assert agg_log.id == original_log_id, (
            f"Aggregation context {all_ctx} should still reference the same log. "
            f"Expected {original_log_id}, got {agg_log.id}"
        )
