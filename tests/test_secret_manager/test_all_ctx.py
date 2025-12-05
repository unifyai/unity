"""Tests for All/Secrets context mirroring and private field injection."""

from __future__ import annotations

import os
from unittest.mock import patch

import unify
from tests.helpers import _handle_project
from unity.common.log_utils import _derive_all_context
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
def test_log_creates_all_secrets_entry():
    """Creating a secret should mirror to All/<Ctx>."""
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

    # Derive the All/<Ctx> context from the manager's context
    all_ctx = _derive_all_context(sm._ctx)
    assert all_ctx is not None, "All context should be derivable"

    # Verify it was mirrored to All/<Ctx>
    all_logs = unify.get_logs(
        context=all_ctx,
        filter=f"secret_id == {secret_id}",
    )
    assert len(all_logs) >= 1, f"Secret should be mirrored to {all_ctx}"


@_handle_project
def test_assistant_field_injected():
    """Logs should have _assistant field set to assistant name."""
    test_assistant_name = "TestAssistantName"

    with patch(
        "unity.common.log_utils._get_assistant_name",
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
    """Logs should have _user_id field when USER_ID env is set."""
    test_user_id = "test-user-456"

    with patch.dict(os.environ, {"USER_ID": test_user_id}):
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
def test_all_context_created_on_provision():
    """All/<Ctx> context should be created when SecretManager provisions storage."""
    # SecretManager provisions storage via ContextRegistry.get_context() in __init__
    sm = SecretManager()

    # Derive the expected All/<Ctx> context
    all_ctx = _derive_all_context(sm._ctx)
    assert all_ctx is not None, "All context should be derivable"

    # Verify All/<Ctx> exists
    contexts = unify.get_contexts()
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
    assert not hasattr(secret, "_assistant"), "_assistant should not be exposed"
    assert not hasattr(secret, "_assistant_id"), "_assistant_id should not be exposed"
    assert not hasattr(secret, "_user_id"), "_user_id should not be exposed"


@_handle_project
def test_deleting_secret_removes_from_all_ctx():
    """Deleting a secret should also remove it from All/<Ctx>."""
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

    # Derive the All/<Ctx> context
    all_ctx = _derive_all_context(sm._ctx)
    assert all_ctx is not None, "All context should be derivable"

    # Verify it exists in All/<Ctx> before deletion
    all_logs_before = unify.get_logs(
        context=all_ctx,
        filter=f"secret_id == {secret_id}",
    )
    assert len(all_logs_before) >= 1, "Secret should exist in All/<Ctx> before deletion"

    # Delete the secret
    sm._delete_secret(name="delete_test_secret")

    # Verify it's removed from All/<Ctx> after deletion
    all_logs_after = unify.get_logs(
        context=all_ctx,
        filter=f"secret_id == {secret_id}",
    )
    assert (
        len(all_logs_after) == 0
    ), "Secret should be removed from All/<Ctx> after deletion"
