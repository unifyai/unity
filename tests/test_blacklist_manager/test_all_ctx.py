"""Tests for All/BlackList context mirroring and private field injection."""

from __future__ import annotations

import os
from unittest.mock import patch

import unify
from tests.helpers import _handle_project
from unity.common.log_utils import _derive_all_context
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
def test_log_creates_all_blacklist_entry():
    """Creating a blacklist entry should mirror to All/<Ctx>."""
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

    # Derive the All/<Ctx> context from the manager's context
    all_ctx = _derive_all_context(bm._ctx)
    assert all_ctx is not None, "All context should be derivable"

    # Verify it was mirrored to All/<Ctx>
    all_logs = unify.get_logs(
        context=all_ctx,
        filter=f"blacklist_id == {blacklist_id}",
    )
    assert len(all_logs) >= 1, f"Blacklist entry should be mirrored to {all_ctx}"


@_handle_project
def test_assistant_field_injected():
    """Logs should have _assistant field set to assistant name."""
    test_assistant_name = "TestAssistantName"

    with patch(
        "unity.common.log_utils._get_assistant_name",
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
            medium="whatsapp_message",
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
    """Logs should have _user_id field when USER_ID env is set."""
    test_user_id = "test-user-456"

    with patch.dict(os.environ, {"USER_ID": test_user_id}):
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
def test_all_context_created_on_provision():
    """All/<Ctx> context should be created when BlackListManager provisions storage."""
    # BlackListManager provisions storage via ContextRegistry.get_context() in __init__
    bm = BlackListManager()

    # Derive the expected All/<Ctx> context
    all_ctx = _derive_all_context(bm._ctx)
    assert all_ctx is not None, "All context should be derivable"

    # Verify All/<Ctx> exists
    contexts = unify.get_contexts()
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
    assert not hasattr(entry, "_assistant"), "_assistant should not be exposed"
    assert not hasattr(entry, "_assistant_id"), "_assistant_id should not be exposed"
    assert not hasattr(entry, "_user_id"), "_user_id should not be exposed"


@_handle_project
def test_deleting_blacklist_entry_removes_from_all_ctx():
    """Deleting a blacklist entry should also remove it from All/<Ctx>."""
    bm = BlackListManager()

    # Create a blacklist entry
    result = bm.create_blacklist_entry(
        medium="email",
        contact_detail="delete-test@example.com",
        reason="Entry to be deleted",
    )
    assert result["outcome"] == "blacklist entry created"
    blacklist_id = result["details"]["blacklist_id"]

    # Derive the All/<Ctx> context
    all_ctx = _derive_all_context(bm._ctx)
    assert all_ctx is not None, "All context should be derivable"

    # Verify it exists in All/<Ctx> before deletion
    all_logs_before = unify.get_logs(
        context=all_ctx,
        filter=f"blacklist_id == {blacklist_id}",
    )
    assert (
        len(all_logs_before) >= 1
    ), "Blacklist entry should exist in All/<Ctx> before deletion"

    # Delete the blacklist entry
    bm.delete_blacklist_entry(blacklist_id=blacklist_id)

    # Verify it's removed from All/<Ctx> after deletion
    all_logs_after = unify.get_logs(
        context=all_ctx,
        filter=f"blacklist_id == {blacklist_id}",
    )
    assert (
        len(all_logs_after) == 0
    ), "Blacklist entry should be removed from All/<Ctx> after deletion"
