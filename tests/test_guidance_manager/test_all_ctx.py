"""Tests for All/Guidance context mirroring and private field injection."""

from __future__ import annotations

import os
from unittest.mock import patch

import unify
from tests.helpers import _handle_project
from unity.common.log_utils import _derive_all_context
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
def test_log_creates_all_guidance_entry():
    """Creating a guidance entry should mirror to All/<Ctx>."""
    gm = GuidanceManager()

    # Create a guidance entry
    result = gm._add_guidance(
        title="Test Guidance",
        content="Test guidance content for All/Ctx",
    )
    guidance_id = result["details"]["guidance_id"]

    # Verify it exists in the manager's context
    guidance = gm._filter(filter=f"guidance_id == {guidance_id}")
    assert len(guidance) == 1, "Guidance should exist in manager's context"

    # Derive the All/<Ctx> context from the manager's context
    all_ctx = _derive_all_context(gm._ctx)
    assert all_ctx is not None, "All context should be derivable"

    # Verify it was mirrored to All/<Ctx>
    all_logs = unify.get_logs(
        context=all_ctx,
        filter=f"guidance_id == {guidance_id}",
    )
    assert len(all_logs) >= 1, f"Guidance should be mirrored to {all_ctx}"


@_handle_project
def test_assistant_field_injected():
    """Logs should have _assistant field set to assistant name."""
    test_assistant_name = "TestAssistantName"

    with patch(
        "unity.common.log_utils._get_assistant_name",
        return_value=test_assistant_name,
    ):
        gm = GuidanceManager()
        result = gm._add_guidance(
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
        result = gm._add_guidance(
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
    """Logs should have _user_id field when USER_ID env is set."""
    test_user_id = "test-user-456"

    with patch.dict(os.environ, {"USER_ID": test_user_id}):
        gm = GuidanceManager()
        result = gm._add_guidance(
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
def test_all_context_created_on_provision():
    """All/<Ctx> context should be created when GuidanceManager provisions storage."""
    # GuidanceManager provisions storage via ContextRegistry.get_context() in __init__
    gm = GuidanceManager()

    # Derive the expected All/<Ctx> context
    all_ctx = _derive_all_context(gm._ctx)
    assert all_ctx is not None, "All context should be derivable"

    # Verify All/<Ctx> exists
    contexts = unify.get_contexts()
    assert all_ctx in contexts, f"{all_ctx} context should be created"


@_handle_project
def test_private_fields_excluded_from_filter():
    """Private fields should be excluded when reading guidance via public API."""
    gm = GuidanceManager()

    result = gm._add_guidance(
        title="Private Field Test",
        content="Testing private field exclusion",
    )
    guidance_id = result["details"]["guidance_id"]

    # Get guidance via _filter API
    guidance_list = gm._filter(filter=f"guidance_id == {guidance_id}")
    assert len(guidance_list) == 1

    guidance = guidance_list[0]
    # Private fields should NOT be in the Guidance model (they're excluded on read)
    assert not hasattr(guidance, "_assistant"), "_assistant should not be exposed"
    assert not hasattr(guidance, "_assistant_id"), "_assistant_id should not be exposed"
    assert not hasattr(guidance, "_user_id"), "_user_id should not be exposed"


@_handle_project
def test_deleting_guidance_removes_from_all_ctx():
    """Deleting a guidance entry should also remove it from All/<Ctx>."""
    gm = GuidanceManager()

    # Create a guidance entry
    result = gm._add_guidance(
        title="Delete Test Guidance",
        content="Guidance to be deleted",
    )
    guidance_id = result["details"]["guidance_id"]

    # Derive the All/<Ctx> context
    all_ctx = _derive_all_context(gm._ctx)
    assert all_ctx is not None, "All context should be derivable"

    # Verify it exists in All/<Ctx> before deletion
    all_logs_before = unify.get_logs(
        context=all_ctx,
        filter=f"guidance_id == {guidance_id}",
    )
    assert (
        len(all_logs_before) >= 1
    ), "Guidance should exist in All/<Ctx> before deletion"

    # Delete the guidance
    gm._delete_guidance(guidance_id=guidance_id)

    # Verify it's removed from All/<Ctx> after deletion
    all_logs_after = unify.get_logs(
        context=all_ctx,
        filter=f"guidance_id == {guidance_id}",
    )
    assert (
        len(all_logs_after) == 0
    ), "Guidance should be removed from All/<Ctx> after deletion"
