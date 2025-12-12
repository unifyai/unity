"""Tests for aggregation context mirroring and private field injection for FunctionManager."""

from __future__ import annotations

import os
from unittest.mock import patch

import unify
from tests.helpers import _handle_project
from unity.common.log_utils import _derive_all_contexts
from unity.function_manager.function_manager import FunctionManager


def _get_raw_log_by_function_id(ctx: str, function_id: int):
    """Get raw log entry including private fields."""
    logs = unify.get_logs(
        context=ctx,
        filter=f"function_id == {function_id}",
        limit=1,
    )
    return logs[0] if logs else None


@_handle_project
def test_log_creates_all_compositional_entries():
    """Creating a function should mirror to both aggregation contexts."""
    fm = FunctionManager()

    # Add a simple function
    src = "def test_dual_ctx(x):\n    return x * 2\n"
    result = fm.add_functions(implementations=src)
    assert result == {"test_dual_ctx": "added"}

    # Get the function_id
    listing = fm.list_functions()
    assert "test_dual_ctx" in listing
    function_id = listing["test_dual_ctx"]["function_id"]

    # Derive both aggregation contexts from the compositional context
    all_ctxs = _derive_all_contexts(fm._compositional_ctx)
    assert len(all_ctxs) == 2, "Should have user-level and global aggregation contexts"

    # Verify it was mirrored to both aggregation contexts
    for all_ctx in all_ctxs:
        all_logs = unify.get_logs(
            context=all_ctx,
            filter=f"function_id == {function_id}",
        )
        assert len(all_logs) >= 1, f"Function should be mirrored to {all_ctx}"


@_handle_project
def test_user_field_injected():
    """Logs should have _user field set to user name."""
    test_user_name = "TestUserName"

    with patch(
        "unity.common.log_utils._get_user_name",
        return_value=test_user_name,
    ):
        fm = FunctionManager()

        src = "def test_user_field(x):\n    return x + 1\n"
        result = fm.add_functions(implementations=src)
        assert result == {"test_user_field": "added"}

        listing = fm.list_functions()
        function_id = listing["test_user_field"]["function_id"]

        log = _get_raw_log_by_function_id(fm._compositional_ctx, function_id)
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
        fm = FunctionManager()

        src = "def test_assistant_field(x):\n    return x + 1\n"
        result = fm.add_functions(implementations=src)
        assert result == {"test_assistant_field": "added"}

        listing = fm.list_functions()
        function_id = listing["test_assistant_field"]["function_id"]

        log = _get_raw_log_by_function_id(fm._compositional_ctx, function_id)
        assert log is not None, "Log should exist"

        entries = log.entries
        assert (
            entries.get("_assistant") == test_assistant_name
        ), f"_assistant should be '{test_assistant_name}', got {entries.get('_assistant')}"


@_handle_project
def test_assistant_id_field_injected():
    """Logs should have _assistant_id field when ASSISTANT is available."""
    test_assistant_id = "test-agent-789"

    with patch(
        "unity.common.log_utils._get_assistant_id",
        return_value=test_assistant_id,
    ):
        fm = FunctionManager()

        src = "def test_assistant_id_field(x):\n    return x - 1\n"
        result = fm.add_functions(implementations=src)
        assert result == {"test_assistant_id_field": "added"}

        listing = fm.list_functions()
        function_id = listing["test_assistant_id_field"]["function_id"]

        log = _get_raw_log_by_function_id(fm._compositional_ctx, function_id)
        assert log is not None, "Log should exist"

        entries = log.entries
        assert "_assistant_id" in entries, "_assistant_id field should be present"
        assert (
            entries.get("_assistant_id") == test_assistant_id
        ), f"_assistant_id should be '{test_assistant_id}', got {entries.get('_assistant_id')}"


@_handle_project
def test_user_id_field_injected():
    """Logs should have _user_id field when USER_ID env var is set."""
    test_user_id = "user-456"

    with patch.dict(os.environ, {"USER_ID": test_user_id}):
        fm = FunctionManager()

        src = "def test_user_id_field(x):\n    return x * x\n"
        result = fm.add_functions(implementations=src)
        assert result == {"test_user_id_field": "added"}

        listing = fm.list_functions()
        function_id = listing["test_user_id_field"]["function_id"]

        log = _get_raw_log_by_function_id(fm._compositional_ctx, function_id)
        assert log is not None, "Log should exist"

        entries = log.entries
        assert "_user_id" in entries, "_user_id field should be present"
        assert (
            entries.get("_user_id") == test_user_id
        ), f"_user_id should be '{test_user_id}', got {entries.get('_user_id')}"


@_handle_project
def test_all_contexts_created_on_provision():
    """Aggregation contexts should exist after manager instantiation."""
    fm = FunctionManager()

    # Derive the aggregation contexts
    all_ctxs = _derive_all_contexts(fm._compositional_ctx)
    assert len(all_ctxs) == 2, "Should have user-level and global aggregation contexts"

    # Verify both contexts exist by trying to query them
    for all_ctx in all_ctxs:
        try:
            unify.get_logs(context=all_ctx, limit=1)
            context_exists = True
        except Exception:
            context_exists = False
        assert context_exists, f"Context {all_ctx} should exist after manager init"


@_handle_project
def test_private_fields_excluded_from_list_functions():
    """Private fields should not be exposed when listing functions."""
    test_assistant_name = "HiddenAssistant"

    with patch(
        "unity.common.log_utils._get_assistant_name",
        return_value=test_assistant_name,
    ):
        fm = FunctionManager()

        src = "def test_private_hidden(x):\n    return x / 2\n"
        fm.add_functions(implementations=src)

        listing = fm.list_functions()
        func_data = listing.get("test_private_hidden", {})

        # Private fields should not be in the listing
        assert "_user" not in func_data, "_user should not be in listing"
        assert "_user_id" not in func_data, "_user_id should not be in listing"
        assert "_assistant" not in func_data, "_assistant should not be in listing"
        assert (
            "_assistant_id" not in func_data
        ), "_assistant_id should not be in listing"


@_handle_project
def test_batch_create_mirrors_to_all_ctxs():
    """Batch-created functions should all be mirrored to both aggregation contexts."""
    fm = FunctionManager()

    # Add multiple functions at once (triggers batch creation)
    sources = [
        "def batch_func_a(x):\n    return x + 1\n",
        "def batch_func_b(y):\n    return y + 2\n",
        "def batch_func_c(z):\n    return z + 3\n",
    ]
    result = fm.add_functions(implementations=sources)
    assert result == {
        "batch_func_a": "added",
        "batch_func_b": "added",
        "batch_func_c": "added",
    }

    listing = fm.list_functions()

    # Derive both aggregation contexts
    all_ctxs = _derive_all_contexts(fm._compositional_ctx)
    assert len(all_ctxs) == 2

    # Verify all functions were mirrored to both contexts
    for func_name in ["batch_func_a", "batch_func_b", "batch_func_c"]:
        function_id = listing[func_name]["function_id"]
        for all_ctx in all_ctxs:
            all_logs = unify.get_logs(
                context=all_ctx,
                filter=f"function_id == {function_id}",
            )
            assert (
                len(all_logs) >= 1
            ), f"Function {func_name} should be mirrored to {all_ctx}"


@_handle_project
def test_deleting_function_removes_from_all_ctxs():
    """Deleting a function should also remove it from all aggregation contexts."""
    fm = FunctionManager()

    # Create a function
    src = "def delete_test_func(x):\n    return x\n"
    result = fm.add_functions(implementations=src)
    assert result == {"delete_test_func": "added"}

    listing = fm.list_functions()
    function_id = listing["delete_test_func"]["function_id"]

    # Derive the aggregation contexts
    all_ctxs = _derive_all_contexts(fm._compositional_ctx)
    assert len(all_ctxs) == 2, "Should have user-level and global aggregation contexts"

    # Verify it exists in all aggregation contexts before deletion
    for all_ctx in all_ctxs:
        all_logs_before = unify.get_logs(
            context=all_ctx,
            filter=f"function_id == {function_id}",
        )
        assert (
            len(all_logs_before) >= 1
        ), f"Function should exist in {all_ctx} before deletion"

    # Delete the function
    fm.delete_function(function_id=function_id)

    # Verify it's removed from all aggregation contexts after deletion
    for all_ctx in all_ctxs:
        all_logs_after = unify.get_logs(
            context=all_ctx,
            filter=f"function_id == {function_id}",
        )
        assert (
            len(all_logs_after) == 0
        ), f"Function should be removed from {all_ctx} after deletion"
