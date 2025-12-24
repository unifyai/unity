"""Tests for aggregation context mirroring and private field injection."""

from __future__ import annotations

from unittest.mock import patch

import unify
from tests.helpers import _handle_project
from unity.common.log_utils import _derive_all_contexts
from unity.task_scheduler.task_scheduler import TaskScheduler


def _get_raw_log_by_task_id(ctx: str, task_id: int):
    """Get raw log entry including private fields."""
    logs = unify.get_logs(
        context=ctx,
        filter=f"task_id == {task_id}",
        limit=1,
    )
    return logs[0] if logs else None


@_handle_project
def test_log_creates_all_tasks_entries():
    """Creating a task should mirror to both aggregation contexts."""
    ts = TaskScheduler()

    # Create a task
    result = ts._create_task(name="Test Task AllCtx", description="Test description")
    task_id = result["details"]["task_id"]

    # Verify it exists in the manager's context
    tasks = ts._filter_tasks(filter=f"task_id == {task_id}")
    assert len(tasks) == 1, "Task should exist in manager's context"

    # Derive both aggregation contexts from the manager's context
    all_ctxs = _derive_all_contexts(ts._ctx)
    assert len(all_ctxs) == 2, "Should have user-level and global aggregation contexts"

    # Verify it was mirrored to both aggregation contexts
    for all_ctx in all_ctxs:
        all_logs = unify.get_logs(
            context=all_ctx,
            filter=f"task_id == {task_id}",
        )
        assert len(all_logs) >= 1, f"Task should be mirrored to {all_ctx}"


@_handle_project
def test_user_field_injected():
    """Logs should have _user field set to user name."""
    test_user_name = "TestUserName"

    with patch(
        "unity.common.log_utils._get_user_name",
        return_value=test_user_name,
    ):
        ts = TaskScheduler()
        result = ts._create_task(name="UserTest Task", description="Test")
        task_id = result["details"]["task_id"]

        log = _get_raw_log_by_task_id(ts._ctx, task_id)
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
        ts = TaskScheduler()
        result = ts._create_task(name="AssistantTest Task", description="Test")
        task_id = result["details"]["task_id"]

        log = _get_raw_log_by_task_id(ts._ctx, task_id)
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
        ts = TaskScheduler()
        result = ts._create_task(name="AssistantIdTest Task", description="Test")
        task_id = result["details"]["task_id"]

        log = _get_raw_log_by_task_id(ts._ctx, task_id)
        assert log is not None, "Log should exist"

        entries = log.entries
        assert (
            entries.get("_assistant_id") == test_assistant_id
        ), f"_assistant_id should be '{test_assistant_id}', got {entries.get('_assistant_id')}"


@_handle_project
def test_user_id_field_injected():
    """Logs should have _user_id field set to user's id."""
    test_user_id = "test-user-456"

    with patch(
        "unity.common.log_utils._get_user_id",
        return_value=test_user_id,
    ):
        ts = TaskScheduler()
        result = ts._create_task(name="UserIdTest Task", description="Test")
        task_id = result["details"]["task_id"]

        log = _get_raw_log_by_task_id(ts._ctx, task_id)
        assert log is not None, "Log should exist"

        entries = log.entries
        assert (
            entries.get("_user_id") == test_user_id
        ), f"_user_id should be '{test_user_id}', got {entries.get('_user_id')}"


@_handle_project
def test_all_contexts_created_on_provision():
    """Aggregation contexts should be created when TaskScheduler provisions storage."""
    # TaskScheduler provisions storage via ContextRegistry.get_context() in __init__
    ts = TaskScheduler()

    # Derive the expected aggregation contexts
    all_ctxs = _derive_all_contexts(ts._ctx)
    assert len(all_ctxs) == 2, "Should have user-level and global aggregation contexts"

    # Verify both aggregation contexts exist
    contexts = unify.get_contexts()
    for all_ctx in all_ctxs:
        assert all_ctx in contexts, f"{all_ctx} context should be created"


@_handle_project
def test_private_fields_excluded_from_filter_tasks():
    """Private fields should be excluded when reading tasks via internal filter."""
    ts = TaskScheduler()

    result = ts._create_task(name="PrivateTest Task", description="Exclusion test")
    task_id = result["details"]["task_id"]

    # Get task via internal _filter_tasks (which uses exclude_fields)
    tasks = ts._filter_tasks(filter=f"task_id == {task_id}")
    assert len(tasks) == 1

    task = tasks[0]
    # Private fields should NOT be in the Task model (they're excluded on read)
    assert not hasattr(task, "_user"), "_user should not be exposed"
    assert not hasattr(task, "_user_id"), "_user_id should not be exposed"
    assert not hasattr(task, "_assistant"), "_assistant should not be exposed"
    assert not hasattr(task, "_assistant_id"), "_assistant_id should not be exposed"


@_handle_project
def test_create_many_mirrors_to_all_ctxs():
    """Batch-created tasks should also be mirrored to both aggregation contexts."""
    test_assistant_name = "BatchTestAssistant"

    with patch(
        "unity.common.log_utils._get_assistant_name",
        return_value=test_assistant_name,
    ):
        ts = TaskScheduler()

        # Create multiple tasks at once
        result1 = ts._create_task(name="Batch Task 1", description="First batch task")
        result2 = ts._create_task(name="Batch Task 2", description="Second batch task")

        task_id_1 = result1["details"]["task_id"]
        task_id_2 = result2["details"]["task_id"]

        # Derive both aggregation contexts
        all_ctxs = _derive_all_contexts(ts._ctx)
        assert len(all_ctxs) == 2

        # Verify both tasks were mirrored to both contexts
        for all_ctx in all_ctxs:
            all_logs = unify.get_logs(
                context=all_ctx,
                filter=f"task_id in [{task_id_1}, {task_id_2}]",
            )
            assert len(all_logs) >= 2, f"Both tasks should be mirrored to {all_ctx}"


@_handle_project
def test_deleting_task_removes_from_all_ctxs():
    """Deleting a task should also remove it from all aggregation contexts."""
    ts = TaskScheduler()

    # Create a task
    result = ts._create_task(name="DeleteTest Task", description="Task to be deleted")
    task_id = result["details"]["task_id"]

    # Derive the aggregation contexts
    all_ctxs = _derive_all_contexts(ts._ctx)
    assert len(all_ctxs) == 2, "Should have user-level and global aggregation contexts"

    # Verify it exists in all aggregation contexts before deletion
    for all_ctx in all_ctxs:
        all_logs_before = unify.get_logs(
            context=all_ctx,
            filter=f"task_id == {task_id}",
        )
        assert (
            len(all_logs_before) >= 1
        ), f"Task should exist in {all_ctx} before deletion"

    # Delete the task
    ts._delete_task(task_id=task_id)

    # Verify it's removed from all aggregation contexts after deletion
    for all_ctx in all_ctxs:
        all_logs_after = unify.get_logs(
            context=all_ctx,
            filter=f"task_id == {task_id}",
        )
        assert (
            len(all_logs_after) == 0
        ), f"Task should be removed from {all_ctx} after deletion"


@_handle_project
def test_update_syncs_to_all_aggregation_contexts():
    """Updating a task should be immediately visible in all aggregation contexts."""
    ts = TaskScheduler()

    # Create a task with initial values
    result = ts._create_task(name="UpdateSync Task", description="Original description")
    task_id = result["details"]["task_id"]

    # Derive aggregation contexts
    all_ctxs = _derive_all_contexts(ts._ctx)
    assert len(all_ctxs) == 2, "Should have user-level and global aggregation contexts"

    # Verify initial description in all contexts
    for ctx in [ts._ctx, *all_ctxs]:
        log = _get_raw_log_by_task_id(ctx, task_id)
        assert log is not None, f"Log should exist in {ctx}"
        assert (
            log.entries.get("description") == "Original description"
        ), f"Initial description in {ctx}"

    # Update the task's description
    ts._update_task(task_id=task_id, description="Updated description")

    # Verify the update is immediately visible in ALL contexts (primary + aggregations)
    for ctx in [ts._ctx, *all_ctxs]:
        log = _get_raw_log_by_task_id(ctx, task_id)
        assert log is not None, f"Log should exist in {ctx} after update"
        assert log.entries.get("description") == "Updated description", (
            f"Updated description should be visible in {ctx}. "
            f"Expected 'Updated description', got '{log.entries.get('description')}'"
        )


@_handle_project
def test_log_id_unchanged_after_update():
    """Updates should modify the existing log entry, not create a new one."""
    ts = TaskScheduler()

    # Create a task
    result = ts._create_task(name="LogIdTest Task", description="Before update")
    task_id = result["details"]["task_id"]

    # Get the original log ID
    original_log = _get_raw_log_by_task_id(ts._ctx, task_id)
    original_log_id = original_log.id

    # Update the task
    ts._update_task(task_id=task_id, description="After update")

    # Verify the log ID is unchanged (in-place update, not delete+create)
    updated_log = _get_raw_log_by_task_id(ts._ctx, task_id)
    assert updated_log.id == original_log_id, (
        f"Log ID should be unchanged after update. "
        f"Original: {original_log_id}, After update: {updated_log.id}"
    )

    # Verify all aggregation contexts still reference the same log ID
    all_ctxs = _derive_all_contexts(ts._ctx)
    for all_ctx in all_ctxs:
        agg_log = _get_raw_log_by_task_id(all_ctx, task_id)
        assert agg_log.id == original_log_id, (
            f"Aggregation context {all_ctx} should still reference the same log. "
            f"Expected {original_log_id}, got {agg_log.id}"
        )
