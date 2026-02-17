"""
Foreign Key Tests for TaskScheduler

Coverage
========
✓ entrypoint → Functions.function_id (direct FK)
  - Valid reference creation
  - SET NULL: Task survives function deletion with entrypoint=null
  - Read after SET NULL without breaking
  - Write with None
  - Clone after SET NULL

Note: schedule.next_task and schedule.prev_task FKs are not yet implemented,
      so related tests are commented out.
"""

from __future__ import annotations

import unify
from tests.helpers import _handle_project
from unity.function_manager.function_manager import FunctionManager
from unity.task_scheduler.task_scheduler import TaskScheduler

# --------------------------------------------------------------------------- #
#  Unit Tests: entrypoint → Functions.function_id                            #
# --------------------------------------------------------------------------- #


@_handle_project
def test_entrypoint_valid_reference():
    """Test that tasks can reference valid function IDs as entrypoint."""
    fm = FunctionManager()
    ts = TaskScheduler()

    # Create function
    src = "def process_data():\n    return 'processed'\n"
    fm.add_functions(implementations=src)

    # Get function ID
    funcs = unify.get_logs(
        context=fm._compositional_ctx,
        from_fields=["function_id", "name"],
    )
    assert len(funcs) == 1
    func_id = int(funcs[0].entries["function_id"])

    # Create task with valid entrypoint
    ts._create_task(
        name="Process Task",
        description="Task that processes data",
        entrypoint=func_id,
    )

    # Verify task was created with entrypoint
    tasks = unify.get_logs(
        context=ts._ctx,
        from_fields=["task_id", "name", "entrypoint"],
    )
    assert len(tasks) == 1
    assert tasks[0].entries["entrypoint"] == func_id


@_handle_project
def test_entrypoint_set_null_on_delete():
    """Test SET NULL: Task survives function deletion with entrypoint becoming null."""
    fm = FunctionManager()
    ts = TaskScheduler()

    # Create function
    src = "def my_func():\n    return 'result'\n"
    fm.add_functions(implementations=src)

    # Get function ID
    funcs = unify.get_logs(context=fm._compositional_ctx, from_fields=["function_id"])
    func_id = int(funcs[0].entries["function_id"])

    # Create task with this function as entrypoint
    ts._create_task(
        name="Task 1",
        description="Task with entrypoint",
        entrypoint=func_id,
    )

    # Verify task has entrypoint
    tasks = unify.get_logs(context=ts._ctx, from_fields=["task_id", "entrypoint"])
    assert len(tasks) == 1
    assert tasks[0].entries["entrypoint"] == func_id

    # Delete the function
    fm.delete_function(function_id=func_id)

    # Verify task still exists but entrypoint is null (SET NULL behavior)
    tasks_after = unify.get_logs(
        context=ts._ctx,
        from_fields=["task_id", "name", "entrypoint"],
    )
    assert len(tasks_after) == 1  # Task still exists
    assert tasks_after[0].entries["name"] == "Task 1"
    assert tasks_after[0].entries.get("entrypoint") is None  # Entrypoint nulled


@_handle_project
def test_entrypoint_null_does_not_break_scheduler_init():
    """Test that tasks with null entrypoint (after FK SET NULL) load without errors."""
    fm = FunctionManager()
    ts = TaskScheduler()

    # Create function
    src = "def my_function():\n    return 'result'\n"
    fm.add_functions(implementations=src)

    # Get function ID
    funcs = unify.get_logs(context=fm._compositional_ctx, from_fields=["function_id"])
    func_id = int(funcs[0].entries["function_id"])

    # Create task with entrypoint
    result = ts._create_task(
        name="Task with Entrypoint",
        description="Task that will lose its entrypoint",
        entrypoint=func_id,
    )
    tid = result["details"]["task_id"]

    # Verify task has entrypoint
    tasks = unify.get_logs(context=ts._ctx, from_fields=["task_id", "entrypoint"])
    assert len(tasks) == 1
    assert tasks[0].entries["entrypoint"] == func_id

    # Delete function (triggers FK SET NULL)
    fm.delete_function(function_id=func_id)

    # Verify entrypoint is now null
    tasks_after = unify.get_logs(context=ts._ctx, from_fields=["task_id", "entrypoint"])
    assert len(tasks_after) == 1
    assert tasks_after[0].entries.get("entrypoint") is None

    # Create NEW TaskScheduler instance (tests read path on init)
    ts_new = TaskScheduler()

    # Verify the new scheduler can read tasks with null entrypoint without errors
    tasks_from_new = unify.get_logs(
        context=ts_new._ctx,
    )
    assert len(tasks_from_new) == 1
    assert tasks_from_new[0].entries["task_id"] == tid
    assert tasks_from_new[0].entries.get("entrypoint") is None

    # Verify Task model construction succeeds (critical test for Orchestra NULL omission)
    from unity.task_scheduler.types.task import Task

    task_dict = tasks_from_new[0].entries
    task_obj = Task(**task_dict)
    assert task_obj.entrypoint is None
    assert task_obj.task_id == tid


@_handle_project
def test_entrypoint_explicit_none_on_create():
    """Test that tasks can be created with entrypoint=None explicitly."""
    ts = TaskScheduler()

    # Create task with explicit None entrypoint
    result = ts._create_task(
        name="Task without Entrypoint",
        description="Task that never had an entrypoint",
        entrypoint=None,
    )
    tid = result["details"]["task_id"]

    # Verify task was created with null entrypoint
    tasks = unify.get_logs(
        context=ts._ctx,
    )
    assert len(tasks) == 1
    assert tasks[0].entries["task_id"] == tid
    assert tasks[0].entries.get("entrypoint") is None

    # Verify it can be read back successfully
    from unity.task_scheduler.types.task import Task

    task_dict = tasks[0].entries
    task_obj = Task(**task_dict)
    assert task_obj.entrypoint is None


@_handle_project
def test_entrypoint_clone_after_set_null():
    """Test that cloning a task after its entrypoint is nulled works correctly."""
    fm = FunctionManager()
    ts = TaskScheduler()

    # Create function
    src = "def cloneable():\n    return 'clone me'\n"
    fm.add_functions(implementations=src)
    funcs = unify.get_logs(context=fm._compositional_ctx, from_fields=["function_id"])
    func_id = int(funcs[0].entries["function_id"])

    # Create recurring task with entrypoint
    from unity.task_scheduler.types.repetition import RepeatPattern, Frequency

    repeat_pattern = RepeatPattern(frequency=Frequency.DAILY)
    result = ts._create_task(
        name="Recurring Task",
        description="Task that repeats daily",
        entrypoint=func_id,
        repeat=[repeat_pattern],
    )
    tid = result["details"]["task_id"]

    # Verify task has entrypoint
    tasks = unify.get_logs(
        context=ts._ctx,
        filter=f"task_id == {tid}",
        from_fields=["task_id", "entrypoint", "instance_id"],
    )
    assert len(tasks) == 1
    assert tasks[0].entries["entrypoint"] == func_id
    original_instance_id = tasks[0].entries["instance_id"]

    # Delete function (triggers FK SET NULL)
    fm.delete_function(function_id=func_id)

    # Verify entrypoint is now null (include task_id to avoid NULL-only field issue)
    tasks_after_delete = unify.get_logs(
        context=ts._ctx,
        filter=f"task_id == {tid}",
        from_fields=["task_id", "entrypoint"],
    )
    assert len(tasks_after_delete) == 1
    assert tasks_after_delete[0].entries.get("entrypoint") is None

    # Trigger cloning by fetching the task and calling _clone_task_instance
    from unity.task_scheduler.types.task import Task

    task_entries = unify.get_logs(
        context=ts._ctx,
        filter=f"task_id == {tid}",
    )[0].entries
    task_obj = Task(**task_entries)

    # Clone the task
    ts._clone_task_instance(task_obj)

    # Verify clone was created with null entrypoint
    all_instances = unify.get_logs(
        context=ts._ctx,
        filter=f"task_id == {tid}",
        from_fields=["task_id", "instance_id", "entrypoint"],
    )
    assert len(all_instances) == 2  # Original + clone

    # Check both instances have null entrypoint
    for instance in all_instances:
        assert instance.entries.get("entrypoint") is None

    # Verify we have distinct instance_ids
    instance_ids = {int(inst.entries["instance_id"]) for inst in all_instances}
    assert len(instance_ids) == 2
