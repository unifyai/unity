"""Real TaskScheduler tests for Actor.

Tests that Actor correctly calls real TaskScheduler methods and verifies
actual state mutations.
"""

import pytest

from tests.helpers import _handle_project
from tests.test_actor.test_state_managers.utils import (
    assert_memoized_function_used,
    assert_tool_called,
    get_state_manager_tools,
    make_hierarchical_actor,
)
from unity.function_manager.function_manager import FunctionManager
from unity.manager_registry import ManagerRegistry


@pytest.mark.asyncio
@pytest.mark.timeout(300)
@pytest.mark.eval
@_handle_project
async def test_ask_calls_scheduler(mock_verification):
    """Test that Actor calls TaskScheduler.ask for task queries."""
    async with make_hierarchical_actor(impl="real") as actor:

        # Access real TaskScheduler and seed data
        ts = ManagerRegistry.get_task_scheduler()
        ts.actor = actor
        ts._create_task(
            name="Quarterly report",
            description="Quarterly report",
            status="primed",
        )

        # Call actor with natural language query
        handle = await actor.act(
            "Which task is currently primed?",
            persist=False,
        )

        # Wait for result
        result = await handle.result()

        # Assert result contains expected task
        assert "quarterly report" in result.lower()

        # Assert correct tool was called
        assert_tool_called(handle, "primitives.tasks.ask")

        # Assert only tasks tools were used
        state_manager_tools = get_state_manager_tools(handle)
        assert all("tasks" in tool for tool in state_manager_tools)


@pytest.mark.asyncio
@pytest.mark.timeout(300)
@pytest.mark.eval
@_handle_project
async def test_ask_calls_scheduler_memoized(mock_verification):
    """Test that Actor uses memoized function for task queries."""
    async with make_hierarchical_actor(impl="real") as actor:

        # Access real TaskScheduler and seed data
        ts = ManagerRegistry.get_task_scheduler()
        ts.actor = actor
        ts._create_task(
            name="Quarterly report",
            description="Quarterly report",
            status="primed",
        )

        # Create FunctionManager and seed memoized function
        fm = FunctionManager()
        implementation = '''
async def ask_tasks(question: str, response_format=None) -> str:
    """Query the task list via the task scheduler (read-only).

    **Use when** the user is asking about existing tasks: what is due, what is scheduled,
    what is assigned to someone, priorities/statuses, or summaries of the task queue.

    **Do NOT use when**:
    - the user wants to create/update/delete/reorder tasks (use `primitives.tasks.update`)
    - the user wants to execute a task (use `tasks.execute` in the task system; not this test primitive)
    - the user is asking about contacts/transcripts/guidance/web

    Args:
        question: The task-related question to ask.
        response_format: Optional Pydantic model for structured output.

    Returns:
        The answer from the task scheduler as a string.
    """
    handle = await primitives.tasks.ask(question, response_format=response_format)
    result = await handle.result()
    return result
'''
        fm.add_functions(implementations=implementation, overwrite=True)
        actor.function_manager = fm

        # Call actor with natural language query
        handle = await actor.act(
            "Which task is currently primed? Do not ask clarifying questions. Do not create any stubs. Generate the full plan. Proceed with the best interpretation of the request.",
            persist=False,
        )

        # Wait for result
        result = await handle.result()

        # Assert result contains expected task
        assert "quarterly report" in result.lower()

        # Assert memoized function was used
        assert_memoized_function_used(handle, "ask_tasks")

        # Assert underlying primitive was called
        assert_tool_called(handle, "primitives.tasks.ask")


@pytest.mark.asyncio
@pytest.mark.timeout(300)
@pytest.mark.eval
@_handle_project
async def test_update_calls_scheduler(mock_verification):
    """Test that Actor calls TaskScheduler.update for mutations."""
    async with make_hierarchical_actor(impl="real") as actor:

        # Access real TaskScheduler
        ts = ManagerRegistry.get_task_scheduler()
        ts.actor = actor

        # Call actor with update request
        handle = await actor.act(
            "Create a new task called 'Promote Jeff Smith' with the description 'Send an email to Jeff Smith to congratulate him on the promotion.'",
            persist=False,
        )

        # Wait for result
        result = await handle.result()

        # Assert update tool was called (NOT ask)
        assert_tool_called(handle, "primitives.tasks.update")

        # Verify the tool selection - should not call ask for mutation
        state_manager_tools = get_state_manager_tools(handle)
        # May call ask to check if task exists, but must call update
        assert "primitives.tasks.update" in state_manager_tools

        # Verify mutation actually occurred
        tasks = ts._filter_tasks()
        assert tasks and any(
            t.name.lower() == "promote jeff smith"
            or "promote jeff smith" in t.description.lower()
            for t in tasks
        )


@pytest.mark.asyncio
@pytest.mark.timeout(300)
@pytest.mark.eval
@_handle_project
async def test_update_calls_scheduler_memoized(mock_verification):
    """Test that Actor uses memoized function for task updates."""
    async with make_hierarchical_actor(impl="real") as actor:

        # Access real TaskScheduler
        ts = ManagerRegistry.get_task_scheduler()
        ts.actor = actor

        # Create FunctionManager and seed memoized function
        fm = FunctionManager()
        implementation = '''
async def create_or_update_or_delete_tasks(name: str, description: str) -> str:
    """Mutate tasks via the task scheduler (create/update/delete/reorder).

    **Use when** the user requests any change to the task list: create a task, update
    fields like priority/schedule/status, delete tasks, or otherwise modify tasks.

    **Do NOT use when**:
    - the user is asking a read-only question about tasks (use `primitives.tasks.ask`)
    - the user is asking about contacts/transcripts/guidance/web

    Args:
        name: The name/title of the task to create.
        description: The description of the task.

    Returns:
        The result from the task scheduler update operation as a string.
    """
    handle = await primitives.tasks.update(
        f"Create a new task called '{name}' with the description '{description}'."
    )
    result = await handle.result()
    return result
'''
        fm.add_functions(implementations=implementation, overwrite=True)
        actor.function_manager = fm

        # Call actor with update request
        handle = await actor.act(
            "Create a new task called 'Promote Jeff Smith' with the description 'Send an email to Jeff Smith to congratulate him on the promotion.' Do not ask clarifying questions. Do not create any stubs. Generate the full plan. Proceed with the best interpretation of the request.",
            persist=False,
        )

        # Wait for result
        result = await handle.result()

        # Assert memoized function was used
        assert_memoized_function_used(handle, "create_or_update_or_delete_tasks")

        # Assert underlying primitive was called
        assert_tool_called(handle, "primitives.tasks.update")

        # Verify mutation actually occurred
        tasks = ts._filter_tasks()
        assert tasks and any(
            t.name.lower() == "promote jeff smith"
            or "promote jeff smith" in t.description.lower()
            for t in tasks
        )


@pytest.mark.asyncio
@pytest.mark.timeout(300)
@pytest.mark.eval
@_handle_project
async def test_execute_calls_scheduler(mock_verification):
    """Test that Actor calls TaskScheduler.execute for task execution."""
    async with make_hierarchical_actor(impl="real") as actor:

        # Access real TaskScheduler and seed data
        ts = ManagerRegistry.get_task_scheduler()
        ts.actor = actor

        # NOTE: `TaskScheduler.execute` will execute a task's description via an Actor (delegated
        # through `current_task_execution_delegate`). If we seed a realistic, open-ended task
        # description here, this test can exceed the 300s timeout while the Actor tries to
        # "actually do the work".
        #
        # To keep this test deterministic and fast while still exercising delegation, seed a
        # tiny FunctionManager entrypoint and attach it to the task.
        fm = FunctionManager()
        entrypoint_impl = '''
async def run_quick_task_entrypoint() -> str:
    """Fast deterministic task entrypoint for tests."""
    return "ok"
'''
        fm.add_functions(
            implementations=entrypoint_impl,
            verify={"run_quick_task_entrypoint": False},
            overwrite=True,
        )
        actor.function_manager = fm
        entrypoint_id = fm.list_functions()["run_quick_task_entrypoint"]["function_id"]
        assert isinstance(entrypoint_id, int)

        ts._create_task(
            name="Prepare the monthly analytics dashboard",
            description="Prepare the monthly analytics dashboard",
            entrypoint=entrypoint_id,
        )

        # Call actor with execute request
        handle = await actor.act(
            "Run the task named 'Prepare the monthly analytics dashboard' now.",
            persist=False,
        )

        # Wait for result
        result = await handle.result()

        # Assert execute tool was called exactly once
        assert_tool_called(handle, "primitives.tasks.execute")

        # Verify the tool selection - should use execute, NOT update
        state_manager_tools = get_state_manager_tools(handle)
        assert "primitives.tasks.execute" in state_manager_tools
        assert "primitives.tasks.update" not in state_manager_tools
        # May also see primitives.tasks.ask to resolve task ID

        # Verify mutation: task was actually executed
        ts = ManagerRegistry.get_task_scheduler()
        tasks = ts._filter_tasks()
        executed_task = None
        for task in tasks:
            if "monthly analytics dashboard" in task.name.lower():
                executed_task = task
                break

        assert (
            executed_task is not None
        ), "Task 'Prepare the monthly analytics dashboard' not found"
        # Verify execution evidence: status changed or execution metadata exists
        # Task should have transitioned from initial state or have execution timestamp
        assert (
            executed_task.status != "queued"
            or executed_task.last_run is not None
            or executed_task.status in ["running", "completed", "failed"]
        ), f"Task shows no evidence of execution: status={executed_task.status}, last_run={executed_task.last_run}"


@pytest.mark.asyncio
@pytest.mark.timeout(300)
@pytest.mark.eval
@_handle_project
async def test_execute_calls_scheduler_memoized(mock_verification):
    """Test that Actor uses memoized function for task execution."""
    async with make_hierarchical_actor(impl="real") as actor:

        # Access real TaskScheduler and seed data
        ts = ManagerRegistry.get_task_scheduler()
        ts.actor = actor

        # Create FunctionManager and seed:
        # - a fast deterministic task entrypoint (to keep TaskScheduler.execute from wandering)
        # - the memoized function under test
        fm = FunctionManager()
        entrypoint_impl = '''
async def run_quick_task_entrypoint() -> str:
    """Fast deterministic task entrypoint for tests."""
    return "ok"
'''

        implementation = '''
async def execute_task_by_name(task_name: str) -> str:
    """Execute a task by name via the task scheduler.

    **Use when** the user wants to start/run/execute a specific task that exists in the task queue.

    **Do NOT use when**:
    - the user is asking about tasks (use `primitives.tasks.ask`)
    - the user wants to create/update/delete tasks (use `primitives.tasks.update`)
    - the user is asking about contacts/transcripts/guidance/web

    Args:
        task_name: The name of the task to execute.

    Returns:
        The result from the task execution as a string.
    """
    # Use the injected primitives object (convention) rather than importing ManagerRegistry.
    ts = primitives.tasks
    # Fast deterministic lookup: filter by exact name (case-insensitive) and exclude terminal/active.
    candidates = ts._filter_tasks()

    def _norm(s):
        return (s or "").strip().lower()

    target = _norm(task_name)
    runnable = []
    for t in candidates:
        if _norm(getattr(t, "name", None)) != target:
            continue
        status = _norm(getattr(t, "status", None))
        if status in {"completed", "cancelled", "failed", "active"}:
            continue
        runnable.append(t)

    if not runnable:
        raise ValueError(f"No runnable task found with name: {task_name!r}")

    chosen = runnable[0]
    exec_handle = await primitives.tasks.execute(task_id=int(chosen.task_id))
    result = await exec_handle.result()
    return result
'''
        fm.add_functions(
            implementations=[entrypoint_impl, implementation],
            verify={"run_quick_task_entrypoint": False, "execute_task_by_name": False},
            overwrite=True,
        )
        actor.function_manager = fm
        entrypoint_id = fm.list_functions()["run_quick_task_entrypoint"]["function_id"]
        assert isinstance(entrypoint_id, int)

        ts._create_task(
            name="Prepare the monthly analytics dashboard",
            description="Prepare the monthly analytics dashboard",
            entrypoint=entrypoint_id,
        )

        # Call actor with execute request
        handle = await actor.act(
            "Run the task named 'Prepare the monthly analytics dashboard' now. Do not ask clarifying questions. Do not create any stubs. Generate the full plan. Proceed with the best interpretation of the request.",
            persist=False,
        )

        # Wait for result
        result = await handle.result()

        # Assert memoized function was used
        assert_memoized_function_used(handle, "execute_task_by_name")

        # Assert underlying primitive was called
        assert_tool_called(handle, "primitives.tasks.execute")

        # Verify mutation: task was actually executed
        ts = ManagerRegistry.get_task_scheduler()
        tasks = ts._filter_tasks()
        executed_task = None
        for task in tasks:
            if "monthly analytics dashboard" in task.name.lower():
                executed_task = task
                break

        assert (
            executed_task is not None
        ), "Task 'Prepare the monthly analytics dashboard' not found"
        # Verify execution evidence: status changed or execution metadata exists
        # Task should have transitioned from initial state or have execution timestamp
        assert (
            executed_task.status != "queued"
            or executed_task.last_run is not None
            or executed_task.status in ["running", "completed", "failed"]
        ), f"Task shows no evidence of execution: status={executed_task.status}, last_run={executed_task.last_run}"
