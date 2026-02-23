"""Real TaskScheduler routing tests for CodeActActor.

These mirror `test_tasks.py` but use CodeActActor (code-first tool loop).
"""

import pytest

from tests.helpers import _handle_project
from tests.actor.state_managers.utils import (
    assert_code_act_function_manager_used,
    extract_code_act_execute_code_snippets,
    make_code_act_actor,
)
from unity.function_manager.function_manager import FunctionManager
from unity.manager_registry import ManagerRegistry


@pytest.mark.asyncio
@pytest.mark.timeout(600)
@pytest.mark.eval
@_handle_project
async def test_ask_calls_scheduler():
    """CodeAct routes read-only task question → primitives.tasks.ask."""
    async with make_code_act_actor(impl="real") as (actor, _primitives, calls):
        ts = ManagerRegistry.get_task_scheduler()
        ts.actor = actor
        ts._create_task(
            name="Quarterly report",
            description="Quarterly report",
            status="primed",
        )

        handle = await actor.act(
            "Which task is currently primed?",
            clarification_enabled=False,
        )
        result = await handle.result()

        assert "quarterly report" in str(result).lower()
        assert "primitives.tasks.ask" in calls
        assert all(c.startswith("primitives.tasks.") for c in calls)


@pytest.mark.asyncio
@pytest.mark.timeout(600)
@pytest.mark.eval
@_handle_project
async def test_ask_calls_scheduler_memoized():
    """CodeAct uses FunctionManager (when available) for task queries."""
    fm = FunctionManager()
    implementation = """
async def ask_tasks(question: str, response_format=None) -> str:
    \"\"\"Query tasks via the task scheduler (read-only).\"\"\"
    handle = await primitives.tasks.ask(question, response_format=response_format)
    return await handle.result()
"""
    fm.add_functions(implementations=implementation, overwrite=True)

    async with make_code_act_actor(
        impl="real",
        include_function_manager_tools=True,
        function_manager=fm,
    ) as (actor, _primitives, calls):
        ts = ManagerRegistry.get_task_scheduler()
        ts.actor = actor
        ts._create_task(
            name="Quarterly report",
            description="Quarterly report",
            status="primed",
        )

        handle = await actor.act(
            "Which task is currently primed?",
            clarification_enabled=False,
        )
        result = await handle.result()

        assert "quarterly report" in str(result).lower()
        assert_code_act_function_manager_used(handle)
        snippets = "\n\n".join(extract_code_act_execute_code_snippets(handle))
        assert "ask_tasks" in snippets

        assert "primitives.tasks.ask" in calls
        assert all(c.startswith("primitives.tasks.") for c in calls)


@pytest.mark.asyncio
@pytest.mark.timeout(600)
@pytest.mark.eval
@_handle_project
async def test_update_calls_scheduler():
    """CodeAct routes task mutation → primitives.tasks.update."""
    async with make_code_act_actor(impl="real") as (actor, _primitives, calls):
        ts = ManagerRegistry.get_task_scheduler()
        ts.actor = actor

        handle = await actor.act(
            "Create a new task called 'Promote Jeff Smith' with the description "
            "'Send an email to Jeff Smith to congratulate him on the promotion.'",
            clarification_enabled=False,
        )
        await handle.result()

        assert "primitives.tasks.update" in calls
        assert "primitives.tasks.execute" not in calls

        tasks = ts._filter_tasks()
        assert tasks and any(
            t.name.lower() == "promote jeff smith"
            or "promote jeff smith" in (t.description or "").lower()
            for t in tasks
        )


@pytest.mark.asyncio
@pytest.mark.timeout(600)
@pytest.mark.eval
@_handle_project
async def test_update_calls_scheduler_memoized():
    """CodeAct uses FunctionManager (when available) for task mutations."""
    fm = FunctionManager()
    implementation = """
async def create_or_update_or_delete_tasks(name: str, description: str) -> str:
    \"\"\"Create/update/delete tasks via the task scheduler.\"\"\"
    handle = await primitives.tasks.update(
        f"Create a new task called '{name}' with the description '{description}'."
    )
    return await handle.result()
"""
    fm.add_functions(implementations=implementation, overwrite=True)

    async with make_code_act_actor(
        impl="real",
        include_function_manager_tools=True,
        function_manager=fm,
    ) as (actor, _primitives, calls):
        ts = ManagerRegistry.get_task_scheduler()
        ts.actor = actor

        handle = await actor.act(
            "Create a new task called 'Promote Jeff Smith' with the description "
            "'Send an email to Jeff Smith to congratulate him on the promotion.'",
            clarification_enabled=False,
        )
        await handle.result()

        assert_code_act_function_manager_used(handle)
        snippets = "\n\n".join(extract_code_act_execute_code_snippets(handle))
        assert "create_or_update_or_delete_tasks" in snippets

        assert "primitives.tasks.update" in calls
        assert "primitives.tasks.execute" not in calls

        tasks = ts._filter_tasks()
        assert tasks and any(
            t.name.lower() == "promote jeff smith"
            or "promote jeff smith" in (t.description or "").lower()
            for t in tasks
        )


@pytest.mark.asyncio
@pytest.mark.timeout(600)
@pytest.mark.eval
@_handle_project
async def test_execute_calls_scheduler():
    """CodeAct routes task execution request → primitives.tasks.execute."""
    fm = FunctionManager()
    entrypoint_impl = """
async def run_quick_task_entrypoint() -> str:
    return "ok"
"""
    fm.add_functions(
        implementations=entrypoint_impl,
        verify={"run_quick_task_entrypoint": False},
        overwrite=True,
    )
    entrypoint_id = fm.list_functions()["run_quick_task_entrypoint"]["function_id"]
    assert isinstance(entrypoint_id, int)

    async with make_code_act_actor(
        impl="real",
        include_function_manager_tools=False,
        function_manager=fm,
    ) as (actor, _primitives, calls):
        ts = ManagerRegistry.get_task_scheduler()
        ts.actor = actor

        ts._create_task(
            name="Prepare the monthly analytics dashboard",
            description="Prepare the monthly analytics dashboard",
            entrypoint=entrypoint_id,
        )

        handle = await actor.act(
            "Run the task named 'Prepare the monthly analytics dashboard' now.",
            clarification_enabled=False,
        )
        await handle.result()

        assert "primitives.tasks.execute" in calls
        assert "primitives.tasks.update" not in calls

        ts2 = ManagerRegistry.get_task_scheduler()
        tasks = ts2._filter_tasks()
        executed_task = next(
            (t for t in tasks if "monthly analytics dashboard" in t.name.lower()),
            None,
        )
        assert executed_task is not None
        assert (
            executed_task.status != "queued"
            or executed_task.last_run is not None
            or executed_task.status in ["running", "completed", "failed"]
        ), (
            "Task shows no evidence of execution: "
            f"status={executed_task.status}, last_run={executed_task.last_run}"
        )


@pytest.mark.asyncio
@pytest.mark.timeout(600)
@pytest.mark.eval
@_handle_project
async def test_execute_calls_scheduler_memoized():
    """CodeAct uses FunctionManager (when available) for task execution."""
    fm = FunctionManager()
    entrypoint_impl = """
async def run_quick_task_entrypoint() -> str:
    return "ok"
"""
    execute_impl = """
async def execute_task_by_name(task_name: str) -> str:
    tasks = primitives.tasks._filter_tasks()
    target = (task_name or "").strip().lower()
    for t in tasks:
        if (getattr(t, "name", "") or "").strip().lower() != target:
            continue
        exec_handle = await primitives.tasks.execute(task_id=int(t.task_id))
        return await exec_handle.result()
    raise ValueError(f"No runnable task found with name: {task_name!r}")
"""
    fm.add_functions(
        implementations=[entrypoint_impl, execute_impl],
        verify={"run_quick_task_entrypoint": False, "execute_task_by_name": False},
        overwrite=True,
    )
    entrypoint_id = fm.list_functions()["run_quick_task_entrypoint"]["function_id"]
    assert isinstance(entrypoint_id, int)

    async with make_code_act_actor(
        impl="real",
        include_function_manager_tools=True,
        function_manager=fm,
    ) as (actor, _primitives, calls):
        ts = ManagerRegistry.get_task_scheduler()
        ts.actor = actor

        ts._create_task(
            name="Prepare the monthly analytics dashboard",
            description="Prepare the monthly analytics dashboard",
            entrypoint=entrypoint_id,
        )

        handle = await actor.act(
            "Run the task named 'Prepare the monthly analytics dashboard' now.",
            clarification_enabled=False,
        )
        await handle.result()

        assert_code_act_function_manager_used(handle)
        snippets = "\n\n".join(extract_code_act_execute_code_snippets(handle))
        assert "execute_task_by_name" in snippets

        assert "primitives.tasks.execute" in calls
        assert "primitives.tasks.update" not in calls

        ts2 = ManagerRegistry.get_task_scheduler()
        tasks = ts2._filter_tasks()
        executed_task = next(
            (t for t in tasks if "monthly analytics dashboard" in t.name.lower()),
            None,
        )
        assert executed_task is not None
        assert (
            executed_task.status != "queued"
            or executed_task.last_run is not None
            or executed_task.status in ["running", "completed", "failed"]
        ), (
            "Task shows no evidence of execution: "
            f"status={executed_task.status}, last_run={executed_task.last_run}"
        )
