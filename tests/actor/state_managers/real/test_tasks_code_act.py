"""Real TaskScheduler routing tests for CodeActActor.

Validates that CodeActActor uses ``execute_function`` for simple single-primitive
task operations, both with and without FunctionManager discovery tools.
"""

import pytest

from tests.helpers import _handle_project
from tests.actor.state_managers.utils import (
    assert_used_execute_function,
    make_code_act_actor,
)
from unity.function_manager.function_manager import FunctionManager
from unity.manager_registry import ManagerRegistry


@pytest.mark.asyncio
@pytest.mark.timeout(600)
@pytest.mark.eval
@_handle_project
async def test_ask_calls_scheduler():
    """CodeAct routes read-only task question via execute_function."""
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
        assert_used_execute_function(handle)
        assert "primitives.tasks.ask" in calls
        assert all(c.startswith("primitives.tasks.") for c in calls)


@pytest.mark.asyncio
@pytest.mark.timeout(600)
@pytest.mark.eval
@_handle_project
async def test_ask_calls_scheduler_with_fm_tools():
    """CodeAct routes task query via execute_function even with FM discovery tools present."""
    async with make_code_act_actor(
        impl="real",
        include_function_manager_tools=True,
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
        assert_used_execute_function(handle)
        assert "primitives.tasks.ask" in calls
        assert all(c.startswith("primitives.tasks.") for c in calls)


@pytest.mark.asyncio
@pytest.mark.timeout(600)
@pytest.mark.eval
@_handle_project
async def test_update_calls_scheduler():
    """CodeAct routes task mutation via execute_function."""
    async with make_code_act_actor(impl="real") as (actor, _primitives, calls):
        ts = ManagerRegistry.get_task_scheduler()
        ts.actor = actor

        handle = await actor.act(
            "Create a new task called 'Promote Jeff Smith' with the description "
            "'Send an email to Jeff Smith to congratulate him on the promotion.'",
            clarification_enabled=False,
        )
        await handle.result()

        assert_used_execute_function(handle)
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
async def test_update_calls_scheduler_with_fm_tools():
    """CodeAct routes task mutation via execute_function even with FM discovery tools present."""
    async with make_code_act_actor(
        impl="real",
        include_function_manager_tools=True,
    ) as (actor, _primitives, calls):
        ts = ManagerRegistry.get_task_scheduler()
        ts.actor = actor

        handle = await actor.act(
            "Create a new task called 'Promote Jeff Smith' with the description "
            "'Send an email to Jeff Smith to congratulate him on the promotion.'",
            clarification_enabled=False,
        )
        await handle.result()

        assert_used_execute_function(handle)
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
    """CodeAct routes task execution request via execute_function."""
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

        assert_used_execute_function(handle)
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
async def test_execute_calls_scheduler_with_fm_tools():
    """CodeAct routes task execution via execute_function even with FM discovery tools present."""
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

        assert_used_execute_function(handle)
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
