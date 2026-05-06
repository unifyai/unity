"""Regression coverage for CodeAct-owned durable task execution."""

from __future__ import annotations

import pytest

from tests.helpers import _handle_project
from unity.actor.code_act_actor import CodeActActor
from unity.actor.environments.state_managers import StateManagerEnvironment
from unity.common.task_execution_context import current_task_execution_delegate
from unity.function_manager.function_manager import FunctionManager
from unity.function_manager.primitives import PrimitiveScope, Primitives
from unity.manager_registry import ManagerRegistry
from unity.task_scheduler.task_scheduler import TaskScheduler
from unity.task_scheduler.types.status import Status


@pytest.mark.asyncio
@pytest.mark.llm_call
@pytest.mark.timeout(120)
@_handle_project
async def test_codeact_task_primitive_delegates_execution_without_fallback_actor():
    """CodeAct should execute a scheduled task through the real task primitive."""

    ManagerRegistry.clear()
    scheduler = TaskScheduler()
    function_manager = FunctionManager()
    primitives = Primitives(
        primitive_scope=PrimitiveScope(scoped_managers=frozenset({"tasks"})),
    )
    actor = CodeActActor(
        environments=[StateManagerEnvironment(primitives)],
        function_manager=function_manager,
    )

    try:
        function_manager.add_functions(
            implementations=[
                """
def delegated_task_entrypoint():
    return "delegated task completed"
""".strip(),
            ],
        )
        functions = function_manager.list_functions()
        task_entrypoint_id = functions["delegated_task_entrypoint"]["function_id"]

        task_id = scheduler._create_task(
            name="delegated task",
            description="Run through the current CodeAct actor.",
            entrypoint=task_entrypoint_id,
        )["details"]["task_id"]

        handle = await actor.act(
            (
                f"Execute scheduled task {task_id} now by calling "
                f"`primitives.tasks.execute(task_id={task_id})`, awaiting the "
                "returned handle, and reporting the task result."
            ),
            clarification_enabled=False,
        )
        assert current_task_execution_delegate.get() is None

        result = await handle.result()

        assert "delegated task completed" in result
        assert current_task_execution_delegate.get() is None
        assert scheduler.__dict__.get("_TaskScheduler__actor") is None
        assert "_actor" not in scheduler.__dict__
        assert scheduler._active_task is None

        task = scheduler._get_task_or_raise(task_id)
        assert task.status == Status.completed
    finally:
        await actor.close()
        ManagerRegistry.clear()
