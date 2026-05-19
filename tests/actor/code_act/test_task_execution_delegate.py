"""Regression coverage for CodeAct-owned durable task execution."""

from __future__ import annotations

import pytest

from tests.helpers import _handle_project
from unity.actor.code_act_actor import CodeActActor, _CodeActTaskExecutionDelegate
from unity.actor.environments.state_managers import StateManagerEnvironment
from unity.actor.simulated import SimulatedActor
from unity.common.task_execution_context import current_task_execution_delegate
from unity.conversation_manager.domains.task_activation import (
    _ConversationTaskExecutionDelegate,
)
from unity.function_manager.function_manager import FunctionManager
from unity.function_manager.primitives import PrimitiveScope, Primitives
from unity.manager_registry import ManagerRegistry
from unity.task_scheduler.task_scheduler import TaskScheduler
from unity.task_scheduler.types.status import Status


@pytest.mark.asyncio
async def test_codeact_task_delegate_runs_description_tasks_in_child_actor_slot():
    calls = []
    actor = SimulatedActor(steps=0)

    original_act = actor.act

    async def _spy_act(*args, **kwargs):
        calls.append(kwargs)
        return await original_act(*args, **kwargs)

    actor.act = _spy_act  # type: ignore[method-assign]
    delegate = _CodeActTaskExecutionDelegate(actor)  # type: ignore[arg-type]

    handle = await delegate.start_task_run(
        task_description="Run the description-driven task.",
        entrypoint=None,
        parent_chat_context=None,
        clarification_up_q=None,
        clarification_down_q=None,
    )
    await handle.result()

    assert calls[0]["_reuse_actor_slot"] is False
    assert calls[0]["persist"] is False


@pytest.mark.asyncio
async def test_codeact_task_delegate_reuses_actor_slot_for_entrypoint_tasks():
    calls = []
    actor = SimulatedActor(steps=0)

    original_act = actor.act

    async def _spy_act(*args, **kwargs):
        calls.append(kwargs)
        return await original_act(*args, **kwargs)

    actor.act = _spy_act  # type: ignore[method-assign]
    delegate = _CodeActTaskExecutionDelegate(actor)  # type: ignore[arg-type]

    handle = await delegate.start_task_run(
        task_description="Run the function-backed task.",
        entrypoint=123,
        parent_chat_context=None,
        clarification_up_q=None,
        clarification_down_q=None,
    )
    await handle.result()

    assert calls[0]["_reuse_actor_slot"] is True
    assert calls[0]["entrypoint"] == 123


@pytest.mark.asyncio
async def test_task_execution_delegates_accept_shared_protocol_kwargs():
    class _FakeHandle:
        async def result(self):
            return "ok"

    class _FakeActor:
        def __init__(self):
            self.calls = []

        async def act(self, *args, **kwargs):
            self.calls.append({"args": args, "kwargs": kwargs})
            return _FakeHandle()

    for delegate_cls in (
        _CodeActTaskExecutionDelegate,
        _ConversationTaskExecutionDelegate,
    ):
        actor = _FakeActor()
        delegate = delegate_cls(actor)  # type: ignore[arg-type]

        handle = await delegate.start_task_run(
            task_description="Run with the shared task delegate protocol.",
            entrypoint=None,
            parent_chat_context=None,
            clarification_up_q=None,
            clarification_down_q=None,
            images=[],
            guidelines="Follow task-specific execution guidelines.",
            future_option=True,
        )

        assert await handle.result() == "ok"
        call = actor.calls[0]
        assert call["args"][0] == "Run with the shared task delegate protocol."
        assert (
            call["kwargs"]["guidelines"] == "Follow task-specific execution guidelines."
        )
        assert call["kwargs"]["future_option"] is True


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
