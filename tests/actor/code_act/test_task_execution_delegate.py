"""Regression coverage for CodeAct-owned durable task execution."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

import unity.actor.code_act_actor as code_act_actor_module
from tests.helpers import _handle_project
from unity.actor.code_act_actor import CodeActActor
from unity.actor.simulated import SimulatedActor
from unity.common.task_execution_context import current_task_execution_delegate
from unity.function_manager.function_manager import FunctionManager
from unity.manager_registry import ManagerRegistry
from unity.task_scheduler.task_scheduler import TaskScheduler


class _FailingFallbackActor(SimulatedActor):
    """Actor used to prove `TaskScheduler` did not take its fallback branch."""

    def __init__(self) -> None:
        super().__init__(steps=1)
        self.calls = 0

    async def act(self, *args: Any, **kwargs: Any):
        self.calls += 1
        raise AssertionError("TaskScheduler fallback actor should not execute")


class _StaticToolLoopHandle:
    """Minimal handle returned by a deterministic tool-loop replacement."""

    def __init__(self, task: asyncio.Task[str]) -> None:
        self._task = task

    async def result(self) -> str:
        return await self._task

    async def stop(self, reason: str | None = None) -> None:
        _ = reason
        self._task.cancel()

    async def pause(self) -> None:
        return None

    async def resume(self) -> None:
        return None

    def done(self) -> bool:
        return self._task.done()


@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_codeact_tool_loop_task_inherits_task_execution_delegate(
    monkeypatch: pytest.MonkeyPatch,
):
    """The normal CodeAct loop should inherit the run-scoped task delegate."""

    captured: dict[str, object | None] = {}

    async def _observe_loop_context() -> str:
        captured["delegate_inside_loop"] = current_task_execution_delegate.get()
        return "loop completed"

    def _start_tool_loop(*args: Any, **kwargs: Any) -> _StaticToolLoopHandle:
        captured["delegate_while_starting_loop"] = current_task_execution_delegate.get()
        _ = args, kwargs
        return _StaticToolLoopHandle(asyncio.create_task(_observe_loop_context()))

    monkeypatch.setattr(
        code_act_actor_module,
        "start_async_tool_loop",
        _start_tool_loop,
    )

    actor = CodeActActor(can_store=False)
    try:
        handle = await actor.act("run the normal CodeAct loop")
        assert current_task_execution_delegate.get() is None

        assert await handle.result() == "loop completed"

        assert captured["delegate_while_starting_loop"] is not None
        assert (
            captured["delegate_inside_loop"] is captured["delegate_while_starting_loop"]
        )
        assert current_task_execution_delegate.get() is None
    finally:
        await actor.close()


@pytest.mark.asyncio
@pytest.mark.timeout(120)
@_handle_project
async def test_codeact_entrypoint_routes_task_execution_through_current_actor():
    """A task started inside CodeAct should execute through that CodeAct run."""

    ManagerRegistry.clear()
    fallback_actor = _FailingFallbackActor()
    scheduler = TaskScheduler(actor=fallback_actor)
    function_manager = FunctionManager()
    actor = CodeActActor(function_manager=function_manager)
    actor._act_semaphore = asyncio.Semaphore(1)
    actor._act_semaphore_timeout_s = 0.1

    try:
        function_manager.add_functions(
            implementations=[
                """
def delegated_task_entrypoint():
    return "delegated task completed"
""".strip(),
                """
async def start_task_from_codeact(task_id: int):
    handle = await primitives.tasks.execute(task_id=task_id)
    result = await handle.result()
    return f"outer saw: {result}"
""".strip(),
            ],
        )
        functions = function_manager.list_functions()
        task_entrypoint_id = functions["delegated_task_entrypoint"]["function_id"]
        outer_entrypoint_id = functions["start_task_from_codeact"]["function_id"]

        created = scheduler._create_task(
            name="delegated task",
            description="Run through the current CodeAct actor.",
            entrypoint=task_entrypoint_id,
        )
        task_id = created["details"]["task_id"]

        handle = await actor.act(
            "start the task",
            entrypoint=outer_entrypoint_id,
            entrypoint_kwargs={"task_id": task_id},
        )
        assert current_task_execution_delegate.get() is None

        result = await handle.result()

        assert result == "outer saw: delegated task completed"
        assert fallback_actor.calls == 0
        assert current_task_execution_delegate.get() is None
    finally:
        await actor.close()
        ManagerRegistry.clear()
