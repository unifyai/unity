from __future__ import annotations

import asyncio
import json

import pytest

from tests.helpers import _handle_project

from unity.conductor.conductor import Conductor
from unity.actor.simulated import SimulatedActor
from unity.common.async_tool_loop import AsyncToolLoopHandle
from unity.conductor.types import StateManager
from tests.test_async_tool_loop.async_helpers import (
    _wait_for_tool_request,
    _wait_for_condition,
)


@pytest.mark.asyncio
@_handle_project
async def test_actor_handle_present_for_direct_actor(monkeypatch):
    """
    Starting a Conductor.request that triggers Actor.act should make
    Conductor.actor_handle return the same live handle during execution, and None
    after completion.
    """

    # Keep actor session alive until explicitly stopped in cleanup
    actor = SimulatedActor(steps=None, duration=None)
    c = Conductor(actor=actor)

    # Start a live request that clearly implies an interactive session (Actor.act)
    h = await c.request(
        text=(
            "Let's do a live, step-by-step demo now. Please open a browser window "
            "and show a quick demo. Act now inside this chat."
        ),
    )
    try:
        # Wait deterministically until the assistant has requested Actor_act
        client = getattr(h, "_client", None)  # internal test-only access
        assert (
            client is not None
        ), "Expected AsyncToolLoopHandle to expose its client for tests"
        await _wait_for_tool_request(client, "Actor_act")

        # Wait until the actor handle is actually registered and visible via nested_structure
        async def _actor_child_adopted():
            try:
                tree = await h.nested_structure()

                def _has_actor(node: dict) -> bool:
                    try:
                        label = str(node.get("handle", "")).strip()
                    except Exception:
                        label = ""
                    if label.startswith("ActorHandle("):
                        return True
                    for ch in node.get("children", []) or []:
                        if _has_actor(ch):
                            return True
                    return False

                return _has_actor(tree)
            except Exception:
                return False

        await _wait_for_condition(_actor_child_adopted, poll=0.02, timeout=60.0)

        # Now also confirm Conductor exposes the handle
        async def _actor_handle_visible():
            try:
                return (await c.actor_handle()) is not None
            except Exception:
                return False

        await _wait_for_condition(_actor_handle_visible, poll=0.02, timeout=60.0)

        # After the child is visible, the handle method should return non-None
        assert await c.actor_handle() is not None

        # Method should return the same live request handle during execution
        assert await c.actor_handle() is h

        # Drive two steps to complete deterministically
        h.pause()
        h.resume()
        await asyncio.wait_for(h.result(), timeout=30)

        # After completion, the handle should clear
        assert await c.actor_handle() is None
    finally:
        h.stop("cleanup")


@pytest.mark.asyncio
@_handle_project
async def test_actor_handle_present_with_active_task(monkeypatch):
    """
    When executing a task (ActiveTask), actor_handle should also exist and be the
    same object as task_handle.
    """

    # Keep the execute session alive until explicit cleanup
    actor = SimulatedActor(steps=None, duration=None)
    c = Conductor(actor=actor)

    # Clean tasks and create one
    c.clear(StateManager.TASKS)
    ts = c._task_scheduler  # type: ignore[attr-defined]
    tid = ts._create_task(name="Y", description="Y")["details"]["task_id"]  # type: ignore[index]

    # Start by snapshot helper – registers automatically inside start_task
    h = await c.start_task(task_id=int(tid), trigger_reason="test-actor-has-task")
    try:
        # Wait deterministically until the assistant has requested TaskScheduler_execute
        client = getattr(h, "_client", None)  # internal test-only access
        assert (
            client is not None
        ), "Expected AsyncToolLoopHandle to expose its client for tests"
        await _wait_for_tool_request(client, "TaskScheduler_execute")

        # Ensure the nested structure has adopted the ActiveQueue/ActiveTask child
        async def _execute_child_adopted():
            try:
                tree = await h.nested_structure()

                def _has_exec(node: dict) -> bool:
                    try:
                        label = str(node.get("handle", "")).strip()
                    except Exception:
                        label = ""
                    if label.startswith("ActiveQueue(") or label.startswith(
                        "ActiveTask(",
                    ):
                        return True
                    for ch in node.get("children", []) or []:
                        if _has_exec(ch):
                            return True
                    return False

                return _has_exec(tree)
            except Exception:
                return False

        await _wait_for_condition(_execute_child_adopted, poll=0.02, timeout=60.0)

        # Both task_handle and actor_handle should exist and be the same
        th = await c.task_handle()
        ah = await c.actor_handle()
        assert th is not None
        assert ah is not None
        assert th is ah is h

        # Complete deterministically
        h.pause()
        h.resume()
        await asyncio.wait_for(h.result(), timeout=30)

        assert await c.actor_handle() is None
        assert await c.task_handle() is None
    finally:
        h.stop("cleanup")


@pytest.mark.asyncio
@_handle_project
async def test_actor_handle_absent_for_read_only_request():
    """
    A read-only Conductor.request loop (no Actor.act and no TaskScheduler.execute)
    should leave actor_handle as None.
    """

    c = Conductor()

    tools = dict(c.get_tools("request"))
    ask_tool_name = next(
        (k for k in tools.keys() if k.lower().endswith("_ask")),
        None,
    )
    assert ask_tool_name is not None, "No ask tool available on Conductor.request"

    snapshot = {
        "version": 1,
        "entrypoint": {"class_name": "Conductor", "method_name": "request"},
        "loop_id": "Conductor.request",
        "initial_user_message": "<read-only>",
        "assistant_steps": [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "tc_ro",
                        "type": "function",
                        "function": {
                            "name": ask_tool_name,
                            "arguments": json.dumps({"text": "Hello?"}),
                        },
                    },
                ],
            },
        ],
        "tool_results": [],
    }

    h = AsyncToolLoopHandle.deserialize(snapshot)
    c._live_requests.add(h)  # type: ignore[attr-defined]
    try:
        # No actor_handle expected for read-only flows
        assert await c.actor_handle() is None

        await asyncio.wait_for(h.result(), timeout=30)
    finally:
        h.stop("cleanup")
