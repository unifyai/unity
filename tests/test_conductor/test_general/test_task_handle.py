from __future__ import annotations

import asyncio
import json

import pytest

from tests.helpers import _handle_project

from unity.conductor.conductor import Conductor
from unity.conductor.types import StateManager
from unity.actor.simulated import SimulatedActor
from unity.common.async_tool_loop import AsyncToolLoopHandle
from tests.test_async_tool_loop.async_helpers import (
    _wait_for_tool_request,
    _wait_for_condition,
)


@pytest.mark.asyncio
@_handle_project
async def test_task_handle_present_with_deserialized_execute(monkeypatch):
    """
    Starting a Conductor.request loop via deserialization that immediately executes
    a task should make Conductor.task_handle return the same live request handle
    while the execution is in-flight, and None after completion.
    """

    # Keep actor session alive until explicitly stopped in cleanup
    actor = SimulatedActor(steps=None, duration=None)
    c = Conductor(actor=actor)

    # Ensure a clean task table
    c.clear(StateManager.TASKS)

    # Pre-create a simple task and obtain its id
    ts = c._task_scheduler  # type: ignore[attr-defined]
    tid = ts._create_task(name="X", description="X")["details"]["task_id"]  # type: ignore[index]

    h = await c.start_task(task_id=int(tid), trigger_reason="test")
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

        # Method should expose the same live request handle during execution
        assert await c.task_handle() is not None
        # The handle should point to the same object (logging wrappers are not used here)
        assert await c.task_handle() is h

        # Drive two steps (pause/resume) to complete deterministically
        h.pause()
        h.resume()

        # Completion clears the handle
        await asyncio.wait_for(h.result(), timeout=30)
        assert await c.task_handle() is None
    finally:
        h.stop("cleanup")


@pytest.mark.asyncio
@_handle_project
async def test_task_handle_none_with_deserialized_non_execute():
    """
    Deserializing a Conductor.request loop that calls a read-only tool (not execute)
    should keep Conductor.task_handle as None.
    """

    c = Conductor()

    # Pick a safe read-only tool on the request surface (e.g., ContactManager.ask)
    tools = dict(c.get_tools("request"))
    ask_tool_name = next(
        (
            k
            for k in tools.keys()
            if k.lower().startswith("contactmanager_") and k.lower().endswith("ask")
        ),
        None,
    )
    # Fallback to any other *ask* tool if ContactManager.ask is unavailable
    if ask_tool_name is None:
        ask_tool_name = next(
            (k for k in tools.keys() if k.lower().endswith("_ask")),
            None,
        )
    assert ask_tool_name is not None, "No read-only ask tool found on Conductor.request"

    # Build a minimal v1 snapshot that schedules a single read-only ask tool call
    call_id = "tc_test"
    snapshot = {
        "version": 1,
        "entrypoint": {"class_name": "Conductor", "method_name": "request"},
        "loop_id": f"Conductor.request",
        "initial_user_message": "<test: read-only request>",
        "assistant_steps": [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": ask_tool_name,
                            "arguments": json.dumps({"text": "Who is Alice?"}),
                        },
                    },
                ],
            },
        ],
        "tool_results": [],
    }

    h = AsyncToolLoopHandle.deserialize(snapshot)

    # Register with Conductor so properties scan this live request
    # (tests are allowed to use private attributes as in other suites)
    c._live_requests.add(h)  # type: ignore[attr-defined]
    try:
        # During this read-only request, there must be no task_handle
        assert await c.task_handle() is None

        # Finish the loop to avoid background tasks lingering
        await asyncio.wait_for(h.result(), timeout=30)
    finally:
        h.stop("cleanup")
