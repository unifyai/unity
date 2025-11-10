from __future__ import annotations

import asyncio
import json

import pytest

from tests.helpers import _handle_project

from unity.conductor.conductor import Conductor
from unity.conductor.types import StateManager
from unity.actor.simulated import SimulatedActor
from unity.common.async_tool_loop import AsyncToolLoopHandle


@pytest.mark.asyncio
@_handle_project
async def test_task_handle_present_with_deserialized_execute():
    """
    Starting a Conductor.request loop via deserialization that immediately executes
    a task should make Conductor.task_handle return the same live request handle
    while the execution is in-flight, and None after completion.
    """

    # Use step-based actor so the task does not complete before we can assert
    actor = SimulatedActor(steps=2, duration=None)
    c = Conductor(actor=actor)

    # Ensure a clean task table
    c.clear(StateManager.TASKS)

    # Pre-create a simple task and obtain its id
    ts = c._task_scheduler  # type: ignore[attr-defined]
    tid = ts._create_task(name="X", description="X")["details"]["task_id"]  # type: ignore[index]

    # Start via deserialization helper – runs TaskScheduler.execute immediately
    # Spy to emit an event when the target task turns active
    active_evt: asyncio.Event = asyncio.Event()
    orig_update = ts._update_task_status_instance

    def _spy_update(*, task_id: int, instance_id: int, new_status: str, activated_by=None):  # type: ignore[override]
        res = orig_update(
            task_id=task_id,
            instance_id=instance_id,
            new_status=new_status,
            activated_by=activated_by,
        )
        try:
            if task_id == int(tid) and str(new_status) == "active":
                active_evt.set()
        except Exception:
            pass
        return res

    # Install the spy
    setattr(ts, "_update_task_status_instance", _spy_update)

    h = await c.start_task(task_id=int(tid), trigger_reason="test")

    # Wait deterministically until the task is active
    await asyncio.wait_for(active_evt.wait(), timeout=10)

    # Property should expose the same live request handle during execution
    assert c.task_handle is not None
    # The property should point to the same object (logging wrappers are not used here)
    assert c.task_handle is h

    # Drive two steps (pause/resume) to complete deterministically
    h.pause()
    h.resume()

    # Completion clears the handle
    await asyncio.wait_for(h.result(), timeout=30)
    assert c.task_handle is None


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

    # During this read-only request, there must be no task_handle
    assert c.task_handle is None

    # Finish the loop to avoid background tasks lingering
    await asyncio.wait_for(h.result(), timeout=30)
