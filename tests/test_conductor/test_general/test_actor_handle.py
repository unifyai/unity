from __future__ import annotations

import asyncio
import json

import pytest

from tests.helpers import _handle_project

from unity.conductor.conductor import Conductor
from unity.actor.simulated import SimulatedActor
from unity.common.async_tool_loop import AsyncToolLoopHandle
from unity.conductor.types import StateManager


@pytest.mark.asyncio
@_handle_project
async def test_actor_handle_present_for_direct_actor(monkeypatch):
    """
    Starting a Conductor.request that triggers Actor.act should make
    Conductor.actor_handle return the same live handle during execution, and None
    after completion.
    """

    # Step-based actor so it won't auto-complete before assertions
    actor = SimulatedActor(steps=2, duration=None)
    c = Conductor(actor=actor)

    # Trigger: spy session adoption to avoid timing races (no sleeps)
    adopt_evt: asyncio.Event = asyncio.Event()
    orig_adopt = c._session_guard.adopt  # type: ignore[attr-defined]

    async def _spy_adopt(handle, kind):  # type: ignore[override]
        await orig_adopt(handle, kind)
        try:
            if str(kind) == "actor" and handle is not None:
                adopt_evt.set()
        except Exception:
            pass

    monkeypatch.setattr(c._session_guard, "adopt", _spy_adopt, raising=True)  # type: ignore[attr-defined]

    # Start a live request that clearly implies an interactive session (Actor.act)
    h = await c.request(
        text=(
            "Let's do a live, step-by-step demo now. Please open a browser window "
            "and show a quick demo. Act now inside this chat."
        ),
    )

    # Wait deterministically for actor adoption
    await asyncio.wait_for(adopt_evt.wait(), timeout=120)

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


@pytest.mark.asyncio
@_handle_project
async def test_actor_handle_present_with_active_task(monkeypatch):
    """
    When executing a task (ActiveTask), actor_handle should also exist and be the
    same object as task_handle.
    """

    actor = SimulatedActor(steps=2, duration=None)
    c = Conductor(actor=actor)

    # Clean tasks and create one
    c.clear(StateManager.TASKS)
    ts = c._task_scheduler  # type: ignore[attr-defined]
    tid = ts._create_task(name="Y", description="Y")["details"]["task_id"]  # type: ignore[index]

    # Spy a session adoption event for execute to avoid timing races
    adopt_evt: asyncio.Event = asyncio.Event()
    orig_adopt = c._session_guard.adopt  # type: ignore[attr-defined]

    async def _spy_adopt(handle, kind):  # type: ignore[override]
        await orig_adopt(handle, kind)
        try:
            if str(kind) == "execute" and handle is not None:
                adopt_evt.set()
        except Exception:
            pass

    monkeypatch.setattr(c._session_guard, "adopt", _spy_adopt, raising=True)  # type: ignore[attr-defined]

    # Start by snapshot helper – registers automatically inside start_task
    h = await c.start_task(task_id=int(tid), trigger_reason="test-actor-has-task")

    # Wait deterministically until execute session is adopted
    await asyncio.wait_for(adopt_evt.wait(), timeout=120)

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

    # No actor_handle expected for read-only flows
    assert await c.actor_handle() is None

    await asyncio.wait_for(h.result(), timeout=30)
