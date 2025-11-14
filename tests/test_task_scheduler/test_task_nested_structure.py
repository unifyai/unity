import asyncio
import pytest

from unity.task_scheduler.task_scheduler import TaskScheduler
from tests.helpers import _handle_project
from tests.test_async_tool_loop.async_helpers import (
    _wait_for_tool_request,
    _wait_for_condition,
)


@pytest.mark.asyncio
@_handle_project
async def test_nested_structure_flat_taskscheduler_ask():
    """
    Verify a flat, in‑flight TaskScheduler.ask loop reports a minimal structure.
    """
    ts = TaskScheduler()

    h = await ts.ask("What tasks are scheduled for today?")

    try:
        structure = await h.nested_structure()  # type: ignore[attr-defined]
        expected = {
            "handle": "ReadOnlyAskGuardHandle(AsyncToolLoopHandle)",
            "tool": "TaskScheduler.ask",
            "children": [],
        }
        assert structure == expected
    finally:
        try:
            h.stop("cleanup")  # type: ignore[attr-defined]
        except Exception:
            pass


@pytest.mark.asyncio
@_handle_project
async def test_nested_structure_taskscheduler_execute():
    """
    Verify TaskScheduler.execute returns an ActiveQueue handle that wraps an ActiveTask,
    and assert the full nested structure directly for readability.
    """
    ts = TaskScheduler()

    # Create a runnable task and capture its id
    res = ts.create_task(name="Demo", description="Run a demo task.")
    tid = int(res["details"]["task_id"])

    # Trigger task execution (by id)
    h = await ts.execute(tid)

    try:
        structure = await h.nested_structure()  # type: ignore[attr-defined]

        expected = {
            "handle": "ActiveQueue(SteerableToolHandle)",
            "children": [
                {
                    "handle": "ActiveTask(SteerableToolHandle)",
                    "children": [
                        {
                            # Simulated actor handle is canonicalized to drop 'Simulated' prefix
                            "handle": "ActorHandle(SteerableToolHandle)",
                            "children": [],
                        },
                    ],
                },
            ],
        }
        assert structure == expected
    finally:
        try:
            h.stop("cleanup")  # type: ignore[attr-defined]
        except Exception:
            pass
        try:
            await asyncio.wait_for(h.result(), timeout=180)  # type: ignore[attr-defined]
        except Exception:
            pass


@pytest.mark.asyncio
@_handle_project
async def test_nested_structure_flat_taskscheduler_update_before_nested(monkeypatch):
    """
    Verify a flat, in‑flight TaskScheduler.update loop reports a minimal structure
    when the first‑turn nested ask has been requested but not yet adopted.
    """
    gate = asyncio.Event()

    # Gate the manager's ask so it does not return a handle yet (keeps structure flat)
    original_ask = TaskScheduler.ask

    async def _gated_ask(self, *args, **kwargs):
        await gate.wait()
        # Return a simple string result to avoid creating a nested handle in this test
        return "ok"

    # Ensure the dynamic tool name exposed to the LLM remains exactly "ask"
    _gated_ask.__name__ = "ask"  # type: ignore[attr-defined]
    _gated_ask.__qualname__ = "ask"  # type: ignore[attr-defined]

    monkeypatch.setattr(TaskScheduler, "ask", _gated_ask, raising=True)

    ts = TaskScheduler()
    h = await ts.update(
        "Please update the task queue priorities.",
    )  # should require ask first

    try:
        # Wait until the assistant has requested the first-turn 'ask' tool
        client = getattr(h, "_client", None)  # internal test-only access
        assert (
            client is not None
        ), "Expected AsyncToolLoopHandle to expose its client for tests"
        await _wait_for_tool_request(client, "ask")

        # Ask has been requested but is still blocked by the gate → no nested handle yet
        structure = await h.nested_structure()  # type: ignore[attr-defined]
        expected = {
            "handle": "AsyncToolLoopHandle",
            "tool": "TaskScheduler.update",
            "children": [],
        }
        assert structure == expected
    finally:
        # Release the gate so the loop can finish cleanly
        try:
            gate.set()
        except Exception:
            pass
        try:
            await asyncio.wait_for(h.result(), timeout=120)  # type: ignore[attr-defined]
        except Exception:
            try:
                h.stop("cleanup")  # type: ignore[attr-defined]
            except Exception:
                pass


@pytest.mark.asyncio
@_handle_project
async def test_nested_structure_taskscheduler_update_then_ask_nested():
    """
    Verify a nested structure for TaskScheduler.update → TaskScheduler.ask
    (hard-coded policy requires 'ask' on the first turn).
    """
    ts = TaskScheduler()
    h = await ts.update(
        "Reorder tasks to ensure the highest priority items run first.",
    )

    try:
        # Wait deterministically until the nested ask handle has been adopted
        async def _ask_child_adopted():
            try:
                task_info = getattr(getattr(h, "_task", None), "task_info", {})  # type: ignore[attr-defined]
                if isinstance(task_info, dict):
                    return any(
                        getattr(meta, "name", None) == "ask"
                        and getattr(meta, "handle", None) is not None
                        for meta in task_info.values()
                    )
            except Exception:
                return False
            return False

        await _wait_for_condition(_ask_child_adopted, poll=0.01, timeout=60.0)

        structure = await h.nested_structure()  # type: ignore[attr-defined]
        expected = {
            "handle": "AsyncToolLoopHandle",
            "tool": "TaskScheduler.update",
            "children": [
                {
                    "handle": "ReadOnlyAskGuardHandle(AsyncToolLoopHandle)",
                    "tool": "TaskScheduler.ask",
                    "children": [],
                },
            ],
        }
        assert structure == expected
    finally:
        try:
            h.stop("cleanup")  # type: ignore[attr-defined]
        except Exception:
            pass
        try:
            await asyncio.wait_for(h.result(), timeout=120)  # type: ignore[attr-defined]
        except Exception:
            pass
