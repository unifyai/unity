import asyncio
import pytest

from unity.task_scheduler.task_scheduler import TaskScheduler
from unity.common.async_tool_loop import AsyncToolLoopHandle
from tests.helpers import _handle_project
from tests.test_async_tool_loop.async_helpers import (
    _wait_for_tool_request,
    _wait_for_condition,
)


def _assert_dict_subset(expected: dict, actual: dict):
    """
    Recursively assert that `expected` is a subset of `actual`.
    Values in `expected` must match exactly for corresponding keys in `actual`.
    """
    assert isinstance(actual, dict), "Actual value is not a dict"
    for k, v in expected.items():
        assert k in actual, f"Missing key in actual snapshot: {k!r}"
        if isinstance(v, dict):
            assert isinstance(
                actual[k],
                dict,
            ), f"Expected dict at key {k!r}, got {type(actual[k]).__name__}"
            _assert_dict_subset(v, actual[k])
        elif isinstance(v, list):
            assert isinstance(
                actual[k],
                list,
            ), f"Expected list at key {k!r}, got {type(actual[k]).__name__}"
            # For lists, ensure each expected item is present (subset semantics).
            # - If dict: require at least one actual item to be a superset of this dict.
            # - Else: require exact membership.
            for exp_item in v:
                if isinstance(exp_item, dict):
                    found = False
                    for act_item in actual[k]:
                        if isinstance(act_item, dict):
                            try:
                                _assert_dict_subset(exp_item, act_item)
                                found = True
                                break
                            except AssertionError:
                                continue
                    assert (
                        found
                    ), f"List at key {k!r} missing an item matching subset {exp_item!r}"
                else:
                    assert (
                        exp_item in actual[k]
                    ), f"List at key {k!r} missing item {exp_item!r}"
        else:
            assert (
                actual[k] == v
            ), f"Value mismatch at key {k!r}: {actual[k]!r} != {v!r}"


@pytest.mark.asyncio
@_handle_project
async def test_serialize_flat_ask():
    """
    Verify a flat TaskScheduler.ask snapshot contains the expected minimal shape.
    """
    ts = TaskScheduler()
    h = await ts.ask("What tasks are scheduled for today?")

    try:
        snap = h.serialize()  # type: ignore[attr-defined]

        # Human-readable expected subset
        expected = {
            "version": 1,
            "root": {
                "tool": "TaskScheduler.ask",
                "handle": "AsyncToolLoopHandle",
            },
        }
        _assert_dict_subset(expected, snap)
        assert snap.get("loop_id", "").startswith(
            "TaskScheduler.ask",
        ), "loop_id must start with TaskScheduler.ask"
        assert isinstance(snap.get("assistant"), list)
        assert isinstance(snap.get("tools"), list)
        # For non-recursive serialize, 'children' should be absent or an empty list
        if "children" in snap:
            assert isinstance(snap["children"], list) and len(snap["children"]) == 0
    finally:
        try:
            h.stop("cleanup")  # type: ignore[attr-defined]
        except Exception:
            pass
        try:
            await asyncio.wait_for(h.result(), timeout=60)  # type: ignore[attr-defined]
        except Exception:
            pass


@pytest.mark.asyncio
@_handle_project
async def test_serialize_execute_recursive():
    """
    Verify structure for TaskScheduler.execute now that it returns an ActiveQueue directly
    (no outer async tool loop or recursive serialization).
    """
    ts = TaskScheduler()

    # Create a runnable task and capture its id
    res = ts.create_task(name="Demo", description="Run a demo task.")
    tid = int(res["details"]["task_id"])

    h = await ts.execute(tid)

    try:
        # Direct structure check (mirrors nested_structure execute test)
        structure = await h.nested_structure()  # type: ignore[attr-defined]
        expected = {
            "handle": "ActiveQueue(SteerableToolHandle)",
            "children": [
                {
                    "handle": "ActiveTask(SteerableToolHandle)",
                    "children": [
                        {
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
async def test_serialize_flat_update_before_nested(monkeypatch):
    """
    Verify a flat TaskScheduler.update snapshot when the first-turn nested ask
    has been requested but not adopted (keeps snapshot flat and non-recursive).
    """
    gate = asyncio.Event()

    # Gate the manager's ask so it does not return a handle yet (keeps snapshot flat)
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
    )

    try:
        # Wait until the assistant has requested the first-turn 'ask' tool
        client = getattr(h, "_client", None)  # internal test-only access
        assert (
            client is not None
        ), "Expected AsyncToolLoopHandle to expose its client for tests"
        await _wait_for_tool_request(client, "ask")

        # Ask has been requested but is still blocked by the gate → no nested handle yet
        snap = h.serialize()  # type: ignore[attr-defined]
        expected = {
            "version": 1,
            "root": {
                "tool": "TaskScheduler.update",
                "handle": "AsyncToolLoopHandle",
            },
        }
        _assert_dict_subset(expected, snap)
        assert snap.get("loop_id", "").startswith(
            "TaskScheduler.update",
        ), "loop_id must start with TaskScheduler.update"
        assert isinstance(snap.get("assistant"), list)
        assert isinstance(snap.get("tools"), list)
        if "children" in snap:
            assert isinstance(snap["children"], list) and len(snap["children"]) == 0
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
async def test_serialize_update_then_ask_nested():
    """
    Verify a recursive snapshot for TaskScheduler.update → TaskScheduler.ask
    (policy requires 'ask' on the first turn).
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

        snap = h.serialize(recursive=True)  # type: ignore[attr-defined]

        # Human-readable overview of expected nested shape (subset semantics).
        expected_overview = {
            "version": 1,
            "root": {
                "tool": "TaskScheduler.update",
                "handle": "AsyncToolLoopHandle",
            },
            "children": [
                {
                    "tool": "TaskScheduler.ask",
                    "handle": "ReadOnlyAskGuardHandle(AsyncToolLoopHandle)",
                },
            ],
        }
        _assert_dict_subset(expected_overview, snap)
        assert snap.get("loop_id", "").startswith("TaskScheduler.update")
        assert isinstance(snap.get("assistant"), list)
        assert isinstance(snap.get("tools"), list)

        # Children schema: expect at least one child and one of them must be TaskScheduler.ask
        children = snap.get("children") or []
        assert isinstance(children, list) and len(children) >= 1
        child = None
        for ch in children:
            if (ch or {}).get("tool") == "TaskScheduler.ask":
                child = ch
                break
        assert (
            child is not None
        ), "Expected a child for TaskScheduler.ask in recursive snapshot"

        # Assert child subset (human-readable, stable shape)
        expected_child_subset = {
            "tool": "TaskScheduler.ask",
            "handle": "ReadOnlyAskGuardHandle(AsyncToolLoopHandle)",
        }
        _assert_dict_subset(expected_child_subset, child)

        # State must be 'in_flight' or 'done'
        assert child.get("state") in ("in_flight", "done")
        if child.get("state") == "in_flight":
            assert isinstance(
                child.get("snapshot"),
                dict,
            ), "In-flight child must include an inline snapshot"
    finally:
        try:
            h.stop("cleanup")  # type: ignore[attr-defined]
        except Exception:
            pass
        try:
            await asyncio.wait_for(h.result(), timeout=120)  # type: ignore[attr-defined]
        except Exception:
            pass


# ──────────────────────────────────────────────────────────────────────────────
# Deserialization tests (hard-coded snapshots)
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
@_handle_project
async def test_deserialize_and_continue_ask_with_interjection():
    """
    Start from a flat ask snapshot, resume, add an interjection, and verify completion and transcript.
    """
    snap = {
        "version": 1,
        "loop_id": "TaskScheduler.ask(static)",
        "root": {"tool": "TaskScheduler.ask", "handle": "AsyncToolLoopHandle"},
        "system_message": "You are helpful.",
        "initial_user_message": "What tasks are scheduled for today?",
        "assistant": [],
        "tools": [],
    }

    resumed: AsyncToolLoopHandle = AsyncToolLoopHandle.deserialize(snap)
    interjection_text = "Prefer concise output"
    await resumed.interject(interjection_text)  # type: ignore[attr-defined]
    out = await asyncio.wait_for(resumed.result(), timeout=240)  # type: ignore[attr-defined]
    assert isinstance(out, str) and len(out) > 0

    # Verify the interjection appears once in the resumed transcript
    hist = resumed.get_history()  # type: ignore[attr-defined]
    assert isinstance(hist, list)
    seen = [
        m
        for m in hist
        if isinstance(m, dict)
        and m.get("role") == "user"
        and interjection_text in str(m.get("content", ""))
    ]
    assert len(seen) == 1


@pytest.mark.asyncio
@_handle_project
async def test_deserialize_and_continue_update_before_nested_with_interjection():
    """
    Start from a flat update snapshot taken before nested 'ask' is adopted, resume,
    add an interjection, and verify completion and transcript.
    """
    snap = {
        "version": 1,
        "loop_id": "TaskScheduler.update(static)",
        "root": {"tool": "TaskScheduler.update", "handle": "AsyncToolLoopHandle"},
        "system_message": "You are helpful.",
        "initial_user_message": "Please update the task queue priorities.",
        "assistant": [],
        "tools": [],
    }

    resumed: AsyncToolLoopHandle = AsyncToolLoopHandle.deserialize(snap)
    interjection_text = "Please proceed safely"
    await resumed.interject(interjection_text)  # type: ignore[attr-defined]
    out = await asyncio.wait_for(resumed.result(), timeout=240)  # type: ignore[attr-defined]
    assert isinstance(out, str) and len(out) > 0

    hist = resumed.get_history()  # type: ignore[attr-defined]
    assert isinstance(hist, list)
    seen = [
        m
        for m in hist
        if isinstance(m, dict)
        and m.get("role") == "user"
        and interjection_text in str(m.get("content", ""))
    ]
    assert len(seen) == 1


@pytest.mark.asyncio
@_handle_project
async def test_deserialize_and_continue_update_then_ask_nested_with_interjection():
    """
    Start from a recursive update→ask snapshot, resume, add an interjection, and verify continuation.
    """
    snap = {
        "version": 1,
        "loop_id": "TaskScheduler.update(static-nested)",
        "root": {"tool": "TaskScheduler.update", "handle": "AsyncToolLoopHandle"},
        "system_message": "You are helpful.",
        "initial_user_message": "Reorder tasks to ensure the highest priority items run first.",
        "assistant": [],
        "tools": [],
        "children": [
            {
                "call_id": None,
                "tool": "TaskScheduler.ask",
                "handle": "ReadOnlyAskGuardHandle(AsyncToolLoopHandle)",
                "passthrough": False,
                "state": "done",
            },
        ],
    }

    resumed: AsyncToolLoopHandle = AsyncToolLoopHandle.deserialize(snap)
    interjection_text = "Prefer compact layout"
    await resumed.interject(interjection_text)  # type: ignore[attr-defined]
    out = await asyncio.wait_for(resumed.result(), timeout=240)  # type: ignore[attr-defined]
    assert isinstance(out, str) and len(out) > 0

    hist = resumed.get_history()  # type: ignore[attr-defined]
    assert isinstance(hist, list)
    seen = [
        m
        for m in hist
        if isinstance(m, dict)
        and m.get("role") == "user"
        and interjection_text in str(m.get("content", ""))
    ]
    assert len(seen) == 1
