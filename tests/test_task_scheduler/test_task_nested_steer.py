import asyncio
import pytest

from unity.task_scheduler.task_scheduler import TaskScheduler
from tests.helpers import _handle_project
from tests.test_async_tool_loop.async_helpers import _wait_for_condition
from unity.common.handle_wrappers import discover_wrapped_handles


@pytest.mark.asyncio
@_handle_project
async def test_task_nested_steer_interject_reaches_inner_ask():
    """
    Verify that nested_steer can target the inner TaskScheduler.ask loop within
    an update→ask nested configuration and that an interjection is delivered to
    the inner loop (bypassing the outer update loop).
    """
    ts = TaskScheduler()

    # Start an update flow that will run an inner ask on the first turn
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

        # Send an interjection only to the inner ask loop via nested_steer
        msg = "nested-steer interjection for inner ask"
        spec = {
            "children": [
                {
                    "tool": "TaskScheduler.ask",
                    "steps": [
                        {"method": "interject", "args": msg},
                    ],
                },
            ],
        }

        await h.nested_steer(spec)  # type: ignore[attr-defined]

        # Locate the adopted inner ask handle
        child_handle = None
        task_info = getattr(getattr(h, "_task", None), "task_info", {})  # type: ignore[attr-defined]
        if isinstance(task_info, dict):
            for meta in task_info.values():
                if (
                    getattr(meta, "name", None) == "ask"
                    and getattr(meta, "handle", None) is not None
                ):
                    child_handle = getattr(meta, "handle", None)
                    break
        assert child_handle is not None, "Expected inner ask handle to be adopted"

        # Assert the interjection is recorded on the inner ask handle's user-visible history
        async def _interjection_visible_on_inner():
            try:
                hist = getattr(child_handle, "_user_visible_history", [])  # type: ignore[attr-defined]
                for m in hist or []:
                    if isinstance(m, dict) and m.get("role") == "user":
                        c = m.get("content")
                        if isinstance(c, str) and c == msg:
                            return True
                        if isinstance(c, dict) and c.get("message") == msg:
                            return True
                return False
            except Exception:
                return False

        await _wait_for_condition(
            _interjection_visible_on_inner,
            poll=0.01,
            timeout=60.0,
        )

        # And confirm it did NOT land on the outer update handle
        outer_hist = getattr(h, "_user_visible_history", [])  # type: ignore[attr-defined]
        assert not any(
            isinstance(m, dict)
            and m.get("role") == "user"
            and (
                (isinstance(m.get("content"), str) and m.get("content") == msg)
                or (
                    isinstance(m.get("content"), dict)
                    and m.get("content", {}).get("message") == msg
                )
            )
            for m in (outer_hist or [])
        ), "Interjection should bypass the outer update loop"
    finally:
        try:
            h.stop("cleanup")  # type: ignore[attr-defined]
        except Exception:
            pass
        try:
            await asyncio.wait_for(h.result(), timeout=120)  # type: ignore[attr-defined]
        except Exception:
            pass


@pytest.mark.asyncio
@_handle_project
async def test_nested_steer_targets_actor_handle_and_applies_interject():
    """
    Verify that nested_steer can target the inner ActorHandle within the
    TaskScheduler.execute chain (ActiveQueue → ActiveTask → ActorHandle), and
    that an interjection is delivered to the ActorHandle (not just the outer layers).
    """
    ts = TaskScheduler()

    # Create a runnable task and capture its id
    res = ts.create_task(name="Demo", description="Run a demo task.")
    tid = int(res["details"]["task_id"])

    # Start execution using natural language so the outer execute loop remains in play
    h = await ts.execute(f"Please execute task id {tid} now.")

    try:
        # Wait until the nested execute tool has been adopted with a live handle
        async def _exec_child_adopted():
            try:
                ti = getattr(h, "_task", None)  # type: ignore[attr-defined]
                task_info = getattr(ti, "task_info", {}) if ti is not None else {}
                if isinstance(task_info, dict):
                    for meta in task_info.values():
                        nm = getattr(meta, "name", None)
                        hd = getattr(meta, "handle", None)
                        if (
                            nm in ("execute_by_id", "execute_isolated_by_id")
                            and hd is not None
                        ):
                            return True
                return False
            except Exception:
                return False

        await _wait_for_condition(_exec_child_adopted, poll=0.02, timeout=60.0)

        # Capture the adopted ActiveQueue handle immediately to avoid races with completion
        def _get_adopted_queue_handle():
            try:
                ti = getattr(h, "_task", None)  # type: ignore[attr-defined]
                task_info = getattr(ti, "task_info", {}) if ti is not None else {}
                if isinstance(task_info, dict):
                    for meta in task_info.values():
                        nm = getattr(meta, "name", None)
                        hd = getattr(meta, "handle", None)
                        if (
                            nm in ("execute_by_id", "execute_isolated_by_id")
                            and hd is not None
                        ):
                            return hd
                return None
            except Exception:
                return None

        aq_handle = _get_adopted_queue_handle()
        assert (
            aq_handle is not None
        ), "Expected ActiveQueue handle to be adopted after execute start"

        # Send an interjection directly to the inner ActorHandle via nested_steer
        msg = "actor-level interjection from nested_steer"
        spec = {
            "children": [
                {
                    "handle": "ActiveQueue",
                    "children": [
                        {
                            "handle": "ActiveTask",
                            "children": [
                                {
                                    "handle": "ActorHandle",
                                    "steps": [{"method": "interject", "args": msg}],
                                },
                            ],
                        },
                    ],
                },
            ],
        }

        await h.nested_steer(spec)  # type: ignore[attr-defined]

        # Walk wrappers to find the inner ActorHandle (SimulatedActorHandle)
        def _find_actor_handle(root):
            # BFS over wrapper graph
            from collections import deque

            dq = deque([root])
            seen = set()
            while dq:
                cur = dq.popleft()
                try:
                    cid = id(cur)
                    if cid in seen:
                        continue
                    seen.add(cid)
                except Exception:
                    pass

                # Actor handle class is SimulatedActorHandle (canonicalized to ActorHandle in structure)
                try:
                    if "ActorHandle" in getattr(cur, "__class__", object).__name__ or (
                        "SimulatedActorHandle"
                        in getattr(cur, "__class__", object).__name__
                    ):
                        return cur
                except Exception:
                    pass

                try:
                    pairs = list(discover_wrapped_handles(cur) or [])
                except Exception:
                    pairs = []
                for _src, child in pairs:
                    if child is not None:
                        dq.append(child)
            return None

        actor_handle = _find_actor_handle(aq_handle)
        assert (
            actor_handle is not None
        ), "Expected to discover inner ActorHandle via wrappers"

        # Confirm the interjection arrived in the actor's LLM message history
        async def _interjection_visible_on_actor():
            try:
                llm = getattr(actor_handle, "_llm", None)
                msgs = getattr(llm, "messages", None) if llm is not None else None
                if not isinstance(msgs, list):
                    return False
                for m in reversed(msgs):
                    if isinstance(m, dict) and m.get("role") == "user":
                        c = m.get("content")
                        if isinstance(c, list):
                            for item in c:
                                if (
                                    isinstance(item, dict)
                                    and item.get("type") == "text"
                                    and item.get("text") == msg
                                ):
                                    return True
                        elif isinstance(c, str) and c == msg:
                            return True
                return False
            except Exception:
                return False

        await _wait_for_condition(
            _interjection_visible_on_actor,
            poll=0.01,
            timeout=60.0,
        )
    finally:
        try:
            h.stop("cleanup")  # type: ignore[attr-defined]
        except Exception:
            pass
        try:
            await asyncio.wait_for(h.result(), timeout=180)  # type: ignore[attr-defined]
        except Exception:
            pass
