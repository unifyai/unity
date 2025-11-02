import asyncio

import pytest
import unify

from unity.common.async_tool_loop import (
    start_async_tool_loop,
    SteerableToolHandle,
)
from tests.helpers import _handle_project, SETTINGS
from tests.test_async_tool_loop.async_helpers import (
    _wait_for_tool_request,
    _wait_for_condition,
)


class ToyHandle(SteerableToolHandle):
    """Minimal nested handle that records pause/resume/stop and never finishes
    until stopped. Used to assert that nested_steer applies targeted methods.
    """

    def __init__(self) -> None:
        self._done = asyncio.Event()
        self.paused = 0
        self.resumed = 0
        self.stopped = 0
        self.interjections: list[str] = []

    async def ask(self, question: str, *, parent_chat_context_cont=None):  # type: ignore[override]
        return self

    async def interject(self, message: str, **_):  # type: ignore[override]
        self.interjections.append(message)
        return None

    def stop(self, *_, **__):  # type: ignore[override]
        self.stopped += 1
        self._done.set()
        return "stopped"

    def pause(self, *_, **__):  # type: ignore[override]
        self.paused += 1
        return "paused"

    def resume(self, *_, **__):  # type: ignore[override]
        self.resumed += 1
        return "resumed"

    def done(self) -> bool:  # type: ignore[override]
        return self._done.is_set()

    async def result(self) -> str:  # type: ignore[override]
        await self._done.wait()
        return "inner done"

    async def next_clarification(self) -> dict:  # type: ignore[override]
        return {}

    async def next_notification(self) -> dict:  # type: ignore[override]
        return {}

    async def answer_clarification(self, call_id: str, answer: str) -> None:  # type: ignore[override]
        return None


@pytest.mark.asyncio
@_handle_project
async def test_nested_steer_targets_child_and_applies_method():
    """nested_steer should target a live child by tool-name selector and apply the method."""

    inner = ToyHandle()

    async def Outer_spawn():  # type: ignore[valid-type]
        return inner

    # Real LLM client; direct it to call our tool in the first turn
    client = unify.AsyncUnify(
        endpoint="gpt-5@openai",
        reasoning_effort="high",
        service_tier="priority",
        cache=SETTINGS.UNIFY_CACHE,
        traced=SETTINGS.UNIFY_TRACED,
    )
    client.set_system_message(
        "You are running inside an automated test. In your FIRST assistant turn, call `Outer_spawn` with no arguments. "
        "Then wait for it to complete before replying.",
    )

    outer = start_async_tool_loop(
        client=client,  # type: ignore[arg-type]
        message="start",
        tools={"Outer_spawn": Outer_spawn},
    )

    try:
        # Wait until the assistant has requested our tool, ensuring the call is scheduled
        await _wait_for_tool_request(client, "Outer_spawn")

        # Also wait until the nested handle is adopted and visible in task_info
        async def _child_adopted():
            try:
                ti = getattr(outer._task, "task_info", {})  # type: ignore[attr-defined]
                if isinstance(ti, dict):
                    return any(
                        getattr(meta, "name", None) == "Outer_spawn"
                        and getattr(meta, "handle", None) is not None
                        for meta in ti.values()
                    )
            except Exception:
                return False
            return False

        await _wait_for_condition(_child_adopted, poll=0.01, timeout=60.0)

        # Apply nested steer: interject at root, pause the child by matching tool name
        spec = {
            "method": "interject",
            "args": "root-info",  # informational only
            "children": {
                "Outer_spawn": {"method": "pause"},
            },
        }

        res = await outer.nested_steer(spec)  # type: ignore[attr-defined]

        # Wait until the ToyHandle recorded the pause
        async def _paused():
            return inner.paused >= 1

        await _wait_for_condition(_paused, poll=0.01, timeout=30.0)

        # Assert pause applied and nested_steer reported an application targeting the child
        assert inner.paused >= 1, "pause was not applied to the nested handle"
        assert any(
            (item.get("method") == "pause")
            and any(
                isinstance(p, str) and ("Outer_spawn" in p)
                for p in (item.get("path") or [])
            )
            for item in (res.get("applied") or [])
        ), "nested_steer did not report applying pause to the child path"
    finally:
        # Ensure both outer and inner are stopped and finished to avoid pending tasks
        try:
            outer.stop("cleanup")
        except Exception:
            pass
        try:
            await asyncio.wait_for(outer.result(), timeout=30.0)
        except Exception:
            pass
        try:
            if not inner.done():
                inner.stop("cleanup")
        except Exception:
            pass


@pytest.mark.asyncio
@_handle_project
async def test_nested_steer_wrapper_fallback_descends_when_single_child():
    """When a child selector doesn't match but exactly one child-node is provided, nested_steer should
    descend through common wrapper attributes (e.g., _current_handle) and apply the method to the inner handle.
    """

    inner = ToyHandle()

    class WrapperHandle(SteerableToolHandle):
        def __init__(self, h: ToyHandle):
            self._current_handle = (
                h  # wrapper fallback attribute expected by nested_steer
            )

        async def ask(self, question: str, *, parent_chat_context_cont=None):  # type: ignore[override]
            return self

        async def interject(self, message: str, **_):  # type: ignore[override]
            return None

        def stop(self, *_, **__):  # type: ignore[override]
            return self._current_handle.stop()

        def pause(self, *_, **__):  # type: ignore[override]
            return self._current_handle.pause()

        def resume(self, *_, **__):  # type: ignore[override]
            return self._current_handle.resume()

        def done(self) -> bool:  # type: ignore[override]
            return self._current_handle.done()

        async def result(self) -> str:  # type: ignore[override]
            return await self._current_handle.result()

        async def next_clarification(self) -> dict:  # type: ignore[override]
            return {}

        async def next_notification(self) -> dict:  # type: ignore[override]
            return {}

        async def answer_clarification(self, call_id: str, answer: str) -> None:  # type: ignore[override]
            return None

    wrapper = WrapperHandle(inner)

    async def Wrapper_run():  # type: ignore[valid-type]
        return wrapper

    client = unify.AsyncUnify(
        endpoint="gpt-5@openai",
        reasoning_effort="high",
        service_tier="priority",
        cache=SETTINGS.UNIFY_CACHE,
        traced=SETTINGS.UNIFY_TRACED,
    )
    client.set_system_message(
        "You are running inside an automated test. In your FIRST assistant turn, call `Wrapper_run` with no arguments. "
        "Then wait for it to complete before replying.",
    )

    outer = start_async_tool_loop(
        client=client,  # type: ignore[arg-type]
        message="start",
        tools={"Wrapper_run": Wrapper_run},
    )

    try:
        await _wait_for_tool_request(client, "Wrapper_run")

        async def _child_adopted():
            try:
                ti = getattr(outer._task, "task_info", {})  # type: ignore[attr-defined]
                if isinstance(ti, dict):
                    return any(
                        getattr(meta, "name", None) == "Wrapper_run"
                        and getattr(meta, "handle", None) is not None
                        for meta in ti.values()
                    )
            except Exception:
                return False
            return False

        await _wait_for_condition(_child_adopted, poll=0.01, timeout=60.0)

        # Important: the first child-level selector intentionally does NOT match the tool name
        # so that nested_steer, once inside the wrapper handle, uses the single-child fallback
        # to descend into `_current_handle`.
        spec = {
            "children": {
                "Wrapper_run": {
                    "children": {
                        "IGNORED": {"method": "pause"},
                    },
                },
            },
        }

        await outer.nested_steer(spec)  # type: ignore[attr-defined]

        async def _paused():
            return inner.paused >= 1

        await _wait_for_condition(_paused, poll=0.01, timeout=30.0)

        assert (
            inner.paused >= 1
        ), "pause was not applied to the wrapped inner handle via fallback traversal"
    finally:
        # Ensure both outer and inner are stopped and finished to avoid pending tasks
        try:
            outer.stop("cleanup")
        except Exception:
            pass
        try:
            await asyncio.wait_for(outer.result(), timeout=30.0)
        except Exception:
            pass
        try:
            if not inner.done():
                inner.stop("cleanup")
        except Exception:
            pass
