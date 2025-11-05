import asyncio

import pytest
import unify

from unity.common.async_tool_loop import start_async_tool_loop, SteerableToolHandle
from tests.helpers import _handle_project, SETTINGS
from tests.test_async_tool_loop.async_helpers import (
    _wait_for_tool_request,
    _wait_for_condition,
)


class _TaskInfoMeta:
    def __init__(self, name: str, call_id: str, handle):
        self.name = name
        self.call_id = call_id
        self.handle = handle
        self.is_passthrough = False


class _TaskContainer:
    def __init__(self, info: dict):
        self.task_info = info


class ToyHandle(SteerableToolHandle):
    def __init__(self) -> None:
        self._done = asyncio.Event()

    async def ask(self, question: str, *, parent_chat_context_cont=None):  # type: ignore[override]
        return self

    async def interject(self, message: str, **_):  # type: ignore[override]
        return None

    def stop(self, *_, **__):  # type: ignore[override]
        self._done.set()
        return "stopped"

    def pause(self, *_, **__):  # type: ignore[override]
        return "paused"

    def resume(self, *_, **__):  # type: ignore[override]
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


class NestedHandle(SteerableToolHandle):
    def __init__(self) -> None:
        self.child = ToyHandle()
        # Expose a nested child via task_info so it appears in structure
        info = {id(self.child): _TaskInfoMeta("Child_spawn", "cid-1", self.child)}
        self._task = _TaskContainer(info)
        self._done = asyncio.Event()

    async def ask(self, question: str, *, parent_chat_context_cont=None):  # type: ignore[override]
        return self

    async def interject(self, message: str, **_):  # type: ignore[override]
        return None

    def stop(self, *_, **__):  # type: ignore[override]
        self.child.stop()
        self._done.set()
        return "stopped"

    def pause(self, *_, **__):  # type: ignore[override]
        return "paused"

    def resume(self, *_, **__):  # type: ignore[override]
        return "resumed"

    def done(self) -> bool:  # type: ignore[override]
        return self._done.is_set()

    async def result(self) -> str:  # type: ignore[override]
        await self._done.wait()
        return "nested done"

    async def next_clarification(self) -> dict:  # type: ignore[override]
        return {}

    async def next_notification(self) -> dict:  # type: ignore[override]
        return {}

    async def answer_clarification(self, call_id: str, answer: str) -> None:  # type: ignore[override]
        return None


class WrapperHandle(SteerableToolHandle):
    def __init__(self, h: ToyHandle):
        self._current_handle = h

    # Adopt standardized wrapper registration for nested_structure
    def get_wrapped_handles(self):  # type: ignore[override]
        return {"current": self._current_handle}

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


def _find_child(children: list[dict], tool_name: str) -> dict | None:
    for ch in children or []:
        if ch.get("tool_name") == tool_name:
            return ch
    return None


@pytest.mark.asyncio
@_handle_project
async def test_nested_structure_reports_child_tool_and_handle():
    inner = ToyHandle()

    async def Outer_spawn():  # type: ignore[valid-type]
        return inner

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
        await _wait_for_tool_request(client, "Outer_spawn")

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

        s = await outer.nested_structure()  # type: ignore[attr-defined]
        assert isinstance(s, dict)
        ch = _find_child(s.get("children", []), "Outer_spawn")
        assert ch is not None, "Expected Outer_spawn child in structure"
        h = ch.get("handle") or {}
        assert (h.get("class") == "ToyHandle") or (
            (h.get("label") or "").endswith("ToyHandle")
        )
        assert ch.get("origin") == "task_info"
        assert ch.get("state") in {"in_flight", "pending", "done"}
    finally:
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
async def test_nested_structure_reports_deep_hierarchy_via_task_info():
    nested = NestedHandle()

    async def Outer_spawn():  # type: ignore[valid-type]
        return nested

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
        await _wait_for_tool_request(client, "Outer_spawn")

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

        s = await outer.nested_structure()  # type: ignore[attr-defined]
        first = _find_child(s.get("children", []), "Outer_spawn")
        assert first is not None, "Expected Outer_spawn child"
        inner_node = first.get("handle") or {}
        deep = _find_child(inner_node.get("children", []), "Child_spawn")
        assert deep is not None, "Expected Child_spawn nested under Outer_spawn handle"
        deep_handle = deep.get("handle") or {}
        assert (deep_handle.get("class") == "ToyHandle") or (
            (deep_handle.get("label") or "").endswith("ToyHandle")
        )
    finally:
        try:
            outer.stop("cleanup")
        except Exception:
            pass
        try:
            await asyncio.wait_for(outer.result(), timeout=30.0)
        except Exception:
            pass
        try:
            nested.stop("cleanup")
        except Exception:
            pass


@pytest.mark.asyncio
@_handle_project
async def test_nested_structure_includes_wrapper_attribute_children():
    inner = ToyHandle()
    wrapped = WrapperHandle(inner)

    async def Outer_spawn():  # type: ignore[valid-type]
        return wrapped

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
        await _wait_for_tool_request(client, "Outer_spawn")

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

        s = await outer.nested_structure()  # type: ignore[attr-defined]
        first = _find_child(s.get("children", []), "Outer_spawn")
        assert first is not None, "Expected Outer_spawn child"
        inner_node = first.get("handle") or {}
        wrapper_children = inner_node.get("children", [])
        # Expect an entry discovered via standardized wrapper method
        assert any(
            (c.get("origin") == "wrapper")
            and str(c.get("wrapper_attr", "")).startswith("get_wrapped_handles")
            and (
                (c.get("handle") or {}).get("class") == "ToyHandle"
                or (c.get("handle") or {}).get("label", "").endswith("ToyHandle")
            )
            for c in wrapper_children
        ), "Expected wrapper child pointing at ToyHandle via get_wrapped_handles"
    finally:
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
