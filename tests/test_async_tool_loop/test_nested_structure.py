import asyncio

import pytest

from unity.common.async_tool_loop import (
    start_async_tool_loop,
    SteerableToolHandle,
    SteerableHandle,
    _nested_structure_on,
)
from tests.helpers import _handle_project
from unity.common.llm_client import new_llm_client
from tests.test_async_tool_loop.async_helpers import (
    _wait_for_tool_request,
    _wait_for_condition,
)


class _TaskInfoMeta:
    def __init__(self, name: str, call_id: str, handle):
        self.name = name
        self.call_id = call_id
        self.handle = handle


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

    async def pause(self, *_, **__):  # type: ignore[override]
        return "paused"

    async def resume(self, *_, **__):  # type: ignore[override]
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

    async def pause(self, *_, **__):  # type: ignore[override]
        return "paused"

    async def resume(self, *_, **__):  # type: ignore[override]
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
    def _get_wrapped_handles(self):  # type: ignore[override]
        return {"current": self._current_handle}

    async def ask(self, question: str, *, parent_chat_context_cont=None):  # type: ignore[override]
        return self

    async def interject(self, message: str, **_):  # type: ignore[override]
        return None

    def stop(self, *_, **__):  # type: ignore[override]
        return self._current_handle.stop()

    async def pause(self, *_, **__):  # type: ignore[override]
        return await self._current_handle.pause()

    async def resume(self, *_, **__):  # type: ignore[override]
        return await self._current_handle.resume()

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


def _base_name(val: str | None) -> str:
    s = str(val) if val is not None else ""
    return s.split("(", 1)[0]


def _find_child_by_handle(children: list[dict], handle_name: str) -> dict | None:
    for ch in children or []:
        h = ch.get("handle")
        t = ch.get("tool")
        if _base_name(h) == handle_name or _base_name(t) == handle_name:
            return ch
    return None


@pytest.mark.asyncio
@_handle_project
async def test_nested_structure_reports_child_tool_and_handle(model):
    inner = ToyHandle()

    async def Outer_spawn():  # type: ignore[valid-type]
        return inner

    client = new_llm_client(model=model)
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
        # Root should resolve to the async loop handle sentinel
        assert s.get("handle") == "AsyncToolLoopHandle"
        assert isinstance(s, dict)
        # Minimal format: direct child node for the returned handle
        ch = _find_child_by_handle(s.get("children", []), "ToyHandle")
        assert ch is not None, "Expected ToyHandle child in structure"
        # Child should itself be a minimal node with no further children in this test
        assert (
            _base_name(ch.get("handle")) == "ToyHandle"
            or _base_name(ch.get("tool")) == "ToyHandle"
        )
        # And the sentinel encountered for ToyHandle should be SteerableToolHandle
        assert "SteerableToolHandle" in str(ch.get("handle"))
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
async def test_nested_structure_reports_deep_hierarchy(model):
    nested = NestedHandle()

    async def Outer_spawn():  # type: ignore[valid-type]
        return nested

    client = new_llm_client(model=model)
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
        # First layer: NestedHandle
        first = _find_child_by_handle(s.get("children", []), "NestedHandle")
        assert first is not None, "Expected NestedHandle child"
        # Second layer under NestedHandle: ToyHandle
        deep = _find_child_by_handle(first.get("children", []), "ToyHandle")
        assert deep is not None, "Expected ToyHandle nested under NestedHandle"
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
async def test_nested_structure_includes_wrapper_attribute_children(model):
    inner = ToyHandle()
    wrapped = WrapperHandle(inner)

    async def Outer_spawn():  # type: ignore[valid-type]
        return wrapped

    client = new_llm_client(model=model)
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
        # First layer: WrapperHandle
        first = _find_child_by_handle(s.get("children", []), "WrapperHandle")
        assert first is not None, "Expected WrapperHandle child"
        # The wrapper should expose ToyHandle as a nested live child
        wrapper_children = first.get("children", [])
        assert any(
            (_base_name(c.get("handle")) == "ToyHandle")
            or (_base_name(c.get("tool")) == "ToyHandle")
            for c in wrapper_children
        ), "Expected ToyHandle child discovered via wrapper get_wrapped_handles"
    finally:
        try:
            outer.stop("cleanup")
        except Exception:
            pass


@pytest.mark.asyncio
async def test_handle_chain_includes_steerable_handle_sentinel():
    class DirectSH(SteerableHandle):
        async def ask(  # type: ignore[override]
            self,
            question: str,
            *,
            parent_chat_context_cont=None,
            images=None,
        ):
            return self

        async def interject(  # type: ignore[override]
            self,
            message: str,
            *,
            parent_chat_context_cont=None,
        ):
            return None

    h = DirectSH()
    s = await _nested_structure_on(h)
    # The handle string should include the SteerableHandle sentinel for a direct subclass
    assert s.get("handle") == "DirectSH(SteerableHandle)"


@pytest.mark.asyncio
async def test_handle_chain_canonicalizes_leaf_and_drops_base():
    class BaseCustomHandle(SteerableHandle):
        async def ask(  # type: ignore[override]
            self,
            question: str,
            *,
            parent_chat_context_cont=None,
            images=None,
        ):
            return self

        async def interject(  # type: ignore[override]
            self,
            message: str,
            *,
            parent_chat_context_cont=None,
        ):
            return None

    class V3CustomHandle(BaseCustomHandle):
        pass

    h = V3CustomHandle()
    s = await _nested_structure_on(h)
    # Leaf "V3CustomHandle" → canonicalized to "CustomHandle", "BaseCustomHandle" is dropped, sentinel included
    assert s.get("handle") == "CustomHandle(SteerableHandle)"


@pytest.mark.asyncio
async def test_handle_chain_canonicalizes_simulated_leaf():
    class BaseCustomHandle(SteerableHandle):
        async def ask(  # type: ignore[override]
            self,
            question: str,
            *,
            parent_chat_context_cont=None,
            images=None,
        ):
            return self

        async def interject(  # type: ignore[override]
            self,
            message: str,
            *,
            parent_chat_context_cont=None,
        ):
            return None

    class SimulatedCustomHandle(BaseCustomHandle):
        pass

    h = SimulatedCustomHandle()
    s = await _nested_structure_on(h)
    # Leaf "SimulatedCustomHandle" → canonicalized to "CustomHandle", base dropped, sentinel included
    assert s.get("handle") == "CustomHandle(SteerableHandle)"


@pytest.mark.asyncio
async def test_handle_chain_canonicalizes_intermediate_and_drops_base():
    class BaseBaz(SteerableHandle):
        async def ask(  # type: ignore[override]
            self,
            question: str,
            *,
            parent_chat_context_cont=None,
            images=None,
        ):
            return self

        async def interject(  # type: ignore[override]
            self,
            message: str,
            *,
            parent_chat_context_cont=None,
        ):
            return None

    class SimulatedBar(BaseBaz):
        pass

    class V2Foo(SimulatedBar):
        pass

    h = V2Foo()
    s = await _nested_structure_on(h)
    # Leaf "V2Foo" → "Foo"; intermediate "SimulatedBar" → "Bar"; "BaseBaz" dropped; sentinel included
    assert s.get("handle") == "Foo(Bar(SteerableHandle))"


@pytest.mark.asyncio
async def test_tool_name_canonicalizes_simulated_prefix():
    class Dummy:
        def __init__(self):
            self._loop_id = "SimulatedSomethingManager.ask"

    d = Dummy()
    s = await _nested_structure_on(d)
    # Tool field should canonicalize class segment: SimulatedX → X
    assert s.get("tool") == "SomethingManager.ask"


@pytest.mark.asyncio
async def test_tool_name_canonicalizes_base_prefix():
    class Dummy:
        def __init__(self):
            self._loop_id = "BaseAnotherManager.ask"

    d = Dummy()
    s = await _nested_structure_on(d)
    # Tool field should canonicalize class segment: BaseX → X
    assert s.get("tool") == "AnotherManager.ask"
