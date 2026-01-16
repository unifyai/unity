import asyncio

import pytest

from unity.common.async_tool_loop import _nested_structure_on


class _TaskInfoMeta:
    def __init__(self, name: str, call_id: str, handle):
        self.name = name
        self.call_id = call_id
        self.handle = handle


class _TaskContainer:
    def __init__(self, info: dict):
        self.task_info = info


class ToyHandle:
    def __init__(self) -> None:
        self._done = asyncio.Event()

    async def ask(self, question: str, *, parent_chat_context_cont=None):
        return self

    async def interject(self, message: str, **_):
        return None

    def stop(self, *_, **__):
        self._done.set()
        return "stopped"

    async def pause(self, *_, **__):
        return "paused"

    async def resume(self, *_, **__):
        return "resumed"

    def done(self) -> bool:
        return self._done.is_set()

    async def result(self) -> str:
        await self._done.wait()
        return "inner done"

    async def next_clarification(self) -> dict:
        return {}

    async def next_notification(self) -> dict:
        return {}

    async def answer_clarification(self, call_id: str, answer: str) -> None:
        return None


class SteeringHandle:
    """A handle that steers a deeper loop via task_info."""

    def __init__(self) -> None:
        self._done = asyncio.Event()
        self._inner = ToyHandle()
        info = {id(self._inner): _TaskInfoMeta("Final_loop", "cid-final", self._inner)}
        self._task = _TaskContainer(info)

    async def ask(self, question: str, *, parent_chat_context_cont=None):
        return self

    async def interject(self, message: str, **_):
        return None

    def stop(self, *_, **__):
        self._inner.stop()
        self._done.set()
        return "stopped"

    async def pause(self, *_, **__):
        return "paused"

    async def resume(self, *_, **__):
        return "resumed"

    def done(self) -> bool:
        return self._done.is_set()

    async def result(self) -> str:
        await self._done.wait()
        return "steering done"


class WrapperWithMethod:
    """Wrapper that uses the standardized _get_wrapped_handles method."""

    def __init__(self, handle):
        self._handles = [handle]

    def _get_wrapped_handles(self):
        # Demonstrate dict form support too
        return {"actor": self._handles[0]}


from unity.common.handle_wrappers import HandleWrapperMixin


def _base_name(val: str | None) -> str:
    s = str(val) if val is not None else ""
    return s.split("(", 1)[0]


class WrapperWithMixin(HandleWrapperMixin):
    """Wrapper that uses the mixin and _wrap_handle registration."""

    def __init__(self, handle):
        self._wrap_handle(handle)


def _find_child(
    children: list[dict],
    *,
    origin: str = None,
    wrapper_attr: str = None,
    tool_name: str = None,
):
    for ch in children or []:
        if origin is not None and ch.get("origin") != origin:
            continue
        if wrapper_attr is not None and ch.get("wrapper_attr") != wrapper_attr:
            continue
        if tool_name is not None and _base_name(ch.get("tool_name")) != tool_name:
            continue
        return ch
    return None


@pytest.mark.asyncio
async def test_nested_structure_with_get_wrapped_handles_method():
    inner = SteeringHandle()
    wrapped = WrapperWithMethod(inner)

    s = await _nested_structure_on(wrapped)

    # Minimal structure: child node directly represents SteeringHandle
    wchild = None
    for ch in s.get("children", []):
        if (_base_name(ch.get("handle")) == "SteeringHandle") or (
            _base_name(ch.get("tool")) == "SteeringHandle"
        ):
            wchild = ch
            break
    assert (
        wchild is not None
    ), "Expected wrapped SteeringHandle child via get_wrapped_handles"

    # The wrapped SteeringHandle should itself steer a deeper loop; its child must be ToyHandle
    deep_child = None
    for ch in wchild.get("children", []):
        if (_base_name(ch.get("handle")) == "ToyHandle") or (
            _base_name(ch.get("tool")) == "ToyHandle"
        ):
            deep_child = ch
            break
    assert deep_child is not None, "Expected ToyHandle nested under SteeringHandle"


@pytest.mark.asyncio
async def test_nested_structure_with_mixin_registration():
    inner = SteeringHandle()
    wrapped = WrapperWithMixin(inner)

    s = await _nested_structure_on(wrapped)

    # Minimal structure: child node directly represents SteeringHandle
    wchild = None
    for ch in s.get("children", []):
        if (_base_name(ch.get("handle")) == "SteeringHandle") or (
            _base_name(ch.get("tool")) == "SteeringHandle"
        ):
            wchild = ch
            break
    assert (
        wchild is not None
    ), "Expected SteeringHandle child discovered via mixin get_wrapped_handles"

    # And nested ToyHandle under SteeringHandle
    deep_child = None
    for ch in wchild.get("children", []):
        if (_base_name(ch.get("handle")) == "ToyHandle") or (
            _base_name(ch.get("tool")) == "ToyHandle"
        ):
            deep_child = ch
            break
    assert deep_child is not None, "Expected ToyHandle nested under SteeringHandle"
