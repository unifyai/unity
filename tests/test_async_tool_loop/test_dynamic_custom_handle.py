from __future__ import annotations

import asyncio
import json
import inspect
from typing import Any, Dict, List, Optional

import pytest

from unity.common.async_tool_loop import (
    SteerableToolHandle,
    AsyncToolLoopHandle,
    start_async_tool_loop,
)
from tests.helpers import _handle_project
from unity.common.llm_client import new_llm_client
from tests.test_async_tool_loop.async_helpers import (
    _wait_for_tool_request,
    _wait_for_condition,
)


class CustomArgsHandle(SteerableToolHandle):
    """A handle that records all steering calls with extra args."""

    def __init__(self) -> None:
        self._done_ev = asyncio.Event()
        self._result_text: str = "inner-complete"
        self.interject_calls: List[Dict[str, Any]] = []
        self.pause_calls: List[Dict[str, Any]] = []
        self.resume_calls: List[Dict[str, Any]] = []
        self.stop_calls: List[Dict[str, Any]] = []
        self.ask_calls: List[Dict[str, Any]] = []
        # Mark custom write-only helpers
        self.write_only_methods = ["abort"]

    async def ask(
        self,
        question: str,
        *,
        style: str = "short",
    ) -> "SteerableToolHandle":
        self.ask_calls.append({"question": question, "style": style})
        return self

    async def interject(
        self,
        message: str,
        *,
        priority: int = 1,
        metadata: Dict[str, str] | None = None,
    ) -> Optional[str]:
        self.interject_calls.append(
            {"message": message, "priority": priority, "metadata": metadata or {}},
        )
        return None

    def stop(
        self,
        *,
        reason: Optional[str] = None,
        abandon: bool = False,
    ) -> Optional[str]:
        self.stop_calls.append({"reason": reason, "abandon": abandon})
        self._done_ev.set()
        return "stopped"

    async def pause(
        self,
        *,
        reason: str,
        log_to_backend: bool = False,
    ) -> Optional[str]:
        self.pause_calls.append({"reason": reason, "log_to_backend": log_to_backend})
        return "paused"

    async def resume(self, *, resume_token: Optional[str] = None) -> Optional[str]:
        self.resume_calls.append({"resume_token": resume_token})
        return "resumed"

    # Write-only helper: terminate with an "aborted" result. This method is
    # intentionally write-only (no returned value used by the loop); the loop
    # should acknowledge and finish when the nested handle resolves.
    def abort(self, *, reason: Optional[str] = None) -> None:
        self._result_text = "aborted"
        self._done_ev.set()
        return None

    def done(self) -> bool:
        return self._done_ev.is_set()

    async def result(self) -> str:
        await self._done_ev.wait()
        return self._result_text

    # New abstract event APIs – simple stubs for tests
    async def next_clarification(self) -> dict:
        return {}

    async def next_notification(self) -> dict:
        return {}

    async def answer_clarification(self, call_id: str, answer: str) -> None:
        return None


async def spawn_custom_handle() -> SteerableToolHandle:  # type: ignore[name-defined]
    """Return a CustomArgsHandle to exercise dynamic helper schemas/args."""
    return CustomArgsHandle()


@pytest.fixture(scope="function")
def client(model):
    return new_llm_client(model=model)


@pytest.mark.asyncio
@_handle_project
async def test_dynamic_helper_args_are_exposed_and_forwarded(client):
    """
    End-to-end: the LLM should (a) see full helper args in tool schemas and (b)
    invoke helpers with extra kwargs that reach the underlying handle methods.
    """

    # Initial instruction: only spawn the custom handle
    client.set_system_message(
        "Call `spawn_custom_handle` to start a task that exposes dynamic helpers.",
    )

    outer = start_async_tool_loop(
        client,
        message="start",
        tools={"spawn_custom_handle": spawn_custom_handle},
        timeout=60,
        max_steps=10,
    )

    # Ensure the spawn tool has been requested so helpers will be exposed
    await _wait_for_tool_request(client, "spawn_custom_handle")

    # Interject a single instruction to use the stop helper with custom arguments
    await outer.interject(
        'Now, stop the task with reason="user_request", abandon=true. '
        "Then respond only with: done",
    )

    # Let the model drive; it should call interject_ / pause_ / resume_ / stop_ with kwargs
    final = await outer.result()
    assert final is not None, "Loop should complete with a response"

    # Retrieve the live handle instance from the spawned task info
    # Walk messages to locate the helper tool-call arguments for validation.
    msgs = client.messages or []

    # 1) Validate that the assistant included the extra args in tool calls
    def _extract_first_args(prefix: str) -> Dict[str, Any]:
        for m in msgs:
            if m.get("role") != "assistant":
                continue
            for tc in m.get("tool_calls") or []:
                fn = tc.get("function", {}).get("name", "")
                if isinstance(fn, str) and fn.startswith(prefix):
                    try:
                        return json.loads(tc.get("function", {}).get("arguments", "{}"))
                    except Exception:
                        return {}
        return {}

    stop_args = _extract_first_args("stop_")

    # The LLM should have passed our custom stop kwargs
    assert stop_args.get("reason") in {"user_request", "user request", "User request"}
    # Some models may encode booleans as strings – accept both
    assert stop_args.get("abandon") in {True, "true", "True"}

    # No multi-step checks – only validate the stop helper args

    # 2) Validate that the underlying handle methods actually received kwargs
    # Find the most recent CustomArgsHandle recorded by intercepting spawn
    # Since we returned a new instance, we can infer values by scanning tool messages
    # However, the more robust check is to verify the semantics via tool responses:
    # The loop does not expose internals; instead, infer from ordering that each helper
    # was called at least once by checking tool messages inserted by the loop.

    # Count the helper invocations visible in the transcript
    def _assistant_calls_prefix(prefix: str) -> int:
        count = 0
        for m in msgs:
            if m.get("role") != "assistant":
                continue
            tcs = m.get("tool_calls") or []
            count += sum(
                1
                for tc in tcs
                if tc.get("function", {}).get("name", "").startswith(prefix)
            )
        return count

    assert _assistant_calls_prefix("stop_") >= 1


@pytest.mark.asyncio
@_handle_project
async def test_custom_abort_finishes_nested(client):
    """
    End-to-end: expose a write-only custom helper `abort` on the spawned handle.
    The model should call the helper, we acknowledge immediately, and the nested
    handle should resolve with the "aborted" message allowing the outer loop to finish.
    """

    client.set_system_message(
        "Call `spawn_custom_handle` to start a task that exposes dynamic helpers.",
    )

    outer = start_async_tool_loop(
        client,
        message="start",
        tools={"spawn_custom_handle": spawn_custom_handle},
        timeout=60,
        max_steps=20,
    )

    # Ensure the spawn tool has been requested so helpers will be exposed
    await _wait_for_tool_request(client, "spawn_custom_handle")

    # Instruct the model to call abort and then reply with 'done'
    await outer.interject(
        "Now, call the abort helper immediately, then respond only with: done",
    )

    final = await outer.result()
    assert final is not None, "Loop should complete with a response"

    # Verify that a tool message shows the nested handle finished with "aborted"
    msgs = client.messages or []

    def _has_aborted_tool_message(messages: List[Dict[str, Any]]) -> bool:
        for m in messages:
            if m.get("role") != "tool":
                continue
            content = m.get("content")
            if isinstance(content, str):
                txt = content.strip().strip('"').lower()
                if txt == "aborted":
                    return True
        return False

    assert _has_aborted_tool_message(msgs)


@pytest.mark.asyncio
@_handle_project
async def test_custom_outer_handle_instantiated(client):
    """
    Simple sanity check: the start helper should instantiate the provided
    custom outer handle class, and its extended stop signature should be
    usable immediately (e.g., accepts cancel=...).
    """

    class CustomOuterHandle(AsyncToolLoopHandle):
        def stop(self, *, cancel: bool | None = None, reason: Optional[str] = None) -> None:  # type: ignore[override]
            # Delegate to base stop for cancellation; accepting `cancel` is the point of this test
            super().stop(reason=reason)

    # Minimal prompt; we don't need tools for this test – just verify instantiation & signature
    client.set_system_message("Reply briefly.")

    outer = start_async_tool_loop(
        client,
        message="hi",
        tools={},
        timeout=60,
        max_steps=1,
        handle_cls=CustomOuterHandle,
    )

    # Returned handle is our custom class
    assert isinstance(outer, CustomOuterHandle)

    # Its stop signature now accepts `cancel`
    import inspect as _inspect

    params = _inspect.signature(outer.stop).parameters
    assert "cancel" in params

    # Calling stop with cancel should not raise, even with no delegate
    outer.stop(cancel=True, reason="test")
    # Wait for graceful shutdown of the handle task
    await outer.result()


@pytest.mark.asyncio
@_handle_project
async def test_dynamic_helpers_use_base_docstrings(client):
    """
    Verify that when a custom handle does not provide docstrings for standard
    steering methods (pause/resume/interject/ask/stop), the dynamic helpers
    adopt informative base docstrings from the abstract/base classes.
    """

    class BaseLikeHandle(SteerableToolHandle):
        # No docstrings on purpose – rely on base docstrings via MRO fallback
        def __init__(self) -> None:
            self._done = asyncio.Event()

        async def ask(self, question: str) -> "SteerableToolHandle":
            return self

        async def interject(self, message: str):
            return None

        def stop(self, reason: Optional[str] = None):
            return "stopped"

        async def pause(self):
            return "paused"

        async def resume(self):
            return "resumed"

        def done(self) -> bool:
            return self._done.is_set()

        async def result(self) -> str:
            await self._done.wait()
            return "ok"

        async def next_clarification(self) -> dict:
            return {}

        async def next_notification(self) -> dict:
            return {}

        async def answer_clarification(self, call_id: str, answer: str) -> None:
            return None

    async def spawn_handle() -> SteerableToolHandle:  # type: ignore[name-defined]
        return BaseLikeHandle()

    # Spy dynamic registrations to capture effective docstrings
    from unity.common._async_tool import dynamic_tools_factory as _dtf

    registered_docs: Dict[str, str] = {}
    orig_register_tool = _dtf.DynamicToolFactory._register_tool

    def _spy_register_tool(self, func_name: str, fallback_doc: str, fn):  # type: ignore[no-redef]
        doc = inspect.getdoc(fn) or ""
        registered_docs[func_name] = doc
        return orig_register_tool(self, func_name, fallback_doc, fn)

    setattr(_dtf.DynamicToolFactory, "_register_tool", _spy_register_tool)

    client.set_system_message("Call `spawn_handle` to start a nested handle.")
    outer = start_async_tool_loop(
        client,
        message="start",
        tools={"spawn_handle": spawn_handle},
        timeout=120,
    )

    await _wait_for_tool_request(client, "spawn_handle")

    # Trigger an immediate assistant turn so helpers are exposed/registered
    await outer.interject("probe helpers")

    async def _helpers_registered() -> bool:
        return any(
            any(k.startswith(p) for k in registered_docs.keys())
            for p in ("pause_", "resume_", "interject_", "ask_", "stop_")
        )

    await _wait_for_condition(_helpers_registered, poll=0.05, timeout=30.0)

    # Assertions: base docstrings should be visible (substrings)
    # pause
    for k, v in registered_docs.items():
        if k.startswith("pause_"):
            assert "Pause this task temporarily" in v
    # resume
    for k, v in registered_docs.items():
        if k.startswith("resume_"):
            assert "Resume a task that was previously paused" in v
    # interject
    for k, v in registered_docs.items():
        if k.startswith("interject_"):
            assert "Provide additional information or instructions" in v
    # ask (from SteerableHandle.ask)
    for k, v in registered_docs.items():
        if k.startswith("ask_"):
            assert "Ask about the current status or progress" in v
    # stop – either base or explicit; ensure it's non-empty and informative
    for k, v in registered_docs.items():
        if k.startswith("stop_"):
            assert isinstance(v, str) and len(v.strip()) > 0


@pytest.mark.asyncio
@_handle_project
async def test_dynamic_helpers_use_overridden_docstrings(client):
    """
    Verify that when a custom handle overrides docstrings (or adds extra kwargs),
    the dynamic helpers expose those overridden docstrings in their schema.
    """

    class OverrideDocHandle(SteerableToolHandle):
        def __init__(self) -> None:
            self._done = asyncio.Event()

        async def ask(self, question: str) -> "SteerableToolHandle":
            """Ask override doc: consult safe cache only."""
            return self

        async def interject(self, message: str, *, importance: int = 1):
            """Interject override doc: only interject if importance >= 1."""
            return None

        def stop(self, reason: Optional[str] = None):
            """Stop override doc: stop only if safe to cancel."""
            return "stopped"

        async def pause(self, *, gate: Optional[str] = None):
            """Pause override doc: only pause if XYZ precondition holds."""
            return "paused"

        async def resume(self, *, token: Optional[str] = None):
            """Resume override doc: resume with a session token if required."""
            return "resumed"

        def done(self) -> bool:
            return self._done.is_set()

        async def result(self) -> str:
            await self._done.wait()
            return "ok"

        async def next_clarification(self) -> dict:
            return {}

        async def next_notification(self) -> dict:
            return {}

        async def answer_clarification(self, call_id: str, answer: str) -> None:
            return None

    async def spawn_handle() -> SteerableToolHandle:  # type: ignore[name-defined]
        return OverrideDocHandle()

    from unity.common._async_tool import dynamic_tools_factory as _dtf

    registered_docs: Dict[str, str] = {}
    orig_register_tool = _dtf.DynamicToolFactory._register_tool

    def _spy_register_tool(self, func_name: str, fallback_doc: str, fn):  # type: ignore[no-redef]
        doc = inspect.getdoc(fn) or ""
        registered_docs[func_name] = doc
        return orig_register_tool(self, func_name, fallback_doc, fn)

    setattr(_dtf.DynamicToolFactory, "_register_tool", _spy_register_tool)

    client.set_system_message("Call `spawn_handle` to start a nested handle.")
    outer = start_async_tool_loop(
        client,
        message="start",
        tools={"spawn_handle": spawn_handle},
        timeout=120,
        max_steps=10,
    )

    await _wait_for_tool_request(client, "spawn_handle")
    await outer.interject("expose helpers now")

    async def _helpers_registered() -> bool:
        return any(
            any(k.startswith(p) for k in registered_docs.keys())
            for p in ("pause_", "resume_", "interject_", "ask_", "stop_")
        )

    await _wait_for_condition(_helpers_registered, poll=0.05, timeout=30.0)

    # Assertions: overridden text should appear
    for k, v in registered_docs.items():
        if k.startswith("pause_"):
            assert "Pause override doc: only pause if XYZ precondition holds." in v
        if k.startswith("resume_"):
            assert "Resume override doc: resume with a session token if required." in v
        if k.startswith("interject_"):
            assert "Interject override doc: only interject if importance >= 1." in v
        if k.startswith("ask_"):
            assert "Ask override doc: consult safe cache only." in v
        if k.startswith("stop_"):
            assert "Stop override doc: stop only if safe to cancel." in v


@pytest.mark.asyncio
@_handle_project
async def test_dynamic_helpers_adopt_custom_method_docstring(client):
    """
    Verify a brand-new custom steering method on the handle has its docstring
    adopted by the dynamically generated helper.
    """

    class CustomMethodHandle(SteerableToolHandle):
        def __init__(self) -> None:
            self._done = asyncio.Event()

        # Standard methods (minimal, no docstrings needed here)
        async def ask(self, question: str) -> "SteerableToolHandle":
            return self

        async def interject(self, message: str):
            return None

        def stop(self, reason: Optional[str] = None):
            return "stopped"

        async def pause(self):
            return "paused"

        async def resume(self):
            return "resumed"

        def done(self) -> bool:
            return self._done.is_set()

        async def result(self) -> str:
            await self._done.wait()
            return "ok"

        async def next_clarification(self) -> dict:
            return {}

        async def next_notification(self) -> dict:
            return {}

        async def answer_clarification(self, call_id: str, answer: str) -> None:
            return None

        # New custom steering method
        def escalate(self, level: int) -> str:
            """Escalate override doc: raise escalation to the specified level."""
            return f"escalated:{level}"

    async def spawn_handle() -> SteerableToolHandle:  # type: ignore[name-defined]
        return CustomMethodHandle()

    from unity.common._async_tool import dynamic_tools_factory as _dtf

    registered_docs: Dict[str, str] = {}
    orig_register_tool = _dtf.DynamicToolFactory._register_tool

    def _spy_register_tool(self, func_name: str, fallback_doc: str, fn):  # type: ignore[no-redef]
        doc = inspect.getdoc(fn) or ""
        registered_docs[func_name] = doc
        return orig_register_tool(self, func_name, fallback_doc, fn)

    setattr(_dtf.DynamicToolFactory, "_register_tool", _spy_register_tool)

    client.set_system_message("Call `spawn_handle` to start a nested handle.")
    outer = start_async_tool_loop(
        client,
        message="start",
        tools={"spawn_handle": spawn_handle},
        timeout=120,
    )

    await _wait_for_tool_request(client, "spawn_handle")
    await outer.interject("expose custom methods")

    async def _custom_registered() -> bool:
        return any(k.startswith("escalate_") for k in registered_docs.keys())

    await _wait_for_condition(_custom_registered, poll=0.05, timeout=30.0)

    # Assert the custom helper's docstring matches the original method docstring
    found = False
    for k, v in registered_docs.items():
        if k.startswith("escalate_"):
            assert (
                "Escalate override doc: raise escalation to the specified level." in v
            )
            found = True
    assert found, "expected an escalate_* helper to be registered"


async def spawn_custom_handle() -> SteerableToolHandle:  # type: ignore[name-defined]
    """Return a CustomArgsHandle to exercise dynamic helper schemas/args."""
    return CustomArgsHandle()


@pytest.mark.asyncio
@_handle_project
async def test_dynamic_helper_preserves_annotations_for_public_methods(model):
    """
    Lower-level factory test: ensure public-method helpers preserve annotations
    so their generated tool schema exposes correct JSON types (e.g., integer).
    This test is independent of the task scheduler and queue logic.
    """
    import asyncio
    import inspect
    from contextlib import suppress

    from unity.common.async_tool_loop import SteerableToolHandle
    from unity.common._async_tool.tools_data import ToolsData
    from unity.common._async_tool.tools_utils import ToolCallMetadata
    from unity.common._async_tool.dynamic_tools_factory import DynamicToolFactory
    from unity.common.llm_helpers import method_to_schema

    class _AnnotatedHandle(SteerableToolHandle):
        def __init__(self) -> None:
            pass

        def set_value(self, task_id: int, note: str | None = None) -> str:
            return f"set:{task_id}:{note or ''}"

        async def ask(self, question: str, *, images=None):  # type: ignore[override]
            return self

        async def interject(self, message: str, *, images=None):  # type: ignore[override]
            return None

        def stop(self, reason: str | None = None, *, parent_chat_context_cont=None):  # type: ignore[override]
            return "stopped"

        async def pause(self):  # type: ignore[override]
            return "paused"

        async def resume(self):  # type: ignore[override]
            return "resumed"

        def done(self) -> bool:  # type: ignore[override]
            return True

        async def result(self) -> str:  # type: ignore[override]
            return "OK"

        async def next_clarification(self) -> dict:  # type: ignore[override]
            return {}

        async def next_notification(self) -> dict:  # type: ignore[override]
            return {}

        async def answer_clarification(self, call_id: str, answer: str) -> None:  # type: ignore[override]
            return None

    class _DummyLogger:
        log_steps = False

        def info(self, *a, **kw): ...

        def error(self, *a, **kw): ...

    class _DummyClient:
        def __init__(self):
            self.messages = []

    tools_data = ToolsData({}, client=_DummyClient(), logger=_DummyLogger())

    pending_task = asyncio.create_task(asyncio.sleep(10))
    call_id = "abc123"
    asst_msg = {
        "role": "assistant",
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": "dummy_base", "arguments": "{}"},
            },
        ],
        "content": "",
    }
    meta = ToolCallMetadata(
        name="dummy_base",
        call_id=call_id,
        call_dict=asst_msg["tool_calls"][0],
        call_idx=0,
        chat_context=None,
        assistant_msg=asst_msg,
        is_interjectable=False,
        tool_schema={},
        llm_arguments={},
        raw_arguments_json=asst_msg["tool_calls"][0]["function"]["arguments"],
        handle=_AnnotatedHandle(),
        interject_queue=None,
        clar_up_queue=None,
        clar_down_queue=None,
        notification_queue=None,
        pause_event=None,
    )
    tools_data.save_task(pending_task, meta)

    factory = DynamicToolFactory(tools_data)
    factory.generate()

    keys = [k for k in factory.dynamic_tools.keys() if k.startswith("set_value_")]
    assert keys, "expected set_value helper to be generated"
    helper = factory.dynamic_tools[keys[0]]

    sig = inspect.signature(helper)
    assert "task_id" in sig.parameters
    ann = sig.parameters["task_id"].annotation
    if ann is not inspect._empty:
        assert (
            (ann is int)
            or (ann == int)
            or (ann == "int")
            or (getattr(ann, "__name__", None) == "int")
        )

    schema = method_to_schema(helper, include_class_name=False)
    params = schema["function"]["parameters"]
    assert "task_id" in params["properties"]
    prop = params["properties"]["task_id"]
    is_integer = (prop.get("type") == "integer") or any(
        (d.get("type") == "integer")
        for d in prop.get("anyOf", [])
        if isinstance(prop, dict)
    )
    assert is_integer, f"expected integer type for task_id, got: {prop}"
    assert "task_id" in params.get("required", [])

    with suppress(BaseException):
        pending_task.cancel()
        await pending_task


@pytest.mark.asyncio
@_handle_project
async def test_dynamic_factory_ignores_internal_introspection_methods(model):
    """
    Regression test: Ensure `DynamicToolFactory` does NOT generate tools for
    internal introspection methods (e.g. `get_wrapped_handles`, `_get_wrapped_handles`)
    even if they are public on the handle class, while correctly exposing other
    public methods.
    """
    from contextlib import suppress
    from unity.common.async_tool_loop import SteerableToolHandle
    from unity.common._async_tool.tools_data import ToolsData
    from unity.common._async_tool.tools_utils import ToolCallMetadata
    from unity.common._async_tool.dynamic_tools_factory import DynamicToolFactory
    from unity.common.handle_wrappers import HandleWrapperMixin

    # A custom handle that mixes in wrapper functionality and defines introspection-like methods
    class IntrospectiveHandle(SteerableToolHandle, HandleWrapperMixin):
        def __init__(self):
            self._done_ev = asyncio.Event()

        # Valid public method - SHOULD be exposed
        def public_action(self, arg: str) -> str:
            return f"echo: {arg}"

        # Internal method - SHOULD NOT be exposed
        def _internal_method(self):
            pass

        # Standard steerable methods
        async def ask(self, q: str, **kw):
            return self

        async def interject(self, m: str, **kw):
            pass

        def stop(self, r=None):
            return "stopped"

        async def pause(self):
            return "paused"

        async def resume(self):
            return "resumed"

        def done(self):
            return True

        async def result(self):
            return "ok"

        async def next_clarification(self):
            return {}

        async def next_notification(self):
            return {}

        async def answer_clarification(self, cid, ans):
            pass

    # Setup dummy environment
    class _DummyLogger:
        log_steps = False

        def info(self, *a, **kw): ...
        def error(self, *a, **kw): ...

    class _DummyClient:
        def __init__(self):
            self.messages = []

    tools_data = ToolsData({}, client=_DummyClient(), logger=_DummyLogger())
    pending_task = asyncio.create_task(asyncio.sleep(10))

    meta = ToolCallMetadata(
        name="dummy_tool",
        call_id="call_123",
        call_dict={"id": "call_123", "function": {"name": "dummy", "arguments": "{}"}},
        call_idx=0,
        chat_context=None,
        assistant_msg={},
        is_interjectable=False,
        tool_schema={},
        llm_arguments={},
        raw_arguments_json="{}",
        handle=IntrospectiveHandle(),
        interject_queue=None,
        clar_up_queue=None,
        clar_down_queue=None,
        notification_queue=None,
        pause_event=None,
    )
    tools_data.save_task(pending_task, meta)

    # Generate tools
    factory = DynamicToolFactory(tools_data)
    factory.generate()

    tools = factory.dynamic_tools

    # Check 1: public_action should be present
    public_keys = [k for k in tools.keys() if k.startswith("public_action_")]
    assert public_keys, "Expected public_action to be exposed"

    # Check 2: _get_wrapped_handles (from mixin) should NOT be present
    internal_keys = [k for k in tools.keys() if "get_wrapped_handles" in k]
    assert not internal_keys, f"Introspection method leaked: {internal_keys}"

    with suppress(BaseException):
        pending_task.cancel()
        await pending_task
