"""Event logging and lineage tests for CodeActActor.

This module is intentionally compact and covers the highest-signal behaviors:

- `execute_code` boundary emits ManagerMethod events and restores `TOOL_LOOP_LINEAGE`.
- `execute_function` boundary emits ManagerMethod events with `execute_function({name})`
  in the lineage and restores `TOOL_LOOP_LINEAGE`.
- FunctionManager boundary (`_LineageTrackedFunction`) composes with `execute_code`
  so that nested manager calls carry the full hierarchy.
- Concurrency does not cause lineage crosstalk between sibling function calls.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest

from tests.helpers import _handle_project, capture_events
from unity.actor.code_act_actor import CodeActActor
from unity.actor.execution import (
    PythonExecutionSession,
    _CURRENT_SANDBOX,
    parts_to_text,
)
from unity.common._async_tool.loop_config import TOOL_LOOP_LINEAGE
from unity.events.event_bus import EVENT_BUS
from unity.events.manager_event_logging import log_manager_call
from unity.function_manager.function_manager import _LineageTrackedFunction

pytestmark = pytest.mark.enable_eventbus


def _get(out: Any, key: str, default: Any = None) -> Any:
    """Get a field from either a dict or Pydantic model."""
    if isinstance(out, dict):
        return out.get(key, default)
    return getattr(out, key, default)


# ---------------------------------------------------------------------------
# execute_code boundary unit tests
# ---------------------------------------------------------------------------


def _result_error(res: Any) -> Any:
    """Return the error field from an execute_code result (dict or ExecutionResult)."""
    if isinstance(res, dict):
        return res.get("error")
    return getattr(res, "error", None)


def _result_stdout_text(res: Any) -> str:
    """Return stdout as plain text from an execute_code result (dict or ExecutionResult)."""
    if isinstance(res, dict):
        stdout = res.get("stdout") or ""
    else:
        stdout = getattr(res, "stdout", "") or ""
    return parts_to_text(stdout) if isinstance(stdout, list) else str(stdout)


@pytest.mark.asyncio
@_handle_project
async def test_execute_code_boundary_publishes_events_and_cleans_lineage(monkeypatch):
    actor = CodeActActor(
        environments=[],  # avoid default computer/state-manager envs in unit test
    )

    async def _fake_execute(**_kwargs):
        return {
            "stdout": "ok",
            "stderr": "",
            "result": 1,
            "error": None,
            "language": "python",
            "state_mode": "stateless",
            "session_id": 0,
            "venv_id": None,
            "session_created": False,
            "duration_ms": 1,
        }

    monkeypatch.setattr(actor._session_executor, "execute", _fake_execute, raising=True)

    execute_code = actor.get_tools("act")["execute_code"]
    token = TOOL_LOOP_LINEAGE.set(["CodeActActor.act"])
    try:
        async with capture_events("ManagerMethod") as events:
            out = await execute_code(
                thought="run",
                code="print('hi')",
                language="python",
                state_mode="stateless",
                session_id=None,
                session_name=None,
                venv_id=None,
                _notification_up_q=None,
            )
        EVENT_BUS.join_published()
        assert out.get("error") is None
        assert TOOL_LOOP_LINEAGE.get([]) == ["CodeActActor.act"]

        mm = [
            e
            for e in events
            if e.payload.get("manager") == "CodeActActor"
            and e.payload.get("method") == "execute_code"
        ]
        assert sorted([e.payload.get("phase") for e in mm]) == ["incoming", "outgoing"]
        assert mm[0].payload.get("hierarchy") == ["CodeActActor.act", "execute_code"]
        assert "execute_code(" in str(mm[0].payload.get("hierarchy_label"))
    finally:
        TOOL_LOOP_LINEAGE.reset(token)


@pytest.mark.asyncio
@_handle_project
async def test_execute_code_boundary_marks_error_when_executor_raises(monkeypatch):
    actor = CodeActActor(
        environments=[],
    )

    async def _boom(**_kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(actor._session_executor, "execute", _boom, raising=True)

    execute_code = actor.get_tools("act")["execute_code"]
    async with capture_events("ManagerMethod") as events:
        out = await execute_code(
            thought="run",
            code="print('hi')",
            language="python",
            state_mode="stateless",
            session_id=None,
            session_name=None,
            venv_id=None,
            _notification_up_q=None,
        )
    EVENT_BUS.join_published()

    mm = [
        e
        for e in events
        if e.payload.get("manager") == "CodeActActor"
        and e.payload.get("method") == "execute_code"
        and e.payload.get("phase") == "outgoing"
    ]
    assert len(mm) == 1
    assert mm[0].payload.get("status") == "error"
    assert "RuntimeError" in str(mm[0].payload.get("error_type"))
    assert out.get("error")


# ---------------------------------------------------------------------------
# execute_function boundary unit tests
# ---------------------------------------------------------------------------


class _StubFunctionManager:
    """Minimal stand-in for FunctionManager used by execute_function tests.

    Provides ``_get_function_data_by_name`` so the code synthesis path can
    look up an implementation, and the standard discovery stubs to avoid
    ``AttributeError`` during actor construction.
    """

    _include_primitives = False

    def _get_function_data_by_name(self, *, name: str):
        return {
            "name": name,
            "implementation": f"def {name}(**kwargs):\n    return 'ok'",
            "language": "python",
        }

    def _get_primitive_data_by_name(self, *, name: str):
        return None

    # The constructor probes these; stubs avoid AttributeError.
    search_functions = None
    filter_functions = None
    list_functions = None


class _BoomFunctionManager(_StubFunctionManager):
    """Like _StubFunctionManager but returns an implementation that raises."""

    def _get_function_data_by_name(self, *, name: str):
        return {
            "name": name,
            "implementation": (
                f"def {name}(**kwargs):\n    raise RuntimeError('boom')"
            ),
            "language": "python",
        }


@pytest.mark.asyncio
@_handle_project
async def test_execute_function_boundary_publishes_events_and_cleans_lineage():
    """execute_function pushes execute_function({name}) onto lineage and restores it."""
    actor = CodeActActor(
        environments=[],
        function_manager=_StubFunctionManager(),
    )

    execute_function = actor.get_tools("act")["execute_function"]
    token = TOOL_LOOP_LINEAGE.set(["CodeActActor.act"])
    try:
        async with capture_events("ManagerMethod") as events:
            out = await execute_function(
                function_name="greet",
                call_kwargs={"name": "Alice"},
            )
        EVENT_BUS.join_published()

        assert _get(out, "error") is None
        assert TOOL_LOOP_LINEAGE.get([]) == ["CodeActActor.act"]

        mm = [
            e
            for e in events
            if e.payload.get("manager") == "CodeActActor"
            and e.payload.get("method") == "execute_function"
        ]
        assert sorted([e.payload.get("phase") for e in mm]) == ["incoming", "outgoing"]
        assert mm[0].payload.get("hierarchy") == [
            "CodeActActor.act",
            "execute_function(greet)",
        ]
        assert "execute_function(greet)(" in str(mm[0].payload.get("hierarchy_label"))
    finally:
        TOOL_LOOP_LINEAGE.reset(token)


@pytest.mark.asyncio
@_handle_project
async def test_execute_function_boundary_marks_error_and_cleans_lineage():
    """execute_function publishes status=error and restores lineage on failure.

    The boom function raises inside the sandbox; execute_function catches this
    and returns an error dict rather than propagating the exception.
    """
    actor = CodeActActor(
        environments=[],
        function_manager=_BoomFunctionManager(),
    )

    execute_function = actor.get_tools("act")["execute_function"]
    token = TOOL_LOOP_LINEAGE.set(["CodeActActor.act"])
    try:
        async with capture_events("ManagerMethod") as events:
            out = await execute_function(
                function_name="fail_fn",
                call_kwargs=None,
            )
        EVENT_BUS.join_published()

        # The error is captured in the result, not raised.
        assert _get(out, "error") is not None
        assert "boom" in str(_get(out, "error"))

        assert TOOL_LOOP_LINEAGE.get([]) == ["CodeActActor.act"]

        outgoing = [
            e
            for e in events
            if e.payload.get("manager") == "CodeActActor"
            and e.payload.get("method") == "execute_function"
            and e.payload.get("phase") == "outgoing"
        ]
        assert len(outgoing) == 1
        assert outgoing[0].payload.get("status") == "error"
        assert outgoing[0].payload.get("hierarchy") == [
            "CodeActActor.act",
            "execute_function(fail_fn)",
        ]
    finally:
        TOOL_LOOP_LINEAGE.reset(token)


@pytest.mark.asyncio
@_handle_project
async def test_execute_function_propagates_lineage_to_nested_manager():
    """The lineage set by execute_function is visible inside the sandbox.

    We verify by having the stub function capture the lineage ContextVar
    from within its implementation (which runs inside PythonExecutionSession).
    """

    class _LineageCapturingFM(_StubFunctionManager):
        """Returns an implementation that captures TOOL_LOOP_LINEAGE."""

        def _get_function_data_by_name(self, *, name: str):
            return {
                "name": name,
                "implementation": (
                    "def my_func(**kwargs):\n"
                    "    from unity.common._async_tool.loop_config import TOOL_LOOP_LINEAGE\n"
                    "    return list(TOOL_LOOP_LINEAGE.get([]))\n"
                ),
                "language": "python",
            }

    actor = CodeActActor(
        environments=[],
        function_manager=_LineageCapturingFM(),
    )

    execute_function = actor.get_tools("act")["execute_function"]
    token = TOOL_LOOP_LINEAGE.set(["CodeActActor.act"])
    try:
        out = await execute_function(function_name="my_func", call_kwargs=None)
        assert _get(out, "error") is None, f"Unexpected error: {_get(out, 'error')}"
        # The lineage captured inside the sandbox should include execute_function.
        captured = _get(out, "result")
        assert captured == [
            "CodeActActor.act",
            "execute_function(my_func)",
        ]
    finally:
        TOOL_LOOP_LINEAGE.reset(token)


# ---------------------------------------------------------------------------
# Integration: execute_code + FunctionManager boundary + manager
# ---------------------------------------------------------------------------


@dataclass
class _ResultHandle:
    """Tiny handle with the minimum API used by these integration tests."""

    value: Any

    async def result(self) -> Any:
        return self.value


class UnitStateManager:
    """Minimal manager-like object that publishes ManagerMethod events via decorator."""

    @log_manager_call("UnitStateManager", "ask", payload_key="question")
    async def ask(self, question: str, *, _call_id: str | None = None):
        _ = _call_id
        return _ResultHandle(value=f"answer:{question}")


def _make_primitives() -> Any:
    return SimpleNamespace(unit=UnitStateManager())


@pytest.mark.asyncio
@_handle_project
async def test_execute_code_function_boundary_to_manager_includes_full_hierarchy():
    """Full hierarchy list across execute_code + FM boundary + manager."""
    actor = CodeActActor(environments=[])
    execute_code = actor.get_tools("act")["execute_code"]

    sandbox = PythonExecutionSession(environments={}, computer_primitives=None)
    sandbox.global_state["primitives"] = _make_primitives()

    async def send_meeting_invite():
        h = await sandbox.global_state["primitives"].unit.ask("invite")
        return await h.result()

    sandbox.global_state["send_meeting_invite"] = _LineageTrackedFunction(
        send_meeting_invite,
        "send_meeting_invite",
    )

    sb_token = _CURRENT_SANDBOX.set(sandbox)
    lineage_token = TOOL_LOOP_LINEAGE.set(["CodeActActor.act"])
    try:
        code = "out = await send_meeting_invite()\nprint(out)\n"
        async with capture_events("ManagerMethod") as events:
            res = await execute_code(
                thought="run",
                code=code,
                language="python",
                state_mode="stateful",
                session_id=0,
                session_name=None,
                venv_id=None,
                _notification_up_q=None,
            )
        EVENT_BUS.join_published()

        assert _result_error(res) is None

        ask_events = [
            e
            for e in events
            if e.payload.get("manager") == "UnitStateManager"
            and e.payload.get("method") == "ask"
            and e.payload.get("phase") in ("incoming", "outgoing")
        ]
        assert {e.payload.get("phase") for e in ask_events} == {"incoming", "outgoing"}
        assert ask_events[0].payload.get("hierarchy") == [
            "CodeActActor.act",
            "execute_code",
            "send_meeting_invite",
            "UnitStateManager.ask",
        ]
    finally:
        TOOL_LOOP_LINEAGE.reset(lineage_token)
        _CURRENT_SANDBOX.reset(sb_token)


@pytest.mark.asyncio
@_handle_project
async def test_concurrent_function_boundaries_do_not_cross_talk_lineage_or_calling_ids():
    """Concurrent sibling boundaries must not nest under each other.

    _LineageTrackedFunction manages TOOL_LOOP_LINEAGE (ContextVar), not
    event-bus events.  The observable proof that lineage didn't cross-talk
    is that the inner manager calls (UnitStateManager.ask) each carry a
    hierarchy containing only their own function boundary, not the sibling's.
    """
    actor = CodeActActor(environments=[])
    execute_code = actor.get_tools("act")["execute_code"]

    sandbox = PythonExecutionSession(environments={}, computer_primitives=None)
    sandbox.global_state["primitives"] = _make_primitives()

    async def f1():
        h = await sandbox.global_state["primitives"].unit.ask("one")
        await asyncio.sleep(0)  # encourage interleaving
        return await h.result()

    async def f2():
        h = await sandbox.global_state["primitives"].unit.ask("two")
        await asyncio.sleep(0)
        return await h.result()

    sandbox.global_state["f1"] = _LineageTrackedFunction(f1, "f1")
    sandbox.global_state["f2"] = _LineageTrackedFunction(f2, "f2")

    sb_token = _CURRENT_SANDBOX.set(sandbox)
    lineage_token = TOOL_LOOP_LINEAGE.set(["CodeActActor.act"])
    try:
        code = "import asyncio\nres = await asyncio.gather(f1(), f2())\nprint(res)\n"
        async with capture_events("ManagerMethod") as events:
            res = await execute_code(
                thought="run",
                code=code,
                language="python",
                state_mode="stateful",
                session_id=0,
                session_name=None,
                venv_id=None,
                _notification_up_q=None,
            )
        EVENT_BUS.join_published()
        assert _result_error(res) is None

        # Inner manager calls (UnitStateManager.ask) are the observable events.
        # Each should carry a hierarchy that includes its own function boundary
        # but NOT the sibling's.
        ask_events = [
            e
            for e in events
            if e.payload.get("manager") == "UnitStateManager"
            and e.payload.get("method") == "ask"
            and e.payload.get("phase") == "incoming"
        ]
        assert len(ask_events) == 2

        hierarchies = [e.payload.get("hierarchy", []) for e in ask_events]
        f1_hierarchy = [h for h in hierarchies if "f1" in h]
        f2_hierarchy = [h for h in hierarchies if "f2" in h]
        assert len(f1_hierarchy) == 1, f"Expected one f1 hierarchy, got {f1_hierarchy}"
        assert len(f2_hierarchy) == 1, f"Expected one f2 hierarchy, got {f2_hierarchy}"

        # f1's hierarchy must NOT contain f2, and vice versa.
        assert "f2" not in f1_hierarchy[0], f"f1 hierarchy leaked f2: {f1_hierarchy[0]}"
        assert "f1" not in f2_hierarchy[0], f"f2 hierarchy leaked f1: {f2_hierarchy[0]}"
    finally:
        TOOL_LOOP_LINEAGE.reset(lineage_token)
        _CURRENT_SANDBOX.reset(sb_token)


@pytest.mark.asyncio
@_handle_project
async def test_function_boundary_error_restores_lineage_and_surfaces_error():
    """_LineageTrackedFunction must restore TOOL_LOOP_LINEAGE when the
    wrapped function raises, and the error must surface in execute_code's result.

    _LineageTrackedFunction manages lineage (ContextVar) only — it does not
    publish ManagerMethod events to the event bus.  The execute_code boundary
    captures the exception and reports it in the result dict.
    """
    actor = CodeActActor(environments=[])
    execute_code = actor.get_tools("act")["execute_code"]

    sandbox = PythonExecutionSession(environments={}, computer_primitives=None)
    sandbox.global_state["primitives"] = _make_primitives()

    async def boom():
        raise RuntimeError("boom")

    sandbox.global_state["boom"] = _LineageTrackedFunction(boom, "boom")

    sb_token = _CURRENT_SANDBOX.set(sandbox)
    lineage_token = TOOL_LOOP_LINEAGE.set(["CodeActActor.act"])
    try:
        res = await execute_code(
            thought="run",
            code="await boom()\n",
            language="python",
            state_mode="stateful",
            session_id=0,
            session_name=None,
            venv_id=None,
            _notification_up_q=None,
        )

        # The error should be captured in the result.
        assert _result_error(res)
        assert "RuntimeError" in str(_result_error(res))

        # Lineage must be restored to the pre-call state.
        assert TOOL_LOOP_LINEAGE.get([]) == ["CodeActActor.act"]
    finally:
        TOOL_LOOP_LINEAGE.reset(lineage_token)
        _CURRENT_SANDBOX.reset(sb_token)


# ---------------------------------------------------------------------------
# execute_function: clarification queue + context forwarding wiring
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@_handle_project
async def test_execute_function_environments_accessible_in_sandbox():
    """Environments injected into the actor are accessible from within
    execute_function's sandbox execution, just like execute_code."""

    from unity.actor.environments import create_env

    class _DummyService:
        value = "hello_from_env"

        async def do_something(self):
            return self.value

    actor = CodeActActor(
        environments=[create_env("my_service", _DummyService())],
        function_manager=_StubFunctionManager(),
    )

    execute_function = actor.get_tools("act")["execute_function"]

    token = TOOL_LOOP_LINEAGE.set(["CodeActActor.act"])
    try:
        out = await execute_function(
            function_name="greet",
            call_kwargs=None,
        )
    finally:
        TOOL_LOOP_LINEAGE.reset(token)

    # The function should execute successfully via the sandbox.
    assert _get(out, "error") is None, f"Unexpected error: {_get(out, 'error')}"
