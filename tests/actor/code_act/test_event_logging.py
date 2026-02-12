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
        headless=True,
        computer_mode="mock",
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
        headless=True,
        computer_mode="mock",
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
    """Minimal stand-in for FunctionManager used by execute_function tests."""

    async def execute_function(self, **kwargs):
        return {"result": "ok", "error": None, "stdout": "", "stderr": ""}

    # The constructor probes these; stubs avoid AttributeError.
    search_functions = None
    filter_functions = None
    list_functions = None


class _BoomFunctionManager(_StubFunctionManager):
    """Like _StubFunctionManager but raises on execute_function."""

    async def execute_function(self, **kwargs):
        raise RuntimeError("boom")


@pytest.mark.asyncio
@_handle_project
async def test_execute_function_boundary_publishes_events_and_cleans_lineage():
    """execute_function pushes execute_function({name}) onto lineage and restores it."""
    actor = CodeActActor(
        environments=[],
        headless=True,
        computer_mode="mock",
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

        assert out.get("error") is None
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
    """execute_function publishes status=error and restores lineage on failure."""
    actor = CodeActActor(
        environments=[],
        headless=True,
        computer_mode="mock",
        function_manager=_BoomFunctionManager(),
    )

    execute_function = actor.get_tools("act")["execute_function"]
    token = TOOL_LOOP_LINEAGE.set(["CodeActActor.act"])
    try:
        async with capture_events("ManagerMethod") as events:
            with pytest.raises(RuntimeError, match="boom"):
                await execute_function(
                    function_name="fail_fn",
                    call_kwargs=None,
                )
        EVENT_BUS.join_published()

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
        assert "RuntimeError" in str(outgoing[0].payload.get("error_type"))
        assert outgoing[0].payload.get("hierarchy") == [
            "CodeActActor.act",
            "execute_function(fail_fn)",
        ]
    finally:
        TOOL_LOOP_LINEAGE.reset(token)


@pytest.mark.asyncio
@_handle_project
async def test_execute_function_propagates_lineage_to_nested_manager():
    """State managers called inside execute_function inherit the full lineage."""

    captured_lineage: list[str] = []

    class _CapturingFunctionManager(_StubFunctionManager):
        async def execute_function(self, **kwargs):
            captured_lineage.extend(TOOL_LOOP_LINEAGE.get([]))
            return {"result": "captured", "error": None, "stdout": "", "stderr": ""}

    actor = CodeActActor(
        environments=[],
        headless=True,
        computer_mode="mock",
        function_manager=_CapturingFunctionManager(),
    )

    execute_function = actor.get_tools("act")["execute_function"]
    token = TOOL_LOOP_LINEAGE.set(["CodeActActor.act"])
    try:
        await execute_function(function_name="my_func", call_kwargs=None)
        assert captured_lineage == [
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
    actor = CodeActActor(environments=[], headless=True, computer_mode="mock")
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
    """Concurrent sibling boundaries must not nest under each other."""
    actor = CodeActActor(environments=[], headless=True, computer_mode="mock")
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

        def _call_ids(name: str) -> set[str]:
            evts = [
                e
                for e in events
                if e.payload.get("manager") == "FunctionManager"
                and e.payload.get("method") == name
            ]
            assert evts
            return {e.calling_id for e in evts}

        f1_ids = _call_ids("f1")
        f2_ids = _call_ids("f2")
        assert len(f1_ids) == 1
        assert len(f2_ids) == 1
        assert next(iter(f1_ids)) != next(iter(f2_ids))
    finally:
        TOOL_LOOP_LINEAGE.reset(lineage_token)
        _CURRENT_SANDBOX.reset(sb_token)


@pytest.mark.asyncio
@_handle_project
async def test_function_boundary_error_emits_outgoing_error_and_does_not_leak_lineage():
    """FM boundary errors must publish status=error and always restore lineage."""
    actor = CodeActActor(environments=[], headless=True, computer_mode="mock")
    execute_code = actor.get_tools("act")["execute_code"]

    sandbox = PythonExecutionSession(environments={}, computer_primitives=None)
    sandbox.global_state["primitives"] = _make_primitives()

    async def boom():
        raise RuntimeError("boom")

    sandbox.global_state["boom"] = _LineageTrackedFunction(boom, "boom")

    sb_token = _CURRENT_SANDBOX.set(sandbox)
    lineage_token = TOOL_LOOP_LINEAGE.set(["CodeActActor.act"])
    try:
        async with capture_events("ManagerMethod") as events:
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
        EVENT_BUS.join_published()

        assert _result_error(res)
        assert TOOL_LOOP_LINEAGE.get([]) == ["CodeActActor.act"]

        outgoing = [
            e
            for e in events
            if e.payload.get("manager") == "FunctionManager"
            and e.payload.get("method") == "boom"
            and e.payload.get("phase") == "outgoing"
        ]
        assert len(outgoing) == 1
        assert outgoing[0].payload.get("status") == "error"
        assert "RuntimeError" in str(outgoing[0].payload.get("error_type"))
    finally:
        TOOL_LOOP_LINEAGE.reset(lineage_token)
        _CURRENT_SANDBOX.reset(sb_token)


# ---------------------------------------------------------------------------
# execute_function: clarification queue + context forwarding wiring
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@_handle_project
async def test_execute_function_wraps_namespaces_with_clarification_injector():
    """When _clarification_up_q / _clarification_down_q are provided,
    execute_function wraps environment instances with
    _ClarificationQueueInjector before passing to FunctionManager."""

    captured_ns: dict = {}

    class _CapturingFM(_StubFunctionManager):
        async def execute_function(self, **kwargs):
            captured_ns.update(kwargs.get("extra_namespaces") or {})
            return {"result": "ok", "error": None, "stdout": "", "stderr": ""}

    from unity.actor.environments import create_env
    from unity.actor.environments.base import _ClarificationQueueInjector

    class _DummyService:
        async def do_something(self):
            pass

    actor = CodeActActor(
        environments=[create_env("my_service", _DummyService())],
        headless=True,
        computer_mode="mock",
        function_manager=_CapturingFM(),
    )

    execute_function = actor.get_tools("act")["execute_function"]

    up_q: asyncio.Queue[str] = asyncio.Queue()
    down_q: asyncio.Queue[str] = asyncio.Queue()

    token = TOOL_LOOP_LINEAGE.set(["CodeActActor.act"])
    try:
        await execute_function(
            function_name="greet",
            call_kwargs=None,
            _clarification_up_q=up_q,
            _clarification_down_q=down_q,
        )
    finally:
        TOOL_LOOP_LINEAGE.reset(token)

    # The environment instance should be wrapped with _ClarificationQueueInjector.
    assert "my_service" in captured_ns
    assert isinstance(captured_ns["my_service"], _ClarificationQueueInjector)


@pytest.mark.asyncio
@_handle_project
async def test_execute_function_no_wrapping_without_clarification_queues():
    """Without clarification queues, environment instances are passed unwrapped."""

    captured_ns: dict = {}

    class _CapturingFM(_StubFunctionManager):
        async def execute_function(self, **kwargs):
            captured_ns.update(kwargs.get("extra_namespaces") or {})
            return {"result": "ok", "error": None, "stdout": "", "stderr": ""}

    from unity.actor.environments import create_env
    from unity.actor.environments.base import _ClarificationQueueInjector

    class _DummyService:
        async def do_something(self):
            pass

    svc = _DummyService()
    actor = CodeActActor(
        environments=[create_env("my_service", svc)],
        headless=True,
        computer_mode="mock",
        function_manager=_CapturingFM(),
    )

    execute_function = actor.get_tools("act")["execute_function"]

    token = TOOL_LOOP_LINEAGE.set(["CodeActActor.act"])
    try:
        await execute_function(
            function_name="greet",
            call_kwargs=None,
        )
    finally:
        TOOL_LOOP_LINEAGE.reset(token)

    # Without clarification queues, the raw instance should be passed.
    assert "my_service" in captured_ns
    assert not isinstance(captured_ns["my_service"], _ClarificationQueueInjector)
