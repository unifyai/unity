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

    async def pause(self, *_, **__):  # type: ignore[override]
        self.paused += 1
        return "paused"

    async def resume(self, *_, **__):  # type: ignore[override]
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
            "steps": [
                {"method": "interject", "args": "root-info"},  # informational only
            ],
            "children": [
                {"handle": "ToyHandle", "steps": [{"method": "pause"}]},
            ],
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
                isinstance(p, str) and ("ToyHandle" in p)
                for p in (item.get("path") or [])
            )
            for item in (res.get("applied") or [])
        ), "nested_steer did not report applying pause to the child path"

        # New assertions: the OUTER transcript should include a synthetic helper tool_call
        # for the child-level pause, targeted to the specific in-flight tool (Outer_spawn).
        def _find_helper(prefix: str, tool_name: str):
            msgs = client.messages
            for idx, m in enumerate(msgs):
                if m.get("role") != "assistant":
                    continue
                tcs = m.get("tool_calls") or []
                if not tcs:
                    continue
                for tc in tcs:
                    fn = (tc.get("function") or {}).get("name") or ""
                    if isinstance(fn, str) and fn.startswith(f"{prefix}_{tool_name}_"):
                        return idx, fn, m
            return None

        # Wait until the helper appears (mirror is async but fast)
        async def _helper_present():
            return _find_helper("pause", "Outer_spawn") is not None

        await _wait_for_condition(_helper_present, poll=0.01, timeout=30.0)
        found = _find_helper("pause", "Outer_spawn")
        assert (
            found is not None
        ), "Expected synthetic pause_* helper tool_call in outer transcript"
        helper_asst_idx, helper_name, helper_asst_msg = found

        # The acknowledgement tool message must be directly after the assistant helper
        # with the same helper tool name, containing a standard pause acknowledgement.
        msgs = client.messages
        mm = msgs[helper_asst_idx + 1]
        assert (
            mm.get("role") == "tool" and mm.get("name") == helper_name
        ), "Expected immediate tool ack after helper"
        assert "Pause request acknowledged." in str(
            mm.get("content", ""),
        ), "Pause acknowledgement content mismatch"
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
async def test_nested_steer_noop_when_child_selector_does_not_match():
    """When a child selector does not match any in-flight child, nested_steer should do nothing (no fallback)."""

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

        # Intentionally provide a non-matching child-level selector. Without any fallback,
        # no method should be applied.
        spec = {
            "children": [
                {"handle": "IGNORED", "steps": [{"method": "pause"}]},
            ],
        }

        res = await outer.nested_steer(spec)  # type: ignore[attr-defined]

        # No-op expected: inner remains unpaused and selector is recorded as skipped
        assert (
            inner.paused == 0
        ), "No child matched; nested_steer should not apply any method"
        assert any(
            isinstance(item.get("child"), dict)
            and (item["child"].get("handle") == "IGNORED")
            for item in (res.get("skipped") or [])
        ), "Expected non-matching selector to be recorded as skipped with the correct path"

        # Also ensure NO helper pause_* was injected for Wrapper_run (no target match)
        def _any_helper(prefix: str, tool_name: str) -> bool:
            for m in client.messages:
                if m.get("role") != "assistant":
                    continue
                for tc in m.get("tool_calls") or []:
                    fn = (tc.get("function") or {}).get("name") or ""
                    if isinstance(fn, str) and fn.startswith(f"{prefix}_{tool_name}_"):
                        return True
            return False

        assert not _any_helper(
            "pause",
            "Wrapper_run",
        ), "Helper pause_* should not be injected for a non-matching child"
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
async def test_nested_steer_applies_serial_steps_on_child():
    """Verify that multiple serial steps (pause → resume → interject) apply on the same child loop."""

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

        msg = "serial-steps-info"
        spec = {
            "children": [
                {
                    "handle": "ToyHandle",
                    "steps": [
                        {"method": "pause"},
                        {"method": "resume"},
                        {"method": "interject", "args": msg},
                    ],
                },
            ],
        }

        await outer.nested_steer(spec)  # type: ignore[attr-defined]

        async def _serial_done():
            try:
                return (
                    inner.paused >= 1
                    and inner.resumed >= 1
                    and (msg in inner.interjections)
                )
            except Exception:
                return False

        await _wait_for_condition(_serial_done, poll=0.01, timeout=30.0)

        assert inner.paused >= 1, "pause step did not apply"
        assert inner.resumed >= 1, "resume step did not apply"
        assert msg in inner.interjections, "interject step did not apply"

        # Assert that helper tool_calls for pause, resume and interject were injected and acknowledged
        def _find_helpers(prefix: str, tool_name: str) -> list[tuple[int, str]]:
            out: list[tuple[int, str]] = []
            msgs2 = client.messages
            for idx, m in enumerate(msgs2):
                if m.get("role") != "assistant":
                    continue
                tcs = m.get("tool_calls") or []
                if not tcs:
                    continue
                for tc in tcs:
                    fn = (tc.get("function") or {}).get("name") or ""
                    if isinstance(fn, str) and fn.startswith(f"{prefix}_{tool_name}_"):
                        out.append((idx, fn))
            return out

        # Wait until helpers appear (mirrors are async relative to nested_steer return)
        async def _helpers_present():
            return (
                len(_find_helpers("pause", "Outer_spawn")) > 0
                and len(_find_helpers("resume", "Outer_spawn")) > 0
                and len(_find_helpers("interject", "Outer_spawn")) > 0
            )

        await _wait_for_condition(_helpers_present, poll=0.01, timeout=30.0)

        helpers_pause = _find_helpers("pause", "Outer_spawn")
        helpers_resume = _find_helpers("resume", "Outer_spawn")
        helpers_interject = _find_helpers("interject", "Outer_spawn")

        assert (
            helpers_pause
        ), "Expected synthetic pause_* helper tool_call in outer transcript"
        assert (
            helpers_resume
        ), "Expected synthetic resume_* helper tool_call in outer transcript"
        assert (
            helpers_interject
        ), "Expected synthetic interject_* helper tool_call in outer transcript"

        # Check acknowledgement for interject is immediately after helper and includes the message content
        ii_idx, ii_name = helpers_interject[-1]
        mm = client.messages[ii_idx + 1]
        assert (
            mm.get("role") == "tool" and mm.get("name") == ii_name
        ), "Expected immediate interject ack after helper"
        assert msg in str(
            mm.get("content", ""),
        ), "Interject acknowledgement should include the interjection content"
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
