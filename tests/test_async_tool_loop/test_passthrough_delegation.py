import asyncio

import pytest
import unify

from unity.common.async_tool_loop import (
    start_async_tool_loop,
    AsyncToolLoopHandle,
    SteerableToolHandle,
)
from tests.helpers import _handle_project
from unity.common.llm_client import new_llm_client
from tests.test_async_tool_loop.async_helpers import (
    _wait_for_tool_request,
    _wait_for_condition,
    _wait_for_tool_scheduled,
    _wait_for_tool_requested_and_scheduled,
    _wait_for_tools_requested_and_scheduled,
)


# ---------------------------------------------------------------------------
#  TOOLS
# ---------------------------------------------------------------------------


@unify.traced
async def sleeper(delay: float = 1.0) -> str:  # noqa: D401 – simple async
    """Sleep *delay* seconds then return."""
    await asyncio.sleep(delay)
    return "slept"


def _make_delegating_tool(model: str):
    """Factory to create delegating_tool with a specific model."""

    async def delegating_tool() -> AsyncToolLoopHandle:  # type: ignore[valid-type]
        """Return a nested async-tool loop *handle* that requests pass-through."""
        inner_client = new_llm_client(model=model)
        # Start an inner loop that runs one sleeper tool.
        inner_handle = start_async_tool_loop(
            inner_client,
            message="Run sleeper please.",
            tools={"sleeper": sleeper},
        )
        # 🎯 mark for pass-through so the outer handle *adopts* this one.
        inner_handle.__passthrough__ = True  # type: ignore[attr-defined]
        return inner_handle  # outer tool returns instantly

    delegating_tool.__name__ = "delegating_tool"
    delegating_tool.__qualname__ = "delegating_tool"
    return delegating_tool


# ---------------------------------------------------------------------------
#  Early interjection pass-through
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@_handle_project
async def test_outer_interjection_forwarded_to_inner(model, monkeypatch):
    """An *early* interjection (sent before delegate adoption) must be forwarded
    to the inner handle once the outer loop adopts it.

    Prior to the buffering logic introduced in `llm_helpers.py` this behaviour
    was missing – the outer loop would consume the interjection itself and the
    nested handle would *never* see it.  The test therefore fails on the old
    implementation and passes now.
    """

    # ---- helper tool -----------------------------------------------------
    @unify.traced  # noqa: D401 – simple async tool (no sleep for determinism)
    async def sleeper(delay: float = 0.1) -> str:
        return "slept"

    # ---- counter to verify delegate.interject was called ------------------
    counter: dict[str, list] = {"msgs": []}

    # Gate the return of the nested handle until after we send the early interjection
    return_handle_gate = asyncio.Event()

    async def delegating_tool() -> AsyncToolLoopHandle:  # type: ignore[valid-type]
        """Return a nested handle marked for pass-through with patched interject."""
        inner_client = new_llm_client(model=model)

        inner_handle = start_async_tool_loop(
            inner_client,
            message="Run sleeper please.",
            tools={"sleeper": sleeper},
        )

        # Patch the inner handle's interject *before* it is returned so that the
        # outer loop's forwarding flush can be observed deterministically.
        orig_interject = inner_handle.interject

        async def _patched_interject(self, msg: str):  # type: ignore[valid-type]
            counter["msgs"].append(msg)
            return await orig_interject(msg)

        import types as _types

        inner_handle.interject = _types.MethodType(_patched_interject, inner_handle)  # type: ignore[method-assign]

        # Wait for the test to signal that the early interjection has been sent
        await return_handle_gate.wait()

        # Flag for pass-through so the outer loop forwards interjections
        inner_handle.__passthrough__ = True  # type: ignore[attr-defined]
        return inner_handle

    # Give the tool a stable name for the LLM prompt.
    delegating_tool.__name__ = "delegating_tool_interject"
    delegating_tool.__qualname__ = "delegating_tool_interject"

    # ---- start outer loop -------------------------------------------------
    # Real client; strongly instruct the model to call our delegating tool
    client = new_llm_client(model=model)
    client.set_system_message(
        "You are running inside an automated test. In your FIRST assistant turn, call "
        "`delegating_tool_interject` with no arguments. Then wait for it to complete before replying.",
    )

    outer_handle = start_async_tool_loop(
        client,
        message="go",
        tools={"delegating_tool_interject": delegating_tool},
    )

    # Wait until assistant requested and the loop has scheduled the delegating tool
    await _wait_for_tool_requested_and_scheduled(
        client,
        outer_handle,
        "delegating_tool_interject",
    )

    # ---- send interjection within scheduling window (before adoption) ----
    early_msg = "EARLY_INTERJECTION"
    await outer_handle.interject(early_msg)
    # Release the delegating tool to return its handle only after the early interjection
    return_handle_gate.set()

    # ---- await completion -------------------------------------------------
    await outer_handle.result()

    # ---- assertions -------------------------------------------------------
    assert (
        early_msg in counter["msgs"]
    ), "Interjection was not forwarded to inner handle"


# ---------------------------------------------------------------------------
#  Passthrough handover: outer performs a follow-up LLM turn
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
#  Additional tests for new passthrough behaviour
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@_handle_project
async def test_interject_multicasts_passthrough(model, monkeypatch):
    """An early interjection must be forwarded to ALL passthrough handles."""

    # Counters to verify both inner handles receive the interjection
    recv_one: list[str] = []
    recv_two: list[str] = []

    class _InnerHandle(AsyncToolLoopHandle):  # type: ignore[misc]
        __passthrough__ = True  # signal passthrough mode

    async def _make_inner(counter: list[str]) -> AsyncToolLoopHandle:
        client = new_llm_client(model=model)

        async def _noop():
            return "ok"

        h = start_async_tool_loop(
            client,
            message="noop",
            tools={"noop": _noop},
            handle_cls=_InnerHandle,
        )

        # Patch interject to record payload (must accept self)
        orig = h.interject

        async def _patched(self, msg: str, **_):  # type: ignore[valid-type]
            counter.append(msg)
            return await orig(msg)

        import types as _types

        h.interject = _types.MethodType(_patched, h)  # type: ignore[method-assign]
        h.__passthrough__ = True  # type: ignore[attr-defined]
        return h

    inner_one = await _make_inner(recv_one)
    inner_two = await _make_inner(recv_two)

    # Use an event gate to ensure the outer interjection is sent before returning handles
    gate = asyncio.Event()

    async def delegating_one() -> AsyncToolLoopHandle:  # type: ignore[valid-type]
        await gate.wait()
        return inner_one

    async def delegating_two() -> AsyncToolLoopHandle:  # type: ignore[valid-type]
        await gate.wait()
        return inner_two

    client = new_llm_client(model=model)
    client.set_system_message(
        "You are running inside an automated test. In your FIRST assistant turn, call both `delegate_one` and `delegate_two` "
        "with no arguments, in the same turn, then wait for completion before replying.",
    )

    outer = start_async_tool_loop(
        client=client,  # type: ignore[arg-type]
        message="start",
        tools={"delegate_one": delegating_one, "delegate_two": delegating_two},
    )

    # Ensure both delegates have been requested and scheduled before interjecting
    await _wait_for_tools_requested_and_scheduled(
        client,
        outer,
        ["delegate_one", "delegate_two"],
    )

    # Interject after scheduling (but before adoption) → should multicast to both
    await outer.interject("BROADCAST")
    # Release delegates deterministically now that the interjection is queued
    gate.set()

    # Now wait until both patched interjects are observed
    async def _both_received():
        return ("BROADCAST" in recv_one) and ("BROADCAST" in recv_two)

    await _wait_for_condition(_both_received, poll=0.01, timeout=60.0)

    assert (
        "BROADCAST" in recv_one
    ), "first passthrough handle did not receive interjection"
    assert (
        "BROADCAST" in recv_two
    ), "second passthrough handle did not receive interjection"


@pytest.mark.asyncio
@_handle_project
async def test_ask_multicasts_passthrough(model, monkeypatch):
    """Programmatic ask() on the outer handle should be sent to every passthrough handle."""

    class MockPassthrough(SteerableToolHandle):
        __passthrough__ = True

        def __init__(self):
            self._done = asyncio.Event()
            self.ask_count = 0

        async def ask(
            self,
            question: str,
            *,
            parent_chat_context_cont: list[dict] | None = None,
        ) -> "SteerableToolHandle":
            self.ask_count += 1
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
            return "ok"

        async def next_clarification(self) -> dict:
            return {}

        async def next_notification(self) -> dict:
            return {}

        async def answer_clarification(self, call_id: str, answer: str) -> None:
            return None

    h1 = MockPassthrough()
    h2 = MockPassthrough()

    async def d1():  # type: ignore[valid-type]
        return h1

    async def d2():  # type: ignore[valid-type]
        return h2

    client = new_llm_client(model=model)
    client.set_system_message(
        "You are running inside an automated test. In your FIRST assistant turn, call both tools `d1` and `d2` "
        "with no arguments, then wait for completion before replying.",
    )
    outer = start_async_tool_loop(
        client=client,  # type: ignore[arg-type]
        message="start",
        tools={"d1": d1, "d2": d2},
    )

    # Wait deterministically until both tool requests were made
    await _wait_for_tool_request(client, "d1")
    await _wait_for_tool_request(client, "d2")

    # Wait until both passthrough handles are registered in task_info
    async def _wait_for_passthrough_handles(timeout: float = 5.0):
        import time as _time

        start = _time.perf_counter()
        while _time.perf_counter() - start < timeout:
            try:
                ti = getattr(outer._task, "task_info", {})  # type: ignore[attr-defined]
                if isinstance(ti, dict):
                    pts = [
                        _inf
                        for _inf in ti.values()
                        if getattr(_inf, "handle", None) is not None
                        and getattr(_inf, "is_passthrough", False)
                    ]
                    if len(pts) >= 2:
                        return
            except Exception:
                pass
            await asyncio.sleep(0)
        raise TimeoutError("timeout waiting for passthrough handles to register")

    await _wait_for_passthrough_handles()

    # Ask the outer handle; forwarding to both passthrough handles is performed
    # asynchronously via the mirrored steering path inside the inner loop.
    ask_handle = await outer.ask("STATUS?")

    # Wait until both inner passthrough handles observed ask()
    from tests.test_async_tool_loop.async_helpers import _wait_for_condition

    async def _both_asked():
        return (h1.ask_count >= 1) and (h2.ask_count >= 1)

    await _wait_for_condition(_both_asked, poll=0.01, timeout=60.0)

    assert h1.ask_count == 1 and h2.ask_count == 1, "ask() was not multicasted"
    # Cleanup the spawned inspection loop
    try:
        ask_handle.stop("done")
        await ask_handle.result()
    except Exception:
        pass


@pytest.mark.asyncio
@_handle_project
async def test_passthrough_clarification_bubbles(model, monkeypatch):
    """Clarification from a passthrough handle bubbles to the outer loop and can be answered programmatically."""

    from unity.common.async_tool_loop import SteerableToolHandle

    class ClarHandle(SteerableToolHandle):
        __passthrough__ = True

        def __init__(self):
            self._done = asyncio.Event()
            # Expose clarification queues so the outer loop can wire them
            self.clarification_up_q: asyncio.Queue[str] = asyncio.Queue()
            self.clarification_down_q: asyncio.Queue[str] = asyncio.Queue()

        async def ask(self, question: str, **_):
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
            # Emit a clarification request, wait for the answer, then finish.
            await self.clarification_up_q.put("favorite color?")
            ans = await self.clarification_down_q.get()
            self._done.set()
            return f"answer={ans}"

        # event APIs (not used by this test)
        async def next_clarification(self) -> dict:
            return {}

        async def next_notification(self) -> dict:
            return {}

        async def answer_clarification(self, call_id: str, answer: str) -> None:
            return None

    inner = ClarHandle()

    async def spawn() -> SteerableToolHandle:  # type: ignore[name-defined]
        return inner

    # Force a single tool call to spawn the passthrough handle (via instruction to real LLM)
    client = new_llm_client(model=model)
    client.set_system_message(
        "You are running inside an automated test. In your FIRST assistant turn, call the tool `spawn` with no arguments. "
        "Wait for it to finish before replying with a brief final message.",
    )
    outer = start_async_tool_loop(
        client=client,  # type: ignore[arg-type]
        message="start",
        tools={"spawn": spawn},
    )

    # Wait for clarification to bubble up
    evt = await outer.next_clarification()
    assert evt.get("question", "").startswith(
        "favorite",
    ), "clarification did not bubble up"

    call_id = evt.get("call_id")
    assert call_id, "missing call_id in clarification event"

    await outer.answer_clarification(call_id, "blue")

    final = await outer.result()

    # Outer transcript should include a tool message acknowledging clarification
    assert any(
        m.get("role") == "tool"
        and isinstance(m.get("name"), str)
        and m["name"].startswith("clarification_request_")
        for m in client.messages
    ), "expected clarification tool message in outer transcript"

    # The outer *final* text is driven by the spy and need not include the answer.
    # Verify the inner tool's final result with the answered clarification appears in the transcript.
    assert any(
        m.get("role") == "tool" and "answer=blue" in (m.get("content") or "")
        for m in client.messages
    ), "expected inner final result containing the answered clarification in transcript"


@pytest.mark.asyncio
@_handle_project
async def test_steering_propagates_passthrough(model):
    """Outer pause/resume/stop should be forwarded to all active passthrough handles."""

    from unity.common.async_tool_loop import SteerableToolHandle

    class CtlHandle(SteerableToolHandle):
        __passthrough__ = True

        def __init__(self):
            self._done = asyncio.Event()
            self.paused = 0
            self.resumed = 0
            self.stopped = 0

        async def ask(self, question: str, **_):  # type: ignore[override]
            return self  # self is also a SteerableToolHandle

        async def interject(self, message: str, **_):  # type: ignore[override]
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
            return "ok"

        async def next_clarification(self) -> dict:  # type: ignore[override]
            return {}

        async def next_notification(self) -> dict:  # type: ignore[override]
            return {}

        async def answer_clarification(self, call_id: str, answer: str) -> None:  # type: ignore[override]
            return None

    h1, h2 = CtlHandle(), CtlHandle()

    async def t1():  # type: ignore[valid-type]
        return h1

    async def t2():  # type: ignore[valid-type]
        return h2

    client = new_llm_client(model=model)
    client.set_system_message(
        "You are running inside an automated test. In your FIRST assistant turn, call both tools `t1` and `t2` "
        "with no arguments in the same turn, then wait for completion before replying.",
    )
    outer = start_async_tool_loop(
        client=client,  # type: ignore[arg-type]
        message="start",
        tools={"t1": t1, "t2": t2},
    )

    # Deterministically wait until both delegates were requested by the assistant
    from tests.test_async_tool_loop.async_helpers import _wait_for_tool_request

    await _wait_for_tool_request(client, "t1")
    await _wait_for_tool_request(client, "t2")

    # Programmatic pause → wait counters
    await outer.pause()
    from tests.test_async_tool_loop.async_helpers import _wait_for_condition

    async def _paused_both():
        return h1.paused >= 1 and h2.paused >= 1

    await _wait_for_condition(_paused_both, poll=0.05, timeout=60.0)

    # Programmatic resume → wait counters
    await outer.resume()

    async def _resumed_both():
        return h1.resumed >= 1 and h2.resumed >= 1

    await _wait_for_condition(_resumed_both, poll=0.05, timeout=60.0)

    # Programmatic stop → wait counters and complete handles
    outer.stop("done")

    async def _stopped_both():
        return h1.stopped >= 1 and h2.stopped >= 1

    await _wait_for_condition(_stopped_both, poll=0.05, timeout=60.0)

    assert h1.paused >= 1 and h2.paused >= 1, "pause did not propagate to all handles"
    assert (
        h1.resumed >= 1 and h2.resumed >= 1
    ), "resume did not propagate to all handles"
    assert h1.stopped >= 1 and h2.stopped >= 1, "stop did not propagate to all handles"


# ---------------------------------------------------------------------------
#  New tests: kwargs passthrough for standard steering on passthrough handles
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@_handle_project
async def test_interject_kwargs_forwarded(
    model,
    monkeypatch,
):
    """Programmatic outer.interject(..., priority=..., metadata=...) forwards kwargs to a passthrough child."""

    from unity.common.async_tool_loop import SteerableToolHandle

    class InterjectKwargsHandle(SteerableToolHandle):
        __passthrough__ = True

        def __init__(self):
            self._done = asyncio.Event()
            self.calls: list[dict] = []
            self._interject_ev = asyncio.Event()

        async def ask(self, question: str, **_):
            return self

        async def interject(
            self,
            message: str,
            *,
            priority: int = 1,
            metadata: dict | None = None,
        ):
            self.calls.append(
                {"message": message, "priority": priority, "metadata": metadata or {}},
            )
            self._interject_ev.set()
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
            return "ok"

        async def next_clarification(self) -> dict:
            return {}

        async def next_notification(self) -> dict:
            return {}

        async def answer_clarification(self, call_id: str, answer: str) -> None:
            return None

    inner = InterjectKwargsHandle()

    async def spawn() -> SteerableToolHandle:  # type: ignore[name-defined]
        return inner

    client = new_llm_client(model=model)
    client.set_system_message(
        "You are running inside an automated test. In your FIRST assistant turn, call the tool `spawn` with no arguments. "
        "Wait for it to finish before replying.",
    )

    outer = start_async_tool_loop(
        client=client,  # type: ignore[arg-type]
        message="start",
        tools={"spawn": spawn},
    )

    # Wait until spawn requested and adoption complete
    await _wait_for_tool_request(client, "spawn")

    async def _wait_adopted(timeout: float = 5.0):
        import time as _time

        start = _time.perf_counter()
        while _time.perf_counter() - start < timeout:
            try:
                ti = getattr(outer._task, "task_info", {})  # type: ignore[attr-defined]
                if isinstance(ti, dict):
                    pts = [
                        _inf
                        for _inf in ti.values()
                        if getattr(_inf, "handle", None) is not None
                        and getattr(_inf, "is_passthrough", False)
                    ]
                    if pts:
                        return
            except Exception:
                pass
            await asyncio.sleep(0)
        raise TimeoutError("timeout waiting for passthrough adoption")

    await _wait_adopted()

    # Programmatic interject with extra kwargs
    await outer.interject("HELLO", priority=3, metadata={"k": "v"})  # type: ignore[arg-type]

    async def _got_kwargs():
        return (
            bool(inner.calls)
            and inner.calls[-1]["priority"] == 3
            and inner.calls[-1]["metadata"].get("k") == "v"
        )

    await _wait_for_condition(_got_kwargs, poll=0.01, timeout=60.0)
    assert inner.calls and inner.calls[-1]["priority"] == 3
    assert inner.calls[-1]["metadata"].get("k") == "v"

    outer.stop("done")
    await outer.result()


@pytest.mark.asyncio
@_handle_project
async def test_steering_kwargs_forwarded(model, monkeypatch):
    """Programmatic pause/resume/stop kwargs are forwarded to a passthrough child that overrides their signatures."""

    from unity.common.async_tool_loop import SteerableToolHandle

    class CtlHandleKwargs(SteerableToolHandle):
        __passthrough__ = True

        def __init__(self):
            self._done = asyncio.Event()
            self.pauses: list[dict] = []
            self.resumes: list[dict] = []
            self.stops: list[dict] = []

        async def ask(self, question: str, **_):
            return self

        async def interject(self, message: str, **_):
            return None

        def stop(self, *, reason: str | None = None, abandon: bool = False):
            self.stops.append({"reason": reason, "abandon": abandon})
            self._done.set()
            return "stopped"

        async def pause(self, *, reason: str, log_to_backend: bool = False):
            self.pauses.append({"reason": reason, "log_to_backend": log_to_backend})
            return "paused"

        async def resume(self, *, token: str | None = None):
            self.resumes.append({"token": token})
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

    inner = CtlHandleKwargs()

    async def spawn() -> SteerableToolHandle:  # type: ignore[name-defined]
        return inner

    client = new_llm_client(model=model)
    client.set_system_message(
        "You are running inside an automated test. In your FIRST assistant turn, call the tool `spawn` with no arguments. "
        "Wait for it to finish before replying.",
    )
    outer = start_async_tool_loop(
        client=client,  # type: ignore[arg-type]
        message="start",
        tools={"spawn": spawn},
    )

    # Wait until the tool request and adoption
    await _wait_for_tool_request(client, "spawn")

    async def _wait_adopted(timeout: float = 5.0):
        import time as _time

        start = _time.perf_counter()
        while _time.perf_counter() - start < timeout:
            try:
                ti = getattr(outer._task, "task_info", {})  # type: ignore[attr-defined]
                if isinstance(ti, dict):
                    pts = [
                        _inf
                        for _inf in ti.values()
                        if getattr(_inf, "handle", None) is not None
                        and getattr(_inf, "is_passthrough", False)
                    ]
                    if pts:
                        return
            except Exception:
                pass
            await asyncio.sleep(0)
        raise TimeoutError("timeout waiting for passthrough adoption")

    await _wait_adopted()

    # Programmatic kwargs steering
    await outer.pause(reason="maintenance", log_to_backend=True)  # type: ignore[arg-type]
    await outer.resume(token="session-123")  # type: ignore[arg-type]
    outer.stop(reason="done", abandon=True)  # type: ignore[arg-type]

    async def _all_seen():
        return (
            inner.pauses
            and inner.pauses[-1]["reason"] == "maintenance"
            and inner.pauses[-1]["log_to_backend"] is True
            and inner.resumes
            and inner.resumes[-1]["token"] == "session-123"
            and inner.stops
            and inner.stops[-1]["reason"] == "done"
            and inner.stops[-1]["abandon"] is True
        )

    await _wait_for_condition(_all_seen, poll=0.01, timeout=60.0)

    assert inner.pauses and inner.pauses[-1]["log_to_backend"] is True
    assert inner.resumes and inner.resumes[-1]["token"] == "session-123"
    assert inner.stops and inner.stops[-1]["abandon"] is True


@pytest.mark.asyncio
@_handle_project
async def test_no_extra_turn_passthrough_handover(model, monkeypatch):
    """Outer loop continues after passthrough and performs exactly one follow-up
    LLM step once the inner tool completes.

    New semantics: The outer loop no longer hands over and exits when a tool
    returns a passthrough handle. Instead, it keeps running and, after the
    inner finishes, publishes the tool result to the outer transcript and gives
    the model one more turn. Therefore we expect exactly **two** outer LLM
    generate calls:
      1) to schedule the delegating tool, and
      2) a follow-up turn after the inner returns its final result.
    """

    # Inner real client; instruct it to call `sleeper` then finish
    inner_client = new_llm_client(model=model)
    inner_client.set_system_message(
        'You are running inside an automated test. In your FIRST assistant turn, call `sleeper` with {"delay": 0.01}. '
        "After it finishes, reply exactly with the single word DONE.",
    )

    # Tool: quick async sleep
    @unify.traced
    async def sleeper(delay: float = 0.01) -> str:
        return "slept"

    # Delegating tool: returns a pass-through inner handle immediately
    async def delegating_tool_regression() -> AsyncToolLoopHandle:  # type: ignore[valid-type]
        inner_handle = start_async_tool_loop(
            inner_client,
            message="Run sleeper then finish",
            tools={"sleeper": sleeper},
        )
        inner_handle.__passthrough__ = True  # type: ignore[attr-defined]
        return inner_handle

    # Name the tool as referenced by the outer spy's assistant message
    delegating_tool_regression.__name__ = "delegating_tool_regression"
    delegating_tool_regression.__qualname__ = "delegating_tool_regression"

    # Outer client; spy wrapper records calls while still hitting the real LLM
    outer_client = new_llm_client(model=model)
    outer_client.set_system_message(
        "You are running inside an automated test. In your FIRST assistant turn, call `delegating_tool_regression` "
        "with no arguments, then wait for the inner task to complete before replying.",
    )
    snapshots: list[list[dict]] = []
    outer_orig = outer_client.generate

    async def _outer_driver(**kwargs):
        import copy as _copy

        snapshots.append(_copy.deepcopy(outer_client.messages))
        return await outer_orig(**kwargs)

    monkeypatch.setattr(outer_client, "generate", _outer_driver, raising=True)

    outer_handle = start_async_tool_loop(
        client=outer_client,  # type: ignore[arg-type]
        message="please delegate",
        tools={"delegating_tool_regression": delegating_tool_regression},
    )

    # Await final result bubbling from the inner loop
    final = await outer_handle.result()

    # Assert: outer LLM was invoked exactly twice under new passthrough semantics
    #   1) initial planning/tool request
    #   2) follow-up after the tool result (inner DONE) is inserted
    assert (
        len(snapshots) == 2
    ), f"Expected exactly 2 outer LLM calls, got {len(snapshots)}"

    # The second invocation should see the tool result from the inner loop.
    # Look for a tool message carrying the inner final content ("DONE").
    seen_second = snapshots[1]
    assert any(
        (
            m.get("role") == "tool"
            and m.get("name") == "delegating_tool_regression"
            and "DONE" in (m.get("content") or "")
        )
        for m in seen_second
    ), "Expected outer transcript to include inner final result before second LLM call"

    # Inner finished successfully and the outer returned a non-empty result
    assert isinstance(final, str) and final, "Outer result should be a non-empty string"


@pytest.mark.asyncio
@_handle_project
async def test_ask_images_multicasts_passthrough(model, monkeypatch):
    """
    Programmatic outer.ask(..., images=...) should be forwarded to all active
    passthrough handles' ask methods, carrying the images payload.
    """

    from unity.image_manager.types import ImageRefs, RawImageRef

    class MockPassthrough(SteerableToolHandle):
        __passthrough__ = True

        def __init__(self):
            self._done = asyncio.Event()
            self.ask_payloads: list[object | None] = []

        async def ask(
            self,
            question: str,
            *,
            images: object | None = None,
            parent_chat_context_cont: list[dict] | None = None,
        ) -> "SteerableToolHandle":
            self.ask_payloads.append(images)
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
            return "ok"

        async def next_clarification(self) -> dict:
            return {}

        async def next_notification(self) -> dict:
            return {}

        async def answer_clarification(self, call_id: str, answer: str) -> None:
            return None

    h1 = MockPassthrough()
    h2 = MockPassthrough()

    async def d1():  # type: ignore[valid-type]
        return h1

    async def d2():  # type: ignore[valid-type]
        return h2

    client = new_llm_client(model=model)
    client.set_system_message(
        "You are running inside an automated test. In your FIRST assistant turn, call both tools `d1` and `d2` "
        "with no arguments, then wait for completion before replying.",
    )
    outer = start_async_tool_loop(
        client=client,  # type: ignore[arg-type]
        message="start",
        tools={"d1": d1, "d2": d2},
    )

    await _wait_for_tool_request(client, "d1")
    await _wait_for_tool_request(client, "d2")

    # Wait until both passthrough handles are registered in task_info
    async def _wait_for_passthrough_handles(timeout: float = 5.0):
        import time as _time

        start = _time.perf_counter()
        while _time.perf_counter() - start < timeout:
            try:
                ti = getattr(outer._task, "task_info", {})  # type: ignore[attr-defined]
                if isinstance(ti, dict):
                    pts = [
                        _inf
                        for _inf in ti.values()
                        if getattr(_inf, "handle", None) is not None
                        and getattr(_inf, "is_passthrough", False)
                    ]
                    if len(pts) >= 2:
                        return
            except Exception:
                pass
            await asyncio.sleep(0)
        raise TimeoutError("timeout waiting for passthrough handles to register")

    await _wait_for_passthrough_handles()

    imgs = ImageRefs([RawImageRef(image_id=123)])
    # Forward ask with images – multicasts to all passthrough handles
    ask_handle = await outer.ask("STATUS?", images=imgs)

    async def _both_received():
        return len(h1.ask_payloads) >= 1 and len(h2.ask_payloads) >= 1

    await _wait_for_condition(_both_received, poll=0.01, timeout=60.0)

    assert len(h1.ask_payloads) == 1 and len(h2.ask_payloads) == 1
    assert h1.ask_payloads[0] is not None and h2.ask_payloads[0] is not None
    # Cleanup the spawned inspection loop
    try:
        ask_handle.stop("done")
        await ask_handle.result()
    except Exception:
        pass


# ---------------------------------------------------------------------------
#  New tests for early steering replay on adoption (post‑refactor)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@_handle_project
async def test_early_ask_forwarded_on_adoption(model, monkeypatch):
    """An early outer.ask(...) issued after scheduling but before adoption is replayed to the adopted passthrough handle."""

    from unity.common.async_tool_loop import SteerableToolHandle

    class AskSpy(SteerableToolHandle):
        __passthrough__ = True

        def __init__(self):
            self._done = asyncio.Event()
            self.ask_count = 0

        async def ask(self, question: str, **_):
            self.ask_count += 1
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
            # Finish shortly after adoption so the outer loop can proceed
            await asyncio.sleep(0.01)
            self._done.set()
            return "ok"

        async def next_clarification(self) -> dict:
            return {}

        async def next_notification(self) -> dict:
            return {}

        async def answer_clarification(self, call_id: str, answer: str) -> None:
            return None

    # Gate to delay returning the inner handle until after the early ask
    gate = asyncio.Event()
    inner = AskSpy()

    async def spawn() -> SteerableToolHandle:  # type: ignore[name-defined]
        await gate.wait()
        return inner

    client = new_llm_client(model=model)
    client.set_system_message(
        "You are running inside an automated test. In your FIRST assistant turn, call the tool `spawn` with no arguments. "
        "Wait for it to finish before replying.",
    )
    outer = start_async_tool_loop(
        client=client,  # type: ignore[arg-type]
        message="start",
        tools={"spawn": spawn},
    )

    # Ensure the tool request has been scheduled, but the handle not yet returned.
    await _wait_for_tool_request(client, "spawn")

    # Ensure the spawn tool-call is scheduled before early ask, so adoption replay applies
    await _wait_for_tool_scheduled(outer, "spawn", timeout=30.0, poll=0.01)

    # EARLY steering: programmatic ask before adoption
    ask_handle = await outer.ask("EARLY_STATUS?")

    # Now allow adoption
    gate.set()

    # Wait until the ask was replayed to the adopted passthrough handle
    from tests.test_async_tool_loop.async_helpers import _wait_for_condition

    async def _asked_once():
        return inner.ask_count >= 1

    await _wait_for_condition(_asked_once, poll=0.01, timeout=60.0)
    assert inner.ask_count >= 1, "early ask() was not replayed to the adopted handle"
    # Cleanup the spawned inspection loop
    try:
        ask_handle.stop("done")
        await ask_handle.result()
    except Exception:
        pass


@pytest.mark.asyncio
@_handle_project
async def test_adoption_syncs_pause_state_when_paused(model, monkeypatch):
    """If the outer loop is paused at adoption time, the adopted passthrough handle receives pause() once."""

    from unity.common.async_tool_loop import SteerableToolHandle

    class PauseResumeSpy(SteerableToolHandle):
        __passthrough__ = True

        def __init__(self):
            self._done = asyncio.Event()
            self.paused = 0
            self.resumed = 0

        async def ask(self, question: str, **_):
            return self

        async def interject(self, message: str, **_):
            return None

        def stop(self, *_, **__):
            self._done.set()
            return "stopped"

        async def pause(self, *_, **__):
            self.paused += 1
            return "paused"

        async def resume(self, *_, **__):
            self.resumed += 1
            return "resumed"

        def done(self) -> bool:
            return self._done.is_set()

        async def result(self) -> str:
            # Complete shortly after adoption to let the outer loop complete cleanly
            await asyncio.sleep(0.01)
            self._done.set()
            return "ok"

        async def next_clarification(self) -> dict:
            return {}

        async def next_notification(self) -> dict:
            return {}

        async def answer_clarification(self, call_id: str, answer: str) -> None:
            return None

    gate = asyncio.Event()
    inner = PauseResumeSpy()

    async def spawn() -> SteerableToolHandle:  # type: ignore[name-defined]
        await gate.wait()
        return inner

    client = new_llm_client(model=model)
    client.set_system_message(
        "You are running inside an automated test. In your FIRST assistant turn, call the tool `spawn` with no arguments. "
        "Wait for it to finish before replying.",
    )
    outer = start_async_tool_loop(
        client=client,  # type: ignore[arg-type]
        message="start",
        tools={"spawn": spawn},
    )

    # Ensure the tool is requested first so the pause occurs between scheduling and adoption
    await _wait_for_tool_request(client, "spawn")

    # Pause BEFORE adoption so adoption-time state sync applies pause() to the child
    await outer.pause()

    # Now allow adoption
    gate.set()

    from tests.test_async_tool_loop.async_helpers import _wait_for_condition

    async def _paused_synced():
        return inner.paused >= 1 and inner.resumed == 0

    await _wait_for_condition(_paused_synced, poll=0.01, timeout=60.0)

    assert (
        inner.paused >= 1
    ), "pause() was not applied to the adopted handle when paused at adoption"
    assert (
        inner.resumed == 0
    ), "resume() should not be applied at adoption when outer remains paused"

    # Let the outer loop finish cleanly – paused outer cannot complete without resume
    await outer.resume()
    await outer.result()


@pytest.mark.asyncio
@_handle_project
async def test_adoption_respects_resumed_state(model, monkeypatch):
    """If the outer loop is resumed by adoption time, the adopted passthrough handle receives no pause/resume replay."""

    from unity.common.async_tool_loop import SteerableToolHandle

    class PauseResumeSpy(SteerableToolHandle):
        __passthrough__ = True

        def __init__(self):
            self._done = asyncio.Event()
            self.paused = 0
            self.resumed = 0

        async def ask(self, question: str, **_):
            return self

        async def interject(self, message: str, **_):
            return None

        def stop(self, *_, **__):
            self._done.set()
            return "stopped"

        async def pause(self, *_, **__):
            self.paused += 1
            return "paused"

        async def resume(self, *_, **__):
            self.resumed += 1
            return "resumed"

        def done(self) -> bool:
            return self._done.is_set()

        async def result(self) -> str:
            await asyncio.sleep(0.01)
            self._done.set()
            return "ok"

        async def next_clarification(self) -> dict:
            return {}

        async def next_notification(self) -> dict:
            return {}

        async def answer_clarification(self, call_id: str, answer: str) -> None:
            return None

    gate = asyncio.Event()
    inner = PauseResumeSpy()

    async def spawn() -> SteerableToolHandle:  # type: ignore[name-defined]
        await gate.wait()
        return inner

    client = new_llm_client(model=model)
    client.set_system_message(
        "You are running inside an automated test. In your FIRST assistant turn, call the tool `spawn` with no arguments. "
        "Wait for it to finish before replying.",
    )
    outer = start_async_tool_loop(
        client=client,  # type: ignore[arg-type]
        message="start",
        tools={"spawn": spawn},
    )

    # Ensure the tool is requested first so the steering occurs between scheduling and adoption
    await _wait_for_tool_request(client, "spawn")

    # EARLY steering: pause then resume BEFORE adoption (outer is resumed at adoption)
    await outer.pause()
    await outer.resume()

    # Now allow adoption
    gate.set()

    # Wait for outer loop to complete to ensure adoption occurred
    await outer.result()

    # Under new semantics, pause/resume are NOT replayed; since outer was resumed by adoption,
    # no pause/resume should be applied to the child at adoption time.
    assert (
        inner.paused == 0
    ), "pause() should not be replayed when outer is resumed by adoption"
    assert inner.resumed == 0, "resume() should not be replayed at adoption"


# ---------------------------------------------------------------------------
#  New mirror synthesis tests: immediate forward + transcript tool_calls
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@_handle_project
async def test_interject_immediate_and_mirrored(model, monkeypatch):
    """Programmatic interject should:
    - forward immediately to adopted passthrough handle(s) (no LLM step)
    - produce an assistant helper tool_call 'interject_*' and an ack tool message
    """

    recv: list[str] = []

    class Inner(SteerableToolHandle):
        __passthrough__ = True

        def __init__(self):
            self._done = asyncio.Event()

        async def ask(self, question: str, **_):
            return self

        async def interject(self, message: str, **_):
            recv.append(message)
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
            return "ok"

        async def next_clarification(self) -> dict:
            return {}

        async def next_notification(self) -> dict:
            return {}

        async def answer_clarification(self, call_id: str, answer: str) -> None:
            return None

    inner = Inner()

    async def spawn() -> SteerableToolHandle:  # type: ignore[name-defined]
        return inner

    client = new_llm_client(model=model)
    client.set_system_message(
        "You are running inside an automated test. In your FIRST assistant turn, call the tool `spawn` with no arguments. "
        "Wait for it to finish before replying.",
    )
    outer = start_async_tool_loop(
        client=client,  # type: ignore[arg-type]
        message="start",
        tools={"spawn": spawn},
    )

    # Wait until the tool request for spawn is made and the child is adopted
    await _wait_for_tool_request(client, "spawn")

    # Wait until passthrough handle is registered
    async def _wait_adopted(timeout: float = 5.0):
        import time as _time

        start = _time.perf_counter()
        while _time.perf_counter() - start < timeout:
            try:
                ti = getattr(outer._task, "task_info", {})  # type: ignore[attr-defined]
                if isinstance(ti, dict):
                    pts = [
                        _inf
                        for _inf in ti.values()
                        if getattr(_inf, "handle", None) is not None
                        and getattr(_inf, "is_passthrough", False)
                    ]
                    if pts:
                        return
            except Exception:
                pass
            await asyncio.sleep(0)
        raise TimeoutError("timeout waiting for passthrough adoption")

    await _wait_adopted()

    # Immediate interject → should record instantly
    msg = "GUIDE_NOW"
    await outer.interject(msg)

    async def _received():
        return msg in recv

    await _wait_for_condition(_received, poll=0.01, timeout=60.0)

    # Wait for mirror synthesis to appear in transcript
    async def _helper_tool_present():
        return any(
            m.get("role") == "assistant"
            and any(
                (tc.get("function", {}) or {}).get("name", "").startswith("interject_")
                for tc in (m.get("tool_calls") or [])
            )
            for m in client.messages
        )

    async def _ack_present():
        # Find all assistant tool_call ids for interject_*
        helper_ids = [
            tc["id"]
            for m in client.messages
            if m.get("role") == "assistant"
            for tc in (m.get("tool_calls") or [])
            if isinstance(tc, dict)
            and isinstance(tc.get("function"), dict)
            and str(tc["function"].get("name", "")).startswith("interject_")
        ]
        # Ack present if any tool message ties back to those ids
        return any(
            m.get("role") == "tool" and m.get("tool_call_id") in helper_ids
            for m in client.messages
        )

    await _wait_for_condition(_helper_tool_present, poll=0.01, timeout=60.0)
    await _wait_for_condition(_ack_present, poll=0.01, timeout=60.0)
    helper_seen = await _helper_tool_present()
    ack_seen = await _ack_present()
    assert helper_seen, "expected assistant helper tool_call interject_* in transcript"
    assert ack_seen, "expected ack tool message interject_* in transcript"


@pytest.mark.asyncio
@_handle_project
async def test_ask_immediate_and_mirrored(model, monkeypatch):
    """Programmatic ask should:
    - forward immediately to adopted passthrough handle(s) (no LLM step)
    - produce an assistant helper tool_call 'ask_*' and an ack tool message
    """

    class AskSpy(SteerableToolHandle):
        __passthrough__ = True

        def __init__(self):
            self._done = asyncio.Event()
            self.ask_count = 0

        async def ask(self, question: str, **_):
            self.ask_count += 1
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
            return "ok"

        async def next_clarification(self) -> dict:
            return {}

        async def next_notification(self) -> dict:
            return {}

        async def answer_clarification(self, call_id: str, answer: str) -> None:
            return None

    inner = AskSpy()

    async def spawn() -> SteerableToolHandle:  # type: ignore[name-defined]
        return inner

    client = new_llm_client(model=model)
    client.set_system_message(
        "You are running inside an automated test. In your FIRST assistant turn, call the tool `spawn` with no arguments. "
        "Wait for it to finish before replying.",
    )
    outer = start_async_tool_loop(
        client=client,  # type: ignore[arg-type]
        message="start",
        tools={"spawn": spawn},
    )

    await _wait_for_tool_request(client, "spawn")

    # Wait until passthrough handle is registered
    async def _wait_adopted(timeout: float = 5.0):
        import time as _time

        start = _time.perf_counter()
        while _time.perf_counter() - start < timeout:
            try:
                ti = getattr(outer._task, "task_info", {})  # type: ignore[attr-defined]
                if isinstance(ti, dict):
                    pts = [
                        _inf
                        for _inf in ti.values()
                        if getattr(_inf, "handle", None) is not None
                        and getattr(_inf, "is_passthrough", False)
                    ]
                    if pts:
                        return
            except Exception:
                pass
            await asyncio.sleep(0)
        raise TimeoutError("timeout waiting for passthrough adoption")

    await _wait_adopted()

    # Programmatic ask should be forwarded promptly via mirror; wait for it
    ask_handle = await outer.ask("STATUS?")

    async def _asked():
        return inner.ask_count >= 1

    await _wait_for_condition(_asked, poll=0.01, timeout=60.0)

    # Wait for mirrored helper tool_call and ack tool message to be present
    async def _ask_helper_tool_present():
        return any(
            m.get("role") == "assistant"
            and any(
                (tc.get("function", {}) or {}).get("name", "").startswith("ask_")
                for tc in (m.get("tool_calls") or [])
            )
            for m in client.messages
        )

    async def _ask_ack_present():
        # Find all assistant tool_call ids for ask_*
        helper_ids = [
            tc["id"]
            for m in client.messages
            if m.get("role") == "assistant"
            for tc in (m.get("tool_calls") or [])
            if isinstance(tc, dict)
            and isinstance(tc.get("function"), dict)
            and str(tc["function"].get("name", "")).startswith("ask_")
        ]
        # Ack present if any tool message ties back to those ids
        return any(
            m.get("role") == "tool" and m.get("tool_call_id") in helper_ids
            for m in client.messages
        )

    await _wait_for_condition(_ask_helper_tool_present, poll=0.01, timeout=60.0)
    await _wait_for_condition(_ask_ack_present, poll=0.01, timeout=60.0)
    helper_seen = await _ask_helper_tool_present()
    ack_seen = await _ask_ack_present()
    assert helper_seen, "expected assistant helper tool_call ask_* in transcript"
    assert ack_seen, "expected ack tool message ask_* in transcript"
    # Cleanup the spawned inspection loop
    try:
        ask_handle.stop("done")
        await ask_handle.result()
    except Exception:
        pass


@pytest.mark.asyncio
@_handle_project
async def test_custom_method_propagates_matching(
    model,
    monkeypatch,
):
    """
    Outer has a custom steering method. Two passthrough children are in-flight:
      - one child that implements the same custom method,
      - one child using a standard outer handle (no custom method).
    Assert: calling the outer custom steering method is delivered ONLY to the implementing child.
    """
    from unity.common.async_tool_loop import custom_steering_method

    # Custom outer handle exposing a custom steering method
    class CustomOuterHandle(AsyncToolLoopHandle):  # type: ignore[misc]
        @custom_steering_method()
        def append_to_queue(self, payload: str) -> None:
            # No local effect; decorator handles record + mirror
            return None

    # Passthrough child that implements the custom method
    class ChildCustomHandle(AsyncToolLoopHandle):  # type: ignore[misc]
        __passthrough__ = True

        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self.received: list[str] = []
            # Gate to keep the child "in flight" while the test asserts propagation
            self._done_gate: asyncio.Event = asyncio.Event()

        def append_to_queue(self, payload: str) -> str:
            self.received.append(payload)
            return "ok"

        async def result(self) -> str:  # type: ignore[override]
            await self._done_gate.wait()
            return "ok"

    # Passthrough child with NO custom method
    class BasePassthroughHandle(AsyncToolLoopHandle):  # type: ignore[misc]
        __passthrough__ = True

        # No append_to_queue method on purpose
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self._done_gate: asyncio.Event = asyncio.Event()

        async def result(self) -> str:  # type: ignore[override]
            await self._done_gate.wait()
            return "ok"

    # Two inner clients
    client_one = new_llm_client(model=model)
    client_two = new_llm_client(model=model)

    @unify.traced
    async def noop():
        return "ok"

    # Build two delegates: one returns ChildCustomHandle; the other BasePassthroughHandle
    async def delegate_custom() -> AsyncToolLoopHandle:  # type: ignore[valid-type]
        h = start_async_tool_loop(
            client_one,
            message="noop",
            tools={"noop": noop},
            handle_cls=ChildCustomHandle,
        )
        return h

    async def delegate_base() -> AsyncToolLoopHandle:  # type: ignore[valid-type]
        h = start_async_tool_loop(
            client_two,
            message="noop",
            tools={"noop": noop},
            handle_cls=BasePassthroughHandle,
        )
        return h

    # Outer client instructs model to call both delegates in the first turn
    outer_client = new_llm_client(model=model)
    outer_client.set_system_message(
        "In your FIRST assistant turn, call both tools `delegate_custom` and `delegate_base` "
        "with no arguments, then wait for completion before replying.",
    )

    outer = start_async_tool_loop(
        client=outer_client,  # type: ignore[arg-type]
        message="start",
        tools={"delegate_custom": delegate_custom, "delegate_base": delegate_base},
        handle_cls=CustomOuterHandle,
    )

    # Ensure both tool requests and scheduling happened
    await _wait_for_tools_requested_and_scheduled(
        outer_client,
        outer,
        ["delegate_custom", "delegate_base"],
    )

    # Wait until both passthrough handles are adopted
    async def _two_adopted(timeout: float = 5.0):
        import time as _time

        start = _time.perf_counter()
        while _time.perf_counter() - start < timeout:
            try:
                ti = getattr(outer._task, "task_info", {})  # type: ignore[attr-defined]
                if isinstance(ti, dict):
                    pts = [
                        _inf
                        for _inf in ti.values()
                        if getattr(_inf, "handle", None) is not None
                        and getattr(_inf, "is_passthrough", False)
                    ]
                    if len(pts) >= 2:
                        return True
            except Exception:
                pass
            await asyncio.sleep(0)
        return False

    assert await _two_adopted(), "expected both delegates to be adopted as passthrough"

    # Locate the custom child handle instance to assert its state later
    def _find_child_custom_handle():
        try:
            ti = getattr(outer._task, "task_info", {})  # type: ignore[attr-defined]
            if isinstance(ti, dict):
                for _inf in ti.values():
                    h = getattr(_inf, "handle", None)
                    if isinstance(h, ChildCustomHandle):
                        return h
        except Exception:
            return None
        return None

    def _find_base_passthrough_handle():
        try:
            ti = getattr(outer._task, "task_info", {})  # type: ignore[attr-defined]
            if isinstance(ti, dict):
                for _inf in ti.values():
                    h = getattr(_inf, "handle", None)
                    if isinstance(h, BasePassthroughHandle):
                        return h
        except Exception:
            return None
        return None

    custom_child = _find_child_custom_handle()
    base_child = _find_base_passthrough_handle()
    assert custom_child is not None, "could not locate custom child handle"
    assert base_child is not None, "could not locate base passthrough child handle"

    # Issue outer custom steering method
    outer.append_to_queue(payload="ADDME")  # type: ignore[attr-defined]

    # Wait until the custom child received the call
    async def _custom_received():
        ch = _find_child_custom_handle()
        return bool(ch and getattr(ch, "received", []) and "ADDME" in ch.received)

    await _wait_for_condition(_custom_received, poll=0.01, timeout=60.0)

    # Assert delivery reached ONLY the matching child
    assert "ADDME" in custom_child.received
    # The base child has no method and therefore no state to check; simply ensure no exception was raised
    # Optionally ensure no unintended attribute appeared
    assert not hasattr(base_child, "received")

    # Cleanup: release child gates to avoid lingering tasks
    try:
        custom_child._done_gate.set()  # type: ignore[attr-defined]
    except Exception:
        pass
    try:
        base_child._done_gate.set()  # type: ignore[attr-defined]
    except Exception:
        pass
    # Ensure the outer loop stops and cleans up (prevents stray task hangs)
    try:
        outer.stop("done")
    except Exception:
        pass
    try:
        await outer.result()
    except Exception:
        pass


@pytest.mark.asyncio
@_handle_project
async def test_adoption_replay_mirrors_interject(model, monkeypatch):
    """An interject sent before adoption should be mirrored on adoption and functionally forwarded once."""

    calls = {"count": 0}

    class InterjectSpy(SteerableToolHandle):
        __passthrough__ = True

        def __init__(self):
            self._done = asyncio.Event()

        async def ask(self, question: str, **_):
            return self

        async def interject(self, message: str, **_):
            calls["count"] += 1
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
            return "ok"

        async def next_clarification(self) -> dict:
            return {}

        async def next_notification(self) -> dict:
            return {}

        async def answer_clarification(self, call_id: str, answer: str) -> None:
            return None

    gate = asyncio.Event()
    inner = InterjectSpy()

    async def spawn() -> SteerableToolHandle:  # type: ignore[name-defined]
        await gate.wait()
        return inner

    client = new_llm_client(model=model)
    client.set_system_message(
        "You are running inside an automated test. In your FIRST assistant turn, call the tool `spawn` with no arguments. "
        "Wait for it to finish before replying.",
    )
    outer = start_async_tool_loop(
        client=client,  # type: ignore[arg-type]
        message="start",
        tools={"spawn": spawn},
    )

    await _wait_for_tool_request(client, "spawn")

    # Ensure the spawn tool-call is scheduled before early interject so adoption replay applies
    await _wait_for_tool_scheduled(outer, "spawn", timeout=30.0, poll=0.01)

    # Send interject BEFORE adoption so replay should deliver it once on adoption
    await outer.interject("HELLO_BEFORE_ADOPTION")
    gate.set()

    async def _got_once():
        return calls["count"] >= 1

    await _wait_for_condition(_got_once, poll=0.01, timeout=60.0)
    assert calls["count"] == 1, "interject should be delivered exactly once on adoption"

    # Assert mirrored helper tool_call exists for interject_*
    helper_seen = any(
        m.get("role") == "assistant"
        and any(
            (tc.get("function", {}) or {}).get("name", "").startswith("interject_")
            for tc in (m.get("tool_calls") or [])
        )
        for m in client.messages
    )
    assert (
        helper_seen
    ), "expected assistant helper tool_call interject_* mirrored on adoption"


@pytest.mark.asyncio
@_handle_project
async def test_interject_replayed_to_new_child(model, monkeypatch):
    """
    Multi-child adoption: send interject after both delegates are scheduled but before
    the second delegate returns its handle. Expect:
      - immediate forward to the already-adopted child,
      - replay to the newly adopted child on adoption,
      - no duplication.
    This would fail prior to per-child forwarded tracking (global had_passthrough)."""

    from unity.common.async_tool_loop import SteerableToolHandle

    early_msgs: list[str] = []
    late_msgs: list[str] = []

    # Gates to keep inner handles alive long enough for deterministic adoption checks.
    # We release these near the end of the test to avoid racy, ultra-short lifetimes.
    early_done_gate = asyncio.Event()
    late_done_gate = asyncio.Event()

    class SpyHandle(SteerableToolHandle):
        __passthrough__ = True

        def __init__(self, bucket: list[str], done_gate: asyncio.Event | None = None):
            self._done = asyncio.Event()
            self._bucket = bucket
            self._done_gate = done_gate

        async def ask(self, question: str, **_):
            return self

        async def interject(self, message: str, **_):
            self._bucket.append(message)
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
            # Keep the handle alive until the test signals completion, to ensure
            # adoption remains observable and interjections can be forwarded.
            if self._done_gate is not None:
                try:
                    await self._done_gate.wait()
                except asyncio.CancelledError:
                    # Allow graceful cancellation during outer.stop()
                    self._done.set()
                    raise
            else:
                await asyncio.sleep(0.05)
            self._done.set()
            return "ok"

        async def next_clarification(self) -> dict:
            return {}

        async def next_notification(self) -> dict:
            return {}

        async def answer_clarification(self, call_id: str, answer: str) -> None:
            return None

    early = SpyHandle(early_msgs, done_gate=early_done_gate)
    late = SpyHandle(late_msgs, done_gate=late_done_gate)
    gate = asyncio.Event()

    async def delegate_early() -> SteerableToolHandle:  # type: ignore[name-defined]
        return early

    async def delegate_late() -> SteerableToolHandle:  # type: ignore[name-defined]
        await gate.wait()
        return late

    client = new_llm_client(model=model)
    client.set_system_message(
        "You are running inside an automated test. In your FIRST assistant turn, call both tools "
        "`delegate_early` and `delegate_late` with no arguments, then wait for completion before replying.",
    )

    outer = start_async_tool_loop(
        client=client,  # type: ignore[arg-type]
        message="start",
        tools={"delegate_early": delegate_early, "delegate_late": delegate_late},
    )

    await _wait_for_tools_requested_and_scheduled(
        client,
        outer,
        ["delegate_early", "delegate_late"],
    )

    # Wait until exactly one passthrough handle is adopted (the early one)
    async def _one_adopted(timeout: float = 5.0):
        import time as _time

        start = _time.perf_counter()
        while _time.perf_counter() - start < timeout:
            try:
                ti = getattr(outer._task, "task_info", {})  # type: ignore[attr-defined]
                if isinstance(ti, dict):
                    pts = [
                        _inf
                        for _inf in ti.values()
                        if getattr(_inf, "handle", None) is not None
                        and getattr(_inf, "is_passthrough", False)
                    ]
                    if len(pts) == 1:
                        return True
            except Exception:
                pass
            await asyncio.sleep(0)
        return False

    assert await _one_adopted(), "expected early delegate to be adopted first"

    # Interject before late adoption
    msg = "GUIDE_ONCE"
    await outer.interject(msg)

    # Allow late delegate to return its handle
    gate.set()

    # Wait for both handles to be adopted
    async def _two_adopted(timeout: float = 5.0):
        import time as _time

        start = _time.perf_counter()
        while _time.perf_counter() - start < timeout:
            try:
                ti = getattr(outer._task, "task_info", {})  # type: ignore[attr-defined]
                if isinstance(ti, dict):
                    pts = [
                        _inf
                        for _inf in ti.values()
                        if getattr(_inf, "handle", None) is not None
                        and getattr(_inf, "is_passthrough", False)
                    ]
                    if len(pts) >= 2:
                        return True
            except Exception:
                pass
            await asyncio.sleep(0)
        return False

    assert await _two_adopted(), "expected both delegates to be adopted"

    async def _both_once():
        return early_msgs.count(msg) == 1 and late_msgs.count(msg) == 1

    await _wait_for_condition(_both_once, poll=0.01, timeout=60.0)

    assert (
        early_msgs.count(msg) == 1
    ), "early child should receive interject once (immediate)"
    assert (
        late_msgs.count(msg) == 1
    ), "late child should receive interject once (replay)"

    # Release inner handles to complete cleanly, then stop the outer loop.
    early_done_gate.set()
    late_done_gate.set()

    outer.stop("done")
    await outer.result()
