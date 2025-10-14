import asyncio

import pytest
import unify

from unity.common.async_tool_loop import (
    start_async_tool_loop,
    AsyncToolLoopHandle,
    SteerableToolHandle,
)
from tests.helpers import _handle_project, SETTINGS
from tests.test_async_tool_loop.async_helpers import _wait_for_tool_request


# ---------------------------------------------------------------------------
#  TOOLS
# ---------------------------------------------------------------------------


@unify.traced
async def sleeper(delay: float = 1.0) -> str:  # noqa: D401 – simple async
    """Sleep *delay* seconds then return."""
    await asyncio.sleep(delay)
    return "slept"


async def delegating_tool() -> AsyncToolLoopHandle:  # type: ignore[valid-type]
    """Return a nested async-tool loop *handle* that requests pass-through."""
    inner_client = unify.AsyncUnify(
        endpoint="gpt-5@openai",
        reasoning_effort="high",
        service_tier="priority",
        cache=SETTINGS.UNIFY_CACHE,
        traced=SETTINGS.UNIFY_TRACED,
    )
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


# ---------------------------------------------------------------------------
#  Early interjection pass-through
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@_handle_project
async def test_outer_interjection_forwarded_to_inner(monkeypatch):
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
        inner_client = unify.AsyncUnify(
            endpoint="gpt-5@openai",
            reasoning_effort="high",
            service_tier="priority",
            cache=SETTINGS.UNIFY_CACHE,
            traced=SETTINGS.UNIFY_TRACED,
        )

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
    client = unify.AsyncUnify(
        endpoint="gpt-5@openai",
        reasoning_effort="high",
        service_tier="priority",
        cache=SETTINGS.UNIFY_CACHE,
        traced=SETTINGS.UNIFY_TRACED,
    )
    client.set_system_message(
        "You are running inside an automated test. In your FIRST assistant turn, call "
        "`delegating_tool_interject` with no arguments. Then wait for it to complete before replying.",
    )

    outer_handle = start_async_tool_loop(
        client,
        message="go",
        tools={"delegating_tool_interject": delegating_tool},
    )

    # ---- send *early* interjection ---------------------------------------
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
async def test_interject_multicasts_to_multiple_passthrough_handles(monkeypatch):
    """An early interjection must be forwarded to ALL passthrough handles."""

    # Counters to verify both inner handles receive the interjection
    recv_one: list[str] = []
    recv_two: list[str] = []

    class _InnerHandle(AsyncToolLoopHandle):  # type: ignore[misc]
        __passthrough__ = True  # signal passthrough mode

    async def _make_inner(counter: list[str]) -> AsyncToolLoopHandle:
        client = unify.AsyncUnify(
            endpoint="gpt-5@openai",
            reasoning_effort="high",
            service_tier="priority",
            cache=SETTINGS.UNIFY_CACHE,
            traced=SETTINGS.UNIFY_TRACED,
        )

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

    client = unify.AsyncUnify(
        endpoint="gpt-5@openai",
        reasoning_effort="high",
        service_tier="priority",
        cache=SETTINGS.UNIFY_CACHE,
        traced=SETTINGS.UNIFY_TRACED,
    )
    client.set_system_message(
        "You are running inside an automated test. In your FIRST assistant turn, call both `delegate_one` and `delegate_two` "
        "with no arguments, in the same turn, then wait for completion before replying.",
    )

    outer = start_async_tool_loop(
        client=client,  # type: ignore[arg-type]
        message="start",
        tools={"delegate_one": delegating_one, "delegate_two": delegating_two},
    )

    # Early interjection before inner handles are returned
    await outer.interject("BROADCAST")
    # Release delegates deterministically now that the interjection is queued
    gate.set()

    # Wait until both passthrough handles are registered in task_info
    from tests.test_async_tool_loop.async_helpers import _wait_for_condition

    async def _pts_registered():
        try:
            ti = getattr(outer._task, "task_info", {})  # type: ignore[attr-defined]
            if isinstance(ti, dict):
                pts = [
                    _inf
                    for _inf in ti.values()
                    if getattr(_inf, "handle", None) is not None
                    and getattr(_inf, "is_passthrough", False)
                ]
                return len(pts) >= 2
        except Exception:
            return False
        return False

    await _wait_for_condition(_pts_registered, poll=0.01, timeout=60.0)

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
async def test_ask_multicasts_to_all_passthrough_handles(monkeypatch):
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

        def pause(self, *_, **__):
            return "paused"

        def resume(self, *_, **__):
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

    client = unify.AsyncUnify(
        endpoint="gpt-5@openai",
        reasoning_effort="high",
        service_tier="priority",
        cache=SETTINGS.UNIFY_CACHE,
        traced=SETTINGS.UNIFY_TRACED,
    )
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

    # Ask the outer handle; forwarding to both passthrough handles happens synchronously within ask()
    await outer.ask("STATUS?")

    assert h1.ask_count == 1 and h2.ask_count == 1, "ask() was not multicasted"


@pytest.mark.asyncio
@_handle_project
async def test_passthrough_clarification_bubbles_and_can_be_answered(monkeypatch):
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

        def pause(self, *_, **__):
            return "paused"

        def resume(self, *_, **__):
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
    client = unify.AsyncUnify(
        endpoint="gpt-5@openai",
        reasoning_effort="high",
        service_tier="priority",
        cache=SETTINGS.UNIFY_CACHE,
        traced=SETTINGS.UNIFY_TRACED,
    )
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
async def test_programmatic_pause_resume_stop_propagate_to_all_passthrough_handles():
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

    client = unify.AsyncUnify(
        endpoint="gpt-5@openai",
        reasoning_effort="high",
        service_tier="priority",
        cache=SETTINGS.UNIFY_CACHE,
        traced=SETTINGS.UNIFY_TRACED,
    )
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
    outer.pause()
    from tests.test_async_tool_loop.async_helpers import _wait_for_condition

    async def _paused_both():
        return h1.paused >= 1 and h2.paused >= 1

    await _wait_for_condition(_paused_both, poll=0.05, timeout=60.0)

    # Programmatic resume → wait counters
    outer.resume()

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


@pytest.mark.asyncio
@_handle_project
async def test_no_extra_llm_turn_during_passthrough_handover(monkeypatch):
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
    inner_client = unify.AsyncUnify(
        endpoint="gpt-5@openai",
        reasoning_effort="high",
        service_tier="priority",
        cache=SETTINGS.UNIFY_CACHE,
        traced=SETTINGS.UNIFY_TRACED,
    )
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
    outer_client = unify.AsyncUnify(
        endpoint="gpt-5@openai",
        reasoning_effort="high",
        service_tier="priority",
        cache=SETTINGS.UNIFY_CACHE,
        traced=SETTINGS.UNIFY_TRACED,
    )
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
            and '"DONE"' in (m.get("content") or "")
        )
        for m in seen_second
    ), "Expected outer transcript to include inner final result before second LLM call"

    # Inner finished successfully and the outer returned a non-empty result
    assert isinstance(final, str) and final, "Outer result should be a non-empty string"
