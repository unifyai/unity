import asyncio
import types

import pytest
import unify

from unity.common.async_tool_loop import (
    start_async_tool_use_loop,
    AsyncToolUseLoopHandle,
)
from tests.helpers import _handle_project, SETTINGS


# ---------------------------------------------------------------------------
#  TOOLS
# ---------------------------------------------------------------------------


@unify.traced
async def sleeper(delay: float = 1.0) -> str:  # noqa: D401 – simple async
    """Sleep *delay* seconds then return."""
    await asyncio.sleep(delay)
    return "slept"


async def delegating_tool() -> AsyncToolUseLoopHandle:  # type: ignore[valid-type]
    """Return a nested async-tool loop *handle* that requests pass-through."""
    inner_client = unify.AsyncUnify(
        endpoint="o4-mini@openai",
        cache=SETTINGS.UNIFY_CACHE,
        traced=SETTINGS.UNIFY_TRACED,
    )
    # Start an inner loop that runs one sleeper tool.
    inner_handle = start_async_tool_use_loop(
        inner_client,
        message="Run sleeper please.",
        tools={"sleeper": sleeper},
        log_steps=False,
    )
    # 🎯 mark for pass-through so the outer handle *adopts* this one.
    inner_handle.__passthrough__ = True  # type: ignore[attr-defined]
    return inner_handle  # outer tool returns instantly


delegating_tool.__name__ = "delegating_tool"
delegating_tool.__qualname__ = "delegating_tool"


# ---------------------------------------------------------------------------
#  TEST
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@_handle_project
async def test_outer_handle_delegates_to_inner_pause_resume(monkeypatch):
    """The outer handle's pause/resume must forward to the adopted inner handle."""

    # ── set up outer loop
    client = unify.AsyncUnify(
        endpoint="o4-mini@openai",
        cache=SETTINGS.UNIFY_CACHE,
        traced=SETTINGS.UNIFY_TRACED,
    )
    client.set_system_message(
        "Call `delegating_tool` once then wait for it to finish before replying DONE.",
    )

    outer_handle = start_async_tool_use_loop(
        client,
        message="go",
        tools={"delegating_tool": delegating_tool},
        log_steps=False,
    )

    # ── wait until the pass-through adoption has happened ─────────────────
    async def _delegated() -> bool:
        return getattr(outer_handle, "_delegate", None) is not None

    start = asyncio.get_event_loop().time()
    while not await _delegated():
        if asyncio.get_event_loop().time() - start > 30:
            raise TimeoutError("Delegate not adopted within 30 s")
        await asyncio.sleep(0.05)

    delegate: AsyncToolUseLoopHandle = outer_handle._delegate  # type: ignore[attr-defined]

    # Patch *this specific* delegate's pause method so we can count invocations.
    pause_counter = {"count": 0}
    original_pause = delegate.pause

    def _patched_pause(self):
        pause_counter["count"] += 1
        return original_pause()

    # Bind the patched method to the delegate instance.
    delegate.pause = types.MethodType(_patched_pause, delegate)  # type: ignore[method-assign]

    # ── invoke pause via the *outer* handle – should route to delegate.
    outer_handle.pause()

    # Verify that the delegate.pause was called exactly once.
    assert pause_counter["count"] == 1, "Outer pause was not forwarded to inner handle"

    # Resume so the inner loop can finish.
    outer_handle.resume()

    # Final result must bubble through.
    result = await outer_handle.result()


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
    @unify.traced  # noqa: D401 – simple async sleep tool
    async def sleeper(delay: float = 0.1) -> str:
        await asyncio.sleep(delay)
        return "slept"

    # ---- counter to verify delegate.interject was called ------------------
    counter: dict[str, list] = {"msgs": []}

    async def delegating_tool() -> AsyncToolUseLoopHandle:  # type: ignore[valid-type]
        """Return a nested handle marked for pass-through with patched interject."""
        inner_client = unify.AsyncUnify(
            endpoint="o4-mini@openai",
            cache=SETTINGS.UNIFY_CACHE,
            traced=SETTINGS.UNIFY_TRACED,
        )

        inner_handle = start_async_tool_use_loop(
            inner_client,
            message="Run sleeper please.",
            tools={"sleeper": sleeper},
            log_steps=False,
        )

        # Patch the inner handle's interject *before* it is returned so that the
        # outer loop's adoption flush can be observed.
        orig_interject = inner_handle.interject

        async def _patched_interject(self, msg: str):  # type: ignore[valid-type]
            counter["msgs"].append(msg)
            return await orig_interject(msg)

        import types as _types

        inner_handle.interject = _types.MethodType(_patched_interject, inner_handle)  # type: ignore[method-assign]

        # Artificial delay gives the outer test a chance to send an interjection
        # *before* the nested handle is adopted.
        await asyncio.sleep(0.5)

        # Flag for pass-through so the outer handle adopts this one.
        inner_handle.__passthrough__ = True  # type: ignore[attr-defined]
        return inner_handle

    # Give the tool a stable name for the LLM prompt.
    delegating_tool.__name__ = "delegating_tool_interject"
    delegating_tool.__qualname__ = "delegating_tool_interject"

    # ---- start outer loop -------------------------------------------------
    client = unify.AsyncUnify(
        endpoint="o4-mini@openai",
        cache=SETTINGS.UNIFY_CACHE,
        traced=SETTINGS.UNIFY_TRACED,
    )
    client.set_system_message(
        "Call `delegating_tool_interject` once then wait for it to finish before replying DONE.",
    )

    outer_handle = start_async_tool_use_loop(
        client,
        message="go",
        tools={"delegating_tool_interject": delegating_tool},
        log_steps=False,
    )

    # ---- send *early* interjection ---------------------------------------
    early_msg = "EARLY_INTERJECTION"
    await outer_handle.interject(early_msg)

    # ---- await completion -------------------------------------------------
    await outer_handle.result()

    # ---- assertions -------------------------------------------------------
    assert (
        early_msg in counter["msgs"]
    ), "Interjection was not forwarded to inner handle"


# ---------------------------------------------------------------------------
#  Regression: no extra outer LLM turn during passthrough handover
# ---------------------------------------------------------------------------


class _SpyAsyncUnify:
    """Minimal AsyncUnify-compatible stub that records generate invocations.

    It returns a single assistant turn that requests a tool, then (if called
    again) returns a plain assistant message. Tests assert the outer loop does
    not perform this second call in passthrough handover scenarios.
    """

    def __init__(self):
        self.messages: list[dict] = []
        self.seen_messages: list[list[dict]] = []
        self._step = 0

    def append_messages(self, msgs):
        self.messages.extend(msgs)

    async def generate(self, **_):
        # Snapshot what the model "saw" at invocation time
        import copy as _copy

        self.seen_messages.append(_copy.deepcopy(self.messages))

        if self._step == 0:
            self._step += 1
            assistant_msg = {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_outer_1",
                        "type": "function",
                        "function": {
                            "name": "delegating_tool_regression",
                            "arguments": "{}",
                        },
                    },
                ],
            }
        else:
            # Any second outer LLM call would be a regression
            self._step += 1
            assistant_msg = {
                "role": "assistant",
                "content": "unexpected_extra_outer_turn",
                "tool_calls": [],
            }

        self.messages.append(assistant_msg)
        return assistant_msg

    @property
    def system_message(self) -> str:  # for logging access in the loop
        return ""


@pytest.mark.asyncio
@_handle_project
async def test_no_extra_llm_turn_during_passthrough_handover():
    """Outer loop must not perform an additional LLM step after adopting a
    passthrough delegate. Prior to the guard, the outer loop could start a
    stray LLM step between adoption and the top-of-loop handover.
    Deterministic: this test would have failed by observing 2 outer generate
    calls; with the fix it observes exactly 1.
    """

    # Inner spy client drives the inner loop to (1) request sleeper, then (2) finish.
    class _InnerSpyClient(_SpyAsyncUnify):
        async def generate(self, **_):
            import copy as _copy

            self.seen_messages.append(_copy.deepcopy(self.messages))
            if self._step == 0:
                self._step += 1
                assistant_msg = {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_inner_1",
                            "type": "function",
                            "function": {
                                "name": "sleeper",
                                "arguments": '{"delay": 0.01}',
                            },
                        },
                    ],
                }
            else:
                self._step += 1
                assistant_msg = {
                    "role": "assistant",
                    "content": "DONE",
                    "tool_calls": [],
                }
            self.messages.append(assistant_msg)
            return assistant_msg

    # Tool: quick async sleep
    @unify.traced
    async def sleeper(delay: float = 0.01) -> str:
        await asyncio.sleep(delay)
        return "slept"

    # Delegating tool: returns a pass-through inner handle immediately
    async def delegating_tool_regression() -> AsyncToolUseLoopHandle:  # type: ignore[valid-type]
        inner_client = _InnerSpyClient()
        inner_handle = start_async_tool_use_loop(
            inner_client,
            message="Run sleeper then finish",
            tools={"sleeper": sleeper},
            log_steps=False,
        )
        inner_handle.__passthrough__ = True  # type: ignore[attr-defined]
        return inner_handle

    # Name the tool as referenced by the outer spy's assistant message
    delegating_tool_regression.__name__ = "delegating_tool_regression"
    delegating_tool_regression.__qualname__ = "delegating_tool_regression"

    # Outer spy client drives only one assistant turn (tool request)
    outer_client = _SpyAsyncUnify()

    outer_handle = start_async_tool_use_loop(
        client=outer_client,  # type: ignore[arg-type]
        message="please delegate",
        tools={"delegating_tool_regression": delegating_tool_regression},
        log_steps=False,
    )

    # Await final result bubbling from the inner loop
    final = await outer_handle.result()

    # Assert: outer LLM was invoked exactly once (no stray turn during handover)
    assert (
        len(outer_client.seen_messages) == 1
    ), f"Expected exactly 1 outer LLM call, got {len(outer_client.seen_messages)}"

    # Inner finished successfully
    assert isinstance(final, str) and final, "Outer result should be a non-empty string"


# ---------------------------------------------------------------------------
#  Regression: result() must not raise CancelledError after outer stop
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@_handle_project
async def test_result_after_outer_stop_no_cancelled_error():
    """After pass-through adoption, calling outer_handle.stop() cancels the
    outer loop task; result() must still return the inner result (not raise).

    Prior to swallowing asyncio.CancelledError in AsyncToolUseLoopHandle.result,
    awaiting result() would propagate the cancellation of the outer task even
    though the delegate finished cleanly. This test fails on the old behavior
    and passes after the fix.
    """

    # Inner tool: quick async sleep
    @unify.traced
    async def sleeper(delay: float = 0.01) -> str:
        await asyncio.sleep(delay)
        return "slept"

    # Delegating tool: returns a pass-through inner handle that completes normally
    async def delegating_tool_pass():  # type: ignore[valid-type]
        inner_client = unify.AsyncUnify(
            endpoint="o4-mini@openai",
            cache=SETTINGS.UNIFY_CACHE,
            traced=SETTINGS.UNIFY_TRACED,
        )
        inner_handle = start_async_tool_use_loop(
            inner_client,
            message="Run sleeper then finish",
            tools={"sleeper": sleeper},
            log_steps=False,
        )
        inner_handle.__passthrough__ = True  # type: ignore[attr-defined]
        return inner_handle

    delegating_tool_pass.__name__ = "delegating_tool_pass"
    delegating_tool_pass.__qualname__ = "delegating_tool_pass"

    # Start outer loop
    client = unify.AsyncUnify(
        endpoint="o4-mini@openai",
        cache=SETTINGS.UNIFY_CACHE,
        traced=SETTINGS.UNIFY_TRACED,
    )
    client.set_system_message(
        "Call `delegating_tool_pass` once then wait for it to finish before replying DONE.",
    )

    outer_handle = start_async_tool_use_loop(
        client,
        message="go",
        tools={"delegating_tool_pass": delegating_tool_pass},
        log_steps=False,
    )

    # Wait until delegate is adopted
    async def _delegated() -> bool:
        return getattr(outer_handle, "_delegate", None) is not None

    start = asyncio.get_event_loop().time()
    while not await _delegated():
        if asyncio.get_event_loop().time() - start > 30:
            raise TimeoutError("Delegate not adopted within 30 s")
        await asyncio.sleep(0.01)

    delegate = outer_handle._delegate  # type: ignore[attr-defined]

    # Prevent outer stop() from forwarding to the inner handle so the inner can
    # complete normally; still cancel the OUTER loop task.
    try:
        setattr(delegate, "stop", None)  # type: ignore[attr-defined]
    except Exception:
        pass

    # Cancel the outer loop; this used to make result() bubble CancelledError
    outer_handle.stop(reason="test-outer-cancel")

    # Must NOT raise – should return the inner result (e.g., "slept")
    final = await outer_handle.result()
    assert (
        isinstance(final, str) and final
    ), "result() should return inner result, not raise"


# ---------------------------------------------------------------------------
#  Duplicate interjection forwarding during adopt
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@_handle_project
async def test_interjection_not_duplicated_on_adopt():
    """Simulate a pre-adoption interjection that exists in both the outer
    handle's queue and early buffer when adoption occurs. The ideal behavior
    is to forward it to the delegate exactly once. Current implementation
    forwards it twice (once from the queue drain in _adopt, once from the
    early buffer replay), so this test should FAIL until deduplication is added.
    """

    # Create a minimal outer handle with its own queue/events
    async def _noop():
        return ""

    task = asyncio.create_task(_noop())
    interject_q: asyncio.Queue[str] = asyncio.Queue()
    cancel_ev = asyncio.Event()
    stop_ev = asyncio.Event()
    pause_ev = asyncio.Event()
    pause_ev.set()  # start unpaused so _adopt won't call delegate.pause()

    outer = AsyncToolUseLoopHandle(
        task=task,
        interject_queue=interject_q,
        cancel_event=cancel_ev,
        stop_event=stop_ev,
        pause_event=pause_ev,
        client=None,
        loop_id="dup-test",
        initial_user_message=None,
    )

    # Prime both buffers with the same interjection before adoption
    dup_msg = "DUPLICATE_ME"
    outer._early_interjects.append(dup_msg)  # type: ignore[attr-defined]
    interject_q.put_nowait(dup_msg)

    # Spy delegate that counts interject calls
    class _Delegate:
        def __init__(self):
            self.calls: list[str] = []

        async def interject(self, message: str, **_):  # type: ignore[valid-type]
            self.calls.append(message)
            return None

    delegate = _Delegate()

    # Adopt the delegate (this will forward from both the queue and early buffer)
    outer._adopt(delegate)  # type: ignore[attr-defined]

    # Allow any scheduled forwards to run
    await asyncio.sleep(0)

    # EXPECTATION (ideal): exactly one forward
    # ACTUAL (current): two forwards → this assertion should FAIL until fixed
    assert (
        delegate.calls.count(dup_msg) == 1
    ), "Interjection was forwarded to the delegate more than once during adopt"
