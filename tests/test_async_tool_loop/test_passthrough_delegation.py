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

    Prior to the buffering logic introduced in `async_tool_loop.py` this behaviour
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
