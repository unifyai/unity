import os
import json
import unify
import functools
import inspect
import sys
import traceback
from os import sep
from typing import Any, Callable
from unity.events.event_bus import EVENT_BUS

# --------------------------------------------------------------------------- #
#  SHARED TEST HELPERS                                                        #
# --------------------------------------------------------------------------- #

from typing import List
import asyncio


@unify.traced
def count_assistant_tool_calls(msgs: List[dict], tool_name: str) -> int:
    """Return the number of *assistant* turns whose visible ``tool_calls``
    reference *tool_name* (exact match).  Helper used by many tests to verify
    how often a particular tool was requested.
    """

    return sum(
        1
        for m in msgs
        if m.get("role") == "assistant"
        and any(
            tc.get("function", {}).get("name") == tool_name
            for tc in (m.get("tool_calls") or [])
        )
    )


@unify.traced
async def _wait_for_tool_request(
    client: "unify.AsyncUnify",
    tool_name: str,
    *,
    timeout: float = 15.0,
    poll: float = 0.05,
) -> None:
    """Poll *client.messages* until the assistant has issued **at least one**
    visible tool call to *tool_name* or *timeout* seconds elapse.

    This helper is meant to replace fragile ``asyncio.sleep`` calls that assume
    the LLM will have scheduled a tool within an arbitrary time window.  It
    deterministically blocks *only* until the relevant tool request is present
    in the chat history (or raises ``TimeoutError`` otherwise).
    """

    import time as _time

    start_ts = _time.perf_counter()
    while _time.perf_counter() - start_ts < timeout:
        msgs = client.messages or []  # unify may return None initially
        if count_assistant_tool_calls(msgs, tool_name) >= 1:
            return  # tool has been requested – safe to proceed
        await asyncio.sleep(poll)

    raise TimeoutError(
        f"Timed out after {timeout}s waiting for assistant to request {tool_name!r}.",
    )


def _handle_project(
    test_fn: Callable | None = None,
    *,
    try_reuse_prev_ctx: bool = False,
    delete_ctx_on_exit: bool = False,
):
    if json.loads(os.getenv("UNIFY_DELETE_CONTEXT_ON_EXIT", "false")):
        delete_ctx_on_exit = True
    if test_fn is None:  # called with parameters → return real decorator
        return lambda f: _handle_project(
            f,
            try_reuse_prev_ctx=try_reuse_prev_ctx,
            delete_ctx_on_exit=delete_ctx_on_exit,
        )

    # ---------- helper -------------------------------------------------
    def _ctx_name(fn: Callable, fn_name: str) -> str:
        file_path = fn.__code__.co_filename
        test_path = "/".join(file_path.split(f"{sep}tests{sep}")[1].split(sep))[:-3]
        return f"{test_path}/{fn_name}" if test_path else fn_name

    async def _call(fn: Callable, *a: Any, **kw: Any):
        """Call *fn* and await it if it returns an awaitable."""
        if json.loads(os.environ.get("UNIFY_TRACED", "true")):
            result = unify.traced(fn)(*a, **kw)
        else:
            result = fn(*a, **kw)
        if inspect.isawaitable(result):
            return await result
        return result

    # ---------- build the right kind of wrapper ------------------------
    if inspect.iscoroutinefunction(test_fn):
        # -------- ASYNC TESTS ------------------------------------------
        @functools.wraps(test_fn)
        async def wrapper(*args, **kwargs):
            try:
                test_fn_name = getattr(wrapper, "_unity_pytest_nodeid")
            except AttributeError:
                test_fn_name = test_fn.__name__

            ctx = _ctx_name(test_fn, test_fn_name)

            if not try_reuse_prev_ctx and unify.get_contexts(prefix=ctx):
                unify.delete_context(ctx)

            try:
                with unify.Context(ctx):
                    EVENT_BUS.reset()
                    # Ensure EVENT_BUS has been initialised – in case the
                    # global pytest_sessionstart hook was bypassed (e.g. when
                    # running an individual test without the full suite).
                    if not EVENT_BUS:
                        import unity as _unity_mod

                        _unity_mod.init("UnityTests")
                        EVENT_BUS.reset()
                    if json.loads(os.environ.get("UNIFY_TRACED", "true")):
                        unify.set_trace_context("Traces")
                    await _call(test_fn, *args, **kwargs)

                if delete_ctx_on_exit:
                    unify.delete_context(ctx)

            except Exception:
                if delete_ctx_on_exit:
                    unify.delete_context(ctx)
                exc_type, exc_value, exc_tb = sys.exc_info()
                tb = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
                raise Exception(tb)

    else:
        # -------- SYNC TESTS -------------------------------------------
        @functools.wraps(test_fn)
        def wrapper(*args, **kwargs):
            try:
                test_fn_name = getattr(wrapper, "_unity_pytest_nodeid")
            except AttributeError:
                test_fn_name = test_fn.__name__

            ctx = _ctx_name(test_fn, test_fn_name)

            if not try_reuse_prev_ctx and unify.get_contexts(prefix=ctx):
                unify.delete_context(ctx)

            try:
                with unify.Context(ctx):
                    EVENT_BUS.reset()
                    # Ensure EVENT_BUS has been initialised – in case the
                    # global pytest_sessionstart hook was bypassed (e.g. when
                    # running an individual test without the full suite).
                    if not EVENT_BUS:
                        import unity as _unity_mod

                        _unity_mod.init("UnityTests")
                        EVENT_BUS.reset()
                    if json.loads(os.environ.get("UNIFY_TRACED", "true")):
                        unify.set_trace_context("Traces")
                        unify.traced(test_fn)(*args, **kwargs)
                    else:
                        test_fn(*args, **kwargs)

                if delete_ctx_on_exit:
                    unify.delete_context(ctx)

            except Exception:
                if delete_ctx_on_exit:
                    unify.delete_context(ctx)
                exc_type, exc_value, exc_tb = sys.exc_info()
                tb = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
                raise Exception(tb)

    return wrapper
