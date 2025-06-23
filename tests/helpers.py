import os
import json
import unify
import functools
import inspect
import sys
import traceback
from os import sep
from typing import Any, Callable


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
    def _ctx_name(fn: Callable) -> str:
        file_path = fn.__code__.co_filename
        test_path = "/".join(file_path.split(f"{sep}tests{sep}")[1].split(sep))[:-3]
        return f"{test_path}/{fn.__name__}" if test_path else fn.__name__

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
            ctx = _ctx_name(test_fn)

            if not try_reuse_prev_ctx and unify.get_contexts(prefix=ctx):
                unify.delete_context(ctx)

            try:
                with unify.Context(ctx):
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
            ctx = _ctx_name(test_fn)

            if not try_reuse_prev_ctx and unify.get_contexts(prefix=ctx):
                unify.delete_context(ctx)

            try:
                with unify.Context(ctx):
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
