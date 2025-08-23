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

# Contexts that were pre-created during collection;
PRECREATED_CONTEXTS: set[str] = set()

TESTS_DEFAULT_ENV_VARS = {
    "UNIFY_TRACED": "true",
    "UNIFY_CACHE": "true",
    "UNIFY_DELETE_CONTEXT_ON_EXIT": "false",
    "UNIFY_OVERWRITE_PROJECT": "false",
    "UNIFY_REGISTER_SUMMARY_CALLBACKS": "false",
    "UNIFY_REGISTER_UPDATE_CALLBACKS": "false",
    "UNIFY_TESTS_RAND_PROJ": "false",
    "UNIFY_TESTS_DELETE_PROJ_ON_EXIT": "false",
    "UNIFY_CACHE_BENCHMARK": "false",
}


def _get_unity_test_env_var(name):
    """
    Get the value of an environment variable for tests. If requested variable is not set, return the default value.
    If requested variable is not one of the test environment variables, return None.
    """
    return json.loads(os.environ.get(name, TESTS_DEFAULT_ENV_VARS.get(name, "false")))


# ---------- helper -------------------------------------------------
def _ctx_name(fn: Callable, fn_name: str) -> str:
    file_path = fn.__code__.co_filename
    test_path = "/".join(file_path.split(f"{sep}tests{sep}")[1].split(sep))[:-3]
    return f"tests/{test_path}/{fn_name}" if test_path else fn_name


def _handle_project(
    test_fn: Callable | None = None,
    *,
    delete_ctx_on_exit: bool = False,
):
    if _get_unity_test_env_var("UNIFY_DELETE_CONTEXT_ON_EXIT"):
        delete_ctx_on_exit = True
    if test_fn is None:  # called with parameters → return real decorator
        return lambda f: _handle_project(
            f,
            delete_ctx_on_exit=delete_ctx_on_exit,
        )

    async def _call(fn: Callable, *a: Any, **kw: Any):
        """Call *fn* and await it if it returns an awaitable."""
        if _get_unity_test_env_var("UNIFY_TRACED"):
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

            try:
                unify.set_context(
                    ctx,
                    relative=False,
                    skip_create=ctx in PRECREATED_CONTEXTS,
                )
                EVENT_BUS.reset(delete_contexts=False)
                # Ensure EVENT_BUS has been initialised – in case the
                # global pytest_sessionstart hook was bypassed (e.g. when
                # running an individual test without the full suite).
                if not EVENT_BUS:
                    import unity as _unity_mod

                    _unity_mod.init("UnityTests")
                    EVENT_BUS.reset()
                if _get_unity_test_env_var("UNIFY_TRACED"):
                    unify.set_trace_context("Traces")
                await _call(test_fn, *args, **kwargs)

            except Exception:
                exc_type, exc_value, exc_tb = sys.exc_info()
                tb = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
                raise Exception(tb)
            finally:
                if delete_ctx_on_exit:
                    unify.delete_context(ctx)
                unify.unset_context()

    else:
        # -------- SYNC TESTS -------------------------------------------
        @functools.wraps(test_fn)
        def wrapper(*args, **kwargs):
            try:
                test_fn_name = getattr(wrapper, "_unity_pytest_nodeid")
            except AttributeError:
                test_fn_name = test_fn.__name__

            ctx = _ctx_name(test_fn, test_fn_name)

            try:
                unify.set_context(
                    ctx,
                    relative=False,
                    skip_create=ctx in PRECREATED_CONTEXTS,
                )
                EVENT_BUS.reset(delete_contexts=False)
                # Ensure EVENT_BUS has been initialised – in case the
                # global pytest_sessionstart hook was bypassed (e.g. when
                # running an individual test without the full suite).
                if not EVENT_BUS:
                    import unity as _unity_mod

                    _unity_mod.init("UnityTests")
                    EVENT_BUS.reset()
                if _get_unity_test_env_var("UNIFY_TRACED"):
                    unify.set_trace_context("Traces")
                    unify.traced(test_fn)(*args, **kwargs)
                else:
                    test_fn(*args, **kwargs)

            except Exception:
                exc_type, exc_value, exc_tb = sys.exc_info()
                tb = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
                raise Exception(tb)
            finally:
                if delete_ctx_on_exit:
                    unify.delete_context(ctx)
                unify.unset_context()

    return wrapper
