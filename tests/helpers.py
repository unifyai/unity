import unify
import functools
import inspect
import sys
import traceback
from os import sep
from typing import Any, Callable
from unity.events.event_bus import EVENT_BUS
from pydantic_settings import BaseSettings, SettingsConfigDict

# Contexts that were pre-created during collection;
PRECREATED_CONTEXTS: set[str] = set()


# Settings for the testing environment
class TestingSettings(BaseSettings):
    UNIFY_TRACED: bool = False
    UNIFY_CACHE: bool = True
    UNIFY_DELETE_CONTEXT_ON_EXIT: bool = False
    UNIFY_OVERWRITE_PROJECT: bool = False
    UNIFY_REGISTER_SUMMARY_CALLBACKS: bool = False
    UNIFY_REGISTER_UPDATE_CALLBACKS: bool = False
    UNIFY_TESTS_RAND_PROJ: bool = False
    UNIFY_TESTS_DELETE_PROJ_ON_EXIT: bool = False
    UNIFY_CACHE_BENCHMARK: bool = False
    UNIFY_PRETEST_CONTEXT_CREATE: bool = False

    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=True,
        extra="ignore",
    )


SETTINGS = TestingSettings()


# ---------- CURSOR DEBUG LOGGER --------------------------------
# Re-export the leaf logger from the production module to keep a single
# grep-able function name while avoiding circular imports in production code.
from unity.common.debug import CURSOR_DEBUG_LOG  # noqa: E402,F401


# ---------- helper -------------------------------------------------
def _ctx_name(fn: Callable, fn_name: str) -> str:
    file_path = fn.__code__.co_filename
    test_path = "/".join(file_path.split(f"{sep}tests{sep}")[1].split(sep))[:-3]
    return f"tests/{test_path}/{fn_name}" if test_path else fn_name


def _handle_project(
    test_fn: Callable | None = None,
    *,
    try_reuse_prev_ctx: bool = False,
    delete_ctx_on_exit: bool = False,
):
    if SETTINGS.UNIFY_DELETE_CONTEXT_ON_EXIT:
        delete_ctx_on_exit = True
    if test_fn is None:  # called with parameters → return real decorator
        return lambda f: _handle_project(
            f,
            try_reuse_prev_ctx=try_reuse_prev_ctx,
            delete_ctx_on_exit=delete_ctx_on_exit,
        )

    async def _call(fn: Callable, *a: Any, **kw: Any):
        """Call *fn* and await it if it returns an awaitable."""
        if SETTINGS.UNIFY_TRACED:
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
            skip_ctx_create = False
            if SETTINGS.UNIFY_PRETEST_CONTEXT_CREATE:
                skip_ctx_create = ctx in PRECREATED_CONTEXTS
            else:
                if not try_reuse_prev_ctx and ctx in unify.get_contexts(prefix=ctx):
                    unify.delete_context(ctx)
                    skip_ctx_create = False

            try:
                unify.set_context(
                    ctx,
                    relative=False,
                    skip_create=skip_ctx_create,
                )
                EVENT_BUS.clear(delete_contexts=False)
                # Ensure EVENT_BUS has been initialised – in case the
                # global pytest_sessionstart hook was bypassed (e.g. when
                # running an individual test without the full suite).
                if not EVENT_BUS:
                    import unity as _unity_mod

                    _unity_mod.init("UnityTests")
                    EVENT_BUS.clear()
                if SETTINGS.UNIFY_TRACED:
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
            skip_ctx_create = False
            if SETTINGS.UNIFY_PRETEST_CONTEXT_CREATE:
                skip_ctx_create = ctx in PRECREATED_CONTEXTS
            else:
                if not try_reuse_prev_ctx and ctx in unify.get_contexts(prefix=ctx):
                    unify.delete_context(ctx)
                    skip_ctx_create = False

            try:
                unify.set_context(
                    ctx,
                    relative=False,
                    skip_create=skip_ctx_create,
                )
                EVENT_BUS.clear(delete_contexts=False)
                # Ensure EVENT_BUS has been initialised – in case the
                # global pytest_sessionstart hook was bypassed (e.g. when
                # running an individual test without the full suite).
                if not EVENT_BUS:
                    import unity as _unity_mod

                    _unity_mod.init("UnityTests")
                    EVENT_BUS.clear()
                if SETTINGS.UNIFY_TRACED:
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


# ---------------- Additional shared test helpers ----------------
import asyncio
import re
import uuid

DEFAULT_TIMEOUT = 60


def _assert_non_empty_str(val: object) -> None:
    assert isinstance(val, str) and val.strip(), "Expected a non-empty string"


def _normalize_alnum_lower(s: str) -> str:
    return re.sub(r"\W+", "", s.strip().lower())


def _contains_any(text: str, substrings: list[str]) -> bool:
    t = text.lower()
    return any(sub.lower() in t for sub in substrings)


def _ack_ok(reply: str) -> bool:
    return _contains_any(
        reply,
        ["ack", "acknowledged", "noted", "received", "okay", "ok"],
    )


async def _assert_blocks_while_paused(result_coro, delay: float = 0.1) -> None:
    t = asyncio.create_task(result_coro)
    await asyncio.sleep(delay)
    assert not t.done(), "result() should block while paused"
    # caller should resume and then await the same task


def _unique_token(prefix: str = "TOKEN") -> str:
    return f"{prefix}-{uuid.uuid4()}"


def make_queues():
    up_q: asyncio.Queue[str] = asyncio.Queue()
    down_q: asyncio.Queue[str] = asyncio.Queue()
    return up_q, down_q
