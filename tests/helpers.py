import unify
import functools
import inspect
import sys
import time
import traceback
from os import sep
from pathlib import Path
from typing import Any, Callable, List
from unity.events.event_bus import EVENT_BUS
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Contexts that were pre-created during collection;
PRECREATED_CONTEXTS: set[str] = set()

# Session-level tags for duration logging (set via --test-tags CLI option)
_SESSION_TAGS: List[str] = []


def set_session_tags(tags: List[str]) -> None:
    """Set the session-level tags for duration logging."""
    global _SESSION_TAGS
    _SESSION_TAGS = list(tags)


def get_session_tags() -> List[str]:
    """Get the session-level tags for duration logging."""
    return list(_SESSION_TAGS)


def _parse_cache_value(v: Any) -> bool | str:
    """
    Parse a cache setting value.

    Returns True/False for boolean strings, otherwise passes through as string.
    """
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        lower = v.lower()
        if lower in ("true", "yes", "1"):
            return True
        if lower in ("false", "no", "0"):
            return False
        return v
    return bool(v)


# Settings for the testing environment
class TestingSettings(BaseSettings):
    UNIFY_MODEL: str = "gpt-5.1@openai"
    UNIFY_CACHE: bool | str = True
    UNIFY_DELETE_CONTEXT_ON_EXIT: bool = False
    UNIFY_OVERWRITE_PROJECT: bool = False
    UNIFY_REGISTER_SUMMARY_CALLBACKS: bool = False
    UNIFY_REGISTER_UPDATE_CALLBACKS: bool = False
    UNIFY_TESTS_RAND_PROJ: bool = False
    UNIFY_TESTS_DELETE_PROJ_ON_EXIT: bool = False
    UNIFY_CACHE_BENCHMARK: bool = False
    UNIFY_PRETEST_CONTEXT_CREATE: bool = False
    UNIFY_TEST_TAGS: str = ""  # Comma-separated list of tags for duration logging
    UNIFY_SKIP_SESSION_SETUP: bool = False  # Skip project/context creation (pre-done)

    @field_validator("UNIFY_CACHE", mode="before")
    @classmethod
    def parse_cache(cls, v: Any) -> bool | str:
        return _parse_cache_value(v)

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


def _test_fpath(fn: Callable, fn_name: str) -> str:
    """Return the test path in format 'folder_a/folder_b/fname.py::test_name'."""
    file_path = fn.__code__.co_filename
    # Extract path relative to 'tests/' directory
    parts = file_path.split(f"{sep}tests{sep}")
    if len(parts) > 1:
        rel_path = parts[1].replace(sep, "/")
    else:
        # Fallback: just use the filename
        rel_path = file_path.split(sep)[-1]
    return f"{rel_path}::{fn_name}"


def _get_llm_io_dir() -> Path | None:
    """Get the LLM I/O debug directory for the current session."""
    try:
        from unity.constants import SESSION_ID
    except ImportError:
        return None

    root = Path(".llm_io_debug")
    if not root.exists():
        return None

    # Match the sanitization used in the async_tool loop
    import re

    session_safe = re.sub(r"[^0-9A-Za-z._-]", "-", SESSION_ID)
    session_dir = root / session_safe
    if session_dir.exists():
        return session_dir
    return None


def _list_llm_io_files() -> set[Path]:
    """List all LLM I/O files in the current session directory."""
    llm_io_dir = _get_llm_io_dir()
    if llm_io_dir is None:
        return set()
    return set(llm_io_dir.glob("*.txt"))


def _collect_llm_io_contents(files: set[Path]) -> List[str]:
    """Read contents of LLM I/O files."""
    contents = []
    for f in sorted(files):
        try:
            contents.append(f.read_text(encoding="utf-8"))
        except Exception:
            pass
    return contents


def _log_test_combined(
    test_fpath: str,
    duration: float,
    llm_io: List[str],
) -> None:
    """Log test duration and LLM I/O to the Combined context."""
    try:
        unify.log(
            context="Combined",
            test_fpath=test_fpath,
            tags=get_session_tags(),
            duration=duration,
            llm_io=llm_io,
            new=True,
        )
    except Exception:
        # Logging is best-effort; don't fail tests if it errors
        pass


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
            fpath = _test_fpath(test_fn, test_fn_name)
            skip_ctx_create = False
            if SETTINGS.UNIFY_PRETEST_CONTEXT_CREATE:
                skip_ctx_create = ctx in PRECREATED_CONTEXTS
            else:
                if not try_reuse_prev_ctx and ctx in unify.get_contexts(prefix=ctx):
                    unify.delete_context(ctx)
                    skip_ctx_create = False

            # Track LLM I/O files before test starts
            llm_io_before = _list_llm_io_files()

            start_time = time.perf_counter()
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
                await _call(test_fn, *args, **kwargs)

            except Exception:
                exc_type, exc_value, exc_tb = sys.exc_info()
                tb = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
                raise Exception(tb)
            finally:
                duration = time.perf_counter() - start_time
                # Collect new LLM I/O files created during test
                llm_io_after = _list_llm_io_files()
                new_llm_io_files = llm_io_after - llm_io_before
                llm_io_contents = _collect_llm_io_contents(new_llm_io_files)
                _log_test_combined(fpath, duration, llm_io_contents)
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
            fpath = _test_fpath(test_fn, test_fn_name)
            skip_ctx_create = False
            if SETTINGS.UNIFY_PRETEST_CONTEXT_CREATE:
                skip_ctx_create = ctx in PRECREATED_CONTEXTS
            else:
                if not try_reuse_prev_ctx and ctx in unify.get_contexts(prefix=ctx):
                    unify.delete_context(ctx)
                    skip_ctx_create = False

            # Track LLM I/O files before test starts
            llm_io_before = _list_llm_io_files()

            start_time = time.perf_counter()
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
                test_fn(*args, **kwargs)

            except Exception:
                exc_type, exc_value, exc_tb = sys.exc_info()
                tb = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
                raise Exception(tb)
            finally:
                duration = time.perf_counter() - start_time
                # Collect new LLM I/O files created during test
                llm_io_after = _list_llm_io_files()
                new_llm_io_files = llm_io_after - llm_io_before
                llm_io_contents = _collect_llm_io_contents(new_llm_io_files)
                _log_test_combined(fpath, duration, llm_io_contents)
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


async def _assert_blocks_while_paused(result_or_coro, delay: float = 0.1):
    # Accept either a coroutine or a Task/Future; create a task only when needed.
    try:
        is_task = isinstance(result_or_coro, asyncio.Task)
        is_future = asyncio.isfuture(result_or_coro)
    except Exception:
        is_task = False
        is_future = False
    t = (
        result_or_coro
        if (is_task or is_future)
        else asyncio.create_task(result_or_coro)
    )
    await asyncio.sleep(delay)
    assert not t.done(), "result() should block while paused"
    return t


def _unique_token(prefix: str = "TOKEN") -> str:
    return f"{prefix}-{uuid.uuid4()}"


def make_queues():
    up_q: asyncio.Queue[str] = asyncio.Queue()
    down_q: asyncio.Queue[str] = asyncio.Queue()
    return up_q, down_q


from contextlib import asynccontextmanager
from typing import List


@asynccontextmanager
async def capture_events(
    event_type: str,
    filter: str | None = None,
    every_n: int = 1,
):
    """
    Context manager to capture events live.
    Avoids slow/blocking backend searches in tests by hooking directly
    into the in-memory EventBus publishing pipeline.
    """
    captured: List[Any] = []

    async def _cb(evts):
        captured.extend(evts)

    sub_id = await EVENT_BUS.register_callback(
        event_type=event_type,
        callback=_cb,
        filter=filter,
        every_n=every_n,
    )
    try:
        yield captured
    finally:
        # Ensure all callback tasks (including the one for the final event)
        # have finished execution before returning control to the test.
        # This avoids race conditions where the test asserts on 'captured'
        # while the EventBus is still processing the last publication.
        #
        # We wrap in suppress() because join_callbacks is an async method
        # on the real EventBus but might be missing on mocks if they aren't
        # fully compliant (though our conftest mock likely needs it).
        try:
            await EVENT_BUS.join_callbacks()
        except Exception:
            pass
