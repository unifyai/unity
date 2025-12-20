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
from unity.common.context_registry import ContextRegistry
from unity.session_details import DEFAULT_ASSISTANT_CONTEXT, DEFAULT_USER_CONTEXT

from tests.settings import SETTINGS

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


def _derive_socket_name() -> str:
    """Derive a unique socket name from the terminal's TTY device.

    Mirrors the logic in tests/_shell_common.sh::_derive_socket_name() to ensure
    consistent naming whether tests are run via parallel_run.sh or directly via pytest.

    Returns:
        - 'unity_dev_ttysXXX' if running in a TTY (e.g., terminal session)
        - 'unity_pidXXX' if not running in a TTY (e.g., background process)
    """
    import os
    import sys

    try:
        # Try to get the TTY device path (e.g., '/dev/ttys042')
        tty_path = os.ttyname(sys.stdout.fileno())
        # Sanitize: /dev/ttys042 -> unity_dev_ttys042
        tty_id = tty_path.replace("/", "_")
        return f"unity{tty_id}"
    except (OSError, AttributeError):
        # Not a TTY (e.g., piped output, background process)
        return f"unity_pid{os.getpid()}"


def _get_socket_subdir() -> str:
    """Determine the log subdirectory for test-related files.

    Returns a datetime-prefixed directory name for natural time-based ordering:
        - UNITY_LOG_SUBDIR if set (e.g., '2025-12-05T14-30-45_unity_dev_ttys042')
        - Falls back to UNITY_TEST_SOCKET for legacy compatibility
        - Derives terminal ID for direct pytest invocations (same as parallel_run.sh would)
    """
    import os
    from datetime import datetime

    # Prefer the datetime-prefixed log subdir if available
    log_subdir = os.environ.get("UNITY_LOG_SUBDIR", "").strip()
    if log_subdir:
        return log_subdir
    # Fallback to socket name for backward compatibility
    socket = os.environ.get("UNITY_TEST_SOCKET", "").strip()
    if socket:
        return socket
    # Derive terminal ID (same logic as _shell_common.sh) for direct pytest invocations
    socket_name = _derive_socket_name()
    return f"{datetime.now().strftime('%Y-%m-%dT%H-%M-%S')}_{socket_name}"


def _get_repo_root() -> Path:
    """Determine the repository root directory.

    Prefers UNITY_LOG_ROOT env var if set, allowing explicit worktree targeting.
    Otherwise derives from this file's location, which correctly resolves to
    the worktree when running from one.

    This fixes the issue where Background Agents (which use git worktrees)
    would have logs written to the main repo instead of their worktree.
    """
    import os

    # Allow explicit override for flexibility
    log_root = os.environ.get("UNITY_LOG_ROOT", "").strip()
    if log_root:
        return Path(log_root)

    # Derive repo root from this file's location (works correctly in worktrees)
    # __file__ is tests/helpers.py, so go up 1 level to repo root
    try:
        return Path(__file__).resolve().parent.parent
    except Exception:
        # Fallback to cwd if __file__ resolution fails
        return Path(os.getcwd())


def _get_llm_io_dir() -> Path | None:
    """Get the LLM I/O debug directory for the current session.

    Directory structure:
        .llm_io_debug/{datetime}_{socket_name}/{session_id}/
    """
    try:
        from unity.constants import SESSION_ID
    except ImportError:
        return None

    import re

    socket_subdir = _get_socket_subdir()
    root = _get_repo_root() / ".llm_io_debug" / socket_subdir
    if not root.exists():
        return None

    # Match the sanitization used in the async_tool loop
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
    """Log test duration, LLM I/O, and settings to the Combined context."""
    try:
        unify.log(
            context="Combined",
            test_fpath=test_fpath,
            tags=get_session_tags(),
            duration=duration,
            llm_io=llm_io,
            settings=SETTINGS.model_dump(),
            new=True,
        )
    except Exception:
        # Logging is best-effort; don't fail tests if it errors
        pass


class _TestContext:
    """Manages test context setup, teardown, and timing for _handle_project."""

    def __init__(
        self,
        test_fn: Callable,
        wrapper: Callable,
        try_reuse_prev_ctx: bool,
        delete_ctx_on_exit: bool,
    ):
        self.test_fn = test_fn
        self.wrapper = wrapper
        self.try_reuse_prev_ctx = try_reuse_prev_ctx
        self.delete_ctx_on_exit = delete_ctx_on_exit
        self.ctx: str = ""
        self.fpath: str = ""
        self.llm_io_before: set[Path] = set()
        self.start_time: float = 0.0

    def setup(self) -> None:
        """Prepare test context before execution."""
        try:
            test_fn_name = getattr(self.wrapper, "_unity_pytest_nodeid")
        except AttributeError:
            test_fn_name = self.test_fn.__name__

        test_path = _ctx_name(self.test_fn, test_fn_name)
        # Append default user/assistant to create proper context hierarchy for testing
        # This results in: tests/.../test_name/DefaultUser/Assistant
        # Which mirrors production structure and enables proper All context derivation
        self.ctx = f"{test_path}/{DEFAULT_USER_CONTEXT}/{DEFAULT_ASSISTANT_CONTEXT}"
        self.fpath = _test_fpath(self.test_fn, test_fn_name)

        skip_ctx_create = False
        if SETTINGS.UNIFY_PRETEST_CONTEXT_CREATE:
            skip_ctx_create = self.ctx in PRECREATED_CONTEXTS
        elif not self.try_reuse_prev_ctx:
            # EAFP: Just try to delete - don't check first.
            # The context may not exist or may have been deleted by a parallel test.
            try:
                unify.delete_context(self.ctx)
            except Exception:
                pass

        self.llm_io_before = _list_llm_io_files()
        self.start_time = time.perf_counter()

        unify.set_context(self.ctx, relative=False, skip_create=skip_ctx_create)
        ContextRegistry.clear()
        EVENT_BUS.clear(delete_contexts=False)

        if not EVENT_BUS:
            import unity as _unity_mod

            _unity_mod.init("UnityTests")
            EVENT_BUS.clear()

    def teardown(self) -> None:
        """Clean up test context after execution."""
        duration = time.perf_counter() - self.start_time
        llm_io_after = _list_llm_io_files()
        new_llm_io_files = llm_io_after - self.llm_io_before
        llm_io_contents = _collect_llm_io_contents(new_llm_io_files)
        _log_test_combined(self.fpath, duration, llm_io_contents)

        if self.delete_ctx_on_exit:
            unify.delete_context(self.ctx)
        unify.unset_context()


def _handle_project(
    test_fn: Callable | None = None,
    *,
    try_reuse_prev_ctx: bool = False,
    delete_ctx_on_exit: bool = False,
):
    if SETTINGS.UNIFY_DELETE_CONTEXT_ON_EXIT:
        delete_ctx_on_exit = True
    if test_fn is None:
        return lambda f: _handle_project(
            f,
            try_reuse_prev_ctx=try_reuse_prev_ctx,
            delete_ctx_on_exit=delete_ctx_on_exit,
        )

    if inspect.iscoroutinefunction(test_fn):

        @functools.wraps(test_fn)
        async def wrapper(*args, **kwargs):
            ctx = _TestContext(
                test_fn,
                wrapper,
                try_reuse_prev_ctx,
                delete_ctx_on_exit,
            )
            ctx.setup()
            try:
                result = test_fn(*args, **kwargs)
                if inspect.isawaitable(result):
                    await result
            except Exception:
                exc_type, exc_value, exc_tb = sys.exc_info()
                tb = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
                raise Exception(tb)
            finally:
                ctx.teardown()

    else:

        @functools.wraps(test_fn)
        def wrapper(*args, **kwargs):
            ctx = _TestContext(
                test_fn,
                wrapper,
                try_reuse_prev_ctx,
                delete_ctx_on_exit,
            )
            ctx.setup()
            try:
                test_fn(*args, **kwargs)
            except Exception:
                exc_type, exc_value, exc_tb = sys.exc_info()
                tb = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
                raise Exception(tb)
            finally:
                ctx.teardown()

    return wrapper


# ---------------- Additional shared test helpers ----------------
import asyncio
import re
import uuid
import random

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
    rand_bits = random.getrandbits(128)
    return f"{prefix}-{uuid.UUID(int=rand_bits, version=4)}"


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
