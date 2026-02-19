import unify
import functools
import inspect
import sys
import time
import os
from os import sep
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, List

if TYPE_CHECKING:
    from unity.contact_manager.contact_manager import ContactManager
    from unity.task_scheduler.task_scheduler import TaskScheduler
from unity.events.event_bus import EVENT_BUS
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
        logs/unillm/{datetime}_{socket_name}/{session_id}/
    """
    try:
        from unity.logger import SESSION_ID
    except ImportError:
        return None

    import re

    socket_subdir = _get_socket_subdir()
    root = _get_repo_root() / "logs" / "unillm" / socket_subdir
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


# ---------- OTEL Trace Upload to Context ----------

# Field schema for the Trace context
_TRACE_FIELDS = {
    "trace_id": {"type": "str", "mutable": False},
    "span_id": {"type": "str", "mutable": False},
    "parent_span_id": {"type": "str", "mutable": False},
    "name": {"type": "str", "mutable": False},
    "service": {"type": "str", "mutable": False},
    "start_time": {"type": "datetime", "mutable": False},
    "end_time": {"type": "datetime", "mutable": False},
    "duration_ms": {"type": "float", "mutable": False},
    "status": {"type": "str", "mutable": False},
    "attributes": {"type": "dict", "mutable": False},
}


def _get_trace_file_path(trace_id: str) -> Path | None:
    """Get the path to the trace file for a given trace_id."""
    repo_root = _get_repo_root()
    trace_file = repo_root / "logs" / "all" / f"{trace_id}.jsonl"
    if trace_file.exists():
        return trace_file
    return None


def _flush_otel_spans() -> None:
    """Flush any pending OpenTelemetry spans to ensure trace file is complete."""
    try:
        from opentelemetry import trace

        provider = trace.get_tracer_provider()
        if hasattr(provider, "force_flush"):
            provider.force_flush(timeout_millis=5000)
    except Exception:
        pass


def _should_include_span(span: dict) -> bool:
    """Check if a span should be included based on SETTINGS filters.

    Filters:
        - UNITY_TRACE_SERVICES: "all" or comma-separated list of services
        - UNITY_TRACE_EXCLUDE_PATTERNS: comma-separated span name patterns to exclude
    """
    # Service filter
    services_setting = SETTINGS.UNITY_TRACE_SERVICES.strip().lower()
    if services_setting != "all":
        allowed_services = {s.strip() for s in services_setting.split(",") if s.strip()}
        span_service = (span.get("service") or "").lower()
        if span_service not in allowed_services:
            return False

    # Exclusion pattern filter
    exclude_setting = SETTINGS.UNITY_TRACE_EXCLUDE_PATTERNS.strip()
    if exclude_setting:
        exclude_patterns = [p.strip() for p in exclude_setting.split(",") if p.strip()]
        span_name = span.get("name") or ""
        for pattern in exclude_patterns:
            if pattern in span_name:
                return False

    return True


def _upload_trace_to_context(
    test_ctx: str,
    trace_id: str | None,
    max_spans: int = 1000,
) -> None:
    """Upload trace data from JSONL file to {TestContext}/Trace context.

    Controlled by SETTINGS:
        - UNITY_TRACE_UPLOAD: Enable/disable upload entirely
        - UNITY_TRACE_SERVICES: Filter by service (e.g., "unity" or "unity,orchestra")
        - UNITY_TRACE_EXCLUDE_PATTERNS: Exclude spans matching patterns

    Args:
        test_ctx: The test context path (e.g., tests/.../test_name/{user_id}/{assistant_id})
        trace_id: The 32-char hex trace_id for this test run
        max_spans: Maximum number of spans to upload (default 1000 to avoid slow uploads)
    """
    # Check if upload is enabled
    if not SETTINGS.UNITY_TRACE_UPLOAD:
        return

    if not trace_id:
        return  # OTEL disabled or no trace_id captured

    # Flush pending spans to ensure trace file is complete
    _flush_otel_spans()

    # Find the trace file
    trace_file = _get_trace_file_path(trace_id)
    if not trace_file:
        return  # No trace file exists (OTEL disabled or no spans created)

    try:
        import json

        # Read and filter spans from the JSONL file
        spans = []
        with open(trace_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    span = json.loads(line)
                    if _should_include_span(span):
                        spans.append(span)
                        if len(spans) >= max_spans:
                            break

        if not spans:
            return  # No spans after filtering

        # Detach from trace context to avoid recursive span creation.
        # Without this, each unify.log() call generates ~26 Orchestra spans,
        # turning a 600-span upload into 15,000+ additional spans.
        try:
            from opentelemetry import context

            upload_token = context.attach(context.Context())
        except ImportError:
            upload_token = None

        try:
            # Create the Trace context with explicit field types
            trace_ctx = f"{test_ctx}/Trace"
            try:
                unify.create_context(trace_ctx)
            except Exception:
                pass  # Context may already exist

            # Create fields with explicit types (idempotent)
            try:
                unify.create_fields(context=trace_ctx, fields=_TRACE_FIELDS)
            except Exception:
                pass  # Fields may already exist

            # Batch upload all spans in a single request
            entries = [
                {
                    "trace_id": span.get("trace_id"),
                    "span_id": span.get("span_id"),
                    "parent_span_id": span.get("parent_span_id"),
                    "name": span.get("name"),
                    "service": span.get("service"),
                    "start_time": span.get("start_time"),
                    "end_time": span.get("end_time"),
                    "duration_ms": span.get("duration_ms"),
                    "status": span.get("status"),
                    "attributes": span.get("attributes", {}),
                }
                for span in spans
            ]
            try:
                unify.create_logs(context=trace_ctx, entries=entries)
            except Exception:
                pass  # Best-effort logging
        finally:
            # Restore original trace context
            if upload_token is not None:
                context.detach(upload_token)

    except Exception:
        # Trace upload is best-effort; don't fail tests if it errors
        pass


def _get_trace_id_from_span(span) -> str | None:
    """Extract trace_id as 32-char hex string from an OTel span."""
    if span is None:
        return None
    try:
        ctx = span.get_span_context()
        if ctx and ctx.is_valid:
            return f"{ctx.trace_id:032x}"
    except Exception:
        pass
    return None


def _get_current_trace_id() -> str | None:
    """Get the current trace_id from the active OTel span.

    This retrieves the trace_id from the span created by the _trace_test fixture
    in conftest.py, which wraps each test in an OTel span.
    """
    try:
        from opentelemetry import trace

        span = trace.get_current_span()
        ctx = span.get_span_context()
        if ctx and ctx.is_valid:
            return f"{ctx.trace_id:032x}"
    except Exception:
        pass
    return None


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
        self.trace_id: str | None = None

    def setup(self) -> None:
        """Prepare test bookkeeping before execution.

        NOTE:
        Unify context management is handled centrally in tests/conftest.py
        (pytest_runtest_setup/teardown) so it wraps fixture setup + teardown.
        Doing unify.set_context()/unset_context() here (inside the test call
        phase) can cause flaky cross-test interference when fixtures create
        or clear managers that delete contexts.
        """
        try:
            test_fn_name = getattr(self.wrapper, "_unity_pytest_nodeid")
        except AttributeError:
            test_fn_name = self.test_fn.__name__

        test_path = _ctx_name(self.test_fn, test_fn_name)
        # Append default user_id/assistant_id for proper context hierarchy
        # Mirrors production structure and enables proper All context derivation
        self.ctx = f"{test_path}/{DEFAULT_USER_CONTEXT}/{DEFAULT_ASSISTANT_CONTEXT}"
        self.fpath = _test_fpath(self.test_fn, test_fn_name)

        self.llm_io_before = _list_llm_io_files()
        self.start_time = time.perf_counter()

    def set_trace_id(self, trace_id: str | None) -> None:
        """Store the trace_id for this test run."""
        self.trace_id = trace_id

    def teardown(self) -> None:
        """Finalize bookkeeping after execution (no Unify context mutation)."""
        duration = time.perf_counter() - self.start_time
        llm_io_after = _list_llm_io_files()
        new_llm_io_files = llm_io_after - self.llm_io_before
        llm_io_contents = _collect_llm_io_contents(new_llm_io_files)
        _log_test_combined(self.fpath, duration, llm_io_contents)

        # Upload trace data to {TestContext}/Trace
        _upload_trace_to_context(self.ctx, self.trace_id)


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
            # Capture trace_id from the _trace_test fixture's span (in root conftest.py)
            ctx.set_trace_id(_get_current_trace_id())
            try:
                result = test_fn(*args, **kwargs)
                if inspect.isawaitable(result):
                    await result
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
            # Capture trace_id from the _trace_test fixture's span (in root conftest.py)
            ctx.set_trace_id(_get_current_trace_id())
            try:
                test_fn(*args, **kwargs)
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
        # Use ajoin_callbacks (async version) to avoid deadlocks with nest_asyncio.
        try:
            await EVENT_BUS.ajoin_callbacks()
        except Exception:
            pass


# ---------- Idempotent Seeding Helpers for Parallel Tests ----------
#
# These helpers enable safe parallel execution of tests that share scenario
# data. The approach is simple: make individual operations idempotent rather
# than using distributed locks.
#
# - get_or_create_contact: Handles race conditions at the contact level
# - rebuild_id_mapping: Reconstructs local state from existing shared data
# - is_scenario_seeded: Checks if scenario data already exists
# --------------------------------------------------------------------------


def get_or_create_contact(
    cm: "ContactManager",
    *,
    email_address: str,
    **fields,
) -> int:
    """
    Idempotent contact creation - handles parallel race conditions.

    Strategy: Check first (fast path), then try create. If creation fails
    due to unique constraint (another process won the race), query again.
    This is race-safe because the unique constraint is enforced by the DB.

    Args:
        cm: ContactManager instance to use
        email_address: Unique email for the contact (used for deduplication)
        **fields: Additional contact fields (first_name, surname, etc.)

    Returns:
        The contact_id (either existing or newly created)

    Raises:
        Exception: If creation fails for reasons other than unique constraint
    """
    # Fast path: check if it already exists
    existing = cm.filter_contacts(
        filter=f"email_address == '{email_address}'",
    )["contacts"]
    if existing:
        return existing[0].contact_id

    # Try to create
    try:
        result = cm._create_contact(email_address=email_address, **fields)
        return result["details"]["contact_id"]
    except (ValueError, Exception) as e:
        # Unique constraint violation - another process won the race
        err_str = str(e).lower()
        if "unique" in err_str or "already exists" in err_str:
            existing = cm.filter_contacts(
                filter=f"email_address == '{email_address}'",
            )["contacts"]
            if existing:
                return existing[0].contact_id
        raise


def rebuild_id_mapping(
    cm: "ContactManager",
    contact_defs: list[dict],
) -> dict[str, int]:
    """
    Rebuild first_name -> contact_id mapping from existing contacts.

    Used by processes that find the scenario already seeded and need to
    reconstruct consistent ID mappings to work with the shared data.

    Args:
        cm: ContactManager instance to query
        contact_defs: List of contact definitions (dicts with email_address, first_name)

    Returns:
        Dict mapping lowercase first_name to contact_id
    """
    id_by_name: dict[str, int] = {}
    for c in contact_defs:
        email = c.get("email_address")
        first_name = c.get("first_name", "").lower()
        if email and first_name:
            existing = cm.filter_contacts(
                filter=f"email_address == '{email}'",
            )["contacts"]
            if existing:
                id_by_name[first_name] = existing[0].contact_id
    return id_by_name


def is_scenario_seeded(
    cm: "ContactManager",
    contact_defs: list[dict],
    transcript_context: str | None = None,
) -> bool:
    """
    Check if scenario data already exists (seeded by another process).

    Checks both contacts AND transcripts (if specified) to avoid race conditions
    where contacts are created but transcript seeding is still in progress.

    Args:
        cm: ContactManager instance to query
        contact_defs: List of contact definitions to check
        transcript_context: Optional transcript context path to check for logs

    Returns:
        True if scenario appears to be seeded (contacts AND transcripts exist)
    """
    # Check for contacts
    contacts_exist = False
    for c in contact_defs:
        email = c.get("email_address")
        if email:
            existing = cm.filter_contacts(
                filter=f"email_address == '{email}'",
            )["contacts"]
            if existing:
                contacts_exist = True
                break

    if not contacts_exist:
        return False

    # If no transcript context specified, just check contacts
    if not transcript_context:
        return True

    # Check for transcripts - scenario is only fully seeded if both exist
    try:
        logs = unify.get_logs(context=transcript_context, limit=1)
        return bool(logs)
    except Exception:
        # If we can't check transcripts, fall back to contacts-only check
        return True


# ---------- File-Based Scenario Lock for Parallel Tests ----------
#
# For scenarios with high data volume or no unique constraints (like transcript
# messages), use this file lock to coordinate seeding across parallel processes.
# Only one process seeds while others wait, then all rebuild local state.
#
# For scenarios with low volume and DB-level uniqueness (like contacts), the
# simpler idempotent check-before-create pattern works fine without a lock.
# --------------------------------------------------------------------------

import tempfile
import time
from contextlib import contextmanager

# Cross-platform file locking
if sys.platform == "win32":
    import msvcrt

    def _lock_file_nb(file_obj):
        """Acquire an exclusive non-blocking lock on the file (Windows)."""
        msvcrt.locking(file_obj.fileno(), msvcrt.LK_NBLCK, 1)

    def _unlock_file(file_obj):
        """Release the lock on the file (Windows)."""
        file_obj.seek(0)
        msvcrt.locking(file_obj.fileno(), msvcrt.LK_UNLCK, 1)

else:
    import fcntl

    def _lock_file_nb(file_obj):
        """Acquire an exclusive non-blocking lock on the file (Unix)."""
        fcntl.flock(file_obj.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

    def _unlock_file(file_obj):
        """Release the lock on the file (Unix)."""
        fcntl.flock(file_obj.fileno(), fcntl.LOCK_UN)


def _acquire_file_lock_with_timeout(
    lock_file,
    timeout: float,
    lock_name: str,
) -> None:
    """
    Acquire a file lock with timeout to prevent indefinite hangs.

    Uses non-blocking lock attempts with polling to implement a timeout.
    If the timeout is exceeded, raises TimeoutError with a descriptive message.

    This prevents a hung process from blocking all other parallel tests
    indefinitely - instead, waiting tests will fail with a clear error.
    """
    start = time.monotonic()
    while True:
        try:
            _lock_file_nb(lock_file)
            return  # Successfully acquired lock
        except (BlockingIOError, OSError):
            elapsed = time.monotonic() - start
            if elapsed >= timeout:
                raise TimeoutError(
                    f"Timeout after {timeout}s waiting for lock '{lock_name}'. "
                    f"Another test process may be hung while holding this lock.",
                )
            time.sleep(0.1)  # Brief sleep before retry


@contextmanager
def scenario_file_lock(lock_name: str, timeout: float | None = None):
    """
    File-based lock for coordinating parallel test scenario seeding.

    Use this when seeding involves high data volume or resources without
    unique constraints (e.g., transcript messages). All parallel processes
    block on this lock, ensuring only one seeds at a time.

    Args:
        lock_name: Unique name for this scenario's lock file.
                   Will be created in system temp directory.
        timeout: Maximum seconds to wait for the lock. Defaults to
                 SETTINGS.UNITY_FILE_LOCK_TIMEOUT (3600s / 1 hour).

    Raises:
        TimeoutError: If the lock cannot be acquired within the timeout.
            This indicates another process is hung while holding the lock.

    Example:
        with scenario_file_lock("tm_scenario"):
            if is_scenario_seeded(cm, CONTACTS, transcript_context=ctx):
                # Rebuild local state
                ids = rebuild_id_mapping(cm, CONTACTS)
            else:
                # Seed the scenario
                seed_all_data()
    """
    if timeout is None:
        timeout = SETTINGS.UNITY_FILE_LOCK_TIMEOUT
    lock_path = os.path.join(tempfile.gettempdir(), f"unity_{lock_name}.lock")
    lock_file = open(lock_path, "w")
    try:
        _acquire_file_lock_with_timeout(lock_file, timeout, lock_name)
        yield
    finally:
        _unlock_file(lock_file)
        lock_file.close()


@contextmanager
def mutation_test_lock(lock_name: str, timeout: float | None = None):
    """
    File-based lock for serializing mutation tests in parallel execution.

    When tests run via parallel_run.sh (per-test concurrency), tests that
    mutate shared context data can race with each other's rollbacks. This
    lock ensures only one mutation test runs at a time, preventing:

    1. Test A updates data
    2. Test B starts and rolls back (wiping A's changes)
    3. Test A's verification fails

    Read-only tests don't need this lock and can run fully in parallel.

    Args:
        lock_name: Unique name for this lock (e.g., "cm_mutation").
                   Will be created in system temp directory.
        timeout: Maximum seconds to wait for the lock. Defaults to
                 SETTINGS.UNITY_FILE_LOCK_TIMEOUT (3600s / 1 hour).

    Raises:
        TimeoutError: If the lock cannot be acquired within the timeout.
            This indicates another process is hung while holding the lock.

    Example:
        @pytest.fixture
        def mutation_scenario(base_scenario):
            cm, id_map = base_scenario
            with mutation_test_lock("cm_mutation"):
                yield cm, id_map
    """
    if timeout is None:
        timeout = SETTINGS.UNITY_FILE_LOCK_TIMEOUT
    lock_path = os.path.join(tempfile.gettempdir(), f"unity_{lock_name}.lock")
    lock_file = open(lock_path, "w")
    try:
        _acquire_file_lock_with_timeout(lock_file, timeout, lock_name)
        yield
    finally:
        _unlock_file(lock_file)
        lock_file.close()


# --------------------------------------------------------------------------
# TaskScheduler scenario helpers
# --------------------------------------------------------------------------


def is_task_scenario_seeded(ts: "TaskScheduler", task_defs: list[dict]) -> bool:
    """
    Check if task scenario data already exists (seeded by another process).

    Args:
        ts: TaskScheduler instance to query
        task_defs: List of task definitions to check (each has 'name' key)

    Returns:
        True if all expected tasks exist
    """
    if not task_defs:
        return False

    # Check if at least one expected task exists
    first_task_name = task_defs[0]["name"]
    try:
        existing = ts._filter_tasks(filter=f"name == {first_task_name!r}", limit=1)
        return len(existing) > 0
    except Exception:
        return False


def rebuild_task_id_mapping(
    ts: "TaskScheduler",
    task_defs: list[dict],
) -> list[int]:
    """
    Rebuild task ID list from existing shared scenario data.

    Used when scenario was seeded by another parallel process and we need
    to reconstruct the local ID list.

    Args:
        ts: TaskScheduler instance to query
        task_defs: List of task definitions (each has 'name' key)

    Returns:
        List of task IDs in the same order as task_defs
    """
    id_list: list[int] = []
    for task_data in task_defs:
        name = task_data["name"]
        try:
            existing = ts._filter_tasks(filter=f"name == {name!r}", limit=1)
            if existing:
                id_list.append(existing[0].task_id)
        except Exception:
            pass
    return id_list
