"""
tests/conftest.py
=================

Global pytest configuration for Unity test suite.

Sections:
  1. Imports and logging guard
  2. Test stubs (Redis, ComputerWorker, DateTime)
  3. Singleton isolation
  4. Command-line options
  5. Custom logging helpers
  6. Session lifecycle hooks
  7. Test run hooks
  8. HTTP client cleanup
  9. Pre-run context creation
"""

from __future__ import annotations

import asyncio
import logging
import os
import random
import re
import threading
import hashlib

import httpx
import pytest
import unify
from pytest_metadata.plugin import metadata_key

from datetime import datetime, timezone

# --------------------------------------------------------------------------- #
# 1. Early logging guard                                                      #
# --------------------------------------------------------------------------- #
# Ensure a handler exists before imports that might call logging.basicConfig()
_root_logger_early = logging.getLogger()
if not _root_logger_early.handlers:
    _root_logger_early.addHandler(logging.NullHandler())

from tests.helpers import PRECREATED_CONTEXTS, set_session_tags
from tests.settings import SETTINGS
from unity.session_details import DEFAULT_ASSISTANT_CONTEXT, DEFAULT_USER_CONTEXT


# --------------------------------------------------------------------------- #
# Orchestra availability check for requires_orchestra marker                   #
# --------------------------------------------------------------------------- #
def _check_orchestra_available() -> bool:
    """Check if Orchestra server is reachable. Cached after first call."""
    if hasattr(_check_orchestra_available, "_cached"):
        return _check_orchestra_available._cached

    orchestra_url = os.environ.get("ORCHESTRA_URL", "http://localhost:8000")
    try:
        with httpx.Client(timeout=2.0) as client:
            # Check a known endpoint - /v0/projects works and 404 on root is fine
            resp = client.get(f"{orchestra_url}/v0/projects")
            # 200 = success, 401/403 = auth required but server is up
            _check_orchestra_available._cached = resp.status_code in (200, 401, 403)
    except Exception:
        _check_orchestra_available._cached = False

    return _check_orchestra_available._cached


def _derive_test_context(item: pytest.Item) -> str:
    """
    Derive a per-test Unify context path that is stable and unique.

    Matches the intent of tests.helpers._TestContext.setup(), but runs early enough
    (pytest_runtest_setup) to wrap fixture setup + teardown, preventing cross-test
    interference when fixtures create/clear managers that delete contexts.
    """
    # Build "tests/<relpath-without-.py>/<func_name>" prefix
    file_path = str(getattr(item, "fspath", "") or "")
    parts = file_path.split(f"{os.sep}tests{os.sep}")
    if len(parts) > 1:
        rel_path = parts[1].replace(os.sep, "/")
        if rel_path.endswith(".py"):
            rel_path = rel_path[:-3]
        test_path = f"tests/{rel_path}"
    else:
        # Fallback (should be rare): use nodeid as the "path"
        test_path = "tests/unknown"

    func_name = getattr(item, "originalname", None) or getattr(item, "name", "test")

    # Parametrized tests: include a stable suffix so contexts don't collide
    nodeid = getattr(item, "nodeid", "")
    if "[" in nodeid:
        normalized = _normalize_pytest_nodeid(nodeid)
        if normalized is None:
            normalized = hashlib.md5(nodeid.encode("utf-8")).hexdigest()[:8]
        func_name = f"{func_name}/{normalized}"

    # Mirror production-like hierarchy: .../<DefaultUser>/<Assistant>
    return f"{test_path}/{func_name}/{DEFAULT_USER_CONTEXT}/{DEFAULT_ASSISTANT_CONTEXT}"


def _set_unify_context_for_test(item: pytest.Item) -> None:
    """Set a unique per-test Unify context early (before fixtures)."""
    ctx = _derive_test_context(item)
    setattr(item, "_unity_unify_test_ctx", ctx)

    # Clean slate unless contexts are pre-created during collection.
    skip_ctx_create = False
    if SETTINGS.UNIFY_PRETEST_CONTEXT_CREATE:
        skip_ctx_create = ctx in PRECREATED_CONTEXTS
    else:
        try:
            unify.delete_context(ctx)
        except Exception:
            pass

    unify.set_context(ctx, relative=False, skip_create=skip_ctx_create)

    # Ensure singleton registries don't leak across tests and that fixtures see
    # the correct context for any context-derived subcontexts (e.g. FunctionManager).
    try:
        from unity.common.context_registry import ContextRegistry
        from unity.manager_registry import ManagerRegistry
        from unity.events.event_bus import EVENT_BUS

        ManagerRegistry.clear()
        ContextRegistry.clear()
        EVENT_BUS.clear(delete_contexts=False)
    except Exception:
        pass


def _unset_unify_context_for_test(item: pytest.Item) -> None:
    """Unset (and optionally delete) the per-test Unify context after fixture teardown."""
    ctx = getattr(item, "_unity_unify_test_ctx", None)
    try:
        if ctx and SETTINGS.UNIFY_DELETE_CONTEXT_ON_EXIT:
            try:
                unify.delete_context(ctx)
            except Exception:
                pass
    finally:
        try:
            unify.unset_context()
        except Exception:
            pass


def pytest_report_header(config):
    settings_str = [f"{k}={v}" for k, v in SETTINGS.model_dump().items()]
    return [
        f"orchestra_url={os.environ.get('ORCHESTRA_URL')}",
        f"unity_comms_url={os.environ.get('UNITY_COMMS_URL')}",
        f"unify_project={unify.active_project()}",
        f"UNILLM_CACHE={os.environ.get('UNILLM_CACHE', 'not set')}",
    ] + settings_str


# --------------------------------------------------------------------------- #
# 2. Test stubs (Redis, ComputerWorker, DateTime)                              #
# --------------------------------------------------------------------------- #

_FIXED_DATETIME = datetime(2025, 6, 13, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture(scope="session")
def static_now():
    return _FIXED_DATETIME


@pytest.fixture(autouse=True)
def stub_external_deps(monkeypatch):
    """
    This fixture automatically stubs heavy external dependencies for tests,
    namely Redis. It runs for every test.
    """

    # --- Redis stub -----------------------------------------------------------
    class _FakePubSub:
        def __init__(self):
            self._messages = []
            self._thread = None

        def subscribe(self, *args, **kwargs):
            # Support both positional and keyword arguments
            # Keyword args are channel_name=handler_function pairs
            pass

        def listen(self):
            while self._messages:
                yield self._messages.pop()
            while True:
                # Keep the loop alive without blocking
                yield {"type": "noop"}

        def get_message(self):
            return None

        def run_in_thread(self, daemon=True):
            # Mock implementation that returns a fake thread with stop() method
            class StoppableThread(threading.Thread):
                def __init__(self, *args, **kwargs):
                    super().__init__(*args, **kwargs)
                    self._stop_flag = False

                def stop(self):
                    self._stop_flag = True

            self._thread = StoppableThread(target=lambda: None, daemon=daemon)
            return self._thread

    class _FakeRedis:
        def __init__(self, *a, **k):
            self._pubsub = _FakePubSub()
            self.published: list[tuple[str, str]] = []

        def pubsub(self, **kwargs):
            return self._pubsub

        def publish(self, chan, msg):
            self.published.append((chan, msg))

    # Safely patch redis.Redis with our fake version
    monkeypatch.setattr("redis.Redis", _FakeRedis)

    # --- DateTime stub for prompts (centralized) -----------------------------------
    # All timestamps in prompts come from prompt_helpers.now() which returns either:
    # - A formatted string (as_string=True): "Friday, June 13, 2025 at 12:00 PM UTC"
    # - A datetime object (as_string=False): for timestamp comparisons
    #
    # When UNITY_INCREMENTING_TIMESTAMPS is enabled (e.g., ConversationManager tests),
    # datetime objects auto-increment by microseconds so last_snapshot < message.timestamp
    # comparisons work correctly for **NEW** markers.

    from datetime import timedelta

    _timestamp_counter = {"value": 0}

    def _static_now(time_only: bool = False, as_string: bool = True):
        """Return a fixed timestamp for testing."""
        if SETTINGS.UNITY_INCREMENTING_TIMESTAMPS and not as_string:
            # Return incrementing datetime for **NEW** marker comparisons
            _timestamp_counter["value"] += 1
            return _FIXED_DATETIME + timedelta(microseconds=_timestamp_counter["value"])

        if not as_string:
            return _FIXED_DATETIME

        label = "UTC"
        if time_only:
            return _FIXED_DATETIME.strftime("%I:%M %p ") + label
        return _FIXED_DATETIME.strftime("%A, %B %d, %Y at %I:%M %p ") + label

    # Patch prompt_helpers.now everywhere it's imported
    monkeypatch.setattr("unity.common.prompt_helpers.now", _static_now)
    monkeypatch.setattr("unity.guidance_manager.prompt_builders.now", _static_now)
    monkeypatch.setattr("unity.secret_manager.prompt_builders.now", _static_now)
    monkeypatch.setattr("unity.image_manager.prompt_builders.now", _static_now)
    monkeypatch.setattr("unity.memory_manager.prompt_builders.now", _static_now)
    monkeypatch.setattr("unity.file_manager.prompt_builders.now", _static_now)
    monkeypatch.setattr("unity.conversation_manager.prompt_builders.now", _static_now)
    monkeypatch.setattr("unity.conversation_manager.events.prompt_now", _static_now)
    monkeypatch.setattr(
        "unity.conversation_manager.domains.contact_index.prompt_now",
        _static_now,
    )
    monkeypatch.setattr(
        "unity.conversation_manager.domains.managers_utils.prompt_now",
        _static_now,
    )
    monkeypatch.setattr(
        "unity.conversation_manager.conversation_manager.prompt_now",
        _static_now,
    )

    def _static_perf_counter() -> float:
        return 1000.0

    monkeypatch.setattr(
        "unity.common._async_tool.time_context.perf_counter",
        _static_perf_counter,
    )


# --------------------------------------------------------------------------- #
# 3. Singleton isolation                                                      #
# --------------------------------------------------------------------------- #

from unity.common.context_registry import ContextRegistry
from unity.manager_registry import ManagerRegistry


@pytest.fixture(autouse=True)
def _clear_singletons_between_tests():
    """Ensure *singleton* instances never leak from one test to the next."""
    yield
    ManagerRegistry.clear()  # Clear the registry after each test
    ContextRegistry.clear()  # Clear the context handler after each test


@pytest.fixture(autouse=True)
def _enable_eventbus_for_marked_tests(request):
    """Enable EventBus publishing for tests marked with @pytest.mark.enable_eventbus.

    By default, EventBus publishing is disabled during tests (via SETTINGS).
    Tests that need to verify event publishing behavior opt-in via the marker.
    """
    from unity.events.event_bus import EventBus

    if request.node.get_closest_marker("enable_eventbus"):
        EventBus._publishing_enabled = True
        yield
        EventBus._publishing_enabled = SETTINGS.EVENTBUS_PUBLISHING_ENABLED
    else:
        yield


# --------------------------------------------------------------------------- #
# 4. Command-line options                                                     #
# --------------------------------------------------------------------------- #


def pytest_addoption(parser):
    parser.addoption(
        "--unify-stub",
        action="store_true",
        help="Use an in-memory stub for unite.log / projects whilst "
        "leaving LLM calls intact.",
    )
    parser.addoption(
        "--overwrite-scenarios",
        action="store_true",
        default=False,
        help="Delete and recreate all test scenarios from scratch.",
    )
    parser.addoption(
        "--test-tags",
        action="store",
        default="",
        help="Comma-separated list of tags to associate with this test run "
        "(logged to the Combined context). Falls back to UNIFY_TEST_TAGS env var.",
    )

    group = parser.getgroup("custom-logging")
    group.addoption(
        "--test-log-enable",
        action="store_true",
        default=False,
        help="Enable test-aware logging (adds test name to log records).",
    )
    group.addoption(
        "--test-log-file",
        action="store",
        default="tests.log",
        help="Filename to write test-aware logs to (only applies if --test-log-enable is used).",
    )
    group.addoption(
        "--test-log-format",
        action="store",
        default="[%(levelname)s] %(asctime)s - %(test_name)s: %(message)s",
        help="Custom log format string (only applies if --test-log-enable is used).",
    )


# --------------------------------------------------------------------------- #
# 5. Custom logging helpers                                                   #
# --------------------------------------------------------------------------- #


class TestNameLogFilter(logging.Filter):
    def __init__(self):
        super().__init__()
        self.test_name = None

    def set_test_name(self, test_name):
        self.test_name = test_name.split("tests/")[-1]

    def reset_test_name(self):
        self.test_name = ""

    def filter(self, record):
        record.test_name = self.test_name or "UNKNOWN"
        return True


test_name_log_filter = TestNameLogFilter()


@pytest.fixture(scope="session", autouse=True)
def configure_logging(request):
    config = request.config
    if not is_test_logging_enabled(config):
        return

    logger = logging.getLogger()
    file_handler = logging.FileHandler(get_test_log_file(config), mode="w")
    formatter = logging.Formatter(
        get_test_log_format(config),
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(formatter)
    file_handler.addFilter(test_name_log_filter)
    logger.addHandler(file_handler)


def is_test_logging_enabled(config):
    return config.getoption("--test-log-enable")


def get_test_log_file(config):
    return config.getoption("--test-log-file")


def get_test_log_format(config):
    return config.getoption("--test-log-format")


# --------------------------------------------------------------------------- #
# 6. Session lifecycle hooks                                                  #
# --------------------------------------------------------------------------- #


def pytest_sessionstart(session):
    # ------------------------------------------------------------------
    #  Initialize Unity's OpenTelemetry TracerProvider FIRST
    #  This ensures Unity owns the provider (service: "unity") before
    #  any library (unify, unillm) makes traced calls.
    # ------------------------------------------------------------------
    from unity.logger import get_tracer

    get_tracer()  # Creates TracerProvider with service="unity" if OTEL enabled

    # ------------------------------------------------------------------
    #  Optionally delete the project before starting (clean slate)
    #  Skip in shared project mode (UNIFY_SKIP_SESSION_SETUP) because
    #  parallel_run.sh handles deletion at the script level to avoid
    #  race conditions between parallel sessions.
    # ------------------------------------------------------------------

    project_name = SETTINGS.test_project_name

    if (
        SETTINGS.UNIFY_TESTS_DELETE_PROJ_ON_START
        and not SETTINGS.UNIFY_SKIP_SESSION_SETUP
    ):
        try:
            unify.delete_project(project_name)
        except Exception:
            pass  # Project may not exist yet

    # ------------------------------------------------------------------
    #  Activate the UnityTests project
    # ------------------------------------------------------------------

    if os.environ.get("CI"):
        import unillm

        unillm.set_cache_backend("local_separate")

    if SETTINGS.UNIFY_SKIP_SESSION_SETUP:
        # Project and shared contexts already prepared externally (e.g., by
        # ._prepare_shared_project.sh). Just activate without overwrite.
        unify.activate(project_name, overwrite=False)
        unify.set_user_logging(False)
    else:
        unify.activate(
            project_name,
            overwrite=SETTINGS.UNIFY_OVERWRITE_PROJECT,
        )
        unify.set_user_logging(False)

    # ------------------------------------------------------------------
    #  Ensure the unity runtime is fully initialised for the test suite
    # ------------------------------------------------------------------

    import unity  # local import to avoid affecting stub installation order

    try:
        unity.init(project_name)
    except Exception:
        # Fallback to default project if UnityTests not available yet
        unity.init()

    # ------------------------------------------------------------------
    #  Configure EventBus publishing (disabled by default in tests)
    # ------------------------------------------------------------------
    from unity.events.event_bus import EventBus

    EventBus._publishing_enabled = SETTINGS.EVENTBUS_PUBLISHING_ENABLED

    # ------------------------------------------------------------------
    #  Parse and store session-level test tags for duration logging
    #  Priority: CLI --test-tags > env var UNIFY_TEST_TAGS
    # ------------------------------------------------------------------
    tags_raw = session.config.getoption("--test-tags", default="")
    if not tags_raw:
        tags_raw = SETTINGS.UNIFY_TEST_TAGS
    tags = [t.strip() for t in tags_raw.split(",") if t.strip()]
    set_session_tags(tags)

    # ------------------------------------------------------------------
    #  Ensure the Combined context exists for duration and LLM I/O logging
    #  (idempotent: tolerates pre-existing context/fields and concurrent
    #  creation attempts from parallel pytest sessions)
    # ------------------------------------------------------------------
    if SETTINGS.UNIFY_SKIP_SESSION_SETUP:
        # Combined context already prepared externally; skip creation
        pass
    else:
        unify.create_context("Combined")
        try:
            unify.create_fields(
                context="Combined",
                fields={
                    "test_fpath": {"type": "str", "mutable": True},
                    "tags": {"type": "list", "mutable": True},
                    "duration": {"type": "float", "mutable": True},
                    "llm_io": {"type": "list", "mutable": True},
                    "settings": {"type": "dict", "mutable": True},
                },
            )
        except Exception:
            pass  # Fields already exist or transient failure


def pytest_sessionfinish(session, exitstatus):
    # Write cache stats to a temp file for parallel_run.sh to consume
    # The file is keyed by UNITY_TMUX_SESSION_ID env var (set by parallel_run.sh)
    try:
        import unillm

        stats = unillm.get_cache_stats()
        session_id = os.environ.get("UNITY_TMUX_SESSION_ID", "")
        if session_id:
            stats_file = f"/tmp/parallel_run_cache_{session_id}.txt"
            with open(stats_file, "w") as f:
                f.write(f"{stats.hits}|{stats.misses}\n")
    except Exception:
        pass  # Don't fail the test run if cache stats writing fails

    if SETTINGS.UNIFY_TESTS_DELETE_PROJ_ON_EXIT:
        unify.delete_project(unify.active_project())


def pytest_unconfigure(config):
    """Restore HOME and clean up the temporary test home directory."""
    import shutil

    test_home = os.environ.get("HOME", "")
    if _original_home is None:
        os.environ.pop("HOME", None)
    else:
        os.environ["HOME"] = _original_home
    if test_home.startswith("/tmp/") and "unity_test_home_" in test_home:
        shutil.rmtree(test_home, ignore_errors=True)


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    if SETTINGS.UNITY_CACHE_STATS:
        import unillm

        stats = unillm.get_cache_stats()
        terminalreporter.section(
            f"Unify cache report | Hits ({stats.get_percentage_of_cache_hits():.2f}%): {stats.hits} | Misses ({stats.get_percentage_of_cache_misses():.2f}%): {stats.misses} | Reads: {stats.reads} | Writes: {stats.writes}",
        )


# --------------------------------------------------------------------------- #
# 7. Test run hooks                                                           #
# --------------------------------------------------------------------------- #


_original_home: str | None = None


def pytest_configure(config):
    # ------------------------------------------------------------------
    # Isolate HOME so that tests never touch the real home directory.
    # get_local_root() defaults to ~/Unity/Local, and the process cwd
    # is set to the same path at startup.  By pointing HOME at a temp
    # dir we keep Downloads/, .env, snapshots, etc. sandboxed.
    # ------------------------------------------------------------------
    import tempfile

    global _original_home
    _original_home = os.environ.get("HOME")
    test_home = tempfile.mkdtemp(prefix="unity_test_home_")
    os.environ["HOME"] = test_home

    config.addinivalue_line(
        "markers",
        "requires_real_unify: mark test as requiring the real unify implementation",
    )
    config.addinivalue_line(
        "markers",
        "eval: mark a test as a fuzzy evaluation test for English language APIs",
    )
    config.addinivalue_line(
        "markers",
        "enable_eventbus: enable EventBus publishing for this test",
    )
    config.addinivalue_line(
        "markers",
        "requires_orchestra: mark test as requiring a running Orchestra server",
    )

    # Required to disable explicit log level if set from pytest.ini or command line options
    if os.environ.get("UNITY_TESTS_CLI_LOGGING", "true").lower() == "false":
        config.option.log_cli_level = None
        config.option.showcapture = "no"
        config.option.capture = "no"

    config.stash[metadata_key]["Settings"] = SETTINGS.model_dump(mode="json")

    # ------------------------------------------------------------------ #
    # Prune non-pytest console handlers so only pytest live logs appear. #
    # Keeps any file handlers (e.g., when --test-log-enable is used).    #
    # ------------------------------------------------------------------ #
    try:
        root = logging.getLogger()
        kept_handlers: list[logging.Handler] = []
        for h in list(root.handlers):
            mod = getattr(h.__class__, "__module__", "")
            is_stream = isinstance(h, logging.StreamHandler)
            is_pytest = mod.startswith("_pytest.logging")
            # Retain pytest's handlers and any non-stream handlers (file, etc.)
            if is_stream and not is_pytest:
                continue
            kept_handlers.append(h)
        root.handlers = kept_handlers
    except Exception:
        # Never fail configuration due to logging hygiene adjustments.
        pass


# Skip tests marked with requires_real_unify when using the unify stub
# Skip tests marked with requires_orchestra when Orchestra is not available
def pytest_runtest_setup(item):
    test_name_log_filter.set_test_name(item.nodeid)
    _set_unify_context_for_test(item)

    # Skip requires_orchestra tests if Orchestra is not running
    if item.get_closest_marker("requires_orchestra"):
        if not _check_orchestra_available():
            pytest.skip("Orchestra server not available")


def _normalize_pytest_nodeid(nodeid):
    """
    Try to normalize the pytest nodeid to an alphanumeric string that is
    accepted for unify.Context path. If not possible, return None.
    Will fallback to invocation count if empty.
    """
    bracket_match = re.search(r"\[([^\]]+)\]", nodeid)
    if bracket_match:
        bracket_content = bracket_match.group(1)
    else:
        bracket_content = ""

    # Try to normalize to alphanumeric
    normalized = re.sub(r"[^a-zA-Z0-9]", "", bracket_content)

    if len(normalized) == 0:
        return None

    return normalized[:24]


def pytest_runtest_call(item):
    import types

    func_name = item.originalname

    # For class-based tests, item.obj is a bound method. We need to access
    # the underlying function via __func__ to set/get attributes.
    target_obj = item.obj
    if isinstance(target_obj, types.MethodType):
        target_obj = target_obj.__func__

    if "[" in item.nodeid:  # Any parametrization (markers, fixtures, etc.)
        # Need to keep track of invocation count for parametrized tests
        # In case of a later failure.
        current_count = getattr(target_obj, "_unity_pytest_invocation_count", 0)
        setattr(target_obj, "_unity_pytest_invocation_count", current_count + 1)

        normalized_id = _normalize_pytest_nodeid(item.nodeid)
        if normalized_id is None:
            normalized_id = f"_{current_count}_"
        func_name = f"{func_name}/{normalized_id}"

    setattr(target_obj, "_unity_pytest_nodeid", func_name)


def pytest_runtest_teardown(item, nextitem=None):
    _unset_unify_context_for_test(item)
    test_name_log_filter.reset_test_name()


def pytest_html_results_summary(prefix, summary, postfix):
    if SETTINGS.UNITY_CACHE_STATS:
        import unillm

        stats = unillm.get_cache_stats()
        prefix.extend(
            [
                f"<h4>Unify Cache Stats Report:</h4>",
                f"<p>Hits ({stats.get_percentage_of_cache_hits():.2f}%): {stats.hits} | Misses ({stats.get_percentage_of_cache_misses():.2f}%): {stats.misses}</p>",
                f"<p>Reads: {stats.reads} | Writes: {stats.writes}</p>",
            ],
        )


# --------------------------------------------------------------------------- #
# 8. HTTP client cleanup                                                      #
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="session", autouse=True)
def _close_httpx_clients_at_session_end():
    """
    Track every httpx.AsyncClient that gets created during the session
    and close it gracefully *before* pytest tears the event-loop down.
    """
    created: list[httpx.AsyncClient] = []

    # monkey-patch __init__ to collect instances
    orig_init = httpx.AsyncClient.__init__

    def _patched_init(self, *a, **kw):
        orig_init(self, *a, **kw)
        created.append(self)

    httpx.AsyncClient.__init__ = _patched_init  # type: ignore[assignment]

    yield  # ← tests run here

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    for c in created:
        if not c.is_closed:
            # swallow "loop closed" if it still happens for a stray client
            try:
                loop.run_until_complete(c.aclose())
            except RuntimeError:
                pass
    loop.run_until_complete(loop.shutdown_asyncgens())
    loop.close()


# --------------------------------------------------------------------------- #
# 9. Pre-run context creation                                                 #
# --------------------------------------------------------------------------- #


def _get_context_name_for_item(item):
    """
    Return the unify context name for a collected pytest item.

    This uses the same logic as tests.helpers._ctx_name to generate a unique context
    path for the test, including any parametrization suffixes.
    """
    original_name = item.originalname or item.name

    # Append normalized parametrization suffix if available
    normalized = _normalize_pytest_nodeid(item.nodeid)

    func_name = original_name
    if normalized is not None:
        func_name = f"{original_name}/{normalized}"

    path = item._nodeid.split(".py")[0]
    return f"{path}/{func_name}"


def pytest_collection_finish(session):
    # Compute all contexts and fire off background creation tasks
    # Skip when UNIFY_SKIP_SESSION_SETUP is set (shared project mode)
    if SETTINGS.UNIFY_PRETEST_CONTEXT_CREATE and not SETTINGS.UNIFY_SKIP_SESSION_SETUP:
        contexts: set[str] = set()
        for item in session.items:
            ctx = _get_context_name_for_item(item)
            contexts.add(ctx)
            contexts.add(f"{ctx}/Events/_callbacks/")

        # TODO: Should delete contexts before creating them
        # But this is mostly fine now for CI purpose, as we create
        # a fresh project anyway
        unify.create_contexts(list(contexts))
        PRECREATED_CONTEXTS.update(contexts)


@pytest.fixture(autouse=True)
def _set_random_seed():
    random.seed(42)
