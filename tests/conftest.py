"""
tests/conftest.py
=================

Global pytest configuration for Unity test suite.

Sections:
  1. Imports and logging guard
  2. Test stubs (Redis, BrowserWorker, DateTime)
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


def pytest_report_header(config):
    settings_str = [f"{k}={v}" for k, v in SETTINGS.model_dump().items()]
    return [
        f"unify_base_url={os.environ.get('UNIFY_BASE_URL')}",
        f"unify_project={unify.active_project()}",
    ] + settings_str


# --------------------------------------------------------------------------- #
# 2. Test stubs (Redis, BrowserWorker, DateTime)                              #
# --------------------------------------------------------------------------- #

_FIXED_DATETIME = datetime(2025, 6, 13, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture(scope="session")
def static_now():
    return _FIXED_DATETIME


@pytest.fixture(autouse=True)
def stub_controller_deps(monkeypatch):
    """
    This fixture automatically stubs heavy dependencies for the Controller tests,
    namely Redis and the BrowserWorker. It runs for every test.
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

    # --- BrowserWorker stub ---------------------------------------------------
    class _DummyWorker:
        def __init__(self, *a, **k):
            self.started = False
            self.stopped = False

        def start(self):
            self.started = True

        def stop(self):
            self.stopped = True

        def join(self, *a, **k):
            pass

    # Safely patch the BrowserWorker class where it's defined
    monkeypatch.setattr(
        "unity.controller.playwright_utils.worker.BrowserWorker",
        _DummyWorker,
    )

    # --- DateTime stub for prompts (centralized) -----------------------------------
    # Two sources of timestamps appear in LLM prompts:
    # 1. prompt_helpers.now() -> string for "Current UTC time is..." footer
    # 2. Event/Message timestamps -> datetime for "@ Friday, June 13, 2025..." in messages
    # We stub both to the same fixed point in time for cache consistency.

    def _static_now(time_only: bool = False):
        """Return a fixed timestamp string for prompt footers."""
        label = "UTC"
        return (
            _FIXED_DATETIME.strftime("%H:%M:%S ") + label
            if time_only
            else _FIXED_DATETIME.strftime("%Y-%m-%d %H:%M:%S ") + label
        )

    # Patch prompt_helpers.now for prompt footers
    monkeypatch.setattr("unity.common.prompt_helpers.now", _static_now)
    monkeypatch.setattr("unity.guidance_manager.prompt_builders.now", _static_now)
    monkeypatch.setattr("unity.secret_manager.prompt_builders.now", _static_now)
    monkeypatch.setattr("unity.image_manager.prompt_builders.now", _static_now)
    monkeypatch.setattr("unity.memory_manager.prompt_builders.now", _static_now)
    monkeypatch.setattr("unity.file_manager.prompt_builders.now", _static_now)

    # Patch events._get_now for Event/Message timestamps in renderer output
    monkeypatch.setattr(
        "unity.conversation_manager.events._get_now",
        lambda: _FIXED_DATETIME,
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
        unify.set_cache_backend("local_separate")

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
    if SETTINGS.UNIFY_TESTS_DELETE_PROJ_ON_EXIT:
        unify.delete_project(unify.active_project())


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    if SETTINGS.UNIFY_CACHE_BENCHMARK:
        stats = unify.get_cache_stats()
        terminalreporter.section(
            f"Unify cache report | Hits ({stats.get_percentage_of_cache_hits():.2f}%): {stats.hits} | Misses ({stats.get_percentage_of_cache_misses():.2f}%): {stats.misses} | Reads: {stats.reads} | Writes: {stats.writes}",
        )


# --------------------------------------------------------------------------- #
# 7. Test run hooks                                                           #
# --------------------------------------------------------------------------- #


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "requires_real_unify: mark test as requiring the real unify implementation",
    )
    config.addinivalue_line(
        "markers",
        "eval: mark a test as a fuzzy evaluation test for English language APIs",
    )

    # Required to disable explicit log level if set from pytest.ini or command line options
    if os.environ.get("UNITY_TESTS_CLI_LOGGING", "true").lower() == "false":
        config.option.log_cli_level = None
        config.option.showcapture = "no"
        config.option.capture = "no"

    config.stash[metadata_key]["Settings"] = SETTINGS.model_dump()

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
def pytest_runtest_setup(item):
    test_name_log_filter.set_test_name(item.nodeid)


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


def pytest_runtest_teardown(item):
    test_name_log_filter.reset_test_name()


def pytest_html_results_summary(prefix, summary, postfix):
    if SETTINGS.UNIFY_CACHE_BENCHMARK:
        stats = unify.get_cache_stats()
        prefix.extend(
            [
                f"<h4>Unify Cache Benchmark Report:</h4>",
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
