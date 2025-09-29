"""
tests/conftest.py
=================

Global pytest configuration.

• `--unify-stub` *or* `USE_UNIFY_STUB=1` ➜ replace the **persistence** parts
  of the `unify` SDK with an in-memory implementation, while *optionally*
  keeping the real `unify.Unify` class for live LLM calls.

  – With flag         → in-memory logs, live LLM
  – Without flag      → untouched, everything goes to real backend
"""

from __future__ import annotations

import logging
import os
from pytest_metadata.plugin import metadata_key
import httpx
import asyncio
import pytest
import re
import threading
import random
import string
import unify

from tests.helpers import (
    SETTINGS,
    PRECREATED_CONTEXTS,
)


def pytest_report_header(config):
    settings_str = [f"{k}={v}" for k, v in SETTINGS.model_dump().items()]
    return [
        f"unify_base_url={os.environ.get('UNIFY_BASE_URL')}",
        f"unify_project={unify.active_project()}",
    ] + settings_str


# --------------------------------------------------------------------------- #
#  Controller Dependency Stubbing Fixture                                     #
# --------------------------------------------------------------------------- #


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

    # --- DateTime stub for prompt builders for all managers -----------------------------------
    def _static_now():
        """Return a fixed timestamp for consistent test caching."""
        return "2025-06-13 12:00:00 UTC"  # Friday, June 13, 2025 at noon UTC

    # Patch all _now functions in prompt builders
    monkeypatch.setattr("unity.contact_manager.prompt_builders._now", _static_now)
    monkeypatch.setattr("unity.knowledge_manager.prompt_builders._now", _static_now)
    monkeypatch.setattr("unity.conductor.prompt_builders._now", _static_now)
    monkeypatch.setattr("unity.task_scheduler.prompt_builders._now", _static_now)
    monkeypatch.setattr("unity.transcript_manager.prompt_builders._now", _static_now)


# --------------------------------------------------------------------------- #
#  Singleton-registry isolation fixture                                        #
# --------------------------------------------------------------------------- #

from unity.singleton_registry import SingletonRegistry


@pytest.fixture(autouse=True)
def _clear_singletons_between_tests():
    """Ensure *singleton* instances never leak from one test to the next."""
    yield
    SingletonRegistry.clear()  # Clear the registry after each test


# --------------------------------------------------------------------------- #
#  Command-line flag                                                          #
# --------------------------------------------------------------------------- #


def pytest_addoption(parser):
    parser.addoption(
        "--unify-stub",
        action="store_true",
        help="Use an in-memory stub for unite.log / projects whilst "
        "leaving LLM calls intact.",
    )
    parser.addoption(
        "--no-reuse-scenario",
        action="store_true",
        default=False,
        help="Force fresh scenario creation.",
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


# -------------------------
# Custom Logging Helpers
# -------------------------


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
#  Session-wide hook – install stub *before* any project imports              #
# --------------------------------------------------------------------------- #


def _generate_random_project_name():
    return f"UnityTests_{''.join(random.choices(string.ascii_letters + string.digits, k=8))}"


def pytest_sessionstart(session):
    # ------------------------------------------------------------------
    #  Activate the UnityTests project
    # ------------------------------------------------------------------

    project_name = (
        _generate_random_project_name()
        if SETTINGS.UNIFY_TESTS_RAND_PROJ
        else "UnityTests"
    )

    if os.environ.get("CI"):
        unify.set_cache_backend("local_separate")

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


def pytest_sessionfinish(session, exitstatus):
    if SETTINGS.UNIFY_TESTS_DELETE_PROJ_ON_EXIT:
        unify.delete_project(unify.active_project())


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    if SETTINGS.UNIFY_CACHE_BENCHMARK:
        stats = unify.get_cache_stats()
        terminalreporter.section(
            f"Unify cache report | Hits ({stats.get_percentage_of_cache_hits():.2f}%): {stats.hits} | Misses ({stats.get_percentage_of_cache_misses():.2f}%): {stats.misses} | Reads: {stats.reads} | Writes: {stats.writes}",
        )


# Define a marker for tests that require the real unify
def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "requires_real_unify: mark test as requiring the real unify implementation",
    )
    config.addinivalue_line(
        "markers",
        "unit: mark a test as a deterministic unit test",
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

    # Enable xdist only on CI
    if os.environ.get("CI"):
        setattr(config.option, "numprocesses", "auto")
        setattr(config.option, "dist", "loadfile")

    config.stash[metadata_key]["Settings"] = SETTINGS.model_dump()


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


def _is_test_parametrized(item):
    return any(i.name == "parametrize" for i in item.iter_markers())


def pytest_runtest_call(item):
    func_name = item.originalname
    if _is_test_parametrized(item):
        # Need to keep track of invocation count for parametrized tests
        # In case of a later failure.
        current_count = getattr(item.obj, "_unity_pytest_invocation_count", 0)
        setattr(item.obj, "_unity_pytest_invocation_count", current_count + 1)

        normalized_id = _normalize_pytest_nodeid(item.nodeid)
        if normalized_id is None:
            normalized_id = f"_{current_count}_"
        func_name = f"{func_name}/{normalized_id}"

    setattr(item.obj, "_unity_pytest_nodeid", func_name)


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
#  httpx cleanup
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
#  Pre-run context creation (to minimize API calls)
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
    if SETTINGS.UNIFY_PRETEST_CONTEXT_CREATE:
        contexts: set[str] = set()
        for item in session.items:
            ctx = _get_context_name_for_item(item)
            contexts.add(ctx)
            contexts.add(f"{ctx}/Events/_callbacks/")
            if SETTINGS.UNIFY_TRACED:
                contexts.add(f"{ctx}/Traces/")

        # TODO: Should delete contexts before creating them
        # But this is mostly fine now for CI purpose, as we create
        # a fresh project anyway
        unify.create_contexts(list(contexts))
        PRECREATED_CONTEXTS.update(contexts)
