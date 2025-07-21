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
import json
import os
import sys
import types
import importlib
from typing import Any, Dict, List, Optional
import httpx
import asyncio
import pytest
import re
import threading
import unify

unify.activate(
    "UnityTests",
    overwrite=json.loads(os.getenv("UNIFY_OVERWRITE_PROJECT", "false")),
)
unify.set_user_logging(False)


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
#  Command-line flag                                                          #
# --------------------------------------------------------------------------- #

# Flag to track if we're using the stub version
_using_unify_stub = False


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


def pytest_sessionstart(session):
    global _using_unify_stub
    cmd_flag = session.config.getoption("--unify-stub")
    env_var = os.getenv("USE_UNIFY_STUB")

    # Only consider env_var as True if it's set to a non-zero/non-empty value
    use_env_var = env_var and env_var.lower() not in ("0", "false", "no", "")

    use_stub = cmd_flag or use_env_var

    if use_stub:
        _using_unify_stub = True
        _install_unify_stub()
        _install_requests_mock()
    else:
        _using_unify_stub = False

    # ------------------------------------------------------------------
    #  Ensure the unity runtime is fully initialised for the test suite
    # ------------------------------------------------------------------

    import unity  # local import to avoid affecting stub installation order

    try:
        unity.init("UnityTests")
    except Exception:
        # Fallback to default project if UnityTests not available yet
        unity.init()


# Function to check if we're using the unify stub
def is_using_unify_stub():
    """Return True if tests are running with the unify stub."""
    global _using_unify_stub
    return _using_unify_stub


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


# Skip tests marked with requires_real_unify when using the unify stub
def pytest_runtest_setup(item):
    if any(mark.name == "requires_real_unify" for mark in item.iter_markers()):
        if is_using_unify_stub():
            pytest.skip("Test requires real unify implementation")

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


# Pytest fixture to skip tests that require the real unify
@pytest.fixture
def requires_real_unify(request):
    """Skip tests if unify stub is being used."""
    if is_using_unify_stub():
        pytest.skip("Test requires real unify implementation")


# --------------------------------------------------------------------------- #
#  Helper: mock requests library                                              #
# --------------------------------------------------------------------------- #


def _install_requests_mock():
    """Mock the requests library for unify API calls during tests."""
    import sys
    import types

    class MockResponse:
        def __init__(self, json_data, status_code=200):
            self.json_data = json_data
            self.status_code = status_code
            self.text = str(json_data)

        def json(self):
            return self.json_data

        def raise_for_status(self):
            if self.status_code >= 400:
                from requests.exceptions import HTTPError

                raise HTTPError(f"{self.status_code} Error", response=self)

    class MockRequests:
        @staticmethod
        def _get_columns_from_log(context):
            """Helper function to extract column metadata from logs."""
            import unify

            # Get direct access to the internal logs storage
            store = None
            unify_module = sys.modules.get("unify")
            if unify_module and hasattr(unify_module, "_ctx_store"):
                try:
                    # Get the logs directly from the store
                    ctx_store = getattr(unify_module, "_ctx_store")
                    store = ctx_store(context)
                except:
                    # Fall back to regular API
                    pass

            # If we couldn't get direct access, use the normal API
            if store is None:
                store = unify.get_logs(context=context)

            # Find column metadata logs
            column_logs = [log for log in store if "__columns__" in log.entries]

            # Return column definitions if found
            if column_logs:
                return column_logs[0].entries.get("__columns__", {})

            return {}

        @staticmethod
        def request(method, url, json=None, headers=None, **kwargs):
            # Get or extract table name from URL
            import re

            if json:
                pass

            table_match = re.search(r"Knowledge/([^/]+)", url)
            table_name = table_match.group(1) if table_match else None

            # Process requests based on URL pattern
            if url == "https://api.unify.ai/v0/logs/rename_field":
                # Handle rename field request
                import unify

                if json:
                    context = json.get("context")
                    old_field_name = json.get("old_field_name")
                    new_field_name = json.get("new_field_name")

                    if context and old_field_name and new_field_name:
                        unify._rename_column(context, old_field_name, new_field_name)
                    else:
                        pass
                else:
                    pass

                return MockResponse({"success": True, "message": "Column renamed"})
            elif "/columns" in url:
                # Creating/modifying columns
                if table_name and json and method == "POST":
                    # Access the unify module directly
                    import unify

                    # Extract column definitions
                    column_definitions = {}
                    if json and "columns" in json:
                        for col_name, col_type in json["columns"].items():
                            # Store the column type directly, not as a dict
                            column_definitions[col_name] = col_type

                    # Find or create a log with __columns__ entry
                    column_logs = [
                        log
                        for log in unify.get_logs(context=f"Knowledge/{table_name}")
                        if "__columns__" in log.entries
                    ]

                    if column_logs:
                        # Update existing column metadata
                        column_log = column_logs[0]
                        existing = column_log.entries.get("__columns__", {})
                        column_log.update_entries(
                            __columns__={
                                **existing,
                                **column_definitions,
                            },
                        )
                    else:
                        # Create new column metadata log
                        unify.log(
                            context=f"Knowledge/{table_name}",
                            __columns__=column_definitions,
                        )

                return MockResponse(
                    {"success": True, "message": "Column operation successful"},
                )
            elif "/rename" in url and "contexts" in url:
                # Renaming tables
                import unify
                import re

                url_parts = url.split("/")

                old_context = None
                for i, part in enumerate(url_parts):
                    if part == "contexts" and i + 2 < len(url_parts):
                        old_context = f"{url_parts[i+1]}/{url_parts[i+2]}"
                        break

                new_context = json.get("name")

                if old_context and new_context:
                    unify.create_context(new_context)

                    # Get the logs from the old context
                    logs = unify.get_logs(context=old_context)

                    # For each log, copy its entries to the new context
                    for log in logs:
                        unify.log(context=new_context, **log.entries)

                    # Delete the old context
                    unify.delete_context(old_context)

                return MockResponse(
                    {"success": True, "message": "Table renamed successfully"},
                )
            elif "/logs/fields" in url:
                # This endpoint is called by knowledge_manager._get_columns
                import unify

                if method == "POST" and json:
                    project = json.get("project")
                    context = json.get("context")
                    fields = json.get("fields", {})

                    if context and fields:
                        # Find or create column metadata log
                        column_logs = [
                            log
                            for log in unify.get_logs(context=context)
                            if "__columns__" in log.entries
                        ]

                        if column_logs:
                            # Update existing column metadata
                            column_log = column_logs[0]
                            existing = column_log.entries.get("__columns__", {})
                            column_log.update_entries(
                                __columns__={**existing, **fields},
                            )
                        else:
                            # Create new column metadata log
                            unify.log(
                                context=context,
                                __columns__=fields,
                            )

                    return MockResponse(
                        {"success": True, "message": "Columns created successfully"},
                    )

                # Handle GET requests to retrieve column information
                # Extract query parameters
                import urllib.parse

                query = url.split("?")[-1] if "?" in url else ""
                params = dict(urllib.parse.parse_qsl(query))

                # Get context parameter - handle both direct parameter and URL pattern
                context = params.get("context")
                if not context and table_name:
                    context = f"Knowledge/{table_name}"

                if context:
                    # Use unify.get_fields to retrieve the column definitions for the context.
                    # This helper already inspects the raw log store (including metadata logs) and
                    # therefore reflects the authoritative list of columns for a context, exactly
                    # as the real Unify backend would.
                    column_data = unify.get_fields(context=context) or {}

                    # Format the payload in the same shape the real API returns – a mapping from
                    # field name to an object that at least contains the "data_type" key.
                    formatted_columns = {
                        name: {"data_type": dtype}
                        for name, dtype in column_data.items()
                    }

                    return MockResponse(formatted_columns)
            elif "/logs/derived" in url:
                # Creating derived columns
                if json:
                    context = json.get("context")
                    column_name = json.get("key")
                    equation = json.get("equation", "")
                    referenced_logs = json.get("referenced_logs", {})

                    if context and column_name and equation:
                        # Store in columns metadata using unify directly
                        import unify

                        # Get metadata logs
                        column_logs = [
                            log
                            for log in unify.get_logs(context=context)
                            if "__columns__" in log.entries
                        ]

                        # Store column type and equation in metadata
                        if column_logs:
                            # Update existing column metadata
                            column_log = column_logs[0]
                            column_log.update_entries(
                                __columns__={
                                    **column_log.entries.get("__columns__", {}),
                                    **{column_name: "derived"},
                                },
                                __equations__={
                                    **column_log.entries.get("__equations__", {}),
                                    **{column_name: equation},
                                },
                            )
                        else:
                            # Create new column metadata log
                            unify.log(
                                context=context,
                                __columns__={column_name: "derived"},
                                __equations__={column_name: equation},
                            )

                        # Get all logs except metadata
                        logs = [
                            log
                            for log in unify.get_logs(context=context)
                            if "__columns__" not in log.entries
                            and "__equations__" not in log.entries
                        ]

                        # Apply the derived column to all logs immediately
                        if logs:
                            # Simple equation parser - handle basic expressions with field references
                            # Try to evaluate with each log's fields
                            for log in logs:
                                try:
                                    # Replace field references with values
                                    eval_equation = equation

                                    # Handle field references like {lg:fieldname}
                                    import re

                                    field_refs = re.findall(
                                        r"\{([^{}]+):([^{}]+)\}",
                                        eval_equation,
                                    )

                                    # First get values for referenced fields
                                    local_vars = {}
                                    for ref_name, field_name in field_refs:
                                        ref_context = referenced_logs.get(
                                            ref_name,
                                            {},
                                        ).get("context", context)
                                        field_value = log.entries.get(field_name)
                                        if field_value is not None:
                                            local_vars[f"{ref_name}_{field_name}"] = (
                                                field_value
                                            )
                                            eval_equation = eval_equation.replace(
                                                f"{{{ref_name}:{field_name}}}",
                                                f"{ref_name}_{field_name}",
                                            )

                                    # Handle direct field references like {fieldname}
                                    direct_refs = re.findall(
                                        r"\{([^{}]+)\}",
                                        eval_equation,
                                    )
                                    for field_name in direct_refs:
                                        field_value = log.entries.get(field_name)
                                        if field_value is not None:
                                            local_vars[field_name] = field_value
                                            eval_equation = eval_equation.replace(
                                                f"{{{field_name}}}",
                                                field_name,
                                            )

                                    # Evaluate the equation with the field values
                                    result = eval(
                                        eval_equation,
                                        {"__builtins__": {}},
                                        local_vars,
                                    )
                                    log.entries[column_name] = result
                                except Exception as e:
                                    pass

                return MockResponse(
                    {"success": True, "message": "Derived column created"},
                )
            elif "/logs/rename_field" in url.lower():
                # Renaming columns
                import unify

                if json:
                    context = json.get("context")
                    old_field_name = json.get("old_field_name")
                    new_field_name = json.get("new_field_name")

                    if context and old_field_name and new_field_name:
                        # Get all non-metadata logs in the context
                        logs = [
                            log
                            for log in unify._ctx_store(context)
                            if "__columns__" not in log.entries
                        ]

                        # Rename the field in each log entry
                        for log in logs:
                            if old_field_name in log.entries:
                                # Preserve position of the field in the entries
                                old_value = log.entries.pop(old_field_name)

                                # Get the keys of the entries in their original order
                                keys = list(log.entries.keys())

                                # Create a new ordered dict with the new field name in place of the old one
                                new_entries = {}

                                # Find where the original field was in the order
                                # If it's a new field (not in the original), we'll add it at the beginning
                                original_keys = list(log.entries.keys())

                                # Loop through adding each key in original order
                                added_new_field = False

                                # Handle an empty log case
                                if not keys:
                                    new_entries[new_field_name] = old_value
                                else:
                                    # If the field was the first one, maintain that position
                                    if (
                                        len(original_keys) == 0
                                        or old_field_name < original_keys[0]
                                    ):
                                        new_entries[new_field_name] = old_value
                                        added_new_field = True

                                    # Add all other fields in their original order
                                    for k, v in log.entries.items():
                                        # If we haven't added the new field yet and we're past where
                                        # the old field would have been alphabetically, add it now
                                        if not added_new_field and k > old_field_name:
                                            new_entries[new_field_name] = old_value
                                            added_new_field = True
                                        new_entries[k] = v

                                    # If we haven't added the new field yet, add it at the end
                                    if not added_new_field:
                                        new_entries[new_field_name] = old_value

                                log.entries = new_entries

                        # Also update column metadata
                        column_logs = [
                            log
                            for log in unify._ctx_store(context)
                            if "__columns__" in log.entries
                        ]
                        if column_logs:
                            column_log = column_logs[0]
                            columns = column_log.entries.get("__columns__", {})
                            if old_field_name in columns:
                                updated_columns = {}
                                for col_name, col_type in columns.items():
                                    if col_name == old_field_name:
                                        updated_columns[new_field_name] = col_type
                                    else:
                                        updated_columns[col_name] = col_type
                                column_log.entries["__columns__"] = updated_columns

                    else:
                        pass

                return MockResponse({"success": True, "message": "Column renamed"})
            elif method == "DELETE" and "/columns/" in url:
                # Handle DELETE request to remove a column
                import re
                import unify

                # Extract context and column name from URL
                # Pattern example: /project/.../contexts/Knowledge/MyTable/columns/x
                column_pattern = re.search(
                    r"/contexts/([^/]+)/([^/]+)/columns/([^/?]+)",
                    url,
                )
                if column_pattern:
                    context = f"{column_pattern.group(1)}/{column_pattern.group(2)}"
                    column_name = column_pattern.group(3)

                    # Get all non-metadata logs in the context
                    logs = [
                        log
                        for log in unify._ctx_store(context)
                        if "__columns__" not in log.entries
                    ]

                    # Remove the column from each log entry
                    for log in logs:
                        if column_name in log.entries:
                            log.entries.pop(column_name, None)

                    # Also update column metadata
                    column_logs = [
                        log
                        for log in unify._ctx_store(context)
                        if "__columns__" in log.entries
                    ]
                    if column_logs:
                        column_log = column_logs[0]
                        columns = column_log.entries.get("__columns__", {})
                        if column_name in columns:
                            updated_columns = {
                                k: v for k, v in columns.items() if k != column_name
                            }
                            column_log.entries["__columns__"] = updated_columns

                return MockResponse(
                    {"success": True, "message": "Column deleted via DELETE"},
                )
            elif "/logs?delete_empty_logs=True" in url:
                # Deleting columns
                import re
                import unify
                import urllib.parse

                if json:
                    context = json.get("context")

                    # Extract column name from ids_and_fields format: [[log_id, field_name], ...]
                    ids_and_fields = json.get("ids_and_fields", [])
                    if (
                        ids_and_fields
                        and isinstance(ids_and_fields, list)
                        and len(ids_and_fields) > 0
                    ):
                        # The format appears to be [[log_id, field_name], ...] where log_id can be None
                        # to indicate deletion from all logs
                        first_field = ids_and_fields[0]
                        if isinstance(first_field, list) and len(first_field) > 1:
                            column_name = first_field[1]

                # If context and column were found, perform the deletion
                if context and column_name:
                    # Get all non-metadata logs in the context
                    logs = [
                        log
                        for log in unify._ctx_store(context)
                        if "__columns__" not in log.entries
                    ]

                    # Remove the column from each log entry
                    for log in logs:
                        if column_name in log.entries:
                            log.entries.pop(column_name, None)

                    # Also update column metadata
                    column_logs = [
                        log
                        for log in unify._ctx_store(context)
                        if "__columns__" in log.entries
                    ]
                    if column_logs:
                        column_log = column_logs[0]
                        columns = column_log.entries.get("__columns__", {})
                        if column_name in columns:
                            updated_columns = {
                                k: v for k, v in columns.items() if k != column_name
                            }
                            column_log.entries["__columns__"] = updated_columns

                return MockResponse({"success": True, "message": "Column deleted"})
            elif "/logs" in url:
                # Generic logs endpoint
                return MockResponse(
                    {"success": True, "message": "Log operation successful"},
                )
            else:
                # Default response
                return MockResponse(
                    {"success": True, "message": "Operation successful"},
                )

    # Create a module-like object
    mock_requests = types.ModuleType("requests")

    # Copy all the original requests attributes
    try:
        import requests as original_requests

        for attr in dir(original_requests):
            if not attr.startswith("__"):
                setattr(mock_requests, attr, getattr(original_requests, attr))
    except ImportError:
        pass  # If requests isn't available, we'll just use our mock

    # Override the request method
    mock_requests.request = MockRequests.request

    # Install our mock
    sys.modules["requests"] = mock_requests


# --------------------------------------------------------------------------- #
#  Helper: stub implementation                                                #
# --------------------------------------------------------------------------- #


def _install_unify_stub() -> None:  # noqa: C901 – long but linear
    """
    Monkey-patch the `unify` module so that:

      • log / project APIs are fully in-memory (no network / DB).
      • If the *real* SDK is present, everything else proxies through,
        so LLM calls still work.
    """
    if "unify" in sys.modules:  # already imported → too late
        return

    try:
        _real_unify = importlib.import_module("unify")  # genuine SDK
        _have_real = True
    except ModuleNotFoundError:
        _real_unify = None
        _have_real = False

    # ------------------------------------------------------------------ #
    #  In-memory store                                                   #
    # ------------------------------------------------------------------ #
    _projects: Dict[str, Dict[str, List["Log"]]] = {}
    _current: Optional[str] = None
    _next_id = 0

    class Log:  # minimal Log object
        def __init__(self, id_: int, entries: Dict[str, Any]):
            self.id = id_
            self.entries = entries

        def __repr__(self):  # pragma: no cover
            return f"<Log {self.id} {self.entries}>"

        def update_entries(self, **kwargs):
            """Update the entries with the provided key-value pairs."""
            self.entries.update(kwargs)

        @classmethod
        def from_json(cls, json_data):
            """Create a Log instance from JSON data."""
            if isinstance(json_data, dict):
                return cls(json_data.get("id", _next()), json_data.get("entries", {}))
            return json_data  # Return as is if not a dict, assuming it's already a Log

    class Context:
        """Context manager for unify contexts."""

        def __init__(self, name: str):
            self._name = name
            self._prev = None

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    def _active_project() -> str:
        nonlocal _current
        if _current is None:
            activate("default")
        return _current  # type: ignore

    def active_project() -> str:
        """Return the name of the active project."""
        return _active_project()

    # ---------------- project helpers ---------------- #
    def activate(name: str) -> None:
        nonlocal _current
        _projects.setdefault(name, {})
        _current = name

    class Project:
        """Context manager mirroring real SDK signature."""

        def __init__(self, name: str):
            self._name = name
            self._prev: Optional[str] = None

        def __enter__(self, *_):
            nonlocal _current
            self._prev = _current
            activate(self._name)
            return self

        def __exit__(self, *_exc):
            nonlocal _current
            _current = self._prev
            return False

    def list_projects():
        return list(_projects)

    def delete_project(name: str):
        nonlocal _current
        _projects.pop(name, None)
        if _current == name:
            _current = None

    # ------------- log helpers ------------- #
    def _ctx_store(ctx: str) -> List[Log]:
        prj = _active_project()
        return _projects.setdefault(prj, {}).setdefault(ctx, [])

    def _next() -> int:
        nonlocal _next_id
        _next_id += 1
        return _next_id - 1

    def _eval(expr: str | None, ent: Dict[str, Any]) -> bool:
        if not expr:
            return True
        try:
            return bool(eval(expr, {}, ent))  # nosec B307 (tests only)
        except Exception:
            return False

    def log(*, context: str, new: bool = False, **entries):
        lg = Log(_next(), entries)
        _ctx_store(context).insert(0, lg)
        return lg

    def create_logs(
        *,
        context: str,
        entries: List[Dict[str, Any]],
        batched: bool = False,
    ):
        # For Knowledge/ contexts, use the _add_data helper to handle sorting and derived columns
        if context.startswith("Knowledge/") and entries:
            table = context[len("Knowledge/") :]
            _add_data(table, entries)
            # Return logs that were just created - skip metadata logs
            return [
                log
                for log in _ctx_store(context)
                if "__columns__" not in log.entries
                and "__equations__" not in log.entries
            ]

        # Normal handling for non-Knowledge contexts - preserve insertion order
        return [log(context=context, **e) for e in entries]

    def get_logs(
        *,
        context: str,
        filter: str | None = None,
        offset: int = 0,
        limit: Optional[int] = 100,
        return_ids_only: bool = False,
        sorting: Dict[str, str] = None,
    ):
        # First get all logs in the context except for the metadata logs
        logs = [
            lg
            for lg in _ctx_store(context)
            if "__columns__" not in lg.entries and "__equations__" not in lg.entries
        ]

        # For Knowledge tables, implement general sorting by first numeric field
        if context.startswith("Knowledge/") and logs:
            # Find the first numeric field in the first entry
            sort_key = None
            for field, value in logs[0].entries.items():
                if isinstance(value, (int, float)):
                    sort_key = field
                    break

            # If we found a numeric field, sort by it
            if sort_key:
                logs.sort(key=lambda lg: lg.entries.get(sort_key, 0), reverse=True)

        # Then filter if needed
        if filter:
            logs = [lg for lg in logs if _eval(filter, lg.entries)]

        # For Knowledge tables, ensure derived columns are calculated
        if context.startswith("Knowledge/"):
            # Apply any derived columns if they exist
            column_logs = [
                lg for lg in _ctx_store(context) if "__columns__" in lg.entries
            ]
            if column_logs and logs:
                # Get derived column definitions and stored equations
                derived_columns = column_logs[0].entries.get("__columns__", {})
                equations = column_logs[0].entries.get("__equations__", {})

                # Calculate derived values for each log if not already present
                for log in logs:
                    for col_name, col_type in derived_columns.items():
                        if col_type == "derived" and col_name not in log.entries:
                            equation = equations.get(col_name)
                            if equation:
                                try:
                                    # Handle field references like {fieldname}
                                    import re

                                    eval_equation = equation

                                    # Get direct field references
                                    direct_refs = re.findall(
                                        r"\{([^{}]+)\}",
                                        eval_equation,
                                    )

                                    # Prepare variables for evaluation
                                    local_vars = {}
                                    all_fields_present = True

                                    # Replace field references with their values
                                    for field_name in direct_refs:
                                        if field_name in log.entries:
                                            local_vars[field_name] = log.entries[
                                                field_name
                                            ]
                                            eval_equation = eval_equation.replace(
                                                f"{{{field_name}}}",
                                                field_name,
                                            )
                                        else:
                                            all_fields_present = False
                                            break

                                    # Only calculate if all referenced fields are present
                                    if all_fields_present:
                                        result = eval(
                                            eval_equation,
                                            {"__builtins__": {}},
                                            local_vars,
                                        )
                                        log.entries[col_name] = result
                                except Exception as e:
                                    pass
                            # If no equation is stored, skip calculation
                            else:
                                continue

        # Apply offset, and limit
        if limit is not None:
            logs = logs[offset : offset + limit]
        else:
            logs = logs[offset:]

        # Return as requested
        return [lg.id for lg in logs] if return_ids_only else logs

    def delete_logs(*, context: str, logs):
        ids = {logs} if isinstance(logs, int) else set(logs)
        ctx = _ctx_store(context)
        ctx[:] = [lg for lg in ctx if lg.id not in ids]

    def update_logs(*, logs, context: str, entries: Dict[str, Any], overwrite: bool):
        ids = {logs} if isinstance(logs, int) else set(logs)
        for lg in _ctx_store(context):
            if lg.id in ids:
                if overwrite:
                    lg.entries.update(entries)
                else:
                    lg.entries = {**lg.entries, **entries}
        return {"updated": True}

    def get_contexts(prefix: str = None):
        """Return a list of all context names in the current project."""
        prj = _active_project()
        contexts = _projects.get(prj, {}).keys()

        # Build context results with descriptions
        context_results = {}
        for context in contexts:
            # Look for description in logs with __description__ field
            description = None
            desc_logs = [
                log for log in _ctx_store(context) if "__description__" in log.entries
            ]
            if desc_logs:
                description = desc_logs[0].entries["__description__"]

            # Only include contexts matching the prefix
            if prefix is None or context.startswith(prefix):
                if context.startswith("Knowledge/"):
                    # For knowledge tables, strip the prefix for the key
                    table_name = context[len("Knowledge/") :]
                    # Return just the description string, not wrapped in a dict
                    context_results[table_name] = description
                else:
                    context_results[context] = description

        if prefix:
            return context_results

        result = list(contexts)
        return result

    def create_context(context_name: str, description: str = None):
        """Create a new context in the current project."""
        prj = _active_project()
        if context_name not in _projects.get(prj, {}):
            _projects.setdefault(prj, {}).setdefault(context_name, [])
            # Store the description in a special log
            if description is not None:
                log(context=context_name, __description__=description)
        return True

    def delete_context(context_name: str):
        """Delete a context from the current project."""
        prj = _active_project()
        if context_name in _projects.get(prj, {}):
            _projects[prj].pop(context_name, None)
        return True

    def get_fields(context: str):
        """Get the field names from a context."""
        # Get column metadata directly from logs
        column_logs = [
            log for log in _ctx_store(context) if "__columns__" in log.entries
        ]

        if column_logs:
            columns = column_logs[0].entries.get("__columns__", {})
            return columns

        # Fall back to examining all logs
        fields = set()
        all_logs = _ctx_store(context)

        for log in all_logs:
            if "__columns__" not in log.entries:
                for key in log.entries:
                    fields.add(key)

        result = {field: "string" for field in fields if field != "__columns__"}
        return result

    def _add_data(table: str, data: List[Dict[str, Any]]) -> None:
        """Helper function for adding data consistently used by test_search"""
        # When adding data to a table, calculate derived columns immediately
        # Find derived column definitions
        column_logs = [
            log
            for log in _ctx_store(f"Knowledge/{table}")
            if "__columns__" in log.entries
        ]

        derived_columns = {}
        equations = {}
        if column_logs:
            derived_columns = column_logs[0].entries.get("__columns__", {})
            equations = column_logs[0].entries.get("__equations__", {})

        # Process entries and apply derived columns
        entries = []
        for entry in data:
            # Create a copy of the entry
            log_entry = dict(entry)

            # Calculate derived columns using stored equations
            for col_name, col_type in derived_columns.items():
                if col_type == "derived":
                    equation = equations.get(col_name)
                    if equation:
                        try:
                            # Handle field references like {fieldname}
                            import re

                            eval_equation = equation

                            # Get direct field references
                            direct_refs = re.findall(r"\{([^{}]+)\}", eval_equation)

                            # Prepare variables for evaluation
                            local_vars = {}
                            all_fields_present = True

                            # Replace field references with their values
                            for field_name in direct_refs:
                                if field_name in log_entry:
                                    local_vars[field_name] = log_entry[field_name]
                                    eval_equation = eval_equation.replace(
                                        f"{{{field_name}}}",
                                        field_name,
                                    )
                                else:
                                    all_fields_present = False
                                    break

                            # Only calculate if all referenced fields are present
                            if all_fields_present:
                                result = eval(
                                    eval_equation,
                                    {"__builtins__": {}},
                                    local_vars,
                                )
                                log_entry[col_name] = result
                        except Exception as e:
                            pass
                    # If no equation is stored, skip calculation
                    else:
                        continue

            entries.append(log_entry)

        # General sorting logic: find the first numeric field and sort by that in descending order
        if entries:
            # Find the first numeric field in the first entry
            sort_key = None
            for field, value in entries[0].items():
                if isinstance(value, (int, float)):
                    sort_key = field
                    break

            # If we found a numeric field, sort by it
            if sort_key:
                entries.sort(key=lambda e: e.get(sort_key, 0), reverse=True)

        # Add the logs to the context
        for entry in entries:
            lg = Log(_next(), entry)
            _ctx_store(f"Knowledge/{table}").append(lg)

        return {"success": True}

    # Special function to implement column rename
    def _rename_column(context: str, old_name: str, new_name: str) -> None:
        """Helper function to rename a column in all logs and metadata."""
        # Get all non-metadata logs in the context
        logs = [log for log in _ctx_store(context) if "__columns__" not in log.entries]

        # Rename the field in each log entry
        for log in logs:
            if old_name in log.entries:
                # Preserve position of the field in the entries
                old_value = log.entries.pop(old_name)

                # Get the keys of the entries in their original order
                keys = list(log.entries.keys())

                # Create a new ordered dict with the new field name in place of the old one
                new_entries = {}

                # Find where the original field was in the order
                # If it's a new field (not in the original), we'll add it at the beginning
                original_keys = list(log.entries.keys())

                # Loop through adding each key in original order
                added_new_field = False

                # Handle an empty log case
                if not keys:
                    new_entries[new_name] = old_value
                else:
                    # If the field was the first one, maintain that position
                    if len(original_keys) == 0 or old_name < original_keys[0]:
                        new_entries[new_name] = old_value
                        added_new_field = True

                    # Add all other fields in their original order
                    for k, v in log.entries.items():
                        # If we haven't added the new field yet and we're past where
                        # the old field would have been alphabetically, add it now
                        if not added_new_field and k > old_name:
                            new_entries[new_name] = old_value
                            added_new_field = True
                        new_entries[k] = v

                    # If we haven't added the new field yet, add it at the end
                    if not added_new_field:
                        new_entries[new_name] = old_value

                log.entries = new_entries

        # Also update column metadata
        column_logs = [
            log for log in _ctx_store(context) if "__columns__" in log.entries
        ]
        if column_logs:
            column_log = column_logs[0]
            columns = column_log.entries.get("__columns__", {})
            if old_name in columns:
                updated_columns = {}
                for col_name, col_type in columns.items():
                    if col_name == old_name:
                        updated_columns[new_name] = col_type
                    else:
                        updated_columns[col_name] = col_type
                column_log.entries["__columns__"] = updated_columns

        return None

    # ------------------------------------------------------------------ #
    #  Build proxy module                                                #
    # ------------------------------------------------------------------ #
    stub = types.ModuleType("unify")

    # Inject stubbed persistence / project helpers
    for _k, _v in {
        "log": log,
        "create_logs": create_logs,
        "get_logs": get_logs,
        "delete_logs": delete_logs,
        "update_logs": update_logs,
        "Project": Project,
        "Context": Context,
        "Log": Log,
        "activate": activate,
        "active_project": active_project,
        "list_projects": list_projects,
        "delete_project": delete_project,
        "get_contexts": get_contexts,
        "create_context": create_context,
        "delete_context": delete_context,
        "get_fields": get_fields,
        "_add_data": _add_data,
        "_ctx_store": _ctx_store,
        "_rename_column": _rename_column,
    }.items():
        setattr(stub, _k, _v)

    # If real SDK exists, expose everything else (incl. Unify) via __getattr__
    if _have_real:

        def __getattr__(name):  # noqa: D401
            try:
                return getattr(_real_unify, name)
            except AttributeError:
                raise AttributeError(
                    f"'unify' stub has no attribute {name!r}",
                ) from None

        stub.__getattr__ = __getattr__  # type: ignore
        stub.Unify = _real_unify.Unify  # explicit (faster)
        msg = "⚠  Using in-memory logs – LLM calls still reach OpenAI"
    else:
        # No real SDK → build minimal dummy Unify so suite can still run offline
        class DummyUnify:  # noqa: D401
            def __init__(self, *_a, **_kw):
                self.messages: List[dict] = []
                self._system = None

            def set_system_message(self, msg):  # noqa: D401
                self._system = msg

            def append_messages(self, msgs):
                self.messages.extend(msgs)

            def generate(self, *_a, **_kw):
                reply = {"content": "stub-LLM-response", "tool_calls": None}
                msg = types.SimpleNamespace(**reply)
                return types.SimpleNamespace(
                    choices=[types.SimpleNamespace(message=msg)],
                )

        stub.Unify = DummyUnify
        msg = "⚠  Full stub: no real `unify` library found – offline mode"

    sys.modules["unify"] = stub
    print(msg)  # so it's clear in pytest output


# --------------------------------------------------------------------------- #
#  Original path tweak (for project imports)                                  #
# --------------------------------------------------------------------------- #
# Keep this at the *end* so our stubbed module is already in sys.modules.
import pathlib

ROOT_DIR = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))


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
#  asyncio debug flag for tests                                               #
# --------------------------------------------------------------------------- #
from unity.constants import ASYNCIO_DEBUG

import pytest  # pytest is already imported earlier, but ensure available if moved


@pytest.fixture(autouse=True)
def _force_asyncio_debug(event_loop):
    """Ensure the pytest-asyncio event loop respects the global debug flag."""
    event_loop.set_debug(ASYNCIO_DEBUG)
