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

import os
import sys
import types
import importlib
from typing import Any, Dict, List, Optional


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


# --------------------------------------------------------------------------- #
#  Session-wide hook – install stub *before* any project imports              #
# --------------------------------------------------------------------------- #


def pytest_sessionstart(session):
    use_stub = session.config.getoption("--unify-stub") or os.getenv("USE_UNIFY_STUB")
    if use_stub:
        _install_unify_stub()
        _install_requests_mock()


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
        def request(method, url, json=None, headers=None, **kwargs):
            # Get or extract table name from URL
            import re

            table_match = re.search(r"Knowledge/([^/]+)", url)
            table_name = table_match.group(1) if table_match else None

            # Handle different API endpoints
            if "/columns" in url:
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
            elif "/rename" in url:
                # Renaming tables
                return MockResponse(
                    {"success": True, "message": "Table renamed successfully"},
                )
            elif "/logs/fields" in url or "/logs/fields?" in url:
                # Getting columns for a table
                import unify

                # Extract context from URL query params
                import urllib.parse

                query = url.split("?")[-1] if "?" in url else ""
                params = dict(urllib.parse.parse_qsl(query))
                context = params.get("context")
                project = params.get("project")

                # Handle formats like /v0/logs/fields?project=XX&context=Knowledge/MyTable
                if context:
                    # Already have the context
                    pass
                elif table_name:
                    # Extract from path component
                    context = f"Knowledge/{table_name}"
                else:
                    # Extract the context path from the URL if not in params
                    ctx_match = re.search(r"contexts/([^/]+)", url)
                    if ctx_match:
                        context = ctx_match.group(1)

                if context:
                    # Look for column metadata in the logs
                    column_logs = [
                        log
                        for log in unify.get_logs(context=context)
                        if "__columns__" in log.entries
                    ]

                    if column_logs:
                        # Format column data to match API expected by _get_columns
                        # This is the format returned by the real API and expected by
                        # knowledge_manager._get_columns method
                        column_data = column_logs[0].entries.get("__columns__", {})
                        formatted_columns = {}
                        for name, type_val in column_data.items():
                            formatted_columns[name] = {"data_type": type_val}
                        return MockResponse(formatted_columns)

                # Default empty response
                return MockResponse({})
            elif "/logs/derived" in url:
                # Creating derived columns
                if json:
                    context = json.get("context")
                    column_name = json.get("key")
                    if context and column_name:
                        # Store in columns metadata using unify directly
                        import unify

                        column_logs = [
                            log
                            for log in unify.get_logs(context=context)
                            if "__columns__" in log.entries
                        ]

                        if column_logs:
                            # Update existing column metadata
                            column_log = column_logs[0]
                            column_log.update_entries(
                                __columns__={
                                    **column_log.entries.get("__columns__", {}),
                                    **{column_name: "derived"},
                                },
                            )
                        else:
                            # Create new column metadata log
                            unify.log(
                                context=context,
                                __columns__={column_name: "derived"},
                            )

                return MockResponse(
                    {"success": True, "message": "Derived column created"},
                )
            elif "/logs/rename_field" in url:
                # Renaming columns
                return MockResponse({"success": True, "message": "Column renamed"})
            elif "/logs?delete_empty_logs=True" in url:
                # Deleting columns
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
        return [log(context=context, **e) for e in entries]

    def get_logs(
        *,
        context: str,
        filter: str | None = None,
        offset: int = 0,
        limit: int = 100,
        return_ids_only: bool = False,
    ):
        logs = [lg for lg in _ctx_store(context) if _eval(filter, lg.entries)]
        logs = logs[offset : offset + limit]
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
        if prefix:
            # This is what the real API returns for Knowledge/ prefix
            # and what knowledge_manager._list_tables expects
            # Format: {"Knowledge/MyTable": None, ...}
            return {k: None for k in contexts if k.startswith(prefix)}
        return list(contexts)

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
        fields = set()

        # Check for column metadata first
        for log in _ctx_store(context):
            if "__columns__" in log.entries:
                # Convert from {"column": "type"} to proper format
                return {
                    col: col_type
                    for col, col_type in log.entries["__columns__"].items()
                }

        # Fall back to examining all logs
        for log in _ctx_store(context):
            fields.update(log.entries.keys())

        return {field: "string" for field in fields if field != "__columns__"}

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
