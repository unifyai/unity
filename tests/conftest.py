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

    def _active_project() -> str:
        nonlocal _current
        if _current is None:
            activate("default")
        return _current  # type: ignore

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
        _ctx_store(context).append(lg)
        return lg

    def create_logs(*, context: str, entries: List[Dict[str, Any]]):
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
        "activate": activate,
        "list_projects": list_projects,
        "delete_project": delete_project,
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
