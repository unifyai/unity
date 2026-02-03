"""
UI process entrypoint for the multi-process ConversationManager sandbox.

This is spawned by the sandbox entrypoint with:
- `ui_to_worker`: IPC queue for UI → worker messages
- `worker_to_ui`: IPC queue for worker → UI messages
- `config`: dict of runtime parameters (actor_type, project_name, voice, etc.)

The UI process is intentionally thin: it renders and sends commands over IPC.
"""

from __future__ import annotations

from typing import Any


def main(ui_to_worker: Any, worker_to_ui: Any, config: dict):
    """UI process entrypoint."""

    # Textual relies on `sys.__stdin__` for terminal setup. In some spawned-process
    # environments, `sys.__stdin__` may be closed; repair it best-effort so the UI
    # can still start.
    try:  # best-effort; never crash the UI just for this
        import os as _os
        import sys as _sys

        def _ensure_stdin_open() -> None:
            try:
                s = getattr(_sys, "__stdin__", None)
                if s is not None and not getattr(s, "closed", False):
                    return
            except Exception:
                pass
            try:
                fh = open(
                    "/dev/tty",
                    "r",
                    encoding="utf-8",
                    errors="ignore",
                )  # noqa: P201
            except Exception:
                try:
                    fh = open(_os.devnull, "r", encoding="utf-8")  # noqa: P201
                except Exception:
                    return
            try:
                _sys.stdin = fh
                _sys.__stdin__ = fh  # type: ignore[attr-defined]
            except Exception:
                return

        _ensure_stdin_open()
        # If we still don't have a usable TTY, Textual can't start reliably.
        try:
            s = getattr(_sys, "__stdin__", None)
            if s is None or getattr(s, "closed", False) or (not s.isatty()):
                print(
                    "GUI could not acquire a usable TTY (stdin is closed or not a terminal). "
                    "Please run the sandbox from a terminal session.",
                )
                return
        except Exception:
            pass
    except Exception:
        pass

    from sandboxes.conversation_manager.gui import ModernizedMessagingApp

    app = ModernizedMessagingApp(
        ui_to_worker=ui_to_worker,
        worker_to_ui=worker_to_ui,
        config=config or {},
    )
    app.run()
