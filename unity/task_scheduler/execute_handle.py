from __future__ import annotations

import asyncio
from typing import Optional

from ..constants import LOGGER
from ..common.async_tool_loop import AsyncToolLoopHandle


class ExecuteLoopHandle(AsyncToolLoopHandle):
    """
    Specialized outer handle used by TaskScheduler.execute.

    Extends the general async-loop handle with a stop(cancel=..., reason=...)
    signature so upstream callers (e.g., sandbox/router) can pass task-level
    intent. When a delegate has been adopted (e.g., ActiveQueue/ActiveTask),
    the cancel flag is forwarded; otherwise the flag is ignored and only the
    outer loop is cancelled as before.
    """

    def stop(self, *, cancel: bool | None = None, reason: Optional[str] = None) -> None:  # type: ignore[override]
        # Idempotent guard: if already stopping, do nothing and DO NOT log again
        if self._cancel_event.is_set():
            return

        # Flip the cancel event first so concurrent callers see we are stopping
        self._cancel_event.set()

        # Only the root/top-level handle logs the stop request
        if getattr(self, "_is_root_handle", False):
            _label = getattr(self, "_log_label", None) or self._loop_id
            try:
                LOGGER.info(
                    f"🛑 [{_label}] Stop requested"
                    + (f" – reason: {reason}" if reason else "")
                    + (f" – cancel={cancel}" if cancel is not None else ""),
                )
            except Exception:
                pass

        # Best-effort forwarding to a delegate, including cancel flag when supported
        if self._delegate is not None:
            try:
                # Preferred (keyword-only in most implementations)
                self._delegate.stop(cancel=bool(cancel), reason=reason)  # type: ignore[misc]
            except TypeError:
                # Legacy fallbacks
                try:
                    self._delegate.stop(reason=reason)  # type: ignore[misc]
                except TypeError:
                    try:
                        if reason is not None:
                            self._delegate.stop(reason)  # type: ignore[misc]
                        else:
                            self._delegate.stop()  # type: ignore[misc]
                    except Exception:
                        pass
            except Exception:
                # Defensive: failure to forward must not break outer shutdown
                pass

        # Expedite shutdown of the outer task and signal stop_event for any waiters
        try:
            self._task.cancel()
        except Exception:
            pass
        try:
            self._stop_event.set()
        except Exception:
            pass

    # ─────────────────────────────────────────────────────────────────────
    # Queue helper: accept early append() and replay once queue is adopted
    # ─────────────────────────────────────────────────────────────────────
    def append_to_queue(self, task_id: int) -> Optional[str]:
        """
        Append `task_id` to the tail of the in-flight ActiveQueue.

        Behaviour
        ---------
        - If an ActiveQueue child has already been adopted and is currently in
          passthrough mode (singleton), forward immediately and return its result.
        - Otherwise, buffer the request locally and replay it as soon as an
          ActiveQueue is adopted in passthrough mode.
        - If the execute loop completes without adopting an ActiveQueue (e.g. no
          matching task was found), the buffered requests are dropped (no-op).

        Returns
        -------
        str | None
            Human-readable summary from the queue when forwarded immediately;
            None when buffered for later replay.
        """
        # Lazy init to avoid modifying __init__
        if not hasattr(self, "_append_q_buffer"):
            self._append_q_buffer: list[int] = []
        if not hasattr(self, "_append_q_watcher"):
            self._append_q_watcher: asyncio.Task | None = None

        # Try immediate forward if an ActiveQueue is already adopted and in passthrough
        aq = self._find_active_queue_child()
        if aq is not None and self._is_passthrough_queue(aq):
            try:
                return aq.append_to_queue(int(task_id))
            except Exception:
                # If immediate forward fails, fall back to buffering
                pass

        # Buffer for later replay
        try:
            self._append_q_buffer.append(int(task_id))
        except Exception:
            return None

        # Start a watcher (once) to flush when an ActiveQueue is adopted in passthrough
        if self._append_q_watcher is None or self._append_q_watcher.done():
            try:
                self._append_q_watcher = asyncio.create_task(
                    self._watch_and_flush_appends(),
                )
            except Exception:
                # If scheduling fails, buffer remains inert (safe no-op)
                pass

        return None

    # ── internal helpers ─────────────────────────────────────────────────
    def _find_active_queue_child(self):
        """
        Return the first adopted ActiveQueue child handle, or None if not present.
        """
        try:
            # Local import to avoid cycles during module import
            from .active_queue import ActiveQueue  # type: ignore
        except Exception:
            ActiveQueue = None  # type: ignore

        try:
            task_info = getattr(self._task, "task_info", {}) or {}
        except Exception:
            task_info = {}

        for meta in list(task_info.values()):
            try:
                h = getattr(meta, "handle", None)
            except Exception:
                h = None
            if h is None:
                continue
            # Prefer isinstance; fall back to name check for robustness
            try:
                if ActiveQueue is not None and isinstance(h, ActiveQueue):  # type: ignore[arg-type]
                    return h
            except Exception:
                pass
            try:
                cls_name = getattr(getattr(h, "__class__", object), "__name__", "")
                if cls_name == "ActiveQueue":
                    return h
            except Exception:
                continue
        return None

    @staticmethod
    def _is_passthrough_queue(aq) -> bool:
        """
        True when the queue is currently in singleton passthrough mode.
        """
        try:
            return bool(getattr(aq, "_should_passthrough")() is True)
        except Exception:
            return False

    async def _watch_and_flush_appends(self) -> None:
        """
        Watch for an adopted ActiveQueue in passthrough mode and flush buffered appends.

        Drops the buffer silently if the execute loop finishes without adopting a queue.
        """
        poll_s = 0.05
        try:
            while True:
                # If loop finished without adopting a queue → drop buffer and exit
                if self.done():
                    try:
                        self._append_q_buffer.clear()
                    except Exception:
                        pass
                    return

                aq = self._find_active_queue_child()
                if aq is not None and self._is_passthrough_queue(aq):
                    pending: list[int] = []
                    try:
                        pending = list(getattr(self, "_append_q_buffer", []) or [])
                    except Exception:
                        pending = []

                    for tid in pending:
                        try:
                            aq.append_to_queue(int(tid))
                        except Exception:
                            # Best-effort: continue flushing remaining entries
                            pass
                    try:
                        self._append_q_buffer.clear()
                    except Exception:
                        pass
                    return

                await asyncio.sleep(poll_s)
        except asyncio.CancelledError:
            return
        except Exception:
            # Never let a watcher failure affect the outer handle
            return
