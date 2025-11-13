from __future__ import annotations

from typing import Optional

from ..constants import LOGGER
from ..common.async_tool_loop import AsyncToolLoopHandle, custom_steering_method


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

    @custom_steering_method()
    async def append_to_queue(self, task_id: int) -> None:
        """
        Request appending an existing runnable task to the live task queue.

        Behaviour
        ---------
        - Adds a concise user-visible interjection on the current tool loop.
        - The custom steering decorator mirrors this call (no extra LLM step) and
          forwards it to any adopted passthrough child that implements the same
          method signature (e.g., ActiveQueue.append_to_queue).
        """
        await self.interject(
            (
                f"outer append_to_queue({int(task_id)}) called, requesting for task "
                f"with task_id={int(task_id)} to be added to the live task queue, ready for execution."
            ),
            trigger_immediate_llm_turn=False,
        )
