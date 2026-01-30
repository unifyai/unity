"""
Trace display for CodeActActor execution.

This module captures and renders CodeActActor "turn" execution:
- code string sent to the execution boundary
- ExecutionResult (stdout/stderr/result/error)

It is sandbox-only and intentionally decoupled from any UI. The REPL/GUI can call
`render_recent()` or install the execution wrapper at the appropriate boundary.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

from sandboxes.display.formatting import draw_box, join_blocks, truncate

LG = logging.getLogger("conversation_manager_sandbox")


@dataclass(frozen=True)
class TraceEntry:
    turn_index: int
    timestamp: float
    event_id: str
    code: str
    result: Any
    error: Optional[str] = None


class TraceDisplay:
    """In-memory bounded execution trace store + renderer."""

    def __init__(self, *, max_entries: int = 100) -> None:
        self._max_entries = int(max(1, max_entries))
        self._entries: list[TraceEntry] = []
        self._turn_counter: int = 0
        self._event_id: str = ""
        self._last_capture_error: str | None = None
        self._last_capture_error_at: float | None = None

    def set_event_context(self, *, event_id: str) -> None:
        """Set the current event context (resets turn counter for the event)."""
        self._event_id = str(event_id or "")
        self._turn_counter = 0

    def reset_history(self) -> None:
        self._entries.clear()
        self._turn_counter = 0
        self._event_id = ""
        self._last_capture_error = None
        self._last_capture_error_at = None

    def entry_count(self) -> int:
        return len(self._entries)

    def entry_count_for_current_event(self) -> int:
        """Return number of captured turns for the current event context."""
        eid = self._event_id
        if not eid:
            return 0
        return sum(1 for e in self._entries if e.event_id == eid)

    def last_capture_error(self) -> str | None:
        return self._last_capture_error

    def capture_execution(self, *, code: str, result: Any) -> TraceEntry:
        self._turn_counter += 1
        entry = TraceEntry(
            turn_index=self._turn_counter,
            timestamp=time.time(),
            event_id=self._event_id,
            code=str(code or ""),
            result=result,
            error=_get_error(result),
        )
        self._entries.append(entry)
        if len(self._entries) > self._max_entries:
            # FIFO eviction
            self._entries = self._entries[-self._max_entries :]
        return entry

    def render_recent(self, count: int = 3) -> str:
        n = int(max(1, count))
        recent = self._entries[-n:]
        return self._render_entries(recent)

    def render_current_event(self) -> str:
        """
        Render all captured turns for the current event context.

        The ConversationManager sandbox calls `set_event_context()` on ActorHandleStarted
        so the trace panel can show the complete CodeAct trajectory for the active event.
        """
        eid = self._event_id
        if not eid:
            return self.render_recent(3)
        ev_entries = [e for e in self._entries if e.event_id == eid]
        if not ev_entries:
            return self.render_recent(3)
        return self._render_entries(ev_entries)

    def render_all(self) -> str:
        """Render all captured turns currently in memory (across events)."""
        if not self._entries:
            return "(no trace entries yet)"
        return self._render_entries(list(self._entries))

    def _render_entries(self, entries: list[TraceEntry]) -> str:
        blocks: list[str] = []
        for e in entries:
            title = f"TRACE — Turn {e.turn_index}"
            if e.event_id:
                title += f" (event={truncate(e.event_id, 16)})"

            code_box = draw_box(
                (e.code or "").rstrip(),
                title="Code",
            )

            out_box = draw_box(
                _format_execution_result(e.result),
                title="Output" if not e.error else "Output (error)",
            )

            blocks.append(
                draw_box(join_blocks([code_box, out_box], separator="\n"), title=title),
            )
        return join_blocks(blocks, separator="\n\n" + ("═" * 60) + "\n\n")

    def install_executor_wrapper(
        self,
        *,
        execute_fn: Callable[..., Any],
        after_capture: Callable[[TraceEntry], Any] | None = None,
    ) -> Callable[..., Any]:
        """
        Wrap a code execution boundary (async callable) to capture traces.

        The wrapped callable must accept a `code=` kwarg (or positional `code`) and
        return an object/dict with an `error` field when execution failed.
        """

        async def _wrapped(*args: Any, **kwargs: Any) -> Any:
            # Extract code best-effort.
            code = ""
            if "code" in kwargs:
                code = kwargs.get("code") or ""
            elif args:
                # The canonical boundary uses keyword-only code, but tests may use positional.
                code = str(args[0] or "")

            res = await execute_fn(*args, **kwargs)
            try:
                entry = self.capture_execution(code=str(code or ""), result=res)
                if after_capture is not None:
                    after_capture(entry)
            except Exception as e:
                # Do not crash the sandbox if trace capture fails, but record an
                # explicit error so UIs can surface it instead of showing empty output.
                try:
                    self._last_capture_error = f"{type(e).__name__}: {e}"
                    self._last_capture_error_at = time.time()
                except Exception:
                    pass
                try:
                    LG.warning("trace capture failed: %s", self._last_capture_error)
                except Exception:
                    pass
            return res

        return _wrapped


def _get_error(result: Any) -> Optional[str]:
    try:
        if isinstance(result, dict):
            err = result.get("error")
            return str(err) if err else None
        err = getattr(result, "error", None)
        return str(err) if err else None
    except Exception:
        return None


def _format_execution_result(result: Any) -> str:
    """
    Best-effort rendering for CodeAct `ExecutionResult` or dict results.

    Images in rich stdout/stderr are represented as placeholders (`[image]`)
    since the sandbox output is text-first.
    """
    try:
        if result is None:
            return "(no output)"

        if isinstance(result, dict):
            stdout = result.get("stdout", "")
            stderr = result.get("stderr", "")
            err = result.get("error")
            lines: list[str] = []
            if stdout:
                lines.append(f"stdout:\n{_coerce_output(stdout)}")
            if stderr:
                lines.append(f"stderr:\n{_coerce_output(stderr)}")
            if err:
                lines.append(f"error:\n{err}")
            if not lines:
                return "(no output)"
            return "\n\n".join(lines).rstrip()

        # ExecutionResult-like
        stdout = getattr(result, "stdout", None)
        stderr = getattr(result, "stderr", None)
        err = getattr(result, "error", None)
        meta_bits: list[str] = []
        for k in (
            "language",
            "state_mode",
            "session_id",
            "duration_ms",
            "computer_used",
        ):
            v = getattr(result, k, None)
            if v is not None and v != "" and v is not False:
                meta_bits.append(f"{k}={v}")
        meta = ("meta: " + ", ".join(meta_bits)) if meta_bits else ""

        lines: list[str] = []
        if meta:
            lines.append(meta)
        if stdout:
            lines.append(f"stdout:\n{_coerce_output(stdout)}")
        if stderr:
            lines.append(f"stderr:\n{_coerce_output(stderr)}")
        if err:
            lines.append(f"error:\n{err}")
        if not lines:
            return "(no output)"
        return "\n\n".join(lines).rstrip()
    except Exception:
        return str(result)


def _coerce_output(stdout_or_parts: Any) -> str:
    # Rich: list[TextPart|ImagePart]
    if isinstance(stdout_or_parts, list):
        out_lines: list[str] = []
        for p in stdout_or_parts:
            if hasattr(p, "text"):
                out_lines.append(str(getattr(p, "text") or ""))
            else:
                # ImagePart or unknown object
                out_lines.append("[image]")
        return "".join(out_lines).rstrip()
    return str(stdout_or_parts).rstrip()
