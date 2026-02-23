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
from unity.events.types.manager_method import ManagerMethodPayload

LG = logging.getLogger("conversation_manager_sandbox")


@dataclass(frozen=True)
class TraceEntry:
    turn_index: int
    timestamp: float
    event_id: str
    code: str
    result: Any
    error: Optional[str] = None
    handle_id: Optional[int] = None  # Actor handle ID for concurrent tracking


class TraceDisplay:
    """In-memory bounded execution trace store + renderer."""

    def __init__(self, *, max_entries: int = 100) -> None:
        self._max_entries = int(max(1, max_entries))
        self._entries: list[TraceEntry] = []
        self._turn_counter: int = 0
        self._event_id: str = ""
        self._handle_id: int | None = None  # Current handle context
        self._last_capture_error: str | None = None
        self._last_capture_error_at: float | None = None
        # Optional mapping used by IPC-driven tracing:
        # call_id (ManagerMethod) → index into `_entries`
        self._call_id_to_index: dict[str, int] = {}
        # Per-handle turn counters for independent numbering
        self._handle_turn_counters: dict[int, int] = {}

    def set_event_context(self, *, event_id: str, handle_id: int | None = None) -> None:
        """Set the current event context (resets turn counter for the event).

        Args:
            event_id: Event identifier string
            handle_id: Optional Actor handle ID for concurrent tracking.
                       When provided, turn counters are tracked per-handle.
        """
        self._event_id = str(event_id or "")
        self._handle_id = handle_id
        # Only reset the global counter if no handle_id (legacy behavior)
        if handle_id is None:
            self._turn_counter = 0
        elif handle_id not in self._handle_turn_counters:
            self._handle_turn_counters[handle_id] = 0

    def reset_history(self) -> None:
        self._entries.clear()
        self._turn_counter = 0
        self._event_id = ""
        self._handle_id = None
        self._last_capture_error = None
        self._last_capture_error_at = None
        self._call_id_to_index.clear()
        self._handle_turn_counters.clear()

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

    def capture_execution(
        self,
        *,
        code: str,
        result: Any,
        handle_id: int | None = None,
    ) -> TraceEntry:
        # Use provided handle_id or fall back to current context
        effective_handle_id = handle_id if handle_id is not None else self._handle_id

        # Get the appropriate turn counter
        if effective_handle_id is not None:
            if effective_handle_id not in self._handle_turn_counters:
                self._handle_turn_counters[effective_handle_id] = 0
            self._handle_turn_counters[effective_handle_id] += 1
            turn = self._handle_turn_counters[effective_handle_id]
        else:
            self._turn_counter += 1
            turn = self._turn_counter

        entry = TraceEntry(
            turn_index=turn,
            timestamp=time.time(),
            event_id=self._event_id,
            code=str(code or ""),
            result=result,
            error=_get_error(result),
            handle_id=effective_handle_id,
        )
        self._entries.append(entry)
        if len(self._entries) > self._max_entries:
            # FIFO eviction
            self._entries = self._entries[-self._max_entries :]
            # Rebuild call_id mapping after eviction (small, bounded list).
            try:
                self._call_id_to_index = {
                    k: i
                    for i, (k, _idx) in enumerate(
                        sorted(
                            self._call_id_to_index.items(),
                            key=lambda kv: kv[1],
                        ),
                    )
                    if i < len(self._entries)
                }
            except Exception:
                self._call_id_to_index.clear()
        return entry

    def handle_codeact_execute_code(
        self,
        *,
        call_id: str,
        payload: ManagerMethodPayload,
        handle_id: int | None = None,
    ) -> None:
        """
        Ingest CodeActActor `execute_code` ManagerMethod events (IPC mode).

        In multi-process GUI mode, the worker streams EventBus `ManagerMethod` events
        across IPC. The UI process owns this TraceDisplay and uses these events to
        render a best-effort CodeAct trace without directly instrumenting the actor.

        We only handle:
        - manager == "CodeActActor"
        - method == "execute_code"
        - phase in {"incoming", "outgoing"}

        Args:
            call_id: Unique ID for this manager method call
            payload: The ManagerMethodPayload event
            handle_id: Optional Actor handle ID for concurrent tracking
        """

        try:
            if payload.manager != "CodeActActor" or payload.method != "execute_code":
                return
        except Exception:
            return

        phase = (payload.phase or "").strip().lower()
        cid = str(call_id or "")
        if not cid:
            return

        # Use provided handle_id or fall back to current context
        effective_handle_id = handle_id if handle_id is not None else self._handle_id

        # Prefer a stable "execution id" per hierarchy label.
        try:
            if payload.hierarchy_label:
                self._event_id = str(payload.hierarchy_label)
        except Exception:
            pass

        if phase == "incoming":
            code = (payload.instructions or payload.question or "").rstrip()

            # Get turn counter for this handle
            if effective_handle_id is not None:
                if effective_handle_id not in self._handle_turn_counters:
                    self._handle_turn_counters[effective_handle_id] = 0
                self._handle_turn_counters[effective_handle_id] += 1
                turn = self._handle_turn_counters[effective_handle_id]
            else:
                self._turn_counter += 1
                turn = self._turn_counter

            # Create a placeholder result; it will be updated on outgoing.
            entry = TraceEntry(
                turn_index=turn,
                timestamp=time.time(),
                event_id=self._event_id,
                code=code,
                result={"stdout": "", "stderr": "", "error": None},
                error=None,
                handle_id=effective_handle_id,
            )
            self._entries.append(entry)
            self._call_id_to_index[cid] = len(self._entries) - 1
            if len(self._entries) > self._max_entries:
                self._entries = self._entries[-self._max_entries :]
                # Conservative: mapping may be stale after eviction.
                self._call_id_to_index.clear()
            return

        if phase == "outgoing":
            idx = self._call_id_to_index.get(cid)
            if idx is None or idx < 0 or idx >= len(self._entries):
                # Missing incoming; create a minimal entry.
                self.capture_execution(
                    code=(payload.instructions or payload.question or "").rstrip(),
                    result={
                        "stdout": (payload.answer or "").rstrip(),
                        "stderr": "",
                        "error": payload.error,
                    },
                    handle_id=effective_handle_id,
                )
                return
            try:
                prior = self._entries[idx]
                res = {
                    "stdout": (payload.answer or "").rstrip(),
                    "stderr": "",
                    "error": payload.error,
                }
                self._entries[idx] = TraceEntry(
                    turn_index=prior.turn_index,
                    timestamp=prior.timestamp,
                    event_id=prior.event_id,
                    code=prior.code,
                    result=res,
                    error=str(payload.error) if payload.error else None,
                    handle_id=prior.handle_id,  # Preserve handle_id from incoming
                )
            except Exception:
                return

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

    def render_all(self, *, group_by_handle: bool = False) -> str:
        """Render all captured turns currently in memory (across events).

        Args:
            group_by_handle: If True, group entries by handle_id with visual separation.
        """
        if not self._entries:
            return "(no trace entries yet)"
        if group_by_handle:
            return self._render_entries_grouped(list(self._entries))
        return self._render_entries(list(self._entries))

    def render_for_handle(self, handle_id: int) -> str:
        """Render all captured turns for a specific Actor handle."""
        entries = [e for e in self._entries if e.handle_id == handle_id]
        if not entries:
            return f"(no trace entries for handle {handle_id})"
        return self._render_entries(entries, show_handle=False)

    def get_active_handles(self) -> list[int]:
        """Return list of handle_ids that have trace entries."""
        handles = set()
        for e in self._entries:
            if e.handle_id is not None:
                handles.add(e.handle_id)
        return sorted(handles)

    def get_entries_by_handle(self) -> dict[int | None, list[TraceEntry]]:
        """Return entries grouped by handle_id for concurrent visualization."""
        result: dict[int | None, list[TraceEntry]] = {}
        for e in self._entries:
            hid = e.handle_id
            if hid not in result:
                result[hid] = []
            result[hid].append(e)
        return result

    def _render_entries(
        self,
        entries: list[TraceEntry],
        *,
        show_handle: bool = True,
    ) -> str:
        blocks: list[str] = []
        for e in entries:
            title = f"TRACE — Turn {e.turn_index}"
            if show_handle and e.handle_id is not None:
                title += f" [H{e.handle_id}]"
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

    def _render_entries_grouped(self, entries: list[TraceEntry]) -> str:
        """Render entries grouped by handle_id with visual separation."""
        by_handle = self.get_entries_by_handle()
        if not by_handle:
            return "(no trace entries yet)"

        sections: list[str] = []
        for hid in sorted(by_handle.keys(), key=lambda x: (x is None, x)):
            handle_entries = by_handle[hid]
            if hid is None:
                header = "━━━ Traces (no handle) ━━━"
            else:
                header = f"━━━ Handle {hid} ({len(handle_entries)} turn(s)) ━━━"
            section_content = self._render_entries(handle_entries, show_handle=False)
            sections.append(f"{header}\n\n{section_content}")

        return "\n\n".join(sections)

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
