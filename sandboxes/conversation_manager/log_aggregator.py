"""
Log aggregation for the ConversationManager sandbox.

This component is designed to consume:
- structured broker events (default mode)
- optional Python logging records (only when --debug is enabled)

It maintains bounded buffers and provides "summary" vs "expanded" rendering.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Callable, Literal, Optional

from sandboxes.display.formatting import colorize

Category = Literal["cm", "actor", "manager"]


@dataclass(frozen=True)
class LogEntry:
    timestamp: float
    category: Category
    level: str
    message: str
    subcategory: str | None = None
    event_id: str | None = None
    handle_id: int | None = None  # Actor handle ID for concurrent tracking


class LogAggregator:
    """Bounded log buffers + expansion state."""

    def __init__(self, *, max_entries_per_category: int = 1000) -> None:
        self._max = int(max(100, max_entries_per_category))
        self._buf: dict[Category, list[LogEntry]] = {
            "cm": [],
            "actor": [],
            "manager": [],
        }
        self._expanded: set[Category] = set()
        self._execution_id: str | None = None
        self._current_handle_id: int | None = None  # Current handle context

        self._debug_handler: logging.Handler | None = None

    def reset(self) -> None:
        for k in self._buf:
            self._buf[k].clear()
        self._expanded.clear()
        self._execution_id = None
        self._current_handle_id = None

    def reset_expansion(self) -> None:
        self._expanded.clear()

    def set_handle_context(self, *, handle_id: int | None) -> None:
        """Set the current Actor handle context for tagging subsequent events."""
        self._current_handle_id = handle_id

    def counts(self) -> dict[Category, int]:
        return {k: len(v) for k, v in self._buf.items()}

    def start_execution(self, *, execution_id: str | None = None) -> None:
        """Mark an execution boundary and clear per-execution buffers."""
        self._execution_id = execution_id
        for k in self._buf:
            self._buf[k].clear()

    def expand(self, category: Category) -> None:
        self._expanded.add(category)

    def collapse(self, category: Category) -> None:
        self._expanded.discard(category)

    def handle_structured_event(
        self,
        *,
        category: Category,
        message: str,
        level: str = "INFO",
        subcategory: str | None = None,
        event_id: str | None = None,
        handle_id: int | None = None,
    ) -> None:
        # Use provided handle_id or fall back to current context
        effective_handle_id = (
            handle_id if handle_id is not None else self._current_handle_id
        )
        self._append(
            LogEntry(
                timestamp=time.time(),
                category=category,
                level=str(level),
                message=str(message),
                subcategory=subcategory,
                event_id=event_id,
                handle_id=effective_handle_id,
            ),
        )

    def enable_python_logging_capture(
        self,
        *,
        category_mapper: Optional[
            Callable[[logging.LogRecord], Category | None]
        ] = None,
    ) -> None:
        """
        Attach a logging.Handler to capture Python logs into buffers.

        This is intended for `--debug` mode only.
        """
        if self._debug_handler is not None:
            return

        mapper = category_mapper or (lambda _r: None)

        class _Handler(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:  # type: ignore[override]
                try:
                    cat = mapper(record)
                    if cat is None:
                        return
                    msg = record.getMessage()
                    self_._append(
                        LogEntry(
                            timestamp=time.time(),
                            category=cat,
                            level=record.levelname,
                            message=msg,
                            subcategory=record.name,
                            event_id=None,
                        ),
                    )
                except Exception:
                    return

        self_ = self
        self._debug_handler = _Handler(level=logging.DEBUG)
        logging.getLogger().addHandler(self._debug_handler)

    def disable_python_logging_capture(self) -> None:
        if self._debug_handler is None:
            return
        try:
            logging.getLogger().removeHandler(self._debug_handler)
        except Exception:
            pass
        self._debug_handler = None

    def render_summary(self) -> str:
        """One-line summaries per category."""
        lines: list[str] = []
        for cat in ("cm", "actor", "manager"):
            buf = self._buf[cat]  # type: ignore[index]
            n = len(buf)
            color = _color_for(cat)  # type: ignore[arg-type]
            label = cat.upper() if cat != "cm" else "CM"
            lines.append(colorize(f"[{label}] {n} log line(s)", color))
        return "\n".join(lines)

    def render_expanded(
        self,
        category: Category,
        *,
        group_by_handle: bool = False,
        max_message_length: int = 160,
    ) -> str:
        """Render logs for a category.

        Args:
            category: The log category to render
            group_by_handle: If True, group logs by handle_id with visual separation
            max_message_length: Maximum message length for display (0 = no limit)
        """
        buf = self._buf[category]
        if not buf:
            return "(no logs)"
        if group_by_handle:
            return self._render_grouped(
                buf,
                category,
                max_message_length=max_message_length,
            )
        return self._render_flat(
            buf,
            category,
            show_handle=True,
            max_message_length=max_message_length,
        )

    def _render_flat(
        self,
        buf: list[LogEntry],
        category: Category,
        *,
        show_handle: bool = False,
        max_message_length: int = 160,
    ) -> str:
        """Render logs in flat chronological order."""
        lines: list[str] = []
        for e in buf:
            prefix = f"[{category}]"
            if show_handle and e.handle_id is not None:
                prefix += f"[H{e.handle_id}]"
            if e.subcategory:
                prefix += f"[{e.subcategory}]"
            # Truncate message for display if needed
            msg = e.message
            if max_message_length > 0 and len(msg) > max_message_length:
                msg = msg[:max_message_length] + "..."
            lines.append(f"{prefix} {e.level}: {msg}")
        return "\n".join(lines)

    def _render_grouped(
        self,
        buf: list[LogEntry],
        category: Category,
        *,
        max_message_length: int = 160,
    ) -> str:
        """Render logs grouped by handle_id with visual separation."""
        by_handle: dict[int | None, list[LogEntry]] = {}
        for e in buf:
            hid = e.handle_id
            if hid not in by_handle:
                by_handle[hid] = []
            by_handle[hid].append(e)

        sections: list[str] = []
        for hid in sorted(by_handle.keys(), key=lambda x: (x is None, x)):
            entries = by_handle[hid]
            if hid is None:
                header = f"─── {category.upper()} (no handle) ───"
            else:
                header = (
                    f"─── {category.upper()} Handle {hid} ({len(entries)} entries) ───"
                )
            content = self._render_flat(
                entries,
                category,
                show_handle=False,
                max_message_length=max_message_length,
            )
            sections.append(f"{header}\n{content}")

        return "\n\n".join(sections)

    def get_active_handles(self, category: Category) -> list[int]:
        """Return list of handle_ids that have log entries in this category."""
        handles = set()
        for e in self._buf[category]:
            if e.handle_id is not None:
                handles.add(e.handle_id)
        return sorted(handles)

    # ──────────────────────────────────────────────────────────────────
    # Internals
    # ──────────────────────────────────────────────────────────────────

    def _append(self, entry: LogEntry) -> None:
        buf = self._buf[entry.category]
        buf.append(entry)
        if len(buf) > self._max:
            self._buf[entry.category] = buf[-self._max :]


def _color_for(category: Category) -> str:
    return {"cm": "cyan", "actor": "yellow", "manager": "magenta"}.get(category, "dim")
