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

        self._debug_handler: logging.Handler | None = None

    def reset(self) -> None:
        for k in self._buf:
            self._buf[k].clear()
        self._expanded.clear()
        self._execution_id = None

    def reset_expansion(self) -> None:
        self._expanded.clear()

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
    ) -> None:
        self._append(
            LogEntry(
                timestamp=time.time(),
                category=category,
                level=str(level),
                message=str(message),
                subcategory=subcategory,
                event_id=event_id,
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

    def render_expanded(self, category: Category) -> str:
        buf = self._buf[category]
        if not buf:
            return "(no logs)"
        lines: list[str] = []
        for e in buf:
            prefix = f"[{category}]"
            if e.subcategory:
                prefix += f"[{e.subcategory}]"
            lines.append(f"{prefix} {e.level}: {e.message}")
        return "\n".join(lines)

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
