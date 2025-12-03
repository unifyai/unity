"""Progress reporting system for FileManager pipeline.

This module provides a pluggable progress reporting system that replaces
the complex Rich-based progress_display.py. It decouples progress tracking
from UI rendering, treating progress as structured data.

Key components:
- ProgressEvent: Structured event data for each progress update
- ProgressReporter: Protocol for progress reporting implementations
- Implementations: JsonFileReporter, CallbackReporter, ConsoleReporter, NoOpReporter

The system is designed to be non-blocking and thread-safe for use in
concurrent task execution scenarios.
"""

from __future__ import annotations

import json
import logging
import sys
import threading
import time
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Protocol,
    TextIO,
    TypedDict,
    runtime_checkable,
)

logger = logging.getLogger(__name__)


class ProgressEvent(TypedDict, total=False):
    """Structured event for progress reporting.

    This TypedDict defines the shape of all progress events in the pipeline.
    Each event captures the state of a specific operation at a point in time.

    Attributes
    ----------
    timestamp : str
        ISO-formatted timestamp of the event.
    file_path : str
        The file being processed.
    phase : str
        The pipeline phase: "file_record", "ingest_content", "embed_content",
        "ingest_table", "embed_table", "file_complete".
    status : str
        The status: "started", "progress", "completed", "failed", "retry", "cancelled".
    duration_ms : float
        Time in milliseconds for THIS specific operation. Always present for
        completed/failed events. For "started" events, this is 0.0.
    elapsed_ms : float
        Cumulative time in milliseconds since file processing started. Provides
        a running total to understand overall progress timeline.
    error : str | None
        Error message if status is "failed" or "retry".
    traceback : str | None
        Full traceback string for error events (included at medium/high verbosity).
    meta : dict | None
        Phase-specific metadata. Contents vary by phase and verbosity level:
        - low: minimal (timestamp, file_path, phase, status, duration_ms, elapsed_ms)
        - medium: + chunk, total_chunks, table_label, row_count
        - high: + batch_size, inserted_ids_count, strategy, retries
    """

    timestamp: str
    file_path: str
    phase: str
    status: str
    duration_ms: float
    elapsed_ms: float
    error: Optional[str]
    traceback: Optional[str]
    meta: Optional[Dict[str, Any]]


@runtime_checkable
class ProgressReporter(Protocol):
    """Protocol for progress reporting implementations.

    Implementations should be thread-safe as they may be called from
    multiple concurrent tasks.
    """

    def report(self, event: ProgressEvent) -> None:
        """Report a progress event.

        Parameters
        ----------
        event : ProgressEvent
            The event to report.
        """
        ...

    def flush(self) -> None:
        """Flush any buffered output.

        Called at the end of pipeline execution to ensure all events
        are persisted.
        """
        ...


class NoOpReporter:
    """Silent reporter that discards all events.

    Use this when no progress output is desired.
    """

    def report(self, event: ProgressEvent) -> None:
        """Discard the event."""

    def flush(self) -> None:
        """No-op flush."""


class ConsoleReporter:
    """Simple console reporter that prints events to stdout/stderr.

    Formats events as human-readable single-line messages.
    Thread-safe via a lock.

    Parameters
    ----------
    stream : TextIO
        Output stream (default: sys.stdout).
    show_timestamps : bool
        Whether to include timestamps in output.
    emoji : bool
        Whether to use emoji in output.
    """

    # Status emoji mapping
    _STATUS_EMOJI = {
        "started": "🔄",
        "progress": "⏳",
        "completed": "✅",
        "failed": "❌",
        "retry": "🔁",
        "cancelled": "⏹️",
    }

    # Phase descriptions
    _PHASE_NAMES = {
        "parse": "Parse",
        "file_record": "File Record",
        "ingest_content": "Ingest Content",
        "embed_content": "Embed Content",
        "ingest_table": "Ingest Table",
        "embed_table": "Embed Table",
        "file_complete": "File Complete",
    }

    def __init__(
        self,
        stream: Optional[TextIO] = None,
        show_timestamps: bool = False,
        emoji: bool = True,
    ) -> None:
        self._stream = stream or sys.stdout
        self._show_timestamps = show_timestamps
        self._emoji = emoji
        self._lock = threading.Lock()

    def report(self, event: ProgressEvent) -> None:
        """Print the event to the console."""
        with self._lock:
            try:
                msg = self._format_event(event)
                print(msg, file=self._stream)
            except Exception as e:
                logger.debug(f"Console report failed: {e}")

    def flush(self) -> None:
        """Flush the output stream."""
        try:
            self._stream.flush()
        except Exception:
            pass

    def _format_event(self, event: ProgressEvent) -> str:
        """Format an event as a human-readable string."""
        parts = []

        # Timestamp (optional)
        if self._show_timestamps:
            parts.append(f"[{event['timestamp']}]")

        # Status emoji or text
        status = event["status"]
        if self._emoji:
            parts.append(self._STATUS_EMOJI.get(status, "❓"))
        else:
            parts.append(f"[{status.upper()}]")

        # Phase
        phase = event["phase"]
        phase_name = self._PHASE_NAMES.get(phase, phase)
        parts.append(phase_name)

        # File path (shortened)
        file_path = event["file_path"]
        if len(file_path) > 40:
            file_path = "..." + file_path[-37:]
        parts.append(f"'{file_path}'")

        # Extract meta for chunk/table info
        meta = event.get("meta") or {}

        # Table label (if present in meta)
        if meta.get("table_label"):
            parts.append(f"[{meta['table_label']}]")

        # Chunk progress (if present in meta)
        if meta.get("chunk") is not None:
            chunk = meta["chunk"]
            total = meta.get("total_chunks")
            if total:
                parts.append(f"({chunk}/{total})")
            else:
                parts.append(f"(chunk {chunk})")

        # Timing information (if present and non-zero)
        duration_ms = event.get("duration_ms", 0.0)
        elapsed_ms = event.get("elapsed_ms", 0.0)
        if duration_ms > 0 or elapsed_ms > 0:
            timing_parts = []
            if duration_ms > 0:
                timing_parts.append(f"{duration_ms:.0f}ms")
            if elapsed_ms > 0:
                timing_parts.append(f"@{elapsed_ms:.0f}ms")
            if timing_parts:
                parts.append(f"[{' '.join(timing_parts)}]")

        # Error (if present)
        if event.get("error"):
            parts.append(f"- {event['error']}")

        return " ".join(parts)


class JsonFileReporter:
    """Reporter that writes JSON-lines to a file.

    Each event is written as a single JSON line, making the file
    easy to parse and stream. Thread-safe via a lock.

    By default, the file is overwritten at the start of each pipeline run
    to ensure clean logs for each job.

    Parameters
    ----------
    file_path : str | Path
        Path to the output file. Created if it doesn't exist.
    append : bool
        If True, append to existing file. If False (default), overwrite at start.
    """

    def __init__(
        self,
        file_path: str | Path,
        append: bool = False,
    ) -> None:
        self._file_path = Path(file_path)
        self._append = append
        self._lock = threading.Lock()
        self._file: Optional[TextIO] = None
        self._opened = False

    def _ensure_open(self) -> TextIO:
        """Ensure the file is open for writing."""
        if self._file is None or self._file.closed:
            mode = "a" if self._append else "w"
            # Ensure parent directory exists
            self._file_path.parent.mkdir(parents=True, exist_ok=True)
            self._file = open(self._file_path, mode, encoding="utf-8")
            self._opened = True
        return self._file

    def report(self, event: ProgressEvent) -> None:
        """Write the event as a JSON line with immediate flush."""
        with self._lock:
            try:
                f = self._ensure_open()
                line = json.dumps(dict(event), ensure_ascii=False)
                f.write(line + "\n")
                f.flush()  # Ensure event is written immediately
            except Exception as e:
                logger.warning(f"JSON file report failed: {e}")

    def flush(self) -> None:
        """Flush and close the file."""
        with self._lock:
            if self._file is not None and not self._file.closed:
                try:
                    self._file.flush()
                    self._file.close()
                except Exception:
                    pass
            self._file = None

    def __del__(self) -> None:
        """Ensure file is closed on cleanup."""
        try:
            self.flush()
        except Exception:
            pass


class CallbackReporter:
    """Reporter that invokes a user-provided callback for each event.

    The callback receives the event and can perform any custom handling.
    Thread-safe via a lock.

    Parameters
    ----------
    callback : Callable[[ProgressEvent], None]
        The callback function to invoke for each event.
    """

    def __init__(self, callback: Callable[[ProgressEvent], None]) -> None:
        self._callback = callback
        self._lock = threading.Lock()

    def report(self, event: ProgressEvent) -> None:
        """Invoke the callback with the event."""
        with self._lock:
            try:
                self._callback(event)
            except Exception as e:
                logger.warning(f"Callback report failed: {e}")

    def flush(self) -> None:
        """No-op flush."""


class CompositeReporter:
    """Reporter that delegates to multiple child reporters.

    Useful for sending events to multiple destinations simultaneously,
    such as both console and file.

    Parameters
    ----------
    reporters : list[ProgressReporter]
        Child reporters to delegate to.
    """

    def __init__(self, reporters: List[ProgressReporter]) -> None:
        self._reporters = list(reporters)

    def report(self, event: ProgressEvent) -> None:
        """Delegate to all child reporters."""
        for reporter in self._reporters:
            try:
                reporter.report(event)
            except Exception as e:
                logger.warning(f"Child reporter failed: {e}")

    def flush(self) -> None:
        """Flush all child reporters."""
        for reporter in self._reporters:
            try:
                reporter.flush()
            except Exception as e:
                logger.warning(f"Child reporter flush failed: {e}")


def generate_progress_file_path() -> str:
    """Generate an auto-generated progress file path with timestamp.

    Returns
    -------
    str
        Path in format: ./pipeline_progress_{timestamp}.jsonl
    """
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    return f"./pipeline_progress_{timestamp}.jsonl"


def create_reporter(
    mode: Literal["json_file", "callback", "off"] = "json_file",
    *,
    file_path: Optional[str] = None,
    callback: Optional[Callable[[ProgressEvent], None]] = None,
    verbosity: Literal["low", "medium", "high"] = "low",
    append: bool = False,
) -> ProgressReporter:
    """Factory function to create a progress reporter.

    Parameters
    ----------
    mode : Literal["json_file", "callback", "off"]
        The type of reporter to create.
    file_path : str | None
        For json_file mode, the output file path.
        If not provided, auto-generates: ./pipeline_progress_{timestamp}.jsonl
    callback : Callable | None
        For callback mode, the callback function.
    verbosity : Literal["low", "medium", "high"]
        Controls detail level of events (reserved for future filtering).
    append : bool
        For json_file mode: if True, append to existing file; if False (default),
        overwrite to start fresh for each pipeline run.

    Returns
    -------
    ProgressReporter
        The configured reporter instance.

    Raises
    ------
    ValueError
        If required parameters are missing for the selected mode.
    """
    if mode == "off":
        return NoOpReporter()

    if mode == "json_file":
        # Auto-generate file path if not provided
        actual_path = file_path or generate_progress_file_path()
        logger.info(f"📝 Progress events will be logged to: {actual_path}")
        return JsonFileReporter(actual_path, append=append)

    if mode == "callback":
        if callback is None:
            raise ValueError("callback is required for callback mode")
        return CallbackReporter(callback)

    raise ValueError(f"Unknown progress mode: {mode}")


def create_progress_event(
    file_path: str,
    phase: str,
    status: str,
    *,
    duration_ms: float = 0.0,
    elapsed_ms: float = 0.0,
    error: Optional[str] = None,
    traceback_str: Optional[str] = None,
    meta: Optional[Dict[str, Any]] = None,
    verbosity: Literal["low", "medium", "high"] = "low",
) -> ProgressEvent:
    """Helper to create a progress event with current timestamp and timing.

    Parameters
    ----------
    file_path : str
        The file being processed.
    phase : str
        The pipeline phase: "file_record", "ingest_content", "embed_content",
        "ingest_table", "embed_table", "file_complete".
    status : str
        The status: "started", "progress", "completed", "failed", "retry", "cancelled".
    duration_ms : float
        Time in milliseconds for THIS specific operation. Default 0.0 for "started" events.
    elapsed_ms : float
        Cumulative time in milliseconds since file processing started.
    error : str | None
        Error message if applicable.
    traceback_str : str | None
        Full traceback string for error events.
    meta : dict | None
        Phase-specific metadata. The verbosity parameter controls what gets included:
        - low: meta is omitted entirely (timing fields are always included)
        - medium: meta includes chunk, total_chunks, table_label, row_count
        - high: meta includes all available fields
    verbosity : Literal["low", "medium", "high"]
        Controls the detail level of the event. Timing (duration_ms, elapsed_ms)
        is ALWAYS included regardless of verbosity. Meta contents are filtered
        based on verbosity level.

    Returns
    -------
    ProgressEvent
        The constructed event with timing information.

    Examples
    --------
    >>> # Low verbosity - timing only, no meta
    >>> create_progress_event("file.xlsx", "ingest_content", "completed",
    ...                       duration_ms=1500.0, elapsed_ms=3200.0)

    >>> # Medium verbosity - timing + basic meta
    >>> create_progress_event("file.xlsx", "ingest_content", "completed",
    ...                       duration_ms=1500.0, elapsed_ms=3200.0,
    ...                       meta={"chunk": 1, "total_chunks": 5, "row_count": 1000},
    ...                       verbosity="medium")

    >>> # High verbosity - all details
    >>> create_progress_event("file.xlsx", "ingest_content", "completed",
    ...                       duration_ms=1500.0, elapsed_ms=3200.0,
    ...                       meta={"chunk": 1, "total_chunks": 5, "row_count": 1000,
    ...                             "batch_size": 1000, "inserted_ids_count": 1000},
    ...                       verbosity="high")
    """
    event: ProgressEvent = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "file_path": file_path,
        "phase": phase,
        "status": status,
        "duration_ms": duration_ms,
        "elapsed_ms": elapsed_ms,
    }

    # Add error fields if present
    if error is not None:
        event["error"] = error
    if traceback_str is not None:
        event["traceback"] = traceback_str

    # Add meta based on verbosity level
    if meta is not None and verbosity != "low":
        if verbosity == "medium":
            # Medium: include basic chunk/progress info
            filtered_meta = {
                k: v
                for k, v in meta.items()
                if k
                in ("chunk", "total_chunks", "table_label", "row_count", "strategy")
            }
            if filtered_meta:
                event["meta"] = filtered_meta
        else:  # high
            # High: include everything
            event["meta"] = meta

    return event
