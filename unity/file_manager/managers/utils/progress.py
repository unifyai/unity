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
import re
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
        Error message if status is "failed" or "retry". For large errors (e.g.,
        HTTP errors with full payloads), this is truncated to the essential detail
        and the full error is dumped to a separate file.
    error_detail_file : str | None
        Path to file containing full error details when error was truncated.
        Only present when the original error exceeded max_error_length.
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
    error_detail_file: Optional[str]
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

    For events with large error messages (e.g., HTTP errors containing full
    request payloads), the error is truncated to the essential detail and the
    full error is dumped to a separate file. The path to the full error file
    is included in the event as `error_detail_file`.

    Error files are organized into per-run subfolders based on the timestamp
    when the reporter was created, e.g.:
        error_details/
        ├── 2025-01-29T10-30-45/
        │   ├── ingest_table_0001.json
        │   └── ingest_table_0002.json
        └── 2025-01-29T11-45-00/
            └── ingest_table_0001.json

    Parameters
    ----------
    file_path : str | Path
        Path to the output file. Created if it doesn't exist.
    append : bool
        If True, append to existing file. If False (default), overwrite at start.
    max_error_length : int
        Maximum length of error messages before truncation (default: 500).
        Errors longer than this are truncated and full details dumped to file.
    error_dir : str | Path | None
        Base directory for error detail files. If None (default), creates an
        `error_details` subdirectory next to the progress file. Each pipeline
        run creates a timestamped subfolder within this directory.
    """

    # Regex pattern to extract HTTP error detail from full error string
    # Matches: ...failed with status code XXX: {"detail":"..."}
    _HTTP_ERROR_PATTERN = re.compile(
        r"failed with status code (\d+):\s*(\{.*\})",
        re.DOTALL,
    )

    def __init__(
        self,
        file_path: str | Path,
        append: bool = False,
        max_error_length: int = 500,
        error_dir: Optional[str | Path] = None,
    ) -> None:
        self._file_path = Path(file_path)
        self._append = append
        self._max_error_length = max_error_length
        # Generate run timestamp for unique subfolder per pipeline run
        self._run_timestamp = time.strftime("%Y-%m-%dT%H-%M-%S")
        # Default error dir: sibling to progress file, with per-run subfolder
        base_error_dir = (
            Path(error_dir) if error_dir else self._file_path.parent / "error_details"
        )
        self._error_dir = base_error_dir / self._run_timestamp
        self._lock = threading.Lock()
        self._file: Optional[TextIO] = None
        self._opened = False
        self._error_counter = 0  # For unique error file names

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
        """Write the event as a JSON line with immediate flush.

        If the event has a large error message, it is truncated and the full
        error is dumped to a separate file. The path to the full error file
        is included in the event as `error_detail_file`.
        """
        with self._lock:
            try:
                # Truncate large errors and dump full details to file
                event = self._maybe_truncate_error(event)
                f = self._ensure_open()
                line = json.dumps(dict(event), ensure_ascii=False)
                f.write(line + "\n")
                f.flush()  # Ensure event is written immediately
            except Exception as e:
                logger.warning(f"JSON file report failed: {e}")

    def _maybe_truncate_error(self, event: ProgressEvent) -> ProgressEvent:
        """Truncate long errors and dump full details to separate file.

        Parameters
        ----------
        event : ProgressEvent
            The event to potentially truncate.

        Returns
        -------
        ProgressEvent
            The event with truncated error and error_detail_file if applicable,
            or the original event if no truncation was needed.
        """
        error = event.get("error")
        if not error or len(error) <= self._max_error_length:
            return event

        # Extract the useful part (e.g., HTTP status + detail)
        truncated = self._extract_error_summary(error)

        # Dump full error to separate file
        error_file = self._dump_full_error(event, error)

        # Return modified event with truncated error + file reference
        new_event = dict(event)
        new_event["error"] = truncated
        new_event["error_detail_file"] = str(error_file)
        return new_event  # type: ignore[return-value]

    def _extract_error_summary(self, error: str) -> str:
        """Extract the useful HTTP error detail from a full error string.

        Attempts to parse common HTTP error patterns and extract the status
        code and detail message. Falls back to simple truncation if pattern
        doesn't match.

        Parameters
        ----------
        error : str
            The full error string.

        Returns
        -------
        str
            A truncated/summarized version of the error.
        """
        match = self._HTTP_ERROR_PATTERN.search(error)
        if match:
            status_code = match.group(1)
            try:
                detail_json = json.loads(match.group(2))
                detail_msg = detail_json.get("detail", "Unknown error")
                return f"HTTP {status_code}: {detail_msg}"
            except json.JSONDecodeError:
                # JSON parse failed, extract raw detail string
                raw_detail = match.group(2)
                if len(raw_detail) > 200:
                    raw_detail = raw_detail[:200] + "..."
                return f"HTTP {status_code}: {raw_detail}"

        # Fallback: just truncate with indicator
        return (
            error[: self._max_error_length] + "... [truncated, see error_detail_file]"
        )

    def _dump_full_error(self, event: ProgressEvent, full_error: str) -> Path:
        """Dump full error to a separate file and return its path.

        Creates a JSON file containing the full error details along with
        context from the event (timestamp, file_path, phase, meta).

        Files are stored in a per-run timestamped subfolder under error_dir.

        Parameters
        ----------
        event : ProgressEvent
            The event containing error context.
        full_error : str
            The full error string to dump.

        Returns
        -------
        Path
            Path to the created error detail file.
        """
        self._error_dir.mkdir(parents=True, exist_ok=True)

        # Create unique filename using phase and counter
        # (timestamp subfolder already provides run isolation)
        phase = event.get("phase", "unknown")
        self._error_counter += 1
        error_file = self._error_dir / f"{phase}_{self._error_counter:04d}.json"

        error_payload = {
            "timestamp": event.get("timestamp"),
            "file_path": event.get("file_path"),
            "phase": phase,
            "status": event.get("status"),
            "full_error": full_error,
            "traceback": event.get("traceback"),
            "meta": event.get("meta"),
        }

        try:
            with open(error_file, "w", encoding="utf-8") as f:
                json.dump(error_payload, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"Failed to dump error details to {error_file}: {e}")

        return error_file

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
    max_error_length: int = 500,
    error_dir: Optional[str] = None,
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
    max_error_length : int
        For json_file mode: maximum length of error messages before truncation
        (default: 500). Errors longer than this are truncated and full details
        are dumped to a separate file.
    error_dir : str | None
        For json_file mode: base directory for error detail files. If None
        (default), creates an `error_details` subdirectory next to the progress
        file. Each pipeline run creates a timestamped subfolder within this
        directory to isolate errors from different runs.

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
        return JsonFileReporter(
            actual_path,
            append=append,
            max_error_length=max_error_length,
            error_dir=error_dir,
        )

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
