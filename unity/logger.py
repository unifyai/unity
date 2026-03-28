"""
unity/logger.py
===============

Unity's runtime logging and OpenTelemetry tracing configuration.

File-based logging:
    When UNITY_LOG_DIR is set (via env var or configure_log_dir()),
    Unity's LOGGER output is written to two files:
      - {UNITY_LOG_DIR}/unity.log           (DEBUG + INFO)
      - {UNITY_LOG_DIR}/unity_info_only.log (INFO only)
    This captures async tool loop events, manager operations, etc.

OpenTelemetry tracing:
    When UNITY_OTEL is enabled, manager operations and async tool loops
    create OTel spans that propagate trace context to downstream libraries.

    - UNITY_OTEL: Master switch (default: false)
    - UNITY_OTEL_ENDPOINT: OTLP endpoint for trace export (optional)
    - UNITY_OTEL_LOG_DIR: Directory for file-based span export (optional)

    Unity acts as the root TracerProvider when enabled. Child libraries
    (unillm, unify) will detect the existing provider and create child spans.

File-based span export:
    When UNITY_OTEL_LOG_DIR is set, spans are written to JSONL files keyed
    by trace_id: {UNITY_OTEL_LOG_DIR}/{trace_id}.jsonl

    This enables full-stack trace correlation across processes. Orchestra
    (running in a separate FastAPI process) receives the traceparent header
    from Unify HTTP calls and can write its spans to the same directory.
"""

from __future__ import annotations

import json
import logging
import os
import time
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from unity.settings import SETTINGS

# ─────────────────────────────────────────────────────────────────────────────
# Cgroup Memory Reporting
#
# Reads container memory from cgroup v1/v2 pseudo-files and formats a compact
# tag like ``[4821/8192 MiB]`` for inclusion in every log line.  Cached with
# a short TTL to avoid per-record syscall overhead.  Returns an empty string
# when cgroup files are absent (local dev, macOS, tests).
# ─────────────────────────────────────────────────────────────────────────────

_CGROUP_MEM_FILE: str | None = None  # path to current-usage pseudo-file
_CGROUP_MEM_MAX: int | None = None  # limit in bytes (None = no limit)
_CGROUP_CACHE: tuple[float, int] = (0.0, 0)  # (monotonic_ts, usage_bytes)
_CGROUP_CACHE_TTL = 1.0  # seconds


def _init_cgroup_paths() -> None:
    """Detect cgroup v1/v2 memory files (called once, lazily)."""
    global _CGROUP_MEM_FILE, _CGROUP_MEM_MAX
    if _CGROUP_MEM_FILE is not None:
        return

    # cgroup v2
    if os.path.isfile("/sys/fs/cgroup/memory.current"):
        _CGROUP_MEM_FILE = "/sys/fs/cgroup/memory.current"
        try:
            with open("/sys/fs/cgroup/memory.max") as f:
                val = f.read().strip()
                _CGROUP_MEM_MAX = None if val == "max" else int(val)
        except (OSError, ValueError):
            _CGROUP_MEM_MAX = None
    # cgroup v1
    elif os.path.isfile("/sys/fs/cgroup/memory/memory.usage_in_bytes"):
        _CGROUP_MEM_FILE = "/sys/fs/cgroup/memory/memory.usage_in_bytes"
        try:
            with open("/sys/fs/cgroup/memory/memory.limit_in_bytes") as f:
                _CGROUP_MEM_MAX = int(f.read().strip())
        except (OSError, ValueError):
            _CGROUP_MEM_MAX = None
    else:
        _CGROUP_MEM_FILE = ""  # sentinel: no cgroup available


def _get_memory_tag() -> str:
    """Return e.g. ``[4821/8192 MiB]`` or ``""`` if cgroup unavailable."""
    global _CGROUP_CACHE

    _init_cgroup_paths()
    if not _CGROUP_MEM_FILE:
        return ""

    now = time.monotonic()
    cached_ts, cached_bytes = _CGROUP_CACHE
    if now - cached_ts < _CGROUP_CACHE_TTL:
        current = cached_bytes
    else:
        try:
            with open(_CGROUP_MEM_FILE) as f:
                current = int(f.read().strip())
            _CGROUP_CACHE = (now, current)
        except (OSError, ValueError):
            return ""

    mib = current >> 20  # // 1048576
    if _CGROUP_MEM_MAX:
        max_mib = _CGROUP_MEM_MAX >> 20
        pct = current * 100 // _CGROUP_MEM_MAX
        return f"[{mib}/{max_mib} MiB ({pct}%)]"
    return f"[{mib} MiB]"


# ─────────────────────────────────────────────────────────────────────────────
# Logger Instance
# ─────────────────────────────────────────────────────────────────────────────

LOGGER = logging.getLogger("unity")

# Unique identifier for this process lifetime (used for log correlation)
SESSION_ID = datetime.now(timezone.utc).isoformat()

# File handler state (managed by configure_log_dir)
_FILE_HANDLER: Optional[logging.FileHandler] = None
_INFO_FILE_HANDLER: Optional[logging.FileHandler] = None
_LOG_DIR: Optional[Path] = None

# ─────────────────────────────────────────────────────────────────────────────
# OpenTelemetry Setup
# ─────────────────────────────────────────────────────────────────────────────

_OTEL_ENABLED = SETTINGS.UNITY_OTEL
_OTEL_ENDPOINT = SETTINGS.UNITY_OTEL_ENDPOINT
_OTEL_LOG_DIR = SETTINGS.UNITY_OTEL_LOG_DIR
_OTEL_INITIALIZED = False
_TRACER = None


# ─────────────────────────────────────────────────────────────────────────────
# File-based Span Exporter
# ─────────────────────────────────────────────────────────────────────────────


class FileSpanExporter:
    """Exports spans to JSONL files, one file per trace_id.

    This enables full-stack trace correlation across processes (Unity, Orchestra).
    Each span is written as a JSON line to {log_dir}/{trace_id}.jsonl.
    """

    def __init__(self, log_dir: Path, service_name: str = "unity"):
        self.log_dir = log_dir
        self.service_name = service_name
        self._lock = threading.Lock()
        # Ensure directory exists
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def export(self, spans) -> int:
        """Export a batch of spans to files.

        Args:
            spans: Sequence of ReadableSpan objects

        Returns:
            SpanExportResult.SUCCESS (0) or SpanExportResult.FAILURE (1)
        """
        try:
            for span in spans:
                self._write_span(span)
            return 0  # SUCCESS
        except Exception as e:
            LOGGER.warning(f"FileSpanExporter failed to export spans: {e}")
            return 1  # FAILURE

    def _write_span(self, span) -> None:
        """Write a single span to its trace file."""
        ctx = span.get_span_context()
        if ctx is None or not ctx.is_valid:
            return

        trace_id = f"{ctx.trace_id:032x}"
        span_id = f"{ctx.span_id:016x}"

        # Get parent span ID if exists
        parent_span_id = None
        if span.parent is not None:
            parent_span_id = f"{span.parent.span_id:016x}"

        # Build span data
        span_data: dict[str, Any] = {
            "trace_id": trace_id,
            "span_id": span_id,
            "parent_span_id": parent_span_id,
            "name": span.name,
            "service": self.service_name,
            "start_time": (
                datetime.fromtimestamp(
                    span.start_time / 1e9,
                    tz=timezone.utc,
                ).isoformat()
                if span.start_time
                else None
            ),
            "end_time": (
                datetime.fromtimestamp(
                    span.end_time / 1e9,
                    tz=timezone.utc,
                ).isoformat()
                if span.end_time
                else None
            ),
            "duration_ms": (
                (span.end_time - span.start_time) / 1e6
                if span.end_time and span.start_time
                else None
            ),
            "status": span.status.status_code.name if span.status else None,
            "attributes": dict(span.attributes) if span.attributes else {},
        }

        # Write to trace file (append mode, one span per line)
        trace_file = self.log_dir / f"{trace_id}.jsonl"
        with self._lock:
            with open(trace_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(span_data, default=str) + "\n")

    def shutdown(self) -> None:
        """Shutdown the exporter (no-op for file exporter)."""

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        """Force flush any buffered spans (no-op for file exporter)."""
        return True


def _setup_otel() -> None:
    """Initialize OpenTelemetry if enabled and not already configured.

    Unity is typically the outermost layer, so we create the TracerProvider.
    Child libraries (unillm, unify) will detect the existing provider
    and use it to create child spans in the same trace.

    When UNITY_OTEL_LOG_DIR is set, spans are also written to JSONL files
    keyed by trace_id for full-stack trace correlation.
    """
    global _OTEL_INITIALIZED, _TRACER

    if _OTEL_INITIALIZED or not _OTEL_ENABLED:
        return

    _OTEL_INITIALIZED = True

    try:
        from opentelemetry import trace
        from opentelemetry.sdk.resources import SERVICE_NAME, Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor

        # Check if a TracerProvider already exists
        # Note: ProxyTracerProvider is a lazy wrapper that delegates to NoOpTracerProvider
        # by default until a real provider is set. We should replace it with our own.
        existing = trace.get_tracer_provider()
        if existing and not isinstance(
            existing,
            (trace.NoOpTracerProvider, trace.ProxyTracerProvider),
        ):
            # Someone else already configured OTel - use theirs
            _TRACER = trace.get_tracer("unity")
            LOGGER.debug("Using existing OTel TracerProvider")
            return

        # We're the outermost layer - set up our own provider
        resource = Resource.create({SERVICE_NAME: "unity"})
        provider = TracerProvider(resource=resource)

        # Add OTLP exporter if endpoint configured
        if _OTEL_ENDPOINT:
            try:
                from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
                    OTLPSpanExporter,
                )
                from opentelemetry.sdk.trace.export import BatchSpanProcessor

                exporter = OTLPSpanExporter(endpoint=_OTEL_ENDPOINT, insecure=True)
                provider.add_span_processor(BatchSpanProcessor(exporter))
                LOGGER.debug(f"Configured OTLP exporter at {_OTEL_ENDPOINT}")
            except ImportError:
                LOGGER.warning(
                    "OTLP exporter not available - install opentelemetry-exporter-otlp",
                )
            except Exception as e:
                LOGGER.warning(f"Failed to configure OTLP exporter: {e}")

        # Add file exporter if log directory configured
        if _OTEL_LOG_DIR:
            try:
                log_dir = Path(_OTEL_LOG_DIR)
                file_exporter = FileSpanExporter(log_dir, service_name="unity")
                # Use SimpleSpanProcessor for immediate writes (not batched)
                provider.add_span_processor(SimpleSpanProcessor(file_exporter))
                LOGGER.debug(f"Configured file span exporter at {_OTEL_LOG_DIR}")
            except Exception as e:
                LOGGER.warning(f"Failed to configure file span exporter: {e}")

        trace.set_tracer_provider(provider)
        _TRACER = trace.get_tracer("unity")
        LOGGER.debug("Initialized OTel TracerProvider for unity")

    except ImportError:
        LOGGER.debug("OpenTelemetry not available - tracing disabled")
    except Exception as e:
        LOGGER.warning(f"Failed to initialize OpenTelemetry: {e}")


def get_tracer():
    """Get the OpenTelemetry tracer, initializing if needed.

    Returns:
        The tracer instance, or None if OTel is disabled/unavailable.
    """
    global _TRACER
    if _TRACER is None and _OTEL_ENABLED:
        _setup_otel()
    return _TRACER


def is_otel_enabled() -> bool:
    """Check if OpenTelemetry tracing is enabled."""
    return _OTEL_ENABLED


@contextmanager
def unity_span(name: str, **attributes):
    """Create an OTel span for a Unity operation.

    Args:
        name: The span name (e.g., "ContactManager.ask", "async_tool_loop")
        **attributes: Additional span attributes

    Yields:
        The span (or None if OTel disabled)

    Example:
        with unity_span("ContactManager.ask", query="find john") as span:
            # ... do work ...
            if span:
                span.set_attribute("result.count", 5)
    """
    tracer = get_tracer()
    if tracer is None:
        yield None
        return

    try:
        from opentelemetry.trace import SpanKind, Status, StatusCode
    except ImportError:
        yield None
        return

    with tracer.start_as_current_span(
        name,
        kind=SpanKind.INTERNAL,
    ) as span:
        for key, value in attributes.items():
            if value is not None:
                if isinstance(value, (int, float, bool)):
                    span.set_attribute(f"unity.{key}", value)
                else:
                    span.set_attribute(f"unity.{key}", str(value))

        try:
            yield span
        except Exception as e:
            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.set_attribute("error.type", type(e).__name__)
            span.set_attribute("error.message", str(e))
            raise


def set_span_ok(span) -> None:
    """Set the span status to OK."""
    if span is None:
        return
    try:
        from opentelemetry.trace import Status, StatusCode

        span.set_status(Status(StatusCode.OK))
    except Exception:
        pass


def get_current_trace_id() -> Optional[str]:
    """Get the current trace ID if an OTel span is active.

    Returns:
        The trace ID as a 32-char hex string, or None if no active span.
    """
    if not _OTEL_ENABLED:
        return None

    try:
        from opentelemetry import trace

        span = trace.get_current_span()
        if span is None:
            return None
        ctx = span.get_span_context()
        if ctx is not None and ctx.is_valid:
            return f"{ctx.trace_id:032x}"
    except Exception:
        pass
    return None


def get_otel_log_dir() -> Optional[Path]:
    """Get the OTel span log directory, if configured.

    Returns:
        Path to the UNITY_OTEL_LOG_DIR, or None if not configured.
    """
    if not _OTEL_LOG_DIR:
        return None
    return Path(_OTEL_LOG_DIR)


# ─────────────────────────────────────────────────────────────────────────────
# Console (Terminal) Logging
#
# This is the single authority for all Unity log output.  No other module
# should call logging.basicConfig(), add handlers, or filter the root logger.
# ─────────────────────────────────────────────────────────────────────────────

# Prevent unity records from propagating to the root logger.  This eliminates
# duplicate output from any root-level handler (e.g. logging.basicConfig())
# that third-party code may install.
LOGGER.propagate = False


from unity.syntax_highlight import highlight_code_blocks  # noqa: E402


class _MillisFormatter(logging.Formatter):
    """Formatter that prepends ``HH:MM:SS.mmm`` to each log line.

    Messages that don't already start with a non-ASCII character (i.e. an
    emoji icon from the hierarchical logger) are auto-prefixed with ``⬥``
    so every terminal line has a consistent visual anchor.

    When *stream* is a TTY, markdown-fenced code blocks are syntax-
    highlighted via Pygments.
    """

    _DEFAULT_ICON = "⬥"

    def __init__(self, *args, stream=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._is_tty = getattr(stream, "isatty", lambda: False)()

    def format(self, record: logging.LogRecord) -> str:
        dt = datetime.fromtimestamp(record.created, tz=timezone.utc).astimezone()
        ts = dt.strftime("%H:%M:%S") + f".{int(dt.microsecond / 1000):03d}"
        msg = record.getMessage()
        if msg and ord(msg[0]) < 128:
            msg = f"{self._DEFAULT_ICON} {msg}"
        if self._is_tty:
            msg = highlight_code_blocks(msg)
        return f"{ts} {msg}"


LOGGER.setLevel(logging.DEBUG)

if SETTINGS.UNITY_TERMINAL_LOG:
    import sys

    _handler = logging.StreamHandler(sys.stdout)
    _handler.setFormatter(_MillisFormatter(stream=sys.stdout))
    _handler.setLevel(getattr(logging, SETTINGS.UNITY_TERMINAL_LOG_LEVEL, logging.INFO))

    _already_configured = any(
        isinstance(h, logging.StreamHandler) and getattr(h, "_unity_terminal", False)
        for h in LOGGER.handlers
    )

    if not _already_configured:
        _handler._unity_terminal = True  # type: ignore[attr-defined]
        LOGGER.addHandler(_handler)

# Mute noisy third-party loggers so only unity.* output reaches the terminal.
for _lib in (
    "httpx",
    "urllib3",
    "openai",
    "LiteLLM",
    "LiteLLM Proxy",
    "LiteLLM Router",
):
    logging.getLogger(_lib).setLevel(logging.WARNING)

# RapidOCR reconfigures its own logger on import (unconditionally calling
# setLevel(INFO)), so setLevel here would be clobbered.  A filter survives.
logging.getLogger("RapidOCR").addFilter(
    lambda record: record.levelno >= logging.WARNING,
)

# File-only loggers: cut propagation so nothing reaches the root/terminal
# handlers, and attach the DEBUG file handler in configure_log_dir() so the
# output still lands in unity.log.  NOT wired to unity_info_only.log — that
# file mirrors the terminal exactly (Unity INFO+ only).
#
# "py.warnings" captures Python warnings (e.g. unawaited coroutine
# RuntimeWarnings from LiteLLM's async bridge during task cancellation).
# They are harmless but noisy on the terminal; routing through the logging
# system keeps them in unity.log for debugging without cluttering stdout.
_FILE_ONLY_LOGGERS = [
    logging.getLogger(name)
    for name in (
        "livekit",
        "livekit.agents",
        "livekit.plugins",
        "PIL",
        "py.warnings",
    )
]
for _fo in _FILE_ONLY_LOGGERS:
    _fo.setLevel(logging.DEBUG)
    _fo.propagate = False

logging.captureWarnings(True)

# ─────────────────────────────────────────────────────────────────────────────
# File-based Logging Configuration
# ─────────────────────────────────────────────────────────────────────────────


class _MemoryFileFormatter(logging.Formatter):
    """File formatter that prepends cgroup memory usage to each line."""

    def format(self, record: logging.LogRecord) -> str:
        mem = _get_memory_tag()
        base = super().format(record)
        if mem:
            # Insert memory tag after the log-level field
            return f"{base} {mem}"
        return base


def configure_log_dir(log_dir: Optional[str] = None) -> Optional[Path]:
    """Configure or reconfigure the Unity LOGGER file output directory.

    When configured, LOGGER output is written to two files:
      - {log_dir}/unity.log           (everything: Unity DEBUG+, plus
                                        third-party file-only loggers)
      - {log_dir}/unity_info_only.log (Unity INFO+ only — mirrors the
                                        terminal exactly)

    This captures async tool loop events, manager operations, hierarchical
    session logs, and any other code using LOGGER.

    Call this after setting UNITY_LOG_DIR if the env var was set
    after this module was imported.

    Args:
        log_dir: Explicit log directory path. If None, reads from
                 UNITY_LOG_DIR env var (or SETTINGS.UNITY_LOG_DIR).

    Returns:
        The configured log directory Path, or None if disabled.
    """
    global _FILE_HANDLER, _INFO_FILE_HANDLER, _LOG_DIR

    # Remove existing file handlers if any
    if _FILE_HANDLER is not None:
        LOGGER.removeHandler(_FILE_HANDLER)
        for _fo in _FILE_ONLY_LOGGERS:
            _fo.removeHandler(_FILE_HANDLER)
        _FILE_HANDLER.close()
        _FILE_HANDLER = None
    if _INFO_FILE_HANDLER is not None:
        LOGGER.removeHandler(_INFO_FILE_HANDLER)
        _INFO_FILE_HANDLER.close()
        _INFO_FILE_HANDLER = None
    _LOG_DIR = None

    # Determine log directory
    if log_dir is not None:
        os.environ["UNITY_LOG_DIR"] = log_dir
        dir_path = log_dir
    else:
        dir_path = os.environ.get("UNITY_LOG_DIR", "").strip() or SETTINGS.UNITY_LOG_DIR

    if not dir_path:
        return None

    try:
        log_path = Path(dir_path)
        log_path.mkdir(parents=True, exist_ok=True)

        fmt = "%(asctime)s %(levelname)7s %(message)s"

        log_file = log_path / "unity.log"
        handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        handler.setFormatter(_MemoryFileFormatter(fmt))
        handler.setLevel(logging.DEBUG)
        handler._unity_file_handler = True  # type: ignore[attr-defined]
        LOGGER.addHandler(handler)
        for _fo in _FILE_ONLY_LOGGERS:
            _fo.addHandler(handler)
        _FILE_HANDLER = handler

        info_log_file = log_path / "unity_info_only.log"
        info_handler = logging.FileHandler(info_log_file, mode="a", encoding="utf-8")
        info_handler.setFormatter(logging.Formatter(fmt))
        info_handler.setLevel(logging.INFO)
        info_handler._unity_file_handler = True  # type: ignore[attr-defined]
        LOGGER.addHandler(info_handler)
        _INFO_FILE_HANDLER = info_handler

        _LOG_DIR = log_path

        LOGGER.debug(f"Unity file logging enabled: {log_file}")
        return log_path

    except Exception as e:
        # Best-effort: log to console if file logging fails
        logging.warning(f"Failed to configure Unity log directory {dir_path}: {e}")
        return None


def get_log_dir() -> Optional[Path]:
    """Get the current Unity log directory, if configured."""
    return _LOG_DIR


# Auto-configure from settings on module load (if UNITY_LOG_DIR is set)
if SETTINGS.UNITY_LOG_DIR:
    configure_log_dir(SETTINGS.UNITY_LOG_DIR)
