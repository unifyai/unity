"""Format-aware file parser facade.

This module is the single entry point for all file parsing.  The
``FileParser`` class selects a backend by file format, enforces output
invariants, and orchestrates concurrent/subprocess batch parsing.

Infrastructure that does **not** belong to the core parse flow is
delegated to focused utility modules:

- ``utils.subprocess_pool``  — warm ``ProcessPoolExecutor``, forkserver
  setup, RLIMIT_AS enforcement, subprocess entry point.
- ``utils.memory_scheduler`` — format-agnostic heavy/light file
  classification based on dynamic system-memory percentages.
- ``utils.diagnostics``      — optional ``SAVE_PARSED_RESULTS`` debug
  serialisation to disk.
"""

from __future__ import annotations

import logging
import os
import time
from concurrent.futures import BrokenExecutor, Future, ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

from unity.file_manager.file_parsers.registry import BackendRegistry
from unity.file_manager.file_parsers.settings import FILE_PARSER_SETTINGS
from unity.file_manager.file_parsers.types.backend import BaseFileParserBackend
from unity.file_manager.file_parsers.types.contracts import (
    FileParseRequest,
    FileParseResult,
    FileParseTrace,
    StepStatus,
)
from unity.file_manager.file_parsers.types.formats import (
    FileFormat,
    MimeType,
    extension_to_format,
    extension_to_mime,
)
from unity.file_manager.file_parsers.utils.diagnostics import save_result_to_disk
from unity.file_manager.file_parsers.utils.memory_scheduler import (
    classify_file,
    fmt_bytes,
    get_system_memory_bytes,
)
from unity.file_manager.file_parsers.utils.postconditions import (
    enforce_parse_success_invariants,
)
from unity.file_manager.file_parsers.utils.subprocess_pool import (
    force_reset_pool,
    get_or_create_pool,
    shutdown_pool,
    subprocess_parse_single,
)
from unity.file_manager.types.config import ParseConfig

logger = logging.getLogger(__name__)


class FileParser:
    """Format-aware file parser facade.

    The FileManager should depend on this class (not on concrete backends).

    Guarantees
    ----------
    ``FileParser.parse()`` is designed to be *safe* to call from batch
    pipelines:

    - It **never raises** for backend failures; unexpected backend
      exceptions are caught and converted into
      ``FileParseResult(status="error")``.
    - It enforces key invariants on outputs (trace identity, format/mime
      defaults, and minimal retrieval fields on success).
    - It keeps the parser boundary stable: input is ``FileParseRequest``,
      output is ``FileParseResult`` (no dicts, no legacy models).

    Format awareness
    ----------------
    Backends are selected by ``FileFormat``, but the facade also enforces
    format-aware output policy:

    - Spreadsheets should not dump huge data into text fields.
    - ``summary`` and ``metadata`` should be present on success.

    Backends are resolved via a config-driven registry so that:

    - Swapping a single format backend (e.g. XLSX) is a config change.
    - Swapping an entire backend set is a config change.
    - ``FileParser`` remains agnostic of which parsing library is used.
    """

    def __init__(
        self,
        *,
        backends: Optional[Sequence[BaseFileParserBackend]] = None,
        registry: Optional[BackendRegistry] = None,
    ) -> None:
        # For tests / power-users: allow injecting already-instantiated backends.
        # If provided, FileParser will use these instances directly and will not
        # do any dynamic importing.
        self._backends: Optional[List[BaseFileParserBackend]] = (
            list(backends) if backends is not None else None
        )

        # Default registry uses built-in mapping; config can override per-call.
        self._default_registry: BackendRegistry = (
            registry or BackendRegistry.from_config()
        )

    # ------------------------------------------------------------------
    # Backend selection
    # ------------------------------------------------------------------

    def _pick_backend(
        self,
        fmt: Optional[FileFormat],
        *,
        registry: BackendRegistry,
    ) -> Optional[BaseFileParserBackend]:
        if self._backends is not None:
            for b in self._backends:
                try:
                    if b.can_handle(fmt):
                        return b
                except Exception as e:
                    # Do not silently ignore backend predicate failures; log for observability.
                    logger.debug(
                        "Injected backend can_handle failed (backend=%s fmt=%s): %s",
                        type(b).__name__,
                        fmt,
                        e,
                        exc_info=True,
                    )
                    continue
            return None
        return registry.pick_backend(fmt)

    # ------------------------------------------------------------------
    # Single-file parse
    # ------------------------------------------------------------------

    def parse(
        self,
        request: FileParseRequest,
        *,
        parse_config: Optional[ParseConfig] = None,
    ) -> FileParseResult:
        """Parse a single file.

        Parameters
        ----------
        request
            Canonical parse request.  ``logical_path`` is the stable
            external identifier (used for contexts/records).
            ``source_local_path`` is the actual on-disk path to read
            (may be a temp export).
        parse_config
            Optional typed parser configuration.  This is the canonical
            way to:

            - Override per-format backend implementations
              (``backend_class_paths_by_format``).
            - Control parse-stage concurrency in ``parse_batch``
              (``max_concurrent_parses``).

        Returns
        -------
        FileParseResult
            Always returned.  On failure ``status="error"`` and ``error``
            is populated.
        """
        reg = (
            BackendRegistry.from_config(
                backend_class_paths_by_format=getattr(
                    parse_config,
                    "backend_class_paths_by_format",
                    None,
                ),
            )
            if parse_config is not None
            else self._default_registry
        )
        return self._parse_single(request, registry=reg)

    def _parse_single(
        self,
        request: FileParseRequest,
        *,
        registry: BackendRegistry,
    ) -> FileParseResult:
        """Internal parse implementation that uses a pre-built registry."""
        t0 = time.perf_counter()
        logical_path = str(request.logical_path)
        source_p = Path(request.source_local_path).expanduser().resolve()

        # Prefer deriving format from the actual source path when not provided.
        fmt = request.file_format or extension_to_format(source_p.suffix)
        mt = request.mime_type or extension_to_mime(source_p.suffix)

        # Fallback: treat unknown extensions as plain text (best-effort parsing).
        if fmt == FileFormat.UNKNOWN:
            fmt = FileFormat.TXT
            if request.mime_type is None:
                mt = MimeType.TEXT_PLAIN

        ctx = request.model_copy(
            update={
                "file_format": fmt,
                "mime_type": mt,
                "source_local_path": str(source_p),
            },
        )

        backend = self._pick_backend(fmt, registry=registry)
        if backend is None:
            trace = FileParseTrace(
                logical_path=logical_path,
                backend="none",
                file_format=fmt,
                mime_type=mt,
                status=StepStatus.FAILED,
                source_local_path=str(source_p),
                parsed_local_path=str(source_p),
                warnings=[f"No backend for format: {fmt}"],
            )
            trace.duration_ms = (time.perf_counter() - t0) * 1000.0
            return FileParseResult(
                logical_path=logical_path,
                status="error",
                error=f"No backend for format: {fmt}",
                file_format=fmt,
                mime_type=mt,
                trace=trace,
            )

        backend_name = str(getattr(backend, "name", type(backend).__name__))
        try:
            out = backend.parse(ctx)
            if not isinstance(out, FileParseResult):
                raise TypeError(
                    f"Backend {backend_name} returned invalid type: "
                    f"{type(out)!r} (expected FileParseResult)",
                )
        except Exception as e:
            # Backends should generally return FileParseResult on error, but this
            # wrapper ensures FileParser never crashes the FileManager pipeline.
            logger.exception(
                "Backend parse raised (backend=%s path=%s): %s",
                backend_name,
                logical_path,
                e,
            )
            trace = FileParseTrace(
                logical_path=logical_path,
                backend=backend_name,
                file_format=fmt,
                mime_type=mt,
                status=StepStatus.FAILED,
                source_local_path=str(source_p),
                parsed_local_path=str(source_p),
                warnings=[f"backend_exception: {e}"],
            )
            trace.duration_ms = (time.perf_counter() - t0) * 1000.0
            return FileParseResult(
                logical_path=logical_path,
                status="error",
                error=str(e),
                file_format=fmt,
                mime_type=mt,
                trace=trace,
            )

        # Enforce trace presence for observability.
        if out.trace is None:
            out.trace = FileParseTrace(
                logical_path=logical_path,
                backend=backend_name,
                file_format=out.file_format or fmt,
                mime_type=out.mime_type or mt,
                status=(
                    StepStatus.SUCCESS if out.status == "success" else StepStatus.FAILED
                ),
                source_local_path=str(source_p),
                parsed_local_path=str(source_p),
            )

        # The parser does not perform any FileManager-specific lowering/adaptation.
        # Ensure outward-facing identity is always the logical path.
        out.logical_path = logical_path
        out.file_format = out.file_format or fmt
        out.mime_type = out.mime_type or mt
        out.trace.logical_path = logical_path
        out.trace.source_local_path = str(source_p)
        if out.trace.parsed_local_path is None:
            out.trace.parsed_local_path = out.trace.source_local_path
        if out.status == "error":
            out.trace.status = StepStatus.FAILED

        # -------------------- Postconditions / invariants -------------------- #
        enforce_parse_success_invariants(out, settings=FILE_PARSER_SETTINGS)

        # Make trace duration cover the full facade work (backend + invariants/enforcement).
        out.trace.duration_ms = (time.perf_counter() - t0) * 1000.0

        save_result_to_disk(out)
        return out

    # ------------------------------------------------------------------
    # Batch parsing
    # ------------------------------------------------------------------

    def parse_batch(
        self,
        requests: Iterable[FileParseRequest],
        *,
        raises_on_error: bool = False,
        parse_config: Optional[ParseConfig] = None,
    ) -> List[FileParseResult]:
        """Parse multiple files with optional bounded concurrency.

        Notes
        -----
        - This method preserves input ordering in its outputs.
        - Concurrency is implemented with a ThreadPool by default.  When
          ``parse_config.subprocess_isolation`` is True a ProcessPool is
          used instead, ensuring that each file's parse runs in a
          dedicated child process whose memory is fully reclaimed by the
          OS on exit.
        - Concurrency is derived from ``parse_config.max_concurrent_parses``
          and is conservatively capped to 8 to avoid resource exhaustion
          in typical environments.
        - When ``raises_on_error=True``, the first error result raises a
          ``RuntimeError``.  In ingestion pipelines this should generally
          remain ``False``.

        Adaptive scheduling (subprocess mode)
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        When subprocess isolation is enabled, files are classified as
        *heavy* or *light* (based on file size, a configurable expansion
        factor, and total system RAM via ``psutil``).  Heavy files are
        parsed **one at a time** to prevent concurrent OOM; light files
        are parsed in parallel.  All subprocess workers use
        ``max_tasks_per_child=1`` so memory (including Python arena
        fragments) is fully reclaimed by the OS after each file.
        """
        reqs = list(requests)
        if not reqs:
            return []

        use_subprocess = bool(
            getattr(parse_config, "subprocess_isolation", False),
        )

        reg = (
            BackendRegistry.from_config(
                backend_class_paths_by_format=getattr(
                    parse_config,
                    "backend_class_paths_by_format",
                    None,
                ),
            )
            if parse_config is not None
            else self._default_registry
        )

        # Enforce a conservative hard cap for safety.
        requested = (
            getattr(parse_config, "max_concurrent_parses", 1)
            if parse_config is not None
            else 1
        )
        workers = max(
            1,
            min(int(requested) if isinstance(requested, int) else 1, 8),
        )

        # Sequential fallback (no subprocess isolation needed for serial)
        if (workers <= 1 or len(reqs) <= 1) and not use_subprocess:
            outcomes: List[FileParseResult] = []
            for req in reqs:
                out = self._parse_single(req, registry=reg)
                if raises_on_error and out.status == "error":
                    raise RuntimeError(out.error or "parse failed")
                outcomes.append(out)
            return outcomes

        if use_subprocess:
            return self._parse_batch_subprocess(
                reqs,
                parse_config=parse_config,
                workers=workers,
                raises_on_error=raises_on_error,
            )

        # Parallel parse (threaded): preserve request order in outputs.
        outcomes = []
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(self._parse_single, req, registry=reg) for req in reqs]
            for req, fut in zip(reqs, futures):
                try:
                    out = fut.result()
                except Exception as e:
                    out = _error_result(req, f"parse_batch_exception: {e}")
                if raises_on_error and out.status == "error":
                    raise RuntimeError(out.error or "parse failed")
                outcomes.append(out)
        return outcomes

    # ------------------------------------------------------------------
    # Subprocess-isolated batch with adaptive scheduling
    # ------------------------------------------------------------------

    def _parse_batch_subprocess(
        self,
        reqs: List[FileParseRequest],
        *,
        parse_config: Optional[ParseConfig],
        workers: int,
        raises_on_error: bool,
    ) -> List[FileParseResult]:
        """Run ``parse_batch`` with process-level isolation and adaptive scheduling.

        Files are classified as *heavy* or *light*.  Light files are
        submitted to the warm pool concurrently; heavy files are
        submitted one at a time.  Results are placed back at their
        original indices to preserve ordering.

        Fault tolerance
        ~~~~~~~~~~~~~~~
        Every ``fut.result()`` is guarded by ``parse_timeout_seconds``.
        If a child hangs (e.g., ``MemoryError`` leaves it wedged) or the
        pool breaks (e.g., OOM killer terminated a child), the pool is
        force-reset and processing continues for the remaining files.
        This guarantees the parent process **never** hangs indefinitely.
        """
        config = parse_config or ParseConfig()
        sys_mem = get_system_memory_bytes()
        effective_workers = max(1, min(len(reqs), workers))

        heavy: List[Tuple[int, FileParseRequest]] = []
        light: List[Tuple[int, FileParseRequest]] = []
        for idx, req in enumerate(reqs):
            bucket = classify_file(req, sys_memory=sys_mem, config=config)
            if bucket == "heavy":
                heavy.append((idx, req))
            else:
                light.append((idx, req))

        if heavy:
            logger.info(
                "Adaptive scheduler: %d heavy, %d light files "
                "(sys_mem=%s, threshold=%.0f%%)",
                len(heavy),
                len(light),
                fmt_bytes(sys_mem),
                config.heavy_file_memory_pct * 100,
            )

        results: List[Optional[FileParseResult]] = [None] * len(reqs)
        pool_needs_reset = False

        # Light files — dispatched concurrently to the pool.
        if light:
            pool = get_or_create_pool(effective_workers)
            light_futures: List[Tuple[Future, int]] = []  # type: ignore[type-arg]
            for idx, req in light:
                try:
                    fut = pool.submit(
                        subprocess_parse_single,
                        req,
                        parse_config,
                    )
                    light_futures.append((fut, idx))
                except BrokenExecutor:
                    pool_needs_reset = True
                    results[idx] = _error_result(
                        req,
                        "subprocess_pool_broken_on_submit",
                    )
                    break

            for fut, idx in light_futures:
                file_timeout = _adaptive_timeout(reqs[idx], config)
                out, needs_reset = _collect_future(
                    fut,
                    reqs[idx],
                    raises_on_error,
                    timeout=file_timeout,
                )
                results[idx] = out
                if needs_reset:
                    pool_needs_reset = True

        # Heavy files — dispatched one at a time to avoid concurrent OOM.
        for idx, req in heavy:
            if pool_needs_reset:
                force_reset_pool(effective_workers)
                pool_needs_reset = False

            pool = get_or_create_pool(effective_workers)
            try:
                fut = pool.submit(subprocess_parse_single, req, parse_config)
            except BrokenExecutor:
                force_reset_pool(effective_workers)
                pool = get_or_create_pool(effective_workers)
                try:
                    fut = pool.submit(
                        subprocess_parse_single,
                        req,
                        parse_config,
                    )
                except Exception as e:
                    results[idx] = _error_result(
                        req,
                        f"subprocess_submit_failed: {e}",
                    )
                    continue

            file_timeout = _adaptive_timeout(req, config)
            out, needs_reset = _collect_future(
                fut,
                req,
                raises_on_error,
                timeout=file_timeout,
            )
            results[idx] = out
            if needs_reset:
                pool_needs_reset = True

        # Final cleanup: if the pool was left in a bad state, reset it so
        # the next parse_batch call starts with a healthy pool.
        if pool_needs_reset:
            force_reset_pool(effective_workers)

        return [r for r in results if r is not None]  # type: ignore[misc]

    # ------------------------------------------------------------------
    # Pool lifecycle (delegates to utils.subprocess_pool)
    # ------------------------------------------------------------------

    @staticmethod
    def shutdown_pool() -> None:
        """Shut down the persistent subprocess pool (if any)."""
        shutdown_pool()


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _adaptive_timeout(
    req: FileParseRequest,
    config: ParseConfig,
) -> float:
    """Compute per-file timeout that scales with file size.

    Returns ``max(parse_timeout_seconds, file_size_mb * timeout_seconds_per_mb)``
    so that large files automatically get proportionally more time while small
    files still have a reasonable floor.  If the source file cannot be stat'd
    (e.g., already deleted), falls back to the base ``parse_timeout_seconds``.
    """
    base = config.parse_timeout_seconds
    per_mb = config.timeout_seconds_per_mb
    if per_mb <= 0:
        return base
    try:
        size_bytes = os.path.getsize(req.source_local_path)
    except OSError:
        return base
    size_mb = size_bytes / (1024 * 1024)
    return max(base, size_mb * per_mb)


def _collect_future(
    fut: Future,  # type: ignore[type-arg]
    req: FileParseRequest,
    raises_on_error: bool,
    *,
    timeout: Optional[float] = None,
) -> Tuple[FileParseResult, bool]:
    """Collect a result from a subprocess future with timeout and error handling.

    Returns ``(result, pool_needs_reset)``.  The second element is ``True``
    when the failure mode indicates the pool is unhealthy (timeout or broken
    executor) and the caller should call ``force_reset_pool`` before
    submitting more work.

    Three failure modes are handled so the parent **never** hangs:

    1. **Normal exception** from the child (e.g., parse error) — logged and
       wrapped in an error ``FileParseResult``.
    2. **Timeout** (child stuck, e.g., ``MemoryError`` left it wedged) —
       the pool must be force-reset.
    3. **Broken pool** (child crashed, e.g., OOM killer) — the pool must
       be force-reset.
    """
    pool_needs_reset = False
    try:
        out = fut.result(timeout=timeout)
    except FuturesTimeoutError:
        logger.error(
            "Subprocess parse timed out after %.0fs for %s — "
            "child is likely stuck; pool will be force-reset",
            timeout,
            req.logical_path,
        )
        out = _error_result(
            req,
            f"subprocess_parse_timeout: exceeded {timeout}s",
        )
        pool_needs_reset = True
    except BrokenExecutor as e:
        logger.error(
            "Subprocess pool broken while parsing %s: %s",
            req.logical_path,
            e,
        )
        out = _error_result(req, f"subprocess_pool_broken: {e}")
        pool_needs_reset = True
    except Exception as e:
        logger.exception(
            "Subprocess parse failed for %s: %s",
            req.logical_path,
            e,
        )
        out = _error_result(req, f"subprocess_parse_exception: {e}")
    if raises_on_error and out.status == "error":
        raise RuntimeError(out.error or "parse failed")
    return out, pool_needs_reset


def _error_result(req: FileParseRequest, warning: str) -> FileParseResult:
    """Build an error ``FileParseResult`` for a failed request."""
    trace = FileParseTrace(
        logical_path=str(req.logical_path),
        backend="file_parser",
        file_format=req.file_format,
        mime_type=req.mime_type,
        status=StepStatus.FAILED,
        source_local_path=req.source_local_path,
        parsed_local_path=req.source_local_path,
        warnings=[warning],
    )
    return FileParseResult(
        logical_path=str(req.logical_path),
        status="error",
        error=warning,
        file_format=req.file_format,
        mime_type=req.mime_type,
        trace=trace,
    )
