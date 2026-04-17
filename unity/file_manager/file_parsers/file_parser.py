"""Format-aware file parser facade.

This module is the single entry point for all file parsing.  The
``FileParser`` class selects a backend by file format, enforces output
invariants, and orchestrates concurrent/subprocess batch parsing.

Infrastructure that does **not** belong to the core parse flow is
delegated to focused utility modules:

- ``utils.subprocess_pool``  — per-file isolated process spawning,
  forkserver setup, subprocess entry point.
- ``utils.memory_scheduler`` — format-and-size-based heavy/light file
  classification for batch scheduling.
- ``utils.diagnostics``      — optional ``SAVE_PARSED_RESULTS`` debug
  serialisation to disk.
"""

from __future__ import annotations

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
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
from unity.file_manager.file_parsers.utils.memory_scheduler import classify_file
from unity.file_manager.file_parsers.utils.postconditions import (
    enforce_parse_success_invariants,
)
from unity.file_manager.file_parsers.utils.enrich import enrich_parse_result
from unity.file_manager.file_parsers.utils.subprocess_pool import run_isolated
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
        return self._parse_and_enrich(request, registry=reg)

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

        out.trace.duration_ms = (time.perf_counter() - t0) * 1000.0

        save_result_to_disk(out)
        return out

    def _parse_and_enrich(
        self,
        request: FileParseRequest,
        *,
        registry: BackendRegistry,
    ) -> FileParseResult:
        """Parse a file and run LLM enrichment (summary + metadata).

        This is the standard entry point for all non-subprocess callers.
        Subprocesses call ``_parse_single`` directly (no LLM) and
        enrichment runs in the parent process after the child returns.
        """
        out = self._parse_single(request, registry=registry)
        if out.status == "success":
            try:
                enrich_parse_result(out)
            except Exception:
                logger.debug(
                    "Enrichment failed for %s",
                    out.logical_path,
                    exc_info=True,
                )
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
        """Parse multiple files with LLM enrichment.

        Execution paths
        ~~~~~~~~~~~~~~~
        - **Single file or no subprocess isolation** → sequential
          in-process via ``_parse_and_enrich``.  Used for interactive
          call sites and test fixtures with injected backends.
        - **Multi-file with subprocess isolation** → each file gets its
          own ``multiprocessing.Process`` (via ``_parse_batch_subprocess``).
          Files are classified by format as *heavy* (Docling → serialised)
          or *light* (concurrent).  LLM enrichment runs in the parent
          process after all subprocesses complete.

        Notes
        -----
        - Input ordering is preserved in outputs.
        - Concurrency is capped to ``min(max_concurrent_parses, 8)``.
        - ``raises_on_error=True`` raises on the first error result.
        """
        reqs = list(requests)
        if not reqs:
            return []

        use_subprocess = bool(
            getattr(parse_config, "subprocess_isolation", False),
        )
        if use_subprocess and self._backends is not None:
            logger.debug(
                "Ignoring subprocess isolation because FileParser is using injected backends",
            )
            use_subprocess = False

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

        # Single file or no subprocess isolation → sequential in-process.
        # This covers: single files, test fixtures with injected backends,
        # and any explicitly non-subprocess configuration.
        if len(reqs) <= 1 or not use_subprocess:
            outcomes: List[FileParseResult] = []
            for req in reqs:
                out = self._parse_and_enrich(req, registry=reg)
                if raises_on_error and out.status == "error":
                    raise RuntimeError(out.error or "parse failed")
                outcomes.append(out)
            return outcomes

        # Multi-file with subprocess isolation — the production path.
        return self._parse_batch_subprocess(
            reqs,
            parse_config=parse_config,
            workers=workers,
            raises_on_error=raises_on_error,
        )

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
        """Run ``parse_batch`` with per-file process isolation.

        Each file is parsed in a **fully isolated** ``multiprocessing.Process``
        (via ``run_isolated``).

        Scheduling
        ~~~~~~~~~~
        Files are split into two lanes:

        - **Heavy** (Docling formats OR files ≥ 100 MB) — run one at a
          time to prevent concurrent memory exhaustion.
        - **Light** (everything else) — run concurrently up to
          ``max_concurrent_parses``.

        Fault isolation & retry
        ~~~~~~~~~~~~~~~~~~~~~~~
        Every file gets its own process and IPC channel.  If a child
        crashes (OOM, segfault, SIGKILL), only that file's result is an
        error.  **Crashed light files are retried once in isolation**
        (concurrency=1) on the assumption that the crash was caused by
        concurrent memory pressure rather than the file itself being
        unparseable.
        """
        config = parse_config or ParseConfig()
        effective_workers = max(1, min(len(reqs), workers))

        heavy: List[Tuple[int, FileParseRequest]] = []
        light: List[Tuple[int, FileParseRequest]] = []
        for idx, req in enumerate(reqs):
            if classify_file(req) == "heavy":
                heavy.append((idx, req))
            else:
                light.append((idx, req))

        if heavy:
            heavy_names = [os.path.basename(r.logical_path) for _, r in heavy]
            logger.info(
                "Batch scheduler: %d heavy (serialised), %d light "
                "(concurrency=%d): %s",
                len(heavy),
                len(light),
                effective_workers,
                heavy_names,
            )

        results: List[Optional[FileParseResult]] = [None] * len(reqs)

        # --- Light files: concurrent with hard concurrency cap ----------- #
        if light:
            with ThreadPoolExecutor(max_workers=effective_workers) as tp:
                futures = {}
                for idx, req in light:
                    timeout = _adaptive_timeout(req, config)
                    fut = tp.submit(run_isolated, req, parse_config, timeout)
                    futures[fut] = (idx, req)

                for fut in futures:
                    idx, req = futures[fut]
                    results[idx] = fut.result()

        # Retry crashed light files once in isolation.
        retry_items = [
            (idx, req)
            for idx, req in light
            if results[idx] is not None
            and results[idx].status == "error"  # type: ignore[union-attr]
            and _is_subprocess_crash(results[idx])  # type: ignore[arg-type]
        ]
        if retry_items:
            logger.info(
                "Retrying %d crashed light file(s) in isolation: %s",
                len(retry_items),
                [os.path.basename(r.logical_path) for _, r in retry_items],
            )
            for idx, req in retry_items:
                timeout = _adaptive_timeout(req, config)
                results[idx] = run_isolated(req, parse_config, timeout)

        # --- Heavy files: one at a time ---------------------------------- #
        for idx, req in heavy:
            timeout = _adaptive_timeout(req, config)
            results[idx] = run_isolated(req, parse_config, timeout)

        # --- LLM enrichment in parent process ---------------------------- #
        for r in results:
            if r is not None and r.status == "success":
                try:
                    enrich_parse_result(r)
                except Exception:
                    logger.debug(
                        "Enrichment failed for %s",
                        r.logical_path,
                        exc_info=True,
                    )

        # Check for errors after all attempts.
        if raises_on_error:
            for r in results:
                if r is not None and r.status == "error":
                    raise RuntimeError(r.error or "parse failed")

        return [r for r in results if r is not None]  # type: ignore[misc]


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


_CRASH_ERROR_PREFIXES = ("subprocess_crash:", "subprocess_parse_timeout:")


def _is_subprocess_crash(result: FileParseResult) -> bool:
    """True if the error indicates the child process crashed or was killed."""
    err = result.error or ""
    return any(err.startswith(p) for p in _CRASH_ERROR_PREFIXES)
