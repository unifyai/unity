from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

from unity.file_manager.file_parsers.registry import BackendRegistry
from unity.file_manager.file_parsers.settings import FILE_PARSER_SETTINGS
from unity.file_manager.file_parsers.types.backend import BaseFileParserBackend
from unity.file_manager.file_parsers.types.contracts import (
    FileParseRequest,
    FileParseResult,
    FileParseTrace,
    StepStatus,
)
from unity.file_manager.types.config import ParseConfig
from unity.file_manager.file_parsers.types.formats import (
    FileFormat,
    MimeType,
    extension_to_format,
    extension_to_mime,
)
from unity.file_manager.file_parsers.utils.postconditions import (
    enforce_parse_success_invariants,
)

logger = logging.getLogger(__name__)


class FileParser:
    """
    Format-aware file parser facade.

    The FileManager should depend on this class (not on concrete backends).

    Guarantees
    ----------
    `FileParser.parse()` is designed to be *safe* to call from batch pipelines:
    - It **never raises** for backend failures; unexpected backend exceptions are
      caught and converted into `FileParseResult(status="error")`.
    - It enforces key invariants on outputs (trace identity, format/mime defaults,
      and minimal retrieval fields on success).
    - It keeps the parser boundary stable: input is `FileParseRequest`, output is
      `FileParseResult` (no dicts, no legacy models).

    Format awareness
    ---------------
    Backends are selected by `FileFormat`, but the facade also enforces
    format-aware output policy:
    - spreadsheets should not dump huge data into text fields
    - `summary` and `metadata` should be present on success across formats

    Backends are resolved via a config-driven registry so that:
    - swapping a single format backend (e.g. XLSX) is a config change
    - swapping an entire backend set is a config change
    - FileParser remains agnostic of which parsing library is used
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

    def parse(
        self,
        request: FileParseRequest,
        *,
        parse_config: Optional[ParseConfig] = None,
    ) -> FileParseResult:
        """
        Parse a single file.

        Parameters
        ----------
        request:
            Canonical parse request. `logical_path` is the stable external identifier
            (used for contexts/records). `source_local_path` is the actual on-disk
            path to read (may be a temp export).
        parse_config:
            Optional typed parser configuration. This is the canonical way to:
            - override per-format backend implementations (`backend_class_paths_by_format`)
            - control parse-stage concurrency in `parse_batch` (`max_concurrent_parses`)

        Returns
        -------
        FileParseResult
            Always returned. On failure, `status="error"` and `error` is populated.
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
                    f"Backend {backend_name} returned invalid type: {type(out)!r} (expected FileParseResult)",
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
        return out

    def parse_batch(
        self,
        requests: Iterable[FileParseRequest],
        *,
        raises_on_error: bool = False,
        parse_config: Optional[ParseConfig] = None,
    ) -> List[FileParseResult]:
        """
        Parse multiple files with optional bounded concurrency.

        Notes
        -----
        - This method preserves input ordering in its outputs.
        - Concurrency is implemented with a ThreadPool (backends tend to be I/O
          bound and/or call external processes).
        - Concurrency is derived from `parse_config.max_concurrent_parses` and is
          conservatively capped to 8 to avoid resource exhaustion in typical environments.
        - When `raises_on_error=True`, the first error result raises a RuntimeError.
          In ingestion pipelines this should generally remain False.
        """
        reqs = list(requests)
        if not reqs:
            return []

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
        workers = int(requested) if isinstance(requested, int) else 1
        workers = max(1, workers)
        workers = min(workers, 8)

        # Sequential fallback
        if workers <= 1 or len(reqs) <= 1:
            outcomes: List[FileParseResult] = []
            for req in reqs:
                out = self._parse_single(req, registry=reg)
                if raises_on_error and out.status == "error":
                    raise RuntimeError(out.error or "parse failed")
                outcomes.append(out)
            return outcomes

        # Parallel parse (threaded): preserve request order in outputs.
        outcomes: List[FileParseResult] = []
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(self._parse_single, req, registry=reg) for req in reqs]
            for req, fut in zip(reqs, futures):
                try:
                    out = fut.result()
                except Exception as e:
                    # Extremely defensive: FileParser.parse should already never raise.
                    trace = FileParseTrace(
                        logical_path=str(req.logical_path),
                        backend="file_parser",
                        file_format=req.file_format,
                        mime_type=req.mime_type,
                        status=StepStatus.FAILED,
                        source_local_path=req.source_local_path,
                        parsed_local_path=req.source_local_path,
                        warnings=[f"parse_batch_exception: {e}"],
                    )
                    out = FileParseResult(
                        logical_path=str(req.logical_path),
                        status="error",
                        error=str(e),
                        file_format=req.file_format,
                        mime_type=req.mime_type,
                        trace=trace,
                    )
                if raises_on_error and out.status == "error":
                    raise RuntimeError(out.error or "parse failed")
                outcomes.append(out)
        return outcomes
