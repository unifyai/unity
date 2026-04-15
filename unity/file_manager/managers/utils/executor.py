"""Thin orchestration layer for the FileManager ingest pipeline.

All heavy lifting (chunking, parallelism, embedding, retry) is delegated to
``DataManager.ingest()`` via the task functions in ``task_functions.py`` and
the bridge helpers in ``ingest_ops.py``.

This module provides:

- ``process_single_file``: Process one parsed file (file record + content + tables).
- ``run_pipeline``: Process multiple files (sequential or parallel).
- Result aggregation and progress reporting helpers.

The flow per file is straightforward::

    1. Create file record  (must succeed first)
    2. Ingest content      -+
    3. Ingest tables (xN)  -+  independent, run concurrently
    4. Aggregate and return Pydantic model
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
)

from unity.file_manager.file_parsers.types.contracts import FileParseResult
from unity.file_manager.types.ingest import (
    BaseIngestedFile,
    IngestedMinimal,
    ContentRef,
    FileMetrics,
    FileResultType,
)

if TYPE_CHECKING:
    from .progress import ProgressReporter

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Retry helper
# ---------------------------------------------------------------------------


@dataclass
class _StepResult:
    """Outcome of a single pipeline step."""

    success: bool
    value: Any = None
    error: Optional[str] = None
    duration_ms: float = 0.0
    retries: int = 0


def _run_with_retry(
    fn: Callable[..., Any],
    kwargs: Dict[str, Any],
    *,
    max_retries: int = 2,
    retry_delay: float = 1.0,
    label: str = "",
) -> _StepResult:
    """Call *fn* with retries and exponential backoff."""
    last_error = ""
    for attempt in range(1 + max_retries):
        t0 = time.perf_counter()
        try:
            value = fn(**kwargs)
            return _StepResult(
                success=True,
                value=value,
                duration_ms=(time.perf_counter() - t0) * 1000,
                retries=attempt,
            )
        except Exception as exc:
            last_error = str(exc)
            elapsed = (time.perf_counter() - t0) * 1000
            if attempt < max_retries:
                delay = retry_delay * (2**attempt)
                logger.warning(
                    f"[Pipeline] {label} attempt {attempt + 1} failed "
                    f"({elapsed:.0f}ms): {exc} -- retrying in {delay:.1f}s",
                )
                time.sleep(delay)
            else:
                logger.error(
                    f"[Pipeline] {label} failed after {attempt + 1} attempts "
                    f"({elapsed:.0f}ms): {exc}",
                )
                return _StepResult(
                    success=False,
                    error=last_error,
                    duration_ms=elapsed,
                    retries=attempt,
                )
    return _StepResult(success=False, error=last_error)


# ---------------------------------------------------------------------------
# Result aggregation
# ---------------------------------------------------------------------------


def _aggregate_results(
    file_path: str,
    *,
    file_record_result: _StepResult,
    content_result: Optional[_StepResult],
    table_results: List[_StepResult],
    file_start_time: float,
    parse_result: FileParseResult,
) -> Dict[str, Any]:
    """Build a file-level summary dict from individual step outcomes."""
    total_duration_ms = (time.perf_counter() - file_start_time) * 1000

    timing = {
        "file_record_ms": file_record_result.duration_ms,
        "ingest_content_ms": content_result.duration_ms if content_result else 0.0,
        "ingest_table_ms": sum(r.duration_ms for r in table_results),
    }

    counts = {
        "content_ingested": 1 if content_result and content_result.success else 0,
        "tables_ingested": sum(1 for r in table_results if r.success),
    }

    failed_labels: List[str] = []
    if content_result and not content_result.success:
        failed_labels.append("content")
    for i, r in enumerate(table_results):
        if not r.success:
            failed_labels.append(f"table_{i}")

    failures = {
        "ingest_failures": len(failed_labels),
        "failed_task_ids": failed_labels,
    }

    retries_used = (
        file_record_result.retries
        + (content_result.retries if content_result else 0)
        + sum(r.retries for r in table_results)
    )

    status = "error" if failures["ingest_failures"] > 0 else "success"

    return {
        "file_path": file_path,
        "status": status,
        "total_duration_ms": total_duration_ms,
        "timing_breakdown": timing,
        "chunks": counts,
        "failures": failures,
        "retries_used": retries_used,
        "parse_result": parse_result,
    }


# Backward-compatible aliases used by __init__.py exports
build_file_result = _aggregate_results  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Progress reporting
# ---------------------------------------------------------------------------


def report_file_complete(
    reporter: Optional["ProgressReporter"],
    file_path: str,
    file_start_time: float,
    result: Dict[str, Any],
    verbosity: str,
) -> None:
    """Emit the final progress event for a file."""
    if reporter is None:
        return
    try:
        from .progress import create_progress_event

        total_ms = (time.perf_counter() - file_start_time) * 1000
        status = "completed" if result["status"] == "success" else "failed"

        meta = {
            "total_duration_ms": total_ms,
            **result.get("timing_breakdown", {}),
            **result.get("chunks", {}),
            "ingest_failures": result.get("failures", {}).get("ingest_failures", 0),
            "retries_used": result.get("retries_used", 0),
        }

        event = create_progress_event(
            file_path,
            "file_complete",
            status,
            duration_ms=total_ms,
            elapsed_ms=total_ms,
            meta=meta,
            verbosity=verbosity,  # type: ignore[arg-type]
        )
        reporter.report(event)
    except Exception as e:
        logger.debug(f"File complete report failed: {e}")


# ---------------------------------------------------------------------------
# Return model builders (private)
# ---------------------------------------------------------------------------


def _build_return_model(
    file_manager: Any,
    file_path: str,
    parse_result: FileParseResult,
    config: Any,
    return_mode: str,
    adapted: Any,
) -> FileResultType:
    """Build the appropriate Pydantic result based on *return_mode*."""
    if return_mode == "full":
        from unity.file_manager.types.ingest import IngestedFullFile
        from .file_ops import build_compact_ingest_model

        compact = build_compact_ingest_model(
            file_manager,
            file_path=file_path,
            parse_result=parse_result,
            config=config,
        )

        return IngestedFullFile(
            file_path=file_path,
            status=parse_result.status,
            error=parse_result.error,
            file_format=parse_result.file_format,
            mime_type=parse_result.mime_type,
            summary=getattr(parse_result, "summary", "") or "",
            full_text=getattr(parse_result, "full_text", "") or "",
            trace=getattr(parse_result, "trace", None),
            metadata=(
                parse_result.metadata.model_dump(mode="json", exclude_none=True)
                if getattr(parse_result, "metadata", None) is not None
                else None
            ),
            graph=getattr(parse_result, "graph", None),
            tables=list(getattr(parse_result, "tables", []) or []),
            content_rows=list(adapted.content_rows or []),
            content_ref=getattr(compact, "content_ref", None),
            tables_ref=list(getattr(compact, "tables_ref", []) or []),
            metrics=getattr(compact, "metrics", None),
        )

    if return_mode == "none":
        total_records = len(list(adapted.content_rows or []))
        return IngestedMinimal(
            file_path=file_path,
            status=parse_result.status,
            error=parse_result.error,
            total_records=total_records,
            file_format=(
                str(parse_result.file_format) if parse_result.file_format else None
            ),
        )

    from .file_ops import build_compact_ingest_model

    return build_compact_ingest_model(
        file_manager,
        file_path=file_path,
        parse_result=parse_result,
        config=config,
    )


def _build_error_model(
    file_path: str,
    error: str,
    elapsed_ms: float,
    return_mode: str,
) -> FileResultType:
    """Build an error Pydantic result based on *return_mode*."""
    if return_mode == "full":
        from unity.file_manager.types.ingest import IngestedFullFile

        return IngestedFullFile(
            file_path=file_path,
            status="error",
            error=error,
            content_rows=[],
            tables=[],
        )

    if return_mode == "none":
        return IngestedMinimal(
            file_path=file_path,
            status="error",
            error=error,
            total_records=0,
            file_format=None,
        )

    return BaseIngestedFile(
        file_path=file_path,
        status="error",
        error=error,
        content_ref=ContentRef(context="", record_count=0, text_chars=0),
        metrics=FileMetrics(processing_time=elapsed_ms / 1000.0),
    )


# ---------------------------------------------------------------------------
# Single file processing
# ---------------------------------------------------------------------------


def process_single_file(
    file_manager,
    *,
    parse_result,
    file_path: str,
    config,
    reporter=None,
    enable_progress: bool = False,
    verbosity: str = "low",
):
    """Process a single parsed document through the ingest pipeline.

    Sequentially creates a file record, then ingests content and tables
    (concurrently where possible), delegating all heavy lifting to
    ``DataManager.ingest()``.

    Parameters
    ----------
    file_manager : FileManager
        The file manager instance.
    parse_result : FileParseResult
        The FileParseResult from the file parser.
    file_path : str
        The original file path.
    config : FilePipelineConfig
        Pipeline configuration.
    reporter : ProgressReporter | None
        Optional progress reporter.
    enable_progress : bool
        Whether to emit progress events.
    verbosity : str
        Verbosity level: "low", "medium", "high".

    Returns
    -------
    FileResultType
        Pydantic model whose shape depends on ``config.output.return_mode``.
    """
    import time as _time
    import traceback as _tb
    from concurrent.futures import ThreadPoolExecutor as _TPE, as_completed as _asc

    file_start_time = _time.perf_counter()
    return_mode = getattr(getattr(config, "output", None), "return_mode", "compact")
    max_retries = getattr(config.retry, "max_retries", 2)
    retry_delay = getattr(config.retry, "retry_delay_seconds", 1.0)

    try:
        from unity.file_manager.parse_adapter import adapt_parse_result_for_file_manager
        from .task_functions import (
            execute_create_file_record,
            execute_ingest_content,
            execute_ingest_table,
        )

        adapted = adapt_parse_result_for_file_manager(parse_result, config=config)
        content_rows = list(adapted.content_rows or [])

        # 1. File record (sequential -- everything else depends on this)
        fr_result = _run_with_retry(
            execute_create_file_record,
            kwargs={
                "file_manager": file_manager,
                "file_path": file_path,
                "parse_result": parse_result,
                "config": config,
                "document_summary": adapted.document_summary,
                "total_records": len(content_rows),
            },
            max_retries=max_retries,
            retry_delay=retry_delay,
            label=f"file_record({file_path})",
        )
        if not fr_result.success:
            raise RuntimeError(f"File record creation failed: {fr_result.error}")

        # 2. Build work items for content + tables
        content_result = None
        table_results = []

        tables = list(getattr(parse_result, "tables", []) or [])
        do_tables = bool(config.ingest.table_ingest and tables)

        work_items = []

        if content_rows:
            work_items.append(
                {
                    "kind": "content",
                    "fn": execute_ingest_content,
                    "kwargs": {
                        "file_manager": file_manager,
                        "file_path": file_path,
                        "content_rows": content_rows,
                        "config": config,
                    },
                    "label": f"content({file_path})",
                },
            )

        if do_tables:
            for i, tbl in enumerate(tables, start=1):
                table_label = str(getattr(tbl, "label", None) or f"{i:02d}")
                columns = list(getattr(tbl, "columns", []) or [])
                rows = list(getattr(tbl, "rows", []) or [])
                table_input = getattr(adapted.bundle, "table_inputs", {}).get(
                    str(getattr(tbl, "table_id", "")),
                )
                if not columns and rows and isinstance(rows[0], dict):
                    try:
                        columns = [str(k) for k in rows[0].keys()]
                    except Exception:
                        columns = []
                row_count = getattr(tbl, "num_rows", None)
                if row_count is None:
                    row_count = len(rows)
                if table_input is None and not rows:
                    continue
                work_items.append(
                    {
                        "kind": "table",
                        "fn": execute_ingest_table,
                        "kwargs": {
                            "file_manager": file_manager,
                            "file_path": file_path,
                            "table_label": table_label,
                            "table_rows": rows,
                            "table_input": table_input,
                            "columns": columns,
                            "config": config,
                        },
                        "label": f"table({file_path}/{table_label})",
                        "row_count": row_count,
                    },
                )

        # 3. Execute work items (concurrent when multiple)
        if len(work_items) <= 1:
            for item in work_items:
                res = _run_with_retry(
                    item["fn"],
                    item["kwargs"],
                    max_retries=max_retries,
                    retry_delay=retry_delay,
                    label=item["label"],
                )
                if item["kind"] == "content":
                    content_result = res
                else:
                    table_results.append(res)
        elif work_items:
            max_workers = getattr(config.execution, "max_embed_workers", 8)
            with _TPE(max_workers=min(len(work_items), max_workers)) as pool:
                futures = {
                    pool.submit(
                        _run_with_retry,
                        item["fn"],
                        item["kwargs"],
                        max_retries=max_retries,
                        retry_delay=retry_delay,
                        label=item["label"],
                    ): item
                    for item in work_items
                }
                for future in _asc(futures):
                    item = futures[future]
                    try:
                        res = future.result()
                    except Exception as exc:
                        res = _StepResult(success=False, error=str(exc))
                    if item["kind"] == "content":
                        content_result = res
                    else:
                        table_results.append(res)

        # 4. Aggregate
        result_dict = _aggregate_results(
            file_path,
            file_record_result=fr_result,
            content_result=content_result,
            table_results=table_results,
            file_start_time=file_start_time,
            parse_result=parse_result,
        )

        if enable_progress and reporter:
            report_file_complete(
                reporter,
                file_path,
                file_start_time,
                result_dict,
                verbosity,
            )

        # 5. Build return model
        return _build_return_model(
            file_manager,
            file_path,
            parse_result,
            config,
            return_mode,
            adapted,
        )

    except Exception as e:
        tb_str = _tb.format_exc()
        logger.error(f"Fatal error processing {file_path}: {e}\n{tb_str}")
        elapsed_ms = (_time.perf_counter() - file_start_time) * 1000

        if enable_progress and reporter:
            from .progress import create_progress_event

            event = create_progress_event(
                file_path,
                "file_complete",
                "failed",
                duration_ms=elapsed_ms,
                elapsed_ms=elapsed_ms,
                error=str(e),
                traceback_str=tb_str,
                verbosity=verbosity,
            )
            reporter.report(event)

        return _build_error_model(file_path, str(e), elapsed_ms, return_mode)


# ---------------------------------------------------------------------------
# Multi-file pipeline
# ---------------------------------------------------------------------------


def run_pipeline(
    file_manager,
    *,
    parse_results,
    file_paths,
    config,
    reporter=None,
    enable_progress: bool = False,
    verbosity: str = "low",
):
    """Run the ingest pipeline for multiple files.

    Each file is processed through ``process_single_file``.  Files can be
    processed in parallel or sequentially based on *config*.

    Parameters
    ----------
    file_manager : FileManager
        The file manager instance.
    parse_results : list[FileParseResult]
        Parsed file results.
    file_paths : list[str]
        Corresponding file paths.
    config : FilePipelineConfig
        Pipeline configuration.
    reporter : ProgressReporter | None
        Optional progress reporter.
    enable_progress : bool
        Whether to emit progress events.
    verbosity : str
        Verbosity level: "low", "medium", "high".

    Returns
    -------
    IngestPipelineResult
    """
    import time as _time
    from concurrent.futures import ThreadPoolExecutor as _TPE, as_completed as _asc

    pipeline_start = _time.perf_counter()

    if not parse_results or not file_paths:
        from unity.file_manager.types.ingest import IngestPipelineResult as _IPR

        return _IPR()

    return_mode = getattr(getattr(config, "output", None), "return_mode", "compact")
    results = {}

    parallel = getattr(getattr(config, "execution", None), "parallel_files", False)
    max_workers = getattr(getattr(config, "execution", None), "max_file_workers", 4)

    if enable_progress:
        logger.info(
            f"Processing {len(parse_results)} files "
            f"({'parallel' if parallel else 'sequential'}, "
            f"max_workers={max_workers})",
        )

    if not parallel or len(parse_results) == 1:
        for idx, (pr, path) in enumerate(zip(parse_results, file_paths)):
            if enable_progress:
                logger.info(f"Processing file {idx + 1}/{len(parse_results)}: {path}")
            results[path] = process_single_file(
                file_manager,
                parse_result=pr,
                file_path=path,
                config=config,
                reporter=reporter,
                enable_progress=enable_progress,
                verbosity=verbosity,
            )
            # Release the heavy FileParseResult for this file so its memory
            # (graph, tables, full_text) can be reclaimed before the next file
            # starts.  In "full" return_mode the result model already copied
            # what it needs; all other modes are lightweight references.
            try:
                parse_results[idx] = None  # type: ignore[index]
            except (TypeError, IndexError):
                pass
    else:
        with _TPE(max_workers=min(len(parse_results), max_workers)) as pool:
            futures = {
                pool.submit(
                    process_single_file,
                    file_manager,
                    parse_result=pr,
                    file_path=path,
                    config=config,
                    reporter=reporter,
                    enable_progress=enable_progress,
                    verbosity=verbosity,
                ): path
                for pr, path in zip(parse_results, file_paths)
            }
            for future in _asc(futures):
                path = futures[future]
                try:
                    results[path] = future.result()
                except Exception as e:
                    results[path] = _build_error_model(
                        path,
                        str(e),
                        0.0,
                        return_mode,
                    )

    if enable_progress and reporter:
        reporter.flush()

    total_ms = (_time.perf_counter() - pipeline_start) * 1000
    from unity.file_manager.types.ingest import IngestPipelineResult as _IPR

    return _IPR.from_results(results, total_duration_ms=total_ms)
