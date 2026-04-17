"""Thin orchestration layer for the FileManager ingest pipeline.

All heavy lifting (chunking, parallelism, embedding, retry) is delegated to
``DataManager.ingest()`` via the task functions in ``task_functions.py`` and
the bridge helpers in ``ingest_ops.py``.

This module provides:

- ``fm_process_file``: FM-specific per-file callback for the shared pipeline.
- ``run_pipeline``: Multi-file dispatch using ``PipelineInstrumentation``.
- Result aggregation and progress reporting helpers.

The flow per file is straightforward::

    1. Adapt parse result for FM layout (content rows + table handles)
    2. Create file record  (must succeed first)
    3. Ingest content      -+
    4. Ingest tables (xN)  -+  via ingest_artifacts() (concurrent)
    5. Aggregate and return Pydantic model
"""

from __future__ import annotations

import logging
import time
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Optional,
)

from unity.common.pipeline import (
    ArtifactWorkItem,
    PipelineInstrumentation,
    ingest_artifacts,
    run_with_retry,
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
# Parse-cost metric extraction (keeps parser imports out of cost_ledger.py)
# ---------------------------------------------------------------------------


def _extract_parse_cost_metrics(
    parse_result,
    file_path: str,
    parse_config,
) -> dict:
    """Extract pre-computed metrics from a FileParseResult for cost estimation.

    Returns kwargs suitable for ``build_parse_cost_line_items``.
    """
    from pathlib import Path as _Path

    from unity.file_manager.file_parsers.types.contracts import FileParseRequest
    from unity.file_manager.file_parsers.utils.memory_scheduler import (
        estimate_peak_memory_bytes,
    )

    trace = getattr(parse_result, "trace", None)
    duration_ms = float(getattr(trace, "duration_ms", 0.0) or 0.0)
    trace_status = getattr(getattr(trace, "status", None), "value", None)
    backend = getattr(trace, "backend", None)

    estimated_peak_bytes = 0
    source_candidates = [
        getattr(trace, "source_local_path", None),
        getattr(trace, "parsed_local_path", None),
        file_path,
    ]
    for candidate in source_candidates:
        if not candidate:
            continue
        try:
            path = _Path(str(candidate)).expanduser()
        except Exception:
            continue
        if not path.exists():
            continue
        request = FileParseRequest(
            logical_path=file_path,
            source_local_path=str(path),
        )
        estimated_peak_bytes = estimate_peak_memory_bytes(request)
        break

    llm_calls = 0
    for step in list(getattr(trace, "steps", []) or []):
        counters = getattr(step, "counters", {}) or {}
        for key in ("llm_calls", "summary_calls", "metadata_calls"):
            value = counters.get(key)
            if value:
                llm_calls += int(value)

    return {
        "parse_duration_seconds": duration_ms / 1000.0,
        "estimated_peak_memory_bytes": estimated_peak_bytes,
        "llm_enrichment_calls": llm_calls,
        "parse_backend": backend,
        "trace_status": trace_status,
    }


# ---------------------------------------------------------------------------
# Progress reporting
# ---------------------------------------------------------------------------


def report_file_complete(
    reporter: Optional["ProgressReporter"],
    file_path: str,
    file_start_time: float,
    result: Dict[str, Any],
    verbosity: str,
    *,
    run_id: str | None = None,
    file_id: int | None = None,
    storage_id: str | None = None,
    trace_id: str | None = None,
) -> None:
    """Emit the final progress event for a file."""
    if reporter is None:
        return
    try:
        from .progress import create_progress_event

        total_ms = (time.perf_counter() - file_start_time) * 1000
        status = "completed" if result["status"] == "success" else "failed"
        parse_result = result.get("parse_result")
        parse_trace = getattr(parse_result, "trace", None)

        meta = {
            "total_duration_ms": total_ms,
            **result.get("timing_breakdown", {}),
            **result.get("chunks", {}),
            "ingest_failures": result.get("failures", {}).get("ingest_failures", 0),
            "retries_used": result.get("retries_used", 0),
            "parse_backend": getattr(parse_trace, "backend", None),
            "parse_trace_status": (
                getattr(getattr(parse_trace, "status", None), "value", None)
                if parse_trace is not None
                else None
            ),
            "file_format": str(getattr(parse_result, "file_format", "") or ""),
            "mime_type": str(getattr(parse_result, "mime_type", "") or ""),
        }

        event = create_progress_event(
            file_path,
            "file_complete",
            status,
            run_id=run_id,
            file_id=file_id,
            storage_id=storage_id,
            trace_id=trace_id,
            duration_ms=total_ms,
            elapsed_ms=total_ms,
            meta=meta,
            verbosity=verbosity,  # type: ignore[arg-type]
        )
        reporter.report(event)
    except Exception as e:
        logger.debug(f"File complete report failed: {e}")


def _report_stage_progress(
    reporter: Optional["ProgressReporter"],
    *,
    run_id: str | None,
    file_path: str,
    stage_name: str,
    success: bool,
    duration_ms: float,
    retries: int,
    error: str | None,
    failure_kind: str | None,
    file_start_time: float,
    verbosity: str,
    stage_id: str | None = None,
    file_id: int | None = None,
    storage_id: str | None = None,
    table_id: str | None = None,
    trace_id: str | None = None,
    meta: Optional[Dict[str, Any]] = None,
) -> None:
    if reporter is None:
        return
    try:
        from .progress import create_progress_event

        event = create_progress_event(
            file_path,
            stage_name,
            "completed" if success else "failed",
            run_id=run_id,
            stage_id=stage_id,
            file_id=file_id,
            storage_id=storage_id,
            table_id=table_id,
            trace_id=trace_id,
            duration_ms=duration_ms,
            elapsed_ms=(time.perf_counter() - file_start_time) * 1000,
            error=error if not success else None,
            meta={
                **dict(meta or {}),
                "retries_used": retries,
                **({"failure_kind": failure_kind} if failure_kind is not None else {}),
            },
            verbosity=verbosity,  # type: ignore[arg-type]
        )
        reporter.report(event)
    except Exception as exc:
        logger.debug("Stage progress report failed: %s", exc)


# ---------------------------------------------------------------------------
# Result aggregation
# ---------------------------------------------------------------------------


def _aggregate_results(
    file_path: str,
    *,
    file_record_success: bool,
    file_record_duration_ms: float,
    file_record_retries: int,
    content_result: Optional[Any],
    table_results: List[Any],
    file_start_time: float,
    parse_result: FileParseResult,
) -> Dict[str, Any]:
    """Build a file-level summary dict from individual step outcomes."""
    total_duration_ms = (time.perf_counter() - file_start_time) * 1000

    timing = {
        "file_record_ms": file_record_duration_ms,
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
        file_record_retries
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


build_file_result = _aggregate_results  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Return model builders (private)
# ---------------------------------------------------------------------------


def _build_return_model(
    file_manager: Any,
    file_path: str,
    parse_result: FileParseResult,
    config: Any,
    return_mode: str,
    ingest_payload: Any,
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
            content_rows=list(ingest_payload.content_rows or []),
            content_ref=getattr(compact, "content_ref", None),
            tables_ref=list(getattr(compact, "tables_ref", []) or []),
            metrics=getattr(compact, "metrics", None),
        )

    if return_mode == "none":
        total_records = len(list(ingest_payload.content_rows or []))
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
# FM-specific per-file callback
# ---------------------------------------------------------------------------


def fm_process_file(
    file_manager,
    *,
    parse_result,
    file_path: str,
    config,
    instrumentation: PipelineInstrumentation,
    reporter=None,
    enable_progress: bool = False,
    verbosity: str = "low",
):
    """Process a single parsed document through the FM ingest pipeline.

    Uses ``ingest_artifacts()`` for parallel content + table dispatch.
    """
    import traceback as _tb

    file_start_time = time.perf_counter()
    return_mode = getattr(getattr(config, "output", None), "return_mode", "compact")

    try:
        from unity.file_manager.parse_adapter import adapt_parse_result_for_file_manager
        from .task_functions import (
            execute_create_file_record,
            execute_ingest_content,
            execute_ingest_table,
        )

        ingest_payload = adapt_parse_result_for_file_manager(
            parse_result,
            config=config,
        )
        content_rows = list(ingest_payload.content_rows or [])
        parse_trace = getattr(parse_result, "trace", None)
        trace_id = str(getattr(parse_trace, "trace_id", "") or "") or None
        parse_backend = getattr(parse_trace, "backend", None)

        # 1. File record (sequential -- everything else depends on this)
        file_record_stage_id = instrumentation.make_stage_id(
            file_path=file_path,
            stage_name="file_record",
        )
        fr_outcome = run_with_retry(
            execute_create_file_record,
            kwargs={
                "file_manager": file_manager,
                "file_path": file_path,
                "parse_result": parse_result,
                "config": config,
                "document_summary": ingest_payload.document_summary,
                "total_records": len(content_rows),
            },
            retry_config=config.retry,
            label=f"file_record({file_path})",
        )
        file_record_value = (
            fr_outcome.value if isinstance(fr_outcome.value, dict) else {}
        )
        file_id = file_record_value.get("file_id")
        storage_id = file_record_value.get("storage_id")

        instrumentation.record_stage(
            file_path=file_path,
            stage_name="file_record",
            status="success" if fr_outcome.success else "error",
            duration_ms=fr_outcome.duration_ms,
            retries_used=fr_outcome.retries,
            error=fr_outcome.error,
            stage_id=file_record_stage_id,
            file_id=file_id,
            storage_id=storage_id,
            meta={
                "content_row_count": len(content_rows),
                "parse_backend": parse_backend,
            },
        )
        _report_stage_progress(
            reporter if enable_progress else None,
            run_id=instrumentation.run_id,
            file_path=file_path,
            stage_name="file_record",
            success=fr_outcome.success,
            duration_ms=fr_outcome.duration_ms,
            retries=fr_outcome.retries,
            error=fr_outcome.error,
            failure_kind=fr_outcome.failure_kind,
            file_start_time=file_start_time,
            verbosity=verbosity,
            stage_id=file_record_stage_id,
            file_id=file_id,
            storage_id=storage_id,
            trace_id=trace_id,
            meta={
                "content_row_count": len(content_rows),
                "parse_backend": parse_backend,
            },
        )
        if not fr_outcome.success:
            raise RuntimeError(f"File record creation failed: {fr_outcome.error}")

        instrumentation.add_transport_costs(
            file_path=file_path,
            file_id=file_id,
            storage_id=storage_id,
            bundle=ingest_payload.bundle,
            retention_days=getattr(config.cost, "artifact_retention_days", 30),
        )

        # 2. Build ArtifactWorkItems for content + tables
        work_items: List[ArtifactWorkItem] = []
        tables = list(getattr(parse_result, "tables", []) or [])
        do_tables = bool(config.ingest.table_ingest and tables)

        if content_rows:
            work_items.append(
                ArtifactWorkItem(
                    kind="content",
                    label="content",
                    stage_name="ingest_content",
                    payload={
                        "file_manager": file_manager,
                        "file_path": file_path,
                        "content_rows": content_rows,
                        "config": config,
                    },
                    row_count=len(content_rows),
                    stage_id=instrumentation.make_stage_id(
                        file_path=file_path,
                        stage_name="ingest_content",
                    ),
                    meta={
                        "row_count": len(content_rows),
                        "context": "content",
                        "file_id": file_id,
                        "storage_id": storage_id,
                    },
                ),
            )

        if do_tables:
            for i, tbl in enumerate(tables, start=1):
                table_label = str(getattr(tbl, "label", None) or f"{i:02d}")
                columns = list(getattr(tbl, "columns", []) or [])
                rows = list(getattr(tbl, "rows", []) or [])
                table_input = getattr(ingest_payload.bundle, "table_inputs", {}).get(
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
                tbl_id = str(getattr(tbl, "table_id", "") or "") or None
                work_items.append(
                    ArtifactWorkItem(
                        kind="table",
                        label=table_label,
                        stage_name="ingest_table",
                        payload={
                            "file_manager": file_manager,
                            "file_path": file_path,
                            "table_label": table_label,
                            "table_rows": rows,
                            "table_input": table_input,
                            "columns": columns,
                            "config": config,
                        },
                        columns=columns,
                        row_count=row_count,
                        table_id=tbl_id,
                        stage_id=instrumentation.make_stage_id(
                            file_path=file_path,
                            stage_name="ingest_table",
                            discriminator=str(
                                getattr(tbl, "table_id", "") or table_label,
                            ),
                        ),
                        meta={
                            "row_count": row_count,
                            "table_label": table_label,
                            "column_count": len(columns),
                            "source_handle_type": (
                                type(table_input).__name__
                                if table_input is not None
                                else "InlineRowsHandle"
                            ),
                            "file_id": file_id,
                            "storage_id": storage_id,
                        },
                    ),
                )

        # 3. FM-specific ingest dispatcher
        def _fm_ingest_fn(item: ArtifactWorkItem) -> dict:
            if item.kind == "content":
                return execute_ingest_content(**item.payload)
            return execute_ingest_table(**item.payload)

        max_workers = getattr(config.execution, "max_embed_workers", 8)
        artifact_results = ingest_artifacts(
            work_items=work_items,
            ingest_fn=_fm_ingest_fn,
            instrumentation=instrumentation,
            source_path=file_path,
            max_workers=max_workers,
            retry_config=config.retry,
        )

        # 4. Report progress for each artifact
        if enable_progress and reporter:
            item_by_label = {item.label: item for item in work_items}
            for ar in artifact_results:
                src_item = item_by_label.get(ar.label)
                _report_stage_progress(
                    reporter,
                    run_id=instrumentation.run_id,
                    file_path=file_path,
                    stage_name=f"ingest_{ar.kind}",
                    success=ar.success,
                    duration_ms=ar.duration_ms,
                    retries=ar.retries,
                    error=ar.error,
                    failure_kind=ar.failure_kind,
                    file_start_time=file_start_time,
                    verbosity=verbosity,
                    stage_id=src_item.stage_id if src_item else None,
                    file_id=file_id,
                    storage_id=storage_id,
                    table_id=src_item.table_id if src_item else None,
                    trace_id=trace_id,
                    meta={
                        "label": ar.label,
                        **(src_item.meta if src_item else {}),
                    },
                )

        instrumentation.add_observability_costs(
            progress_event_count=(
                1 + len(work_items) + 1 if enable_progress and reporter else 0
            ),
            file_path=file_path,
            file_id=file_id,
            storage_id=storage_id,
        )

        # 5. Aggregate
        content_ar = next((r for r in artifact_results if r.kind == "content"), None)
        table_ars = [r for r in artifact_results if r.kind == "table"]

        result_dict = _aggregate_results(
            file_path,
            file_record_success=fr_outcome.success,
            file_record_duration_ms=fr_outcome.duration_ms,
            file_record_retries=fr_outcome.retries,
            content_result=content_ar,
            table_results=table_ars,
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
                run_id=instrumentation.run_id,
                file_id=file_id,
                storage_id=storage_id,
                trace_id=trace_id,
            )

        instrumentation.record_file(
            file_path=file_path,
            status=result_dict["status"],
            total_duration_ms=result_dict["total_duration_ms"],
            retries_used=result_dict["retries_used"],
            file_id=file_id,
            storage_id=storage_id,
            meta={
                "ingest_failures": result_dict["failures"]["ingest_failures"],
                **result_dict["timing_breakdown"],
                **result_dict["chunks"],
                "parse_backend": parse_backend,
            },
        )

        # 6. Build return model
        return _build_return_model(
            file_manager,
            file_path,
            parse_result,
            config,
            return_mode,
            ingest_payload,
        )

    except Exception as e:
        import traceback as _tb

        tb_str = _tb.format_exc()
        logger.error(f"Fatal error processing {file_path}: {e}\n{tb_str}")
        elapsed_ms = (time.perf_counter() - file_start_time) * 1000

        if enable_progress and reporter:
            from .progress import create_progress_event

            event = create_progress_event(
                file_path,
                "file_complete",
                "failed",
                run_id=instrumentation.run_id,
                duration_ms=elapsed_ms,
                elapsed_ms=elapsed_ms,
                error=str(e),
                traceback_str=tb_str,
                verbosity=verbosity,
            )
            reporter.report(event)

        instrumentation.record_file(
            file_path=file_path,
            status="error",
            total_duration_ms=elapsed_ms,
            retries_used=0,
            meta={"fatal_error": str(e)},
        )

        return _build_error_model(file_path, str(e), elapsed_ms, return_mode)


# Backward compat alias
process_single_file = fm_process_file


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
    all_parse_results=None,
    run_id: str | None = None,
    enable_progress: bool = False,
    verbosity: str = "low",
):
    """Run the ingest pipeline for multiple files.

    Uses ``PipelineInstrumentation`` for all observability wiring and delegates
    per-file processing to ``fm_process_file``.

    Parameters
    ----------
    file_manager : FileManager
        The file manager instance.
    parse_results : list[FileParseResult]
        Parsed file results (successful parses only).
    file_paths : list[str]
        Corresponding file paths.
    config : FilePipelineConfig
        Pipeline configuration.
    reporter : ProgressReporter | None
        Optional progress reporter.
    all_parse_results : list | None
        All parse results (including failures) for cost accounting.
    run_id : str | None
        Pipeline run ID.
    enable_progress : bool
        Whether to emit progress events.
    verbosity : str
        Verbosity level: "low", "medium", "high".

    Returns
    -------
    IngestPipelineResult
    """
    from uuid import uuid4
    from concurrent.futures import ThreadPoolExecutor as _TPE, as_completed as _asc

    pipeline_start = time.perf_counter()

    if not parse_results or not file_paths:
        from unity.file_manager.types.ingest import IngestPipelineResult as _IPR

        return _IPR()

    return_mode = getattr(getattr(config, "output", None), "return_mode", "compact")
    parallel = getattr(getattr(config, "execution", None), "parallel_files", False)
    max_workers = getattr(getattr(config, "execution", None), "max_file_workers", 4)
    run_id = run_id or uuid4().hex

    instrumentation = PipelineInstrumentation.from_config(
        config,
        run_id=run_id,
        parallel_files=parallel,
        file_count=len(parse_results),
        meta={"max_file_workers": max_workers},
    )

    with instrumentation:
        # Record parse costs for all files (including failures)
        if instrumentation.has_cost_tracking:
            for pr in list(all_parse_results or parse_results):
                logical_path = str(getattr(pr, "logical_path", "") or "")
                instrumentation.add_parse_costs(
                    file_path=logical_path,
                    **_extract_parse_cost_metrics(pr, logical_path, config.parse),
                )

        if enable_progress:
            logger.info(
                f"Processing {len(parse_results)} files "
                f"({'parallel' if parallel else 'sequential'}, "
                f"max_workers={max_workers})",
            )

        results = {}

        if not parallel or len(parse_results) == 1:
            for idx, (pr, path) in enumerate(zip(parse_results, file_paths)):
                if enable_progress:
                    logger.info(
                        f"Processing file {idx + 1}/{len(parse_results)}: {path}",
                    )
                results[path] = fm_process_file(
                    file_manager,
                    parse_result=pr,
                    file_path=path,
                    config=config,
                    instrumentation=instrumentation,
                    reporter=reporter,
                    enable_progress=enable_progress,
                    verbosity=verbosity,
                )
                try:
                    parse_results[idx] = None  # type: ignore[index]
                except (TypeError, IndexError):
                    pass
        else:
            with _TPE(max_workers=min(len(parse_results), max_workers)) as pool:
                futures = {
                    pool.submit(
                        fm_process_file,
                        file_manager,
                        parse_result=pr,
                        file_path=path,
                        config=config,
                        instrumentation=instrumentation,
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
                        instrumentation.record_file(
                            file_path=path,
                            status="error",
                            total_duration_ms=0.0,
                            retries_used=0,
                            meta={"fatal_error": str(e)},
                        )

        if enable_progress and reporter:
            reporter.flush()

        # Final run-level observability costs
        instrumentation.add_observability_costs(
            progress_event_count=(
                2 * len(list(all_parse_results or parse_results))
                if enable_progress and reporter
                else 0
            ),
        )

        # Update file count for the completed manifest
        success_count = sum(
            1
            for value in results.values()
            if getattr(value, "status", "error") == "success"
        )
        instrumentation._file_count = len(results)
        instrumentation._meta["success_count"] = success_count
        instrumentation._meta["failure_count"] = len(results) - success_count

    total_ms = (time.perf_counter() - pipeline_start) * 1000

    from unity.file_manager.types.ingest import IngestPipelineResult as _IPR

    return _IPR.from_results(results, total_duration_ms=total_ms)
