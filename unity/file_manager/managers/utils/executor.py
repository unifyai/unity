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
from uuid import NAMESPACE_URL, uuid5
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
        estimated_peak_bytes = estimate_peak_memory_bytes(request, config=parse_config)
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
# Retry helper
# ---------------------------------------------------------------------------


@dataclass
class _IngestWorkItem:
    """Typed descriptor for one content or table ingest unit of work."""

    kind: str
    fn: Callable[..., Any]
    kwargs: Dict[str, Any]
    label: str
    stage_name: str
    stage_id: Optional[str] = None
    table_id: Optional[str] = None
    meta: Optional[Dict[str, Any]] = None


@dataclass
class PipelineStepOutcome:
    """Outcome of a single pipeline step."""

    success: bool
    value: Any = None
    error: Optional[str] = None
    failure_kind: Optional[str] = None
    duration_ms: float = 0.0
    retries: int = 0


def _run_with_retry(
    fn: Callable[..., Any],
    kwargs: Dict[str, Any],
    *,
    retry_config=None,
    label: str = "",
) -> PipelineStepOutcome:
    """Call *fn* with typed retry policy, backoff, jitter, and deadline."""
    from unity.file_manager.pipeline import ResilientRequestPolicy

    policy = ResilientRequestPolicy.from_config(retry_config)
    last_error = ""
    started_at = time.perf_counter()
    for attempt in range(policy.max_retries + 1):
        t0 = time.perf_counter()
        try:
            value = fn(**kwargs)
            return PipelineStepOutcome(
                success=True,
                value=value,
                duration_ms=(time.perf_counter() - t0) * 1000,
                retries=attempt,
            )
        except Exception as exc:
            last_error = str(exc)
            elapsed = (time.perf_counter() - t0) * 1000
            decision = policy.check_retry(
                exc,
                attempt_index=attempt,
                started_at=started_at,
            )
            if decision.should_retry:
                delay = policy.compute_delay(attempt_index=attempt)
                logger.warning(
                    f"[Pipeline] {label} attempt {attempt + 1} failed "
                    f"({elapsed:.0f}ms): {exc} -- retrying in {delay:.1f}s",
                )
                if delay > 0:
                    time.sleep(delay)
            else:
                logger.error(
                    f"[Pipeline] {label} failed after {attempt + 1} attempts "
                    f"({elapsed:.0f}ms): {exc}",
                )
                return PipelineStepOutcome(
                    success=False,
                    error=last_error,
                    failure_kind=decision.failure_kind,
                    duration_ms=elapsed,
                    retries=attempt,
                )
    return PipelineStepOutcome(success=False, error=last_error)


# ---------------------------------------------------------------------------
# Result aggregation
# ---------------------------------------------------------------------------


def _aggregate_results(
    file_path: str,
    *,
    file_record_result: PipelineStepOutcome,
    content_result: Optional[PipelineStepOutcome],
    table_results: List[PipelineStepOutcome],
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


def _make_stage_id(
    *,
    run_id: str | None,
    file_path: str,
    stage_name: str,
    discriminator: str | None = None,
) -> str | None:
    if not run_id:
        return None
    raw = "::".join(
        part
        for part in (run_id, file_path, stage_name, discriminator or "")
        if part is not None
    )
    return uuid5(NAMESPACE_URL, raw).hex


def _report_stage_progress(
    reporter: Optional["ProgressReporter"],
    *,
    run_id: str | None,
    file_path: str,
    stage_name: str,
    result: PipelineStepOutcome,
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
            "completed" if result.success else "failed",
            run_id=run_id,
            stage_id=stage_id,
            file_id=file_id,
            storage_id=storage_id,
            table_id=table_id,
            trace_id=trace_id,
            duration_ms=result.duration_ms,
            elapsed_ms=(time.perf_counter() - file_start_time) * 1000,
            error=result.error if not result.success else None,
            meta={
                **dict(meta or {}),
                "retries_used": result.retries,
                **(
                    {"failure_kind": result.failure_kind}
                    if result.failure_kind is not None
                    else {}
                ),
            },
            verbosity=verbosity,  # type: ignore[arg-type]
        )
        reporter.report(event)
    except Exception as exc:
        logger.debug("Stage progress report failed: %s", exc)


def _record_stage_manifest(
    ledger,
    *,
    run_id: str | None,
    file_path: str,
    stage_name: str,
    result: PipelineStepOutcome,
    stage_id: str | None = None,
    file_id: int | None = None,
    storage_id: str | None = None,
    table_id: str | None = None,
    meta: Optional[Dict[str, Any]] = None,
) -> None:
    if ledger is None or not run_id:
        return
    try:
        from unity.file_manager.pipeline import PipelineStageManifest

        ledger.write(
            PipelineStageManifest(
                run_id=run_id,
                stage_id=stage_id,
                file_path=file_path,
                file_id=file_id,
                storage_id=storage_id,
                table_id=table_id,
                stage_name=stage_name,
                status="success" if result.success else "error",
                duration_ms=result.duration_ms,
                retries_used=result.retries,
                error=result.error,
                meta={
                    **dict(meta or {}),
                    **(
                        {"failure_kind": result.failure_kind}
                        if result.failure_kind is not None
                        else {}
                    ),
                },
            ),
        )
    except Exception as exc:
        logger.debug("Stage ledger write failed: %s", exc)


def _record_file_manifest(
    ledger,
    *,
    run_id: str | None,
    file_path: str,
    status: str,
    total_duration_ms: float,
    retries_used: int,
    file_id: int | None = None,
    storage_id: str | None = None,
    meta: Optional[Dict[str, Any]] = None,
) -> None:
    if ledger is None or not run_id:
        return
    try:
        from unity.file_manager.pipeline import PipelineFileManifest

        ledger.write(
            PipelineFileManifest(
                run_id=run_id,
                file_path=file_path,
                file_id=file_id,
                storage_id=storage_id,
                status="success" if status == "success" else "error",
                total_duration_ms=total_duration_ms,
                retries_used=retries_used,
                meta=dict(meta or {}),
            ),
        )
    except Exception as exc:
        logger.debug("File ledger write failed: %s", exc)


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
# Single file processing
# ---------------------------------------------------------------------------


def process_single_file(
    file_manager,
    *,
    parse_result,
    file_path: str,
    config,
    reporter=None,
    ledger=None,
    cost_accumulator=None,
    run_id: str | None = None,
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

    try:
        from unity.file_manager.parse_adapter import adapt_parse_result_for_file_manager
        from unity.file_manager.pipeline import (
            build_ingest_cost_line_items,
            build_observability_cost_line_items,
            build_transport_cost_line_items,
        )
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
        file_record_stage_id = _make_stage_id(
            run_id=run_id,
            file_path=file_path,
            stage_name="file_record",
        )
        fr_result = _run_with_retry(
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
        file_record_value = fr_result.value if isinstance(fr_result.value, dict) else {}
        file_id = file_record_value.get("file_id")
        storage_id = file_record_value.get("storage_id")
        _record_stage_manifest(
            ledger,
            run_id=run_id,
            file_path=file_path,
            stage_name="file_record",
            result=fr_result,
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
            run_id=run_id,
            file_path=file_path,
            stage_name="file_record",
            result=fr_result,
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
        if not fr_result.success:
            raise RuntimeError(f"File record creation failed: {fr_result.error}")

        if cost_accumulator is not None and run_id:
            cost_accumulator.add_line_items(
                build_transport_cost_line_items(
                    run_id=run_id,
                    file_path=file_path,
                    file_id=file_id,
                    storage_id=storage_id,
                    bundle=ingest_payload.bundle,
                    rate_card=cost_accumulator.rate_card,
                    retention_days=getattr(config.cost, "artifact_retention_days", 30),
                ),
            )

        # 2. Build work items for content + tables
        content_result = None
        table_results: List[PipelineStepOutcome] = []

        tables = list(getattr(parse_result, "tables", []) or [])
        do_tables = bool(config.ingest.table_ingest and tables)

        work_items: List[_IngestWorkItem] = []

        if content_rows:
            work_items.append(
                _IngestWorkItem(
                    kind="content",
                    fn=execute_ingest_content,
                    kwargs={
                        "file_manager": file_manager,
                        "file_path": file_path,
                        "content_rows": content_rows,
                        "config": config,
                    },
                    label=f"content({file_path})",
                    stage_name="ingest_content",
                    stage_id=_make_stage_id(
                        run_id=run_id,
                        file_path=file_path,
                        stage_name="ingest_content",
                    ),
                    meta={
                        "row_count": len(content_rows),
                        "context": "content",
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
                work_items.append(
                    _IngestWorkItem(
                        kind="table",
                        stage_name="ingest_table",
                        fn=execute_ingest_table,
                        kwargs={
                            "file_manager": file_manager,
                            "file_path": file_path,
                            "table_label": table_label,
                            "table_rows": rows,
                            "table_input": table_input,
                            "columns": columns,
                            "config": config,
                        },
                        label=f"table({file_path}/{table_label})",
                        table_id=str(getattr(tbl, "table_id", "") or "") or None,
                        stage_id=_make_stage_id(
                            run_id=run_id,
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
                        },
                    ),
                )

        # 3. Execute work items (concurrent when multiple)
        def _handle_completed(item: _IngestWorkItem, res: PipelineStepOutcome) -> None:
            nonlocal content_result
            if item.kind == "content":
                content_result = res
            else:
                table_results.append(res)
            item_meta = {"label": item.label, **dict(item.meta or {})}
            _record_stage_manifest(
                ledger,
                run_id=run_id,
                file_path=file_path,
                stage_name=item.stage_name,
                result=res,
                stage_id=item.stage_id,
                file_id=file_id,
                storage_id=storage_id,
                table_id=item.table_id,
                meta=item_meta,
            )
            _report_stage_progress(
                reporter if enable_progress else None,
                run_id=run_id,
                file_path=file_path,
                stage_name=item.stage_name,
                result=res,
                file_start_time=file_start_time,
                verbosity=verbosity,
                stage_id=item.stage_id,
                file_id=file_id,
                storage_id=storage_id,
                table_id=item.table_id,
                trace_id=trace_id,
                meta=item_meta,
            )
            if cost_accumulator is not None and run_id:
                cost_accumulator.add_line_items(
                    build_ingest_cost_line_items(
                        run_id=run_id,
                        file_path=file_path,
                        file_id=file_id,
                        storage_id=storage_id,
                        stage_name=item.stage_name,
                        stage_id=item.stage_id,
                        table_id=item.table_id,
                        stage_value=res.value,
                        rate_card=cost_accumulator.rate_card,
                    ),
                )

        if len(work_items) <= 1:
            for item in work_items:
                res = _run_with_retry(
                    item.fn,
                    item.kwargs,
                    retry_config=config.retry,
                    label=item.label,
                )
                _handle_completed(item, res)
        elif work_items:
            max_workers = getattr(config.execution, "max_embed_workers", 8)
            with _TPE(max_workers=min(len(work_items), max_workers)) as pool:
                futures = {
                    pool.submit(
                        _run_with_retry,
                        item.fn,
                        item.kwargs,
                        retry_config=config.retry,
                        label=item.label,
                    ): item
                    for item in work_items
                }
                for future in _asc(futures):
                    item = futures[future]
                    try:
                        res = future.result()
                    except Exception as exc:
                        res = PipelineStepOutcome(success=False, error=str(exc))
                    _handle_completed(item, res)

        if cost_accumulator is not None and run_id:
            cost_accumulator.add_line_items(
                build_observability_cost_line_items(
                    run_id=run_id,
                    rate_card=cost_accumulator.rate_card,
                    progress_event_count=(
                        1 + len(work_items) + 1 if enable_progress and reporter else 0
                    ),
                    run_manifest_count=0,
                    file_manifest_count=1 if ledger is not None else 0,
                    stage_manifest_count=(
                        (1 + len(work_items)) if ledger is not None else 0
                    ),
                    cost_ledger_count=0,
                    file_path=file_path,
                    file_id=file_id,
                    storage_id=storage_id,
                ),
            )

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
                run_id=run_id,
                file_id=file_id,
                storage_id=storage_id,
                trace_id=trace_id,
            )

        _record_file_manifest(
            ledger,
            run_id=run_id,
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

        # 5. Build return model
        return _build_return_model(
            file_manager,
            file_path,
            parse_result,
            config,
            return_mode,
            ingest_payload,
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
                run_id=run_id,
                duration_ms=elapsed_ms,
                elapsed_ms=elapsed_ms,
                error=str(e),
                traceback_str=tb_str,
                verbosity=verbosity,
            )
            reporter.report(event)

        _record_file_manifest(
            ledger,
            run_id=run_id,
            file_path=file_path,
            status="error",
            total_duration_ms=elapsed_ms,
            retries_used=0,
            file_id=None,
            storage_id=None,
            meta={"fatal_error": str(e)},
        )

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
    all_parse_results=None,
    run_id: str | None = None,
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
    from uuid import uuid4
    from concurrent.futures import ThreadPoolExecutor as _TPE, as_completed as _asc

    pipeline_start = _time.perf_counter()

    if not parse_results or not file_paths:
        from unity.file_manager.types.ingest import IngestPipelineResult as _IPR

        return _IPR()

    return_mode = getattr(getattr(config, "output", None), "return_mode", "compact")
    results = {}

    parallel = getattr(getattr(config, "execution", None), "parallel_files", False)
    max_workers = getattr(getattr(config, "execution", None), "max_file_workers", 4)
    run_id = run_id or uuid4().hex
    ledger = None
    cost_accumulator = None
    cost_ledger = None

    if getattr(getattr(config, "diagnostics", None), "enable_run_ledger", False):
        from unity.file_manager.pipeline import JsonlRunLedger, PipelineRunManifest
        from unity.file_manager.pipeline import generate_run_ledger_path

        ledger_path = getattr(config.diagnostics, "run_ledger_file", None)
        ledger = JsonlRunLedger(path=ledger_path or generate_run_ledger_path())
        ledger.write(
            PipelineRunManifest(
                run_id=run_id,
                status="started",
                file_count=len(parse_results),
                parallel_files=parallel,
                meta={"max_file_workers": max_workers},
            ),
        )

    if getattr(getattr(config, "cost", None), "enable_cost_ledger", False):
        from unity.file_manager.pipeline import (
            JsonlCostLedger,
            PipelineCostAccumulator,
            PipelineCostRateCard,
            build_parse_cost_line_items,
            generate_cost_ledger_path,
        )

        rate_card = PipelineCostRateCard.from_config(config.cost)
        cost_accumulator = PipelineCostAccumulator(
            run_id=run_id,
            rate_card=rate_card,
            environment=getattr(config.cost, "environment", "local"),
            tenant_id=getattr(config.cost, "tenant_id", None),
        )
        cost_ledger_path = getattr(config.cost, "cost_ledger_file", None)
        cost_ledger = JsonlCostLedger(
            path=cost_ledger_path or generate_cost_ledger_path(),
        )
        for parse_result in list(all_parse_results or parse_results):
            logical_path = str(getattr(parse_result, "logical_path", "") or "")
            cost_accumulator.add_line_items(
                build_parse_cost_line_items(
                    run_id=run_id,
                    file_path=logical_path,
                    rate_card=rate_card,
                    **_extract_parse_cost_metrics(
                        parse_result,
                        logical_path,
                        config.parse,
                    ),
                ),
            )

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
                ledger=ledger,
                cost_accumulator=cost_accumulator,
                run_id=run_id,
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
                    ledger=ledger,
                    cost_accumulator=cost_accumulator,
                    run_id=run_id,
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
                    _record_file_manifest(
                        ledger,
                        run_id=run_id,
                        file_path=path,
                        status="error",
                        total_duration_ms=0.0,
                        retries_used=0,
                        meta={"fatal_error": str(e)},
                    )

    if enable_progress and reporter:
        reporter.flush()

    total_ms = (_time.perf_counter() - pipeline_start) * 1000
    if ledger is not None:
        try:
            from unity.file_manager.pipeline import PipelineRunManifest

            success_count = sum(
                1
                for value in results.values()
                if getattr(value, "status", "error") == "success"
            )
            failure_count = len(results) - success_count
            ledger.write(
                PipelineRunManifest(
                    run_id=run_id,
                    status="completed",
                    file_count=len(results),
                    success_count=success_count,
                    failure_count=failure_count,
                    parallel_files=parallel,
                    total_duration_ms=total_ms,
                    meta={"max_file_workers": max_workers},
                ),
            )
        finally:
            ledger.flush()
            ledger.close()

    if cost_accumulator is not None and cost_ledger is not None:
        try:
            from unity.file_manager.pipeline import build_observability_cost_line_items

            cost_accumulator.add_line_items(
                build_observability_cost_line_items(
                    run_id=run_id,
                    rate_card=cost_accumulator.rate_card,
                    progress_event_count=(
                        2 * len(list(all_parse_results or parse_results))
                        if enable_progress and reporter
                        else 0
                    ),
                    run_manifest_count=2 if ledger is not None else 0,
                    file_manifest_count=0,
                    stage_manifest_count=0,
                    cost_ledger_count=1,
                ),
            )
            cost_ledger.write(cost_accumulator.build_ledger())
        finally:
            cost_ledger.flush()
            cost_ledger.close()

    from unity.file_manager.types.ingest import IngestPipelineResult as _IPR

    return _IPR.from_results(results, total_duration_ms=total_ms)
