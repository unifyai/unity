"""Shared orchestration helpers for ingestion pipelines.

Provides two composable entry points consumed identically by FileManager
and DataManager paths:

- ``run_ingestion_pipeline``: top-level parse → dispatch → finalize flow.
- ``ingest_artifacts``: per-source parallel artifact dispatch with retry.

Both are callback-driven: callers supply domain-specific ``ProcessFileFn``
and ``ArtifactIngestFn`` callables while the shared engine handles
parallelism, retry, instrumentation, memory release, and result aggregation.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    Literal,
    Protocol,
    runtime_checkable,
)

from .instrumentation import PipelineInstrumentation
from .retry_policy import ResilientRequestPolicy

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Typed result models
# ---------------------------------------------------------------------------


@dataclass
class ArtifactWorkItem:
    """Describes one artifact to ingest (content section or table)."""

    kind: Literal["content", "table"]
    label: str
    stage_name: str
    payload: Any = None
    columns: list[str] = field(default_factory=list)
    row_count: int = 0
    stage_id: str | None = None
    table_id: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class ArtifactWorkResult:
    """Result of ingesting one artifact."""

    kind: str
    label: str
    success: bool
    duration_ms: float = 0.0
    retries: int = 0
    rows_inserted: int = 0
    error: str | None = None
    failure_kind: str | None = None
    value: Any = None


@dataclass
class FileWorkResult:
    """Result of processing one file through the pipeline."""

    file_path: str
    status: str = "success"
    duration_ms: float = 0.0
    artifact_results: list[ArtifactWorkResult] = field(default_factory=list)
    error: str | None = None
    value: Any = None


@dataclass
class PipelineRunResult:
    """Aggregate result of a full pipeline run."""

    file_results: dict[str, FileWorkResult] = field(default_factory=dict)
    total_duration_ms: float = 0.0
    success_count: int = 0
    failure_count: int = 0


@dataclass
class FileProcessingContext:
    """Context object passed to ``ProcessFileFn`` callbacks."""

    parse_result: Any
    file_path: str
    instrumentation: PipelineInstrumentation
    retry_config: Any | None = None
    reporter: Any | None = None
    enable_progress: bool = False
    verbosity: str = "low"


# ---------------------------------------------------------------------------
# Callback protocols
# ---------------------------------------------------------------------------

ArtifactIngestFn = Callable[[ArtifactWorkItem], Any]
"""Signature for the per-artifact ingest callback.

Receives an ``ArtifactWorkItem`` and returns an opaque result
(domain-specific, e.g. dict with ingest_result)."""


@runtime_checkable
class ProcessFileFn(Protocol):
    """Signature for the per-file processing callback."""

    def __call__(self, ctx: FileProcessingContext) -> FileWorkResult: ...


# ---------------------------------------------------------------------------
# Step-level retry helper (shared)
# ---------------------------------------------------------------------------


@dataclass
class _StepOutcome:
    success: bool
    value: Any = None
    error: str | None = None
    failure_kind: str | None = None
    duration_ms: float = 0.0
    retries: int = 0


def run_with_retry(
    fn: Callable[..., Any],
    kwargs: Dict[str, Any],
    *,
    retry_config: Any | None = None,
    label: str = "",
) -> _StepOutcome:
    """Call *fn* with typed retry policy, backoff, jitter, and deadline."""
    if retry_config is not None:
        policy = ResilientRequestPolicy.from_config(retry_config)
    else:
        policy = ResilientRequestPolicy()
    last_error = ""
    started_at = time.perf_counter()
    for attempt in range(policy.max_retries + 1):
        t0 = time.perf_counter()
        try:
            value = fn(**kwargs)
            return _StepOutcome(
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
                    "[Pipeline] %s attempt %d failed (%.0fms): %s -- retrying in %.1fs",
                    label,
                    attempt + 1,
                    elapsed,
                    exc,
                    delay,
                )
                if delay > 0:
                    time.sleep(delay)
            else:
                logger.error(
                    "[Pipeline] %s failed after %d attempts (%.0fms): %s",
                    label,
                    attempt + 1,
                    elapsed,
                    exc,
                )
                return _StepOutcome(
                    success=False,
                    error=last_error,
                    failure_kind=decision.failure_kind,
                    duration_ms=elapsed,
                    retries=attempt,
                )
    return _StepOutcome(success=False, error=last_error)


# ---------------------------------------------------------------------------
# ingest_artifacts
# ---------------------------------------------------------------------------


def ingest_artifacts(
    *,
    work_items: list[ArtifactWorkItem],
    ingest_fn: ArtifactIngestFn,
    instrumentation: PipelineInstrumentation,
    source_path: str,
    max_workers: int = 8,
    retry_config: Any | None = None,
) -> list[ArtifactWorkResult]:
    """Dispatch artifact ingestion with retry, instrumentation, and optional parallelism.

    Handles both content and table artifacts uniformly. Each ``ArtifactWorkItem``
    is passed to *ingest_fn*, wrapped in retry logic, and its outcome is recorded
    as a stage manifest via *instrumentation*.
    """
    if not work_items:
        return []

    results: list[ArtifactWorkResult] = []

    def _process_item(item: ArtifactWorkItem) -> ArtifactWorkResult:
        outcome = run_with_retry(
            lambda **_kw: ingest_fn(item),
            {},
            retry_config=retry_config,
            label=f"{item.kind}({source_path}/{item.label})",
        )
        result = ArtifactWorkResult(
            kind=item.kind,
            label=item.label,
            success=outcome.success,
            duration_ms=outcome.duration_ms,
            retries=outcome.retries,
            error=outcome.error,
            failure_kind=outcome.failure_kind,
            value=outcome.value,
        )

        instrumentation.record_stage(
            file_path=source_path,
            stage_name=item.stage_name,
            status="success" if outcome.success else "error",
            duration_ms=outcome.duration_ms,
            retries_used=outcome.retries,
            error=outcome.error,
            stage_id=item.stage_id,
            table_id=item.table_id,
            meta={
                "label": item.label,
                **item.meta,
                **(
                    {"failure_kind": outcome.failure_kind}
                    if outcome.failure_kind
                    else {}
                ),
            },
        )

        if outcome.success:
            instrumentation.add_ingest_costs(
                file_path=source_path,
                file_id=item.meta.get("file_id"),
                storage_id=item.meta.get("storage_id"),
                stage_name=item.stage_name,
                stage_id=item.stage_id,
                stage_value=outcome.value,
                table_id=item.table_id,
            )

        return result

    if len(work_items) <= 1:
        for item in work_items:
            results.append(_process_item(item))
    else:
        effective_workers = min(len(work_items), max_workers)
        with ThreadPoolExecutor(max_workers=effective_workers) as pool:
            futures = {pool.submit(_process_item, item): item for item in work_items}
            for future in as_completed(futures):
                item = futures[future]
                try:
                    results.append(future.result())
                except Exception as exc:
                    results.append(
                        ArtifactWorkResult(
                            kind=item.kind,
                            label=item.label,
                            success=False,
                            error=str(exc),
                        ),
                    )

    return results


# ---------------------------------------------------------------------------
# run_ingestion_pipeline
# ---------------------------------------------------------------------------


def run_ingestion_pipeline(
    *,
    parse_requests: list,
    parse_config: Any,
    process_file_fn: ProcessFileFn,
    instrumentation: PipelineInstrumentation,
    execution_config: Any | None = None,
    retry_config: Any | None = None,
    reporter: Any | None = None,
    enable_progress: bool = False,
    verbosity: str = "low",
) -> PipelineRunResult:
    """Top-level shared orchestrator for parse → ingest pipelines.

    Steps:
      1. Parse all files via ``FileParser.parse_batch``.
      2. Record parse costs.
      3. Dispatch ``process_file_fn`` per successful parse result (parallel or serial).
      4. Record file manifests for failures.
      5. Null out parse results after each file to release memory.
      6. Return ``PipelineRunResult``.

    Both the FM executor and the DM ingestion script delegate to this function,
    supplying their own ``process_file_fn`` callback.
    """
    from unity.file_manager.file_parsers.file_parser import FileParser

    pipeline_start = time.perf_counter()

    if not parse_requests:
        return PipelineRunResult()

    parallel = (
        getattr(execution_config, "parallel_files", False)
        if execution_config
        else False
    )
    max_workers = (
        getattr(execution_config, "max_file_workers", 4) if execution_config else 4
    )

    # 1. Parse
    parser = FileParser()
    parse_results: list = parser.parse_batch(
        parse_requests,
        parse_config=parse_config,
    )

    if not parse_results:
        return PipelineRunResult()

    # 2. Record parse costs
    all_parse_results = list(parse_results)
    for pr in all_parse_results:
        logical_path = str(getattr(pr, "logical_path", "") or "")
        trace = getattr(pr, "trace", None)
        if trace and instrumentation.has_cost_tracking:
            from unity.file_manager.managers.utils.executor import (
                _extract_parse_cost_metrics,
            )

            metrics = _extract_parse_cost_metrics(pr, logical_path, parse_config)
            instrumentation.add_parse_costs(
                file_path=logical_path,
                **metrics,
            )

    # 3. Split successes vs failures
    successful: list[tuple[Any, str]] = []
    for pr in parse_results:
        status = getattr(pr, "status", "error")
        logical_path = str(getattr(pr, "logical_path", "") or "")
        if status == "success":
            successful.append((pr, logical_path))
        else:
            error_msg = str(getattr(pr, "error", "Parse failed") or "Parse failed")
            instrumentation.record_file(
                file_path=logical_path,
                status="error",
                meta={"parse_error": error_msg},
            )

    if not successful:
        total_ms = (time.perf_counter() - pipeline_start) * 1000
        return PipelineRunResult(
            total_duration_ms=total_ms,
            failure_count=len(parse_results),
        )

    # 4. Dispatch per-file processing
    file_results: dict[str, FileWorkResult] = {}

    def _dispatch_file(pr_and_path: tuple, idx: int) -> tuple[str, FileWorkResult]:
        pr, file_path = pr_and_path
        ctx = FileProcessingContext(
            parse_result=pr,
            file_path=file_path,
            instrumentation=instrumentation,
            retry_config=retry_config,
            reporter=reporter,
            enable_progress=enable_progress,
            verbosity=verbosity,
        )
        try:
            result = process_file_fn(ctx)
        except Exception as exc:
            logger.error("Fatal error processing %s: %s", file_path, exc)
            result = FileWorkResult(
                file_path=file_path,
                status="error",
                error=str(exc),
            )
            instrumentation.record_file(
                file_path=file_path,
                status="error",
                meta={"fatal_error": str(exc)},
            )
        return file_path, result

    if not parallel or len(successful) == 1:
        for idx, item in enumerate(successful):
            file_path, result = _dispatch_file(item, idx)
            file_results[file_path] = result
            try:
                parse_results[idx] = None
            except (TypeError, IndexError):
                pass
    else:
        with ThreadPoolExecutor(max_workers=min(len(successful), max_workers)) as pool:
            futures = {
                pool.submit(_dispatch_file, item, idx): item
                for idx, item in enumerate(successful)
            }
            for future in as_completed(futures):
                pr_and_path = futures[future]
                file_path = pr_and_path[1]
                try:
                    _, result = future.result()
                except Exception as exc:
                    result = FileWorkResult(
                        file_path=file_path,
                        status="error",
                        error=str(exc),
                    )
                file_results[file_path] = result

    total_ms = (time.perf_counter() - pipeline_start) * 1000
    success_count = sum(1 for r in file_results.values() if r.status == "success")
    failure_count = len(file_results) - success_count
    parse_failures = len(parse_results) - len(successful)
    failure_count += parse_failures

    return PipelineRunResult(
        file_results=file_results,
        total_duration_ms=total_ms,
        success_count=success_count,
        failure_count=failure_count,
    )
