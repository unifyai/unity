"""Shared orchestration helpers for ingestion pipelines.

Provides the per-source artifact dispatch primitive consumed identically
by FileManager and DataManager paths:

- ``ingest_artifacts``: per-source parallel artifact dispatch with retry.

Callers supply a domain-specific ``ArtifactIngestFn`` callable while the
shared engine handles parallelism, retry, instrumentation, and result
aggregation.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    Literal,
    Optional,
)

from .instrumentation import PipelineInstrumentation
from .retry_policy import ResilientRequestPolicy
from .work_queue import CancellationCheck, PipelineCancelled

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


# ---------------------------------------------------------------------------
# Callback protocols
# ---------------------------------------------------------------------------

ArtifactIngestFn = Callable[[ArtifactWorkItem], Any]
"""Signature for the per-artifact ingest callback.

Receives an ``ArtifactWorkItem`` and returns an opaque result
(domain-specific, e.g. dict with ingest_result)."""


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
    is_cancelled: Optional[CancellationCheck] = None,
) -> list[ArtifactWorkResult]:
    """Dispatch artifact ingestion with retry, instrumentation, and optional parallelism.

    Handles both content and table artifacts uniformly. Each ``ArtifactWorkItem``
    is passed to *ingest_fn*, wrapped in retry logic, and its outcome is recorded
    as a stage manifest via *instrumentation*.

    When *is_cancelled* is supplied, each work item is checked before
    dispatch. In threaded mode, remaining futures are cancelled when
    cancellation is detected.
    """
    if not work_items:
        return []

    results: list[ArtifactWorkResult] = []

    def _process_item(item: ArtifactWorkItem) -> ArtifactWorkResult:
        if is_cancelled and is_cancelled():
            raise PipelineCancelled(f"Cancelled before {item.kind}({item.label})")

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
            try:
                results.append(_process_item(item))
            except PipelineCancelled:
                results.append(
                    ArtifactWorkResult(
                        kind=item.kind,
                        label=item.label,
                        success=False,
                        error="cancelled",
                    ),
                )
    else:
        effective_workers = min(len(work_items), max_workers)
        with ThreadPoolExecutor(max_workers=effective_workers) as pool:
            futures: dict[Future, ArtifactWorkItem] = {
                pool.submit(_process_item, item): item for item in work_items
            }
            for future in as_completed(futures):
                item = futures[future]
                try:
                    results.append(future.result())
                except PipelineCancelled:
                    results.append(
                        ArtifactWorkResult(
                            kind=item.kind,
                            label=item.label,
                            success=False,
                            error="cancelled",
                        ),
                    )
                    for pending in futures:
                        pending.cancel()
                    break
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
