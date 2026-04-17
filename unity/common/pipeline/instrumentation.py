"""Reusable pipeline instrumentation context manager.

Encapsulates all observability wiring (run/cost ledgers, stage/file/run manifests,
cost accumulation) behind a single object that is safe to call unconditionally.
When disabled, every method is a no-op -- callers never need conditionals.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any
from uuid import NAMESPACE_URL, uuid4, uuid5

from .cost_ledger import (
    CostLedger,
    JsonlCostLedger,
    PipelineCostAccumulator,
    PipelineCostLineItem,
    PipelineCostRateCard,
    build_ingest_cost_line_items,
    build_observability_cost_line_items,
    build_parse_cost_line_items,
    build_transport_cost_line_items,
    generate_cost_ledger_path,
)
from .run_ledger import (
    JsonlRunLedger,
    PipelineFileManifest,
    PipelineRunManifest,
    PipelineStageManifest,
    RunLedger,
    generate_run_ledger_path,
)

logger = logging.getLogger(__name__)


@dataclass
class _Counters:
    """Mutable counters for observability cost estimation."""

    progress_events: int = 0
    run_manifests: int = 0
    file_manifests: int = 0
    stage_manifests: int = 0
    cost_ledger_writes: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)

    def increment(
        self,
        *,
        progress: int = 0,
        run: int = 0,
        filec: int = 0,
        stage: int = 0,
        cost: int = 0,
    ) -> None:
        with self.lock:
            self.progress_events += progress
            self.run_manifests += run
            self.file_manifests += filec
            self.stage_manifests += stage
            self.cost_ledger_writes += cost


class PipelineInstrumentation:
    """Context manager that wires up run/cost ledger and manifest recording.

    Usage::

        instr = PipelineInstrumentation.from_config(config, run_id="abc123")
        with instr:
            instr.record_stage(...)
            instr.record_file(...)
            instr.add_parse_costs(...)
        # On exit: run-completed manifest + cost ledger are flushed

    All public methods are safe to call when instrumentation is disabled (no-op).
    """

    def __init__(
        self,
        *,
        run_id: str | None = None,
        run_ledger: RunLedger | None = None,
        cost_accumulator: PipelineCostAccumulator | None = None,
        cost_ledger: CostLedger | None = None,
        parallel_files: bool = False,
        file_count: int = 0,
        meta: dict[str, Any] | None = None,
    ) -> None:
        self.run_id: str = run_id or uuid4().hex
        self._run_ledger = run_ledger
        self._cost_accumulator = cost_accumulator
        self._cost_ledger = cost_ledger
        self._parallel_files = parallel_files
        self._file_count = file_count
        self._meta = dict(meta or {})
        self._counters = _Counters()
        self._started_at: float | None = None

    @property
    def enabled(self) -> bool:
        return self._run_ledger is not None or self._cost_accumulator is not None

    @property
    def has_run_ledger(self) -> bool:
        return self._run_ledger is not None

    @property
    def has_cost_tracking(self) -> bool:
        return self._cost_accumulator is not None

    @property
    def cost_accumulator(self) -> PipelineCostAccumulator | None:
        return self._cost_accumulator

    @property
    def rate_card(self) -> PipelineCostRateCard | None:
        return self._cost_accumulator.rate_card if self._cost_accumulator else None

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(
        cls,
        config,
        *,
        run_id: str | None = None,
        parallel_files: bool = False,
        file_count: int = 0,
        meta: dict[str, Any] | None = None,
    ) -> "PipelineInstrumentation":
        """Build from a ``FilePipelineConfig`` (or any object with .diagnostics/.cost)."""
        run_id = run_id or uuid4().hex
        run_ledger: RunLedger | None = None
        cost_accumulator: PipelineCostAccumulator | None = None
        cost_ledger_inst: CostLedger | None = None

        diagnostics = getattr(config, "diagnostics", None)
        if diagnostics and getattr(diagnostics, "enable_run_ledger", False):
            ledger_path = getattr(diagnostics, "run_ledger_file", None)
            run_ledger = JsonlRunLedger(path=ledger_path or generate_run_ledger_path())

        cost_cfg = getattr(config, "cost", None)
        if cost_cfg and getattr(cost_cfg, "enable_cost_ledger", False):
            rate_card = PipelineCostRateCard.from_config(cost_cfg)
            cost_accumulator = PipelineCostAccumulator(
                run_id=run_id,
                rate_card=rate_card,
                environment=getattr(cost_cfg, "environment", "local"),
                tenant_id=getattr(cost_cfg, "tenant_id", None),
            )
            cost_ledger_path = getattr(cost_cfg, "cost_ledger_file", None)
            cost_ledger_inst = JsonlCostLedger(
                path=cost_ledger_path or generate_cost_ledger_path(),
            )

        return cls(
            run_id=run_id,
            run_ledger=run_ledger,
            cost_accumulator=cost_accumulator,
            cost_ledger=cost_ledger_inst,
            parallel_files=parallel_files,
            file_count=file_count,
            meta=meta,
        )

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "PipelineInstrumentation":
        self._started_at = time.perf_counter()
        if self._run_ledger is not None:
            try:
                self._run_ledger.write(
                    PipelineRunManifest(
                        run_id=self.run_id,
                        status="started",
                        file_count=self._file_count,
                        parallel_files=self._parallel_files,
                        meta=self._meta,
                    ),
                )
                self._counters.increment(run=1)
            except Exception as exc:
                logger.debug("Run-started manifest write failed: %s", exc)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        total_ms = (
            (time.perf_counter() - self._started_at) * 1000
            if self._started_at is not None
            else 0.0
        )
        self._finalize_run_ledger(total_ms)
        self._finalize_cost_ledger()
        return None

    # ------------------------------------------------------------------
    # Stage recording
    # ------------------------------------------------------------------

    def record_stage(
        self,
        *,
        file_path: str,
        stage_name: str,
        status: str,
        duration_ms: float = 0.0,
        retries_used: int = 0,
        error: str | None = None,
        stage_id: str | None = None,
        file_id: int | None = None,
        storage_id: str | None = None,
        table_id: str | None = None,
        meta: dict[str, Any] | None = None,
    ) -> None:
        if self._run_ledger is None:
            return
        try:
            self._run_ledger.write(
                PipelineStageManifest(
                    run_id=self.run_id,
                    stage_id=stage_id,
                    file_path=file_path,
                    file_id=file_id,
                    storage_id=storage_id,
                    table_id=table_id,
                    stage_name=stage_name,
                    status="success" if status == "success" else "error",
                    duration_ms=duration_ms,
                    retries_used=retries_used,
                    error=error,
                    meta=dict(meta or {}),
                ),
            )
            self._counters.increment(stage=1)
        except Exception as exc:
            logger.debug("Stage ledger write failed: %s", exc)

    def record_file(
        self,
        *,
        file_path: str,
        status: str,
        total_duration_ms: float = 0.0,
        retries_used: int = 0,
        file_id: int | None = None,
        storage_id: str | None = None,
        meta: dict[str, Any] | None = None,
    ) -> None:
        if self._run_ledger is None:
            return
        try:
            self._run_ledger.write(
                PipelineFileManifest(
                    run_id=self.run_id,
                    file_path=file_path,
                    file_id=file_id,
                    storage_id=storage_id,
                    status="success" if status == "success" else "error",
                    total_duration_ms=total_duration_ms,
                    retries_used=retries_used,
                    meta=dict(meta or {}),
                ),
            )
            self._counters.increment(filec=1)
        except Exception as exc:
            logger.debug("File ledger write failed: %s", exc)

    # ------------------------------------------------------------------
    # Cost accumulation
    # ------------------------------------------------------------------

    def add_cost_line_items(self, items: list[PipelineCostLineItem]) -> None:
        if self._cost_accumulator is not None:
            self._cost_accumulator.add_line_items(items)

    def add_parse_costs(
        self,
        *,
        file_path: str,
        parse_duration_seconds: float,
        estimated_peak_memory_bytes: int = 0,
        llm_enrichment_calls: int = 0,
        parse_backend: str | None = None,
        trace_status: str | None = None,
    ) -> None:
        if self._cost_accumulator is None:
            return
        self._cost_accumulator.add_line_items(
            build_parse_cost_line_items(
                run_id=self.run_id,
                file_path=file_path,
                parse_duration_seconds=parse_duration_seconds,
                estimated_peak_memory_bytes=estimated_peak_memory_bytes,
                llm_enrichment_calls=llm_enrichment_calls,
                parse_backend=parse_backend,
                trace_status=trace_status,
                rate_card=self._cost_accumulator.rate_card,
            ),
        )

    def add_ingest_costs(
        self,
        *,
        file_path: str,
        file_id: int | None,
        storage_id: str | None,
        stage_name: str,
        stage_id: str | None,
        stage_value: Any,
        table_id: str | None = None,
    ) -> None:
        if self._cost_accumulator is None:
            return
        self._cost_accumulator.add_line_items(
            build_ingest_cost_line_items(
                run_id=self.run_id,
                file_path=file_path,
                file_id=file_id,
                storage_id=storage_id,
                stage_name=stage_name,
                stage_id=stage_id,
                stage_value=stage_value,
                rate_card=self._cost_accumulator.rate_card,
                table_id=table_id,
            ),
        )

    def add_transport_costs(
        self,
        *,
        file_path: str,
        file_id: int | None,
        storage_id: str | None,
        bundle,
        retention_days: int = 30,
    ) -> None:
        if self._cost_accumulator is None:
            return
        self._cost_accumulator.add_line_items(
            build_transport_cost_line_items(
                run_id=self.run_id,
                file_path=file_path,
                file_id=file_id,
                storage_id=storage_id,
                bundle=bundle,
                rate_card=self._cost_accumulator.rate_card,
                retention_days=retention_days,
            ),
        )

    def add_observability_costs(
        self,
        *,
        progress_event_count: int = 0,
        file_path: str | None = None,
        file_id: int | None = None,
        storage_id: str | None = None,
    ) -> None:
        if self._cost_accumulator is None:
            return
        self._cost_accumulator.add_line_items(
            build_observability_cost_line_items(
                run_id=self.run_id,
                rate_card=self._cost_accumulator.rate_card,
                progress_event_count=progress_event_count,
                run_manifest_count=self._counters.run_manifests,
                file_manifest_count=self._counters.file_manifests,
                stage_manifest_count=self._counters.stage_manifests,
                cost_ledger_count=1 if self._cost_ledger is not None else 0,
                file_path=file_path,
                file_id=file_id,
                storage_id=storage_id,
            ),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def make_stage_id(
        self,
        *,
        file_path: str,
        stage_name: str,
        discriminator: str | None = None,
    ) -> str | None:
        if not self.run_id:
            return None
        raw = "::".join(
            part
            for part in (self.run_id, file_path, stage_name, discriminator or "")
            if part is not None
        )
        return uuid5(NAMESPACE_URL, raw).hex

    def increment_progress_events(self, count: int = 1) -> None:
        self._counters.increment(progress=count)

    # ------------------------------------------------------------------
    # Finalization (private)
    # ------------------------------------------------------------------

    def _finalize_run_ledger(self, total_ms: float) -> None:
        if self._run_ledger is None:
            return
        try:
            self._run_ledger.write(
                PipelineRunManifest(
                    run_id=self.run_id,
                    status="completed",
                    file_count=self._file_count,
                    parallel_files=self._parallel_files,
                    total_duration_ms=total_ms,
                    meta=self._meta,
                ),
            )
            self._counters.increment(run=1)
        except Exception as exc:
            logger.debug("Run-completed manifest write failed: %s", exc)
        finally:
            try:
                self._run_ledger.flush()
                self._run_ledger.close()
            except Exception:
                pass

    def _finalize_cost_ledger(self) -> None:
        if self._cost_accumulator is None or self._cost_ledger is None:
            return
        try:
            self._cost_ledger.write(self._cost_accumulator.build_ledger())
            self._counters.increment(cost=1)
        except Exception as exc:
            logger.debug("Cost ledger finalization failed: %s", exc)
        finally:
            try:
                self._cost_ledger.flush()
                self._cost_ledger.close()
            except Exception:
                pass
