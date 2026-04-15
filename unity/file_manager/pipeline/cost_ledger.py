from __future__ import annotations

import json
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Literal, Protocol
from urllib.parse import unquote, urlparse

from pydantic import BaseModel, Field

from unity.file_manager.file_parsers.types.contracts import (
    FileParseRequest,
    FileParseResult,
)
from unity.file_manager.file_parsers.utils.memory_scheduler import (
    estimate_peak_memory_bytes,
)

from .types import ObjectStoreArtifactHandle, ParsedFileBundle

if TYPE_CHECKING:
    from unity.file_manager.types.config import CostLedgerConfig, ParseConfig


CostConfidence = Literal["high", "medium", "low"]


class PipelineCostRateCard(BaseModel):
    """Versioned unit-cost rates used for immediate pipeline cost estimates."""

    version: str = "local-default-v1"
    currency: str = "USD"
    parse_cpu_per_second: float = 0.000011
    parse_memory_gb_second: float = 0.0000015
    artifact_storage_gb_month: float = 0.020
    row_ingest_request: float = 0.0005
    row_ingest_row: float = 0.000001
    embedding_row: float = 0.00002
    llm_enrichment_call: float = 0.002
    observability_event: float = 0.0000005

    @classmethod
    def from_config(cls, cost_config: "CostLedgerConfig") -> "PipelineCostRateCard":
        rate_card = getattr(cost_config, "rate_card", None)
        if rate_card is None:
            return cls()
        return cls(
            version=str(
                getattr(rate_card, "version", cls.model_fields["version"].default),
            ),
            currency=str(
                getattr(rate_card, "currency", cls.model_fields["currency"].default),
            ),
            parse_cpu_per_second=float(
                getattr(
                    rate_card,
                    "parse_cpu_per_second",
                    cls.model_fields["parse_cpu_per_second"].default,
                ),
            ),
            parse_memory_gb_second=float(
                getattr(
                    rate_card,
                    "parse_memory_gb_second",
                    cls.model_fields["parse_memory_gb_second"].default,
                ),
            ),
            artifact_storage_gb_month=float(
                getattr(
                    rate_card,
                    "artifact_storage_gb_month",
                    cls.model_fields["artifact_storage_gb_month"].default,
                ),
            ),
            row_ingest_request=float(
                getattr(
                    rate_card,
                    "row_ingest_request",
                    cls.model_fields["row_ingest_request"].default,
                ),
            ),
            row_ingest_row=float(
                getattr(
                    rate_card,
                    "row_ingest_row",
                    cls.model_fields["row_ingest_row"].default,
                ),
            ),
            embedding_row=float(
                getattr(
                    rate_card,
                    "embedding_row",
                    cls.model_fields["embedding_row"].default,
                ),
            ),
            llm_enrichment_call=float(
                getattr(
                    rate_card,
                    "llm_enrichment_call",
                    cls.model_fields["llm_enrichment_call"].default,
                ),
            ),
            observability_event=float(
                getattr(
                    rate_card,
                    "observability_event",
                    cls.model_fields["observability_event"].default,
                ),
            ),
        )


class PipelineCostLineItem(BaseModel):
    """One estimated or reconciled cost component attributed to the pipeline run."""

    run_id: str
    component: str
    usage_unit: str
    quantity: float
    unit_rate: float
    estimated_cost: float
    actual_cost: float | None = None
    currency: str = "USD"
    source: str = "rate_card_estimate"
    confidence: CostConfidence = "medium"
    file_path: str | None = None
    file_id: int | None = None
    storage_id: str | None = None
    stage_name: str | None = None
    stage_id: str | None = None
    table_id: str | None = None
    reconciliation_ref: str | None = None
    meta: dict[str, Any] = Field(default_factory=dict)


class PipelineCostLedger(BaseModel):
    """Typed per-run pipeline cost ledger."""

    record_type: Literal["cost_ledger"] = "cost_ledger"
    run_id: str
    tenant_id: str | None = None
    environment: str = "local"
    currency: str = "USD"
    rate_card_version: str
    line_items: list[PipelineCostLineItem] = Field(default_factory=list)
    estimated_total: float = 0.0
    reconciled_total: float | None = None
    recorded_at: str = Field(default_factory=lambda: _utc_now().isoformat())


class CostLedger(Protocol):
    """Port for emitting typed pipeline cost ledgers."""

    def write(self, ledger: PipelineCostLedger) -> None: ...

    def flush(self) -> None: ...

    def close(self) -> None: ...


class JsonlCostLedger:
    """Thread-safe JSONL writer for pipeline cost ledgers."""

    def __init__(self, *, path: str | Path):
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("a", encoding="utf-8", newline="\n")
        self._lock = threading.Lock()

    def write(self, ledger: PipelineCostLedger) -> None:
        payload = ledger.model_dump(mode="json", exclude_none=True)
        with self._lock:
            self._fh.write(json.dumps(payload, ensure_ascii=False))
            self._fh.write("\n")

    def flush(self) -> None:
        with self._lock:
            self._fh.flush()

    def close(self) -> None:
        with self._lock:
            try:
                self._fh.flush()
            finally:
                self._fh.close()


class PipelineCostAccumulator:
    """Thread-safe in-memory accumulator for estimated cost line items."""

    def __init__(
        self,
        *,
        run_id: str,
        rate_card: PipelineCostRateCard,
        environment: str = "local",
        tenant_id: str | None = None,
    ):
        self.run_id = run_id
        self.rate_card = rate_card
        self.environment = environment
        self.tenant_id = tenant_id
        self._line_items: list[PipelineCostLineItem] = []
        self._lock = threading.Lock()

    def add_line_items(self, line_items: Iterable[PipelineCostLineItem]) -> None:
        items = [
            item for item in line_items if item.quantity > 0 or item.estimated_cost > 0
        ]
        if not items:
            return
        with self._lock:
            self._line_items.extend(items)

    def build_ledger(self) -> PipelineCostLedger:
        with self._lock:
            items = list(self._line_items)
        estimated_total = round(sum(item.estimated_cost for item in items), 6)
        actuals = [item.actual_cost for item in items if item.actual_cost is not None]
        reconciled_total = round(sum(actuals), 6) if actuals else None
        return PipelineCostLedger(
            run_id=self.run_id,
            tenant_id=self.tenant_id,
            environment=self.environment,
            currency=self.rate_card.currency,
            rate_card_version=self.rate_card.version,
            line_items=items,
            estimated_total=estimated_total,
            reconciled_total=reconciled_total,
        )


def build_parse_cost_line_items(
    *,
    run_id: str,
    file_path: str,
    parse_result: FileParseResult,
    parse_config: "ParseConfig",
    rate_card: PipelineCostRateCard,
) -> list[PipelineCostLineItem]:
    """Estimate parse-stage cost from measured duration and bounded memory heuristics."""

    items: list[PipelineCostLineItem] = []
    trace = getattr(parse_result, "trace", None)
    parse_duration_seconds = float(getattr(trace, "duration_ms", 0.0) or 0.0) / 1000.0
    trace_status = getattr(getattr(trace, "status", None), "value", None)
    backend = getattr(trace, "backend", None)

    if parse_duration_seconds > 0:
        items.append(
            _line_item(
                run_id=run_id,
                component="parse_compute_cpu",
                usage_unit="cpu_seconds",
                quantity=parse_duration_seconds,
                unit_rate=rate_card.parse_cpu_per_second,
                currency=rate_card.currency,
                confidence="medium",
                file_path=file_path,
                stage_name="parse",
                meta={
                    "parse_backend": backend,
                    "trace_status": trace_status,
                },
            ),
        )

    estimated_peak_bytes = _estimate_peak_memory_bytes_for_result(
        file_path=file_path,
        parse_result=parse_result,
        parse_config=parse_config,
    )
    estimated_memory_gb_seconds = (
        (estimated_peak_bytes / (1024.0**3)) * parse_duration_seconds
        if estimated_peak_bytes > 0 and parse_duration_seconds > 0
        else 0.0
    )
    if estimated_memory_gb_seconds > 0:
        items.append(
            _line_item(
                run_id=run_id,
                component="parse_compute_memory",
                usage_unit="gb_seconds",
                quantity=estimated_memory_gb_seconds,
                unit_rate=rate_card.parse_memory_gb_second,
                currency=rate_card.currency,
                confidence="low",
                file_path=file_path,
                stage_name="parse",
                meta={
                    "estimated_peak_bytes": estimated_peak_bytes,
                    "parse_backend": backend,
                },
            ),
        )

    llm_calls = _estimate_llm_enrichment_calls(parse_result)
    if llm_calls > 0:
        items.append(
            _line_item(
                run_id=run_id,
                component="llm_enrichment",
                usage_unit="calls",
                quantity=float(llm_calls),
                unit_rate=rate_card.llm_enrichment_call,
                currency=rate_card.currency,
                confidence="low",
                file_path=file_path,
                stage_name="parse",
                meta={"parse_backend": backend},
            ),
        )

    return items


def build_transport_cost_line_items(
    *,
    run_id: str,
    file_path: str,
    file_id: int | None,
    storage_id: str | None,
    bundle: ParsedFileBundle,
    rate_card: PipelineCostRateCard,
    retention_days: int,
) -> list[PipelineCostLineItem]:
    """Estimate transport/storage cost for materialized artifact handles."""

    artifact_bytes = _estimate_artifact_bytes(bundle)
    if artifact_bytes <= 0:
        return []
    gb_months = (artifact_bytes / (1024.0**3)) * (max(retention_days, 0) / 30.0)
    return [
        _line_item(
            run_id=run_id,
            component="artifact_storage",
            usage_unit="gb_months",
            quantity=gb_months,
            unit_rate=rate_card.artifact_storage_gb_month,
            currency=rate_card.currency,
            confidence="medium",
            file_path=file_path,
            file_id=file_id,
            storage_id=storage_id,
            stage_name="transport",
            meta={
                "artifact_bytes": artifact_bytes,
                "artifact_count": sum(
                    1
                    for handle in bundle.table_inputs.values()
                    if isinstance(handle, ObjectStoreArtifactHandle)
                ),
                "retention_days": retention_days,
            },
        ),
    ]


def build_ingest_cost_line_items(
    *,
    run_id: str,
    file_path: str,
    file_id: int | None,
    storage_id: str | None,
    stage_name: str,
    stage_id: str | None,
    stage_value: Any,
    rate_card: PipelineCostRateCard,
    table_id: str | None = None,
) -> list[PipelineCostLineItem]:
    """Estimate ingest and embedding cost from measured DataManager ingest outcomes."""

    ingest_result = (
        stage_value.get("ingest_result") if isinstance(stage_value, dict) else None
    )
    if ingest_result is None:
        return []
    rows_inserted = int(getattr(ingest_result, "rows_inserted", 0) or 0)
    rows_embedded = int(getattr(ingest_result, "rows_embedded", 0) or 0)
    chunks_processed = int(getattr(ingest_result, "chunks_processed", 0) or 0)
    items: list[PipelineCostLineItem] = []

    if chunks_processed > 0:
        items.append(
            _line_item(
                run_id=run_id,
                component="row_ingest_requests",
                usage_unit="requests",
                quantity=float(chunks_processed),
                unit_rate=rate_card.row_ingest_request,
                currency=rate_card.currency,
                confidence="high",
                file_path=file_path,
                file_id=file_id,
                storage_id=storage_id,
                stage_name=stage_name,
                stage_id=stage_id,
                table_id=table_id,
                meta={"context": getattr(ingest_result, "context", None)},
            ),
        )

    if rows_inserted > 0:
        items.append(
            _line_item(
                run_id=run_id,
                component="row_ingest_storage",
                usage_unit="rows",
                quantity=float(rows_inserted),
                unit_rate=rate_card.row_ingest_row,
                currency=rate_card.currency,
                confidence="high",
                file_path=file_path,
                file_id=file_id,
                storage_id=storage_id,
                stage_name=stage_name,
                stage_id=stage_id,
                table_id=table_id,
                meta={"context": getattr(ingest_result, "context", None)},
            ),
        )

    if rows_embedded > 0:
        items.append(
            _line_item(
                run_id=run_id,
                component="embeddings",
                usage_unit="rows",
                quantity=float(rows_embedded),
                unit_rate=rate_card.embedding_row,
                currency=rate_card.currency,
                confidence="high",
                file_path=file_path,
                file_id=file_id,
                storage_id=storage_id,
                stage_name=stage_name,
                stage_id=stage_id,
                table_id=table_id,
                meta={"context": getattr(ingest_result, "context", None)},
            ),
        )

    return items


def build_observability_cost_line_items(
    *,
    run_id: str,
    rate_card: PipelineCostRateCard,
    progress_event_count: int,
    run_manifest_count: int,
    file_manifest_count: int,
    stage_manifest_count: int,
    cost_ledger_count: int = 1,
    file_path: str | None = None,
    file_id: int | None = None,
    storage_id: str | None = None,
) -> list[PipelineCostLineItem]:
    """Estimate observability cost from emitted event/manifest volume."""

    total_events = (
        max(progress_event_count, 0)
        + max(run_manifest_count, 0)
        + max(file_manifest_count, 0)
        + max(stage_manifest_count, 0)
        + max(cost_ledger_count, 0)
    )
    if total_events <= 0:
        return []
    return [
        _line_item(
            run_id=run_id,
            component="observability",
            usage_unit="events",
            quantity=float(total_events),
            unit_rate=rate_card.observability_event,
            currency=rate_card.currency,
            confidence="low",
            file_path=file_path,
            file_id=file_id,
            storage_id=storage_id,
            meta={
                "progress_event_count": progress_event_count,
                "run_manifest_count": run_manifest_count,
                "file_manifest_count": file_manifest_count,
                "stage_manifest_count": stage_manifest_count,
                "cost_ledger_count": cost_ledger_count,
            },
        ),
    ]


def generate_cost_ledger_path() -> str:
    """Return a timestamped local path for pipeline cost-ledger output."""

    stamp = _utc_now().strftime("%Y%m%d_%H%M%S")
    path = Path("logs/file_manager_runs") / f"cost_ledger_{stamp}.jsonl"
    return str(path.resolve())


def _line_item(
    *,
    run_id: str,
    component: str,
    usage_unit: str,
    quantity: float,
    unit_rate: float,
    currency: str,
    confidence: CostConfidence,
    file_path: str | None = None,
    file_id: int | None = None,
    storage_id: str | None = None,
    stage_name: str | None = None,
    stage_id: str | None = None,
    table_id: str | None = None,
    meta: dict[str, Any] | None = None,
) -> PipelineCostLineItem:
    raw_quantity = float(quantity)
    estimated_cost = round(raw_quantity * float(unit_rate), 12)
    reconciliation_parts = [run_id, file_path or "", stage_name or "", component]
    if table_id:
        reconciliation_parts.append(table_id)
    return PipelineCostLineItem(
        run_id=run_id,
        component=component,
        usage_unit=usage_unit,
        quantity=raw_quantity,
        unit_rate=float(unit_rate),
        estimated_cost=estimated_cost,
        currency=currency,
        confidence=confidence,
        file_path=file_path,
        file_id=file_id,
        storage_id=storage_id,
        stage_name=stage_name,
        stage_id=stage_id,
        table_id=table_id,
        reconciliation_ref="::".join(reconciliation_parts),
        meta=dict(meta or {}),
    )


def _estimate_peak_memory_bytes_for_result(
    *,
    file_path: str,
    parse_result: FileParseResult,
    parse_config: "ParseConfig",
) -> int:
    source_candidates = [
        getattr(getattr(parse_result, "trace", None), "source_local_path", None),
        getattr(getattr(parse_result, "trace", None), "parsed_local_path", None),
        file_path,
    ]
    for candidate in source_candidates:
        if not candidate:
            continue
        try:
            path = Path(str(candidate)).expanduser()
        except Exception:
            continue
        if not path.exists():
            continue
        request = FileParseRequest(
            logical_path=file_path,
            source_local_path=str(path),
        )
        return estimate_peak_memory_bytes(request, config=parse_config)
    return 0


def _estimate_artifact_bytes(bundle: ParsedFileBundle) -> int:
    total = 0
    for handle in bundle.table_inputs.values():
        if not isinstance(handle, ObjectStoreArtifactHandle):
            continue
        total += _resolve_local_size_bytes(handle.storage_uri)
    return total


def _resolve_local_size_bytes(storage_uri: str) -> int:
    if not storage_uri:
        return 0
    parsed = urlparse(storage_uri)
    if parsed.scheme in ("", "file"):
        raw_path = unquote(parsed.path or storage_uri)
        try:
            path = Path(raw_path).expanduser()
        except Exception:
            return 0
        try:
            return path.stat().st_size
        except OSError:
            return 0
    return 0


def _estimate_llm_enrichment_calls(parse_result: FileParseResult) -> int:
    trace = getattr(parse_result, "trace", None)
    if trace is None:
        return 0
    total = 0
    for step in list(getattr(trace, "steps", []) or []):
        counters = getattr(step, "counters", {}) or {}
        for key in ("llm_calls", "summary_calls", "metadata_calls"):
            value = counters.get(key)
            if value:
                total += int(value)
    return total


def _utc_now() -> datetime:
    return datetime.now(UTC)
