"""Run ledger and cost ledger tests.

Validates JSONL serialization round-trips, rate card line item generation,
accumulator finalization, and cost builder coverage for all components.
"""

from __future__ import annotations

import json


from unity.common.pipeline import (
    JsonlCostLedger,
    JsonlRunLedger,
    PipelineCostAccumulator,
    PipelineCostLedger,
    PipelineCostLineItem,
    PipelineCostRateCard,
    PipelineFileManifest,
    PipelineRunManifest,
    PipelineStageManifest,
    build_observability_cost_line_items,
    build_parse_cost_line_items,
    build_transport_cost_line_items,
)
from unity.common.pipeline.types import (
    InlineRowsHandle,
    ObjectStoreArtifactHandle,
    ParsedFileBundle,
)
from unity.file_manager.file_parsers.types.contracts import FileParseResult


class TestRunLedgerJSONL:

    def test_writes_and_reads_stage_manifest(self, tmp_path):
        ledger_path = tmp_path / "run.jsonl"
        ledger = JsonlRunLedger(path=ledger_path)

        manifest = PipelineStageManifest(
            run_id="run-001",
            file_path="repairs.csv",
            stage_name="parse",
            status="success",
            duration_ms=1234.5,
        )
        ledger.write(manifest)
        ledger.flush()
        ledger.close()

        lines = ledger_path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 1
        loaded = json.loads(lines[0])
        assert loaded["record_type"] == "stage"
        assert loaded["run_id"] == "run-001"
        assert loaded["stage_name"] == "parse"
        assert loaded["status"] == "success"

    def test_writes_file_and_run_manifests(self, tmp_path):
        ledger_path = tmp_path / "run.jsonl"
        ledger = JsonlRunLedger(path=ledger_path)

        ledger.write(
            PipelineFileManifest(
                run_id="run-002",
                file_path="data.xlsx",
                status="success",
                total_duration_ms=5000.0,
            ),
        )
        ledger.write(
            PipelineRunManifest(
                run_id="run-002",
                status="completed",
                file_count=1,
                success_count=1,
            ),
        )
        ledger.close()

        lines = ledger_path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 2
        assert json.loads(lines[0])["record_type"] == "file"
        assert json.loads(lines[1])["record_type"] == "run"


class TestCostLedgerJSONL:

    def test_writes_and_reads_cost_ledger(self, tmp_path):
        ledger_path = tmp_path / "cost.jsonl"
        writer = JsonlCostLedger(path=ledger_path)

        cost = PipelineCostLedger(
            run_id="run-001",
            rate_card_version="test-v1",
            line_items=[
                PipelineCostLineItem(
                    run_id="run-001",
                    component="parse_compute_cpu",
                    usage_unit="cpu_seconds",
                    quantity=10.0,
                    unit_rate=0.000011,
                    estimated_cost=0.00011,
                ),
            ],
            estimated_total=0.00011,
        )
        writer.write(cost)
        writer.close()

        lines = ledger_path.read_text(encoding="utf-8").strip().split("\n")
        loaded = json.loads(lines[0])
        assert loaded["record_type"] == "cost_ledger"
        assert loaded["run_id"] == "run-001"
        assert len(loaded["line_items"]) == 1


class TestParseCostLineItems:

    def test_generates_cpu_and_memory_items(self):
        rate_card = PipelineCostRateCard()
        items = build_parse_cost_line_items(
            run_id="r1",
            file_path="big.csv",
            parse_duration_seconds=30.0,
            estimated_peak_memory_bytes=2 * 1024**3,
            llm_enrichment_calls=0,
            rate_card=rate_card,
        )

        components = {item.component for item in items}
        assert "parse_compute_cpu" in components
        assert "parse_compute_memory" in components

        cpu_item = next(i for i in items if i.component == "parse_compute_cpu")
        assert cpu_item.quantity == 30.0
        assert cpu_item.estimated_cost > 0

    def test_generates_llm_enrichment_item(self):
        rate_card = PipelineCostRateCard()
        items = build_parse_cost_line_items(
            run_id="r1",
            file_path="report.pdf",
            parse_duration_seconds=5.0,
            llm_enrichment_calls=3,
            rate_card=rate_card,
        )

        components = {item.component for item in items}
        assert "llm_enrichment" in components
        llm_item = next(i for i in items if i.component == "llm_enrichment")
        assert llm_item.quantity == 3.0

    def test_skips_zero_duration(self):
        rate_card = PipelineCostRateCard()
        items = build_parse_cost_line_items(
            run_id="r1",
            file_path="empty.csv",
            parse_duration_seconds=0.0,
            rate_card=rate_card,
        )
        assert items == []


class TestTransportCostLineItems:

    def test_generates_artifact_storage_item(self, tmp_path):
        artifact_path = tmp_path / "artifact.jsonl"
        artifact_path.write_text('{"a":1}\n{"a":2}\n', encoding="utf-8")

        rate_card = PipelineCostRateCard()
        bundle = ParsedFileBundle(
            result=FileParseResult(logical_path="data.csv", status="success"),
            table_inputs={
                "t1": ObjectStoreArtifactHandle(
                    storage_uri=artifact_path.as_uri(),
                    logical_path="data.csv",
                    artifact_format="jsonl",
                    columns=["a"],
                    row_count=2,
                ),
            },
        )

        items = build_transport_cost_line_items(
            run_id="r1",
            file_path="data.csv",
            file_id=None,
            storage_id=None,
            bundle=bundle,
            rate_card=rate_card,
            retention_days=30,
        )

        assert len(items) == 1
        assert items[0].component == "artifact_storage"
        assert items[0].estimated_cost > 0

    def test_skips_when_no_object_store_handles(self):
        rate_card = PipelineCostRateCard()
        bundle = ParsedFileBundle(
            result=FileParseResult(logical_path="data.csv", status="success"),
            table_inputs={
                "t1": InlineRowsHandle(
                    rows=[{"a": 1}],
                    columns=["a"],
                ),
            },
        )

        items = build_transport_cost_line_items(
            run_id="r1",
            file_path="data.csv",
            file_id=None,
            storage_id=None,
            bundle=bundle,
            rate_card=rate_card,
            retention_days=30,
        )

        assert items == []


class TestObservabilityCostLineItems:

    def test_generates_observability_item(self):
        rate_card = PipelineCostRateCard()
        items = build_observability_cost_line_items(
            run_id="r1",
            rate_card=rate_card,
            progress_event_count=100,
            run_manifest_count=1,
            file_manifest_count=5,
            stage_manifest_count=15,
        )

        assert len(items) == 1
        assert items[0].component == "observability"
        assert items[0].quantity == 100 + 1 + 5 + 15 + 1

    def test_skips_zero_events(self):
        rate_card = PipelineCostRateCard()
        items = build_observability_cost_line_items(
            run_id="r1",
            rate_card=rate_card,
            progress_event_count=0,
            run_manifest_count=0,
            file_manifest_count=0,
            stage_manifest_count=0,
            cost_ledger_count=0,
        )
        assert items == []


class TestCostAccumulator:

    def test_accumulates_and_finalizes_ledger(self):
        rate_card = PipelineCostRateCard()
        acc = PipelineCostAccumulator(run_id="r1", rate_card=rate_card)

        parse_items = build_parse_cost_line_items(
            run_id="r1",
            file_path="data.csv",
            parse_duration_seconds=10.0,
            estimated_peak_memory_bytes=1024**3,
            rate_card=rate_card,
        )
        acc.add_line_items(parse_items)

        obs_items = build_observability_cost_line_items(
            run_id="r1",
            rate_card=rate_card,
            progress_event_count=50,
            run_manifest_count=1,
            file_manifest_count=1,
            stage_manifest_count=3,
        )
        acc.add_line_items(obs_items)

        ledger = acc.build_ledger()
        assert ledger.run_id == "r1"
        assert ledger.rate_card_version == rate_card.version
        assert len(ledger.line_items) == len(parse_items) + len(obs_items)
        assert ledger.estimated_total > 0
        assert ledger.reconciled_total is None

    def test_filters_zero_quantity_items(self):
        rate_card = PipelineCostRateCard()
        acc = PipelineCostAccumulator(run_id="r1", rate_card=rate_card)

        acc.add_line_items(
            [
                PipelineCostLineItem(
                    run_id="r1",
                    component="empty",
                    usage_unit="units",
                    quantity=0.0,
                    unit_rate=1.0,
                    estimated_cost=0.0,
                ),
            ],
        )

        ledger = acc.build_ledger()
        assert len(ledger.line_items) == 0
        assert ledger.estimated_total == 0.0

    def test_rate_card_from_config_defaults(self):
        from pydantic import BaseModel

        class FakeRateCard(BaseModel):
            parse_cpu_per_second: float = 0.00005

        class FakeCostConfig(BaseModel):
            rate_card: FakeRateCard = FakeRateCard()

        card = PipelineCostRateCard.from_config(FakeCostConfig())
        assert card.parse_cpu_per_second == 0.00005
        assert card.artifact_storage_gb_month == 0.020
