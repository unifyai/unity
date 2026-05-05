"""Tests for shared pipeline instrumentation and orchestration."""

from __future__ import annotations

import json
import time
from pathlib import Path

from unity.common.pipeline import (
    ArtifactWorkItem,
    PipelineCancelled,
    PipelineInstrumentation,
    ingest_artifacts,
    run_with_retry,
)
from unity.common.pipeline.work_queue import CancellationCheck

# ---------------------------------------------------------------------------
# PipelineInstrumentation lifecycle
# ---------------------------------------------------------------------------


class TestPipelineInstrumentationLifecycle:
    """Verify instrumentation wires up ledgers and produces JSONL correctly."""

    def test_enabled_mode_produces_run_manifests(self, tmp_path: Path):
        from unity.common.pipeline.run_ledger import JsonlRunLedger

        ledger_path = tmp_path / "run_ledger.jsonl"
        ledger = JsonlRunLedger(path=ledger_path)

        instr = PipelineInstrumentation(
            run_id="test-run-001",
            run_ledger=ledger,
            file_count=3,
            parallel_files=True,
            meta={"test_key": "test_val"},
        )

        assert instr.enabled
        assert instr.has_run_ledger
        assert not instr.has_cost_tracking

        with instr:
            instr.record_stage(
                file_path="file_a.csv",
                stage_name="ingest_table",
                status="success",
                duration_ms=42.0,
            )
            instr.record_file(
                file_path="file_a.csv",
                status="success",
                total_duration_ms=100.0,
            )

        records = [
            json.loads(line) for line in ledger_path.read_text().strip().splitlines()
        ]

        assert len(records) == 4
        assert records[0]["record_type"] == "run"
        assert records[0]["status"] == "started"
        assert records[0]["run_id"] == "test-run-001"
        assert records[0]["file_count"] == 3
        assert records[0]["parallel_files"] is True

        assert records[1]["record_type"] == "stage"
        assert records[1]["stage_name"] == "ingest_table"
        assert records[1]["status"] == "success"

        assert records[2]["record_type"] == "file"
        assert records[2]["file_path"] == "file_a.csv"

        assert records[3]["record_type"] == "run"
        assert records[3]["status"] == "completed"

    def test_disabled_mode_is_noop(self):
        instr = PipelineInstrumentation(run_id="noop-run")

        assert not instr.enabled
        assert not instr.has_run_ledger
        assert not instr.has_cost_tracking

        with instr:
            instr.record_stage(
                file_path="x.csv",
                stage_name="test",
                status="success",
            )
            instr.record_file(
                file_path="x.csv",
                status="success",
            )
            instr.add_parse_costs(
                file_path="x.csv",
                parse_duration_seconds=1.0,
            )

    def test_cost_accumulation(self, tmp_path: Path):
        from unity.common.pipeline.cost_ledger import (
            JsonlCostLedger,
            PipelineCostAccumulator,
            PipelineCostRateCard,
        )

        rate_card = PipelineCostRateCard()
        accumulator = PipelineCostAccumulator(
            run_id="cost-run",
            rate_card=rate_card,
        )
        cost_path = tmp_path / "cost_ledger.jsonl"
        cost_ledger = JsonlCostLedger(path=cost_path)

        instr = PipelineInstrumentation(
            run_id="cost-run",
            cost_accumulator=accumulator,
            cost_ledger=cost_ledger,
            file_count=1,
        )

        assert instr.has_cost_tracking
        assert instr.rate_card is rate_card

        with instr:
            instr.add_parse_costs(
                file_path="data.xlsx",
                parse_duration_seconds=5.0,
                estimated_peak_memory_bytes=1024 * 1024 * 100,
                parse_backend="native_xlsx",
            )

        records = [
            json.loads(line) for line in cost_path.read_text().strip().splitlines()
        ]
        assert len(records) == 1
        assert records[0]["record_type"] == "cost_ledger"
        assert records[0]["run_id"] == "cost-run"
        assert len(records[0]["line_items"]) >= 1

    def test_from_config_with_diagnostics_enabled(self, tmp_path: Path):
        """Verify from_config wires up run ledger from config.diagnostics."""

        class _MockDiagnostics:
            enable_run_ledger = True
            run_ledger_file = str(tmp_path / "from_config_ledger.jsonl")

        class _MockCost:
            enable_cost_ledger = False

        class _MockConfig:
            diagnostics = _MockDiagnostics()
            cost = _MockCost()

        instr = PipelineInstrumentation.from_config(
            _MockConfig(),
            run_id="cfg-test",
            file_count=2,
        )
        assert instr.has_run_ledger
        assert not instr.has_cost_tracking

        with instr:
            pass

        records = [
            json.loads(line)
            for line in Path(_MockDiagnostics.run_ledger_file)
            .read_text()
            .strip()
            .splitlines()
        ]
        assert len(records) == 2
        assert records[0]["status"] == "started"
        assert records[1]["status"] == "completed"


class TestMakeStageId:
    def test_generates_deterministic_ids(self):
        instr = PipelineInstrumentation(run_id="run-abc")
        id1 = instr.make_stage_id(
            file_path="file.csv",
            stage_name="ingest_table",
            discriminator="Sheet1",
        )
        id2 = instr.make_stage_id(
            file_path="file.csv",
            stage_name="ingest_table",
            discriminator="Sheet1",
        )
        assert id1 is not None
        assert id1 == id2

    def test_different_discriminators_yield_different_ids(self):
        instr = PipelineInstrumentation(run_id="run-abc")
        id1 = instr.make_stage_id(
            file_path="file.csv",
            stage_name="ingest_table",
            discriminator="Sheet1",
        )
        id2 = instr.make_stage_id(
            file_path="file.csv",
            stage_name="ingest_table",
            discriminator="Sheet2",
        )
        assert id1 != id2


# ---------------------------------------------------------------------------
# ingest_artifacts
# ---------------------------------------------------------------------------


class TestIngestArtifacts:
    def test_dispatches_mixed_content_and_table_items(self):
        call_log: list[str] = []

        def mock_ingest_fn(item: ArtifactWorkItem):
            call_log.append(f"{item.kind}:{item.label}")
            return {"ingest_result": None, "rows_inserted": item.row_count}

        instr = PipelineInstrumentation(run_id="test-artifacts")
        items = [
            ArtifactWorkItem(
                kind="content",
                label="content",
                stage_name="ingest_content",
                row_count=10,
                meta={},
            ),
            ArtifactWorkItem(
                kind="table",
                label="Sheet1",
                stage_name="ingest_table",
                row_count=100,
                table_id="t1",
                meta={},
            ),
            ArtifactWorkItem(
                kind="table",
                label="Sheet2",
                stage_name="ingest_table",
                row_count=50,
                table_id="t2",
                meta={},
            ),
        ]

        results = ingest_artifacts(
            work_items=items,
            ingest_fn=mock_ingest_fn,
            instrumentation=instr,
            source_path="test_file.xlsx",
            max_workers=4,
        )

        assert len(results) == 3
        assert all(r.success for r in results)
        assert set(call_log) == {"content:content", "table:Sheet1", "table:Sheet2"}

    def test_empty_work_items_returns_empty(self):
        instr = PipelineInstrumentation(run_id="empty-test")
        results = ingest_artifacts(
            work_items=[],
            ingest_fn=lambda item: None,
            instrumentation=instr,
            source_path="nothing.csv",
        )
        assert results == []

    def test_failure_is_captured_in_result(self):
        def failing_fn(item: ArtifactWorkItem):
            raise ValueError("bad data")

        instr = PipelineInstrumentation(run_id="fail-test")
        items = [
            ArtifactWorkItem(
                kind="table",
                label="bad_sheet",
                stage_name="ingest_table",
                meta={},
            ),
        ]

        results = ingest_artifacts(
            work_items=items,
            ingest_fn=failing_fn,
            instrumentation=instr,
            source_path="bad.csv",
        )

        assert len(results) == 1
        assert not results[0].success
        assert "bad data" in (results[0].error or "")

    def test_records_stage_manifests(self, tmp_path: Path):
        from unity.common.pipeline.run_ledger import JsonlRunLedger

        ledger_path = tmp_path / "stages.jsonl"
        ledger = JsonlRunLedger(path=ledger_path)

        instr = PipelineInstrumentation(
            run_id="manifest-test",
            run_ledger=ledger,
        )

        items = [
            ArtifactWorkItem(
                kind="table",
                label="Sheet1",
                stage_name="ingest_table",
                meta={"row_count": 42},
            ),
        ]

        with instr:
            ingest_artifacts(
                work_items=items,
                ingest_fn=lambda item: {"ingest_result": None},
                instrumentation=instr,
                source_path="data.csv",
            )

        records = [
            json.loads(line) for line in ledger_path.read_text().strip().splitlines()
        ]
        stage_records = [r for r in records if r["record_type"] == "stage"]
        assert len(stage_records) == 1
        assert stage_records[0]["stage_name"] == "ingest_table"
        assert stage_records[0]["status"] == "success"


# ---------------------------------------------------------------------------
# run_with_retry
# ---------------------------------------------------------------------------


class TestRunWithRetry:
    def test_succeeds_on_first_attempt(self):
        result = run_with_retry(lambda: "ok", {}, label="test")
        assert result.success
        assert result.value == "ok"
        assert result.retries == 0

    def test_retries_transient_errors(self):
        from unity.file_manager.types import RetryConfig

        attempts = {"count": 0}

        def flaky():
            attempts["count"] += 1
            if attempts["count"] < 3:
                raise TimeoutError("timed out")
            return "recovered"

        cfg = RetryConfig(
            max_retries=5,
            retry_delay_seconds=0.0,
            jitter_ratio=0.0,
        )
        result = run_with_retry(flaky, {}, retry_config=cfg, label="flaky")
        assert result.success
        assert result.value == "recovered"
        assert result.retries == 2

    def test_non_retryable_fails_immediately(self):
        from unity.file_manager.types import RetryConfig

        cfg = RetryConfig(
            max_retries=3,
            retry_delay_seconds=0.0,
            jitter_ratio=0.0,
            retry_mode="transient_only",
        )

        def fatal():
            raise ValueError("bad input")

        result = run_with_retry(fatal, {}, retry_config=cfg, label="fatal")
        assert not result.success
        assert result.failure_kind == "non_retryable"


# ---------------------------------------------------------------------------
# Cancellation integration with ingest_artifacts
# ---------------------------------------------------------------------------


class TestIngestArtifactsCancellation:
    """Verify cancellation callback is respected by ingest_artifacts."""

    def test_cancellation_before_first_item_marks_cancelled(self):
        instr = PipelineInstrumentation(run_id="cancel-test")
        items = [
            ArtifactWorkItem(
                kind="table",
                label="Sheet1",
                stage_name="ingest_table",
                meta={},
            ),
        ]

        results = ingest_artifacts(
            work_items=items,
            ingest_fn=lambda item: {"rows": 0},
            instrumentation=instr,
            source_path="test.csv",
            is_cancelled=lambda: True,
        )

        assert len(results) == 1
        assert not results[0].success
        assert results[0].error == "cancelled"

    def test_cancellation_mid_flight_cancels_remaining(self):
        call_count = {"n": 0}

        def slow_ingest(item: ArtifactWorkItem):
            call_count["n"] += 1
            time.sleep(0.01)
            return {"rows": 1}

        cancel_after = {"n": 0}

        def check_cancelled() -> bool:
            cancel_after["n"] += 1
            return cancel_after["n"] > 1

        instr = PipelineInstrumentation(run_id="cancel-mid-test")
        items = [
            ArtifactWorkItem(
                kind="table",
                label=f"t{i}",
                stage_name="ingest_table",
                meta={},
            )
            for i in range(5)
        ]

        results = ingest_artifacts(
            work_items=items,
            ingest_fn=slow_ingest,
            instrumentation=instr,
            source_path="test.xlsx",
            max_workers=1,
            is_cancelled=check_cancelled,
        )

        cancelled_results = [r for r in results if r.error == "cancelled"]
        assert len(cancelled_results) >= 1

    def test_no_cancellation_all_succeed(self):
        instr = PipelineInstrumentation(run_id="no-cancel-test")
        items = [
            ArtifactWorkItem(
                kind="table",
                label=f"t{i}",
                stage_name="ingest_table",
                meta={},
            )
            for i in range(3)
        ]

        results = ingest_artifacts(
            work_items=items,
            ingest_fn=lambda item: {"rows": 10},
            instrumentation=instr,
            source_path="test.csv",
            is_cancelled=lambda: False,
        )

        assert len(results) == 3
        assert all(r.success for r in results)


# ---------------------------------------------------------------------------
# PipelineCancelled and CancellationCheck types
# ---------------------------------------------------------------------------


class TestPipelineCancelledTypes:
    def test_pipeline_cancelled_is_exception(self):
        exc = PipelineCancelled("job X cancelled")
        assert isinstance(exc, Exception)
        assert str(exc) == "job X cancelled"

    def test_cancellation_check_type_alias(self):
        def checker() -> bool:
            return True

        fn: CancellationCheck = checker
        assert fn() is True
