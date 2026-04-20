"""Tests for streaming ingestion via ``table_input_handle`` in SimulatedDataManager.

Verifies that ``dm.ingest(table_input_handle=InlineRowsHandle(...))`` correctly
materializes rows through the streaming path and produces a valid IngestResult.
"""

from __future__ import annotations

import pytest

from unity.common.pipeline.types import InlineRowsHandle
from unity.data_manager.simulated import SimulatedDataManager
from unity.data_manager.types import IngestResult


@pytest.fixture
def dm() -> SimulatedDataManager:
    sdm = SimulatedDataManager()
    yield sdm
    sdm.clear()


class TestStreamingIngestViaInlineRows:

    def test_ingest_creates_table_and_rows(self, dm: SimulatedDataManager):
        handle = InlineRowsHandle(
            rows=[
                {"name": "Alice", "age": 30},
                {"name": "Bob", "age": 25},
            ],
            columns=["name", "age"],
            row_count=2,
        )
        result = dm.ingest(
            "test/people",
            table_input_handle=handle,
            description="People table",
            fields={"name": "str", "age": "int"},
        )
        assert isinstance(result, IngestResult)
        assert result.rows_inserted == 2

        stored = dm.filter("test/people")
        assert len(stored) == 2
        names = {r["name"] for r in stored}
        assert names == {"Alice", "Bob"}

    def test_ingest_with_no_rows_and_no_handle(self, dm: SimulatedDataManager):
        result = dm.ingest(
            "test/empty",
            description="Empty table",
        )
        assert isinstance(result, IngestResult)
        assert result.rows_inserted == 0

    def test_handle_takes_precedence_over_none_rows(self, dm: SimulatedDataManager):
        """When rows=None and table_input_handle is set, handle wins."""
        handle = InlineRowsHandle(
            rows=[{"x": 1}, {"x": 2}, {"x": 3}],
            columns=["x"],
            row_count=3,
        )
        result = dm.ingest(
            "test/nums",
            None,
            table_input_handle=handle,
            fields={"x": "int"},
        )
        assert result.rows_inserted == 3
        stored = dm.filter("test/nums")
        assert [r["x"] for r in stored] == [1, 2, 3]

    def test_empty_handle_ingests_zero_rows(self, dm: SimulatedDataManager):
        handle = InlineRowsHandle(rows=[], columns=["a"], row_count=0)
        result = dm.ingest(
            "test/empty_handle",
            table_input_handle=handle,
            fields={"a": "str"},
        )
        assert isinstance(result, IngestResult)
        assert result.rows_inserted == 0

    def test_multiple_columns_via_handle(self, dm: SimulatedDataManager):
        """Verify multi-column rows are stored correctly via handle."""
        handle = InlineRowsHandle(
            rows=[
                {"a": "x", "b": 1},
                {"a": "y", "b": 2},
            ],
            columns=["a", "b"],
            row_count=2,
        )
        result = dm.ingest(
            "test/multi",
            table_input_handle=handle,
            fields={"a": "str", "b": "int"},
        )
        assert result.rows_inserted == 2
        stored = dm.filter("test/multi")
        assert len(stored) == 2
        assert {r["a"] for r in stored} == {"x", "y"}
