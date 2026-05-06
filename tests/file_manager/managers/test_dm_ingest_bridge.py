"""Regression tests for FileManager helpers that delegate into DataManager.ingest."""

from __future__ import annotations

import pytest

from unity.data_manager.simulated import SimulatedDataManager
from unity.file_manager.managers.utils.ingest_ops import ingest_table_batch


def test_ingest_table_batch_forwards_resume_row_count_validation():
    dm = SimulatedDataManager()

    result = ingest_table_batch(
        data_manager=dm,
        context="test/fm_bridge",
        rows=[{"x": 1}, {"x": 2}, {"x": 3}],
        fields={"x": "int"},
        skip_rows=1,
        expected_total_rows=3,
    )

    assert result.rows_inserted == 2
    assert [row["x"] for row in dm.filter("test/fm_bridge")] == [2, 3]


def test_ingest_table_batch_rejects_resume_row_count_mismatch():
    dm = SimulatedDataManager()

    with pytest.raises(ValueError, match="expected 4"):
        ingest_table_batch(
            data_manager=dm,
            context="test/fm_bridge_mismatch",
            rows=[{"x": 1}, {"x": 2}, {"x": 3}],
            fields={"x": "int"},
            skip_rows=1,
            expected_total_rows=4,
        )

    assert dm.filter("test/fm_bridge_mismatch") == []
