"""Artifact store tests for LocalArtifactStore.

Validates materialization from InlineRowsHandle, CsvFileHandle, and
ObjectStoreArtifactHandle round-trip, plus format rejection.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from unity.common.pipeline import (
    CsvFileHandle,
    InlineRowsHandle,
    LocalArtifactStore,
    ObjectStoreArtifactHandle,
)


class TestInlineRowsMaterialization:

    def test_materializes_inline_rows_to_jsonl(self, tmp_path):
        store = LocalArtifactStore(root_dir=tmp_path)
        handle = InlineRowsHandle(
            rows=[
                {"name": "Alice", "age": 30},
                {"name": "Bob", "age": 25},
            ],
            columns=["name", "age"],
            row_count=2,
        )

        result = store.materialize_table_input(
            handle,
            logical_path="people.csv",
            table_id="main",
            artifact_format="jsonl",
        )

        assert isinstance(result, ObjectStoreArtifactHandle)
        assert result.artifact_format == "jsonl"
        assert result.row_count == 2
        assert result.columns == ["name", "age"]

        artifact_path = Path(result.storage_uri.removeprefix("file://"))
        assert artifact_path.exists()
        lines = artifact_path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 2
        assert json.loads(lines[0]) == {"name": "Alice", "age": 30}

    def test_materializes_empty_inline_rows(self, tmp_path):
        store = LocalArtifactStore(root_dir=tmp_path)
        handle = InlineRowsHandle(rows=[], columns=["x"], row_count=0)

        result = store.materialize_table_input(
            handle,
            logical_path="empty.csv",
            table_id="sheet1",
            artifact_format="jsonl",
        )

        assert result.row_count == 0
        artifact_path = Path(result.storage_uri.removeprefix("file://"))
        assert artifact_path.read_text(encoding="utf-8").strip() == ""

    def test_columns_inferred_from_first_row_when_not_provided(self, tmp_path):
        store = LocalArtifactStore(root_dir=tmp_path)
        handle = InlineRowsHandle(
            rows=[{"x": 1, "y": 2}, {"x": 3, "y": 4}],
            columns=[],
        )

        result = store.materialize_table_input(
            handle,
            logical_path="data.csv",
            table_id="t1",
            artifact_format="jsonl",
        )

        assert result.columns == ["x", "y"]


class TestCsvFileHandleMaterialization:

    def test_materializes_csv_file_to_jsonl(self, tmp_path):
        csv_file = tmp_path / "source.csv"
        csv_file.write_text("name,score\nAlice,95\nBob,87\n", encoding="utf-8")

        store = LocalArtifactStore(root_dir=tmp_path / "artifacts")
        handle = CsvFileHandle(
            storage_uri=csv_file.as_uri(),
            logical_path="source.csv",
            source_local_path=str(csv_file),
            columns=["name", "score"],
            encoding="utf-8",
            delimiter=",",
            has_header=True,
        )

        result = store.materialize_table_input(
            handle,
            logical_path="source.csv",
            table_id="main",
            artifact_format="jsonl",
        )

        assert result.row_count == 2
        assert result.artifact_format == "jsonl"
        artifact_path = Path(result.storage_uri.removeprefix("file://"))
        lines = artifact_path.read_text(encoding="utf-8").strip().split("\n")
        rows = [json.loads(line) for line in lines]
        assert rows[0]["name"] == "Alice"
        assert rows[1]["name"] == "Bob"


class TestObjectStoreHandlePassthrough:

    def test_already_materialized_handle_is_returned_as_is(self, tmp_path):
        store = LocalArtifactStore(root_dir=tmp_path)
        existing = ObjectStoreArtifactHandle(
            storage_uri="gs://bucket/artifact.jsonl",
            logical_path="data.csv",
            artifact_format="jsonl",
            columns=["a", "b"],
            row_count=100,
        )

        result = store.materialize_table_input(
            existing,
            logical_path="data.csv",
            table_id="t1",
            artifact_format="jsonl",
        )

        assert result is existing


class TestFormatRejection:

    def test_rejects_unsupported_artifact_format(self, tmp_path):
        store = LocalArtifactStore(root_dir=tmp_path)
        handle = InlineRowsHandle(
            rows=[{"a": 1}],
            columns=["a"],
        )

        with pytest.raises(ValueError, match="Unsupported artifact format"):
            store.materialize_table_input(
                handle,
                logical_path="data.csv",
                table_id="t1",
                artifact_format="parquet",
            )


class TestJSONLRoundTrip:
    """Verify that materialized JSONL artifacts can be read back through ObjectStoreArtifactHandle."""

    def test_jsonl_round_trip_through_row_streaming(self, tmp_path):
        from unity.file_manager.parse_adapter.row_streaming import (
            iter_table_input_rows,
        )

        store = LocalArtifactStore(root_dir=tmp_path)
        original_rows = [
            {"id": 1, "city": "London", "pop": 9000000},
            {"id": 2, "city": "Birmingham", "pop": 1150000},
            {"id": 3, "city": "Manchester", "pop": 550000},
        ]
        handle = InlineRowsHandle(
            rows=original_rows,
            columns=["id", "city", "pop"],
        )

        artifact = store.materialize_table_input(
            handle,
            logical_path="cities.csv",
            table_id="cities",
            artifact_format="jsonl",
        )

        read_back = list(iter_table_input_rows(artifact))
        assert len(read_back) == 3
        assert read_back[0]["city"] == "London"
        assert read_back[2]["pop"] == 550000
