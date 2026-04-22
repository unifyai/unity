"""Tests for pipeline transport handle construction and materialization gating."""

from __future__ import annotations

from unity.common.pipeline.transport import _can_materialize, build_table_handles
from unity.common.pipeline.types import (
    CsvFileHandle,
    InlineRowsHandle,
    ObjectStoreArtifactHandle,
    XlsxSheetHandle,
)

# ---------------------------------------------------------------------------
# _can_materialize gating
# ---------------------------------------------------------------------------


class TestCanMaterialize:
    """Verify materialization is skipped for handles backed by GCS."""

    def test_inline_rows_with_data_is_materializable(self):
        handle = InlineRowsHandle(rows=[{"a": 1}], columns=["a"], row_count=1)
        assert _can_materialize(handle) is True

    def test_inline_rows_empty_is_not_materializable(self):
        handle = InlineRowsHandle(rows=[], columns=["a"], row_count=0)
        assert _can_materialize(handle) is False

    def test_object_store_handle_is_not_materializable(self):
        handle = ObjectStoreArtifactHandle(
            storage_uri="gs://bucket/artifact.jsonl",
            artifact_format="jsonl",
            source_local_path="",
            logical_path="data.csv",
        )
        assert _can_materialize(handle) is False

    def test_csv_with_local_uri_is_materializable(self):
        handle = CsvFileHandle(
            storage_uri="file:///tmp/data.csv",
            logical_path="data.csv",
            source_local_path="/tmp/data.csv",
            columns=["a", "b"],
        )
        assert _can_materialize(handle) is True

    def test_csv_with_gs_uri_is_not_materializable(self):
        handle = CsvFileHandle(
            storage_uri="gs://bucket/data.csv",
            logical_path="data.csv",
            source_local_path="/tmp/data.csv",
            columns=["a", "b"],
        )
        assert _can_materialize(handle) is False

    def test_xlsx_with_local_uri_is_materializable(self):
        handle = XlsxSheetHandle(
            storage_uri="file:///tmp/data.xlsx",
            logical_path="data.xlsx",
            source_local_path="/tmp/data.xlsx",
            sheet_name="Sheet1",
            columns=["a", "b"],
        )
        assert _can_materialize(handle) is True

    def test_xlsx_with_gs_uri_is_not_materializable(self):
        handle = XlsxSheetHandle(
            storage_uri="gs://bucket/data.xlsx",
            logical_path="data.xlsx",
            source_local_path="/tmp/data.xlsx",
            sheet_name="Sheet1",
            columns=["a", "b"],
        )
        assert _can_materialize(handle) is False


# ---------------------------------------------------------------------------
# build_table_handles with source_gs_uri
# ---------------------------------------------------------------------------


class TestBuildTableHandlesSourceGsUri:
    """Verify source_gs_uri flows into handle storage_uri fields."""

    def _make_parse_result(
        self,
        *,
        tables=None,
        file_format=None,
        status="success",
    ):
        """Minimal duck-typed parse result for testing handle construction."""

        class _Table:
            def __init__(self, *, rows=None, columns=None, table_id=None, **kw):
                self.rows = rows or []
                self.columns = columns or []
                self.table_id = table_id
                self.num_rows = len(self.rows) if self.rows else 0
                for k, v in kw.items():
                    setattr(self, k, v)

        class _Result:
            def __init__(self):
                self.tables = tables or []
                self.file_format = file_format
                self.status = status
                self.logical_path = "data.csv"
                self.trace = None

        return _Result(), _Table

    def test_no_source_gs_uri_uses_local_path(self):
        result, Table = self._make_parse_result(
            tables=[],
        )
        handles = build_table_handles(result)
        assert handles == {}

    def test_inline_rows_ignore_source_gs_uri(self):
        """InlineRowsHandle does not carry storage_uri, so source_gs_uri
        has no effect on them."""

        class _Table:
            table_id = "t1"
            rows = [{"a": 1}]
            columns = ["a"]
            num_rows = 1

        class _Result:
            tables = [_Table()]
            file_format = None
            status = "success"
            logical_path = "data.csv"
            trace = None

        handles = build_table_handles(
            _Result(),
            source_gs_uri="gs://bucket/data.csv",
        )
        assert len(handles) == 1
        handle = handles["t1"]
        assert isinstance(handle, InlineRowsHandle)
