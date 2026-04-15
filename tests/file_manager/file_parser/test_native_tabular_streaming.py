from __future__ import annotations

from pathlib import Path

from tests.helpers import _handle_project
from unity.file_manager.file_parsers import FileParseRequest, FileParser, FileFormat
from unity.file_manager.file_parsers.settings import FILE_PARSER_SETTINGS
from unity.file_manager.parse_adapter import adapt_parse_result_for_file_manager
from unity.file_manager.pipeline import (
    CsvFileHandle,
    ObjectStoreArtifactHandle,
    XlsxSheetHandle,
)
from unity.file_manager.pipeline.row_streaming import iter_table_input_row_batches
from unity.file_manager.types.config import FilePipelineConfig


@_handle_project
def test_large_csv_stays_reference_first_and_streams_batches(tmp_path: Path):
    row_count = int(FILE_PARSER_SETTINGS.TABULAR_INLINE_ROW_LIMIT) + 25
    csv_path = tmp_path / "large_people.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        fh.write("Name,Age,City\n")
        for index in range(row_count):
            fh.write(f"Person {index},{20 + (index % 50)},City {index}\n")

    result = FileParser().parse(
        FileParseRequest(logical_path=str(csv_path), source_local_path=str(csv_path)),
    )

    assert result.status == "success"
    assert result.file_format == FileFormat.CSV
    assert result.trace is not None
    assert result.trace.backend == "native_csv_backend"
    assert len(result.tables) == 1

    table = result.tables[0]
    assert table.num_rows == row_count
    assert table.rows == []
    assert len(table.sample_rows) == int(FILE_PARSER_SETTINGS.TABULAR_SAMPLE_ROWS)

    adapted = adapt_parse_result_for_file_manager(result, config=FilePipelineConfig())
    handle = adapted.bundle.table_inputs[table.table_id]
    assert isinstance(handle, CsvFileHandle)
    assert handle.row_count == row_count

    first_batch = next(iter_table_input_row_batches(handle, batch_size=128))
    assert len(first_batch) == 128
    assert first_batch[0]["Name"] == "Person 0"
    assert first_batch[0]["City"] == "City 0"


@_handle_project
def test_large_xlsx_stays_reference_first_and_streams_batches(
    tmp_path: Path,
    write_minimal_xlsx,
):
    row_count = int(FILE_PARSER_SETTINGS.TABULAR_INLINE_ROW_LIMIT) + 10
    rows = [["Name", "Age", "Department"]]
    rows.extend(
        [
            [f"Employee {index}", str(25 + (index % 20)), f"Dept {index % 5}"]
            for index in range(row_count)
        ],
    )

    xlsx_path = tmp_path / "large_people.xlsx"
    write_minimal_xlsx(xlsx_path, sheets=[("Employees", rows)])

    result = FileParser().parse(
        FileParseRequest(logical_path=str(xlsx_path), source_local_path=str(xlsx_path)),
    )

    assert result.status == "success"
    assert result.file_format == FileFormat.XLSX
    assert result.trace is not None
    assert result.trace.backend == "native_excel_backend"
    assert len(result.tables) == 1

    table = result.tables[0]
    assert table.sheet_name == "Employees"
    assert table.num_rows == row_count
    assert table.rows == []
    assert len(table.sample_rows) == int(FILE_PARSER_SETTINGS.TABULAR_SAMPLE_ROWS)

    adapted = adapt_parse_result_for_file_manager(result, config=FilePipelineConfig())
    handle = adapted.bundle.table_inputs[table.table_id]
    assert isinstance(handle, XlsxSheetHandle)
    assert handle.sheet_name == "Employees"
    assert handle.row_count == row_count

    first_batch = next(iter_table_input_row_batches(handle, batch_size=64))
    assert len(first_batch) == 64
    assert first_batch[0]["Name"] == "Employee 0"
    assert first_batch[0]["Department"] == "Dept 0"


@_handle_project
def test_large_csv_can_materialize_to_artifact_handle(tmp_path: Path):
    row_count = int(FILE_PARSER_SETTINGS.TABULAR_INLINE_ROW_LIMIT) + 15
    csv_path = tmp_path / "materialized_people.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        fh.write("Name,Age,City\n")
        for index in range(row_count):
            fh.write(f"Person {index},{30 + (index % 10)},City {index}\n")

    result = FileParser().parse(
        FileParseRequest(logical_path=str(csv_path), source_local_path=str(csv_path)),
    )

    cfg = FilePipelineConfig(
        transport={
            "table_input_mode": "materialized_artifact",
            "artifact_root_dir": str(tmp_path / "artifacts"),
        },
    )
    adapted = adapt_parse_result_for_file_manager(result, config=cfg)

    table = result.tables[0]
    handle = adapted.bundle.table_inputs[table.table_id]
    assert isinstance(handle, ObjectStoreArtifactHandle)
    assert handle.artifact_format == "jsonl"
    assert handle.row_count == row_count

    artifact_path = Path(handle.storage_uri.removeprefix("file://"))
    assert artifact_path.exists()

    first_batch = next(iter_table_input_row_batches(handle, batch_size=128))
    assert len(first_batch) == 128
    assert first_batch[0]["Name"] == "Person 0"
    assert first_batch[0]["City"] == "City 0"
