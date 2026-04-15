from __future__ import annotations

from pathlib import Path

import pytest

from tests.helpers import _handle_project
from unity.file_manager.file_parsers import FileParseRequest, FileParser, FileFormat
from unity.file_manager.file_parsers.settings import FILE_PARSER_SETTINGS


def _assert_non_empty(s: str, *, label: str) -> None:
    assert isinstance(s, str), f"{label} must be str"
    assert s.strip(), f"{label} must be non-empty"


@_handle_project
def test_csv_employee_records_extracts_expected_schema_and_rows(sample_file):
    p = sample_file("employee_records.csv")
    logical = "sample/employee_records.csv"

    res = FileParser().parse(
        FileParseRequest(logical_path=logical, source_local_path=str(p)),
    )

    assert res.status == "success"
    assert res.file_format == FileFormat.CSV
    assert res.trace is not None
    assert res.trace.backend == "native_csv_backend"

    # CSV is parsed as a single-sheet spreadsheet; full_text is bounded profile text
    _assert_non_empty(res.full_text, label="full_text")
    assert "Spreadsheet:" in res.full_text
    _assert_non_empty(res.summary, label="summary")

    assert res.metadata is not None
    _assert_non_empty(res.metadata.key_topics, label="metadata.key_topics")
    _assert_non_empty(res.metadata.content_tags, label="metadata.content_tags")
    assert res.metadata.confidence_score is not None

    assert len(res.tables) >= 1
    tbl = res.tables[0]
    assert tbl.columns == [
        "EmployeeID",
        "Name",
        "Department",
        "HireDate",
        "Salary",
        "FullTime",
    ]
    assert tbl.num_rows == 6
    assert tbl.num_cols == 6

    all_values = [str(v) for r in tbl.rows for v in r.values() if v is not None]
    assert any("Alice Johnson" in v for v in all_values)
    assert any("Bob Smith" in v for v in all_values)
    assert any("Engineering" in v for v in all_values)


@pytest.mark.parametrize(
    ("name", "content", "expected_cols", "expected_rows", "expected_value_substrings"),
    [
        (
            "semicolon.csv",
            "Name;Age;City\nAlice;30;Paris\nBob;25;Berlin\nCharlie;35;Madrid\n",
            ["Name", "Age", "City"],
            3,
            ["Alice", "Paris", "Berlin", "Madrid"],
        ),
        (
            "pipe.csv",
            "Name|Department|Salary\nJohn|Engineering|95000\nMary|Sales|75000\nSteve|HR|65000\n",
            ["Name", "Department", "Salary"],
            3,
            ["John", "Engineering", "95000", "Mary", "Sales", "75000"],
        ),
        (
            "unicode.csv",
            "Name,City,Greeting\nJosé,São Paulo,Olá\nFrançois,Montréal,Bonjour\n李明,北京,你好\n",
            ["Name", "City", "Greeting"],
            3,
            ["José", "São Paulo", "François", "Montréal", "李明", "北京"],
        ),
        (
            "empty_cells.csv",
            "Name,Age,City,Country\nJohn,30,,USA\nJane,,London,\nBob,35,Sydney,Australia\n,28,Toronto,Canada\n",
            ["Name", "Age", "City", "Country"],
            4,
            ["John", "USA", "London", "Sydney", "Australia", "Toronto", "Canada"],
        ),
    ],
)
@_handle_project
def test_csv_delimiters_unicode_and_empty_cells(
    tmp_path: Path,
    name: str,
    content: str,
    expected_cols: list[str],
    expected_rows: int,
    expected_value_substrings: list[str],
):
    p = tmp_path / name
    p.write_text(content, encoding="utf-8")

    res = FileParser().parse(
        FileParseRequest(logical_path=str(p), source_local_path=str(p)),
    )

    assert res.status == "success"
    assert res.file_format == FileFormat.CSV
    assert len(res.tables) == 1

    tbl = res.tables[0]
    assert tbl.columns == expected_cols
    assert tbl.num_rows == expected_rows
    assert tbl.num_cols == len(expected_cols)

    values = {str(v) for r in tbl.rows for v in r.values() if v is not None}
    for needle in expected_value_substrings:
        assert any(needle in v for v in values), f"Expected '{needle}' in parsed values"


@_handle_project
def test_csv_complex_quotes_and_commas(tmp_path: Path):
    """
    Port of legacy `test_csv_complex`: quoted fields + commas should parse correctly.
    """
    p = tmp_path / "complex.csv"
    p.write_text(
        "Name,Department,Notes\n"
        '"Doe, John","R&D","Joined 2020, promoted 2022"\n'
        '"Smith, Jane","Sales","Top performer, Q4"\n',
        encoding="utf-8",
    )

    res = FileParser().parse(
        FileParseRequest(logical_path=str(p), source_local_path=str(p)),
    )
    assert res.status == "success"
    assert res.file_format == FileFormat.CSV
    assert len(res.tables) == 1
    tbl = res.tables[0]
    assert tbl.columns == ["Name", "Department", "Notes"]
    assert tbl.num_rows == 2

    values = {str(v) for r in tbl.rows for v in r.values() if v is not None}
    assert any("Doe, John" == v for v in values)
    assert any("Smith, Jane" == v for v in values)
    assert any("R&D" == v for v in values)


@_handle_project
def test_spreadsheet_full_text_is_bounded_for_embedding_budget(sample_file):
    p = sample_file("project_status.xlsx")

    res = FileParser().parse(
        FileParseRequest(
            logical_path="sample/project_status.xlsx",
            source_local_path=str(p),
        ),
    )
    assert res.status == "success"
    assert res.file_format == FileFormat.XLSX

    # Bounded profile text should fit within the embedding input budget.
    from unity.common.token_utils import conservative_token_estimate

    est = conservative_token_estimate(
        res.full_text,
        FILE_PARSER_SETTINGS.EMBEDDING_ENCODING,
    )
    assert est <= FILE_PARSER_SETTINGS.EMBEDDING_MAX_INPUT_TOKENS


@_handle_project
def test_xlsx_project_status_extracts_tables_and_has_bounded_profile_text(sample_file):
    p = sample_file("project_status.xlsx")
    logical = "sample/project_status.xlsx"

    res = FileParser().parse(
        FileParseRequest(logical_path=logical, source_local_path=str(p)),
    )

    assert res.status == "success"
    assert res.file_format == FileFormat.XLSX
    assert res.trace is not None
    assert res.trace.backend == "native_excel_backend"

    _assert_non_empty(res.full_text, label="full_text")
    assert "Spreadsheet:" in res.full_text
    _assert_non_empty(res.summary, label="summary")

    assert res.metadata is not None
    _assert_non_empty(res.metadata.key_topics, label="metadata.key_topics")
    _assert_non_empty(res.metadata.content_tags, label="metadata.content_tags")

    assert len(res.tables) >= 1
    assert res.tables[0].columns, "Expected non-empty column list"
    assert (res.tables[0].num_rows or 0) > 0


@_handle_project
def test_xlsx_table_labels_are_unique_within_file(sample_file):
    p = sample_file("retail_data.xlsx")
    res = FileParser().parse(
        FileParseRequest(
            logical_path="sample/retail_data.xlsx",
            source_local_path=str(p),
        ),
    )
    assert res.status == "success"
    assert res.file_format == FileFormat.XLSX

    labels = [str(t.label) for t in res.tables]
    assert len(labels) == len(set(labels)), "Expected unique table labels within a file"


@_handle_project
def test_xlsx_multiple_sheets_generated_without_openpyxl(
    tmp_path: Path,
    write_minimal_xlsx,
):
    p = tmp_path / "multi_sheet.xlsx"
    write_minimal_xlsx(
        p,
        sheets=[
            (
                "Q1 Sales",
                [["Product", "Revenue"], ["Widget", "50000"], ["Gadget", "75000"]],
            ),
            (
                "Q1 Expenses",
                [["Category", "Amount"], ["Salaries", "30000"], ["Marketing", "15000"]],
            ),
        ],
    )

    res = FileParser().parse(
        FileParseRequest(logical_path=str(p), source_local_path=str(p)),
    )
    assert res.status == "success"
    assert res.file_format == FileFormat.XLSX
    assert len(res.tables) >= 2

    sheet_names = {str(t.sheet_name or "") for t in res.tables}
    assert "Q1 Sales" in sheet_names
    assert "Q1 Expenses" in sheet_names

    values = {
        str(v) for t in res.tables for r in t.rows for v in r.values() if v is not None
    }
    for needle in [
        "Widget",
        "Gadget",
        "Salaries",
        "Marketing",
        "50000",
        "75000",
        "30000",
        "15000",
    ]:
        assert needle in values


@_handle_project
def test_xlsx_with_formulas_generated_without_openpyxl(
    tmp_path: Path,
    write_minimal_xlsx,
):
    # We do not require formula evaluation; we require parsing succeeds and inputs are present.
    p = tmp_path / "formulas.xlsx"
    write_minimal_xlsx(
        p,
        sheets=[
            (
                "Calculations",
                [
                    ["Value1", "Value2", "Sum"],
                    ["100", "200", "=A2+B2"],
                    ["50", "75", "=A3+B3"],
                ],
            ),
        ],
    )

    res = FileParser().parse(
        FileParseRequest(logical_path=str(p), source_local_path=str(p)),
    )
    assert res.status == "success"
    assert res.file_format == FileFormat.XLSX
    assert len(res.tables) >= 1

    values = {
        str(v) for t in res.tables for r in t.rows for v in r.values() if v is not None
    }
    assert "100" in values
    assert "200" in values
    assert "50" in values
    assert "75" in values


@_handle_project
def test_xlsx_metadata_extraction_generated_without_openpyxl(
    tmp_path: Path,
    write_minimal_xlsx,
):
    p = tmp_path / "metadata_test.xlsx"
    write_minimal_xlsx(
        p,
        sheets=[
            ("Sheet 1", [["Name", "Value"], ["Test", "123"]]),
        ],
    )

    res = FileParser().parse(
        FileParseRequest(logical_path=str(p), source_local_path=str(p)),
    )
    assert res.status == "success"
    assert res.file_format == FileFormat.XLSX
    assert res.metadata is not None
    _assert_non_empty(res.metadata.key_topics, label="metadata.key_topics")
    _assert_non_empty(res.metadata.content_tags, label="metadata.content_tags")


@pytest.mark.parametrize(
    ("fname", "expected_sheet_names"),
    [
        ("workforce_data.xlsx", ["Employees", "Attendance", "Salaries"]),
        ("retail_data.xlsx", ["Stores", "Sales", "Inventory", "Returns"]),
    ],
)
@_handle_project
def test_xlsx_multi_sheet_samples_surface_sheet_names_in_tables(
    fname: str,
    expected_sheet_names: list[str],
    sample_file,
):
    p = sample_file(fname)
    logical = f"sample/{fname}"

    res = FileParser().parse(
        FileParseRequest(logical_path=logical, source_local_path=str(p)),
    )

    assert res.status == "success"
    assert res.file_format == FileFormat.XLSX
    assert len(res.tables) >= len(expected_sheet_names)

    got = {str(t.sheet_name or "").strip().lower() for t in res.tables}
    for s in expected_sheet_names:
        assert s.strip().lower() in got


@_handle_project
def test_workforce_data_xlsx_values_present_in_expected_sheets(sample_file):
    p = sample_file("workforce_data.xlsx")
    res = FileParser().parse(
        FileParseRequest(
            logical_path="sample/workforce_data.xlsx",
            source_local_path=str(p),
        ),
    )
    assert res.status == "success"
    assert res.file_format == FileFormat.XLSX
    assert len(res.tables) >= 3

    by_sheet: dict[str, list[dict]] = {}
    for t in res.tables:
        if t.sheet_name:
            by_sheet.setdefault(t.sheet_name, []).extend(list(t.rows or []))

    for s in ("Employees", "Attendance", "Salaries"):
        assert s in by_sheet, f"Missing sheet table: {s}"

    employees_values = {
        str(v) for r in by_sheet["Employees"] for v in r.values() if v is not None
    }
    for name in [
        "Aria Patel",
        "Bilal Khan",
        "Chen Li",
        "Diego Reyes",
        "Emma Novak",
        "Farah Qureshi",
    ]:
        assert name in employees_values
    for emp_id in ["301", "302", "303", "304", "305", "306"]:
        assert emp_id in employees_values
    for dept in ["Engineering", "Design", "Sales", "Finance", "HR"]:
        assert dept in employees_values

    attendance_values = {
        str(v) for r in by_sheet["Attendance"] for v in r.values() if v is not None
    }
    for status in ["Present", "Absent", "Remote"]:
        assert any(status.lower() == str(v).strip().lower() for v in attendance_values)

    salaries_values = {
        str(v) for r in by_sheet["Salaries"] for v in r.values() if v is not None
    }
    for salary in ["98000", "105000", "86000", "45000", "92000", "52000"]:
        assert any(salary in str(v).replace(",", "") for v in salaries_values)
    for bonus in ["10000", "15000", "8000", "2000", "12000", "3000"]:
        assert any(bonus in str(v).replace(",", "") for v in salaries_values)


@_handle_project
def test_retail_data_xlsx_values_present_in_expected_sheets(sample_file):
    p = sample_file("retail_data.xlsx")
    res = FileParser().parse(
        FileParseRequest(
            logical_path="sample/retail_data.xlsx",
            source_local_path=str(p),
        ),
    )
    assert res.status == "success"
    assert res.file_format == FileFormat.XLSX
    assert len(res.tables) >= 4

    by_sheet: dict[str, list[dict]] = {}
    for t in res.tables:
        if t.sheet_name:
            by_sheet.setdefault(t.sheet_name, []).extend(list(t.rows or []))

    for s in ("Stores", "Sales", "Inventory", "Returns"):
        assert s in by_sheet, f"Missing sheet table: {s}"

    stores_values = {
        str(v) for r in by_sheet["Stores"] for v in r.values() if v is not None
    }
    for name in ["Gulshan", "DHA", "Blue Area", "Saddar"]:
        assert any(name.lower() == str(v).strip().lower() for v in stores_values)
    for city in ["Karachi", "Lahore", "Islamabad", "Rawalpindi"]:
        assert any(city.lower() == str(v).strip().lower() for v in stores_values)

    sales_values = {
        str(v) for r in by_sheet["Sales"] for v in r.values() if v is not None
    }
    for sale_id in ["5001", "5002", "5003", "5004", "5005", "5006"]:
        assert sale_id in sales_values
    for sku in ["LTP-15", "MOU-01", "KBD-02", "MON-27", "PRN-10"]:
        assert any(sku.lower() == str(v).strip().lower() for v in sales_values)

    inventory_values = {
        str(v) for r in by_sheet["Inventory"] for v in r.values() if v is not None
    }
    for item in ["Laptop", "Mouse", "Keyboard", "Monitor", "Printer"]:
        assert any(item.lower() in str(v).strip().lower() for v in inventory_values)

    returns_values = {
        str(v) for r in by_sheet["Returns"] for v in r.values() if v is not None
    }
    for rid in ["9001", "9002"]:
        assert rid in returns_values
    assert any("defective" in str(v).lower() for v in returns_values)
    assert any("damaged" in str(v).lower() for v in returns_values)
