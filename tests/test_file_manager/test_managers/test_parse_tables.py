"""
Per-table ingestion tests for spreadsheets (CSV/XLSX with multi-tab).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from tests.helpers import _handle_project
from unity.file_manager.types import (
    FilePipelineConfig,
    TableBusinessContextSpec,
    FileBusinessContextSpec,
    BusinessContextsConfig,
)


@pytest.mark.asyncio
@_handle_project
async def test_csv_per_table_context(file_manager, tmp_path: Path):
    # Create a simple CSV
    csv_path = tmp_path / "people.csv"
    csv_path.write_text(
        "Name,Age,City\nJohn,30,NYC\nJane,25,LDN\n",
        encoding="utf-8",
    )
    display_name = str(csv_path)

    # Parse file with business context to enrich table/column descriptions
    cfg = FilePipelineConfig()
    cfg.ingest.business_contexts = BusinessContextsConfig(
        global_rules=[],
        file_contexts=[
            FileBusinessContextSpec(
                file_path=display_name,
                file_rules=[],
                table_contexts=[
                    TableBusinessContextSpec(
                        table="people",  # CSV table name derived from filename
                        table_rules=[],
                        column_descriptions={
                            "Name": "Person's full name",
                            "Age": "Person's age in years",
                            "City": "City where the person lives",
                        },
                        table_description="People directory with basic contact information",
                    ),
                ],
            ),
        ],
    )

    result = file_manager.ingest_files(display_name, config=cfg)
    _item = result[display_name]
    _item = _item if isinstance(_item, dict) else _item.model_dump()
    assert _item["status"] == "success"

    # Verify a per-table context exists
    import unify

    ctxs = unify.get_contexts()
    table_ctx_candidates = [
        name for name in ctxs.keys() if "/Tables/" in name and "people" in name
    ]
    assert table_ctx_candidates, "No table context candidates found"

    # Optionally, fetch a few rows from one table context
    sample_ctx = table_ctx_candidates[0]
    rows = unify.get_logs(context=sample_ctx, limit=10)
    # Should have header-derived columns
    assert rows, "No rows found"
    assert set(["Name", "Age", "City"]).issubset(
        set(rows[0].entries.keys()),
    ), "Columns not found"
    assert len(rows) > 0, "No rows found"
    assert len(rows[0].entries) == 4, "Columns not found"
    assert len(rows[1].entries) == 4, "Columns not found"
    assert "row_id" in rows[0].entries
    assert "row_id" in rows[1].entries
    assert rows[0].entries["Name"] == "Jane", "Name not found"
    assert rows[0].entries["Age"] == "25", "Age not found"
    assert rows[0].entries["City"] == "LDN", "City not found"
    assert rows[1].entries["Name"] == "John", "Name not found"
    assert rows[1].entries["Age"] == "30", "Age not found"
    assert rows[1].entries["City"] == "NYC", "City not found"


@pytest.mark.asyncio
@_handle_project
async def test_xlsx_multi_tab_per_table_context(file_manager, tmp_path: Path):
    # Use sample multi-tab workbooks from tests
    sample_dir = Path(__file__).parents[1] / "sample"
    retail = sample_dir / "retail_data.xlsx"
    workforce = sample_dir / "workforce_data.xlsx"

    # Create config file with business contexts for multiple tables
    config_data = {
        "ingest": {
            "business_contexts": {
                "global_rules": [],
                "file_contexts": [
                    {
                        "file_path": str(retail) if retail.exists() else "",
                        "file_rules": [],
                        "table_contexts": [
                            {
                                "table": "Sales",
                                "table_rules": [],
                                "column_descriptions": {
                                    "Product": "Product name or SKU",
                                    "Quantity": "Number of units sold",
                                },
                                "table_description": "Sales transactions data",
                            },
                        ],
                    },
                    {
                        "file_path": str(workforce) if workforce.exists() else "",
                        "file_rules": [],
                        "table_contexts": [
                            {
                                "table": "Employees",
                                "table_rules": [],
                                "column_descriptions": {
                                    "Name": "Employee full name",
                                    "Department": "Department assignment",
                                },
                                "table_description": "Employee directory",
                            },
                        ],
                    },
                ],
            },
        },
    }
    config_file = tmp_path / "multi_table_config.json"
    config_file.write_text(json.dumps(config_data))

    cfg = FilePipelineConfig.from_file(str(config_file))

    for path in [retail, workforce]:
        if path.exists():
            display_name = str(path)
            # Use config with business context
            res = file_manager.ingest_files(display_name, config=cfg)
            _item = res[display_name]
            _item = _item if isinstance(_item, dict) else _item.model_dump()
            assert _item["status"] == "success"
        else:
            print(f"Path {path} does not exist")
            assert False

    # Verify multiple per-table contexts exist (one per tab)
    import unify

    ctxs = unify.get_contexts()
    table_ctx_candidates = [
        name
        for name in ctxs.keys()
        if "/Tables/" in name
        and any(k in name for k in ["retail_data", "workforce_data"])
    ]
    assert len(table_ctx_candidates) >= 2


@pytest.mark.asyncio
@_handle_project
async def test_csv_with_business_context_from_file(file_manager, tmp_path: Path):
    """Test CSV parsing with business context loaded from JSON config file."""
    # Create a CSV file
    csv_path = tmp_path / "products.csv"
    csv_path.write_text(
        "ProductID,Name,Price,Stock\nP001,Widget,19.99,100\nP002,Gadget,29.99,50\n",
        encoding="utf-8",
    )
    display_name = str(csv_path)

    # Create config file with business context
    config_data = {
        "ingest": {
            "business_contexts": {
                "global_rules": [],
                "file_contexts": [
                    {
                        "file_path": display_name,
                        "file_rules": [],
                        "table_contexts": [
                            {
                                "table": "products",
                                "table_rules": [],
                                "column_descriptions": {
                                    "ProductID": "Unique product identifier",
                                    "Name": "Product display name",
                                    "Price": "Retail price in USD",
                                    "Stock": "Available inventory quantity",
                                },
                                "table_description": "Product catalog with pricing and inventory",
                            },
                        ],
                    },
                ],
            },
        },
    }
    config_file = tmp_path / "products_config.json"
    config_file.write_text(json.dumps(config_data))

    # Load config from file
    cfg = FilePipelineConfig.from_file(str(config_file))

    # Parse with config
    result = file_manager.ingest_files(display_name, config=cfg)
    _item = result[display_name]
    _item = _item if isinstance(_item, dict) else _item.model_dump()
    assert _item["status"] == "success"

    # Verify business context was loaded
    assert cfg.ingest.business_contexts is not None
    assert len(cfg.ingest.business_contexts.file_contexts) == 1
    fc = cfg.ingest.business_contexts.file_contexts[0]
    assert fc.file_path == display_name
    assert len(fc.table_contexts) == 1
    table_spec = fc.table_contexts[0]
    assert table_spec.table == "products"
    assert "ProductID" in table_spec.column_descriptions
