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
    file_manager.clear()

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

    # Avoid LLM calls during table catalog summarization; we validate business-context
    # enrichment by patching the summarizer and asserting the built profile text contains
    # the business-context fields. Note: `/Content/` table rows must NOT store `content_text`.
    from unittest.mock import patch

    def _stub_summary(*args, **kwargs) -> str:
        profile_text = kwargs.get("profile_text") if isinstance(kwargs, dict) else None
        if profile_text is None and args:
            profile_text = args[0]
        profile_text = str(profile_text or "")
        assert "Table Label: people" in profile_text
        assert (
            "Table Description: People directory with basic contact information"
            in profile_text
        )
        assert "- Name: Person's full name" in profile_text
        assert "- Age: Person's age in years" in profile_text
        assert "- City: City where the person lives" in profile_text
        return "stub table summary"

    with patch(
        "unity.file_manager.parse_adapter.lowering.content_rows.summarize_table_profile",
        side_effect=_stub_summary,
    ):
        result = file_manager.ingest_files(display_name, config=cfg)
    item = result[display_name]
    # All returns are now Pydantic models - use attribute access
    assert item.status == "success"

    # Verify `/Content/` contains sheet + table catalog rows (RAG navigation surface)
    import unify

    content_rows = unify.get_logs(context=item.content_ref.context, limit=50)
    assert content_rows, "Expected at least one /Content/ row"
    entries = [r.entries for r in content_rows]
    sheet_rows = [e for e in entries if e.get("content_type") == "sheet"]
    table_rows = [e for e in entries if e.get("content_type") == "table"]
    assert sheet_rows, "Expected at least one sheet catalog row in /Content/"
    assert table_rows, "Expected at least one table catalog row in /Content/"

    # Table rows are catalog-only; content_text is centralized to be None for sheet/table rows.
    assert table_rows[0].get("content_text") is None
    # Summary is stubbed (LLM patched)
    assert table_rows[0].get("summary") == "stub table summary"

    # Verify a per-table context exists
    ctxs = unify.get_contexts()
    # unify.get_contexts() returns a list of dicts with 'name' field
    ctx_names = (
        [ctx.get("name", "") for ctx in ctxs]
        if isinstance(ctxs, list)
        else list(ctxs.keys())
    )
    table_ctx_candidates = [
        name for name in ctx_names if "/Tables/" in name and "people" in name
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
    assert len(rows[0].entries) > 0, "Columns not found"
    assert len(rows[1].entries) > 0, "Columns not found"
    assert "row_id" in rows[0].entries
    assert "row_id" in rows[1].entries
    # Order is backend-dependent; validate presence rather than position.
    seen = {
        (r.entries.get("Name"), str(r.entries.get("Age")), r.entries.get("City"))
        for r in rows
    }
    assert ("Jane", "25", "LDN") in seen
    assert ("John", "30", "NYC") in seen


@pytest.mark.asyncio
@_handle_project
async def test_xlsx_multi_tab_per_table_context(file_manager, tmp_path: Path):
    file_manager.clear()

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

    from unittest.mock import patch

    for path in [retail, workforce]:
        if path.exists():
            display_name = str(path)
            # Use config with business context
            with patch(
                "unity.file_manager.parse_adapter.lowering.content_rows.summarize_table_profile",
                return_value="stub table summary",
            ):
                res = file_manager.ingest_files(display_name, config=cfg)
            item = res[display_name]
            # All returns are now Pydantic models - use attribute access
            assert item.status == "success"
        else:
            print(f"Path {path} does not exist")
            assert False

    # Verify multiple per-table contexts exist (one per tab)
    import unify

    ctxs = unify.get_contexts()
    # unify.get_contexts() returns a list of dicts with 'name' field
    ctx_names = (
        [ctx.get("name", "") for ctx in ctxs]
        if isinstance(ctxs, list)
        else list(ctxs.keys())
    )
    table_ctx_candidates = [
        name
        for name in ctx_names
        if "/Tables/" in name
        and any(k in name for k in ["retail_data", "workforce_data"])
    ]
    assert len(table_ctx_candidates) >= 2


@pytest.mark.asyncio
@_handle_project
async def test_csv_with_business_context_from_file(file_manager, tmp_path: Path):
    """Test CSV parsing with business context loaded from JSON config file."""
    file_manager.clear()

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
    from unittest.mock import patch

    with patch(
        "unity.file_manager.parse_adapter.lowering.content_rows.summarize_table_profile",
        return_value="stub table summary",
    ):
        result = file_manager.ingest_files(display_name, config=cfg)
    item = result[display_name]
    # All returns are now Pydantic models - use attribute access
    assert item.status == "success"

    # Verify business context was loaded
    assert cfg.ingest.business_contexts is not None
    assert len(cfg.ingest.business_contexts.file_contexts) == 1
    fc = cfg.ingest.business_contexts.file_contexts[0]
    assert fc.file_path == display_name
    assert len(fc.table_contexts) == 1
    table_spec = fc.table_contexts[0]
    assert table_spec.table == "products"
    assert "ProductID" in table_spec.column_descriptions
