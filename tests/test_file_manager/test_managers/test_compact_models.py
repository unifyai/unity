from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_parse_compact_returns_typed_models(
    file_manager,
    supported_file_examples: dict,
):
    from unity.file_manager.types import (
        BaseParsedFile,
        ParsedPDF,
        ParsedDocx,
        ParsedXlsx,
        ParsedCsv,
        ContentRef,
        TableRef,
        FileMetrics,
        FileFormat,
        MimeType,
    )

    # Map extensions to expected compact model classes
    ext_to_model = {
        ".pdf": ParsedPDF,
        ".docx": ParsedDocx,
        ".xlsx": ParsedXlsx,
        ".csv": ParsedCsv,
    }

    # Iterate known sample files and assert typed models are returned by default (compact mode)
    for filename, example_data in supported_file_examples.items():
        ext = example_data["format"].lower()
        if ext not in ext_to_model:
            continue  # skip formats without a concrete subclass

        display_name = str(example_data["path"])  # absolute path
        res = file_manager.parse(display_name)  # default compact
        assert display_name in res
        item = res[display_name]

        # Type: should be the specific subclass (not a dict)
        assert isinstance(item, ext_to_model[ext])
        assert isinstance(item, BaseParsedFile)

        # Common fields and types
        assert isinstance(item.content_ref, ContentRef)
        assert isinstance(item.tables_ref, list)
        assert all(isinstance(t, TableRef) for t in item.tables_ref)
        assert isinstance(item.metrics, FileMetrics)
        assert item.status in ("success", "error")

        # Enums are present and correctly typed
        assert item.file_format is None or isinstance(item.file_format, FileFormat)
        assert item.mime_type is None or isinstance(item.mime_type, MimeType)

        # Subclass-specific field presence (shape checks only)
        if isinstance(item, ParsedPDF):
            assert hasattr(item, "page_count")
            assert hasattr(item, "total_sections")
        if isinstance(item, ParsedXlsx):
            assert hasattr(item, "sheet_count")
            assert isinstance(item.sheet_names, list)
        if isinstance(item, ParsedCsv):
            assert hasattr(item, "table_count")
