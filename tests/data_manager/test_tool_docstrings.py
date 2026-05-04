from __future__ import annotations

from unity.data_manager.base import BaseDataManager


def test_data_write_tools_expose_destination_guidance():
    for method_name in (
        "create_table",
        "delete_table",
        "rename_table",
        "create_column",
        "delete_column",
        "rename_column",
        "create_derived_column",
        "join_tables",
        "insert_rows",
        "update_rows",
        "delete_rows",
        "ingest",
        "ensure_vector_column",
        "vectorize_rows",
    ):
        doc = (getattr(BaseDataManager, method_name).__doc__ or "").strip()

        assert "destination : str | None" in doc
        assert "Accessible" in doc
        assert "shared" in doc
        assert "spaces" in doc
        assert "space:<id>" in doc
        assert "personal" in doc
        assert "request_clarification" in doc
