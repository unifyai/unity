"""
Helpers for loading and composing table schemas from intranet/flat_schema.json.

This centralises schema concerns so core modules do not need edits when the
schema evolves.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional


def get_schema_path() -> Path:
    """Return absolute path to intranet/flat_schema.json."""
    # This file lives at intranet/scripts/schema_utils.py → project_root = parents[2]
    # parents[0] = scripts, [1] = intranet, [2] = repo root
    project_root = Path(__file__).resolve().parents[2]
    return project_root / "intranet" / "flat_schema.json"


def load_flat_schema(schema_path: Optional[str | Path] = None) -> Dict[str, Any]:
    """Load the flat schema JSON into a dict."""
    path = Path(schema_path) if schema_path else get_schema_path()
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_unique_key_mapping(schema: Dict[str, Any], table: str) -> Dict[str, str]:
    """Return mapping of unique key column(s) → type for the table."""
    return (schema.get("unique_key_configuration", {}) or {}).get("tables", {}).get(
        table,
        {},
    ) or {}


def get_unique_key_name(schema: Dict[str, Any], table: str) -> Optional[str]:
    """Return the first unique key name for the table if present."""
    uk = get_unique_key_mapping(schema, table)
    return next(iter(uk.keys())) if uk else None


def get_auto_counting_mapping(
    schema: Dict[str, Any],
    table: str,
) -> Dict[str, Optional[str]]:
    """Return mapping of auto-counted columns → parent counter for the table."""
    return (schema.get("auto_counting_configuration", {}) or {}).get("tables", {}).get(
        table,
        {},
    ) or {}


def get_full_embedding_and_counters(
    schema: Dict[str, Any],
    table: str,
) -> Dict[str, Any]:
    """Bundle embedding configuration and auto-counting for convenience."""
    return {
        "embedding": get_embedding_configuration(schema, table),
        "auto_counting": get_auto_counting_mapping(schema, table),
    }


def get_table_columns_definition(
    schema: Dict[str, Any],
    table: str,
) -> Dict[str, Dict[str, Any]]:
    """Return the raw columns block from schema["tables"][table]["columns"]."""
    return (schema.get("tables", {}) or {}).get(table, {}).get("columns", {}) or {}


def compose_table_columns(schema: Dict[str, Any], table: str) -> Dict[str, str]:
    """
    Compose the full columns mapping to create fields for a table by merging:
    1) unique key columns (as provided types),
    2) auto-counted columns (type "int"), then
    3) declared table columns (use their "type" field).

    Later entries with the same name do not override earlier ones to preserve
    the configured type for unique/auto columns.
    """
    unique_map = get_unique_key_mapping(schema, table)
    auto_map = get_auto_counting_mapping(schema, table)
    raw_cols = get_table_columns_definition(schema, table)

    composed: Dict[str, str] = {}

    # 1) unique keys: preserve provided types
    for name, type_str in (unique_map or {}).items():
        if name not in composed:
            composed[name] = type_str

    # 2) auto-counted columns: always ints
    for name in (auto_map or {}).keys():
        if name not in composed:
            composed[name] = "int"

    # 3) declared table columns
    for name, info in (raw_cols or {}).items():
        if name not in composed:
            type_str = (info or {}).get("type", "str")
            composed[name] = type_str

    return composed


def get_embedding_configuration(
    schema: Dict[str, Any],
    table: str,
) -> Optional[Dict[str, Any]]:
    """Return the embedding configuration for the provided table, if any."""
    tables_cfg = (schema.get("embedding_configuration", {}) or {}).get(
        "tables",
        {},
    ) or {}
    tbl_cfg = tables_cfg.get(table)
    if not tbl_cfg:
        return None
    return {"tables": {table: tbl_cfg}}
