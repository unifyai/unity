"""
Table-related type definitions for DataManager.

This module defines Pydantic models for table metadata, schemas, and column information
returned by DataManager's table management and query operations.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class ColumnInfo(BaseModel):
    """
    Metadata for a single table column.

    Attributes
    ----------
    name : str
        Column name as it appears in queries.
    dtype : str
        Data type string (e.g., "str", "int", "float", "bool", "datetime", "list", "dict").
    description : str | None
        Optional human-readable description of the column's purpose.
    """

    name: str
    dtype: str
    description: Optional[str] = None


class TableSchema(BaseModel):
    """
    Schema definition for a table context.

    This represents the column structure and constraints of a Unify table context.

    Attributes
    ----------
    columns : list[ColumnInfo]
        List of column definitions with type info.
    unique_keys : dict[str, str] | None
        Mapping of unique key column names to their types.
        These columns enforce uniqueness constraints during inserts.
    auto_counting : dict[str, str | None] | None
        Columns with auto-increment behavior. Keys are column names,
        values are scoping columns (or None for global auto-increment).
    """

    columns: List[ColumnInfo] = Field(default_factory=list)
    unique_keys: Optional[Dict[str, str]] = None
    auto_counting: Optional[Dict[str, Optional[str]]] = None

    # TODO: The Unify backend's `get_context` / `get_contexts` API returns `unique_keys`
    # as a list (e.g., ["column_name"]) instead of a dict (e.g., {"column_name": "int"}).
    # This is inconsistent with `create_context` which expects a dict.
    #
    # Failing example: Actor calls `primitives.data.describe_table("Data/Test/...")` which
    # internally calls `unify.get_context()`. The response has `unique_keys: []` (empty list)
    # which fails Pydantic validation expecting `dict[str, str] | None`.
    #
    # Workaround: Convert list to None until backend is updated to return dict format.
    # See: unity/data_manager/ops/table_ops.py describe_table_impl
    @field_validator("unique_keys", mode="before")
    @classmethod
    def _normalize_unique_keys(cls, v):
        if isinstance(v, list):
            # Backend returns list, but we need dict or None
            return None
        return v

    @property
    def column_names(self) -> List[str]:
        """List of all column names."""
        return [c.name for c in self.columns]

    @property
    def column_types(self) -> Dict[str, str]:
        """Mapping of column names to their data types."""
        return {c.name: c.dtype for c in self.columns}


class TableDescription(BaseModel):
    """
    Complete metadata for a Unify table context.

    This model is returned by ``DataManager.describe_table()`` and provides
    everything needed to understand a table's structure.

    Attributes
    ----------
    context : str
        Fully-qualified context path (e.g., "Data/MH/arrears").
    description : str | None
        Human-readable description of the table's purpose.
    table_schema : TableSchema
        Column definitions and constraints.
    has_embeddings : bool
        True if any column has associated vector embeddings.
    embedding_columns : list[str]
        Names of embedding columns present (pattern: ``_<name>_emb``).

    Notes
    -----
    row_count is intentionally NOT included as it's expensive to compute.
    Use ``dm.reduce(context, metric="count", columns="id")`` if needed.

    Usage Examples
    --------------
    >>> desc = dm.describe_table("Data/MH/arrears")
    >>> print(f"Table: {desc.context}")
    >>> print(f"Columns: {desc.table_schema.column_names}")
    """

    context: str
    description: Optional[str] = None
    table_schema: TableSchema = Field(default_factory=TableSchema)
    has_embeddings: bool = False
    embedding_columns: List[str] = Field(default_factory=list)

    # Convenience aliases
    @property
    def columns(self) -> Dict[str, str]:
        """Column name to type mapping (convenience accessor)."""
        return self.table_schema.column_types

    @property
    def unique_keys(self) -> Optional[Dict[str, str]]:
        """Unique key constraints (convenience accessor)."""
        return self.table_schema.unique_keys
