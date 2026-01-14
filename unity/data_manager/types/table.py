"""
Table-related type definitions for DataManager.

This module defines Pydantic models for table metadata, schemas, and column information
returned by DataManager's table management and query operations.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class ColumnInfo(BaseModel):
    """
    Metadata for a single table column.

    Attributes
    ----------
    name : str
        Column name as it appears in queries.
    dtype : str
        Data type string (e.g., "str", "int", "float", "bool", "datetime", "list", "dict").
    searchable : bool
        True if this column has an associated embedding column (``_<name>_emb``),
        enabling semantic search via ``search()`` method.
    description : str | None
        Optional human-readable description of the column's purpose.
    """

    name: str
    dtype: str
    searchable: bool = False
    description: Optional[str] = None


class TableSchema(BaseModel):
    """
    Schema definition for a table context.

    This represents the column structure and constraints of a Unify table context.

    Attributes
    ----------
    columns : list[ColumnInfo]
        List of column definitions with type and searchability info.
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

    @property
    def column_names(self) -> List[str]:
        """List of all column names."""
        return [c.name for c in self.columns]

    @property
    def searchable_columns(self) -> List[str]:
        """List of column names that have vector embeddings (searchable)."""
        return [c.name for c in self.columns if c.searchable]

    @property
    def column_types(self) -> Dict[str, str]:
        """Mapping of column names to their data types."""
        return {c.name: c.dtype for c in self.columns}


class TableDescription(BaseModel):
    """
    Complete metadata for a Unify table context.

    This model is returned by ``DataManager.describe_table()`` and provides
    everything needed to understand a table's structure and current state.

    Attributes
    ----------
    context : str
        Fully-qualified context path (e.g., "Data/MH/arrears").
    description : str | None
        Human-readable description of the table's purpose.
    table_schema : TableSchema
        Column definitions and constraints.
    row_count : int
        Current number of rows in the table.
    has_embeddings : bool
        True if any column has associated vector embeddings.
    embedding_columns : list[str]
        Names of columns that have embeddings (pattern: ``_<name>_emb``).

    Usage Examples
    --------------
    >>> desc = dm.describe_table("Data/MH/arrears")
    >>> print(f"Table: {desc.context}")
    >>> print(f"Rows: {desc.row_count}")
    >>> print(f"Columns: {desc.table_schema.column_names}")
    >>> print(f"Searchable: {desc.table_schema.searchable_columns}")
    """

    context: str
    description: Optional[str] = None
    table_schema: TableSchema = Field(default_factory=TableSchema)
    row_count: int = 0
    has_embeddings: bool = False
    embedding_columns: List[str] = Field(default_factory=list)

    # Convenience aliases for backward compatibility with spec
    @property
    def columns(self) -> Dict[str, str]:
        """Column name to type mapping (convenience accessor)."""
        return self.table_schema.column_types

    @property
    def unique_keys(self) -> Optional[Dict[str, str]]:
        """Unique key constraints (convenience accessor)."""
        return self.table_schema.unique_keys
