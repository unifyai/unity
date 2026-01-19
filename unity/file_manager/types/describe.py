"""
Pydantic models for the FileManager describe() API.

These types provide a holistic view of a file's storage representation
in the Unify backend, enabling agents to discover all queryable contexts,
schemas, and identifiers for accurate retrieval operations.
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class ColumnInfo(BaseModel):
    """
    Schema information for a single column in a context.

    Provides type, description, and embedding status for a column,
    enabling agents to understand the queryable structure of a context.
    """

    name: str = Field(
        ...,
        description="Column name as stored in the Unify context.",
    )
    data_type: str = Field(
        default="unknown",
        description="Column data type (e.g., 'str', 'int', 'float', 'json').",
    )
    description: Optional[str] = Field(
        default=None,
        description="Human-readable description of the column's purpose.",
    )
    is_searchable: bool = Field(
        default=False,
        description="True if this column has vector embeddings for semantic search.",
    )
    embedding_column: Optional[str] = Field(
        default=None,
        description="Name of the embedding column (e.g., '_summary_emb') if searchable.",
    )


class ContextSchema(BaseModel):
    """
    Schema representation for a Unify context.

    Contains column information enabling agents to construct
    accurate filter expressions, search queries, and aggregations.
    """

    columns: List[ColumnInfo] = Field(
        default_factory=list,
        description="List of columns in this context with type and searchability info.",
    )

    @property
    def column_names(self) -> List[str]:
        """Return list of column names."""
        return [c.name for c in self.columns]

    @property
    def searchable_columns(self) -> List[str]:
        """Return list of columns that support semantic search."""
        return [c.name for c in self.columns if c.is_searchable]

    def get_column(self, name: str) -> Optional[ColumnInfo]:
        """Get column info by name."""
        for col in self.columns:
            if col.name == name:
                return col
        return None


class DocumentInfo(BaseModel):
    """
    Information about a file's /Content context.

    The /Content context stores hierarchical document structure
    (sections, paragraphs, sentences) with semantic annotations.
    Used for unstructured documents like PDFs, DOCX, etc.
    """

    context_path: str = Field(
        ...,
        description=(
            "Full Unify context path for document content. "
            "Use this exact path with filter/search/reduce operations."
        ),
    )
    column_schema: ContextSchema = Field(
        default_factory=ContextSchema,
        description="Schema with columns, types, and searchability info.",
    )
    row_count: Optional[int] = Field(
        default=None,
        description="Number of content rows (may be None if not yet computed).",
    )


class TableInfo(BaseModel):
    """
    Information about a single table context within a file.

    Tables can come from spreadsheet sheets, extracted tables in documents,
    or CSV data. Each table has its own schema and searchable columns.
    """

    name: str = Field(
        ...,
        description="Logical table name (e.g., 'Sheet1', 'extracted_table_1').",
    )
    context_path: str = Field(
        ...,
        description=(
            "Full Unify context path for this table. "
            "Use this exact path with filter/search/reduce operations."
        ),
    )
    column_schema: ContextSchema = Field(
        default_factory=ContextSchema,
        description="Schema with columns, types, and searchability info.",
    )
    row_count: Optional[int] = Field(
        default=None,
        description="Number of rows in this table (may be None if not yet computed).",
    )


class FileStorageMap(BaseModel):
    """
    Complete storage representation of a file in the Unify backend.

    This is the primary output of FileManager.describe() and provides
    all information an agent needs to construct accurate queries:

    - Status: filesystem_exists, indexed_exists, parsed_status
    - Identity: file_id, file_path, storage_id, source_uri, source_provider
    - Storage: document (/Content), tables (/Tables/<name>), index_context
    - Schemas: column info with types and searchability

    Usage Pattern
    -------------
    1. Call describe(file_path="/reports/Q4.csv") to get FileStorageMap
    2. Check storage.indexed_exists and storage.parsed_status before querying
    3. Use storage.tables[0].context_path with filter/search/reduce
    4. Reference storage.tables[0].column_schema for column names and types

    Status Interpretation
    ---------------------
    - filesystem_exists=True, indexed_exists=False: File exists but not ingested
    - indexed_exists=True, parsed_status='failed': Ingested but parsing failed
    - indexed_exists=True, parsed_status='success': Full storage info available

    Example
    -------
    >>> storage = file_manager.describe(file_path="/reports/Q4.csv")
    >>> if not storage.indexed_exists:
    ...     file_manager.ingest_files("/reports/Q4.csv")
    ...     storage = file_manager.describe(file_path="/reports/Q4.csv")
    >>> if storage.parsed_status == "success" and storage.has_tables:
    ...     results = data_manager.filter(
    ...         context=storage.tables[0].context_path,
    ...         filter="revenue > 1000000",
    ...     )
    """

    # -------------------------------------------------------------------------
    # Status fields
    # -------------------------------------------------------------------------
    filesystem_exists: bool = Field(
        default=False,
        description=(
            "True if the file exists on the filesystem at the given path. "
            "False if the file was deleted or path is invalid."
        ),
    )
    indexed_exists: bool = Field(
        default=False,
        description=(
            "True if the file has a row in the FileRecords index. "
            "False means the file hasn't been ingested yet."
        ),
    )
    parsed_status: Optional[str] = Field(
        default=None,
        description=(
            "Parse outcome: 'success', 'failed', 'partial', or None if not parsed. "
            "Check this before accessing document/tables storage info."
        ),
    )

    # -------------------------------------------------------------------------
    # Storage configuration
    # -------------------------------------------------------------------------
    storage_id: str = Field(
        default="",
        description=(
            "Context path identifier for this file's storage. "
            "Use to query shared contexts or reference in filter/search operations. "
            "Files sharing the same storage_id have their content in a shared context."
        ),
    )
    table_ingest: bool = Field(
        default=True,
        description="True if table extraction was enabled during ingestion.",
    )
    file_format: Optional[str] = Field(
        default=None,
        description="Detected file format (e.g., 'pdf', 'xlsx', 'csv', 'docx').",
    )

    # -------------------------------------------------------------------------
    # Core identification
    # -------------------------------------------------------------------------
    file_id: Optional[int] = Field(
        default=None,
        description=(
            "Stable unique identifier for this file. "
            "None if file is not indexed. Use for cross-context joins."
        ),
    )
    file_path: str = Field(
        ...,
        description="Filesystem path as provided or resolved from file_id.",
    )
    source_uri: Optional[str] = Field(
        default=None,
        description="Canonical provider URI (e.g., local:///abs/path, gdrive://fileId).",
    )
    source_provider: Optional[str] = Field(
        default=None,
        description="Provider/adapter name (e.g., Local, GoogleDrive).",
    )

    # Storage contexts
    document: Optional[DocumentInfo] = Field(
        default=None,
        description=(
            "Document content context (/Content). "
            "Present for PDFs, DOCX, and other unstructured documents."
        ),
    )
    tables: List[TableInfo] = Field(
        default_factory=list,
        description=(
            "List of table contexts (/Tables/<name>). "
            "Present for spreadsheets, CSVs, and extracted tables."
        ),
    )

    # Index reference
    index_context: str = Field(
        ...,
        description="FileRecords index context path for this file's metadata row.",
    )

    # Structural flags (from manifest, fast to check)
    has_document: bool = Field(
        default=False,
        description="True if this file has a /Content context.",
    )
    has_tables: bool = Field(
        default=False,
        description="True if this file has one or more /Tables contexts.",
    )

    @property
    def table_names(self) -> List[str]:
        """Return list of table names."""
        return [t.name for t in self.tables]

    @property
    def all_context_paths(self) -> List[str]:
        """Return all queryable context paths for this file."""
        paths = []
        if self.document:
            paths.append(self.document.context_path)
        for table in self.tables:
            paths.append(table.context_path)
        return paths

    def get_table(self, name: str) -> Optional[TableInfo]:
        """Get table info by name."""
        for table in self.tables:
            if table.name == name:
                return table
        return None

    def get_table_by_context(self, context_path: str) -> Optional[TableInfo]:
        """Get table info by context path."""
        for table in self.tables:
            if table.context_path == context_path:
                return table
        return None
