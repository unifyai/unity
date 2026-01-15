from __future__ import annotations

import asyncio
from abc import abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from unity.common.async_tool_loop import SteerableToolHandle
from unity.common.global_docstrings import CLEAR_METHOD_DOCSTRING
from unity.common.state_managers import BaseStateManager

if TYPE_CHECKING:
    from unity.file_manager.types.ingest import IngestPipelineResult


class BaseFileManager(BaseStateManager):
    """
    Public contract that every concrete file-manager must satisfy.

    Exposes read-only discovery/analysis over a single filesystem.

    Responsibilities
    ----------------
    • "ask_about_file" — answer questions about one specific file (read-only)

    For filesystem‑wide operations (asking questions across files, organizing
    files), use ``FunctionManager`` to compose bespoke logic combining lexical
    and semantic search with shell scripts.

    Contexts & Joins
    ----------------
    A concrete FileManager typically manages:
    - a global index context, and
    - per-file contexts, optionally with nested per-table contexts.

    Implementations expose read-only join tools to combine these contexts for
    efficient retrieval:
    - filter_join / search_join: join two contexts and then filter or perform
      semantic search over the joined result.
    - filter_multi_join / search_multi_join: chain multiple joins (the special
      placeholder '$prev' may be used to refer to the previous step at call-time).

    Reference conventions for join tools are implementation-specific. The
    concrete class must document how callers identify the global index and
    per-file/per-table contexts.
    """

    _as_caller_description: str = (
        "a FileManager, analyzing files on behalf of the end user"
    )

    # ------------------------------------------------------------------ #
    # Basic inventory operations                                          #
    # ------------------------------------------------------------------ #
    @abstractmethod
    def exists(self, file_path: str) -> bool:
        """Return True if a file with the given file path exists in this filesystem."""

    @abstractmethod
    def list(self) -> List[str]:
        """Return the list of file paths (stable order) for files in this filesystem."""

    @abstractmethod
    def ingest_files(
        self,
        file_paths: Union[str, List[str]],
        **options: Any,
    ) -> "IngestPipelineResult":
        """
        Run the complete file processing pipeline: parse, ingest, and embed.

        This method orchestrates the full file processing workflow:
        1. Parse files using the configured parser to extract structured content
        2. Ingest parsed content into storage contexts (per-file or unified)
        3. Create embeddings based on the configured strategy (along, after, or off)

        Parameters
        ----------
        file_paths : str | list[str]
            Single file path or a list of file paths to process.
        **options : Any
            Pipeline options (forwarded as-is).

        Returns
        -------
        IngestPipelineResult
            Structured container with per-file ingest results and global statistics.
            Supports dict-like access: result[file_path].

            - result.files: Dict[str, IngestedFileUnion] - per-file results
            - result.statistics: PipelineStatistics - global counts and timing
            - result[file_path]: direct access to individual file result

            Each file result is a typed Pydantic model (IngestedPDF, IngestedXlsx, etc.)
            containing reference-first pointers (content_ref, tables_ref) and metadata.

        Options
        -------
        config : FilePipelineConfig | dict | None
            Complete pipeline configuration controlling parsing, ingestion, embeddings,
            and output return mode. When a dict is provided, it will be coerced
            to `FilePipelineConfig` (unknown keys are ignored).

            Key sub-models and fields:
            - parse.max_concurrent_parses: int (parse-stage parallelism; capped conservatively)
            - parse.backend_class_paths_by_format: dict[str, str] (format -> dotted backend class path)
            - ingest.mode: "per_file" | "unified" (destination layout)
            - ingest.table_ingest: bool (ingest extracted tables)
            - embed.strategy: "along" | "after" | "off" (when to embed)
            - embed.file_specs: list[FileEmbeddingSpec] (which columns to embed)

        Notes
        -----
        - Implementations SHOULD accept `config` in **options and default to a sensible
          `FilePipelineConfig()` when omitted.
        """

    # ------------------------------------------------------------------ #
    # File export operations (for parsing)                               #
    # ------------------------------------------------------------------ #
    @abstractmethod
    def export_file(self, file_path: str, destination_dir: str) -> str:
        """
        Export a file from the underlying filesystem to a local destination directory.

        This method is used by parse operations to bring files from the adapter's
        filesystem into a local temporary directory with their original file paths preserved.

        Parameters
        ----------
        file_path : str
            The file path of the file to export.
        destination_dir : str
            Local directory path where the file should be exported.

        Returns
        -------
        str
            Full path to the exported file in the destination directory.

        Raises
        ------
        FileNotFoundError
            If the source file doesn't exist.
        """

    @abstractmethod
    def export_directory(self, directory: str, destination_dir: str) -> List[str]:
        """
        Export all files from a directory to a local destination directory.

        This is a batch operation that exports multiple files at once, optimizing
        for the underlying filesystem's capabilities (e.g., zip downloads).

        Parameters
        ----------
        directory : str
            The directory path to export files from.
        destination_dir : str
            Local directory path where files should be exported.

        Returns
        -------
        list[str]
            List of full paths to exported files in the destination directory.
        """

    # ------------------------------------------------------------------ #
    # Unify-backed retrieval (public tools)                              #
    # ------------------------------------------------------------------ #
    @abstractmethod
    def describe(
        self,
        *,
        file_path: Optional[str] = None,
        file_id: Optional[int] = None,
    ) -> Any:
        """
        Return a complete storage representation of a file in the Unify backend.

        This is the primary discovery tool for understanding how a file's data
        is stored. It returns all context paths, schemas, and identifiers needed
        for accurate filter/search/reduce operations.

        Use describe() BEFORE calling filter_files(), search_files(), or reduce()
        to obtain the exact context paths for your queries.

        Parameters
        ----------
        file_path : str | None
            The filesystem path of the file. Either file_path or file_id must be provided.
        file_id : int | None
            The stable unique identifier from FileRecords. Either file_path or
            file_id must be provided. Using file_id is preferred when available.

        Returns
        -------
        FileStorageMap
            Complete storage representation including file_id, file_path,
            document info, table infos, index_context, and flags.
        """

    @abstractmethod
    def file_info(self, *, identifier: Union[str, int]) -> Any:
        """
        Return comprehensive information about a file's status and ingest identity.

        .. deprecated::
            Use describe(file_path=...) or describe(file_id=...) instead for
            comprehensive file discovery with exact context paths and schemas.

        Parameters
        ----------
        identifier : str | int
            File identifier. Accepted forms:
            - Absolute file path: "/path/to/file.pdf"
            - Provider URI: "local:///path/to/file.pdf", "gdrive://fileId"
            - File ID (int): The numeric file_id from FileRecords

        Returns
        -------
        FileInfo
            Pydantic model with filesystem_exists, indexed_exists, parsed_status,
            source_provider, source_uri, ingest_mode, unified_label, table_ingest,
            file_format fields.
        """

    @abstractmethod
    def tables_overview(
        self,
        *,
        include_column_info: bool = True,
        file: Optional[str] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Return an overview of available tables/contexts managed by this FileManager.

        .. deprecated::
            Use describe(file_path=...) instead for file-specific discovery.
            describe() returns FileStorageMap with exact context paths and schemas.

        Parameters
        ----------
        include_column_info : bool, default True
            When True and file is None, include the index schema (columns→types).
        file : str | None, default None
            When None: returns ONLY the global FileRecords index overview.
            When provided: returns file-scoped overview with Content and Tables
            for that specific file (respecting its ingest mode).

        Returns
        -------
        dict[str, dict]
            Logical table names → metadata (context path, description, columns).
        """

    @abstractmethod
    def schema_explain(self, *, table: str) -> str:
        """
        Return a natural-language explanation of a table's structure and purpose.

        Parameters
        ----------
        table : str
            Table reference (path-first preferred):
            - "<file_path>" for per-file Content
            - "<file_path>.Tables.<label>" for per-file tables
            - "FileRecords" for the global file index

        Returns
        -------
        str
            Compact natural-language explanation including what the table
            represents, key fields and their meanings, and approximate row count.
        """

    @abstractmethod
    def list_columns(
        self,
        *,
        include_types: bool = True,
        context: Optional[str] = None,
    ) -> Dict[str, Any] | List[str]:
        """
        Return the schema for a context managed by this FileManager.

        Use describe() first to obtain the exact context path, then use
        list_columns() for detailed schema inspection if needed.

        Parameters
        ----------
        include_types : bool, default True
            When True, return a mapping of column → logical type. When False,
            return just the list of column names.
        context : str | None, default None
            Full Unify context path to inspect. Obtain this from describe():
            - storage.document.context_path for document content
            - storage.tables[0].context_path for a specific table
            - storage.index_context for the FileRecords index
            When None, returns the FileRecords (index) columns.

        Returns
        -------
        dict[str, Any] | list[str]
            Column→type mapping when include_types=True, otherwise a list of column names.
        """

    @abstractmethod
    def filter_files(
        self,
        *,
        context: Optional[str] = None,
        filter: Optional[str] = None,
        offset: int = 0,
        limit: int = 100,
        columns: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Filter rows from a context using exact match expressions.

        Use describe() first to get the exact context path.

        Parameters
        ----------
        context : str | None
            Full Unify context path to filter. Obtain this from describe():
            - storage.document.context_path for document content
            - storage.tables[0].context_path for a specific table
            - storage.index_context for the FileRecords index
            When None, defaults to the FileRecords index.
        filter : str | None
            Row-level predicate evaluated per row (column names in scope).
        offset : int
            Pagination offset.
        limit : int
            Maximum rows to return.
        columns : list[str] | None
            Specific columns to return. When None, returns all columns.

        Returns
        -------
        list[dict]
            Flat list of matching rows from the context.
        """

    @abstractmethod
    def search_files(
        self,
        *,
        context: Optional[str] = None,
        references: Optional[Dict[str, str]] = None,
        limit: int = 10,
        filter: Optional[str] = None,
        columns: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Semantic search over a context using reference texts for similarity matching.

        Use describe() first to get the exact context path and check which
        columns are searchable (have embeddings).

        Parameters
        ----------
        context : str | None
            Full Unify context path to search. Obtain this from describe():
            - storage.document.context_path for document content
            - storage.tables[0].context_path for a specific table
            - storage.index_context for the FileRecords index
            When None, defaults to the FileRecords index.
        references : dict[str, str] | None
            Mapping of column names → reference texts for semantic matching.
            Only searchable columns (with embeddings) can be used.
        limit : int
            Maximum rows to return.
        filter : str | None
            Row-level predicate applied before ranking.
        columns : list[str] | None
            Specific columns to return. When None, returns all columns.

        Returns
        -------
        list[dict]
            Top-k rows ranked by semantic similarity.
        """

    @abstractmethod
    def reduce(
        self,
        *,
        context: Optional[str] = None,
        metric: str,
        column: str,
        filter: Optional[str] = None,
        group_by: Optional[Union[str, List[str]]] = None,
    ) -> Any:
        """
        Compute aggregation metrics over a context.

        Use describe() first to get the exact context path.

        Parameters
        ----------
        context : str | None
            Full Unify context path to aggregate. Obtain this from describe():
            - storage.document.context_path for document content
            - storage.tables[0].context_path for a specific table
            - storage.index_context for the FileRecords index
            When None, aggregates over the FileRecords index.
        metric : str
            Reduction metric: "count", "sum", "mean", "min", "max", "median",
            "mode", "var", "std".
        column : str
            Column to aggregate.
        filter : str | None
            Optional row-level filter expression applied before aggregation.
        group_by : str | list[str] | None
            Optional column(s) to group by.

        Returns
        -------
        Any
            Metric value(s) computed over the context.
        """

    @abstractmethod
    def visualize(
        self,
        *,
        tables: Union[str, List[str]],
        plot_type: str,
        x_axis: str,
        y_axis: Optional[str] = None,
        group_by: Optional[str] = None,
        filter: Optional[str] = None,
        title: Optional[str] = None,
        aggregate: Optional[str] = None,
        scale_x: Optional[str] = None,
        scale_y: Optional[str] = None,
        bin_count: Optional[int] = None,
        show_regression: Optional[bool] = None,
    ) -> Any:
        """
        Generate plot visualizations from table data via the Plot API.

        Parameters
        ----------
        tables : str | list[str]
            Table reference(s) to visualize. When a list is provided, the same
            plot configuration is applied to each table.
        plot_type : str
            Chart type: "bar", "line", "scatter", "histogram".
        x_axis : str
            Column name for the x-axis.
        y_axis : str | None
            Column name for the y-axis.
        group_by : str | None
            Column to group/color data points by.
        filter : str | None
            Row-level filter expression.
        title : str | None
            Plot title.
        aggregate : str | None
            Aggregation function: "sum", "mean", "count", "min", "max".
        scale_x, scale_y : str | None
            Axis scale: "linear" or "log".
        bin_count : int | None
            Number of bins for histogram plots.
        show_regression : bool | None
            Show regression line (scatter plots only).

        Returns
        -------
        PlotResult | list[PlotResult]
            Single table returns PlotResult, multiple tables returns list.
            Each result has: url, token, expires_in_hours, title, error, succeeded.
        """

    @abstractmethod
    def rename_file(
        self,
        *,
        file_id_or_path: Union[str, int],
        new_name: str,
    ) -> Dict[str, Any]:
        """
        Rename a file in the underlying filesystem.

        Parameters
        ----------
        file_id_or_path : str | int
            Either the file_id (int) as preserved in the FileRecords index, or the
            fully-qualified file_path (str) as stored in the FileRecords index/context.
            When a file_id is provided, it is resolved to the corresponding file_path.
        new_name : str
            New file name; adapter determines full path semantics.

        Returns
        -------
        dict
            Adapter reference payload or a summary dict of the rename result.
        """

    @abstractmethod
    def move_file(
        self,
        *,
        file_id_or_path: Union[str, int],
        new_parent_path: str,
    ) -> Dict[str, Any]:
        """
        Move a file to a new directory in the underlying filesystem.

        Parameters
        ----------
        file_id_or_path : str | int
            Either the file_id (int) as preserved in the FileRecords index, or the
            fully-qualified file_path (str) as stored in the FileRecords index/context.
            When a file_id is provided, it is resolved to the corresponding file_path.
        new_parent_path : str
            Destination directory path in adapter-native form.

        Returns
        -------
        dict
            Adapter reference payload or a summary dict of the move result.
        """

    @abstractmethod
    def delete_file(self, *, file_id_or_path: Union[str, int]) -> Dict[str, Any]:
        """
        Delete a file record from the Unify table and, if supported by the adapter,
        from the underlying filesystem.

        Parameters
        ----------
        file_id_or_path : str | int
            Either the file_id (int) as preserved in the FileRecords index, or the
            fully-qualified file_path (str) as stored in the FileRecords index/context.
            When a file_id is provided, it is resolved to the corresponding file_path.

        Returns
        -------
        dict
            Result dictionary with 'outcome' and 'details' keys.

        Raises
        ------
        ValueError
            If no file with the given file_id_or_path exists.
        PermissionError
            If the file is protected or the adapter doesn't support deletion.
        """

    # ------------------------------------------------------------------ #
    # File-specific Q&A                                                  #
    # ------------------------------------------------------------------ #
    @abstractmethod
    async def ask_about_file(
        self,
        file_path: str,
        question: str,
        *,
        _return_reasoning_steps: bool = False,
        _parent_chat_context: Optional[List[Dict[str, Any]]] = None,
        _clarification_up_q: Optional[asyncio.Queue[str]] = None,
        _clarification_down_q: Optional[asyncio.Queue[str]] = None,
        response_format: Optional[Any] = None,
    ) -> SteerableToolHandle:
        """
        Interrogate **one specific file** (read‑only) and obtain a live
        :class:`SteerableToolHandle`.

        Purpose
        -------
        Use this method when the caller already knows which file is relevant
        and wants a focused analysis (e.g., summarise this PDF, extract a key
        value from a document).

        Clarifications
        --------------
        Do not use this method to ask the human follow‑up questions. If the
        file_path is ambiguous and a clarification tool is available, route a
        targeted question via ``request_clarification``; if no channel exists,
        proceed with sensible defaults/best‑guess values and state assumptions
        in the outer reply.

        Parameters
        ----------
        file_path : str
            Logical identifier/path of the target file.
        question : str
            Natural‑language question about the specific file.

        Returns
        -------
        SteerableToolHandle
            Handle that eventually yields the answer text (and optionally the
            hidden reasoning steps) for this file‑scoped query.
        """

    @abstractmethod
    def clear(self) -> None:
        raise NotImplementedError

    # ------------------------------------------------------------------ #
    # Public sync                                                        #
    # ------------------------------------------------------------------ #
    @abstractmethod
    def sync(self, *, file_path: str) -> Dict[str, Any]:
        """
        Synchronize a previously ingested file with the underlying filesystem.

        Purge existing rows in relevant contexts and re-ingest. Implementations
        must respect ingest layout (per_file vs unified) when purging.
        """


# Attach centralised docstring
BaseFileManager.clear.__doc__ = CLEAR_METHOD_DOCSTRING
