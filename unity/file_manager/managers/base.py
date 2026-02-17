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
        """
        Check if a file exists in the adapter-backed filesystem.

        This checks ONLY the raw filesystem, NOT the index. Use this to answer
        "does this path currently exist on disk?" regardless of whether it has
        been parsed or indexed.

        For complete status (filesystem + index + parse status), use describe() instead.

        Parameters
        ----------
        file_path : str
            Adapter-native file path/identifier to check. Use absolute paths
            for reliable results.

        Returns
        -------
        bool
            True if the adapter reports the file exists, False otherwise or
            when no adapter is configured.

        Usage Examples
        --------------
        # Check if a file exists on disk:
        if exists("/path/to/document.pdf"):
            # File is present in filesystem
            pass

        # Verify before attempting to ingest:
        if exists("/path/to/new_file.xlsx"):
            ingest_files("/path/to/new_file.xlsx")

        # Check multiple files:
        files = ["/path/to/a.pdf", "/path/to/b.pdf"]
        existing = [f for f in files if exists(f)]

        Anti-patterns
        -------------
        - WRONG: Using exists() to check if a file has been indexed
          CORRECT: Use describe() to check indexed_exists

        - WRONG: Assuming exists() == True means content is queryable
          CORRECT: File must also be indexed; use describe() for complete status

        - WRONG: Using relative paths without knowing the adapter's working directory
          CORRECT: Use absolute paths for reliable results
        """

    @abstractmethod
    def list(self) -> List[str]:
        """
        List all files visible to the underlying adapter.

        Returns file paths from the RAW FILESYSTEM, not the index. Use this to
        discover what files are available for ingestion or to compare with
        indexed files.

        For querying indexed files, use filter_files() on the FileRecords table.

        Returns
        -------
        list[str]
            Adapter-native file paths/identifiers discoverable in the underlying
            filesystem, or an empty list if no adapter is configured or listing fails.

        Usage Examples
        --------------
        # List all files in the filesystem:
        files = list()
        # Returns: ["/path/to/file1.pdf", "/path/to/file2.xlsx", ...]

        # Find files not yet indexed:
        filesystem_files = set(list())
        indexed = filter_files()
        indexed_paths = {r["file_path"] for r in indexed}
        not_indexed = filesystem_files - indexed_paths

        # Count available files:
        file_count = len(list())

        Anti-patterns
        -------------
        - WRONG: Using list() to query file metadata (status, parse info)
          CORRECT: Use filter_files() on FileRecords for indexed file queries

        - WRONG: Expecting list() to show only successfully parsed files
          CORRECT: list() shows raw filesystem; filter_files(filter="status == 'success'")
                   shows successfully indexed files

        - WRONG: Using list() for large filesystems then filtering in Python
          CORRECT: Use filter_files() with appropriate filters for efficient queries
        """

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
        file_path: str,
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
        file_path : str
            The filesystem path of the file as used in FileRecords.

        Returns
        -------
        FileStorageMap
            Complete storage representation including:

            - **file_id** (int): Stable identifier for cross-referencing and joins.
            - **file_path** (str): Original filesystem path.
            - **source_uri** (str | None): Canonical provider URI.
            - **source_provider** (str | None): Provider/adapter name.
            - **document** (DocumentInfo | None): Info about /Content context including:
              - context_path: Full path for filter/search/reduce operations
              - schema: Columns with types and searchability info
            - **tables** (list[TableInfo]): List of /Tables/<name> contexts including:
              - name: Logical table name (e.g., 'Sheet1')
              - context_path: Full path for filter/search/reduce operations
              - schema: Columns with types and searchability info
            - **index_context** (str): Path to FileRecords index.
            - **has_document** (bool): Quick check if /Content exists.
            - **has_tables** (bool): Quick check if any /Tables exist.

        Usage Examples
        --------------
        Basic discovery by file path:

        >>> storage = file_manager.describe(file_path="/reports/Q4.csv")
        >>> print(f"File ID: {storage.file_id}")
        File ID: 42
        >>> print(f"Has tables: {storage.has_tables}")
        Has tables: True

        Get context path for querying:

        >>> storage = file_manager.describe(file_path="/reports/Q4.csv")
        >>> results = file_manager.filter_files(
        ...     context=storage.tables[0].context_path,
        ...     filter="revenue > 1000000",
        ...     columns=["region", "revenue"]
        ... )

        Anti-patterns
        -------------
        - WRONG: Guessing context paths without calling describe()

          >>> filter_files(context="Files/Local/reports/Q4.csv/Tables/Sheet1")

          CORRECT: Always use describe() to get exact paths

          >>> storage = describe(file_path="/reports/Q4.csv")
          >>> filter_files(context=storage.tables[0].context_path)

        - WRONG: Assuming all files have both document and tables

          >>> storage.tables[0].context_path  # Will fail if no tables

          CORRECT: Check has_tables/has_document first

          >>> if storage.has_tables:
          ...     # Safe to access tables
          ...     storage.tables[0].context_path

        - WRONG: Using file_path in context paths directly

          >>> # Unstable - breaks if file is renamed
          >>> "Files/Local/" + file_path + "/Content"

          CORRECT: Context paths use file_id internally for stability

          >>> storage.document.context_path  # Uses file_id

        Notes
        -----
        - Context paths use file_id (not file_path) for stability across renames.
        - The describe() method queries the backend live for fresh schema information.
        - Row counts are not included by default; use reduce(metric='count', ...) when needed.
        - For CSV/spreadsheet files, primary data is in tables (has_document may be False).
        - For PDF/DOCX files, primary data is in document with extracted tables optional.
        - Schema includes searchability info - check column_schema.searchable_columns before
          using search_files() to ensure the columns you need are embedded.

        See Also
        --------
        filter_files : Filter rows from a context path obtained from describe()
        search_files : Semantic search on a context path obtained from describe()
        reduce : Aggregate data from a context path obtained from describe()
        """

    @abstractmethod
    def list_columns(
        self,
        *,
        include_types: bool = True,
        context: Optional[str] = None,
    ) -> Dict[str, Any] | List[str]:
        """
        List columns for the FileRecords index or a specific context.

        Use this to inspect a context's schema before writing filter expressions
        or selecting columns for retrieval.

        IMPORTANT: Use describe() first to get the exact context path, then use
        list_columns() for detailed schema inspection if needed:

        >>> storage = describe(file_path="/reports/Q4.csv")
        >>> columns = list_columns(context=storage.tables[0].context_path)

        Parameters
        ----------
        include_types : bool, default True
            When True, return a mapping of column → type (e.g., {"name": "str"}).
            When False, return just the list of column names.
        context : str | None, default None
            Full Unify context path to inspect. Obtain this from describe():
            - storage.document.context_path for document content
            - storage.tables[0].context_path for a specific table
            - storage.index_context for the FileRecords index
            When None, returns the FileRecords index columns.

        Returns
        -------
        dict[str, str] | list[str]
            Column→type mapping (when include_types=True) or list of column names.

        Usage Examples
        --------------
        # Get FileRecords index schema:
        list_columns()
        # Returns: {"file_id": "int", "file_path": "str", "status": "str", ...}

        # Get columns for a context from describe():
        storage = describe(file_path="/path/to/document.pdf")
        list_columns(context=storage.document.context_path)
        # Returns: {"content_id": "dict", "content_type": "str", "content_text": "str", ...}

        # Get columns for a table context:
        storage = describe(file_path="/path/to/data.xlsx")
        list_columns(context=storage.tables[0].context_path)
        # Returns: {"order_id": "str", "amount": "float", "customer": "str", ...}

        # Get just column names (no types):
        list_columns(context=storage.tables[0].context_path, include_types=False)
        # Returns: ["order_id", "amount", "customer", ...]
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

        Use this tool for exact matches on structured fields (ids, statuses, dates).
        For semantic/meaning-based search, use search_files() instead.

        IMPORTANT: Use describe() first to get the exact context path:

        >>> storage = describe(file_path="/reports/Q4.csv")
        >>> results = filter_files(
        ...     context=storage.tables[0].context_path,
        ...     filter="revenue > 1000000"
        ... )

        Parameters
        ----------
        context : str, optional
            Full Unify context path to filter. Obtain this from describe():
            - storage.document.context_path for document content
            - storage.tables[0].context_path for a specific table
            - storage.index_context for the FileRecords index
            When None, defaults to the FileRecords index.

        filter : str, optional
            Python expression evaluated per row with column names in scope.
            Any valid Python syntax returning a boolean is supported.
            String values must be quoted. Use .get() for safe dict access.

        offset : int, default 0
            Zero-based pagination offset.

        limit : int, default 100
            Maximum rows to return.

        columns : list[str], optional
            Specific columns to return. When None, returns all columns.
            Use this to reduce response size when only specific fields are needed.

        Returns
        -------
        list[dict]
            Flat list of matching rows from the context.

        Per-file Content: content_id structure
        --------------------------------------
        The `content_id` column in per-file Content encodes document hierarchy:
        - document row: {"document": 0}
        - section row: {"document": 0, "section": 2}
        - paragraph row: {"document": 0, "section": 2, "paragraph": 1}
        - sentence row: {"document": 0, "section": 2, "paragraph": 1, "sentence": 3}
        - table row: {"document": 0, "section": 2, "table": 0}
        - image row: {"document": 0, "section": 2, "image": 0}

        Use .get() accessor for safe filtering:
        - filter="content_id.get('section') == 2"
        - filter="content_id.get('document') == 0 and content_type == 'paragraph'"

        Usage Examples
        --------------
        First, always call describe() to get context paths:

        >>> storage = describe(file_path="/reports/data.xlsx")

        Filter a table from the file:

        >>> results = filter_files(
        ...     context=storage.tables[0].context_path,
        ...     filter="created_at >= '2024-01-01' and created_at < '2024-02-01'",
        ...     columns=["id", "name", "created_at"]
        ... )

        Filter document content by hierarchy:

        >>> storage = describe(file_path="/docs/report.pdf")
        >>> results = filter_files(
        ...     context=storage.document.context_path,
        ...     filter="content_type == 'paragraph' and content_id.get('section') == 2"
        ... )

        Filter the FileRecords index (no context needed):

        >>> results = filter_files(filter="status == 'success'")

        Paginate through results:

        >>> page1 = filter_files(context=ctx, filter="...", offset=0, limit=30)
        >>> page2 = filter_files(context=ctx, filter="...", offset=30, limit=30)

        Anti-patterns
        -------------
        - WRONG: Guessing context paths

          >>> filter_files(context="Files/Local/reports/Q4.csv/Tables/Sheet1")

          CORRECT: Always use describe() to get exact paths

          >>> storage = describe(file_path="/reports/Q4.csv")
          >>> filter_files(context=storage.tables[0].context_path, ...)

        - WRONG: Using filter for meaning-based search

          >>> filter_files(filter="description.contains('budget')")

          CORRECT: Use search_files(references={'description': 'budget'})

        - WRONG: Fetching rows just to count them

          CORRECT: Use reduce(metric='count', columns='id') instead

        - WRONG: filter="content_id['section'] == 2" (direct dict indexing fails)

          CORRECT: filter="content_id.get('section') == 2"

        Notes
        -----
        - Context paths use file_id internally (e.g., Files/Local/42/Tables/Sheet1)
          for stability across file renames. Always use describe() to get paths.
        - For large result sets, use pagination (offset/limit) to avoid memory issues.
        - Check column_schema.column_names from describe() to see available columns.

        See Also
        --------
        describe : Get context paths and schema for a file
        search_files : Semantic search (for meaning-based queries)
        reduce : Aggregate metrics without fetching rows
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

        Use this tool when searching by meaning, topics, or concepts in text fields.
        For exact matches on structured fields (ids, statuses), use filter_files instead.

        IMPORTANT: Use describe() first to get the exact context path and check
        which columns are searchable (have embeddings):

        >>> storage = describe(file_path="/docs/report.pdf")
        >>> print(storage.document.column_schema.searchable_columns)
        ['summary', 'content_text']
        >>> results = search_files(
        ...     context=storage.document.context_path,
        ...     references={'summary': 'fire safety regulations'}
        ... )

        Parameters
        ----------
        context : str, optional
            Full Unify context path to search. Obtain this from describe():
            - storage.document.context_path for document content
            - storage.tables[0].context_path for a specific table
            - storage.index_context for the FileRecords index
            When None, defaults to the FileRecords index.

        references : dict[str, str], optional
            Mapping of column_name → reference_text for semantic matching.
            Only use columns that are searchable (have embeddings) - check
            column_schema.searchable_columns from describe().
            When omitted, returns recent rows without semantic ranking.

        limit : int, default 10
            Number of results to return. Start with 10, increase if needed.
            Maximum 100, but prefer smaller values to avoid context overflow.

        filter : str, optional
            Optional row-level predicate to narrow results before semantic ranking.
            Supports arbitrary Python expressions evaluated per row.

        columns : list[str], optional
            Specific columns to return. When None, returns all columns.
            Use this to reduce response size when only specific fields are needed.

        Returns
        -------
        list[dict]
            Up to `limit` rows ranked by similarity from the context.

        Usage Examples
        --------------
        First, always call describe() to get context paths and check searchability:

        >>> storage = describe(file_path="/docs/report.pdf")
        >>> print(storage.document.column_schema.searchable_columns)

        Search document content:

        >>> results = search_files(
        ...     context=storage.document.context_path,
        ...     references={'summary': 'fire safety regulations'},
        ...     limit=10
        ... )

        Search a specific table (check searchable columns first):

        >>> storage = describe(file_path="/data/telematics.xlsx")
        >>> table = storage.get_table("July_2025")
        >>> results = search_files(
        ...     context=table.context_path,
        ...     references={'Vehicle': 'Stuart Birks'},
        ...     limit=20
        ... )

        Combine semantic search with exact filter:

        >>> results = search_files(
        ...     context=storage.document.context_path,
        ...     references={'content_text': 'payment terms'},
        ...     filter="content_type == 'paragraph'",
        ...     limit=5
        ... )

        Tiered search strategy for documents:

        >>> # 1. Paragraphs first (highest precision):
        >>> results = search_files(
        ...     context=storage.document.context_path,
        ...     references={'summary': 'quarterly metrics'},
        ...     filter="content_type == 'paragraph'",
        ...     limit=10
        ... )
        >>> # 2. Sections if paragraphs insufficient:
        >>> results = search_files(
        ...     context=storage.document.context_path,
        ...     references={'summary': 'quarterly metrics'},
        ...     filter="content_type == 'section'",
        ...     limit=5
        ... )

        Search FileRecords index (no context needed):

        >>> results = search_files(references={'summary': 'fire safety'}, limit=10)

        Anti-patterns
        -------------
        - WRONG: Guessing context paths

          >>> search_files(context="Files/Local/docs/report.pdf/Content", ...)

          CORRECT: Always use describe() to get exact paths

          >>> storage = describe(file_path="/docs/report.pdf")
          >>> search_files(context=storage.document.context_path, ...)

        - WRONG: Using non-searchable columns in references

          >>> search_files(references={'id': 'search term'})  # id has no embeddings

          CORRECT: Check searchable_columns first

          >>> storage.document.column_schema.searchable_columns

        - WRONG: Using search_files for exact id/status lookup

          CORRECT: Use filter_files(filter="id == 123") for exact matches

        - WRONG: limit=500 (too many results flood context)

          CORRECT: Start with limit=10, increase only if needed

        Notes
        -----
        - Only columns with embeddings can be used in references. Check
          column_schema.searchable_columns from describe() before searching.
        - Context paths use file_id internally for stability across renames.
        - For best results, combine with filter to narrow the search scope.

        See Also
        --------
        describe : Get context paths, schema, and searchable columns
        filter_files : Exact match filtering (for structured fields)
        reduce : Aggregate metrics without fetching rows
        """

    @abstractmethod
    def reduce(
        self,
        *,
        context: Optional[str] = None,
        metric: str,
        columns: Union[str, List[str]],
        filter: Optional[str] = None,
        group_by: Optional[Union[str, List[str]]] = None,
    ) -> Any:
        """
        Compute aggregation metrics over a context.

        This is the PRIMARY tool for any quantitative question (counts, sums,
        averages, statistics). Always use reduce instead of fetching rows and
        computing aggregates in-memory.

        IMPORTANT: Use describe() first to get the exact context path:

        >>> storage = describe(file_path="/reports/sales.xlsx")
        >>> count = reduce(
        ...     context=storage.tables[0].context_path,
        ...     metric='count',
        ...     columns='id'
        ... )

        Parameters
        ----------
        context : str, optional
            Full Unify context path to aggregate. Obtain this from describe():
            - storage.document.context_path for document content
            - storage.tables[0].context_path for a specific table
            - storage.index_context for the FileRecords index
            When None, aggregates over the FileRecords index.

        metric : str
            Reduction metric to compute. Supported values (case-insensitive):
            "count", "sum", "mean", "min", "max", "median", "mode", "var", "std".

        columns : str | list[str]
            Column(s) to aggregate. Single column or list for multi-column metrics.
            Check column_schema.column_names from describe() for available columns.

        filter : str, optional
            Optional row-level filter expression to narrow the dataset
            BEFORE aggregation. Supports Python expressions evaluated per row.

        group_by : str | list[str], optional
            Column(s) to group by. Single column for one grouping level,
            or a list for hierarchical grouping. Result becomes a nested dict
            keyed by group values.

        Returns
        -------
        Any
            Metric value(s) computed over the context:
            - No grouping → scalar (float/int)
            - With group_by → dict keyed by group values
            - Hierarchical grouping → nested dict

        Usage Examples
        --------------
        First, always call describe() to get context paths:

        >>> storage = describe(file_path="/reports/sales.xlsx")
        >>> table_ctx = storage.tables[0].context_path

        Count rows in a table:

        >>> count = reduce(context=table_ctx, metric='count', columns='id')
        >>> # Returns: 42

        Count with filter:

        >>> count = reduce(
        ...     context=table_ctx,
        ...     metric='count',
        ...     columns='id',
        ...     filter="status == 'complete'"
        ... )
        >>> # Returns: 38

        Sum a numeric column:

        >>> total = reduce(
        ...     context=table_ctx,
        ...     metric='sum',
        ...     columns='amount',
        ...     filter="status == 'complete'"
        ... )
        >>> # Returns: 15420.50

        Average with grouping:

        >>> avg_by_category = reduce(
        ...     context=table_ctx,
        ...     metric='mean',
        ...     columns='amount',
        ...     group_by='category'
        ... )
        >>> # Returns: {'Electronics': 245.00, 'Furniture': 890.50, ...}

        Hierarchical grouping:

        >>> breakdown = reduce(
        ...     context=table_ctx,
        ...     metric='sum',
        ...     columns='revenue',
        ...     group_by=['region', 'quarter']
        ... )
        >>> # Returns: {'North': {'Q1': 1000, 'Q2': 1200}, 'South': {...}}

        Count files in FileRecords index (no context needed):

        >>> total_files = reduce(metric='count', columns='file_id')

        Anti-patterns
        -------------
        - WRONG: Guessing context paths

          >>> reduce(context="Files/Local/sales.xlsx/Tables/Orders", ...)

          CORRECT: Always use describe() to get exact paths

          >>> storage = describe(file_path="/sales.xlsx")
          >>> reduce(context=storage.tables[0].context_path, ...)

        - WRONG: filter_files(...) then count rows in Python

          CORRECT: reduce(metric='count', columns='id', filter=...)

        - WRONG: filter_files(...) then sum values in Python

          CORRECT: reduce(metric='sum', columns='amount', filter=...)

        - WRONG: Using reduce for text/categorical analysis

          CORRECT: Use search_files for semantic queries

        Notes
        -----
        - Context paths use file_id internally for stability across renames.
        - Check column_schema.column_names from describe() for available columns.
        - Use filter to narrow the dataset BEFORE aggregation for efficiency.

        See Also
        --------
        describe : Get context paths and schema for a file
        filter_files : Fetch rows with exact match filters
        search_files : Semantic search for meaning-based queries
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
        metric: Optional[str] = None,
        aggregate: Optional[str] = None,
        scale_x: Optional[str] = None,
        scale_y: Optional[str] = None,
        bin_count: Optional[int] = None,
        show_regression: Optional[bool] = None,
    ) -> Any:
        """
        Generate plot visualizations from table data via the Plot API.

        This tool creates interactive charts from the specified table(s). Before
        calling, use `describe()` to discover available contexts and then
        `list_columns` for detailed column information.

        Parameters
        ----------
        tables : str | list[str]
            Table reference(s) to visualize. Accepted forms:
            - "<storage_id>.Tables.<label>" for per-file tables
            - "<storage_id>" for per-file Content

            When a LIST is provided, the same plot configuration is applied to
            EACH table. Use this for tables with identical schemas that you want
            to compare (e.g., monthly data tables like "July_2025", "August_2025").
            Each table produces a separate plot with the table name in the title.

            For tables with DIFFERENT schemas, make separate visualize calls.

        plot_type : str
            Chart type. One of: "bar", "line", "scatter", "histogram".
            - bar: Compare values across categories
            - line: Show trends over time or sequences
            - scatter: Show correlations between two numeric variables
            - histogram: Show distribution of a single variable

        x_axis : str
            Column name for the x-axis. Must exist in the table schema.

        y_axis : str | None
            Column name for the y-axis. Required for bar, line, scatter.
            Optional for histogram (uses count by default).

        group_by : str | None
            Column to group/color data points by. Creates multiple series.

        filter : str | None
            Row-level filter expression. Same syntax as `reduce` and `filter_files`.
            Applied before plotting.

        title : str | None
            Plot title. Auto-generated if not provided.

        metric : str | None
            Metric to aggregate: "sum", "mean", "var", "std", "min", "max", "median", "mode", "count".

        aggregate : str | None
            Aggregation function for grouped data: "sum", "mean", "count", "min", "max",
            "median", "mode", "var", "std". Use when comparing aggregated values across
            groups (e.g., mean completion rate by region).

        scale_x, scale_y : str | None
            Axis scale: "linear" (default) or "log".

        bin_count : int | None
            Number of bins for histogram plots.

        show_regression : bool | None
            Show regression line (scatter plots only).

        Returns
        -------
        PlotResult | list[PlotResult]
            Single table → PlotResult with attrs: url, token, expires_in_hours, title, error
            Multiple tables → list of PlotResult, one per table

            On success: result.url contains the plot URL, result.succeeded is True
            On failure: result.error contains error message, result.succeeded is False

        Usage Examples
        --------------
        # Bar chart of job counts by operative:
        visualize(
            tables='/path/to/data.xlsx.Tables.Jobs',
            plot_type='bar',
            x_axis='OperativeName',
            y_axis='JobCount',
            title='Jobs per Operative'
        )

        # Line chart showing trend over time:
        visualize(
            tables='/path/to/data.xlsx.Tables.Sales',
            plot_type='line',
            x_axis='Date',
            y_axis='Revenue',
            group_by='Region'
        )

        # Same plot across multiple monthly tables (identical schemas):
        visualize(
            tables=[
                '/path/to/data.xlsx.Tables.July_2025',
                '/path/to/data.xlsx.Tables.August_2025',
                '/path/to/data.xlsx.Tables.September_2025',
            ],
            plot_type='bar',
            x_axis='Driver',
            y_axis='TotalDistance',
            metric='sum'
        )
        # Returns: [PlotResult(url="...", title="... (July_2025)"), ...]

        # Histogram of value distribution:
        visualize(
            tables='/path/to/data.xlsx.Tables.Orders',
            plot_type='histogram',
            x_axis='OrderValue',
            bin_count=20
        )

        # Scatter with regression line:
        visualize(
            tables='/path/to/data.xlsx.Tables.Metrics',
            plot_type='scatter',
            x_axis='TimeOnSite',
            y_axis='JobsCompleted',
            show_regression=True
        )

        Anti-patterns
        -------------
        - WRONG: Calling visualize without first checking column names
          CORRECT: Use list_columns(table='...') first to discover schema

        - WRONG: Passing tables with different schemas in same call
          CORRECT: Use separate visualize calls for different schemas

        - WRONG: Using non-existent column names
          CORRECT: Column names must match exactly (case-sensitive)

        - WRONG: histogram with y_axis (histogram uses x_axis distribution)
          CORRECT: For histogram, only specify x_axis
        """

    # ------------------------------------------------------------------ #
    # Join operations (read-only)                                        #
    # ------------------------------------------------------------------ #
    @abstractmethod
    def filter_join(
        self,
        *,
        tables: Union[str, List[str]],
        join_expr: str,
        select: Dict[str, str],
        mode: str = "inner",
        left_where: Optional[str] = None,
        right_where: Optional[str] = None,
        result_where: Optional[str] = None,
        result_limit: int = 100,
        result_offset: int = 0,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Join two sources and return rows from the joined result with optional filtering.

        Use this tool to correlate data across two tables (e.g., repairs data
        with telematics data). Push filters into left_where/right_where to
        reduce data before joining for efficiency.

        Parameters
        ----------
        tables : list[str] | str
            Exactly two table references. Accepted forms:
            - Path-first (preferred): "<storage_id>" for per-file Content,
              "<storage_id>.Tables.<label>" for per-file tables
            - Logical names from `describe()`
        join_expr : str
            Join predicate using table references as prefixes.
            Example: "Repairs.operative == Telematics.driver"
            The identifiers in join_expr should match the table references in tables.
        select : dict[str, str]
            Mapping of source expressions → output column names.
            Keys use "TableRef.column" format; values are the output names.
        mode : str
            Join mode: "inner" (default), "left", "right", or "outer".
        left_where, right_where : str | None
            Optional filters applied to left/right tables BEFORE joining.
            Supports arbitrary Python expressions. Use to narrow inputs.
        result_where : str | None
            Filter applied to the joined result. Supports arbitrary Python
            expressions but can only reference column names from `select` output.
        result_limit, result_offset : int
            Pagination parameters for the result. Start with limit=30.

        Returns
        -------
        dict[str, list[dict[str, Any]]]
            Rows from the materialized join context.

        Usage Examples
        --------------
        # Join repairs with telematics by operative name:
        filter_join(
            tables=[
                '/path/repairs.xlsx.Tables.Jobs',
                '/path/telematics.xlsx.Tables.July_2025'
            ],
            join_expr="Jobs.OperativeWhoCompletedJob == July_2025.Driver",
            select={
                'Jobs.JobTicketReference': 'job_ref',
                'Jobs.FullAddress': 'address',
                'July_2025.Departure': 'trip_start',
                'July_2025.Arrival': 'trip_end'
            },
            mode='inner',
            result_limit=30
        )

        # With input filters to narrow before joining:
        filter_join(
            tables=['TableA', 'TableB'],
            join_expr="TableA.id == TableB.a_id",
            select={'TableA.id': 'id', 'TableB.score': 'score'},
            left_where="TableA.status == 'active'",
            right_where="TableB.score > 0",
            result_limit=50
        )

        # Left join to keep all left rows:
        filter_join(
            tables=['Orders', 'Shipments'],
            join_expr="Orders.order_id == Shipments.order_id",
            select={'Orders.order_id': 'oid', 'Shipments.shipped_date': 'shipped'},
            mode='left',
            result_limit=30
        )

        # Filter the joined result (use output column names from select):
        filter_join(
            tables=['A', 'B'],
            join_expr="A.id == B.id",
            select={'A.name': 'name', 'B.value': 'val'},
            result_where="val > 100",  # Uses 'val' from select, not 'B.value'
            result_limit=30
        )

        Anti-patterns
        -------------
        - WRONG: result_where="B.value > 100" (references original table column)
          CORRECT: result_where="val > 100" (uses output name from select)

        - WRONG: Joining without input filters when tables are large
          CORRECT: Use left_where/right_where to narrow before joining

        - WRONG: result_limit=500 (too many rows)
          CORRECT: Start with result_limit=30, paginate with result_offset
        """

    @abstractmethod
    def search_join(
        self,
        *,
        tables: Union[str, List[str]],
        join_expr: str,
        select: Dict[str, str],
        mode: str = "inner",
        left_where: Optional[str] = None,
        right_where: Optional[str] = None,
        references: Optional[Dict[str, str]] = None,
        k: int = 10,
        filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Perform a semantic search over the result of joining two sources.

        Use this when you need to join tables AND rank results by semantic
        similarity to a query. Combines the power of joins with semantic search.

        Parameters
        ----------
        tables : list[str] | str
            Exactly two table references. Use path-first format:
            - Path-first (preferred): "<storage_id>" for per-file Content,
              "<storage_id>.Tables.<label>" for per-file tables
            - Logical names from `describe()`
        join_expr : str
            Join predicate using table references as prefixes.
        select : dict[str, str]
            Mapping of source expressions → output column names.
            Include text columns you want to search semantically.
        mode : str
            Join mode: "inner" (default), "left", "right", or "outer".
        left_where, right_where : str | None
            Optional filters applied to tables BEFORE joining.
            Supports arbitrary Python expressions.
        references : dict[str, str] | None
            Mapping of OUTPUT column names → reference text for semantic ranking.
            Use column names from the `select` values, not original table columns.
        k : int
            Maximum rows to return. Start with 10.
        filter : str | None
            Optional predicate over the joined result before ranking.
            Supports arbitrary Python expressions; uses output column names.

        Returns
        -------
        list[dict[str, Any]]
            Top-k rows ranked by semantic similarity.

        Usage Examples
        --------------
        # Join repairs and telematics, then search by address:
        search_join(
            tables=[
                '/path/repairs.xlsx.Tables.Jobs',
                '/path/telematics.xlsx.Tables.July_2025'
            ],
            join_expr="Jobs.OperativeWhoCompletedJob == July_2025.Driver",
            select={
                'Jobs.JobTicketReference': 'job_ref',
                'Jobs.FullAddress': 'address',
                'July_2025.End location': 'destination'
            },
            references={'address': 'Birmingham city centre'},
            k=10
        )

        # Combine semantic search with exact filter on joined result:
        search_join(
            tables=['Products', 'Reviews'],
            join_expr="Products.id == Reviews.product_id",
            select={
                'Products.name': 'product_name',
                'Reviews.text': 'review_text',
                'Reviews.rating': 'rating'
            },
            references={'review_text': 'excellent quality'},
            filter="rating >= 4",
            k=20
        )

        Anti-patterns
        -------------
        - WRONG: references={'Jobs.FullAddress': 'query'} (uses original column)
          CORRECT: references={'address': 'query'} (uses select output name)

        - WRONG: filter="Jobs.status == 'complete'" (uses original column)
          CORRECT: Include status in select, then filter="status == 'complete'"
        """

    @abstractmethod
    def filter_multi_join(
        self,
        *,
        joins: List[Dict[str, Any]],
        result_where: Optional[str] = None,
        result_limit: int = 100,
        result_offset: int = 0,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Chain multiple joins, then return rows from the final joined result.

        Use this for complex queries requiring more than two tables. Each step
        can reference the previous step's result using "$prev".

        Parameters
        ----------
        joins : list[dict]
            Ordered steps. Each step is a dict with:
            - tables: list of 2 refs. Use "$prev" to reference previous step's result.
            - join_expr: join predicate using table refs or "prev" as prefix.
            - select: dict mapping source expressions → output names.
            - mode (optional): "inner", "left", "right", "outer".
            - left_where, right_where (optional): input filters (arbitrary Python).
        result_where : str | None
            Filter applied to the FINAL result. Supports arbitrary Python
            expressions; uses column names from the last step's select output.
        result_limit, result_offset : int
            Pagination parameters. Start with limit=30.

        Returns
        -------
        dict[str, list[dict[str, Any]]]
            Rows from the final materialized context.

        Usage Examples
        --------------
        # Chain three tables: A → B → C
        filter_multi_join(
            joins=[
                # Step 1: Join A and B
                {
                    'tables': ['TableA', 'TableB'],
                    'join_expr': 'TableA.id == TableB.a_id',
                    'select': {'TableA.id': 'id', 'TableB.value': 'b_value'}
                },
                # Step 2: Join previous result with C
                {
                    'tables': ['$prev', 'TableC'],
                    'join_expr': 'prev.id == TableC.ref_id',
                    'select': {'prev.id': 'id', 'prev.b_value': 'b_val', 'TableC.name': 'c_name'}
                }
            ],
            result_where="b_val > 100",
            result_limit=30
        )

        # Real example: Repairs → Telematics July → Telematics August
        filter_multi_join(
            joins=[
                {
                    'tables': [
                        '/path/repairs.xlsx.Tables.Jobs',
                        '/path/telematics.xlsx.Tables.July_2025'
                    ],
                    'join_expr': 'Jobs.OperativeWhoCompletedJob == July_2025.Driver',
                    'select': {
                        'Jobs.JobTicketReference': 'job_ref',
                        'Jobs.OperativeWhoCompletedJob': 'operative',
                        'July_2025.Trip': 'july_trip'
                    },
                    'left_where': "Jobs.status == 'complete'"
                },
                {
                    'tables': ['$prev', '/path/telematics.xlsx.Tables.August_2025'],
                    'join_expr': 'prev.operative == August_2025.Driver',
                    'select': {
                        'prev.job_ref': 'job_ref',
                        'prev.operative': 'operative',
                        'prev.july_trip': 'july_trip',
                        'August_2025.Trip': 'august_trip'
                    }
                }
            ],
            result_limit=30
        )

        Anti-patterns
        -------------
        - WRONG: Using original table column in result_where
          CORRECT: result_where uses column names from the LAST step's select

        - WRONG: Referencing "$prev" in the first step (there's no previous result)
          CORRECT: First step must use two actual table references

        - WRONG: Not propagating needed columns through each step's select
          CORRECT: Each step's select must include columns needed by later steps
        """

    @abstractmethod
    def search_multi_join(
        self,
        *,
        joins: List[Dict[str, Any]],
        references: Optional[Dict[str, str]] = None,
        k: int = 10,
        filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Perform a semantic search over a chain of joined results.

        Combines multi-step joins with semantic ranking. Use when you need to
        correlate 3+ tables and rank results by meaning.

        Parameters
        ----------
        joins : list[dict]
            Ordered steps. Each step is a dict with:
            - tables: list of 2 refs. Use "$prev" to reference previous step.
            - join_expr: join predicate.
            - select: dict mapping source expressions → output names.
            - mode, left_where, right_where (optional; arbitrary Python for filters).
        references : dict[str, str] | None
            Mapping of FINAL output column names → reference text for ranking.
            Column names must be from the LAST step's select output.
        k : int
            Maximum rows to return. Start with 10.
        filter : str | None
            Optional predicate applied before semantic ranking.
            Supports arbitrary Python expressions; uses column names from the
            LAST step's select output.

        Returns
        -------
        list[dict[str, Any]]
            Top-k rows ranked by semantic similarity.

        Usage Examples
        --------------
        # Join three tables and search semantically:
        search_multi_join(
            joins=[
                {
                    'tables': ['Jobs', 'Visits'],
                    'join_expr': 'Jobs.id == Visits.job_id',
                    'select': {
                        'Jobs.id': 'job_id',
                        'Jobs.description': 'job_desc',
                        'Visits.notes': 'visit_notes'
                    }
                },
                {
                    'tables': ['$prev', 'Operatives'],
                    'join_expr': 'prev.job_id == Operatives.job_id',
                    'select': {
                        'prev.job_id': 'job_id',
                        'prev.job_desc': 'description',
                        'Operatives.name': 'operative_name'
                    }
                }
            ],
            references={'description': 'heating system repair'},
            k=15
        )

        # With filter before semantic ranking:
        search_multi_join(
            joins=[...],
            references={'summary': 'budget updates'},
            filter="status == 'approved'",
            k=10
        )

        Anti-patterns
        -------------
        - WRONG: references={'Jobs.description': 'query'} (uses original column)
          CORRECT: references={'description': 'query'} (uses final select output)

        - WRONG: Not including semantic target column in final step's select
          CORRECT: Ensure the column you want to search is in the last select
        """

    # ------------------------------------------------------------------ #
    # Mutation operations                                                #
    # ------------------------------------------------------------------ #
    @abstractmethod
    def rename_file(
        self,
        *,
        file_id_or_path: Union[str, int],
        new_name: str,
    ) -> Dict[str, Any]:
        """
        Rename a file in the underlying filesystem and update index/context metadata.

        This tool renames the file on disk AND updates all associated metadata
        in the index. The file stays in its current directory; only the filename
        changes. For moving to a different directory, use move_file instead.

        Parameters
        ----------
        file_id_or_path : str | int
            Either the file_id (int) from FileRecords, or the fully-qualified
            file_path (str) as stored in the index. Use absolute paths for reliability.
        new_name : str
            New filename (not full path). Must include the file extension.
            Example: "report_2024.pdf", not "/new/path/report.pdf"

        Returns
        -------
        dict
            Result containing the new path and any adapter-specific metadata.
            {"file_path": str, "file_name": str, ...}

        Raises
        ------
        PermissionError
            If the file is protected or rename is not permitted.
        ValueError
            If the file does not exist in the index.

        Usage Examples
        --------------
        # Rename by file path:
        rename_file(
            file_id_or_path="/path/to/old_name.pdf",
            new_name="new_name.pdf"
        )

        # Rename by file_id:
        rename_file(file_id_or_path=42, new_name="renamed_file.xlsx")

        # Rename with extension change (if adapter permits):
        rename_file(
            file_id_or_path="/path/to/document.doc",
            new_name="document.docx"
        )

        Anti-patterns
        -------------
        - WRONG: new_name="/full/path/to/file.pdf" (including directory)
          CORRECT: new_name="file.pdf" (filename only)

        - WRONG: Renaming without verifying the file exists first
          CORRECT: Use describe() or ask() to verify file exists before renaming

        - WRONG: new_name="file" (missing extension)
          CORRECT: new_name="file.pdf" (include extension)

        - WRONG: Renaming protected files
          CORRECT: Check is_protected() first or handle PermissionError
        """

    @abstractmethod
    def move_file(
        self,
        *,
        file_id_or_path: Union[str, int],
        new_parent_path: str,
    ) -> Dict[str, Any]:
        """
        Move a file to a different directory and update index/context metadata.

        This tool moves the file on disk AND updates all associated metadata
        in the index. The filename stays the same; only the directory changes.
        For renaming the file, use rename_file instead.

        Parameters
        ----------
        file_id_or_path : str | int
            Either the file_id (int) from FileRecords, or the fully-qualified
            file_path (str) as stored in the index. Use absolute paths for reliability.
        new_parent_path : str
            Destination directory path. Must be an absolute path to an existing
            directory. Example: "/new/destination/folder"

        Returns
        -------
        dict
            Result containing the new path and any adapter-specific metadata.
            {"file_path": str, ...}

        Raises
        ------
        PermissionError
            If the file is protected or move is not permitted.
        ValueError
            If the file does not exist in the index.

        Usage Examples
        --------------
        # Move file to a different directory:
        move_file(
            file_id_or_path="/path/to/file.pdf",
            new_parent_path="/archive/2024"
        )

        # Move by file_id:
        move_file(file_id_or_path=42, new_parent_path="/processed")

        # Organize files into folders:
        # First query files to move:
        files = filter_files(filter="status == 'success' and file_format == 'pdf'")
        for f in files:
            move_file(file_id_or_path=f["file_path"], new_parent_path="/documents/pdfs")

        Anti-patterns
        -------------
        - WRONG: new_parent_path="/destination/newname.pdf" (including filename)
          CORRECT: new_parent_path="/destination" (directory only)

        - WRONG: Moving to a non-existent directory
          CORRECT: Ensure destination directory exists first

        - WRONG: Moving without verifying the file exists
          CORRECT: Use describe() or ask() to verify file before moving

        - WRONG: Cross-filesystem moves (moving between different adapters)
          CORRECT: Move only within the same filesystem adapter
        """

    @abstractmethod
    def delete_file(
        self,
        *,
        file_id_or_path: Union[str, int],
    ) -> Dict[str, Any]:
        """
        Delete a file from the filesystem and purge all related index data.

        This tool removes the file from disk (when adapter supports it) AND
        purges all associated index entries, Content rows, and per-file tables.
        This is a DESTRUCTIVE operation that cannot be undone.

        Parameters
        ----------
        file_id_or_path : str | int
            Either the file_id (int) from FileRecords, or the fully-qualified
            file_path (str) as stored in the index. Use absolute paths for reliability.

        Returns
        -------
        dict
            {"outcome": "file deleted", "details": {"file_id": int, "file_path": str}}

        Raises
        ------
        ValueError
            If the file does not exist in the index.
        PermissionError
            If the file is protected.

        Usage Examples
        --------------
        # Delete by file path:
        delete_file(file_id_or_path="/path/to/obsolete_file.pdf")
        # Returns: {"outcome": "file deleted", "details": {...}}

        # Delete by file_id:
        delete_file(file_id_or_path=42)

        # Clean up failed ingests:
        failed = filter_files(filter="status == 'error'")
        for f in failed:
            delete_file(file_id_or_path=f["file_path"])

        # Delete outdated files:
        old_files = filter_files(filter="created_at < '2023-01-01'")
        for f in old_files:
            delete_file(file_id_or_path=f["file_path"])

        Anti-patterns
        -------------
        - WRONG: Deleting without verifying the file first
          CORRECT: Use ask() or describe() to confirm file identity before deleting

        - WRONG: Attempting to delete protected files without handling errors
          CORRECT: Check is_protected() first or catch PermissionError

        - WRONG: Deleting files that other processes depend on
          CORRECT: Verify file is not referenced elsewhere before deletion

        - WRONG: Bulk deletion without confirmation
          CORRECT: For bulk operations, verify the filter returns expected files first
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

        Use this when a file's contents have changed on disk and you need to
        update the index. This tool purges existing Content/Table rows and
        re-parses the file from scratch, ensuring the index reflects current
        file contents.

        Parameters
        ----------
        file_path : str
            The file identifier/path as used in FileRecords.file_path.
            Use absolute paths for reliability.

        Returns
        -------
        dict
            Outcome with details:
            - "outcome": "sync complete" or "sync failed"
            - "purged": {"content_rows": int, "table_rows": int}
            - "error": str (only on failure)

        Usage Examples
        --------------
        # Sync a single file after it was updated:
        sync(file_path="/path/to/updated_document.pdf")
        # Returns: {"outcome": "sync complete", "purged": {"content_rows": 45, "table_rows": 3}}

        # Re-index after external modifications:
        modified_files = ["/path/to/a.pdf", "/path/to/b.xlsx"]
        for f in modified_files:
            result = sync(file_path=f)
            if result["outcome"] == "sync failed":
                print(f"Failed to sync {f}: {result.get('error')}")

        # Verify sync worked:
        result = sync(file_path="/path/to/file.pdf")
        if result["outcome"] == "sync complete":
            # Query the updated content
            search_files(table="/path/to/file.pdf", ...)

        Anti-patterns
        -------------
        - WRONG: Using sync on a file that doesn't exist in the filesystem
          CORRECT: Verify exists() returns True before syncing

        - WRONG: Syncing a file that hasn't changed (unnecessary work)
          CORRECT: Only sync when you know the source file was modified

        - WRONG: Expecting sync to preserve manual index modifications
          CORRECT: Sync purges ALL existing rows and re-ingests from source

        - WRONG: Using sync for initial ingestion
          CORRECT: Use ingest_files for new files; sync is for updates only
        """


# Attach centralised docstring
BaseFileManager.clear.__doc__ = CLEAR_METHOD_DOCSTRING
