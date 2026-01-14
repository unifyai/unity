"""
Base abstract class for DataManager.

This module defines the public contract for the canonical data operations layer.
All docstrings are defined here and inherited by concrete implementations via
``@functools.wraps``.

IMPORTANT: Do not duplicate docstrings in concrete implementations.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from unity.common.state_managers import BaseStateManager

if TYPE_CHECKING:
    from unity.data_manager.types.table import TableDescription
    from unity.data_manager.types.plot import PlotResult


class BaseDataManager(BaseStateManager):
    """
    Public contract for the canonical data operations layer.

    DataManager provides low-level primitives for data manipulation that work
    on ANY Unify context. It is the single source of truth for:

    - **Query operations**: filter, search, reduce
    - **Join operations**: filter_join, search_join, filter_multi_join, search_multi_join
    - **Mutation operations**: insert_rows, update_rows, delete_rows
    - **Embedding operations**: ensure_vector_column, vectorize_rows
    - **Visualization**: plot, plot_batch

    Semantic Ownership
    ------------------
    DataManager is the semantic owner of the ``Data/*`` namespace, which
    represents datasets as first-class entities (not tied to file lifecycle).

    However, DataManager's primitives work on ANY context, including ``Files/*``.
    This allows FileManager to delegate its data operations internally while
    retaining semantic ownership of file-derived contexts.

    Context Resolution
    ------------------
    All methods accepting a ``context`` parameter support:

    - **Relative paths**: Resolved against DataManager's base context.
      Example: ``"MH/arrears"`` → ``"{base_ctx}/MH/arrears"``
    - **Absolute owned paths**: Used as-is.
      Example: ``"Data/MH/arrears"``
    - **Foreign paths**: Used as-is for cross-namespace operations.
      Example: ``"Files/Local/120/Tables/Sheet1"``

    Usage Patterns
    --------------
    - **Direct usage**: Actor/FunctionManager calls DataManager primitives directly
      for pipeline/API data stored in ``Data/*``.
    - **Via FileManager**: FileManager's convenience methods (filter_files, search_files)
      internally delegate to DataManager for execution.

    No Tool Loops
    -------------
    DataManager exposes pure primitives with no ask/update tool loops.
    High-level orchestration is handled by Actor composing these primitives.

    Docstring Requirements for Subclass Methods
    -------------------------------------------
    All public methods (primitives) MUST include comprehensive docstrings with:

    1. **One-line summary** - What the method does
    2. **Extended description** - When to use, contrasted with similar methods
    3. **Parameters section** - EVERY parameter with:
       - Type annotation
       - Detailed description of expected values
       - Default behavior when optional
       - Context resolution behavior (for context params)
    4. **Returns section** - Return type and structure
    5. **Raises section** - Exceptions that may be raised
    6. **Usage Examples** - Concrete code examples showing common patterns
    7. **Anti-patterns section** - What NOT to do and why
    8. **Notes section** - Additional context, invariants, edge cases

    This is CRITICAL because:
    - Actor/FunctionManager reads docstrings to understand primitive usage
    - LLMs compose primitives based on docstring content
    - No external documentation - docstrings ARE the documentation

    See ``FileManager.filter_files``, ``FileManager.search_files`` for reference patterns.
    """

    _as_caller_description: str = (
        "a DataManager, performing data operations on behalf of the system"
    )

    # ──────────────────────────────────────────────────────────────────────────
    # Table Management
    # ──────────────────────────────────────────────────────────────────────────

    @abstractmethod
    def create_table(
        self,
        context: str,
        *,
        description: Optional[str] = None,
        fields: Optional[Dict[str, str]] = None,
        unique_keys: Optional[Dict[str, str]] = None,
        auto_counting: Optional[Dict[str, Optional[str]]] = None,
    ) -> str:
        """
        Create a new table context in Unify.

        Use this method to create a new table with optional schema definition.
        Tables can be created empty and populated later via ``insert_rows``,
        or created with a predefined schema for type enforcement.

        This is NOT for creating file-derived tables (use FileManager.ingest_files
        for that). Use this for:
        - Pipeline outputs stored in ``Data/*``
        - Intermediate calculation results
        - API/warehouse data ingestion targets

        Parameters
        ----------
        context : str
            Target context path for the new table. Accepts:

            - **Relative paths**: Resolved against DataManager's base context.
              Example: ``"MH/arrears"`` → ``"{base_ctx}/MH/arrears"``
            - **Absolute owned paths**: Used as-is.
              Example: ``"Data/MH/arrears"``
            - **Foreign paths**: Used as-is for cross-namespace operations.
              Example: ``"Files/Local/120/Tables/Sheet1"``

        description : str | None, default ``None``
            Human-readable description of the table purpose. Strongly recommended
            for discoverability when Actor explores available contexts.

        fields : dict[str, str] | None, default ``None``
            Mapping of field names to Unify types. Supported types:
            ``"str"``, ``"int"``, ``"float"``, ``"bool"``, ``"datetime"``,
            ``"list"``, ``"dict"``.

            Example: ``{"name": "str", "amount": "float", "created_at": "datetime"}``

            When ``None``, the table is created without a fixed schema and
            fields are inferred from the first ``insert_rows`` call.

        unique_keys : dict[str, str] | None, default ``None``
            Mapping of unique key columns to their types. Unify enforces
            uniqueness on these columns during inserts.

            Example: ``{"property_id": "int"}``

        auto_counting : dict[str, str | None] | None, default ``None``
            Columns with auto-incrementing behavior. Keys are column names,
            values are the scoping column (or ``None`` for global auto-increment).

            Example: ``{"row_id": None}`` → row_id auto-increments globally
            Example: ``{"instance_id": "task_id"}`` → instance_id auto-increments per task_id

        Returns
        -------
        str
            The fully resolved context path that was created.

        Raises
        ------
        ValueError
            If the context already exists and schema conflicts.

        Usage Examples
        --------------
        # Create a table in DataManager's owned namespace
        dm.create_table(
            "MH/arrears",  # Relative path → Data/MH/arrears
            description="Tenant arrears aggregated weekly",
            fields={"tenant_id": "int", "amount": "float", "week": "str"},
            unique_keys={"tenant_id": "int"},
        )

        # Create a table with explicit absolute path
        dm.create_table(
            "Data/Analytics/metrics",
            fields={"metric_name": "str", "value": "float"},
        )

        # Create without schema (fields inferred on first insert)
        dm.create_table("staging/raw_api_response")

        # Create with auto-incrementing ID
        dm.create_table(
            "Data/logs/events",
            fields={"event_id": "int", "message": "str", "timestamp": "datetime"},
            auto_counting={"event_id": None},  # Global auto-increment
        )

        Anti-patterns
        -------------
        - WRONG: Creating tables for file-derived data directly.
          CORRECT: Let FileManager.ingest_files create the table structure.

        - WRONG: Using this for temporary intermediate results that fit in memory.
          CORRECT: Process in Python, only persist when needed.

        - WRONG: Creating many small single-use tables without cleanup.
          CORRECT: Use meaningful, reusable table names or delete after use.

        Notes
        -----
        - For ``Data/*`` contexts, DataManager is the semantic owner.
        - For ``Files/*`` contexts, FileManager is the semantic owner but may
          use this method internally during ingestion.
        - If the table already exists with a compatible schema, no error is raised.
        """

    @abstractmethod
    def describe_table(self, context: str) -> "TableDescription":
        """
        Get schema, row count, and metadata for a table.

        Use this to understand a table's structure before querying, especially
        when working with unfamiliar contexts or validating schema assumptions.

        Parameters
        ----------
        context : str
            Full context path of the table to describe. Accepts:

            - **Relative paths**: Resolved against DataManager's base context.
            - **Absolute paths**: Used as-is (e.g., ``"Data/MH/arrears"`` or
              ``"Files/Local/120/Tables/Sheet1"``).

        Returns
        -------
        TableDescription
            Typed model containing:
            - ``context``: The resolved context path
            - ``description``: Human-readable description (if set)
            - ``schema``: Column definitions with types and searchability
            - ``row_count``: Current number of rows
            - ``has_embeddings``: Whether any columns have vector embeddings
            - ``embedding_columns``: List of embedding column names

        Raises
        ------
        ValueError
            If the context does not exist.

        Usage Examples
        --------------
        # Describe a table in Data/*
        desc = dm.describe_table("MH/arrears")
        print(f"Columns: {desc.schema.column_names}")
        print(f"Rows: {desc.row_count}")
        print(f"Searchable columns: {desc.schema.searchable_columns}")

        # Check if table has embeddings before semantic search
        desc = dm.describe_table("Data/products")
        if desc.has_embeddings:
            results = dm.search(desc.context, query="premium quality", k=5)

        Anti-patterns
        -------------
        - WRONG: Calling describe_table in a loop for many tables.
          CORRECT: Use list_tables with prefix filter, then batch describe if needed.

        - WRONG: Assuming column types without checking.
          CORRECT: Use describe_table to verify schema before operations.

        Notes
        -----
        - Row count is fetched live; may have minor latency for large tables.
        - Embedding columns are detected by the ``_<name>_emb`` naming pattern.
        """

    @abstractmethod
    def list_tables(self, *, prefix: Optional[str] = None) -> List[str]:
        """
        List all table contexts, optionally filtered by prefix.

        Use this to discover available tables within a namespace or find
        tables matching a naming pattern.

        Parameters
        ----------
        prefix : str | None, default ``None``
            Context path prefix to filter by. Only tables whose paths start
            with this prefix are returned.

            Example: ``"Data/MidlandHeart"`` returns all MH tables.
            Example: ``"Files/Local/120"`` returns all contexts for file ID 120.

            When ``None``, returns all accessible tables (may be slow/large).

        Returns
        -------
        list[str]
            List of fully-qualified context paths matching the prefix.
            Sorted alphabetically.

        Usage Examples
        --------------
        # List all tables in a project namespace
        tables = dm.list_tables(prefix="Data/MidlandHeart")
        # Returns: ["Data/MidlandHeart/arrears", "Data/MidlandHeart/properties", ...]

        # List all tables for a specific file
        file_tables = dm.list_tables(prefix="Files/Local/120")
        # Returns: ["Files/Local/120/Content", "Files/Local/120/Tables/Sheet1", ...]

        # List all Data/* tables
        all_data = dm.list_tables(prefix="Data/")

        Anti-patterns
        -------------
        - WRONG: Calling list_tables() without prefix to iterate all tables.
          CORRECT: Always use a meaningful prefix to limit scope.

        - WRONG: Parsing table names to extract metadata.
          CORRECT: Use describe_table for structured metadata.

        Notes
        -----
        - Results are limited by access permissions.
        - Large result sets may be truncated; use specific prefixes.
        """

    @abstractmethod
    def delete_table(
        self,
        context: str,
        *,
        dangerous_ok: bool = False,
    ) -> None:
        """
        Delete a table context and all its data.

        **WARNING**: This is a destructive operation that cannot be undone.
        All data in the table will be permanently deleted.

        Parameters
        ----------
        context : str
            Full context path of the table to delete.

        dangerous_ok : bool, default ``False``
            Safety flag that MUST be set to ``True`` to confirm the deletion.
            This prevents accidental data loss.

        Raises
        ------
        ValueError
            If ``dangerous_ok`` is ``False`` (safety guard).
        ValueError
            If the context does not exist.

        Usage Examples
        --------------
        # Delete a temporary staging table
        dm.delete_table("Data/staging/temp_import", dangerous_ok=True)

        # Clean up after tests
        dm.delete_table(test_context, dangerous_ok=True)

        Anti-patterns
        -------------
        - WRONG: Setting dangerous_ok=True without understanding consequences.
          CORRECT: Only delete tables you are certain should be removed.

        - WRONG: Deleting FileManager-owned contexts directly.
          CORRECT: Use FileManager methods for file-derived contexts.

        Notes
        -----
        - Deletion is immediate and permanent.
        - Associated embeddings are also deleted.
        - References from other tables are NOT automatically cleaned up.
        """

    # ──────────────────────────────────────────────────────────────────────────
    # Query Operations
    # ──────────────────────────────────────────────────────────────────────────

    @abstractmethod
    def filter(
        self,
        context: str,
        *,
        filter: Optional[str] = None,
        columns: Optional[List[str]] = None,
        limit: int = 100,
        offset: int = 0,
        order_by: Optional[str] = None,
        descending: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Filter rows from a table by expression.

        Use this tool for **exact matches** on structured fields (ids, statuses,
        dates, numeric comparisons). For **semantic/meaning-based** lookups,
        use ``search()`` instead.

        This is the workhorse query method for retrieving rows that match
        specific criteria. It supports flexible filtering, pagination, and
        column selection.

        Parameters
        ----------
        context : str
            Target context path. Accepts:

            - **Relative paths**: Resolved against DataManager's base context.
              Example: ``"MH/arrears"`` → ``"{base_ctx}/MH/arrears"``
            - **Absolute owned paths**: Used as-is.
              Example: ``"Data/MH/arrears"``
            - **Foreign paths**: Used as-is for cross-namespace operations.
              Example: ``"Files/Local/120/Tables/Sheet1"``

        filter : str | None, default ``None``
            Arbitrary Python expression evaluated with column names in scope.
            The expression is evaluated per row; any valid Python syntax that
            returns a boolean is supported. **String values MUST be quoted.**
            Use ``.get()`` for safe dict access on nested fields.

            When ``None``, all rows are returned (subject to limit/offset).

            Filter syntax examples:
            - Equality: ``"status == 'active'"``
            - Comparison: ``"amount > 1000"``
            - Range: ``"amount >= 100 and amount <= 500"``
            - Date range: ``"created_at >= '2024-01-01' and created_at < '2024-02-01'"``
            - IN-style: ``"status in ('pending', 'review', 'approved')"``
            - Nested access: ``"metadata.get('category') == 'urgent'"``
            - Boolean logic: ``"is_active and not is_deleted"``

        columns : list[str] | None, default ``None``
            Columns to return in the result. When ``None``, all columns are
            returned. Use this to reduce payload size when only specific
            fields are needed.

            Example: ``columns=["id", "name", "status"]``

        limit : int, default ``100``
            Maximum rows to return. Use for pagination or limiting large result sets.
            Set higher for exhaustive queries, but be mindful of memory.

        offset : int, default ``0``
            Zero-based pagination offset. Skip this many rows before returning.
            Use with ``limit`` for pagination: page N = ``offset=N*limit``.

        order_by : str | None, default ``None``
            Column name to sort results by. Combined with ``descending`` to
            control sort direction. When ``None``, order is not guaranteed.

        descending : bool, default ``False``
            When ``True`` and ``order_by`` is set, sort in descending order.

        Returns
        -------
        list[dict]
            List of row dictionaries containing the requested columns.
            Each dict maps column names to values. Empty list if no matches.

        Usage Examples
        --------------
        # Get all rows (up to limit)
        rows = dm.filter("MH/arrears")

        # Filter by exact match
        dm.filter("MH/arrears", filter="status == 'overdue'")

        # Filter with numeric comparison
        dm.filter("MH/arrears", filter="amount > 1000 and amount < 5000")

        # Filter with date range
        dm.filter(
            "MH/arrears",
            filter="created_at >= '2024-01-01' and created_at < '2024-02-01'"
        )

        # Paginate through results
        page1 = dm.filter("MH/arrears", limit=50, offset=0)
        page2 = dm.filter("MH/arrears", limit=50, offset=50)

        # Select specific columns only
        dm.filter("MH/arrears", columns=["tenant_id", "amount"], filter="amount > 0")

        # Filter with safe dict access (for nested fields)
        dm.filter("Data/logs", filter="metadata.get('level') == 'error'")

        # Cross-namespace operation (FileManager context)
        dm.filter(
            "Files/Local/120/Tables/Sheet1",
            filter="region == 'East'"
        )

        # Sorted results
        dm.filter("MH/arrears", order_by="amount", descending=True, limit=10)

        Anti-patterns
        -------------
        - WRONG: ``filter="visit_date == '2024-01'"`` (partial date equality fails)
          CORRECT: ``filter="visit_date >= '2024-01-01' and visit_date < '2024-02-01'"``

        - WRONG: ``filter="description.contains('budget')"`` (substring for meaning)
          CORRECT: Use ``search(query='budget related content')`` instead

        - WRONG: ``filter="metadata['level'] == 'error'"`` (direct dict indexing may fail)
          CORRECT: ``filter="metadata.get('level') == 'error'"``

        - WRONG: Fetching all rows just to count them
          CORRECT: Use ``reduce(metric='count')`` instead

        - WRONG: Large limit without pagination for unknown result size
          CORRECT: Use reasonable limit and paginate with offset

        Notes
        -----
        - Filter expressions are evaluated in Python; use ``and``/``or`` not ``AND``/``OR``
        - String values in filter expressions MUST be quoted: ``"status == 'active'"``
        - For OR conditions: ``"status == 'active' or status == 'pending'"``
        - For IN-style queries: ``"status in ('active', 'pending', 'review')"``
        - Empty filter (``None``) returns all rows up to the limit
        - Results are not guaranteed to be in any order unless ``order_by`` is specified
        """

    @abstractmethod
    def search(
        self,
        context: str,
        *,
        query: str,
        k: int = 10,
        filter: Optional[str] = None,
        vector_column: Optional[str] = None,
        columns: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Semantic search over embedded column.

        Use this tool when searching by **meaning, topics, or concepts** in text fields.
        For **exact matches** on structured fields, use ``filter()`` instead.

        This performs vector similarity search using embeddings. The table must
        have an embedding column (``_<name>_emb``) for the column being searched.
        Use ``describe_table()`` to check which columns are searchable.

        Parameters
        ----------
        context : str
            Target context path. Accepts relative, absolute owned, or foreign paths.
            See ``filter()`` for full context resolution documentation.

        query : str
            Natural language query to embed and match against the vector column.
            The query is embedded using the same model as the column embeddings.

            Good queries are:
            - Descriptive: "budget allocation for Q4 marketing"
            - Conceptual: "customer complaints about delivery"
            - Topic-based: "renewable energy initiatives"

        k : int, default ``10``
            Number of results to return. Results are ranked by similarity score.

        filter : str | None, default ``None``
            Additional row-level predicate to narrow results BEFORE semantic ranking.
            Same syntax as ``filter()`` method.

            Example: ``filter="year == 2024"`` to only search 2024 documents.

        vector_column : str | None, default ``None``
            Specific embedding column to search. If ``None``, uses the default
            searchable column (usually ``text`` or the first embedded column).

            Use ``describe_table()`` to find available embedding columns:
            ``desc.schema.searchable_columns``

        columns : list[str] | None, default ``None``
            Columns to return in results. When ``None``, all columns returned.

        Returns
        -------
        list[dict]
            List of row dictionaries ranked by semantic similarity.
            Each dict includes a ``_similarity`` score (0-1, higher is better).
            Results are sorted by similarity descending.

        Raises
        ------
        ValueError
            If the context has no embedding columns (not searchable).

        Usage Examples
        --------------
        # Basic semantic search
        results = dm.search(
            "Data/documents",
            query="budget planning for next fiscal year",
            k=5
        )

        # Search with filter to narrow scope
        results = dm.search(
            "Data/support_tickets",
            query="delivery problems",
            k=10,
            filter="status == 'open'"
        )

        # Search specific embedded column
        results = dm.search(
            "Data/products",
            query="eco-friendly materials",
            vector_column="description"
        )

        # Cross-namespace search (file content)
        results = dm.search(
            "Files/Local/120/Content",
            query="executive summary recommendations",
            k=5
        )

        Anti-patterns
        -------------
        - WRONG: Using search for exact matches like ``query="status:active"``
          CORRECT: Use ``filter(filter="status == 'active'")`` instead

        - WRONG: Very short queries like ``query="budget"``
          CORRECT: Use descriptive queries: ``query="Q4 marketing budget allocation"``

        - WRONG: Assuming embeddings exist without checking
          CORRECT: Use ``describe_table()`` to verify searchable columns

        - WRONG: Large k values (k=1000) for exhaustive retrieval
          CORRECT: Use reasonable k; combine with filter for large datasets

        Notes
        -----
        - Requires the column to have embeddings (``ensure_vector_column`` + ``vectorize_rows``)
        - Similarity scores are normalized 0-1; higher is more similar
        - Performance depends on embedding index; large tables may have latency
        - Combine filter + search for best results: filter narrows, search ranks
        """

    @abstractmethod
    def reduce(
        self,
        context: str,
        *,
        metric: str,
        column: Optional[str] = None,
        filter: Optional[str] = None,
        group_by: Optional[Union[str, List[str]]] = None,
    ) -> Any:
        """
        Compute aggregate metrics over rows.

        Use this tool for **aggregations** like counts, sums, averages, min/max.
        Much more efficient than fetching all rows and computing in Python.

        Parameters
        ----------
        context : str
            Target context path. Accepts relative, absolute owned, or foreign paths.
            See ``filter()`` for full context resolution documentation.

        metric : str
            Aggregation function to apply. Supported values:

            - ``"count"``: Number of rows (column optional)
            - ``"count_distinct"``: Number of unique values in column
            - ``"sum"``: Sum of numeric column values
            - ``"avg"`` or ``"mean"``: Average of numeric column values
            - ``"min"``: Minimum value in column
            - ``"max"``: Maximum value in column

        column : str | None, default ``None``
            Column to aggregate. Required for sum/avg/min/max.
            For ``count``, if provided, counts non-null values in that column.

        filter : str | None, default ``None``
            Filter expression to apply BEFORE aggregation.
            Same syntax as ``filter()`` method.

        group_by : str | list[str] | None, default ``None``
            Column(s) to group by before aggregation. When provided, returns
            one result per unique group.

        Returns
        -------
        Any
            - **Scalar** if no ``group_by``: int for count, float for avg/sum, etc.
            - **List of dicts** if ``group_by``: Each dict has group columns + metric result.

        Usage Examples
        --------------
        # Count all rows
        total = dm.reduce("MH/arrears", metric="count")

        # Sum with filter
        total_overdue = dm.reduce(
            "MH/arrears",
            metric="sum",
            column="amount",
            filter="status == 'overdue'"
        )

        # Average
        avg_amount = dm.reduce("MH/arrears", metric="avg", column="amount")

        # Group by aggregation
        by_region = dm.reduce(
            "Data/sales",
            metric="sum",
            column="revenue",
            group_by="region"
        )
        # Returns: [{"region": "East", "sum": 150000}, {"region": "West", "sum": 200000}]

        # Multiple group-by columns
        by_region_quarter = dm.reduce(
            "Data/sales",
            metric="sum",
            column="revenue",
            group_by=["region", "quarter"]
        )

        # Count distinct values
        unique_categories = dm.reduce(
            "Data/products",
            metric="count_distinct",
            column="category"
        )

        Anti-patterns
        -------------
        - WRONG: ``dm.filter(ctx, limit=10000)`` then ``len(results)`` for counting
          CORRECT: ``dm.reduce(ctx, metric="count")``

        - WRONG: Fetching all rows to compute average in Python
          CORRECT: ``dm.reduce(ctx, metric="avg", column="value")``

        - WRONG: Using metric="count" with column for row count
          CORRECT: Use column=None for total row count

        Notes
        -----
        - Aggregations are computed server-side for efficiency
        - For ``group_by``, results are not guaranteed sorted; sort in Python if needed
        - ``count_distinct`` may be approximate for very large datasets
        """

    # ──────────────────────────────────────────────────────────────────────────
    # Join Operations
    # ──────────────────────────────────────────────────────────────────────────

    @abstractmethod
    def filter_join(
        self,
        *,
        left_context: str,
        right_context: str,
        join_column: str,
        filter: Optional[str] = None,
        columns: Optional[List[str]] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Join two tables and filter the result.

        Use this to combine data from two related tables based on a shared column.
        This is an inner join: only rows with matching values in both tables are returned.

        Parameters
        ----------
        left_context : str
            Left table context path. Columns from this table are included first.

        right_context : str
            Right table context path. Columns from this table are appended.

        join_column : str
            Column name to join on. Must exist in BOTH tables with compatible types.

        filter : str | None, default ``None``
            Filter expression applied AFTER the join. Can reference columns
            from either table. Same syntax as ``filter()`` method.

        columns : list[str] | None, default ``None``
            Columns to return. Can include columns from either table.
            When ``None``, all columns from both tables are returned.

        limit : int | None, default ``None``
            Maximum rows to return. When ``None``, all matching rows returned.

        Returns
        -------
        list[dict]
            Joined rows containing columns from both tables.

        Usage Examples
        --------------
        # Join orders with customers
        results = dm.filter_join(
            left_context="Data/orders",
            right_context="Data/customers",
            join_column="customer_id",
            filter="amount > 100"
        )

        # Cross-namespace join (file table + data table)
        results = dm.filter_join(
            left_context="Files/Local/120/Tables/Orders",
            right_context="Data/MH/properties",
            join_column="property_id"
        )

        # Select specific columns
        results = dm.filter_join(
            left_context="Data/orders",
            right_context="Data/products",
            join_column="product_id",
            columns=["order_id", "product_name", "quantity", "price"]
        )

        Anti-patterns
        -------------
        - WRONG: Joining on columns with different types
          CORRECT: Ensure join columns have compatible types (e.g., both int)

        - WRONG: Joining very large tables without filter
          CORRECT: Add filter to limit result size

        Notes
        -----
        - This is an INNER join; rows without matches in both tables are excluded
        - Column name conflicts: right table columns may be prefixed
        - For outer joins or complex logic, fetch separately and join in Python
        """

    @abstractmethod
    def search_join(
        self,
        *,
        left_context: str,
        right_context: str,
        join_column: str,
        query: str,
        k: int = 10,
        filter: Optional[str] = None,
        vector_column: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Join two tables and perform semantic search on the result.

        Combines a join operation with semantic search. Useful when you need
        to find semantically similar rows across related data.

        Parameters
        ----------
        left_context : str
            Left table context path.

        right_context : str
            Right table context path.

        join_column : str
            Column to join on (must exist in both tables).

        query : str
            Natural language query for semantic search.

        k : int, default ``10``
            Number of results to return.

        filter : str | None, default ``None``
            Filter expression applied after join, before search.

        vector_column : str | None, default ``None``
            Embedding column to search (must exist in one of the tables).

        Returns
        -------
        list[dict]
            Joined rows ranked by semantic similarity.

        Usage Examples
        --------------
        # Search products with order info
        results = dm.search_join(
            left_context="Data/orders",
            right_context="Data/products",
            join_column="product_id",
            query="eco-friendly sustainable materials",
            k=5
        )

        Notes
        -----
        - The embedding column must exist in one of the joined tables
        - Results are ranked by similarity after the join
        """

    @abstractmethod
    def filter_multi_join(
        self,
        *,
        contexts: List[str],
        join_columns: List[str],
        filter: Optional[str] = None,
        columns: Optional[List[str]] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Chain multiple joins across several tables and filter the result.

        Use this for complex queries spanning 3+ related tables.

        Parameters
        ----------
        contexts : list[str]
            Ordered list of context paths to join. First is leftmost table.
            Example: ``["orders", "products", "categories"]``

        join_columns : list[str]
            List of columns to join on (one per join, so ``len = len(contexts) - 1``).
            Example: ``["product_id", "category_id"]`` for the above contexts.

        filter : str | None, default ``None``
            Filter expression applied after all joins.

        columns : list[str] | None, default ``None``
            Columns to return from any of the joined tables.

        limit : int | None, default ``None``
            Maximum rows to return.

        Returns
        -------
        list[dict]
            Joined and filtered rows.

        Usage Examples
        --------------
        # Join three tables: orders → products → categories
        results = dm.filter_multi_join(
            contexts=[
                "Data/orders",
                "Data/products",
                "Data/categories"
            ],
            join_columns=["product_id", "category_id"],
            filter="category_name == 'Electronics'"
        )

        Notes
        -----
        For contexts [A, B, C] with join_columns [j1, j2], this performs:
        A JOIN B ON j1 JOIN C ON j2

        The joins are performed left-to-right in sequence.
        """

    @abstractmethod
    def search_multi_join(
        self,
        *,
        contexts: List[str],
        join_columns: List[str],
        query: str,
        k: int = 10,
        filter: Optional[str] = None,
        vector_column: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Chain multiple joins and perform semantic search.

        Combines multi-table join with semantic ranking.

        Parameters
        ----------
        contexts : list[str]
            Ordered list of context paths to join.

        join_columns : list[str]
            Columns to join on (one per join).

        query : str
            Semantic search query.

        k : int, default ``10``
            Number of results to return.

        filter : str | None, default ``None``
            Filter expression applied after joins, before search.

        vector_column : str | None, default ``None``
            Embedding column for search (must exist in one of the contexts).

        Returns
        -------
        list[dict]
            Joined rows ranked by semantic similarity.
        """

    # ──────────────────────────────────────────────────────────────────────────
    # Mutation Operations
    # ──────────────────────────────────────────────────────────────────────────

    @abstractmethod
    def insert_rows(
        self,
        context: str,
        rows: List[Dict[str, Any]],
        *,
        dedupe_key: Optional[str] = None,
    ) -> int:
        """
        Insert rows into a table.

        Use this to add new data to an existing table. Supports bulk inserts
        and deduplication based on a key column.

        Parameters
        ----------
        context : str
            Target context path. Accepts relative, absolute owned, or foreign paths.

        rows : list[dict]
            Row dictionaries to insert. Each dict maps column names to values.
            All rows should have consistent structure.

            Example: ``[{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]``

        dedupe_key : str | None, default ``None``
            If provided, rows with duplicate key values are updated instead of
            creating duplicates. This enables "upsert" behavior.

            Example: ``dedupe_key="id"`` - if row with same id exists, update it.

        Returns
        -------
        int
            Number of rows inserted (or updated if dedupe_key caused updates).

        Usage Examples
        --------------
        # Simple insert
        dm.insert_rows("Data/customers", [
            {"id": 1, "name": "Alice", "email": "alice@example.com"},
            {"id": 2, "name": "Bob", "email": "bob@example.com"},
        ])

        # Upsert with dedupe_key
        dm.insert_rows(
            "Data/products",
            [{"sku": "ABC123", "price": 29.99, "stock": 100}],
            dedupe_key="sku"  # Updates if SKU exists
        )

        # Batch insert from API response
        api_data = fetch_from_api()
        transformed = [transform_row(r) for r in api_data]
        count = dm.insert_rows("Data/MH/arrears", transformed)
        print(f"Inserted {count} rows")

        Anti-patterns
        -------------
        - WRONG: Inserting one row at a time in a loop
          CORRECT: Batch rows into a single insert_rows call

        - WRONG: Inserting without dedupe_key when updates are expected
          CORRECT: Use dedupe_key for idempotent inserts

        Notes
        -----
        - Empty rows list returns 0 without error
        - If table doesn't exist, it's created with inferred schema
        - With dedupe_key, existing rows are replaced (delete + insert)
        """

    @abstractmethod
    def update_rows(
        self,
        context: str,
        updates: Dict[str, Any],
        *,
        filter: str,
    ) -> int:
        """
        Update rows matching a filter.

        Use this to modify existing rows based on a condition. All matching
        rows receive the same updates.

        Parameters
        ----------
        context : str
            Target context path.

        updates : dict
            Column values to set. Only specified columns are updated;
            other columns remain unchanged.

            Example: ``{"status": "processed", "processed_at": "2024-01-15"}``

        filter : str
            Filter expression to match rows for update. **Required** to prevent
            accidental mass updates. Same syntax as ``filter()`` method.

        Returns
        -------
        int
            Number of rows updated.

        Usage Examples
        --------------
        # Update status for specific records
        dm.update_rows(
            "Data/orders",
            updates={"status": "shipped", "shipped_at": "2024-01-15"},
            filter="order_id in (101, 102, 103)"
        )

        # Mark all overdue items
        dm.update_rows(
            "Data/invoices",
            updates={"is_overdue": True},
            filter="due_date < '2024-01-01' and status == 'pending'"
        )

        Anti-patterns
        -------------
        - WRONG: Trying to use empty filter to update all rows
          CORRECT: Filter is required; use very broad filter if intentional

        - WRONG: Updating primary/unique key columns
          CORRECT: Delete and re-insert if key changes are needed

        Notes
        -----
        - Filter is required as a safety measure
        - Updates are applied atomically per row
        - For complex updates, consider delete + insert pattern
        """

    @abstractmethod
    def delete_rows(
        self,
        context: str,
        *,
        filter: str,
        dangerous_ok: bool = False,
    ) -> int:
        """
        Delete rows matching a filter.

        **WARNING**: This is a destructive operation. Deleted rows cannot be recovered.

        Parameters
        ----------
        context : str
            Target context path.

        filter : str
            Filter expression to match rows for deletion. **Required** to prevent
            accidental mass deletion. Same syntax as ``filter()`` method.

        dangerous_ok : bool, default ``False``
            Safety flag that MUST be set to ``True`` to confirm deletion.

        Returns
        -------
        int
            Number of rows deleted.

        Raises
        ------
        ValueError
            If ``dangerous_ok`` is ``False``.

        Usage Examples
        --------------
        # Delete specific records
        dm.delete_rows(
            "Data/temp_imports",
            filter="created_at < '2024-01-01'",
            dangerous_ok=True
        )

        # Clean up test data
        dm.delete_rows(
            "Data/orders",
            filter="is_test == True",
            dangerous_ok=True
        )

        Anti-patterns
        -------------
        - WRONG: Using dangerous_ok=True without verifying filter
          CORRECT: Test filter with dm.filter() first to see what would be deleted

        - WRONG: Deleting from FileManager-owned contexts
          CORRECT: Use FileManager methods for file-derived data

        Notes
        -----
        - Filter is required as a safety measure
        - Deletion is immediate and permanent
        - Associated references from other tables are NOT cleaned up
        """

    # ──────────────────────────────────────────────────────────────────────────
    # Embedding Operations
    # ──────────────────────────────────────────────────────────────────────────

    @abstractmethod
    def ensure_vector_column(
        self,
        context: str,
        *,
        source_column: str,
        target_column: Optional[str] = None,
    ) -> str:
        """
        Ensure an embedding column exists for a source column.

        Creates the vector column structure if it doesn't exist. This must be
        called before ``vectorize_rows`` to prepare the table for embeddings.

        Parameters
        ----------
        context : str
            Target context path.

        source_column : str
            Column containing text to embed. Must be a string column.

        target_column : str | None, default ``None``
            Name for the embedding column. If ``None``, defaults to
            ``"_{source_column}_emb"`` following the convention.

        Returns
        -------
        str
            The name of the embedding column (either provided or generated).

        Usage Examples
        --------------
        # Create embedding column with default name
        emb_col = dm.ensure_vector_column("Data/docs", source_column="content")
        # Returns: "_content_emb"

        # Create with custom name
        emb_col = dm.ensure_vector_column(
            "Data/products",
            source_column="description",
            target_column="_desc_vectors"
        )

        Notes
        -----
        - Idempotent: safe to call multiple times
        - Does NOT generate embeddings; use vectorize_rows for that
        """

    @abstractmethod
    def vectorize_rows(
        self,
        context: str,
        *,
        source_column: str,
        target_column: Optional[str] = None,
        row_ids: Optional[List[int]] = None,
        batch_size: int = 100,
    ) -> int:
        """
        Generate embeddings for rows.

        Creates vector embeddings for the specified text column. Call
        ``ensure_vector_column`` first to set up the column structure.

        Parameters
        ----------
        context : str
            Target context path.

        source_column : str
            Column containing text to embed.

        target_column : str | None, default ``None``
            Embedding column name. If ``None``, defaults to ``"_{source_column}_emb"``.

        row_ids : list[int] | None, default ``None``
            Specific row IDs to embed. If ``None``, embeds all rows that don't
            already have embeddings.

        batch_size : int, default ``100``
            Number of rows to embed per batch. Larger batches are faster but
            use more memory.

        Returns
        -------
        int
            Number of rows that were embedded.

        Usage Examples
        --------------
        # Embed all unembedded rows
        count = dm.vectorize_rows("Data/documents", source_column="content")
        print(f"Embedded {count} rows")

        # Embed specific rows
        dm.vectorize_rows(
            "Data/products",
            source_column="description",
            row_ids=[1, 2, 3, 4, 5]
        )

        Notes
        -----
        - Requires ensure_vector_column to be called first
        - Skips rows that already have embeddings
        - Uses the default embedding model configured for the project
        """

    # ──────────────────────────────────────────────────────────────────────────
    # Visualization
    # ──────────────────────────────────────────────────────────────────────────

    @abstractmethod
    def plot(
        self,
        context: str,
        *,
        plot_type: str,
        x: str,
        y: Optional[str] = None,
        group_by: Optional[str] = None,
        aggregate: Optional[str] = None,
        filter: Optional[str] = None,
        title: Optional[str] = None,
        scale_x: Optional[str] = None,
        scale_y: Optional[str] = None,
        bin_count: Optional[int] = None,
        show_regression: Optional[bool] = None,
    ) -> "PlotResult":
        """
        Generate a plot visualization from table data.

        Creates visual charts from table data. Supports scatter plots, bar charts,
        histograms, and line charts.

        Parameters
        ----------
        context : str
            Target context path containing the data to visualize.

        plot_type : str
            Chart type. One of:

            - ``"scatter"``: Scatter plot for correlations (requires x and y)
            - ``"bar"``: Bar chart for comparing categories (requires x and y)
            - ``"histogram"``: Distribution of single variable (requires x only)
            - ``"line"``: Trend/time series (requires x and y)

        x : str
            X-axis column name.

        y : str | None, default ``None``
            Y-axis column name. Required for scatter, bar, line plots.
            Not used for histogram.

        group_by : str | None, default ``None``
            Column to group/color data points by. Creates multiple series.

        aggregate : str | None, default ``None``
            Aggregation for bar charts: ``"sum"``, ``"count"``, ``"avg"``,
            ``"min"``, ``"max"``. When set, bars show aggregate values.

        filter : str | None, default ``None``
            Filter expression to subset data before plotting.

        title : str | None, default ``None``
            Chart title displayed above the plot.

        scale_x : str | None, default ``None``
            X-axis scale: ``"linear"`` or ``"log"``.

        scale_y : str | None, default ``None``
            Y-axis scale: ``"linear"`` or ``"log"``.

        bin_count : int | None, default ``None``
            Number of bins for histogram plots.

        show_regression : bool | None, default ``None``
            Show regression line on scatter plots.

        Returns
        -------
        PlotResult
            Result with URL, token, or error information.
            Check ``result.succeeded`` to verify generation worked.
            Access ``result.url`` for the plot image URL.

        Usage Examples
        --------------
        # Bar chart: revenue by region
        result = dm.plot(
            "Data/sales",
            plot_type="bar",
            x="region",
            y="revenue",
            aggregate="sum",
            title="Revenue by Region"
        )
        if result.succeeded:
            print(f"Plot URL: {result.url}")

        # Scatter plot with regression
        result = dm.plot(
            "Data/performance",
            plot_type="scatter",
            x="experience_years",
            y="salary",
            show_regression=True
        )

        # Histogram of price distribution
        result = dm.plot(
            "Data/products",
            plot_type="histogram",
            x="price",
            bin_count=20,
            title="Price Distribution"
        )

        # Line chart with groups
        result = dm.plot(
            "Data/metrics",
            plot_type="line",
            x="date",
            y="value",
            group_by="metric_name"
        )

        Anti-patterns
        -------------
        - WRONG: Plotting very large datasets without filter
          CORRECT: Filter to a reasonable sample for visualization

        - WRONG: Using histogram with both x and y
          CORRECT: Histogram only uses x (single variable distribution)

        Notes
        -----
        - Plot URLs expire after a period (check expires_in_hours)
        - Large datasets are sampled for plotting performance
        - Error details available in result.error and result.traceback_str
        """

    @abstractmethod
    def plot_batch(
        self,
        contexts: List[str],
        *,
        plot_type: str,
        x: str,
        y: Optional[str] = None,
        group_by: Optional[str] = None,
        aggregate: Optional[str] = None,
        filter: Optional[str] = None,
        title: Optional[str] = None,
        **kwargs: Any,
    ) -> List["PlotResult"]:
        """
        Generate the same plot across multiple tables.

        Convenience method for creating comparable visualizations across
        multiple data sources with identical configuration.

        Parameters
        ----------
        contexts : list[str]
            List of context paths to generate plots for.

        plot_type, x, y, group_by, aggregate, filter, title, **kwargs
            Same parameters as ``plot()``. Applied to all contexts.

        Returns
        -------
        list[PlotResult]
            One result per context, in the same order as inputs.
            Check each result's ``succeeded`` property.

        Usage Examples
        --------------
        # Compare quarterly data
        results = dm.plot_batch(
            contexts=[
                "Data/sales/Q1",
                "Data/sales/Q2",
                "Data/sales/Q3",
                "Data/sales/Q4"
            ],
            plot_type="bar",
            x="category",
            y="revenue",
            aggregate="sum"
        )

        for ctx, result in zip(contexts, results):
            if result.succeeded:
                print(f"{ctx}: {result.url}")
        """
