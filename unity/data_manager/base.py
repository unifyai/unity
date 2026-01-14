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
        references: Optional[Dict[str, str]] = None,
        k: int = 10,
        filter: Optional[str] = None,
        columns: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Semantic search over embedded columns using reference text.

        Use this tool when searching by **meaning, topics, or concepts** in text fields.
        For **exact matches** on structured fields, use ``filter()`` instead.

        This performs vector similarity search using embeddings. The table must
        have embedding columns (``_<name>_emb``) for the columns being searched.
        Use ``describe_table()`` to check which columns have embeddings.

        Parameters
        ----------
        context : str
            Target context path. Accepts relative, absolute owned, or foreign paths.
            See ``filter()`` for full context resolution documentation.

        references : dict[str, str] | None, default ``None``
            Mapping of source_column/expression → reference_text for semantic matching.

            The keys specify WHICH columns to search (must have embeddings).
            The values are the reference text to match against.

            Examples:
            - ``{"text": "budget allocation"}`` — search the ``text`` column
            - ``{"description": "eco-friendly"}`` — search the ``description`` column
            - ``{"content": "Q4 priorities", "summary": "budget planning"}`` — multi-column search

            For multi-column search, results are ranked by average similarity across
            all specified columns.

            When ``None`` or empty, returns rows without semantic ranking (equivalent
            to ``filter()`` with backfill).

        k : int, default ``10``
            Number of results to return. Results are ranked by similarity score.

        filter : str | None, default ``None``
            Additional row-level predicate to narrow results BEFORE semantic ranking.
            Same syntax as ``filter()`` method.

            Example: ``filter="year == 2024"`` to only search 2024 documents.

        columns : list[str] | None, default ``None``
            Columns to return in results. When ``None``, all columns returned.

        Returns
        -------
        list[dict]
            List of row dictionaries ranked by semantic similarity.
            Each dict may include a ``_similarity`` score (0-1, higher is better).
            Results are sorted by similarity descending (best match first).

        Usage Examples
        --------------
        # Basic semantic search on single column
        results = dm.search(
            "Data/documents",
            references={"text": "budget planning for next fiscal year"},
            k=5
        )

        # Search with filter to narrow scope
        results = dm.search(
            "Data/support_tickets",
            references={"description": "delivery problems"},
            k=10,
            filter="status == 'open'"
        )

        # Multi-column semantic search
        results = dm.search(
            "Data/products",
            references={
                "name": "eco-friendly",
                "description": "sustainable materials"
            },
            k=10
        )

        # Cross-namespace search (file content)
        results = dm.search(
            "Files/Local/120/Content",
            references={"text": "executive summary recommendations"},
            k=5
        )

        # Search with derived expression (combine columns)
        results = dm.search(
            "Data/articles",
            references={"str({title}) + ' ' + str({abstract})": "machine learning"},
            k=10
        )

        Anti-patterns
        -------------
        - WRONG: Using search for exact matches like ``references={"status": "active"}``
          CORRECT: Use ``filter(filter="status == 'active'")`` instead

        - WRONG: Very short references like ``references={"text": "budget"}``
          CORRECT: Use descriptive text: ``references={"text": "Q4 marketing budget allocation"}``

        - WRONG: Assuming embeddings exist without checking
          CORRECT: Use ``describe_table()`` to verify embedding columns exist

        - WRONG: Large k values (k=1000) for exhaustive retrieval
          CORRECT: Use reasonable k; combine with filter for large datasets

        Notes
        -----
        - Requires columns to have embeddings (``ensure_vector_column`` + ``vectorize_rows``)
        - For multi-column search, similarity is averaged across all specified columns
        - Combine filter + search for best results: filter narrows, search ranks
        - The ``references`` keys can be plain column names or derived expressions
        """

    @abstractmethod
    def reduce(
        self,
        context: str,
        *,
        metric: str,
        columns: Union[str, List[str]],
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

            - ``"count"``: Number of rows (counts non-null values in column)
            - ``"count_distinct"``: Number of unique values in column
            - ``"sum"``: Sum of numeric column values
            - ``"mean"`` or ``"avg"``: Average of numeric column values
            - ``"var"``: Variance of numeric column values
            - ``"std"``: Standard deviation of numeric column values
            - ``"min"``: Minimum value in column
            - ``"max"``: Maximum value in column
            - ``"median"``: Median value in column
            - ``"mode"``: Most frequent value in column

        columns : str | list[str]
            Column(s) to aggregate. **Required parameter**.

            - **Single column** (str): Returns a single aggregate value.
              Example: ``columns="amount"``

            - **Multiple columns** (list[str]): Returns dict mapping column → aggregate.
              Example: ``columns=["amount", "quantity"]``

            For ``count``, use any column (e.g., ``"id"``).
            For ``sum``/``avg``/``min``/``max``, must be numeric columns.

        filter : str | None, default ``None``
            Filter expression to apply BEFORE aggregation.
            Same syntax as ``filter()`` method.

        group_by : str | list[str] | None, default ``None``
            Column(s) to group by before aggregation. When provided, returns
            one result per unique group.

        Returns
        -------
        Any
            Depends on ``columns`` and ``group_by``:

            - **Single column, no group_by**: Scalar (int for count, float for avg/sum)
            - **Multiple columns, no group_by**: Dict mapping column_name → aggregate_value
            - **Single column, with group_by**: List of dicts with group columns + metric
            - **Multiple columns, with group_by**: List of dicts with group columns + all metrics

        Raises
        ------
        ValueError
            If metric is not supported.
        ValueError
            If columns is empty.

        Usage Examples
        --------------
        # Count all rows (single column)
        total = dm.reduce("MH/arrears", metric="count", columns="id")

        # Sum single column with filter
        total_overdue = dm.reduce(
            "MH/arrears",
            metric="sum",
            columns="amount",
            filter="status == 'overdue'"
        )

        # Sum multiple columns at once
        totals = dm.reduce(
            "MH/arrears",
            metric="sum",
            columns=["amount", "fees", "penalties"]
        )
        # Returns: {"amount": 150000, "fees": 5000, "penalties": 2000}

        # Average multiple columns
        averages = dm.reduce(
            "Data/orders",
            metric="mean",
            columns=["quantity", "price", "discount"]
        )
        # Returns: {"quantity": 5.2, "price": 99.50, "discount": 0.15}

        # Group by aggregation (single column)
        by_region = dm.reduce(
            "Data/sales",
            metric="sum",
            columns="revenue",
            group_by="region"
        )
        # Returns: [{"region": "East", "sum": 150000}, {"region": "West", "sum": 200000}]

        # Group by with multiple columns
        by_region = dm.reduce(
            "Data/sales",
            metric="sum",
            columns=["revenue", "units"],
            group_by="region"
        )
        # Returns: [{"region": "East", "revenue": 150000, "units": 500}, ...]

        # Multiple group-by columns
        by_region_quarter = dm.reduce(
            "Data/sales",
            metric="sum",
            columns="revenue",
            group_by=["region", "quarter"]
        )

        # Count distinct values
        unique_categories = dm.reduce(
            "Data/products",
            metric="count_distinct",
            columns="category"
        )

        Anti-patterns
        -------------
        - WRONG: ``dm.filter(ctx, limit=10000)`` then ``len(results)`` for counting
          CORRECT: ``dm.reduce(ctx, metric="count", columns="id")``

        - WRONG: Fetching all rows to compute average in Python
          CORRECT: ``dm.reduce(ctx, metric="mean", columns="value")``

        - WRONG: Calling reduce multiple times for different columns
          CORRECT: ``dm.reduce(ctx, metric="sum", columns=["a", "b", "c"])``

        Notes
        -----
        - Aggregations are computed server-side for efficiency
        - For ``group_by``, results are not guaranteed sorted; sort in Python if needed
        - ``count_distinct`` may be approximate for very large datasets
        - Multiple columns are aggregated in a single query for efficiency
        """

    # ──────────────────────────────────────────────────────────────────────────
    # Join Operations
    # ──────────────────────────────────────────────────────────────────────────

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
    ) -> List[Dict[str, Any]]:
        """
        Join two tables and filter the result.

        Use this to combine data from two related tables with full control over
        join expression, column selection, pre/post filters, and join mode.

        This follows the same pattern as KnowledgeManager's join implementation.

        Parameters
        ----------
        tables : str | list[str]
            Exactly TWO table names/context paths to join.
            Example: ``["Orders", "Customers"]`` or ``["Data/orders", "Data/customers"]``

        join_expr : str
            Join condition expression using table names as prefixes.
            Example: ``"Orders.customer_id == Customers.id"``

            The table names in the expression are automatically rewritten to
            fully-qualified context paths.

        select : dict[str, str]
            Mapping of source columns to output column names.
            Keys use table names as prefixes; values are the output alias.

            Example::

                {
                    "Orders.id": "order_id",
                    "Orders.amount": "amount",
                    "Customers.name": "customer_name",
                    "Customers.email": "email"
                }

        mode : str, default ``"inner"``
            Join mode: ``"inner"``, ``"left"``, ``"right"``, or ``"outer"``.

        left_where : str | None, default ``None``
            Filter expression applied to LEFT table BEFORE the join.
            Uses left table's column names without prefix.
            Example: ``"status == 'active'"``

        right_where : str | None, default ``None``
            Filter expression applied to RIGHT table BEFORE the join.
            Uses right table's column names without prefix.
            Example: ``"created_at >= '2024-01-01'"``

        result_where : str | None, default ``None``
            Filter expression applied AFTER the join on the result.
            Must use column names from ``select`` values (the output aliases).
            Example: ``"amount > 100"``

        result_limit : int, default ``100``
            Maximum rows to return. Must be <= 1000.

        result_offset : int, default ``0``
            Pagination offset for results.

        Returns
        -------
        list[dict]
            Joined rows with columns as specified in ``select``.

        Usage Examples
        --------------
        # Basic inner join
        results = dm.filter_join(
            tables=["Data/orders", "Data/customers"],
            join_expr="Data/orders.customer_id == Data/customers.id",
            select={
                "Data/orders.id": "order_id",
                "Data/orders.amount": "amount",
                "Data/customers.name": "customer_name"
            },
            result_where="amount > 100"
        )

        # Left join with pre-filters
        results = dm.filter_join(
            tables=["Data/orders", "Data/products"],
            join_expr="Data/orders.product_id == Data/products.id",
            select={
                "Data/orders.id": "order_id",
                "Data/products.name": "product_name",
                "Data/products.price": "price"
            },
            mode="left",
            left_where="status == 'completed'",
            right_where="category == 'Electronics'",
            result_limit=50
        )

        # Cross-namespace join (file table + data table)
        results = dm.filter_join(
            tables=["Files/Local/120/Tables/Sheet1", "Data/MH/properties"],
            join_expr="Files/Local/120/Tables/Sheet1.property_ref == Data/MH/properties.id",
            select={
                "Files/Local/120/Tables/Sheet1.amount": "arrears_amount",
                "Data/MH/properties.address": "property_address"
            }
        )

        Anti-patterns
        -------------
        - WRONG: Using output aliases in ``join_expr``
          CORRECT: Use source table.column references in ``join_expr``

        - WRONG: Using source references in ``result_where``
          CORRECT: Use output column names (values from ``select``) in ``result_where``

        - WRONG: ``result_where`` references columns not in ``select``
          CORRECT: Add all referenced columns to ``select``

        Notes
        -----
        - Pre-filters (``left_where``, ``right_where``) are applied before joining,
          which can significantly improve performance on large tables.
        - The temporary join context is created, queried, and cleaned up automatically.
        - Column references in ``join_expr`` and ``select`` keys use table/context prefixes.
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
        Join two tables and perform semantic search on the result.

        Combines a join operation with semantic search. The join is performed first,
        then results are ranked by semantic similarity to the reference text.

        Parameters
        ----------
        tables : str | list[str]
            Exactly TWO table names/context paths to join.

        join_expr : str
            Join condition expression using table names as prefixes.
            Example: ``"Orders.product_id == Products.id"``

        select : dict[str, str]
            Mapping of source columns to output column names.
            Example: ``{"Products.description": "description", "Orders.id": "order_id"}``

        mode : str, default ``"inner"``
            Join mode: ``"inner"``, ``"left"``, ``"right"``, or ``"outer"``.

        left_where : str | None, default ``None``
            Filter expression applied to LEFT table BEFORE the join.

        right_where : str | None, default ``None``
            Filter expression applied to RIGHT table BEFORE the join.

        references : dict[str, str] | None, default ``None``
            Mapping of column → reference_text for semantic similarity.
            Keys must be columns that exist in the joined result (output aliases from ``select``).

            Example: ``{"description": "eco-friendly sustainable materials"}``

        k : int, default ``10``
            Number of top results to return (1 to 1000).

        filter : str | None, default ``None``
            Filter expression applied after join, before semantic ranking.
            Uses output column names (values from ``select``).

        Returns
        -------
        list[dict]
            Top-k joined rows ranked by semantic similarity.

        Usage Examples
        --------------
        # Search products with order info
        results = dm.search_join(
            tables=["Data/orders", "Data/products"],
            join_expr="Data/orders.product_id == Data/products.id",
            select={
                "Data/orders.id": "order_id",
                "Data/orders.quantity": "quantity",
                "Data/products.name": "product_name",
                "Data/products.description": "description"
            },
            references={"description": "eco-friendly sustainable materials"},
            k=10
        )

        # With pre-filters and post-filter
        results = dm.search_join(
            tables=["Data/support_tickets", "Data/customers"],
            join_expr="Data/support_tickets.customer_id == Data/customers.id",
            select={
                "Data/support_tickets.id": "ticket_id",
                "Data/support_tickets.description": "issue_description",
                "Data/customers.name": "customer_name",
                "Data/customers.tier": "customer_tier"
            },
            left_where="status == 'open'",
            references={"issue_description": "payment processing errors"},
            filter="customer_tier == 'enterprise'",
            k=5
        )

        Notes
        -----
        - The embedding column must exist in the joined result for semantic search
        - Results are ranked by similarity after the join and filter are applied
        """

    @abstractmethod
    def filter_multi_join(
        self,
        *,
        joins: List[Dict[str, Any]],
        result_where: Optional[str] = None,
        result_limit: int = 100,
        result_offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """
        Execute a sequence of joins and filter the final result.

        Use this for complex queries spanning 3+ tables. Each join step is
        performed in sequence, with the result of each step available to the next.

        Parameters
        ----------
        joins : list[dict]
            Ordered list of join steps. Each step is a dict with:

            - ``"tables"`` (list[str], required): Exactly TWO table names.
              Use ``"$prev"``, ``"__prev__"``, or ``"_"`` to reference the
              previous join result.
            - ``"join_expr"`` (str, required): Join condition expression.
            - ``"select"`` (dict[str, str], required): Column mapping.
            - ``"mode"`` (str, optional): Join mode (default: ``"inner"``).
            - ``"left_where"`` (str | None, optional): Pre-join filter on left.
            - ``"right_where"`` (str | None, optional): Pre-join filter on right.

        result_where : str | None, default ``None``
            Filter expression applied to the FINAL joined result.
            Must use column names from the final step's ``select`` values.

        result_limit : int, default ``100``
            Maximum rows to return. Must be <= 1000.

        result_offset : int, default ``0``
            Pagination offset for results.

        Returns
        -------
        list[dict]
            Joined and filtered rows from the final result.

        Usage Examples
        --------------
        # Three-table join: Orders → Products → Categories
        results = dm.filter_multi_join(
            joins=[
                {
                    "tables": ["Data/orders", "Data/products"],
                    "join_expr": "Data/orders.product_id == Data/products.id",
                    "select": {
                        "Data/orders.id": "order_id",
                        "Data/orders.quantity": "quantity",
                        "Data/products.name": "product_name",
                        "Data/products.category_id": "category_id"
                    }
                },
                {
                    "tables": ["$prev", "Data/categories"],
                    "join_expr": "$prev.category_id == Data/categories.id",
                    "select": {
                        "$prev.order_id": "order_id",
                        "$prev.quantity": "quantity",
                        "$prev.product_name": "product_name",
                        "Data/categories.name": "category_name"
                    }
                }
            ],
            result_where="category_name == 'Electronics'",
            result_limit=50
        )

        # Four-table join with mode variations
        results = dm.filter_multi_join(
            joins=[
                {
                    "tables": ["Data/employees", "Data/departments"],
                    "join_expr": "Data/employees.dept_id == Data/departments.id",
                    "select": {
                        "Data/employees.id": "emp_id",
                        "Data/employees.name": "emp_name",
                        "Data/departments.name": "dept_name",
                        "Data/departments.location_id": "location_id"
                    },
                    "mode": "left"
                },
                {
                    "tables": ["$prev", "Data/locations"],
                    "join_expr": "$prev.location_id == Data/locations.id",
                    "select": {
                        "$prev.emp_id": "emp_id",
                        "$prev.emp_name": "emp_name",
                        "$prev.dept_name": "dept_name",
                        "Data/locations.city": "city"
                    }
                }
            ],
            result_where="city == 'London'"
        )

        Notes
        -----
        - ``$prev`` references the result of the previous join step
        - Column names in ``$prev`` references use the output aliases from that step
        - Intermediate join results are stored in temporary contexts and cleaned up
        - ``result_where`` must only reference columns from the final ``select``
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
        Execute a sequence of joins and perform semantic search.

        Combines multi-table join with semantic ranking. The joins are performed
        first, then results are ranked by semantic similarity.

        Parameters
        ----------
        joins : list[dict]
            Ordered list of join steps. Same format as ``filter_multi_join``:

            - ``"tables"`` (list[str], required): Exactly TWO table names.
              Use ``"$prev"`` to reference previous join result.
            - ``"join_expr"`` (str, required): Join condition.
            - ``"select"`` (dict[str, str], required): Column mapping.
            - ``"mode"`` (str, optional): Join mode.
            - ``"left_where"``, ``"right_where"`` (str | None, optional): Pre-filters.

        references : dict[str, str] | None, default ``None``
            Mapping of column → reference_text for semantic similarity.
            Keys must be columns from the final step's ``select`` output.

        k : int, default ``10``
            Number of top results to return (1 to 1000).

        filter : str | None, default ``None``
            Filter expression applied after all joins, before semantic ranking.

        Returns
        -------
        list[dict]
            Top-k joined rows ranked by semantic similarity.

        Usage Examples
        --------------
        # Search across three tables
        results = dm.search_multi_join(
            joins=[
                {
                    "tables": ["Data/tickets", "Data/customers"],
                    "join_expr": "Data/tickets.customer_id == Data/customers.id",
                    "select": {
                        "Data/tickets.id": "ticket_id",
                        "Data/tickets.description": "issue",
                        "Data/customers.name": "customer",
                        "Data/customers.product_id": "product_id"
                    }
                },
                {
                    "tables": ["$prev", "Data/products"],
                    "join_expr": "$prev.product_id == Data/products.id",
                    "select": {
                        "$prev.ticket_id": "ticket_id",
                        "$prev.issue": "issue",
                        "$prev.customer": "customer",
                        "Data/products.name": "product_name"
                    }
                }
            ],
            references={"issue": "login authentication failures"},
            k=10
        )

        Notes
        -----
        - Embeddings must exist on the searched column in the final result
        - Results are ranked by similarity after all joins complete
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
