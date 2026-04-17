"""
Concrete DataManager implementation.

This module provides the concrete DataManager class that implements
BaseDataManager. It delegates to ops/ functions for implementation
and stays thin (orchestration only).

Docstrings are inherited from BaseDataManager via @functools.wraps.
"""

from __future__ import annotations

import functools
import logging
from typing import Any, Dict, List, Optional, Tuple, Union


from unity.data_manager.base import BaseDataManager
from unity.data_manager.types.table import TableDescription
from unity.data_manager.types.ingest import (
    IngestExecutionConfig,
    IngestResult,
    PostIngestConfig,
)
from unity.data_manager.ops.table_ops import (
    create_table_impl,
    describe_table_impl,
    get_columns_impl,
    get_table_impl,
    list_tables_impl,
    delete_table_impl,
    rename_table_impl,
    create_column_impl,
    delete_column_impl,
    rename_column_impl,
    create_derived_column_impl,
)
from unity.data_manager.ops.query_ops import (
    filter_impl,
    search_impl,
    reduce_impl,
)
from unity.data_manager.ops.mutation_ops import (
    insert_rows_impl,
    update_rows_impl,
    delete_rows_impl,
)
from unity.data_manager.ops.join_ops import (
    join_tables_impl,
    filter_join_impl,
    reduce_join_impl,
    search_join_impl,
    filter_multi_join_impl,
    search_multi_join_impl,
)
from unity.common.embed_utils import ensure_vector_column as _ensure_vector_column
from unity.common.join_utils import rewrite_join_paths
from unity.data_manager.ops.ingest_ops import run_ingest
from unity.common.context_registry import ContextRegistry, TableContext

logger = logging.getLogger(__name__)


# Known absolute prefixes that indicate a path should not be resolved
_ABSOLUTE_PREFIXES = (
    "Data/",
    "Files/",
    "FileRecords/",
    "Contacts",
    "Knowledge/",
    "Tasks",
    "Messages",
    "Exchanges",
)


class DataManager(BaseDataManager):
    """
    Canonical implementation of data operations for any Unify context.

    See BaseDataManager for full API documentation.
    """

    class Config:
        """Context registration for DataManager's owned namespace."""

        required_contexts = [
            TableContext(
                name="Data",
                description=(
                    "Root namespace for pipeline/API-derived datasets. "
                    "Sub-contexts are created dynamically as Data/<project>/<table>."
                ),
                fields=None,  # No fixed schema - tables created dynamically
                unique_keys=None,
                auto_counting=None,
            ),
        ]

    def __init__(self) -> None:
        """Initialize DataManager with context registration."""
        super().__init__()
        self.include_in_multi_assistant_table = True

        # Resolve owned base context via ContextRegistry
        # This gives us the fully-qualified path like "User/Assistant/Data"
        try:
            self._base_ctx = ContextRegistry.get_context(self, "Data")
        except Exception:
            # Fallback for tests or offline scenarios
            self._base_ctx = "Data"

        logger.debug("DataManager initialized with base context: %s", self._base_ctx)

    def _resolve_unique_keys_and_auto_counting(
        self,
        resolved: str,
        unique_keys: Optional[Dict[str, str]],
        auto_counting: Optional[Dict[str, Optional[str]]],
    ) -> Tuple[Optional[Dict[str, str]], Optional[Dict[str, Optional[str]]]]:
        """Apply DM-owned defaults for unique keys and auto-counting.

        When *resolved* lives under DataManager's own namespace and the
        caller hasn't supplied explicit values, defaults to a global
        auto-incrementing ``row_id`` column — matching the convention used
        by every other state manager.  For foreign contexts the caller's
        values pass through unchanged.
        """
        base = self._base_ctx or "Data"
        if resolved == base or resolved.startswith(base + "/"):
            if unique_keys is None:
                unique_keys = {"row_id": "int"}
            if auto_counting is None:
                auto_counting = {"row_id": None}
        return unique_keys, auto_counting

    def _resolve_context(self, context: str) -> str:
        """
        Resolve a context path, handling relative and absolute paths.

        Parameters
        ----------
        context : str
            Context path. Can be:
            - Relative: "projects/housing" → resolved to "{base_ctx}/projects/housing"
            - Short-form absolute: "Data/MH/arrears", "Contacts" → used as-is
            - Fully-qualified: "org123/42/Contacts", "org123/42/Data/foo" → as-is

        Returns
        -------
        str
            Fully resolved context path.
        """
        context = context.lstrip("/")
        if not context:
            raise ValueError("Empty context path")

        # Short-form absolute: starts with a known context root name
        if any(context.startswith(p) for p in _ABSOLUTE_PREFIXES):
            return context

        # Fully-qualified: shares the org/assistant scope with our base context.
        # _base_ctx = "org/42/Data" → scope = "org/42/" →
        # "org/42/Contacts" is recognised as already-qualified.
        if self._base_ctx and "/" in self._base_ctx:
            scope = self._base_ctx.rsplit("/", 1)[0] + "/"
            if context.startswith(scope):
                return context

        # Relative path: prepend base context
        return f"{self._base_ctx}/{context}" if self._base_ctx else context

    # ──────────────────────────────────────────────────────────────────────────
    # Table Management
    # ──────────────────────────────────────────────────────────────────────────

    @functools.wraps(BaseDataManager.create_table, updated=())
    def create_table(
        self,
        context: str,
        *,
        description: Optional[str] = None,
        fields: Optional[Dict[str, Any]] = None,
        unique_keys: Optional[Dict[str, str]] = None,
        auto_counting: Optional[Dict[str, Optional[str]]] = None,
    ) -> str:
        resolved = self._resolve_context(context)
        unique_keys, auto_counting = self._resolve_unique_keys_and_auto_counting(
            resolved,
            unique_keys,
            auto_counting,
        )
        return create_table_impl(
            resolved,
            description=description,
            fields=fields,
            unique_keys=unique_keys,
            auto_counting=auto_counting,
        )

    @functools.wraps(BaseDataManager.describe_table, updated=())
    def describe_table(self, context: str) -> TableDescription:
        resolved = self._resolve_context(context)
        return describe_table_impl(resolved)

    @functools.wraps(BaseDataManager.get_columns, updated=())
    def get_columns(self, table: str) -> Dict[str, Any]:
        resolved = self._resolve_context(table)
        return get_columns_impl(resolved)

    @functools.wraps(BaseDataManager.get_table, updated=())
    def get_table(self, context: str) -> Dict[str, Any]:
        resolved = self._resolve_context(context)
        return get_table_impl(resolved)

    @functools.wraps(BaseDataManager.list_tables, updated=())
    def list_tables(
        self,
        *,
        prefix: Optional[str] = None,
        include_column_info: bool = True,
    ) -> Union[List[str], Dict[str, Any]]:
        resolved_prefix = self._resolve_context(prefix) if prefix else None
        return list_tables_impl(
            prefix=resolved_prefix,
            include_column_info=include_column_info,
        )

    @functools.wraps(BaseDataManager.delete_table, updated=())
    def delete_table(
        self,
        context: str,
        *,
        dangerous_ok: bool = False,
    ) -> None:
        resolved = self._resolve_context(context)
        delete_table_impl(resolved, dangerous_ok=dangerous_ok)

    @functools.wraps(BaseDataManager.rename_table, updated=())
    def rename_table(
        self,
        old_context: str,
        new_context: str,
    ) -> Dict[str, str]:
        resolved_old = self._resolve_context(old_context)
        resolved_new = self._resolve_context(new_context)
        return rename_table_impl(resolved_old, resolved_new)

    # ──────────────────────────────────────────────────────────────────────────
    # Column Operations
    # ──────────────────────────────────────────────────────────────────────────

    @functools.wraps(BaseDataManager.create_column, updated=())
    def create_column(
        self,
        context: str,
        *,
        column_name: str,
        column_type: str,
        mutable: bool = True,
        backfill_logs: bool = False,
    ) -> Dict[str, str]:
        resolved = self._resolve_context(context)
        return create_column_impl(
            resolved,
            column_name=column_name,
            column_type=column_type,
            mutable=mutable,
            backfill_logs=backfill_logs,
        )

    @functools.wraps(BaseDataManager.delete_column, updated=())
    def delete_column(
        self,
        context: str,
        *,
        column_name: str,
    ) -> Dict[str, str]:
        resolved = self._resolve_context(context)
        return delete_column_impl(resolved, column_name=column_name)

    @functools.wraps(BaseDataManager.rename_column, updated=())
    def rename_column(
        self,
        context: str,
        *,
        old_name: str,
        new_name: str,
    ) -> Dict[str, str]:
        resolved = self._resolve_context(context)
        return rename_column_impl(resolved, old_name=old_name, new_name=new_name)

    @functools.wraps(BaseDataManager.create_derived_column, updated=())
    def create_derived_column(
        self,
        context: str,
        *,
        column_name: str,
        equation: str,
    ) -> Dict[str, str]:
        resolved = self._resolve_context(context)
        return create_derived_column_impl(
            resolved,
            column_name=column_name,
            equation=equation,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Query Operations
    # ──────────────────────────────────────────────────────────────────────────

    @functools.wraps(BaseDataManager.filter, updated=())
    def filter(
        self,
        context: str,
        *,
        filter: Optional[str] = None,
        columns: Optional[List[str]] = None,
        exclude_columns: Optional[List[str]] = None,
        limit: int = 100,
        offset: int = 0,
        order_by: Optional[str] = None,
        descending: bool = False,
        return_ids_only: bool = False,
    ) -> Union[List[Dict[str, Any]], List[int]]:
        resolved = self._resolve_context(context)
        return filter_impl(
            resolved,
            filter=filter,
            columns=columns,
            exclude_columns=exclude_columns,
            limit=limit,
            offset=offset,
            order_by=order_by,
            descending=descending,
            return_ids_only=return_ids_only,
        )

    @functools.wraps(BaseDataManager.search, updated=())
    def search(
        self,
        context: str,
        *,
        references: Optional[Dict[str, str]] = None,
        k: int = 10,
        filter: Optional[str] = None,
        columns: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        resolved = self._resolve_context(context)
        return search_impl(
            resolved,
            references=references,
            k=k,
            filter=filter,
            columns=columns,
        )

    @functools.wraps(BaseDataManager.reduce, updated=())
    def reduce(
        self,
        context: str,
        *,
        metric: str,
        columns: Union[str, List[str]],
        filter: Optional[str] = None,
        group_by: Optional[Union[str, List[str]]] = None,
    ) -> Any:
        resolved = self._resolve_context(context)
        return reduce_impl(
            resolved,
            metric=metric,
            columns=columns,
            filter=filter,
            group_by=group_by,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Join Operations
    # ──────────────────────────────────────────────────────────────────────────

    @functools.wraps(BaseDataManager.join_tables, updated=())
    def join_tables(
        self,
        *,
        left_table: str,
        right_table: str,
        join_expr: str,
        dest_table: str,
        select: Dict[str, str],
        mode: str = "inner",
        left_where: Optional[str] = None,
        right_where: Optional[str] = None,
    ) -> str:
        resolved_left = self._resolve_context(left_table)
        resolved_right = self._resolve_context(right_table)
        resolved_dest = self._resolve_context(dest_table)
        join_expr, select = rewrite_join_paths(
            [left_table, right_table],
            [resolved_left, resolved_right],
            join_expr,
            select,
        )
        return join_tables_impl(
            left_table=resolved_left,
            right_table=resolved_right,
            join_expr=join_expr,
            dest_table=resolved_dest,
            select=select,
            mode=mode,
            left_where=left_where,
            right_where=right_where,
        )

    @functools.wraps(BaseDataManager.filter_join, updated=())
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
        # Resolve table contexts
        if isinstance(tables, str):
            tables = [tables]
        resolved_tables = [self._resolve_context(t) for t in tables]
        join_expr, select = rewrite_join_paths(
            tables,
            resolved_tables,
            join_expr,
            select,
        )

        return filter_join_impl(
            tables=resolved_tables,
            join_expr=join_expr,
            select=select,
            mode=mode,
            left_where=left_where,
            right_where=right_where,
            result_where=result_where,
            result_limit=result_limit,
            result_offset=result_offset,
        )

    @functools.wraps(BaseDataManager.reduce_join, updated=())
    def reduce_join(
        self,
        *,
        tables: Union[str, List[str]],
        join_expr: str,
        select: Dict[str, str],
        metric: str,
        columns: Union[str, List[str]],
        mode: str = "inner",
        left_where: Optional[str] = None,
        right_where: Optional[str] = None,
        result_where: Optional[str] = None,
        group_by: Optional[Union[str, List[str]]] = None,
    ) -> Any:
        if isinstance(tables, str):
            tables = [tables]
        resolved_tables = [self._resolve_context(t) for t in tables]
        join_expr, select = rewrite_join_paths(
            tables,
            resolved_tables,
            join_expr,
            select,
        )

        return reduce_join_impl(
            tables=resolved_tables,
            join_expr=join_expr,
            select=select,
            metric=metric,
            columns=columns,
            mode=mode,
            left_where=left_where,
            right_where=right_where,
            result_where=result_where,
            group_by=group_by,
        )

    @functools.wraps(BaseDataManager.search_join, updated=())
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
        # Resolve table contexts
        if isinstance(tables, str):
            tables = [tables]
        resolved_tables = [self._resolve_context(t) for t in tables]
        join_expr, select = rewrite_join_paths(
            tables,
            resolved_tables,
            join_expr,
            select,
        )

        return search_join_impl(
            tables=resolved_tables,
            join_expr=join_expr,
            select=select,
            mode=mode,
            left_where=left_where,
            right_where=right_where,
            references=references,
            k=k,
            filter=filter,
            tmp_context_prefix=self._base_ctx,
        )

    @functools.wraps(BaseDataManager.filter_multi_join, updated=())
    def filter_multi_join(
        self,
        *,
        joins: List[Dict[str, Any]],
        result_where: Optional[str] = None,
        result_limit: int = 100,
        result_offset: int = 0,
    ) -> List[Dict[str, Any]]:
        return filter_multi_join_impl(
            joins=joins,
            context_resolver=self._resolve_context,
            result_where=result_where,
            result_limit=result_limit,
            result_offset=result_offset,
            tmp_context_prefix=self._base_ctx,
        )

    @functools.wraps(BaseDataManager.search_multi_join, updated=())
    def search_multi_join(
        self,
        *,
        joins: List[Dict[str, Any]],
        references: Optional[Dict[str, str]] = None,
        k: int = 10,
        filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        return search_multi_join_impl(
            joins=joins,
            context_resolver=self._resolve_context,
            references=references,
            k=k,
            filter=filter,
            tmp_context_prefix=self._base_ctx,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Mutation Operations
    # ──────────────────────────────────────────────────────────────────────────

    @functools.wraps(BaseDataManager.insert_rows, updated=())
    def insert_rows(
        self,
        context: str,
        rows: List[Dict[str, Any]],
        *,
        add_to_all_context: bool = False,
        batched: bool = True,
    ) -> List[int]:
        resolved = self._resolve_context(context)
        return insert_rows_impl(
            resolved,
            rows,
            add_to_all_context=add_to_all_context,
            batched=batched,
        )

    @functools.wraps(BaseDataManager.update_rows, updated=())
    def update_rows(
        self,
        context: str,
        updates: Dict[str, Any],
        *,
        filter: str,
    ) -> int:
        resolved = self._resolve_context(context)
        return update_rows_impl(resolved, updates, filter=filter)

    @functools.wraps(BaseDataManager.delete_rows, updated=())
    def delete_rows(
        self,
        context: str,
        *,
        filter: Optional[str] = None,
        log_ids: Optional[List[int]] = None,
        dangerous_ok: bool = False,
        delete_empty_rows: bool = False,
    ) -> int:
        resolved = self._resolve_context(context)
        return delete_rows_impl(
            resolved,
            filter=filter,
            log_ids=log_ids,
            dangerous_ok=dangerous_ok,
            delete_empty_rows=delete_empty_rows,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # High-Level Ingestion
    # ──────────────────────────────────────────────────────────────────────────

    @functools.wraps(BaseDataManager.ingest, updated=())
    def ingest(
        self,
        context: str,
        rows: Optional[List[Dict[str, Any]]] = None,
        *,
        table_input_handle=None,
        description: Optional[str] = None,
        fields: Optional[Dict[str, Any]] = None,
        unique_keys: Optional[Dict[str, str]] = None,
        embed_columns: Optional[List[str]] = None,
        embed_strategy: str = "along",
        chunk_size: int = 1000,
        auto_counting: Optional[Dict[str, Optional[str]]] = None,
        infer_untyped_fields: bool = False,
        add_to_all_context: bool = False,
        execution: Optional["IngestExecutionConfig"] = None,
        post_ingest: Optional["PostIngestConfig"] = None,
        on_task_complete=None,
        coerce_types: bool = True,
    ) -> "IngestResult":
        resolved = self._resolve_context(context)
        unique_keys, auto_counting = self._resolve_unique_keys_and_auto_counting(
            resolved,
            unique_keys,
            auto_counting,
        )
        return run_ingest(
            self,
            resolved,
            rows,
            table_input_handle=table_input_handle,
            description=description,
            fields=fields,
            unique_keys=unique_keys,
            embed_columns=embed_columns,
            embed_strategy=embed_strategy,
            chunk_size=chunk_size,
            auto_counting=auto_counting,
            infer_untyped_fields=infer_untyped_fields,
            add_to_all_context=add_to_all_context,
            execution=execution,
            post_ingest=post_ingest,
            on_task_complete=on_task_complete,
            coerce_types=coerce_types,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Embedding Operations
    # ──────────────────────────────────────────────────────────────────────────

    @functools.wraps(BaseDataManager.ensure_vector_column, updated=())
    def ensure_vector_column(
        self,
        context: str,
        *,
        source_column: str,
        target_column: Optional[str] = None,
        async_embeddings: bool = False,
    ) -> str:
        resolved = self._resolve_context(context)
        target = target_column or f"_{source_column}_emb"
        _ensure_vector_column(
            context=resolved,
            embed_column=target,
            source_column=source_column,
            derived_expr=None,
            from_ids=None,
            async_embeddings=async_embeddings,
        )
        return target

    @functools.wraps(BaseDataManager.vectorize_rows, updated=())
    def vectorize_rows(
        self,
        context: str,
        *,
        source_column: str,
        target_column: Optional[str] = None,
        row_ids: Optional[List[int]] = None,
        batch_size: int = 100,
        async_embeddings: bool = False,
    ) -> int:
        resolved = self._resolve_context(context)
        target = target_column or f"_{source_column}_emb"
        _ensure_vector_column(
            context=resolved,
            embed_column=target,
            source_column=source_column,
            derived_expr=None,
            from_ids=row_ids,
            async_embeddings=async_embeddings,
        )
        return len(row_ids) if row_ids else 0
