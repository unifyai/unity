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
from typing import Any, Dict, List, Optional, Union


from unity.data_manager.base import BaseDataManager
from unity.data_manager.types.table import TableDescription
from unity.data_manager.types.plot import PlotConfig, PlotResult
from unity.data_manager.ops.table_ops import (
    create_table_impl,
    describe_table_impl,
    list_tables_impl,
    delete_table_impl,
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
    filter_join_impl,
    search_join_impl,
    filter_multi_join_impl,
    search_multi_join_impl,
)
from unity.data_manager.ops.embedding_ops import (
    ensure_vector_column_impl,
    vectorize_rows_impl,
)
from unity.data_manager.ops.plot_ops import (
    generate_plot,
    generate_plots_batch,
)
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

        # Resolve owned base context via ContextRegistry
        # This gives us the fully-qualified path like "User/Assistant/Data"
        try:
            self._base_ctx = ContextRegistry.get_context(self, "Data")
        except Exception:
            # Fallback for tests or offline scenarios
            self._base_ctx = "Data"

        logger.debug("DataManager initialized with base context: %s", self._base_ctx)

    def _resolve_context(self, context: str) -> str:
        """
        Resolve a context path, handling relative and absolute paths.

        Parameters
        ----------
        context : str
            Context path. Can be:
            - Relative: "projects/housing" → resolved to "{base_ctx}/projects/housing"
            - Absolute owned: "Data/MH/arrears" → used as-is
            - Foreign: "Files/Local/..." → used as-is

        Returns
        -------
        str
            Fully resolved context path.
        """
        # Check for known absolute prefixes
        if any(context.startswith(p) for p in _ABSOLUTE_PREFIXES):
            return context

        # Check if it looks like a fully-qualified path already
        if self._base_ctx and context.startswith(self._base_ctx):
            return context

        # Heuristic: if it has many path parts and doesn't start with base,
        # it might be fully qualified
        parts = context.split("/")
        if len(parts) > 3:
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
        fields: Optional[Dict[str, str]] = None,
        unique_keys: Optional[Dict[str, str]] = None,
        auto_counting: Optional[Dict[str, Optional[str]]] = None,
    ) -> str:
        resolved = self._resolve_context(context)
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

    @functools.wraps(BaseDataManager.list_tables, updated=())
    def list_tables(self, *, prefix: Optional[str] = None) -> List[str]:
        resolved_prefix = self._resolve_context(prefix) if prefix else None
        return list_tables_impl(prefix=resolved_prefix)

    @functools.wraps(BaseDataManager.delete_table, updated=())
    def delete_table(
        self,
        context: str,
        *,
        dangerous_ok: bool = False,
    ) -> None:
        resolved = self._resolve_context(context)
        delete_table_impl(resolved, dangerous_ok=dangerous_ok)

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
        limit: int = 100,
        offset: int = 0,
        order_by: Optional[str] = None,
        descending: bool = False,
    ) -> List[Dict[str, Any]]:
        resolved = self._resolve_context(context)
        return filter_impl(
            resolved,
            filter=filter,
            columns=columns,
            limit=limit,
            offset=offset,
            order_by=order_by,
            descending=descending,
        )

    @functools.wraps(BaseDataManager.search, updated=())
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
        resolved = self._resolve_context(context)
        return search_impl(
            resolved,
            query=query,
            k=k,
            filter=filter,
            vector_column=vector_column,
            columns=columns,
        )

    @functools.wraps(BaseDataManager.reduce, updated=())
    def reduce(
        self,
        context: str,
        *,
        metric: str,
        column: Optional[str] = None,
        filter: Optional[str] = None,
        group_by: Optional[Union[str, List[str]]] = None,
    ) -> Any:
        resolved = self._resolve_context(context)
        return reduce_impl(
            resolved,
            metric=metric,
            column=column,
            filter=filter,
            group_by=group_by,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Join Operations
    # ──────────────────────────────────────────────────────────────────────────

    @functools.wraps(BaseDataManager.filter_join, updated=())
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
        left_resolved = self._resolve_context(left_context)
        right_resolved = self._resolve_context(right_context)
        return filter_join_impl(
            left_context=left_resolved,
            right_context=right_resolved,
            join_column=join_column,
            filter=filter,
            columns=columns,
            limit=limit,
        )

    @functools.wraps(BaseDataManager.search_join, updated=())
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
        left_resolved = self._resolve_context(left_context)
        right_resolved = self._resolve_context(right_context)
        return search_join_impl(
            left_context=left_resolved,
            right_context=right_resolved,
            join_column=join_column,
            query=query,
            k=k,
            filter=filter,
            vector_column=vector_column,
        )

    @functools.wraps(BaseDataManager.filter_multi_join, updated=())
    def filter_multi_join(
        self,
        *,
        contexts: List[str],
        join_columns: List[str],
        filter: Optional[str] = None,
        columns: Optional[List[str]] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        resolved_contexts = [self._resolve_context(c) for c in contexts]
        return filter_multi_join_impl(
            contexts=resolved_contexts,
            join_columns=join_columns,
            filter=filter,
            columns=columns,
            limit=limit,
        )

    @functools.wraps(BaseDataManager.search_multi_join, updated=())
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
        resolved_contexts = [self._resolve_context(c) for c in contexts]
        return search_multi_join_impl(
            contexts=resolved_contexts,
            join_columns=join_columns,
            query=query,
            k=k,
            filter=filter,
            vector_column=vector_column,
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
        dedupe_key: Optional[str] = None,
    ) -> int:
        resolved = self._resolve_context(context)
        return insert_rows_impl(resolved, rows, dedupe_key=dedupe_key)

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
        filter: str,
        dangerous_ok: bool = False,
    ) -> int:
        resolved = self._resolve_context(context)
        return delete_rows_impl(resolved, filter=filter, dangerous_ok=dangerous_ok)

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
    ) -> str:
        resolved = self._resolve_context(context)
        return ensure_vector_column_impl(
            resolved,
            source_column=source_column,
            target_column=target_column,
        )

    @functools.wraps(BaseDataManager.vectorize_rows, updated=())
    def vectorize_rows(
        self,
        context: str,
        *,
        source_column: str,
        target_column: Optional[str] = None,
        row_ids: Optional[List[int]] = None,
        batch_size: int = 100,
    ) -> int:
        resolved = self._resolve_context(context)
        return vectorize_rows_impl(
            resolved,
            source_column=source_column,
            target_column=target_column,
            row_ids=row_ids,
            batch_size=batch_size,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Visualization
    # ──────────────────────────────────────────────────────────────────────────

    @functools.wraps(BaseDataManager.plot, updated=())
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
    ) -> PlotResult:
        resolved = self._resolve_context(context)
        config = PlotConfig(
            plot_type=plot_type,
            x_axis=x,
            y_axis=y,
            group_by=group_by,
            aggregate=aggregate,
            scale_x=scale_x,
            scale_y=scale_y,
            bin_count=bin_count,
            show_regression=show_regression,
            title=title,
        )
        return generate_plot(
            config=config,
            context=resolved,
            filter_expr=filter,
        )

    @functools.wraps(BaseDataManager.plot_batch, updated=())
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
    ) -> List[PlotResult]:
        resolved_contexts = [self._resolve_context(c) for c in contexts]
        config = PlotConfig(
            plot_type=plot_type,
            x_axis=x,
            y_axis=y,
            group_by=group_by,
            aggregate=aggregate,
            title=title,
            **{k: v for k, v in kwargs.items() if v is not None},
        )
        return generate_plots_batch(
            contexts=resolved_contexts,
            config=config,
            filter_expr=filter,
        )
