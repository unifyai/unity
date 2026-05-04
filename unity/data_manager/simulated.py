"""
Simulated DataManager implementation.

This module provides a drop-in, side-effect-free replacement for DataManager
that returns plausible simulated data. It is useful for testing, demos, and
development scenarios where no real backend is available.

The simulated manager maintains an in-memory store for tables and rows,
allowing realistic multi-turn interactions without any external dependencies.
"""

from __future__ import annotations

import functools
import logging
import time
from typing import Any, Dict, List, Optional, Union
from collections import defaultdict

from unity.data_manager.base import BaseDataManager
from unity.data_manager.types.table import (
    TableDescription,
    TableSchema,
    ColumnInfo,
)
from unity.data_manager.types.ingest import (
    IngestExecutionConfig,
    IngestResult,
    PostIngestConfig,
)

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Shared reduce helpers (used by both reduce() and reduce_join())
# ──────────────────────────────────────────────────────────────────────────────


def _compute_metric(data: List[Dict[str, Any]], col: str, metric: str) -> Any:
    """Compute a single aggregate metric over *col* in *data*."""
    values = [row.get(col) for row in data if row.get(col) is not None]
    if metric == "count":
        return len(values)
    elif metric == "count_distinct":
        return len(set(values))
    elif metric == "sum":
        return sum(float(v) for v in values if v is not None)
    elif metric in ("avg", "mean"):
        if not values:
            return 0.0
        return sum(float(v) for v in values) / len(values)
    elif metric == "min":
        return min(values) if values else None
    elif metric == "max":
        return max(values) if values else None
    return 0


def _reduce_rows(
    rows: List[Dict[str, Any]],
    *,
    metric: str,
    columns: Union[str, List[str]],
    group_by: Optional[Union[str, List[str]]] = None,
) -> Any:
    """Shared reduce logic for both ``reduce()`` and ``reduce_join()``."""
    cols_list = [columns] if isinstance(columns, str) else list(columns)

    if group_by is None:
        if len(cols_list) == 1:
            return _compute_metric(rows, cols_list[0], metric)
        return {col: _compute_metric(rows, col, metric) for col in cols_list}

    groups_list = [group_by] if isinstance(group_by, str) else list(group_by)
    grouped: Dict[tuple, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = tuple(row.get(g) for g in groups_list)
        grouped[key].append(row)

    results = []
    for key, group_rows in grouped.items():
        result = dict(zip(groups_list, key))
        if len(cols_list) == 1:
            result[metric] = _compute_metric(group_rows, cols_list[0], metric)
        else:
            for col in cols_list:
                result[col] = _compute_metric(group_rows, col, metric)
        results.append(result)
    return results


class SimulatedDataManager(BaseDataManager):
    """
    A drop-in, side-effect-free replacement for DataManager.

    This simulated manager maintains an in-memory store of tables and rows,
    allowing realistic interactions without any backend. Useful for testing,
    demos, and development scenarios.

    The simulated manager:
    - Creates tables in memory with inferred or explicit schemas
    - Stores inserted rows and supports filtering/search
    - Returns deterministic placeholder data for consistency
    - Does NOT persist data between instances

    Usage Examples
    --------------
    >>> dm = SimulatedDataManager()
    >>> dm.create_table("test/data", fields={"id": "int", "name": "str"})
    >>> dm.insert_rows("test/data", [{"id": 1, "name": "Alice"}])
    >>> rows = dm.filter("test/data")
    >>> print(rows)  # [{"id": 1, "name": "Alice"}]
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initialize simulated DataManager with empty in-memory stores."""
        super().__init__()

        # In-memory storage: {context_path: list of row dicts}
        self._tables: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

        # Schema storage: {context_path: {field_name: field_type}}
        self._schemas: Dict[str, Dict[str, str]] = {}

        # Table metadata: {context_path: description}
        self._descriptions: Dict[str, str] = {}

        # Unique keys: {context_path: list of key column names}
        self._unique_keys: Dict[str, Any] = {}

        # Auto counting config: {context_path: auto_counting config}
        self._auto_counting: Dict[str, Any] = {}

        # Embedding columns: {context_path: set of embedding column names}
        self._embeddings: Dict[str, set] = defaultdict(set)

        # Simulated base context
        self._base_ctx = "Data"

        # Auto-incrementing log ID counter
        self._next_log_id: int = 1

        logger.debug("SimulatedDataManager initialized")

    def _resolve_context(self, context: str) -> str:
        """Resolve context path (passthrough for simulated manager)."""
        context = context.lstrip("/")
        if not context:
            raise ValueError("Empty context path")
        if context.startswith(("Data/", "Files/", "Knowledge/")):
            return context
        return f"{self._base_ctx}/{context}"

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
        destination: str | None = None,
    ) -> str:
        resolved = self._resolve_context(context)
        if fields:
            self._schemas[resolved] = dict(fields)
        if description:
            self._descriptions[resolved] = description
        if unique_keys is not None:
            self._unique_keys[resolved] = unique_keys
        if auto_counting is not None:
            self._auto_counting[resolved] = auto_counting
        # Ensure table exists in _tables
        if resolved not in self._tables:
            self._tables[resolved] = []
        logger.debug("Simulated: created table %s", resolved)
        return resolved

    @functools.wraps(BaseDataManager.describe_table, updated=())
    def describe_table(self, context: str) -> TableDescription:
        resolved = self._resolve_context(context)

        # Build column info from schema
        columns = []
        schema = self._schemas.get(resolved, {})
        for name, dtype in schema.items():
            if not name.startswith("_"):
                columns.append(ColumnInfo(name=name, dtype=dtype))

        # Get embedding columns
        emb_cols = list(self._embeddings.get(resolved, set()))

        return TableDescription(
            context=resolved,
            description=self._descriptions.get(resolved),
            table_schema=TableSchema(columns=columns),
            has_embeddings=bool(emb_cols),
            embedding_columns=emb_cols,
        )

    @functools.wraps(BaseDataManager.get_columns, updated=())
    def get_columns(self, table: str) -> Dict[str, Any]:
        resolved = self._resolve_context(table)
        schema = self._schemas.get(resolved, {})
        # Convert schema to column info format
        columns: Dict[str, Any] = {}
        for name, dtype in schema.items():
            columns[name] = {"data_type": dtype}
        return columns

    @functools.wraps(BaseDataManager.get_table, updated=())
    def get_table(self, context: str) -> Dict[str, Any]:
        resolved = self._resolve_context(context)
        if resolved not in self._tables:
            raise ValueError(f"Table not found: {resolved}")
        return {
            "description": self._descriptions.get(resolved),
            # Note: Real Unify API returns [] for unique_keys, which gets normalized
            # to None by TableSchema validator. SimulatedDataManager mimics this.
            "unique_keys": self._unique_keys.get(resolved),
            "auto_counting": self._auto_counting.get(resolved),
        }

    @functools.wraps(BaseDataManager.list_tables, updated=())
    def list_tables(
        self,
        *,
        prefix: Optional[str] = None,
        include_column_info: bool = True,
    ) -> Union[List[str], Dict[str, Any]]:
        all_contexts = list(self._tables.keys())
        if prefix:
            all_contexts = [c for c in all_contexts if c.startswith(prefix)]

        if include_column_info:
            result: Dict[str, Any] = {}
            for ctx in sorted(all_contexts):
                result[ctx] = {
                    "description": self._descriptions.get(ctx),
                }
            return result
        else:
            return sorted(all_contexts)

    @functools.wraps(BaseDataManager.delete_table, updated=())
    def delete_table(
        self,
        context: str,
        *,
        dangerous_ok: bool = False,
        destination: str | None = None,
    ) -> None:
        if not dangerous_ok:
            raise ValueError(
                "delete_table is a destructive operation. "
                "Set dangerous_ok=True to confirm.",
            )
        resolved = self._resolve_context(context)
        self._tables.pop(resolved, None)
        self._schemas.pop(resolved, None)
        self._descriptions.pop(resolved, None)
        self._embeddings.pop(resolved, None)
        logger.debug("Simulated: deleted table %s", resolved)

    @functools.wraps(BaseDataManager.rename_table, updated=())
    def rename_table(
        self,
        old_context: str,
        new_context: str,
        *,
        destination: str | None = None,
    ) -> Dict[str, str]:
        old_resolved = self._resolve_context(old_context)
        new_resolved = self._resolve_context(new_context)

        if old_resolved not in self._tables:
            raise ValueError(f"Table {old_resolved} does not exist")
        if new_resolved in self._tables:
            raise ValueError(f"Table {new_resolved} already exists")

        # Move data
        self._tables[new_resolved] = self._tables.pop(old_resolved)
        if old_resolved in self._schemas:
            self._schemas[new_resolved] = self._schemas.pop(old_resolved)
        if old_resolved in self._descriptions:
            self._descriptions[new_resolved] = self._descriptions.pop(old_resolved)
        if old_resolved in self._embeddings:
            self._embeddings[new_resolved] = self._embeddings.pop(old_resolved)

        logger.debug("Simulated: renamed table %s -> %s", old_resolved, new_resolved)
        return {
            "status": "renamed",
            "old_context": old_resolved,
            "new_context": new_resolved,
        }

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
        destination: str | None = None,
    ) -> Dict[str, str]:
        if column_name == "id":
            raise ValueError("Cannot create a column with reserved name 'id'.")

        resolved = self._resolve_context(context)
        if resolved not in self._schemas:
            self._schemas[resolved] = {}
        self._schemas[resolved][column_name] = column_type
        logger.debug("Simulated: created column %s in %s", column_name, resolved)
        return {"status": "created", "column": column_name, "type": column_type}

    @functools.wraps(BaseDataManager.delete_column, updated=())
    def delete_column(
        self,
        context: str,
        *,
        column_name: str,
        destination: str | None = None,
    ) -> Dict[str, str]:
        resolved = self._resolve_context(context)
        if resolved in self._schemas and column_name in self._schemas[resolved]:
            del self._schemas[resolved][column_name]

        # Remove column from rows
        for row in self._tables.get(resolved, []):
            row.pop(column_name, None)

        logger.debug("Simulated: deleted column %s from %s", column_name, resolved)
        return {"status": "deleted", "column": column_name}

    @functools.wraps(BaseDataManager.rename_column, updated=())
    def rename_column(
        self,
        context: str,
        *,
        old_name: str,
        new_name: str,
        destination: str | None = None,
    ) -> Dict[str, str]:
        if old_name == new_name:
            return {"info": "no-op: old and new names are identical"}
        if new_name == "id":
            raise ValueError("Cannot rename a column to reserved name 'id'.")

        resolved = self._resolve_context(context)

        # Rename in schema
        if resolved in self._schemas and old_name in self._schemas[resolved]:
            self._schemas[resolved][new_name] = self._schemas[resolved].pop(old_name)

        # Rename in rows
        for row in self._tables.get(resolved, []):
            if old_name in row:
                row[new_name] = row.pop(old_name)

        logger.debug(
            "Simulated: renamed column %s -> %s in %s",
            old_name,
            new_name,
            resolved,
        )
        return {"status": "renamed", "old_name": old_name, "new_name": new_name}

    @functools.wraps(BaseDataManager.create_derived_column, updated=())
    def create_derived_column(
        self,
        context: str,
        *,
        column_name: str,
        equation: str,
        destination: str | None = None,
    ) -> Dict[str, str]:
        resolved = self._resolve_context(context)

        # Add to schema
        if resolved not in self._schemas:
            self._schemas[resolved] = {}
        self._schemas[resolved][column_name] = "derived"

        # Compute values for existing rows (simple simulation)
        for row in self._tables.get(resolved, []):
            try:
                # Replace {col} with row[col] for evaluation
                eval_equation = equation
                import re

                for col in re.findall(r"\{(\w+)\}", equation):
                    if col in row:
                        eval_equation = eval_equation.replace(
                            f"{{{col}}}",
                            repr(row[col]),
                        )
                row[column_name] = eval(eval_equation, {"__builtins__": {}})
            except Exception:
                row[column_name] = None

        logger.debug(
            "Simulated: created derived column %s in %s",
            column_name,
            resolved,
        )
        return {"status": "created", "column": column_name, "equation": equation}

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
        rows = list(self._tables.get(resolved, []))

        # Apply filter expression (simple eval for simulated)
        if filter:
            filtered = []
            for row in rows:
                try:
                    if eval(filter, {"__builtins__": {}}, row):
                        filtered.append(row)
                except Exception:
                    pass
            rows = filtered

        # Apply ordering
        if order_by and rows:
            try:
                rows = sorted(
                    rows,
                    key=lambda r: r.get(order_by, ""),
                    reverse=descending,
                )
            except Exception:
                pass

        # Apply pagination
        rows = rows[offset : offset + limit]

        # If return_ids_only, return actual log IDs
        if return_ids_only:
            return [row.get("_log_id", i + 1) for i, row in enumerate(rows)]

        # Select columns
        if columns:
            rows = [{k: row.get(k) for k in columns} for row in rows]

        # Exclude columns
        if exclude_columns:
            rows = [
                {k: v for k, v in row.items() if k not in exclude_columns}
                for row in rows
            ]

        return rows

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
        # For simulated search, just return filtered results with fake similarity
        resolved = self._resolve_context(context)
        rows = list(self._tables.get(resolved, []))

        # Apply filter if provided
        if filter:
            filtered = []
            for row in rows:
                try:
                    if eval(filter, {"__builtins__": {}}, row):
                        filtered.append(row)
                except Exception:
                    pass
            rows = filtered

        # Without references, just return rows without ranking
        if not references:
            results = rows[:k]
            if columns:
                results = [{c: row.get(c) for c in columns} for row in results]
            return results

        # Simulate semantic ranking by checking if reference words appear in target columns
        def _score(row: Dict[str, Any]) -> float:
            total_score = 0.0
            for col, ref_text in references.items():
                query_words = set(ref_text.lower().split())
                col_value = str(row.get(col, "")).lower()
                matches = sum(1 for w in query_words if w in col_value)
                total_score += matches / len(query_words) if query_words else 0.0
            return total_score / len(references) if references else 0.0

        scored = [(row, _score(row)) for row in rows]
        scored.sort(key=lambda x: x[1], reverse=True)

        # Add similarity score and limit
        results = []
        for row, score in scored[:k]:
            result = dict(row)
            result["_similarity"] = score
            results.append(result)

        # Select columns
        if columns:
            results = [
                {c: row.get(c) for c in columns + ["_similarity"]} for row in results
            ]

        return results

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
        rows = list(self._tables.get(resolved, []))

        if filter:
            filtered = []
            for row in rows:
                try:
                    if eval(filter, {"__builtins__": {}}, row):
                        filtered.append(row)
                except Exception:
                    pass
            rows = filtered

        return _reduce_rows(rows, metric=metric, columns=columns, group_by=group_by)

    # ──────────────────────────────────────────────────────────────────────────
    # Join Operations
    # ──────────────────────────────────────────────────────────────────────────

    def _simple_join(
        self,
        left_ctx: str,
        right_ctx: str,
        select: Dict[str, str],
        left_where: Optional[str] = None,
        right_where: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Helper to perform a simple simulated join based on select column mappings."""
        left_rows = list(self._tables.get(left_ctx, []))
        right_rows = list(self._tables.get(right_ctx, []))

        # Apply pre-filters
        if left_where:
            filtered = []
            for row in left_rows:
                try:
                    if eval(left_where, {"__builtins__": {}}, row):
                        filtered.append(row)
                except Exception:
                    pass
            left_rows = filtered

        if right_where:
            filtered = []
            for row in right_rows:
                try:
                    if eval(right_where, {"__builtins__": {}}, row):
                        filtered.append(row)
                except Exception:
                    pass
            right_rows = filtered

        # Simulated cartesian join with column selection
        # In practice, joins would match on join_expr, but for simulation
        # we just do a simple merge based on select
        results = []
        for left_row in left_rows:
            for right_row in right_rows:
                merged: Dict[str, Any] = {}
                for src, alias in select.items():
                    # src is like "Context.column" - extract column name
                    if "." in src:
                        col = src.split(".")[-1]
                    else:
                        col = src
                    # Try to get from left first, then right
                    if col in left_row:
                        merged[alias] = left_row[col]
                    elif col in right_row:
                        merged[alias] = right_row[col]
                results.append(merged)
                # For simulation, just take first match per left row
                break

        return results

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
        destination: str | None = None,
    ) -> str:
        left_ctx = self._resolve_context(left_table)
        right_ctx = self._resolve_context(right_table)
        dest_ctx = self._resolve_context(dest_table)

        # Perform the join and store in dest_table
        joined_rows = self._simple_join(
            left_ctx,
            right_ctx,
            select,
            left_where=left_where,
            right_where=right_where,
        )

        # Store in dest context
        self._tables[dest_ctx] = joined_rows
        logger.debug(
            "Simulated: join_tables %s + %s -> %s (%d rows)",
            left_ctx,
            right_ctx,
            dest_ctx,
            len(joined_rows),
        )
        return dest_ctx

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
        if isinstance(tables, str):
            tables = [tables]
        if len(tables) != 2:
            raise ValueError("Exactly TWO tables are required.")

        left_resolved = self._resolve_context(tables[0])
        right_resolved = self._resolve_context(tables[1])

        results = self._simple_join(
            left_resolved,
            right_resolved,
            select,
            left_where,
            right_where,
        )

        # Apply result_where filter
        if result_where:
            filtered = []
            for row in results:
                try:
                    if eval(result_where, {"__builtins__": {}}, row):
                        filtered.append(row)
                except Exception:
                    pass
            results = filtered

        # Apply pagination
        results = results[result_offset : result_offset + result_limit]

        return results

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
        if len(tables) != 2:
            raise ValueError("Exactly TWO tables are required.")

        left_resolved = self._resolve_context(tables[0])
        right_resolved = self._resolve_context(tables[1])

        rows = self._simple_join(
            left_resolved,
            right_resolved,
            select,
            left_where,
            right_where,
        )

        if result_where:
            filtered = []
            for row in rows:
                try:
                    if eval(result_where, {"__builtins__": {}}, row):
                        filtered.append(row)
                except Exception:
                    pass
            rows = filtered

        return _reduce_rows(rows, metric=metric, columns=columns, group_by=group_by)

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
        # Perform join then return first k results (simulated ranking)
        joined = self.filter_join(
            tables=tables,
            join_expr=join_expr,
            select=select,
            mode=mode,
            left_where=left_where,
            right_where=right_where,
            result_where=filter,
            result_limit=k,
        )
        return joined

    @functools.wraps(BaseDataManager.filter_multi_join, updated=())
    def filter_multi_join(
        self,
        *,
        joins: List[Dict[str, Any]],
        result_where: Optional[str] = None,
        result_limit: int = 100,
        result_offset: int = 0,
    ) -> List[Dict[str, Any]]:
        if not joins:
            raise ValueError("`joins` must contain at least one join step.")

        # For simulation, chain the joins
        previous_result: Optional[List[Dict[str, Any]]] = None

        for idx, step in enumerate(joins):
            raw_tables = step.get("tables", [])
            if isinstance(raw_tables, str):
                raw_tables = [raw_tables]
            if len(raw_tables) != 2:
                raise ValueError(f"Step {idx} must have exactly TWO tables")

            select = step.get("select", {})

            if idx == 0:
                # First join - use actual tables
                left_ctx = self._resolve_context(raw_tables[0])
                right_ctx = self._resolve_context(raw_tables[1])
                previous_result = self._simple_join(
                    left_ctx,
                    right_ctx,
                    select,
                    step.get("left_where"),
                    step.get("right_where"),
                )
            else:
                # Subsequent joins - use previous result as "left"
                # For simulation, just continue with last result
                right_table = raw_tables[1]
                if right_table in {"$prev", "__prev__", "_"}:
                    continue
                right_ctx = self._resolve_context(right_table)
                right_rows = list(self._tables.get(right_ctx, []))

                # Simple merge
                new_result = []
                for prev_row in previous_result or []:
                    for right_row in right_rows:
                        merged: Dict[str, Any] = {}
                        for src, alias in select.items():
                            col = src.split(".")[-1] if "." in src else src
                            if col in prev_row:
                                merged[alias] = prev_row[col]
                            elif col in right_row:
                                merged[alias] = right_row[col]
                        new_result.append(merged)
                        break
                previous_result = new_result

        results = previous_result or []

        # Apply result_where
        if result_where:
            filtered = []
            for row in results:
                try:
                    if eval(result_where, {"__builtins__": {}}, row):
                        filtered.append(row)
                except Exception:
                    pass
            results = filtered

        # Apply pagination
        results = results[result_offset : result_offset + result_limit]

        return results

    @functools.wraps(BaseDataManager.search_multi_join, updated=())
    def search_multi_join(
        self,
        *,
        joins: List[Dict[str, Any]],
        references: Optional[Dict[str, str]] = None,
        k: int = 10,
        filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        # Apply filter then return first k results
        result = self.filter_multi_join(
            joins=joins,
            result_where=filter,
            result_limit=k,
        )
        return result

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
        destination: str | None = None,
    ) -> List[int]:
        if not rows:
            return []

        resolved = self._resolve_context(context)

        inserted_ids: List[int] = []
        for row in rows:
            log_id = self._next_log_id
            self._next_log_id += 1
            row_with_id = {**row, "_log_id": log_id}
            self._tables[resolved].append(row_with_id)
            inserted_ids.append(log_id)

        return inserted_ids

    @functools.wraps(BaseDataManager.update_rows, updated=())
    def update_rows(
        self,
        context: str,
        updates: Dict[str, Any],
        *,
        filter: str,
        destination: str | None = None,
    ) -> int:
        resolved = self._resolve_context(context)
        updated = 0

        new_rows = []
        for row in self._tables.get(resolved, []):
            try:
                if eval(filter, {"__builtins__": {}}, row):
                    row = {**row, **updates}
                    updated += 1
            except Exception:
                pass
            new_rows.append(row)

        self._tables[resolved] = new_rows
        return updated

    @functools.wraps(BaseDataManager.delete_rows, updated=())
    def delete_rows(
        self,
        context: str,
        *,
        filter: Optional[str] = None,
        log_ids: Optional[List[int]] = None,
        dangerous_ok: bool = False,
        delete_empty_rows: bool = False,
        destination: str | None = None,
    ) -> int:
        if not dangerous_ok:
            raise ValueError(
                "delete_rows is a destructive operation. "
                "Set dangerous_ok=True to confirm.",
            )

        if filter is None and log_ids is None:
            raise ValueError(
                "Either filter or log_ids must be provided for delete_rows",
            )

        resolved = self._resolve_context(context)
        original_count = len(self._tables.get(resolved, []))

        if log_ids is not None:
            # Delete rows matching the log IDs
            log_id_set = set(log_ids)
            rows = self._tables.get(resolved, [])
            new_rows = [r for r in rows if r.get("_log_id") not in log_id_set]
            deleted_count = len(rows) - len(new_rows)
            self._tables[resolved] = new_rows
            return deleted_count

        if filter is not None:
            new_rows = []
            for row in self._tables.get(resolved, []):
                try:
                    if not eval(filter, {"__builtins__": {}}, row):
                        new_rows.append(row)
                except Exception:
                    new_rows.append(row)

            self._tables[resolved] = new_rows
            return original_count - len(new_rows)

        return 0

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
        execution: Optional[IngestExecutionConfig] = None,
        post_ingest: Optional[PostIngestConfig] = None,
        on_task_complete=None,
        coerce_types: bool = True,
        storage_client=None,
        skip_rows: int = 0,
        destination: str | None = None,
    ) -> IngestResult:
        if table_input_handle is not None:
            from unity.common.pipeline.row_streaming import (
                iter_table_input_rows,
            )

            rows = list(
                iter_table_input_rows(
                    table_input_handle,
                    storage_client=storage_client,
                    skip_rows=skip_rows,
                ),
            )
        elif rows is None:
            rows = []

        start = time.perf_counter()
        resolved = self._resolve_context(context)

        # Step 1: create table (idempotent)
        self.create_table(
            resolved,
            description=description,
            fields=fields,
            unique_keys=unique_keys,
            auto_counting=auto_counting,
        )

        # Step 2: insert rows
        log_ids = self.insert_rows(resolved, rows)

        # Step 3: optional embedding
        rows_embedded = 0
        if embed_columns:
            for col in embed_columns:
                self.ensure_vector_column(resolved, source_column=col)
                rows_embedded += self.vectorize_rows(
                    resolved,
                    source_column=col,
                    row_ids=log_ids,
                )

        duration_ms = (time.perf_counter() - start) * 1000
        chunks_processed = (
            max(1, (len(rows) + chunk_size - 1) // chunk_size) if rows else 0
        )

        return IngestResult(
            context=resolved,
            rows_inserted=len(log_ids),
            rows_embedded=rows_embedded,
            log_ids=log_ids,
            duration_ms=duration_ms,
            chunks_processed=chunks_processed,
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
        destination: str | None = None,
    ) -> str:
        target = target_column or f"_{source_column}_emb"
        resolved = self._resolve_context(context)
        self._embeddings[resolved].add(target)
        logger.debug("Simulated: ensured vector column %s in %s", target, resolved)
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
        destination: str | None = None,
    ) -> int:
        # Simulated: just count rows that would be embedded
        resolved = self._resolve_context(context)
        rows = self._tables.get(resolved, [])
        if row_ids:
            return len(row_ids)
        return len(rows)

    # ──────────────────────────────────────────────────────────────────────────
    # Utility Methods
    # ──────────────────────────────────────────────────────────────────────────

    def clear(self, *, destination: str | None = None) -> None:
        """Clear all in-memory tables and reset the simulated manager."""
        self._tables.clear()
        self._schemas.clear()
        self._descriptions.clear()
        self._unique_keys.clear()
        self._auto_counting.clear()
        self._embeddings.clear()
        self._next_log_id = 1
        logger.debug("SimulatedDataManager cleared")
