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
from typing import Any, Dict, List, Optional, Union
from collections import defaultdict

from unity.data_manager.base import BaseDataManager
from unity.data_manager.types.table import (
    TableDescription,
    TableSchema,
    ColumnInfo,
)
from unity.data_manager.types.plot import PlotResult

logger = logging.getLogger(__name__)


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

    def __init__(self) -> None:
        """Initialize simulated DataManager with empty in-memory stores."""
        super().__init__()

        # In-memory storage: {context_path: list of row dicts}
        self._tables: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

        # Schema storage: {context_path: {field_name: field_type}}
        self._schemas: Dict[str, Dict[str, str]] = {}

        # Table metadata: {context_path: description}
        self._descriptions: Dict[str, str] = {}

        # Embedding columns: {context_path: set of embedding column names}
        self._embeddings: Dict[str, set] = defaultdict(set)

        # Simulated base context
        self._base_ctx = "Data"

        logger.debug("SimulatedDataManager initialized")

    def _resolve_context(self, context: str) -> str:
        """Resolve context path (passthrough for simulated manager)."""
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
        fields: Optional[Dict[str, str]] = None,
        unique_keys: Optional[Dict[str, str]] = None,
        auto_counting: Optional[Dict[str, Optional[str]]] = None,
    ) -> str:
        resolved = self._resolve_context(context)
        if fields:
            self._schemas[resolved] = dict(fields)
        if description:
            self._descriptions[resolved] = description
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

    @functools.wraps(BaseDataManager.list_tables, updated=())
    def list_tables(self, *, prefix: Optional[str] = None) -> List[str]:
        all_contexts = list(self._tables.keys())
        if prefix:
            all_contexts = [c for c in all_contexts if c.startswith(prefix)]
        return sorted(all_contexts)

    @functools.wraps(BaseDataManager.delete_table, updated=())
    def delete_table(
        self,
        context: str,
        *,
        dangerous_ok: bool = False,
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

        # Select columns
        if columns:
            rows = [{k: row.get(k) for k in columns} for row in rows]

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

        # Apply filter
        if filter:
            filtered = []
            for row in rows:
                try:
                    if eval(filter, {"__builtins__": {}}, row):
                        filtered.append(row)
                except Exception:
                    pass
            rows = filtered

        # Normalize columns to list
        cols_list = [columns] if isinstance(columns, str) else list(columns)

        def _compute_metric(data: List[Dict[str, Any]], col: str, m: str) -> Any:
            values = [row.get(col) for row in data if row.get(col) is not None]
            if m == "count":
                return len(values)
            elif m == "count_distinct":
                return len(set(values))
            elif m == "sum":
                return sum(float(v) for v in values if v is not None)
            elif m in ("avg", "mean"):
                if not values:
                    return 0.0
                return sum(float(v) for v in values) / len(values)
            elif m == "min":
                return min(values) if values else None
            elif m == "max":
                return max(values) if values else None
            return 0

        if group_by is None:
            # No grouping
            if len(cols_list) == 1:
                # Single column: return scalar
                return _compute_metric(rows, cols_list[0], metric)
            else:
                # Multiple columns: return dict
                return {col: _compute_metric(rows, col, metric) for col in cols_list}

        # Group by
        groups_list = [group_by] if isinstance(group_by, str) else list(group_by)
        grouped: Dict[tuple, List[Dict[str, Any]]] = defaultdict(list)
        for row in rows:
            key = tuple(row.get(g) for g in groups_list)
            grouped[key].append(row)

        results = []
        for key, group_rows in grouped.items():
            result = dict(zip(groups_list, key))
            if len(cols_list) == 1:
                # Single column: add metric result
                result[metric] = _compute_metric(group_rows, cols_list[0], metric)
            else:
                # Multiple columns: add all metric results
                for col in cols_list:
                    result[col] = _compute_metric(group_rows, col, metric)
            results.append(result)

        return results

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
        dedupe_key: Optional[str] = None,
    ) -> int:
        if not rows:
            return 0

        resolved = self._resolve_context(context)

        if dedupe_key:
            # Upsert mode: remove existing rows with same key
            existing_keys = {row.get(dedupe_key) for row in self._tables[resolved]}
            new_rows = []
            updated = 0
            for row in rows:
                key_val = row.get(dedupe_key)
                if key_val in existing_keys:
                    # Remove old row
                    self._tables[resolved] = [
                        r
                        for r in self._tables[resolved]
                        if r.get(dedupe_key) != key_val
                    ]
                    updated += 1
                new_rows.append(row)
            self._tables[resolved].extend(new_rows)
            return len(new_rows)
        else:
            self._tables[resolved].extend(rows)
            return len(rows)

    @functools.wraps(BaseDataManager.update_rows, updated=())
    def update_rows(
        self,
        context: str,
        updates: Dict[str, Any],
        *,
        filter: str,
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
        filter: str,
        dangerous_ok: bool = False,
    ) -> int:
        if not dangerous_ok:
            raise ValueError(
                "delete_rows is a destructive operation. "
                "Set dangerous_ok=True to confirm.",
            )

        resolved = self._resolve_context(context)
        original_count = len(self._tables.get(resolved, []))

        new_rows = []
        for row in self._tables.get(resolved, []):
            try:
                if not eval(filter, {"__builtins__": {}}, row):
                    new_rows.append(row)
            except Exception:
                new_rows.append(row)

        self._tables[resolved] = new_rows
        return original_count - len(new_rows)

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
    ) -> int:
        # Simulated: just count rows that would be embedded
        resolved = self._resolve_context(context)
        rows = self._tables.get(resolved, [])
        if row_ids:
            return len(row_ids)
        return len(rows)

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
        return PlotResult(
            url=f"https://simulated-plot.example.com/{resolved}/{plot_type}",
            token="simulated-token",
            expires_in_hours=24,
            title=title or f"Simulated {plot_type} plot",
            context=resolved,
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
        return [
            self.plot(
                ctx,
                plot_type=plot_type,
                x=x,
                y=y,
                group_by=group_by,
                aggregate=aggregate,
                filter=filter,
                title=title,
            )
            for ctx in contexts
        ]

    # ──────────────────────────────────────────────────────────────────────────
    # Utility Methods
    # ──────────────────────────────────────────────────────────────────────────

    def clear(self) -> None:
        """Clear all in-memory tables and reset the simulated manager."""
        self._tables.clear()
        self._schemas.clear()
        self._descriptions.clear()
        self._embeddings.clear()
        logger.debug("SimulatedDataManager cleared")
