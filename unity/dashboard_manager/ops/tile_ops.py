"""Tile operations for DashboardManager.

Helper functions for building tile records and validating data bindings.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING, List, Optional, Sequence, Union

from unity.dashboard_manager.types.tile import (
    FilterBinding,
    JoinBinding,
    JoinReduceBinding,
    ReduceBinding,
    TileRecordRow,
)

if TYPE_CHECKING:
    from unity.data_manager.base import BaseDataManager


AnyBinding = Union[FilterBinding, ReduceBinding, JoinBinding, JoinReduceBinding]


def _contexts_for_binding(binding: AnyBinding) -> List[str]:
    """Extract context paths from a binding (single context or join tables)."""
    if isinstance(binding, (FilterBinding, ReduceBinding)):
        return [binding.context]
    return list(binding.tables)


def build_tile_record_row(
    token: str,
    html: str,
    title: str,
    description: Optional[str] = None,
    data_bindings: Optional[Sequence[AnyBinding]] = None,
) -> TileRecordRow:
    """Build a TileRecordRow ready for insertion into the Unify context."""
    has_bindings = bool(data_bindings)
    binding_contexts: Optional[str] = None
    if data_bindings:
        all_ctxs: List[str] = []
        for b in data_bindings:
            all_ctxs.extend(_contexts_for_binding(b))
        binding_contexts = ",".join(dict.fromkeys(all_ctxs))
    now = datetime.now(timezone.utc).isoformat()

    return TileRecordRow(
        token=token,
        title=title,
        description=description,
        html_content=html,
        has_data_bindings=has_bindings,
        data_binding_contexts=binding_contexts,
        created_at=now,
        updated_at=now,
    )


def validate_data_bindings(
    data_bindings: Optional[List[AnyBinding]],
) -> Optional[List[AnyBinding]]:
    """Validate and clean data bindings. Returns typed list or None."""
    if not data_bindings:
        return None
    cleaned: List[AnyBinding] = []
    for b in data_bindings:
        if isinstance(b, (FilterBinding, ReduceBinding)):
            ctx = b.context.strip()
            if not ctx:
                continue
        elif isinstance(b, (JoinBinding, JoinReduceBinding)):
            if not b.tables or not any(t.strip() for t in b.tables):
                continue
        else:
            continue
        cleaned.append(b)
    return cleaned or None


def _filter_binding_has_query_params(binding: FilterBinding) -> bool:
    """True if a filter binding declares params beyond just ``context``."""
    return (
        binding.filter is not None
        or binding.columns is not None
        or binding.exclude_columns is not None
        or binding.order_by is not None
        or binding.descending is not False
        or binding.limit is not None
        or binding.offset is not None
        or binding.group_by is not None
    )


def verify_data_bindings(
    data_bindings: List[AnyBinding],
    dm: BaseDataManager,
) -> None:
    """Dry-run each binding through the corresponding DataManager method.

    Dispatches to the appropriate DM primitive per binding type:

    - ``FilterBinding``  -> ``dm.filter(limit=5)``
    - ``ReduceBinding``  -> ``dm.reduce(...)``
    - ``JoinBinding``    -> ``dm.filter_join(result_limit=5)``
    - ``JoinReduceBinding`` -> ``dm.reduce_join(...)``

    Called inside ``create_tile`` before the tile is stored.  If any
    binding references an invalid context, column, or expression,
    this raises and ``create_tile`` returns ``TileResult(error=...)``.
    """
    for binding in data_bindings:
        if isinstance(binding, FilterBinding):
            if not _filter_binding_has_query_params(binding):
                continue
            dm.filter(
                binding.context,
                filter=binding.filter,
                columns=binding.columns,
                exclude_columns=binding.exclude_columns,
                order_by=binding.order_by,
                descending=binding.descending,
                limit=5,
            )

        elif isinstance(binding, ReduceBinding):
            dm.reduce(
                binding.context,
                metric=binding.metric,
                columns=binding.columns,
                filter=binding.filter,
                group_by=binding.group_by,
            )

        elif isinstance(binding, JoinBinding):
            dm.filter_join(
                tables=binding.tables,
                join_expr=binding.join_expr,
                select=binding.select,
                mode=binding.mode,
                left_where=binding.left_where,
                right_where=binding.right_where,
                result_where=binding.result_where,
                result_limit=5,
                result_offset=0,
            )

        elif isinstance(binding, JoinReduceBinding):
            dm.reduce_join(
                tables=binding.tables,
                join_expr=binding.join_expr,
                select=binding.select,
                metric=binding.metric,
                columns=binding.columns,
                mode=binding.mode,
                left_where=binding.left_where,
                right_where=binding.right_where,
                result_where=binding.result_where,
                group_by=binding.group_by,
            )
