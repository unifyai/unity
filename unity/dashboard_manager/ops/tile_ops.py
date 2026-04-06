"""Tile operations for DashboardManager.

Helper functions for building tile records and validating data bindings.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Optional

from unity.dashboard_manager.types.tile import DataBinding, TileRecordRow


def build_tile_record_row(
    token: str,
    html: str,
    title: str,
    description: Optional[str] = None,
    data_bindings: Optional[List[DataBinding]] = None,
) -> TileRecordRow:
    """Build a TileRecordRow ready for insertion into the Unify context."""
    has_bindings = bool(data_bindings)
    binding_contexts = (
        ",".join(b.context for b in data_bindings) if data_bindings else None
    )
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
    data_bindings: Optional[List[DataBinding]],
) -> Optional[List[DataBinding]]:
    """Validate data bindings if provided. Returns cleaned list or None."""
    if not data_bindings:
        return None
    cleaned = []
    for b in data_bindings:
        ctx = b.context.strip()
        if not ctx:
            continue
        cleaned.append(
            DataBinding(
                context=ctx,
                alias=b.alias,
                filter=b.filter,
                columns=b.columns,
                exclude_columns=b.exclude_columns,
                order_by=b.order_by,
                descending=b.descending,
            ),
        )
    return cleaned or None


def _binding_has_query_params(binding: DataBinding) -> bool:
    """True if the binding declares any query params beyond just ``context``."""
    return (
        binding.filter is not None
        or binding.columns is not None
        or binding.exclude_columns is not None
        or binding.order_by is not None
        or binding.descending is not False
    )


def verify_data_bindings(
    data_bindings: List[DataBinding],
    dm,
) -> None:
    """Dry-run each binding's query params through DataManager.filter().

    Only bindings that declare at least one query param (filter, columns,
    exclude_columns, order_by, or descending=True) are validated.
    Bindings with only ``context`` pass through without a backend call,
    since the context might be created before the tile is rendered.

    Called inside ``create_tile`` before the tile is stored.  If any
    binding references an invalid context, column, or filter expression,
    this raises and ``create_tile`` returns ``TileResult(error=...)``.
    """
    for binding in data_bindings:
        if not _binding_has_query_params(binding):
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
