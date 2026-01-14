"""
Table management operations for DataManager.

Implementation functions for create_table, describe_table, list_tables, delete_table.
These are called by DataManager methods and should not be used directly.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import unify

if TYPE_CHECKING:
    from unity.data_manager.types.table import TableDescription

logger = logging.getLogger(__name__)


def create_table_impl(
    context: str,
    *,
    description: Optional[str] = None,
    fields: Optional[Dict[str, str]] = None,
    unique_keys: Optional[Dict[str, str]] = None,
    auto_counting: Optional[Dict[str, Optional[str]]] = None,
) -> str:
    """
    Implementation of create_table operation.

    Creates a Unify context with optional schema definition.
    """
    logger.debug("Creating table context: %s", context)

    try:
        unify.create_context(
            context,
            description=description,
            unique_keys=unique_keys,
            auto_counting=auto_counting,
        )
    except Exception as e:
        # Context may already exist - log and continue
        logger.debug("Context creation note: %s", e)

    if fields:
        try:
            unify.create_fields(fields, context=context)
        except Exception as e:
            logger.debug("Fields creation note: %s", e)

    return context


def describe_table_impl(context: str) -> "TableDescription":
    """
    Implementation of describe_table operation.

    Fetches schema, row count, and metadata for a table.
    """
    from unity.data_manager.types.table import (
        TableDescription,
        TableSchema,
        ColumnInfo,
    )

    logger.debug("Describing table: %s", context)

    # Get context info
    ctx_info: Any = None
    try:
        ctx_info = unify.get_context(context)
    except Exception as e:
        logger.debug("Could not get context info: %s", e)

    # Get fields/columns
    columns_raw: Dict[str, Any] = {}
    try:
        columns_raw = unify.get_fields(context=context) or {}
    except Exception as e:
        logger.debug("Could not get fields: %s", e)

    # Get row count via metric
    row_count = 0
    try:
        count_result = unify.get_logs_metric(
            metric="count",
            key="id",
            context=context,
        )
        if isinstance(count_result, (int, float)):
            row_count = int(count_result)
    except Exception as e:
        logger.debug("Could not get row count: %s", e)

    # Detect embedding columns (pattern: _<name>_emb)
    embedding_cols = [
        c for c in columns_raw.keys() if c.startswith("_") and c.endswith("_emb")
    ]

    # Build column info list
    columns: List[ColumnInfo] = []
    for name, info in columns_raw.items():
        if name.startswith("_"):
            # Skip internal/private columns
            continue

        # Check if this column has an embedding sibling
        searchable = f"_{name}_emb" in columns_raw

        # Extract type info
        dtype = "unknown"
        col_desc = None
        if isinstance(info, dict):
            dtype = info.get("data_type", "unknown")
            col_desc = info.get("description")
        elif isinstance(info, str):
            dtype = info

        columns.append(
            ColumnInfo(
                name=name,
                dtype=dtype,
                searchable=searchable,
                description=col_desc,
            ),
        )

    # Extract unique_keys and auto_counting from context info
    unique_keys = None
    auto_counting = None
    description = None
    if isinstance(ctx_info, dict):
        unique_keys = ctx_info.get("unique_keys")
        auto_counting = ctx_info.get("auto_counting")
        description = ctx_info.get("description")

    schema = TableSchema(
        columns=columns,
        unique_keys=unique_keys,
        auto_counting=auto_counting,
    )

    return TableDescription(
        context=context,
        description=description,
        schema=schema,
        row_count=row_count,
        has_embeddings=len(embedding_cols) > 0,
        embedding_columns=embedding_cols,
    )


def list_tables_impl(*, prefix: Optional[str] = None) -> List[str]:
    """
    Implementation of list_tables operation.

    Lists all table contexts, optionally filtered by prefix.
    """
    logger.debug("Listing tables with prefix: %s", prefix)

    try:
        all_contexts = unify.get_contexts() or {}
        context_names = list(all_contexts.keys())
    except Exception as e:
        logger.warning("Could not list contexts: %s", e)
        return []

    if prefix:
        context_names = [c for c in context_names if c.startswith(prefix)]

    return sorted(context_names)


def delete_table_impl(
    context: str,
    *,
    dangerous_ok: bool = False,
) -> None:
    """
    Implementation of delete_table operation.

    Deletes a table context and all its data.
    """
    if not dangerous_ok:
        raise ValueError(
            "delete_table is a destructive operation. "
            "Set dangerous_ok=True to confirm.",
        )

    logger.info("Deleting table context: %s", context)
    unify.delete_context(context)
