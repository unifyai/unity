"""
Table management operations for DataManager.

Implementation functions for create_table, describe_table, list_tables, delete_table.
These are called by DataManager methods and should not be used directly.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Dict, List, Optional

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

    Parameters
    ----------
    context : str
        Fully-qualified Unify context path.
    description : str | None
        Human-readable description of the table purpose.
    fields : dict[str, str] | None
        Mapping of field names to Unify types.
    unique_keys : dict[str, str] | None
        Mapping of unique key columns to their types.
    auto_counting : dict[str, str | None] | None
        Columns with auto-increment behavior.

    Returns
    -------
    str
        The context path that was created.

    Raises
    ------
    Exception
        If context creation fails.
    """
    logger.debug("Creating table context: %s", context)

    unify.create_context(
        context,
        description=description,
        unique_keys=unique_keys,
        auto_counting=auto_counting,
    )

    if fields:
        unify.create_fields(fields, context=context)

    return context


def describe_table_impl(context: str) -> "TableDescription":
    """
    Implementation of describe_table operation.

    Fetches schema and metadata for a table.

    Parameters
    ----------
    context : str
        Fully-qualified Unify context path.

    Returns
    -------
    TableDescription
        Schema and metadata for the table.

    Notes
    -----
    - row_count is NOT included (expensive to compute).
    - Private columns (starting with "_") are excluded from the schema.
    """
    from unity.data_manager.types.table import (
        TableDescription,
        TableSchema,
        ColumnInfo,
    )

    logger.debug("Describing table: %s", context)

    # Get context info
    ctx_info = unify.get_context(context)

    # Get fields/columns
    columns_raw = unify.get_fields(context=context) or {}

    # Detect embedding columns (pattern: _<name>_emb)
    embedding_cols = [
        c for c in columns_raw.keys() if c.startswith("_") and c.endswith("_emb")
    ]

    # Build column info list (excluding private columns)
    columns: List[ColumnInfo] = []
    for name, info in columns_raw.items():
        if name.startswith("_"):
            # Skip internal/private columns (like _id, _*_emb, etc.)
            continue

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
        table_schema=schema,
        has_embeddings=len(embedding_cols) > 0,
        embedding_columns=embedding_cols,
    )


def list_tables_impl(*, prefix: Optional[str] = None) -> List[str]:
    """
    Implementation of list_tables operation.

    Lists all table contexts, optionally filtered by prefix.

    Parameters
    ----------
    prefix : str | None
        Context prefix to filter by.

    Returns
    -------
    list[str]
        Sorted list of context paths.

    Raises
    ------
    Exception
        If context listing fails.
    """
    logger.debug("Listing tables with prefix: %s", prefix)

    all_contexts = unify.get_contexts() or {}
    context_names = list(all_contexts.keys())

    if prefix:
        context_names = [c for c in context_names if c.startswith(prefix)]

    return sorted(context_names)


def delete_table_impl(context: str) -> None:
    """
    Implementation of delete_table operation.

    Deletes a table context and all its data.

    Parameters
    ----------
    context : str
        Fully-qualified Unify context path.

    Raises
    ------
    Exception
        If context deletion fails.

    Notes
    -----
    The dangerous_ok check is performed at the DataManager level,
    not in this implementation function.
    """
    logger.info("Deleting table context: %s", context)
    unify.delete_context(context)
