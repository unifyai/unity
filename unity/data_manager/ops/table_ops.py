"""
Table management operations for DataManager.

Implementation functions for create_table, describe_table, list_tables, delete_table.
These are called by DataManager methods and should not be used directly.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import unify

if TYPE_CHECKING:
    from unity.data_manager.types.table import TableDescription

logger = logging.getLogger(__name__)


def create_table_impl(
    context: str,
    *,
    description: Optional[str] = None,
    fields: Optional[Dict[str, Any]] = None,
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
    fields : dict[str, Any] | None
        Mapping of field names to Unify types (simple ``{name: type_str}``)
        or richer payloads (``{name: {"type": ..., "description": ...}}``).
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


def list_tables_impl(
    *,
    prefix: Optional[str] = None,
    include_column_info: bool = True,
) -> Union[List[str], Dict[str, Any]]:
    """
    Implementation of list_tables operation.

    Lists all table contexts, optionally filtered by prefix.
    Returns either table names only or table names with metadata.

    Parameters
    ----------
    prefix : str | None
        Context prefix to filter by.
    include_column_info : bool, default True
        If True, returns dict mapping table paths to metadata.
        If False, returns just a sorted list of table names.

    Returns
    -------
    list[str] | dict[str, Any]
        List of context paths (if include_column_info=False) or
        mapping of context_path -> context_info dict.

    Raises
    ------
    Exception
        If context listing fails.
    """
    logger.debug(
        "Listing tables with prefix: %s, include_column_info: %s",
        prefix,
        include_column_info,
    )

    raw_contexts = unify.get_contexts(prefix=prefix) if prefix else unify.get_contexts()

    if not raw_contexts:
        return {} if include_column_info else []

    # unify.get_contexts returns either:
    # - A list of dicts with 'name' and other fields (newer SDK)
    # - A dict mapping context_path -> context_info (older SDK)
    if isinstance(raw_contexts, list):
        all_contexts = {ctx.get("name"): ctx for ctx in raw_contexts if ctx.get("name")}
    elif isinstance(raw_contexts, dict):
        # Check if it's {name: info_dict} or {name: string_description}
        sample_val = next(iter(raw_contexts.values()), None) if raw_contexts else None
        if isinstance(sample_val, dict):
            all_contexts = dict(raw_contexts)
        else:
            # It's {name: description_string}, need to convert
            all_contexts = {k: {"description": v} for k, v in raw_contexts.items()}
    else:
        all_contexts = {}

    # Filter by prefix if needed (unify.get_contexts may not filter perfectly)
    if prefix:
        all_contexts = {k: v for k, v in all_contexts.items() if k.startswith(prefix)}

    if include_column_info:
        return all_contexts
    else:
        return sorted(all_contexts.keys())


def delete_table_impl(context: str, *, dangerous_ok: bool = False) -> None:
    """
    Implementation of delete_table operation.

    Deletes a table context and all its data.

    Parameters
    ----------
    context : str
        Fully-qualified Unify context path.
    dangerous_ok : bool
        Safety flag that must be True to confirm deletion.

    Raises
    ------
    ValueError
        If dangerous_ok is False.
    Exception
        If context deletion fails.
    """
    if not dangerous_ok:
        raise ValueError(
            "delete_table requires dangerous_ok=True to confirm deletion. "
            "This is a safety guard to prevent accidental data loss.",
        )
    logger.info("Deleting table context: %s", context)
    unify.delete_context(context)


def get_columns_impl(table: str) -> Dict[str, Any]:
    """
    Implementation of get_columns operation.

    Retrieves raw column definitions for a table.

    Parameters
    ----------
    table : str
        Fully-qualified Unify context path.

    Returns
    -------
    dict[str, Any]
        Mapping of column_name -> column_info dict.
        Returns empty dict if table has no columns.

    Raises
    ------
    Exception
        If table does not exist or access fails.
    """
    logger.debug("Getting columns for table: %s", table)
    try:
        columns = unify.get_fields(context=table)
        return dict(columns) if columns else {}
    except Exception as e:
        logger.warning("Failed to get columns for %s: %s", table, e)
        raise


def get_table_impl(context: str) -> Dict[str, Any]:
    """
    Implementation of get_table operation.

    Retrieves context/table metadata without full schema.

    Parameters
    ----------
    context : str
        Fully-qualified Unify context path.

    Returns
    -------
    dict[str, Any]
        Context metadata including unique_keys, auto_counting, description.

    Raises
    ------
    Exception
        If table does not exist or access fails.
    """
    logger.debug("Getting table metadata for: %s", context)
    try:
        ctx_info = unify.get_context(context)
        return dict(ctx_info) if isinstance(ctx_info, dict) else {}
    except Exception as e:
        logger.warning("Failed to get table %s: %s", context, e)
        raise


def rename_table_impl(old_context: str, new_context: str) -> Dict[str, str]:
    """
    Implementation of rename_table operation.

    Renames a table context to a new name.

    Parameters
    ----------
    old_context : str
        Current fully-qualified Unify context path.
    new_context : str
        New fully-qualified Unify context path.

    Returns
    -------
    dict[str, str]
        Backend response containing the operation result.

    Raises
    ------
    Exception
        If rename operation fails.
    """
    logger.info("Renaming table context: %s -> %s", old_context, new_context)
    return unify.rename_context(old_context, new_context)


def create_column_impl(
    context: str,
    *,
    column_name: str,
    column_type: str,
    mutable: bool = True,
    backfill_logs: bool = False,
) -> Dict[str, str]:
    """
    Implementation of create_column operation.

    Creates a new column in a table.

    Parameters
    ----------
    context : str
        Fully-qualified Unify context path.
    column_name : str
        Name for the new column.
    column_type : str
        Unify data type for the column.
    mutable : bool
        Whether values can be updated.
    backfill_logs : bool
        Whether to backfill existing rows.

    Returns
    -------
    dict[str, str]
        Backend response confirming column creation.

    Raises
    ------
    ValueError
        If column_name is reserved ('id').
    Exception
        If column creation fails.
    """
    if column_name == "id":
        raise ValueError("Cannot create a column with reserved name 'id'.")

    logger.debug(
        "Creating column %s (type=%s) in %s",
        column_name,
        column_type,
        context,
    )
    return unify.create_fields(
        context=context,
        fields={column_name: {"type": column_type, "mutable": mutable}},
        backfill_logs=backfill_logs,
    )


def delete_column_impl(
    context: str,
    *,
    column_name: str,
) -> Dict[str, str]:
    """
    Implementation of delete_column operation.

    Removes a column from a table.

    Parameters
    ----------
    context : str
        Fully-qualified Unify context path.
    column_name : str
        Name of the column to delete.

    Returns
    -------
    dict[str, str]
        Backend response confirming column deletion.

    Raises
    ------
    Exception
        If column deletion fails.
    """
    logger.info("Deleting column %s from %s", column_name, context)
    return unify.delete_fields(fields=[column_name], context=context)


def rename_column_impl(
    context: str,
    *,
    old_name: str,
    new_name: str,
) -> Dict[str, str]:
    """
    Implementation of rename_column operation.

    Renames a column in a table.

    Parameters
    ----------
    context : str
        Fully-qualified Unify context path.
    old_name : str
        Current name of the column.
    new_name : str
        New name for the column.

    Returns
    -------
    dict[str, str]
        Backend response confirming the rename.

    Raises
    ------
    ValueError
        If new_name is reserved ('id').
    Exception
        If rename operation fails.
    """
    if old_name == new_name:
        return {
            "info": "no-op: old and new names are identical",
            "old_name": old_name,
            "new_name": new_name,
        }
    if new_name == "id":
        raise ValueError("Cannot rename a column to reserved name 'id'.")

    logger.debug("Renaming column %s -> %s in %s", old_name, new_name, context)
    return unify.rename_field(name=old_name, new_name=new_name, context=context)


def create_derived_column_impl(
    context: str,
    *,
    column_name: str,
    equation: str,
) -> Dict[str, str]:
    """
    Implementation of create_derived_column operation.

    Creates a computed column based on an equation.

    Parameters
    ----------
    context : str
        Fully-qualified Unify context path.
    column_name : str
        Name for the new derived column.
    equation : str
        Python expression evaluated per-row.
        Column references should use {column_name} syntax.

    Returns
    -------
    dict[str, str]
        Backend response confirming column creation.

    Raises
    ------
    Exception
        If derived column creation fails.

    Notes
    -----
    The equation is transformed to use Unify's internal syntax
    by prefixing column references with 'lg:'.
    """
    logger.debug(
        "Creating derived column %s in %s with equation: %s",
        column_name,
        context,
        equation,
    )
    # Transform equation to Unify's internal format
    transformed_equation = equation.replace("{", "{lg:")
    return unify.create_derived_logs(
        context=context,
        key=column_name,
        equation=transformed_equation,
        referenced_logs={"lg": {"context": context}},
    )
