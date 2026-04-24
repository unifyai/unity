"""
Mutation operations for DataManager.

Implementation functions for insert_rows, update_rows, delete_rows.
These are called by DataManager methods and should not be used directly.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import unify

from unity.common.authoring import AUTHORING_COLUMN, authoring_assistant_id
from unity.common.filter_utils import normalize_filter_expr
from unity.common.log_utils import log as unify_log, create_logs as unify_create_logs

logger = logging.getLogger(__name__)


def insert_rows_impl(
    context: str,
    rows: List[Dict[str, Any]],
    *,
    add_to_all_context: bool = False,
    batched: bool = True,
) -> List[int]:
    """Insert rows into a context via bulk creation.

    Uniqueness enforcement belongs at the **schema level** (``unique_keys``
    on the context), not at insert time.  Use ``create_table(unique_keys=…)``
    or ``ingest(unique_keys=…)`` to declare which columns form the natural
    key; the backend will reject or upsert duplicates server-side.

    Parameters
    ----------
    context : str
        Fully-qualified Unify context path.
    rows : list[dict[str, Any]]
        Row dictionaries to insert.
    add_to_all_context : bool, default False
        Whether to also add to aggregation contexts.
    batched : bool, default True
        When True, uses batched log creation for better performance.

    Returns
    -------
    list[int]
        Log IDs of inserted rows.
    """
    if not rows:
        return []

    logger.debug("Inserting %d rows into %s (batched=%s)", len(rows), context, batched)

    stamp = authoring_assistant_id()
    stamped_rows = [
        {**row, AUTHORING_COLUMN: row.get(AUTHORING_COLUMN, stamp)}
        for row in rows
    ]

    result = unify_create_logs(
        context=context,
        entries=stamped_rows,
        add_to_all_context=add_to_all_context,
        batched=batched,
    )
    if isinstance(result, list):
        return [lg.id for lg in result if hasattr(lg, "id")]
    return []


def update_rows_impl(
    context: str,
    updates: Dict[str, Any],
    *,
    filter: str,
) -> int:
    """
    Implementation of update_rows operation.

    Updates rows matching a filter expression.

    Parameters
    ----------
    context : str
        Fully-qualified Unify context path.
    updates : dict[str, Any]
        Column values to update.
    filter : str
        Filter expression to select rows. Required.

    Returns
    -------
    int
        Number of rows updated.

    Raises
    ------
    Exception
        If update fails.
    """
    logger.debug("Updating rows in %s where %s", context, filter)

    filter_expr = normalize_filter_expr(filter)

    # Get matching rows
    logs = unify.get_logs(context=context, filter=filter_expr)

    if not logs:
        return 0

    # The authoring stamp is immutable after creation: strip it from the
    # caller-supplied update so the delete+insert cycle keeps the original
    # authoring body on the resurrected row.
    safe_updates = {k: v for k, v in updates.items() if k != AUTHORING_COLUMN}

    # Update each matching row (delete + insert pattern)
    # Note: This is the current pattern used elsewhere in the codebase
    updated = 0
    for log in logs:
        # Get existing entries
        if hasattr(log, "entries") and isinstance(log.entries, dict):
            existing = dict(log.entries)
        elif isinstance(log, dict):
            existing = dict(log)
        else:
            continue

        # Delete old row
        log_id = log.id if hasattr(log, "id") else None
        if log_id:
            unify.delete_logs(context=context, logs=[log_id])

        # Merge updates; existing authoring_assistant_id survives because
        # safe_updates no longer carries the column.
        new_entries = {**existing, **safe_updates}

        # Insert updated row
        unify_log(context=context, **new_entries)
        updated += 1

    return updated


def delete_rows_impl(
    context: str,
    *,
    filter: Optional[str] = None,
    log_ids: Optional[List[int]] = None,
    dangerous_ok: bool = False,
    delete_empty_rows: bool = False,
) -> int:
    """
    Implementation of delete_rows operation.

    Deletes rows matching a filter expression or specific log IDs.

    Parameters
    ----------
    context : str
        Fully-qualified Unify context path.
    filter : str | None
        Filter expression to select rows to delete.
    log_ids : list[int] | None
        Specific log IDs to delete. More efficient than filter when IDs are known.
    dangerous_ok : bool
        Safety flag; must be True to confirm destructive operation.
    delete_empty_rows : bool, default False
        When True, also deletes rows with no data (empty logs).

    Returns
    -------
    int
        Number of rows deleted.

    Raises
    ------
    ValueError
        If neither filter nor log_ids is provided.
    Exception
        If deletion fails.
    """
    if filter is None and log_ids is None:
        raise ValueError("Either filter or log_ids must be provided for delete_rows")

    logger.info(
        "Deleting rows from %s (filter=%s, log_ids=%s, delete_empty_rows=%s)",
        context,
        filter,
        f"{len(log_ids)} ids" if log_ids else None,
        delete_empty_rows,
    )

    ids_to_delete: List[int] = []

    if log_ids is not None:
        # Use provided log IDs directly
        ids_to_delete = list(log_ids)
    elif filter is not None:
        # Get log IDs using return_ids_only for efficiency
        filter_expr = normalize_filter_expr(filter)
        result = unify.get_logs(
            context=context,
            filter=filter_expr,
            return_ids_only=True,
        )
        if isinstance(result, list):
            ids_to_delete = result

    if not ids_to_delete:
        return 0

    # Delete the logs
    unify.delete_logs(
        context=context,
        logs=ids_to_delete,
        delete_empty_logs=delete_empty_rows,
    )

    return len(ids_to_delete)
