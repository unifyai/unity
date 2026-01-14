"""
Mutation operations for DataManager.

Implementation functions for insert_rows, update_rows, delete_rows.
These are called by DataManager methods and should not be used directly.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import unify

from unity.common.filter_utils import normalize_filter_expr
from unity.common.log_utils import log as unify_log, create_logs as unify_create_logs

logger = logging.getLogger(__name__)


def insert_rows_impl(
    context: str,
    rows: List[Dict[str, Any]],
    *,
    dedupe_key: Optional[str] = None,
    add_to_all_context: bool = False,
) -> int:
    """
    Implementation of insert_rows operation.

    Inserts rows into a context with optional deduplication.

    Parameters
    ----------
    context : str
        Fully-qualified Unify context path.
    rows : list[dict[str, Any]]
        List of row dictionaries to insert.
    dedupe_key : str | None
        If provided, existing rows with matching key values are replaced.
    add_to_all_context : bool, default False
        Whether to also add to aggregation contexts.

    Returns
    -------
    int
        Number of rows inserted.

    Raises
    ------
    Exception
        If insertion fails.
    """
    if not rows:
        return 0

    logger.debug(
        "Inserting %d rows into %s (dedupe_key=%s)",
        len(rows),
        context,
        dedupe_key,
    )

    if dedupe_key:
        # Upsert mode: check and replace existing rows
        inserted = 0
        for row in rows:
            key_val = row.get(dedupe_key)
            if key_val is not None:
                # Check if row with this key exists
                existing = unify.get_logs(
                    context=context,
                    filter=f"{dedupe_key} == {key_val!r}",
                    limit=1,
                )
                if existing:
                    # Delete existing row
                    log_id = existing[0].id if hasattr(existing[0], "id") else None
                    if log_id:
                        unify.delete_logs(context=context, logs=[log_id])

            # Insert the row
            unify_log(
                context=context,
                add_to_all_context=add_to_all_context,
                **row,
            )
            inserted += 1

        return inserted
    else:
        # Bulk insert mode
        result = unify_create_logs(
            context=context,
            entries=rows,
            add_to_all_context=add_to_all_context,
        )
        if isinstance(result, dict):
            return len(result.get("log_event_ids", []))
        elif isinstance(result, list):
            return len(result)
        return len(rows)


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

        # Merge updates
        new_entries = {**existing, **updates}

        # Insert updated row
        unify_log(context=context, **new_entries)
        updated += 1

    return updated


def delete_rows_impl(
    context: str,
    *,
    filter: str,
) -> int:
    """
    Implementation of delete_rows operation.

    Deletes rows matching a filter expression.

    Parameters
    ----------
    context : str
        Fully-qualified Unify context path.
    filter : str
        Filter expression to select rows to delete. Required.

    Returns
    -------
    int
        Number of rows deleted.

    Raises
    ------
    Exception
        If deletion fails.
    """
    logger.info("Deleting rows from %s where %s", context, filter)

    filter_expr = normalize_filter_expr(filter)

    # Get matching rows
    logs = unify.get_logs(context=context, filter=filter_expr)

    if not logs:
        return 0

    # Collect log IDs
    log_ids = []
    for log in logs:
        log_id = log.id if hasattr(log, "id") else None
        if log_id:
            log_ids.append(log_id)

    if log_ids:
        unify.delete_logs(context=context, logs=log_ids)

    return len(log_ids)
