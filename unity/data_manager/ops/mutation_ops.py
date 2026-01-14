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
) -> int:
    """
    Implementation of insert_rows operation.

    Inserts rows into a table with optional deduplication.
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
                try:
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
                except Exception as e:
                    logger.debug("Dedupe check failed: %s", e)

            # Insert the row
            try:
                unify_log(context=context, **row)
                inserted += 1
            except Exception as e:
                logger.warning("Row insert failed: %s", e)

        return inserted
    else:
        # Bulk insert mode
        try:
            result = unify_create_logs(context=context, entries=rows)
            if isinstance(result, dict):
                return len(result.get("log_event_ids", []))
            elif isinstance(result, list):
                return len(result)
            return len(rows)
        except Exception as e:
            logger.warning("Bulk insert failed: %s", e)
            return 0


def update_rows_impl(
    context: str,
    updates: Dict[str, Any],
    *,
    filter: str,
) -> int:
    """
    Implementation of update_rows operation.

    Updates rows matching a filter.
    """
    logger.debug("Updating rows in %s where %s", context, filter)

    filter_expr = normalize_filter_expr(filter)

    # Get matching rows
    try:
        logs = unify.get_logs(context=context, filter=filter_expr)
    except Exception as e:
        logger.warning("Update query failed: %s", e)
        return 0

    if not logs:
        return 0

    # Update each matching row (delete + insert pattern)
    updated = 0
    for log in logs:
        try:
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
        except Exception as e:
            logger.warning("Row update failed: %s", e)

    return updated


def delete_rows_impl(
    context: str,
    *,
    filter: str,
    dangerous_ok: bool = False,
) -> int:
    """
    Implementation of delete_rows operation.

    Deletes rows matching a filter.
    """
    if not dangerous_ok:
        raise ValueError(
            "delete_rows is a destructive operation. "
            "Set dangerous_ok=True to confirm.",
        )

    logger.info("Deleting rows from %s where %s", context, filter)

    filter_expr = normalize_filter_expr(filter)

    # Get matching rows
    try:
        logs = unify.get_logs(context=context, filter=filter_expr)
    except Exception as e:
        logger.warning("Delete query failed: %s", e)
        return 0

    if not logs:
        return 0

    # Delete matching rows
    log_ids = []
    for log in logs:
        log_id = log.id if hasattr(log, "id") else None
        if log_id:
            log_ids.append(log_id)

    if log_ids:
        try:
            unify.delete_logs(context=context, logs=log_ids)
        except Exception as e:
            logger.warning("Row deletion failed: %s", e)
            return 0

    return len(log_ids)
