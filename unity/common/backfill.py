"""
Utility for backfilling _assistant field on existing logs.

When the private field injection was added, existing logs won't have
_assistant, _assistant_id, or _user_id fields. This module provides
utilities to backfill these fields in bulk.

Usage
-----
    from unity.common.backfill import (
        backfill_assistant_field,
        backfill_all_contexts_for_assistant,
    )

    # Backfill a single context
    result = backfill_assistant_field("JohnDoe/Contacts", "JohnDoe")

    # Backfill all contexts for an assistant
    results = backfill_all_contexts_for_assistant("JohnDoe")
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import unify


def backfill_assistant_field(
    context: str,
    assistant_name: str,
    *,
    batch_size: int = 100,
    filter: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Backfill _assistant field for existing logs in a context.

    Finds all logs where _assistant is missing or None and updates them with
    the provided assistant_name. Processes in batches to handle large contexts.

    Note: Uses client-side filtering because `_assistant is None` only matches
    logs where the field exists but is NULL, not logs where the field was
    never set.

    Parameters
    ----------
    context : str
        The context to backfill (e.g., "JohnDoe/Contacts")
    assistant_name : str
        The assistant name to set for _assistant field
    batch_size : int, default 100
        Number of logs to fetch per batch
    filter : str, optional
        Additional filter expression to narrow scope

    Returns
    -------
    Dict[str, Any]
        Summary with total_updated count and context
    """
    total_updated = 0
    offset = 0

    while True:
        # Fetch logs with entries to check _assistant client-side
        logs = unify.get_logs(
            context=context,
            filter=filter,  # Only apply user's additional filter
            offset=offset,
            limit=batch_size,
            return_ids_only=False,  # Need entries to check _assistant
        )

        if not logs:
            break

        # Filter client-side for logs missing _assistant or with None value
        logs_to_update = [lg.id for lg in logs if lg.entries.get("_assistant") is None]

        if logs_to_update:
            # Batch update
            unify.update_logs(
                logs=logs_to_update,
                context=context,
                entries={"_assistant": assistant_name},
                overwrite=False,  # Don't overwrite other fields
            )
            total_updated += len(logs_to_update)

        # Move to next batch
        offset += len(logs)

        # If we got fewer than batch_size, we're done
        if len(logs) < batch_size:
            break

    return {"total_updated": total_updated, "context": context}


def backfill_all_contexts_for_assistant(
    assistant_name: str,
    *,
    contexts: Optional[List[str]] = None,
    batch_size: int = 100,
) -> Dict[str, Dict[str, Any]]:
    """
    Backfill _assistant for all contexts belonging to an assistant.

    If contexts is None, discovers them by prefix "{assistant_name}/".

    Parameters
    ----------
    assistant_name : str
        The assistant name (used as context prefix and field value)
    contexts : List[str], optional
        Explicit list of contexts to backfill. If None, discovers via prefix.
    batch_size : int, default 100
        Number of logs to fetch per batch per context

    Returns
    -------
    Dict[str, Dict[str, Any]]
        Mapping of context -> result dict (total_updated or error)
    """
    if contexts is None:
        all_contexts = unify.get_contexts(prefix=f"{assistant_name}/")
        contexts = list(all_contexts.keys())

    results: Dict[str, Dict[str, Any]] = {}
    for ctx in contexts:
        try:
            result = backfill_assistant_field(
                ctx,
                assistant_name,
                batch_size=batch_size,
            )
            results[ctx] = result
        except Exception as e:
            results[ctx] = {"error": str(e), "context": ctx}

    return results
