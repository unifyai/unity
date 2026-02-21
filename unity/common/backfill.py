"""
Utility for backfilling private fields on existing logs.

When the private field injection was added, existing logs won't have
_user, _user_id, _assistant, or _assistant_id fields. This module provides
utilities to backfill these fields in bulk.

Usage
-----
    from unity.common.backfill import (
        backfill_private_fields,
        backfill_all_contexts_for_user_assistant,
    )

    # Backfill a single context
    result = backfill_private_fields(
        "42/7/Contacts",
        user_context="42",
        assistant_context="7",
    )

    # Backfill all contexts for a user/assistant combination
    results = backfill_all_contexts_for_user_assistant(
        user_context="42",
        assistant_context="7",
    )
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import unify


def backfill_private_fields(
    context: str,
    *,
    user_context: Optional[str] = None,
    assistant_context: Optional[str] = None,
    batch_size: int = 100,
    filter: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Backfill _user and/or _assistant fields for existing logs in a context.

    Finds all logs where _user/_assistant is missing or None and updates them
    with the provided values. Processes in batches to handle large contexts.

    Note: Uses client-side filtering because `_field is None` only matches
    logs where the field exists but is NULL, not logs where the field was
    never set.

    Parameters
    ----------
    context : str
        The context to backfill (e.g., "42/7/Contacts")
    user_context : str, optional
        The user ID to set for _user field
    assistant_context : str, optional
        The assistant ID to set for _assistant field
    batch_size : int, default 100
        Number of logs to fetch per batch
    filter : str, optional
        Additional filter expression to narrow scope

    Returns
    -------
    Dict[str, Any]
        Summary with total_updated count and context
    """
    if user_context is None and assistant_context is None:
        return {
            "total_updated": 0,
            "context": context,
            "error": "No fields to backfill",
        }

    total_updated = 0
    offset = 0

    while True:
        # Fetch logs with entries to check fields client-side
        logs = unify.get_logs(
            context=context,
            filter=filter,  # Only apply user's additional filter
            offset=offset,
            limit=batch_size,
            return_ids_only=False,  # Need entries to check fields
        )

        if not logs:
            break

        # Filter client-side for logs needing updates
        logs_to_update = []
        for lg in logs:
            needs_update = False
            if user_context is not None and lg.entries.get("_user") is None:
                needs_update = True
            if assistant_context is not None and lg.entries.get("_assistant") is None:
                needs_update = True
            if needs_update:
                logs_to_update.append(lg.id)

        if logs_to_update:
            entries: Dict[str, str] = {}
            if user_context is not None:
                entries["_user"] = user_context
            if assistant_context is not None:
                entries["_assistant"] = assistant_context

            # Batch update
            unify.update_logs(
                logs=logs_to_update,
                context=context,
                entries=entries,
                overwrite=False,  # Don't overwrite other fields
            )
            total_updated += len(logs_to_update)

        # Move to next batch
        offset += len(logs)

        # If we got fewer than batch_size, we're done
        if len(logs) < batch_size:
            break

    return {"total_updated": total_updated, "context": context}


def backfill_all_contexts_for_user_assistant(
    *,
    user_context: str,
    assistant_context: str,
    contexts: Optional[List[str]] = None,
    batch_size: int = 100,
) -> Dict[str, Dict[str, Any]]:
    """
    Backfill _user and _assistant for all contexts belonging to a user/assistant.

    If contexts is None, discovers them by prefix "{user_context}/{assistant_context}/".

    Parameters
    ----------
    user_context : str
        The user ID (used as context prefix and _user field value)
    assistant_context : str
        The assistant ID (used as context prefix and _assistant field value)
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
        prefix = f"{user_context}/{assistant_context}/"
        all_contexts = unify.get_contexts(prefix=prefix)
        contexts = list(all_contexts.keys())

    results: Dict[str, Dict[str, Any]] = {}
    for ctx in contexts:
        try:
            result = backfill_private_fields(
                ctx,
                user_context=user_context,
                assistant_context=assistant_context,
                batch_size=batch_size,
            )
            results[ctx] = result
        except Exception as e:
            results[ctx] = {"error": str(e), "context": ctx}

    return results


# Backward compatibility aliases
def backfill_assistant_field(
    context: str,
    assistant_context: str,
    *,
    batch_size: int = 100,
    filter: Optional[str] = None,
) -> Dict[str, Any]:
    """Backward-compatible wrapper for backfill_private_fields."""
    return backfill_private_fields(
        context,
        assistant_context=assistant_context,
        batch_size=batch_size,
        filter=filter,
    )


def backfill_all_contexts_for_assistant(
    assistant_context: str,
    *,
    contexts: Optional[List[str]] = None,
    batch_size: int = 100,
) -> Dict[str, Dict[str, Any]]:
    """Backward-compatible wrapper - discovers contexts by old single-prefix pattern."""
    if contexts is None:
        all_contexts = unify.get_contexts(prefix=f"{assistant_context}/")
        contexts = list(all_contexts.keys())

    results: Dict[str, Dict[str, Any]] = {}
    for ctx in contexts:
        try:
            result = backfill_private_fields(
                ctx,
                assistant_context=assistant_context,
                batch_size=batch_size,
            )
            results[ctx] = result
        except Exception as e:
            results[ctx] = {"error": str(e), "context": ctx}

    return results
