"""
Wrappers around unify.log/create_logs with:
1. _assistant injection (assistant's name from ASSISTANT_CONTEXT)
2. _assistant_id injection (assistant's ID from ASSISTANT["agent_id"])
3. _user_id injection (from USER_ID environment variable)
4. Automatic mirroring to All/<Ctx> by reference (copy=False)

Usage
-----
Replace direct unify.log/create_logs calls with these wrappers:

    from unity.common.log_utils import log, create_logs

    # Instead of: unify.log(context=ctx, **entries)
    log(context=ctx, **entries)

    # Instead of: unify.create_logs(context=ctx, entries=entries_list)
    create_logs(context=ctx, entries=entries_list)

The wrappers automatically:
- Inject _assistant, _assistant_id, _user_id as private fields
- Mirror logs to All/<Ctx> by reference (unless mirror_to_all=False)
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import unify


def _get_assistant_name() -> Optional[str]:
    """Retrieve assistant's name from ASSISTANT_CONTEXT."""
    try:
        from unity import ASSISTANT_CONTEXT

        return ASSISTANT_CONTEXT
    except Exception:
        return None


def _get_assistant_id() -> Optional[str]:
    """Retrieve assistant's ID from ASSISTANT dict."""
    try:
        from unity import ASSISTANT

        if ASSISTANT is not None:
            return ASSISTANT.get("agent_id")
    except Exception:
        pass
    return None


def _get_user_id() -> Optional[str]:
    """Retrieve user_id from environment."""
    return os.environ.get("USER_ID")


def _inject_private_fields(entries: Dict[str, Any]) -> Dict[str, Any]:
    """Inject _assistant, _assistant_id, and _user_id into entries."""
    result = dict(entries)

    assistant_name = _get_assistant_name()
    if assistant_name is not None:
        result["_assistant"] = assistant_name

    assistant_id = _get_assistant_id()
    if assistant_id is not None:
        result["_assistant_id"] = assistant_id

    user_id = _get_user_id()
    if user_id is not None:
        result["_user_id"] = user_id

    return result


def _derive_all_context(context: str) -> Optional[str]:
    """
    Derive the All/<suffix> context from an assistant-scoped context.

    Examples:
        "JohnDoe/Contacts" -> "All/Contacts"
        "JohnDoe/Tasks" -> "All/Tasks"
        "Contacts" -> None (no assistant prefix)
    """
    if "/" not in context:
        return None
    _, suffix = context.split("/", 1)
    return f"All/{suffix}"


def _mirror_to_all(log_ids: List[int], context: str) -> None:
    """Add logs by reference to the All/<suffix> context (best-effort)."""
    if not log_ids:
        return
    all_ctx = _derive_all_context(context)
    if all_ctx is None:
        return
    try:
        unify.add_logs_to_context(
            log_ids,
            context=all_ctx,
            project=unify.active_project(),
        )
    except Exception:
        pass  # Best-effort: don't fail the main operation


def log(
    context: str,
    *,
    mirror_to_all: bool = True,
    new: bool = True,
    mutable: bool = False,
    **entries: Any,
) -> unify.Log:
    """
    Wrapper around unify.log with private field injection and All/<Ctx> mirroring.

    Parameters
    ----------
    context : str
        The context to log to (e.g., "JohnDoe/Contacts")
    mirror_to_all : bool, default True
        If True, mirror the log to All/<Ctx> by reference
    new : bool, default True
        Whether to create a new log entry
    mutable : bool, default False
        Whether the log entry is mutable
    **entries
        Field values to log

    Returns
    -------
    unify.Log
        The created log object
    """
    entries = _inject_private_fields(entries)
    result = unify.log(context=context, new=new, mutable=mutable, **entries)

    if mirror_to_all:
        try:
            _mirror_to_all([result.id], context)
        except Exception:
            pass

    return result


def create_logs(
    context: str,
    *,
    entries: List[Dict[str, Any]],
    mirror_to_all: bool = True,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Wrapper around unify.create_logs with private field injection and All/<Ctx> mirroring.

    Parameters
    ----------
    context : str
        The context to log to (e.g., "JohnDoe/Tasks")
    entries : List[Dict[str, Any]]
        List of entry dicts to create
    mirror_to_all : bool, default True
        If True, mirror logs to All/<Ctx> by reference
    **kwargs
        Additional arguments passed to unify.create_logs (e.g., batched=True)

    Returns
    -------
    Dict[str, Any]
        Response from unify.create_logs containing log_event_ids, etc.
    """
    entries = [_inject_private_fields(e) for e in entries]
    result = unify.create_logs(context=context, entries=entries, **kwargs)

    if mirror_to_all:
        log_ids = result.get("log_event_ids", [])
        if log_ids:
            _mirror_to_all(log_ids, context)

    return result
