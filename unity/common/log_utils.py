"""
Wrappers around unify.log/create_logs with:
1. _user injection (user's name from USER_CONTEXT)
2. _user_id injection (from USER_ID environment variable)
3. _assistant injection (assistant's name from ASSISTANT_CONTEXT)
4. _assistant_id injection (assistant's ID from ASSISTANT["agent_id"])
5. Automatic addition to aggregation contexts by reference (copy=False)

Usage
-----
Replace direct unify.log/create_logs calls with these wrappers:

    from unity.common.log_utils import log, create_logs

    # Instead of: unify.log(context=ctx, **entries)
    log(context=ctx, **entries)

    # Instead of: unify.create_logs(context=ctx, entries=entries_list)
    create_logs(context=ctx, entries=entries_list)

The wrappers automatically:
- Inject _user, _user_id, _assistant, _assistant_id as private fields
- Add logs to aggregation contexts by reference (when add_to_all_context=True):
  - {UserName}/All/{Suffix} - user-level aggregation (all assistants for this user)
  - All/{Suffix} - global aggregation (all users, all assistants)

For test contexts (starting with "tests/"), aggregation is scoped to the test root:
  - {test_root}/{UserName}/All/{Suffix}
  - {test_root}/All/{Suffix}
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import unify

from unity.session_details import SESSION_DETAILS


def _get_user_name() -> Optional[str]:
    """Retrieve user's context name from SESSION_DETAILS.

    Returns the user context name, including default values like "DefaultUser".
    These are always injected because they are part of the context path structure
    and are needed by orchestra for 3-tier deletion cascade.
    """
    return SESSION_DETAILS.user_context or None


def _get_user_id() -> Optional[str]:
    """Retrieve user_id from SESSION_DETAILS."""
    return SESSION_DETAILS.user.id or None


def _get_assistant_name() -> Optional[str]:
    """Retrieve assistant's context name from SESSION_DETAILS.

    Returns the assistant context name, including default values like "Assistant".
    These are always injected because they are part of the context path structure
    and are needed by orchestra for 3-tier deletion cascade.
    """
    return SESSION_DETAILS.assistant_context or None


def _get_assistant_id() -> Optional[str]:
    """Retrieve assistant's ID from SESSION_DETAILS."""
    if SESSION_DETAILS.assistant_record is not None:
        return SESSION_DETAILS.assistant_record.get("agent_id")
    return None


def _inject_private_fields(entries: Dict[str, Any]) -> Dict[str, Any]:
    """Inject _user, _user_id, _assistant, and _assistant_id into entries."""
    result = dict(entries)

    user_name = _get_user_name()
    if user_name is not None:
        result["_user"] = user_name

    user_id = _get_user_id()
    if user_id is not None:
        result["_user_id"] = user_id

    assistant_name = _get_assistant_name()
    if assistant_name is not None:
        result["_assistant"] = assistant_name

    assistant_id = _get_assistant_id()
    if assistant_id is not None:
        result["_assistant_id"] = assistant_id

    return result


def _derive_all_contexts(context: str) -> List[str]:
    """
    Derive aggregation contexts from a user/assistant-scoped context.

    Returns two contexts for cross-assistant and cross-user aggregation:
      - {UserName}/All/{suffix} - all assistants for this user
      - All/{suffix}            - all users, all assistants

    For test contexts (starting with "tests/"), the aggregation contexts are
    scoped to the test root for proper isolation:
      - {test_root}/{UserName}/All/{suffix}
      - {test_root}/All/{suffix}

    Examples:
        Production:
        "JohnDoe/MyAssistant/Contacts" -> ["JohnDoe/All/Contacts", "All/Contacts"]

        Testing:
        "tests/foo/DefaultUser/Assistant/Contacts" ->
            ["tests/foo/DefaultUser/All/Contacts", "tests/foo/All/Contacts"]

        Invalid (too few parts):
        "JohnDoe/Contacts" -> []
        "Contacts" -> []
    """
    parts = context.split("/")
    if len(parts) < 3:
        return []

    # Handle test contexts: tests/.../DefaultUser/Assistant/Suffix
    # Find the User position by looking for "DefaultUser" marker
    if parts[0] == "tests":
        from unity.session_details import DEFAULT_USER_CONTEXT

        try:
            user_idx = parts.index(DEFAULT_USER_CONTEXT)
        except ValueError:
            # Can't determine structure without the DefaultUser marker
            return []

        # Need at least User/Assistant/Suffix after the test root
        if user_idx + 2 >= len(parts):
            return []

        test_root = "/".join(parts[:user_idx])
        user_ctx = parts[user_idx]
        suffix = "/".join(parts[user_idx + 2 :])

        return [
            f"{test_root}/{user_ctx}/All/{suffix}",  # User-level aggregation
            f"{test_root}/All/{suffix}",  # Global aggregation
        ]

    # Production path: User/Assistant/Suffix
    user_ctx = parts[0]
    suffix = "/".join(parts[2:])
    return [
        f"{user_ctx}/All/{suffix}",  # User-level aggregation
        f"All/{suffix}",  # Global aggregation
    ]


def _add_to_all(log_ids: List[int], context: str) -> None:
    """Add logs by reference to all aggregation contexts (best-effort)."""
    if not log_ids:
        return
    all_ctxs = _derive_all_contexts(context)
    for all_ctx in all_ctxs:
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
    add_to_all_context: bool = False,
    new: bool = True,
    mutable: bool = False,
    **entries: Any,
) -> unify.Log:
    """
    Wrapper around unify.log with private field injection and aggregation context addition.

    Parameters
    ----------
    context : str
        The context to log to (e.g., "JohnDoe/MyAssistant/Contacts")
    add_to_all_context : bool, default False
        If True, add the log to aggregation contexts by reference:
        - {UserName}/All/{Ctx} (user-level)
        - All/{Ctx} (global)
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

    if add_to_all_context:
        try:
            _add_to_all([result.id], context)
        except Exception:
            pass

    return result


def create_logs(
    context: str,
    *,
    entries: List[Dict[str, Any]],
    add_to_all_context: bool = False,
    **kwargs: Any,
) -> Any:
    """
    Wrapper around unify.create_logs with private field injection and aggregation context addition.

    Parameters
    ----------
    context : str
        The context to log to (e.g., "JohnDoe/MyAssistant/Tasks")
    entries : List[Dict[str, Any]]
        List of entry dicts to create
    add_to_all_context : bool, default False
        If True, add logs to aggregation contexts by reference:
        - {UserName}/All/{Ctx} (user-level)
        - All/{Ctx} (global)
    **kwargs
        Additional arguments passed to unify.create_logs (e.g., batched=True)

    Returns
    -------
    Dict[str, Any] | List[unify.Log]
        Response from unify.create_logs. Returns a dict with log_event_ids normally,
        or a list of Log objects when batched=True.
    """
    entries = [_inject_private_fields(e) for e in entries]
    result = unify.create_logs(context=context, entries=entries, **kwargs)

    if add_to_all_context:
        # Handle both dict (normal) and list (batched=True) return types
        if isinstance(result, dict):
            log_ids = result.get("log_event_ids", [])
        elif isinstance(result, list):
            # batched=True returns a list of Log objects
            log_ids = [lg.id for lg in result if hasattr(lg, "id")]
        else:
            log_ids = []

        if log_ids:
            _add_to_all(log_ids, context)

    return result
