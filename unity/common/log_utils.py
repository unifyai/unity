"""
Wrappers around unify.log/create_logs with:
1. _user injection (user ID, matches user_context path component)
2. _user_id injection (user ID from SESSION_DETAILS)
3. _assistant injection (assistant ID, matches assistant_context path component)
4. _assistant_id injection (assistant's agent_id from SESSION_DETAILS.assistant.agent_id)
5. _org injection (organization ID from SESSION_DETAILS, None for personal context)
6. _org_id injection (organization ID from SESSION_DETAILS, None for personal context)
7. Automatic addition to aggregation contexts by reference (copy=False)

Usage
-----
Replace direct unify.log/create_logs calls with these wrappers:

    from unity.common.log_utils import log, create_logs

    # Instead of: unify.log(context=ctx, **entries)
    log(context=ctx, **entries)

    # Instead of: unify.create_logs(context=ctx, entries=entries_list)
    create_logs(context=ctx, entries=entries_list)

The wrappers automatically:
- Inject _user, _user_id, _assistant, _assistant_id, _org, _org_id as private fields
- Add logs to aggregation contexts by reference (when add_to_all_context=True):
  - {user_id}/All/{Suffix} - user-level aggregation (all assistants for this user)
  - All/{Suffix} - global aggregation (all users, all assistants)

For test contexts (starting with "tests/"), aggregation is scoped to the test root:
  - {test_root}/{user_id}/All/{Suffix}
  - {test_root}/All/{Suffix}
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import httpx
import unify

from unity.session_details import SESSION_DETAILS
from unity.settings import SETTINGS

logger = logging.getLogger(__name__)


def _get_user_context() -> Optional[str]:
    """Retrieve user's context path component (user ID) from SESSION_DETAILS.

    Injected as _user into every log entry. Matches the user_id path segment
    in context paths like {user_id}/{assistant_id}/Contacts.
    Needed by orchestra for 3-tier deletion cascade.
    """
    return SESSION_DETAILS.user_context or None


def _get_user_id() -> Optional[str]:
    """Retrieve user_id from SESSION_DETAILS."""
    return SESSION_DETAILS.user.id or None


def _get_assistant_context() -> Optional[str]:
    """Retrieve assistant's context path component (assistant ID) from SESSION_DETAILS.

    Injected as _assistant into every log entry. Matches the assistant_id path
    segment in context paths like {user_id}/{assistant_id}/Contacts.
    Needed by orchestra for 3-tier deletion cascade.
    """
    return SESSION_DETAILS.assistant_context or None


def _get_assistant_id() -> Optional[str]:
    """Retrieve assistant's agent_id from SESSION_DETAILS as a string."""
    aid = SESSION_DETAILS.assistant.agent_id
    return str(aid) if aid is not None else None


def _get_org_id() -> Optional[int]:
    """Retrieve organization ID from SESSION_DETAILS.

    Returns None for personal (non-org) context.
    """
    return SESSION_DETAILS.org_id


def _get_org_context() -> Optional[str]:
    """Retrieve organization name from SESSION_DETAILS.

    Returns None/empty for personal (non-org) context.
    """
    return SESSION_DETAILS.org_name or None


def _inject_private_fields(entries: Dict[str, Any]) -> Dict[str, Any]:
    """Inject _user, _user_id, _assistant, _assistant_id, _org, and _org_id into entries."""
    result = dict(entries)

    user_ctx = _get_user_context()
    if user_ctx is not None:
        result["_user"] = user_ctx

    user_id = _get_user_id()
    if user_id is not None:
        result["_user_id"] = user_id

    assistant_ctx = _get_assistant_context()
    if assistant_ctx is not None:
        result["_assistant"] = assistant_ctx

    assistant_id = _get_assistant_id()
    if assistant_id is not None:
        result["_assistant_id"] = assistant_id

    org_ctx = _get_org_context()
    if org_ctx is not None:
        result["_org"] = org_ctx

    org_id = _get_org_id()
    if org_id is not None:
        result["_org_id"] = org_id

    return result


def _derive_all_contexts(context: str) -> List[str]:
    """
    Derive aggregation contexts from a user/assistant-scoped context.

    Returns two contexts for cross-assistant and cross-user aggregation:
      - {user_id}/All/{suffix} - all assistants for this user
      - All/{suffix}           - all users, all assistants

    For test contexts (starting with "tests/"), the aggregation contexts are
    scoped to the test root for proper isolation:
      - {test_root}/{user_id}/All/{suffix}
      - {test_root}/All/{suffix}

    Examples:
        Production:
        "42/7/Contacts" -> ["42/All/Contacts", "All/Contacts"]

        Testing:
        "tests/foo/default/default-assistant/Contacts" ->
            ["tests/foo/default/All/Contacts", "tests/foo/All/Contacts"]

        Invalid (too few parts):
        "42/Contacts" -> []
        "Contacts" -> []
    """
    parts = context.split("/")
    if len(parts) < 3:
        return []

    # Handle test contexts: tests/.../{default_user_id}/{default_assistant_id}/Suffix
    # Find the user position by looking for the UNASSIGNED_USER_CONTEXT marker
    if parts[0] == "tests":
        from unity.session_details import UNASSIGNED_USER_CONTEXT

        try:
            user_idx = parts.index(UNASSIGNED_USER_CONTEXT)
        except ValueError:
            # Can't determine structure without the UNASSIGNED_USER_CONTEXT marker
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
        The context to log to (e.g., "42/7/Contacts")
    add_to_all_context : bool, default False
        If True, add the log to aggregation contexts by reference:
        - {user_id}/All/{Ctx} (user-level)
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
        The context to log to (e.g., "42/7/Tasks")
    entries : List[Dict[str, Any]]
        List of entry dicts to create
    add_to_all_context : bool, default False
        If True, add logs to aggregation contexts by reference:
        - {user_id}/All/{Ctx} (user-level)
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


# =============================================================================
# Atomic Upsert for Spending Tracking
# =============================================================================


@dataclass
class AtomicUpsertResult:
    """Result of an atomic upsert operation."""

    log_id: int
    new_value: float
    created: bool
    mirrored_contexts: List[str]


async def atomic_upsert(
    context: str,
    *,
    unique_keys: Dict[str, str],
    field: str,
    operation: str,
    initial_data: Optional[Dict[str, Any]] = None,
    add_to_all_context: bool = False,
    project: Optional[str] = None,
    data_overrides: Optional[Dict[str, Any]] = None,
) -> AtomicUpsertResult:
    """
    Atomically upsert a field value in a log entry.

    This function calls Orchestra's `/v0/logs/atomic` endpoint which:
    1. Ensures context exists with correct unique_keys configuration
    2. Acquires advisory lock on unique key values (prevents race on first insert)
    3. Finds log by unique_keys or creates it with initial_data
    4. Applies atomic operation to field
    5. Optionally mirrors to All/* archive context

    Parameters
    ----------
    context : str
        The context to upsert to (e.g., "42/7/Spending/Monthly")
    unique_keys : Dict[str, str]
        Key names to types for matching/creating logs
        (e.g., {"_assistant_id": "str", "month": "str"})
    field : str
        The field to update atomically (e.g., "cumulative_spend")
    operation : str
        The atomic operation to apply (e.g., "+5.50" for increment)
    initial_data : Dict[str, Any], optional
        Data for creating a new log if one doesn't exist.
        Must include all unique key values.
    add_to_all_context : bool, default False
        If True, mirror the log to All/* archive context
    project : str, optional
        The project name. Defaults to the active project.
    data_overrides : Dict[str, Any], optional
        Values applied after private field injection to override specific
        injected fields (e.g., ``{"_user_id": "..."}`` for per-user cost
        attribution).

    Returns
    -------
    AtomicUpsertResult
        Result containing log_id, new_value, created flag, and mirrored contexts

    Raises
    ------
    httpx.HTTPStatusError
        If the API request fails
    """
    if project is None:
        project = unify.active_project()

    # Inject private fields into initial_data
    if initial_data is None:
        initial_data = {}
    initial_data = _inject_private_fields(initial_data)
    if data_overrides:
        initial_data.update(data_overrides)

    # Build request payload
    payload = {
        "project": project,
        "context": context,
        "unique_keys": unique_keys,
        "field": field,
        "operation": operation,
        "initial_data": initial_data,
        "add_to_all_context": add_to_all_context,
    }

    # Get API credentials
    api_key = SESSION_DETAILS.unify_key
    base_url = SETTINGS.ORCHESTRA_URL

    # Make the HTTP request to Orchestra
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            f"{base_url}/logs/atomic",
            json=payload,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        )
        response.raise_for_status()
        data = response.json()

    return AtomicUpsertResult(
        log_id=data.get("log_id", 0),
        new_value=data.get("new_value", 0.0),
        created=data.get("created", False),
        mirrored_contexts=data.get("mirrored_contexts", []),
    )


def atomic_upsert_sync(
    context: str,
    *,
    unique_keys: Dict[str, str],
    field: str,
    operation: str,
    initial_data: Optional[Dict[str, Any]] = None,
    add_to_all_context: bool = False,
    project: Optional[str] = None,
) -> AtomicUpsertResult:
    """
    Synchronous version of atomic_upsert for use in non-async contexts.

    See atomic_upsert() for full documentation.
    """
    if project is None:
        project = unify.active_project()

    # Inject private fields into initial_data
    if initial_data is None:
        initial_data = {}
    initial_data = _inject_private_fields(initial_data)

    # Build request payload
    payload = {
        "project": project,
        "context": context,
        "unique_keys": unique_keys,
        "field": field,
        "operation": operation,
        "initial_data": initial_data,
        "add_to_all_context": add_to_all_context,
    }

    # Get API credentials
    api_key = SESSION_DETAILS.unify_key
    base_url = SETTINGS.ORCHESTRA_URL

    # Make the HTTP request to Orchestra
    with httpx.Client(timeout=30.0) as client:
        response = client.post(
            f"{base_url}/logs/atomic",
            json=payload,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        )
        response.raise_for_status()
        data = response.json()

    return AtomicUpsertResult(
        log_id=data.get("log_id", 0),
        new_value=data.get("new_value", 0.0),
        created=data.get("created", False),
        mirrored_contexts=data.get("mirrored_contexts", []),
    )
