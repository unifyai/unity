"""
Wrappers around unisdk.log/create_logs with:
1. _user injection (user ID, matches user_context path component)
2. _user_id injection (user ID from SESSION_DETAILS)
3. _assistant injection (assistant ID, matches assistant_context path component)
4. _assistant_id injection (assistant's agent_id from SESSION_DETAILS.assistant.agent_id)
5. _org injection (organization ID from SESSION_DETAILS, None for personal context)
6. _org_id injection (organization ID from SESSION_DETAILS, None for personal context)
7. Automatic addition to aggregation contexts by reference (copy=False)

Usage
-----
Replace direct unisdk.log/create_logs calls with these wrappers:

    from unify.common.log_utils import log, create_logs

    # Instead of: unisdk.log(context=ctx, **entries)
    log(context=ctx, **entries)

    # Instead of: unisdk.create_logs(context=ctx, entries=entries_list)
    create_logs(context=ctx, entries=entries_list)

The wrappers automatically inject _user, _user_id, _assistant, _assistant_id,
_org, _org_id as private fields.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import httpx
import unisdk

from unify.common.authorship import (
    AUTHORING_ASSISTANT_ID_FIELD,
    current_authoring_assistant_id,
)
from unify.session_details import SESSION_DETAILS
from unify.settings import SETTINGS

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


class MissingRowIdentityError(RuntimeError):
    """A created row came back without its auto-counted identity column."""


def assigned_row_id(log: unisdk.Log, column: str, *, context: str) -> int:
    """Return the identity value the backend assigned to a freshly created row.

    A context configured with unique-key auto-counting stamps *column* on
    every insert and echoes it back on the created log. A missing echo means
    the context is live without that configuration — it was created
    implicitly by a row write reaching the backend before provisioning — so
    the row can never be addressed by id. The orphan row is deleted so junk
    does not accumulate, and the corruption is raised to the caller: no API
    can retrofit the configuration, so the context must be deleted and
    re-provisioned before further writes.
    """
    value = log.entries.get(column)
    if value is not None:
        return int(value)
    unisdk.delete_logs(logs=log.id, context=context)
    raise MissingRowIdentityError(
        f"Context {context!r} accepted a row without assigning {column!r}: "
        "the context is live without its unique-key/auto-counting "
        "configuration, so rows written to it cannot be addressed by id. "
        "The orphan row was deleted. Delete and re-provision the context "
        "before writing again.",
    )


# EventBus metadata columns stored alongside an event's spread payload fields.
_EVENT_META_KEYS = frozenset(
    {"row_id", "event_id", "calling_id", "event_timestamp", "payload_cls", "type"},
)


def payload_from_log_entries(entries: Dict[str, Any]) -> Dict[str, Any]:
    """Reconstruct an event payload from a spread event log row.

    Events are stored in their per-type context with payload fields spread into
    top-level columns. This drops the EventBus metadata columns and the injected
    private fields (all underscore-prefixed, see :func:`_inject_private_fields`)
    so the result matches the payload dict that was originally published.
    """
    return {
        key: value
        for key, value in entries.items()
        if key not in _EVENT_META_KEYS and not key.startswith("_")
    }


def log(
    context: str,
    *,
    new: bool = True,
    mutable: bool = False,
    project: Optional[str] = None,
    stamp_authoring: bool = False,
    **entries: Any,
) -> unisdk.Log:
    """
    Wrapper around unisdk.log with private field injection.

    Parameters
    ----------
    context : str
        The context to log to (e.g., "42/7/Contacts")
    new : bool, default True
        Whether to create a new log entry
    mutable : bool, default False
        Whether the log entry is mutable
    **entries
        Field values to log

    Returns
    -------
    unisdk.Log
        The created log object
    """
    if stamp_authoring:
        entries[AUTHORING_ASSISTANT_ID_FIELD] = current_authoring_assistant_id()
    entries = _inject_private_fields(entries)
    return unisdk.log(
        project=project,
        context=context,
        new=new,
        mutable=mutable,
        **entries,
    )


def create_logs(
    context: str,
    *,
    entries: List[Dict[str, Any]],
    project: Optional[str] = None,
    stamp_authoring: bool = False,
    **kwargs: Any,
) -> Any:
    """
    Wrapper around unisdk.create_logs with private field injection.

    Parameters
    ----------
    context : str
        The context to log to (e.g., "42/7/Tasks")
    entries : List[Dict[str, Any]]
        List of entry dicts to create
    **kwargs
        Additional arguments passed to unisdk.create_logs (e.g., batched=True)

    Returns
    -------
    Dict[str, Any] | List[unisdk.Log]
        Response from unisdk.create_logs. Returns a dict with log_event_ids normally,
        or a list of Log objects when batched=True.
    """
    authoring_assistant_id = (
        current_authoring_assistant_id() if stamp_authoring else None
    )
    entries = [
        _inject_private_fields(
            {
                **entry,
                **(
                    {AUTHORING_ASSISTANT_ID_FIELD: authoring_assistant_id}
                    if stamp_authoring
                    else {}
                ),
            },
        )
        for entry in entries
    ]
    return unisdk.create_logs(
        project=project,
        context=context,
        entries=entries,
        **kwargs,
    )


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
        project = unisdk.active_project()

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
    project: Optional[str] = None,
) -> AtomicUpsertResult:
    """
    Synchronous version of atomic_upsert for use in non-async contexts.

    See atomic_upsert() for full documentation.
    """
    if project is None:
        project = unisdk.active_project()

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
