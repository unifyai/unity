from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

import unisdk
from pydantic import ValidationError

from ..common.authorship import (
    AUTHORING_ASSISTANT_ID_FIELD,
    strip_authoring_assistant_id,
)
from ..common.log_utils import assigned_row_id, log as unity_log
from ..common.tool_outcome import ToolOutcome
from .types.contact import Contact

# Named create/update params (plus plumbing). The Contact schema is fixed;
# any key outside this set is rejected. Authorship is stamped separately and
# must not appear in named kwargs.
_NAMED_CREATE_FIELDS = frozenset(
    {
        "first_name",
        "surname",
        "email_address",
        "phone_number",
        "whatsapp_number",
        "discord_id",
        "slack_user_id",
        "bio",
        "job_title",
        "timezone",
        "rolling_summary",
        "should_respond",
        "response_policy",
        "is_system",
        "custom_key",
        "custom_hash",
        "user_id",
        "agent_id",
        "destination",
        "context",
        "data_store",
        "_contact_id",
    },
)
_NAMED_UPDATE_FIELDS = frozenset(
    {
        "contact_id",
        "first_name",
        "surname",
        "email_address",
        "phone_number",
        "whatsapp_number",
        "discord_id",
        "slack_user_id",
        "bio",
        "job_title",
        "timezone",
        "rolling_summary",
        "should_respond",
        "response_policy",
        "is_system",
        "custom_key",
        "custom_hash",
        "user_id",
        "agent_id",
        "destination",
        "context",
        "data_store",
        "_log_id",
        "_contact_id",
    },
)


def partition_create_kwargs(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Split a splat dict into closed ``_create_contact`` kwargs."""
    payload = strip_authoring_assistant_id(payload)
    named: dict[str, Any] = {}
    for key, value in payload.items():
        if key == AUTHORING_ASSISTANT_ID_FIELD:
            continue
        if key == "contact_id":
            named["_contact_id"] = value
        elif key in _NAMED_CREATE_FIELDS:
            named[key] = value
        else:
            raise ValueError(f"Unknown contact field: {key}")
    return named


def partition_update_kwargs(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Split a splat dict into closed ``update_contact`` kwargs."""
    # Authorship is stamped at the log boundary, never as a create/update kwarg.
    payload = strip_authoring_assistant_id(payload)
    named: dict[str, Any] = {}
    for key, value in payload.items():
        if key == AUTHORING_ASSISTANT_ID_FIELD:
            continue
        if key in _NAMED_UPDATE_FIELDS:
            named[key] = value
        else:
            raise ValueError(f"Unknown contact field: {key}")
    return named


def _get_assistant_id() -> int | None:
    """Get assistant_id from SESSION_DETAILS, returning None if unavailable."""
    from ..session_details import SESSION_DETAILS

    if not SESSION_DETAILS.is_initialized:
        return None
    return SESSION_DETAILS.assistant.agent_id


def _maybe_sync_timezone_to_backend(
    self,
    contact_id: int,
    timezone: str,
    *,
    context: str | None = None,
    data_store: Any = None,
) -> None:
    """
    Fire-and-forget sync of timezone to backend for system contacts.

    - assistant self contact → sync to assistant
    - boss or other system contact → sync to user via email
    """
    from .backend_sync import sync_assistant_timezone, sync_user_timezone

    assistant_id = _get_assistant_id()
    if assistant_id is None:
        return

    from ..session_details import SESSION_DETAILS

    if contact_id == SESSION_DETAILS.self_contact_id:
        sync_assistant_timezone(assistant_id, timezone)
        return

    store = data_store or self._data_store
    context_name = context or self._ctx

    # User or org member - need email and is_system check
    try:
        row = store.get(contact_id)
    except KeyError:
        row = None

    if row is None:
        try:
            rows = unisdk.get_logs(
                context=context_name,
                filter=f"contact_id == {contact_id}",
                limit=1,
                from_fields=["contact_id", "is_system", "email_address"],
            )
            row = rows[0].entries if rows else None
        except Exception:
            return

    if row and row.get("is_system") and row.get("email_address"):
        sync_user_timezone(assistant_id, row["email_address"], timezone)


def _maybe_sync_bio_to_backend(
    self,
    contact_id: int,
    bio: str,
    *,
    context: str | None = None,
    data_store: Any = None,
) -> None:
    """
    Fire-and-forget sync of bio to backend for system contacts.

    - assistant self contact → sync to assistant (as 'about')
    - boss or other system contact → sync to user (as 'bio')
    """
    from .backend_sync import sync_assistant_about, sync_user_bio

    assistant_id = _get_assistant_id()
    if assistant_id is None:
        return

    from ..session_details import SESSION_DETAILS

    if contact_id == SESSION_DETAILS.self_contact_id:
        sync_assistant_about(assistant_id, bio)
        return

    store = data_store or self._data_store
    context_name = context or self._ctx

    # User or org member - need email and is_system check
    try:
        row = store.get(contact_id)
    except KeyError:
        row = None

    if row is None:
        try:
            rows = unisdk.get_logs(
                context=context_name,
                filter=f"contact_id == {contact_id}",
                limit=1,
                from_fields=["contact_id", "is_system", "email_address"],
            )
            row = rows[0].entries if rows else None
        except Exception:
            return

    if row and row.get("is_system") and row.get("email_address"):
        sync_user_bio(assistant_id, row["email_address"], bio)


def _maybe_sync_job_title_to_backend(
    self,
    contact_id: int,
    job_title: str,
) -> None:
    """
    Fire-and-forget sync of ``job_title`` to backend for the assistant contact.

    Mirrors :func:`_maybe_sync_bio_to_backend` but for the free-text job title
    / specialization. Only the assistant's self contact flows back to the
    Assistant table; for any other contact the value is purely local metadata.
    """
    from .backend_sync import sync_assistant_job_title

    from ..session_details import SESSION_DETAILS

    if contact_id != SESSION_DETAILS.self_contact_id:
        return
    assistant_id = _get_assistant_id()
    if assistant_id is None:
        return
    sync_assistant_job_title(assistant_id, job_title)


def create_contact(
    self,
    *,
    first_name: Optional[str] = None,
    surname: Optional[str] = None,
    email_address: Optional[str] = None,
    phone_number: Optional[str] = None,
    whatsapp_number: Optional[str] = None,
    discord_id: Optional[str] = None,
    slack_user_id: Optional[str] = None,
    bio: Optional[str] = None,
    job_title: Optional[str] = None,
    timezone: Optional[str] = None,
    rolling_summary: Optional[str] = None,
    should_respond: bool = True,
    response_policy: Optional[str] = None,
    is_system: bool = False,
    custom_key: Optional[str] = None,
    custom_hash: Optional[str] = None,
    user_id: Optional[str] = None,
    agent_id: Optional[str] = None,
    contact_id: Optional[int] = None,
    context: str | None = None,
    data_store: Any = None,
) -> ToolOutcome:
    context_name = context or self._ctx
    store = data_store or self._data_store

    contact_details = {
        "first_name": first_name,
        "surname": surname,
        "email_address": email_address,
        "phone_number": phone_number,
        "whatsapp_number": whatsapp_number,
        "discord_id": discord_id,
        "slack_user_id": slack_user_id,
        "bio": bio,
        "job_title": job_title,
        "timezone": timezone,
        "rolling_summary": rolling_summary,
        "should_respond": should_respond,
        "response_policy": response_policy,
        "is_system": is_system,
        "custom_key": custom_key,
        "custom_hash": custom_hash,
    }
    if user_id is not None:
        contact_details["user_id"] = user_id
    if agent_id is not None:
        contact_details["agent_id"] = agent_id
    if contact_id is not None:
        contact_details["contact_id"] = contact_id
    if contact_details["response_policy"] is None:
        contact_details["response_policy"] = self.DEFAULT_RESPONSE_POLICY

    contact_details = strip_authoring_assistant_id(contact_details)

    if not any(v is not None for v in contact_details.values()):
        raise AssertionError("At least one contact detail must be provided.")

    # Validate against Pydantic model
    try:
        Contact(**contact_details)
    except ValidationError as e:
        msg = str(e)
        try:
            err = e.errors()[0]
            msg = err.get("msg", str(e))
            if err.get("type") == "value_error":
                ctx = err.get("ctx", {})
                if "error" in ctx:
                    msg = str(ctx["error"])
        except Exception:
            pass
        raise ValueError(msg) from e

    log = unity_log(
        context=context_name,
        **contact_details,
        new=True,
        mutable=True,
        stamp_authoring=True,
    )
    contact_id = assigned_row_id(log, "contact_id", context=context_name)
    try:
        store.put(log.entries)
    except Exception:
        pass
    return {
        "outcome": "contact created successfully",
        "details": {"contact_id": contact_id},
    }


def update_contact(
    self,
    *,
    contact_id: int,
    first_name: Optional[str] = None,
    surname: Optional[str] = None,
    email_address: Optional[str] = None,
    phone_number: Optional[str] = None,
    whatsapp_number: Optional[str] = None,
    discord_id: Optional[str] = None,
    slack_user_id: Optional[str] = None,
    bio: Optional[str] = None,
    job_title: Optional[str] = None,
    timezone: Optional[str] = None,
    rolling_summary: Optional[str] = None,
    should_respond: Optional[bool] = None,
    response_policy: Optional[str] = None,
    is_system: Optional[bool] = None,
    custom_key: Optional[str] = None,
    custom_hash: Optional[str] = None,
    user_id: Optional[str] = None,
    agent_id: Optional[str] = None,
    _log_id: Optional[int] = None,
    context: str | None = None,
    data_store: Any = None,
) -> ToolOutcome:
    context_name = context or self._ctx
    store = data_store or self._data_store

    contact_details = {
        "first_name": first_name,
        "surname": surname,
        "email_address": email_address,
        "phone_number": phone_number,
        "whatsapp_number": whatsapp_number,
        "discord_id": discord_id,
        "slack_user_id": slack_user_id,
        "bio": bio,
        "job_title": job_title,
        "timezone": timezone,
        "rolling_summary": rolling_summary,
        "should_respond": should_respond,
        "response_policy": response_policy,
        "is_system": is_system,
        "custom_key": custom_key,
        "custom_hash": custom_hash,
        "user_id": user_id,
        "agent_id": agent_id,
    }

    updates_dict = strip_authoring_assistant_id(
        {k: v for k, v in contact_details.items() if v is not None},
    )
    if not updates_dict:
        raise ValueError("At least one contact detail must be provided for an update.")

    # Validate and normalize via Pydantic model (e.g. "" → None for unique fields)
    try:
        validated = Contact(contact_id=contact_id, **updates_dict)
    except ValidationError as e:
        msg = str(e)
        try:
            err = e.errors()[0]
            msg = err.get("msg", str(e))
            if err.get("type") == "value_error":
                ctx = err.get("ctx", {})
                if "error" in ctx:
                    msg = str(ctx["error"])
        except Exception:
            pass
        raise ValueError(msg) from e
    updates_dict = {
        k: getattr(validated, k)
        for k in updates_dict
        if getattr(validated, k) is not None
    }
    if not updates_dict:
        return ToolOutcome(output="No effective changes after normalization.")

    if _log_id is None:
        target_ids = unisdk.get_logs(
            context=context_name,
            filter=f"contact_id == {contact_id}",
            return_ids_only=True,
        )
        if not target_ids:
            raise ValueError(
                f"No contact found with contact_id {contact_id} to update.",
            )
        if len(target_ids) > 1:
            raise ValueError(
                f"Multiple contacts found with contact_id {contact_id}. Data integrity issue.",
            )
        log_to_update_id = target_ids[0]
    else:
        log_to_update_id = _log_id

    unisdk.update_logs(
        logs=[log_to_update_id],
        context=context_name,
        entries=updates_dict,
        overwrite=True,
    )
    try:
        rows = unisdk.get_logs(
            context=context_name,
            filter=f"contact_id == {contact_id}",
            limit=1,
            from_fields=self._allowed_fields(),
        )
        if rows:
            store.put(rows[0].entries)
    except Exception:
        pass

    # Fire-and-forget sync to backend for system contacts
    if timezone is not None:
        try:
            _maybe_sync_timezone_to_backend(
                self,
                contact_id,
                timezone,
                context=context_name,
                data_store=store,
            )
        except Exception:
            pass
    if bio is not None:
        try:
            _maybe_sync_bio_to_backend(
                self,
                contact_id,
                bio,
                context=context_name,
                data_store=store,
            )
        except Exception:
            pass
    if job_title is not None:
        try:
            _maybe_sync_job_title_to_backend(self, contact_id, job_title)
        except Exception:
            pass

    return {"outcome": "contact updated", "details": {"contact_id": contact_id}}


def delete_contact(
    self,
    *,
    contact_id: int,
    _log_id: Optional[int] = None,
    context: str | None = None,
    data_store: Any = None,
) -> ToolOutcome:
    context_name = context or self._ctx
    store = data_store or self._data_store
    from ..session_details import SESSION_DETAILS

    protected_contact_ids = {
        int(SESSION_DETAILS.self_contact_id),
        int(SESSION_DETAILS.boss_contact_id),
    }

    if _log_id is None:
        # Fetch with is_system to check for org member protection
        rows = unisdk.get_logs(
            context=context_name,
            filter=f"contact_id == {contact_id}",
            limit=2,
            from_fields=["contact_id", "is_system"],
        )
        if not rows:
            raise ValueError(
                f"No contact found with contact_id {contact_id} to delete.",
            )
        if len(rows) > 1:
            raise RuntimeError(
                f"Multiple contacts found with contact_id {contact_id}. Data integrity issue.",
            )
        row = rows[0]
        if row.entries.get("is_system"):
            raise RuntimeError(
                f"Cannot delete system contact with id {contact_id}. "
                "System contacts include the assistant, primary user, and org members.",
            )
        if context_name == self._ctx and contact_id in protected_contact_ids:
            raise RuntimeError("Cannot delete assistant self or boss system contacts.")
        resolved_id = row.id
    else:
        if context_name == self._ctx and contact_id in protected_contact_ids:
            raise RuntimeError("Cannot delete assistant self or boss system contacts.")
        resolved_id = _log_id

    # Snapshot Knowledge provenance debt before Contacts FK CASCADE pops
    # ``source_refs[*].contact_id``.
    try:
        from unify.common.stale_reason import StaleReason
        from unify.knowledge_manager.knowledge_manager import (
            mark_knowledge_stale_for_deleted_sources,
        )

        mark_knowledge_stale_for_deleted_sources(
            reasons=[
                StaleReason(
                    dep_kind="contact",
                    id=int(contact_id),
                    message=f"missing contact_id={int(contact_id)}",
                ),
            ],
        )
    except Exception:
        pass
    unisdk.delete_logs(context=context_name, logs=resolved_id)
    try:
        store.delete(contact_id)
    except Exception:
        pass
    return {"outcome": "contact deleted", "details": {"contact_id": contact_id}}


def merge_contacts(
    self,
    *,
    contact_id_1: int,
    contact_id_2: int,
    overrides: Optional[Dict[str, int]] = None,
    context: str | None = None,
    data_store: Any = None,
) -> ToolOutcome:
    context_name = context or self._ctx
    store = data_store or self._data_store
    if contact_id_1 == contact_id_2:
        raise ValueError("contact_id_1 and contact_id_2 must be distinct.")
    if overrides is not None and any(v not in (1, 2) for v in overrides.values()):
        raise ValueError(
            "Override values must be 1 or 2, referring to the corresponding contact id argument.",
        )
    overrides = overrides or {}

    rows = unisdk.get_logs(
        context=context_name,
        filter=f"contact_id in [{contact_id_1}, {contact_id_2}]",
        limit=2,
        from_fields=self._allowed_fields(),
    )
    if not rows or len(rows) < 2:
        present_ids: set[int] = set()
        for lg in rows or []:
            try:
                present_ids.add(int(lg.entries.get("contact_id")))
            except Exception:
                pass
        missing = contact_id_1 if contact_id_1 not in present_ids else contact_id_2
        raise ValueError(f"No contact found with contact_id {missing}.")

    by_id: Dict[int, Any] = {}
    for lg in rows:
        try:
            by_id[int(lg.entries.get("contact_id"))] = lg
        except Exception:
            continue
    log1 = by_id[contact_id_1]
    log2 = by_id[contact_id_2]

    keep_id = contact_id_1 if overrides.get("contact_id", 1) == 1 else contact_id_2
    delete_id = contact_id_2 if keep_id == contact_id_1 else contact_id_1
    from ..session_details import SESSION_DETAILS

    protected_contact_ids = {
        int(SESSION_DETAILS.self_contact_id),
        int(SESSION_DETAILS.boss_contact_id),
    }
    if delete_id in protected_contact_ids:
        raise RuntimeError(
            "Cannot delete assistant self or boss system contacts during merge.",
        )

    entries1 = log1.entries
    entries2 = log2.entries
    all_cols = set(entries1.keys()) | set(entries2.keys())
    all_cols.discard("contact_id")

    consolidated: Dict[str, Any] = {}
    for col in all_cols:
        if col.endswith("_emb"):
            continue
        if col in overrides:
            source = overrides[col]
            value = entries1.get(col) if source == 1 else entries2.get(col)
        else:
            value = (
                entries1.get(col)
                if entries1.get(col) is not None
                else entries2.get(col)
            )
        if value is not None:
            consolidated[col] = value

    builtin_updates = {
        k: v for k, v in consolidated.items() if k in self._BUILTIN_FIELDS
    }

    if builtin_updates:
        kept_log_id = getattr(by_id[keep_id], "id", None)
        update_contact(
            self,
            contact_id=keep_id,
            _log_id=kept_log_id,
            context=context_name,
            data_store=store,
            **{
                k: builtin_updates.get(k)
                for k in self._BUILTIN_FIELDS
                if k in builtin_updates
            },
        )

    # Rewrite transcripts BEFORE deleting the merged contact to avoid FK SET NULL
    try:
        ctxs = unisdk.get_active_context()
        read_ctx = ctxs.get("read")
    except Exception:
        read_ctx = None
    transcripts_ctx = f"{read_ctx}/Transcripts" if read_ctx else "Transcripts"

    try:
        referenced = unisdk.get_logs(
            context=transcripts_ctx,
            filter=f"(sender_id == {delete_id}) or ({delete_id} in receiver_ids)",
            limit=1,
            return_ids_only=True,
        )
    except Exception:
        referenced = []
    if referenced:
        from unify.manager_registry import ManagerRegistry  # local import

        tm = ManagerRegistry.get_transcript_manager()
        tm.update_contact_id(original_contact_id=delete_id, new_contact_id=keep_id)

    # Finally, delete the merged contact (FK SET NULL won't fire since no references remain)
    delete_log_id = getattr(by_id[delete_id], "id", None)
    delete_contact(
        self,
        contact_id=delete_id,
        _log_id=delete_log_id,
        context=context_name,
        data_store=store,
    )

    return {
        "outcome": "contacts merged successfully",
        "details": {"kept_contact_id": keep_id, "deleted_contact_id": delete_id},
    }
