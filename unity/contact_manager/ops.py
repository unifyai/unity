from __future__ import annotations

from typing import Any, Dict, Optional

import unify
from pydantic import ValidationError

from ..common.log_utils import log as unity_log
from ..common.tool_outcome import ToolOutcome
from .types.contact import Contact
from .custom_columns import sanitize_custom_columns


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
) -> None:
    """
    Fire-and-forget sync of timezone to backend for system contacts.

    - contact_id=0 → sync to assistant
    - contact_id=1 or other is_system=True → sync to user via email
    """
    from .backend_sync import sync_assistant_timezone, sync_user_timezone

    assistant_id = _get_assistant_id()
    if assistant_id is None:
        return

    if contact_id == 0:
        sync_assistant_timezone(assistant_id, timezone)
        return

    # User or org member - need email and is_system check
    try:
        row = self._data_store.get(contact_id)
    except KeyError:
        row = None

    if row is None:
        try:
            rows = unify.get_logs(
                context=self._ctx,
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
) -> None:
    """
    Fire-and-forget sync of bio to backend for system contacts.

    - contact_id=0 → sync to assistant (as 'about')
    - contact_id=1 or other is_system=True → sync to user (as 'bio')
    """
    from .backend_sync import sync_assistant_about, sync_user_bio

    assistant_id = _get_assistant_id()
    if assistant_id is None:
        return

    if contact_id == 0:
        sync_assistant_about(assistant_id, bio)
        return

    # User or org member - need email and is_system check
    try:
        row = self._data_store.get(contact_id)
    except KeyError:
        row = None

    if row is None:
        try:
            rows = unify.get_logs(
                context=self._ctx,
                filter=f"contact_id == {contact_id}",
                limit=1,
                from_fields=["contact_id", "is_system", "email_address"],
            )
            row = rows[0].entries if rows else None
        except Exception:
            return

    if row and row.get("is_system") and row.get("email_address"):
        sync_user_bio(assistant_id, row["email_address"], bio)


def create_contact(
    self,
    *,
    first_name: Optional[str] = None,
    surname: Optional[str] = None,
    email_address: Optional[str] = None,
    phone_number: Optional[str] = None,
    bio: Optional[str] = None,
    timezone: Optional[str] = None,
    rolling_summary: Optional[str] = None,
    should_respond: bool = True,
    response_policy: Optional[str] = None,
    **kwargs: Any,
) -> ToolOutcome:
    if "kwargs" in kwargs:
        kwargs = {**kwargs, **kwargs.pop("kwargs")}

    contact_details = {
        "first_name": first_name,
        "surname": surname,
        "email_address": email_address,
        "phone_number": phone_number,
        "bio": bio,
        "timezone": timezone,
        "rolling_summary": rolling_summary,
        "should_respond": should_respond,
        "response_policy": response_policy,
        "is_system": False,
    }
    if contact_details["response_policy"] is None:
        contact_details["response_policy"] = self.DEFAULT_RESPONSE_POLICY

    if kwargs:
        safe_custom = sanitize_custom_columns(kwargs)
        contact_details.update(safe_custom)
        try:
            for k in safe_custom.keys():
                if k not in self._BUILTIN_FIELDS:
                    if hasattr(self, "_known_custom_fields"):
                        self._known_custom_fields.add(k)  # type: ignore[attr-defined]
        except Exception:
            pass

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
        context=self._ctx,
        **contact_details,
        new=True,
        mutable=True,
        add_to_all_context=self.include_in_multi_assistant_table,
    )
    try:
        self._data_store.put(log.entries)
    except Exception:
        pass
    return {
        "outcome": "contact created successfully",
        "details": {"contact_id": log.entries["contact_id"]},
    }


def update_contact(
    self,
    *,
    contact_id: int,
    first_name: Optional[str] = None,
    surname: Optional[str] = None,
    email_address: Optional[str] = None,
    phone_number: Optional[str] = None,
    bio: Optional[str] = None,
    timezone: Optional[str] = None,
    rolling_summary: Optional[str] = None,
    should_respond: Optional[bool] = None,
    response_policy: Optional[str] = None,
    _log_id: Optional[int] = None,
    **kwargs: Any,
) -> ToolOutcome:
    if "kwargs" in kwargs:
        kwargs = {**kwargs, **kwargs.pop("kwargs")}

    contact_details = {
        "first_name": first_name,
        "surname": surname,
        "email_address": email_address,
        "phone_number": phone_number,
        "bio": bio,
        "timezone": timezone,
        "rolling_summary": rolling_summary,
        "should_respond": should_respond,
        "response_policy": response_policy,
    }
    if kwargs:
        safe_custom = sanitize_custom_columns(kwargs)
        contact_details.update(safe_custom)
        try:
            for k in safe_custom.keys():
                if k not in self._BUILTIN_FIELDS:
                    if hasattr(self, "_known_custom_fields"):
                        self._known_custom_fields.add(k)  # type: ignore[attr-defined]
        except Exception:
            pass

    updates_dict = {k: v for k, v in contact_details.items() if v is not None}
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
        target_ids = unify.get_logs(
            context=self._ctx,
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

    unify.update_logs(
        logs=[log_to_update_id],
        context=self._ctx,
        entries=updates_dict,
        overwrite=True,
    )
    try:
        rows = unify.get_logs(
            context=self._ctx,
            filter=f"contact_id == {contact_id}",
            limit=1,
            from_fields=self._allowed_fields(),
        )
        if rows:
            self._data_store.put(rows[0].entries)
    except Exception:
        pass

    # Fire-and-forget sync to backend for system contacts
    if timezone is not None:
        try:
            _maybe_sync_timezone_to_backend(self, contact_id, timezone)
        except Exception:
            pass
    if bio is not None:
        try:
            _maybe_sync_bio_to_backend(self, contact_id, bio)
        except Exception:
            pass

    return {"outcome": "contact updated", "details": {"contact_id": contact_id}}


def delete_contact(
    self,
    *,
    contact_id: int,
    _log_id: Optional[int] = None,
) -> ToolOutcome:
    # Fast path: hard-coded protection for assistant and primary user
    if contact_id in (0, 1):
        raise RuntimeError("Cannot delete system contacts with id 0 or 1.")

    if _log_id is None:
        # Fetch with is_system to check for org member protection
        rows = unify.get_logs(
            context=self._ctx,
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
        resolved_id = row.id
    else:
        resolved_id = _log_id

    unify.delete_logs(context=self._ctx, logs=resolved_id)
    try:
        self._data_store.delete(contact_id)
    except Exception:
        pass
    return {"outcome": "contact deleted", "details": {"contact_id": contact_id}}


def merge_contacts(
    self,
    *,
    contact_id_1: int,
    contact_id_2: int,
    overrides: Optional[Dict[str, int]] = None,
) -> ToolOutcome:
    if contact_id_1 == contact_id_2:
        raise ValueError("contact_id_1 and contact_id_2 must be distinct.")
    if overrides is not None and any(v not in (1, 2) for v in overrides.values()):
        raise ValueError(
            "Override values must be 1 or 2, referring to the corresponding contact id argument.",
        )
    overrides = overrides or {}

    rows = unify.get_logs(
        context=self._ctx,
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
    if delete_id in (0, 1):
        raise RuntimeError("Cannot delete system contacts with id 0 or 1 during merge.")

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
    custom_updates = {
        k: v for k, v in consolidated.items() if k not in self._BUILTIN_FIELDS
    }

    if builtin_updates or custom_updates:
        kept_log_id = getattr(by_id[keep_id], "id", None)
        update_contact(
            self,
            contact_id=keep_id,
            _log_id=kept_log_id,
            **{
                k: builtin_updates.get(k)
                for k in self._BUILTIN_FIELDS
                if k in builtin_updates
            },
            **(custom_updates or {}),
        )

    # Rewrite transcripts BEFORE deleting the merged contact to avoid FK SET NULL
    try:
        ctxs = unify.get_active_context()
        read_ctx = ctxs.get("read")
    except Exception:
        read_ctx = None
    transcripts_ctx = f"{read_ctx}/Transcripts" if read_ctx else "Transcripts"

    try:
        referenced = unify.get_logs(
            context=transcripts_ctx,
            filter=f"(sender_id == {delete_id}) or ({delete_id} in receiver_ids)",
            limit=1,
            return_ids_only=True,
        )
    except Exception:
        referenced = []
    if referenced:
        from unity.manager_registry import ManagerRegistry  # local import

        tm = ManagerRegistry.get_transcript_manager()
        tm.update_contact_id(original_contact_id=delete_id, new_contact_id=keep_id)

    # Finally, delete the merged contact (FK SET NULL won't fire since no references remain)
    delete_log_id = getattr(by_id[delete_id], "id", None)
    delete_contact(self, contact_id=delete_id, _log_id=delete_log_id)

    return {
        "outcome": "contacts merged successfully",
        "details": {"kept_contact_id": keep_id, "deleted_contact_id": delete_id},
    }
