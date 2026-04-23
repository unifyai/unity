from __future__ import annotations

from typing import Any, Dict, Final, Optional

import unify
from pydantic import ValidationError

from ..common.context_registry import ContextRegistry
from ..common.log_utils import log as unity_log
from ..common.tool_outcome import ToolOutcome
from .settings import (
    RELATIONSHIP_COWORKER,
    RELATIONSHIP_OTHER,
)
from .types.contact import Contact, ContactMembership
from .custom_columns import sanitize_custom_columns

# Audit-only fields on the shared Contact row: internal provenance
# stamps that should never count as a user-supplied "contact detail"
# in create/update surface-area guards.
_AUDIT_FIELDS: Final[frozenset[str]] = frozenset(
    {"assistant_id", "authoring_assistant_id", "is_system"},
)


def _get_assistant_id() -> int | None:
    """Get assistant_id from SESSION_DETAILS, returning None if unavailable."""
    from ..session_details import SESSION_DETAILS

    if not SESSION_DETAILS.is_initialized:
        return None
    return SESSION_DETAILS.assistant.agent_id


def _relationship_for(self, contact_id: int) -> Optional[str]:
    """Return this body's overlay relationship for *contact_id*, if any.

    Reads the per-body ContactMembership overlay. Returns ``None`` when
    no overlay row exists so callers can fall back to the shared-row
    ``is_system`` signal (pre-deploy solo bodies, brand-new Hive members
    that haven't minted their system overlays yet).
    """
    try:
        rows = unify.get_logs(
            context=self._membership_ctx,
            filter=f"contact_id == {int(contact_id)}",
            limit=1,
            from_fields=["contact_id", "relationship"],
        )
    except Exception:
        return None
    if not rows:
        return None
    try:
        return rows[0].entries.get("relationship")
    except Exception:
        return None


def _write_membership_overlay(
    self,
    *,
    contact_id: int,
    relationship: Optional[str] = None,
    should_respond: Optional[bool] = None,
    response_policy: Optional[str] = None,
    can_edit: Optional[bool] = None,
) -> None:
    """Upsert this body's ContactMembership row for a shared contact.

    Writes a single log to ``{user}/{assistant}/ContactMembership`` with
    only the supplied fields. If a row for *contact_id* already exists
    it is updated in place; otherwise a new row is inserted.
    ``overwrite=True`` ensures the provided fields replace prior values
    without clobbering fields the caller chose not to pass.
    """
    overlay_fields: Dict[str, Any] = {"contact_id": int(contact_id)}
    if relationship is not None:
        overlay_fields["relationship"] = relationship
    if should_respond is not None:
        overlay_fields["should_respond"] = bool(should_respond)
    if response_policy is not None:
        overlay_fields["response_policy"] = response_policy
    if can_edit is not None:
        overlay_fields["can_edit"] = bool(can_edit)

    try:
        existing = unify.get_logs(
            context=self._membership_ctx,
            filter=f"contact_id == {int(contact_id)}",
            limit=1,
            return_ids_only=True,
        )
    except Exception:
        existing = []

    if existing:
        unify.update_logs(
            logs=[existing[0]],
            context=self._membership_ctx,
            entries=overlay_fields,
            overwrite=True,
        )
    else:
        # Default the unspecified fields when minting a brand-new row so
        # downstream readers get a complete overlay (callers that want
        # partial state call update_logs on an existing row instead).
        if "relationship" not in overlay_fields:
            overlay_fields["relationship"] = RELATIONSHIP_OTHER
        if "should_respond" not in overlay_fields:
            overlay_fields["should_respond"] = True
        if "response_policy" not in overlay_fields:
            overlay_fields["response_policy"] = None
        if "can_edit" not in overlay_fields:
            overlay_fields["can_edit"] = True
        unity_log(
            context=self._membership_ctx,
            **overlay_fields,
            new=True,
            mutable=True,
        )


def _default_relationship_for(
    self,
    *,
    explicit: Optional[str],
    shared_assistant_id: Optional[int],
) -> str:
    """Resolve the default ``relationship`` for a freshly-created contact.

    Precedence:
    1. Explicit ``_relationship`` kwarg (used by system-contact
       provisioning to mint ``self`` / ``boss``).
    2. Shared row carries an ``assistant_id`` (the contact IS a
       Hive-member body) → ``coworker``.
    3. Fallback → ``other``.
    """
    if explicit is not None:
        return explicit
    if shared_assistant_id is not None:
        return RELATIONSHIP_COWORKER
    return RELATIONSHIP_OTHER


def _get_shared_row(
    self,
    contact_id: int,
    *,
    fields: list[str],
) -> Optional[Dict[str, Any]]:
    """Return the shared Contact row for *contact_id* via cache-then-backend."""
    try:
        return self._data_store.get(contact_id)
    except KeyError:
        pass
    try:
        rows = unify.get_logs(
            context=self._ctx,
            filter=f"contact_id == {int(contact_id)}",
            limit=1,
            from_fields=fields,
        )
    except Exception:
        return None
    return rows[0].entries if rows else None


def _maybe_sync_timezone_to_backend(
    self,
    contact_id: int,
    timezone: str,
) -> None:
    """Fire-and-forget sync of timezone to backend for system contacts.

    Routing is driven by the per-body ContactMembership overlay when
    present (``relationship == "self"`` → assistant backend,
    ``"boss"`` → user backend via email) and falls back to the shared
    ``is_system`` flag when the overlay has not yet been minted
    (pre-deploy solo bodies, freshly-provisioned Hive bodies).

    Contacts that represent another Hive-member body (shared row
    carries ``assistant_id``) are treated as ``"coworker"`` and are
    no-ops — we don't reach into another body's Assistant backend
    record from this body's update path.
    """
    from .backend_sync import sync_assistant_timezone, sync_user_timezone

    assistant_id = _get_assistant_id()
    if assistant_id is None:
        return

    relationship = _relationship_for(self, contact_id)
    if relationship == "self":
        sync_assistant_timezone(assistant_id, timezone)
        return
    if relationship == "coworker":
        return

    row = _get_shared_row(
        self,
        contact_id,
        fields=["contact_id", "is_system", "email_address", "assistant_id"],
    )
    if not row:
        return

    # A coworker shared row (assistant_id set) is never synced back
    # through this body's sync path even if the overlay is missing.
    if row.get("assistant_id") is not None:
        return

    if relationship == "boss" or row.get("is_system"):
        if row.get("email_address"):
            sync_user_timezone(assistant_id, row["email_address"], timezone)


def _maybe_sync_bio_to_backend(
    self,
    contact_id: int,
    bio: str,
) -> None:
    """Fire-and-forget sync of bio to backend for system contacts.

    Same overlay-driven routing as :func:`_maybe_sync_timezone_to_backend`:
    ``"self"`` → assistant ``about``; ``"boss"`` or fallback
    ``is_system + email`` → user ``bio``; coworker → no-op.
    """
    from .backend_sync import sync_assistant_about, sync_user_bio

    assistant_id = _get_assistant_id()
    if assistant_id is None:
        return

    relationship = _relationship_for(self, contact_id)
    if relationship == "self":
        sync_assistant_about(assistant_id, bio)
        return
    if relationship == "coworker":
        return

    row = _get_shared_row(
        self,
        contact_id,
        fields=["contact_id", "is_system", "email_address", "assistant_id"],
    )
    if not row:
        return

    if row.get("assistant_id") is not None:
        return

    if relationship == "boss" or row.get("is_system"):
        if row.get("email_address"):
            sync_user_bio(assistant_id, row["email_address"], bio)


def _maybe_sync_job_title_to_backend(
    self,
    contact_id: int,
    job_title: str,
) -> None:
    """Fire-and-forget sync of ``job_title`` to the assistant's backend.

    Only the contact that IS this body (``relationship == "self"``)
    flows its job title back to the Assistant table; for every other
    contact this value is purely body-local metadata.
    """
    from .backend_sync import sync_assistant_job_title

    relationship = _relationship_for(self, contact_id)
    if relationship != "self":
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
    bio: Optional[str] = None,
    job_title: Optional[str] = None,
    timezone: Optional[str] = None,
    rolling_summary: Optional[str] = None,
    should_respond: bool = True,
    response_policy: Optional[str] = None,
    _assistant_id: Optional[int] = None,
    _relationship: Optional[str] = None,
    _is_system: Optional[bool] = None,
    **kwargs: Any,
) -> ToolOutcome:
    """Create a shared Contact row and materialize this body's overlay.

    Two-stage write:

    1. Insert into the shared ``Contacts`` table — Hive-scoped for Hive
       members, per-body for solo bodies. The shared row stamps
       ``authoring_assistant_id`` with the caller's body so the row's
       origin is traceable for the lifetime of the contact. The caller
       may also pass ``_assistant_id`` when minting a contact that
       *represents* another Hive-member body (the shared row's
       ``assistant_id`` field).
    2. Materialize a :class:`ContactMembership` overlay on *this*
       body with ``relationship`` / ``should_respond`` /
       ``response_policy``. ``relationship`` defaults via
       :func:`_default_relationship_for` — the explicit ``_relationship``
       kwarg wins, otherwise a shared row with ``assistant_id`` set
       becomes ``"coworker"``, and everything else becomes ``"other"``.
    """
    if "kwargs" in kwargs:
        kwargs = {**kwargs, **kwargs.pop("kwargs")}

    if response_policy is None:
        response_policy = self.DEFAULT_RESPONSE_POLICY

    # Shared-row fields: identity, contactable details, and the audit
    # stamps. should_respond/response_policy are **not** written here —
    # they are body-local and live on the ContactMembership overlay.
    contact_details: Dict[str, Any] = {
        "first_name": first_name,
        "surname": surname,
        "email_address": email_address,
        "phone_number": phone_number,
        "bio": bio,
        "job_title": job_title,
        "timezone": timezone,
        "rolling_summary": rolling_summary,
        "is_system": bool(_is_system),
        "assistant_id": _assistant_id,
        "authoring_assistant_id": _get_assistant_id(),
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

    if not any(
        v is not None for k, v in contact_details.items() if k not in _AUDIT_FIELDS
    ):
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

    new_contact_id = int(log.entries["contact_id"])
    relationship = _default_relationship_for(
        self,
        explicit=_relationship,
        shared_assistant_id=_assistant_id,
    )
    try:
        _write_membership_overlay(
            self,
            contact_id=new_contact_id,
            relationship=relationship,
            should_respond=should_respond,
            response_policy=response_policy,
        )
    except Exception:
        # Overlay write must not fail the shared-row create. If the
        # overlay is missing, downstream readers fall back to shared-row
        # signals and the lazy-materialization helper will backfill it
        # at the next interaction boundary.
        pass

    return {
        "outcome": "contact created successfully",
        "details": {"contact_id": new_contact_id},
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
    bio: Optional[str] = None,
    job_title: Optional[str] = None,
    timezone: Optional[str] = None,
    rolling_summary: Optional[str] = None,
    should_respond: Optional[bool] = None,
    response_policy: Optional[str] = None,
    _log_id: Optional[int] = None,
    **kwargs: Any,
) -> ToolOutcome:
    """Update a contact, splitting shared-row and overlay-only fields.

    ``should_respond`` and ``response_policy`` are overlay-only and
    route to this body's :class:`ContactMembership` row (materialized
    if missing). Everything else — identity, contactable details, free
    text, custom columns — goes to the shared ``Contacts`` row as
    before. ``authoring_assistant_id`` is a write-once audit stamp and
    is never rewritten by updates.
    """
    if "kwargs" in kwargs:
        kwargs = {**kwargs, **kwargs.pop("kwargs")}

    # Split the shared-row fields from the overlay-only ones up front
    # so every downstream guard only ever sees its own slice.
    shared_kwargs: Dict[str, Any] = {
        "first_name": first_name,
        "surname": surname,
        "email_address": email_address,
        "phone_number": phone_number,
        "whatsapp_number": whatsapp_number,
        "bio": bio,
        "job_title": job_title,
        "timezone": timezone,
        "rolling_summary": rolling_summary,
    }
    overlay_kwargs: Dict[str, Any] = {
        "should_respond": should_respond,
        "response_policy": response_policy,
    }

    if kwargs:
        safe_custom = sanitize_custom_columns(kwargs)
        # authoring_assistant_id is a write-once audit field; drop it
        # defensively if a caller tries to sneak an override through
        # the custom-column path.
        safe_custom.pop("authoring_assistant_id", None)
        shared_kwargs.update(safe_custom)
        try:
            for k in safe_custom.keys():
                if k not in self._BUILTIN_FIELDS:
                    if hasattr(self, "_known_custom_fields"):
                        self._known_custom_fields.add(k)  # type: ignore[attr-defined]
        except Exception:
            pass

    shared_updates = {k: v for k, v in shared_kwargs.items() if v is not None}
    overlay_updates = {k: v for k, v in overlay_kwargs.items() if v is not None}

    if not shared_updates and not overlay_updates:
        raise ValueError("At least one contact detail must be provided for an update.")

    # Validate and normalize the shared-row slice via the Contact model
    # (e.g. "" → None for unique fields). The overlay fields are kept
    # separate and validated via ContactMembership.
    if shared_updates:
        try:
            validated = Contact(contact_id=contact_id, **shared_updates)
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
        shared_updates = {
            k: getattr(validated, k)
            for k in shared_updates
            if getattr(validated, k, None) is not None
        }

    if overlay_updates:
        # ContactMembership constructor exercises validation for
        # should_respond / response_policy shapes.
        ContactMembership(contact_id=contact_id, **overlay_updates)

    if not shared_updates and not overlay_updates:
        return ToolOutcome(output="No effective changes after normalization.")

    if shared_updates:
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
            entries=shared_updates,
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

    if overlay_updates:
        try:
            _write_membership_overlay(
                self,
                contact_id=contact_id,
                should_respond=overlay_updates.get("should_respond"),
                response_policy=overlay_updates.get("response_policy"),
            )
        except Exception:
            # Overlay write failures must not mask a successful shared
            # update; log-silence is acceptable here because the next
            # interaction boundary will backfill the overlay lazily.
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
) -> ToolOutcome:
    # Fast path: an overlay relationship of "self" or "boss" protects
    # the contact from deletion regardless of the shared-row flag.
    # contact_id 0/1 are no longer reserved — Hive auto-counting picks
    # whatever id is next — so the overlay is the authoritative signal.
    relationship = _relationship_for(self, contact_id)
    if relationship in ("self", "boss"):
        raise RuntimeError(
            f"Cannot delete system contact with id {contact_id} "
            f"(relationship={relationship}). "
            "System contacts represent the assistant itself or its primary human principal.",
        )

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
    delete_relationship = _relationship_for(self, delete_id)
    if delete_relationship in ("self", "boss"):
        raise RuntimeError(
            f"Cannot delete system contact with id {delete_id} "
            f"(relationship={delete_relationship}) during merge. "
            "Flip 'contact_id' in overrides to keep the system row instead.",
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
        transcripts_ctx = f"{ContextRegistry.base_for('Transcripts')}/Transcripts"
    except RuntimeError:
        transcripts_ctx = "Transcripts"

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
