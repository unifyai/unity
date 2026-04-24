from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import unify
from unify.utils.http import RequestError

_log = logging.getLogger(__name__)

from ..common.context_registry import HIVE_CONTEXT_PREFIX
from ..knowledge_manager.types import ColumnType
from ..session_details import (
    PLACEHOLDER_ASSISTANT_BIO,
    PLACEHOLDER_ASSISTANT_EMAIL,
    PLACEHOLDER_ASSISTANT_FIRST_NAME,
    PLACEHOLDER_ASSISTANT_PHONE,
    PLACEHOLDER_ASSISTANT_SURNAME,
    PLACEHOLDER_USER_EMAIL,
    PLACEHOLDER_USER_FIRST_NAME,
    PLACEHOLDER_USER_SURNAME,
)
from .settings import (
    RELATIONSHIP_BOSS,
    RELATIONSHIP_SELF,
)


def _ensure_columns_exist(self, extra_fields: Dict[str, Any]) -> None:
    """Create custom columns for *extra_fields* that are not yet present."""
    existing_cols = self._get_columns()
    for col in extra_fields:
        if col in self._REQUIRED_COLUMNS or col in existing_cols:
            continue
        try:
            # Default to string type for new assistant/user metadata columns
            self._create_custom_column(
                column_name=col,
                column_type=ColumnType.str,
            )
        except Exception:
            # Column may have been created concurrently – ignore
            pass


def _is_assistant_populated() -> bool:
    """Return True if SESSION_DETAILS has real assistant profile data."""
    from ..session_details import SESSION_DETAILS

    if not SESSION_DETAILS.is_initialized:
        return False
    return bool(SESSION_DETAILS.assistant.first_name)


def _shared_contact_row_by_assistant_id(self, agent_id: int) -> Optional[Any]:
    """Return the shared ``Contacts`` row whose ``assistant_id`` names *agent_id*.

    Both ``provision_assistant_contact`` and ``provision_self_contact``
    need this lookup to decide whether the current body already owns a
    row in the shared Hive-contacts table — without it, a second body
    joining a Hive would mint a duplicate self contact alongside the
    first body's row.
    """
    try:
        rows = unify.get_logs(
            context=self._ctx,
            filter=f"assistant_id == {int(agent_id)}",
            limit=1,
            from_fields=["contact_id", "assistant_id"],
        )
    except RequestError as exc:
        _log.warning(
            "assistant_id lookup failed on %s for agent_id=%s: %s",
            self._ctx,
            agent_id,
            exc,
        )
        return None
    return rows[0] if rows else None


def _resolve_user_details(self) -> Dict[str, Any]:
    """Resolve user details from SESSION_DETAILS, API, or defaults.

    When SESSION_DETAILS has not been initialized (e.g., during tests),
    returns default user info to avoid calling real APIs.

    In DEMO_MODE, returns empty details because the boss (contact_id==1)
    is the prospect being demoed to — their details are unknown at startup
    and will be learned organically during the demo conversation.

    Returns
    -------
    dict
        User info dict with first_name, last_name, email, and optionally phone_number.
    """
    from ..session_details import SESSION_DETAILS
    from ..settings import SETTINGS

    # In demo mode, there is no real user account backing contact_id==1.
    # The prospect's details will be populated during the demo via
    # set_boss_details / inline communication tools.
    if SETTINGS.DEMO_MODE:
        return {}

    # If SESSION_DETAILS hasn't been initialized, use defaults.
    # This ensures tests don't call real APIs for user info.
    if not SESSION_DETAILS.is_initialized:
        return {
            "first_name": PLACEHOLDER_USER_FIRST_NAME,
            "last_name": PLACEHOLDER_USER_SURNAME,
            "email": PLACEHOLDER_USER_EMAIL,
        }

    # In production (SESSION_DETAILS initialized), fetch real user info
    try:
        data: Any = unify.get_user_basic_info()
    except Exception:
        _log.warning(
            "Failed to fetch user details from Orchestra, using session details",
        )
        return {
            "first_name": SESSION_DETAILS.user.first_name
            or PLACEHOLDER_USER_FIRST_NAME,
            "last_name": SESSION_DETAILS.user.surname or PLACEHOLDER_USER_SURNAME,
            "email": SESSION_DETAILS.user.email or PLACEHOLDER_USER_EMAIL,
        }

    user_info: Dict[str, Any] = {}
    mapped: Dict[str, Any] = {
        "first_name": data.get("first"),
        "last_name": data.get("last"),
        "email": data.get("email"),
        "bio": data.get("bio"),
        "timezone": data.get("timezone"),
        "phone_number": data.get("phone_number"),
        "whatsapp_number": data.get("whatsapp_number"),
        "discord_id": data.get("discord_id"),
    }
    user_info.update({k: v for k, v in mapped.items() if v is not None})

    if "phone_number" not in user_info and SESSION_DETAILS.user.number:
        user_info["phone_number"] = SESSION_DETAILS.user.number

    if "whatsapp_number" not in user_info and SESSION_DETAILS.user.whatsapp_number:
        user_info["whatsapp_number"] = SESSION_DETAILS.user.whatsapp_number

    if user_info:
        return user_info

    return {
        "first_name": PLACEHOLDER_USER_FIRST_NAME,
        "last_name": PLACEHOLDER_USER_SURNAME,
        "email": PLACEHOLDER_USER_EMAIL,
    }


def provision_assistant_contact(self, assistant_log) -> None:
    """Seed (or refresh) the shared ``Contacts`` row representing this body.

    Values come from ``SESSION_DETAILS`` when the assistant record is
    populated and from placeholder defaults otherwise. The authoritative
    "this row is me" signal is the ``"self"`` ``ContactMembership``
    overlay materialized by :func:`provision_self_contact`; this helper
    only ensures the shared row exists.
    """
    from ..session_details import SESSION_DETAILS

    populated = _is_assistant_populated()
    ast = SESSION_DETAILS.assistant

    base_fields = {
        fld: None
        for fld in self._BUILTIN_FIELDS
        if fld not in {"contact_id", "is_system", "assistant_id"}
    }
    base_fields["should_respond"] = True
    base_fields["response_policy"] = ""
    base_fields.update(
        {
            "first_name": (
                ast.first_name if populated else PLACEHOLDER_ASSISTANT_FIRST_NAME
            ),
            "surname": ast.surname if populated else PLACEHOLDER_ASSISTANT_SURNAME,
            "email_address": ast.email if populated else PLACEHOLDER_ASSISTANT_EMAIL,
            "phone_number": ast.number if populated else PLACEHOLDER_ASSISTANT_PHONE,
            "whatsapp_number": (
                ast.whatsapp_number if populated and ast.whatsapp_number else None
            ),
            "discord_id": (
                ast.discord_bot_id if populated and ast.discord_bot_id else None
            ),
            "bio": ast.about if populated else PLACEHOLDER_ASSISTANT_BIO,
            "job_title": (ast.job_title or None) if populated else None,
            "timezone": (ast.timezone or "UTC") if populated else "UTC",
            "rolling_summary": None,
        },
    )
    base_fields["_is_system"] = True
    base_fields["_relationship"] = RELATIONSHIP_SELF
    if ast.agent_id is not None:
        base_fields["_assistant_id"] = int(ast.agent_id)

        # The caller only probed ``contact_id == 0`` — which belongs to
        # the Hive's first body. Every subsequent body must find its own
        # row by ``assistant_id`` and reject the seeded row when it is
        # already owned by a Hive-mate, otherwise the placeholder update
        # below would leave a duplicate self contact behind.
        own_row = _shared_contact_row_by_assistant_id(self, ast.agent_id)
        if own_row is not None:
            assistant_log = own_row
        elif assistant_log is not None:
            seeded_assistant_id = assistant_log.entries.get("assistant_id")
            if seeded_assistant_id is not None and seeded_assistant_id != int(
                ast.agent_id,
            ):
                assistant_log = None

    if assistant_log is not None:
        try:
            entries = assistant_log.entries
            fetched_bio = ast.about if populated else None
            fetched_tz = ast.timezone if populated else None
            fetched_phone = ast.number if populated else None
            fetched_whatsapp = (
                ast.whatsapp_number if populated and ast.whatsapp_number else None
            )
            fetched_discord = (
                ast.discord_bot_id if populated and ast.discord_bot_id else None
            )
            fetched_first_name = ast.first_name if populated else None
            fetched_surname = ast.surname if populated else None
            fetched_job_title = (ast.job_title or None) if populated else None

            needs_timezone = fetched_tz and entries.get("timezone") != fetched_tz
            needs_bio = fetched_bio and entries.get("bio") != fetched_bio
            needs_job_title = (
                populated and (entries.get("job_title") or None) != fetched_job_title
            )
            needs_phone = fetched_phone and entries.get("phone_number") != fetched_phone
            needs_whatsapp = (
                fetched_whatsapp and entries.get("whatsapp_number") != fetched_whatsapp
            )
            needs_discord = (
                fetched_discord and entries.get("discord_id") != fetched_discord
            )
            needs_is_system = entries.get("is_system") is not True
            needs_first_name = (
                fetched_first_name and entries.get("first_name") != fetched_first_name
            )
            needs_surname = (
                fetched_surname and entries.get("surname") != fetched_surname
            )

            if (
                needs_timezone
                or needs_bio
                or needs_job_title
                or needs_phone
                or needs_whatsapp
                or needs_discord
                or needs_is_system
                or needs_first_name
                or needs_surname
            ):
                update_kwargs: Dict[str, Any] = {
                    "contact_id": 0,
                    "_log_id": assistant_log.id,
                }
                if needs_timezone:
                    update_kwargs["timezone"] = fetched_tz
                if needs_bio:
                    update_kwargs["bio"] = fetched_bio
                if needs_job_title:
                    update_kwargs["job_title"] = fetched_job_title
                if needs_phone:
                    update_kwargs["phone_number"] = fetched_phone
                if needs_whatsapp:
                    update_kwargs["whatsapp_number"] = fetched_whatsapp
                if needs_discord:
                    update_kwargs["discord_id"] = fetched_discord
                if needs_is_system:
                    update_kwargs["is_system"] = True
                if needs_first_name:
                    update_kwargs["first_name"] = fetched_first_name
                if needs_surname:
                    update_kwargs["surname"] = fetched_surname
                self.update_contact(**update_kwargs)
            else:
                # Warm local cache when no change needed
                self._data_store.put(entries)
        except Exception:
            pass
        return

    # Insert the assistant row. Race conditions are handled by Orchestra's
    # field-level uniqueness enforcement on email_address / phone_number.
    try:
        self._create_contact(**base_fields)
    except RequestError as e:
        if e.response is not None and e.response.status_code in (400, 500):
            detail = ""
            try:
                detail = str(e.response.json().get("detail", ""))
            except Exception:
                detail = str(getattr(e.response, "text", ""))
            if "unique" in detail.lower():
                return
        raise


def provision_user_contact(self, user_log) -> None:
    """Provision the user system contact (id == 1).

    Creates or updates the user (boss) contact using details resolved from
    SESSION_DETAILS, the Unify API, or default values.

    In DEMO_MODE, the boss contact is the prospect being demoed to. If the
    contact already exists (from a previous session), we preserve whatever
    details were set during the demo (name, phone, email) and only warm
    the local cache. If it doesn't exist yet, we create a minimal placeholder
    with should_respond=True so communication tools work immediately.
    """
    from ..settings import SETTINGS

    if SETTINGS.DEMO_MODE:
        if user_log is not None:
            # Contact already exists — preserve all details set during the demo.
            # Only ensure is_system is True (warm cache either way).
            try:
                entries = user_log.entries
                if entries.get("is_system") is not True:
                    self.update_contact(
                        contact_id=1,
                        _log_id=user_log.id,
                        is_system=True,
                    )
                else:
                    self._data_store.put(entries)
            except Exception:
                pass
            return
        # No existing contact — create a minimal placeholder.
        try:
            self._create_contact(
                should_respond=True,
                _is_system=True,
                _relationship=RELATIONSHIP_BOSS,
                response_policy=self.USER_MANAGER_RESPONSE_POLICY,
                timezone="UTC",
            )
        except (ValueError, RequestError):
            pass
        return

    user_info = _resolve_user_details(self)

    base_fields: Dict[str, Any] = {
        fld: None
        for fld in self._BUILTIN_FIELDS
        if fld not in {"contact_id", "rolling_summary", "is_system", "assistant_id"}
    }
    base_fields["should_respond"] = True
    base_fields.update(
        {
            "first_name": user_info.get("first_name"),
            "surname": user_info.get("last_name"),
            "email_address": user_info.get("email"),
            "phone_number": user_info.get("phone_number"),
            "whatsapp_number": user_info.get("whatsapp_number"),
            "discord_id": user_info.get("discord_id"),
            "bio": user_info.get("bio"),
            "response_policy": self.USER_MANAGER_RESPONSE_POLICY,
        },
    )
    base_fields["_is_system"] = True
    base_fields["_relationship"] = RELATIONSHIP_BOSS

    # Use fetched timezone if available, fallback to UTC
    base_fields["timezone"] = user_info.get("timezone") or "UTC"

    # Store the platform user_id for cost attribution (contact_id -> user_id mapping)
    from ..session_details import SESSION_DETAILS

    if SESSION_DETAILS.is_initialized and SESSION_DETAILS.user.id:
        base_fields["user_id"] = SESSION_DETAILS.user.id

    extra_fields = {
        k: v
        for k, v in user_info.items()
        if k
        not in {
            "first_name",
            "last_name",
            "email",
            "phone_number",
            "whatsapp_number",
            "discord_id",
        }
    }
    if extra_fields:
        _ensure_columns_exist(self, extra_fields)

    if user_log is not None:
        try:
            entries = user_log.entries
            fetched_bio = user_info.get("bio")
            fetched_tz = user_info.get("timezone")
            fetched_phone = user_info.get("phone_number")
            fetched_whatsapp = user_info.get("whatsapp_number")
            fetched_discord = user_info.get("discord_id")

            needs_timezone = fetched_tz and entries.get("timezone") != fetched_tz
            needs_bio = fetched_bio and entries.get("bio") != fetched_bio
            needs_phone = fetched_phone and entries.get("phone_number") != fetched_phone
            needs_whatsapp = (
                fetched_whatsapp and entries.get("whatsapp_number") != fetched_whatsapp
            )
            needs_discord = (
                fetched_discord and entries.get("discord_id") != fetched_discord
            )
            needs_is_system = entries.get("is_system") is not True

            if (
                needs_timezone
                or needs_bio
                or needs_phone
                or needs_whatsapp
                or needs_discord
                or needs_is_system
            ):
                update_kwargs: Dict[str, Any] = {
                    "contact_id": 1,
                    "_log_id": user_log.id,
                }
                if needs_timezone:
                    update_kwargs["timezone"] = fetched_tz
                if needs_bio:
                    update_kwargs["bio"] = fetched_bio
                if needs_phone:
                    update_kwargs["phone_number"] = fetched_phone
                if needs_whatsapp:
                    update_kwargs["whatsapp_number"] = fetched_whatsapp
                if needs_discord:
                    update_kwargs["discord_id"] = fetched_discord
                if needs_is_system:
                    update_kwargs["is_system"] = True
                self.update_contact(**update_kwargs)
            else:
                # Warm local cache when no change needed
                self._data_store.put(entries)
        except Exception:
            pass
        return

    # Insert the user row. Race conditions are handled by Orchestra's
    # field-level uniqueness enforcement on email_address / phone_number.
    try:
        self._create_contact(**{k: v for k, v in base_fields.items() if v is not None})
    except RequestError as e:
        if e.response is not None and e.response.status_code in (400, 500):
            detail = ""
            try:
                detail = str(e.response.json().get("detail", ""))
            except Exception:
                detail = str(getattr(e.response, "text", ""))
            if "unique" in detail.lower():
                return
        raise


def _fetch_org_members() -> List[Dict[str, Any]]:
    """
    Return list of org members for the current organization.

    Uses GET /organizations/members
    Returns empty list if:
    - Personal API key (not org)
    - API unavailable
    - Any error
    """
    from ..session_details import SESSION_DETAILS
    from ..settings import SETTINGS

    base_url = SETTINGS.ORCHESTRA_URL
    api_key = SESSION_DETAILS.unify_key

    if not base_url or not api_key:
        return []

    try:
        from unify.utils import http

        url = f"{base_url}/organizations/members"
        headers = {"Authorization": f"Bearer {api_key}"}
        resp = http.get(url, headers=headers, timeout=30)

        if 200 <= resp.status_code < 300:
            return resp.json() or []
        return []
    except Exception:
        return []


def _overlay_contact_id_for_relationship(
    self,
    relationship: str,
) -> Optional[int]:
    """Return the ``contact_id`` of this body's overlay row for *relationship*.

    Reads the per-body ``ContactMembership`` overlay, returning
    ``None`` when no row exists for that relationship (callers treat
    absence as "overlay not materialized yet" rather than falling
    back to any magic ``contact_id``).
    """
    try:
        rows = unify.get_logs(
            context=self._membership_ctx,
            filter=f"relationship == '{relationship}'",
            limit=1,
            from_fields=["contact_id", "relationship"],
        )
    except Exception:
        return None
    if not rows:
        return None
    try:
        return int(rows[0].entries.get("contact_id"))
    except Exception:
        return None


def _boss_contact_id_for_body(self) -> Optional[int]:
    """Return the contact_id this body treats as the boss, if any."""
    return _overlay_contact_id_for_relationship(self, RELATIONSHIP_BOSS)


def _self_overlay_exists(self) -> bool:
    """Return True when this body already has a ``"self"`` overlay."""
    return _overlay_contact_id_for_relationship(self, RELATIONSHIP_SELF) is not None


def provision_self_contact(self) -> None:
    """Ensure this body has a ``"self"`` ContactMembership overlay.

    Idempotent: if the per-body ``ContactMembership`` overlay already
    has a row with ``relationship == "self"``, this returns without
    doing any work.

    Otherwise it locates (or creates) the shared ``Contact`` row that
    represents this body — matched by ``assistant_id == <session
    agent_id>`` — and materializes the ``"self"`` overlay on top. The
    shared row carries ``assistant_id`` so cross-body readers can
    recognise it as a Hive-member body; the per-body overlay encodes
    this body's *policy* toward that row (``should_respond=True``,
    ``relationship="self"``, etc).
    """
    if _self_overlay_exists(self):
        return

    from ..session_details import SESSION_DETAILS

    populated = _is_assistant_populated()
    ast = SESSION_DETAILS.assistant
    agent_id = (
        getattr(ast, "agent_id", None) if SESSION_DETAILS.is_initialized else None
    )

    shared_contact_id: Optional[int] = None

    if agent_id is not None:
        existing = _shared_contact_row_by_assistant_id(self, agent_id)
        if existing is not None:
            shared_contact_id = int(existing.entries.get("contact_id"))

    if shared_contact_id is None:
        # Claim the seeded ``contact_id == 0`` row only when it is
        # genuinely unstamped (legacy / freshly provisioned) or already
        # belongs to this body. A row stamped with a different
        # ``assistant_id`` represents a Hive-mate body's self row — this
        # body must mint its own shared row instead.
        try:
            seeded = unify.get_logs(
                context=self._ctx,
                filter="contact_id == 0",
                limit=1,
                from_fields=["contact_id", "assistant_id"],
            )
        except Exception:
            seeded = []
        if seeded:
            seeded_assistant_id = seeded[0].entries.get("assistant_id")
            same_body = agent_id is not None and seeded_assistant_id == int(agent_id)
            if seeded_assistant_id is None or same_body:
                try:
                    shared_contact_id = int(seeded[0].entries.get("contact_id"))
                except Exception:
                    shared_contact_id = None
                if (
                    shared_contact_id is not None
                    and seeded_assistant_id is None
                    and agent_id is not None
                ):
                    try:
                        self.update_contact(
                            contact_id=shared_contact_id,
                            assistant_id=int(agent_id),
                        )
                    except Exception:
                        pass

    if shared_contact_id is None:
        create_kwargs: Dict[str, Any] = {
            "first_name": (
                ast.first_name if populated else PLACEHOLDER_ASSISTANT_FIRST_NAME
            ),
            "surname": ast.surname if populated else PLACEHOLDER_ASSISTANT_SURNAME,
            "email_address": (ast.email if populated else PLACEHOLDER_ASSISTANT_EMAIL),
            "phone_number": (ast.number if populated else PLACEHOLDER_ASSISTANT_PHONE),
            "bio": ast.about if populated else PLACEHOLDER_ASSISTANT_BIO,
            "job_title": (ast.job_title or None) if populated else None,
            "timezone": (ast.timezone or "UTC") if populated else "UTC",
            "should_respond": True,
            "response_policy": "",
            "_is_system": True,
            "_relationship": RELATIONSHIP_SELF,
            "_assistant_id": int(agent_id) if agent_id is not None else None,
        }
        if populated and ast.whatsapp_number:
            create_kwargs["whatsapp_number"] = ast.whatsapp_number
        if populated and ast.discord_bot_id:
            create_kwargs["discord_id"] = ast.discord_bot_id

        try:
            outcome = self._create_contact(
                **{k: v for k, v in create_kwargs.items() if v is not None},
            )
            if isinstance(outcome, dict):
                details = outcome.get("details") or {}
                if "contact_id" in details:
                    shared_contact_id = int(details["contact_id"])
        except RequestError as e:
            if e.response is not None and e.response.status_code in (400, 500):
                detail = ""
                try:
                    detail = str(e.response.json().get("detail", ""))
                except Exception:
                    detail = str(getattr(e.response, "text", ""))
                if "unique" not in detail.lower():
                    raise
            else:
                raise

    if shared_contact_id is None:
        return

    try:
        self.materialize_membership_if_missing(
            shared_contact_id,
            relationship=RELATIONSHIP_SELF,
            should_respond=True,
            response_policy="",
        )
    except Exception:
        pass


def provision_hive_boss_contact(
    hive_id: int,
    *,
    email: Optional[str],
    first_name: Optional[str] = None,
    surname: Optional[str] = None,
    phone_number: Optional[str] = None,
    timezone: Optional[str] = None,
    bio: Optional[str] = None,
) -> Optional[int]:
    """Ensure a shared boss ``Contact`` row exists under ``Hives/{hive_id}/Contacts``.

    Orchestra's Hive-creation pipeline calls this helper once at Hive
    birth to seed the shared boss row — every Hive member body then
    materializes its own ``"boss"`` overlay against the same shared
    contact via :func:`provision_boss_overlay`.

    Idempotent by email: a subsequent call with the same email
    returns the existing contact_id instead of inserting a duplicate.
    ``assistant_id`` is intentionally left ``None`` — a boss row
    represents a human, not a Hive-member body.

    Returns the shared ``contact_id`` on success, ``None`` when the
    Hive's Contacts context cannot be resolved (e.g. hive_id=None or
    ContextRegistry not yet set up).
    """
    if not hive_id:
        return None

    hive_contacts_ctx = f"{HIVE_CONTEXT_PREFIX}{int(hive_id)}/Contacts"

    if email:
        try:
            existing = unify.get_logs(
                context=hive_contacts_ctx,
                filter=f"email_address == '{email}'",
                limit=1,
                from_fields=["contact_id", "email_address"],
            )
            if existing:
                try:
                    return int(existing[0].entries.get("contact_id"))
                except Exception:
                    return None
        except Exception:
            pass

    entries: Dict[str, Any] = {
        "first_name": first_name,
        "surname": surname,
        "email_address": email,
        "phone_number": phone_number,
        "timezone": timezone or "UTC",
        "bio": bio,
        "is_system": True,
        "assistant_id": None,
    }

    from ..common.log_utils import log as unity_log

    try:
        log = unity_log(
            context=hive_contacts_ctx,
            **{k: v for k, v in entries.items() if v is not None},
            new=True,
            mutable=True,
        )
    except Exception as e:
        _log.warning(f"provision_hive_boss_contact: shared insert failed: {e}")
        return None

    try:
        return int(log.entries.get("contact_id"))
    except Exception:
        return None


def provision_boss_overlay(self) -> None:
    """Materialize a ``"boss"`` ContactMembership overlay on this body.

    Resolves the boss's shared ``Contact`` row — looking first for an
    existing row flagged ``is_system=True`` with the primary user's
    email (solo bodies and Hive members alike), then falling back to
    the active ``SESSION_DETAILS`` user-email as the match key — and
    writes a ``"boss"`` overlay against it on this body.

    Idempotent: when this body already has a ``"boss"`` overlay
    (regardless of which shared row it points at) the call is a no-op.
    """
    if _overlay_contact_id_for_relationship(self, RELATIONSHIP_BOSS) is not None:
        return

    from ..session_details import SESSION_DETAILS

    candidate_emails: list[str] = []
    if SESSION_DETAILS.is_initialized:
        u_email = getattr(SESSION_DETAILS.user, "email", None)
        if u_email:
            candidate_emails.append(str(u_email))

    shared_contact_id: Optional[int] = None
    for email in candidate_emails:
        try:
            matches = unify.get_logs(
                context=self._ctx,
                filter=f"email_address == '{email}' and is_system == True",
                limit=1,
                from_fields=["contact_id", "email_address", "is_system"],
            )
        except Exception:
            matches = []
        if matches:
            try:
                shared_contact_id = int(matches[0].entries.get("contact_id"))
                break
            except Exception:
                continue

    if shared_contact_id is None:
        # Fall back to the reserved ``contact_id == 1`` row that
        # ``provision_user_contact`` always seeds. A later merge_contacts
        # can still rewrite the overlay to point at a richer row.
        try:
            seeded = unify.get_logs(
                context=self._ctx,
                filter="contact_id == 1",
                limit=1,
                from_fields=["contact_id"],
            )
            if seeded:
                shared_contact_id = int(seeded[0].entries.get("contact_id"))
        except Exception:
            shared_contact_id = None

    if shared_contact_id is None:
        return

    try:
        self.materialize_membership_if_missing(
            shared_contact_id,
            relationship=RELATIONSHIP_BOSS,
            should_respond=True,
            response_policy=getattr(self, "USER_MANAGER_RESPONSE_POLICY", None),
        )
    except Exception:
        pass


def provision_system_overlays(self) -> None:
    """Eagerly mint this body's ``self`` + ``boss`` overlays.

    Called from :meth:`ContactManager.__init__` so every body starts
    life with a ``"self"`` and ``"boss"`` ContactMembership row. The
    overlay is the authoritative signal for every reader; the
    shared-row ``is_system`` flag only serves as a fallback when a
    body's overlay has not yet been materialized.
    """
    try:
        provision_self_contact(self)
    except Exception:
        pass
    try:
        provision_boss_overlay(self)
    except Exception:
        pass


def provision_org_member_contacts(self) -> None:
    """
    Ensure org member contacts exist with is_system=True.

    For each org member:
    - If contact with email exists: ensure is_system=True
    - If no contact exists: create with is_system=True

    Skips the primary user (identified via the per-body
    ``ContactMembership`` overlay's ``relationship == "boss"`` row)
    to avoid duplicating the boss contact.
    """
    members = _fetch_org_members()
    if not members:
        return

    # Resolve the boss email via the per-body overlay → shared row
    # lookup so the exclusion is keyed on this body's declared boss,
    # not a hard-coded ``contact_id == 1``.
    primary_user_email = None
    boss_contact_id = _boss_contact_id_for_body(self)
    if boss_contact_id is not None:
        try:
            boss_rows = unify.get_logs(
                context=self._ctx,
                filter=f"contact_id == {int(boss_contact_id)}",
                limit=1,
                from_fields=["email_address"],
            )
            if boss_rows:
                primary_user_email = boss_rows[0].entries.get("email_address")
        except Exception:
            pass

    for member in members:
        email = member.get("email")
        if not email:
            continue

        # Skip primary user (already synced as id=1)
        if primary_user_email and email.lower() == primary_user_email.lower():
            continue

        # Parse name into first/last
        full_name = member.get("name", "")
        name_parts = full_name.strip().split(maxsplit=1)
        first_name = name_parts[0] if name_parts else None
        surname = name_parts[1] if len(name_parts) > 1 else None

        try:
            # Check if contact with this email already exists
            existing = unify.get_logs(
                context=self._ctx,
                filter=f"email_address == '{email}'",
                limit=1,
            )

            if existing:
                log = existing[0]
                entries = log.entries
                fetched_bio = member.get("bio")
                fetched_tz = member.get("timezone")
                fetched_phone = member.get("phone_number")
                fetched_whatsapp = member.get("whatsapp_number")
                fetched_user_id = member.get("user_id")

                needs_is_system = not entries.get("is_system")
                needs_bio = fetched_bio and entries.get("bio") != fetched_bio
                needs_timezone = fetched_tz and entries.get("timezone") != fetched_tz
                needs_phone = (
                    fetched_phone and entries.get("phone_number") != fetched_phone
                )
                needs_whatsapp = (
                    fetched_whatsapp
                    and entries.get("whatsapp_number") != fetched_whatsapp
                )
                needs_user_id = (
                    fetched_user_id and entries.get("user_id") != fetched_user_id
                )

                if (
                    needs_is_system
                    or needs_bio
                    or needs_timezone
                    or needs_phone
                    or needs_whatsapp
                    or needs_user_id
                ):
                    update_kwargs: Dict[str, Any] = {
                        "contact_id": int(entries["contact_id"]),
                        "_log_id": log.id,
                    }
                    if needs_is_system:
                        update_kwargs["is_system"] = True
                    if needs_bio:
                        update_kwargs["bio"] = fetched_bio
                    if needs_timezone:
                        update_kwargs["timezone"] = fetched_tz
                    if needs_phone:
                        update_kwargs["phone_number"] = fetched_phone
                    if needs_whatsapp:
                        update_kwargs["whatsapp_number"] = fetched_whatsapp
                    if needs_user_id:
                        update_kwargs["user_id"] = fetched_user_id
                    self.update_contact(**update_kwargs)
            else:
                # Create new contact for org member
                create_kwargs: Dict[str, Any] = dict(
                    first_name=first_name,
                    surname=surname,
                    email_address=email,
                    phone_number=member.get("phone_number"),
                    whatsapp_number=member.get("whatsapp_number"),
                    bio=member.get("bio"),
                    timezone=member.get("timezone") or "UTC",
                    is_system=True,
                    should_respond=True,
                    response_policy="",
                )
                if member.get("user_id"):
                    create_kwargs["user_id"] = member["user_id"]
                self._create_contact(**create_kwargs)
        except Exception:
            # Best-effort: continue with other members
            continue
