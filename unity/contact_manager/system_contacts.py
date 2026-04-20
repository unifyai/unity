from __future__ import annotations

import logging
from typing import Any, Dict, List

import unify
from unify.utils.http import RequestError

_log = logging.getLogger(__name__)

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
    """Provision the assistant system contact (id == 0).

    Creates or updates the assistant contact using details from
    SESSION_DETAILS or default values.
    """
    from ..session_details import SESSION_DETAILS

    populated = _is_assistant_populated()
    ast = SESSION_DETAILS.assistant

    base_fields = {fld: None for fld in self._BUILTIN_FIELDS if fld != "contact_id"}
    base_fields["should_respond"] = True
    base_fields["response_policy"] = ""
    base_fields["is_system"] = True
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
                is_system=True,
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
        if fld not in {"contact_id", "rolling_summary"}
    }
    base_fields["should_respond"] = True
    base_fields["is_system"] = True
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


def provision_org_member_contacts(self) -> None:
    """
    Ensure org member contacts exist with is_system=True.

    For each org member:
    - If contact with email exists: ensure is_system=True
    - If no contact exists: create with is_system=True

    Skips the primary user (id=1) to avoid duplicates.
    """
    members = _fetch_org_members()
    if not members:
        return

    # Get primary user email to skip
    primary_user_email = None
    try:
        primary_user_rows = unify.get_logs(
            context=self._ctx,
            filter="contact_id == 1",
            limit=1,
            from_fields=["email_address"],
        )
        if primary_user_rows:
            primary_user_email = primary_user_rows[0].entries.get("email_address")
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
