from __future__ import annotations

from typing import Any, Dict, List

import unify
from unify.utils.http import RequestError

from ..knowledge_manager.types import ColumnType
from ..session_details import (
    DEFAULT_ASSISTANT_BIO,
    DEFAULT_ASSISTANT_EMAIL,
    DEFAULT_ASSISTANT_FIRST_NAME,
    DEFAULT_ASSISTANT_PHONE,
    DEFAULT_ASSISTANT_SURNAME,
    DEFAULT_USER_EMAIL,
    DEFAULT_USER_FIRST_NAME,
    DEFAULT_USER_SURNAME,
)


def _list_assistants(self) -> List[Dict[str, Any]]:
    """Return the list of assistants configured for the current account."""
    return unify.list_assistants()


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


def _resolve_assistant_details(self) -> Dict[str, Any] | None:
    """Resolve assistant details from SESSION_DETAILS, API, or return None for defaults.

    When SESSION_DETAILS has not been initialized (e.g., during tests),
    returns None to indicate defaults should be used, avoiding real API calls.

    Returns
    -------
    dict | None
        Assistant record dict if found, or None to use defaults.
    """
    from ..session_details import SESSION_DETAILS

    if not SESSION_DETAILS.is_initialized:
        return None

    # 1) Prefer the assistant provided by unity.init
    if SESSION_DETAILS.assistant_record is not None:
        return SESSION_DETAILS.assistant_record

    # 2) Otherwise map the active context (if numeric) onto the list index
    assistants = _list_assistants(self)
    ctxs = unify.get_active_context()
    read_ctx = ctxs.get("read")
    try:
        idx = int(read_ctx) if read_ctx is not None else 0
    except (TypeError, ValueError):
        idx = 0

    return assistants[idx] if idx < len(assistants) else None


def _resolve_user_details(self) -> Dict[str, Any]:
    """Resolve user details from SESSION_DETAILS, API, or defaults.

    When SESSION_DETAILS has not been initialized (e.g., during tests),
    returns default user info to avoid calling real APIs.

    Returns
    -------
    dict
        User info dict with first_name, last_name, email, and optionally phone_number.
    """
    from ..session_details import SESSION_DETAILS

    # If SESSION_DETAILS hasn't been initialized, use defaults.
    # This ensures tests don't call real APIs for user info.
    if not SESSION_DETAILS.is_initialized:
        return {
            "first_name": DEFAULT_USER_FIRST_NAME,
            "last_name": DEFAULT_USER_SURNAME,
            "email": DEFAULT_USER_EMAIL,
        }

    # In production (SESSION_DETAILS initialized), fetch real user info
    user_info: Dict[str, Any] = {}
    data: Any = unify.get_user_basic_info()
    mapped: Dict[str, Any] = {
        "first_name": data.get("first"),
        "last_name": data.get("last"),
        "email": data.get("email"),
        "bio": data.get("bio"),
        "timezone": data.get("timezone"),
    }
    user_info.update({k: v for k, v in mapped.items() if v is not None})

    if SESSION_DETAILS.assistant_record is not None:
        phone = SESSION_DETAILS.assistant_record.get("user_phone")
        mapped_extra: Dict[str, Any] = {
            "phone_number": phone,
        }
        user_info.update({k: v for k, v in mapped_extra.items() if v is not None})

    if user_info:
        return user_info

    return {
        "first_name": DEFAULT_USER_FIRST_NAME,
        "last_name": DEFAULT_USER_SURNAME,
        "email": DEFAULT_USER_EMAIL,
    }


def provision_assistant_contact(self, assistant_log) -> None:
    """Provision the assistant system contact (id == 0).

    Creates or updates the assistant contact using details resolved from
    SESSION_DETAILS, the Unify API, or default values.
    """
    selected = _resolve_assistant_details(self)

    # Build the canonical assistant record (real or dummy)
    if selected is not None:
        a = selected
        base_fields = {fld: None for fld in self._BUILTIN_FIELDS if fld != "contact_id"}
        base_fields["respond_to"] = True
        base_fields["response_policy"] = ""
        base_fields["is_system"] = True
        base_fields.update(
            {
                "first_name": a.get("first_name"),
                "surname": a.get("surname"),
                "email_address": a.get("email"),
                "phone_number": a.get("phone"),
                "bio": a.get("about"),
                "rolling_summary": None,
            },
        )
    else:
        base_fields = {fld: None for fld in self._BUILTIN_FIELDS if fld != "contact_id"}
        base_fields["respond_to"] = True
        base_fields["response_policy"] = ""
        base_fields["is_system"] = True
        base_fields.update(
            {
                "first_name": DEFAULT_ASSISTANT_FIRST_NAME,
                "surname": DEFAULT_ASSISTANT_SURNAME,
                "email_address": DEFAULT_ASSISTANT_EMAIL,
                "phone_number": DEFAULT_ASSISTANT_PHONE,
                "bio": DEFAULT_ASSISTANT_BIO,
                "rolling_summary": None,
            },
        )

    # Use fetched timezone if available, fallback to UTC
    if selected is not None:
        base_fields["timezone"] = selected.get("timezone") or "UTC"
    else:
        base_fields["timezone"] = "UTC"

    if assistant_log is not None:
        try:
            entries = assistant_log.entries
            fetched_bio = selected.get("about") if selected else None
            fetched_tz = selected.get("timezone") if selected else None

            needs_timezone = not entries.get("timezone")
            needs_bio = fetched_bio and entries.get("bio") != fetched_bio
            needs_is_system = entries.get("is_system") is not True

            if needs_timezone or needs_bio or needs_is_system:
                update_kwargs: Dict[str, Any] = {
                    "contact_id": 0,
                    "_log_id": assistant_log.id,
                }
                if needs_timezone:
                    update_kwargs["timezone"] = fetched_tz or "UTC"
                if needs_bio:
                    update_kwargs["bio"] = fetched_bio
                if needs_is_system:
                    update_kwargs["is_system"] = True
                self.update_contact(**update_kwargs)
            else:
                # Warm local cache when no change needed
                self._data_store.put(entries)
        except Exception:
            pass
        return

    # Insert the assistant row. Use try-except to handle race conditions where
    # another process creates the contact concurrently.
    try:
        self._create_contact(**base_fields)
    except ValueError as e:
        if "unique fields" in str(e):
            # Another process created the contact concurrently – that's fine
            pass
        else:
            raise
    except RequestError as e:
        # Backend returned 500 due to DB-level race condition – contact exists
        if e.response is not None and e.response.status_code == 500:
            pass
        else:
            raise


def provision_user_contact(self, user_log) -> None:
    """Provision the user system contact (id == 1).

    Creates or updates the user (boss) contact using details resolved from
    SESSION_DETAILS, the Unify API, or default values.
    """
    user_info = _resolve_user_details(self)

    base_fields: Dict[str, Any] = {
        fld: None
        for fld in self._BUILTIN_FIELDS
        if fld not in {"contact_id", "rolling_summary"}
    }
    base_fields["respond_to"] = True
    base_fields["is_system"] = True
    base_fields.update(
        {
            "first_name": user_info.get("first_name"),
            "surname": user_info.get("last_name"),
            "email_address": user_info.get("email"),
            "phone_number": user_info.get("phone_number"),
            "bio": user_info.get("bio"),
            "response_policy": self.USER_MANAGER_RESPONSE_POLICY,
        },
    )

    # Use fetched timezone if available, fallback to UTC
    base_fields["timezone"] = user_info.get("timezone") or "UTC"

    extra_fields = {
        k: v
        for k, v in user_info.items()
        if k
        not in {
            "first_name",
            "last_name",
            "email",
            "phone_number",
        }
    }
    if extra_fields:
        _ensure_columns_exist(self, extra_fields)

    if user_log is not None:
        try:
            entries = user_log.entries
            fetched_bio = user_info.get("bio")
            fetched_tz = user_info.get("timezone")

            needs_timezone = not entries.get("timezone")
            needs_bio = fetched_bio and entries.get("bio") != fetched_bio
            needs_is_system = entries.get("is_system") is not True

            if needs_timezone or needs_bio or needs_is_system:
                update_kwargs: Dict[str, Any] = {
                    "contact_id": 1,
                    "_log_id": user_log.id,
                }
                if needs_timezone:
                    update_kwargs["timezone"] = fetched_tz or "UTC"
                if needs_bio:
                    update_kwargs["bio"] = fetched_bio
                if needs_is_system:
                    update_kwargs["is_system"] = True
                self.update_contact(**update_kwargs)
            else:
                # Warm local cache when no change needed
                self._data_store.put(entries)
        except Exception:
            pass
        return

    # Insert the user row. Use try-except to handle race conditions where
    # another process creates the contact concurrently.
    try:
        self._create_contact(**{k: v for k, v in base_fields.items() if v is not None})
    except ValueError as e:
        if "unique fields" in str(e):
            # Another process created the contact concurrently – that's fine
            pass
        else:
            raise
    except RequestError as e:
        # Backend returned 500 due to DB-level race condition – contact exists
        if e.response is not None and e.response.status_code == 500:
            pass
        else:
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

    base_url = SETTINGS.UNIFY_BASE_URL
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

                needs_is_system = not entries.get("is_system")
                needs_bio = fetched_bio and entries.get("bio") != fetched_bio
                needs_timezone = not entries.get("timezone")

                if needs_is_system or needs_bio or needs_timezone:
                    update_kwargs: Dict[str, Any] = {
                        "contact_id": int(entries["contact_id"]),
                        "_log_id": log.id,
                    }
                    if needs_is_system:
                        update_kwargs["is_system"] = True
                    if needs_bio:
                        update_kwargs["bio"] = fetched_bio
                    if needs_timezone:
                        update_kwargs["timezone"] = fetched_tz or "UTC"
                    self.update_contact(**update_kwargs)
            else:
                # Create new contact for org member
                self._create_contact(
                    first_name=first_name,
                    surname=surname,
                    email_address=email,
                    bio=member.get("bio"),
                    timezone=member.get("timezone") or "UTC",
                    is_system=True,
                    respond_to=True,
                    response_policy="",
                )
        except Exception:
            # Best-effort: continue with other members
            continue
