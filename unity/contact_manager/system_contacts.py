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

    # Hard-code system contacts to UTC so a timezone exists for these
    # canonical contacts until frontend configuration is available.
    base_fields["timezone"] = "UTC"

    if assistant_log is not None:
        try:
            entries = assistant_log.entries
            current = entries.get("timezone")
            if current != "UTC":
                # Only update the timezone field to avoid clobbering other values
                self.update_contact(
                    contact_id=0,
                    timezone="UTC",
                    _log_id=assistant_log.id,
                )
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
        if fld not in {"contact_id", "bio", "rolling_summary"}
    }
    base_fields["respond_to"] = True
    base_fields.update(
        {
            "first_name": user_info.get("first_name"),
            "surname": user_info.get("last_name"),
            "email_address": user_info.get("email"),
            "phone_number": user_info.get("phone_number"),
            "response_policy": self.USER_MANAGER_RESPONSE_POLICY,
        },
    )

    # Hard-code system contacts to UTC so a timezone exists for these
    # canonical contacts until frontend configuration is available.
    base_fields["timezone"] = "UTC"

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
            current = entries.get("timezone")
            if current != "UTC":
                # Only update the timezone field to avoid clobbering other values
                self.update_contact(
                    contact_id=1,
                    timezone="UTC",
                    _log_id=user_log.id,
                )
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
