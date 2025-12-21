from __future__ import annotations

from typing import Any, Dict, List

import unify

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


def fetch_assistant_info(self) -> List[Dict[str, Any]]:
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


def sync_assistant_contact(self, assistant_log) -> None:
    """Ensure assistant contact (id == 0) exists and is correct."""
    from ..session_details import SESSION_DETAILS

    # Determine which assistant record to use
    selected = None

    if SESSION_DETAILS.is_initialized:
        # 1) Prefer the assistant provided by unity.init
        if SESSION_DETAILS.assistant_record is not None:
            selected = SESSION_DETAILS.assistant_record
        else:
            # 2) Otherwise map the active context (if numeric) onto the list index
            assistants = fetch_assistant_info(self)
            ctxs = unify.get_active_context()
            read_ctx = ctxs.get("read")
            try:
                idx = int(read_ctx) if read_ctx is not None else 0
            except (TypeError, ValueError):
                idx = 0
            selected = assistants[idx] if idx < len(assistants) else None
    # If SESSION_DETAILS not initialized (e.g., tests), selected stays None → defaults

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

    # Temporary: hard-code system contacts to UTC so a timezone exists for these
    # canonical contacts until frontend configuration is available.
    base_fields["utc_offset_hours"] = 0.0

    if assistant_log is not None:
        try:
            entries = assistant_log.entries
            current = entries.get("utc_offset_hours")
            if current != 0.0:
                # Only update the timezone field to avoid clobbering other values
                self.update_contact(
                    contact_id=0,
                    utc_offset_hours=0.0,
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


def fetch_user_info(self) -> Dict[str, Any]:
    """Return basic information for the authenticated human user.

    When SESSION_DETAILS has not been initialized (e.g., during tests),
    returns default user info to avoid calling real APIs.
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


def sync_user_contact(self, user_log) -> None:
    """Ensure default user contact (id == 1) exists and is correct."""
    user_info = fetch_user_info(self)

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

    # Temporary: hard-code system contacts to UTC so a timezone exists for these
    # canonical contacts until frontend configuration is available.
    base_fields["utc_offset_hours"] = 0.0

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
            current = entries.get("utc_offset_hours")
            if current != 0.0:
                # Only update the timezone field to avoid clobbering other values
                self.update_contact(
                    contact_id=1,
                    utc_offset_hours=0.0,
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
