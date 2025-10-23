from __future__ import annotations

from typing import Any, Dict, List

import unify

from ..knowledge_manager.types import ColumnType


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


def sync_assistant_contact(self) -> None:
    """Ensure assistant contact (id == 0) exists and is correct."""
    from .. import ASSISTANT as _GLOBAL_ASSISTANT  # local import to avoid cycles

    assistants = fetch_assistant_info(self)

    # 1) Prefer the assistant provided by unity.init
    if _GLOBAL_ASSISTANT is not None:
        selected = _GLOBAL_ASSISTANT
    else:
        # 2) Otherwise map the active context (if numeric) onto the list index
        ctxs = unify.get_active_context()
        read_ctx = ctxs.get("read")
        try:
            idx = int(read_ctx) if read_ctx is not None else 0
        except (TypeError, ValueError):
            idx = 0
        selected = assistants[idx] if idx < len(assistants) else None

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
                "whatsapp_number": a.get("phone"),
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
                "first_name": "Unify",
                "surname": "Assistant",
                "email_address": "unify.assistant@unify.ai",
                "phone_number": "+10000000000",
                "whatsapp_number": "+10000000000",
                "bio": "Your helpful Unify AI assistant.",
                "rolling_summary": None,
            },
        )

    existing_logs = unify.get_logs(
        context=self._ctx,
        filter="contact_id == 0",
        limit=1,
        from_fields=self._allowed_fields(),
    )

    if existing_logs:
        try:
            self._data_store.put(existing_logs[0].entries)
        except Exception:
            pass
        return

    # Insert the assistant row
    if not unify.get_logs(context=self._ctx, limit=1, return_ids_only=True):
        self._create_contact(**base_fields)
    else:
        log = unify.log(
            context=self._ctx,
            contact_id=0,
            **base_fields,
            new=True,
            mutable=True,
        )
        try:
            self._data_store.put(log.entries)
        except Exception:
            pass


def fetch_user_info(self) -> Dict[str, Any]:
    """Return basic information for the authenticated human user."""
    user_info: Dict[str, Any] = {}
    data: Any = unify.get_user_basic_info()
    mapped: Dict[str, Any] = {
        "first_name": data.get("first"),
        "last_name": data.get("last"),
        "email": data.get("email"),
    }
    user_info.update({k: v for k, v in mapped.items() if v is not None})

    from .. import ASSISTANT

    if ASSISTANT is not None:
        phone = ASSISTANT.get("user_phone")
        whatsapp = ASSISTANT.get("user_whatsapp_number")
        mapped_extra: Dict[str, Any] = {
            "phone_number": phone,
            "whatsapp_number": whatsapp,
        }
        user_info.update({k: v for k, v in mapped_extra.items() if v is not None})

    if user_info:
        return user_info

    return {
        "first_name": "John",
        "last_name": "Doe",
        "email": "john.doe@email.com",
    }


def sync_user_contact(self) -> None:
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
            "whatsapp_number": user_info.get("whatsapp_number"),
            "response_policy": self.USER_MANAGER_RESPONSE_POLICY,
        },
    )

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
        }
    }
    if extra_fields:
        _ensure_columns_exist(self, extra_fields)

    existing_logs = unify.get_logs(
        context=self._ctx,
        filter="contact_id == 1",
        limit=1,
        from_fields=self._allowed_fields(),
    )

    if existing_logs:
        try:
            self._data_store.put(existing_logs[0].entries)
        except Exception:
            pass
        return

    self._create_contact(**{k: v for k, v in base_fields.items() if v is not None})
