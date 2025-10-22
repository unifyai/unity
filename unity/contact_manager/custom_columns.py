from __future__ import annotations

import re
from typing import Any, Dict, Optional

import unify

from ..knowledge_manager.types import ColumnType


def create_custom_column(
    self,
    *,
    column_name: str,
    column_type: ColumnType | str,
    column_description: Optional[str] = None,
) -> Dict[str, str]:
    """Create a mutable custom column on the contacts table."""
    assert (
        column_name not in self._REQUIRED_COLUMNS
    ), f"'{column_name}' is a required column and cannot be recreated."

    if not re.fullmatch(r"[a-z][a-z0-9_]*", column_name):
        raise ValueError(
            "column_name must be snake_case: start with a letter, then letters/digits/underscores",
        )

    column_info: Dict[str, Any] = {"type": str(column_type), "mutable": True}
    if column_description is not None:
        column_info["description"] = column_description

    response = unify.create_fields(fields={column_name: column_info}, context=self._ctx)

    # Best-effort hygiene for DataStore rows after schema changes is done by callers/tests
    try:
        if hasattr(self, "_known_custom_fields"):
            self._known_custom_fields.add(column_name)  # type: ignore[attr-defined]
    except Exception:
        pass

    return response


def delete_custom_column(self, *, column_name: str) -> Dict[str, str]:
    """Delete an existing custom column from the contacts table."""
    if column_name in self._REQUIRED_COLUMNS:
        raise ValueError(f"Cannot delete required column '{column_name}'.")

    response = unify.delete_fields(fields=[column_name], context=self._ctx)

    # Update local view of known custom columns on success & scrub DataStore
    try:
        if hasattr(self, "_known_custom_fields") and column_name in self._known_custom_fields:  # type: ignore[attr-defined]
            self._known_custom_fields.discard(column_name)  # type: ignore[attr-defined]
    except Exception:
        pass

    try:
        snap = self._data_store.snapshot()
        for _k, row in snap.items():
            if column_name in row:
                new_row = dict(row)
                del new_row[column_name]
                self._data_store.put(new_row)
    except Exception:
        try:
            self._data_store.clear()
        except Exception:
            pass

    return response


def sanitize_custom_columns(custom_columns: Dict[str, Any]) -> Dict[str, Any]:
    """Return a filtered copy of custom columns safe for JSON logging."""
    import json

    internal_keys = {
        "parent_chat_context",
        "interject_queue",
        "pause_event",
        "clarification_up_q",
        "clarification_down_q",
        "kwargs",
        "_log_id",
    }
    safe: Dict[str, Any] = {}
    for key, value in (custom_columns or {}).items():
        if key in internal_keys:
            continue
        try:
            json.dumps(value)
        except Exception:
            continue
        safe[key] = value
    return safe
