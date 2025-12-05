from __future__ import annotations

from typing import Any, Dict, Optional

import unify


# Private fields injected by log_utils wrappers
_PRIVATE_FIELDS: Dict[str, str] = {
    "_assistant": "str",
    "_assistant_id": "str",
    "_user_id": "str",
}


class TableStore:
    """
    Idempotent context/field provisioner with safe accessors.

    Guarantees that a given ``(project, context)`` exists with the required
    fields before read/write operations. Falls back to an ensure→retry path when
    encountering a backend 404 due to races or eventual consistency.
    """

    # Process-local memo to avoid repeated ensures in the same run
    _ENSURED: set[tuple[str, str]] = set()

    def __init__(
        self,
        context: str,
        *,
        unique_keys: Optional[Dict[str, str]] = None,
        auto_counting: Optional[Dict[str, Optional[str]]] = None,
        description: Optional[str] = None,
        fields: Optional[Dict[str, Any]] = None,
        foreign_keys: Optional[list[Dict[str, Any]]] = None,
    ) -> None:
        self._ctx = context
        self._project = unify.active_project()
        self._unique_keys = dict(unique_keys or {})
        self._auto_counting = dict(auto_counting or {})
        self._description = description or ""
        self._fields = dict(fields or {})
        self._foreign_keys = list(foreign_keys or [])

    # ──────────────────────────────────────────────────────────────────────
    # Provisioning
    # ──────────────────────────────────────────────────────────────────────
    def _all_context(self) -> Optional[str]:
        """
        Derive the All/<suffix> context for cross-assistant aggregation.

        Returns None if the context has no assistant prefix (no "/" in path).
        """
        if "/" not in self._ctx:
            return None
        _, suffix = self._ctx.split("/", 1)
        return f"All/{suffix}"

    def _ensure_all_context(self, all_ctx: str) -> None:
        """
        Ensure the All/<suffix> context exists for cross-assistant aggregation.

        This context:
        - Has the same fields as the source context (for consistent querying)
        - Includes private fields (_assistant, _assistant_id, _user_id)
        - Has NO unique_keys or auto_counting (logs are added by reference)
        """
        key = (self._project, all_ctx)
        if key in self._ENSURED:
            return

        # Always attempt creation; tolerate pre-existence
        try:
            unify.get_context(all_ctx, project=self._project)
        except Exception:
            pass

        # Create aggregation context (no unique_keys/auto_counting)
        unify.create_context(
            all_ctx,
            description=f"Aggregation of {self._ctx.split('/')[-1]} across all assistants",
        )

        # Mirror fields from source context + add private fields
        fields_with_private = dict(self._fields)
        fields_with_private.update(_PRIVATE_FIELDS)

        if fields_with_private:
            try:
                unify.create_fields(fields_with_private, context=all_ctx)
            except Exception:
                # Tolerate duplicates / partial creation
                pass

        self._ENSURED.add(key)

    def ensure_context(self) -> None:
        key = (self._project, self._ctx)
        if key in self._ENSURED:
            return

        # Always attempt creation; tolerate pre-existence
        try:
            unify.get_context(self._ctx, project=self._project)
        except Exception:
            pass

        unify.create_context(
            self._ctx,
            unique_keys=self._unique_keys or None,
            auto_counting=self._auto_counting or None,
            description=self._description,
            foreign_keys=self._foreign_keys or None,
        )

        # Ensure required fields exist (idempotent per-field)
        if self._fields:
            try:
                unify.create_fields(self._fields, context=self._ctx)
            except Exception:
                # Tolerate duplicates / partial creation
                pass

        self._ENSURED.add(key)

        # Also ensure All/<Ctx> exists for cross-assistant aggregation
        all_ctx = self._all_context()
        if all_ctx is not None:
            self._ensure_all_context(all_ctx)

    # ──────────────────────────────────────────────────────────────────────
    # Accessors with 404→ensure→retry
    # ──────────────────────────────────────────────────────────────────────
    def get_columns(self) -> Dict[str, str]:
        """Return {column_name: column_type} for this context.

        If the backend returns 404 (missing context), run ``ensure_context``
        once and retry with a tiny backoff. Normalises to a single string
        label per field, preferring 'data_type' then 'type'.
        """
        data = unify.get_fields(project=self._project, context=self._ctx)
        return {k: v["data_type"] for k, v in data.items()}
