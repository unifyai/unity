from __future__ import annotations

import os
import time
from typing import Any, Dict, Optional

import requests
import unify


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
    ) -> None:
        self._ctx = context
        self._project = unify.active_project()
        self._unique_keys = dict(unique_keys or {})
        self._auto_counting = dict(auto_counting or {})
        self._description = description or ""
        self._fields = dict(fields or {})

    # ──────────────────────────────────────────────────────────────────────
    # Provisioning
    # ──────────────────────────────────────────────────────────────────────
    def ensure_context(self) -> None:
        key = (self._project, self._ctx)
        if key in self._ENSURED:
            return

        # Always attempt creation; tolerate pre-existence
        try:
            unify.get_context
            unify.create_context(
                self._ctx,
                unique_keys=self._unique_keys or None,
                auto_counting=self._auto_counting or None,
                description=self._description,
            )
        except Exception:
            # Best-effort – treat as already exists
            pass

        # Ensure required fields exist (idempotent per-field)
        if self._fields:
            try:
                unify.create_fields(self._fields, context=self._ctx)
            except Exception:
                # Tolerate duplicates / partial creation
                pass

        self._ENSURED.add(key)

    # ──────────────────────────────────────────────────────────────────────
    # Accessors with 404→ensure→retry
    # ──────────────────────────────────────────────────────────────────────
    def get_columns(self) -> Dict[str, str]:
        """Return {column_name: column_type} for this context.

        If the backend returns 404 (missing context), run ``ensure_context``
        once and retry with a tiny backoff.
        """
        url = f"{os.environ['UNIFY_BASE_URL']}/logs/fields?project={self._project}&context={self._ctx}"
        headers = {"Authorization": f"Bearer {os.environ['UNIFY_KEY']}"}
        try:
            resp = requests.request("GET", url, headers=headers)
            if resp.status_code == 404:
                # Ensure then retry once (absorbs races)
                self.ensure_context()
                time.sleep(0.05)
                resp = requests.request("GET", url, headers=headers)
            resp.raise_for_status()
        except Exception:
            # As a last resort, ensure and retry once more
            self.ensure_context()
            time.sleep(0.05)
            resp = requests.request("GET", url, headers=headers)
            resp.raise_for_status()

        data = resp.json()
        return {k: v["data_type"] for k, v in data.items()}
