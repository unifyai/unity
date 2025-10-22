from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

import unify

from ..common.context_store import TableStore
from ..common.model_to_fields import model_to_fields
from .types.contact import Contact


def provision_storage(self) -> None:
    """Ensure Contacts context, schema, and local cache exist (idempotent)."""
    # Ensure context/fields exist deterministically (idempotent)
    self._store = TableStore(
        self._ctx,
        unique_keys={"contact_id": "int"},
        auto_counting={"contact_id": None},
        description="List of contacts, with all contact details stored.",
        fields=model_to_fields(Contact),
    )
    self._store.ensure_context()

    # Prefill known custom fields once to include any preexisting non-private columns
    try:
        existing_cols = get_columns(self)
        for col in existing_cols:
            if col not in self._REQUIRED_COLUMNS and not str(col).startswith("_"):
                try:
                    self._known_custom_fields.add(col)  # type: ignore[attr-defined]
                except Exception:
                    pass
    except Exception:
        # Best-effort only; tools fall back safely
        pass


def get_columns(self) -> Dict[str, str]:
    """Return {column_name: column_type} for the contacts table."""
    return self._store.get_columns()


def get_contact_info(
    self,
    contact_id: Union[int, List[int]],
    fields: Optional[Union[str, List[str]]] = None,
    search_local_storage: bool = True,
) -> Dict[int, Dict[str, Any]]:
    """Return a mapping of requested fields for one or many contacts."""
    allowed = set(self._allowed_fields())

    # Normalise requested fields
    if fields is None or (isinstance(fields, str) and fields.lower() == "all"):
        requested: List[str] = list(allowed)
    elif isinstance(fields, str):
        requested = [fields]
    else:
        requested = list(fields or [])

    # Intersect with allowed set to avoid accidental vector/private columns
    requested = [f for f in requested if f in allowed]
    if not requested:
        requested = list(allowed)

    # Normalise ids list
    if isinstance(contact_id, list):
        ids: List[int] = [int(x) for x in contact_id]
    else:
        ids = [int(contact_id)]

    results: Dict[int, Dict[str, Any]] = {}
    misses: List[int] = []

    # 1) Try local cache
    if search_local_storage:
        for cid in ids:
            try:
                row = self._data_store[cid]
                results[cid] = {k: v for k, v in row.items() if k in requested}
            except KeyError:
                misses.append(cid)
    else:
        misses = list(ids)

    # 2) Backend read for misses (allowed-field superset); write-through to cache
    if misses:
        if len(misses) == 1:
            filt = f"contact_id == {misses[0]}"
        else:
            filt = f"contact_id in [{', '.join(str(x) for x in misses)}]"
        rows = unify.get_logs(
            context=self._ctx,
            filter=filt,
            limit=len(misses),
            from_fields=list(allowed),
        )
        for lg in rows:
            try:
                backend_row = lg.entries
                cid_val = int(backend_row.get("contact_id"))
            except Exception:
                continue
            try:
                self._data_store.put(backend_row)
            except Exception:
                pass
            results[cid_val] = {k: backend_row.get(k) for k in requested}

    return results


def num_contacts(self) -> int:
    """Return total number of contacts in the context."""
    ret = unify.get_logs_metric(
        metric="count",
        key="contact_id",
        context=self._ctx,
    )
    if ret is None:
        return 0
    return int(ret)
