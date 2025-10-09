from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union, Iterable

import copy
import threading

import unify


KeyInput = Union[str, int, Tuple[Any, ...], Iterable[Any]]


class DataStore:
    """
    Process-local, per-context cache of table rows with simple key-based access.

    Purpose
    -------
    Maintain a minimal in-memory mirror of a single Unify context (table),
    keyed by the table's unique key (single or composite). Consumers can index
    by the primary key to retrieve one row at a time and keep the cache
    up-to-date by calling the mutation helpers after backend interactions.

    Scope and non-goals
    -------------------
    - Read API: single-row lookup only; no arbitrary filtering/scans.
    - Write API: explicit put/update/delete; no automatic backend sync.
    - Columns: stores only public columns – keys starting with "_" or ending
      with "_emb" are treated as private and excluded.

    Key forms
    ---------
    - Single-key tables (e.g. Contacts):
      data[contact_id]            → row dict
      data[(contact_id,)]         → row dict
    - Composite-key tables (e.g. Tasks):
      data[(task_id, instance_id)]        → row dict
      data["<task_id>.<instance_id>"]    → row dict

    Notes
    -----
    - On cache miss, lookups raise KeyError.
    - Updates replace nested structures wholesale (no deep merge).
    - Instances are singletons per (project, context) via for_context().
    """

    # Registry of singleton instances per (project, context)
    _REGISTRY: Dict[Tuple[str, str], "DataStore"] = {}

    def __init__(
        self,
        context: str,
        *,
        key_fields: Tuple[str, ...],
        project: Optional[str] = None,
    ) -> None:
        if not key_fields or not isinstance(key_fields, tuple):
            raise ValueError("key_fields must be a non-empty tuple of field names")
        self._project: str = project or unify.active_project()
        self._context: str = context
        self._key_fields: Tuple[str, ...] = key_fields
        self._lock = threading.RLock()
        # Internal storage: normalized key tuple -> sanitized row dict
        self._rows: Dict[Tuple[Any, ...], Dict[str, Any]] = {}

    # ------------------------------------------------------------------ #
    #  Construction / accessors                                          #
    # ------------------------------------------------------------------ #
    @classmethod
    def for_context(
        cls,
        context: str,
        *,
        key_fields: Tuple[str, ...],
        project: Optional[str] = None,
    ) -> "DataStore":
        """Return (and memoize) a DataStore for (project, context)."""
        proj = project or unify.active_project()
        key = (proj, context)
        if key in cls._REGISTRY:
            return cls._REGISTRY[key]
        inst = cls(context, key_fields=key_fields, project=proj)
        cls._REGISTRY[key] = inst
        return inst

    @property
    def project(self) -> str:
        return self._project

    @property
    def context(self) -> str:
        return self._context

    @property
    def key_fields(self) -> Tuple[str, ...]:
        return self._key_fields

    # ------------------------------------------------------------------ #
    #  Public API – reads                                                #
    # ------------------------------------------------------------------ #
    def __getitem__(self, key: KeyInput) -> Dict[str, Any]:
        """
        Return a deep copy of the row for *key*.

        Raises KeyError when not present.
        """
        norm = self._normalize_key(key)
        with self._lock:
            row = self._rows[norm]
            return copy.deepcopy(row)

    def get(
        self,
        key: KeyInput,
        default: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Return row for *key* or *default* if not present."""
        try:
            return self.__getitem__(key)
        except KeyError:
            return default

    def snapshot(self) -> Dict[str, Dict[str, Any]]:
        """
        Return a deep-copied snapshot mapping stringified keys → rows.

        Composite keys are rendered as dot-joined values in key order.
        """
        with self._lock:
            out: Dict[str, Dict[str, Any]] = {}
            for k, v in self._rows.items():
                out[self._stringify_key_tuple(k)] = copy.deepcopy(v)
            return out

    # ------------------------------------------------------------------ #
    #  Public API – writes                                               #
    # ------------------------------------------------------------------ #
    def put(self, row: Dict[str, Any]) -> None:
        """
        Insert or replace a row in the store.

        - Requires all key_fields present in the row.
        - Stores only public columns (drop private keys).
        """
        key = self._key_from_row(row)
        clean = self._sanitize_row(row)
        with self._lock:
            self._rows[key] = clean

    def update(self, key: KeyInput, updates: Dict[str, Any]) -> None:
        """
        Update a subset of fields for an existing row.

        - Replaces nested objects entirely (no deep merge).
        - Drops any private keys in the update payload.
        - Raises KeyError when the row does not exist.
        """
        norm = self._normalize_key(key)
        clean_updates = self._sanitize_row(updates)
        with self._lock:
            if norm not in self._rows:
                raise KeyError(self._stringify_key_tuple(norm))
            merged = dict(self._rows[norm])
            merged.update(clean_updates)
            self._rows[norm] = merged

    def delete(self, key: KeyInput) -> None:
        """Remove the row with *key*. Raises KeyError when not present."""
        norm = self._normalize_key(key)
        with self._lock:
            if norm not in self._rows:
                raise KeyError(self._stringify_key_tuple(norm))
            del self._rows[norm]

    def clear(self) -> None:
        """Remove all cached rows for this context."""
        with self._lock:
            self._rows.clear()

    # ------------------------------------------------------------------ #
    #  Helpers                                                           #
    # ------------------------------------------------------------------ #
    def _sanitize_row(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """
        Return a copy of *row* containing only public columns.

        - Excludes keys starting with '_' (private)
        - Excludes keys ending with '_emb' (private vector columns)
        """
        clean: Dict[str, Any] = {}
        for k, v in (row or {}).items():
            try:
                key_str = str(k)
            except Exception:
                key_str = k  # type: ignore[assignment]
            if isinstance(key_str, str):
                if key_str.startswith("_"):
                    continue
                if key_str.endswith("_emb"):
                    continue
            clean[k] = v
        return clean

    def _key_from_row(self, row: Dict[str, Any]) -> Tuple[Any, ...]:
        """Extract and normalize the key tuple for *row* from key_fields."""
        if not isinstance(row, dict):
            raise ValueError("row must be a dict")
        values: list[Any] = []
        for fld in self._key_fields:
            if fld not in row:
                raise ValueError(f"Missing key field '{fld}' in row")
            values.append(self._coerce_scalar(row[fld]))
        return tuple(values)

    def _normalize_key(self, key: KeyInput) -> Tuple[Any, ...]:
        """
        Normalize supported key forms to a canonical tuple in key_fields order.

        Supported inputs:
        - Single-key: scalar, 1-tuple, or 1-length iterable
        - Composite: tuple/list of len(key_fields), or dot-separated string
        """
        # Composite key
        if len(self._key_fields) > 1:
            if isinstance(key, tuple):
                parts = list(key)
            elif isinstance(key, list):
                parts = list(key)
            elif isinstance(key, str):
                parts = key.split(".")
            else:
                raise TypeError(
                    "Composite key must be a tuple/list or a dot-separated string",
                )
            if len(parts) != len(self._key_fields):
                raise KeyError(
                    f"Expected {len(self._key_fields)} components for composite key",
                )
            return tuple(self._coerce_scalar(p) for p in parts)

        # Single-key
        if isinstance(key, tuple) or isinstance(key, list):
            if len(key) != 1:
                raise KeyError("Single-key table expects exactly one key component")
            scalar = list(key)[0]
        else:
            scalar = key
        return (self._coerce_scalar(scalar),)

    @staticmethod
    def _coerce_scalar(value: Any) -> Any:
        """
        Best-effort scalar normalization for key components.

        - Convert strings that look like integers (including negative) to int.
        - Leave everything else as-is.
        """
        if isinstance(value, str):
            s = value.strip()
            if s.startswith("-") and s[1:].isdigit():
                try:
                    return int(s)
                except Exception:
                    return value
            if s.isdigit():
                try:
                    return int(s)
                except Exception:
                    return value
        return value

    def _stringify_key_tuple(self, key_tuple: Tuple[Any, ...]) -> str:
        """Render key tuple as a dot-joined string for diagnostics/snapshots."""
        if len(key_tuple) == 1:
            return str(key_tuple[0])
        return ".".join(str(x) for x in key_tuple)

    # Convenience dunder helpers
    def __len__(self) -> int:  # pragma: no cover - trivial
        with self._lock:
            return len(self._rows)

    def __contains__(self, key: KeyInput) -> bool:  # pragma: no cover - trivial
        norm = self._normalize_key(key)
        with self._lock:
            return norm in self._rows

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"DataStore(project={self._project!r}, context={self._context!r}, keys={len(self)})"
