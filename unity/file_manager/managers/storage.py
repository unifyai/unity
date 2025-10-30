from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

import unify

from ...common.context_store import TableStore
from ...common.model_to_fields import model_to_fields
from ..types.file import FileRecord as FileRow


def provision_storage(self) -> None:
    """Ensure FileRecords/<alias> context, schema and local view exist (idempotent)."""
    # Create the TableStore if not already initialised by the manager
    if not hasattr(self, "_store") or getattr(self, "_store", None) is None:
        self._store = TableStore(  # type: ignore[attr-defined]
            self._ctx,  # type: ignore[attr-defined]
            unique_keys={"file_id": "int"},
            auto_counting={"file_id": None},
            description=(
                "FileRecords index for a single filesystem; per-file content lives under File/<alias>/<filename>/Tables/<table>."
            ),
            fields=model_to_fields(FileRow),
        )
    try:
        self._store.ensure_context()  # type: ignore[attr-defined]
    except Exception:
        # Best-effort
        pass


def get_columns(self) -> Dict[str, str]:
    """Return {column_name: column_type} for the file table."""
    try:
        return self._store.get_columns()  # type: ignore[attr-defined]
    except Exception:
        return {}


def tables_overview(
    self,
    *,
    include_column_info: bool = True,
    file: Optional[str] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Return an overview of tables/contexts managed by the FileManager.

    Behaviour
    ---------
    - When file is None: return overview for the global index only; include
      column schema for the index when requested.
    - When file is provided: return overview for that file: one entry for the
      per-file context ("File") and one entry per extracted per-table context
      under its Tables/ directory. Do not include columns for per-file/per-table.
    """

    # Global index overview
    if file is None:
        try:
            ctx_info = unify.get_context(self._ctx)  # type: ignore[attr-defined]
        except Exception:
            ctx_info = {}
        try:
            cols = get_columns(self) if include_column_info else None
        except Exception:
            cols = None
        label = "Index"
        out: Dict[str, Dict[str, Any]] = {
            label: {
                "description": (
                    ctx_info.get("description") if isinstance(ctx_info, dict) else ""
                ),
                **(
                    {"columns": cols}
                    if include_column_info and isinstance(cols, dict)
                    else {}
                ),
            },
        }
        return out

    # Per-file overview
    # Derive the per-file root context using manager-provided helpers
    try:
        base = getattr(self, "_per_file_root")
        safe_fn = getattr(self, "_safe")
        per_file_ctx = f"{base}/{safe_fn(file)}"
    except Exception:
        per_file_ctx = None

    out: Dict[str, Dict[str, Any]] = {}
    if isinstance(per_file_ctx, str):
        # File root entry
        try:
            info = unify.get_context(per_file_ctx)
            out["File"] = {
                "description": (
                    info.get("description") if isinstance(info, dict) else ""
                ),
            }
        except Exception:
            out["File"] = {"description": ""}

        # Extract per-table contexts under /Tables/
        try:
            ctxs = unify.get_contexts(prefix=f"{per_file_ctx}/Tables/")
        except Exception:
            ctxs = {}
        for full, desc in (ctxs or {}).items():
            try:
                # table label is the tail after /Tables/
                label = full.split("/Tables/")[-1]
                out[label] = {"description": desc}
            except Exception:
                continue

    return out


def get_file_info(
    self,
    file_id: Union[int, List[int]],
    *,
    fields: Optional[Union[str, List[str]]] = None,
    search_local_storage: bool = True,
) -> Dict[int, Dict[str, Any]]:
    """Return a mapping of requested fields for one or many files."""
    allowed = set(self._allowed_fields())  # type: ignore[attr-defined]

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
    if isinstance(file_id, list):
        ids: List[int] = [int(x) for x in file_id]
    else:
        ids = [int(file_id)]

    results: Dict[int, Dict[str, Any]] = {}
    misses: List[int] = []

    # 1) Try local cache first
    if search_local_storage:
        for fid in ids:
            try:
                row = self._data_store[fid]  # type: ignore[attr-defined]
                results[fid] = {k: v for k, v in row.items() if k in requested}
            except KeyError:
                misses.append(fid)
    else:
        misses = list(ids)

    # 2) Backend read for misses; write-through cache
    if misses:
        filt = (
            f"file_id == {misses[0]}"
            if len(misses) == 1
            else f"file_id in [{', '.join(str(x) for x in misses)}]"
        )
        rows = unify.get_logs(
            context=self._ctx,  # type: ignore[attr-defined]
            filter=filt,
            limit=len(misses),
            from_fields=list(allowed),
        )
        for lg in rows:
            try:
                backend_row = lg.entries
                fid_val = int(backend_row.get("file_id"))
            except Exception:
                continue
            try:
                self._data_store.put(backend_row)  # type: ignore[attr-defined]
            except Exception:
                pass
            results[fid_val] = {k: backend_row.get(k) for k in requested}

    return results


def num_files(self) -> int:
    """Return total number of files in the context."""
    try:
        ret = unify.get_logs_metric(metric="count", key="file_id", context=self._ctx)  # type: ignore[attr-defined]
        return 0 if ret is None else int(ret)
    except Exception:
        return 0
