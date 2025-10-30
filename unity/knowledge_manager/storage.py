from __future__ import annotations

import unify
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Union

from ..common.context_store import TableStore


def ctx_for_table(self, table: str) -> str:
    """
    Return the fully‑qualified Unify context name for ``table``.

    Mirrors KnowledgeManager._ctx_for_table implementation.
    """
    if table == "Contacts":
        if (
            not getattr(self, "_include_contacts", False)
            or getattr(self, "_contacts_ctx", None) is None
        ):
            raise ValueError(
                "This KnowledgeManager instance was initialised with include_contacts=False so it cannot access the Contacts table.",
            )
        return self._contacts_ctx  # type: ignore[attr-defined]
    return f"{self._ctx}/{table}"


def provision_storage(self) -> None:
    """Ensure optional linked storage exists (e.g. root-level Contacts)."""
    contacts_ctx = getattr(self, "_contacts_ctx", None)
    if contacts_ctx is not None:
        try:
            TableStore(
                contacts_ctx,
                unique_keys={"contact_id": "int"},
                auto_counting={"contact_id": None},
            ).ensure_context()
        except Exception:
            # Best-effort; absence of Contacts must not break KM initialisation
            pass


def get_columns(self, *, table: str) -> Dict[str, str]:
    """
    Return ``{column_name: column_type}`` for the given table.
    """
    ret = unify.get_fields(context=ctx_for_table(self, table))
    return {k: v["data_type"] for k, v in ret.items()}


def tables_overview(
    self,
    *,
    include_column_info: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """
    Show the information for all Knowledge tables (and optionally root Contacts).
    """
    km_contexts = unify.get_contexts(prefix=f"{self._ctx}/")
    tables: Dict[str, Dict[str, Any]] = {
        k[len(f"{self._ctx}/") :]: {"description": v} for k, v in km_contexts.items()
    }

    # Optionally expose root-level Contacts when linkage is enabled (single call)
    if (
        getattr(self, "_include_contacts", False)
        and getattr(self, "_contacts_ctx", None) is not None
    ):
        try:
            contacts_info = unify.get_context(self._contacts_ctx)  # type: ignore[attr-defined]
            if isinstance(contacts_info, dict):
                tables["Contacts"] = {
                    "description": contacts_info.get("description", ""),
                }
        except Exception:
            # Best-effort: absence of Contacts must not fail overview
            pass

    if not include_column_info or not tables:
        return tables

    # Fetch column metadata in parallel to avoid N sequential REST calls
    columns_by_table: Dict[str, Dict[str, str]] = {}
    with ThreadPoolExecutor(max_workers=min(8, max(1, len(tables)))) as pool:
        futures = {
            pool.submit(get_columns, self, table=table_name): table_name
            for table_name in tables.keys()
        }
        for fut in as_completed(futures):
            table_name = futures[fut]
            cols = fut.result()
            columns_by_table[table_name] = cols

    return {
        name: {**meta, "columns": columns_by_table.get(name, {})}
        for name, meta in tables.items()
    }


def create_table(
    self,
    *,
    name: str,
    description: str | None = None,
    columns: Dict[str, Any] | None = None,
    unique_key_name: str = "row_id",
    auto_counting: Optional[Dict[str, Optional[str]]] = None,
) -> Dict[str, str]:
    """
    Create a brand-new table in the knowledge store with optional initial columns.
    """
    proj = unify.active_project()
    ctx = f"{self._ctx}/{name}"
    ac = auto_counting or {}
    unify.create_context(
        ctx,
        unique_keys={unique_key_name: "int"},
        auto_counting={unique_key_name: None, **ac},
        description=description,
    )

    # If no initial columns are provided, avoid an unnecessary fields call.
    if not columns:
        return {"info": "Context created", "context": ctx, "project": proj}

    # Make sure fields are always mutable by default and skip backfill for a new context
    materialized_fields = {k: {"type": v, "mutable": True} for k, v in columns.items()}
    return unify.create_fields(
        context=ctx,
        fields=materialized_fields,
        backfill_logs=False,
    )


def rename_table(self, *, old_name: str, new_name: str) -> Dict[str, str]:
    old_name_fq = f"{self._ctx}/{old_name}"
    new_name_fq = f"{self._ctx}/{new_name}"
    return unify.rename_context(old_name_fq, new_name_fq)


def delete_tables(
    self,
    *,
    tables: Union[str, List[str]],
    startswith: Optional[str] = None,
) -> List[Dict[str, str]]:
    # Build a single, de-duplicated list of fully-qualified contexts to delete
    contexts_to_delete: List[str] = []

    if isinstance(tables, str):
        if tables:
            contexts_to_delete.append(ctx_for_table(self, tables))
    elif tables:
        contexts_to_delete.extend(ctx_for_table(self, t) for t in tables)

    if startswith:
        # One backend read to expand the prefix – avoid any further metadata calls
        ctx_map = unify.get_contexts(prefix=f"{self._ctx}/{startswith}")
        contexts_to_delete.extend(list(ctx_map.keys()))

    # De-duplicate while preserving order (explicit tables first, then prefix matches)
    seen: set[str] = set()
    contexts_to_delete = [
        c for c in contexts_to_delete if not (c in seen or seen.add(c))
    ]
    if not contexts_to_delete:
        return []

    # Fast-path: single deletion avoids thread-pool overhead
    if len(contexts_to_delete) == 1:
        return [unify.delete_context(contexts_to_delete[0])]

    # Parallelise deletions to minimise wall-clock time across multiple contexts
    results: List[Dict[str, str]] = []
    max_workers = min(8, len(contexts_to_delete))
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(unify.delete_context, ctx) for ctx in contexts_to_delete]
        for fut in as_completed(futures):
            results.append(fut.result())
    return results
