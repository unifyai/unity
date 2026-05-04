from __future__ import annotations

import unify
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from unity.knowledge_manager.knowledge_manager import KnowledgeManager


def ctx_for_table(
    knowledge_manager: "KnowledgeManager",
    table: str,
    *,
    destination: str | None = None,
) -> str:
    """
    Return the fully-qualified Unify context path for a table.

    Parameters
    ----------
    knowledge_manager : KnowledgeManager
        The KnowledgeManager instance providing context resolution.
    table : str
        Logical table name (e.g., ``"Products"``, ``"Contacts"``).

    Returns
    -------
    str
        Fully-qualified Unify context path.

    Raises
    ------
    ValueError
        If ``table == "Contacts"`` but this instance was initialised with
        ``include_contacts=False``.
    """
    if table == "Contacts":
        if (
            not getattr(knowledge_manager, "_include_contacts", False)
            or getattr(knowledge_manager, "_contacts_ctx", None) is None
        ):
            raise ValueError(
                "This KnowledgeManager instance was initialised with "
                "include_contacts=False so it cannot access the Contacts table.",
            )
        return knowledge_manager._contacts_ctx  # type: ignore[return-value]
    if hasattr(knowledge_manager, "_knowledge_context_for_table"):
        return knowledge_manager._knowledge_context_for_table(  # type: ignore[attr-defined]
            table,
            destination=destination,
        )
    return f"{knowledge_manager._ctx}/{table}"


def contexts_for_table(
    knowledge_manager: "KnowledgeManager",
    table: str,
) -> list[str]:
    """Return personal-first concrete Knowledge contexts for a table."""
    if table == "Contacts":
        return [ctx_for_table(knowledge_manager, table)]
    if hasattr(knowledge_manager, "_read_contexts_for_table"):
        return knowledge_manager._read_contexts_for_table(table)  # type: ignore[attr-defined]
    return [ctx_for_table(knowledge_manager, table)]


def table_contexts_for_read(knowledge_manager: "KnowledgeManager") -> Dict[str, str]:
    """Return the first readable concrete context for each visible Knowledge table."""
    if hasattr(knowledge_manager, "_table_contexts_for_read"):
        return knowledge_manager._table_contexts_for_read()  # type: ignore[attr-defined]

    prefix = f"{knowledge_manager._ctx}/"
    ctx_info = knowledge_manager._data_manager.list_tables(
        prefix=prefix,
        include_column_info=False,
    )
    contexts = ctx_info.keys() if isinstance(ctx_info, dict) else ctx_info
    return {context[len(prefix) :]: context for context in contexts}


def get_columns(
    knowledge_manager: "KnowledgeManager",
    *,
    table: str,
) -> Dict[str, str]:
    """
    Return ``{column_name: column_type}`` for the given table.

    Parameters
    ----------
    knowledge_manager : KnowledgeManager
        The KnowledgeManager instance.
    table : str
        Logical table name.

    Returns
    -------
    dict[str, str]
        Mapping of column names to their Unify data types.
    """
    ctx = ctx_for_table(knowledge_manager, table)
    dm = knowledge_manager._data_manager
    cols = dm.get_columns(ctx)
    return {k: v.get("data_type", "unknown") for k, v in cols.items()}


def tables_overview(
    knowledge_manager: "KnowledgeManager",
    *,
    include_column_info: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """
    Show the information for all Knowledge tables (and optionally root Contacts).

    Parameters
    ----------
    knowledge_manager : KnowledgeManager
        The KnowledgeManager instance.
    include_column_info : bool, default ``True``
        When ``True``, each table entry includes a ``"columns"`` mapping.

    Returns
    -------
    dict[str, dict]
        Mapping ``table_name -> {"description": str, "columns": {...}}``.
    """
    dm = knowledge_manager._data_manager
    table_contexts = table_contexts_for_read(knowledge_manager)

    # Build tables dict stripping each Knowledge namespace prefix.
    tables: Dict[str, Dict[str, Any]] = {}
    for table_name, full_path in table_contexts.items():
        try:
            meta = unify.get_context(full_path)
        except Exception:
            meta = {}
        desc = meta.get("description") if isinstance(meta, dict) else None
        if desc == "":
            desc = None
        tables[table_name] = {"description": desc}

    # Optionally expose root-level Contacts when linkage is enabled
    if (
        getattr(knowledge_manager, "_include_contacts", False)
        and getattr(knowledge_manager, "_contacts_ctx", None) is not None
    ):
        try:
            contacts_info = unify.get_context(knowledge_manager._contacts_ctx)
            if isinstance(contacts_info, dict):
                desc = contacts_info.get("description")
                if desc == "":
                    desc = None
                tables["Contacts"] = {
                    "description": desc,
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
            pool.submit(
                lambda ctx: {
                    k: v.get("data_type", "unknown")
                    for k, v in dm.get_columns(ctx).items()
                },
                table_contexts.get(table_name)
                or ctx_for_table(knowledge_manager, table_name),
            ): table_name
            for table_name in tables.keys()
        }
        for fut in as_completed(futures):
            table_name = futures[fut]
            try:
                cols = fut.result()
                columns_by_table[table_name] = cols
            except Exception:
                columns_by_table[table_name] = {}

    return {
        name: {**meta, "columns": columns_by_table.get(name, {})}
        for name, meta in tables.items()
    }


def create_table(
    knowledge_manager: "KnowledgeManager",
    *,
    name: str,
    description: str | None = None,
    columns: Dict[str, Any] | None = None,
    unique_key_name: str = "row_id",
    auto_counting: Optional[Dict[str, Optional[str]]] = None,
    destination: str | None = None,
) -> Dict[str, str]:
    """
    Create a brand-new table in the knowledge store with optional initial columns.

    Parameters
    ----------
    knowledge_manager : KnowledgeManager
        The KnowledgeManager instance.
    name : str
        Table name (will be created under the Knowledge namespace).
    description : str | None
        Human-readable description.
    columns : dict[str, Any] | None
        Optional initial schema mapping column names to types.
    unique_key_name : str
        Name of the auto-incrementing unique key column.
    auto_counting : dict[str, str | None] | None
        Additional auto-counting configuration.

    Returns
    -------
    dict[str, str]
        Backend response describing success or failure.
    """
    dm = knowledge_manager._data_manager
    ctx = ctx_for_table(knowledge_manager, name, destination=destination)
    ac = auto_counting or {}

    # Prepare fields dict for DataManager
    fields_dict = None
    if columns:
        fields_dict = {k: str(v) for k, v in columns.items()}

    return dm.create_table(
        ctx,
        description=description,
        fields=fields_dict,
        unique_keys={unique_key_name: "int"},
        auto_counting={unique_key_name: None, **ac},
    )


def rename_table(
    knowledge_manager: "KnowledgeManager",
    *,
    old_name: str,
    new_name: str,
    destination: str | None = None,
) -> Dict[str, str]:
    """
    Rename an existing table.

    Parameters
    ----------
    knowledge_manager : KnowledgeManager
        The KnowledgeManager instance.
    old_name : str
        Current table name.
    new_name : str
        New table name.

    Returns
    -------
    dict[str, str]
        Backend acknowledgement / error message.
    """
    dm = knowledge_manager._data_manager
    old_name_fq = ctx_for_table(
        knowledge_manager,
        old_name,
        destination=destination,
    )
    new_name_fq = ctx_for_table(
        knowledge_manager,
        new_name,
        destination=destination,
    )
    return dm.rename_table(old_name_fq, new_name_fq)


def delete_tables(
    knowledge_manager: "KnowledgeManager",
    *,
    tables: Union[str, List[str]],
    startswith: Optional[str] = None,
    destination: str | None = None,
) -> List[Dict[str, str]]:
    """
    Drop one or more tables and all their rows.

    Parameters
    ----------
    knowledge_manager : KnowledgeManager
        The KnowledgeManager instance.
    tables : str | list[str]
        Target table name(s).
    startswith : str | None
        If provided, also delete all tables whose names start with this prefix.

    Returns
    -------
    list[dict[str, str]]
        Confirmations / errors from the backend.
    """
    dm = knowledge_manager._data_manager

    # Build a single, de-duplicated list of fully-qualified contexts to delete
    contexts_to_delete: List[str] = []

    if isinstance(tables, str):
        if tables:
            contexts_to_delete.append(
                ctx_for_table(knowledge_manager, tables, destination=destination),
            )
    elif tables:
        contexts_to_delete.extend(
            ctx_for_table(knowledge_manager, t, destination=destination) for t in tables
        )

    if startswith:
        # One backend read to expand the prefix
        if hasattr(knowledge_manager, "_knowledge_namespace_for_destination"):
            prefix = (
                f"{knowledge_manager._knowledge_namespace_for_destination(destination)}/"  # type: ignore[attr-defined]
                f"{startswith}"
            )
        else:
            prefix = f"{knowledge_manager._ctx}/{startswith}"
        ctx_list = dm.list_tables(prefix=prefix, include_column_info=False)
        if isinstance(ctx_list, list):
            contexts_to_delete.extend(ctx_list)
        elif isinstance(ctx_list, dict):
            contexts_to_delete.extend(ctx_list.keys())

    # De-duplicate while preserving order
    seen: set[str] = set()
    contexts_to_delete = [
        c for c in contexts_to_delete if not (c in seen or seen.add(c))
    ]
    if not contexts_to_delete:
        return []

    def _delete_one(ctx: str) -> Dict[str, str]:
        try:
            dm.delete_table(ctx, dangerous_ok=True)
            return {"status": "deleted", "context": ctx}
        except Exception as e:
            return {"status": "error", "context": ctx, "error": str(e)}

    # Fast-path: single deletion avoids thread-pool overhead
    if len(contexts_to_delete) == 1:
        return [_delete_one(contexts_to_delete[0])]

    # Parallelise deletions to minimise wall-clock time across multiple contexts
    results: List[Dict[str, str]] = []
    max_workers = min(8, len(contexts_to_delete))
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_delete_one, ctx): ctx for ctx in contexts_to_delete}
        for fut in as_completed(futures):
            results.append(fut.result())

    return results
