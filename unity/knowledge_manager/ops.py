from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import unify

from ..common.embed_utils import ensure_vector_column
from .storage import ctx_for_table

if TYPE_CHECKING:
    from unity.knowledge_manager.knowledge_manager import KnowledgeManager


def add_rows(
    knowledge_manager: "KnowledgeManager",
    *,
    table: str,
    rows: List[Dict[str, Any]],
) -> List[int]:
    """
    Insert rows into a table.

    Parameters
    ----------
    knowledge_manager : KnowledgeManager
        The KnowledgeManager instance.
    table : str
        Destination table name.
    rows : list[dict[str, Any]]
        Row dictionaries to insert.

    Returns
    -------
    list[int]
        Log IDs of the inserted rows.
    """
    dm = knowledge_manager._data_manager
    ctx = ctx_for_table(knowledge_manager, table)
    return dm.insert_rows(
        ctx,
        rows=rows,
        add_to_all_context=knowledge_manager.include_in_multi_assistant_table,
    )


def update_rows(
    knowledge_manager: "KnowledgeManager",
    *,
    table: str,
    updates: Dict[int, Dict[str, Any]],
) -> Dict[str, str]:
    """
    Update existing rows identified by their table-specific unique id.

    Parameters
    ----------
    knowledge_manager : KnowledgeManager
        The KnowledgeManager instance.
    table : str
        Target table name.
    updates : dict[int, dict[str, Any]]
        Mapping of unique row ids to new field values.

    Returns
    -------
    dict[str, str]
        Backend response from update operation.

    Raises
    ------
    ValueError
        If no matching rows are found for the given ids.
    """
    dm = knowledge_manager._data_manager
    ctx = ctx_for_table(knowledge_manager, table)

    # Get context info to find the unique key column name
    ctx_info = dm.get_table(ctx)
    keys = ctx_info.get("unique_keys")
    unique_column_name = keys[0] if isinstance(keys, list) and keys else keys

    unique_ids = sorted(int(k) for k in updates.keys())
    filt = f"{unique_column_name} in {unique_ids}"

    # Build a stable mapping from unique ids → log ids
    events = dm.filter(
        ctx,
        filter=filt,
        columns=[unique_column_name],
        limit=len(unique_ids),
    )
    id_to_log: Dict[int, int] = {}
    for ev in events:
        try:
            key_val = ev.get(unique_column_name)
            uid = int(key_val) if key_val is not None else None
        except Exception:
            uid = None
        if uid is None:
            continue
        # Get log id from the internal _id field if available
        log_id = ev.get("id") or ev.get("_id")
        if log_id is not None:
            id_to_log[uid] = int(log_id)

    # Pair only ids present in both the request and the backend
    matched: List[tuple[int, Dict[str, Any]]] = [
        (id_to_log[i], updates[i])
        for i in unique_ids
        if i in updates and i in id_to_log
    ]
    if not matched:
        raise ValueError(
            f"No matching rows for ids={unique_ids} in table '{table}'",
        )

    log_ids = [lid for (lid, _) in matched]
    entries = [entry for (_, entry) in matched]

    # Use unify.update_logs for now (needs DataManager.update_rows enhancement)
    res = unify.update_logs(
        logs=log_ids,
        context=ctx,
        entries=entries,
        overwrite=True,
    )
    return res


def delete_rows(
    knowledge_manager: "KnowledgeManager",
    *,
    filter: Optional[str] = None,
    offset: int = 0,
    limit: int = 100,
    tables: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Delete rows matching a filter across one or more tables.

    Parameters
    ----------
    knowledge_manager : KnowledgeManager
        The KnowledgeManager instance.
    filter : str | None
        Row-level predicate.
    offset : int
        Pagination offset.
    limit : int
        Maximum rows per table.
    tables : list[str] | None
        Subset of tables; ``None`` -> all tables.

    Returns
    -------
    dict[str, Any]
        Mapping ``table_name -> backend message``.

    Raises
    ------
    ValueError
        If limit exceeds 1000.
    """
    if limit > 1000:
        raise ValueError("Limit must be less than 1000")

    dm = knowledge_manager._data_manager

    if tables is None:
        km_prefix = f"{knowledge_manager._ctx}/"
        ctx_list = dm.list_tables(prefix=km_prefix, include_column_info=False)
        if isinstance(ctx_list, dict):
            resolved_tables = [k[len(km_prefix) :] for k in ctx_list.keys()]
        else:
            resolved_tables = [k[len(km_prefix) :] for k in ctx_list]

        if (
            getattr(knowledge_manager, "_include_contacts", False)
            and getattr(knowledge_manager, "_contacts_ctx", None) is not None
        ):
            resolved_tables.append("Contacts")
    else:
        resolved_tables = list(tables)

    if not resolved_tables:
        return {}

    def _delete_for_table(table_name: str) -> tuple[str, Any]:
        ctx = ctx_for_table(knowledge_manager, table_name)
        try:
            # Get log IDs matching filter efficiently
            log_ids = dm.filter(
                ctx,
                filter=filter,
                limit=limit,
                offset=offset,
                return_ids_only=True,
            )
            if not log_ids:
                return table_name, {"status": "no-op"}

            # Delete using log_ids for efficiency
            count = dm.delete_rows(ctx, log_ids=log_ids, dangerous_ok=True)
            return table_name, {"status": "deleted", "count": count}
        except Exception as e:
            return table_name, {"status": "error", "error": str(e)}

    # Parallelise across tables
    if len(resolved_tables) == 1:
        name, msg = _delete_for_table(resolved_tables[0])
        return {name: msg}

    summaries: Dict[str, Any] = {}
    max_workers = min(8, max(1, len(resolved_tables)))
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_delete_for_table, table_name): table_name
            for table_name in resolved_tables
        }
        for fut in as_completed(futures):
            name, msg = fut.result()
            summaries[name] = msg

    return dict(sorted(summaries.items()))


def create_empty_column(
    knowledge_manager: "KnowledgeManager",
    *,
    table: str,
    column_name: str,
    column_type: str,
) -> Dict[str, str]:
    """
    Add a new, initially empty column to a table.

    Parameters
    ----------
    knowledge_manager : KnowledgeManager
        The KnowledgeManager instance.
    table : str
        Target table name.
    column_name : str
        New column identifier (must be snake_case).
    column_type : str
        Logical type (e.g., "str", "float", "datetime").

    Returns
    -------
    dict[str, str]
        Backend response.
    """
    dm = knowledge_manager._data_manager
    ctx = ctx_for_table(knowledge_manager, table)
    return dm.create_column(
        ctx,
        column_name=column_name,
        column_type=column_type,
        mutable=True,
        backfill_logs=False,
    )


def create_derived_column(
    knowledge_manager: "KnowledgeManager",
    *,
    table: str,
    column_name: str,
    equation: str,
) -> Dict[str, str]:
    """
    Create a derived column computed from other columns via an equation.

    Parameters
    ----------
    knowledge_manager : KnowledgeManager
        The KnowledgeManager instance.
    table : str
        Table to modify.
    column_name : str
        Name of the new derived column.
    equation : str
        Python expression evaluated per-row.

    Returns
    -------
    dict[str, str]
        Backend acknowledgement.
    """
    dm = knowledge_manager._data_manager
    ctx = ctx_for_table(knowledge_manager, table)
    return dm.create_derived_column(ctx, column_name=column_name, equation=equation)


def delete_column(
    knowledge_manager: "KnowledgeManager",
    *,
    table: str,
    column_name: str,
) -> Dict[str, str]:
    """
    Remove a column and its data from a table.

    Parameters
    ----------
    knowledge_manager : KnowledgeManager
        The KnowledgeManager instance.
    table : str
        Table name.
    column_name : str
        Column to drop.

    Returns
    -------
    dict[str, str]
        Backend confirmation or error.

    Raises
    ------
    ValueError
        If attempting to delete a required column.
    """
    dm = knowledge_manager._data_manager
    ctx = ctx_for_table(knowledge_manager, table)
    ctx_info = dm.get_table(ctx)
    keys = ctx_info.get("unique_keys")
    unique_column_name = keys[0] if isinstance(keys, list) and keys else keys

    # Guard against removal of mandatory columns
    if table == "Contacts":
        try:
            from unity.contact_manager.types.contact import Contact as _C

            required_cols = set(_C.model_fields.keys()) - {
                "rolling_summary",
                "response_policy",
                "should_respond",
            }
        except Exception:
            required_cols = {"contact_id"}
        if column_name in required_cols:
            raise ValueError(
                f"Cannot delete required Contacts column '{column_name}'. "
                "Contacts core schema is protected. If you need to restructure, "
                "use rename_column or create a new optional column and migrate values.",
            )
    elif column_name == unique_column_name:
        raise ValueError(
            f"Cannot delete primary key column '{column_name}'. "
            "This column uniquely identifies rows. Use rename_column if you need a different name.",
        )

    return dm.delete_column(ctx, column_name=column_name)


def rename_column(
    knowledge_manager: "KnowledgeManager",
    *,
    table: str,
    old_name: str,
    new_name: str,
) -> Dict[str, str]:
    """
    Rename a column inside a table.

    Parameters
    ----------
    knowledge_manager : KnowledgeManager
        The KnowledgeManager instance.
    table : str
        Table identifier.
    old_name : str
        Existing column name.
    new_name : str
        Desired new name.

    Returns
    -------
    dict[str, str]
        Backend response.

    Raises
    ------
    ValueError
        If attempting to rename to reserved name 'id'.
    """
    dm = knowledge_manager._data_manager
    ctx = ctx_for_table(knowledge_manager, table)
    return dm.rename_column(ctx, old_name=old_name, new_name=new_name)


def copy_column(
    knowledge_manager: "KnowledgeManager",
    *,
    source_table: str,
    column_name: str,
    dest_table: str,
) -> Dict[str, Any]:
    """
    Copy a column's values from one table to another.

    Parameters
    ----------
    knowledge_manager : KnowledgeManager
        The KnowledgeManager instance.
    source_table : str
        Table to read values from.
    column_name : str
        Column to copy.
    dest_table : str
        Destination table.

    Returns
    -------
    dict[str, Any]
        Summary of the copy operation.
    """
    dm = knowledge_manager._data_manager
    src_ctx = ctx_for_table(knowledge_manager, source_table)
    dest_ctx = ctx_for_table(knowledge_manager, dest_table)

    # Get rows with non-null values in the column
    rows = dm.filter(src_ctx, filter=f"{column_name} is not None", limit=1000)
    log_ids = [r.get("id") or r.get("_id") for r in rows if r.get("id") or r.get("_id")]

    if log_ids:
        unify.add_logs_to_context(
            log_ids,
            context=dest_ctx,
            project=unify.active_project(),
        )

    return {
        "status": "copied",
        "rows": len(log_ids),
        "from": source_table,
        "to": dest_table,
        "column": column_name,
    }


def move_column(
    knowledge_manager: "KnowledgeManager",
    *,
    source_table: str,
    column_name: str,
    dest_table: str,
) -> Dict[str, Any]:
    """
    Move a column from one table to another.

    Parameters
    ----------
    knowledge_manager : KnowledgeManager
        The KnowledgeManager instance.
    source_table : str
        Source table.
    column_name : str
        Column to move.
    dest_table : str
        Destination table.

    Returns
    -------
    dict[str, Any]
        Summary containing copy and delete results.
    """
    copy_res = copy_column(
        knowledge_manager,
        source_table=source_table,
        column_name=column_name,
        dest_table=dest_table,
    )
    del_res = delete_column(
        knowledge_manager,
        table=source_table,
        column_name=column_name,
    )
    return {"status": "moved", "copy_result": copy_res, "delete_result": del_res}


def transform_column(
    knowledge_manager: "KnowledgeManager",
    *,
    table: str,
    column_name: str,
    equation: str,
) -> Dict[str, Any]:
    """
    Transform a column in-place according to a Python equation.

    Parameters
    ----------
    knowledge_manager : KnowledgeManager
        The KnowledgeManager instance.
    table : str
        Table to modify.
    column_name : str
        Column to transform.
    equation : str
        Per-row Python expression.

    Returns
    -------
    dict[str, Any]
        Summary of create/delete/rename steps.
    """
    import uuid as _uuid

    tmp_name = f"tmp_{column_name}_{_uuid.uuid4().hex[:8]}"
    create_res = create_derived_column(
        knowledge_manager,
        table=table,
        column_name=tmp_name,
        equation=equation,
    )
    delete_res = delete_column(
        knowledge_manager,
        table=table,
        column_name=column_name,
    )
    rename_res = rename_column(
        knowledge_manager,
        table=table,
        old_name=tmp_name,
        new_name=column_name,
    )
    return {
        "status": "transformed",
        "create_result": create_res,
        "delete_result": delete_res,
        "rename_result": rename_res,
    }


def vectorize_column(
    knowledge_manager: "KnowledgeManager",
    table: str,
    source_column: str,
    target_column_name: str,
    *,
    from_ids: List[int] | None = None,
) -> None:
    """
    Ensure a vector column exists and generate embeddings.

    Parameters
    ----------
    knowledge_manager : KnowledgeManager
        The KnowledgeManager instance.
    table : str
        The table to ensure the vector column in.
    source_column : str
        The existing column whose text will be embedded.
    target_column_name : str
        Name of the embedding column to create/ensure.
    from_ids : list[int] | None
        Optional specific row IDs to embed.

    Returns
    -------
    None
    """
    context = ctx_for_table(knowledge_manager, table)
    ensure_vector_column(
        context,
        embed_column=target_column_name,
        source_column=source_column,
        from_ids=from_ids,
    )
