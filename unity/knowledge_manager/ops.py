from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

import unify

from ..common.embed_utils import ensure_vector_column
from .storage import ctx_for_table


def add_rows(self, *, table: str, rows: List[Dict[str, Any]]) -> Dict[str, str]:
    return unify.create_logs(
        context=ctx_for_table(self, table),
        entries=rows,
        batched=True,
    )


def update_rows(
    self,
    *,
    table: str,
    updates: Dict[int, Dict[str, Any]],
) -> Dict[str, str]:
    ctx = ctx_for_table(self, table)
    ctx_info = unify.get_context(ctx)
    keys = ctx_info.get("unique_keys")
    unique_column_name = keys[0] if isinstance(keys, list) and keys else keys
    unique_ids = sorted(int(k) for k in updates.keys())
    filt = f"{unique_column_name} in {unique_ids}"
    log_ids: List[int] = list(
        unify.get_logs(context=ctx, filter=filt, return_ids_only=True),
    )
    entries = [updates[i] for i in unique_ids]
    res = unify.update_logs(logs=log_ids, context=ctx, entries=entries, overwrite=True)
    return res


def delete_rows(
    self,
    *,
    filter: Optional[str] = None,
    offset: int = 0,
    limit: int = 100,
    tables: Optional[List[str]] = None,
) -> Dict[str, Any]:
    if limit > 1000:
        raise ValueError("Limit must be less than 1000")

    if tables is None:
        km_prefix = f"{self._ctx}/"
        ctxs = unify.get_contexts(prefix=km_prefix)
        resolved_tables: List[str] = [k[len(km_prefix) :] for k in ctxs.keys()]
        if (
            getattr(self, "_include_contacts", False)
            and getattr(self, "_contacts_ctx", None) is not None
        ):
            try:
                contacts_info = unify.get_context(self._contacts_ctx)  # type: ignore[attr-defined]
                if isinstance(contacts_info, dict):
                    resolved_tables.append("Contacts")
            except Exception:
                pass
    else:
        resolved_tables = list(tables)

    if not resolved_tables:
        return {}

    project_name = unify.active_project()

    def _delete_for_table(table_name: str) -> tuple[str, Any]:
        ctx = self._ctx_for_table(table_name)
        log_ids = list(
            unify.get_logs(
                context=ctx,
                filter=filter,
                offset=offset,
                limit=limit,
                return_ids_only=True,
            ),
        )
        if not log_ids:
            return table_name, {"status": "no-op"}

        res = unify.delete_logs(
            logs=log_ids,
            context=ctx,
            project=project_name,
            delete_empty_logs=True,
        )
        # Return the full backend response for structured logging
        return table_name, res

    # Parallelise across tables to minimise wall-clock time when multiple tables are targeted.
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

    return summaries


def create_empty_column(
    self,
    *,
    table: str,
    column_name: str,
    column_type: str,
) -> Dict[str, str]:
    return unify.create_fields(
        context=ctx_for_table(self, table),
        fields={column_name: {"type": column_type, "mutable": True}},
        backfill_logs=False,
    )


def create_derived_column(
    self,
    *,
    table: str,
    column_name: str,
    equation: str,
) -> Dict[str, str]:
    equation = equation.replace("{", "{lg:")
    return unify.create_derived_logs(
        context=ctx_for_table(self, table),
        key=column_name,
        equation=equation,
        referenced_logs={"lg": {"context": ctx_for_table(self, table)}},
    )


def delete_column(self, *, table: str, column_name: str) -> Dict[str, str]:
    table_ctx = unify.get_context(ctx_for_table(self, table))
    keys = table_ctx.get("unique_keys")
    unique_column_name = keys[0] if isinstance(keys, list) and keys else keys
    # Guard against removal of mandatory columns
    if table == "Contacts":
        try:
            from unity.contact_manager.types.contact import Contact as _C

            required_cols = set(_C.model_fields.keys()) - {
                "rolling_summary",
                "response_policy",
                "respond_to",
            }
        except Exception:
            required_cols = {"contact_id"}
        if column_name in required_cols:
            raise ValueError(
                (
                    f"Cannot delete required Contacts column '{column_name}'. "
                    "Contacts core schema is protected. If you need to restructure, "
                    "use rename_column or create a new optional column and migrate values."
                ),
            )
    elif column_name == unique_column_name:
        raise ValueError(
            (
                f"Cannot delete primary key column '{column_name}'. "
                "This column uniquely identifies rows. Use rename_column if you need a different name."
            ),
        )

    # Prefer field-level deletion endpoint for efficiency; avoids per-log scans
    return unify.delete_fields(fields=[column_name], context=ctx_for_table(self, table))


def rename_column(self, *, table: str, old_name: str, new_name: str) -> Dict[str, str]:
    # Short-circuit obvious no-op and invalid rename targets to avoid any backend call
    if old_name == new_name:
        return {
            "info": "no-op: old and new names are identical",
            "old_name": old_name,
            "new_name": new_name,
        }
    if new_name == "id":
        raise ValueError("Cannot rename a column to reserved name 'id'.")
    return unify.rename_field(
        name=old_name,
        new_name=new_name,
        context=ctx_for_table(self, table),
    )


def copy_column(
    self,
    *,
    source_table: str,
    column_name: str,
    dest_table: str,
) -> Dict[str, Any]:
    src_ctx = ctx_for_table(self, source_table)
    dest_ctx = ctx_for_table(self, dest_table)
    log_ids = unify.get_logs(
        context=src_ctx,
        filter=f"{column_name} is not None",
        limit=100_000,
        return_ids_only=True,
    )
    unify.add_logs_to_context(log_ids, context=dest_ctx, project=unify.active_project())
    return {
        "status": "copied",
        "rows": len(log_ids),
        "from": source_table,
        "to": dest_table,
        "column": column_name,
    }


def move_column(
    self,
    *,
    source_table: str,
    column_name: str,
    dest_table: str,
) -> Dict[str, Any]:
    copy_res = copy_column(
        self,
        source_table=source_table,
        column_name=column_name,
        dest_table=dest_table,
    )
    del_res = delete_column(self, table=source_table, column_name=column_name)
    return {"status": "moved", "copy_result": copy_res, "delete_result": del_res}


def transform_column(
    self,
    *,
    table: str,
    column_name: str,
    equation: str,
) -> Dict[str, Any]:
    import uuid as _uuid

    tmp_name = f"tmp_{column_name}_{_uuid.uuid4().hex[:8]}"
    create_res = create_derived_column(
        self,
        table=table,
        column_name=tmp_name,
        equation=equation,
    )
    delete_res = delete_column(self, table=table, column_name=column_name)
    rename_res = rename_column(
        self,
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
    self,
    table: str,
    source_column: str,
    target_column_name: str,
    *,
    from_ids: List[int] | None = None,
) -> None:
    context = ctx_for_table(self, table)
    ensure_vector_column(
        context,
        embed_column=target_column_name,
        source_column=source_column,
        from_ids=from_ids,
    )
