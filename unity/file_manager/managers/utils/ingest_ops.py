"""Pure batch ingestion operations for the FileManager pipeline.

This module contains the core ingestion logic extracted from ops.py,
providing clean, stateless functions for ingesting content and table rows.

These functions:
- Are pure batch operations without async orchestration
- Take explicit parameters instead of relying on manager state
- Return results directly without side effects on progress tracking
- Are designed to be called by the PipelineExecutor's task functions

The orchestration of these operations (parallel execution, progress reporting,
retries) is handled by the executor layer.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import unify

from unity.common.log_utils import create_logs as unity_create_logs
from unity.file_manager.types.file import FileContent
from unity.file_manager.types.config import (
    FilePipelineConfig,
)

logger = logging.getLogger(__name__)


def apply_content_ingest_policy(
    records: List[Dict[str, Any]],
    *,
    config: FilePipelineConfig,
    file_format: Optional[str],
) -> List[Dict[str, Any]]:
    """
    Apply per-format content ingestion policy to parsed content rows before insertion.

    - When no policy exists for the format → return rows unchanged.
    - mode == "none" → return []
    - mode == "document_only" → keep (or synthesize) a single document row
      and drop fields listed in policy.omit_fields.
    - mode == "default" → keep rows but drop fields listed in policy.omit_fields.

    Parameters
    ----------
    records : list[dict]
        Parsed content records.
    config : FilePipelineConfig
        Pipeline configuration.
    file_format : str | None
        File format (e.g., 'xlsx', 'csv') for policy lookup.

    Returns
    -------
    list[dict]
        Filtered/transformed records.
    """
    rows: List[Dict[str, Any]] = list(records or [])
    fmt = (file_format or "").strip().lower()
    if not fmt:
        return rows

    policy = (
        getattr(getattr(config, "ingest", None), "content_policy_by_format", {}) or {}
    ).get(fmt)
    if policy is None:
        return rows

    mode = getattr(policy, "mode", "default")
    omit_fields = list(getattr(policy, "omit_fields", []) or [])

    if mode == "none":
        return []

    if mode == "document_only":
        doc_rows = [
            r for r in rows if str(r.get("content_type") or "").lower() == "document"
        ]
        if not doc_rows:
            id_layout = getattr(getattr(config, "ingest", None), "id_layout", "map")
            synthesized: Dict[str, Any] = {"content_type": "document"}
            if id_layout == "columns":
                synthesized["document_id"] = 0
            else:
                synthesized["content_id"] = {"document": 0}
            doc_rows = [synthesized]
        cleaned: List[Dict[str, Any]] = []
        for r in doc_rows:
            r2 = {k: v for k, v in r.items() if k not in omit_fields}
            cleaned.append(r2)
        return cleaned

    # default: keep rows, drop omitted fields if any
    if not omit_fields:
        return rows
    return [{k: v for k, v in r.items() if k not in omit_fields} for r in rows]


def prepare_content_rows(
    records: List[Dict[str, Any]],
    *,
    config: FilePipelineConfig,
    file_format: Optional[str],
) -> List[Dict[str, Any]]:
    """
    Prepare content rows for ingestion by applying policies and column filtering.

    Parameters
    ----------
    records : list[dict]
        Raw parsed content records.
    config : FilePipelineConfig
        Pipeline configuration.
    file_format : str | None
        File format for policy lookup.

    Returns
    -------
    list[dict]
        Prepared records ready for ingestion.
    """
    rows = apply_content_ingest_policy(
        records,
        config=config,
        file_format=file_format,
    )

    # Filter allowed columns if specified
    allowed = (
        set(config.ingest.allowed_columns) if config.ingest.allowed_columns else None
    )
    if allowed:
        rows = [{k: v for k, v in rec.items() if k in allowed} for rec in rows]

    return rows


def ingest_content_batch(
    *,
    context: str,
    rows: List[Dict[str, Any]],
    file_id: int,
    id_layout: str = "map",
    auto_counting: Optional[Dict[str, Optional[str]]] = None,
) -> List[int]:
    """
    Ingest a batch of content rows into a context.

    Parameters
    ----------
    context : str
        The Unify context to ingest into.
    rows : list[dict]
        Content rows to ingest.
    file_id : int
        The file ID to associate with these rows.
    id_layout : str
        ID layout mode ("map", "columns", "string").
    auto_counting : dict | None
        Auto-counting configuration for columns layout.

    Returns
    -------
    list[int]
        Log IDs of the inserted rows.
    """
    if not rows:
        return []

    # Transform rows to file content entries
    file_content_entries: List[Dict[str, Any]] = FileContent.to_file_content_entries(
        file_id=file_id,
        rows=rows,
        id_layout=id_layout,
    )

    # Batch insert
    try:
        result = unity_create_logs(
            context=context,
            entries=file_content_entries,
            batched=True,
        )
        return [lg.id for lg in result]
    except Exception as e:
        logger.error(f"Failed to ingest content batch: {e}")
        raise


def ingest_table_batch(
    *,
    context: str,
    rows: List[Dict[str, Any]],
    columns: List[str],
) -> List[int]:
    """
    Ingest a batch of table rows into a context.

    Parameters
    ----------
    context : str
        The Unify context to ingest into.
    rows : list[dict]
        Table rows to ingest.
    columns : list[str]
        Column names for the table.

    Returns
    -------
    list[int]
        Log IDs of the inserted rows.
    """
    if not rows:
        return []

    # Normalize rows to dict format
    entries: List[Dict[str, Any]] = []
    for r in rows:
        if isinstance(r, dict):
            entry = {str(k): (str(v) if v is not None else "") for k, v in r.items()}
        else:
            # Assume it's a sequence matching columns
            entry = {
                str(col): (str(val) if val is not None else "")
                for col, val in zip(columns, r)
            }
        entries.append(entry)

    # Batch insert
    try:
        result = unity_create_logs(context=context, entries=entries, batched=True)
        return [lg.id for lg in result]
    except Exception as e:
        logger.error(f"Failed to ingest table batch: {e}")
        raise


def delete_content_rows(
    *,
    context: str,
    file_id: Optional[int] = None,
) -> int:
    """
    Delete content rows from a context.

    Parameters
    ----------
    context : str
        The Unify context to delete from.
    file_id : int | None
        If provided, only delete rows with this file_id.
        If None, delete all rows.

    Returns
    -------
    int
        Number of rows deleted.
    """
    try:
        filter_expr = f"file_id == {file_id}" if file_id is not None else None
        if filter_expr:
            ids = list(
                unify.get_logs(
                    context=context,
                    filter=filter_expr,
                    return_ids_only=True,
                ),
            )
        else:
            ids = list(unify.get_logs(context=context, return_ids_only=True))

        if not ids:
            return 0

        unify.delete_logs(
            logs=ids,
            context=context,
            project=unify.active_project(),
            delete_empty_logs=True,
        )
        return len(ids)
    except Exception as e:
        logger.warning(f"Failed to delete content rows: {e}")
        return 0


def delete_table_rows(
    *,
    context: str,
    filter_expr: Optional[str] = None,
) -> int:
    """
    Delete table rows from a context.

    Parameters
    ----------
    context : str
        The Unify context to delete from.
    filter_expr : str | None
        Optional filter expression. If None, delete all rows.

    Returns
    -------
    int
        Number of rows deleted.
    """
    try:
        if filter_expr:
            ids = list(
                unify.get_logs(
                    context=context,
                    filter=filter_expr,
                    return_ids_only=True,
                ),
            )
        else:
            ids = list(unify.get_logs(context=context, return_ids_only=True))

        if not ids:
            return 0

        unify.delete_logs(
            logs=ids,
            context=context,
            project=unify.active_project(),
            delete_empty_logs=True,
        )
        return len(ids)
    except Exception as e:
        logger.warning(f"Failed to delete table rows: {e}")
        return 0


def chunk_records(
    records: List[Dict[str, Any]],
    batch_size: int,
) -> List[List[Dict[str, Any]]]:
    """
    Split records into chunks of the specified batch size.

    Parameters
    ----------
    records : list[dict]
        Records to chunk.
    batch_size : int
        Maximum size of each chunk.

    Returns
    -------
    list[list[dict]]
        List of chunks.
    """
    if batch_size <= 0:
        batch_size = 1000

    chunks = []
    for i in range(0, len(records), batch_size):
        chunks.append(records[i : i + batch_size])
    return chunks


def get_file_id_from_path(
    *,
    index_context: str,
    file_path: str,
) -> Optional[int]:
    """
    Look up file_id from the FileRecords index.

    Parameters
    ----------
    index_context : str
        The FileRecords index context.
    file_path : str
        The file path to look up.

    Returns
    -------
    int | None
        The file_id if found, None otherwise.
    """
    try:
        rows = unify.get_logs(
            context=index_context,
            filter=f"file_path == {file_path!r}",
            limit=1,
            from_fields=["file_id"],
        )
        if rows:
            return rows[0].entries.get("file_id")
    except Exception as e:
        logger.warning(f"Failed to lookup file_id for {file_path}: {e}")
    return None


def extract_table_metadata(document: Any) -> List[Dict[str, Any]]:
    """
    Extract table metadata from a parsed document.

    Parameters
    ----------
    document : Any
        Parsed document object.

    Returns
    -------
    list[dict]
        List of table metadata dicts with keys:
        - columns: list of column names
        - rows: list of row data
        - label: derived table label
        - sheet_name: optional sheet name
        - section_path: optional section path
    """
    tables_meta = []
    try:
        tables = getattr(getattr(document, "metadata", None), "tables", []) or []
        for idx, tbl in enumerate(tables, start=1):
            columns = getattr(tbl, "columns", None)
            rows = getattr(tbl, "rows", None)
            sheet_name = getattr(tbl, "sheet_name", None)
            section_path = getattr(tbl, "section_path", None)

            if not rows:
                continue

            # Derive columns if missing
            if not columns:
                first = rows[0]
                if isinstance(first, dict):
                    columns = list(first.keys())
                else:
                    columns = [str(val) for val in first]
                # First row is header, drop it
                rows = rows[1:]

            # Ensure columns is a list
            columns = list(columns) if columns else []

            # Convert list/tuple rows to dicts using columns
            # This is required because downstream unify.create_logs expects dicts
            if rows and columns:
                converted_rows = []
                for row in rows:
                    if isinstance(row, dict):
                        converted_rows.append(row)
                    elif isinstance(row, (list, tuple)):
                        # Zip columns with row values to create dict
                        converted_rows.append(dict(zip(columns, row)))
                    else:
                        # Unexpected row type - skip or wrap
                        logger.warning(f"Unexpected row type {type(row)}, skipping")
                        continue
                rows = converted_rows

            # Derive label
            label = sheet_name or section_path or f"{idx:02d}"

            tables_meta.append(
                {
                    "columns": columns,
                    "rows": list(rows),
                    "label": str(label),
                    "sheet_name": sheet_name,
                    "section_path": section_path,
                },
            )
    except Exception as e:
        logger.warning(f"Failed to extract table metadata: {e}")

    return tables_meta
