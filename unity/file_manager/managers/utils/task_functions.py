"""Pure task execution functions for the FileManager pipeline.

Each function is a self-contained unit of work executed by the
``PipelineExecutor``.  They are PURE:

- Explicit parameters (no hidden state).
- Single responsibility (does one thing).
- Returns explicit results (dict with status/data).
- Raises exceptions on failure (caught by executor for retry).

Embedding is now delegated to ``DataManager.ingest()`` (via the
``embed_columns`` and ``embed_strategy`` parameters), so there are
no longer separate embed task functions.

The orchestration layer (``executor.py``) calls these functions
directly with retry logic and optional concurrency.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from unity.file_manager.file_parsers.types.contracts import FileParseResult
from unity.common.pipeline import InlineRowsHandle, TableInputHandle
from unity.file_manager.types.config import (
    FilePipelineConfig,
    TableBusinessContextSpec,
)
from unity.file_manager.types.file import FileContentRow

logger = logging.getLogger(__name__)


# =============================================================================
# BUSINESS CONTEXT HELPERS
# =============================================================================


def _lookup_table_business_context(
    file_path: str,
    table_label: str,
    config: FilePipelineConfig,
) -> Optional[TableBusinessContextSpec]:
    """Look up the business context for a specific table.

    Searches ``config.ingest.business_contexts.file_contexts`` for a matching
    *file_path*, then finds the matching table spec within that file's
    ``table_contexts``.

    Returns
    -------
    TableBusinessContextSpec | None
    """
    if not config or not hasattr(config, "ingest"):
        return None

    business_contexts = getattr(config.ingest, "business_contexts", None)
    if not business_contexts:
        return None

    file_contexts = getattr(business_contexts, "file_contexts", [])
    if not file_contexts:
        return None

    for fc in file_contexts:
        if fc.file_path != file_path:
            continue
        for table_spec in fc.table_contexts:
            if table_spec.table == table_label:
                logger.debug(
                    f"[TaskFn] Found business context for table '{table_label}' in {file_path}",
                )
                return table_spec

    logger.debug(
        f"[TaskFn] No business context match for table '{table_label}' in {file_path}",
    )
    return None


def _build_table_description(
    table_label: str,
    business_context: Optional[TableBusinessContextSpec],
) -> Optional[str]:
    """Build a table description string from business context metadata."""
    if not business_context:
        return None

    parts: List[str] = []
    if business_context.table_description:
        parts.append(business_context.table_description)
    if business_context.table_rules:
        parts.append("Rules: " + "; ".join(business_context.table_rules))
    if business_context.column_descriptions:
        col_desc = ", ".join(
            f"{k}: {v}" for k, v in business_context.column_descriptions.items()
        )
        parts.append(f"Columns: {col_desc}")
    return " | ".join(parts) if parts else None


def _build_table_fields(
    columns: List[str],
    example_row: Optional[Dict[str, Any]] = None,
    business_context: Optional[TableBusinessContextSpec] = None,
) -> Optional[Dict[str, str]]:
    """Infer field types from column names, example row, and business context."""
    if not columns:
        return None
    fields: Dict[str, str] = {}
    for name in columns:
        fields[str(name)] = "Any"
    return fields


# =============================================================================
# FILE RECORD TASK
# =============================================================================


def execute_create_file_record(
    *,
    file_manager: Any,
    file_path: str,
    parse_result: FileParseResult,
    config: FilePipelineConfig,
    document_summary: str = "",
    total_records: Optional[int] = None,
) -> Dict[str, Any]:
    """Create the FileRecord entry in the index.

    This task MUST run before any content/table ingestion.  It registers
    the file in the FileRecords index and returns the generated ``file_id``
    and computed ``storage_id``.

    Returns
    -------
    dict
        ``{"file_id": int, "file_path": str, "storage_id": str}``

    Raises
    ------
    Exception
        On failure to create the file record.
    """
    from .ops import create_file_record as _ops_create_file_record
    from .ingest_ops import get_file_id_from_path
    from unity.file_manager.types.file import FileRecord

    dm = file_manager._data_manager
    logger.debug(f"[TaskFn] Creating file record for: {file_path}")

    config_storage_id = config.ingest.storage_id
    table_ingest = config.ingest.table_ingest

    from .source_info import source_info_for_file

    ref = None
    try:
        ref = file_manager._adapter.get_file(file_path)
    except Exception:
        ref = None
    sinfo = source_info_for_file(
        adapter_ref=ref,
        trace=getattr(parse_result, "trace", None),
    )

    source_uri = None
    source_provider = None
    try:
        resolve_to_uri = getattr(file_manager, "_resolve_to_uri", None)
        if resolve_to_uri:
            source_uri = resolve_to_uri(file_path)
    except Exception:
        pass
    try:
        adapter = getattr(file_manager, "_adapter", None)
        source_provider = getattr(adapter, "name", None) or getattr(
            file_manager,
            "_fs_type",
            None,
        )
    except Exception:
        pass

    entry = FileRecord.to_file_record_entry(
        file_path=file_path,
        source_uri=source_uri,
        source_provider=source_provider,
        parse_result=parse_result,
        storage_id=config_storage_id or "",
        table_ingest=table_ingest,
        file_size=sinfo.size_bytes,
        created_at=sinfo.created_at,
        modified_at=sinfo.modified_at,
        total_records=total_records,
        document_summary=document_summary,
    )

    pre_insert_file_id = getattr(entry, "file_id", None)

    created_file_record = _ops_create_file_record(file_manager, entry=entry)

    post_insert_file_id = getattr(entry, "file_id", None)

    logger.info(
        "[TaskFn] create_file_record result for %s: "
        "pre_insert_file_id=%s, post_insert_file_id=%s, ops_result=%s",
        file_path,
        pre_insert_file_id,
        post_insert_file_id,
        created_file_record,
    )

    file_id = get_file_id_from_path(
        data_manager=dm,
        index_context=file_manager._ctx,
        file_path=file_path,
    )

    logger.info(
        "[TaskFn] get_file_id_from_path(%s, ctx=%s) -> %s",
        file_path,
        file_manager._ctx,
        file_id,
    )

    if file_id is None:
        raise ValueError(
            f"Failed to retrieve file_id after creating record for: {file_path}",
        )

    storage_id = config_storage_id if config_storage_id else str(file_id)

    if not config_storage_id:
        dm = file_manager._data_manager
        dm.update_rows(
            context=file_manager._ctx,
            updates={"storage_id": storage_id},
            filter=f"file_id == {file_id}",
        )

    logger.info(
        "[TaskFn] File record finalised: file_path=%s file_id=%s storage_id=%s",
        file_path,
        file_id,
        storage_id,
    )

    return {
        "file_id": file_id,
        "file_path": file_path,
        "storage_id": storage_id,
        "created_file_record": created_file_record,
    }


# =============================================================================
# CONTENT INGEST TASK  (delegates to dm.ingest via ingest_content_batch)
# =============================================================================


def execute_ingest_content(
    *,
    file_manager: Any,
    file_path: str,
    content_rows: Optional[List[FileContentRow]] = None,
    content_rows_handle: Optional[TableInputHandle] = None,
    config: FilePipelineConfig,
    on_task_complete=None,
    storage_client=None,
    skip_rows: int = 0,
) -> Dict[str, Any]:
    """Ingest ALL content rows for a file via ``dm.ingest()``.

    Either ``content_rows`` (in-process path, inline Pydantic models) or
    ``content_rows_handle`` (worker path, handle backed by an object
    store) must be supplied.  When a handle is provided the rows are
    streamed in batches via
    :func:`unity.common.pipeline.row_streaming.iter_table_input_row_batches`
    and rebuilt into ``FileContentRow`` models before being passed to
    ``ingest_content_batch``; the manifest never materialises the full
    document content in memory.

    Chunking, retry, and embedding are handled internally by DM's ingest
    pipeline.  This function:

    1. Resolves ``file_id``, ``storage_id``, and the content context path.
    2. Optionally deletes existing rows (``replace_existing``).
    3. Transforms ``FileContentRow`` objects into document entries.
    4. Resolves embed columns from the FM config.
    5. Calls ``dm.ingest()`` with all rows in a single call.

    Returns
    -------
    dict
        ``{"ingest_result": IngestResult, "context": str, "row_count": int}``

    Raises
    ------
    Exception
        On fatal ingest failure.
    """
    from .ingest_ops import (
        get_file_id_from_path,
        get_storage_id_from_path,
        ingest_content_batch,
        resolve_embed_columns_for_content,
        resolve_embed_strategy,
        build_dm_execution_config,
    )
    from unity.common.model_to_fields import model_to_fields
    from unity.common.pipeline.row_streaming import iter_table_input_row_batches
    from unity.file_manager.types.file import FileContent

    dm = file_manager._data_manager

    # Materialise the content rows from whichever input channel the caller
    # provided.  Handles are drained in batches so memory stays bounded
    # even for very large lowered documents.
    if content_rows_handle is not None and content_rows is not None:
        raise ValueError(
            "execute_ingest_content received both content_rows and content_rows_handle; "
            "supply exactly one.",
        )
    if content_rows_handle is not None:
        streamed: List[FileContentRow] = []
        for batch in iter_table_input_row_batches(
            content_rows_handle,
            batch_size=max(int(config.ingest.content_rows_batch_size or 1000), 1),
            storage_client=storage_client,
            skip_rows=skip_rows,
        ):
            for row in batch:
                streamed.append(FileContentRow.model_validate(row))
        content_rows = streamed
    elif content_rows is None:
        content_rows = []

    logger.debug(
        f"[TaskFn] Ingesting content for {file_path} ({len(content_rows)} rows)",
    )

    file_id = get_file_id_from_path(
        data_manager=dm,
        index_context=file_manager._ctx,
        file_path=file_path,
    )
    if file_id is None:
        raise ValueError(f"File ID not found for {file_path}")

    storage_id = get_storage_id_from_path(
        data_manager=dm,
        index_context=file_manager._ctx,
        file_path=file_path,
    )
    if not storage_id:
        storage_id = str(file_id)

    if not content_rows:
        from unity.data_manager.types.ingest import IngestResult

        return {
            "ingest_result": IngestResult(context=""),
            "context": "",
            "row_count": 0,
        }

    context = file_manager._ctx_for_file_content(storage_id)
    is_shared_storage = storage_id != str(file_id)

    if config.ingest.replace_existing:
        from .ops import delete_file_content_rows

        try:
            if is_shared_storage:
                delete_file_content_rows(
                    file_manager,
                    storage_id=storage_id,
                    filter_expr=f"file_id == {file_id}",
                )
            else:
                delete_file_content_rows(
                    file_manager,
                    storage_id=storage_id,
                    filter_expr=None,
                )
        except Exception as e:
            logger.warning(f"[TaskFn] Failed to delete existing rows: {e}")

    embed_columns = resolve_embed_columns_for_content(file_path, config)
    embed_strategy = resolve_embed_strategy(config)
    execution = build_dm_execution_config(config)

    content_fields = model_to_fields(FileContent)

    result = ingest_content_batch(
        data_manager=dm,
        context=context,
        rows=content_rows,
        file_id=file_id,
        description=f"Content context for storage_id={storage_id}",
        fields=content_fields,
        unique_keys={"row_id": "int"},
        auto_counting={"row_id": None},
        embed_columns=embed_columns or None,
        embed_strategy=embed_strategy,
        chunk_size=config.ingest.content_rows_batch_size,
        infer_untyped_fields=config.ingest.infer_untyped_fields,
        add_to_all_context=file_manager.include_in_multi_assistant_table,
        execution=execution,
        on_task_complete=on_task_complete,
    )

    logger.debug(
        f"[TaskFn] Content ingest complete for {file_path}: "
        f"{result.rows_inserted} rows, {result.rows_embedded} embedded",
    )

    return {
        "ingest_result": result,
        "context": context,
        "row_count": result.rows_inserted,
    }


# =============================================================================
# TABLE INGEST TASK  (delegates to dm.ingest)
# =============================================================================


def execute_ingest_table(
    *,
    file_manager: Any,
    file_path: str,
    table_label: str,
    table_rows: Optional[List[Dict[str, Any]]] = None,
    table_input: Optional[TableInputHandle] = None,
    columns: List[str],
    config: FilePipelineConfig,
    on_task_complete=None,
    storage_client=None,
    skip_rows: int = 0,
) -> Dict[str, Any]:
    """Ingest ALL rows for one table via ``dm.ingest()``.

    Chunking, retry, and embedding are handled internally by DM's ingest
    pipeline.  This function:

    1. Resolves ``file_id``, ``storage_id``, and the table context path.
    2. Resolves table description from business context.
    3. Resolves embed columns from the FM config.
    4. Calls ``dm.ingest()`` with all rows for this table.

    Returns
    -------
    dict
        ``{"ingest_result": IngestResult, "table_label": str, "context": str, "row_count": int}``

    Raises
    ------
    Exception
        On fatal ingest failure.
    """
    from .ingest_ops import (
        get_file_id_from_path,
        get_storage_id_from_path,
        resolve_embed_columns_for_table,
        resolve_embed_strategy,
        build_dm_execution_config,
    )

    dm = file_manager._data_manager
    logger.debug(
        "[TaskFn] Ingesting table '%s' for %s",
        table_label,
        file_path,
    )

    file_id = get_file_id_from_path(
        data_manager=dm,
        index_context=file_manager._ctx,
        file_path=file_path,
    )
    storage_id = get_storage_id_from_path(
        data_manager=dm,
        index_context=file_manager._ctx,
        file_path=file_path,
    )
    if not storage_id and file_id is not None:
        storage_id = str(file_id)

    if not storage_id:
        raise ValueError(f"No storage_id found for {file_path}")

    if table_input is None:
        table_input = InlineRowsHandle(
            rows=list(table_rows or []),
            columns=list(columns or []),
            row_count=len(list(table_rows or [])),
        )

    if isinstance(table_input, InlineRowsHandle) and not table_input.rows:
        from unity.data_manager.types.ingest import IngestResult

        return {
            "ingest_result": IngestResult(context=""),
            "table_label": table_label,
            "context": "",
            "row_count": 0,
        }

    context = file_manager._ctx_for_file_table(storage_id, table_label)

    business_context = _lookup_table_business_context(
        file_path=file_path,
        table_label=table_label,
        config=config,
    )
    description = _build_table_description(table_label, business_context)
    fields = _build_table_fields(columns, business_context=business_context)

    embed_columns = resolve_embed_columns_for_table(
        file_path,
        table_label,
        config,
        safe_fn=file_manager.safe,
    )
    embed_strategy = resolve_embed_strategy(config)
    execution = build_dm_execution_config(config)
    batch_size = config.ingest.table_rows_batch_size

    rows: Optional[List[Dict[str, Any]]] = None
    handle: Optional[TableInputHandle] = None
    if isinstance(table_input, InlineRowsHandle):
        rows = list(table_input.rows)
    else:
        handle = table_input

    result = dm.ingest(
        context,
        rows,
        table_input_handle=handle,
        description=description,
        fields=fields,
        unique_keys={"row_id": "int"},
        auto_counting={"row_id": None},
        embed_columns=embed_columns or None,
        embed_strategy=embed_strategy,
        chunk_size=batch_size,
        infer_untyped_fields=config.ingest.infer_untyped_fields,
        add_to_all_context=file_manager.include_in_multi_assistant_table,
        execution=execution,
        on_task_complete=on_task_complete,
        storage_client=storage_client,
        skip_rows=skip_rows,
    )

    logger.debug(
        f"[TaskFn] Table '{table_label}' ingest complete for {file_path}: "
        f"{result.rows_inserted} rows, {result.rows_embedded} embedded",
    )

    return {
        "ingest_result": result,
        "table_label": table_label,
        "context": context,
        "row_count": result.rows_inserted,
    }
