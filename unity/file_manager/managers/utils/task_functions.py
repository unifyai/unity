"""Pure task execution functions for the FileManager pipeline.

This module contains the actual work functions that are executed as tasks
by the PipelineExecutor. Each function is PURE:
- Explicit parameters (no hidden state)
- Single responsibility (does one thing)
- Returns explicit results (dict with status/data)
- Raises exceptions on failure (caught by executor for retry)

These functions are designed to be called by the executor's task system.
They do NOT handle:
- Retries (handled by executor)
- Progress reporting (handled by executor)
- Timing (handled by executor)
- Parallelism (handled by executor)

The orchestration layer (task_factory.py) wires these functions into
task graphs with proper dependencies.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from unity.file_manager.file_parsers.types.contracts import FileParseResult
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
    """
    Look up the business context for a specific table.

    Searches config.ingest.business_contexts.file_contexts for a matching file_path,
    then finds the matching table spec within that file's table_contexts.

    Parameters
    ----------
    file_path : str
        The file path to match against.
    table_label : str
        The table label to match.
    config : FilePipelineConfig
        Pipeline configuration containing business contexts.

    Returns
    -------
    TableBusinessContextSpec | None
        The matching business context spec, or None if not found.
    """
    if not config or not hasattr(config, "ingest"):
        return None

    business_contexts = getattr(config.ingest, "business_contexts", None)
    if not business_contexts:
        return None

    # Access file_contexts from the BusinessContextsConfig
    file_contexts = getattr(business_contexts, "file_contexts", [])
    if not file_contexts:
        return None

    for fc in file_contexts:
        if fc.file_path != file_path:
            continue
        # Found matching file, now find matching table in table_contexts
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
    """
    Create the FileRecord entry in the index.

    This task MUST run before any content/table ingestion. It registers
    the file in the FileRecords index and returns the generated file_id
    and computed storage_id.

    Parameters
    ----------
    file_manager : FileManager
        The file manager instance providing storage context and identity helpers.
    file_path : str
        The logical file path/identifier.
    parse_result : FileParseResult
        The FileParseResult from the file parser (content_rows + tables + trace).
    config : FilePipelineConfig
        Pipeline configuration for ingest settings.

    Returns
    -------
    dict
        {"file_id": int, "file_path": str, "storage_id": str}

    Raises
    ------
    Exception
        On failure to create the file record. This halts the entire
        file's task graph since all other tasks depend on file_id.
    """
    from .ops import create_file_record as _ops_create_file_record
    from .ingest_ops import get_file_id_from_path
    from unity.file_manager.types.file import FileRecord

    dm = file_manager._data_manager
    logger.debug(f"[TaskFn] Creating file record for: {file_path}")

    # Determine ingest settings from config
    # storage_id from config (None means auto-assign using str(file_id))
    config_storage_id = config.ingest.storage_id
    table_ingest = config.ingest.table_ingest

    # Best-effort: adapter-derived size/timestamps (never raises)
    from .source_info import source_info_for_file

    ref = None
    try:
        ref = file_manager._adapter.get_file(file_path)  # type: ignore[attr-defined]
    except Exception:
        ref = None
    sinfo = source_info_for_file(
        adapter_ref=ref,
        trace=getattr(parse_result, "trace", None),
    )

    # Get source_uri and source_provider from adapter/resolver
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

    # Create the file record entry with empty storage_id initially
    # (will be updated with str(file_id) after we know file_id, unless config specifies one)
    entry = FileRecord.to_file_record_entry(
        file_path=file_path,
        source_uri=source_uri,
        source_provider=source_provider,
        parse_result=parse_result,
        storage_id=config_storage_id or "",  # Empty means auto-assign
        table_ingest=table_ingest,
        file_size=sinfo.size_bytes,
        created_at=sinfo.created_at,
        modified_at=sinfo.modified_at,
        total_records=total_records,
        document_summary=document_summary,
    )

    created_file_record = _ops_create_file_record(file_manager, entry=entry)
    logger.debug(f"[TaskFn] Created file record: {created_file_record}")

    # Lookup the created file_id
    file_id = get_file_id_from_path(
        data_manager=dm,
        index_context=file_manager._ctx,
        file_path=file_path,
    )

    if file_id is None:
        raise ValueError(
            f"Failed to retrieve file_id after creating record for: {file_path}",
        )

    # Compute effective storage_id
    # If config_storage_id is provided, use it; otherwise use str(file_id)
    storage_id = config_storage_id if config_storage_id else str(file_id)

    # Update the record with computed storage_id if we used auto-assign
    if not config_storage_id:
        dm = file_manager._data_manager
        dm.update_rows(
            context=file_manager._ctx,
            updates={"storage_id": storage_id},
            filter=f"file_id == {file_id}",
        )

    logger.debug(
        f"[TaskFn] File record created: file_id={file_id}, storage_id={storage_id}",
    )

    return {
        "file_id": file_id,
        "file_path": file_path,
        "storage_id": storage_id,
        "created_file_record": created_file_record,
    }


# =============================================================================
# CONTENT INGEST TASK
# =============================================================================


def execute_ingest_content_chunk(
    *,
    file_manager: Any,
    file_path: str,
    chunk_records: List[FileContentRow],
    chunk_index: int,
    total_chunks: int,
    config: FilePipelineConfig,
) -> Dict[str, Any]:
    """
    Ingest a single chunk of content rows.

    Parameters
    ----------
    file_manager : FileManager
        The file manager instance.
    file_path : str
        The logical file path/identifier.
    chunk_records : list[FileContentRow]
        The records for this chunk (already chunked by task_factory).
    chunk_index : int
        Zero-based index of this chunk.
    total_chunks : int
        Total number of content chunks for this file.
    config : FilePipelineConfig
        Pipeline configuration.

    Returns
    -------
    dict
        {
            "inserted_ids": list[int],
            "chunk_index": int,
            "row_count": int,
            "context": str,
        }

    Raises
    ------
    Exception
        On ingest failure. This halts the dependency chain - subsequent
        content chunks cannot be ingested.
    """
    from .storage import ensure_file_context as _storage_ensure_file_context
    from .ingest_ops import (
        get_file_id_from_path,
        get_storage_id_from_path,
        ingest_content_batch,
    )

    dm = file_manager._data_manager
    logger.debug(
        f"[TaskFn] Ingesting content chunk {chunk_index + 1}/{total_chunks} "
        f"for {file_path} ({len(chunk_records)} rows)",
    )

    # Get file_id (should exist from file_record task)
    file_id = get_file_id_from_path(
        data_manager=dm,
        index_context=file_manager._ctx,
        file_path=file_path,
    )
    if file_id is None:
        raise ValueError(f"File ID not found for {file_path}")

    # Get storage_id from the file record
    storage_id = get_storage_id_from_path(
        data_manager=dm,
        index_context=file_manager._ctx,
        file_path=file_path,
    )
    if not storage_id:
        storage_id = str(file_id)

    if not chunk_records:
        logger.debug(
            f"[TaskFn] No rows to ingest after preparation for chunk {chunk_index + 1}",
        )
        return {
            "inserted_ids": [],
            "chunk_index": chunk_index,
            "row_count": 0,
            "context": "",
        }

    # Ensure the content context exists (on first chunk)
    if chunk_index == 0:
        try:
            _storage_ensure_file_context(file_manager, storage_id=storage_id)
        except Exception as e:
            logger.warning(f"[TaskFn] Error ensuring content context: {e}")

    # Determine destination context using storage_id
    context = file_manager._ctx_for_file_content(storage_id)

    # Determine if this is shared storage (storage_id != str(file_id))
    is_shared_storage = storage_id != str(file_id)

    # Delete existing rows on first chunk only (if replace_existing)
    if chunk_index == 0 and config.ingest.replace_existing:
        from .ops import delete_file_content_rows

        try:
            if is_shared_storage:
                # Shared storage: delete only this file's rows using filter
                delete_file_content_rows(
                    file_manager,
                    storage_id=storage_id,
                    filter_expr=f"file_id == {file_id}",
                )
            else:
                # Per-file storage: delete all rows
                delete_file_content_rows(
                    file_manager,
                    storage_id=storage_id,
                    filter_expr=None,
                )
        except Exception as e:
            logger.warning(f"[TaskFn] Failed to delete existing rows: {e}")

    # Ingest the batch via DataManager
    inserted_ids = ingest_content_batch(
        data_manager=dm,
        context=context,
        rows=chunk_records,
        file_id=file_id,
        add_to_all_context=file_manager.include_in_multi_assistant_table,
    )

    logger.debug(
        f"[TaskFn] Ingested content chunk {chunk_index + 1}/{total_chunks}: "
        f"{len(inserted_ids)} rows inserted",
    )

    return {
        "inserted_ids": inserted_ids,
        "chunk_index": chunk_index,
        "row_count": len(chunk_records),
        "context": context,
    }


# =============================================================================
# CONTENT EMBED TASK
# =============================================================================


def execute_embed_content_chunk(
    *,
    file_manager: Any,
    file_path: str,
    inserted_ids: List[int],
    chunk_index: int,
    total_chunks: int,
    config: FilePipelineConfig,
) -> Dict[str, Any]:
    """
    Embed a single chunk of content using its inserted_ids.

    This task is NON-BLOCKING in "along" strategy - the executor allows
    ingest N+1 to proceed while this embed runs concurrently.

    Parameters
    ----------
    file_manager : FileManager
        The file manager instance.
    file_path : str
        The logical file path/identifier.
    inserted_ids : list[int]
        Log IDs from the corresponding ingest task.
    chunk_index : int
        Zero-based index of this chunk.
    total_chunks : int
        Total number of content chunks.
    config : FilePipelineConfig
        Pipeline configuration.

    Returns
    -------
    dict
        {
            "chunk_index": int,
            "success": bool,
            "columns_embedded": dict[str, bool],
            "embedded_count": int,
        }

    Notes
    -----
    Embed failures are GRACEFUL - they return success=False but do NOT
    raise exceptions. The executor records these for the final summary
    but does NOT halt the pipeline.
    """
    from .embed_ops import embed_content_batch, get_embedding_specs_for_file
    from .ingest_ops import get_storage_id_from_path, get_file_id_from_path

    dm = file_manager._data_manager
    logger.debug(
        f"[TaskFn] Embedding content chunk {chunk_index + 1}/{total_chunks} "
        f"for {file_path} ({len(inserted_ids)} ids)",
    )

    if not inserted_ids:
        logger.debug(f"[TaskFn] No IDs to embed for chunk {chunk_index + 1}")
        return {
            "chunk_index": chunk_index,
            "success": True,
            "columns_embedded": {},
            "embedded_count": 0,
        }

    # Get embedding specs for this file
    specs = get_embedding_specs_for_file(file_path, config)
    if not specs:
        logger.debug(f"[TaskFn] No embedding specs for {file_path}")
        return {
            "chunk_index": chunk_index,
            "success": True,
            "columns_embedded": {},
            "embedded_count": 0,
        }

    # Get file_id and storage_id for context resolution
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

    # Determine context using storage_id
    if storage_id:
        context = file_manager._ctx_for_file_content(storage_id)
    else:
        # Fallback (should not happen)
        logger.warning(f"[TaskFn] No storage_id found for {file_path}")
        return {
            "chunk_index": chunk_index,
            "success": False,
            "columns_embedded": {},
            "embedded_count": 0,
        }

    # Embed the batch
    results = embed_content_batch(
        context=context,
        specs=specs,
        inserted_ids=inserted_ids,
    )

    success = all(results.values()) if results else True
    if not success:
        failed_cols = [col for col, ok in results.items() if not ok]
        logger.warning(
            f"[TaskFn] Embed chunk {chunk_index + 1} partial failure: {failed_cols}",
        )

    logger.debug(
        f"[TaskFn] Embedded content chunk {chunk_index + 1}/{total_chunks}: "
        f"success={success}, columns={list(results.keys())}",
    )

    return {
        "chunk_index": chunk_index,
        "success": success,
        "columns_embedded": results,
        "embedded_count": len(inserted_ids),
    }


# =============================================================================
# TABLE INGEST TASK
# =============================================================================


def execute_ingest_table_chunk(
    *,
    file_manager: Any,
    file_path: str,
    table_label: str,
    chunk_rows: List[Dict[str, Any]],
    columns: List[str],
    chunk_index: int,
    total_chunks: int,
    config: FilePipelineConfig,
) -> Dict[str, Any]:
    """
    Ingest a single chunk of table rows.

    Parameters
    ----------
    file_manager : FileManager
        The file manager instance.
    file_path : str
        The logical file path/identifier.
    table_label : str
        The table label (e.g., sheet name).
    chunk_rows : list[dict]
        The rows for this chunk.
    columns : list[str]
        Column names for the table.
    chunk_index : int
        Zero-based index of this chunk within the table.
    total_chunks : int
        Total chunks for this table.
    config : FilePipelineConfig
        Pipeline configuration.

    Returns
    -------
    dict
        {
            "inserted_ids": list[int],
            "table_label": str,
            "chunk_index": int,
            "row_count": int,
            "context": str,
        }

    Raises
    ------
    Exception
        On ingest failure. This halts dependent embed tasks for this table.
    """
    from .storage import ensure_file_table_context as _storage_ensure_file_table_context
    from .ops import batch_insert_file_table_rows as _batch_insert_file_table_rows
    from .ingest_ops import get_file_id_from_path, get_storage_id_from_path

    dm = file_manager._data_manager
    logger.debug(
        f"[TaskFn] Ingesting table '{table_label}' chunk {chunk_index + 1}/{total_chunks} "
        f"for {file_path} ({len(chunk_rows)} rows)",
    )

    # Get file_id and storage_id for context resolution
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

    # Ensure the table context exists (on first chunk)
    if chunk_index == 0:
        try:
            # Look up business context for table/column descriptions
            business_context = _lookup_table_business_context(
                file_path=file_path,
                table_label=table_label,
                config=config,
            )
            example = (
                chunk_rows[0]
                if chunk_rows and isinstance(chunk_rows[0], dict)
                else None
            )
            _storage_ensure_file_table_context(
                file_manager,
                storage_id=storage_id,
                table=table_label,
                columns=columns,
                example_row=example,
                business_context=business_context,
            )
        except Exception as e:
            logger.warning(f"[TaskFn] Error ensuring table context: {e}")

    # Get context path and insert rows directly (ensure already done on first chunk)
    context = file_manager._ctx_for_file_table(storage_id, table_label)
    inserted_ids = _batch_insert_file_table_rows(
        file_manager,
        storage_id=storage_id,
        table=table_label,
        rows=chunk_rows,
    )

    logger.debug(
        f"[TaskFn] Ingested table '{table_label}' chunk {chunk_index + 1}/{total_chunks}: "
        f"{len(inserted_ids)} rows inserted",
    )

    return {
        "inserted_ids": inserted_ids,
        "table_label": table_label,
        "chunk_index": chunk_index,
        "row_count": len(chunk_rows),
        "context": context,
    }


# =============================================================================
# TABLE EMBED TASK
# =============================================================================


def execute_embed_table_chunk(
    *,
    file_manager: Any,
    file_path: str,
    table_label: str,
    inserted_ids: List[int],
    chunk_index: int,
    total_chunks: int,
    config: FilePipelineConfig,
) -> Dict[str, Any]:
    """
    Embed a single chunk of table rows.

    Parameters
    ----------
    file_manager : FileManager
        The file manager instance.
    file_path : str
        The logical file path/identifier.
    table_label : str
        The table label.
    inserted_ids : list[int]
        Log IDs from the corresponding ingest task.
    chunk_index : int
        Zero-based index of this chunk.
    total_chunks : int
        Total chunks for this table.
    config : FilePipelineConfig
        Pipeline configuration.

    Returns
    -------
    dict
        {
            "table_label": str,
            "chunk_index": int,
            "success": bool,
            "columns_embedded": dict[str, bool],
            "embedded_count": int,
        }

    Notes
    -----
    Like content embed, table embed failures are GRACEFUL and do not
    halt the pipeline.
    """
    from .embed_ops import embed_table_batch, get_embedding_specs_for_file
    from .ingest_ops import get_file_id_from_path, get_storage_id_from_path

    dm = file_manager._data_manager
    logger.debug(
        f"[TaskFn] Embedding table '{table_label}' chunk {chunk_index + 1}/{total_chunks} "
        f"for {file_path} ({len(inserted_ids)} ids)",
    )

    if not inserted_ids:
        return {
            "table_label": table_label,
            "chunk_index": chunk_index,
            "success": True,
            "columns_embedded": {},
            "embedded_count": 0,
        }

    # Get embedding specs
    specs = get_embedding_specs_for_file(file_path, config)
    if not specs:
        return {
            "table_label": table_label,
            "chunk_index": chunk_index,
            "success": True,
            "columns_embedded": {},
            "embedded_count": 0,
        }

    # Get file_id and storage_id for context resolution
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
        logger.warning(f"[TaskFn] No storage_id found for {file_path}")
        return {
            "table_label": table_label,
            "chunk_index": chunk_index,
            "success": False,
            "columns_embedded": {},
            "embedded_count": 0,
        }

    # Get context using storage_id
    context = file_manager._ctx_for_file_table(storage_id, table_label)

    # Embed the batch
    results = embed_table_batch(
        context=context,
        table_label=table_label,
        specs=specs,
        inserted_ids=inserted_ids,
        safe_fn=file_manager.safe,
    )

    success = all(results.values()) if results else True
    if not success:
        failed_cols = [col for col, ok in results.items() if not ok]
        logger.warning(
            f"[TaskFn] Embed table '{table_label}' chunk {chunk_index + 1} "
            f"partial failure: {failed_cols}",
        )

    logger.debug(
        f"[TaskFn] Embedded table '{table_label}' chunk {chunk_index + 1}/{total_chunks}: "
        f"success={success}",
    )

    return {
        "table_label": table_label,
        "chunk_index": chunk_index,
        "success": success,
        "columns_embedded": results,
        "embedded_count": len(inserted_ids),
    }
