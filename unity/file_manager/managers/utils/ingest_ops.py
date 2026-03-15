"""Pure batch ingestion operations for the FileManager pipeline.

This module provides the bridge between FileManager's file-oriented pipeline
and DataManager's ``ingest()`` API.  Functions here:

- Transform file-specific row formats (``FileContentRow``) into plain dicts.
- Resolve FM configuration (embed specs, retry config, batch sizes) into
  the parameters expected by ``DataManager.ingest()``.
- Provide file-record lookup helpers (``get_file_id_from_path``, etc.)
  used throughout the pipeline.

All heavy lifting (chunking, parallelism, retry, embedding) is delegated
to ``DataManager.ingest()``; this module intentionally stays thin.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, TypeVar

from unity.data_manager.types.ingest import IngestExecutionConfig, IngestResult
from unity.file_manager.types.file import FileContent, FileContentRow

if TYPE_CHECKING:
    from unity.data_manager.data_manager import DataManager
    from unity.file_manager.types.config import (
        FilePipelineConfig,
    )

logger = logging.getLogger(__name__)

T = TypeVar("T")


# ---------------------------------------------------------------------------
# Row helpers
# ---------------------------------------------------------------------------


def with_infer_untyped_fields(
    rows: List[Dict[str, Any]],
    enabled: bool = True,
) -> List[Dict[str, Any]]:
    """Conditionally add ``infer_untyped_fields=True`` to each row dict.

    Parameters
    ----------
    rows : list[dict]
        Rows to augment.
    enabled : bool
        When ``False``, returns *rows* unchanged.

    Returns
    -------
    list[dict]
    """
    if not enabled:
        return rows
    return [{**row, "infer_untyped_fields": True} for row in rows]


def chunk_records(
    records: List[T],
    batch_size: int,
) -> List[List[T]]:
    """Split *records* into sublists of at most *batch_size* elements.

    Parameters
    ----------
    records : list
        Records to chunk.
    batch_size : int
        Maximum size of each chunk.

    Returns
    -------
    list[list]
    """
    if batch_size <= 0:
        batch_size = 1000
    return [records[i : i + batch_size] for i in range(0, len(records), batch_size)]


# ---------------------------------------------------------------------------
# Config translation  (FM config → DM.ingest() parameters)
# ---------------------------------------------------------------------------


def build_dm_execution_config(config: "FilePipelineConfig") -> IngestExecutionConfig:
    """Translate FM pipeline config into :class:`IngestExecutionConfig`.

    Maps:
    - ``config.execution.max_embed_workers``  →  ``max_workers``
    - ``config.retry.max_retries``            →  ``max_retries``
    - ``config.retry.retry_delay_seconds``    →  ``retry_delay_seconds``
    - ``config.retry.fail_fast``              →  ``fail_fast``
    """
    return IngestExecutionConfig(
        max_workers=getattr(config.execution, "max_embed_workers", 4),
        max_retries=getattr(config.retry, "max_retries", 3),
        retry_delay_seconds=getattr(config.retry, "retry_delay_seconds", 3.0),
        fail_fast=getattr(config.retry, "fail_fast", False),
    )


def resolve_embed_columns_for_content(
    file_path: str,
    config: "FilePipelineConfig",
) -> List[str]:
    """Extract source column names to embed for content contexts.

    Reads ``config.embed.file_specs`` and returns the deduplicated list of
    source columns whose ``FileEmbeddingSpec.context`` is ``"per_file"`` or
    ``"unified"`` and whose file path matches *file_path* (or ``"*"``).

    Returns an empty list when embedding is ``"off"`` or no specs match.
    """
    strategy = getattr(config.embed, "strategy", "off")
    if strategy == "off":
        return []

    file_specs = getattr(config.embed, "file_specs", []) or []
    columns: List[str] = []

    for fs in file_specs:
        if fs.file_path != "*" and fs.file_path != file_path:
            continue
        if fs.context not in ("per_file", "unified"):
            continue
        for ts in fs.tables:
            columns.extend(ts.source_columns)

    # Deduplicate while preserving order
    return list(dict.fromkeys(columns))


def resolve_embed_columns_for_table(
    file_path: str,
    table_label: str,
    config: "FilePipelineConfig",
    safe_fn: Optional[callable] = None,
) -> List[str]:
    """Extract source column names to embed for a specific table context.

    Reads ``config.embed.file_specs`` and returns the deduplicated list of
    source columns whose ``FileEmbeddingSpec.context`` is ``"per_file_table"``
    and whose table label matches *table_label*.

    Parameters
    ----------
    file_path : str
        File path for matching against ``FileEmbeddingSpec.file_path``.
    table_label : str
        Table label for matching against ``TableEmbeddingSpec.table``.
    config : FilePipelineConfig
        Pipeline configuration.
    safe_fn : callable | None
        Optional sanitisation function for table label comparison.

    Returns
    -------
    list[str]
        Deduplicated source column names.
    """
    strategy = getattr(config.embed, "strategy", "off")
    if strategy == "off":
        return []

    if safe_fn is None:
        safe_fn = lambda x: x

    file_specs = getattr(config.embed, "file_specs", []) or []
    columns: List[str] = []

    for fs in file_specs:
        if fs.file_path != "*" and fs.file_path != file_path:
            continue
        if fs.context != "per_file_table":
            continue
        for ts in fs.tables:
            table_filter = ts.table
            if table_filter in (None, "*"):
                columns.extend(ts.source_columns)
                continue
            try:
                safe_target = safe_fn(str(table_filter))
            except Exception:
                safe_target = str(table_filter)
            if table_label == safe_target or table_label == str(table_filter):
                columns.extend(ts.source_columns)

    return list(dict.fromkeys(columns))


def resolve_embed_strategy(config: "FilePipelineConfig") -> str:
    """Return the embed strategy string from the FM config (``"off"``, ``"along"``, ``"after"``)."""
    return getattr(config.embed, "strategy", "off")


# ---------------------------------------------------------------------------
# Content ingestion  (delegates to dm.ingest)
# ---------------------------------------------------------------------------


def ingest_content_batch(
    *,
    data_manager: "DataManager",
    context: str,
    rows: List[FileContentRow],
    file_id: int,
    description: Optional[str] = None,
    fields: Optional[Dict[str, str]] = None,
    unique_keys: Optional[Dict[str, str]] = None,
    auto_counting: Optional[Dict[str, Optional[str]]] = None,
    embed_columns: Optional[List[str]] = None,
    embed_strategy: str = "off",
    chunk_size: int = 1000,
    infer_untyped_fields: bool = False,
    add_to_all_context: bool = False,
    execution: Optional[IngestExecutionConfig] = None,
) -> IngestResult:
    """Ingest content rows into a context via ``DataManager.ingest()``.

    Transforms ``FileContentRow`` objects into document entries (attaching
    *file_id*), then delegates chunking, insertion, and optional embedding
    entirely to ``DataManager.ingest()``.

    Parameters
    ----------
    data_manager : DataManager
        The DataManager instance.
    context : str
        Target Unify context.
    rows : list[FileContentRow]
        Content rows to ingest.
    file_id : int
        File ID to stamp onto each row.
    description, fields, unique_keys, auto_counting :
        Passed through to ``dm.ingest()`` for table provisioning.
    embed_columns : list[str] | None
        Source columns to embed (resolved from FM embed specs by the caller).
    embed_strategy : str
        ``"off"``, ``"along"``, or ``"after"``.
    chunk_size : int
        Maximum rows per internal DM chunk.
    infer_untyped_fields : bool
        Instruct backend to infer types for undeclared fields.
    add_to_all_context : bool
        Whether to add rows to cross-assistant aggregation contexts.
    execution : IngestExecutionConfig | None
        Pipeline execution settings.

    Returns
    -------
    IngestResult
        Aggregated ingest outcome from DM.
    """
    if not rows:
        return IngestResult(context=context)

    file_content_rows = FileContent.to_document_entries(file_id=file_id, rows=rows)
    row_dicts: List[Dict[str, Any]] = [
        r.model_dump(mode="json", exclude_none=True) for r in file_content_rows
    ]

    try:
        return data_manager.ingest(
            context,
            row_dicts,
            description=description,
            fields=fields,
            unique_keys=unique_keys,
            auto_counting=auto_counting,
            embed_columns=embed_columns or None,
            embed_strategy=embed_strategy,
            chunk_size=chunk_size,
            infer_untyped_fields=infer_untyped_fields,
            add_to_all_context=add_to_all_context,
            execution=execution,
        )
    except Exception as e:
        logger.error(f"Failed to ingest content batch: {e}")
        raise


# ---------------------------------------------------------------------------
# Table ingestion  (delegates to dm.ingest)
# ---------------------------------------------------------------------------


def ingest_table_batch(
    *,
    data_manager: "DataManager",
    context: str,
    rows: List[Dict[str, Any]],
    description: Optional[str] = None,
    fields: Optional[Dict[str, str]] = None,
    unique_keys: Optional[Dict[str, str]] = None,
    auto_counting: Optional[Dict[str, Optional[str]]] = None,
    embed_columns: Optional[List[str]] = None,
    embed_strategy: str = "off",
    chunk_size: int = 100,
    infer_untyped_fields: bool = False,
    add_to_all_context: bool = False,
    execution: Optional[IngestExecutionConfig] = None,
) -> IngestResult:
    """Ingest table rows into a context via ``DataManager.ingest()``.

    Delegates chunking, insertion, and optional embedding entirely to
    ``DataManager.ingest()``.

    Parameters
    ----------
    data_manager : DataManager
        The DataManager instance.
    context : str
        Target Unify context (e.g. ``Files/<alias>/<sid>/Tables/<label>``).
    rows : list[dict]
        Table rows to ingest.
    description : str | None
        Table description from business context.
    fields : dict[str, str] | None
        Column type schema.
    unique_keys : dict[str, str] | None
        Unique-key column definitions for table creation
        (e.g. ``{"row_id": "int"}``).
    auto_counting : dict[str, str | None] | None
        Auto-incrementing column configuration
        (e.g. ``{"row_id": None}``).
    embed_columns : list[str] | None
        Source columns to embed.
    embed_strategy : str
        ``"off"``, ``"along"``, or ``"after"``.
    chunk_size : int
        Maximum rows per internal DM chunk.
    infer_untyped_fields : bool
        Instruct backend to infer types for undeclared fields.
    add_to_all_context : bool
        Whether to add rows to cross-assistant aggregation contexts.
    execution : IngestExecutionConfig | None
        Pipeline execution settings.

    Returns
    -------
    IngestResult
        Aggregated ingest outcome from DM.
    """
    if not rows:
        return IngestResult(context=context)

    try:
        return data_manager.ingest(
            context,
            rows,
            description=description,
            fields=fields,
            unique_keys=unique_keys,
            auto_counting=auto_counting,
            embed_columns=embed_columns or None,
            embed_strategy=embed_strategy,
            chunk_size=chunk_size,
            infer_untyped_fields=infer_untyped_fields,
            add_to_all_context=add_to_all_context,
            execution=execution,
        )
    except Exception as e:
        logger.error(f"Failed to ingest table batch: {e}")
        raise


# ---------------------------------------------------------------------------
# File-record lookup helpers
# ---------------------------------------------------------------------------


def get_file_id_from_path(
    *,
    data_manager: "DataManager",
    index_context: str,
    file_path: str,
) -> Optional[int]:
    """Look up ``file_id`` from the FileRecords index via DataManager.

    Parameters
    ----------
    data_manager : DataManager
        The DataManager instance.
    index_context : str
        The FileRecords index context.
    file_path : str
        The file path to look up.

    Returns
    -------
    int | None
        The ``file_id`` if found, ``None`` otherwise.
    """
    try:
        rows = data_manager.filter(
            context=index_context,
            filter=f"file_path == {file_path!r}",
            limit=1,
            columns=["file_id"],
        )
        if rows:
            return rows[0].get("file_id")
    except Exception as e:
        logger.warning(f"Failed to lookup file_id for {file_path}: {e}")
    return None


def get_storage_id_from_path(
    *,
    data_manager: "DataManager",
    index_context: str,
    file_path: str,
) -> Optional[str]:
    """Look up ``storage_id`` from the FileRecords index via DataManager.

    Parameters
    ----------
    data_manager : DataManager
        The DataManager instance.
    index_context : str
        The FileRecords index context.
    file_path : str
        The file path to look up.

    Returns
    -------
    str | None
        The ``storage_id`` if found, ``None`` otherwise.
        Falls back to ``str(file_id)`` when the stored value is empty.
    """
    try:
        rows = data_manager.filter(
            context=index_context,
            filter=f"file_path == {file_path!r}",
            limit=1,
            columns=["file_id", "storage_id"],
        )
        if rows:
            entry = rows[0]
            storage_id = entry.get("storage_id", "")
            file_id = entry.get("file_id")
            if not storage_id and file_id is not None:
                return str(file_id)
            return storage_id or None
    except Exception as e:
        logger.warning(f"Failed to lookup storage_id for {file_path}: {e}")
    return None
