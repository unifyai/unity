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

All data operations delegate to DataManager for consistency.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, TypeVar

from unity.file_manager.types.file import FileContent
from unity.file_manager.types.file import FileContentRow

if TYPE_CHECKING:
    from unity.data_manager.data_manager import DataManager

logger = logging.getLogger(__name__)

T = TypeVar("T")


def ingest_content_batch(
    *,
    data_manager: "DataManager",
    context: str,
    rows: List[FileContentRow],
    file_id: int,
    add_to_all_context: bool = False,
) -> List[int]:
    """
    Ingest a batch of content rows into a context via DataManager.

    Parameters
    ----------
    data_manager : DataManager
        The DataManager instance for data operations.
    context : str
        The Unify context to ingest into.
    rows : list[FileContentRow]
        Content rows to ingest.
    file_id : int
        The file ID to associate with these rows.
    add_to_all_context : bool
        Whether to add to multi-assistant aggregation contexts.

    Returns
    -------
    list[int]
        Log IDs of the inserted rows.
    """
    if not rows:
        return []

    # Transform rows to file content entries (attach file_id)
    file_content_rows = FileContent.to_document_entries(file_id=file_id, rows=rows)
    file_content_entries: List[Dict[str, Any]] = [
        r.model_dump(mode="json", exclude_none=True) for r in file_content_rows
    ]

    # Batch insert via DataManager
    try:
        return data_manager.insert_rows(
            context=context,
            rows=file_content_entries,
            add_to_all_context=add_to_all_context,
        )
    except Exception as e:
        logger.error(f"Failed to ingest content batch: {e}")
        raise


def chunk_records(
    records: List[T],
    batch_size: int,
) -> List[List[T]]:
    """
    Split records into chunks of the specified batch size.

    Parameters
    ----------
    records : list
        Records to chunk.
    batch_size : int
        Maximum size of each chunk.

    Returns
    -------
    list[list]
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
    data_manager: "DataManager",
    index_context: str,
    file_path: str,
) -> Optional[int]:
    """
    Look up file_id from the FileRecords index via DataManager.

    Parameters
    ----------
    data_manager : DataManager
        The DataManager instance for data operations.
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
    """
    Look up storage_id from the FileRecords index via DataManager.

    Parameters
    ----------
    data_manager : DataManager
        The DataManager instance for data operations.
    index_context : str
        The FileRecords index context.
    file_path : str
        The file path to look up.

    Returns
    -------
    str | None
        The storage_id if found, None otherwise.
        If storage_id is empty in the record, returns str(file_id).
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
            # If storage_id is empty, use str(file_id)
            if not storage_id and file_id is not None:
                return str(file_id)
            return storage_id or None
    except Exception as e:
        logger.warning(f"Failed to lookup storage_id for {file_path}: {e}")
    return None
