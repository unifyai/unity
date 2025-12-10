"""Task graph factory for the FileManager pipeline.

This module builds task graphs that model the dependencies between
pipeline operations. The graph structure encodes the execution order
and parallelism constraints.

Key insight: The difference between "along" and "after" embed strategies
is purely in the dependency edges - the executor doesn't need to know
about strategies at all.

Dependency Model for "along" (enables non-blocking embed):

    file_record
         │
         ▼
    ingest_chunk_1 ──────────► embed_chunk_1 (async)
         │
         ▼
    ingest_chunk_2 ──────────► embed_chunk_2 (async)
         │
         ▼
    ingest_chunk_3 ──────────► embed_chunk_3 (async)
         │
         ▼
    (same pattern for tables if enabled)

Dependency Model for "after" (all embeds wait for all ingests):

    file_record
         │
         ▼
    ingest_chunk_1 ────┐
         │             │
         ▼             │
    ingest_chunk_2 ────┼──► embed_chunk_1
         │             │    embed_chunk_2
         ▼             │    embed_chunk_3
    ingest_chunk_3 ────┘    (all wait for all ingests)

The executor walks the graph, executing ready tasks (those with all
dependencies satisfied) potentially in parallel.
"""

from __future__ import annotations

import logging
import re
import uuid
from typing import Any, List, Optional, Tuple, TYPE_CHECKING

from unity.file_manager.types.file import ParsedFile
from .ingest_ops import chunk_records, extract_table_metadata
from .embed_ops import has_embedding_work

if TYPE_CHECKING:
    from unity.file_manager.types.config import FilePipelineConfig
    from .executor import TaskGraph

logger = logging.getLogger(__name__)


# =============================================================================
# TASK ID GENERATION
# =============================================================================


def _safe_id_component(value: str, max_len: int = 30) -> str:
    """
    Sanitize a string for use in task IDs.

    Parameters
    ----------
    value : str
        The string to sanitize.
    max_len : int
        Maximum length of the result.

    Returns
    -------
    str
        Sanitized string safe for use in task IDs.
    """
    # Replace non-alphanumeric with underscores
    safe = re.sub(r"[^a-zA-Z0-9]", "_", value)
    # Collapse multiple underscores
    safe = re.sub(r"_+", "_", safe)
    # Strip leading/trailing underscores
    safe = safe.strip("_")
    # Truncate
    if len(safe) > max_len:
        safe = safe[:max_len]
    return safe.lower() or "item"


def create_task_id(
    task_type: str,
    file_path: str,
    chunk_index: Optional[int] = None,
    table_label: Optional[str] = None,
) -> str:
    """
    Create a unique, deterministic, debuggable task ID.

    Format: {type}__{safe_file}__{chunk/table}__{uuid6}
    Example: ingest_content__sales_xlsx__chunk_001__a1b2c3

    Parameters
    ----------
    task_type : str
        The task type value (e.g., "ingest_content", "embed_table").
    file_path : str
        The file path.
    chunk_index : int | None
        Optional chunk index for batched operations.
    table_label : str | None
        Optional table label for table operations.

    Returns
    -------
    str
        A unique task ID.
    """
    # Start with task type
    parts = [task_type]

    # Add sanitized file path
    safe_file = _safe_id_component(file_path, max_len=20)
    parts.append(safe_file)

    # Add table label if present
    if table_label is not None:
        safe_table = _safe_id_component(table_label, max_len=15)
        parts.append(f"tbl_{safe_table}")

    # Add chunk index if present
    if chunk_index is not None:
        parts.append(f"chunk_{chunk_index:03d}")

    # Add short UUID for uniqueness
    parts.append(uuid.uuid4().hex[:6])

    return "__".join(parts)


# =============================================================================
# TASK GRAPH BUILDING
# =============================================================================


def build_file_task_graph(
    file_manager: Any,
    *,
    file_path: str,
    document: Any,
    parse_result: ParsedFile,
    config: "FilePipelineConfig",
    file_start_time: float,
) -> "TaskGraph":
    """
    Build a complete task graph for processing one file.

    The graph structure depends on the embed strategy:
    - "along": embed tasks depend only on their corresponding ingest chunk
    - "after": all embed tasks depend on ALL ingest tasks completing first
    - "off": no embed tasks created

    Parameters
    ----------
    file_manager : FileManager
        The file manager instance.
    file_path : str
        The logical file path/identifier.
    document : Any
        The parsed document object.
    parse_result : ParsedFile
        The ParsedFile Pydantic model from Document.to_parse_result().
    config : FilePipelineConfig
        Pipeline configuration.
    file_start_time : float
        Timestamp when file processing started (for elapsed time calculation).

    Returns
    -------
    TaskGraph
        A complete task graph ready for execution.
    """
    # Import here to avoid circular imports
    from .executor import Task, TaskGraph, TaskType
    from .task_functions import (
        execute_create_file_record,
    )

    logger.debug(f"[TaskFactory] Building task graph for: {file_path}")

    graph = TaskGraph(file_path)

    # Determine embed strategy
    strategy = getattr(config.embed, "strategy", "after")
    do_embed = strategy != "off" and has_embedding_work(config, file_path)

    logger.debug(f"[TaskFactory] Strategy: {strategy}, do_embed: {do_embed}")

    # -------------------------------------------------------------------------
    # 1. FILE RECORD TASK (root of the graph)
    # -------------------------------------------------------------------------
    file_record_task_id = create_task_id("file_record", file_path)
    file_record_task = Task(
        id=file_record_task_id,
        task_type=TaskType.INGEST_CONTENT,  # Using INGEST_CONTENT as closest match
        file_path=file_path,
        func=execute_create_file_record,
        kwargs={
            "file_manager": file_manager,
            "file_path": file_path,
            "parse_result": parse_result,
            "config": config,
        },
        dependencies=set(),  # No dependencies - root task
        chunk_index=None,
        total_chunks=None,
        table_label=None,
        metadata={
            "file_start_time": file_start_time,
            "is_file_record": True,
        },
    )
    graph.add_task(file_record_task)

    # -------------------------------------------------------------------------
    # 2. CONTENT CHUNK TASKS
    # -------------------------------------------------------------------------
    content_ingest_ids, content_embed_ids = _build_content_chunk_tasks(
        graph,
        file_manager=file_manager,
        file_path=file_path,
        parse_result=parse_result,
        config=config,
        file_record_task_id=file_record_task_id,
        file_start_time=file_start_time,
        strategy=strategy,
        do_embed=do_embed,
    )

    # -------------------------------------------------------------------------
    # 3. TABLE CHUNK TASKS (if table_ingest enabled)
    # -------------------------------------------------------------------------
    table_ingest_ids: List[str] = []
    table_embed_ids: List[str] = []

    if config.ingest.table_ingest:
        table_ingest_ids, table_embed_ids = _build_table_chunk_tasks(
            graph,
            file_manager=file_manager,
            file_path=file_path,
            document=document,
            config=config,
            content_ingest_ids=content_ingest_ids,
            file_record_task_id=file_record_task_id,
            file_start_time=file_start_time,
            strategy=strategy,
            do_embed=do_embed,
        )

    total_tasks = len(graph.tasks)
    logger.debug(
        f"[TaskFactory] Graph built: {total_tasks} tasks "
        f"(content_ingest={len(content_ingest_ids)}, content_embed={len(content_embed_ids)}, "
        f"table_ingest={len(table_ingest_ids)}, table_embed={len(table_embed_ids)})",
    )

    return graph


def _build_content_chunk_tasks(
    graph: "TaskGraph",
    *,
    file_manager: Any,
    file_path: str,
    parse_result: ParsedFile,
    config: "FilePipelineConfig",
    file_record_task_id: str,
    file_start_time: float,
    strategy: str,
    do_embed: bool,
) -> Tuple[List[str], List[str]]:
    """
    Build ingest and embed tasks for content chunks.

    Returns (ingest_task_ids, embed_task_ids) for dependency wiring.
    """
    from .executor import Task, TaskType
    from .task_functions import (
        execute_ingest_content_chunk,
        execute_embed_content_chunk,
    )

    records = list(parse_result.records or [])
    if not records:
        logger.debug(f"[TaskFactory] No content records for {file_path}")
        return [], []

    batch_size = config.ingest.content_rows_batch_size
    chunks = chunk_records(records, batch_size)
    total_chunks = len(chunks)

    logger.debug(
        f"[TaskFactory] Content: {len(records)} rows -> {total_chunks} chunks "
        f"(batch_size={batch_size})",
    )

    prev_ingest_id = file_record_task_id
    all_ingest_ids: List[str] = []
    all_embed_ids: List[str] = []

    for idx, chunk in enumerate(chunks):
        # -----------------------------------------------------------
        # INGEST TASK - depends on previous ingest (sequential chain)
        # -----------------------------------------------------------
        ingest_id = create_task_id("ingest_content", file_path, chunk_index=idx)
        ingest_task = Task(
            id=ingest_id,
            task_type=TaskType.INGEST_CONTENT,
            file_path=file_path,
            func=execute_ingest_content_chunk,
            kwargs={
                "file_manager": file_manager,
                "file_path": file_path,
                "chunk_records": chunk,
                "chunk_index": idx,
                "total_chunks": total_chunks,
                "config": config,
                "parse_result": parse_result,
            },
            dependencies={prev_ingest_id},  # Chain: N depends on N-1
            chunk_index=idx,
            total_chunks=total_chunks,
            table_label=None,
            metadata={
                "file_start_time": file_start_time,
                "row_count": len(chunk),
                "batch_size": batch_size,
            },
        )
        graph.add_task(ingest_task)
        all_ingest_ids.append(ingest_id)

        # -----------------------------------------------------------
        # EMBED TASK - dependency depends on strategy
        # -----------------------------------------------------------
        if do_embed:
            embed_id = create_task_id("embed_content", file_path, chunk_index=idx)

            # KEY DIFFERENCE: "along" vs "after"
            if strategy == "along":
                # Along: embed depends only on THIS chunk's ingest
                # This allows N+1 ingest to proceed while N embed runs
                embed_deps = {ingest_id}
            else:
                # After: embed will depend on ALL ingests (updated below)
                # For now, just depend on this chunk's ingest
                embed_deps = {ingest_id}

            embed_task = Task(
                id=embed_id,
                task_type=TaskType.EMBED_CONTENT,
                file_path=file_path,
                func=execute_embed_content_chunk,
                kwargs={
                    "file_manager": file_manager,
                    "file_path": file_path,
                    "inserted_ids": [],  # Will be populated from ingest result
                    "chunk_index": idx,
                    "total_chunks": total_chunks,
                    "config": config,
                },
                dependencies=embed_deps,
                chunk_index=idx,
                total_chunks=total_chunks,
                table_label=None,
                metadata={
                    "file_start_time": file_start_time,
                    "depends_on_ingest": ingest_id,
                },
            )
            graph.add_task(embed_task)
            all_embed_ids.append(embed_id)

        prev_ingest_id = ingest_id

    # For "after" strategy: update all embed deps to include ALL ingests
    if strategy == "after" and all_embed_ids:
        for embed_id in all_embed_ids:
            graph.tasks[embed_id].dependencies = set(all_ingest_ids)

    return all_ingest_ids, all_embed_ids


def _build_table_chunk_tasks(
    graph: "TaskGraph",
    *,
    file_manager: Any,
    file_path: str,
    document: Any,
    config: "FilePipelineConfig",
    content_ingest_ids: List[str],
    file_record_task_id: str,
    file_start_time: float,
    strategy: str,
    do_embed: bool,
) -> Tuple[List[str], List[str]]:
    """
    Build ingest and embed tasks for all tables in the document.

    Tables depend on the file_record task being complete.

    Returns (ingest_task_ids, embed_task_ids).
    """
    from .executor import Task, TaskType
    from .task_functions import (
        execute_ingest_table_chunk,
        execute_embed_table_chunk,
    )

    tables_meta = extract_table_metadata(document)
    if not tables_meta:
        logger.debug(f"[TaskFactory] No tables for {file_path}")
        return [], []

    batch_size = config.ingest.table_rows_batch_size

    all_ingest_ids: List[str] = []
    all_embed_ids: List[str] = []

    for table_meta in tables_meta:
        table_label = table_meta["label"]
        columns = table_meta["columns"]
        rows = table_meta["rows"]

        if not rows:
            continue

        chunks = chunk_records(rows, batch_size)
        total_chunks = len(chunks)

        logger.debug(
            f"[TaskFactory] Table '{table_label}': {len(rows)} rows -> "
            f"{total_chunks} chunks (batch_size={batch_size})",
        )

        # Tables depend on the file record, not on content
        prev_ingest_id = file_record_task_id
        table_ingest_ids: List[str] = []
        table_embed_ids: List[str] = []

        for idx, chunk in enumerate(chunks):
            # -----------------------------------------------------------
            # TABLE INGEST TASK
            # -----------------------------------------------------------
            ingest_id = create_task_id(
                "ingest_table",
                file_path,
                chunk_index=idx,
                table_label=table_label,
            )
            ingest_task = Task(
                id=ingest_id,
                task_type=TaskType.INGEST_TABLE,
                file_path=file_path,
                func=execute_ingest_table_chunk,
                kwargs={
                    "file_manager": file_manager,
                    "file_path": file_path,
                    "table_label": table_label,
                    "chunk_rows": chunk,
                    "columns": columns,
                    "chunk_index": idx,
                    "total_chunks": total_chunks,
                    "config": config,
                },
                dependencies={prev_ingest_id},
                chunk_index=idx,
                total_chunks=total_chunks,
                table_label=table_label,
                metadata={
                    "file_start_time": file_start_time,
                    "row_count": len(chunk),
                    "batch_size": batch_size,
                },
            )
            graph.add_task(ingest_task)
            table_ingest_ids.append(ingest_id)
            all_ingest_ids.append(ingest_id)

            # -----------------------------------------------------------
            # TABLE EMBED TASK
            # -----------------------------------------------------------
            if do_embed:
                embed_id = create_task_id(
                    "embed_table",
                    file_path,
                    chunk_index=idx,
                    table_label=table_label,
                )

                if strategy == "along":
                    embed_deps = {ingest_id}
                else:
                    embed_deps = {ingest_id}

                embed_task = Task(
                    id=embed_id,
                    task_type=TaskType.EMBED_TABLE,
                    file_path=file_path,
                    func=execute_embed_table_chunk,
                    kwargs={
                        "file_manager": file_manager,
                        "file_path": file_path,
                        "table_label": table_label,
                        "inserted_ids": [],
                        "chunk_index": idx,
                        "total_chunks": total_chunks,
                        "config": config,
                    },
                    dependencies=embed_deps,
                    chunk_index=idx,
                    total_chunks=total_chunks,
                    table_label=table_label,
                    metadata={
                        "file_start_time": file_start_time,
                        "depends_on_ingest": ingest_id,
                    },
                )
                graph.add_task(embed_task)
                table_embed_ids.append(embed_id)
                all_embed_ids.append(embed_id)

            prev_ingest_id = ingest_id

        # For "after" strategy: update table embed deps
        if strategy == "after" and table_embed_ids:
            for embed_id in table_embed_ids:
                # Table embeds depend on all table ingests for this table
                graph.tasks[embed_id].dependencies = set(table_ingest_ids)

    return all_ingest_ids, all_embed_ids
