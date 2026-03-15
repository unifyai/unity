"""Ingest operations for DataManager.

Implementation logic for :meth:`DataManager.ingest`.  Orchestrates table
creation, chunked row insertion, and optional embedding via the generic
pipeline engine (``unity.data_manager.utils.pipeline``).

This module is called by ``DataManager.ingest()`` and should not be used
directly by external callers.

Flow
----
1. Validate inputs, chunk *rows* by *chunk_size*.
2. Build a :class:`TaskGraph`:
   - ``create_table`` task (root -- idempotent table provisioning).
   - ``insert_chunk_N`` tasks (one per chunk, depend on ``create_table``).
   - If ``embed_columns`` is provided:
     - ``"along"`` strategy: one ``embed_chunk_N`` task per insert chunk,
       each depending on its corresponding ``insert_chunk_N``.
     - ``"after"`` strategy: a single ``embed_all`` task depending on
       **all** insert chunks.
3. Execute the graph via :class:`PipelineExecutor`.
4. Aggregate results into :class:`IngestResult`.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from unity.common.embed_utils import ensure_vector_column as _ensure_vector_column
from unity.data_manager.ops.mutation_ops import insert_rows_impl
from unity.data_manager.ops.table_ops import create_table_impl
from unity.data_manager.types.ingest import IngestExecutionConfig, IngestResult
from unity.data_manager.utils.pipeline import (
    ExecutionConfig,
    PipelineExecutor,
    Task,
    TaskGraph,
    TaskResult,
)

if TYPE_CHECKING:
    from unity.data_manager.data_manager import DataManager

logger = logging.getLogger(__name__)


_SENTINEL = object()


class _InsertedIdsStore:
    """Thread-safe store for passing inserted log IDs from insert tasks
    to their downstream embed tasks.

    Insert tasks call :meth:`put` after a successful insert.
    Embed tasks call :meth:`get` to retrieve the IDs -- if the upstream
    insert failed (never called ``put``), :meth:`get` returns the sentinel
    so the caller can distinguish *failure* from *empty-but-successful*.
    """

    def __init__(self) -> None:
        self._data: Dict[str, List[int]] = {}
        self._lock = threading.Lock()

    def put(self, key: str, ids: List[int]) -> None:
        with self._lock:
            if key in self._data:
                raise RuntimeError(
                    f"_InsertedIdsStore: duplicate write for key {key!r}. "
                    "Task IDs must be unique within a graph.",
                )
            self._data[key] = ids

    def get(self, key: str) -> Any:
        """Return the ids list, or ``_SENTINEL`` if the key was never written."""
        with self._lock:
            return self._data.get(key, _SENTINEL)

    def all_ids(self) -> List[int]:
        """Return a flat list of all stored IDs (for the "after" strategy)."""
        with self._lock:
            result: List[int] = []
            for v in self._data.values():
                result.extend(v)
            return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _chunk_rows(
    rows: List[Dict[str, Any]],
    chunk_size: int,
) -> List[List[Dict[str, Any]]]:
    """Split *rows* into sublists of at most *chunk_size* elements."""
    return [rows[i : i + chunk_size] for i in range(0, len(rows), chunk_size)]


def _make_create_table_func(
    context: str,
    *,
    description: Optional[str],
    fields: Optional[Dict[str, str]],
    unique_keys: Optional[Dict[str, str]],
    auto_counting: Optional[Dict[str, Optional[str]]],
    infer_untyped_fields: bool,
):
    """Return a zero-arg callable that creates the table context.

    Wrapped in a closure so the task graph can invoke it without arguments.
    """

    def _create():
        create_table_impl(
            context,
            description=description,
            fields=fields,
            unique_keys=unique_keys,
            auto_counting=auto_counting,
        )
        return {"context": context}

    return _create


def _make_insert_chunk_func(
    context: str,
    chunk: List[Dict[str, Any]],
    *,
    add_to_all_context: bool = False,
    ids_store: Optional[_InsertedIdsStore] = None,
    task_id: Optional[str] = None,
):
    """Return a zero-arg callable that inserts a single chunk of rows.

    On success the inserted log IDs are written to *ids_store* so that a
    downstream embed task can retrieve them.  If this task fails (exception
    propagates), ``put`` is never called and the embed task sees the
    sentinel value, allowing it to skip gracefully.
    """

    def _insert():
        log_ids = insert_rows_impl(
            context,
            chunk,
            add_to_all_context=add_to_all_context,
        )
        if ids_store is not None and task_id is not None:
            ids_store.put(task_id, log_ids)
        return {"inserted_ids": log_ids, "row_count": len(chunk)}

    return _insert


def _make_embed_func(
    context: str,
    embed_columns: List[str],
    *,
    async_embeddings: bool = True,
    ids_store: Optional[_InsertedIdsStore] = None,
    insert_task_id: Optional[str] = None,
):
    """Return a zero-arg callable that embeds columns for specific rows.

    Calls :func:`unity.common.embed_utils.ensure_vector_column` directly
    with ``from_ids`` so the backend both creates the derived column
    definition (idempotent, first call only) **and** processes the given
    row IDs.  Each column is embedded in parallel via a thread pool;
    ``ensure_vector_column`` is already thread-safe (per-column locks).

    For the ``"along"`` strategy, *insert_task_id* identifies the upstream
    insert task whose IDs to retrieve from *ids_store*.  If that insert
    failed (sentinel returned), embedding is skipped.

    For the ``"after"`` strategy, all successfully-inserted IDs are
    collected from the store.
    """

    def _embed():
        from concurrent.futures import ThreadPoolExecutor

        from_ids: Optional[List[int]] = None

        if ids_store is not None:
            if insert_task_id is not None:
                value = ids_store.get(insert_task_id)
                if value is _SENTINEL:
                    logger.warning(
                        "Skipping embed for %s -- upstream %s did not produce IDs "
                        "(likely failed)",
                        context,
                        insert_task_id,
                    )
                    return {"rows_embedded": 0, "skipped": True}
                from_ids = value
            else:
                from_ids = ids_store.all_ids() or None

        if not from_ids:
            logger.info("No row IDs to embed for %s -- skipping", context)
            return {"rows_embedded": 0}

        def _do_col(col: str) -> int:
            target = f"_{col}_emb"
            _ensure_vector_column(
                context=context,
                embed_column=target,
                source_column=col,
                derived_expr=None,
                from_ids=from_ids,
                async_embeddings=async_embeddings,
            )
            return len(from_ids)

        if len(embed_columns) == 1:
            total = _do_col(embed_columns[0])
        else:
            with ThreadPoolExecutor(max_workers=len(embed_columns)) as pool:
                total = sum(pool.map(_do_col, embed_columns))

        return {"rows_embedded": total}

    return _embed


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------


def _build_ingest_graph(
    context: str,
    chunks: List[List[Dict[str, Any]]],
    *,
    description: Optional[str],
    fields: Optional[Dict[str, str]],
    unique_keys: Optional[Dict[str, str]],
    embed_columns: Optional[List[str]],
    embed_strategy: str,
    auto_counting: Optional[Dict[str, Optional[str]]],
    infer_untyped_fields: bool,
    add_to_all_context: bool = False,
    total_rows: int = 0,
) -> TaskGraph:
    """Build a :class:`TaskGraph` for a full ingest operation.

    Parameters mirror those of :func:`run_ingest` (minus execution config).
    """
    # 8-char nonce makes every task ID globally unique across concurrent
    # graphs while keeping IDs human-readable in logs/progress output.
    nonce = uuid.uuid4().hex[:8]
    graph = TaskGraph(name=f"ingest:{context}:{nonce}")
    total_chunks = len(chunks)

    # Thread-safe store: insert tasks write their log IDs here so downstream
    # embed tasks can retrieve them.  The dependency edge guarantees ordering
    # (insert completes before embed starts), and the lock inside the store
    # ensures visibility across threads without relying on CPython GIL details.
    ids_store = _InsertedIdsStore()

    # -- root: create_table -------------------------------------------------
    create_id = graph.add_task(
        Task(
            id=f"create_table_{nonce}",
            task_type="create_table",
            func=_make_create_table_func(
                context,
                description=description,
                fields=fields,
                unique_keys=unique_keys,
                auto_counting=auto_counting,
                infer_untyped_fields=infer_untyped_fields,
            ),
            metadata={"context": context},
        ),
    )

    # -- insert chunks ------------------------------------------------------
    # Inserts are chained (each depends on the previous) so the backend's
    # auto_counting counter advances atomically between batches.  Embed
    # chunks still pipeline: embed_N starts as soon as insert_N finishes,
    # overlapping with insert_N+1.
    insert_ids: List[str] = []
    pad = len(str(total_chunks - 1)) if total_chunks > 0 else 1
    prev_insert_id = create_id
    for idx, chunk in enumerate(chunks):
        task_id = f"insert_chunk_{idx:0{pad}d}_{nonce}"
        graph.add_task(
            Task(
                id=task_id,
                task_type="insert_chunk",
                func=_make_insert_chunk_func(
                    context,
                    chunk,
                    add_to_all_context=add_to_all_context,
                    ids_store=ids_store,
                    task_id=task_id,
                ),
                dependencies={prev_insert_id},
                metadata={
                    "context": context,
                    "chunk_index": idx,
                    "chunk_size": len(chunk),
                    "total_chunks": total_chunks,
                    "total_rows": total_rows,
                },
            ),
        )
        insert_ids.append(task_id)
        prev_insert_id = task_id

    # -- optional embedding -------------------------------------------------
    if embed_columns:
        if embed_strategy == "along":
            for idx, ins_id in enumerate(insert_ids):
                embed_task_id = f"embed_chunk_{idx:0{pad}d}_{nonce}"
                # First chunk uses sync embeddings to guarantee the
                # FieldType (field_category="derived_entry") is created
                # via Orchestra's sync path.  The async path queues
                # embeddings but returns NULLs which can prevent
                # FieldType creation.  Subsequent chunks use async.
                use_async = idx > 0
                graph.add_task(
                    Task(
                        id=embed_task_id,
                        task_type="embed_chunk",
                        func=_make_embed_func(
                            context,
                            embed_columns,
                            async_embeddings=use_async,
                            ids_store=ids_store,
                            insert_task_id=ins_id,
                        ),
                        dependencies={ins_id},
                        metadata={
                            "context": context,
                            "chunk_index": idx,
                            "total_chunks": total_chunks,
                            "embed_columns": embed_columns,
                            "depends_on_ingest": ins_id,
                        },
                    ),
                )
        elif embed_strategy == "after":
            graph.add_task(
                Task(
                    id=f"embed_all_{nonce}",
                    task_type="embed_all",
                    func=_make_embed_func(
                        context,
                        embed_columns,
                        ids_store=ids_store,
                    ),
                    dependencies=set(insert_ids),
                    metadata={
                        "context": context,
                        "strategy": "after",
                        "embed_columns": embed_columns,
                        "total_rows": total_rows,
                    },
                ),
            )

    return graph


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _aggregate_results(
    context: str,
    graph: TaskGraph,
    results: Dict[str, TaskResult],
    duration_ms: float,
) -> IngestResult:
    """Collect per-task outcomes into a single :class:`IngestResult`."""
    total_inserted = 0
    total_embedded = 0
    all_log_ids: List[int] = []
    chunks_processed = 0

    for task_id, result in results.items():
        if not result.success or result.value is None:
            continue
        val = result.value
        if isinstance(val, dict):
            if "inserted_ids" in val:
                all_log_ids.extend(val["inserted_ids"])
                total_inserted += val.get("row_count", len(val["inserted_ids"]))
                chunks_processed += 1
            if "rows_embedded" in val:
                total_embedded += val["rows_embedded"]

    return IngestResult(
        context=context,
        rows_inserted=total_inserted,
        rows_embedded=total_embedded,
        log_ids=all_log_ids,
        duration_ms=duration_ms,
        chunks_processed=chunks_processed,
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_ingest(
    dm: "DataManager",
    context: str,
    rows: List[Dict[str, Any]],
    *,
    description: Optional[str] = None,
    fields: Optional[Dict[str, str]] = None,
    unique_keys: Optional[Dict[str, str]] = None,
    embed_columns: Optional[List[str]] = None,
    embed_strategy: str = "along",
    chunk_size: int = 1000,
    auto_counting: Optional[Dict[str, Optional[str]]] = None,
    infer_untyped_fields: bool = False,
    add_to_all_context: bool = False,
    execution: Optional[IngestExecutionConfig] = None,
    on_task_complete=None,
) -> IngestResult:
    """Execute a full ingest pipeline: create table, insert rows, optionally embed.

    This is the implementation behind :meth:`DataManager.ingest`.  It builds
    a :class:`TaskGraph`, runs it through :class:`PipelineExecutor`, and
    returns an aggregated :class:`IngestResult`.

    Parameters
    ----------
    dm : DataManager
        The calling DataManager instance (used for context resolution which
        has already been performed by the caller -- kept here for potential
        future needs).
    context : str
        **Already-resolved** Unify context path.
    rows : list[dict]
        Row data to insert.
    description, fields, unique_keys, embed_columns,
    embed_strategy, chunk_size, auto_counting, infer_untyped_fields :
        See :meth:`DataManager.ingest` for semantics.
    execution : IngestExecutionConfig | None
        Pipeline execution settings.

    Returns
    -------
    IngestResult
        Aggregated outcome of the operation.
    """
    if not rows:
        return IngestResult(context=context)

    exec_cfg = execution or IngestExecutionConfig()

    # Translate IngestExecutionConfig -> pipeline ExecutionConfig
    pipeline_cfg = ExecutionConfig(
        max_workers=exec_cfg.max_workers,
        max_retries=exec_cfg.max_retries,
        retry_delay_seconds=exec_cfg.retry_delay_seconds,
        fail_fast=exec_cfg.fail_fast,
    )

    chunks = _chunk_rows(rows, chunk_size)
    logger.info(
        "Starting ingest into %s: %d rows in %d chunks (chunk_size=%d, embed=%s, strategy=%s)",
        context,
        len(rows),
        len(chunks),
        chunk_size,
        embed_columns,
        embed_strategy,
    )

    graph = _build_ingest_graph(
        context,
        chunks,
        description=description,
        fields=fields,
        unique_keys=unique_keys,
        embed_columns=embed_columns,
        embed_strategy=embed_strategy,
        auto_counting=auto_counting,
        infer_untyped_fields=infer_untyped_fields,
        add_to_all_context=add_to_all_context,
        total_rows=len(rows),
    )

    executor = PipelineExecutor(config=pipeline_cfg, on_task_complete=on_task_complete)
    start = time.perf_counter()
    results = executor.execute(graph)
    duration_ms = (time.perf_counter() - start) * 1000

    ingest_result = _aggregate_results(context, graph, results, duration_ms)

    # Log summary
    summary = graph.get_summary()
    if summary["success"]:
        logger.info(
            "Ingest complete for %s: %d rows inserted, %d embedded in %.0fms (%d chunks)",
            context,
            ingest_result.rows_inserted,
            ingest_result.rows_embedded,
            duration_ms,
            ingest_result.chunks_processed,
        )
    else:
        failed = [tid for tid, r in results.items() if not r.success]
        logger.error(
            "Ingest had failures for %s: %d tasks failed (%s). "
            "Inserted %d rows before failure.",
            context,
            len(failed),
            ", ".join(failed[:5]),
            ingest_result.rows_inserted,
        )

    return ingest_result
