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
from typing import Any, Dict, List, Literal, Optional, TYPE_CHECKING

import unify as _unify

from unity.common.embed_utils import (
    ensure_derived_column as _ensure_derived_column,
    ensure_vector_column as _ensure_vector_column,
)
from unity.common.type_utils import types_match as _types_match
from unity.data_manager.ops.mutation_ops import insert_rows_impl
from unity.data_manager.ops.table_ops import create_table_impl
from unity.data_manager.types.ingest import (
    IngestExecutionConfig,
    IngestResult,
    PostIngestConfig,
)
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


def _chunk_ids(ids: List[int], batch_size: int) -> List[List[int]]:
    """Split log IDs into embedding-sized batches."""
    return [ids[i : i + batch_size] for i in range(0, len(ids), batch_size)]


def _should_serialize_insert_chunks(
    *,
    auto_counting: Optional[Dict[str, Optional[str]]],
    insert_parallelism: Literal["auto", "serial", "parallel"],
) -> bool:
    """Decide whether insert chunks must run sequentially."""
    if insert_parallelism == "serial":
        return True
    if insert_parallelism == "parallel":
        return False
    return bool(auto_counting)


def _derive_target_name(field_name: str, suffix: str) -> str:
    """Build a target column name by appending *suffix* to *field_name*.

    The separator and suffix casing are chosen to match the naming
    convention of the source field:

    * **Whitespace-separated** (``"Subtrip travel time"`` + ``"Date"``)
      → ``"Subtrip travel time Date"`` (space join, suffix as-is)
    * **Underscore-separated** (``"subtrip_travel_time"`` + ``"Date"``)
      → ``"subtrip_travel_time_date"`` (underscore join, suffix
      lowercased)
    * **PascalCase / single word starting uppercase**
      (``"VisitDate"`` + ``"Date"``) → ``"VisitDate_Date"``
      (underscore join, suffix words title-cased)
    * **camelCase / lowercase single word**
      (``"visitDate"`` + ``"Date"``) → ``"visitDate_date"``
      (underscore join, suffix lowercased)
    """
    if " " in field_name:
        return f"{field_name} {suffix}"
    if "_" in field_name:
        return f"{field_name}_{suffix.replace(' ', '_').lower()}"
    if field_name[:1].isupper():
        titled = "".join(w.capitalize() for w in suffix.split())
        return f"{field_name}_{titled}"
    return f"{field_name}_{suffix.replace(' ', '_').lower()}"


def _run_post_ingest_rules(
    context: str,
    config: PostIngestConfig,
) -> list[str]:
    """Execute post-ingest derived column rules for *context*.

    Each rule resolves an ``equation`` (containing a ``{field}``
    placeholder) and creates derived columns via :func:`_ensure_derived_column`.

    Rule dispatch is based on the ``kind`` discriminator:

    * ``"explicit"`` (:class:`ExplicitDerivedColumn`): one derived column
      from a named ``source_field`` / ``target_name`` pair.
    * ``"auto"`` (:class:`AutoDerivedColumn`): scans all fields in the
      context matching ``source_type`` and creates a derived column for each.
      The target name is derived from the source field name and
      ``target_suffix`` using :func:`_derive_target_name`, which
      automatically matches the separator convention of the source field.

    Returns the list of derived column names that were created or
    already existed.
    """
    from unity.data_manager.types.ingest import AutoDerivedColumn, ExplicitDerivedColumn

    if not config.derived_columns:
        return []

    fields: dict | None = None
    created: list[str] = []

    for rule in config.derived_columns:
        if isinstance(rule, ExplicitDerivedColumn):
            equation = rule.equation.replace("{field}", rule.source_field)
            _ensure_derived_column(
                context,
                key=rule.target_name,
                equation=equation,
                referenced_logs_context=context,
            )
            created.append(rule.target_name)

        elif isinstance(rule, AutoDerivedColumn):
            if fields is None:
                fields = _unify.get_fields(context=context) or {}

            for name, info in fields.items():
                if name.startswith("_"):
                    continue
                dtype = (
                    info.get("data_type", "") if isinstance(info, dict) else str(info)
                )
                if not _types_match(rule.source_type, dtype):
                    continue
                target = _derive_target_name(name, rule.target_suffix)
                equation = rule.equation.replace("{field}", name)
                _ensure_derived_column(
                    context,
                    key=target,
                    equation=equation,
                    referenced_logs_context=context,
                )
                created.append(target)

    return created


def _make_create_table_func(
    context: str,
    *,
    description: Optional[str],
    fields: Optional[Dict[str, Any]],
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
    embedding_batch_size: int = 1000,
    sync_first_batch: bool = False,
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

        id_batches = _chunk_ids(from_ids, embedding_batch_size)

        def _do_col(col: str) -> int:
            target = f"_{col}_emb"
            embedded = 0
            for batch_index, id_batch in enumerate(id_batches):
                batch_async = async_embeddings
                if sync_first_batch and batch_index == 0:
                    batch_async = False
                _ensure_vector_column(
                    context=context,
                    embed_column=target,
                    source_column=col,
                    derived_expr=None,
                    from_ids=id_batch,
                    async_embeddings=batch_async,
                )
                embedded += len(id_batch)
            return embedded

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
    fields: Optional[Dict[str, Any]],
    unique_keys: Optional[Dict[str, str]],
    embed_columns: Optional[List[str]],
    embed_strategy: str,
    auto_counting: Optional[Dict[str, Optional[str]]],
    infer_untyped_fields: bool,
    add_to_all_context: bool = False,
    total_rows: int = 0,
    insert_parallelism: Literal["auto", "serial", "parallel"] = "auto",
    embedding_batch_size: int = 1000,
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
    # Insert ordering is execution-configurable:
    # - serial: every chunk depends on the previous chunk
    # - parallel: every chunk fans out from create_table
    # - auto: serial only when auto_counting is configured
    insert_ids: List[str] = []
    pad = len(str(total_chunks - 1)) if total_chunks > 0 else 1
    serialize_inserts = _should_serialize_insert_chunks(
        auto_counting=auto_counting,
        insert_parallelism=insert_parallelism,
    )
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
                dependencies={prev_insert_id} if serialize_inserts else {create_id},
                metadata={
                    "context": context,
                    "chunk_index": idx,
                    "chunk_size": len(chunk),
                    "total_chunks": total_chunks,
                    "total_rows": total_rows,
                    "insert_parallelism": (
                        "serial" if serialize_inserts else "parallel"
                    ),
                },
            ),
        )
        insert_ids.append(task_id)
        if serialize_inserts:
            prev_insert_id = task_id

    # -- optional embedding -------------------------------------------------
    if embed_columns:
        if embed_strategy == "along":
            for idx, ins_id in enumerate(insert_ids):
                embed_task_id = f"embed_chunk_{idx:0{pad}d}_{nonce}"
                graph.add_task(
                    Task(
                        id=embed_task_id,
                        task_type="embed_chunk",
                        func=_make_embed_func(
                            context,
                            embed_columns,
                            async_embeddings=True,
                            ids_store=ids_store,
                            insert_task_id=ins_id,
                            embedding_batch_size=embedding_batch_size,
                            sync_first_batch=(idx == 0),
                        ),
                        dependencies={ins_id},
                        metadata={
                            "context": context,
                            "chunk_index": idx,
                            "total_chunks": total_chunks,
                            "embed_columns": embed_columns,
                            "depends_on_ingest": ins_id,
                            "embedding_batch_size": embedding_batch_size,
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
                        async_embeddings=True,
                        ids_store=ids_store,
                        embedding_batch_size=embedding_batch_size,
                        sync_first_batch=True,
                    ),
                    dependencies=set(insert_ids),
                    metadata={
                        "context": context,
                        "strategy": "after",
                        "embed_columns": embed_columns,
                        "total_rows": total_rows,
                        "embedding_batch_size": embedding_batch_size,
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
    rows: Optional[List[Dict[str, Any]]] = None,
    *,
    table_input_handle: Optional[Any] = None,
    description: Optional[str] = None,
    fields: Optional[Dict[str, Any]] = None,
    unique_keys: Optional[Dict[str, str]] = None,
    embed_columns: Optional[List[str]] = None,
    embed_strategy: str = "along",
    chunk_size: int = 1000,
    auto_counting: Optional[Dict[str, Optional[str]]] = None,
    infer_untyped_fields: bool = False,
    add_to_all_context: bool = False,
    execution: Optional[IngestExecutionConfig] = None,
    post_ingest: Optional[PostIngestConfig] = None,
    on_task_complete=None,
    coerce_types: bool = True,
    storage_client=None,
    skip_rows: int = 0,
) -> IngestResult:
    """Execute a full ingest pipeline: create table, insert rows, optionally embed.

    This is the implementation behind :meth:`DataManager.ingest`.  It builds
    a :class:`TaskGraph`, runs it through :class:`PipelineExecutor`, and
    returns an aggregated :class:`IngestResult`.

    Accepts **either** a materialised row list (``rows``) **or** a typed
    streaming handle (``table_input_handle``).  When a handle is provided
    the rows are streamed from source, prescanned once for type inference,
    and ingested in bounded-memory chunks.

    Parameters
    ----------
    dm : DataManager
        The calling DataManager instance.
    context : str
        **Already-resolved** Unify context path.
    rows : list[dict] | None
        Materialised row data.  Mutually exclusive with *table_input_handle*.
    table_input_handle : TableInputHandle | None
        Typed streaming handle.  Mutually exclusive with *rows*.
    description, fields, unique_keys, embed_columns,
    embed_strategy, chunk_size, auto_counting, infer_untyped_fields :
        See :meth:`DataManager.ingest` for semantics.
    execution : IngestExecutionConfig | None
        Pipeline execution settings.
    post_ingest : PostIngestConfig | None
        Post-ingest derived column rules.

    Returns
    -------
    IngestResult
        Aggregated outcome of the operation.
    """
    if table_input_handle is not None and rows is not None:
        raise ValueError("Provide exactly one of rows or table_input_handle, not both")

    if table_input_handle is not None:
        return _run_ingest_streaming(
            dm,
            context,
            table_input_handle,
            description=description,
            fields=fields,
            unique_keys=unique_keys,
            embed_columns=embed_columns,
            embed_strategy=embed_strategy,
            chunk_size=chunk_size,
            auto_counting=auto_counting,
            infer_untyped_fields=infer_untyped_fields,
            add_to_all_context=add_to_all_context,
            execution=execution,
            post_ingest=post_ingest,
            on_task_complete=on_task_complete,
            coerce_types=coerce_types,
            storage_client=storage_client,
            skip_rows=skip_rows,
        )

    return _run_ingest_materialised(
        dm,
        context,
        rows or [],
        description=description,
        fields=fields,
        unique_keys=unique_keys,
        embed_columns=embed_columns,
        embed_strategy=embed_strategy,
        chunk_size=chunk_size,
        auto_counting=auto_counting,
        infer_untyped_fields=infer_untyped_fields,
        add_to_all_context=add_to_all_context,
        execution=execution,
        post_ingest=post_ingest,
        on_task_complete=on_task_complete,
        coerce_types=coerce_types,
    )


# ---------------------------------------------------------------------------
# Materialised path (original implementation)
# ---------------------------------------------------------------------------


def _run_ingest_materialised(
    dm: "DataManager",
    context: str,
    rows: List[Dict[str, Any]],
    *,
    description: Optional[str] = None,
    fields: Optional[Dict[str, Any]] = None,
    unique_keys: Optional[Dict[str, str]] = None,
    embed_columns: Optional[List[str]] = None,
    embed_strategy: str = "along",
    chunk_size: int = 1000,
    auto_counting: Optional[Dict[str, Optional[str]]] = None,
    infer_untyped_fields: bool = False,
    add_to_all_context: bool = False,
    execution: Optional[IngestExecutionConfig] = None,
    post_ingest: Optional[PostIngestConfig] = None,
    on_task_complete=None,
    coerce_types: bool = True,
) -> IngestResult:
    """Ingest from a fully materialised row list (original code path)."""
    if not rows:
        return IngestResult(context=context)

    from unity.data_manager.ops.type_prescan import (
        coerce_empty_strings,
        coerce_rows,
        prescan_column_types,
    )

    coercion_stats = None
    if coerce_types:
        column_types = prescan_column_types(rows)
        rows, coercion_stats = coerce_rows(rows, column_types)
        logger.info(
            "coerce_types prescan for %s: %d columns typed, %d empty strings coerced, "
            "%d type mismatches coerced (%d total cells)",
            context,
            len(column_types),
            coercion_stats.empty_strings_coerced,
            coercion_stats.type_coerced,
            coercion_stats.total_cells,
        )

        fields = dict(fields or {})
        for col, col_type in column_types.items():
            existing = fields.get(col)
            if existing is None:
                fields[col] = col_type
            elif isinstance(existing, dict) and "type" not in existing:
                existing["type"] = col_type

        explicit_types = {
            col: {"type": col_type} for col, col_type in column_types.items()
        }
        for row in rows:
            row["explicit_types"] = explicit_types
    else:
        rows, empty_count = coerce_empty_strings(rows)
        if empty_count:
            logger.info(
                "Coerced %d empty strings to None for %s (coerce_types=False)",
                empty_count,
                context,
            )

    exec_cfg = execution or IngestExecutionConfig()
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
        insert_parallelism=exec_cfg.insert_parallelism,
        embedding_batch_size=exec_cfg.embedding_batch_size,
    )

    executor = PipelineExecutor(config=pipeline_cfg, on_task_complete=on_task_complete)
    start = time.perf_counter()
    results = executor.execute(graph)
    duration_ms = (time.perf_counter() - start) * 1000

    ingest_result = _aggregate_results(context, graph, results, duration_ms)

    if coercion_stats is not None:
        from dataclasses import asdict

        ingest_result.coercion_stats = asdict(coercion_stats)

    _run_post_ingest_if_needed(context, post_ingest, ingest_result)
    _log_ingest_summary(context, graph, results, ingest_result, duration_ms)

    return ingest_result


# ---------------------------------------------------------------------------
# Streaming path (table_input_handle)
# ---------------------------------------------------------------------------


def _run_ingest_streaming(
    dm: "DataManager",
    context: str,
    handle: Any,
    *,
    description: Optional[str] = None,
    fields: Optional[Dict[str, Any]] = None,
    unique_keys: Optional[Dict[str, str]] = None,
    embed_columns: Optional[List[str]] = None,
    embed_strategy: str = "along",
    chunk_size: int = 1000,
    auto_counting: Optional[Dict[str, Optional[str]]] = None,
    infer_untyped_fields: bool = False,
    add_to_all_context: bool = False,
    execution: Optional[IngestExecutionConfig] = None,
    post_ingest: Optional[PostIngestConfig] = None,
    on_task_complete=None,
    coerce_types: bool = True,
    storage_client=None,
    skip_rows: int = 0,
) -> IngestResult:
    """Ingest from a typed streaming handle in bounded-memory chunks.

    Phase 1 -- drain up to ``_PRESCAN_SAMPLE_SIZE`` rows from the handle
    and build a :class:`TypeMap` for consistent type coercion.

    Phase 2 -- chain the sample rows with the remaining stream, iterate
    in chunks of ``chunk_size``, coerce each chunk against the type map,
    build a per-chunk task graph, execute, and accumulate results.
    """
    import itertools

    from unity.data_manager.ops.type_prescan import (
        TypeMap,
        coerce_batch,
        coerce_empty_strings,
        prescan_from_rows,
    )
    from unity.common.pipeline.row_streaming import iter_table_input_rows

    row_iter = iter_table_input_rows(
        handle,
        storage_client=storage_client,
        skip_rows=skip_rows,
    )

    # Phase 1: drain sample for type inference
    type_map: Optional[TypeMap] = None
    sample: List[Dict[str, Any]] = list(
        itertools.islice(row_iter, _PRESCAN_SAMPLE_SIZE),
    )

    if not sample:
        return IngestResult(context=context)

    if coerce_types:
        type_map = prescan_from_rows(sample)
        logger.info(
            "Streaming prescan for %s: %d columns typed from %d-row sample",
            context,
            len(type_map.column_types),
            type_map.sample_size,
        )
        fields = _merge_prescan_fields(fields, type_map.column_types)

    # Phase 2: stream batches -- sample rows first, then remainder
    all_rows = itertools.chain(sample, row_iter)
    exec_cfg = execution or IngestExecutionConfig()
    pipeline_cfg = ExecutionConfig(
        max_workers=exec_cfg.max_workers,
        max_retries=exec_cfg.max_retries,
        retry_delay_seconds=exec_cfg.retry_delay_seconds,
        fail_fast=exec_cfg.fail_fast,
    )

    overall_start = time.perf_counter()
    result = IngestResult(context=context)
    chunk_idx = 0
    total_rows_seen = 0
    aggregate_coercion: Optional[Dict[str, Any]] = None

    for batch in _iter_chunks(all_rows, chunk_size):
        if coerce_types and type_map is not None:
            batch, batch_stats = coerce_batch(batch, type_map)
            _inject_explicit_types(batch, type_map.column_types)
            if aggregate_coercion is None:
                from dataclasses import asdict

                aggregate_coercion = asdict(batch_stats)
            else:
                aggregate_coercion["total_cells"] += batch_stats.total_cells
                aggregate_coercion[
                    "empty_strings_coerced"
                ] += batch_stats.empty_strings_coerced
                aggregate_coercion["type_coerced"] += batch_stats.type_coerced
        elif not coerce_types:
            batch, _ = coerce_empty_strings(batch)

        total_rows_seen += len(batch)
        chunks = [batch]

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
            total_rows=total_rows_seen,
            insert_parallelism=exec_cfg.insert_parallelism,
            embedding_batch_size=exec_cfg.embedding_batch_size,
        )

        executor = PipelineExecutor(
            config=pipeline_cfg,
            on_task_complete=on_task_complete,
        )
        chunk_results = executor.execute(graph)
        chunk_result = _aggregate_results(context, graph, chunk_results, 0.0)

        result.rows_inserted += chunk_result.rows_inserted
        result.rows_embedded += chunk_result.rows_embedded
        result.log_ids.extend(chunk_result.log_ids)
        result.chunks_processed += chunk_result.chunks_processed
        result.derived_columns_created.extend(chunk_result.derived_columns_created)
        chunk_idx += 1

        # Only pass description/fields for the first chunk (table creation is idempotent)
        description = None

    duration_ms = (time.perf_counter() - overall_start) * 1000
    result.duration_ms = duration_ms

    if aggregate_coercion is not None:
        result.coercion_stats = aggregate_coercion

    logger.info(
        "Streaming ingest into %s: %d rows in %d chunks (%.0fms, embed=%s)",
        context,
        total_rows_seen,
        chunk_idx,
        duration_ms,
        embed_columns,
    )

    _run_post_ingest_if_needed(context, post_ingest, result)

    return result


_PRESCAN_SAMPLE_SIZE = 500


def _merge_prescan_fields(
    fields: Optional[Dict[str, Any]],
    column_types: Dict[str, str],
) -> Dict[str, Any]:
    """Merge prescan type inference into the fields dict for create_table."""
    merged = dict(fields or {})
    for col, col_type in column_types.items():
        existing = merged.get(col)
        if existing is None:
            merged[col] = col_type
        elif isinstance(existing, dict) and "type" not in existing:
            existing["type"] = col_type
    return merged


def _inject_explicit_types(
    batch: List[Dict[str, Any]],
    column_types: Dict[str, str],
) -> None:
    """Inject explicit_types into every row so Orchestra bypasses inference."""
    explicit_types = {col: {"type": col_type} for col, col_type in column_types.items()}
    for row in batch:
        row["explicit_types"] = explicit_types


def _iter_chunks(
    iterable: Any,
    chunk_size: int,
) -> Any:
    """Yield bounded lists from an arbitrary iterable."""
    import itertools

    it = iter(iterable)
    while True:
        batch = list(itertools.islice(it, chunk_size))
        if not batch:
            return
        yield batch


# ---------------------------------------------------------------------------
# Shared post-ingest and logging helpers
# ---------------------------------------------------------------------------


def _run_post_ingest_if_needed(
    context: str,
    post_ingest: Optional[PostIngestConfig],
    ingest_result: IngestResult,
) -> None:
    """Run post-ingest derived column rules if configured."""
    if not post_ingest or not post_ingest.derived_columns:
        return
    try:
        derived_cols = _run_post_ingest_rules(context, post_ingest)
        if derived_cols:
            ingest_result.derived_columns_created = derived_cols
            logger.info(
                "Created %d derived columns for %s: %s",
                len(derived_cols),
                context,
                ", ".join(derived_cols),
            )
    except Exception:
        logger.warning(
            "Failed to create post-ingest derived columns for %s",
            context,
            exc_info=True,
        )


def _log_ingest_summary(
    context: str,
    graph: TaskGraph,
    results: Dict[str, TaskResult],
    ingest_result: IngestResult,
    duration_ms: float,
) -> None:
    """Log a summary of the ingest operation."""
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
