"""
Storage and local view utilities for the Task Scheduler.

- TasksStore: centralizes Unify I/O for the Tasks context (contexts, fields,
  reads/writes, metrics, checkpoints).
- LocalTaskView: best‑effort cache for queue membership, head start_at,
  queue‑id allocation, and log‑id memoization with convenience wrappers.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Union
from enum import Enum
from functools import cached_property
import os

import unify


class TasksStore:
    """
    Adapter around Unify I/O for the Tasks context.

    Centralises reads, writes, field management, metrics, and checkpoint
    helpers used by the scheduler and related utilities.
    """

    def __init__(self, context: str) -> None:
        self._ctx = context

    # ----------------------------- Context ---------------------------------
    def ensure_context(
        self,
        *,
        unique_keys: Dict[str, str],
        auto_counting: Dict[str, Optional[str]],
        description: str,
        fields: Dict[str, str],
    ) -> None:
        """
        Ensure the Tasks context exists with the requested fields.

        If the context already exists, any missing fields are created.
        """
        if self._ctx not in unify.get_contexts():
            unify.create_context(
                self._ctx,
                unique_keys=unique_keys,
                auto_counting=auto_counting,
                description=description,
            )
            unify.create_fields(fields, context=self._ctx)
            # Prime the local fields cache to avoid an immediate second get_fields call.
            try:
                primed = {
                    k: (v.get("data_type") if isinstance(v, dict) else str(v))
                    for k, v in (fields or {}).items()
                }
            except Exception:
                primed = {k: str(v) for k, v in (fields or {}).items()}
            # cached_property can be pre-populated by writing into __dict__
            self.__dict__["fields"] = primed
            return

        try:
            existing = unify.get_fields(context=self._ctx) or {}
        except Exception:
            existing = {}
        missing = {k: v for k, v in fields.items() if k not in existing}
        if missing:
            unify.create_fields(missing, context=self._ctx)
        # Prime/refresh the local fields cache with a normalised view combining existing + newly created
        try:
            normalised = {
                k: (v.get("data_type") if isinstance(v, dict) else str(v))
                for k, v in existing.items()
            }
        except Exception:
            normalised = {k: str(v) for k, v in existing.items()}
        for k, v in (missing or {}).items():
            if k not in normalised:
                try:
                    normalised[k] = (
                        v.get("data_type") if isinstance(v, dict) else str(v)
                    )
                except Exception:
                    normalised[k] = str(v)
        self.__dict__["fields"] = normalised

    # ------------------------------- Reads ---------------------------------
    @cached_property
    def fields(self) -> Dict[str, str]:
        try:
            fields = unify.get_fields(context=self._ctx) or {}
            return {
                k: (v.get("data_type") if isinstance(v, dict) else str(v))
                for k, v in fields.items()
            }
        except Exception:
            return {}

    def get_metric_count(self, *, key: str) -> int:
        ret = unify.get_logs_metric(metric="count", key=key, context=self._ctx)
        return 0 if ret is None else int(ret)

    def get_metric_max(self, *, key: str) -> int:
        """
        Return the maximum value observed for a numeric field within this context.

        When the backend does not return a value (e.g., empty context), 0 is returned.
        """
        ret = unify.get_logs_metric(metric="max", key=key, context=self._ctx)
        return 0 if ret is None else int(ret)

    def get_rows(
        self,
        *,
        filter: Optional[str] = None,
        offset: int = 0,
        limit: int = 100,
        return_ids_only: bool = False,
        exclude_fields: Optional[List[str]] = None,
    ) -> List[Union[int, unify.Log]]:
        return unify.get_logs(
            context=self._ctx,
            filter=filter,
            offset=offset,
            limit=limit,
            return_ids_only=return_ids_only,
            exclude_fields=exclude_fields or [],
        )

    def get_entries(
        self,
        *,
        filter: Optional[str] = None,
        offset: int = 0,
        limit: int = 100,
        exclude_fields: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        return [
            log.entries
            for log in self.get_rows(
                filter=filter,
                offset=offset,
                limit=limit,
                return_ids_only=False,
                exclude_fields=exclude_fields,
            )
        ]

    def get_logs_by_task_ids(
        self,
        *,
        task_ids: Union[int, Iterable[int]],
        return_ids_only: bool = True,
    ) -> List[Union[int, unify.Log]]:
        singular = isinstance(task_ids, int)
        original_id = task_ids if singular else None
        ids_list = [task_ids] if singular else list(task_ids)
        logs = unify.get_logs(
            context=self._ctx,
            filter=f"task_id in {ids_list}",
            return_ids_only=return_ids_only,
        )
        if singular:
            if len(logs) == 0:
                raise ValueError(
                    f"Task with task_id == {original_id} does not exist in the task list.",
                )
            if len(logs) > 1:
                raise AssertionError(
                    f"Expected exactly 1 row for task_id {original_id}, but found {len(logs)}.",
                )
        return logs

    def get_minimal_rows_by_task_ids(
        self,
        *,
        task_ids: Union[int, Iterable[int]],
        fields: Optional[List[str]] = None,
    ) -> List[unify.Log]:
        """
        Fetch a minimal projection of rows for the specified task_ids.

        Only the requested fields are returned in each row's entries payload to
        reduce payload size and backend processing time. The returned objects
        still include their underlying log ids.
        """
        singular = isinstance(task_ids, int)
        original_id = task_ids if singular else None
        ids_list = [task_ids] if singular else list(task_ids)
        # Ensure we always include task_id in the projection for correctness
        fields = list(dict.fromkeys((fields or []) + ["task_id"]))
        logs = unify.get_logs(
            context=self._ctx,
            filter=f"task_id in {ids_list}",
            return_ids_only=False,
            from_fields=fields,
        )
        if singular:
            if len(logs) == 0:
                raise ValueError(
                    f"Task with task_id == {original_id} does not exist in the task list.",
                )
            if len(logs) > 1:
                raise AssertionError(
                    f"Expected exactly 1 row for task_id {original_id}, but found {len(logs)}.",
                )
        # Opportunistically memoize task_id -> log_id mappings
        try:
            for lg in logs or []:
                try:
                    e = getattr(lg, "entries", {}) or {}
                    tid = e.get("task_id")
                    lid = getattr(lg, "id", None)
                    if isinstance(tid, int) and isinstance(lid, int):
                        # This method is on TasksStore; LocalTaskView handles memoization
                        pass
                except Exception:
                    continue
        except Exception:
            pass
        return logs

    # ------------------------------- Writes --------------------------------
    def update(
        self,
        *,
        logs: Union[int, unify.Log, List[Union[int, unify.Log]]],
        entries: Union[Dict[str, Any], List[Dict[str, Any]]],
        overwrite: bool = True,
    ) -> Dict[str, str]:
        def _norm(v: Any) -> Any:
            # Normalize enums to their underlying values
            if isinstance(v, Enum):
                try:
                    from enum import StrEnum  # py311+

                    if isinstance(v, StrEnum):  # type: ignore[arg-type]
                        return v.value
                except Exception:
                    pass
                return v.value
            # Datetime family → ISO-8601 strings
            try:
                import datetime as _dt

                if isinstance(v, (_dt.datetime, _dt.date, _dt.time)):
                    return v.isoformat()
            except Exception:
                pass
            # Pydantic models → plain dict (JSON mode for consistent strings)
            try:
                from pydantic import BaseModel as _BM  # type: ignore

                if isinstance(v, _BM):
                    return _norm(v.model_dump(mode="json"))
            except Exception:
                pass
            if isinstance(v, dict):
                return {k: _norm(x) for k, x in v.items()}
            if isinstance(v, list):
                return [_norm(x) for x in v]
            return v

        def _strip_nones(value: Any, *, top_level: bool) -> Any:
            """
            Remove None values from nested structures so we don't accidentally
            clear existing fields when performing partial updates.

            Policy:
            - At the top level we KEEP explicit None values (e.g., schedule=None) so
              callers can intentionally clear a whole field.
            - For nested dicts/lists we DROP None entries/values entirely.
            """
            if isinstance(value, dict):
                out: Dict[str, Any] = {}
                for k, v in value.items():
                    if v is None:
                        if top_level:
                            out[k] = None
                        else:
                            # omit nested None
                            continue
                    else:
                        out[k] = _strip_nones(v, top_level=False)
                return out
            if isinstance(value, list):
                return [
                    _strip_nones(v, top_level=False) for v in value if v is not None
                ]
            return value

        norm_entries = _strip_nones(_norm(entries), top_level=True)
        return unify.update_logs(
            logs=logs,
            context=self._ctx,
            entries=norm_entries,
            overwrite=True,
        )

    def log(self, *, entries: Dict[str, Any], new: bool = True) -> unify.Log:
        def _norm(v: Any) -> Any:
            # Normalize enums to their underlying values
            if isinstance(v, Enum):
                try:
                    from enum import StrEnum  # py311+

                    if isinstance(v, StrEnum):  # type: ignore[arg-type]
                        return v.value
                except Exception:
                    pass
                return v.value
            # Datetime family → ISO-8601 strings
            try:
                import datetime as _dt

                if isinstance(v, (_dt.datetime, _dt.date, _dt.time)):
                    return v.isoformat()
            except Exception:
                pass
            # Pydantic models → plain dict (JSON mode for consistent strings)
            try:
                from pydantic import BaseModel as _BM  # type: ignore

                if isinstance(v, _BM):
                    return _norm(v.model_dump(mode="json"))
            except Exception:
                pass
            if isinstance(v, dict):
                return {k: _norm(x) for k, x in v.items()}
            if isinstance(v, list):
                return [_norm(x) for x in v]
            return v

        norm_entries = _norm(entries)
        # Create with expanded fields so auto-counting applies when ids are omitted
        return unify.log(context=self._ctx, new=new, **norm_entries)

    def create_many(self, *, entries_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Batch-create multiple logs in a single backend call.

        Returns the raw response from the backend which typically includes
        'log_event_ids' (ids of created log events) and a 'row_ids' structure
        with auto-incremented row identifiers.
        """

        # Normalise all payloads consistently with the single-log path
        def _norm(v: Any) -> Any:
            if isinstance(v, Enum):
                try:
                    from enum import StrEnum  # py311+

                    if isinstance(v, StrEnum):  # type: ignore[arg-type]
                        return v.value
                except Exception:
                    pass
                return v.value
            try:
                import datetime as _dt

                if isinstance(v, (_dt.datetime, _dt.date, _dt.time)):
                    return v.isoformat()
            except Exception:
                pass
            try:
                from pydantic import BaseModel as _BM  # type: ignore

                if isinstance(v, _BM):
                    return _norm(v.model_dump(mode="json"))
            except Exception:
                pass
            if isinstance(v, dict):
                return {k: _norm(x) for k, x in v.items()}
            if isinstance(v, list):
                return [_norm(x) for x in v]
            return v

        normalised = [{**_norm(e)} for e in entries_list]
        try:
            return unify.create_logs(context=self._ctx, entries=normalised)
        except Exception:
            # Fallback: create sequentially (preserves correctness if batch API is unavailable)
            log_ids: list[int] = []
            for e in normalised:
                lg = unify.log(context=self._ctx, new=True, **e)
                try:
                    log_ids.append(int(getattr(lg, "id", None)))
                except Exception:
                    pass
            return {"log_event_ids": log_ids}

    def get_rows_by_log_ids(self, *, log_ids: List[int]) -> List[unify.Log]:
        """
        Fetch full log objects by their log-event ids. This avoids filtering by
        field values and allows precise retrieval of freshly-created rows.
        """
        res = unify.get_logs(
            context=self._ctx,
            from_ids=log_ids,
            return_ids_only=False,
        )
        # The client may return either a list or a dict with 'logs'
        try:
            if isinstance(res, dict):
                logs = res.get("logs") or []
                return logs
        except Exception:
            pass
        return res

    def delete(self, *, logs: Union[int, List[int]]) -> Dict[str, str]:
        return unify.delete_logs(context=self._ctx, logs=logs)

    # ------------------------- Checkpoint helpers -------------------------
    def _checkpoint_context(self) -> str:
        """
        Return the fully-qualified checkpoints context for this task context.

        Example: for "Tasks", returns "Tasks/Checkpoints".
        """
        return f"{self._ctx}/Checkpoints"

    def _ensure_checkpoint_context(self) -> str:
        """
        Ensure the checkpoints context exists with required fields and return its name.
        """
        ctx = self._checkpoint_context()
        try:
            if ctx not in unify.get_contexts():
                unify.create_context(ctx)
                unify.create_fields(
                    {
                        "checkpoint_id": "str",
                        "label": "str",
                        "payload": "json",
                    },
                    context=ctx,
                )
        except Exception:
            # Best-effort: if we cannot create context/fields, subsequent calls may fail
            pass
        return ctx

    def save_checkpoint(
        self,
        *,
        checkpoint_id: str,
        label: Optional[str],
        payload: Dict[str, Any],
    ) -> Optional[unify.Log]:
        """
        Create a checkpoint row in the checkpoints context.
        """
        ctx = self._ensure_checkpoint_context()
        try:
            return unify.log(
                context=ctx,
                new=True,
                checkpoint_id=checkpoint_id,
                label=label,
                payload=payload,
            )
        except Exception:
            return None

    def load_checkpoint(self, *, checkpoint_id: str) -> Optional[Dict[str, Any]]:
        """
        Load a checkpoint payload by its identifier. Returns None if missing.
        """
        ctx = self._checkpoint_context()
        try:
            if ctx not in unify.get_contexts():
                return None
            rows = unify.get_logs(
                context=ctx,
                filter=f"checkpoint_id == {checkpoint_id!r}",
                limit=1,
                return_ids_only=False,
            )
            if not rows:
                return None
            log = rows[0]
            entries = getattr(log, "entries", {}) or {}
            return entries.get("payload")
        except Exception:
            return None

    def get_latest_checkpoint(self) -> Optional[Dict[str, Any]]:
        """
        Return a small descriptor of the most recent checkpoint, or None.

        Ordering semantics are backend-defined.
        """
        ctx = self._checkpoint_context()
        try:
            if ctx not in unify.get_contexts():
                return None
            rows = unify.get_logs(context=ctx, offset=0, limit=1, return_ids_only=False)
            if not rows:
                return None
            log = rows[-1]
            entries = getattr(log, "entries", {}) or {}
            return {
                "checkpoint_id": entries.get("checkpoint_id"),
                "label": entries.get("label"),
            }
        except Exception:
            return None

    def delete_checkpoint(self, *, checkpoint_id: str) -> Dict[str, str]:
        """
        Best-effort deletion of a checkpoint row by identifier.
        """
        ctx = self._checkpoint_context()
        try:
            rows = unify.get_logs(
                context=ctx,
                filter=f"checkpoint_id == {checkpoint_id!r}",
                limit=100,
                return_ids_only=True,
            )
            if not rows:
                return {"detail": "No matching checkpoints"}
            return unify.delete_logs(context=ctx, logs=rows)
        except Exception:
            return {"detail": "Delete failed"}


class LocalTaskView:
    """
    Centralised, best‑effort local view for queue membership, id allocation and
    light caching around TasksStore I/O.

    Goals
    -----
    - Provide a single place for all read optimisation hooks (queue index,
      head start_at cache, reverse membership, log id memoisation).
    - Offer small wrappers for common read/write shapes used by TaskScheduler
      and queue ops to keep tools readable.
    - Never compromise correctness: when in doubt, fall back to TasksStore.

    Notes
    -----
    - All methods are tolerant of cache misses and inconsistent state. They
      either rebuild on demand or degrade to direct store calls.
    - This class intentionally does not enforce invariants; that remains the
      responsibility of higher‑level tools and validators.
    """

    def __init__(self, store: TasksStore) -> None:
        self._store = store
        # Queue membership caches
        self._queue_index: Dict[int, List[int]] = {}
        self._task_to_queue: Dict[int, int] = {}
        self._queue_head_start_at: Dict[int, Optional[str]] = {}
        self._queue_index_stale: bool = False

        # Monotonic allocator and fast id lookups
        self._max_queue_id_seen: Optional[int] = None
        self._task_log_id_cache: Dict[int, int] = {}

    # Expose task context fields via the view for convenience
    @property
    def fields(self) -> Dict[str, str]:
        return self._store.fields

    # ------------------------------- Reads --------------------------------
    def get_rows(
        self,
        *,
        filter: Optional[str] = None,
        offset: int = 0,
        limit: int = 100,
        return_ids_only: bool = False,
        exclude_fields: Optional[List[str]] = None,
    ) -> List[Union[int, unify.Log]]:
        """
        Pass-through to the underlying store for general row retrieval.

        Kept in LocalTaskView so that all read I/O can be centralised and
        optionally instrumented or toggled via environment flags.
        """
        return self._store.get_rows(
            filter=filter,
            offset=offset,
            limit=limit,
            return_ids_only=return_ids_only,
            exclude_fields=exclude_fields,
        )

    def get_entries(
        self,
        *,
        filter: Optional[str] = None,
        offset: int = 0,
        limit: int = 100,
        exclude_fields: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Pass-through to the underlying store for entry dictionaries.

        Exists here so that TaskScheduler and helpers route all generic reads
        through LocalTaskView for consistency and easier optimisation later.
        """
        return self._store.get_entries(
            filter=filter,
            offset=offset,
            limit=limit,
            exclude_fields=exclude_fields,
        )

    # ----------------------------- Queue index -----------------------------
    def queue_index_is_fresh(self) -> bool:
        return (not self._queue_index_stale) and (not self._cache_disabled())

    def mark_queue_changed(self) -> None:
        self._queue_index_stale = True

    def refresh_queue_index_from_rows(self, rows: List[Dict[str, Any]]) -> None:
        """
        Build queue caches from a list of row dicts containing at least:
        task_id, schedule (dict), status, queue_id.
        """
        try:
            # Filter to runnable rows with linkage
            runnable = [
                r
                for r in (rows or [])
                if r.get("schedule") is not None
                and str(r.get("status")) not in ("completed", "cancelled", "failed")
            ]

            rows_by_id: Dict[int, Dict[str, Any]] = {}
            for r in runnable:
                try:
                    tid = int(r.get("task_id"))
                except Exception:
                    continue
                rows_by_id[tid] = r

            # Identify heads by prev_task is None and numeric queue_id
            heads: List[Dict[str, Any]] = []
            for r in runnable:
                sched = r.get("schedule") or {}
                prev = sched.get("prev_task")
                qid = r.get("queue_id")
                if prev is None and isinstance(qid, int):
                    heads.append(r)

            new_index: Dict[int, List[int]] = {}
            new_reverse: Dict[int, int] = {}
            new_head_start: Dict[int, Optional[str]] = {}

            for h in heads:
                try:
                    qid = int(h.get("queue_id"))
                except Exception:
                    continue
                order: List[int] = []
                seen: set[int] = set()
                cur = h
                while cur is not None:
                    try:
                        tid_val = cur.get("task_id")
                        tid = int(tid_val) if tid_val is not None else None
                    except Exception:
                        tid = None
                    if tid is None or tid in seen:
                        break
                    seen.add(tid)
                    order.append(tid)
                    nxt = (cur.get("schedule") or {}).get("next_task")
                    if nxt is None:
                        break
                    try:
                        cur = rows_by_id.get(int(nxt))
                    except Exception:
                        break

                if order:
                    new_index[qid] = order
                    for t in order:
                        new_reverse[t] = qid
                    try:
                        new_head_start[qid] = (h.get("schedule") or {}).get("start_at")
                    except Exception:
                        new_head_start[qid] = None

            self._queue_index = new_index
            self._task_to_queue = new_reverse
            self._queue_head_start_at = new_head_start
            self._queue_index_stale = False
        except Exception:
            # On failure, mark as stale but keep previous view
            self._queue_index_stale = True

    def rebuild_queue_index(self) -> None:
        """Fetch minimal rows from storage and rebuild the queue index."""
        try:
            rows = self._store.get_entries(
                filter=(
                    "schedule is not None and "
                    "status not in ('completed','cancelled','failed')"
                ),
            )
        except Exception:
            rows = []
        self.refresh_queue_index_from_rows(rows)

    def get_member_ids(self, queue_id: int) -> List[int]:
        try:
            if self._cache_disabled() or self._queue_index_stale:
                self.rebuild_queue_index()
            return list(self._queue_index.get(int(queue_id)) or [])
        except Exception:
            return []

    def get_queue_id_for_task(self, task_id: int) -> Optional[int]:
        try:
            if self._cache_disabled() or self._queue_index_stale:
                self.rebuild_queue_index()
            qid = self._task_to_queue.get(int(task_id))
            return int(qid) if isinstance(qid, int) else None
        except Exception:
            return None

    def get_head_start_at(self, queue_id: int) -> Optional[str]:
        try:
            if self._cache_disabled() or self._queue_index_stale:
                self.rebuild_queue_index()
            return self._queue_head_start_at.get(int(queue_id))
        except Exception:
            return None

    def get_all_queue_summaries(self) -> List[Dict[str, Any]]:
        """
        Return cached summaries for all runnable queues.

        Each summary contains: {"queue_id": int, "order": list[int], "start_at": str | None}
        """
        try:
            if self._cache_disabled() or self._queue_index_stale:
                self.rebuild_queue_index()
            out: List[Dict[str, Any]] = []
            for qid, order in self._queue_index.items():
                if not order:
                    continue
                out.append(
                    {
                        "queue_id": int(qid),
                        "order": list(order),
                        "start_at": self._queue_head_start_at.get(int(qid)),
                    },
                )
            return out
        except Exception:
            return []

    def update_after_queue_materialized(
        self,
        *,
        queue_id: int,
        order: List[int],
        head_start_at: Optional[str],
    ) -> None:
        try:
            self._queue_index[int(queue_id)] = list(int(x) for x in order)
            for t in order:
                self._task_to_queue[int(t)] = int(queue_id)
            self._queue_head_start_at[int(queue_id)] = head_start_at
            self._queue_index_stale = False
        except Exception:
            # If anything goes wrong, mark cache as stale to force rebuild later
            self._queue_index_stale = True

    def update_after_reorder(
        self,
        *,
        queue_id: int,
        new_order: List[int],
        head_start_at: Optional[str],
    ) -> None:
        self.update_after_queue_materialized(
            queue_id=int(queue_id),
            order=list(new_order),
            head_start_at=head_start_at,
        )

    def on_tasks_removed_from_queue(
        self,
        *,
        queue_id: int,
        removed_ids: List[int],
    ) -> None:
        try:
            for t in removed_ids:
                self._task_to_queue.pop(int(t), None)
            # If we know the queue, also drop removed ids from the forward index
            if isinstance(queue_id, int):
                cur = list(self._queue_index.get(int(queue_id)) or [])
                if cur:
                    self._queue_index[int(queue_id)] = [
                        x for x in cur if x not in removed_ids
                    ]
        except Exception:
            pass

    # ------------------------ Queue id allocation -------------------------
    def allocate_new_queue_id(self) -> int:
        """
        Return the next candidate queue id without advancing internal state.

        The caller persists a queue with this id and may then call
        `sync_max_queue_id_seen`.
        """
        try:
            if self._cache_disabled():
                # Always consult backend metric when disabled
                return int(self._store.get_metric_max(key="queue_id")) + 1
            if self._max_queue_id_seen is None:
                try:
                    self._max_queue_id_seen = int(
                        self._store.get_metric_max(key="queue_id"),
                    )
                except Exception:
                    # Fallback to any locally indexed queues
                    if not self._queue_index_stale and self._queue_index:
                        self._max_queue_id_seen = max(
                            int(q) for q in self._queue_index.keys()
                        )
                    else:
                        # As a last resort, rebuild then compute
                        self.rebuild_queue_index()
                        self._max_queue_id_seen = (
                            max(int(q) for q in self._queue_index.keys())
                            if self._queue_index
                            else 0
                        )
            return int(self._max_queue_id_seen) + 1
        except Exception:
            return 1

    def sync_max_queue_id_seen(self, candidate: int) -> None:
        try:
            if self._cache_disabled():
                return
            if self._max_queue_id_seen is None or int(candidate) > int(
                self._max_queue_id_seen,
            ):
                self._max_queue_id_seen = int(candidate)
        except Exception:
            pass

    # -------------------------- Log id memoisation ------------------------
    def cache_log_id(self, *, task_id: int, log_id: int) -> None:
        try:
            self._task_log_id_cache[int(task_id)] = int(log_id)
        except Exception:
            pass

    def get_log_ids_by_task_ids(
        self,
        *,
        task_ids: Union[int, Iterable[int]],
        return_ids_only: bool = True,
    ) -> List[Union[int, unify.Log]]:
        """
        Resolve log objects/ids for task_ids, with a read-through memoization
        when callers request ids only.

        For id-only requests, this method will prefer cached mappings and
        fetch only the missing subset (as full rows to learn the mapping),
        returning a list of ints in the same order as the input ids.
        """
        singular = isinstance(task_ids, int)
        ids_list = [task_ids] if singular else list(task_ids)

        if not return_ids_only:
            logs = self._store.get_logs_by_task_ids(
                task_ids=ids_list if not singular else ids_list[0],
                return_ids_only=False,
            )
            # Opportunistically memoize task_id -> log_id
            try:
                for lg in logs or []:
                    try:
                        e = getattr(lg, "entries", {}) or {}
                        tid = e.get("task_id")
                        lid = getattr(lg, "id", None)
                        if isinstance(tid, int) and isinstance(lid, int):
                            self.cache_log_id(task_id=int(tid), log_id=int(lid))
                    except Exception:
                        continue
            except Exception:
                pass
            return logs

        # return_ids_only=True path
        resolved_by_tid: Dict[int, int] = {}
        missing: List[int] = []
        for tid in ids_list:
            try:
                lid = self._task_log_id_cache.get(int(tid))
                if isinstance(lid, int):
                    resolved_by_tid[int(tid)] = int(lid)
                else:
                    missing.append(int(tid))
            except Exception:
                continue

        if missing:
            try:
                logs = self._store.get_logs_by_task_ids(
                    task_ids=missing if len(missing) > 1 else missing[0],
                    return_ids_only=False,
                )
            except Exception:
                logs = []
            for lg in logs or []:
                try:
                    e = getattr(lg, "entries", {}) or {}
                    tid = e.get("task_id")
                    lid = getattr(lg, "id", None)
                    if isinstance(tid, int) and isinstance(lid, int):
                        resolved_by_tid[int(tid)] = int(lid)
                        self.cache_log_id(task_id=int(tid), log_id=int(lid))
                except Exception:
                    continue

        out: List[int] = []
        for tid in ids_list:
            lid = resolved_by_tid.get(int(tid))
            if isinstance(lid, int):
                out.append(int(lid))
        return out

    def get_minimal_rows_by_task_ids(
        self,
        *,
        task_ids: Union[int, Iterable[int]],
        fields: Optional[List[str]] = None,
    ) -> List[unify.Log]:
        return self._store.get_minimal_rows_by_task_ids(
            task_ids=task_ids,
            fields=fields,
        )

    def get_rows_by_log_ids(self, *, log_ids: List[int]) -> List[unify.Log]:
        return self._store.get_rows_by_log_ids(log_ids=log_ids)

    # ------------------------------- Writes --------------------------------
    def create_one(self, *, entries: Dict[str, Any], new: bool = True) -> unify.Log:
        """
        Create a single log row and apply light cache maintenance.

        - Memoises task_id -> log_id when resolvable from the returned object.
        - Conservatively marks queue index stale when lifecycle fields are present.
        """
        log_obj = self._store.log(entries=entries, new=new)
        try:
            e = getattr(log_obj, "entries", {}) or {}
            tid = e.get("task_id")
            lid = getattr(log_obj, "id", None)
            if isinstance(tid, int) and isinstance(lid, int):
                self.cache_log_id(task_id=int(tid), log_id=int(lid))
        except Exception:
            pass
        try:
            touches_lifecycle = any(
                k in entries for k in ("schedule", "queue_id", "status")
            )
            if touches_lifecycle:
                self._queue_index_stale = True
        except Exception:
            # Be conservative when uncertain
            self._queue_index_stale = True
        return log_obj

    def create_many(self, *, entries_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Batch-create multiple rows and apply light cache maintenance.

        Includes cache handling and optional id memoisation.
        """
        result = self._store.create_many(entries_list=entries_list)
        # Attempt to memoise ids when the API returns full log objects
        try:
            if isinstance(result, list):
                for lg in result:
                    try:
                        e = getattr(lg, "entries", {}) or {}
                        tid = e.get("task_id")
                        lid = getattr(lg, "id", None)
                        if isinstance(tid, int) and isinstance(lid, int):
                            self.cache_log_id(task_id=int(tid), log_id=int(lid))
                    except Exception:
                        continue
        except Exception:
            pass
        # Mark queue index stale when any payload contains lifecycle fields
        try:
            if any(
                isinstance(e, dict)
                and any(k in e for k in ("schedule", "queue_id", "status"))
                for e in entries_list
            ):
                self._queue_index_stale = True
        except Exception:
            self._queue_index_stale = True
        return result

    def write_entries_by_task_ids(
        self,
        *,
        entries_by_tid: Dict[int, Dict[str, Any]],
    ) -> Dict[str, str]:
        """
        Resolve log ids for the provided task_ids and apply per-task entries
        in a single backend update.

        Caller is responsible for invariant checks and neighbour symmetry.
        """
        if not entries_by_tid:
            return {"detail": "No updates"}

        target_tids: List[int] = list(
            dict.fromkeys(int(t) for t in entries_by_tid.keys()),
        )

        logs = self.get_minimal_rows_by_task_ids(
            task_ids=target_tids,
            fields=["task_id"],
        )
        by_tid_to_log_id: Dict[int, int] = {}
        for lg in logs or []:
            try:
                e = getattr(lg, "entries", {}) or {}
                tid = e.get("task_id")
                lid = getattr(lg, "id", None)
                if isinstance(tid, int) and isinstance(lid, int):
                    by_tid_to_log_id[int(tid)] = int(lid)
            except Exception:
                continue

        log_ids: List[int] = []
        entries_list: List[Dict[str, Any]] = []
        for tid in target_tids:
            lid = by_tid_to_log_id.get(int(tid))
            if isinstance(lid, int):
                log_ids.append(int(lid))
                entries_list.append(entries_by_tid[int(tid)])

        if not log_ids:
            return {"detail": "No matching task_ids resolved"}

        return self.write_entries(logs=log_ids, entries=entries_list)

    def write_entries(
        self,
        *,
        logs: Union[int, unify.Log, List[Union[int, unify.Log]]],
        entries: Union[Dict[str, Any], List[Dict[str, Any]]],
        overwrite: bool = True,
    ) -> Dict[str, str]:
        """
        Pass-through write with light cache maintenance.

        If the payload appears to touch queue membership (schedule/queue_id),
        we conservatively mark the queue index stale so readers rebuild later.
        """
        try:
            touches_lifecycle = False
            if isinstance(entries, dict):
                touches_lifecycle = any(
                    k in entries for k in ("schedule", "queue_id", "status")
                )
            elif isinstance(entries, list):
                touches_lifecycle = any(
                    isinstance(e, dict)
                    and any(k in e for k in ("schedule", "queue_id", "status"))
                    for e in entries
                )
        except Exception:
            touches_lifecycle = True

        result = self._store.update(logs=logs, entries=entries, overwrite=overwrite)

        if touches_lifecycle:
            # We do not try to micro-update here; the caller can provide
            # precise updates via update_after_* helpers when the new order is known.
            self._queue_index_stale = True

        return result

    # ------------------------------- Delete --------------------------------
    def delete(self, *, logs: Union[int, List[int]]) -> Dict[str, str]:
        """
        Delete one or more logs by their identifiers via the underlying store.

        Deletions can change queue membership; conservatively mark the queue
        index as stale so readers will rebuild on next access.
        """
        try:
            result = self._store.delete(logs=logs)
        finally:
            try:
                self._queue_index_stale = True
            except Exception:
                pass
        return result

    # ------------------------------- Metrics -------------------------------
    def get_metric_count(self, *, key: str) -> int:
        """Expose count metrics via the view (pass-through)."""
        return self._store.get_metric_count(key=key)

    def get_metric_max(self, *, key: str) -> int:
        """Expose max metrics via the view (pass-through)."""
        return self._store.get_metric_max(key=key)

    # ----------------------------- Env helpers -----------------------------
    @staticmethod
    def _cache_disabled() -> bool:
        try:
            raw = os.getenv("UNITY_TS_LOCAL_VIEW_OFF")
            if raw is None:
                return False
            return str(raw).strip().lower() in {"1", "true", "yes", "on"}
        except Exception:
            return False
