"""
Storage and local view utilities for the Task Scheduler.

- TasksStore: centralizes Unify I/O for the Tasks context (contexts, fields,
  reads/writes, metrics, checkpoints).
- LocalTaskView: best‑effort cache for queue membership, head start_at,
  queue‑id allocation, and log‑id memoization with convenience wrappers.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Union, Literal, overload
from enum import Enum
from functools import cached_property

import unify

from unity.settings import SETTINGS
from unify.utils.http import RequestError as _UnifyRequestError
from unity.common.context_store import _PRIVATE_FIELDS
from unity.common.log_utils import log as unity_log, create_logs as unity_create_logs
from unity.task_scheduler.types.queue_summary import QueueSummary
import datetime
from pydantic import BaseModel
from .types.task import Task
from .types.status import Status


class TasksStore:
    """
    Adapter around Unify I/O for the Tasks context.

    Centralises reads, writes, field management, metrics, and checkpoint
    helpers used by the scheduler and related utilities.
    """

    def __init__(self, context: str, *, add_to_all_context: bool = False) -> None:
        self._ctx = context
        self._add_to_all_context = add_to_all_context

    # ----------------------------- Context ---------------------------------
    def ensure_context(
        self,
        *,
        unique_keys: Dict[str, str],
        auto_counting: Dict[str, Optional[str]],
        description: str,
        fields: Dict[str, str],
        foreign_keys: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """
        Ensure the Tasks context exists with the requested fields.

        If the context already exists, any missing fields are created.
        Idempotent: tolerates pre-existing contexts and concurrent creation.
        """
        from unity.common.context_store import _create_context_with_retry

        _create_context_with_retry(
            self._ctx,
            unique_keys=unique_keys,
            auto_counting=auto_counting,
            description=description,
            foreign_keys=foreign_keys,
        )

        # Ensure all required fields exist (idempotent per-field)
        try:
            existing = unify.get_fields(context=self._ctx) or {}
        except Exception:
            existing = {}
        missing = {k: v for k, v in fields.items() if k not in existing}
        if missing:
            try:
                unify.create_fields(missing, context=self._ctx)
            except Exception:
                pass  # Fields already exist or transient failure
        # Refresh the local fields cache from the backend so we only ever
        # read the canonical 'data_type' representation.
        try:
            updated = unify.get_fields(context=self._ctx) or {}
        except Exception:
            updated = existing
        try:
            normalised = {
                k: (v.get("data_type") if isinstance(v, dict) else str(v))
                for k, v in updated.items()
            }
        except Exception:
            normalised = {k: str(v) for k, v in updated.items()}
        self.__dict__["fields"] = normalised

        # Ensure aggregation contexts exist for cross-assistant and cross-user queries
        all_ctxs = self._all_contexts()
        if all_ctxs:
            self._ensure_all_contexts(all_ctxs=all_ctxs, fields=fields)

    def _all_contexts(self) -> List[str]:
        """
        Derive aggregation contexts for this user/assistant-scoped context.

        Returns two contexts for cross-assistant and cross-user aggregation:
          - {user_id}/All/{suffix} - all assistants for this user
          - All/{suffix}           - all users, all assistants

        Example: "42/7/Tasks" returns:
          - "42/All/Tasks"
          - "All/Tasks"

        Returns empty list if context doesn't have user_id/assistant_id prefix.
        """
        parts = self._ctx.split("/")
        if len(parts) < 3:
            return []
        user_ctx = parts[0]
        suffix = "/".join(parts[2:])  # Everything after user_id/assistant_id
        return [
            f"{user_ctx}/All/{suffix}",  # User-level aggregation
            f"All/{suffix}",  # Global aggregation
        ]

    def _ensure_all_contexts(
        self,
        *,
        all_ctxs: List[str],
        fields: Dict[str, str],
    ) -> None:
        """
        Ensure aggregation contexts exist for cross-assistant and cross-user queries.

        Unlike the main context, these do not use unique_keys or auto_counting
        since logs are added by reference from multiple assistant contexts.
        """
        for all_ctx in all_ctxs:
            # Determine description based on aggregation level
            if all_ctx.startswith("All/"):
                description = f"Global aggregation of {self._ctx.split('/')[-1]} across all users and assistants"
            else:
                description = f"Aggregation of {self._ctx.split('/')[-1]} across all assistants for this user"

            unify.create_context(all_ctx, description=description)

            # Merge manager fields with private fields for All context
            fields_with_private = dict(fields)
            fields_with_private.update(_PRIVATE_FIELDS)

            # Ensure all required fields exist (idempotent per-field)
            try:
                existing = unify.get_fields(context=all_ctx) or {}
            except Exception:
                existing = {}
            missing = {
                k: v for k, v in fields_with_private.items() if k not in existing
            }
            if missing:
                try:
                    unify.create_fields(missing, context=all_ctx)
                except Exception:
                    pass  # Fields already exist or transient failure

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

    def _safe_get_logs(self, **kwargs) -> List[unify.Log] | List[int]:
        """Get logs, treating missing contexts as empty during fresh/test runs."""
        try:
            return unify.get_logs(**kwargs)
        except _UnifyRequestError as e:
            status = getattr(getattr(e, "response", None), "status_code", None)
            if status == 404:
                return []
            raise

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
        include_fields: Optional[List[str]] = None,
    ) -> Union[List[int], List[unify.Log]]:
        return self._safe_get_logs(
            context=self._ctx,
            filter=filter,
            offset=offset,
            limit=limit,
            return_ids_only=return_ids_only,
            exclude_fields=exclude_fields,
            from_fields=include_fields,
        )

    def get_logs_by_task_ids(
        self,
        *,
        task_ids: Union[int, Iterable[int]],
        return_ids_only: bool = True,
    ) -> List[Union[int, unify.Log]]:
        singular = isinstance(task_ids, int)
        original_id = task_ids if singular else None
        ids_list = [task_ids] if singular else list(task_ids)
        logs = self._safe_get_logs(
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
        still include their underlying log ids. The log always includes task_id.
        """
        singular = isinstance(task_ids, int)
        original_id = task_ids if singular else None
        ids_list = [task_ids] if singular else list(task_ids)
        # Ensure we always include task_id in the projection for correctness
        fields = list(dict.fromkeys((fields or []) + ["task_id"]))
        logs = self._safe_get_logs(
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

        return logs

    @staticmethod
    def _norm(v: Any) -> Any:
        # Normalize enums to their underlying values
        if isinstance(v, Enum):
            return v.value
        # Datetime family → ISO-8601 strings
        if isinstance(v, (datetime.datetime, datetime.date, datetime.time)):
            return v.isoformat()
        # Pydantic models → plain dict (JSON mode for consistent strings)
        if isinstance(v, BaseModel):
            return TasksStore._norm(v.model_dump(mode="json"))
        if isinstance(v, dict):
            return {k: TasksStore._norm(x) for k, x in v.items()}
        if isinstance(v, list):
            return [TasksStore._norm(x) for x in v]
        return v

    # ------------------------------- Writes --------------------------------
    def update(
        self,
        *,
        logs: Union[int, unify.Log, List[Union[int, unify.Log]]],
        entries: Union[Dict[str, Any], List[Dict[str, Any]]],
    ) -> Dict[str, str]:
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

        norm_entries = _strip_nones(TasksStore._norm(entries), top_level=True)
        return unify.update_logs(
            logs=logs,
            context=self._ctx,
            entries=norm_entries,
            overwrite=True,
        )

    def log(self, *, entries: Dict[str, Any], new: bool = True) -> unify.Log:
        norm_entries = TasksStore._norm(entries)
        # Create with expanded fields so auto-counting applies when ids are omitted
        return unity_log(
            context=self._ctx,
            new=new,
            add_to_all_context=self._add_to_all_context,
            **norm_entries,
        )

    def create_many(self, *, entries_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Batch-create multiple logs in a single backend call.

        Returns the raw response from the backend which typically includes
        'log_event_ids' (ids of created log events) and a 'row_ids' structure
        with auto-incremented row identifiers.
        """

        normalised = [{**TasksStore._norm(e)} for e in entries_list]
        try:
            return unity_create_logs(
                context=self._ctx,
                entries=normalised,
                add_to_all_context=self._add_to_all_context,
            )
        except Exception:
            # Fallback: create sequentially (preserves correctness if batch API is unavailable)
            log_ids: list[int] = []
            for e in normalised:
                lg = unity_log(
                    context=self._ctx,
                    new=True,
                    add_to_all_context=self._add_to_all_context,
                    **e,
                )
                try:
                    log_ids.append(lg.id)
                except Exception:
                    pass
            return {"log_event_ids": log_ids}

    def get_rows_by_log_ids(self, *, log_ids: List[int]) -> List[unify.Log]:
        """
        Fetch full log objects by their log-event ids. This avoids filtering by
        field values and allows precise retrieval of freshly-created rows.
        """
        res = self._safe_get_logs(
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

    # (removed) Checkpoint helpers – checkpoints are in-memory only in TaskScheduler


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
    @overload
    def get_rows(
        self,
        *,
        filter: Optional[str] = None,
        offset: int = 0,
        limit: int = 100,
        return_ids_only: Literal[True],
        exclude_fields: Optional[List[str]] = None,
        include_fields: Optional[List[str]] = None,
    ) -> List[int]: ...

    @overload
    def get_rows(
        self,
        *,
        filter: Optional[str] = None,
        offset: int = 0,
        limit: int = 100,
        return_ids_only: Literal[False] = False,
        exclude_fields: Optional[List[str]] = None,
        include_fields: Optional[List[str]] = None,
    ) -> List[unify.Log]: ...

    def get_rows(
        self,
        *,
        filter: Optional[str] = None,
        offset: int = 0,
        limit: int = 100,
        return_ids_only: bool = False,
        exclude_fields: Optional[List[str]] = None,
        include_fields: Optional[List[str]] = None,
    ) -> Union[List[int], List[unify.Log]]:
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
            include_fields=include_fields,
        )

    # ----------------------------- Queue index -----------------------------
    def mark_queue_changed(self) -> None:
        self._queue_index_stale = True

    def refresh_queue_index_from_rows(self, rows: List[Task]) -> None:
        """
        Build queue caches from a list of row dicts containing at least:
        task_id, schedule (dict), status, queue_id.
        """
        try:
            # Filter to runnable rows with linkage
            runnable = [
                r
                for r in (rows or [])
                if r.schedule is not None
                and r.status not in (Status.completed, Status.cancelled, Status.failed)
            ]

            rows_by_id: Dict[int, Task] = {}
            for r in runnable:
                rows_by_id[r.task_id] = r

            # Identify heads by prev_task is None and numeric queue_id
            heads: List[Task] = []
            for r in runnable:
                if r.schedule is None:
                    continue
                if r.schedule.prev_task is None and r.queue_id is not None:
                    heads.append(r)

            new_index: Dict[int, List[int]] = {}
            new_reverse: Dict[int, int] = {}
            new_head_start: Dict[int, Optional[str]] = {}

            for h in heads:
                qid = h.queue_id

                if qid is None:
                    continue

                order: List[int] = []
                seen: set[int] = set()
                cur = h
                while cur is not None:
                    try:
                        tid_val = cur.task_id
                        tid = int(tid_val) if tid_val is not None else None
                    except Exception:
                        tid = None
                    if tid is None or tid in seen:
                        break
                    seen.add(tid)
                    order.append(tid)
                    nxt = cur.schedule_next
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
                    new_head_start[qid] = (
                        h.schedule_start_at.isoformat()
                        if h.schedule_start_at is not None
                        else None
                    )

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
            rows = self._store.get_rows(
                filter=(
                    "schedule is not None and "
                    "status not in ('completed','cancelled','failed')"
                ),
            )
        except Exception:
            rows = []
        self.refresh_queue_index_from_rows([Task(**r.entries) for r in rows])

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

    def get_all_queue_summaries(self) -> List[QueueSummary]:
        """
        Return cached summaries for all runnable queues.

        Each summary contains: {"queue_id": int, "order": list[int], "start_at": str | None}
        """
        try:
            if self._cache_disabled() or self._queue_index_stale:
                self.rebuild_queue_index()
            out: List[QueueSummary] = []
            for qid, order in self._queue_index.items():
                if not order:
                    continue
                out.append(
                    QueueSummary(
                        queue_id=int(qid),
                        order=list(order),
                        start_at=self._queue_head_start_at.get(int(qid)),
                    ),
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
    ) -> Union[List[int], List[unify.Log]]:
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
                task_ids=ids_list,
                return_ids_only=False,
            )
            # Opportunistically memoize task_id -> log_id
            for lg in logs:
                tid = lg.entries.get("task_id")
                lid = lg.id
                if tid is not None and lid is not None:
                    self.cache_log_id(task_id=tid, log_id=lid)
            return logs

        # return_ids_only=True path
        resolved_by_tid: Dict[int, int] = {}
        missing: List[int] = []
        for tid in ids_list:
            lid = self._task_log_id_cache.get(int(tid))
            if lid is not None:
                resolved_by_tid[int(tid)] = int(lid)
            else:
                missing.append(int(tid))

        if missing:
            try:
                logs = self._store.get_logs_by_task_ids(
                    task_ids=missing if len(missing) > 1 else missing[0],
                    return_ids_only=False,
                )
            except Exception:
                logs = []
            for lg in logs:
                tid = lg.entries.get("task_id")
                lid = lg.id
                if tid is not None and lid is not None:
                    resolved_by_tid[tid] = lid
                    self.cache_log_id(task_id=tid, log_id=lid)

        out: List[int] = []
        for tid in ids_list:
            lid = resolved_by_tid.get(int(tid))
            if lid is not None:
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
            task_id = log_obj.entries.get("task_id")
            log_id = log_obj.id
            if task_id is not None and log_id is not None:
                self.cache_log_id(task_id=task_id, log_id=log_id)
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
                        task_id = lg.entries.get("task_id")
                        log_id = lg.id
                        if task_id is not None and log_id is not None:
                            self.cache_log_id(task_id=task_id, log_id=log_id)
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
            task_id = lg.entries.get("task_id")
            log_id = lg.id
            if task_id is not None and log_id is not None:
                by_tid_to_log_id[task_id] = log_id

        log_ids: List[int] = []
        entries_list: List[Dict[str, Any]] = []
        for task_id in target_tids:
            log_id = by_tid_to_log_id.get(task_id)
            if log_id is not None:
                log_ids.append(log_id)
                entries_list.append(entries_by_tid[task_id])

        if not log_ids:
            return {"detail": "No matching task_ids resolved"}

        return self.write_entries(logs=log_ids, entries=entries_list)

    def write_entries(
        self,
        *,
        logs: Union[int, unify.Log, List[Union[int, unify.Log]]],
        entries: Union[Dict[str, Any], List[Dict[str, Any]]],
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

        result = self._store.update(logs=logs, entries=entries)

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
            self._queue_index_stale = True
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
        return SETTINGS.task.LOCAL_VIEW_OFF
