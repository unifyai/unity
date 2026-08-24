"""
Storage adapter for the Task Scheduler.

TasksStore centralizes Unify I/O for the Tasks context (contexts, fields,
reads, writes, and metrics).
"""

from __future__ import annotations

import datetime
import json
import logging
from typing import Any, Dict, Iterable, List, Optional, Union
from enum import Enum
from functools import cached_property

import unisdk

from unisdk.utils.http import RequestError as _UnifyRequestError
from unify.common.authorship import strip_authoring_assistant_id
from unify.common.log_utils import log as unity_log, create_logs as unity_create_logs
from pydantic import BaseModel

LOGGER = logging.getLogger(__name__)


class TasksStore:
    """
    Adapter around Unify I/O for the Tasks context.

    Centralises reads, writes, field management, and metrics used by the
    scheduler and related utilities.
    """

    def __init__(
        self,
        context: str,
        *,
        project: str | None = None,
    ) -> None:
        self._ctx = context
        self._project = project or unisdk.active_project()

    @property
    def context(self) -> str:
        return self._ctx

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
        from unify.common.context_store import _create_context_with_retry

        _create_context_with_retry(
            self._ctx,
            unique_keys=unique_keys,
            auto_counting=auto_counting,
            description=description,
            foreign_keys=foreign_keys,
            project=self._project,
        )

        # Ensure all required fields exist (idempotent per-field)
        try:
            existing = (
                unisdk.get_fields(
                    project=self._project,
                    context=self._ctx,
                )
                or {}
            )
        except Exception:
            existing = {}
        missing = {k: v for k, v in fields.items() if k not in existing}
        if missing:
            try:
                unisdk.create_fields(
                    missing,
                    project=self._project,
                    context=self._ctx,
                )
            except Exception:
                pass  # Fields already exist or transient failure
        # Refresh the local fields cache from the backend so we only ever
        # read the canonical 'data_type' representation.
        try:
            updated = (
                unisdk.get_fields(
                    project=self._project,
                    context=self._ctx,
                )
                or {}
            )
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

    # ------------------------------- Reads ---------------------------------
    @cached_property
    def fields(self) -> Dict[str, str]:
        try:
            fields = (
                unisdk.get_fields(
                    project=self._project,
                    context=self._ctx,
                )
                or {}
            )
            return {
                k: (v.get("data_type") if isinstance(v, dict) else str(v))
                for k, v in fields.items()
            }
        except Exception:
            return {}

    def _safe_get_logs(self, **kwargs) -> List[unisdk.Log] | List[int]:
        """Get logs, treating missing contexts as empty during fresh/test runs."""
        if "project" not in kwargs:
            kwargs["project"] = self._project
        try:
            return unisdk.get_logs(**kwargs)
        except _UnifyRequestError as e:
            status = getattr(getattr(e, "response", None), "status_code", None)
            if status == 404:
                LOGGER.debug(
                    "TasksStore read returned 404; treating as empty. "
                    "project=%s context=%s filter=%r",
                    kwargs.get("project"),
                    kwargs.get("context"),
                    kwargs.get("filter"),
                )
                return []
            raise

    def get_metric_count(self, *, key: str) -> int:
        ret = unisdk.get_logs_metric(
            metric="count",
            key=key,
            project=self._project,
            context=self._ctx,
        )
        return 0 if ret is None else int(ret)

    def get_metric_max(self, *, key: str) -> int:
        """
        Return the maximum value observed for a numeric field within this context.

        When the backend does not return a value (e.g., empty context), 0 is returned.
        """
        ret = unisdk.get_logs_metric(
            metric="max",
            key=key,
            project=self._project,
            context=self._ctx,
        )
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
    ) -> Union[List[int], List[unisdk.Log]]:
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
    ) -> List[Union[int, unisdk.Log]]:
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
    ) -> List[unisdk.Log]:
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

    @staticmethod
    def _with_explicit_task_types(entries: Any) -> Any:
        if isinstance(entries, list):
            return [TasksStore._with_explicit_task_types(entry) for entry in entries]
        if not isinstance(entries, dict):
            return entries
        out = dict(entries)
        # Legacy rows / callers may still pass trigger as JSON text; the field is
        # registered as dict, so decode before write.
        trigger = out.get("trigger")
        if isinstance(trigger, str):
            out["trigger"] = json.loads(trigger)
        explicit_types = dict(out.get("explicit_types") or {})
        # Nested schedule/trigger columns are registered as dict; declare that on
        # write so Orchestra does not infer a conflicting scalar type.
        if out.get("schedule") is not None:
            schedule_types = dict(explicit_types.get("schedule") or {})
            schedule_types["type"] = "dict"
            explicit_types["schedule"] = schedule_types
        if out.get("trigger") is not None:
            trigger_types = dict(explicit_types.get("trigger") or {})
            trigger_types["type"] = "dict"
            explicit_types["trigger"] = trigger_types
        if explicit_types:
            out["explicit_types"] = explicit_types
        return out

    # ------------------------------- Writes --------------------------------
    def update(
        self,
        *,
        logs: Union[int, unisdk.Log, List[Union[int, unisdk.Log]]],
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

        norm_entries = strip_authoring_assistant_id(
            TasksStore._with_explicit_task_types(
                _strip_nones(TasksStore._norm(entries), top_level=True),
            ),
        )
        return unisdk.update_logs(
            logs=logs,
            context=self._ctx,
            entries=norm_entries,
            overwrite=True,
        )

    def log(self, *, entries: Dict[str, Any], new: bool = True) -> unisdk.Log:
        norm_entries = TasksStore._with_explicit_task_types(TasksStore._norm(entries))
        # Create with expanded fields so auto-counting applies when ids are omitted
        return unity_log(
            project=self._project,
            context=self._ctx,
            new=new,
            stamp_authoring=True,
            **norm_entries,
        )

    def create_many(self, *, entries_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Batch-create multiple logs in a single backend call.

        Returns the raw response from the backend which typically includes
        'log_event_ids' (ids of created log events) and a 'row_ids' structure
        with auto-incremented row identifiers.
        """

        normalised = [
            TasksStore._with_explicit_task_types({**TasksStore._norm(e)})
            for e in entries_list
        ]
        try:
            return unity_create_logs(
                project=self._project,
                context=self._ctx,
                entries=normalised,
                stamp_authoring=True,
            )
        except Exception:
            # Fallback: create sequentially (preserves correctness if batch API is unavailable)
            log_ids: list[int] = []
            for e in normalised:
                lg = unity_log(
                    project=self._project,
                    context=self._ctx,
                    new=True,
                    stamp_authoring=True,
                    **e,
                )
                try:
                    log_ids.append(lg.id)
                except Exception:
                    pass
            return {"log_event_ids": log_ids}

    def get_rows_by_log_ids(self, *, log_ids: List[int]) -> List[unisdk.Log]:
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
        return unisdk.delete_logs(
            project=self._project,
            context=self._ctx,
            logs=logs,
        )
