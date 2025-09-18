from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Union
from enum import Enum
from functools import cached_property

import unify


class TasksStore:
    """
    Thin adapter around Unify I/O for the Tasks context.

    Purpose: centralise all Unify calls used by TaskScheduler and helpers, so
    refactors can adjust behaviour (e.g., overwrite semantics, retries) in one
    place without touching higher-level logic.
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
        return logs

    # ------------------------------- Writes --------------------------------
    def update(
        self,
        *,
        logs: Union[int, unify.Log, List[Union[int, unify.Log]]],
        entries: Union[Dict[str, Any], List[Dict[str, Any]]],
        overwrite: bool = True,
    ) -> Dict[str, str]:
        # Preserve 'activated_by' unless the caller explicitly sets/clears it.

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
