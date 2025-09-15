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
            return

        try:
            existing = unify.get_fields(context=self._ctx) or {}
        except Exception:
            existing = {}
        missing = {k: v for k, v in fields.items() if k not in existing}
        if missing:
            unify.create_fields(missing, context=self._ctx)

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

    # ------------------------------- Writes --------------------------------
    def update(
        self,
        *,
        logs: Union[int, unify.Log, List[Union[int, unify.Log]]],
        entries: Dict[str, Any],
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

    def delete(self, *, logs: Union[int, List[int]]) -> Dict[str, str]:
        return unify.delete_logs(context=self._ctx, logs=logs)
