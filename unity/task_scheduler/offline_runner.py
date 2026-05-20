"""Entry point for Unity's headless offline task lane.

This module runs inside the short-lived Unity job created by Communication when
a scheduled or triggered task should execute without waking the full live
assistant runtime. It exists to answer one simple question:

"How do we run one stored function-backed task in the background, with the
assistant's identity and primitives available, but without booting the whole
ConversationManager?"

The runner is intentionally small and procedural:

1. Read the activation/run payload that Communication injected into env vars.
2. Populate `SESSION_DETAILS` so shared primitives know which assistant is
   acting.
3. Initialize Unity's normal runtime substrate for the assistant context.
4. Enter `TaskScheduler.execute(...)` with a SingleFunctionActor-backed
   execution delegate so scheduler lifecycle and recurring rearm semantics stay
   central.
5. Persist the terminal run state through the scheduler-owned task run lifecycle.

Communication owns orchestration and job creation. The stored function owns the
actual task behavior. This module is the thin bridge that executes that one
function and keeps the durable run record up to date.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
import traceback
from typing import Any

import requests

import unity
from unity.actor.single_function_actor import SingleFunctionActor
from unity.common.task_execution_context import current_task_execution_delegate
from unity.logger import LOGGER
from unity.session_details import SESSION_DETAILS
from unity.task_scheduler.machine_state import (
    TASK_MACHINE_STATE_PROJECT,
    TaskRunProvenance,
    remember_live_task_run_provenance,
)
from unity.task_scheduler.task_scheduler import TaskScheduler
from unity.task_scheduler.types.activated_by import ActivatedBy

TASK_RUN_UPDATE_PATH = "/admin/task-run/update"
HTTP_TIMEOUT_SECONDS = 30
SUMMARY_LIMIT = 4000
SCHEDULER_MANAGED_SOURCE_TYPES = {"scheduled", "triggered"}


@dataclass(frozen=True)
class OfflineTaskConfig:
    """One fully-materialized offline run request from job environment variables.

    Communication injects these values when it creates the short-lived Unity
    job. Together they identify which assistant is acting, which stored
    function should run, why it was activated, and which durable `Tasks/Runs`
    row should be updated as execution progresses.
    """

    assistant_id: str
    run_key: str
    task_id: int
    function_id: int
    request: str
    source_type: str
    source_task_log_id: int
    activation_revision: str
    task_name: str = ""
    task_description: str = ""
    scheduled_for: str = ""
    source_ref: str = ""
    source_medium: str = ""
    source_contact_id: str = ""


def _require_env(name: str) -> str:
    """Return one required environment variable or raise a clear error."""

    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def _load_config_from_env() -> OfflineTaskConfig:
    """Construct one validated offline task config from process environment."""

    return OfflineTaskConfig(
        assistant_id=_require_env("ASSISTANT_ID"),
        run_key=_require_env("UNITY_OFFLINE_TASK_RUN_KEY"),
        task_id=int(_require_env("UNITY_OFFLINE_TASK_ID")),
        function_id=int(_require_env("UNITY_OFFLINE_TASK_FUNCTION_ID")),
        request=_require_env("UNITY_OFFLINE_TASK_REQUEST"),
        source_type=os.environ.get("UNITY_OFFLINE_TASK_SOURCE_TYPE", "scheduled"),
        source_task_log_id=int(_require_env("UNITY_OFFLINE_TASK_SOURCE_TASK_LOG_ID")),
        activation_revision=_require_env("UNITY_OFFLINE_TASK_ACTIVATION_REVISION"),
        task_name=os.environ.get("UNITY_OFFLINE_TASK_NAME", ""),
        task_description=os.environ.get("UNITY_OFFLINE_TASK_DESCRIPTION", ""),
        scheduled_for=os.environ.get("UNITY_OFFLINE_TASK_SCHEDULED_FOR", ""),
        source_ref=os.environ.get("UNITY_OFFLINE_TASK_SOURCE_REF", ""),
        source_medium=os.environ.get("UNITY_OFFLINE_TASK_SOURCE_MEDIUM", ""),
        source_contact_id=os.environ.get("UNITY_OFFLINE_TASK_SOURCE_CONTACT_ID", ""),
    )


def _orchestra_admin_headers() -> dict[str, str]:
    """Return admin auth headers for Orchestra task-run APIs."""

    admin_key = _require_env("ORCHESTRA_ADMIN_KEY")
    return {"Authorization": f"Bearer {admin_key}"}


def _task_run_update_payload(
    assistant_id: str,
    run_key: str,
    updates: dict[str, Any],
) -> dict[str, Any]:
    """Return the admin payload for one partial run update."""

    return {
        "project_name": TASK_MACHINE_STATE_PROJECT,
        "assistant_id": assistant_id,
        "run_key": run_key,
        "updates": updates,
    }


def _update_task_run(assistant_id: str, run_key: str, updates: dict[str, Any]) -> None:
    """Persist one partial run update back to Orchestra."""

    orchestra_url = _require_env("ORCHESTRA_URL")
    response = requests.post(
        f"{orchestra_url}{TASK_RUN_UPDATE_PATH}",
        json=_task_run_update_payload(assistant_id, run_key, updates),
        headers=_orchestra_admin_headers(),
        timeout=HTTP_TIMEOUT_SECONDS,
    )
    response.raise_for_status()


def _now_iso() -> str:
    """Return the current UTC timestamp in ISO-8601 format."""

    return datetime.now(timezone.utc).isoformat()


def _truncate_text(value: str, limit: int = SUMMARY_LIMIT) -> str:
    """Trim long diagnostic strings so run rows stay compact."""

    if len(value) <= limit:
        return value
    return f"{value[: limit - 3]}..."


def _json_safe_value(value: Any) -> Any:
    """Recursively coerce runtime values into JSON-safe structures."""

    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe_value(item) for item in value]
    return repr(value)


def _build_result_summary(config: OfflineTaskConfig, execution_result: Any) -> str:
    """Serialize a compact, hidden summary of the offline execution outcome."""

    payload = {
        "task_id": config.task_id,
        "function_id": config.function_id,
        "task_name": config.task_name,
        "source_type": config.source_type,
        "scheduled_for": config.scheduled_for or None,
        "source_medium": config.source_medium or None,
        "source_contact_id": config.source_contact_id or None,
        "result": _json_safe_value(getattr(execution_result, "result", None)),
        "stdout": _truncate_text(str(getattr(execution_result, "stdout", "") or "")),
        "stderr": _truncate_text(str(getattr(execution_result, "stderr", "") or "")),
    }
    return _truncate_text(json.dumps(payload, default=str, ensure_ascii=True))


def _is_scheduler_managed(config: OfflineTaskConfig) -> bool:
    """Return whether this request represents a durable assistant task run."""

    return config.source_type in SCHEDULER_MANAGED_SOURCE_TYPES


def _activated_by_for_source_type(source_type: str) -> ActivatedBy:
    """Map the offline source type onto the scheduler activation reason."""

    if source_type == "triggered":
        return ActivatedBy.trigger
    return ActivatedBy.schedule


def _trigger_attempt_token(config: OfflineTaskConfig) -> str | None:
    """Return the pending-provenance claim token for one triggered run."""

    if config.source_type == "triggered":
        return config.run_key
    return None


def _build_offline_provenance(config: OfflineTaskConfig) -> TaskRunProvenance:
    """Return scheduler run provenance matching the Communication run identity."""

    return TaskRunProvenance(
        assistant_id=config.assistant_id,
        task_id=config.task_id,
        source_type=config.source_type,
        execution_mode="offline",
        source_task_log_id=config.source_task_log_id,
        activation_revision=config.activation_revision,
        scheduled_for=config.scheduled_for or None,
        source_medium=config.source_medium or None,
        source_ref=config.source_ref or None,
        source_contact_id=config.source_contact_id or None,
        task_name=config.task_name or None,
        task_description=config.task_description or config.request or None,
        attempt_token=_trigger_attempt_token(config),
    )


class _OfflineTaskHandle:
    """Handle wrapper that converts function execution results into task outcomes."""

    def __init__(self, config: OfflineTaskConfig, inner_handle: Any) -> None:
        self._config = config
        self._inner_handle = inner_handle

    async def result(self) -> str:
        execution_result = await self._inner_handle.result()
        error = str(getattr(execution_result, "error", "") or "").strip()
        if error:
            raise RuntimeError(error)
        return _build_result_summary(self._config, execution_result)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner_handle, name)


class _OfflineTaskExecutionDelegate:
    """Task execution delegate that runs one scheduler-owned entrypoint headlessly."""

    def __init__(self, config: OfflineTaskConfig) -> None:
        self._config = config
        self._actor: SingleFunctionActor | None = None

    async def start_task_run(
        self,
        *,
        task_description: str,
        entrypoint: int | None = None,
        parent_chat_context: list[dict] | None = None,
        clarification_up_q: asyncio.Queue[str] | None = None,
        clarification_down_q: asyncio.Queue[str] | None = None,
        images: list[str] | None = None,
        guidelines: str | None = None,
        **kwargs: Any,
    ) -> _OfflineTaskHandle:
        if entrypoint is None:
            raise RuntimeError(
                f"Offline task {self._config.task_id} has no stored entrypoint.",
            )
        if int(entrypoint) != self._config.function_id:
            raise RuntimeError(
                "Offline task entrypoint mismatch: "
                f"activation requested {self._config.function_id}, "
                f"task row provides {entrypoint}.",
            )

        self._actor = SingleFunctionActor(headless=True)
        handle = await self._actor.act(
            request=task_description,
            function_id=int(entrypoint),
        )
        return _OfflineTaskHandle(self._config, handle)

    async def close(self) -> None:
        if self._actor is not None:
            await self._actor.close()
            self._actor = None


async def _execute_direct_function(config: OfflineTaskConfig) -> Any:
    """Run one non-scheduler function request through SingleFunctionActor."""

    actor = SingleFunctionActor(headless=True)
    try:
        handle = await actor.act(
            request=config.request,
            function_id=config.function_id,
        )
        return await handle.result()
    finally:
        await actor.close()


async def _execute_scheduler_managed_task(config: OfflineTaskConfig) -> Any:
    """Execute one offline task through the scheduler-owned lifecycle."""

    remember_live_task_run_provenance(_build_offline_provenance(config))
    delegate = _OfflineTaskExecutionDelegate(config)
    token = current_task_execution_delegate.set(delegate)
    try:
        scheduler = TaskScheduler()
        handle = await scheduler.execute(
            task_id=config.task_id,
            trigger_attempt_token=_trigger_attempt_token(config),
            _activated_by=_activated_by_for_source_type(config.source_type),
            isolated=True,
        )
        return await handle.result()
    finally:
        current_task_execution_delegate.reset(token)
        await delegate.close()


async def _execute_offline_task(config: OfflineTaskConfig) -> Any:
    """Execute one stored function entrypoint with assistant session context."""

    SESSION_DETAILS.populate_from_env()
    unity.ensure_initialised(project_name=TASK_MACHINE_STATE_PROJECT)
    if _is_scheduler_managed(config):
        return await _execute_scheduler_managed_task(config)
    return await _execute_direct_function(config)


def main() -> int:
    """Run one offline task to completion and persist the final run state."""

    config = _load_config_from_env()
    LOGGER.info(
        "Starting offline task runner for task %s (function_id=%s, run_key=%s)",
        config.task_id,
        config.function_id,
        config.run_key,
    )
    if not _is_scheduler_managed(config):
        _update_task_run(
            config.assistant_id,
            config.run_key,
            {
                "state": "running",
                "started_at": _now_iso(),
            },
        )
    try:
        execution_result = asyncio.run(_execute_offline_task(config))
    except Exception as exc:
        error_text = _truncate_text(traceback.format_exc())
        LOGGER.exception(
            "Offline task runner failed for task %s (run_key=%s)",
            config.task_id,
            config.run_key,
        )
        _update_task_run(
            config.assistant_id,
            config.run_key,
            {
                "state": "failed",
                "completed_at": _now_iso(),
                "error": error_text,
                "result_summary": _truncate_text(
                    json.dumps(
                        {
                            "task_id": config.task_id,
                            "function_id": config.function_id,
                            "error": str(exc),
                        },
                        ensure_ascii=True,
                    ),
                ),
            },
        )
        return 1

    if _is_scheduler_managed(config):
        LOGGER.info(
            "Offline task scheduler lifecycle completed for task %s (run_key=%s)",
            config.task_id,
            config.run_key,
        )
        return 0

    error = str(getattr(execution_result, "error", "") or "").strip()
    updates = {
        "completed_at": _now_iso(),
        "result_summary": _build_result_summary(config, execution_result),
    }
    if error:
        updates["state"] = "failed"
        updates["error"] = _truncate_text(error)
        LOGGER.error(
            "Offline task execution failed for task %s (run_key=%s): %s",
            config.task_id,
            config.run_key,
            error,
        )
        _update_task_run(config.assistant_id, config.run_key, updates)
        return 1

    updates["state"] = "completed"
    LOGGER.info(
        "Offline task execution completed for task %s (run_key=%s)",
        config.task_id,
        config.run_key,
    )
    _update_task_run(config.assistant_id, config.run_key, updates)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
