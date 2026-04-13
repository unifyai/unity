"""Headless executor for offline function-backed task activations.

This module runs inside a short-lived Unity Kubernetes Job created by
Communication's offline task lane. It intentionally avoids booting the full
ConversationManager runtime; instead it populates `SESSION_DETAILS`, executes
one stored function entrypoint via `SingleFunctionActor`, and writes run-state
updates back to Orchestra's internal `Tasks/Runs` API.
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

from unity.actor.single_function_actor import SingleFunctionActor
from unity.logger import LOGGER
from unity.session_details import SESSION_DETAILS
from unity.task_scheduler.machine_state import TASK_MACHINE_STATE_PROJECT

TASK_RUN_UPDATE_PATH = "/admin/task-run/update"
HTTP_TIMEOUT_SECONDS = 30
SUMMARY_LIMIT = 4000


@dataclass(frozen=True)
class OfflineTaskConfig:
    """Environment-backed configuration for one offline execution attempt."""

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


def _task_run_update_payload(run_key: str, updates: dict[str, Any]) -> dict[str, Any]:
    """Return the admin payload for one partial run update."""

    return {
        "project_name": TASK_MACHINE_STATE_PROJECT,
        "run_key": run_key,
        "updates": updates,
    }


def _update_task_run(run_key: str, updates: dict[str, Any]) -> None:
    """Persist one partial run update back to Orchestra."""

    orchestra_url = _require_env("ORCHESTRA_URL")
    response = requests.post(
        f"{orchestra_url}{TASK_RUN_UPDATE_PATH}",
        json=_task_run_update_payload(run_key, updates),
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


async def _execute_offline_task(config: OfflineTaskConfig) -> Any:
    """Execute one stored function entrypoint with assistant session context."""

    SESSION_DETAILS.populate_from_env()
    actor = SingleFunctionActor(headless=True)
    try:
        handle = await actor.act(
            request=config.request,
            function_id=config.function_id,
        )
        return await handle.result()
    finally:
        await actor.close()


def main() -> int:
    """Run one offline task to completion and persist the terminal run state."""

    config = _load_config_from_env()
    LOGGER.info(
        "Starting offline task runner for task %s (function_id=%s, run_key=%s)",
        config.task_id,
        config.function_id,
        config.run_key,
    )
    _update_task_run(
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
        _update_task_run(config.run_key, updates)
        return 1

    updates["state"] = "completed"
    LOGGER.info(
        "Offline task execution completed for task %s (run_key=%s)",
        config.task_id,
        config.run_key,
    )
    _update_task_run(config.run_key, updates)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
