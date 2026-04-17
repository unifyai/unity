"""Read-only helpers for Orchestra-projected assistant task machine state.

The user-authored `Tasks` context remains the source of truth for scheduler
mutations. Orchestra mirrors the machine-facing activation and run state into
`Tasks/Activations` and `Tasks/Runs`; Unity reads those contexts to validate
scheduled wakeups and to narrow triggered-task candidates without polling the
full user task table.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping

import requests

from unity.session_details import SESSION_DETAILS
from unity.settings import SETTINGS

from .storage import TasksStore

TASKS_CONTEXT_NAME = "Tasks"
TASK_ACTIVATIONS_CONTEXT_NAME = "Tasks/Activations"
TASK_RUNS_CONTEXT_NAME = "Tasks/Runs"
TASK_OUTBOUND_OPERATIONS_CONTEXT_NAME = "Tasks/OutboundOperations"
_TASK_ACTIVATIONS_CONTEXT_LEAF = "Activations"
_TASK_RUNS_CONTEXT_LEAF = "Runs"
_TASK_OUTBOUND_OPERATIONS_CONTEXT_LEAF = "OutboundOperations"
TASK_MACHINE_STATE_PROJECT = "Assistants"
_TASK_RUN_CREATE_OR_ADOPT_PATH = "/admin/task-run/create-or-adopt"
_TASK_RUN_UPDATE_PATH = "/admin/task-run/update"
_TASK_OUTBOUND_OPERATION_CREATE_OR_ADOPT_PATH = (
    "/admin/task-outbound-operation/create-or-adopt"
)
_TASK_OUTBOUND_OPERATION_UPDATE_PATH = "/admin/task-outbound-operation/update"
_TASK_RUN_HTTP_TIMEOUT_SECONDS = 15
_ACTIVATION_QUERY_FIELDS = [
    "assistant_id",
    "activation_key",
    "task_id",
    "source_task_log_id",
    "activation_kind",
    "execution_mode",
    "status",
    "task_name",
    "task_description",
    "next_due_at",
    "trigger_medium",
    "trigger_from_contact_ids",
    "trigger_omit_contact_ids",
    "interrupt",
    "trigger_recurring",
    "entrypoint",
    "repeat",
    "activation_revision",
]
_DEFAULT_TRIGGER_PAGE_SIZE = 200
_PENDING_LIVE_TASK_RUNS: dict[int, "TaskRunProvenance"] = {}
_PENDING_TRIGGER_LIVE_TASK_RUNS: dict[str, "TaskRunProvenance"] = {}

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TaskActivationSnapshot:
    """Machine-facing activation facts for one assistant/task pair.

    Attributes capture the projected activation contract Unity consumes for due
    validation and trigger matching. The snapshot is intentionally read-only and
    mirrors the Orchestra row closely so callers can reason about staleness and
    execution routing without re-reading the user-authored `Tasks` context.
    """

    assistant_id: str | None
    activation_key: str
    task_id: int
    source_task_log_id: int | None = None
    activation_kind: str | None = None
    execution_mode: str | None = None
    status: str | None = None
    task_name: str | None = None
    task_description: str | None = None
    next_due_at: str | None = None
    trigger_medium: str | None = None
    trigger_from_contact_ids: list[int] = field(default_factory=list)
    trigger_omit_contact_ids: list[int] = field(default_factory=list)
    interrupt: bool = False
    trigger_recurring: bool = False
    entrypoint: int | None = None
    repeat: list[Any] | None = None
    activation_revision: str | None = None


@dataclass(frozen=True)
class TaskRunProvenance:
    """Live or offline provenance facts used to materialize one task run row."""

    assistant_id: str
    task_id: int
    source_type: str
    execution_mode: str = "live"
    source_task_log_id: int | None = None
    activation_revision: str | None = None
    scheduled_for: str | None = None
    source_medium: str | None = None
    source_ref: str | None = None
    source_contact_id: str | None = None
    source_contact_display_name: str | None = None
    task_name: str | None = None
    task_description: str | None = None
    attempt_token: str | None = None


@dataclass(frozen=True)
class TaskRunReference:
    """Stable identifiers needed to patch a materialized task run later."""

    assistant_id: str
    run_key: str


@dataclass(frozen=True)
class TaskOutboundOperationProvenance:
    """Durable provenance facts for one assistant-owned outbound operation."""

    assistant_id: str
    task_run_key: str
    operation_index: int
    method_name: str
    medium: str
    target_kind: str
    target_metadata: Mapping[str, Any] = field(default_factory=dict)
    task_id: int | None = None
    source_task_log_id: int | None = None
    contact_id: int | None = None


@dataclass(frozen=True)
class TaskOutboundOperationReference:
    """Stable identifiers needed to patch one outbound ledger row later."""

    assistant_id: str
    operation_key: str


@dataclass(frozen=True)
class TaskOutboundOperationRecord:
    """Materialized outbound operation row returned by Orchestra admin APIs."""

    reference: TaskOutboundOperationReference
    payload: dict[str, Any]
    created: bool


def build_activation_key(*, assistant_id: str | int | None, task_id: int) -> str:
    """Return the assistant-scoped activation key used by Orchestra."""

    normalized_assistant_id = _coerce_str(assistant_id)
    if normalized_assistant_id:
        return f"{normalized_assistant_id}:{task_id}"
    return str(task_id)


def build_task_activation_context_name(
    *,
    user_context: str | None = None,
    assistant_context: str | None = None,
) -> str:
    """Return the assistant-scoped Orchestra context for activation reads."""

    return _build_task_machine_context_name(
        leaf_name=_TASK_ACTIVATIONS_CONTEXT_LEAF,
        user_context=user_context,
        assistant_context=assistant_context,
    )


def build_task_runs_context_name(
    *,
    user_context: str | None = None,
    assistant_context: str | None = None,
) -> str:
    """Return the assistant-scoped Orchestra context for run reads/writes."""

    return _build_task_machine_context_name(
        leaf_name=_TASK_RUNS_CONTEXT_LEAF,
        user_context=user_context,
        assistant_context=assistant_context,
    )


def build_task_outbound_operations_context_name(
    *,
    user_context: str | None = None,
    assistant_context: str | None = None,
) -> str:
    """Return the assistant-scoped Orchestra context for outbound operation rows."""

    return _build_task_machine_context_name(
        leaf_name=_TASK_OUTBOUND_OPERATIONS_CONTEXT_LEAF,
        user_context=user_context,
        assistant_context=assistant_context,
    )


def _build_task_machine_context_name(
    *,
    leaf_name: str,
    user_context: str | None = None,
    assistant_context: str | None = None,
) -> str:
    """Return one assistant-scoped task-machine context path."""

    resolved_user_context = _coerce_str(user_context) or SESSION_DETAILS.user_context
    resolved_assistant_context = (
        _coerce_str(assistant_context) or SESSION_DETAILS.assistant_context
    )
    return (
        f"{resolved_user_context}/{resolved_assistant_context}/"
        f"{TASKS_CONTEXT_NAME}/{leaf_name}"
    )


def remember_live_task_run_provenance(provenance: TaskRunProvenance) -> None:
    """Remember one pending live-run provenance until the task actually starts."""

    normalized_attempt_token = _normalize_pending_trigger_attempt_token(
        provenance.attempt_token,
    )
    if provenance.source_type == "triggered" and normalized_attempt_token:
        _PENDING_TRIGGER_LIVE_TASK_RUNS[normalized_attempt_token] = provenance
        return
    _PENDING_LIVE_TASK_RUNS[provenance.task_id] = provenance


def consume_live_task_run_provenance(
    *,
    assistant_id: str | int | None,
    task_id: int,
    source_type: str,
    source_task_log_id: int | None = None,
    trigger_attempt_token: str | None = None,
) -> TaskRunProvenance | None:
    """Claim the pending live-run provenance for one task, or build a fallback."""

    normalized_assistant_id = _coerce_str(assistant_id)
    pending = _claim_pending_live_task_run_provenance(
        assistant_id=normalized_assistant_id,
        task_id=task_id,
        source_type=source_type,
        trigger_attempt_token=trigger_attempt_token,
    )
    if pending is not None:
        return pending
    if not normalized_assistant_id:
        return None
    activation = None
    if source_type in {"scheduled", "triggered"}:
        activation = get_task_activation(
            assistant_id=normalized_assistant_id,
            task_id=task_id,
        )
    return TaskRunProvenance(
        assistant_id=normalized_assistant_id,
        task_id=task_id,
        source_type=source_type,
        execution_mode="live",
        source_task_log_id=source_task_log_id
        or (activation.source_task_log_id if activation is not None else None),
        activation_revision=(
            activation.activation_revision if activation is not None else None
        ),
        scheduled_for=(
            activation.next_due_at
            if source_type == "scheduled" and activation
            else None
        ),
        source_medium=(
            activation.trigger_medium
            if source_type == "triggered" and activation
            else None
        ),
        task_name=(activation.task_name if activation is not None else None),
        task_description=(
            activation.task_description if activation is not None else None
        ),
    )


def _claim_pending_live_task_run_provenance(
    *,
    assistant_id: str | None,
    task_id: int,
    source_type: str,
    trigger_attempt_token: str | None,
) -> TaskRunProvenance | None:
    """Claim one pending provenance entry without misattributing another attempt."""

    pending: TaskRunProvenance | None
    if source_type == "triggered":
        normalized_attempt_token = _normalize_pending_trigger_attempt_token(
            trigger_attempt_token,
        )
        if not normalized_attempt_token:
            return None
        pending = _PENDING_TRIGGER_LIVE_TASK_RUNS.pop(normalized_attempt_token, None)
    else:
        pending = _PENDING_LIVE_TASK_RUNS.pop(task_id, None)
    if pending is None:
        return None
    if pending.task_id != task_id:
        logger.warning(
            "Discarding pending live task provenance for mismatched task id "
            "(expected=%s, actual=%s, source_type=%s)",
            task_id,
            pending.task_id,
            source_type,
        )
        return None
    if assistant_id and pending.assistant_id != assistant_id:
        logger.warning(
            "Discarding pending live task provenance for mismatched assistant "
            "(expected=%s, actual=%s, task_id=%s, source_type=%s)",
            assistant_id,
            pending.assistant_id,
            task_id,
            source_type,
        )
        return None
    return pending


def _normalize_pending_trigger_attempt_token(attempt_token: str | None) -> str | None:
    """Return the normalized pending-provenance key for one trigger attempt token."""

    return _normalize_run_key_component(attempt_token)


def source_type_from_activation_reason(reason: str | None) -> str:
    """Normalize scheduler activation reasons into the persisted run source type."""

    normalized_reason = _coerce_str(reason) or "explicit"
    if normalized_reason == "schedule":
        return "scheduled"
    if normalized_reason == "trigger":
        return "triggered"
    return normalized_reason


def create_or_adopt_live_task_run(
    provenance: TaskRunProvenance,
    *,
    started_at: str | None = None,
) -> TaskRunReference | None:
    """Create or adopt one live run row at the moment execution begins."""

    run_key = build_task_run_key(provenance)
    response_body = _orchestra_admin_post(
        _TASK_RUN_CREATE_OR_ADOPT_PATH,
        _drop_none_values(
            {
                "project_name": TASK_MACHINE_STATE_PROJECT,
                "run_key": run_key,
                "assistant_id": provenance.assistant_id,
                "task_id": provenance.task_id,
                "source_task_log_id": provenance.source_task_log_id,
                "source_type": provenance.source_type,
                "execution_mode": provenance.execution_mode,
                "activation_revision": provenance.activation_revision,
                "scheduled_for": provenance.scheduled_for,
                "source_medium": provenance.source_medium,
                "source_ref": provenance.source_ref,
                "source_contact_id": provenance.source_contact_id,
                "source_contact_display_name": provenance.source_contact_display_name,
                "task_name": provenance.task_name,
                "task_description": provenance.task_description,
                "started_at": started_at or _now_iso(),
                "state": "running",
            },
        ),
    )
    if not isinstance(response_body, Mapping):
        return None
    run_payload = response_body.get("run")
    if not isinstance(run_payload, Mapping):
        return None
    persisted_run_key = _coerce_str(run_payload.get("run_key")) or run_key
    return TaskRunReference(
        assistant_id=provenance.assistant_id,
        run_key=persisted_run_key,
    )


def update_task_run_record(
    run_reference: TaskRunReference | None,
    updates: Mapping[str, Any],
) -> None:
    """Patch one previously materialized task run row back in Orchestra."""

    if run_reference is None:
        return
    _orchestra_admin_post(
        _TASK_RUN_UPDATE_PATH,
        {
            "project_name": TASK_MACHINE_STATE_PROJECT,
            "assistant_id": run_reference.assistant_id,
            "run_key": run_reference.run_key,
            "updates": _drop_none_values(dict(updates)),
        },
    )


def create_or_adopt_task_outbound_operation(
    provenance: TaskOutboundOperationProvenance,
    *,
    created_at: str | None = None,
) -> TaskOutboundOperationRecord | None:
    """Create or adopt one outbound operation row for offline send idempotency."""

    operation_key = build_task_outbound_operation_key(provenance)
    response_body = _orchestra_admin_post(
        _TASK_OUTBOUND_OPERATION_CREATE_OR_ADOPT_PATH,
        _drop_none_values(
            {
                "project_name": TASK_MACHINE_STATE_PROJECT,
                "operation_key": operation_key,
                "assistant_id": provenance.assistant_id,
                "task_run_key": provenance.task_run_key,
                "task_id": provenance.task_id,
                "source_task_log_id": provenance.source_task_log_id,
                "operation_index": provenance.operation_index,
                "method_name": provenance.method_name,
                "medium": provenance.medium,
                "target_kind": provenance.target_kind,
                "contact_id": provenance.contact_id,
                "target_metadata": dict(provenance.target_metadata),
                "created_at": created_at or _now_iso(),
                "status": "pending",
            },
        ),
    )
    if not isinstance(response_body, Mapping):
        return None
    operation_payload = response_body.get("operation")
    if not isinstance(operation_payload, Mapping):
        return None
    persisted_operation_key = (
        _coerce_str(operation_payload.get("operation_key")) or operation_key
    )
    return TaskOutboundOperationRecord(
        reference=TaskOutboundOperationReference(
            assistant_id=provenance.assistant_id,
            operation_key=persisted_operation_key,
        ),
        payload=dict(operation_payload),
        created=bool(response_body.get("created")),
    )


def update_task_outbound_operation_record(
    operation_reference: TaskOutboundOperationReference | None,
    updates: Mapping[str, Any],
) -> None:
    """Patch one previously materialized outbound operation row in Orchestra."""

    if operation_reference is None:
        return
    _orchestra_admin_post(
        _TASK_OUTBOUND_OPERATION_UPDATE_PATH,
        {
            "project_name": TASK_MACHINE_STATE_PROJECT,
            "assistant_id": operation_reference.assistant_id,
            "operation_key": operation_reference.operation_key,
            "updates": _drop_none_values(dict(updates)),
        },
    )


def build_task_run_key(provenance: TaskRunProvenance) -> str:
    """Build the canonical run-key shape shared across live and offline lanes.

    The trigger-attempt token is intentionally excluded from the persisted key.
    It only disambiguates pending live trigger provenance before execution
    starts; once a run is materialized, live and offline lanes share the same
    provenance-based identity contract.
    """

    revision_digest = hashlib.sha256(
        str(provenance.activation_revision or "").encode("utf-8"),
    ).hexdigest()[:12]
    tail_parts: list[str] = []
    if provenance.scheduled_for:
        normalized_due = _normalize_run_datetime_fragment(provenance.scheduled_for)
        if normalized_due:
            tail_parts.append(normalized_due)
    normalized_contact_id = _normalize_run_key_component(provenance.source_contact_id)
    if normalized_contact_id:
        tail_parts.append(f"contact-{normalized_contact_id[:24]}")
    normalized_medium = _normalize_run_key_component(provenance.source_medium)
    if normalized_medium:
        tail_parts.append(normalized_medium[:24])
    if provenance.source_ref:
        tail_parts.append(
            hashlib.sha256(provenance.source_ref.encode("utf-8")).hexdigest()[:12],
        )
    tail = "-".join(tail_parts) or "once"
    return (
        f"{provenance.execution_mode}:{provenance.source_type}:"
        f"{provenance.assistant_id}:{provenance.task_id}:{revision_digest}:{tail}"
    )


def build_task_outbound_operation_key(
    provenance: TaskOutboundOperationProvenance,
) -> str:
    """Build the canonical outbound-operation key shared across retry attempts."""

    target_identity = _drop_none_values(
        {
            "contact_id": provenance.contact_id,
            "target_kind": provenance.target_kind,
            "target_metadata": dict(provenance.target_metadata),
        },
    )
    target_digest = hashlib.sha256(
        json.dumps(target_identity, sort_keys=True, default=str).encode("utf-8"),
    ).hexdigest()[:12]
    method_fragment = (
        _normalize_run_key_component(provenance.method_name) or "operation"
    )
    return (
        f"{provenance.task_run_key}:op-{provenance.operation_index}:"
        f"{method_fragment[:24]}:{target_digest}"
    )


def get_task_activation(
    *,
    assistant_id: str | int | None,
    task_id: int,
) -> TaskActivationSnapshot | None:
    """Return the current activation row for one assistant/task pair, if any."""

    activation_key = build_activation_key(
        assistant_id=assistant_id,
        task_id=task_id,
    )
    rows = _activation_store().get_rows(
        filter=f"activation_key == '{activation_key}'",
        limit=1,
        include_fields=_ACTIVATION_QUERY_FIELDS,
    )
    if not rows:
        return None
    return _row_to_activation(rows[0])


def list_trigger_activations(
    *,
    assistant_id: str | int | None,
    medium: str | None = None,
    limit: int = _DEFAULT_TRIGGER_PAGE_SIZE,
) -> list[TaskActivationSnapshot]:
    """List trigger activations for one assistant, optionally scoped by medium."""

    normalized_assistant_id = _coerce_str(assistant_id)
    if not normalized_assistant_id:
        return []
    filter_clauses = [
        f"assistant_id == '{normalized_assistant_id}'",
        "activation_kind == 'triggered'",
    ]
    normalized_medium = _coerce_str(medium)
    if normalized_medium:
        filter_clauses.append(f"trigger_medium == '{normalized_medium}'")
    rows = _activation_store().get_rows(
        filter=" and ".join(filter_clauses),
        limit=limit,
        include_fields=_ACTIVATION_QUERY_FIELDS,
    )
    activations: list[TaskActivationSnapshot] = []
    for row in rows:
        activation = _row_to_activation(row)
        if activation is not None:
            activations.append(activation)
    return activations


def validate_task_due_activation(
    *,
    assistant_id: str | int | None,
    task_id: int,
    activation_revision: str,
    source_task_log_id: int,
    scheduled_for: str,
) -> tuple[TaskActivationSnapshot | None, str | None]:
    """Validate that a scheduled due event still matches the current activation."""

    activation = get_task_activation(
        assistant_id=assistant_id,
        task_id=task_id,
    )
    if activation is None:
        return None, "activation_missing"
    if activation.activation_kind != "scheduled":
        return None, "activation_kind_changed"
    if activation.execution_mode != "live":
        return None, "execution_mode_changed"
    if activation.activation_revision != activation_revision:
        return None, "activation_revision_mismatch"
    if activation.source_task_log_id != source_task_log_id:
        return None, "source_task_log_id_mismatch"
    if _normalize_datetime_string(activation.next_due_at) != _normalize_datetime_string(
        scheduled_for,
    ):
        return None, "scheduled_for_mismatch"
    return activation, None


def _activation_store() -> TasksStore:
    """Return a lightweight reader for the internal activations context."""

    return TasksStore(
        build_task_activation_context_name(),
        project=TASK_MACHINE_STATE_PROJECT,
    )


def _row_to_activation(row: Any) -> TaskActivationSnapshot | None:
    """Convert a Unify log row or raw mapping into a typed activation snapshot."""

    entries = getattr(row, "entries", row)
    if not isinstance(entries, Mapping):
        return None
    task_id = _coerce_int(entries.get("task_id"))
    if task_id is None:
        return None
    assistant_id = _coerce_str(entries.get("assistant_id"))
    activation_key = _coerce_str(entries.get("activation_key")) or build_activation_key(
        assistant_id=assistant_id,
        task_id=task_id,
    )
    return TaskActivationSnapshot(
        assistant_id=assistant_id,
        activation_key=activation_key,
        task_id=task_id,
        source_task_log_id=_coerce_int(entries.get("source_task_log_id")),
        activation_kind=_coerce_str(entries.get("activation_kind")),
        execution_mode=_coerce_str(entries.get("execution_mode")),
        status=_coerce_str(entries.get("status")),
        task_name=_coerce_str(entries.get("task_name")),
        task_description=_coerce_str(entries.get("task_description")),
        next_due_at=_coerce_str(entries.get("next_due_at")),
        trigger_medium=_coerce_str(entries.get("trigger_medium")),
        trigger_from_contact_ids=_coerce_int_list(
            entries.get("trigger_from_contact_ids"),
        ),
        trigger_omit_contact_ids=_coerce_int_list(
            entries.get("trigger_omit_contact_ids"),
        ),
        interrupt=bool(entries.get("interrupt", False)),
        trigger_recurring=bool(entries.get("trigger_recurring", False)),
        entrypoint=_coerce_int(entries.get("entrypoint")),
        repeat=_coerce_list(entries.get("repeat")),
        activation_revision=_coerce_str(entries.get("activation_revision")),
    )


def _orchestra_admin_post(
    path: str,
    payload: Mapping[str, Any],
) -> dict[str, Any] | None:
    """POST one task-machine admin payload back to Orchestra."""

    orchestra_url = (SETTINGS.ORCHESTRA_URL or "").rstrip("/")
    admin_key = SETTINGS.ORCHESTRA_ADMIN_KEY.get_secret_value()
    if not orchestra_url or not admin_key:
        logger.warning(
            "Skipping task-run persistence because ORCHESTRA_URL or ORCHESTRA_ADMIN_KEY is missing.",
        )
        return None
    response = requests.post(
        f"{orchestra_url}{path}",
        json=dict(payload),
        headers={"Authorization": f"Bearer {admin_key}"},
        timeout=_TASK_RUN_HTTP_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    body = response.json()
    return body if isinstance(body, dict) else None


def _drop_none_values(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return a shallow copy without `None` values."""

    return {key: value for key, value in payload.items() if value is not None}


def _normalize_run_key_component(value: str | None) -> str | None:
    """Normalize one free-form run-key component into a compact identifier."""

    text = _coerce_str(value)
    if not text:
        return None
    normalized_chars = [
        char.lower() if char.isalnum() else "-" for char in text.strip()
    ]
    normalized = "".join(normalized_chars).strip("-")
    while "--" in normalized:
        normalized = normalized.replace("--", "-")
    return normalized or None


def _normalize_run_datetime_fragment(value: str | None) -> str | None:
    """Normalize one datetime string into the canonical run-key timestamp fragment."""

    normalized = _normalize_datetime_string(value)
    if not normalized:
        return None
    try:
        return datetime.fromisoformat(normalized.replace("Z", "+00:00")).strftime(
            "%Y%m%dT%H%M%SZ",
        )
    except ValueError:
        return None


def _now_iso() -> str:
    """Return the current UTC timestamp in ISO-8601 format."""

    return datetime.now(timezone.utc).isoformat()


def _coerce_int(value: Any) -> int | None:
    """Best-effort integer coercion for JSON-backed activation rows."""

    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _coerce_int_list(value: Any) -> list[int]:
    """Return a list of integer ids, dropping non-coercible elements."""

    values = _coerce_list(value) or []
    coerced: list[int] = []
    for item in values:
        int_value = _coerce_int(item)
        if int_value is not None:
            coerced.append(int_value)
    return coerced


def _coerce_list(value: Any) -> list[Any] | None:
    """Normalize list-like values while preserving `None`."""

    if value is None:
        return None
    if isinstance(value, list):
        return value
    return [value]


def _coerce_str(value: Any) -> str | None:
    """Normalize scalar values to strings while preserving `None`."""

    if value is None:
        return None
    text = str(value)
    return text if text else None


def _normalize_datetime_string(value: str | None) -> str | None:
    """Return one canonical UTC ISO-8601 string for datetime comparisons."""

    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return value
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).isoformat()
