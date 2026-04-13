"""Read-only helpers for Orchestra-projected Unity task machine state.

The user-authored `Tasks` context remains the source of truth for scheduler
mutations. Orchestra mirrors the machine-facing activation and run state into
`Tasks/Activations` and `Tasks/Runs`; Unity reads those contexts to validate
scheduled wakeups and to narrow triggered-task candidates without polling the
full user task table.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping

from .storage import TasksStore

TASK_ACTIVATIONS_CONTEXT_NAME = "Tasks/Activations"
TASK_RUNS_CONTEXT_NAME = "Tasks/Runs"
_ACTIVATION_QUERY_FIELDS = [
    "assistant_id",
    "activation_key",
    "task_id",
    "source_task_log_id",
    "activation_kind",
    "execution_mode",
    "status",
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
    next_due_at: str | None = None
    trigger_medium: str | None = None
    trigger_from_contact_ids: list[int] = field(default_factory=list)
    trigger_omit_contact_ids: list[int] = field(default_factory=list)
    interrupt: bool = False
    trigger_recurring: bool = False
    entrypoint: str | None = None
    repeat: list[Any] | None = None
    activation_revision: str | None = None


def build_activation_key(*, assistant_id: str | int | None, task_id: int) -> str:
    """Return the assistant-scoped activation key used by Orchestra."""

    normalized_assistant_id = _coerce_str(assistant_id)
    if normalized_assistant_id:
        return f"{normalized_assistant_id}:{task_id}"
    return str(task_id)


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

    return TasksStore(TASK_ACTIVATIONS_CONTEXT_NAME)


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
        entrypoint=_coerce_str(entries.get("entrypoint")),
        repeat=_coerce_list(entries.get("repeat")),
        activation_revision=_coerce_str(entries.get("activation_revision")),
    )


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
