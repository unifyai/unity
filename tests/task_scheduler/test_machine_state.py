from unity.task_scheduler import machine_state
from unity.task_scheduler.machine_state import (
    TaskActivationSnapshot,
    validate_task_due_activation,
)


def test_validate_task_due_activation_accepts_current_activation(monkeypatch):
    activation = TaskActivationSnapshot(
        assistant_id="42",
        activation_key="42:101",
        task_id=101,
        source_task_log_id=555,
        activation_kind="scheduled",
        execution_mode="live",
        next_due_at="2026-04-10T10:00:00+01:00",
        activation_revision="rev-1",
    )
    monkeypatch.setattr(
        machine_state,
        "get_task_activation",
        lambda **_: activation,
    )

    current_activation, stale_reason = validate_task_due_activation(
        assistant_id="42",
        task_id=101,
        activation_revision="rev-1",
        source_task_log_id=555,
        scheduled_for="2026-04-10T09:00:00+00:00",
    )

    assert current_activation == activation
    assert stale_reason is None


def test_validate_task_due_activation_rejects_revision_mismatch(monkeypatch):
    activation = TaskActivationSnapshot(
        assistant_id="42",
        activation_key="42:101",
        task_id=101,
        source_task_log_id=555,
        activation_kind="scheduled",
        execution_mode="live",
        next_due_at="2026-04-10T09:00:00+00:00",
        activation_revision="rev-current",
    )
    monkeypatch.setattr(
        machine_state,
        "get_task_activation",
        lambda **_: activation,
    )

    current_activation, stale_reason = validate_task_due_activation(
        assistant_id="42",
        task_id=101,
        activation_revision="rev-stale",
        source_task_log_id=555,
        scheduled_for="2026-04-10T09:00:00+00:00",
    )

    assert current_activation is None
    assert stale_reason == "activation_revision_mismatch"
