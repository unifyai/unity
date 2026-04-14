from unity.task_scheduler import machine_state
from unity.task_scheduler.machine_state import (
    TASK_MACHINE_STATE_PROJECT,
    TaskActivationSnapshot,
    get_task_activation,
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


def test_get_task_activation_queries_assistants_machine_state_project(monkeypatch):
    captured: dict[str, object] = {}

    class _FakeRow:
        entries = {
            "assistant_id": "42",
            "activation_key": "42:101",
            "task_id": 101,
            "activation_kind": "scheduled",
            "execution_mode": "live",
            "activation_revision": "rev-1",
        }

    def _fake_get_logs(**kwargs):
        captured.update(kwargs)
        return [_FakeRow()]

    monkeypatch.setattr("unity.task_scheduler.storage.unify.get_logs", _fake_get_logs)

    activation = get_task_activation(assistant_id="42", task_id=101)

    assert activation is not None
    assert activation.task_id == 101
    assert captured["project"] == TASK_MACHINE_STATE_PROJECT
