from unity.task_scheduler import machine_state
from unity.task_scheduler.machine_state import (
    TASK_MACHINE_STATE_PROJECT,
    TaskActivationSnapshot,
    TaskRunProvenance,
    build_task_activation_context_name,
    consume_live_task_run_provenance,
    get_task_activation,
    remember_live_task_run_provenance,
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
    monkeypatch.setattr(machine_state.SESSION_DETAILS.user, "id", "user-1")
    monkeypatch.setattr(machine_state.SESSION_DETAILS.assistant, "agent_id", 42)

    activation = get_task_activation(assistant_id="42", task_id=101)

    assert activation is not None
    assert activation.task_id == 101
    assert captured["project"] == TASK_MACHINE_STATE_PROJECT
    assert captured["context"] == build_task_activation_context_name(
        user_context="user-1",
        assistant_context="42",
    )


def test_trigger_provenance_keeps_attempts_separate(monkeypatch):
    monkeypatch.setattr(machine_state, "_PENDING_LIVE_TASK_RUNS", {})
    monkeypatch.setattr(machine_state, "_PENDING_TRIGGER_LIVE_TASK_RUNS", {})

    activation = TaskActivationSnapshot(
        assistant_id="42",
        activation_key="42:301",
        task_id=301,
        source_task_log_id=555,
        activation_kind="triggered",
        execution_mode="live",
        trigger_medium="sms_message",
        activation_revision="rev-1",
    )
    monkeypatch.setattr(
        machine_state,
        "get_task_activation",
        lambda **_: activation,
    )

    remember_live_task_run_provenance(
        TaskRunProvenance(
            assistant_id="42",
            task_id=301,
            source_type="triggered",
            source_medium="sms_message",
            source_ref="message-1",
            source_contact_id="2",
            attempt_token="attempt-a",
        ),
    )
    remember_live_task_run_provenance(
        TaskRunProvenance(
            assistant_id="42",
            task_id=301,
            source_type="triggered",
            source_medium="sms_message",
            source_ref="message-2",
            source_contact_id="2",
            attempt_token="attempt-b",
        ),
    )

    first = consume_live_task_run_provenance(
        assistant_id="42",
        task_id=301,
        source_type="triggered",
        trigger_attempt_token="attempt-a",
    )
    second = consume_live_task_run_provenance(
        assistant_id="42",
        task_id=301,
        source_type="triggered",
        trigger_attempt_token="attempt-b",
    )
    fallback = consume_live_task_run_provenance(
        assistant_id="42",
        task_id=301,
        source_type="triggered",
    )

    assert first is not None
    assert first.source_ref == "message-1"
    assert second is not None
    assert second.source_ref == "message-2"
    assert fallback is not None
    assert fallback.source_ref is None
    assert fallback.source_contact_id is None
    assert fallback.source_medium == "sms_message"
