import hashlib

from unity.task_scheduler import machine_state
from unity.task_scheduler.machine_state import (
    TASK_MACHINE_STATE_PROJECT,
    TaskActivationSnapshot,
    TaskOutboundOperationProvenance,
    TaskRunProvenance,
    build_task_activation_context_name,
    build_task_outbound_operation_key,
    build_task_run_key,
    create_or_adopt_task_outbound_operation,
    create_or_adopt_live_task_run,
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


def test_build_task_run_key_ignores_trigger_attempt_token():
    revision_digest = hashlib.sha256(b"rev-1").hexdigest()[:12]
    source_ref_digest = hashlib.sha256(b"message-1").hexdigest()[:12]
    with_attempt = TaskRunProvenance(
        assistant_id="42",
        task_id=301,
        source_type="triggered",
        execution_mode="live",
        activation_revision="rev-1",
        source_medium="sms_message",
        source_ref="message-1",
        source_contact_id="2",
        attempt_token="attempt-a",
    )
    without_attempt = TaskRunProvenance(
        assistant_id="42",
        task_id=301,
        source_type="triggered",
        execution_mode="live",
        activation_revision="rev-1",
        source_medium="sms_message",
        source_ref="message-1",
        source_contact_id="2",
    )

    expected = (
        f"live:triggered:42:301:{revision_digest}:"
        f"contact-2-sms-message-{source_ref_digest}"
    )

    assert build_task_run_key(with_attempt) == expected
    assert build_task_run_key(without_attempt) == expected


def test_create_or_adopt_live_task_run_persists_display_fields(monkeypatch):
    captured: dict[str, object] = {}
    provenance = TaskRunProvenance(
        assistant_id="42",
        task_id=301,
        source_type="triggered",
        execution_mode="live",
        activation_revision="rev-1",
        source_medium="sms_message",
        source_ref="message-1",
        source_contact_id="2",
        source_contact_display_name="Alice Owner",
        task_name="Follow up on invoice",
        task_description="Ask Alice for the missing invoice details.",
    )

    def _fake_post(path: str, payload: dict[str, object]):
        captured["path"] = path
        captured["payload"] = payload
        return {"run": {"run_key": "live:triggered:42:301:rev-1:once"}}

    monkeypatch.setattr(machine_state, "_orchestra_admin_post", _fake_post)

    reference = create_or_adopt_live_task_run(
        provenance,
        started_at="2026-04-16T10:15:00+00:00",
    )

    assert reference is not None
    payload = captured["payload"]
    assert isinstance(payload, dict)
    assert payload["source_contact_display_name"] == "Alice Owner"
    assert payload["task_name"] == "Follow up on invoice"
    assert payload["task_description"] == "Ask Alice for the missing invoice details."
    assert payload["started_at"] == "2026-04-16T10:15:00+00:00"


def test_build_task_outbound_operation_key_is_stable_for_same_target():
    provenance = TaskOutboundOperationProvenance(
        assistant_id="42",
        task_run_key="offline:scheduled:42:301:rev:once",
        operation_index=2,
        method_name="send_sms",
        medium="sms_message",
        target_kind="contact",
        contact_id=7,
        target_metadata={"phone_number": "+15555550123"},
    )
    equivalent = TaskOutboundOperationProvenance(
        assistant_id="42",
        task_run_key="offline:scheduled:42:301:rev:once",
        operation_index=2,
        method_name="send_sms",
        medium="sms_message",
        target_kind="contact",
        contact_id=7,
        target_metadata={"phone_number": "+15555550123"},
    )

    assert build_task_outbound_operation_key(
        provenance,
    ) == build_task_outbound_operation_key(
        equivalent,
    )


def test_create_or_adopt_task_outbound_operation_persists_target_metadata(monkeypatch):
    captured: dict[str, object] = {}
    provenance = TaskOutboundOperationProvenance(
        assistant_id="42",
        task_run_key="offline:scheduled:42:301:rev:once",
        operation_index=1,
        method_name="send_email",
        medium="email",
        target_kind="contact",
        contact_id=17,
        task_id=301,
        source_task_log_id=555,
        target_metadata={
            "email_address": "alice@example.com",
            "display_name": "Alice Owner",
        },
    )

    def _fake_post(path: str, payload: dict[str, object]):
        captured["path"] = path
        captured["payload"] = payload
        return {
            "created": True,
            "operation": {
                "operation_key": payload["operation_key"],
                "status": "pending",
            },
        }

    monkeypatch.setattr(machine_state, "_orchestra_admin_post", _fake_post)

    record = create_or_adopt_task_outbound_operation(
        provenance,
        created_at="2026-04-16T10:15:00+00:00",
    )

    assert record is not None
    payload = captured["payload"]
    assert isinstance(payload, dict)
    assert payload["task_run_key"] == provenance.task_run_key
    assert payload["contact_id"] == 17
    assert payload["target_metadata"] == {
        "email_address": "alice@example.com",
        "display_name": "Alice Owner",
    }
    assert payload["created_at"] == "2026-04-16T10:15:00+00:00"


def test_update_task_outbound_operation_record_posts_partial_updates(monkeypatch):
    captured: dict[str, object] = {}

    def _fake_post(path: str, payload: dict[str, object]):
        captured["path"] = path
        captured["payload"] = payload
        return {"operation": {"operation_key": "offline:op:1"}}

    monkeypatch.setattr(machine_state, "_orchestra_admin_post", _fake_post)

    machine_state.update_task_outbound_operation_record(
        machine_state.TaskOutboundOperationReference(
            assistant_id="42",
            operation_key="offline:op:1",
        ),
        {
            "status": "completed",
            "provider_message_id": "msg-123",
            "history_message_id": 9,
        },
    )

    payload = captured["payload"]
    assert isinstance(payload, dict)
    assert payload["assistant_id"] == "42"
    assert payload["operation_key"] == "offline:op:1"
    assert payload["updates"] == {
        "status": "completed",
        "provider_message_id": "msg-123",
        "history_message_id": 9,
    }
