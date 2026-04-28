from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from unity.comms import offline_support
from unity.comms.primitives import CommsPrimitives
from unity.conversation_manager.cm_types import Medium
from unity.conversation_manager.domains import comms_utils
from unity.task_scheduler.machine_state import (
    TaskOutboundOperationRecord,
    TaskOutboundOperationReference,
)


def _seed_offline_env(monkeypatch):
    monkeypatch.setenv(
        "UNITY_OFFLINE_TASK_RUN_KEY",
        "offline:scheduled:42:101:rev:once",
    )
    monkeypatch.setenv("UNITY_OFFLINE_TASK_ID", "101")
    monkeypatch.setenv("UNITY_OFFLINE_TASK_SOURCE_TASK_LOG_ID", "555")
    monkeypatch.setattr(offline_support, "_OPERATION_COUNTER", 0)
    monkeypatch.setattr(offline_support.SESSION_DETAILS.assistant, "agent_id", 42)
    monkeypatch.setattr(offline_support.SESSION_DETAILS.assistant, "contact_id", 0)
    monkeypatch.setattr(
        offline_support.SESSION_DETAILS.assistant,
        "number",
        "+15555550000",
    )
    monkeypatch.setattr(
        offline_support.SESSION_DETAILS.assistant,
        "email",
        "assistant@test.com",
    )
    monkeypatch.setattr(
        offline_support.SESSION_DETAILS.assistant,
        "email_provider",
        "google_workspace",
    )
    monkeypatch.setattr(
        offline_support.SESSION_DETAILS.assistant,
        "whatsapp_number",
        "+15555550001",
    )
    monkeypatch.setattr(
        offline_support.SESSION_DETAILS.assistant,
        "first_name",
        "Offline",
        raising=False,
    )


def _stub_offline_tracking(monkeypatch, *, operation_key: str):
    updated_records: list[tuple[TaskOutboundOperationReference, dict]] = []
    transcript_calls: list[tuple[dict, dict]] = []

    monkeypatch.setattr(
        offline_support,
        "create_or_adopt_task_outbound_operation",
        lambda provenance, created_at=None: TaskOutboundOperationRecord(
            reference=TaskOutboundOperationReference(
                assistant_id="42",
                operation_key=operation_key,
            ),
            payload={"status": "pending"},
            created=True,
        ),
    )
    monkeypatch.setattr(
        offline_support,
        "update_task_outbound_operation_record",
        lambda reference, payload: updated_records.append((reference, payload)),
    )
    monkeypatch.setattr(
        offline_support,
        "ManagerRegistry",
        SimpleNamespace(
            get_transcript_manager=lambda: SimpleNamespace(
                log_first_message_in_new_exchange=lambda message, *, exchange_initial_metadata=None: (
                    transcript_calls.append((message, exchange_initial_metadata or {}))
                    or (77, 88)
                ),
            ),
        ),
    )
    return updated_records, transcript_calls


def test_reserve_outbound_operation_dedupes_completed_row(monkeypatch):
    _seed_offline_env(monkeypatch)

    monkeypatch.setattr(
        offline_support,
        "create_or_adopt_task_outbound_operation",
        lambda provenance, created_at=None: TaskOutboundOperationRecord(
            reference=TaskOutboundOperationReference(
                assistant_id="42",
                operation_key="offline:scheduled:42:101:rev:once:op-1",
            ),
            payload={"status": "completed"},
            created=False,
        ),
    )

    decision = offline_support.reserve_outbound_operation(
        method_name="send_sms",
        medium=Medium.SMS_MESSAGE,
        target_kind="contact",
        target_metadata={"contact_id": 7, "phone_number": "+15555550123"},
        contact_id=7,
    )

    assert decision.reservation is not None
    assert decision.response == {"status": "ok", "deduped": True}


def test_finalize_outbound_operation_success_logs_history_and_updates_ledger(
    monkeypatch,
):
    _seed_offline_env(monkeypatch)

    fake_transcript_manager = SimpleNamespace()
    transcript_calls: list[tuple[dict, dict]] = []

    def _log_first_message_in_new_exchange(message, *, exchange_initial_metadata=None):
        transcript_calls.append((message, exchange_initial_metadata or {}))
        return (11, 22)

    fake_transcript_manager.log_first_message_in_new_exchange = (
        _log_first_message_in_new_exchange
    )
    monkeypatch.setattr(
        offline_support,
        "ManagerRegistry",
        SimpleNamespace(get_transcript_manager=lambda: fake_transcript_manager),
    )

    updates: list[tuple[TaskOutboundOperationReference, dict]] = []
    monkeypatch.setattr(
        offline_support,
        "update_task_outbound_operation_record",
        lambda reference, payload: updates.append((reference, payload)),
    )

    reservation = offline_support.OfflineOutboundReservation(
        reference=TaskOutboundOperationReference(
            assistant_id="42",
            operation_key="op-1",
        ),
        task_run_key="offline:scheduled:42:101:rev:once",
        operation_key="op-1",
        medium=Medium.EMAIL,
        target_kind="email",
        target_metadata={"to": ["alice@example.com"]},
    )

    offline_support.finalize_outbound_operation_success(
        reservation,
        attempted_content="Subject: Hello\n\nBody",
        receiver_ids=[7],
        target_metadata={"to": ["alice@example.com"]},
        metadata={"reply_to_email_id": "msg-1"},
        attachments=[{"id": "att-1", "filename": "brief.txt"}],
        provider_response={"id": "provider-1"},
    )

    assert transcript_calls
    message, exchange_metadata = transcript_calls[0]
    assert message["medium"] == Medium.EMAIL
    assert message["sender_id"] == 0
    assert message["receiver_ids"] == [7]
    assert message["content"] == "Subject: Hello\n\nBody"
    assert message["metadata"]["delivery_status"] == "completed"
    assert message["metadata"]["provider_message_id"] == "provider-1"
    assert message["metadata"]["task_run_key"] == "offline:scheduled:42:101:rev:once"
    assert exchange_metadata["offline_outbound"] is True
    assert exchange_metadata["operation_key"] == "op-1"

    assert updates
    reference, payload = updates[0]
    assert reference.operation_key == "op-1"
    assert payload["status"] == "completed"
    assert payload["provider_message_id"] == "provider-1"
    assert payload["history_exchange_id"] == 11
    assert payload["history_message_id"] == 22


def test_finalize_outbound_operation_failure_logs_history_and_updates_ledger(
    monkeypatch,
):
    _seed_offline_env(monkeypatch)

    fake_transcript_manager = SimpleNamespace()
    transcript_calls: list[tuple[dict, dict]] = []

    def _log_first_message_in_new_exchange(message, *, exchange_initial_metadata=None):
        transcript_calls.append((message, exchange_initial_metadata or {}))
        return (55, 66)

    fake_transcript_manager.log_first_message_in_new_exchange = (
        _log_first_message_in_new_exchange
    )
    monkeypatch.setattr(
        offline_support,
        "ManagerRegistry",
        SimpleNamespace(get_transcript_manager=lambda: fake_transcript_manager),
    )

    updates: list[tuple[TaskOutboundOperationReference, dict]] = []
    monkeypatch.setattr(
        offline_support,
        "update_task_outbound_operation_record",
        lambda reference, payload: updates.append((reference, payload)),
    )

    reservation = offline_support.OfflineOutboundReservation(
        reference=TaskOutboundOperationReference(
            assistant_id="42",
            operation_key="op-2",
        ),
        task_run_key="offline:scheduled:42:101:rev:once",
        operation_key="op-2",
        medium=Medium.SMS_MESSAGE,
        target_kind="contact",
        target_metadata={"contact_id": 7, "phone_number": "+15555550123"},
    )

    offline_support.finalize_outbound_operation_failure(
        reservation,
        error="Provider rejected the send.",
        attempted_content="Hello from offline",
        receiver_ids=[7],
        target_metadata={"contact_id": 7, "phone_number": "+15555550123"},
        metadata={"contact_display_name": "Alice Owner"},
    )

    assert transcript_calls
    message, exchange_metadata = transcript_calls[0]
    assert message["medium"] == Medium.SMS_MESSAGE
    assert message["receiver_ids"] == [7]
    assert message["content"].startswith("[Send Failed] Provider rejected the send.")
    assert message["metadata"]["delivery_status"] == "failed"
    assert message["metadata"]["error"] == "Provider rejected the send."
    assert exchange_metadata["operation_key"] == "op-2"

    assert updates
    reference, payload = updates[0]
    assert reference.operation_key == "op-2"
    assert payload["status"] == "failed"
    assert payload["error"] == "Provider rejected the send."
    assert payload["history_exchange_id"] == 55
    assert payload["history_message_id"] == 66


@pytest.mark.anyio
async def test_send_sms_offline_success_reserves_and_finalizes(monkeypatch):
    _seed_offline_env(monkeypatch)

    created_records: list[tuple[object, str | None]] = []
    updated_records: list[tuple[TaskOutboundOperationReference, dict]] = []
    transcript_calls: list[tuple[dict, dict]] = []

    monkeypatch.setattr(
        offline_support,
        "create_or_adopt_task_outbound_operation",
        lambda provenance, created_at=None: (
            created_records.append((provenance, created_at))
            or TaskOutboundOperationRecord(
                reference=TaskOutboundOperationReference(
                    assistant_id="42",
                    operation_key="op-1",
                ),
                payload={"status": "pending"},
                created=True,
            )
        ),
    )
    monkeypatch.setattr(
        offline_support,
        "update_task_outbound_operation_record",
        lambda reference, payload: updated_records.append((reference, payload)),
    )
    monkeypatch.setattr(
        offline_support,
        "ManagerRegistry",
        SimpleNamespace(
            get_transcript_manager=lambda: SimpleNamespace(
                log_first_message_in_new_exchange=lambda message, *, exchange_initial_metadata=None: (
                    transcript_calls.append((message, exchange_initial_metadata or {}))
                    or (33, 44)
                ),
            ),
        ),
    )

    comms = CommsPrimitives()
    comms._get_contact = lambda **kwargs: {
        "contact_id": 5,
        "first_name": "Alice",
        "surname": "Owner",
        "phone_number": "+15555550123",
        "should_respond": True,
    }
    comms._contact_manager = lambda: SimpleNamespace(
        should_respond_to=lambda _cid: True,
    )
    comms._event_broker.publish = AsyncMock()

    async def _fake_send_sms_message_via_number(*, to_number: str, content: str):
        assert to_number == "+15555550123"
        assert content == "Hello from offline"
        return {"success": True, "sid": "sms-1"}

    monkeypatch.setattr(
        comms_utils,
        "send_sms_message_via_number",
        _fake_send_sms_message_via_number,
    )

    result = await comms.send_sms(contact_id=5, content="Hello from offline")

    assert result == {"status": "ok"}
    assert created_records
    assert transcript_calls
    assert updated_records
    assert updated_records[0][1]["status"] == "completed"
    assert updated_records[0][1]["provider_message_id"] == "sms-1"
    assert transcript_calls[0][0]["metadata"]["delivery_status"] == "completed"


@pytest.mark.anyio
async def test_send_sms_offline_reservation_uses_normalized_phone_number(monkeypatch):
    _seed_offline_env(monkeypatch)

    created_records: list[tuple[object, str | None]] = []
    updated_records: list[tuple[TaskOutboundOperationReference, dict]] = []
    transcript_calls: list[tuple[dict, dict]] = []
    contact_state = {
        "contact_id": 5,
        "first_name": "Alice",
        "surname": "Owner",
        "should_respond": True,
    }
    update_calls: list[tuple[int, dict]] = []

    monkeypatch.setattr(
        offline_support,
        "create_or_adopt_task_outbound_operation",
        lambda provenance, created_at=None: (
            created_records.append((provenance, created_at))
            or TaskOutboundOperationRecord(
                reference=TaskOutboundOperationReference(
                    assistant_id="42",
                    operation_key="op-normalized-sms",
                ),
                payload={"status": "pending"},
                created=True,
            )
        ),
    )
    monkeypatch.setattr(
        offline_support,
        "update_task_outbound_operation_record",
        lambda reference, payload: updated_records.append((reference, payload)),
    )
    monkeypatch.setattr(
        offline_support,
        "ManagerRegistry",
        SimpleNamespace(
            get_transcript_manager=lambda: SimpleNamespace(
                log_first_message_in_new_exchange=lambda message, *, exchange_initial_metadata=None: (
                    transcript_calls.append((message, exchange_initial_metadata or {}))
                    or (33, 44)
                ),
            ),
        ),
    )

    comms = CommsPrimitives()

    def _fake_get_contact(**kwargs):
        if kwargs.get("contact_id") == 5:
            return dict(contact_state)
        phone_number = kwargs.get("phone_number")
        if phone_number and contact_state.get("phone_number") == phone_number:
            return dict(contact_state)
        return None

    def _fake_update_contact(*, contact_id: int, **kwargs):
        update_calls.append((contact_id, dict(kwargs)))
        contact_state.update(kwargs)

    comms._get_contact = _fake_get_contact
    comms._contact_manager = lambda: SimpleNamespace(
        update_contact=_fake_update_contact,
        should_respond_to=lambda _cid: True,
    )
    comms._event_broker.publish = AsyncMock()

    async def _fake_send_sms_message_via_number(*, to_number: str, content: str):
        assert to_number == "+15555550123"
        assert content == "Hello from offline"
        return {"success": True, "sid": "sms-1"}

    monkeypatch.setattr(
        comms_utils,
        "send_sms_message_via_number",
        _fake_send_sms_message_via_number,
    )

    result = await comms.send_sms(
        contact_id=5,
        content="Hello from offline",
        phone_number="+15555550123",
    )

    assert result == {"status": "ok"}
    assert update_calls == [(5, {"phone_number": "+15555550123"})]
    assert created_records
    provenance, _created_at = created_records[0]
    assert provenance.target_metadata == {
        "contact_id": 5,
        "phone_number": "+15555550123",
    }
    assert updated_records
    assert transcript_calls


@pytest.mark.anyio
async def test_send_sms_offline_duplicate_skips_transport(monkeypatch):
    _seed_offline_env(monkeypatch)

    comms = CommsPrimitives()
    comms._get_contact = lambda **kwargs: {
        "contact_id": 5,
        "phone_number": "+15555550123",
        "should_respond": True,
    }
    comms._contact_manager = lambda: SimpleNamespace(
        should_respond_to=lambda _cid: True,
    )
    comms._event_broker.publish = AsyncMock()

    monkeypatch.setattr(
        "unity.comms.primitives.reserve_outbound_operation",
        lambda **kwargs: offline_support.OfflineOutboundDecision(
            reservation=None,
            response={"status": "ok", "deduped": True},
        ),
    )

    send_mock = AsyncMock()
    monkeypatch.setattr(comms_utils, "send_sms_message_via_number", send_mock)

    result = await comms.send_sms(contact_id=5, content="Hello from offline")

    assert result == {"status": "ok", "deduped": True}
    send_mock.assert_not_awaited()


@pytest.mark.anyio
async def test_send_email_missing_assistant_email_does_not_reserve_or_attach(
    monkeypatch,
):
    comms = CommsPrimitives()
    comms._event_broker.publish = AsyncMock()
    comms._contact_manager = lambda: SimpleNamespace(
        update_contact=lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("contact details should not be attached"),
        ),
    )
    monkeypatch.setattr(comms, "_assistant_email", lambda: "")
    monkeypatch.setattr(
        "unity.comms.primitives.reserve_outbound_operation",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("offline reservation should not happen"),
        ),
    )
    send_mock = AsyncMock()
    monkeypatch.setattr(comms_utils, "send_email_via_address", send_mock)

    result = await comms.send_email(
        to=[{"contact_id": 5, "email_address": "alice@example.com"}],
        subject="Hello",
        body="Email body",
    )

    assert result["status"] == "error"
    assert "email address" in result["error"]
    send_mock.assert_not_awaited()


@pytest.mark.anyio
async def test_send_discord_channel_missing_bot_does_not_reserve_or_send(monkeypatch):
    comms = CommsPrimitives()
    comms._event_broker.publish = AsyncMock()
    monkeypatch.setattr(comms, "_assistant_discord_bot_id", lambda: "")
    monkeypatch.setattr(
        comms,
        "_assistant_anchor_contact",
        lambda: {
            "contact_id": 0,
            "first_name": "Offline",
            "surname": "Assistant",
        },
    )
    monkeypatch.setattr(
        "unity.comms.primitives.reserve_outbound_operation",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("offline reservation should not happen"),
        ),
    )
    send_mock = AsyncMock()
    monkeypatch.setattr(comms_utils, "send_discord_message", send_mock)

    result = await comms.send_discord_channel_message(
        channel_id="123456",
        content="Hello Discord",
    )

    assert result["status"] == "error"
    assert "Discord is not enabled" in result["error"]
    send_mock.assert_not_awaited()


@pytest.mark.anyio
async def test_send_whatsapp_template_offline_does_not_claim_pending_resend(
    monkeypatch,
):
    _seed_offline_env(monkeypatch)
    updated_records, transcript_calls = _stub_offline_tracking(
        monkeypatch,
        operation_key="op-wa-template",
    )

    comms = CommsPrimitives()
    comms._get_contact = lambda **kwargs: {
        "contact_id": 5,
        "first_name": "Alice",
        "surname": "Owner",
        "whatsapp_number": "+15555550123",
        "should_respond": True,
    }
    comms._contact_manager = lambda: SimpleNamespace(
        should_respond_to=lambda _cid: True,
    )
    comms._event_broker.publish = AsyncMock()

    async def _fake_send_whatsapp_message(**kwargs):
        assert kwargs["to_number"] == "+15555550123"
        assert kwargs["content"] == "Hello from offline"
        return {"success": True, "method": "template", "sid": "wa-1"}

    monkeypatch.setattr(
        comms_utils,
        "send_whatsapp_message",
        _fake_send_whatsapp_message,
    )

    result = await comms.send_whatsapp(contact_id=5, content="Hello from offline")

    assert result["status"] == "ok"
    assert "pending_resend" not in result
    assert "not queued for automatic resend" in result["note"]
    assert transcript_calls
    assert updated_records
    assert updated_records[0][1]["status"] == "completed"


@pytest.mark.anyio
async def test_make_whatsapp_call_invite_offline_does_not_claim_pending_callback(
    monkeypatch,
):
    _seed_offline_env(monkeypatch)
    updated_records, transcript_calls = _stub_offline_tracking(
        monkeypatch,
        operation_key="op-wa-call",
    )

    comms = CommsPrimitives()
    comms._get_contact = lambda **kwargs: {
        "contact_id": 5,
        "first_name": "Alice",
        "surname": "Owner",
        "whatsapp_number": "+15555550123",
        "should_respond": True,
    }
    comms._contact_manager = lambda: SimpleNamespace(
        should_respond_to=lambda _cid: True,
    )
    comms._event_broker.publish = AsyncMock()

    monkeypatch.setattr(
        "unity.conversation_manager.domains.call_manager.make_room_name",
        lambda assistant_id, medium: f"{assistant_id}-{medium}",
    )

    async def _fake_start_whatsapp_call(**kwargs):
        assert kwargs["to_number"] == "+15555550123"
        return {"success": True, "method": "invite", "call_sid": "wa-call-1"}

    monkeypatch.setattr(
        comms_utils,
        "start_whatsapp_call",
        _fake_start_whatsapp_call,
    )

    result = await comms.make_whatsapp_call(
        contact_id=5,
        context="Ask whether Alice is free to chat now.",
    )

    assert result["status"] == "ok"
    assert "pending_callback" not in result
    assert "not queued with your briefing context" in result["note"]
    assert transcript_calls
    assert updated_records
    assert updated_records[0][1]["status"] == "completed"
