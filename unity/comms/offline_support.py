"""Offline comms durability for headless task execution.

When assistant-owned comms run inside the live ConversationManager, the normal
event and transcript machinery already captures what happened. The offline task
lane is different: a stored function runs headlessly through
`task_scheduler.offline_runner` with no live ConversationManager session around
it.

This module provides the missing bookkeeping for that headless path:

1. detect whether the current process is an offline task run
2. reserve one durable outbound-operation row before a real-world send
3. dedupe retries so replayed jobs do not send the same message twice
4. write assistant-visible history for the outcome
5. finalize the durable `Tasks/OutboundOperations` record as completed,
   failed, or one of the pending fallback states

`offline_runner.py` owns executing the stored function itself. This module owns
the extra comms-specific durability and provenance needed when that function
sends messages while running without a live ConversationManager.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import os
from typing import Any, Mapping

from unity.conversation_manager.cm_types import Medium
from unity.logger import LOGGER
from unity.manager_registry import ManagerRegistry
from unity.session_details import SESSION_DETAILS
from unity.task_scheduler.machine_state import (
    TaskOutboundOperationProvenance,
    TaskOutboundOperationReference,
    create_or_adopt_task_outbound_operation,
    update_task_outbound_operation_record,
)

_OPERATION_COUNTER = 0


def offline_tracking_enabled() -> bool:
    """Return whether the current process should use offline outbound tracking."""

    return bool(os.environ.get("UNITY_OFFLINE_TASK_RUN_KEY", "").strip())


def _now_iso() -> str:
    """Return the current UTC timestamp in ISO-8601 format."""

    return datetime.now(timezone.utc).isoformat()


def _next_operation_index() -> int:
    """Return the next stable per-process outbound operation ordinal."""

    global _OPERATION_COUNTER
    _OPERATION_COUNTER += 1
    return _OPERATION_COUNTER


def _optional_int_env(name: str) -> int | None:
    """Return one optional integer env var, or None when unset/invalid."""

    value = os.environ.get(name, "").strip()
    if not value:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _coerce_text(value: Any) -> str:
    """Return a compact string representation for stored metadata fields."""

    if value is None:
        return ""
    return str(value).strip()


def _extract_provider_message_id(response: Mapping[str, Any] | None) -> str | None:
    """Return one best-effort provider-side identifier from a transport response."""

    if not isinstance(response, Mapping):
        return None
    for key in ("message_id", "id", "sid", "call_sid"):
        value = _coerce_text(response.get(key))
        if value:
            return value
    return None


def _dedupe_response_for_existing_operation(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the duplicate-adoption response for one existing ledger row."""

    status = _coerce_text(payload.get("status")) or "pending"
    if status == "completed":
        return {"status": "ok", "deduped": True}
    if status == "pending_resend":
        return {
            "status": "ok",
            "pending_resend": True,
            "deduped": True,
            "note": (
                "This outbound operation already used the WhatsApp fallback flow "
                "during a previous attempt, so it was not sent again."
            ),
        }
    if status == "pending_callback":
        return {
            "status": "ok",
            "pending_callback": True,
            "deduped": True,
            "note": (
                "This outbound operation already sent a WhatsApp call invite "
                "during a previous attempt, so it was not sent again."
            ),
        }
    if status == "failed":
        return {
            "status": "error",
            "error": _coerce_text(payload.get("error"))
            or "This outbound operation already failed earlier.",
            "deduped": True,
        }
    return {
        "status": "error",
        "error": (
            "This offline outbound operation was already reserved by a previous "
            "attempt and will not be sent again."
        ),
        "deduped": True,
    }


def _build_history_content(
    *,
    attempted_content: str,
    success: bool,
    error: str | None = None,
) -> str:
    """Return transcript-visible content for one offline outbound outcome."""

    normalized_attempt = attempted_content.strip()
    if success:
        return normalized_attempt or "<Assistant outbound communication>"
    normalized_error = _coerce_text(error) or "Outbound communication failed."
    if normalized_attempt:
        return f"[Send Failed] {normalized_error}\n\n{normalized_attempt}"
    return f"[Send Failed] {normalized_error}"


def _log_outbound_history(
    *,
    medium: Medium,
    content: str,
    receiver_ids: list[int],
    metadata: Mapping[str, Any],
    attachments: list[dict] | None = None,
) -> tuple[int | None, int | None]:
    """Persist one offline outbound message into assistant history."""

    transcript_manager = ManagerRegistry.get_transcript_manager()
    exchange_metadata = {
        "offline_outbound": True,
        "task_run_key": metadata.get("task_run_key"),
        "operation_key": metadata.get("operation_key"),
        "target_kind": metadata.get("target_kind"),
        "target_metadata": metadata.get("target_metadata") or {},
    }
    exchange_id, message_id = transcript_manager.log_first_message_in_new_exchange(
        {
            "medium": medium,
            "sender_id": SESSION_DETAILS.assistant.contact_id or 0,
            "receiver_ids": receiver_ids or [SESSION_DETAILS.assistant.contact_id or 0],
            "timestamp": datetime.now(timezone.utc),
            "content": content,
            "attachments": list(attachments or []),
            "metadata": dict(metadata),
        },
        exchange_initial_metadata=exchange_metadata,
    )
    return exchange_id, message_id


@dataclass(frozen=True)
class OfflineOutboundReservation:
    """Reserved offline outbound operation state for one send attempt."""

    reference: TaskOutboundOperationReference
    task_run_key: str
    operation_key: str
    medium: Medium
    target_kind: str
    target_metadata: dict[str, Any]


@dataclass(frozen=True)
class OfflineOutboundDecision:
    """Reservation result for one offline outbound operation."""

    reservation: OfflineOutboundReservation | None = None
    response: dict[str, Any] | None = None


def reserve_outbound_operation(
    *,
    method_name: str,
    medium: Medium,
    target_kind: str,
    target_metadata: Mapping[str, Any],
    contact_id: int | None = None,
) -> OfflineOutboundDecision:
    """Reserve one durable outbound operation before transport sends externally."""

    if not offline_tracking_enabled():
        return OfflineOutboundDecision()

    assistant_id = _coerce_text(SESSION_DETAILS.assistant.agent_id)
    task_run_key = _coerce_text(os.environ.get("UNITY_OFFLINE_TASK_RUN_KEY"))
    if not assistant_id or not task_run_key:
        return OfflineOutboundDecision(
            response={
                "status": "error",
                "error": "Offline outbound tracking is not configured correctly.",
            },
        )

    try:
        record = create_or_adopt_task_outbound_operation(
            TaskOutboundOperationProvenance(
                assistant_id=assistant_id,
                task_run_key=task_run_key,
                operation_index=_next_operation_index(),
                method_name=method_name,
                medium=str(medium),
                target_kind=target_kind,
                target_metadata=dict(target_metadata),
                task_id=_optional_int_env("UNITY_OFFLINE_TASK_ID"),
                source_task_log_id=_optional_int_env(
                    "UNITY_OFFLINE_TASK_SOURCE_TASK_LOG_ID",
                ),
                contact_id=contact_id,
            ),
            created_at=_now_iso(),
        )
    except Exception:
        LOGGER.exception("Failed to reserve offline outbound operation.")
        return OfflineOutboundDecision(
            response={
                "status": "error",
                "error": (
                    "Could not reserve a durable offline outbound operation, so "
                    "the communication was not sent."
                ),
            },
        )

    if record is None:
        return OfflineOutboundDecision(
            response={
                "status": "error",
                "error": (
                    "Could not reserve a durable offline outbound operation, so "
                    "the communication was not sent."
                ),
            },
        )

    reservation = OfflineOutboundReservation(
        reference=record.reference,
        task_run_key=task_run_key,
        operation_key=record.reference.operation_key,
        medium=medium,
        target_kind=target_kind,
        target_metadata=dict(target_metadata),
    )
    if record.created:
        return OfflineOutboundDecision(reservation=reservation)
    return OfflineOutboundDecision(
        reservation=reservation,
        response=_dedupe_response_for_existing_operation(record.payload),
    )


def finalize_outbound_operation_success(
    reservation: OfflineOutboundReservation | None,
    *,
    attempted_content: str,
    receiver_ids: list[int] | None,
    target_metadata: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
    attachments: list[dict] | None = None,
    provider_response: Mapping[str, Any] | None = None,
    status: str = "completed",
) -> None:
    """Persist success history and finalize one reserved outbound operation."""

    if reservation is None:
        return

    resolved_target_metadata = dict(target_metadata or reservation.target_metadata)
    history_metadata = {
        "offline_outbound": True,
        "task_run_key": reservation.task_run_key,
        "operation_key": reservation.operation_key,
        "target_kind": reservation.target_kind,
        "target_metadata": resolved_target_metadata,
        "delivery_status": status,
        "provider_message_id": _extract_provider_message_id(provider_response),
    }
    if metadata:
        history_metadata.update(dict(metadata))

    history_exchange_id = None
    history_message_id = None
    try:
        history_exchange_id, history_message_id = _log_outbound_history(
            medium=reservation.medium,
            content=_build_history_content(
                attempted_content=attempted_content,
                success=True,
            ),
            receiver_ids=[
                receiver_id
                for receiver_id in (receiver_ids or [])
                if receiver_id is not None
            ],
            metadata=history_metadata,
            attachments=attachments,
        )
    except Exception:
        LOGGER.exception("Failed to persist offline outbound success into history.")

    update_task_outbound_operation_record(
        reservation.reference,
        {
            "status": status,
            "target_metadata": resolved_target_metadata,
            "provider_message_id": _extract_provider_message_id(provider_response),
            "history_exchange_id": history_exchange_id,
            "history_message_id": history_message_id,
            "updated_at": _now_iso(),
            "completed_at": _now_iso(),
        },
    )


def finalize_outbound_operation_failure(
    reservation: OfflineOutboundReservation | None,
    *,
    error: str,
    attempted_content: str,
    receiver_ids: list[int] | None,
    target_metadata: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
    attachments: list[dict] | None = None,
) -> None:
    """Persist failure history and finalize one reserved outbound operation."""

    if reservation is None:
        return

    resolved_target_metadata = dict(target_metadata or reservation.target_metadata)
    history_metadata = {
        "offline_outbound": True,
        "task_run_key": reservation.task_run_key,
        "operation_key": reservation.operation_key,
        "target_kind": reservation.target_kind,
        "target_metadata": resolved_target_metadata,
        "delivery_status": "failed",
        "error": error,
    }
    if metadata:
        history_metadata.update(dict(metadata))

    history_exchange_id = None
    history_message_id = None
    try:
        history_exchange_id, history_message_id = _log_outbound_history(
            medium=reservation.medium,
            content=_build_history_content(
                attempted_content=attempted_content,
                success=False,
                error=error,
            ),
            receiver_ids=[
                receiver_id
                for receiver_id in (receiver_ids or [])
                if receiver_id is not None
            ],
            metadata=history_metadata,
            attachments=attachments,
        )
    except Exception:
        LOGGER.exception("Failed to persist offline outbound failure into history.")

    update_task_outbound_operation_record(
        reservation.reference,
        {
            "status": "failed",
            "error": error,
            "target_metadata": resolved_target_metadata,
            "history_exchange_id": history_exchange_id,
            "history_message_id": history_message_id,
            "updated_at": _now_iso(),
            "completed_at": _now_iso(),
        },
    )
