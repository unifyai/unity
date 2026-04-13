"""Task-activation wake-reason helpers for conversation-manager handlers."""

import asyncio
import hashlib
from typing import TYPE_CHECKING, Any

import requests

from unity.conversation_manager.cm_types import Medium
from unity.conversation_manager.events import FastBrainNotification, TaskDue
from unity.session_details import SESSION_DETAILS
from unity.task_scheduler.machine_state import (
    TaskActivationSnapshot,
    list_trigger_activations,
    validate_task_due_activation,
)

if TYPE_CHECKING:
    from unity.conversation_manager.conversation_manager import ConversationManager


def _current_task_assistant_id() -> str | None:
    """Return the current assistant id in the string form task state expects."""

    assistant_id = SESSION_DETAILS.assistant.agent_id
    return str(assistant_id) if assistant_id is not None else None


def _task_due_notification_text(event: TaskDue) -> str:
    """Return the slow-brain instruction for a validated due task."""

    return (
        f"Scheduled task {event.task_id} is due now "
        f"(scheduled for {event.scheduled_for}). Start it by calling "
        f"primitives.tasks.execute(task_id={event.task_id})."
    )


def _task_due_fast_brain_context(
    event: TaskDue,
    activation: TaskActivationSnapshot | None,
) -> str:
    """Return the silent fast-brain context for one validated due task."""

    label = (
        activation.task_name
        if activation and activation.task_name
        else f"task {event.task_id}"
    )
    return (
        f"Background context: scheduled {label} is due now "
        f"(task_id={event.task_id}, scheduled_for={event.scheduled_for}). "
        "The slow brain is handling the task wake reason."
    )


def _trigger_candidate_fast_brain_context(
    *,
    medium: Medium,
    sender_name: str,
    candidates: list[TaskActivationSnapshot],
) -> str:
    """Return silent fast-brain context for live trigger candidates."""

    labels = [
        candidate.task_name or f"task {candidate.task_id}"
        for candidate in candidates[:5]
    ]
    if len(candidates) > 5:
        labels.append("...")
    return (
        f"Background context: this {medium.value.replace('_', ' ')} from {sender_name} "
        f"mechanically matches live trigger candidates: {', '.join(labels)}. "
        "The slow brain is deciding whether a trigger truly applies."
    )


def _voice_fast_brain_available(cm: "ConversationManager") -> bool:
    """Return True when a voice fast-brain subprocess exists or is about to start."""

    call_manager = getattr(cm, "call_manager", None)
    if call_manager is None:
        return False
    return bool(
        getattr(cm.mode, "is_voice", False)
        or getattr(call_manager, "has_active_call", False)
        or getattr(call_manager, "has_active_google_meet", False)
        or getattr(call_manager, "_whatsapp_call_joining", False)
        or getattr(call_manager, "_gmeet_joining", False)
        or getattr(call_manager, "_socket_server", None) is not None,
    )


def _append_initial_call_notification(cm: "ConversationManager", content: str) -> None:
    """Append silent task context to the next call-start notification payload."""

    existing = getattr(cm.call_manager, "initial_notification", "") or ""
    if existing:
        cm.call_manager.initial_notification = f"{existing}\n\n{content}"
    else:
        cm.call_manager.initial_notification = content


async def _queue_fast_brain_task_context(
    cm: "ConversationManager",
    *,
    content: str,
    source: str,
    contact: dict | None = None,
) -> None:
    """Send silent task context to the voice fast brain before it speaks."""

    if not content or not _voice_fast_brain_available(cm):
        return
    notification = FastBrainNotification(
        contact=contact or getattr(cm.call_manager, "_disconnect_contact", None) or {},
        content=content,
        should_speak=False,
        source=source,
    )
    socket_server = getattr(cm.call_manager, "_socket_server", None)
    if socket_server is None:
        _append_initial_call_notification(cm, content)
        return
    await socket_server.queue_for_clients(
        "app:call:notification",
        notification.to_json(),
    )


def _task_due_event_from_wake_reason(reason: Any) -> TaskDue | None:
    """Convert one startup wake-reason payload into a typed `TaskDue` event."""

    if not isinstance(reason, dict) or reason.get("type") != "task_due":
        return None
    task_id = reason.get("task_id")
    source_task_log_id = reason.get("source_task_log_id")
    activation_revision = str(reason.get("activation_revision") or "")
    scheduled_for = str(reason.get("scheduled_for") or "")
    if task_id is None or source_task_log_id is None:
        return None
    if not activation_revision or not scheduled_for:
        return None
    try:
        return TaskDue(
            task_id=int(task_id),
            source_task_log_id=int(source_task_log_id),
            activation_revision=activation_revision,
            scheduled_for=scheduled_for,
            execution_mode=str(reason.get("execution_mode") or "live"),
            source_type=str(reason.get("source_type") or "scheduled"),
            reason=f"Scheduled task {task_id} became due.",
        )
    except (TypeError, ValueError):
        return None


async def _handle_task_due_event(event: TaskDue, cm: "ConversationManager") -> bool:
    """Validate and surface one due-task event to the notification bar."""

    assistant_id = _current_task_assistant_id()
    activation, stale_reason = validate_task_due_activation(
        assistant_id=assistant_id,
        task_id=event.task_id,
        activation_revision=event.activation_revision,
        source_task_log_id=event.source_task_log_id,
        scheduled_for=event.scheduled_for,
    )
    if stale_reason is not None:
        cm._session_logger.info(
            "task_due",
            (
                f"Rejected due task {event.task_id}: {stale_reason} "
                f"(assistant_id={assistant_id or '-'})"
            ),
        )
        return False
    cm.notifications_bar.push_notif(
        "Tasks",
        _task_due_notification_text(event),
        event.timestamp,
    )
    cm._session_logger.info(
        "task_due",
        (
            f"Accepted due task {event.task_id} "
            f"(activation_revision={activation.activation_revision or '-'})"
        ),
    )
    await _queue_fast_brain_task_context(
        cm,
        content=_task_due_fast_brain_context(event, activation),
        source="task_due",
    )
    return True


async def _consume_startup_wake_reasons(cm: "ConversationManager") -> None:
    """Replay startup wake reasons once managers are initialized."""

    wake_reasons = list(getattr(cm, "_startup_wake_reasons", []) or [])
    cm._startup_wake_reasons = []
    if not wake_reasons:
        return
    cm._session_logger.info(
        "task_due",
        f"Replaying {len(wake_reasons)} startup wake reason(s)",
    )
    for wake_reason in wake_reasons:
        task_due_event = _task_due_event_from_wake_reason(wake_reason)
        if task_due_event is None:
            cm._session_logger.info(
                "task_due",
                f"Ignoring unparseable startup wake reason: {wake_reason!r}",
            )
            continue
        await _handle_task_due_event(task_due_event, cm)


def _filter_trigger_candidates(
    *,
    medium: Medium,
    contact_id: int | None,
) -> tuple[list[TaskActivationSnapshot], list[TaskActivationSnapshot]]:
    """Return mechanically matching live and offline trigger activations."""

    assistant_id = _current_task_assistant_id()
    candidates = list_trigger_activations(
        assistant_id=assistant_id,
        medium=medium.value,
    )
    matching: list[TaskActivationSnapshot] = []
    for candidate in candidates:
        if contact_id is not None and contact_id in candidate.trigger_omit_contact_ids:
            continue
        if candidate.trigger_from_contact_ids and (
            contact_id is None or contact_id not in candidate.trigger_from_contact_ids
        ):
            continue
        matching.append(candidate)
    live_candidates = [
        candidate for candidate in matching if candidate.execution_mode == "live"
    ]
    offline_candidates = [
        candidate for candidate in matching if candidate.execution_mode == "offline"
    ]
    return live_candidates, offline_candidates


def _trigger_candidate_notification_text(
    *,
    medium: Medium,
    sender_name: str,
    candidates: list[TaskActivationSnapshot],
) -> str:
    """Return the slow-brain instruction for mechanically matched trigger tasks."""

    candidate_labels = [
        f"{candidate.task_id} ({'interrupt' if candidate.interrupt else 'non-interrupt'})"
        for candidate in candidates[:5]
    ]
    if len(candidates) > 5:
        candidate_labels.append("...")
    task_word = "task" if len(candidates) == 1 else "tasks"
    return (
        f"This inbound {medium.value.replace('_', ' ')} from {sender_name} "
        f"mechanically matches trigger criteria for {task_word} "
        f"{', '.join(candidate_labels)}. Decide whether this communication truly "
        "satisfies any candidate trigger based on the task descriptions and this "
        "inbound. If yes, immediately start the matching task with "
        "primitives.tasks.execute(task_id=<task_id>) and treat this inbound as the "
        "triggering communication."
    )


def _build_trigger_source_ref(
    *,
    event: Any,
    medium: Medium,
    contact_id: int | None,
) -> str:
    """Return a stable idempotency key fragment for one inbound trigger event."""

    explicit_ref = (
        getattr(event, "api_message_id", None)
        or getattr(event, "email_id", None)
        or getattr(event, "conference_name", None)
        or getattr(event, "room_name", None)
        or getattr(event, "meet_url", None)
    )
    if explicit_ref:
        return str(explicit_ref)
    content = (
        getattr(event, "content", None)
        or getattr(event, "body", None)
        or getattr(event, "subject", None)
        or ""
    )
    digest = hashlib.sha256(str(content).encode("utf-8")).hexdigest()[:12]
    timestamp = getattr(event, "timestamp", None)
    timestamp_component = timestamp.isoformat() if timestamp is not None else "unknown"
    contact_component = str(contact_id) if contact_id is not None else "unknown"
    return (
        f"{event.__class__.__name__}:{medium.value}:{contact_component}:"
        f"{timestamp_component}:{digest}"
    )


def _dispatch_offline_trigger_candidate(
    *,
    candidate: TaskActivationSnapshot,
    event: Any,
    medium: Medium,
    contact_id: int | None,
) -> dict[str, Any]:
    """Ask Communication to execute one offline trigger candidate headlessly."""

    from unity.settings import SETTINGS

    comms_url = (SETTINGS.conversation.COMMS_URL or "").rstrip("/")
    admin_key = SETTINGS.ORCHESTRA_ADMIN_KEY.get_secret_value()
    if not comms_url:
        raise RuntimeError("UNITY_COMMS_URL is not configured")
    if not admin_key:
        raise RuntimeError("ORCHESTRA_ADMIN_KEY is not configured")
    assistant_id = candidate.assistant_id or (_current_task_assistant_id() or "")
    response = requests.post(
        f"{comms_url}/infra/task-activation/offline-dispatch",
        json={
            "assistant_id": assistant_id,
            "task_id": candidate.task_id,
            "source_task_log_id": candidate.source_task_log_id,
            "activation_revision": candidate.activation_revision,
            "execution_mode": "offline",
            "source_type": "triggered",
            "source_ref": _build_trigger_source_ref(
                event=event,
                medium=medium,
                contact_id=contact_id,
            ),
            "source_medium": medium.value,
            "source_contact_id": contact_id,
        },
        headers={"Authorization": f"Bearer {admin_key}"},
        timeout=15,
    )
    response.raise_for_status()
    return response.json()


async def _surface_trigger_task_candidates(
    *,
    cm: "ConversationManager",
    event: Any,
    medium: Medium,
    contact_id: int | None,
    sender_name: str,
    timestamp: Any,
) -> bool:
    """Push one trigger-candidate notification when any live candidates match."""

    live_candidates, offline_candidates = _filter_trigger_candidates(
        medium=medium,
        contact_id=contact_id,
    )
    if offline_candidates:
        offline_statuses: list[str] = []
        for candidate in offline_candidates:
            try:
                result = await asyncio.to_thread(
                    _dispatch_offline_trigger_candidate,
                    candidate=candidate,
                    event=event,
                    medium=medium,
                    contact_id=contact_id,
                )
                offline_statuses.append(
                    f"{candidate.task_id}:{result.get('status', 'unknown')}",
                )
            except Exception as exc:
                offline_statuses.append(f"{candidate.task_id}:error")
                cm._session_logger.info(
                    "task_trigger",
                    (
                        f"Offline trigger dispatch failed for task {candidate.task_id} "
                        f"(medium={medium.value}, contact_id={contact_id}): {exc}"
                    ),
                )
        if offline_statuses:
            cm._session_logger.info(
                "task_trigger",
                (
                    f"Offline trigger candidates dispatched for medium={medium.value} "
                    f"contact_id={contact_id}: {', '.join(offline_statuses)}"
                ),
            )
    if not live_candidates:
        return False
    candidate_ids = [candidate.task_id for candidate in live_candidates]
    cm.notifications_bar.push_notif(
        "Tasks",
        _trigger_candidate_notification_text(
            medium=medium,
            sender_name=sender_name,
            candidates=live_candidates,
        ),
        timestamp,
    )
    cm._session_logger.info(
        "task_trigger",
        (
            f"Matched trigger candidates {candidate_ids} for medium={medium.value} "
            f"contact_id={contact_id}"
        ),
    )
    await _queue_fast_brain_task_context(
        cm,
        content=_trigger_candidate_fast_brain_context(
            medium=medium,
            sender_name=sender_name,
            candidates=live_candidates,
        ),
        source="task_trigger",
        contact=getattr(event, "contact", None),
    )
    return True
