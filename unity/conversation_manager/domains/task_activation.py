"""Task-activation wake-reason helpers for conversation-manager handlers."""

import asyncio
import hashlib
import uuid
from typing import TYPE_CHECKING, Any

import requests

from unity.conversation_manager.cm_types import Medium
from unity.conversation_manager.events import FastBrainNotification, TaskDue
from unity.session_details import SESSION_DETAILS
from unity.task_scheduler.machine_state import (
    TaskActivationSnapshot,
    TaskRunProvenance,
    list_trigger_activations,
    remember_live_task_run_provenance,
    validate_task_due_activation,
)

if TYPE_CHECKING:
    from unity.conversation_manager.conversation_manager import ConversationManager

_TASK_CONTEXT_SUMMARY_MAX_CHARS = 220
_TRIGGER_CONTEXT_CANDIDATE_LIMIT = 3


def _current_task_assistant_id() -> str | None:
    """Return the current assistant id in the string form task state expects."""

    assistant_id = SESSION_DETAILS.assistant.agent_id
    return str(assistant_id) if assistant_id is not None else None


def _compact_task_text(text: str | None, *, fallback: str) -> str:
    """Return one compact human-readable task summary line."""

    candidate = " ".join(str(text or "").split())
    if not candidate:
        candidate = " ".join(fallback.split())
    candidate = candidate.rstrip(" .")
    if len(candidate) <= _TASK_CONTEXT_SUMMARY_MAX_CHARS:
        return candidate
    truncated = candidate[: _TASK_CONTEXT_SUMMARY_MAX_CHARS - 3].rstrip(" ,.;:")
    return f"{truncated}..."


def _sender_display_name(sender_name: str, *, contact_id: int | None) -> str | None:
    """Return a human-readable sender label when the trigger has a known contact."""

    if contact_id is None:
        return None
    candidate = " ".join(str(sender_name or "").split())
    if not candidate or candidate.lower() == "unknown":
        return None
    return candidate


def _task_due_label(
    event: TaskDue,
    activation: TaskActivationSnapshot | None,
) -> str:
    """Return the human-facing label for one due-task wake."""

    if event.task_label:
        return event.task_label
    if activation and activation.task_name:
        return activation.task_name
    return f"task {event.task_id}"


def _task_due_summary(
    event: TaskDue,
    activation: TaskActivationSnapshot | None,
) -> str:
    """Return one compact summary for one due-task wake."""

    label = _task_due_label(event, activation)
    activation_description = (
        activation.task_description
        if activation and activation.task_description
        else ""
    )
    return _compact_task_text(
        event.task_summary or activation_description,
        fallback=label,
    )


def _task_due_recurrence_hint(
    event: TaskDue,
    activation: TaskActivationSnapshot | None,
) -> str:
    """Return whether the due task should be treated as recurring or one-off."""

    if activation and isinstance(activation.repeat, list) and activation.repeat:
        return "recurring"
    return event.recurrence_hint or "one_off"


def _task_due_notification_text(
    event: TaskDue,
    activation: TaskActivationSnapshot | None,
) -> str:
    """Return the slow-brain instruction for a validated due task."""

    label = _task_due_label(event, activation)
    summary = _task_due_summary(event, activation)
    parts = [f"Scheduled task due now: '{label}'."]
    if summary and summary != label:
        parts.append(f"Summary: {summary}.")
    parts.append(f"Due time: {event.scheduled_for}.")
    if _task_due_recurrence_hint(event, activation) == "recurring":
        parts.append("This is a recurring task.")
    if event.visibility_policy == "silent_by_default":
        parts.append(
            "Default behavior: work silently unless you genuinely need the user.",
        )
    parts.append(
        f"If the task still applies, start it with primitives.tasks.execute(task_id={event.task_id}).",
    )
    return " ".join(parts)


def _task_due_fast_brain_context(
    event: TaskDue,
    activation: TaskActivationSnapshot | None,
) -> str:
    """Return the silent fast-brain context for one validated due task."""

    label = _task_due_label(event, activation)
    summary = _task_due_summary(event, activation)
    parts = [f"Background context: the scheduled task '{label}' is due now."]
    if summary and summary != label:
        parts.append(f"Summary: {summary}.")
    if _task_due_recurrence_hint(event, activation) == "recurring":
        parts.append("This is a recurring task.")
    if event.visibility_policy == "silent_by_default":
        parts.append("Default is silent action unless the user is needed.")
    parts.append("The slow brain is handling the wake reason.")
    return " ".join(parts)


def _activation_label(candidate: TaskActivationSnapshot) -> str:
    """Return one human-facing label for a trigger candidate."""

    return candidate.task_name or f"task {candidate.task_id}"


def _activation_summary(candidate: TaskActivationSnapshot) -> str:
    """Return one compact summary for a trigger candidate."""

    label = _activation_label(candidate)
    return _compact_task_text(candidate.task_description, fallback=label)


def _describe_trigger_candidate(candidate: TaskActivationSnapshot) -> str:
    """Render one live trigger candidate for slow-brain review."""

    label = _activation_label(candidate)
    summary = _activation_summary(candidate)
    urgency = "interrupting" if candidate.interrupt else "non-interrupting"
    if summary and summary != label:
        return f"'{label}' ({urgency}): {summary}"
    return f"'{label}' ({urgency})"


def _trigger_candidate_fast_brain_context(
    *,
    medium: Medium,
    sender_name: str,
    candidates: list[TaskActivationSnapshot],
) -> str:
    """Return silent fast-brain context for live trigger candidates."""

    candidate_descriptions = []
    for candidate in candidates[:_TRIGGER_CONTEXT_CANDIDATE_LIMIT]:
        label = _activation_label(candidate)
        summary = _activation_summary(candidate)
        if summary and summary != label:
            candidate_descriptions.append(f"'{label}' ({summary})")
        else:
            candidate_descriptions.append(f"'{label}'")
    if len(candidates) > _TRIGGER_CONTEXT_CANDIDATE_LIMIT:
        candidate_descriptions.append("...")
    candidate_text = "; ".join(candidate_descriptions)
    return (
        f"Background context: this {medium.value.replace('_', ' ')} from {sender_name} "
        f"may relate to live trigger candidates {candidate_text}. "
        "The slow brain is still deciding whether the trigger truly applies. "
        "Do not mention the task unless it naturally helps the conversation."
    )


def _build_trigger_execute_call(*, task_id: int, attempt_token: str) -> str:
    """Return the exact execute call the slow brain should use for one trigger."""

    return (
        "primitives.tasks.execute("
        f'task_id={task_id}, trigger_attempt_token="{attempt_token}"'
        ")"
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
        or getattr(call_manager, "has_active_teams_meet", False)
        or getattr(call_manager, "_whatsapp_call_joining", False)
        or getattr(call_manager, "_meet_joining", False)
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
            task_label=str(reason.get("task_label") or ""),
            task_summary=str(reason.get("task_summary") or ""),
            visibility_policy=str(
                reason.get("visibility_policy") or "silent_by_default",
            ),
            recurrence_hint=str(reason.get("recurrence_hint") or "one_off"),
            reason=(
                f"Scheduled task '{reason['task_label']}' became due."
                if reason.get("task_label")
                else f"Scheduled task {task_id} became due."
            ),
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
    assistant_id_for_run = assistant_id or activation.assistant_id or ""
    if assistant_id_for_run:
        remember_live_task_run_provenance(
            TaskRunProvenance(
                assistant_id=assistant_id_for_run,
                task_id=event.task_id,
                source_type="scheduled",
                execution_mode="live",
                source_task_log_id=event.source_task_log_id,
                activation_revision=event.activation_revision,
                scheduled_for=event.scheduled_for,
                task_name=(activation.task_name if activation is not None else None),
                task_description=(
                    activation.task_description if activation is not None else None
                ),
            ),
        )
    cm.notifications_bar.push_notif(
        "Tasks",
        _task_due_notification_text(event, activation),
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
    candidates: list[tuple[TaskActivationSnapshot, str]],
) -> str:
    """Return the slow-brain instruction for mechanically matched trigger tasks."""

    candidate_labels = [
        _describe_trigger_candidate(candidate)
        for candidate, _attempt_token in candidates[:_TRIGGER_CONTEXT_CANDIDATE_LIMIT]
    ]
    execute_calls = [
        f"{candidate.task_id} -> {_build_trigger_execute_call(task_id=candidate.task_id, attempt_token=attempt_token)}"
        for candidate, attempt_token in candidates[:_TRIGGER_CONTEXT_CANDIDATE_LIMIT]
    ]
    if len(candidates) > _TRIGGER_CONTEXT_CANDIDATE_LIMIT:
        candidate_labels.append("...")
        execute_calls.append("...")
    return (
        f"This inbound {medium.value.replace('_', ' ')} from {sender_name} "
        "mechanically matched live trigger candidates. "
        f"Candidates: {'; '.join(candidate_labels)}. "
        "Why they matched: the inbound medium and sender fit these trigger filters. "
        "Semantic judgement is still pending. Decide whether this communication "
        "truly satisfies any candidate based on the task summaries and the inbound "
        "itself. If yes, immediately start the best match with its exact execute "
        f"call so the triggering inbound stays attached: {'; '.join(execute_calls)}."
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
    sender_name: str,
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
            "source_contact_display_name": _sender_display_name(
                sender_name,
                contact_id=contact_id,
            ),
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
                    sender_name=sender_name,
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
    source_ref = _build_trigger_source_ref(
        event=event,
        medium=medium,
        contact_id=contact_id,
    )
    live_candidates_with_tokens: list[tuple[TaskActivationSnapshot, str]] = []
    for candidate in live_candidates:
        assistant_id = candidate.assistant_id or (_current_task_assistant_id() or "")
        if not assistant_id:
            continue
        attempt_token = uuid.uuid4().hex[:12]
        remember_live_task_run_provenance(
            TaskRunProvenance(
                assistant_id=assistant_id,
                task_id=candidate.task_id,
                source_type="triggered",
                execution_mode="live",
                source_task_log_id=candidate.source_task_log_id,
                activation_revision=candidate.activation_revision,
                source_medium=medium.value,
                source_ref=source_ref,
                source_contact_id=(str(contact_id) if contact_id is not None else None),
                source_contact_display_name=_sender_display_name(
                    sender_name,
                    contact_id=contact_id,
                ),
                task_name=candidate.task_name,
                task_description=candidate.task_description,
                attempt_token=attempt_token,
            ),
        )
        live_candidates_with_tokens.append((candidate, attempt_token))
    if not live_candidates_with_tokens:
        return False
    candidate_ids = [
        candidate.task_id for candidate, _attempt_token in live_candidates_with_tokens
    ]
    cm.notifications_bar.push_notif(
        "Tasks",
        _trigger_candidate_notification_text(
            medium=medium,
            sender_name=sender_name,
            candidates=live_candidates_with_tokens,
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
            candidates=[
                candidate for candidate, _attempt_token in live_candidates_with_tokens
            ],
        ),
        source="task_trigger",
        contact=getattr(event, "contact", None),
    )
    return True
