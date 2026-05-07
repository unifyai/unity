"""Product-facing activity publishing for Coordinator setup work."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Literal
from uuid import uuid4

from unity.events.event_bus import EVENT_BUS, Event
from unity.events.types.coordinator_activity import (
    CoordinatorActivityEntity,
    CoordinatorActivityEntityType,
    CoordinatorActivityPayload,
    CoordinatorActivityPhase,
    CoordinatorActivityStage,
    CoordinatorActivitySurface,
)
from unity.session_details import SESSION_DETAILS

LOGGER = logging.getLogger(__name__)

_SENSITIVE_TERMS = frozenset(
    {
        "api key",
        "api_key",
        "apikey",
        "authorization",
        "bearer",
        "credential",
        "password",
        "private key",
        "refresh token",
        "secret",
        "token",
    },
)
_MAX_CARD_TEXT_LENGTH = 160
_MAX_ENTITY_NAME_LENGTH = 80
_PUBLISH_TASKS: set[asyncio.Task[None]] = set()


def coordinator_activity_id(prefix: str = "activity") -> str:
    """Return an opaque id suitable for grouping Coordinator activity events."""

    return f"{prefix}-{uuid4().hex}"


def activity_entity(
    entity_type: CoordinatorActivityEntityType,
    *,
    name: object,
    entity_id: object | None = None,
) -> CoordinatorActivityEntity:
    """Build a redacted, display-ready related entity for an activity card."""

    return CoordinatorActivityEntity(
        type=entity_type,
        id=None if entity_id is None else str(entity_id),
        name=safe_activity_text(
            name,
            fallback=_fallback_entity_name(entity_type),
            max_length=_MAX_ENTITY_NAME_LENGTH,
        ),
    )


def safe_activity_text(
    value: object,
    *,
    fallback: str,
    max_length: int = _MAX_CARD_TEXT_LENGTH,
) -> str:
    """Return concise card text without secret-looking values."""

    if value is None:
        return fallback
    text = str(value).strip()
    if not text:
        return fallback
    lowered = text.lower()
    if any(term in lowered for term in _SENSITIVE_TERMS):
        return fallback
    return _truncate(text, max_length)


def publish_coordinator_activity(
    *,
    phase: CoordinatorActivityPhase,
    stage: CoordinatorActivityStage,
    title: str,
    surfaces: Sequence[CoordinatorActivitySurface] = (),
    summary: str | None = None,
    checklist_item_id: int | None = None,
    related_entities: Sequence[CoordinatorActivityEntity | dict] = (),
    chat_prompt: str | None = None,
    chat_prompt_label: str | None = None,
    correlation_id: str | None = None,
    activity_id: str | None = None,
    status: Literal["ok", "error"] | None = None,
    error: str | None = None,
) -> str | None:
    """Publish a Coordinator setup activity event without affecting setup work.

    Returns the activity id used for the event, or ``None`` when the active
    session is not a Coordinator session.
    """

    if not SESSION_DETAILS.is_coordinator:
        return None

    resolved_activity_id = activity_id or coordinator_activity_id(_stage_prefix(stage))
    resolved_status = status or ("error" if phase == "failed" else "ok")
    try:
        payload = CoordinatorActivityPayload(
            activity_id=resolved_activity_id,
            phase=phase,
            stage=stage,
            surfaces=list(dict.fromkeys(surfaces)),
            title=safe_activity_text(title, fallback=_fallback_title(stage, phase)),
            summary=(
                None
                if summary is None
                else safe_activity_text(
                    summary,
                    fallback="",
                    max_length=_MAX_CARD_TEXT_LENGTH,
                )
            ),
            checklist_item_id=checklist_item_id,
            related_entities=[
                (
                    entity
                    if isinstance(entity, CoordinatorActivityEntity)
                    else CoordinatorActivityEntity.model_validate(entity)
                )
                for entity in related_entities
            ],
            chat_prompt=(
                None
                if chat_prompt is None
                else safe_activity_text(
                    chat_prompt,
                    fallback="",
                    max_length=_MAX_CARD_TEXT_LENGTH,
                )
            ),
            chat_prompt_label=(
                None
                if chat_prompt_label is None
                else safe_activity_text(
                    chat_prompt_label,
                    fallback="Continue setup",
                    max_length=_MAX_ENTITY_NAME_LENGTH,
                )
            ),
            correlation_id=correlation_id,
            occurred_at=datetime.now(UTC),
            status=resolved_status,
            error=(
                None
                if error is None
                else safe_activity_text(
                    error,
                    fallback="The setup step could not finish.",
                )
            ),
        )
        event = Event(type="CoordinatorActivity", payload=payload)
        _schedule_publish(event)
    except Exception:
        LOGGER.exception("Failed to schedule Coordinator activity event")
    return resolved_activity_id


def _schedule_publish(event: Event) -> None:
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        asyncio.run(EVENT_BUS.publish(event, blocking=True))
        return
    task = loop.create_task(EVENT_BUS.publish(event))
    _PUBLISH_TASKS.add(task)
    task.add_done_callback(_publish_done)


async def join_coordinator_activity_publishes() -> None:
    """Wait for in-flight Coordinator activity publish tasks."""

    tasks = tuple(_PUBLISH_TASKS)
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)


def _publish_done(task: asyncio.Task[None]) -> None:
    _PUBLISH_TASKS.discard(task)
    try:
        task.result()
    except Exception:
        LOGGER.exception("Coordinator activity publish task failed")


def _truncate(text: str, max_length: int) -> str:
    if len(text) <= max_length:
        return text
    return f"{text[: max_length - 3].rstrip()}..."


def _fallback_entity_name(entity_type: CoordinatorActivityEntityType) -> str:
    return {
        "human": "Authorized human",
        "colleague": "Colleague",
        "workspace": "Workspace",
        "credential": "Credential",
        "task": "Task",
        "knowledge": "Knowledge",
        "guidance": "Guidance",
        "dashboard": "Dashboard",
        "function": "Function",
        "data": "Data",
        "invitation": "Invitation",
        "validation": "Validation",
    }[entity_type]


def _fallback_title(
    stage: CoordinatorActivityStage,
    phase: CoordinatorActivityPhase,
) -> str:
    if phase == "failed":
        return "Setup step needs attention"
    return {
        "discovery": "Learning about the workflow",
        "requirements": "Updating the setup plan",
        "proposal": "Drafting the setup proposal",
        "confirmation": "Waiting for setup confirmation",
        "implementation": "Updating the workspace setup",
        "integration_setup": "Waiting on integration setup",
        "validation": "Validating the setup",
        "handoff": "Handing off the workflow",
    }[stage]


def _stage_prefix(stage: CoordinatorActivityStage) -> str:
    return stage.replace("_", "-")
