from __future__ import annotations

from collections import deque
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

import unity.coordinator_manager.activity as activity_module
from tests.helpers import _handle_project, capture_events
from unity.coordinator_manager.activity import (
    activity_entity,
    flush_pending_coordinator_activity_publishes,
    join_coordinator_activity_publishes,
    publish_coordinator_activity,
    safe_activity_text,
)
from unity.conversation_manager.domains.coordinator_tools import CoordinatorTools
from unity.events.stream_filters import is_streaming_noise
from unity.events.types.coordinator_activity import CoordinatorActivityPayload
from unity.session_details import SESSION_DETAILS


def test_activity_schema_rejects_extra_fields_and_shapes_entities():
    entity = activity_entity("colleague", name="Revenue Ops", entity_id=7002)

    payload = CoordinatorActivityPayload(
        activity_id="activity-1",
        phase="completed",
        stage="integration_setup",
        surfaces=["colleagues"],
        title="Created Revenue Ops colleague",
        related_entities=[entity],
        occurred_at=datetime.now(UTC),
    )

    assert payload.related_entities == [entity]
    assert payload.occurred_at.tzinfo is UTC
    with pytest.raises(ValidationError):
        CoordinatorActivityPayload(
            activity_id="activity-2",
            phase="completed",
            stage="implementation",
            title="Created Revenue Ops colleague",
            occurred_at=datetime.now(UTC),
            unexpected="not allowed",
        )


def test_activity_stage_uses_integration_setup_contract():
    payload = CoordinatorActivityPayload(
        activity_id="activity-integration-setup",
        phase="needs_input",
        stage="integration_setup",
        title="Connect Salesforce",
        occurred_at=datetime.now(UTC),
    )

    assert payload.stage == "integration_setup"

    with pytest.raises(ValidationError):
        CoordinatorActivityPayload(
            activity_id="activity-credential-setup",
            phase="needs_input",
            stage="credential_setup",
            title="Connect Salesforce",
            occurred_at=datetime.now(UTC),
        )


def test_activity_text_redacts_secret_like_content_and_truncates():
    assert (
        safe_activity_text(
            "Salesforce API key configured",
            fallback="Credential updated",
        )
        == "Credential updated"
    )
    assert safe_activity_text("a" * 200, fallback="fallback", max_length=20) == (
        "a" * 17 + "..."
    )


def test_activity_publisher_noops_for_non_coordinator_session():
    SESSION_DETAILS.reset()

    assert (
        publish_coordinator_activity(
            phase="progress",
            stage="requirements",
            title="Added setup step",
        )
        is None
    )


def test_activity_publish_queues_until_event_bus_is_available(monkeypatch):
    class _FakeBus:
        def __init__(self) -> None:
            self.ready = False
            self.published: list[tuple[str, bool]] = []

        def __bool__(self) -> bool:
            return self.ready

        async def publish(self, event, *, blocking: bool = False) -> None:
            if not self.ready:
                raise RuntimeError("EVENT_BUS has not been initialised yet")
            self.published.append((event.payload["title"], blocking))

    fake_bus = _FakeBus()
    monkeypatch.setattr(activity_module, "EVENT_BUS", fake_bus)
    monkeypatch.setattr(activity_module, "_PUBLISH_TASKS", set())
    monkeypatch.setattr(activity_module, "_PENDING_PUBLISH_EVENTS", deque())
    monkeypatch.setattr(activity_module, "_PENDING_DRAIN_TASK", None)
    SESSION_DETAILS.reset()
    SESSION_DETAILS.is_coordinator = True

    publish_coordinator_activity(
        phase="progress",
        stage="requirements",
        title="Queued until init",
    )
    assert fake_bus.published == []

    fake_bus.ready = True
    flush_pending_coordinator_activity_publishes()

    assert fake_bus.published == [("Queued until init", True)]


def test_activity_publish_flush_preserves_fifo_order(monkeypatch):
    class _FakeBus:
        def __init__(self) -> None:
            self.ready = False
            self.published: list[str] = []

        def __bool__(self) -> bool:
            return self.ready

        async def publish(self, event, *, blocking: bool = False) -> None:
            if not self.ready:
                raise RuntimeError("EVENT_BUS has not been initialised yet")
            self.published.append(event.payload["title"])

    fake_bus = _FakeBus()
    monkeypatch.setattr(activity_module, "EVENT_BUS", fake_bus)
    monkeypatch.setattr(activity_module, "_PUBLISH_TASKS", set())
    monkeypatch.setattr(activity_module, "_PENDING_PUBLISH_EVENTS", deque())
    monkeypatch.setattr(activity_module, "_PENDING_DRAIN_TASK", None)
    SESSION_DETAILS.reset()
    SESSION_DETAILS.is_coordinator = True

    publish_coordinator_activity(
        phase="progress",
        stage="requirements",
        title="First queued activity",
    )
    publish_coordinator_activity(
        phase="progress",
        stage="requirements",
        title="Second queued activity",
    )

    fake_bus.ready = True
    flush_pending_coordinator_activity_publishes()

    assert fake_bus.published == [
        "First queued activity",
        "Second queued activity",
    ]


def test_coordinator_activity_is_not_streaming_noise():
    payload = CoordinatorActivityPayload(
        activity_id="activity-1",
        phase="progress",
        stage="requirements",
        surfaces=["tasks"],
        title="Added setup step",
        occurred_at=datetime.now(UTC),
    )

    assert (
        is_streaming_noise("CoordinatorActivity", payload.model_dump(mode="json"))
        is False
    )


@pytest.mark.enable_eventbus
@pytest.mark.asyncio
@_handle_project
async def test_setup_tools_emit_lightweight_activity_for_progress():
    SESSION_DETAILS.is_coordinator = True
    tools = CoordinatorTools(cm=object()).as_tools()

    async with capture_events("CoordinatorActivity") as events:
        added = tools["add_setup_checklist_item"](
            title="Connect Salesforce",
            description="Store read-only access and validate renewal data.",
            kind="integration",
            chat_prompt="I have Salesforce open. Walk me through the safest setup path.",
            chat_prompt_label="Start guided setup",
        )
        item_id = added["details"]["item_id"]
        tools["update_setup_checklist_item"](
            item_id=item_id,
            description="Start with read-only Salesforce access, then pause.",
            chat_prompt="Should we start with Salesforce first, or pause after planning?",
            chat_prompt_label="Choose first slice",
        )
        tools["update_setup_checklist_item"](
            item_id=item_id,
            status="done",
            chat_prompt="Let's review the next integration before continuing.",
            chat_prompt_label="Review next step",
        )
        await join_coordinator_activity_publishes()

    payloads = [event.payload for event in events]
    assert len(payloads) >= 3
    added_payload = next(
        payload
        for payload in payloads
        if payload["phase"] == "progress" and payload["checklist_item_id"] == item_id
    )
    assert "description" not in added_payload
    assert added_payload["chat_prompt"] == (
        "I have Salesforce open. Walk me through the safest setup path."
    )
    assert added_payload["chat_prompt_label"] == "Start guided setup"
    needs_input_payload = next(
        payload
        for payload in payloads
        if payload["phase"] == "needs_input" and payload["checklist_item_id"] == item_id
    )
    assert needs_input_payload["activity_id"] == added_payload["activity_id"]
    assert needs_input_payload["chat_prompt"] == (
        "Should we start with Salesforce first, or pause after planning?"
    )
    assert needs_input_payload["chat_prompt_label"] == "Choose first slice"
    completed_payload = next(
        payload
        for payload in payloads
        if payload["phase"] == "completed" and payload["checklist_item_id"] == item_id
    )
    assert completed_payload["correlation_id"] == added_payload["correlation_id"]
    assert completed_payload["activity_id"] == added_payload["activity_id"]
    assert completed_payload["chat_prompt"] == (
        "Let's review the next integration before continuing."
    )
    assert completed_payload["chat_prompt_label"] == "Review next step"
