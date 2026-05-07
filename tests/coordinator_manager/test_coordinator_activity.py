from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from tests.helpers import _handle_project, capture_events
from unity.coordinator_manager.activity import (
    activity_entity,
    join_coordinator_activity_publishes,
    publish_coordinator_activity,
    safe_activity_text,
)
from unity.coordinator_manager.coordinator_manager import CoordinatorOnboardingManager
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
async def test_onboarding_manager_emits_lightweight_activity_for_progress():
    SESSION_DETAILS.is_coordinator = True
    manager = CoordinatorOnboardingManager()

    async with capture_events("CoordinatorActivity") as events:
        added = manager.add_checklist_item(
            title="Connect Salesforce",
            description="Store read-only access and validate renewal data.",
            kind="integration",
        )
        item_id = added["details"]["item_id"]
        manager.update_checklist_item(item_id=item_id, status="done")
        manager.set_state(mode="ready_to_go")
        manager.set_state(mode="ready_to_go")
        await join_coordinator_activity_publishes()

    payloads = [event.payload for event in events]
    assert len(payloads) >= 3
    added_payload = next(
        payload
        for payload in payloads
        if payload["phase"] == "progress" and payload["checklist_item_id"] == item_id
    )
    assert "description" not in added_payload
    completed_payload = next(
        payload
        for payload in payloads
        if payload["phase"] == "completed" and payload["checklist_item_id"] == item_id
    )
    assert completed_payload["correlation_id"] == added_payload["correlation_id"]
    assert completed_payload["activity_id"] == added_payload["activity_id"]
    handoff_payloads = [
        payload
        for payload in payloads
        if payload["stage"] == "handoff" and payload["title"] == "Setup is ready to go"
    ]
    assert len(handoff_payloads) == 1
