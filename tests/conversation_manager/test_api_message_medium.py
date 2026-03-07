"""Symbolic tests for the API message medium in the ConversationManager."""

from __future__ import annotations

from unity.conversation_manager.events import (
    ApiMessageReceived,
    ApiMessageSent,
    Event,
)
from unity.conversation_manager.types import Medium, MEDIUM_REGISTRY, Mode


def test_medium_enum_has_api_message():
    assert Medium.API_MESSAGE == "api_message"
    assert Medium.API_MESSAGE in MEDIUM_REGISTRY


def test_medium_info():
    info = MEDIUM_REGISTRY[Medium.API_MESSAGE]
    assert info.mode == Mode.TEXT
    assert info.value == "api_message"
    assert info.description


def test_medium_properties():
    m = Medium.API_MESSAGE
    assert m.mode == Mode.TEXT
    assert m.description
    assert not m.mode.is_voice


def test_api_message_received_event_registered():
    assert "ApiMessageReceived" in Event._registry
    assert Event._registry["ApiMessageReceived"] is ApiMessageReceived


def test_api_message_sent_event_registered():
    assert "ApiMessageSent" in Event._registry
    assert Event._registry["ApiMessageSent"] is ApiMessageSent


def test_api_message_received_roundtrip():
    event = ApiMessageReceived(
        contact={"contact_id": 1, "first_name": "Boss"},
        content="Hello from the API",
        api_message_id="msg-uuid-123",
    )
    assert event.content == "Hello from the API"
    assert event.api_message_id == "msg-uuid-123"

    serialized = event.to_json()
    deserialized = Event.from_json(serialized)

    assert isinstance(deserialized, ApiMessageReceived)
    assert deserialized.content == "Hello from the API"
    assert deserialized.api_message_id == "msg-uuid-123"
    assert deserialized.contact["contact_id"] == 1


def test_api_message_sent_roundtrip():
    event = ApiMessageSent(
        contact={"contact_id": 1, "first_name": "Boss"},
        content="Here is my response",
        api_message_id="msg-uuid-456",
    )
    serialized = event.to_json()
    deserialized = Event.from_json(serialized)

    assert isinstance(deserialized, ApiMessageSent)
    assert deserialized.content == "Here is my response"
    assert deserialized.api_message_id == "msg-uuid-456"


def test_api_message_received_topic():
    assert ApiMessageReceived.topic == "app:comms:api_message_message"


def test_api_message_sent_topic():
    assert ApiMessageSent.topic == "app:comms:api_message_sent"


def test_api_message_events_are_content_logged():
    assert ApiMessageReceived.content_logged is True
    assert ApiMessageSent.content_logged is True


def test_event_handler_registers_api_events():
    """Verify both API message events are handled by the shared comms handler."""
    from unity.conversation_manager.domains.event_handlers import EventHandler

    for event_cls in (ApiMessageReceived, ApiMessageSent):
        assert (
            event_cls in EventHandler._registry
        ), f"{event_cls.__name__} not registered in EventHandler"
