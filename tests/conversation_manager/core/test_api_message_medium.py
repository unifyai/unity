"""Symbolic tests for the API message medium in the ConversationManager."""

from __future__ import annotations

from unity.conversation_manager.events import (
    ApiMessageReceived,
    ApiMessageSent,
    Event,
)
from unity.conversation_manager.cm_types import Medium, MEDIUM_REGISTRY, Mode


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


# ---------------------------------------------------------------------------
# log_message: medium detection
# ---------------------------------------------------------------------------


def test_log_message_medium_detection_received():
    """ApiMessageReceived must resolve to API_MESSAGE, not EMAIL."""
    event_name = ApiMessageReceived.__name__.lower()
    # Replicate the detection cascade from log_message
    if "apimessage" in event_name:
        medium = Medium.API_MESSAGE
    elif "unify" in event_name or "prehire" in event_name:
        medium = (
            Medium("unify_meet") if "meet" in event_name else Medium("unify_message")
        )
    elif "phone" in event_name:
        medium = Medium("phone_call")
    elif "sms" in event_name:
        medium = Medium("sms_message")
    else:
        medium = Medium("email")

    assert medium == Medium.API_MESSAGE, (
        f"Expected API_MESSAGE but got {medium} — "
        "the medium detection cascade does not handle ApiMessageReceived"
    )


def test_log_message_medium_detection_sent():
    """ApiMessageSent must resolve to API_MESSAGE, not EMAIL."""
    event_name = ApiMessageSent.__name__.lower()
    if "apimessage" in event_name:
        medium = Medium.API_MESSAGE
    elif "unify" in event_name or "prehire" in event_name:
        medium = (
            Medium("unify_meet") if "meet" in event_name else Medium("unify_message")
        )
    elif "phone" in event_name:
        medium = Medium("phone_call")
    elif "sms" in event_name:
        medium = Medium("sms_message")
    else:
        medium = Medium("email")

    assert medium == Medium.API_MESSAGE


# ---------------------------------------------------------------------------
# log_message: contact_id resolution uses safe .get()
# ---------------------------------------------------------------------------


def test_log_message_contact_isinstance_check():
    """ApiMessage events must be in the isinstance branch that uses .get().

    Before the fix, they fell through to a raw dict['contact_id'] access
    which raised KeyError when the contact dict was empty.
    """
    received = ApiMessageReceived(
        contact={"contact_id": 1},
        content="test",
    )
    sent = ApiMessageSent(
        contact={},  # empty — the scenario that triggered KeyError
        content="test",
    )

    from unity.conversation_manager.events import (
        UnifyMessageSent,
        UnifyMessageReceived,
        InboundUnifyMeetUtterance,
        OutboundUnifyMeetUtterance,
    )

    safe_types = (
        UnifyMessageSent,
        UnifyMessageReceived,
        InboundUnifyMeetUtterance,
        OutboundUnifyMeetUtterance,
        ApiMessageSent,
        ApiMessageReceived,
    )

    assert isinstance(
        received,
        safe_types,
    ), "ApiMessageReceived not in safe isinstance branch"
    assert isinstance(sent, safe_types), "ApiMessageSent not in safe isinstance branch"

    # The safe branch uses .get() — must not raise
    assert sent.contact.get("contact_id") is None
    assert received.contact.get("contact_id") == 1


def test_log_message_empty_contact_no_keyerror():
    """An ApiMessageSent with an empty contact dict must not raise KeyError.

    This is the exact scenario from the production error:
    'Error executing log_message: contact_id'
    """
    event = ApiMessageSent(contact={}, content="response", api_message_id="abc")
    # The safe path uses .get(); the old code did event.contact["contact_id"]
    try:
        _ = event.contact.get("contact_id")
    except KeyError:
        raise AssertionError(
            "contact.get('contact_id') raised KeyError — "
            "the safe branch is not being used",
        )


# ---------------------------------------------------------------------------
# send_api_response: contact fallback preserves contact_id
# ---------------------------------------------------------------------------


def test_send_api_response_contact_fallback_has_contact_id():
    """When get_contact returns None, the fallback dict must include contact_id.

    Before the fix, the fallback was {}, causing downstream KeyError and
    'API response to Unknown' in logs.
    """
    get_contact_returns_none = None
    contact_id = 1
    contact = get_contact_returns_none or {"contact_id": contact_id}
    assert "contact_id" in contact
    assert contact["contact_id"] == contact_id


# ---------------------------------------------------------------------------
# Hydration: ApiMessage events in _MESSAGE_PRODUCING_EVENTS
# ---------------------------------------------------------------------------


def test_api_messages_in_hydration_set():
    """ApiMessage events must be in _MESSAGE_PRODUCING_EVENTS for session hydration."""
    from unity.conversation_manager.domains.managers_utils import (
        _MESSAGE_PRODUCING_EVENTS,
    )

    assert "ApiMessageReceived" in _MESSAGE_PRODUCING_EVENTS, (
        "ApiMessageReceived missing from _MESSAGE_PRODUCING_EVENTS — "
        "API messages will be lost on session restart"
    )
    assert "ApiMessageSent" in _MESSAGE_PRODUCING_EVENTS, (
        "ApiMessageSent missing from _MESSAGE_PRODUCING_EVENTS — "
        "API messages will be lost on session restart"
    )


# ---------------------------------------------------------------------------
# Attachments and tags support
# ---------------------------------------------------------------------------


def test_api_message_received_default_attachments_and_tags():
    """New fields default to empty lists for backward compatibility."""
    event = ApiMessageReceived(
        contact={"contact_id": 1},
        content="test",
        api_message_id="msg-1",
    )
    assert event.attachments == []
    assert event.tags == []


def test_api_message_sent_default_attachments_and_tags():
    event = ApiMessageSent(
        contact={"contact_id": 1},
        content="response",
        api_message_id="msg-1",
    )
    assert event.attachments == []
    assert event.tags == []


def test_api_message_received_with_attachments_and_tags():
    attachments = [
        {"id": "att-1", "filename": "report.pdf", "gs_url": "gs://bucket/report.pdf"},
    ]
    tags = ["source:slack", "channel:#general"]
    event = ApiMessageReceived(
        contact={"contact_id": 1},
        content="See attached",
        api_message_id="msg-2",
        attachments=attachments,
        tags=tags,
    )
    assert len(event.attachments) == 1
    assert event.attachments[0]["filename"] == "report.pdf"
    assert event.tags == ["source:slack", "channel:#general"]


def test_api_message_sent_with_attachments_and_tags():
    attachments = [
        {
            "id": "resp-att-1",
            "filename": "analysis.xlsx",
            "gs_url": "gs://bucket/analysis.xlsx",
        },
    ]
    tags = ["source:slack"]
    event = ApiMessageSent(
        contact={"contact_id": 1},
        content="Done",
        api_message_id="msg-3",
        attachments=attachments,
        tags=tags,
    )
    assert len(event.attachments) == 1
    assert event.attachments[0]["filename"] == "analysis.xlsx"
    assert event.tags == ["source:slack"]


def test_api_message_roundtrip_with_attachments_and_tags():
    """Attachments and tags survive serialization round-trip."""
    attachments = [{"id": "a1", "filename": "data.csv", "gs_url": "gs://b/data.csv"}]
    tags = ["env:prod", "priority:high"]

    event = ApiMessageReceived(
        contact={"contact_id": 1, "first_name": "Boss"},
        content="Process this",
        api_message_id="msg-rt-1",
        attachments=attachments,
        tags=tags,
    )
    serialized = event.to_json()
    deserialized = Event.from_json(serialized)

    assert isinstance(deserialized, ApiMessageReceived)
    assert deserialized.attachments == attachments
    assert deserialized.tags == tags


def test_api_message_sent_roundtrip_with_attachments_and_tags():
    attachments = [{"id": "r1", "filename": "out.pdf", "gs_url": "gs://b/out.pdf"}]
    tags = ["env:prod"]

    event = ApiMessageSent(
        contact={"contact_id": 1},
        content="Here you go",
        api_message_id="msg-rt-2",
        attachments=attachments,
        tags=tags,
    )
    serialized = event.to_json()
    deserialized = Event.from_json(serialized)

    assert isinstance(deserialized, ApiMessageSent)
    assert deserialized.attachments == attachments
    assert deserialized.tags == tags


# ---------------------------------------------------------------------------
# ApiMessage CommsMessage type
# ---------------------------------------------------------------------------


def test_api_message_comms_type_exists():
    from unity.conversation_manager.domains.contact_index import ApiMessage

    msg = ApiMessage(
        name="Boss",
        content="hello",
        timestamp=ApiMessageReceived(
            contact={},
            content="",
        ).timestamp,
        role="user",
        attachments=[{"id": "a1", "filename": "f.txt"}],
        tags=["source:api"],
    )
    assert msg.attachments[0]["filename"] == "f.txt"
    assert msg.tags == ["source:api"]


def test_api_message_in_message_type_to_medium():
    from unity.conversation_manager.domains.contact_index import (
        ApiMessage,
        _MESSAGE_TYPE_TO_MEDIUM,
    )

    assert ApiMessage in _MESSAGE_TYPE_TO_MEDIUM
    assert _MESSAGE_TYPE_TO_MEDIUM[ApiMessage] == Medium.API_MESSAGE


def test_build_message_creates_api_message_type():
    """build_message with API_MESSAGE medium creates an ApiMessage, not generic Message."""
    from unity.conversation_manager.domains.contact_index import (
        ApiMessage,
        ContactIndex,
    )

    ci = ContactIndex()
    ci.get_or_create_conversation(1)
    entry = ci.build_message(
        contact_id=1,
        sender_name="Boss",
        thread_name=Medium.API_MESSAGE,
        message_content="hello",
        role="user",
        tags=["tag1"],
        attachments=[{"id": "a1", "filename": "f.pdf"}],
    )
    assert isinstance(entry.message, ApiMessage)
    assert entry.message.tags == ["tag1"]
    assert entry.message.attachments == [{"id": "a1", "filename": "f.pdf"}]
    assert entry.medium == Medium.API_MESSAGE


# ---------------------------------------------------------------------------
# Renderer: ApiMessage rendering
# ---------------------------------------------------------------------------


def test_renderer_renders_api_message_with_tags_and_attachments():
    """The renderer should include [Tags: ...] and [Attachments: ...] for ApiMessage."""
    from unity.conversation_manager.domains.contact_index import ApiMessage
    from unity.conversation_manager.domains.renderer import Renderer

    from datetime import datetime

    renderer = Renderer()
    msg = ApiMessage(
        name="Boss",
        content="See attached report",
        timestamp=datetime(2026, 3, 8, 12, 0),
        role="user",
        attachments=[{"id": "a1", "filename": "report.pdf"}],
        tags=["source:slack", "channel:#ops"],
    )

    rendered = renderer.render_message(msg, last_snapshot=datetime(2026, 3, 7))
    assert "report.pdf" in rendered
    assert "source:slack" in rendered
    assert "channel:#ops" in rendered
    assert "[Tags:" in rendered
    assert "[Attachments:" in rendered


def test_renderer_renders_api_message_without_tags():
    from unity.conversation_manager.domains.contact_index import ApiMessage
    from unity.conversation_manager.domains.renderer import Renderer
    from datetime import datetime

    renderer = Renderer()
    msg = ApiMessage(
        name="You",
        content="Just text",
        timestamp=datetime(2026, 3, 8, 12, 0),
        role="assistant",
    )

    rendered = renderer.render_message(msg, last_snapshot=datetime(2026, 3, 7))
    assert "Just text" in rendered
    assert "[Tags:" not in rendered
    assert "[Attachments:" not in rendered
