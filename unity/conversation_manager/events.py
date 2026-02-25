import json
import uuid
from typing import Optional, Any, ClassVar
from datetime import datetime
from dataclasses import dataclass, asdict, field

from pydantic import BaseModel

from unity.common.prompt_helpers import now as prompt_now


def custom_dict_factory(kv):
    d = {}
    for k, v in kv:
        if isinstance(v, datetime):
            d[k] = v.isoformat()
        elif isinstance(v, BaseModel):
            d[k] = v.model_dump()
        else:
            d[k] = v
    return d


class _TruncatedReprMixin:
    """Mixin for events that need truncated repr (to avoid logging huge payloads)."""

    def __str__(self) -> str:
        return self._repr_truncated()

    def __repr__(self) -> str:
        return self._repr_truncated()

    def _repr_truncated(self) -> str:
        raise NotImplementedError


def _now_datetime() -> datetime:
    """Wrapper for prompt_now that returns datetime for dataclass default_factory."""
    return prompt_now(as_string=False)


@dataclass(kw_only=True)
class Event:
    timestamp: datetime = field(default_factory=_now_datetime)

    _registry: ClassVar[dict[str, "Event"]] = {}
    loggable: ClassVar[bool] = True
    content_logged: ClassVar[bool] = False
    topic: ClassVar[str | None] = None

    def to_json(self):
        return json.dumps(self.to_dict())

    def to_dict(self):
        return {
            "event_name": self.__class__.__name__,
            "payload": asdict(self, dict_factory=custom_dict_factory),
        }

    def to_bus_event(self):
        from unity.events.event_bus import Event as BusEvent

        payload = self.to_dict()["payload"]
        return BusEvent(
            calling_id="",
            type="Comms",
            timestamp=self.timestamp.isoformat(),
            payload=payload,
            payload_cls=self.__class__.__name__,
        )

    @classmethod
    def from_dict(cls, data) -> "Event":
        import dataclasses

        target_cls = cls._registry.get(data["event_name"])
        if not target_cls:
            raise Exception(f"Class {data['event_name']} is not registered.")
        kwargs = data["payload"].copy()
        timestamp = kwargs.pop("timestamp")

        # Filter to only fields the target dataclass accepts
        valid_fields = {f.name for f in dataclasses.fields(target_cls)}
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_fields}

        return target_cls(
            **filtered_kwargs,
            timestamp=datetime.fromisoformat(timestamp),
        )

    @classmethod
    def from_json(cls, json_data):
        data = json.loads(json_data)
        return cls.from_dict(data)

    @classmethod
    def from_bus_event(cls, event):
        # Use mode="json" to ensure datetime objects are serialized to ISO strings,
        # which from_dict() expects for the timestamp field
        event_dump = event.model_dump(mode="json")
        data = {
            "event_name": event_dump["payload_cls"],
            "payload": event_dump["payload"],
        }
        return cls.from_dict(data)

    def __init_subclass__(cls):
        if cls.__name__ not in Event._registry:
            Event._registry[cls.__name__] = cls
        return cls


# --------------------------------------------------------------------------- #
# Comms Events
# --------------------------------------------------------------------------- #


@dataclass
class PhoneCallReceived(Event):
    topic: ClassVar[str | None] = "app:comms:call_received"

    contact: dict
    conference_name: str = ""


@dataclass
class PhoneCallAnswered(Event):
    topic: ClassVar[str | None] = "app:comms:call_answered"

    contact: dict


@dataclass
class PhoneCallNotAnswered(Event):
    """Outbound call was not answered (no-answer, busy, failed, etc.)."""

    topic: ClassVar[str | None] = "app:comms:call_not_answered"

    contact: dict
    reason: str = "no-answer"  # Twilio status: no-answer, busy, canceled, failed


@dataclass
class UnifyMeetReceived(Event):
    """Frontend/worker confirmed agent connected to room; begin LLM."""

    topic: ClassVar[str | None] = "app:comms:unify_meet_received"

    contact: dict
    room_name: str | None = None


@dataclass
class PhoneCallStarted(Event):
    topic: ClassVar[str | None] = "app:comms:phone_call_started"

    contact: dict


@dataclass
class UnifyMeetStarted(Event):
    """A web-based voice/video meeting session has started (no phone number).

    "contact" should reference the boss/user contact id (typically 1).
    """

    topic: ClassVar[str | None] = "app:comms:unify_meet_started"

    contact: dict


@dataclass
class InboundPhoneUtterance(Event):
    """Utterance received from the other party during a phone call."""

    topic: ClassVar[str | None] = "app:comms:phone_utterance"

    contact: dict
    content: str


@dataclass
class InboundUnifyMeetUtterance(Event):
    """Utterance received from the other party during a web-based voice/video meeting."""

    topic: ClassVar[str | None] = "app:comms:unify_utterance"

    contact: dict
    content: str


@dataclass
class VoiceInterrupt(Event):
    """User interrupted the assistant during a voice call."""

    topic: ClassVar[str | None] = "app:comms:voice_interrupt"

    contact: dict


@dataclass
class PhoneCallEnded(Event):
    topic: ClassVar[str | None] = "app:comms:phone_call_ended"

    contact: dict


@dataclass
class UnifyMeetEnded(Event):
    """The web-based voice/video meeting session has ended."""

    topic: ClassVar[str | None] = "app:comms:unify_meet_ended"

    contact: dict


@dataclass
class RecordingReady(Event):
    """A call/meet recording has been processed and is available in GCS."""

    topic: ClassVar[str | None] = "app:comms:recording_ready"

    conference_name: str
    recording_url: str


@dataclass
class SMSReceived(Event):
    topic: ClassVar[str | None] = "app:comms:msg_message"
    content_logged: ClassVar[bool] = True

    contact: dict
    content: str


@dataclass
class UnifyMessageReceived(Event):
    """A message was received via the Unify console chat interface.

    Attachments are downloaded asynchronously to the Downloads folder.
    Each attachment is a dict with keys: id, filename, gs_url, content_type, size_bytes.
    The actual files are saved to Downloads/ and can be accessed via FileManager.
    """

    topic: ClassVar[str | None] = "app:comms:unify_message_message"
    content_logged: ClassVar[bool] = True

    contact: dict
    content: str
    # List of attachment dicts with full metadata (files are saved to Downloads/).
    attachments: list[dict] = field(default_factory=list)


@dataclass
class PhoneCallSent(Event):
    topic: ClassVar[str | None] = "app:comms:make_call"

    contact: dict


@dataclass
class OutboundPhoneUtterance(Event):
    """Utterance sent by the assistant during a phone call."""

    topic: ClassVar[str | None] = "app:comms:phone_utterance"

    contact: dict
    content: str


@dataclass
class OutboundUnifyMeetUtterance(Event):
    """Utterance sent by the assistant during a web-based voice/video meeting."""

    topic: ClassVar[str | None] = "app:comms:unify_utterance"

    contact: dict
    content: str


@dataclass
class CallGuidance(Event):
    """
    Guidance from the Main CM Brain sent to the Voice Agent during a call.


    When should_speak is True, response_text contains the exact text the fast
    brain should utter via session.say(), bypassing its own LLM. When
    should_speak is False, the fast brain absorbs the notification silently
    and must NOT speak in response.
    """

    topic: ClassVar[str | None] = "app:comms:assistant_call_guidance"

    contact: dict
    content: str
    response_text: str = ""
    should_speak: bool = False
    source: str = ""
    agent_service_url: str = ""


@dataclass
class EmailReceived(Event):
    """An email was received from a contact.

    Attachments are downloaded asynchronously to the Downloads folder. The
    ``attachments`` field contains only filenames (not paths or binary data) so
    the LLM can acknowledge them and, if needed, access them via FileManager.
    """

    topic: ClassVar[str | None] = "app:comms:email_message"
    content_logged: ClassVar[bool] = True

    contact: dict
    subject: str
    body: str
    # Email provider identifier used for threading (e.g., RFC Message-ID header value).
    # This is *not* the TranscriptManager's auto-incremented message_id.
    email_id: Optional[str] = None
    # List of attachment filenames (actual files are saved to Downloads/).
    attachments: list[str] = field(default_factory=list)
    # Recipients from the original email (for reply-all functionality)
    to: list[str] = field(default_factory=list)
    cc: list[str] = field(default_factory=list)
    bcc: list[str] = field(default_factory=list)


# assistant events
@dataclass
class SMSSent(Event):
    topic: ClassVar[str | None] = "app:comms:sms_sent"
    content_logged: ClassVar[bool] = True

    contact: dict
    content: str


@dataclass
class UnifyMessageSent(Event):
    """A message was sent via the Unify console chat interface.

    Attachments are uploaded to GCS. Each attachment is a dict with keys:
    id, filename, gs_url, content_type, size_bytes.
    """

    topic: ClassVar[str | None] = "app:comms:unify_message_sent"
    content_logged: ClassVar[bool] = True

    contact: dict
    content: str
    # List of attachment dicts with full metadata.
    attachments: list[dict] = field(default_factory=list)


@dataclass
class EmailSent(Event):
    """An email was sent to a contact.

    Attachments are specified by filepath and uploaded with the email. The
    ``attachments`` field contains only filenames (not paths) for display.
    """

    topic: ClassVar[str | None] = "app:comms:email_sent"
    content_logged: ClassVar[bool] = True

    contact: dict
    subject: str
    body: str
    # Email provider identifier used for threading (e.g., RFC Message-ID header value).
    # This is *not* the TranscriptManager's auto-incremented message_id.
    email_id_replied_to: str | None = None
    # List of attachment filenames that were sent with the email.
    attachments: list[str] = field(default_factory=list)
    # Recipients the email was sent to
    to: list[str] = field(default_factory=list)
    cc: list[str] = field(default_factory=list)
    bcc: list[str] = field(default_factory=list)


@dataclass
class UnknownContactCreated(Event):
    """A new contact was automatically created from an unknown inbound message.

    This event is published when an inbound SMS, email, or call arrives from
    a sender that is not in the Contacts table and not in the BlackList.

    The contact is created with:
    - Only the medium field populated (phone_number or email_address)
    - should_respond=False to prevent automatic responses
    - A response_policy guiding the assistant to seek boss guidance

    The ConversationManager should use this event to potentially notify the
    boss and ask for guidance on how to handle this new contact.
    """

    topic: ClassVar[str | None] = "app:comms:unknown_contact_created"

    contact: dict
    medium: str  # The communication medium (e.g., "sms_message", "email", "phone_call")
    message_preview: str = ""  # Optional preview of the initial message


@dataclass
class _SessionConfigBase(Event):
    """Base class for session configuration events (StartupEvent, AssistantUpdateEvent)."""

    loggable: ClassVar[bool] = False
    api_key: str
    medium: str
    assistant_id: str
    user_id: str
    assistant_first_name: str
    assistant_surname: str
    assistant_age: str
    assistant_nationality: str
    assistant_about: str
    assistant_number: str
    assistant_email: str
    user_first_name: str
    user_surname: str
    user_number: str
    user_email: str
    voice_id: str
    voice_provider: str = "cartesia"
    voice_mode: str = "tts"
    assistant_timezone: str = (
        ""  # IANA timezone identifier; default empty for backward compat
    )
    desktop_mode: str = "ubuntu"
    desktop_url: str | None = None
    user_desktop_mode: str | None = None
    user_desktop_filesys_sync: bool = False
    user_desktop_url: str | None = None
    # Demo assistant metadata ID. If set, this is a demo session.
    # Unity derives demo_mode from (demo_id is not None).
    demo_id: int | None = None


@dataclass
class StartupEvent(_SessionConfigBase):
    """Initial session configuration sent when ConversationManager starts."""


@dataclass
class InitializationComplete(Event):
    """Published when ConversationManager has fully initialized all managers."""

    loggable: ClassVar[bool] = False


@dataclass
class AssistantUpdateEvent(_SessionConfigBase):
    """Updated session configuration sent to a running ConversationManager."""


@dataclass
class Ping(Event):
    loggable: ClassVar[bool] = False
    kind: str


@dataclass
class Error(Event):
    message: str


@dataclass
class LogMessageResponse(Event):
    medium: str
    exchange_id: int


@dataclass
class ContactInfoResponse(Event):
    contact_details: dict[str, Any]


@dataclass(repr=False)
class StoreChatHistory(_TruncatedReprMixin, Event):
    chat_history: list[dict]

    def _repr_truncated(self) -> str:
        return f"{self.__class__.__name__}(chat_history_len={len(self.chat_history)})"


@dataclass(repr=False)
class GetChatHistory(_TruncatedReprMixin, Event):
    loggable: ClassVar[bool] = False
    chat_history: list[dict]

    def _repr_truncated(self) -> str:
        return f"{self.__class__.__name__}(chat_history_len={len(self.chat_history)})"


@dataclass(repr=False)
class GetBusEventsResponse(_TruncatedReprMixin, Event):
    loggable: ClassVar[bool] = False
    events: list[dict[str, Any]]

    def _repr_truncated(self) -> str:
        return f"{self.__class__.__name__}(events_len={len(self.events)})"


@dataclass
class PreHireMessage(Event):
    content: str
    role: str
    exchange_id: int


# --------------------------------------------------------------------------- #
# LLM inference events
# --------------------------------------------------------------------------- #
@dataclass
class LLMInput(Event):
    chat_history: list[dict]


@dataclass
class UpdateContactRollingSummaryResponse(Event):
    rolling_summaries: list[tuple[int, str]]


# --------------------------------------------------------------------------- #
# ConversationManagerHandle Events
# --------------------------------------------------------------------------- #


@dataclass
class NotificationInjectedEvent(Event):
    """Event to inject a notification into the ConversationManager."""

    content: str
    source: str
    target_conversation_id: str
    interjection_id: str = field(default_factory=lambda: str(uuid.uuid4().hex[:12]))
    pinned: bool = False


@dataclass
class NotificationUnpinnedEvent(Event):
    """Event to unpin a previously pinned interjection."""

    interjection_id: str
    target_conversation_id: str


# --------------------------------------------------------------------------- #
# Actor Events
# --------------------------------------------------------------------------- #


@dataclass(repr=False)
class ActorRequest(_TruncatedReprMixin, Event):
    """Event to ask or request the Actor to perform a task."""

    action_name: str
    query: str
    parent_chat_context: list[dict]

    def _repr_truncated(self) -> str:
        return (
            f"{self.__class__.__name__}(action_name={self.action_name}, "
            f"query={self.query}, "
            f"parent_chat_context_len={len(self.parent_chat_context)})"
        )


@dataclass
class ActorResponse(Event):
    """Event to respond to an Actor request."""

    handle_id: int
    action_name: str
    query: str
    response: str


@dataclass(repr=False)
class ActorHandleRequest(_TruncatedReprMixin, Event):
    """Event to any action on an existing Actor handle."""

    handle_id: int
    action_name: str
    query: str
    parent_chat_context: list[dict]

    def _repr_truncated(self) -> str:
        return (
            f"{self.__class__.__name__}(handle_id={self.handle_id}, "
            f"action_name={self.action_name}, "
            f"query={self.query}, "
            f"parent_chat_context_len={len(self.parent_chat_context)})"
        )


@dataclass
class ActorHandleResponse(Event):
    """Event to respond to an Actor handle request."""

    handle_id: int
    action_name: str
    query: str
    response: str
    call_id: str


@dataclass
class ActorResult(Event):
    """Event to the result of an Actor task."""

    handle_id: int
    success: bool
    result: dict | str | None = None
    error: str | None = None


@dataclass
class ActorClarificationRequest(Event):
    """Event to request clarification from the Actor."""

    handle_id: int
    query: str
    call_id: str


@dataclass
class ActorClarificationResponse(Event):
    """Event to respond to an Actor clarification request."""

    handle_id: int
    response: str
    call_id: str


@dataclass
class ActorNotification(Event):
    """Event to forward a progress notification from an Actor handle.

    Notifications arrive while the actor is still working. They carry
    status/progress updates and do not indicate turn completion.
    """

    handle_id: int
    response: str


@dataclass
class ActorSessionResponse(Event):
    """Event signalling that a persistent actor session has completed a turn.

    Unlike ``ActorNotification``, a session response means the actor has
    finished its current work and is **waiting for the next instruction**
    (via ``interject``).  The ``content`` field carries the actor's output
    for this turn.
    """

    handle_id: int
    content: str


@dataclass
class ActorHandleStarted(Event):
    action_name: str
    handle_id: id
    query: str
    response_format: dict | None = None


# --------------------------------------------------------------------------- #
# Meet Interaction Events (screen share / remote control)
# --------------------------------------------------------------------------- #


@dataclass
class AssistantScreenShareStarted(Event):
    """User enabled assistant screen sharing during a Unify Meet session.

    The assistant's desktop is now visible to the user.
    """

    topic: ClassVar[str | None] = "app:comms:assistant_screen_share_started"

    reason: str = ""


@dataclass
class AssistantScreenShareStopped(Event):
    """User disabled assistant screen sharing during a Unify Meet session.

    The assistant's desktop is no longer visible to the user.
    """

    topic: ClassVar[str | None] = "app:comms:assistant_screen_share_stopped"

    reason: str = ""


@dataclass
class UserScreenShareStarted(Event):
    """User started sharing their screen during a Unify Meet session.

    The user's screen is now being streamed to the assistant.
    """

    topic: ClassVar[str | None] = "app:comms:user_screen_share_started"

    reason: str = ""


@dataclass
class UserScreenShareStopped(Event):
    """User stopped sharing their screen during a Unify Meet session."""

    topic: ClassVar[str | None] = "app:comms:user_screen_share_stopped"

    reason: str = ""


@dataclass
class UserWebcamStarted(Event):
    """User enabled their webcam during a Unify Meet session.

    The user's webcam feed is now being streamed to the assistant.
    """

    topic: ClassVar[str | None] = "app:comms:user_webcam_started"

    reason: str = ""


@dataclass
class UserWebcamStopped(Event):
    """User disabled their webcam during a Unify Meet session."""

    topic: ClassVar[str | None] = "app:comms:user_webcam_stopped"

    reason: str = ""


@dataclass
class UserRemoteControlStarted(Event):
    """User took remote control of the assistant's desktop.

    The user now has mouse and keyboard control. The actor should pause
    computer-related execution to avoid conflicting with user input.
    """

    topic: ClassVar[str | None] = "app:comms:user_remote_control_started"

    reason: str = ""


@dataclass
class UserRemoteControlStopped(Event):
    """User released remote control of the assistant's desktop.

    The actor may resume computer-related execution.
    """

    topic: ClassVar[str | None] = "app:comms:user_remote_control_stopped"

    reason: str = ""


@dataclass
class SyncContacts(Event):
    """Signal to re-sync system contacts from the API (assistant, user, org members)."""

    reason: str = ""


@dataclass
class BackupContactsEvent(Event):
    """
    Fallback contacts from inbound messages for use before ContactManager initializes.

    When an inbound message arrives before the ContactManager is ready, this event
    carries the contacts list so they can be cached locally in ContactIndex. Once
    ContactManager is initialized, this local cache is cleared and all contact
    lookups go through ContactManager.
    """

    loggable: ClassVar[bool] = False
    contacts: list[dict[str, Any]]


@dataclass
class LLMUserMessage(Event):
    content: str


@dataclass
class LLMAssistantMessage(Event):
    content: str


@dataclass
class SummarizeContext(Event):
    pass


@dataclass
class DirectMessageEvent(Event):
    """
    Send a message directly to the user via the current medium,
    bypassing the Main CM Brain's decision-making.

    Used by ConversationManagerHandle.ask for questions and acknowledgments
    that should be delivered verbatim without LLM processing.
    """

    content: str
    source: str = "system"
