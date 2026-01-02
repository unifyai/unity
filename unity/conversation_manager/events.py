import json
import uuid
from typing import Optional, Any, ClassVar
from datetime import datetime
from dataclasses import dataclass, asdict, field

from pydantic import BaseModel

from unity.settings import SETTINGS


def _get_now() -> datetime:
    """Return current datetime.

    In test mode (when UNITY_FIXED_DATETIME is set in SETTINGS), returns the
    fixed datetime to ensure LLM cache hits across test runs. The value should
    be an ISO format datetime string (e.g., "2025-06-13T12:00:00+00:00").
    """
    if SETTINGS.UNITY_FIXED_DATETIME:
        return datetime.fromisoformat(SETTINGS.UNITY_FIXED_DATETIME)
    return datetime.now()


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


@dataclass(kw_only=True)
class Event:
    timestamp: datetime = field(default_factory=_get_now)

    _registry: ClassVar[dict[str, "Event"]] = {}
    loggable: ClassVar[bool] = True

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
        event_dump = event.model_dump()
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
    contact: dict
    conference_name: str = ""


@dataclass
class PhoneCallAnswered(Event):
    contact: dict


@dataclass
class UnifyMeetReceived(Event):
    """Frontend/worker confirmed agent connected to room; begin LLM."""

    contact: dict
    agent_name: str | None = None
    room_name: str | None = None


@dataclass
class PhoneCallStarted(Event):
    contact: dict


@dataclass
class UnifyMeetStarted(Event):
    """A browser-based voice/video meeting session has started (no phone number).

    "contact" should reference the boss/user contact id (typically 1).
    """

    contact: dict


@dataclass
class InboundPhoneUtterance(Event):
    """Utterance received from the other party during a phone call."""

    contact: dict
    content: str


@dataclass
class InboundUnifyMeetUtterance(Event):
    """Utterance received from the other party during a browser-based voice/video meeting."""

    contact: dict
    content: str


@dataclass
class VoiceInterrupt(Event):
    """User interrupted the assistant during a voice call."""

    contact: dict


@dataclass
class PhoneCallEnded(Event):
    contact: dict


@dataclass
class UnifyMeetEnded(Event):
    """The browser-based voice/video meeting session has ended."""

    contact: dict


@dataclass
class SMSReceived(Event):
    contact: dict
    content: str


@dataclass
class UnifyMessageReceived(Event):
    contact: dict
    content: str


@dataclass
class PhoneCallSent(Event):
    contact: dict


@dataclass
class OutboundPhoneUtterance(Event):
    """Utterance sent by the assistant during a phone call."""

    contact: dict
    content: str


@dataclass
class OutboundUnifyMeetUtterance(Event):
    """Utterance sent by the assistant during a browser-based voice/video meeting."""

    contact: dict
    content: str


@dataclass
class CallGuidance(Event):
    """
    Guidance from the Main CM Brain sent to the Voice Agent during a call.

    Used in both TTS and STS voice modes. The Voice Agent (fast brain) handles
    all conversational responses autonomously; this guidance provides data,
    notifications, or requests that the Main CM Brain needs to communicate.
    """

    contact: dict
    content: str


@dataclass
class EmailReceived(Event):
    contact: dict
    subject: str
    body: str
    # Email provider identifier used for threading (e.g., RFC Message-ID header value).
    # This is *not* the TranscriptManager's auto-incremented message_id.
    email_id: Optional[str] = None


# assistant events
@dataclass
class SMSSent(Event):
    contact: dict
    content: str


@dataclass
class UnifyMessageSent(Event):
    contact: dict
    content: str


@dataclass
class EmailSent(Event):
    contact: dict
    subject: str
    body: str
    # Email provider identifier used for threading (e.g., RFC Message-ID header value).
    # This is *not* the TranscriptManager's auto-incremented message_id.
    email_id_replied_to: str | None = None


@dataclass
class StartupEvent(Event):
    loggable: ClassVar[bool] = False
    api_key: str
    medium: str
    assistant_id: str
    user_id: str
    assistant_name: str
    assistant_age: str
    assistant_nationality: str
    assistant_about: str
    assistant_number: str
    assistant_email: str
    user_name: str
    user_number: str
    user_email: str
    voice_id: str
    voice_provider: str = "cartesia"
    voice_mode: str = "tts"
    assistant_timezone: str = (
        ""  # IANA timezone identifier; default empty for backward compat
    )


@dataclass
class InitializationComplete(Event):
    """Published when ConversationManager has fully initialized all managers."""

    loggable: ClassVar[bool] = False


@dataclass
class AssistantUpdateEvent(Event):
    loggable: ClassVar[bool] = False
    api_key: str
    medium: str
    assistant_id: str
    user_id: str
    assistant_name: str
    assistant_age: str
    assistant_nationality: str
    assistant_about: str
    assistant_number: str
    assistant_email: str
    user_name: str
    user_number: str
    user_email: str
    voice_id: str
    voice_provider: str = "cartesia"
    voice_mode: str = "tts"
    assistant_timezone: str = (
        ""  # IANA timezone identifier; default empty for backward compat
    )


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
class GetContactsResponse(Event):
    contacts: list[dict[str, Any]]


@dataclass
class ContactInfoResponse(Event):
    contact_details: dict[str, Any]


@dataclass
class StoreChatHistory(Event):
    chat_history: list[dict]

    def __str__(self) -> str:
        return self._repr_truncated()

    def __repr__(self) -> str:
        return self._repr_truncated()

    def _repr_truncated(self) -> str:
        return f"{self.__class__.__name__}(chat_history_len={len(self.chat_history)})"


@dataclass
class GetChatHistory(Event):
    loggable: ClassVar[bool] = False
    chat_history: list[dict]

    def __str__(self) -> str:
        return self._repr_truncated()

    def __repr__(self) -> str:
        return self._repr_truncated()

    def _repr_truncated(self) -> str:
        return f"{self.__class__.__name__}(chat_history_len={len(self.chat_history)})"


@dataclass
class GetBusEventsResponse(Event):
    loggable: ClassVar[bool] = False
    events: list[dict[str, Any]]

    def __str__(self) -> str:
        return self._repr_truncated()

    def __repr__(self) -> str:
        return self._repr_truncated()

    def _repr_truncated(self) -> str:
        return f"{self.__class__.__name__}(events_len={len(self.events)})"


@dataclass
class PreHireMessage(Event):
    content: str
    role: str
    exchange_id: int
    metadata: dict[str, str]


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
# Conductor Events
# --------------------------------------------------------------------------- #


@dataclass
class ConductorRequest(Event):
    """Event to ask or request the Conductor to perform a task."""

    action_name: str
    query: str
    parent_chat_context: list[dict]

    def __str__(self) -> str:
        return self._repr_truncated()

    def __repr__(self) -> str:
        return self._repr_truncated()

    def _repr_truncated(self) -> str:
        return (
            f"{self.__class__.__name__}(action_name={self.action_name}, "
            f"query={self.query}, "
            f"parent_chat_context_len={len(self.parent_chat_context)})"
        )


@dataclass
class ConductorResponse(Event):
    """Event to respond to a Conductor request."""

    handle_id: int
    action_name: str
    query: str
    response: str


@dataclass
class ConductorHandleRequest(Event):
    """Event to any action on an existing Conductor handle."""

    handle_id: int
    action_name: str
    query: str
    parent_chat_context: list[dict]

    def __str__(self) -> str:
        return self._repr_truncated()

    def __repr__(self) -> str:
        return self._repr_truncated()

    def _repr_truncated(self) -> str:
        return (
            f"{self.__class__.__name__}(handle_id={self.handle_id}, "
            f"action_name={self.action_name}, "
            f"query={self.query}, "
            f"parent_chat_context_len={len(self.parent_chat_context)})"
        )


@dataclass
class ConductorHandleResponse(Event):
    """Event to respond to a Conductor handle request."""

    handle_id: int
    action_name: str
    query: str
    response: str
    call_id: str


@dataclass
class ConductorResult(Event):
    """Event to the result of a Conductor task."""

    handle_id: int
    success: bool
    result: dict | str | None = None
    error: str | None = None


@dataclass
class ConductorClarificationRequest(Event):
    """Event to request clarification from the Conductor."""

    handle_id: int
    query: str
    call_id: str


@dataclass
class ConductorClarificationResponse(Event):
    """Event to respond to a Conductor clarification request."""

    handle_id: int
    response: str
    call_id: str


@dataclass
class ConductorNotification(Event):
    """Event to forward a notification from a Conductor handle."""

    handle_id: int
    response: str


@dataclass
class ConductorHandleStarted(Event):
    action_name: str
    handle_id: id
    query: str


@dataclass
class ConductorPauseActor(Event):
    """Signal to pause any in-flight Actor/TaskScheduler execution for the session."""

    reason: str = ""


@dataclass
class ConductorResumeActor(Event):
    """Signal to resume any previously paused Actor/TaskScheduler execution for the session."""

    reason: str = ""


@dataclass
class SyncContacts(Event):
    """Signal to re-sync system contacts from the API (assistant, user, org members)."""

    reason: str = ""


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
