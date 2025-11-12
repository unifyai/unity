import json
import uuid
from typing import Optional, Any, ClassVar
from datetime import datetime
from dataclasses import dataclass, asdict, field

from pydantic import BaseModel



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
    timestamp: datetime = field(default_factory=datetime.now)

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
        cls = cls._registry.get(data["event_name"])
        if not cls:
            raise Exception(f"Class {data['event_name']} is not registered.")
        kwargs = data["payload"].copy()
        timestamp = kwargs.pop("timestamp")
        return cls(**kwargs, timestamp=datetime.fromisoformat(timestamp))

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
class UnifyCallReceived(Event):
    """Frontend/worker confirmed agent connected to room; begin LLM."""

    contact: dict
    agent_name: str | None = None
    room_name: str | None = None


@dataclass
class PhoneCallStarted(Event):
    contact: dict


@dataclass
class UnifyCallStarted(Event):
    """A browser-based voice call session has started (no phone number).

    "contact" should reference the boss/user contact id (typically 1).
    """

    contact: dict


@dataclass
class PhoneUtterance(Event):
    contact: dict
    content: str


@dataclass
class UnifyCallUtterance(Event):
    """User utterance during a browser-based voice call session."""

    contact: dict
    content: str


@dataclass
class Interrupt(Event):
    contact: dict


@dataclass
class PhoneCallEnded(Event):
    contact: dict


@dataclass
class UnifyCallEnded(Event):
    """The browser-based voice call session has ended."""

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
class AssistantPhoneUtterance(Event):
    contact: dict
    content: str


@dataclass
class AssistantUnifyCallUtterance(Event):
    """Assistant utterance during a browser-based voice call session."""

    contact: dict
    content: str


@dataclass
class EmailReceived(Event):
    contact: dict
    subject: str
    body: str
    message_id: Optional[str] = None


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
    message_id: str | None = None


@dataclass
class StartupEvent(Event):
    loggable: ClassVar[bool] = False
    api_key: str
    medium: str
    assistant_id: str
    user_id: str
    assistant_name: str
    assistant_age: str
    assistant_region: str
    assistant_about: str
    assistant_number: str
    assistant_email: str
    user_name: str
    user_number: str
    user_whatsapp_number: str
    user_email: str
    voice_id: str
    voice_provider: str = "cartesia"


@dataclass
class AssistantUpdateEvent(Event):
    loggable: ClassVar[bool] = False
    api_key: str
    medium: str
    assistant_id: str
    user_id: str
    assistant_name: str
    assistant_age: str
    assistant_region: str
    assistant_about: str
    assistant_number: str
    assistant_email: str
    user_name: str
    user_number: str
    user_whatsapp_number: str
    user_email: str
    voice_id: str
    voice_provider: str = "cartesia"


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
class GetBusEventsResponse(Event):
    loggable: ClassVar[bool] = False
    events: list[dict[str, Any]]

    def __str__(self) -> str:
        return self._repr_truncated()

    def __repr__(self) -> str:
        return self._repr_truncated()

    def _repr_truncated(self) -> str:
        return f"{self.__class__.__name__}(events_len={len(self.events)})"

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