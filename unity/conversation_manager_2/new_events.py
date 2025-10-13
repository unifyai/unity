import json
from typing import Optional, Any
from datetime import datetime
from dataclasses import dataclass, asdict, field


from typing import ClassVar


def datetime_aware_dict_factory(kv):
    d = {}
    for k, v in kv:
        if isinstance(v, datetime):
            d[k] = v.isoformat()
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
            "payload": asdict(self, dict_factory=datetime_aware_dict_factory),
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


@dataclass
class PhoneCallRecieved(Event):
    contact: str
    conference_name: str = ""


@dataclass
class PhoneCallStarted(Event):
    contact: str


@dataclass
class PhoneUtterance(Event):
    contact: str
    content: str


@dataclass
class Interrupt(Event):
    contact: str


@dataclass
class PhoneCallEnded(Event):
    contact: str


@dataclass
class SMSRecieved(Event):
    contact: str
    content: str


@dataclass
class PhoneCallSent(Event):
    contact: str


@dataclass
class AssistantPhoneUtterance(Event):
    contact: str
    content: str


@dataclass
class EmailRecieved(Event):
    contact: str
    subject: str
    body: str
    message_id: Optional[str]


# assistant events
@dataclass
class SMSSent(Event):
    contact: str
    content: str


@dataclass
class EmailSent(Event):
    contact: str
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


# managers worker events
@dataclass
class ManagersStartupInput(Event):
    agent_id: str
    first_name: str
    age: str
    region: str
    about: str
    phone: str
    email: str
    user_phone: str
    user_whatsapp_number: str
    assistant_whatsapp_number: str


@dataclass
class ManagersStartupOutput(Event):
    loggable: ClassVar[bool] = False
    initialized: bool


@dataclass
class LogMessageInput(Event):
    medium: str
    sender_id: int
    receiver_ids: list[int]
    content: str
    exchange_id: int
    call_utterance_timestamp: str
    call_url: str
    metadata: dict[str, Any]


@dataclass
class GetContactsInput(Event):
    pass


@dataclass
class CreateContactInput(Event):
    first_name: str
    surname: str
    email_address: str
    phone_number: str


@dataclass
class LogMessageOutput(Event):
    medium: str
    exchange_id: int


@dataclass
class GetContactsOutput(Event):
    contacts: list[dict[str, Any]]


@dataclass
class GetBusEventsInput(Event):
    pass


@dataclass
class GetBusEventsOutput(Event):
    events: list[dict[str, Any]]


@dataclass
class PublishBusEvent(Event):
    event: dict[str, Any]


# --------------------------------------------------------------------------- #
# LLM inference events
# --------------------------------------------------------------------------- #
@dataclass
class LLMInput(Event):
    content: list[dict]


@dataclass
class UpdateContactRollingSummaryRequest(Event):
    contacts_ids: list[int]
    transcripts: list[str]


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
