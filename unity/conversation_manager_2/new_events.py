import json
from typing import Optional
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

    def to_json(self):
        return json.dumps(self.to_dict())

    def to_dict(self):
        return {
            "event_name": self.__class__.__name__,
            "payload": asdict(self, dict_factory=datetime_aware_dict_factory),
        }

    @classmethod
    def from_dict(cls, data):
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

    def __init_subclass__(cls):
        if cls.__name__ not in Event._registry:
            Event._registry[cls.__name__] = cls
        return cls


@dataclass
class PhoneCallInitiated(Event):
    contact: str


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
    message_id: Optional[str]


@dataclass
class StartupEvent(Event):
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
    contact: str | None = None


@dataclass
class Ping(Event):
    kind: str
