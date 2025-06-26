from __future__ import annotations

from datetime import datetime, timezone
import json
from typing import Any, Dict, Type
from unity.events.event_bus import Event as BusEvent


class _EventRegistry(type):
    """Metaclass that keeps a registry mapping event class names to the class itself."""

    _registry: Dict[str, Type["Event"]] = {}

    def __new__(mcls, name, bases, namespace, **kwargs):
        cls = super().__new__(mcls, name, bases, namespace, **kwargs)
        if name != "Event":
            _EventRegistry._registry[name] = cls
        return cls

    @classmethod
    def get(cls, name: str) -> Type["Event"] | None:
        return cls._registry.get(name)


class Event(metaclass=_EventRegistry):
    """Base event class with symmetric to_dict / from_dict helpers."""

    @staticmethod
    def _parse_timestamp(ts: Any | None) -> datetime:
        if ts is None:
            return datetime.now(timezone.utc)
        if isinstance(ts, datetime):
            return ts
        if isinstance(ts, str):
            try:
                return datetime.fromisoformat(ts)
            except ValueError:
                return datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
        raise TypeError(f"Unsupported timestamp type: {type(ts)}")

    def __init__(
        self,
        *,
        timestamp: datetime | str | None = None,
        is_urgent: bool = False,
        transient: bool = False,
        content: str | None = None,
        role: str | None = None,
    ):
        self.timestamp = self._parse_timestamp(timestamp)
        self.fmt_timestamp = self.timestamp.strftime("%Y-%m-%d %H:%M:%S")
        self.is_urgent = is_urgent
        self.transient = transient
        self.content = content
        self.role = role

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "timestamp": self.timestamp.isoformat(),
            "is_urgent": self.is_urgent,
            "transient": self.transient,
            "content": self.content,
            "role": self.role,
        }
        return {"event_name": self.__class__.__name__, "payload": payload}

    def to_bus_event(self) -> BusEvent:
        payload = self.to_dict()["payload"]
        return BusEvent(
            calling_id="",
            type=self.__class__.__name__,
            timestamp=self.timestamp.isoformat(),
            payload=payload,
            payload_cls=self.__class__.__name__,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Event":
        # If wrapper present, pick subclass and recurse
        if "event_name" in data:
            event_cls = _EventRegistry.get(data["event_name"])
            if event_cls is None:
                raise ValueError(f"Unknown event_name {data['event_name']}")
            return event_cls.from_dict(data["payload"])
        # We are now dealing with payload only
        payload = {**data}
        if "timestamp" in payload:
            payload["timestamp"] = cls._parse_timestamp(payload["timestamp"])
        return cls(**payload)  # type: ignore[arg-type]

    @classmethod
    def from_bus_event(cls, event: BusEvent) -> "Event":
        event_dump = event.model_dump()
        data = {"event_name": event_dump["type"], "payload": event_dump["payload"]}
        return cls.from_dict(data)

    def humanize_time_ago(self) -> str:
        now = datetime.now(timezone.utc) if self.timestamp.tzinfo else datetime.now()
        seconds = int((now - self.timestamp).total_seconds())
        if seconds <= 5:
            return "Now"
        periods = [
            ("year", 60 * 60 * 24 * 365),
            ("month", 60 * 60 * 24 * 30),
            ("week", 60 * 60 * 24 * 7),
            ("day", 60 * 60 * 24),
            ("hour", 60 * 60),
            ("minute", 60),
        ]
        for name, length in periods:
            if seconds >= length:
                count = seconds // length
                return f"{count} {name}{'s' if count != 1 else ''} ago"
        return f"{seconds} seconds ago"

    def __str__(self):
        return f"[{self.__class__.__name__} @ {self.fmt_timestamp}]"


class UserTyping(Event):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, transient=True)


class _Message(Event):
    platform: str = ""
    direction: str = "Sent"

    def __str__(self):
        sender = "Assistant" if self.direction == "Sent" else "User"
        return f'[{self.platform} Message {self.direction} @ {self.fmt_timestamp}] {sender}: "{self.content}"'


class WhatsappMessageSentEvent(_Message):
    platform = "Whatsapp"
    direction = "Sent"


class SMSMessageSentEvent(_Message):
    platform = "SMS"
    direction = "Sent"


class WhatsappMessageRecievedEvent(_Message):
    platform = "Whatsapp"
    direction = "Recieved"


class SMSMessageRecievedEvent(_Message):
    platform = "SMS"
    direction = "Recieved"


# this should be either done by user or assistant, should
# make variants (cleanly)
class PhoneCallStartedEvent(Event):
    def __init__(self, **kwargs):
        # kwargs.pop("content", None)
        super().__init__(**kwargs)

    def __str__(self):
        return f"[Phone Call Started @ {self.fmt_timestamp}]"


# this should be either done by user or assistant, should
# make variants (cleanly)
class PhoneCallInitiatedEvent(Event):
    def __init__(self, **kwargs):
        kwargs.pop("content", None)
        super().__init__(**kwargs)

    def __str__(self):
        return f"[Phone Call Initiated... @ {self.fmt_timestamp}]"


class PhoneCallInitiatedCustomEvent(Event):
    def __init__(self, contact_id: int, purpose: str, **kwargs):
        kwargs.pop("content", None)
        kwargs.pop("purpose", None)
        kwargs.pop("contact_id", None)

        self.contact_id = contact_id
        self.purpose = purpose
        super().__init__(**kwargs)

    def to_dict(self) -> dict[str, Any]:
        base_dict = super().to_dict()
        base_dict["payload"].update(
            {"contact_id": self.contact_id, "purpose": self.purpose},
        )
        return base_dict

    def __str__(self):
        return f"[Phone Call Initiated to ID {self.contact_id} for {self.purpose}... @ {self.fmt_timestamp}]"


class PhoneCallEndedEvent(Event):
    def __str__(self):
        return f"[Phone Call Ended @ {self.fmt_timestamp}]"


class PhoneUtteranceEvent(Event):
    def __init__(self, role: str, content: str, *, is_urgent: bool = True, **kwargs):
        """Phone utterances are *always* urgent by default but allow override."""
        # Remove potential duplicates coming from deserialisation
        kwargs.pop("role", None)
        kwargs.pop("content", None)
        kwargs.pop("is_urgent", None)
        super().__init__(role=role, content=content, is_urgent=is_urgent, **kwargs)

    def __str__(self):
        return f'[Phone Utterance @ {self.fmt_timestamp}] {self.role}: "{self.content}"'


class InterruptEvent(Event):
    def __str__(self):
        return f"[INTERRUPT @ {self.fmt_timestamp}] User interrupted"


class ConductorStartedEvent(Event):
    def __init__(
        self,
        chat_history: list[dict[str, str]],
        query: str,
        *args,
        **kwargs,
    ):
        kwargs.pop("chat_history", None)
        kwargs.pop("query", None)

        self.chat_history = chat_history
        self.query = query
        super().__init__(*args, **kwargs)

    def to_dict(self) -> dict[str, Any]:
        base_dict = super().to_dict()
        base_dict["payload"].update(
            {"chat_history": self.chat_history, "query": self.query},
        )
        return base_dict

    def __str__(self):
        return f"""[CONDUCTOR STARTED @ {self.fmt_timestamp}]
        {self.query}"""


class ConductorProgressEvent(Event):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def to_dict(self) -> dict[str, Any]:
        base_dict = super().to_dict()
        base_dict["payload"].update({"query": self.query})
        return base_dict

    def __str__(self):
        return f"""[CONDUCTOR PROGRESS @ {self.fmt_timestamp}]
        {json.dumps(self.payload)}"""


class ConductorInterjectedEvent(Event):
    def __init__(self, query: str, *args, **kwargs):
        kwargs.pop("query", None)

        self.query = query
        super().__init__(*args, **kwargs)

    def to_dict(self) -> dict[str, Any]:
        base_dict = super().to_dict()
        base_dict["payload"].update({"query": self.query})
        return base_dict

    def __str__(self):
        return f"""[CONDUCTOR INTERJECTED @ {self.fmt_timestamp}]
        {self.query}"""


class ConductorInterjectFailedEvent(Event):
    def __init__(self, query: str, *args, **kwargs):
        kwargs.pop("query", None)

        self.query = query
        super().__init__(*args, **kwargs)

    def to_dict(self) -> dict[str, Any]:
        base_dict = super().to_dict()
        base_dict["payload"].update({"query": self.query})
        return base_dict

    def __str__(self):
        return f"""[CONDUCTOR INTERJECT FAILED @ {self.fmt_timestamp}]
        {self.query}"""


class ConductorEndedEvent(Event):
    def __init__(self, manager_name: str, query: str, *args, **kwargs):
        kwargs.pop("manager_name", None)
        kwargs.pop("query", None)

        self.manager_name = manager_name
        self.query = query
        super().__init__(*args, **kwargs)

    def to_dict(self) -> dict[str, Any]:
        base_dict = super().to_dict()
        base_dict["payload"].update({"query": self.query})
        return base_dict

    def __str__(self):
        return f"""[CONDUCTOR ENDED @ {self.fmt_timestamp}]
        {self.query}"""
