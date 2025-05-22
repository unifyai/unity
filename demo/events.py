from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Type


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


class TelegramMessageSentEvent(_Message):
    platform = "Telegram"
    direction = "Sent"


class SMSMessageSentEvent(_Message):
    platform = "SMS"
    direction = "Sent"


class WhatsappMessageRecievedEvent(_Message):
    platform = "Whatsapp"
    direction = "Recieved"


class TelegramMessageRecievedEvent(_Message):
    platform = "Telegram"
    direction = "Recieved"


class SMSMessageRecievedEvent(_Message):
    platform = "SMS"
    direction = "Recieved"


class PhoneCallStartedEvent(Event):
    def __init__(self, content: str, **kwargs):
        kwargs.pop("content", None)
        super().__init__(content=content, **kwargs)

    def __str__(self):
        return f"[Phone Call Started @ {self.fmt_timestamp}]"


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

    def __str__(self):
        return f"[INTERRUPT @ {self.fmt_timestamp}] User interrupted"
