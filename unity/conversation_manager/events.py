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

    def to_bus_event(self):
        from unity.events.event_bus import Event as BusEvent

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
    def from_bus_event(cls, event) -> "Event":
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


class StartupEvent(Event):
    def __init__(self, **kwargs):
        self.api_key = kwargs.pop("api_key")
        self.assistant_id = kwargs.pop("assistant_id")
        self.assistant_name = kwargs.pop("assistant_name")
        self.assistant_age = kwargs.pop("assistant_age")
        self.assistant_region = kwargs.pop("assistant_region")
        self.assistant_about = kwargs.pop("assistant_about")
        self.assistant_number = kwargs.pop("assistant_number")
        self.user_name = kwargs.pop("user_name")
        self.user_number = kwargs.pop("user_number")
        self.user_phone_number = kwargs.pop("user_phone_number")
        self.user_email = kwargs.pop("user_email")
        super().__init__(**kwargs)

    def to_dict(self) -> dict[str, Any]:
        base_dict = super().to_dict()
        base_dict["payload"].update(
            {
                "api_key": self.api_key,
                "assistant_id": self.assistant_id,
                "assistant_name": self.assistant_name,
                "assistant_age": self.assistant_age,
                "assistant_region": self.assistant_region,
                "assistant_about": self.assistant_about,
                "assistant_number": self.assistant_number,
                "user_name": self.user_name,
                "user_number": self.user_number,
                "user_phone_number": self.user_phone_number,
                "user_email": self.user_email,
            },
        )
        return base_dict

    def __str__(self):
        return f"[Startup... @ {self.fmt_timestamp} for assistant {self.assistant_id}]"


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


class EmailSentEvent(_Message):
    platform = "Email"
    direction = "Sent"


class WhatsappMessageRecievedEvent(_Message):
    platform = "Whatsapp"
    direction = "Recieved"


class SMSMessageRecievedEvent(_Message):
    platform = "SMS"
    direction = "Recieved"


class EmailRecievedEvent(_Message):
    platform = "Email"
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
    def __init__(
        self,
        purpose: str = None,
        task_context: Dict[str, str] = None,
        target_number: str = None,
        meet_id: str = None,
        **kwargs,
    ):
        kwargs.pop("content", None)
        kwargs.pop("purpose", None)
        kwargs.pop("task_context", None)
        kwargs.pop("target_number", None)
        kwargs.pop("meet_id", None)

        self.purpose = purpose if purpose else "general"
        self.task_context = task_context
        self.target_number = target_number
        self.meet_id = meet_id
        super().__init__(**kwargs)

    def to_dict(self) -> dict[str, Any]:
        base_dict = super().to_dict()
        base_dict["payload"].update(
            {
                "purpose": self.purpose,
                "task_context": self.task_context,
                "target_number": self.target_number,
                "meet_id": self.meet_id,
            },
        )
        return base_dict

    def __str__(self):
        return f"[Phone Call Initiated... to {self.target_number} {'(Meet: ' + self.meet_id + ')' if self.meet_id else ''} @ {self.fmt_timestamp} for purpose {self.purpose} with task context {self.task_context}]"


class PhoneCallEndedEvent(Event):
    def __str__(self):
        return f"[Phone Call Ended @ {self.fmt_timestamp}]"


class PhoneCallStopEvent(Event):
    def __str__(self):
        return f"[Phone Call Stopped @ {self.fmt_timestamp}]"


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


class ToolUseStartedEvent(Event):
    def __init__(
        self,
        chat_history: list[dict[str, str]],
        query: str,
        *,
        is_urgent: bool = True,
        **kwargs,
    ):
        kwargs.pop("chat_history", None)
        kwargs.pop("query", None)
        kwargs.pop("is_urgent", None)

        self.chat_history = chat_history
        self.query = query
        super().__init__(is_urgent=is_urgent, **kwargs)

    def to_dict(self) -> dict[str, Any]:
        base_dict = super().to_dict()
        base_dict["payload"].update(
            {"chat_history": self.chat_history, "query": self.query},
        )
        return base_dict

    def __str__(self):
        return f"""[TOOL_USE STARTED @ {self.fmt_timestamp}]
        {self.query}"""


class ToolUseEndedEvent(Event):
    def __init__(self, query: str, *, is_urgent: bool = True, **kwargs):
        kwargs.pop("query", None)
        kwargs.pop("is_urgent", None)

        self.query = query
        super().__init__(is_urgent=is_urgent, **kwargs)

    def to_dict(self) -> dict[str, Any]:
        base_dict = super().to_dict()
        base_dict["payload"].update({"query": self.query})
        return base_dict

    def __str__(self):
        return f"""[TOOL_USE ENDED @ {self.fmt_timestamp}]
        {self.query}"""


class ToolUseHandleSuccessEvent(Event):
    def __init__(
        self,
        query: str,
        handle_type: str,
        *,
        is_urgent: bool = True,
        **kwargs,
    ):
        kwargs.pop("query", None)
        kwargs.pop("handle_type", None)
        kwargs.pop("is_urgent", None)

        self.query = query
        self.handle_type = handle_type
        super().__init__(is_urgent=is_urgent, **kwargs)

    def to_dict(self) -> dict[str, Any]:
        base_dict = super().to_dict()
        base_dict["payload"].update(
            {"query": self.query, "handle_type": self.handle_type},
        )
        return base_dict

    def __str__(self):
        return f"""[TOOL_USE HANDLE ACTION @ {self.fmt_timestamp}]
        {self.handle_type}: {self.query}"""


class ToolUseHandleFailedEvent(Event):
    def __init__(self, query: str, handle_type: str, **kwargs):
        kwargs.pop("query", None)
        kwargs.pop("handle_type", None)

        self.query = query
        self.handle_type = handle_type
        super().__init__(**kwargs)

    def to_dict(self) -> dict[str, Any]:
        base_dict = super().to_dict()
        base_dict["payload"].update(
            {"query": self.query, "handle_type": self.handle_type},
        )
        return base_dict

    def __str__(self):
        return f"""[TOOL_USE HANDLE FAILED @ {self.fmt_timestamp}]
        {self.handle_type}: {self.query}"""


# Public variable with all event class names (excluding internal/abstract ones)
EVENT_TYPES = [
    name
    for name in _EventRegistry._registry.keys()
    if not name.startswith("_") and name != "Event"
]
