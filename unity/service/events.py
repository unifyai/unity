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


# Task events
class CommsTaskCreatedEvent(Event):
    def __init__(
        self,
        contact_name: str,
        contact_number: str,
        task_desc: str,
        agent_id: str,
        task_id: str,
        *args,
        **kwargs,
    ):
        # Remove potential duplicates coming from deserialisation
        kwargs.pop("contact_name", None)
        kwargs.pop("contact_number", None)
        kwargs.pop("task_desc", None)
        kwargs.pop("agent_id", None)

        self.agent_id = agent_id
        self.task_id = task_id
        self.contact_name = contact_name
        self.contact_number = contact_number
        self.task_desc = task_desc
        super().__init__(*args, **kwargs)

    def to_dict(self) -> dict[str, Any]:
        base_dict = super().to_dict()
        base_dict["payload"].update(
            {
                "agent_id": self.agent_id,
                "task_id": self.task_id,
                "contact_name": self.contact_name,
                "contact_number": self.contact_number,
                "task_desc": self.task_desc,
            },
        )
        return base_dict

    def __str__(self):
        return f"""[COMMS TASK CREATED AND HANDLED BY AGENT ID: {self.agent_id} @ {self.fmt_timestamp}]
TASK CONTACT NAME: {self.contact_name}
TASK CONTACT NUMBER: {self.contact_number}
TASK DESC: {self.task_desc}
"""


class CommsTaskStartedEvent(Event):
    def __init__(
        self,
        contact_name: str,
        contact_number: str,
        task_desc: str,
        agent_id: str,
        task_id: str,
        *args,
        **kwargs,
    ):
        # Remove potential duplicates coming from deserialisation
        kwargs.pop("contact_name", None)
        kwargs.pop("contact_number", None)
        kwargs.pop("task_desc", None)
        kwargs.pop("agent_id", None)
        kwargs.pop("task_id", None)

        self.agent_id = agent_id
        self.task_id = task_id
        self.contact_name = contact_name
        self.contact_number = contact_number
        self.task_desc = task_desc
        super().__init__(*args, **kwargs)

    def to_dict(self) -> dict[str, Any]:
        base_dict = super().to_dict()
        base_dict["payload"].update(
            {
                "agent_id": self.agent_id,
                "task_id": self.task_id,
                "contact_name": self.contact_name,
                "contact_number": self.contact_number,
                "task_desc": self.task_desc,
            },
        )
        return base_dict

    def __str__(self):
        return f"""[COMMS TASK CREATED @ {self.fmt_timestamp}]
TASK CONTACT NAME: {self.contact_name}
TASK CONTACT NUMBER: {self.contact_number}
TASK DESC: {self.task_desc}
"""


class CommsTaskDoneEvent(Event):
    def __init__(
        self,
        agent_id: str,
        task_id: int,
        task_status: str,
        task_result: str,
        *args,
        **kwargs,
    ):
        # Remove potential duplicates coming from deserialisation
        kwargs.pop("agent_id", None)
        kwargs.pop("task_status", None)
        kwargs.pop("task_result", None)
        kwargs.pop("task_id", None)

        self.agent_id = agent_id
        self.task_id = task_id
        self.task_status = task_status
        self.task_result = task_result
        super().__init__(*args, **kwargs)

    def to_dict(self) -> dict[str, Any]:
        base_dict = super().to_dict()
        base_dict["payload"].update(
            {
                "agent_id": self.agent_id,
                "task_id": self.task_id,
                "task_status": self.task_status,
                "task_result": self.task_result,
            },
        )
        return base_dict

    def __str__(self):
        return f"""[TASK DONE BY AGENT ID: {self.agent_id} @ {self.fmt_timestamp}]
TASK STATUS: {self.task_status}
TASK RESULT: {self.task_result}"""


class AskUserAgentEvent(Event):
    def __init__(self, agent_id: str, task_id: str, query: str, *args, **kwargs):
        kwargs.pop("agent_id", None)
        kwargs.pop("task_id", None)
        kwargs.pop("query", None)

        self.agent_id = agent_id
        self.task_id = task_id
        self.query = query
        super().__init__(*args, **kwargs)

    def to_dict(self) -> dict[str, Any]:
        base_dict = super().to_dict()
        base_dict["payload"].update(
            {
                "agent_id": self.agent_id,
                "task_id": self.task_id,
                "query": self.query,
            },
        )
        return base_dict

    def __str__(self):
        return f"""[AGENT {self.agent_id} NEEDS SOME CLARIFICATION REGARDING THE FOLLOWING QUERY FOR TASK {self.task_id} @ {self.fmt_timestamp}]
{self.query}"""


class UserAgentResponseEvent(Event):
    def __init__(self, task_id: str, response: str, *args, **kwargs):
        kwargs.pop("task_id", None)
        kwargs.pop("response", None)

        self.task_id = task_id
        self.response = response
        super().__init__(*args, **kwargs)

    def to_dict(self) -> dict[str, Any]:
        base_dict = super().to_dict()
        base_dict["payload"].update(
            {
                "task_id": self.task_id,
                "response": self.response,
            },
        )
        return base_dict

    def __str__(self):
        return f"""[USER AGENT RESPONDED TO YOUR QUERY REGARDING {self.task_id} @ {self.fmt_timestamp}]
        {self.response}"""


class ManagerStartedEvent(Event):
    def __init__(
        self,
        agent_id: str,
        manager_name: str,
        chat_history: list[dict[str, str]],
        query: str,
        *args,
        **kwargs,
    ):
        kwargs.pop("agent_id", None)
        kwargs.pop("manager_name", None)
        kwargs.pop("chat_history", None)
        kwargs.pop("query", None)

        self.agent_id = agent_id
        self.manager_name = manager_name
        self.chat_history = chat_history
        self.query = query
        super().__init__(*args, **kwargs, transient=True)

    def to_dict(self) -> dict[str, Any]:
        base_dict = super().to_dict()
        base_dict["payload"].update(
            {
                "agent_id": self.agent_id,
                "manager_name": self.manager_name.upper(),
                "chat_history": self.chat_history,
                "query": self.query,
            },
        )
        return base_dict

    def __str__(self):
        return f"""[{self.manager_name.upper()} MANAGER STARTED @ {self.fmt_timestamp}]
        {self.query}"""


class ManagerInterjectedEvent(Event):
    def __init__(self, agent_id: str, manager_name: str, query: str, *args, **kwargs):
        kwargs.pop("agent_id", None)
        kwargs.pop("manager_name", None)
        kwargs.pop("query", None)

        self.agent_id = agent_id
        self.manager_name = manager_name
        self.query = query
        super().__init__(*args, **kwargs, transient=True)

    def to_dict(self) -> dict[str, Any]:
        base_dict = super().to_dict()
        base_dict["payload"].update(
            {
                "agent_id": self.agent_id,
                "manager_name": self.manager_name.upper(),
                "query": self.query,
            },
        )
        return base_dict

    def __str__(self):
        return f"""[{self.manager_name.upper()} MANAGER INTERJECTED @ {self.fmt_timestamp}]
        {self.query}"""


class ManagerInterjectFailedEvent(Event):
    def __init__(self, agent_id: str, manager_name: str, query: str, *args, **kwargs):
        kwargs.pop("agent_id", None)
        kwargs.pop("manager_name", None)
        kwargs.pop("query", None)

        self.agent_id = agent_id
        self.manager_name = manager_name
        self.query = query
        super().__init__(*args, **kwargs, transient=True)

    def to_dict(self) -> dict[str, Any]:
        base_dict = super().to_dict()
        base_dict["payload"].update(
            {
                "agent_id": self.agent_id,
                "manager_name": self.manager_name,
                "query": self.query,
            },
        )
        return base_dict

    def __str__(self):
        return f"""[{self.manager_name.upper()} MANAGER INTERJECT FAILED @ {self.fmt_timestamp}]
        {self.query}"""


class ManagerEndedEvent(Event):
    def __init__(self, agent_id: str, manager_name: str, query: str, *args, **kwargs):
        kwargs.pop("agent_id", None)
        kwargs.pop("manager_name", None)
        kwargs.pop("query", None)

        self.agent_id = agent_id
        self.manager_name = manager_name
        self.query = query
        super().__init__(*args, **kwargs)

    def to_dict(self) -> dict[str, Any]:
        base_dict = super().to_dict()
        base_dict["payload"].update(
            {
                "agent_id": self.agent_id,
                "manager_name": self.manager_name,
                "query": self.query,
            },
        )
        return base_dict

    def __str__(self):
        return f"""[{self.manager_name.upper()} MANAGER ENDED @ {self.fmt_timestamp}]
        {self.query}"""
