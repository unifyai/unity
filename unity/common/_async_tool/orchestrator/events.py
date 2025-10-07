from __future__ import annotations

from enum import Enum
from typing import TypedDict, Literal, Union, Any, Optional


class State(Enum):
    IDLE = "idle"
    WAITING_LLM = "waiting_llm"
    RUNNING_TOOLS = "running_tools"
    AWAITING_CLARIFICATION = "awaiting_clarification"
    PAUSED = "paused"
    CANCELLING = "cancelling"
    COMPLETED = "completed"


class _BaseEvent(TypedDict):
    type: str


class LLMCompletedEvent(_BaseEvent, total=False):
    type: Literal["llm_completed"]
    message: dict


class LLMFailedEvent(_BaseEvent, total=False):
    type: Literal["llm_failed"]
    error: str


class LLMPreemptedEvent(_BaseEvent, total=False):
    type: Literal["llm_preempted"]


class ToolCompletedEvent(_BaseEvent, total=False):
    type: Literal["tool_completed"]
    call_id: str
    name: str
    result: Any


class ToolFailedEvent(_BaseEvent, total=False):
    type: Literal["tool_failed"]
    call_id: str
    name: str
    error: str


class ClarificationRequestedEvent(_BaseEvent, total=False):
    type: Literal["clarification_requested"]
    call_id: str
    tool_name: str
    question: str


class NotificationReceivedEvent(_BaseEvent, total=False):
    type: Literal["notification_received"]
    call_id: str
    tool_name: str
    message: str


class InterjectedEvent(_BaseEvent, total=False):
    type: Literal["interjected"]
    content: Any


class PauseRequestedEvent(_BaseEvent, total=False):
    type: Literal["pause_requested"]


class ResumeRequestedEvent(_BaseEvent, total=False):
    type: Literal["resume_requested"]


class CancelRequestedEvent(_BaseEvent, total=False):
    type: Literal["cancel_requested"]
    reason: Optional[str]


class TimeoutEvent(_BaseEvent, total=False):
    type: Literal["timeout"]


Event = Union[
    LLMCompletedEvent,
    LLMFailedEvent,
    LLMPreemptedEvent,
    ToolCompletedEvent,
    ToolFailedEvent,
    ClarificationRequestedEvent,
    NotificationReceivedEvent,
    InterjectedEvent,
    PauseRequestedEvent,
    ResumeRequestedEvent,
    CancelRequestedEvent,
    TimeoutEvent,
]
