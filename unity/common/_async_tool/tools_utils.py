from typing import TypedDict, Literal
from dataclasses import dataclass, field
import asyncio
import time
from typing import Any


@dataclass
class ToolCallMetadata:
    name: str
    call_id: str
    call_dict: dict
    call_idx: int
    chat_context: Any
    assistant_msg: dict
    is_interjectable: bool
    tool_schema: dict
    llm_arguments: dict
    raw_arguments_json: str
    waiting_for_clarification: bool = False
    tool_reply_msg: dict | None = None
    clarify_placeholder: dict | None = None
    handle: Any | None = None
    interject_queue: asyncio.Queue[dict | str] | None = None
    clar_up_queue: asyncio.Queue[str] | None = None
    clar_down_queue: asyncio.Queue[str] | None = None
    # Optional notification stream emitted by tools; payload is a dict with arbitrary fields
    notification_queue: asyncio.Queue[dict] | None = None
    pause_event: asyncio.Event | None = None
    scheduled_time: float = field(default_factory=time.perf_counter)
    # Whether this task's handle is running in passthrough mode. When true,
    # outer-loop programmatic interject/ask should be propagated downwards,
    # and upward events (clarifications/notifications) should bubble to the
    # outer loop as well.
    is_passthrough: bool = False


class ToolCallMessage(TypedDict):
    role: Literal["tool"]
    tool_call_id: str
    name: str
    content: str


def create_tool_call_message(name: str, call_id: str, content: str) -> ToolCallMessage:
    return {
        "role": "tool",
        "tool_call_id": call_id,
        "name": name,
        "content": content,
    }


# Optional typed event payloads exposed via the outer handle
class ClarificationEvent(TypedDict):
    type: Literal["clarification"]
    call_id: str
    tool_name: str
    question: str


class NotificationEvent(TypedDict, total=False):
    type: Literal["notification"]
    call_id: str
    tool_name: str
    message: str
    percent: float
    meta: Any


# All image-related helpers have been moved to `images.py`.
