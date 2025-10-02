from typing import TypedDict, Literal
from dataclasses import dataclass, field
import asyncio
import time
from typing import Any
import re


@dataclass
class ToolCallMetadata:
    name: str
    call_id: str
    call_dict: dict
    call_idx: int
    chat_context: str
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


# ── Helpers for arg-scoped span keys (e.g. "question[2:9]") ─────────────———
_ARG_SPAN_RX = re.compile(
    r"^(?P<arg>[A-Za-z_]\w*)\[(?P<start>-?\d+)?\:(?P<end>-?\d+)?\]$",
)


def parse_arg_scoped_span(key: str) -> tuple[str, str] | None:
    """
    Parse a key of the form "<arg_name>[start:end]" and return
    (arg_name, "[start:end]") when valid; else None.

    The bracket portion preserves the original indices; downstream helpers can
    compute concrete ranges or substrings with Python-slice semantics.
    """
    try:
        m = _ARG_SPAN_RX.fullmatch(str(key))
        if not m:
            return None
        arg = m.group("arg")
        span = key[key.find("[") :]
        return arg, span
    except Exception:
        return None


def extract_alignment_text_from_value(value: Any) -> str:
    """
    Return a best-effort string to align spans against using the shared rules:
      - str: use as-is
      - dict: use str(value.get("content", ""))
      - list: if chat messages → first with role=="user"; else first text block
    """
    try:
        if isinstance(value, str):
            return value
        if isinstance(value, dict):
            return str(value.get("content", ""))
        if isinstance(value, list):
            # Case 1: chat messages
            for m in value:
                if isinstance(m, dict) and m.get("role") == "user":
                    c = m.get("content")
                    if isinstance(c, list):
                        parts: list[str] = []
                        for it in c:
                            if isinstance(it, dict) and it.get("type") == "text":
                                parts.append(str(it.get("text", "")))
                            else:
                                parts.append(str(it))
                        return "".join(parts)
                    return str(c)
            # Case 2: content blocks (no roles)
            for it in value:
                if isinstance(it, dict) and it.get("type") == "text":
                    return str(it.get("text", ""))
            return str(value[0]) if value else ""
        # Fallback best-effort stringification
        return str(value)
    except Exception:
        return ""
