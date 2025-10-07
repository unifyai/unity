from typing import TypedDict, Literal
from dataclasses import dataclass, field
import asyncio
import time
from typing import Any
import re
from .loop_config import LIVE_IMAGES_REGISTRY, LIVE_IMAGES_LOG


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


# ── Helpers for source-scoped keys (e.g. "user_message[0:10]", "this[:]") ───
_SRC_SPAN_RX = re.compile(
    r"^(?P<src>(this|user_message|interjection\d+|ask\d+|clar_request\d+|clar_answer\d+|notification\d+))\[(?P<start>-?\d+)?\:(?P<end>-?\d+)?\]$",
)


def parse_source_scoped_span(key: str) -> tuple[str, str] | None:
    """
    Parse a key of the form "<source>[start:end]" where <source> is one of:
    this, user_message, interjectionN, askN, clar_requestN, clar_answerN, notificationN.
    Return (source, "[start:end]") when valid; else None.
    """
    try:
        m = _SRC_SPAN_RX.fullmatch(str(key))
        if not m:
            return None
        source = m.group("src")
        span = key[key.find("[") :]
        return source, span
    except Exception:
        return None


def append_source_scoped_images(images: dict | None, default_source_label: str) -> None:
    """
    Append `images` (source-scoped mapping) into the loop's live image registry and log.

    Behaviour
    ---------
    - Accepts mapping of key → value where key is either `<source>[start:end]` or omitted
      (treated as `this[:]`), and value is an image id or an ImageHandle.
    - Resolves ids using LIVE_IMAGES_REGISTRY, appends handles idempotently.
    - Records a compact log entry "<source>:<id>:[start:end]" for overview display.
    - If `<source>` is literally `this`, it is mapped to `default_source_label`.
    """
    try:
        if not isinstance(images, dict) or not images:
            return
        reg = LIVE_IMAGES_REGISTRY.get()
        log = LIVE_IMAGES_LOG.get()
        for k, v in images.items():
            parsed = parse_source_scoped_span(str(k))
            if parsed:
                src, span = parsed
                if src == "this":
                    src = default_source_label
            else:
                src, span = default_source_label, "[:]"

            handle = None
            try:
                if isinstance(v, int):
                    handle = reg.get(int(v)) if isinstance(reg, dict) else None
                elif hasattr(v, "image_id"):
                    handle = v
            except Exception:
                handle = None
            if handle is None:
                continue
            try:
                reg[int(getattr(handle, "image_id", -1))] = handle
            except Exception:
                pass
            try:
                log.append(f"{src}:{int(getattr(handle, 'image_id', -1))}:{span}")
            except Exception:
                pass
    except Exception:
        return


def next_source_index(prefix: str) -> int:
    """Return the next numeric index for a given source prefix based on LIVE_IMAGES_LOG."""
    try:
        log = LIVE_IMAGES_LOG.get()
        if not isinstance(log, list):
            return 0
        return sum(1 for e in log if isinstance(e, str) and e.startswith(prefix))
    except Exception:
        return 0


def default_source_label(prefix: str) -> str:
    """Return a default `<prefix>N` label using the next available index."""
    return f"{prefix}{next_source_index(prefix)}"
