"""
IPC protocol (UI ↔ worker) for the multi-process ConversationManager sandbox.

This module is intentionally **pure**:
- No imports from Textual (`gui.py`) or worker implementation (`gui_worker.py`)
- No dependency on `cm_init` / `CommandRouter` / other sandbox wiring

It defines:
- Message type constants
- Explicit schemas for all message types
- Serialization helpers for Pydantic events
- Validation / parsing helpers

Protocol versioning:
    The protocol is versioned to allow future evolution without silent breakage.
    Both UI and worker should include `protocol_version` in their initial handshake
    or logs, but the message schema itself is stable and self-describing via `type`.

Message envelope:
    {
        "type": str,        # discriminator
        "id": str | None,   # correlation id (optional)
        "payload": dict     # type-specific payload
    }

Examples
--------

UI → Worker (execute_raw):

    {
        "type": "execute_raw",
        "id": "550e8400-e29b-41d4-a716-446655440000",
        "payload": {"raw": "/ask what are you doing?", "in_call": false}
    }

Worker → UI (lines):

    {
        "type": "lines",
        "id": "550e8400-e29b-41d4-a716-446655440000",
        "payload": {"lines": ["[Actor] started: ...", "[Phone → User] ..."]}
    }

Worker → UI (event):

    {
        "type": "event",
        "id": null,
        "payload": {
            "channel": "app:actor:notification",
            "event": {"handle_id": 123, "response": "Found 3 contacts"}
        }
    }
"""

from __future__ import annotations

import uuid
from typing import Any, Dict, Literal, Union, cast

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

PROTOCOL_VERSION = "1.0"


class MessageType:
    """String constants for IPC message types."""

    # UI → Worker
    EXECUTE_RAW = "execute_raw"
    SHUTDOWN = "shutdown"

    # Worker → UI
    READY = "ready"
    LINES = "lines"
    STATE = "state"
    EVENT = "event"
    ERROR = "error"
    WORKER_EXIT = "worker_exit"


# -----------------------------------------------------------------------------
# Payload schemas
# -----------------------------------------------------------------------------


class ExecuteRawPayload(BaseModel):
    """Payload for `execute_raw` (UI → Worker)."""

    model_config = ConfigDict(extra="forbid")
    raw: str
    in_call: bool = False


class ShutdownPayload(BaseModel):
    """Payload for `shutdown` (UI → Worker)."""

    model_config = ConfigDict(extra="forbid")


class ReadyPayload(BaseModel):
    """Payload for `ready` (Worker → UI)."""

    model_config = ConfigDict(extra="forbid")


class LinesPayload(BaseModel):
    """Payload for `lines` (Worker → UI)."""

    model_config = ConfigDict(extra="forbid")
    lines: list[str] = Field(default_factory=list)


class StatePayload(BaseModel):
    """Payload for `state` (Worker → UI)."""

    model_config = ConfigDict(extra="forbid")
    active: bool
    in_call: bool
    pending_clarification: bool


class EventPayload(BaseModel):
    """Payload for `event` (Worker → UI)."""

    model_config = ConfigDict(extra="forbid")
    channel: str
    event: Dict[str, Any]


class ErrorPayload(BaseModel):
    """Payload for `error` (Worker → UI)."""

    model_config = ConfigDict(extra="forbid")
    message: str
    traceback: str = ""


class WorkerExitPayload(BaseModel):
    """Payload for `worker_exit` (Worker → UI)."""

    model_config = ConfigDict(extra="forbid")
    restart: bool
    config: Dict[str, Any] | None = None


# -----------------------------------------------------------------------------
# Message schemas (explicit for all 8 types)
# -----------------------------------------------------------------------------


class ExecuteRawMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal[MessageType.EXECUTE_RAW]
    id: str | None = None
    payload: ExecuteRawPayload


class ShutdownMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal[MessageType.SHUTDOWN]
    id: str | None = None
    payload: ShutdownPayload = Field(default_factory=ShutdownPayload)


class ReadyMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal[MessageType.READY]
    id: str | None = None
    payload: ReadyPayload = Field(default_factory=ReadyPayload)


class LinesMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal[MessageType.LINES]
    id: str | None = None
    payload: LinesPayload


class StateMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal[MessageType.STATE]
    id: str | None = None
    payload: StatePayload


class EventMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal[MessageType.EVENT]
    id: str | None = None
    payload: EventPayload


class ErrorMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal[MessageType.ERROR]
    id: str | None = None
    payload: ErrorPayload


class WorkerExitMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal[MessageType.WORKER_EXIT]
    id: str | None = None
    payload: WorkerExitPayload


AnyMessage = Union[
    ExecuteRawMessage,
    ShutdownMessage,
    ReadyMessage,
    LinesMessage,
    StateMessage,
    EventMessage,
    ErrorMessage,
    WorkerExitMessage,
]

_ANY_MESSAGE_ADAPTER: TypeAdapter[AnyMessage] = TypeAdapter(AnyMessage)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def new_message_id() -> str:
    """Return a UUID4 correlation id suitable for `message["id"]`."""

    return str(uuid.uuid4())


def serialize_event(event: Any) -> Dict[str, Any]:
    """
    Serialize an event payload to a JSON-safe dict.

    Contract:
    - If `event` is a Pydantic model, we use `model_dump(mode="json")`.
    - If `event` is already a dict, it is returned as-is.

    Raises:
        TypeError: if the input cannot be serialized into a JSON-safe dict.
    """

    if isinstance(event, dict):
        return cast(Dict[str, Any], event)
    if isinstance(event, BaseModel):
        return cast(Dict[str, Any], event.model_dump(mode="json"))
    # duck-typing for pydantic-like objects
    dump = getattr(event, "model_dump", None)
    if callable(dump):
        return cast(Dict[str, Any], dump(mode="json"))
    raise TypeError(f"Cannot serialize event of type {type(event).__name__}")


def create_message(
    type: str,
    *,
    payload: dict | None = None,
    id: str | None = None,
) -> Dict[str, Any]:
    """
    Build and validate a protocol message.

    This is the preferred constructor used by both UI and worker: it ensures
    messages match the schema before they are put on a queue.
    """

    base = {"type": type, "id": id, "payload": payload or {}}
    parsed = _ANY_MESSAGE_ADAPTER.validate_python(base)
    return cast(Dict[str, Any], parsed.model_dump(mode="json"))


def parse_message(msg: Dict[str, Any]) -> AnyMessage:
    """
    Parse and validate a raw message dict.

    Raises:
        pydantic.ValidationError on invalid messages.
    """

    return _ANY_MESSAGE_ADAPTER.validate_python(msg)


def validate_message(msg: Dict[str, Any]) -> bool:
    """Return True if msg conforms to the protocol schema; otherwise False."""

    try:
        _ANY_MESSAGE_ADAPTER.validate_python(msg)
        return True
    except ValidationError:
        return False


def message_type(msg: Dict[str, Any]) -> str:
    """Best-effort helper to read `msg["type"]` as a string."""

    try:
        t = msg.get("type")
    except Exception:
        return ""
    return str(t or "")
