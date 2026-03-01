"""Typed payload models for known EventBus event types.

Each event type has a corresponding Pydantic model that defines its schema.
The EventBus uses these models to:
1. Validate payloads at publish time
2. Eagerly create backend contexts with pre-defined field schemas
3. Provide type safety and documentation
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel

from .comms import CommsPayload
from .desktop_primitive import DesktopActCompletedPayload, DesktopPrimitivePayload
from .llm import LLMPayload
from .manager_method import ManagerMethodPayload
from .message import MessagePayload
from .tool_loop import ToolLoopPayload

__all__ = [
    "ManagerMethodPayload",
    "ToolLoopPayload",
    "MessagePayload",
    "CommsPayload",
    "LLMPayload",
    "DesktopActCompletedPayload",
    "DesktopPrimitivePayload",
    "PAYLOAD_REGISTRY",
]

# Registry: event type string → Pydantic payload model
PAYLOAD_REGISTRY: dict[str, type[BaseModel]] = {
    "ManagerMethod": ManagerMethodPayload,
    "ToolLoop": ToolLoopPayload,
    "Message": MessagePayload,
    "Comms": CommsPayload,
    "LLM": LLMPayload,
    "DesktopPrimitiveInvoked": DesktopPrimitivePayload,
    "DesktopActCompleted": DesktopActCompletedPayload,
}
