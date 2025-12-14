"""
Centralized utilities for dynamic task action generation in ConversationManager.

This module provides a single source of truth for:
- Steering operations derived from SteerableToolHandle
- Action name generation and parsing
- Short name derivation from task queries

All dynamic action names follow the pattern: {operation}_{short_name}_{handle_id}
For answer_clarification: answer_clarification_{short_name}_{handle_id}_{call_id_suffix}
"""

from __future__ import annotations

import inspect
import re
from dataclasses import dataclass
from typing import Optional

from ..common.async_tool_loop import SteerableToolHandle


# ─────────────────────────────────────────────────────────────────────────────
# Steering operations derived from SteerableToolHandle
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class SteeringOperation:
    """Describes a steering operation that can be performed on an active task."""

    name: str  # e.g., "ask", "stop"
    method_name: str  # Method name on SteerableToolHandle
    param_name: str  # Primary parameter name for the action (e.g., "query", "reason")
    requires_clarification: bool = False  # Whether this needs a call_id suffix

    def get_docstring(self) -> str:
        """Get the first line of the docstring from SteerableToolHandle."""
        method = getattr(SteerableToolHandle, self.method_name, None)
        if method is None:
            return ""
        doc = inspect.getdoc(method)
        if not doc:
            return ""
        return doc.strip().split("\n")[0]


# Core steering operations - derived from SteerableToolHandle's abstract methods
STEERING_OPERATIONS: tuple[SteeringOperation, ...] = (
    SteeringOperation("ask", "ask", "query"),
    SteeringOperation("stop", "stop", "reason"),
    SteeringOperation("interject", "interject", "message"),
    SteeringOperation("pause", "pause", ""),
    SteeringOperation("resume", "resume", ""),
    SteeringOperation(
        "answer_clarification",
        "answer_clarification",
        "answer",
        requires_clarification=True,
    ),
)

# Operation name -> SteeringOperation mapping
OPERATION_MAP: dict[str, SteeringOperation] = {
    op.name: op for op in STEERING_OPERATIONS
}


def get_steering_operation(name: str) -> Optional[SteeringOperation]:
    """Get a steering operation by name."""
    return OPERATION_MAP.get(name)


# ─────────────────────────────────────────────────────────────────────────────
# Short name derivation
# ─────────────────────────────────────────────────────────────────────────────


def derive_short_name(query: str, max_words: int = 4) -> str:
    """Derive a short, descriptive name from a task query for use in action names.

    Takes the first few words, lowercased, with non-alphanumeric chars removed,
    joined by underscores.

    Examples:
        "List all contacts" -> "list_all_contacts"
        "What's the weather?" -> "whats_the_weather"
    """
    words = re.sub(r"[^a-zA-Z0-9\s]", "", query.lower()).split()[:max_words]
    return "_".join(words) if words else "task"


# ─────────────────────────────────────────────────────────────────────────────
# Action name generation and parsing
# ─────────────────────────────────────────────────────────────────────────────


def build_action_name(
    operation: str,
    short_name: str,
    handle_id: int,
    call_id_suffix: Optional[str] = None,
) -> str:
    """Build a dynamic action name from its components.

    Args:
        operation: The steering operation (e.g., "ask", "stop")
        short_name: The short name derived from the task query
        handle_id: The task handle ID
        call_id_suffix: Optional suffix for clarification call IDs

    Returns:
        Action name like "ask_list_contacts_0" or "answer_clarification_task_0_abc123"
    """
    base = f"{operation}_{short_name}_{handle_id}"
    if call_id_suffix:
        return f"{base}_{call_id_suffix}"
    return base


def safe_call_id_suffix(call_id: str) -> str:
    """Convert a call_id to a safe suffix for use in action names."""
    if not call_id:
        return "0"
    return call_id.replace("-", "_")[-8:]


@dataclass
class ParsedActionName:
    """Result of parsing a dynamic action name."""

    operation: str
    handle_id: int
    call_id_suffix: Optional[str] = None

    @property
    def steering_operation(self) -> Optional[SteeringOperation]:
        """Get the corresponding SteeringOperation."""
        return get_steering_operation(self.operation)


def parse_action_name(action_name: str) -> ParsedActionName:
    """Parse a dynamic action name to extract its components.

    Examples:
        "ask_list_contacts_0" -> ParsedActionName("ask", 0, None)
        "stop_search_web_1" -> ParsedActionName("stop", 1, None)
        "answer_clarification_task_0_abc123" -> ParsedActionName("answer_clarification", 0, "abc123")
    """
    parts = action_name.split("_")

    # Handle answer_clarification which has the call_id suffix after handle_id
    if action_name.startswith("answer_clarification_"):
        # Format: answer_clarification_{short_name}_{handle_id}_{call_id_suffix}
        # Find the handle_id (last numeric part before potential call_id suffix)
        for i in range(len(parts) - 1, 1, -1):
            if parts[i].isdigit():
                handle_id = int(parts[i])
                call_id = "_".join(parts[i + 1 :]) if i + 1 < len(parts) else None
                return ParsedActionName("answer_clarification", handle_id, call_id)
        return ParsedActionName("answer_clarification", 0, None)

    # For other operations: {operation}_{short_name}_{handle_id}
    # The handle_id is the last numeric part
    for i in range(len(parts) - 1, -1, -1):
        if parts[i].isdigit():
            handle_id = int(parts[i])
            operation = parts[0]
            return ParsedActionName(operation, handle_id, None)

    return ParsedActionName(parts[0] if parts else "", 0, None)


def is_dynamic_action(action_name: str) -> bool:
    """Check if an action name is a dynamic task action."""
    return any(action_name.startswith(f"{op.name}_") for op in STEERING_OPERATIONS)


def iter_available_actions_for_task(
    handle_id: int,
    query: str,
    pending_clarifications: list[dict] | None = None,
) -> list[tuple[str, str]]:
    """Generate (action_name, description) pairs for a task's available actions.

    Args:
        handle_id: The task handle ID
        query: The original task query
        pending_clarifications: List of pending clarification dicts with "call_id" keys

    Yields:
        (action_name, description) tuples
    """
    short_name = derive_short_name(query)
    actions = []

    for op in STEERING_OPERATIONS:
        if op.requires_clarification:
            # Only generate answer_clarification if there are pending ones
            for clar in pending_clarifications or []:
                call_id = clar.get("call_id", "")
                suffix = safe_call_id_suffix(call_id)
                name = build_action_name(op.name, short_name, handle_id, suffix)
                desc = (
                    op.get_docstring()
                    or f"{op.name.replace('_', ' ').title()} this task"
                )
                actions.append((name, desc))
        else:
            name = build_action_name(op.name, short_name, handle_id)
            desc = (
                op.get_docstring() or f"{op.name.replace('_', ' ').title()} this task"
            )
            actions.append((name, desc))

    return actions
