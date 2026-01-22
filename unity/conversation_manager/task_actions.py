"""
Centralized utilities for action steering operations in ConversationManager.

This module provides a single source of truth for:
- Steering operations derived from SteerableToolHandle
- Action name generation and parsing
- Short name derivation from action queries

Dynamic action names use a structured format with __ as the delimiter:
    {operation}_{short_name}__{handle_id}
    {operation}_{short_name}__{handle_id}__{call_id_suffix}  (for answer_clarification)

The __ delimiter clearly separates semantic description from numeric identifiers,
making parsing unambiguous even when components contain digits.
"""

from __future__ import annotations

import inspect
import re
from dataclasses import dataclass
from typing import Optional

from ..common.async_tool_loop import SteerableToolHandle


# Structural delimiter separating semantic parts from identifiers.
# Using __ because neither derive_short_name nor safe_call_id_suffix can produce it.
_DELIM = "__"


# ─────────────────────────────────────────────────────────────────────────────
# Steering operations derived from SteerableToolHandle
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class SteeringOperation:
    """Describes a steering operation that can be performed on an active action."""

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

# Maximum character length for short_name to guarantee tool names stay under
# OpenAI's 64-character limit. Calculated as:
#   64 (max) - 21 (answer_clarification_) - 2 (__) - 5 (handle_id up to 99999)
#            - 2 (__) - 8 (call_id_suffix) = 26 chars
# Using 25 for safety margin.
_MAX_SHORT_NAME_CHARS = 25


def derive_short_name(query: str, max_words: int = 4) -> str:
    """Derive a short, descriptive name from an action query for use in action names.

    Takes the first few words, lowercased, with non-alphanumeric chars replaced
    by spaces (so they act as word separators), joined by underscores.
    Ensures no __ (reserved as structural delimiter) and enforces a character
    limit to guarantee generated tool names never exceed OpenAI's 64-char limit.

    Examples:
        "List all contacts" -> "list_all_contacts"
        "What's the weather?" -> "whats_the_weather"
        "Search transcripts/messages/emails" -> "search_transcripts_messages_emails"
    """
    # Replace non-alphanumeric chars with spaces (so slashes etc. become word breaks)
    normalized = re.sub(r"[^a-zA-Z0-9\s]", " ", query.lower())
    words = normalized.split()[:max_words]
    result = "_".join(words) if words else "task"

    # Collapse any accidental double underscores (__ is our structural delimiter)
    while _DELIM in result:
        result = result.replace(_DELIM, "_")

    # Truncate to max length to guarantee tool names stay under 64 chars
    if len(result) > _MAX_SHORT_NAME_CHARS:
        result = result[:_MAX_SHORT_NAME_CHARS].rstrip("_")

    return result


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

    Uses __ as delimiter to separate semantic description from identifiers:
        {operation}_{short_name}__{handle_id}
        {operation}_{short_name}__{handle_id}__{call_id_suffix}

    Args:
        operation: The steering operation (e.g., "ask", "stop")
        short_name: The short name derived from the action query
        handle_id: The action handle ID
        call_id_suffix: Optional suffix for clarification call IDs

    Returns:
        Action name like "ask_list_contacts__0" or "answer_clarification_action__0__abc123"
    """
    base = f"{operation}_{short_name}{_DELIM}{handle_id}"
    if call_id_suffix:
        return f"{base}{_DELIM}{call_id_suffix}"
    return base


def safe_call_id_suffix(call_id: str) -> str:
    """Convert a call_id to a safe suffix for use in action names.

    Ensures no __ (reserved as structural delimiter).
    """
    if not call_id:
        return "0"
    result = call_id.replace("-", "_")[-8:]
    # Collapse any accidental double underscores
    while _DELIM in result:
        result = result.replace(_DELIM, "_")
    return result


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

    Parses the format: {operation}_{short_name}__{handle_id}[__{call_id_suffix}]

    The __ delimiter makes parsing unambiguous regardless of digits in components.

    Examples:
        "ask_list_contacts__0" -> ParsedActionName("ask", 0, None)
        "stop_search_web__1" -> ParsedActionName("stop", 1, None)
        "answer_clarification_task__0__abc123" -> ParsedActionName("answer_clarification", 0, "abc123")
    """
    # Split by structural delimiter
    parts = action_name.split(_DELIM)

    if len(parts) < 2:
        # No delimiter found - not a valid dynamic action
        # Extract operation for error reporting
        for op in OPERATION_MAP:
            if action_name.startswith(f"{op}_"):
                return ParsedActionName(op, 0, None)
        first_word = action_name.split("_")[0] if "_" in action_name else action_name
        return ParsedActionName(first_word, 0, None)

    # First part is "{operation}_{short_name}", extract operation
    first_part = parts[0]
    operation = None
    for op in OPERATION_MAP:
        if first_part == op or first_part.startswith(f"{op}_"):
            operation = op
            break

    if operation is None:
        return ParsedActionName("", 0, None)

    # Parse handle_id from parts[1]
    try:
        handle_id = int(parts[1])
    except (ValueError, IndexError):
        handle_id = 0

    # For answer_clarification, parts[2] is the call_id_suffix
    call_id_suffix = None
    if operation == "answer_clarification" and len(parts) >= 3:
        call_id_suffix = parts[2]

    return ParsedActionName(operation, handle_id, call_id_suffix)


def is_dynamic_action(action_name: str) -> bool:
    """Check if an action name is a dynamic steering action."""
    return any(action_name.startswith(f"{op.name}_") for op in STEERING_OPERATIONS)


def iter_steering_tools_for_action(
    handle_id: int,
    query: str,
    pending_clarifications: list[dict] | None = None,
    is_paused: bool | None = None,
) -> list[tuple[str, str]]:
    """Generate (action_name, description) pairs for an action's steering tools.

    Args:
        handle_id: The action handle ID
        query: The original action query
        pending_clarifications: List of pending clarification dicts with "call_id" keys
        is_paused: If True, only include resume (skip pause).
                   If False, only include pause (skip resume).
                   If None, include both (backward compatible behavior).

    Returns:
        List of (action_name, description) tuples
    """
    short_name = derive_short_name(query)
    actions = []

    for op in STEERING_OPERATIONS:
        # Conditionally skip pause/resume based on current state
        if is_paused is not None:
            if op.name == "pause" and is_paused:
                continue  # Skip pause when already paused
            if op.name == "resume" and not is_paused:
                continue  # Skip resume when not paused (running)

        if op.requires_clarification:
            # Only generate answer_clarification if there are pending ones
            for clar in pending_clarifications or []:
                call_id = clar.get("call_id", "")
                suffix = safe_call_id_suffix(call_id)
                name = build_action_name(op.name, short_name, handle_id, suffix)
                desc = (
                    op.get_docstring()
                    or f"{op.name.replace('_', ' ').title()} this action"
                )
                actions.append((name, desc))
        else:
            name = build_action_name(op.name, short_name, handle_id)
            desc = (
                op.get_docstring() or f"{op.name.replace('_', ' ').title()} this action"
            )
            actions.append((name, desc))

    return actions
