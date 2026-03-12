"""Payload model and kind taxonomy for ToolLoop events."""

from __future__ import annotations

import json
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class ToolLoopKind(str, Enum):
    """Semantic kind for every distinct ToolLoop event.

    Computed once at publish time so consumers (stream filters, frontend)
    can key off a single discriminator instead of re-deriving the kind
    from ``message.role`` plus scattered ad-hoc flags.
    """

    # ── User-facing ────────────────────────────────────────────────────
    REQUEST = "request"
    INTERJECTION = "interjection"
    THINKING_SENTINEL = "thinking_sentinel"
    THOUGHT = "thought"
    TOOL_CALL = "tool_call"
    RESPONSE = "response"
    TOOL_RESULT = "tool_result"
    STEERING_PAUSE = "steering_pause"
    STEERING_RESUME = "steering_resume"
    STEERING_STOP = "steering_stop"
    STEERING_HELPER = "steering_helper"

    # ── Noise (filtered from stream) ───────────────────────────────────
    RUNTIME_CONTEXT = "runtime_context"
    TIME_EXPLANATION = "time_explanation"
    VISIBILITY_GUIDANCE = "visibility_guidance"
    CONTEXT_CONTINUATION = "context_continuation"
    PLACEHOLDER = "placeholder"
    STATUS_CHECK = "status_check"
    WAIT_NOOP = "wait_noop"
    EARLY_EXIT = "early_exit"
    SYSTEM_NOTICE = "system_notice"


_STEERING_ACTION_MAP: dict[str, ToolLoopKind] = {
    "pause": ToolLoopKind.STEERING_PAUSE,
    "resume": ToolLoopKind.STEERING_RESUME,
    "stop": ToolLoopKind.STEERING_STOP,
}


def classify_tool_loop_message(msg: dict) -> ToolLoopKind:
    """Derive the canonical :class:`ToolLoopKind` from a raw message dict.

    Priority mirrors the order in which the async tool loop publishes
    messages, so each branch is mutually exclusive.
    """
    role = msg.get("role", "")

    if role == "system":
        if msg.get("_steering"):
            action = str(msg.get("_steering_action", "")).lower()
            return _STEERING_ACTION_MAP.get(action, ToolLoopKind.STEERING_PAUSE)
        if msg.get("_visibility_guidance"):
            return ToolLoopKind.VISIBILITY_GUIDANCE
        if msg.get("_time_explanation"):
            return ToolLoopKind.TIME_EXPLANATION
        if msg.get("_runtime_context"):
            return ToolLoopKind.RUNTIME_CONTEXT
        return ToolLoopKind.SYSTEM_NOTICE

    if role == "user":
        if msg.get("_interjection"):
            return ToolLoopKind.INTERJECTION
        if msg.get("_ctx_header"):
            return ToolLoopKind.CONTEXT_CONTINUATION
        return ToolLoopKind.REQUEST

    if role == "tool":
        name = msg.get("name", "")
        if isinstance(name, str) and name.startswith("check_status_"):
            return ToolLoopKind.STATUS_CHECK
        if name == "wait":
            return ToolLoopKind.WAIT_NOOP
        try:
            parsed = json.loads(msg.get("content", ""))
            if isinstance(parsed, dict) and "_placeholder" in parsed:
                return ToolLoopKind.PLACEHOLDER
        except Exception:
            pass
        return ToolLoopKind.TOOL_RESULT

    if role == "assistant":
        if msg.get("_thinking_in_flight"):
            return ToolLoopKind.THINKING_SENTINEL
        has_thinking = bool(
            msg.get("thinking_blocks")
            or msg.get("reasoning_content")
            or (msg.get("provider_specific_fields") or {}).get("thinking_blocks"),
        )
        if has_thinking:
            return ToolLoopKind.THOUGHT
        if msg.get("tool_calls"):
            return ToolLoopKind.TOOL_CALL
        return ToolLoopKind.RESPONSE

    return ToolLoopKind.RESPONSE


class ToolLoopPayload(BaseModel):
    """Payload for tool loop LLM messages.

    Published for each LLM message during async tool loop execution,
    capturing both the raw message and loop context.
    """

    model_config = ConfigDict(extra="allow")

    kind: str = Field(
        description="Semantic event kind — a ToolLoopKind value computed at publish time",
    )
    message: Dict[str, Any] = Field(
        description="Raw LLM message dict (explicitly untyped to handle varied shapes)",
    )
    method: str = Field(description="Public method that spawned this loop (loop_id)")
    hierarchy: List[str] = Field(
        default_factory=list,
        description="Lineage of nested loops",
    )
    # TODO: remove hierarchy_label once frontend migrates to hierarchy-only
    # tree building -- it is now trivially "->".join(hierarchy).
    hierarchy_label: str = Field(
        default="",
        description="Human-readable hierarchy label (deprecated: derivable from hierarchy)",
    )
    origin: Optional[str] = Field(default=None, description="Origin identifier")
    tool_aliases: Optional[Dict[str, str]] = Field(
        default=None,
        description="Sparse mapping of tool_name -> human-readable label for tool calls in this event only",
    )
