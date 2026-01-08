"""
unity/common/hierarchical_logger.py
===================================

Hierarchical logging infrastructure for components that don't use async tool loops
but still need consistent nested logging (e.g., ConversationManager).

This module provides:
1. SessionLogger - session-level logger with consistent label formatting
2. Event-specific emoji icons (distinct from async tool loop icons)
3. Integration with TOOL_LOOP_LINEAGE for nested hierarchy propagation
"""

from __future__ import annotations

from contextvars import ContextVar
from secrets import token_hex
from typing import Optional

from unity.constants import LOGGER

# ─────────────────────────────────────────────────────────────────────────────
# Session lineage context (parallel to TOOL_LOOP_LINEAGE for non-tool-loop components)
# ─────────────────────────────────────────────────────────────────────────────
SESSION_LINEAGE: ContextVar[list[str]] = ContextVar("SESSION_LINEAGE", default=[])


# ─────────────────────────────────────────────────────────────────────────────
# Emoji icons for ConversationManager events (distinct from async tool loop)
# ─────────────────────────────────────────────────────────────────────────────
# Async tool loop already uses: 🤖 🧑‍💻 💬 ⏸️ ▶️ ❓ ✅ 🖼️ 📦 ❌ ⚠️ 🛑 🛠️ 🔄 🔔 📝 🎬
# We use distinct emojis for CM-specific events:

CM_ICONS = {
    # Communication events
    "phone_call_received": "📞",
    "phone_call_started": "📞",
    "phone_call_ended": "📞",
    "phone_call_sent": "📞",
    "phone_call_answered": "📞",
    "unify_meet_received": "🎥",
    "unify_meet_started": "🎥",
    "unify_meet_ended": "🎥",
    "sms_received": "📱",
    "sms_sent": "📱",
    "email_received": "📧",
    "email_sent": "📧",
    "unify_message_received": "💌",
    "unify_message_sent": "💌",
    # Voice/utterance events
    "inbound_utterance": "🎤",
    "outbound_utterance": "🔊",
    "call_guidance": "🎙️",
    # Session lifecycle
    "session_start": "🚀",
    "session_end": "🏁",
    "startup": "⚡",
    # LLM brain
    "llm_thinking": "🧠",
    "llm_response": "💡",
    # State management
    "state_update": "📋",
    "notification_injected": "📨",
    "notification_unpinned": "🗑️",
    "direct_message": "💭",
    # Actor integration
    "actor_request": "🎯",
    "actor_response": "📥",
    "actor_result": "🏆",
    "actor_clarification": "❔",
    # Generic
    "event": "📣",
    "ping": "💓",
    "summarize": "📑",
}


def get_cm_icon(event_type: str) -> str:
    """Get the emoji icon for a ConversationManager event type."""
    return CM_ICONS.get(event_type.lower().replace(" ", "_"), "📣")


# ─────────────────────────────────────────────────────────────────────────────
# SessionLogger - hierarchical logger for session-based components
# ─────────────────────────────────────────────────────────────────────────────


class SessionLogger:
    """
    Hierarchical logger for components like ConversationManager that operate
    as long-lived sessions rather than request-response tool loops.

    Provides consistent log formatting with the same `[label]` pattern as
    async tool loops, enabling unified log viewing across all components.

    Usage:
        logger = SessionLogger("ConversationManager")
        logger.info("phone_call_received", "Incoming call from +1234567890")
        # Output: 📞 [ConversationManager(a1b2)] Incoming call from +1234567890

    When nested inside a tool loop:
        # Output: 📞 [Actor.act->ConversationManager(a1b2)] Incoming call...
    """

    def __init__(
        self,
        component_name: str,
        *,
        suffix: Optional[str] = None,
        parent_lineage: Optional[list[str]] = None,
    ):
        """
        Initialize a session logger.

        Args:
            component_name: The name of the component (e.g., "ConversationManager")
            suffix: Optional 4-hex suffix. If None, generates a new one.
            parent_lineage: Optional explicit parent lineage. If None, reads from
                TOOL_LOOP_LINEAGE or SESSION_LINEAGE context vars.
        """
        self._component_name = component_name
        self._suffix = suffix or token_hex(2)

        # Determine parent lineage
        if parent_lineage is not None:
            self._parent_lineage = list(parent_lineage)
        else:
            # Check TOOL_LOOP_LINEAGE first (async tool loop context)
            try:
                from unity.common._async_tool.loop_config import TOOL_LOOP_LINEAGE

                tool_lineage = TOOL_LOOP_LINEAGE.get([])
                if tool_lineage:
                    self._parent_lineage = list(tool_lineage)
                else:
                    # Fall back to SESSION_LINEAGE
                    self._parent_lineage = list(SESSION_LINEAGE.get([]))
            except Exception:
                self._parent_lineage = list(SESSION_LINEAGE.get([]))

        # Build the label
        self._label = self._build_label()

    def _build_label(self) -> str:
        """Build the hierarchical label string."""
        parts = self._parent_lineage + [self._component_name]
        base = "->".join(parts)
        return f"{base}({self._suffix})"

    @property
    def label(self) -> str:
        """The full hierarchical label for this session."""
        return self._label

    @property
    def suffix(self) -> str:
        """The 4-hex suffix for this session."""
        return self._suffix

    @property
    def lineage(self) -> list[str]:
        """The full lineage including this component."""
        return self._parent_lineage + [self._component_name]

    def child_lineage(self) -> list[str]:
        """
        Get the lineage to pass to child components/loops.

        This should be passed as `parent_lineage` to nested async tool loops
        or child SessionLoggers.
        """
        return self.lineage

    def info(
        self,
        event_type: str,
        message: str,
        *,
        icon_override: Optional[str] = None,
    ) -> None:
        """
        Log an info-level message with event-specific icon.

        Args:
            event_type: The type of event (used to select icon)
            message: The log message
            icon_override: Optional icon to use instead of event-type lookup
        """
        icon = icon_override or get_cm_icon(event_type)
        LOGGER.info(f"{icon} [{self._label}] {message}")

    def debug(
        self,
        event_type: str,
        message: str,
        *,
        icon_override: Optional[str] = None,
    ) -> None:
        """Log a debug-level message with event-specific icon."""
        icon = icon_override or get_cm_icon(event_type)
        LOGGER.debug(f"{icon} [{self._label}] {message}")

    def warning(
        self,
        event_type: str,
        message: str,
        *,
        icon_override: Optional[str] = None,
    ) -> None:
        """Log a warning-level message with event-specific icon."""
        icon = icon_override or get_cm_icon(event_type)
        LOGGER.warning(f"{icon} [{self._label}] {message}")

    def error(
        self,
        event_type: str,
        message: str,
        *,
        icon_override: Optional[str] = None,
    ) -> None:
        """Log an error-level message with event-specific icon."""
        icon = icon_override or get_cm_icon(event_type)
        LOGGER.error(f"{icon} [{self._label}] {message}")

    # ─────────────────────────────────────────────────────────────────────────
    # Convenience methods for common event types
    # ─────────────────────────────────────────────────────────────────────────

    def log_llm_thinking(self, context: str = "") -> None:
        """Log that the LLM brain is processing."""
        msg = "LLM thinking..." if not context else f"LLM thinking: {context}"
        self.info("llm_thinking", msg)

    def log_llm_response(self, summary: str = "") -> None:
        """Log that the LLM brain has responded."""
        msg = "LLM response received" if not summary else f"LLM response: {summary}"
        self.info("llm_response", msg)

    def log_event_received(self, event_name: str, details: str = "") -> None:
        """Log an incoming event."""
        msg = f"{event_name} received"
        if details:
            msg += f": {details}"
        self.info("event", msg)

    def log_actor_started(self, handle_id: int, query: str) -> None:
        """Log an Actor run being started."""
        preview = query[:80] + "..." if len(query) > 80 else query
        self.info("actor_request", f"Actor run #{handle_id}: {preview}")

    def log_actor_result(self, handle_id: int, result_preview: str = "") -> None:
        """Log an Actor run completing."""
        msg = f"Actor result #{handle_id}"
        if result_preview:
            preview = (
                result_preview[:80] + "..."
                if len(result_preview) > 80
                else result_preview
            )
            msg += f": {preview}"
        self.info("actor_result", msg)


# ─────────────────────────────────────────────────────────────────────────────
# Helper functions for integration with async tool loops
# ─────────────────────────────────────────────────────────────────────────────


def get_current_lineage() -> list[str]:
    """
    Get the current hierarchical lineage from context.

    Checks TOOL_LOOP_LINEAGE first, then SESSION_LINEAGE.
    Returns an empty list if no lineage is set.
    """
    try:
        from unity.common._async_tool.loop_config import TOOL_LOOP_LINEAGE

        tool_lineage = TOOL_LOOP_LINEAGE.get([])
        if tool_lineage:
            return list(tool_lineage)
    except Exception:
        pass

    return list(SESSION_LINEAGE.get([]))


def make_child_loop_id(parent_logger: SessionLogger, method_name: str) -> str:
    """
    Build a loop_id for a child async tool loop.

    Args:
        parent_logger: The parent SessionLogger
        method_name: The method name being called (e.g., "ask")

    Returns:
        A loop_id string like "ConversationManager.ask"
    """
    return f"{parent_logger._component_name}.{method_name}"
