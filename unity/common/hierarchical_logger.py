"""
unity/common/hierarchical_logger.py
===================================

Central icon registry and hierarchical logging infrastructure.

This module provides:
1. ``ICONS`` / ``DEFAULT_ICON`` -- the single source of truth for all emoji
   prefixes used across SessionLogger, LoopLogger, FastBrainLogger, and raw
   ``LOGGER`` calls.
2. ``SessionLogger`` -- session-level logger with consistent label formatting.
3. Integration with ``TOOL_LOOP_LINEAGE`` for nested hierarchy propagation.
"""

from __future__ import annotations

import logging
from contextvars import ContextVar
from typing import Optional

from unity.logger import LOGGER

# ─────────────────────────────────────────────────────────────────────────────
# Session lineage context (parallel to TOOL_LOOP_LINEAGE for non-tool-loop components)
# ─────────────────────────────────────────────────────────────────────────────
SESSION_LINEAGE: ContextVar[list[str]] = ContextVar("SESSION_LINEAGE", default=[])


# ─────────────────────────────────────────────────────────────────────────────
# Central icon registry
#
# Every emoji prefix in the codebase MUST be defined here.  No other module
# should hardcode emoji characters in log calls.
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_ICON = "⬥"

ICONS = {
    # ── Communication events ────────────────────────────────────────────
    "phone_call_received": "📞",
    "phone_call_started": "📞",
    "phone_call_ended": "📞",
    "phone_call_sent": "📞",
    "phone_call_answered": "📞",
    "phone_call_not_answered": "📞",
    "unify_meet_received": "🎥",
    "unify_meet_started": "🎥",
    "unify_meet_ended": "🎥",
    "sms_received": "📱",
    "sms_sent": "📱",
    "email_received": "📧",
    "email_sent": "📧",
    "unify_message_received": "💬",
    "unify_message_sent": "💬",
    "comms_outbound": "📤",
    # ── Voice / utterance ───────────────────────────────────────────────
    "inbound_utterance": "🎤",
    "outbound_utterance": "🔊",
    "call_notification": "🎙️",
    "user_speech": "🧑‍💻",
    "user_state": "🎤",
    "assistant_speech": "🔊",
    # ── Notification pipeline ────────────────────────────────────────────
    "notification_received": "📨",
    "notification_buffered": "⏳",
    "notification_say": "🗣️",
    # ── Proactive speech ────────────────────────────────────────────────
    "proactive_speech": "🗣️",
    "proactive_debounce": "⏱️",
    "proactive_decision": "🗣️",
    "proactive_deferred": "⏸️",
    "proactive_dormant": "💤",
    "proactive_speaking": "🗣️",
    "proactive_published": "📤",
    "proactive_cancelled": "🚫",
    "proactive_error": "❌",
    # ── Session / lifecycle ─────────────────────────────────────────────
    "session_start": "🚀",
    "session_end": "🏁",
    "session_ready": "⚡",
    "startup": "⚡",
    "lifecycle": "🚀",
    "shutdown": "🏁",
    "call_status": "📞",
    # ── LLM brain ───────────────────────────────────────────────────────
    "llm_log_file": "📝",
    "llm_thinking": "🔄",
    "llm_response": "🤖",
    "llm_completed": "✅",
    "llm_cancelled": "⏹️",
    "llm_error": "❌",
    # ── Async tool loop ─────────────────────────────────────────────────
    "system_message": "📋",
    "request": "➡️",
    "tool_seeding": "⬇️",
    "stop_requested": "🛑",
    "early_exit": "⏹️",
    "clarification": "❓",
    "notification": "🔔",
    "interjection": "💬",
    "wait": "🕒",
    "auto_cancel": "🔚",
    "completed": "✅",
    "pending": "⏳",
    "pause": "⏸️",
    "resume": "▶️",
    # ── State management ────────────────────────────────────────────────
    "state_update": "📋",
    "notification_injected": "🔔",
    "notification_unpinned": "🗑️",
    "direct_message": "💬",
    # ── Actor integration ───────────────────────────────────────────────
    "actor_request": "🎯",
    "actor_response": "📥",
    "actor_result": "✅",
    "actor_clarification": "❓",
    # ── Infrastructure clusters ─────────────────────────────────────────
    "managers_worker": "⚙️",
    "file_sync": "🔀",
    "ipc": "🔌",
    "liveview": "🖥️",
    "assistant_jobs": "📋",
    "metrics": "📊",
    "windows_exec": "💻",
    "subscription": "📡",
    "process_cleanup": "🧹",
    # ── IPC direction ───────────────────────────────────────────────────
    "ipc_inbound": "⬇️",
    "ipc_outbound": "⬆️",
    "ipc_error": "❌",
    # ── Inbound comms ────────────────────────────────────────────────────
    "participant_comms": "📱",
    # ── Screenshots / media ─────────────────────────────────────────────
    "screenshot": "📸",
    "screenshot_capture": "📸",
    "webcam_on": "📸",
    "webcam_off": "🚫",
    "screen_share": "🖥️",
    "screen_share_off": "🚫",
    "user_screen_share": "📺",
    "user_screen_share_off": "🚫",
    "remote_control": "🕹️",
    "remote_control_off": "🚫",
    "desktop_session": "🖥️",
    # ── Generic / misc ──────────────────────────────────────────────────
    "event": "📣",
    "ping": "🏓",
    "summarize": "📑",
    "config": "📋",
    "dispatch": "🚀",
    "info": "ℹ️",
    "warning": "⚠️",
    "error": "❌",
}


def get_icon(event_type: str) -> str:
    """Look up the emoji for *event_type*, falling back to ``DEFAULT_ICON``."""
    return ICONS.get(event_type.lower().replace(" ", "_"), DEFAULT_ICON)


# ─────────────────────────────────────────────────────────────────────────────
# SessionLogger - hierarchical logger for session-based components
# ─────────────────────────────────────────────────────────────────────────────


class SessionLogger:
    """
    Hierarchical logger for components like ConversationManager that operate
    as long-lived sessions rather than request-response tool loops.

    Provides consistent log formatting with the same `[label]` pattern as
    async tool loops, enabling unified log viewing across all components.

    Unlike async tool loops (which may have many concurrent instances and need
    unique suffixes), session-scoped components have exactly one instance per
    session — so the label is a fixed string with no suffix.

    Usage:
        logger = SessionLogger("ConversationManager")
        logger.info("phone_call_received", "Incoming call from +1234567890")
        # Output: 📞 [ConversationManager] Incoming call from +1234567890
    """

    def __init__(
        self,
        component_name: str,
        *,
        parent_lineage: Optional[list[str]] = None,
    ):
        """
        Initialize a session logger.

        Args:
            component_name: The name of the component (e.g., "ConversationManager")
            parent_lineage: Optional explicit parent lineage. If None, reads from
                TOOL_LOOP_LINEAGE or SESSION_LINEAGE context vars.
        """
        self._component_name = component_name

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
        return "->".join(parts)

    @property
    def label(self) -> str:
        """The full hierarchical label for this session."""
        return self._label

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

    def _log(
        self,
        level: int,
        event_type: str,
        message: str,
        icon_override: Optional[str] = None,
    ) -> None:
        icon = icon_override or get_icon(event_type)
        LOGGER.log(level, f"{icon} [{self._label}] {message}")

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
        self._log(logging.INFO, event_type, message, icon_override)

    def debug(
        self,
        event_type: str,
        message: str,
        *,
        icon_override: Optional[str] = None,
    ) -> None:
        """Log a debug-level message with event-specific icon."""
        self._log(logging.DEBUG, event_type, message, icon_override)

    def warning(
        self,
        event_type: str,
        message: str,
        *,
        icon_override: Optional[str] = None,
    ) -> None:
        """Log a warning-level message with event-specific icon."""
        self._log(logging.WARNING, event_type, message, icon_override)

    def error(
        self,
        event_type: str,
        message: str,
        *,
        icon_override: Optional[str] = None,
    ) -> None:
        """Log an error-level message with event-specific icon."""
        self._log(logging.ERROR, event_type, message, icon_override)

    # ─────────────────────────────────────────────────────────────────────────
    # Convenience methods for common event types
    # ─────────────────────────────────────────────────────────────────────────

    def log_llm_thinking(self, context: str = "") -> None:
        """Log that the LLM brain is processing."""
        msg = "LLM thinking..." if not context else f"LLM thinking... ({context})"
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


# ─────────────────────────────────────────────────────────────────────────────
# Generic helpers for hierarchical labels and boundary logging
# ─────────────────────────────────────────────────────────────────────────────


def build_hierarchy_label(lineage: list[str], suffix: str = "") -> str:
    """Build a hierarchy label from lineage segments.

    With suffixed hierarchy segments, the label is just ``"->".join(lineage)``
    since each segment already carries its own suffix. The ``suffix`` parameter
    is accepted for backward compatibility but ignored.

    TODO: remove this function once all callers are migrated.
    """
    try:
        return "->".join([str(x) for x in (lineage or []) if str(x)]) or ""
    except Exception:
        return ""


def log_boundary_event(
    hierarchy_label: str,
    message: str,
    *,
    icon: str = "🛠️",
    level: str = "info",
) -> None:
    """Log a boundary event with hierarchical label.

    Format: ``{icon} [{hierarchy_label}] {message}``
    """
    try:
        txt = f"{icon} [{hierarchy_label}] {message}"
        log_fn = getattr(LOGGER, str(level).lower(), None)
        if not callable(log_fn):
            log_fn = LOGGER.info
        log_fn(txt)
    except Exception:
        # Best-effort logging; never fail execution.
        return
