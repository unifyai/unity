"""
Incremental context propagation tracking.

This module provides state tracking for parent chat context propagation through
nested tool loops. It ensures that:
1. Initial context is sent once per inner tool
2. Continued context updates are sent incrementally (no repetition)
3. Each nesting level correctly tracks and forwards context to its inner tools
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ContextForwardingState:
    """Tracks what context has been forwarded to a specific inner tool.

    Used to compute incremental updates: only forward context items that
    haven't been sent to this particular inner tool yet.
    """

    # Whether the initial parent_chat_context has been sent
    initial_context_sent: bool = False
    # Index into parent_chat_context_cont_received: items before this have been sent
    last_cont_idx_forwarded: int = 0
    # Index into local messages: messages before this have been sent as part of context
    last_local_msg_idx_forwarded: int = 0


@dataclass
class LoopContextState:
    """Tracks context state for a single tool loop.

    Each tool loop maintains:
    - The initial parent_chat_context it received (immutable snapshot)
    - Continued context updates received from above via interjections
    - Per-inner-tool tracking of what has been forwarded
    """

    # Initial context snapshot received when this loop started (immutable)
    parent_chat_context: list[dict] = field(default_factory=list)

    # Incremental context updates received from above (via interjections)
    # These are appended as they arrive, never modified once added
    parent_chat_context_cont_received: list[dict] = field(default_factory=list)

    # Per-inner-tool-call tracking: call_id -> ContextForwardingState
    # Tracks what has been forwarded to each inner tool instance
    inner_tool_forwarding: dict[str, ContextForwardingState] = field(
        default_factory=dict,
    )

    def receive_context_continuation(self, cont_items: list[dict]) -> None:
        """Record new context continuation items received from above.

        Args:
            cont_items: List of new context messages to append.
        """
        if cont_items:
            self.parent_chat_context_cont_received.extend(cont_items)

    def get_forwarding_state(self, call_id: str) -> ContextForwardingState:
        """Get or create forwarding state for an inner tool call.

        Args:
            call_id: The tool call identifier.

        Returns:
            The forwarding state for this call_id.
        """
        if call_id not in self.inner_tool_forwarding:
            self.inner_tool_forwarding[call_id] = ContextForwardingState()
        return self.inner_tool_forwarding[call_id]

    def compute_context_for_inner_tool(
        self,
        call_id: str,
        current_local_msgs: list[dict],
    ) -> tuple[Optional[list[dict]], Optional[list[dict]]]:
        """Compute what context to pass to an inner tool call.

        This method computes the incremental context update for a specific
        inner tool call, tracking what has already been sent to avoid repetition.

        Args:
            call_id: The tool call identifier.
            current_local_msgs: Current messages in this loop (excluding ctx header).

        Returns:
            Tuple of (parent_chat_context, parent_chat_context_cont):
            - parent_chat_context: Full initial context (only on first call) or None
            - parent_chat_context_cont: Incremental updates since last call, or None
        """
        state = self.get_forwarding_state(call_id)

        result_parent_ctx: Optional[list[dict]] = None
        result_cont: Optional[list[dict]] = None

        # First call to this tool: send full initial context
        if not state.initial_context_sent:
            # Build nested context: parent_chat_context with local msgs as children
            if self.parent_chat_context:
                result_parent_ctx = copy.deepcopy(self.parent_chat_context)
                # Attach all current local messages as children of the last parent message
                local_msgs_to_send = [
                    {"role": m.get("role"), "content": m.get("content")}
                    for m in current_local_msgs
                ]
                if local_msgs_to_send and result_parent_ctx:
                    result_parent_ctx[-1].setdefault("children", []).extend(
                        local_msgs_to_send,
                    )
            elif current_local_msgs:
                # No parent context, just local messages
                result_parent_ctx = [
                    {"role": m.get("role"), "content": m.get("content")}
                    for m in current_local_msgs
                ]

            # Also include any accumulated cont items received so far
            if self.parent_chat_context_cont_received:
                result_cont = list(self.parent_chat_context_cont_received)

            # Mark as sent
            state.initial_context_sent = True
            state.last_cont_idx_forwarded = len(self.parent_chat_context_cont_received)
            state.last_local_msg_idx_forwarded = len(current_local_msgs)

        else:
            # Subsequent call: only send incremental updates
            incremental_cont: list[dict] = []

            # New cont items received from above
            if state.last_cont_idx_forwarded < len(
                self.parent_chat_context_cont_received,
            ):
                new_cont = self.parent_chat_context_cont_received[
                    state.last_cont_idx_forwarded :
                ]
                incremental_cont.extend(new_cont)
                state.last_cont_idx_forwarded = len(
                    self.parent_chat_context_cont_received,
                )

            # New local messages since last forward
            if state.last_local_msg_idx_forwarded < len(current_local_msgs):
                new_local = current_local_msgs[state.last_local_msg_idx_forwarded :]
                new_local_formatted = [
                    {"role": m.get("role"), "content": m.get("content")}
                    for m in new_local
                ]
                incremental_cont.extend(new_local_formatted)
                state.last_local_msg_idx_forwarded = len(current_local_msgs)

            if incremental_cont:
                result_cont = incremental_cont

        return result_parent_ctx, result_cont

    def get_pending_cont_for_active_tools(
        self,
        active_call_ids: set[str],
    ) -> dict[str, list[dict]]:
        """Get pending context continuations for all active inner tools.

        Used when new cont items arrive via interjection and need to be
        forwarded to all currently running inner tools.

        Args:
            active_call_ids: Set of call_ids for currently running inner tools.

        Returns:
            Dict mapping call_id to list of cont items that need forwarding.
        """
        result: dict[str, list[dict]] = {}

        for call_id in active_call_ids:
            state = self.get_forwarding_state(call_id)
            if not state.initial_context_sent:
                # Tool hasn't been called yet, skip
                continue

            if state.last_cont_idx_forwarded < len(
                self.parent_chat_context_cont_received,
            ):
                pending = self.parent_chat_context_cont_received[
                    state.last_cont_idx_forwarded :
                ]
                if pending:
                    result[call_id] = list(pending)
                    # Note: we don't update last_cont_idx_forwarded here
                    # That happens when the interjection is actually delivered

        return result

    def mark_cont_forwarded_to_tool(self, call_id: str) -> None:
        """Mark that all pending cont items have been forwarded to a tool.

        Called after successfully forwarding an interjection with cont to an inner tool.

        Args:
            call_id: The tool call identifier.
        """
        state = self.get_forwarding_state(call_id)
        state.last_cont_idx_forwarded = len(self.parent_chat_context_cont_received)
