"""
Event tree visualization for manager call hierarchy.

This component is fed *ManagerMethodPayload* events (incoming/outgoing) and builds
an in-memory tree that can be rendered for REPL or exposed as structured data for GUI.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from sandboxes.display.formatting import truncate
from unity.events.types.manager_method import ManagerMethodPayload


@dataclass
class TreeNode:
    label: str
    call_id: str | None = None
    status: str = "in_progress"  # in_progress | completed | error
    started_at: float | None = None
    finished_at: float | None = None
    error: str | None = None
    args: str | None = None
    result: str | None = None
    children: list["TreeNode"] = field(default_factory=list)

    def find_or_add_child(self, label: str) -> "TreeNode":
        for c in self.children:
            if c.label == label:
                return c
        n = TreeNode(label=label)
        self.children.append(n)
        return n


class EventTreeDisplay:
    """Build and render a bounded-history manager call hierarchy tree."""

    def __init__(self, *, max_executions: int = 5) -> None:
        self._max_executions = int(max(1, max_executions))
        self._executions: list[TreeNode] = []
        self._current_root: TreeNode | None = None

    def reset_tree(self) -> None:
        self._executions.clear()
        self._current_root = None

    def handle_manager_method(
        self,
        *,
        call_id: str,
        payload: ManagerMethodPayload,
    ) -> None:
        """
        Ingest a single ManagerMethodPayload event.

        We treat a CodeActActor execute_code incoming as an "execution root" when
        present, but otherwise we place nodes under a generic root.
        """
        phase = (payload.phase or "").lower()
        is_boundary = (
            payload.manager == "CodeActActor" and payload.method == "execute_code"
        )

        if phase == "incoming" and is_boundary:
            self._start_new_execution(
                label=payload.hierarchy_label or "execute_code",
                call_id=call_id,
            )

        if self._current_root is None:
            self._start_new_execution(label="Execution", call_id=None)

        node = self._upsert_node(call_id=call_id, payload=payload)

        if phase == "outgoing":
            node.finished_at = time.time()
            node.status = (
                "error" if (payload.status or "").lower() == "error" else "completed"
            )
            node.error = payload.error or None

    def render_tree(self) -> str:
        """Render the current tree as an ASCII snapshot (REPL)."""
        root = self._current_root
        if root is None:
            return "📊 Event Tree (empty)"
        lines: list[str] = []
        lines.append("📊 Event Tree (Current State)")
        lines.append("═" * 58)
        lines.extend(self._render_node(root, prefix=""))
        lines.append("═" * 58)
        return "\n".join(lines)

    def get_tree_data(self) -> TreeNode | None:
        """Return structured tree data for GUI widgets."""
        return self._current_root

    # ──────────────────────────────────────────────────────────────────
    # Internals
    # ──────────────────────────────────────────────────────────────────

    def _start_new_execution(self, *, label: str, call_id: str | None) -> None:
        root = TreeNode(
            label=label or "Execution",
            call_id=call_id,
            started_at=time.time(),
        )
        self._executions.append(root)
        self._current_root = root
        if len(self._executions) > self._max_executions:
            self._executions = self._executions[-self._max_executions :]

    def _upsert_node(self, *, call_id: str, payload: ManagerMethodPayload) -> TreeNode:
        assert self._current_root is not None
        hierarchy = list(payload.hierarchy or [])
        # Prefer readable labels if available.
        hierarchy_labels = (
            hierarchy if hierarchy else [payload.hierarchy_label or payload.manager]
        )

        cur = self._current_root
        for part in hierarchy_labels:
            cur = cur.find_or_add_child(part or "?")

        # Fill leaf metadata.
        cur.call_id = call_id
        if cur.started_at is None:
            cur.started_at = time.time()
        cur.status = "in_progress"
        # Truncate args/result for display; full details can be shown in the UI.
        q = payload.question or payload.instructions or ""
        if q:
            cur.args = truncate(q, 200)
        if payload.answer:
            cur.result = truncate(payload.answer, 200)
        if payload.error:
            cur.error = truncate(payload.error, 2000)
        return cur

    def _render_node(self, node: TreeNode, *, prefix: str) -> list[str]:
        icon = {"completed": "✓", "in_progress": "⏳", "error": "❌"}.get(
            node.status,
            "•",
        )
        label = node.label
        out: list[str] = []
        out.append(f"{prefix}{icon} {label}")

        # Render one level of details (compact, REPL-safe)
        details: list[str] = []
        if node.args:
            details.append(f"args: {truncate(node.args, 120)}")
        if node.error and node.status == "error":
            details.append(f"error: {truncate(node.error, 120)}")
        if details:
            out.append(f"{prefix}  ({'; '.join(details)})")

        # Children
        for i, c in enumerate(node.children):
            is_last = i == len(node.children) - 1
            child_prefix = prefix + ("   " if is_last else "│  ")
            out.extend(self._render_node(c, prefix=child_prefix))
        return out
