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
    handle_id: int | None = None  # Actor handle ID for concurrent tracking
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
        n = TreeNode(label=label, handle_id=self.handle_id)
        self.children.append(n)
        return n


class EventTreeDisplay:
    """Build and render a bounded-history manager call hierarchy tree."""

    def __init__(self, *, max_executions: int = 5) -> None:
        self._max_executions = int(max(1, max_executions))
        self._executions: list[TreeNode] = []
        self._current_root: TreeNode | None = None
        # Map handle_id → root node for concurrent Actor handle tracking
        self._handle_roots: dict[int, TreeNode] = {}
        # Current handle context (set via set_handle_context)
        self._current_handle_id: int | None = None

    def reset_tree(self) -> None:
        self._executions.clear()
        self._current_root = None
        self._handle_roots.clear()
        self._current_handle_id = None

    def set_handle_context(self, *, handle_id: int | None) -> None:
        """Set the current Actor handle context for routing subsequent events."""
        self._current_handle_id = handle_id

    def handle_manager_method(
        self,
        *,
        call_id: str,
        payload: ManagerMethodPayload,
        handle_id: int | None = None,
    ) -> None:
        """
        Ingest a single ManagerMethodPayload event.

        We treat a CodeActActor execute_code incoming as an "execution root" when
        present, but otherwise we place nodes under a generic root.

        Args:
            call_id: Unique ID for this manager method call
            payload: The ManagerMethodPayload event
            handle_id: Optional Actor handle ID for concurrent tracking. Falls back
                       to _current_handle_id if not provided.
        """
        phase = (payload.phase or "").lower()
        is_boundary = (
            payload.manager == "CodeActActor" and payload.method == "execute_code"
        )

        # Use provided handle_id or fall back to current context
        effective_handle_id = (
            handle_id if handle_id is not None else self._current_handle_id
        )

        if phase == "incoming" and is_boundary:
            self._start_new_execution(
                label=payload.hierarchy_label or "execute_code",
                call_id=call_id,
                handle_id=effective_handle_id,
            )

        if self._current_root is None:
            self._start_new_execution(
                label="Execution",
                call_id=None,
                handle_id=effective_handle_id,
            )

        node = self._upsert_node(
            call_id=call_id,
            payload=payload,
            handle_id=effective_handle_id,
        )

        if phase == "outgoing":
            node.finished_at = time.time()
            node.status = (
                "error" if (payload.status or "").lower() == "error" else "completed"
            )
            node.error = payload.error or None

    def mark_handle_completed(self, handle_id: int) -> None:
        """Mark a handle's tree as completed (all nodes finished)."""
        root = self._handle_roots.get(handle_id)
        if root is not None:
            root.status = "completed"
            root.finished_at = time.time()

    def render_tree(self, *, show_all: bool = False) -> str:
        """Render the tree(s) as an ASCII snapshot (REPL).

        Args:
            show_all: If True, render all active execution trees (for concurrent
                      Actor handles). If False, only render the current tree.
        """
        if show_all:
            active = self.get_active_trees()
            if not active:
                return "📊 Event Tree (empty)"
            lines: list[str] = []
            lines.append(f"📊 Event Tree ({len(active)} concurrent execution(s))")
            lines.append("═" * 58)
            for i, root in enumerate(active):
                hid_label = (
                    f" [Handle {root.handle_id}]" if root.handle_id is not None else ""
                )
                lines.append(f"── Execution {i + 1}{hid_label} ──")
                lines.extend(self._render_node(root, prefix="", show_handle=True))
                if i < len(active) - 1:
                    lines.append("")  # Spacing between trees
            lines.append("═" * 58)
            return "\n".join(lines)
        else:
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
        """Return structured tree data for GUI widgets (legacy single-tree API)."""
        return self._current_root

    def get_all_trees(self) -> list[TreeNode]:
        """Return all execution trees (including concurrent ones)."""
        return list(self._executions)

    def get_active_trees(self) -> list[TreeNode]:
        """Return only trees that are still in-progress."""
        return [t for t in self._executions if t.status == "in_progress"]

    def get_trees_by_handle(self) -> dict[int | None, list[TreeNode]]:
        """Return trees grouped by handle_id for concurrent visualization."""
        result: dict[int | None, list[TreeNode]] = {}
        for tree in self._executions:
            hid = tree.handle_id
            if hid not in result:
                result[hid] = []
            result[hid].append(tree)
        return result

    # ──────────────────────────────────────────────────────────────────
    # Internals
    # ──────────────────────────────────────────────────────────────────

    def _start_new_execution(
        self,
        *,
        label: str,
        call_id: str | None,
        handle_id: int | None = None,
    ) -> None:
        root = TreeNode(
            label=label or "Execution",
            call_id=call_id,
            handle_id=handle_id,
            started_at=time.time(),
        )
        self._executions.append(root)
        self._current_root = root
        # Track by handle_id for concurrent access
        if handle_id is not None:
            self._handle_roots[handle_id] = root
        if len(self._executions) > self._max_executions:
            # Clean up handle_roots for evicted trees
            evicted = self._executions[: len(self._executions) - self._max_executions]
            for t in evicted:
                if t.handle_id is not None:
                    self._handle_roots.pop(t.handle_id, None)
            self._executions = self._executions[-self._max_executions :]

    def _upsert_node(
        self,
        *,
        call_id: str,
        payload: ManagerMethodPayload,
        handle_id: int | None = None,
    ) -> TreeNode:
        # If we have a handle_id, try to find the corresponding root first
        tree_root = None
        if handle_id is not None:
            tree_root = self._handle_roots.get(handle_id)
        if tree_root is None:
            tree_root = self._current_root
        assert tree_root is not None
        hierarchy = list(payload.hierarchy or [])
        # Prefer readable labels if available.
        hierarchy_labels = (
            hierarchy if hierarchy else [payload.hierarchy_label or payload.manager]
        )

        cur = tree_root
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

    def _render_node(
        self,
        node: TreeNode,
        *,
        prefix: str,
        show_handle: bool = False,
    ) -> list[str]:
        icon = {"completed": "✓", "in_progress": "⏳", "error": "❌"}.get(
            node.status,
            "•",
        )
        label = node.label
        if show_handle and node.handle_id is not None:
            label = f"[H{node.handle_id}] {label}"
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
            out.extend(
                self._render_node(c, prefix=child_prefix, show_handle=show_handle),
            )
        return out
