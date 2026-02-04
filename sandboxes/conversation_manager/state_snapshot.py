"""
Structured state snapshot for the ConversationManager sandbox.

This module provides functionality to capture and persist a structured snapshot
of the sandbox's display state (logs, event tree, traces) to a file for debugging
and analysis purposes.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from sandboxes.conversation_manager.event_tree_display import EventTreeDisplay, TreeNode
from sandboxes.conversation_manager.trace_display import TraceDisplay, TraceEntry
from sandboxes.conversation_manager.log_aggregator import LogAggregator, LogEntry


@dataclass
class StateSnapshot:
    """A structured snapshot of the sandbox display state."""

    timestamp: str
    timestamp_unix: float

    # Logs by category, grouped by handle
    cm_logs: dict[str | None, list[dict]]
    actor_logs: dict[str | None, list[dict]]
    manager_logs: dict[str | None, list[dict]]

    # Event trees
    event_trees: list[dict]

    # CodeAct traces grouped by handle
    traces: dict[str | None, list[dict]]

    # Summary counts
    summary: dict[str, Any]


def _tree_node_to_dict(node: TreeNode) -> dict:
    """Convert a TreeNode to a serializable dict."""
    return {
        "label": node.label,
        "call_id": node.call_id,
        "handle_id": node.handle_id,
        "status": node.status,
        "started_at": node.started_at,
        "finished_at": node.finished_at,
        "error": node.error,
        "args": node.args,
        "result": node.result,
        "children": [_tree_node_to_dict(c) for c in node.children],
    }


def _log_entry_to_dict(entry: LogEntry) -> dict:
    """Convert a LogEntry to a serializable dict."""
    return {
        "timestamp": entry.timestamp,
        "timestamp_iso": (
            datetime.fromtimestamp(entry.timestamp).isoformat()
            if entry.timestamp
            else None
        ),
        "category": entry.category,
        "level": entry.level,
        "message": entry.message,
        "subcategory": entry.subcategory,
        "event_id": entry.event_id,
        "handle_id": entry.handle_id,
    }


def _trace_entry_to_dict(entry: TraceEntry) -> dict:
    """Convert a TraceEntry to a serializable dict."""
    return {
        "turn_index": entry.turn_index,
        "timestamp": entry.timestamp,
        "timestamp_iso": (
            datetime.fromtimestamp(entry.timestamp).isoformat()
            if entry.timestamp
            else None
        ),
        "event_id": entry.event_id,
        "handle_id": entry.handle_id,
        "code": entry.code,
        "result": entry.result if isinstance(entry.result, dict) else str(entry.result),
        "error": entry.error,
    }


def _group_by_handle(entries: list, to_dict_fn) -> dict[str | None, list[dict]]:
    """Group entries by handle_id and convert to dicts."""
    result: dict[str | None, list[dict]] = {}
    for entry in entries:
        hid = getattr(entry, "handle_id", None)
        key = str(hid) if hid is not None else None
        if key not in result:
            result[key] = []
        result[key].append(to_dict_fn(entry))
    return result


def capture_snapshot(
    *,
    log_aggregator: LogAggregator | None = None,
    event_tree_display: EventTreeDisplay | None = None,
    trace_display: TraceDisplay | None = None,
) -> StateSnapshot:
    """
    Capture a structured snapshot of the current sandbox state.

    Args:
        log_aggregator: The log aggregator instance
        event_tree_display: The event tree display instance
        trace_display: The trace display instance

    Returns:
        A StateSnapshot containing all the structured state
    """
    now = time.time()
    now_iso = datetime.fromtimestamp(now).isoformat()

    # Logs grouped by handle
    cm_logs: dict[str | None, list[dict]] = {}
    actor_logs: dict[str | None, list[dict]] = {}
    manager_logs: dict[str | None, list[dict]] = {}

    if log_aggregator is not None:
        cm_logs = _group_by_handle(
            log_aggregator._buf.get("cm", []),
            _log_entry_to_dict,
        )
        actor_logs = _group_by_handle(
            log_aggregator._buf.get("actor", []),
            _log_entry_to_dict,
        )
        manager_logs = _group_by_handle(
            log_aggregator._buf.get("manager", []),
            _log_entry_to_dict,
        )

    # Event trees
    event_trees: list[dict] = []
    if event_tree_display is not None:
        for tree in event_tree_display.get_all_trees():
            event_trees.append(_tree_node_to_dict(tree))

    # Traces grouped by handle
    traces: dict[str | None, list[dict]] = {}
    if trace_display is not None:
        traces = _group_by_handle(trace_display._entries, _trace_entry_to_dict)

    # Summary
    summary = {
        "total_cm_logs": sum(len(v) for v in cm_logs.values()),
        "total_actor_logs": sum(len(v) for v in actor_logs.values()),
        "total_manager_logs": sum(len(v) for v in manager_logs.values()),
        "total_event_trees": len(event_trees),
        "total_traces": sum(len(v) for v in traces.values()),
        "active_handles": list(
            set(
                k
                for logs in [cm_logs, actor_logs, manager_logs, traces]
                for k in logs.keys()
                if k is not None
            ),
        ),
    }

    return StateSnapshot(
        timestamp=now_iso,
        timestamp_unix=now,
        cm_logs=cm_logs,
        actor_logs=actor_logs,
        manager_logs=manager_logs,
        event_trees=event_trees,
        traces=traces,
        summary=summary,
    )


def save_snapshot(
    snapshot: StateSnapshot,
    path: Path | str,
    *,
    pretty: bool = True,
) -> Path:
    """
    Save a snapshot to a JSON file.

    Args:
        snapshot: The snapshot to save
        path: Output file path
        pretty: Whether to format with indentation

    Returns:
        The path to the saved file
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = asdict(snapshot)
    with open(path, "w") as f:
        if pretty:
            json.dump(data, f, indent=2, default=str)
        else:
            json.dump(data, f, default=str)

    return path


def save_snapshot_auto(
    *,
    log_aggregator: LogAggregator | None = None,
    event_tree_display: EventTreeDisplay | None = None,
    trace_display: TraceDisplay | None = None,
    output_dir: Path | str | None = None,
    prefix: str = "sandbox_state",
) -> Path:
    """
    Capture and save a snapshot with auto-generated filename.

    Args:
        log_aggregator: The log aggregator instance
        event_tree_display: The event tree display instance
        trace_display: The trace display instance
        output_dir: Output directory (defaults to current directory)
        prefix: Filename prefix

    Returns:
        The path to the saved file
    """
    snapshot = capture_snapshot(
        log_aggregator=log_aggregator,
        event_tree_display=event_tree_display,
        trace_display=trace_display,
    )

    if output_dir is None:
        output_dir = Path(".")
    else:
        output_dir = Path(output_dir)

    timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    filename = f"{prefix}_{timestamp}.json"
    path = output_dir / filename

    return save_snapshot(snapshot, path)


def render_snapshot_text(snapshot: StateSnapshot) -> str:
    """
    Render a snapshot as human-readable text (similar to GUI layout).

    Args:
        snapshot: The snapshot to render

    Returns:
        Formatted text representation
    """
    lines: list[str] = []

    lines.append("=" * 80)
    lines.append(f"SANDBOX STATE SNAPSHOT — {snapshot.timestamp}")
    lines.append("=" * 80)
    lines.append("")

    # Summary
    lines.append("## Summary")
    lines.append(f"  - CM Logs: {snapshot.summary['total_cm_logs']}")
    lines.append(f"  - Actor Logs: {snapshot.summary['total_actor_logs']}")
    lines.append(f"  - Manager Logs: {snapshot.summary['total_manager_logs']}")
    lines.append(f"  - Event Trees: {snapshot.summary['total_event_trees']}")
    lines.append(f"  - Traces: {snapshot.summary['total_traces']}")
    lines.append(f"  - Active Handles: {snapshot.summary['active_handles']}")
    lines.append("")

    # CM Logs
    lines.append("=" * 80)
    lines.append("## CM Logs")
    lines.append("=" * 80)
    for hid, entries in sorted(
        snapshot.cm_logs.items(),
        key=lambda x: (x[0] is None, x[0]),
    ):
        hid_label = f"Handle {hid}" if hid is not None else "No Handle"
        lines.append(f"\n─── {hid_label} ({len(entries)} entries) ───")
        for e in entries:
            hid_prefix = (
                f"[H{e['handle_id']}]" if e.get("handle_id") is not None else ""
            )
            lines.append(f"[cm]{hid_prefix} {e['level']}: {e['message']}")
    lines.append("")

    # Actor Logs
    lines.append("=" * 80)
    lines.append("## Actor Logs")
    lines.append("=" * 80)
    for hid, entries in sorted(
        snapshot.actor_logs.items(),
        key=lambda x: (x[0] is None, x[0]),
    ):
        hid_label = f"Handle {hid}" if hid is not None else "No Handle"
        lines.append(f"\n─── {hid_label} ({len(entries)} entries) ───")
        for e in entries:
            lines.append(f"[actor] {e['level']}: {e['message']}")
    lines.append("")

    # Manager Logs
    lines.append("=" * 80)
    lines.append("## Manager Logs")
    lines.append("=" * 80)
    for hid, entries in sorted(
        snapshot.manager_logs.items(),
        key=lambda x: (x[0] is None, x[0]),
    ):
        hid_label = f"Handle {hid}" if hid is not None else "No Handle"
        lines.append(f"\n─── {hid_label} ({len(entries)} entries) ───")
        for e in entries:
            lines.append(f"[manager] {e['level']}: {e['message']}")
    lines.append("")

    # Event Trees
    lines.append("=" * 80)
    lines.append("## Event Trees")
    lines.append("=" * 80)

    def _render_tree_node(node: dict, indent: int = 0) -> list[str]:
        prefix = "  " * indent
        icon = {"completed": "✓", "in_progress": "⏳", "error": "❌"}.get(
            node.get("status", ""),
            "•",
        )
        hid = node.get("handle_id")
        hid_prefix = f"[H{hid}] " if hid is not None else ""
        result = [f"{prefix}{icon} {hid_prefix}{node.get('label', '?')}"]
        if node.get("args"):
            result.append(
                (
                    f"{prefix}  (args: {node['args'][:100]}...)"
                    if len(node.get("args", "")) > 100
                    else f"{prefix}  (args: {node['args']})"
                ),
            )
        if node.get("error"):
            result.append(f"{prefix}  (error: {node['error'][:100]}...)")
        for child in node.get("children", []):
            result.extend(_render_tree_node(child, indent + 1))
        return result

    for i, tree in enumerate(snapshot.event_trees):
        hid = tree.get("handle_id")
        hid_label = f" [Handle {hid}]" if hid is not None else ""
        lines.append(f"\n─── Tree {i + 1}{hid_label} ───")
        lines.extend(_render_tree_node(tree))
    lines.append("")

    # Traces
    lines.append("=" * 80)
    lines.append("## CodeAct Traces")
    lines.append("=" * 80)
    for hid, entries in sorted(
        snapshot.traces.items(),
        key=lambda x: (x[0] is None, x[0]),
    ):
        hid_label = f"Handle {hid}" if hid is not None else "No Handle"
        lines.append(f"\n─── {hid_label} ({len(entries)} turn(s)) ───")
        for e in entries:
            lines.append(
                f"\n  Turn {e['turn_index']} (event={e.get('event_id', '?')[:20]})",
            )
            lines.append(f"  Code:")
            for code_line in (e.get("code") or "").split("\n")[:10]:
                lines.append(f"    {code_line}")
            if (e.get("code") or "").count("\n") > 10:
                lines.append(
                    f"    ... ({(e.get('code') or '').count(chr(10)) - 10} more lines)",
                )
            result = e.get("result", {})
            if isinstance(result, dict):
                stdout = result.get("stdout", "")
                stderr = result.get("stderr", "")
                error = result.get("error", "")
                if stdout:
                    lines.append(
                        f"  stdout: {stdout[:200]}{'...' if len(stdout) > 200 else ''}",
                    )
                if stderr:
                    lines.append(
                        f"  stderr: {stderr[:200]}{'...' if len(stderr) > 200 else ''}",
                    )
                if error:
                    lines.append(
                        f"  error: {error[:200]}{'...' if len(str(error)) > 200 else ''}",
                    )
            else:
                lines.append(f"  result: {str(result)[:200]}")
    lines.append("")

    lines.append("=" * 80)
    lines.append("END OF SNAPSHOT")
    lines.append("=" * 80)

    return "\n".join(lines)
