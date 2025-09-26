from __future__ import annotations

from typing import List


def normalize_tool_name(name: str, manager_prefix: str) -> str:
    """
    Normalise tool names emitted by the tool loop for a given manager.

    Collapses dynamic "continue_..." wrappers and verbose completion labels like
    "Manager_ask({..}) completed successfully ..." to canonical forms:
        - f"{manager_prefix}_ask"
        - f"{manager_prefix}_update"
        - f"{manager_prefix}_execute"
    """
    s = str(name or "")
    if not s:
        return s

    # Dynamic continue wrappers
    if s.startswith(f"continue_{manager_prefix}_ask"):
        return f"{manager_prefix}_ask"
    if s.startswith(f"continue_{manager_prefix}_update"):
        return f"{manager_prefix}_update"
    if s.startswith(f"continue_{manager_prefix}_execute"):
        return f"{manager_prefix}_execute"

    # Verbose completion labels – tool results that echo the call signature
    if s.startswith(f"{manager_prefix}_ask("):
        return f"{manager_prefix}_ask"
    if s.startswith(f"{manager_prefix}_update("):
        return f"{manager_prefix}_update"
    if s.startswith(f"{manager_prefix}_execute("):
        return f"{manager_prefix}_execute"

    return s


def tool_names_from_messages(msgs: List[dict], manager_prefix: str) -> List[str]:
    """Extract normalised tool names from tool-role messages (skips status checks)."""
    names: List[str] = []
    for m in msgs:
        if m.get("role") == "tool":
            name = m.get("name") or ""
            if name and not str(name).startswith("check_status_"):
                names.append(normalize_tool_name(str(name), manager_prefix))
    return names


def assistant_requested_tool_names(msgs: List[dict], manager_prefix: str) -> List[str]:
    """Extract normalised tool names that the assistant requested (skips status checks)."""
    names: List[str] = []
    for m in msgs:
        if m.get("role") == "assistant" and m.get("tool_calls"):
            for tc in m.get("tool_calls") or []:
                fn = (tc or {}).get("function", {}) or {}
                name = fn.get("name") or ""
                if name and not str(name).startswith("check_status_"):
                    names.append(normalize_tool_name(str(name), manager_prefix))
    return names
