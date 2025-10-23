from __future__ import annotations

from typing import Callable, Dict, Tuple


def require_first(
    tool_name: str,
) -> Callable[[int, Dict[str, Callable]], Tuple[str, Dict[str, Callable]]]:
    """Return a policy function requiring `tool_name` on step 0, else auto.

    Mirrors per-manager defaults with a single shared implementation.
    """

    def _policy(step_index: int, current_tools: Dict[str, Callable]):
        if step_index < 1 and tool_name in current_tools:
            return ("required", {tool_name: current_tools[tool_name]})
        return ("auto", current_tools)

    return _policy


def require_one_of_first(
    tool_names: list[str],
) -> Callable[[int, Dict[str, Callable]], Tuple[str, Dict[str, Callable]]]:
    """Return a policy requiring any of `tool_names` on step 0, else auto."""

    def _policy(step_index: int, current_tools: Dict[str, Callable]):
        if step_index < 1:
            allowed = {n: current_tools[n] for n in tool_names if n in current_tools}
            if allowed:
                return ("required", allowed)
        return ("auto", current_tools)

    return _policy
