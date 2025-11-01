from __future__ import annotations

from typing import Any, Dict, List, Optional, Literal

from pydantic import BaseModel, Field, ValidationError


class EntryPointManagerMethod(BaseModel):
    """Entry point describing a manager method to resume.

    This is intentionally minimal in v1. Future versions may introduce
    additional entrypoint types (e.g., inline tools by import path).
    """

    type: Literal["manager_method"] = "manager_method"
    class_name: str
    method_name: str


class LoopSnapshot(BaseModel):
    """Versioned snapshot schema for resuming a tool loop (v1).

    Scope in v1 (flat-only):
    - Captures the minimal information needed to reconstruct the tool registry
      and re-schedule any assistant-declared tool calls that lack results.
    - Nested handles, images, clarifications, and notifications are out of scope
      in v1 and may be added by later versions.
    """

    version: int = Field(default=1, ge=1)
    entrypoint: EntryPointManagerMethod

    # Optional loop identity and prompt header
    loop_id: Optional[str] = None
    system_message: Optional[str] = None

    # Original user input in any of the accepted forms used by the loop
    initial_user_message: Any = None

    # Transcript fragments necessary for preflight backfill to work:
    # - assistant_steps: assistant messages that contain tool_calls
    # - tool_results: tool messages already produced (paired by call_id)
    assistant_steps: List[Dict[str, Any]] = Field(default_factory=list)
    tool_results: List[Dict[str, Any]] = Field(default_factory=list)

    # Reserved extension points for future versions
    options: Optional[Dict[str, Any]] = None
    env: Optional[Dict[str, Any]] = None


def validate_snapshot(snapshot: Dict[str, Any]) -> LoopSnapshot:
    """Validate a loop snapshot dict and enforce v1 constraints.

    Returns a typed LoopSnapshot instance on success or raises ValueError on
    invalid/unsupported inputs.
    """

    try:
        snap = LoopSnapshot.model_validate(snapshot)
    except (
        ValidationError
    ) as exc:  # pragma: no cover - exercised by tests via ValueError
        raise ValueError("Invalid loop snapshot payload") from exc

    if snap.version != 1:
        raise ValueError(f"Unsupported snapshot version: {snap.version}")

    if snap.entrypoint.type != "manager_method":
        raise ValueError("Unsupported entrypoint type for v1")

    return snap


__all__ = (
    "EntryPointManagerMethod",
    "LoopSnapshot",
    "validate_snapshot",
)
