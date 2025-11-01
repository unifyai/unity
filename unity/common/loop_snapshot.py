from __future__ import annotations

from typing import Any, Dict, List, Optional, Literal, Union

from pydantic import BaseModel, Field, ValidationError


class EntryPointManagerMethod(BaseModel):
    """Entry point describing a manager method to resume.

    This is intentionally minimal in v1. Future versions may introduce
    additional entrypoint types (e.g., inline tools by import path).
    """

    type: Literal["manager_method"] = "manager_method"
    class_name: str
    method_name: str


class ToolRef(BaseModel):
    """Reference to a tool by import path and flags.

    - module: the Python module path (e.g. "mypkg.module").
    - qualname: the qualified name within the module (e.g. "func" or "Cls.method").
    - read_only / manager_tool: optional flags mirroring the decorators used in the
      tool registry; when provided, they will be re-applied on deserialization.
    """

    name: str
    module: str
    qualname: str
    read_only: Optional[bool] = None
    manager_tool: Optional[bool] = None


class EntryPointInlineTools(BaseModel):
    """Entry point describing an inline tools registry to resume.

    This supports non-manager loops by listing tools as importable functions.
    """

    type: Literal["inline_tools"] = "inline_tools"
    tools: List[ToolRef]


class LoopSnapshot(BaseModel):
    """Versioned snapshot schema for resuming a tool loop (v1).

    Scope in v1 (flat-only):
    - Captures the minimal information needed to reconstruct the tool registry
      and re-schedule any assistant-declared tool calls that lack results.
    - Nested handles, images, clarifications, and notifications are out of scope
      in v1 and may be added by later versions.
    """

    version: int = Field(default=1, ge=1)
    # Discriminated union of entrypoint types
    entrypoint: Union[EntryPointManagerMethod, EntryPointInlineTools] = Field(
        discriminator="type",
    )

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

    # Optional: outstanding clarifications at snapshot time
    # Each entry captures the call_id, the base tool name and the question text
    clarifications: List[Dict[str, Any]] = Field(default_factory=list)

    # Optional: pending notifications at snapshot time (flat replay only)
    # Each entry mirrors the user-facing event payload placed on the handle's
    # notification queue, typically including keys: {"type": "notification",
    # "tool_name": str, "call_id": str, ...additional fields...}
    notifications: List[Dict[str, Any]] = Field(default_factory=list)

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

    # Allow both manager and inline-tools entrypoints in v1.
    if snap.entrypoint.type == "inline_tools":
        # Minimal sanity checks for inline tools
        if not snap.entrypoint.tools:
            raise ValueError("Inline tools entrypoint must include at least one tool")
        for t in snap.entrypoint.tools:
            if not t.name or not t.module or not t.qualname:
                raise ValueError("Inline tool refs must include name, module, qualname")

    return snap


__all__ = (
    "EntryPointManagerMethod",
    "EntryPointInlineTools",
    "ToolRef",
    "LoopSnapshot",
    "validate_snapshot",
)
