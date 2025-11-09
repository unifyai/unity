from __future__ import annotations

from typing import Any, Dict, List, Optional, Literal

from pydantic import BaseModel, Field, ValidationError
from pydantic import model_validator


class EntryPointManagerMethod(BaseModel):
    """Entry point describing a manager method to resume.

    This is intentionally minimal in v1.
    """

    class_name: str
    method_name: str


class ChildSnapshot(BaseModel):
    """Schema for a nested child loop snapshot reference (v1).

    Minimal, nested_structure-aligned vocabulary:
    - tool: canonical "Class.method" when available; else canonical class name
    - handle: canonicalized inheritance chain up to AsyncToolLoopHandle or sentinels
    - passthrough: whether the child was wired for passthrough steering
    - state: "in_flight" | "done"
    - call_id: optional assistant tool_call id that spawned this child
    - snapshot: required for in_flight children
    """

    tool: Optional[str] = None
    handle: Optional[str] = None
    passthrough: bool = False
    state: Literal["in_flight", "done"]
    call_id: Optional[str] = None
    snapshot: Optional[Dict[str, Any]] = None

    @model_validator(mode="after")
    def _validate_child(self):  # type: ignore[override]
        # Minimal identifiers: at least one of tool or handle should be present for readability
        if not (isinstance(self.tool, str) and self.tool) and not (
            isinstance(self.handle, str) and self.handle
        ):
            raise ValueError("child must include 'tool' or 'handle'")

        # State-dependent payload rules
        if self.state == "in_flight":
            has_inline = isinstance(self.snapshot, dict) and len(self.snapshot) >= 1
            if not has_inline:
                raise ValueError(
                    "child(in_flight) requires inline snapshot",
                )
        elif self.state == "done":
            if self.snapshot is not None:
                raise ValueError(
                    "child(done) must not provide snapshot",
                )

        return self


class LoopSnapshot(BaseModel):
    """Versioned snapshot schema for resuming a tool loop (v1).

    Scope in v1:
    - Captures the minimal information needed to reconstruct the tool registry
      and re-schedule any assistant-declared tool calls that lack results.
    - Optionally carries a nested children manifest under ``meta.children`` for
      in‑flight child loops (each child may be embedded inline or referenced by
      path). Other metadata fields (e.g., images, clarifications, notifications)
      are supported for convenience but may be ignored by deserializers.
    """

    version: int = Field(default=1, ge=1)
    # Manager-only entrypoint in the simplified v1
    entrypoint: EntryPointManagerMethod

    # Optional loop identity and prompt header
    loop_id: Optional[str] = None
    system_message: Optional[str] = None
    # Human-readable root summary (nested_structure-aligned)
    root: Optional[Dict[str, str]] = None

    # Original user input in any of the accepted forms used by the loop
    initial_user_message: Any = None

    # Transcript fragments necessary for preflight backfill to work:
    # - assistant: assistant messages that contain tool_calls
    # - tools: tool messages already produced (paired by call_id)
    assistant: List[Dict[str, Any]] = Field(default_factory=list)
    tools: List[Dict[str, Any]] = Field(default_factory=list)

    # Optional positions (relative to the original client.messages)
    # When present, these allow exact interleaving with other message types.
    assistant_positions: List[int] = Field(default_factory=list)
    tool_positions: List[int] = Field(default_factory=list)

    # System interjections (beyond index 0) and their original positions
    system_interjections: List[Dict[str, Any]] = Field(default_factory=list)
    interjection_positions: List[int] = Field(default_factory=list)

    # Optional: outstanding clarifications at snapshot time
    # Each entry captures the call_id, the base tool and the question text
    clarifications: List[Dict[str, Any]] = Field(default_factory=list)

    # Optional: pending notifications at snapshot time (flat replay only)
    # Each entry mirrors the user-facing event payload placed on the handle's
    # notification queue, typically including keys: {"type": "notification",
    # "tool_name": str, "call_id": str, ...additional fields...}
    notifications: List[Dict[str, Any]] = Field(default_factory=list)

    # Snapshot of live images context (list of {image_id, annotation})
    images: List[Dict[str, Any]] = Field(default_factory=list)

    # Diagnostics/metadata (v1.1+): run identifiers, timestamps, context
    meta: Optional[Dict[str, Any]] = None


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

    # Validate nested children manifest (when provided under meta.children)
    try:
        meta = snap.meta or {}
        children = meta.get("children") if isinstance(meta, dict) else None
        if children is not None:
            if not isinstance(children, list):
                raise ValueError("meta.children must be a list when provided")
            for idx, child in enumerate(children):
                try:
                    ChildSnapshot.model_validate(child)
                except ValidationError as exc:
                    raise ValueError(
                        f"Invalid child snapshot at index {idx}",
                    ) from exc
    except Exception:
        # Re-raise ValueErrors from our checks; ignore others defensively
        raise

    return snap


def migrate_snapshot(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """Return a shallowly normalised snapshot dict compatible with v1.

    Behaviour:
    - If version is missing, assume 1.
    - Drop any legacy ``entrypoint.type`` field if present.
    - Leave unknown fields untouched for forward-compatibility.
    """

    if not isinstance(snapshot, dict):
        return snapshot

    out = dict(snapshot)

    # Default version to 1 when absent
    if "version" not in out:
        out["version"] = 1

    # Drop legacy entrypoint.type if present (no longer used)
    try:
        ep = out.get("entrypoint")
        if isinstance(ep, dict) and "type" in ep:
            ep = {k: v for k, v in ep.items() if k != "type"}
            out["entrypoint"] = ep
    except Exception:
        pass

    # Nothing else required for v1 – meta remains optional.
    return out


__all__ = (
    "EntryPointManagerMethod",
    "ChildSnapshot",
    "LoopSnapshot",
    "validate_snapshot",
    "migrate_snapshot",
)
