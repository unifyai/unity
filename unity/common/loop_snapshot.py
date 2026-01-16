from __future__ import annotations

from typing import Any, Dict, List, Optional, Literal

from pydantic import BaseModel, Field, ValidationError
from pydantic import model_validator


class ChildSnapshot(BaseModel):
    """Schema for a nested child loop snapshot reference (v1).

    Minimal, nested_structure-aligned vocabulary:
    - tool: canonical "Class.method" when available; else canonical class name
    - handle: canonicalized inheritance chain up to AsyncToolLoopHandle or sentinels
    - state: "in_flight" | "done"
    - call_id: optional assistant tool_call id that spawned this child
    - snapshot: required for in_flight children
    """

    tool: Optional[str] = None
    handle: Optional[str] = None
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
    - Optionally carries a nested children manifest under top-level ``children`` for
      in‑flight child loops (each child is embedded inline with a ``snapshot`` when live).
      Other metadata fields (e.g., images, clarifications, notifications) are supported
      for convenience but may be ignored by deserializers.
    """

    version: int = Field(default=1, ge=1)

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

    # Preferred location for nested child snapshots (structure-aligned with nested_structure)
    children: List[Dict[str, Any]] = Field(default_factory=list)

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

    # Validate top-level nested children manifest (preferred)
    for idx, child in enumerate(snap.children or []):
        try:
            ChildSnapshot.model_validate(child)
        except ValidationError as exc:
            raise ValueError(f"Invalid child snapshot at index {idx}") from exc

    return snap


def migrate_snapshot(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """Return a shallowly normalised snapshot dict compatible with v1.

    Behaviour:
    - If version is missing, assume 1.
    - If legacy ``entrypoint`` is present with class_name/method_name, map it to
      ``root.tool`` (\"Class.method\") when missing and drop ``entrypoint``.
    - Leave unknown fields untouched for forward-compatibility.
    """

    if not isinstance(snapshot, dict):
        return snapshot

    out = dict(snapshot)

    # Default version to 1 when absent
    if "version" not in out:
        out["version"] = 1

    # Legacy support: map entrypoint → root.tool, then drop entrypoint
    try:
        ep = out.get("entrypoint")
        if isinstance(ep, dict):
            cls_name = ep.get("class_name")
            meth_name = ep.get("method_name")
            if isinstance(cls_name, str) and isinstance(meth_name, str):
                root = out.get("root")
                if not isinstance(root, dict):
                    root = {}
                if not isinstance(root.get("tool"), str) or not root.get("tool"):
                    root["tool"] = f"{cls_name}.{meth_name}"
                if not isinstance(root.get("handle"), str) or not root.get("handle"):
                    root["handle"] = "AsyncToolLoopHandle"
                out["root"] = root
            # Drop legacy entrypoint unconditionally
            out.pop("entrypoint", None)
    except Exception:
        pass

    # Nothing else required for v1 – meta remains optional.
    return out


__all__ = (
    "ChildSnapshot",
    "LoopSnapshot",
    "validate_snapshot",
    "migrate_snapshot",
)
