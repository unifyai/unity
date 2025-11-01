from __future__ import annotations

from typing import Any, Dict, List, Literal, Mapping, Optional

from pydantic import BaseModel, Field


class EntryPointManagerMethod(BaseModel):
    """
    Identifies a manager entrypoint for resuming a tool loop.

    This mirrors the semantic cache namespace convention where
    a loop is bound to a specific manager class and public method.
    """

    class_name: str = Field(
        ...,
        description="Manager class name, e.g. 'ContactManager'.",
    )
    method_name: str = Field(..., description="Public manager method, e.g. 'ask'.")
    type: Literal["manager_method"] = Field(
        "manager_method",
        description="Entrypoint discriminator (reserved for future kinds).",
    )


class ChildEntry(BaseModel):
    """
    Optional nested child loop description (inline or by reference).

    This is forward-looking for nested loop serialization. For v1 schema,
    presence of children is optional and not required by flat loops.
    """

    tool_name: Optional[str] = Field(
        None,
        description="Tool/function name on the parent assistant call that produced the child handle.",
    )
    is_passthrough: bool = Field(
        False,
        description="Whether the child handle was adopted in passthrough mode.",
    )
    state: Literal["in_flight", "done"] = Field(
        "in_flight",
        description="Child state at snapshot time.",
    )
    snapshot: Optional[Mapping[str, Any]] = Field(
        None,
        description="Inline child snapshot payload (alternative to ref.path).",
    )
    ref: Optional[Mapping[str, Any]] = Field(
        None,
        description="External reference for child snapshot, e.g. {path: str}.",
    )


class LoopSnapshot(BaseModel):
    """
    Versioned snapshot of an async tool loop sufficient to resume execution.

    Minimal v1 fields capture the manager entrypoint, loop identity, system prompt,
    initial user input, assistant tool-call messages and any completed tool results.
    Optional fields (e.g., children, options, env, full_messages, interjections)
    allow incremental expansion without breaking callers.
    """

    version: int = Field(1, description="Snapshot schema version.")
    entrypoint: EntryPointManagerMethod
    loop_id: str = Field(..., description="Human-friendly loop identifier label.")

    system_message: Optional[str] = Field(
        None,
        description="System prompt used for this loop at snapshot time.",
    )
    initial_user_message: Any = Field(
        None,
        description="Original user content (string or structured).",
    )

    assistant_steps: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Assistant messages containing tool_calls (chronological).",
    )
    tool_results: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Tool reply messages already produced (chronological).",
    )

    # Optional, forward-looking fields
    options: Optional[Dict[str, Any]] = Field(
        None,
        description="Additional options used by the loop (reserved).",
    )
    env: Optional[Dict[str, Any]] = Field(
        None,
        description="Environment metadata such as semantic_cache_namespace (reserved).",
    )
    full_messages: Optional[List[Dict[str, Any]]] = Field(
        None,
        description="Full LLM transcript dump at snapshot time (optional).",
    )
    interjections: Optional[List[Dict[str, Any]]] = Field(
        None,
        description="Serialized interjection records (optional).",
    )
    children: Optional[Dict[str, ChildEntry]] = Field(
        None,
        description="Optional nested child loops keyed by parent call_id.",
    )


def validate_snapshot(data: Mapping[str, Any]) -> LoopSnapshot:
    """
    Validate a snapshot dict and enforce supported version.

    Returns a LoopSnapshot model on success. Raises ValueError on unsupported
    version to allow future migrations to upgrade older/newer payloads.
    """

    snap = LoopSnapshot.model_validate(dict(data))
    if int(snap.version) != 1:
        raise ValueError(f"Unsupported loop snapshot version: {snap.version}")
    return snap


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

    # Optional message-order indices (relative to the original client.messages)
    # When present, these allow exact interleaving with other message types.
    assistant_indices: List[int] = Field(default_factory=list)
    tool_results_indices: List[int] = Field(default_factory=list)

    # Interjections (system messages beyond index 0) and their original indices
    interjections: List[Dict[str, Any]] = Field(default_factory=list)
    interjections_indices: List[int] = Field(default_factory=list)

    # Optional: outstanding clarifications at snapshot time
    # Each entry captures the call_id, the base tool name and the question text
    clarifications: List[Dict[str, Any]] = Field(default_factory=list)

    # Optional: pending notifications at snapshot time (flat replay only)
    # Each entry mirrors the user-facing event payload placed on the handle's
    # notification queue, typically including keys: {"type": "notification",
    # "tool_name": str, "call_id": str, ...additional fields...}
    notifications: List[Dict[str, Any]] = Field(default_factory=list)

    # Snapshot of live images context (list of {image_id, annotation})
    images: List[Dict[str, Any]] = Field(default_factory=list)

    # Optional: full raw client.messages dump at snapshot time for debugging
    # This is not used by deserialization logic; it is provided to aid
    # diagnostics and post-mortem analysis when resumes do not behave as
    # expected. The structure mirrors the LLM client transcript and may
    # include assistant/tool/system entries exactly as recorded.
    full_messages: Optional[List[Dict[str, Any]]] = None

    # Reserved extension points for future versions
    options: Optional[Dict[str, Any]] = None
    env: Optional[Dict[str, Any]] = None

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

    # Allow both manager and inline-tools entrypoints in v1.
    if snap.entrypoint.type == "inline_tools":
        # Minimal sanity checks for inline tools
        if not snap.entrypoint.tools:
            raise ValueError("Inline tools entrypoint must include at least one tool")
        for t in snap.entrypoint.tools:
            if not t.name or not t.module or not t.qualname:
                raise ValueError("Inline tool refs must include name, module, qualname")

    return snap


def migrate_snapshot(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """Return a shallowly normalised snapshot dict compatible with v1.

    Behaviour:
    - If version is missing, assume 1.
    - If entrypoint lacks a discriminant ``type`` but has ``class_name`` or
      ``tools``, infer the appropriate type.
    - Leave unknown fields untouched for forward-compatibility.
    """

    if not isinstance(snapshot, dict):
        return snapshot

    out = dict(snapshot)

    # Default version to 1 when absent
    if "version" not in out:
        out["version"] = 1

    # Normalise entrypoint discriminant
    try:
        ep = out.get("entrypoint")
        if isinstance(ep, dict) and "type" not in ep:
            if "class_name" in ep and "method_name" in ep:
                ep = {"type": "manager_method", **ep}
            elif "tools" in ep:
                ep = {"type": "inline_tools", **ep}
            out["entrypoint"] = ep
    except Exception:
        pass

    # Nothing else required for v1 – meta/options/env remain optional.
    return out


__all__ = (
    "EntryPointManagerMethod",
    "EntryPointInlineTools",
    "ToolRef",
    "LoopSnapshot",
    "validate_snapshot",
    "migrate_snapshot",
)
