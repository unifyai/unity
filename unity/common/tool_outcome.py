from typing import Any, TypedDict


class ToolOutcome(TypedDict, total=False):
    """
    **Standard payload** returned by every internal tool that mutates state.

    Keys
    ----
    outcome : str
        Human-friendly summary (“task created successfully”, “3 contacts updated”…)
    details : Any
        Free-form extra data – usually an ID, list of IDs, or a backend
        response object.  It is up to each tool to decide what is most useful.
    """

    outcome: str
    details: Any


class ToolError(TypedDict):
    """Structured error payload returned by tools that can self-correct.

    Keys
    ----
    error_kind : str
        Stable discriminator for the failure class.
    message : str
        Human-friendly explanation safe to surface to the model.
    details : Any
        Optional machine-readable context for the rejected operation.
    """

    error_kind: str
    message: str
    details: Any


class ToolErrorException(Exception):
    """Exception wrapper for code paths that need to carry a tool error payload."""

    def __init__(self, payload: ToolError) -> None:
        self.payload = payload
        super().__init__(payload["message"])
