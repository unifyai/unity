from collections.abc import Mapping, Sequence
from typing import TypedDict, Any


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


def details_with_near_duplicates(
    details: Mapping[str, Any],
    near_duplicates: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Return tool-result details with optional near-duplicate hints.

    The base ``ToolOutcome.details`` contract is intentionally free-form, but
    Knowledge and Guidance save paths share one additive convention: include
    ``near_duplicates`` only when the save path already has useful nearby rows
    to surface. Empty or missing hints leave the existing details shape
    unchanged.
    """

    result = dict(details)
    if near_duplicates:
        result["near_duplicates"] = [dict(item) for item in near_duplicates]
    return result
