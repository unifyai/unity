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
