from typing import Callable, Dict

__all__ = [
    "clarification_guidance",
    "sig_dict",
    "now_utc_str",
    "tool_name",
    "require_tools",
    "parallelism_guidance",
    "images_policy_block",
]


def clarification_guidance(tools: Dict[str, Callable]) -> str:
    """Return a *single* guidance sentence on how to use the clarification tool.

    The helper looks up the first tool whose name contains the substring
    ``"clarification"`` (case-insensitive).  If such a tool is present the
    returned sentence instructs the model to call it whenever further
    information is required.  Otherwise an **empty string** is returned so
    callers can simply concatenate the result without extra conditionals.
    """
    clar_tool = next((n for n in tools if "clarification" in n.lower()), None)
    if not clar_tool:
        return ""

    return (
        f"If anything is unclear or ambiguous, you must always call the `{clar_tool}` *tool* to "
        "ask the user for clarification before proceeding. Do *not* request clarifications with your final response."
    )


# ---------------------------------------------------------------------------
# Shared utilities for prompt builders
# ---------------------------------------------------------------------------


def sig_dict(tools: Dict[str, Callable]) -> Dict[str, str]:
    """Return {tool_name: '(<argspec>)', …} for pretty JSON dumps.

    Centralises the inspect.signature → string conversion so all prompts render
    a consistent tool signature block.
    """
    import inspect

    return {name: str(inspect.signature(fn)) for name, fn in tools.items()}


def now_utc_str() -> str:
    """Return current UTC timestamp as a compact human-readable string."""
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def tool_name(tools: Dict[str, Callable], needle: str) -> str | None:
    """Best-effort lookup: find the first tool whose name contains ``needle``.

    Comparison is case-insensitive. Returns ``None`` if not found.
    """
    lowered = needle.lower()
    return next((name for name in tools if lowered in name.lower()), None)


def require_tools(pairs: Dict[str, str | None], tools: Dict[str, Callable]) -> None:
    """Validate dynamic tool resolution and raise a clear error if any are missing.

    Parameters
    ----------
    pairs: mapping of a human-friendly expected substring → resolved tool name (or None)
    tools: the full tool mapping; used only to produce a helpful error message
    """
    missing = [substr for substr, resolved in pairs.items() if resolved is None]
    if not missing:
        return

    available = ", ".join(sorted(tools.keys())) or "<none>"
    expected = ", ".join(missing)
    raise ValueError(
        f"Missing required tools: expected to find tool names containing: {expected}. "
        f"Available tools: {available}.",
    )


def parallelism_guidance() -> str:
    """Return a shared block encouraging batching/parallel tool use."""
    return (
        "Parallelism and single\u2011call preference\n"
        "-------------------------------------\n"
        "\u2022 Prefer a single comprehensive tool call over several surgical calls when a tool can safely do the whole job.\n"
        "\u2022 When several reads or writes are independent, plan them together and run them in parallel rather than a serial drip of micro\u2011calls.\n"
        "\u2022 Batch arguments where possible and avoid confirmatory re\u2011queries unless new ambiguity arises."
    )


def images_policy_block() -> str:
    """Return a generic images policy block suitable for inclusion in system prompts.

    This block is intentionally phrased to apply only when images are present,
    so it can be safely included unconditionally by managers.
    """
    return (
        "Images policy (when images are present)\n"
        "--------------------------------------\n"
        "- Treat images as arbitrary user-provided visuals (screenshots, photos, attachments, UI snippets). Do not assume assistant-specific identifiers (task_id, contact_id, queue_id) are visible.\n"
        "- When information is needed from an image, call ask_image with a narrowly scoped question to extract concrete, observable facts. Keep questions minimal and goal-directed.\n"
        "- Use annotations, captions, and explicit user text as the sole ground truth for aligning references (e.g., ‘this one’ / ‘that one’). Do NOT invent ordering rules; if alignment is unclear and not provided, ask a concise clarifying question or proceed conservatively.\n"
        "- Use extracted cues (e.g., names, titles, dates/times, UI labels, keywords, organizations) to drive domain tool calls (search_* / filter_*) that resolve precise records, IDs, and schedules. Do not assume the image alone contains all details.\n"
        "- If critical details are not visible and not stated, avoid unsupported assumptions and note uncertainties or ask a short clarifying question.\n"
        "- Attach images (attach_image_raw) when persistent visual context is helpful for follow-up turns; otherwise prefer targeted ask_image calls to minimize noise."
    )
