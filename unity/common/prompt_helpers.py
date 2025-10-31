from typing import Callable, Dict

__all__ = [
    "clarification_guidance",
    "sig_dict",
    "now_utc_str",
    "tool_name",
    "require_tools",
    "parallelism_guidance",
    "images_policy_block",
    "images_forwarding_block",
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
        "- Treat images as freeform user-provided visuals (screenshots, photos, UI, attachments).\n"
        "- Do not assume system-specific identifiers or structured record fields (e.g., ids, names, statuses, queue/thread references,\n"
        "  timestamps, due/deadline dates) are visible unless they are clearly shown. This applies across managers (e.g., tasks,\n"
        "  contacts, transcripts).\n"
        "- Default-first question: if the caption is vague or absent, start with a very simple descriptive question such as\n"
        "  'What is shown in this image? What activity appears to be in progress? Which app/page is visible?' Extract salient,\n"
        "  observable elements (apps, UI sections, headings, steps, key text snippets) — not database fields.\n"
        "- If the caption already clearly describes the scene and intent, you may skip the broad question and proceed directly\n"
        "  to a targeted question about a specific on-screen detail.\n"
        "- When information is needed from a single image, prefer `ask_image` with a narrowly scoped question to extract concrete,\n"
        "  observable details — never invent system-specific fields that may not be present on-screen.\n"
        "- Use any extracted cues (e.g., what is in the image, what appears to be done if this is a screen-share) to guide downstream\n"
        "  tool choices (e.g., semantic searches guided by inferred activity or content).\n"
        "- Forwarding rule: when delegating to another tool that declares an `images` parameter, forward the relevant images and\n"
        "  rewrite/augment their annotations so they align with the delegated question or action (not the original user phrasing).\n"
        "  Prefer AnnotatedImageRefs; include a curated subset and preserve user-referenced ordering when it matters.\n"
        "- Anti-patterns to avoid:\n"
        "  • Asking for system-specific identifiers or structured record fields in the first question unless those are clearly visible.\n"
        "  • Assuming the screenshot is a structured record view from a specific manager.\n"
        "  • Re-asking a broad description when the caption already provides that description.\n"
        "- Attach images (`attach_image_raw`) when persistent visual context is helpful for follow-up turns; otherwise prefer targeted `ask_image` calls."
    )


def images_forwarding_block() -> str:
    """General guidance for forwarding images into nested tools.

    Manager‑agnostic: safe to include in any prompt where nested tool calls may occur.
    """
    return (
        "Images forwarding to nested tools\n"
        "----------------------------------\n"
        "• When delegating to another tool that declares an `images` parameter, forward the relevant images.\n"
        "• Rewrite or augment image annotations so they align with the delegated question/action (not the original phrasing).\n"
        "• Prefer AnnotatedImageRefs; include a curated subset and preserve user‑referenced ordering when it matters.\n"
        "• If no images are relevant, omit them rather than attaching unrelated visuals."
    )
