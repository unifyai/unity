from __future__ import annotations

from datetime import datetime
from ..common.prompt_helpers import now, PromptParts


import textwrap


def build_image_ask_prompt(
    *,
    caption: str | None,
    timestamp: datetime | None,
) -> PromptParts:
    """
    Return a concise system message for image Q&A.

    Notes
    -----
    - The image itself will be provided as an image_url content block in the
      user message. The model should reason over it but respond with text only.
    - Include lightweight context when available (timestamp, caption).
    """

    ts = timestamp.isoformat() if timestamp else "unknown-time"
    cap = caption or "(no caption provided)"

    parts = PromptParts()

    parts.add(
        "You are a helpful vision assistant. An image will be provided in the next user message.",
    )

    context_block = "\n".join(
        [
            "Context",
            "-------",
            f"• Timestamp: {ts}",
            f"• Caption: {cap}",
        ],
    )
    parts.add(context_block)

    requirements_block = textwrap.dedent(
        """
        Requirements
        -----------
        • Look at the provided image and answer the user's question clearly and concisely.
        • Respond with plain text only. Do NOT attempt to call tools or output JSON/function calls.
        • Do not include raw image data or base64 in your response.
        • If relevant, you may cite visual evidence (colors, objects, text seen).
        • If uncertain, state the uncertainty and what would resolve it.
        • Do not assume system-specific identifiers or structured record fields (e.g., ids, names, statuses, queue/thread
          references, timestamps) are present. If the user's question asks for such fields and they are not visibly shown,
          state that they are not visible and describe what is visible instead.
        """,
    ).strip()
    parts.add(requirements_block)

    # Append current time for reproducibility (dynamic content)
    parts.add(f"Current UTC time is {now()}.", static=False)

    return parts
